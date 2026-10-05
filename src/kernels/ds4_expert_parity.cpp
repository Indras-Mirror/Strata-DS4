// src/kernels/ds4_expert_parity.cpp - Phase 2 gate: DeepSeek-V4-Flash's expert formats on both paths.
//
//     build/ds4_expert_parity <shard.gguf> [--layer L] [--expert E] [--cpu-only] [--bench N]
//
// One REAL expert (gate/up IQ2_XXS, down Q2_K) of the 0731 artifact is read as a slice of its tensors - the mmap
// faults in only those pages, never the 80 GB file - and a random input runs it three ways:
//
//     (a) the float reference: ggml's own dequantizer (`to_float`) and an F32 (double-accumulated) matmul;
//     (b) the CPU pool path: `native_quant_act` / `native_gu_rows` / `native_quant_h` / `native_down_rows`,
//         i.e. ggml-cpu's vec_dot with its own quantized activations (Q8_K for both roles here);
//     (c) the GPU native MMVQ path: `native_mmvq` (Q2_K down, and IQ2_XXS gate/up through `iq_mmvq`).
//
// Each role's relative error against (a) must be < 1e-3; the fused expert output is checked too.  --cpu-only
// skips every CUDA call so the CPU half runs before the shared GPU is free.  --bench N reports ms per expert
// matvec on 1 thread, on 8 threads (the pool's row split), and on the GPU.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t H = 4096;    // n_embd
constexpr int64_t FF = 2048;   // n_ff (expert feed-forward length)
constexpr int GU = 16;         // IQ2_XXS
constexpr int DN = 10;         // Q2_K
constexpr double LIMIT = 1e-3; // the gate's relative-error limit

int g_fail = 0;

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += std::fabs((double) a[i] - b[i]);
        d += std::fabs((double) b[i]);
    }
    return n / (d + 1e-30);
}

void check(bool ok, const char* what, double got) {
    std::printf("  %-42s rel %.3e  %s\n", what, got, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds4_expert_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

float silu(float g) { return g / (1.0f + std::exp(-g)); }

// The expert's three role matrices, dequantized by ggml's own to_float.
struct Weights {
    std::vector<float> G, U, D;   // [FF x H], [FF x H], [H x FF]
};

// The reference: gate/up matmul, SwiGLU, down matmul, all in double.
void reference(const Weights& w, const std::vector<float>& x, std::vector<float>& g, std::vector<float>& u,
               std::vector<float>& h, std::vector<float>& out) {
    g.assign(FF, 0.f);
    u.assign(FF, 0.f);
    h.assign(FF, 0.f);
    for (int64_t r = 0; r < FF; ++r) {
        double sg = 0, su = 0;
        for (int64_t i = 0; i < H; ++i) {
            sg += (double) w.G[(size_t) (r * H + i)] * x[(size_t) i];
            su += (double) w.U[(size_t) (r * H + i)] * x[(size_t) i];
        }
        g[(size_t) r] = (float) sg;
        u[(size_t) r] = (float) su;
        h[(size_t) r] = (float) (sg / (1.0 + std::exp(-sg)) * su);
    }
    out.assign(H, 0.f);
    for (int64_t r = 0; r < H; ++r) {
        double so = 0;
        for (int64_t i = 0; i < FF; ++i) so += (double) w.D[(size_t) (r * FF + i)] * h[(size_t) i];
        out[(size_t) r] = (float) so;
    }
}

// The reference for the KERNEL, not for the quantization: the weights come from ggml's dequantizer and the
// activation is the dequantization of the very buffer the kernel consumes (Q8_K on the CPU, q8_1 on the GPU), so
// what is measured is the dot product's arithmetic and not the 8-bit activation rounding.  The pure-float
// reference is printed alongside so the activation rounding is on the record.
void matvec(const std::vector<float>& w, int64_t rows, int64_t cols, const std::vector<float>& x,
            std::vector<float>& y) {
    y.assign((size_t) rows, 0.f);
    for (int64_t r = 0; r < rows; ++r) {
        double s = 0;
        for (int64_t i = 0; i < cols; ++i) s += (double) w[(size_t) (r * cols + i)] * x[(size_t) i];
        y[(size_t) r] = (float) s;
    }
}

// Q8_K (GGML_TYPE_Q8_K): a float scale per 256 values, int8 quants.  The buffer comes from ggml-cpu's from_float.
void deq_q8_k(const uint8_t* q, int64_t n, std::vector<float>& out) {
    struct Block {
        float d;
        int8_t qs[256];
        int16_t bsums[16];
    };
    static_assert(sizeof(Block) == 292, "block_q8_K layout");
    out.resize((size_t) n);
    for (int64_t b = 0; b < n / 256; ++b) {
        const Block* blk = reinterpret_cast<const Block*>(q + b * sizeof(Block));
        for (int i = 0; i < 256; ++i) out[(size_t) (b * 256 + i)] = blk->d * blk->qs[i];
    }
}

// q8_1 (the CUDA activation): a fp16 scale per 32 values.
void deq_q8_1(const uint8_t* q, int64_t n, std::vector<float>& out) {
    struct Block {
        uint16_t d, s;
        int8_t qs[32];
    };
    static_assert(sizeof(Block) == 36, "block_q8_1 layout");
    out.resize((size_t) n);
    for (int64_t b = 0; b < n / 32; ++b) {
        const Block* blk = reinterpret_cast<const Block*>(q + b * sizeof(Block));
        const float d = ggml_fp16_to_fp32(blk->d);
        for (int i = 0; i < 32; ++i) out[(size_t) (b * 32 + i)] = d * blk->qs[i];
    }
}

// The CPU pool's structure: quantize x, split the FF gate/up rows over `nthreads`, quantize h, split the H down
// rows.  The down rows are H = 2 * FF, so thread t's gate/up rows [t*FF/T, (t+1)*FF/T) map to down rows twice as
// wide - a barrier between the two phases, exactly the pool's row split.
double bench_cpu(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, const std::vector<float>& x, int nthreads,
                 int iters) {
    std::vector<uint8_t> act(cpu::kNativeActBytes);
    std::vector<uint8_t> hq(cpu::kNativeHBytes);
    std::vector<float> ff(FF), out(H);
    auto gu_rows = [&](int r0, int r1) {
        const void* a[1] = {act.data()};
        float* fp[1] = {ff.data()};
        cpu::native_gu_rows(f, blob.data(), a, 1, fp, r0, r1);
    };
    auto dn_rows = [&](int r0, int r1) {
        const void* hp[1] = {hq.data()};
        float* op[1] = {out.data()};
        cpu::native_down_rows(f, blob.data(), hp, 1, op, r0, r1);
    };
    auto over = [&](int n, const std::function<void(int, int)>& fn) {
        if (nthreads <= 1) { fn(0, n); return; }
        std::vector<std::thread> th;
        for (int t = 0; t < nthreads; ++t)
            th.emplace_back(fn, (int) ((int64_t) t * n / nthreads), (int) ((int64_t) (t + 1) * n / nthreads));
        for (auto& t : th) t.join();
    };
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        cpu::native_quant_act(f, x.data(), act.data());
        over((int) FF, gu_rows);
        cpu::native_quant_h(f, ff.data(), hq.data());
        over((int) H, dn_rows);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2 || std::string(argv[1]).empty()) {   // the real-file gate needs $DS4_GGUF; without it, skip
        std::printf("ds4_expert_parity: skipped (pass <shard.gguf> or set $DS4_GGUF)\n");
        return 0;
    }
    const std::string path = argv[1];
    int layer = 0, expert = 0, bench = 0;
    bool cpu_only = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layer" && i + 1 < argc) layer = std::atoi(argv[++i]);
        else if (a == "--expert" && i + 1 < argc) expert = std::atoi(argv[++i]);
        else if (a == "--bench" && i + 1 < argc) bench = std::atoi(argv[++i]);
        else if (a == "--cpu-only") cpu_only = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }

    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GU, DN, H, FF, f, err)) {
        std::printf("native_fmt(%s/%s, %lld, %lld): %s\n", ggml_type_name((ggml_type) GU),
                    ggml_type_name((ggml_type) DN), (long long) H, (long long) FF, err.c_str());
        return 1;
    }
    std::printf("expert shape %lld/%lld, types %s/%s, blob %zu B\n", (long long) H, (long long) FF,
                ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), f.bytes);

    // ---- one real expert's slice from the GGUF (mmap: only the touched pages load)
    const strata::GgufModel model(strata::gguf_split_paths(path));
    const strata::TensorInfo* ti[3] = {};
    const uint8_t* data[3] = {};
    const char* roles[3] = {"gate", "up", "down"};
    for (int r = 0; r < 3; ++r) {
        size_t at = 0;
        const std::string name = "blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight";
        ti[r] = model.find(name, &at);
        if (!ti[r]) { std::printf("%s: no such tensor\n", name.c_str()); return 1; }
        data[r] = model.shard(at).tensor_data(*ti[r]);
    }
    if ((int) ti[0]->type != GU || (int) ti[1]->type != GU || (int) ti[2]->type != DN) {
        std::printf("layer %d: expected gate/up %s and down %s, got %s/%s/%s\n", layer, ggml_type_name((ggml_type) GU),
                    ggml_type_name((ggml_type) DN), ti[0]->type_name(), ti[1]->type_name(), ti[2]->type_name());
        return 1;
    }
    const size_t gu_stride = (size_t) f.up_off;                 // bytes per expert, gate/up
    const size_t d_stride = (size_t) f.bytes - f.down_off;      // bytes per expert, down
    std::vector<uint8_t> blob(f.bytes);
    std::memcpy(blob.data(), data[0] + (size_t) expert * gu_stride, gu_stride);
    std::memcpy(blob.data() + f.up_off, data[1] + (size_t) expert * gu_stride, gu_stride);
    std::memcpy(blob.data() + f.down_off, data[2] + (size_t) expert * d_stride, d_stride);

    Weights w;
    w.G.resize((size_t) FF * H);
    w.U.resize((size_t) FF * H);
    w.D.resize((size_t) H * FF);
    ggml_get_type_traits((ggml_type) GU)->to_float(blob.data(), w.G.data(), FF * H);
    ggml_get_type_traits((ggml_type) GU)->to_float(blob.data() + f.up_off, w.U.data(), FF * H);
    ggml_get_type_traits((ggml_type) DN)->to_float(blob.data() + f.down_off, w.D.data(), H * FF);

    std::mt19937 rng(4096 + layer * 131 + expert);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) H);
    for (auto& v : x) v = nd(rng);
    std::vector<float> g, u, h, ref;
    reference(w, x, g, u, h, ref);

    // (b) the CPU pool path, per role and fused
    {
        std::vector<uint8_t> act(cpu::kNativeActBytes);
        cpu::native_quant_act(f, x.data(), act.data());
        const void* a[1] = {act.data()};
        std::vector<float> ff(FF);
        float* fp[1] = {ff.data()};
        cpu::native_gu_rows(f, blob.data(), a, 1, fp, 0, (int) FF);
        // per-role dots against ggml-cpu's own vec_dot on the same Q8_K activation (the pool's fallback path)
        const auto* tg = ggml_get_type_traits_cpu((ggml_type) GU);
        std::vector<float> cg(FF), cu(FF), cd(H), fused_c(H);
        std::vector<uint8_t> hq_ref(cpu::kNativeHBytes), hq_ff(cpu::kNativeHBytes);
        for (int64_t r = 0; r < FF; ++r) {
            tg->vec_dot((int) H, &cg[(size_t) r], 0, blob.data() + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
            tg->vec_dot((int) H, &cu[(size_t) r], 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
        }
        // the kernel reference: dequantize the same Q8_K activation, then the F32 matmul
        std::vector<float> xd;
        deq_q8_k(act.data(), H, xd);
        std::vector<float> rg, ru;
        matvec(w.G, FF, H, xd, rg);
        matvec(w.U, FF, H, xd, ru);
        cpu::native_quant_h(f, h.data(), hq_ref.data());
        std::vector<float> hd;
        deq_q8_k(hq_ref.data(), FF, hd);
        std::vector<float> rd;
        matvec(w.D, H, FF, hd, rd);
        const auto* td = ggml_get_type_traits_cpu((ggml_type) DN);
        for (int64_t r = 0; r < H; ++r)
            td->vec_dot((int) FF, &cd[(size_t) r], 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0,
                        hq_ref.data(), 0, 1);
        // the fused path: the down activation is the Q8_K of the CPU's own SwiGLU(ff)
        cpu::native_quant_h(f, ff.data(), hq_ff.data());
        std::vector<float> hdf;
        deq_q8_k(hq_ff.data(), FF, hdf);
        std::vector<float> rdf;
        matvec(w.D, H, FF, hdf, rdf);
        const void* hp[1] = {hq_ff.data()};
        float* op[1] = {fused_c.data()};
        cpu::native_down_rows(f, blob.data(), hp, 1, op, 0, (int) H);
        check(rel(cg, rg) < LIMIT, "cpu gate (IQ2_XXS) vs dequant+matmul", rel(cg, rg));
        check(rel(cu, ru) < LIMIT, "cpu up   (IQ2_XXS) vs dequant+matmul", rel(cu, ru));
        check(rel(cd, rd) < LIMIT, "cpu down (Q2_K)    vs dequant+matmul", rel(cd, rd));
        check(rel(fused_c, rdf) < LIMIT, "cpu fused expert   vs dequant+matmul", rel(fused_c, rdf));
        std::printf("    (vs the pure-float reference, i.e. including the Q8_K activation rounding: %.2e / %.2e / "
                    "%.2e)\n", rel(cg, g), rel(cu, u), rel(fused_c, ref));
    }

    // (c) the GPU native MMVQ path
    if (!cpu_only) {
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        void *dblob = nullptr, *dx = nullptr, *dxq = nullptr, *dh = nullptr, *dhq = nullptr, *dout = nullptr;
        ck(cudaMalloc(&dblob, blob.size()), "malloc blob");
        ck(cudaMalloc(&dx, (size_t) H * 4), "malloc x");
        ck(cudaMalloc(&dxq, strata::kernels::native_q8_1_bytes((int) H, 1)), "malloc xq");
        ck(cudaMalloc(&dh, (size_t) FF * 4), "malloc h");
        ck(cudaMalloc(&dhq, strata::kernels::native_q8_1_bytes((int) FF, 1)), "malloc hq");
        ck(cudaMalloc(&dout, (size_t) H * 4), "malloc out");
        ck(cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice), "copy blob");
        ck(cudaMemcpy(dx, x.data(), (size_t) H * 4, cudaMemcpyHostToDevice), "copy x");
        auto* dm = (uint8_t*) dblob;
        // gate/up: the engine's native_mmvq entry point (IQ2_XXS delegates to iq_mmvq)
        std::vector<float> gg(FF), gu(FF);
        strata::kernels::quantize_q8_1_rows((const float*) dx, 1, H, dxq, s);
        ck(cudaStreamSynchronize(s), "sync xq");
        std::vector<uint8_t> xq_host(strata::kernels::native_q8_1_bytes((int) H, 1));
        ck(cudaMemcpy(xq_host.data(), dxq, xq_host.size(), cudaMemcpyDeviceToHost), "d2h xq");
        std::vector<float> xdg;
        deq_q8_1(xq_host.data(), H, xdg);
        std::vector<float> rgg, rgu;
        matvec(w.G, FF, H, xdg, rgg);
        matvec(w.U, FF, H, xdg, rgu);
        strata::kernels::native_mmvq(GU, dm, dxq, (float*) dout, (int) H, (int) FF, 1, s);
        ck(cudaStreamSynchronize(s), "sync gate");
        ck(cudaMemcpy(gg.data(), dout, (size_t) FF * 4, cudaMemcpyDeviceToHost), "d2h gate");
        strata::kernels::native_mmvq(GU, dm + f.up_off, dxq, (float*) dout, (int) H, (int) FF, 1, s);
        ck(cudaStreamSynchronize(s), "sync up");
        ck(cudaMemcpy(gu.data(), dout, (size_t) FF * 4, cudaMemcpyDeviceToHost), "d2h up");
        check(rel(gg, rgg) < LIMIT, "gpu gate (IQ2_XXS) vs dequant+matmul", rel(gg, rgg));
        check(rel(gu, rgu) < LIMIT, "gpu up   (IQ2_XXS) vs dequant+matmul", rel(gu, rgu));
        // down: Q2_K against the q8_1 dequantization of the reference h, then the fused expert from the GPU's own h
        std::vector<float> od(H), of(H);
        ck(cudaMemcpy(dh, h.data(), (size_t) FF * 4, cudaMemcpyHostToDevice), "copy h");
        strata::kernels::native_quantize_q8_1((const float*) dh, dhq, (int) FF, 1, s);
        ck(cudaStreamSynchronize(s), "sync hq");
        std::vector<uint8_t> hq_host(strata::kernels::native_q8_1_bytes((int) FF, 1));
        ck(cudaMemcpy(hq_host.data(), dhq, hq_host.size(), cudaMemcpyDeviceToHost), "d2h hq");
        std::vector<float> hdg, rdg;
        deq_q8_1(hq_host.data(), FF, hdg);
        matvec(w.D, H, FF, hdg, rdg);
        strata::kernels::native_mmvq(DN, dm + f.down_off, dhq, (float*) dout, (int) FF, (int) H, 1, s);
        ck(cudaStreamSynchronize(s), "sync down");
        ck(cudaMemcpy(od.data(), dout, (size_t) H * 4, cudaMemcpyDeviceToHost), "d2h down");
        std::vector<float> gh(FF);
        for (int64_t r = 0; r < FF; ++r) gh[(size_t) r] = silu(gg[(size_t) r]) * gu[(size_t) r];
        ck(cudaMemcpy(dh, gh.data(), (size_t) FF * 4, cudaMemcpyHostToDevice), "copy gh");
        strata::kernels::native_quantize_q8_1((const float*) dh, dhq, (int) FF, 1, s);
        ck(cudaStreamSynchronize(s), "sync ghq");
        ck(cudaMemcpy(hq_host.data(), dhq, hq_host.size(), cudaMemcpyDeviceToHost), "d2h ghq");
        std::vector<float> hdfg, rdfg;
        deq_q8_1(hq_host.data(), FF, hdfg);
        matvec(w.D, H, FF, hdfg, rdfg);
        strata::kernels::native_mmvq(DN, dm + f.down_off, dhq, (float*) dout, (int) FF, (int) H, 1, s);
        ck(cudaStreamSynchronize(s), "sync fused");
        ck(cudaMemcpy(of.data(), dout, (size_t) H * 4, cudaMemcpyDeviceToHost), "d2h fused");
        check(rel(od, rdg) < LIMIT, "gpu down (Q2_K)    vs dequant+matmul", rel(od, rdg));
        check(rel(of, rdfg) < LIMIT, "gpu fused expert   vs dequant+matmul", rel(of, rdfg));
        {   // the prompt path's Q2_K dequantizer (dq_q2_k) against ggml's own to_float
            std::vector<float> dq((size_t) H * FF);
            float* ddq = nullptr;
            ck(cudaMalloc(&ddq, dq.size() * 4), "malloc dq");
            strata::kernels::iq_dequant_f32(DN, dm + f.down_off, H * FF, ddq, s);
            ck(cudaStreamSynchronize(s), "sync dq");
            ck(cudaMemcpy(dq.data(), ddq, dq.size() * 4, cudaMemcpyDeviceToHost), "d2h dq");
            check(rel(dq, w.D) < 1e-6, "gpu Q2_K dequant vs ggml to_float", rel(dq, w.D));
            cudaFree(ddq);
        }
        std::printf("    (vs the pure-float reference, i.e. including the q8_1 activation rounding: %.2e / %.2e / "
                    "%.2e)\n", rel(gg, g), rel(gu, u), rel(of, ref));
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dh); cudaFree(dhq); cudaFree(dout);
        cudaStreamDestroy(s);

        if (bench > 0) {
            // GPU: quantize + two IQ2_XXS matvecs + SwiGLU quantization + one Q2_K matvec, timed with events
            cudaStream_t bs;
            ck(cudaStreamCreate(&bs), "bench stream");
            void *bblob = nullptr, *bx = nullptr, *bxq = nullptr, *bh = nullptr, *bhq = nullptr, *bout = nullptr, *bg = nullptr;
            ck(cudaMalloc(&bblob, blob.size()), "b blob");
            ck(cudaMalloc(&bx, (size_t) H * 4), "b x");
            ck(cudaMalloc(&bxq, strata::kernels::native_q8_1_bytes((int) H, 1)), "b xq");
            ck(cudaMalloc(&bh, (size_t) FF * 4), "b h");
            ck(cudaMalloc(&bhq, strata::kernels::native_q8_1_bytes((int) FF, 1)), "b hq");
            ck(cudaMalloc(&bg, (size_t) FF * 4), "b g");
            ck(cudaMalloc(&bout, (size_t) H * 4), "b out");
            ck(cudaMemcpy(bblob, blob.data(), blob.size(), cudaMemcpyHostToDevice), "b blob copy");
            ck(cudaMemcpy(bx, x.data(), (size_t) H * 4, cudaMemcpyHostToDevice), "b x copy");
            auto* bdm = (uint8_t*) bblob;
            auto one = [&]() {
                strata::kernels::native_quantize_q8_1((const float*) bx, bxq, (int) H, 1, bs);
                strata::kernels::native_mmvq(GU, bdm, bxq, (float*) bg, (int) H, (int) FF, 1, bs);
                strata::kernels::native_mmvq(GU, bdm + f.up_off, bxq, (float*) bh, (int) H, (int) FF, 1, bs);
                strata::kernels::native_swiglu_quantize_q8_1((const float*) bg, (const float*) bh, bhq, (int) FF, 1, bs);
                strata::kernels::native_mmvq(DN, bdm + f.down_off, bhq, (float*) bout, (int) FF, (int) H, 1, bs);
            };
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "e0");
            ck(cudaEventCreate(&e1), "e1");
            for (int i = 0; i < 3; ++i) one();   // warm
            ck(cudaEventRecord(e0, bs), "rec e0");
            for (int i = 0; i < bench; ++i) one();
            ck(cudaEventRecord(e1, bs), "rec e1");
            ck(cudaEventSynchronize(e1), "sync e1");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            std::printf("  bench gpu        %8.3f ms/expert (%d iters)\n", ms / bench, bench);
            cudaFree(bblob); cudaFree(bx); cudaFree(bxq); cudaFree(bh); cudaFree(bhq); cudaFree(bg); cudaFree(bout);
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(bs);
        }
    }

    if (bench > 0) {
        std::printf("  bench cpu 1thr   %8.3f ms/expert\n", bench_cpu(f, blob, x, 1, bench));
        std::printf("  bench cpu 8thr   %8.3f ms/expert\n", bench_cpu(f, blob, x, 8, bench));
    }
    std::printf("ds4_expert_parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
