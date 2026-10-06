// tools/ds4/test_ds4_moe.cpp - slice ds4-moe's gate: `Ds4MoeTier` against a dequant+F32 reference.
//
//     build-ds4/test_ds4_moe       [--gguf PATH] [--require-gguf] [--sets N] [--no-synthetic] [--no-real]
//     build-ds4-cuda/test_ds4_moe_gpu [--gpu] [same flags]           (the SAME source, the CUDA flavour)
//     build-ds4-cuda/test_ds4_moe_gpu --timing --routes R [--pred P] [--profile Q] [--tokens N] [--offset O]
//                                     [--slots S] [--arena-gib G] [--pcie-frac F] [--pf-b B] [--gap-ms X]
//                                     [--threads T]     (a FULL-MODEL run: wrap it in tools/ds4/memguard.sh)
//
// THE INVARIANT UNDER TEST.  For any (layer, ids, weights, x) the tier's output equals the weighted sum of the
// dequantised experts computed in F32 - the only permitted difference being the expert kernels' own 8-bit
// activation rounding (`ds4_moe.hpp`).  The reference dequantises the expert weights with ggml's own `to_float`
// and dequantises the ENGINE'S OWN quantized activation, so what is measured is the kernel's arithmetic and not
// the 8-bit activation rounding; that rounding is measured separately and printed alongside.  This is
// `moe_replay --correctness`'s Phase-4a gate 1 moved onto the extracted library and widened from one activation
// to several layers and several expert sets with real router weights.
//
// FOUR ARMS, because they answer different questions:
//
//   * SYNTHETIC (always run): DS4's real geometry (43 x 256, top-6, 4096/2048) with expert blobs quantized from
//     random F32 weights by ggml's own `ggml_quantize_chunk` - real IQ2_XXS / Q2_K blocks, real kernels, real
//     sizes, no external file.  This is the arm that must run everywhere.
//   * REAL (run when the model is present): the packet's sanctioned fallback - up to 12 REAL expert slices read
//     out of the 0731 GGUF through its mmap (6.75 MiB touched per expert; a slice of a tensor, not a model
//     load).  tools/ds4/make_mini_gguf.py's miniature model is NOT used: it writes its routed experts as ZERO
//     blocks on purpose ("the smoke test is about wiring and finiteness, not about expert values",
//     make_mini_gguf.py:154-156), so a numeric gate against it would be vacuous.
//   * GPU (`--gpu`, CUDA flavour only): the four sources a routed expert can come from - a resident VRAM slot,
//     the prefetch staging, the PCIe DMA out of the pinned arena, and the blob source itself (the file tier,
//     computed by the pool) - each against the same reference on 24 real expert slices (162 MiB of the GGUF via
//     its mmap), with the split counted as well as the value.  Its reference quantizes with the CARD's format
//     (q8_1, scales stored as fp16) for the card's experts and the pool's (Q8_K) for the pool's, because the
//     tier really does use both.  See the block comment above `deq_q8_1`.
//   * TIMING (`--timing`, CUDA flavour only): the tier's own `run` on the real model, real routes and the real
//     prediction file - the replay's measured arm (FINDINGS s12) driven through the library.  FULL MODEL: it goes
//     through `tools/ds4/memguard.sh 64 62 -- ...`, never as a bare process.
//
// The clamp probe printed at the end of each arm is the evidence behind the header's SwiGLU note: the largest
// |gate| and |up| pre-activation the sampled experts produced, against the model's `swiglu_clamp_exp` of 10.0.
//
// SAFETY.  `test_ds4_moe` links `ds4_moe_cpu`, which is `ds4_moe.cpp` compiled WITHOUT `DS4_MOE_CUDA`: no CUDA
// call is in it and no CUDA library is on its link line, so it cannot initialise the driver - that property is
// what makes the CPU gate safe to run while another process owns the GPU, and it is why the GPU arm is a second
// binary and not a flag on this one.  Run the CPU gate under the packet's wrapper:
//     nice -n 10 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 build-ds4/test_ds4_moe
#include "ds4_moe.hpp"

#include "strata/kernels/cpu/pool.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(DS4_MOE_CUDA)
#include <cuda_runtime.h>
#endif

namespace cpu = strata::kernels::cpu;
using strata::ds4::Ds4BlobSource;
using strata::ds4::Ds4GgufBlobs;
using strata::ds4::Ds4MemoryBlobs;
using strata::ds4::Ds4MoeConfig;
using strata::ds4::Ds4MoeGeom;
using strata::ds4::Ds4MoeStats;
using strata::ds4::Ds4MoeTier;

namespace {

constexpr double kLimit = 1e-3;        ///< the gate's relative-error limit
constexpr double kWeightsScale = 1.5;  ///< deepseek4.expert_weights_scale, applied after the normalisation
constexpr double kClampExp = 10.0;     ///< deepseek4.swiglu_clamp_exp (the kernels apply it: NativeFmt::swiglu_limit)
/// The forced-clamp arms rerun arm 2/3 with this limit so the clamp binds on a large share of the elements and the
/// kernels' clamp path is checked against the reference's (at 10.0 it binds on ~0 elements of real activations).
constexpr float kForcedClamp = 1.0f;
const char* kDefaultGguf = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf";

int g_fail = 0, g_checks = 0;

/// The largest pre-activation one expert produced on one activation: the clamp probe's raw number.
struct Probe {
    float gate_abs = 0, up_abs = 0;
    int64_t bound = 0, total = 0;   ///< pre-activations the reference clamped (gate > lim or |up| > lim) / seen
    void merge(const Probe& o) {
        gate_abs = std::max(gate_abs, o.gate_abs);
        up_abs = std::max(up_abs, o.up_abs);
        bound += o.bound;
        total += o.total;
    }
};

/// The model's SwiGLU with its clamp (ds4_ref.cpp / llama.cpp DEEPSEEK4): gate -> min(gate, lim), up -> clamp(up,
/// -lim, lim), then silu(gate) * up.  Counts the clamped elements into `probe`.
void swiglu_ref(const float* g, const float* u, float* h, int64_t n, float lim, Probe* probe) {
    for (int64_t j = 0; j < n; ++j) {
        const float gc = std::min(g[j], lim), uc = std::min(std::max(u[j], -lim), lim);
        if (probe) {
            probe->bound += (g[j] > lim) + (std::fabs(u[j]) > lim);
            probe->total += 2;
        }
        h[j] = (gc / (1.0f + std::exp(-gc))) * uc;
    }
}

/// The clamp probe over every expert every arm sampled.
Probe probe_total;

double rel(const float* a, const float* b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        num += std::fabs((double) a[i] - (double) b[i]);
        den += std::fabs((double) b[i]);
    }
    return num / (den + 1e-30);
}

void check(bool ok, const std::string& what, double got) {
    std::printf("  %-58s rel %.3e  %s\n", what.c_str(), got, ok ? "ok" : "FAIL");
    ++g_checks;
    if (!ok) ++g_fail;
}

/// Q8_K: a float scale per 256 values (GGML_TYPE_Q8_K), the vec_dot_type of both IQ2_XXS and Q2_K on the CPU.
void deq_q8_k(const uint8_t* q, int64_t n, float* out) {
    struct Block {
        float d;
        int8_t qs[256];
        int16_t bsums[16];
    };
    static_assert(sizeof(Block) == 292, "block_q8_K layout");
    for (int64_t b = 0; b < n / 256; ++b) {
        const Block* blk = reinterpret_cast<const Block*>(q + b * sizeof(Block));
        for (int i = 0; i < 256; ++i) out[b * 256 + i] = blk->d * blk->qs[i];
    }
}

/// The activation's scale.  The experts consume `fn = rms_norm(...) * ffn_norm.weight` and an RMS-norm output is
/// unit-RMS, so `x_i = ffn_norm.weight[i] * z_i` with z ~ N(0,1) is what the experts actually see - which is what
/// decides how large the gate/up pre-activations (and so the clamp probe) are.  Both real arms read the model's
/// own weights; a tensor that is not the expected F32 vector falls back to unit weights and says so.
void read_ffn_norm(const std::string& gguf, int64_t layer, int64_t n_embd, std::vector<float>& nw) {
    std::fill(nw.begin(), nw.end(), 1.0f);
    try {
        strata::GgufFile gf(gguf);
        char name[64];
        std::snprintf(name, sizeof name, "blk.%lld.ffn_norm.weight", (long long) layer);
        const strata::TensorInfo* ti = gf.find(name);
        if (ti && ti->type == 0 /* GGML_TYPE_F32 */ && ti->elements() == (uint64_t) n_embd)
            std::memcpy(nw.data(), gf.tensor_data(*ti), (size_t) n_embd * 4);
        else
            std::printf("   (no usable %s: falling back to unit ffn_norm weights)\n", name);
    } catch (const std::exception& e) {
        std::printf("   (ffn_norm read failed: %s)\n", e.what());
    }
}

float rms_of(const std::vector<float>& v) {
    double s = 0;
    for (auto x : v) s += (double) x * (double) x;
    return (float) std::sqrt(s / (double) v.size());
}

// ================================ the reference ================================

struct RefWeights {
    std::vector<float> G, U, D;   // [n_ff x n_embd], [n_ff x n_embd], [n_embd x n_ff]
};

void dequant_weights(const uint8_t* blob, const cpu::NativeFmt& f, RefWeights& w) {
    const int64_t H = f.n_embd, FF = f.n_ff;
    w.G.assign((size_t) (FF * H), 0.f);
    w.U.assign((size_t) (FF * H), 0.f);
    w.D.assign((size_t) (H * FF), 0.f);
    ggml_get_type_traits((ggml_type) f.gu_type)->to_float(blob, w.G.data(), FF * H);
    ggml_get_type_traits((ggml_type) f.gu_type)->to_float(blob + f.up_off, w.U.data(), FF * H);
    ggml_get_type_traits((ggml_type) f.d_type)->to_float(blob + f.down_off, w.D.data(), H * FF);
}

/// y = W x, accumulated in double so the reference is not the error term.
void matvec(const float* w, int64_t rows, int64_t cols, const float* x, float* y) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* wr = w + (size_t) r * (size_t) cols;
        double s = 0;
        for (int64_t i = 0; i < cols; ++i) s += (double) wr[i] * (double) x[i];
        y[r] = (float) s;
    }
}

/// The kernel's own arithmetic: the weights dequantized by ggml, the activation the kernel consumes (dequantized
/// Q8_K of `x`, or `x` itself for the pure-float arm), SwiGLU, and the down projection.  `probe` captures the
/// gate/up pre-activations on the way through.
void ref_expert(const cpu::NativeFmt& f, const uint8_t* blob, const float* x, float* out, Probe* probe,
                bool engine_activation) {
    const int64_t H = f.n_embd, FF = f.n_ff;
    std::vector<uint8_t> act((size_t) f.act_bytes), hq((size_t) f.h_bytes);
    std::vector<float> xd((size_t) H), g((size_t) FF), u((size_t) FF), h((size_t) FF), hd((size_t) FF);
    if (engine_activation) {
        cpu::native_quant_act(f, x, act.data());
        deq_q8_k(act.data(), H, xd.data());
    } else {
        std::memcpy(xd.data(), x, (size_t) H * 4);
    }
    RefWeights w;
    dequant_weights(blob, f, w);
    matvec(w.G.data(), FF, H, xd.data(), g.data());
    matvec(w.U.data(), FF, H, xd.data(), u.data());
    if (probe) {
        for (int64_t j = 0; j < FF; ++j) {
            probe->gate_abs = std::max(probe->gate_abs, std::fabs(g[(size_t) j]));
            probe->up_abs = std::max(probe->up_abs, std::fabs(u[(size_t) j]));
        }
    }
    swiglu_ref(g.data(), u.data(), h.data(), FF, f.swiglu_limit, probe);
    if (engine_activation) {
        cpu::native_quant_h(f, h.data(), hq.data());
        deq_q8_k(hq.data(), FF, hd.data());
    } else {
        std::memcpy(hd.data(), h.data(), (size_t) FF * 4);
    }
    matvec(w.D.data(), H, FF, hd.data(), out);
}

// ================================ one set of six experts ================================

struct SetResult {
    double worst_sum = 0, worst_expert = 0, worst_expert_float = 0;
    Probe probe;
};

/// The layer's weighted sum through the tier, each of its six experts alone through the tier's own CPU path, and
/// the pure-float reference for the record.
bool check_set(Ds4MoeTier& tier, Ds4BlobSource& src, const cpu::NativeFmt& f, cpu::ExpertPool& pool, int64_t layer,
               const int32_t* ids, const float* x, const float* w6, const std::string& tag, SetResult& res) {
    const int64_t H = f.n_embd;
    const int64_t K = tier.geom().top_k;
    std::vector<uint8_t> blob((size_t) src.blob_bytes());
    std::vector<float> e_ref((size_t) K * H), e_flo((size_t) K * H), e_tier((size_t) H), ref_sum((size_t) H),
        flo_sum((size_t) H), out((size_t) H), probe_flo((size_t) H);
    std::fill(ref_sum.begin(), ref_sum.end(), 0.0f);
    std::fill(flo_sum.begin(), flo_sum.end(), 0.0f);
    for (int64_t k = 0; k < K; ++k) {
        src.read(layer, ids[k], blob.data());
        ref_expert(f, blob.data(), x, &e_ref[(size_t) k * H], &res.probe, true);
        ref_expert(f, blob.data(), x, &e_flo[(size_t) k * H], nullptr, false);
        for (int64_t i = 0; i < H; ++i) {
            ref_sum[(size_t) i] += w6[k] * e_ref[(size_t) k * H + i];
            flo_sum[(size_t) i] += w6[k] * e_flo[(size_t) k * H + i];
        }
        strata::ds4::ds4_moe_cpu_expert(f, H, f.n_ff, blob.data(), x, e_tier.data(), pool);
        res.worst_expert = std::max(res.worst_expert, rel(e_tier.data(), &e_ref[(size_t) k * H], H));
        res.worst_expert_float = std::max(res.worst_expert_float, rel(e_tier.data(), &e_flo[(size_t) k * H], H));
    }
    if (!tier.run(layer, ids, w6, x, out.data())) {
        std::printf("  %s: tier.run refused\n", tag.c_str());
        ++g_fail;
        return false;
    }
    res.worst_sum = rel(out.data(), ref_sum.data(), H);
    check(res.worst_sum < kLimit, tag + ": weighted sum vs dequant+F32", res.worst_sum);
    check(res.worst_expert < kLimit,
          tag + ": single expert vs dequant+F32 (worst of " + std::to_string(K) + ")", res.worst_expert);
    std::printf("  %-58s      (pure-float, i.e. the 8-bit activation rounding: %.3e)\n", "",
                res.worst_expert_float);
    (void) probe_flo;
    return true;
}

/// `w6`: `top_k` positive draws normalised to 1 and then scaled by the model's `expert_weights_scale`.
void router_weights(std::mt19937& rng, int64_t K, float* w6) {
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    double sum = 0;
    for (int64_t k = 0; k < K; ++k) {
        w6[k] = ud(rng);
        sum += w6[k];
    }
    for (int64_t k = 0; k < K; ++k) w6[k] = (float) (w6[k] / sum * kWeightsScale);
}

struct Args {
    std::string gguf = kDefaultGguf;
    bool require_gguf = false, no_synthetic = false, no_real = false, gpu = false, timing = false;
    int sets = 3;
    // --timing (a FULL-MODEL run: wrap it in tools/ds4/memguard.sh 64 62 -- ...)
    std::string routes, pred, profile;
    int64_t tokens = 32, offset = 512, slots = 2150, threads = 7;
    double arena_gib = 52.0, pcie_frac = 0.25, pf_b = 1.43, gap_ms = 0.41;
};

// ================================ arm 1: synthetic blobs ================================

int arm_synthetic(int sets) {
    Ds4MoeGeom g;
    cpu::NativeFmt f;
    std::string err;
    if (!strata::ds4::ds4_moe_blob_layout(g, f, err)) {
        std::printf("synthetic: %s\n", err.c_str());
        return 1;
    }
    std::printf("\n-- arm 1: synthetic, %lld layers x %lld experts top-%lld, blob %.2f MiB quantized by ggml\n",
                (long long) g.n_layers, (long long) g.n_experts, (long long) g.top_k,
                (double) f.bytes / 1048576.0);

    const int n_sets = std::min(sets, 3);
    const int64_t layer_of[3] = {0, 1, 7};
    Ds4MemoryBlobs blobs(g, f.bytes);
    std::mt19937 rng(20261006);
    // Weights scaled as a well-conditioned model would be: with a unit-RMS activation the pre-activations land
    // at ~N(0, 1).  The REAL magnitudes come from arm 2, which uses the model's own weights.
    std::normal_distribution<float> nd(0.f, 1.0f / std::sqrt((float) f.n_embd));
    std::vector<float> src((size_t) (f.n_embd * f.n_ff));
    std::vector<float> ones((size_t) std::max(f.n_embd, f.n_ff), 1.0f);   // the "no importance" imatrix
    std::vector<uint8_t> blob((size_t) f.bytes);
    const double t0 = (double) std::clock() / CLOCKS_PER_SEC;
    for (int s = 0; s < n_sets; ++s) {
        for (int64_t e = 0; e < g.top_k; ++e) {
            const int32_t id = (int32_t) (s * 6 + e + 1);
            for (int role = 0; role < 2; ++role) {   // gate, up: n_ff rows of n_embd
                for (auto& v : src) v = nd(rng);
                ggml_quantize_chunk((ggml_type) g.gu_type, src.data(), blob.data() + (role ? f.up_off : 0), 0,
                                    f.n_ff, f.n_embd, ones.data());
            }
            for (auto& v : src) v = nd(rng);         // down: n_embd rows of n_ff
            ggml_quantize_chunk((ggml_type) g.d_type, src.data(), blob.data() + f.down_off, 0, f.n_embd, f.n_ff,
                                nullptr);
            blobs.set(layer_of[s], id, blob.data());
        }
    }
    std::printf("   %lld expert blobs built in %.1f s\n", (long long) blobs.count(),
                (double) std::clock() / CLOCKS_PER_SEC - t0);

    Ds4MoeConfig cfg;
    cfg.cpu_only = true;
    cfg.slots = 0;
    cfg.pf_b = 0.0;
    cfg.arena_gib = 0.0;
    Ds4MoeTier tier;
    if (!tier.init(g, &blobs, cfg, err)) {
        std::printf("synthetic: init: %s\n", err.c_str());
        return 1;
    }
    cpu::ExpertPool& pool = tier.pool();
    std::printf("   tier mode: %s, %d pool workers\n", tier.mode(), pool.workers());

    // A unit-RMS activation: what the experts consume in the model, where `fn` is RMS-normed.
    std::normal_distribution<float> xn(0.f, 1.0f);
    std::vector<float> x((size_t) f.n_embd), w6((size_t) g.top_k);
    for (int s = 0; s < n_sets; ++s) {
        std::vector<int32_t> ids((size_t) g.top_k);
        for (int64_t e = 0; e < g.top_k; ++e) ids[(size_t) e] = (int32_t) (s * 6 + e + 1);
        for (auto& v : x) v = xn(rng);
        router_weights(rng, g.top_k, w6.data());
        for (const auto& id : ids)
            if (!blobs.has(layer_of[s], id)) {
                std::printf("  synthetic: blob missing for L%lld e%d\n", (long long) layer_of[s], id);
                return 1;
            }
        SetResult r;
        check_set(tier, blobs, f, pool, layer_of[s], ids.data(), x.data(), w6.data(),
                  "synthetic L" + std::to_string(layer_of[s]), r);
        probe_total.merge(r.probe);
    }
    tier.close();
    return 0;
}

// ================================ arm 2: real GGUF slices ================================

int arm_real(const std::string& gguf, int sets, bool require_gguf, float lim_override = 0.0f) {
    FILE* p = std::fopen(gguf.c_str(), "rb");
    const bool present = p != nullptr;
    if (p) std::fclose(p);
    if (!present) {
        std::printf("\n-- arm 2: real GGUF slices: SKIPPED (%s not readable)%s\n", gguf.c_str(),
                    require_gguf ? ", and --require-gguf was given" : "");
        return require_gguf ? 1 : 0;
    }
    std::string err;
    Ds4MoeGeom g;
    if (!strata::ds4::ds4_moe_geom_from_gguf(gguf, g, err)) {
        std::printf("real: geom: %s\n", err.c_str());
        return 1;
    }
    if (lim_override > 0.0f) {
        g.swiglu_limit = lim_override;
        std::printf("\n-- arm 2 again with the SwiGLU clamp forced to %.2f (the GGUF's is %.1f)\n", lim_override, kClampExp);
    }
    cpu::NativeFmt f;
    if (!strata::ds4::ds4_moe_blob_layout(g, f, err)) {
        std::printf("real: %s\n", err.c_str());
        return 1;
    }
    std::printf("\n-- arm 2: real slices of the GGUF: %lld layers x %lld experts top-%lld, blob %.2f MiB "
                "(%lld slices touched)\n", (long long) g.n_layers, (long long) g.n_experts, (long long) g.top_k,
                (double) f.bytes / 1048576.0, (long long) (std::min(sets, 2) * g.top_k));
    Ds4GgufBlobs blobs(g, gguf);
    if (!blobs.open(err)) {
        std::printf("real: %s\n", err.c_str());
        return 1;
    }
    Ds4MoeConfig cfg;
    cfg.cpu_only = true;
    cfg.slots = 0;
    cfg.pf_b = 0.0;
    cfg.arena_gib = 0.0;
    Ds4MoeTier tier;
    if (!tier.init(g, &blobs, cfg, err)) {
        std::printf("real: init: %s\n", err.c_str());
        return 1;
    }
    cpu::ExpertPool& pool = tier.pool();
    // Layer 0 is hash-routed (its experts are chosen by token id) and layer 20 is router-routed; both hold the
    // same expert tensors, which is the whole point - the tier does not care how the ids were chosen.
    const int64_t layers[2] = {0, 20};
    std::mt19937 rng(20261006);
    std::normal_distribution<float> xn(0.f, 1.0f);
    std::vector<float> x((size_t) f.n_embd), nw((size_t) f.n_embd), w6((size_t) g.top_k);
    for (int s = 0; s < std::min(sets, 2); ++s) {
        // THE ACTIVATION SCALE MATTERS FOR THE CLAMP PROBE, so it is the model's own.
        read_ffn_norm(gguf, layers[s], f.n_embd, nw);
        const double nw_rms = rms_of(nw);

        std::vector<int32_t> ids((size_t) g.top_k);
        for (int64_t e = 0; e < g.top_k; ++e) ids[(size_t) e] = (int32_t) (s * 11 + e * 7 + 3);
        for (int64_t i = 0; i < f.n_embd; ++i) x[(size_t) i] = nw[(size_t) i] * xn(rng);
        router_weights(rng, g.top_k, w6.data());
        std::printf("   L%lld: ffn_norm.weight RMS %.4f (the activation's scale)\n", (long long) layers[s], nw_rms);
        SetResult r;
        check_set(tier, blobs, f, pool, layers[s], ids.data(), x.data(), w6.data(),
                  "real L" + std::to_string(layers[s]), r);
        probe_total.merge(r.probe);
    }
    tier.close();
    return 0;
}

#if defined(DS4_MOE_CUDA)

// ================================ arm 3: the GPU's four sources ================================
//
// The card runs the same expert kernels as the pool but a different activation format: q8_1, not Q8_K
// (`q8_1_store`, src/kernels/cuda/iq_kernels.cu:1372, used by `native_quantize_q8_1_kernel` for x and by the
// gate/up -> SwiGLU -> q8_1 pass for the intermediate h).  Comparing the GPU against arm 1/2's Q8_K reference
// would therefore measure the difference between two 8-bit formats (~2e-2, see moe_replay's own note) and say
// nothing about the kernels, so the reference here quantizes with the CARD's rule - the same construction
// `src/kernels/ds4_expert_parity.cpp` uses (its gate: 3.0e-5), which reads the card's own bytes; the tier keeps
// those private, so this is the host replica of that kernel, block for block:
//
//   q = round(x / (amax / 127)) clamped to the int8 range, with the block's scale then STORED AS fp16
//   (`q8_1_ds` -> `__float2half`) and the dot product consuming fp16(d) * q.
//
// That fp16 store is not a detail: a reference that keeps the scale in fp32 differs from the card by the fp16
// rounding of the scale - one coherent ~5e-4 relative shift over each whole 32-value block, ~2.2e-3 on these
// activations, which is how this arm first failed.  A well-scaled activation is nowhere near the format's finite
// limits (q8_1_finite), so those clamps never bind here; `s`, the block's fp16 sum, is the quantized weights'
// min-term correction, not part of the activation's value.

/// The card's q8_1 activation as the kernels' dot products see it: fp16(d) * q per value.
void deq_q8_1(const float* x, int64_t n, float* out) {
    for (int64_t b = 0; b < n / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[(size_t) (b * 32 + i)]));
        const float d32 = amax / 127.0f;                              // what the quantizer rounded with
        const float d = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d32));    // what the block stores
        for (int i = 0; i < 32; ++i) {
            const float xi = x[(size_t) (b * 32 + i)];
            float q = amax == 0.0f ? 0.0f : std::round(xi / d32);
            q = std::min(127.0f, std::max(-127.0f, q));
            out[(size_t) (b * 32 + i)] = d * q;
        }
    }
}

/// One expert on the GPU's arithmetic: dequantized weights (ggml `to_float`), the card's quantized activation,
/// the kernels' SwiGLU, the card's quantized intermediate, F32 matmuls accumulated in double so the reference is
/// not the error term.
void ref_expert_gpu(const cpu::NativeFmt& f, const uint8_t* blob, const float* x, float* out, Probe* probe) {
    const int64_t H = f.n_embd, FF = f.n_ff;
    std::vector<float> xq((size_t) H), g((size_t) FF), u((size_t) FF), h((size_t) FF), hq((size_t) FF);
    deq_q8_1(x, H, xq.data());
    RefWeights w;
    dequant_weights(blob, f, w);
    matvec(w.G.data(), FF, H, xq.data(), g.data());
    matvec(w.U.data(), FF, H, xq.data(), u.data());
    if (probe) {
        for (int64_t j = 0; j < FF; ++j) {
            probe->gate_abs = std::max(probe->gate_abs, std::fabs(g[(size_t) j]));
            probe->up_abs = std::max(probe->up_abs, std::fabs(u[(size_t) j]));
        }
    }
    swiglu_ref(g.data(), u.data(), h.data(), FF, f.swiglu_limit, probe);
    deq_q8_1(h.data(), FF, hq.data());
    matvec(w.D.data(), H, FF, hq.data(), out);
}

/// The tier in GPU mode on 24 REAL expert slices in the arena: each of the four sources a routed expert can come
/// from (a resident VRAM slot, the prefetch staging, the PCIe DMA from the pinned arena, and the blob source
/// itself) against the same dequant+F32 reference, and then all four at once in one layer's weighted sum.
constexpr int kArena = 24;   ///< real expert slices the arena holds (24 of 6.75 MiB = 162 MiB)
int arm_gpu(const std::string& gguf, int sets, float lim_override = 0.0f) {
    std::printf("\n-- arm 3: the GPU's four sources (VRAM slot / prefetch staging / PCIe DMA / file tier) vs the "
                "same dequant+F32 reference\n");
    std::string err;
    Ds4MoeGeom g;
    if (!strata::ds4::ds4_moe_geom_from_gguf(gguf, g, err)) {
        std::printf("gpu: %s\n", err.c_str());
        return 1;
    }
    if (lim_override > 0.0f) {
        g.swiglu_limit = lim_override;
        std::printf("   (arm 3 again with the SwiGLU clamp forced to %.2f)\n", lim_override);
    }
    cpu::NativeFmt f;
    if (!strata::ds4::ds4_moe_blob_layout(g, f, err)) {
        std::printf("gpu: %s\n", err.c_str());
        return 1;
    }
    if (f.n_embd % 32 || f.n_ff % 32) {
        std::printf("gpu: q8_1 needs both widths to be multiples of 32\n");
        return 1;
    }
    FILE* probe = std::fopen(gguf.c_str(), "rb");
    if (!probe) {
        std::printf("   SKIPPED (%s not readable): the GPU arm needs real blobs\n", gguf.c_str());
        return 1;
    }
    std::fclose(probe);

    // 24 real slices of ONE layer through the GGUF's mmap (6.75 MiB each, 162 MiB in total: a slice of a tensor,
    // NOT a model load).  Layer 0 because the arena's index-order default IS this list, so nothing is read twice -
    // the tier does not care which layer's experts it is handed.
    const int64_t L = 0;
    std::vector<std::pair<int32_t, int32_t>> ranked;
    for (int e = 0; e < kArena; ++e) ranked.push_back({(int32_t) L, (int32_t) e});

    Ds4GgufBlobs blobs(g, gguf);
    if (!blobs.open(err)) {
        std::printf("gpu: %s\n", err.c_str());
        return 1;
    }
    Ds4MoeConfig cfg;
    cfg.slots = 4;         // the VRAM cache: the first four of the 24
    cfg.pf_b = 1.0;        // one staged expert per layer and no dither, so a group's size is exact
    cfg.dither = false;
    cfg.pcie_frac = 1.0;   // every eligible miss goes to the card: PCIe is exercised, not sampled
    cfg.arena_gib = 0.16;  // 24 slots of 6.75 MiB: the arena holds exactly the list above
    cfg.max_arena_gib = 0.16;
    cfg.mem_floor_gib = 0.5;
    Ds4MoeTier tier;
    if (!tier.init(g, &blobs, cfg, err)) {
        std::printf("gpu: init: %s\n", err.c_str());
        return 1;
    }
    if (!tier.seed_from_ranked(ranked, err)) {
        std::printf("gpu: seed: %s\n", err.c_str());
        return 1;
    }
    std::printf("   tier %s: arena %lld experts (%.3f GiB), cache %lld of %lld slots, file tier %lld, pool %d "
                "workers\n", tier.mode(), (long long) tier.arena_experts(), tier.arena_gib(),
                (long long) tier.resident(), (long long) cfg.slots, (long long) tier.file_tier(),
                tier.pool().workers());
    if (tier.resident() < cfg.slots || tier.arena_experts() < 24) {
        std::printf("  gpu: the tier did not take the arena/cache this arm needs (resident %lld, arena %lld)\n",
                    (long long) tier.resident(), (long long) tier.arena_experts());
        ++g_fail;
        ++g_checks;
        return 1;
    }

    const int64_t H = f.n_embd;
    const int64_t K = g.top_k;
    std::mt19937 rng(20261006);
    std::normal_distribution<float> xn(0.f, 1.0f);
    std::vector<float> x((size_t) H), nw((size_t) H), out((size_t) H), ref((size_t) H);
    read_ffn_norm(gguf, L, H, nw);
    for (int64_t i = 0; i < H; ++i) x[(size_t) i] = nw[(size_t) i] * xn(rng);
    std::printf("   L%lld: ffn_norm.weight RMS %.4f (the activation's scale)\n", (long long) L, rms_of(nw));

    // One-hot weights over K repeats of one id, so a single group serves the whole layer and the count the stats
    // report IS that group's size: a hit is the VRAM cache, prefetch staging is a miss the prefetch covered, the
    // PCIe group is a miss DMA'd out of the pinned arena, and an expert outside the arena has to be read by the
    // blob source (the file tier) and computed by the pool.  `cpu_ref` picks the activation the tier used for that
    // source - the pool's Q8_K or the card's q8_1 - because the tier really does use both (see ds4_moe.hpp): the
    // reference has to speak the format of the path it is checking, or it measures the format difference instead.
    std::vector<uint8_t> blob((size_t) blobs.blob_bytes());
    auto one = [&](int32_t e, const char* what, bool pf, bool cpu_ref, int64_t h, int64_t p, int64_t pc, int64_t c,
                   int64_t ft) {
        std::vector<int32_t> ids((size_t) K, e);
        std::vector<float> w6((size_t) K, 0.0f);
        w6[0] = 1.0f;
        if (pf) {
            std::vector<int32_t> p16(cfg.kPredW, e);
            tier.prefetch(L, p16.data());
        }
        tier.reset_stats();
        if (!tier.run(L, ids.data(), w6.data(), x.data(), out.data())) {
            std::printf("  %s: tier.run refused\n", what);
            ++g_fail;
            ++g_checks;
            return;
        }
        blobs.read(L, e, blob.data());
        if (cpu_ref) ref_expert(f, blob.data(), x.data(), ref.data(), &probe_total, true);
        else ref_expert_gpu(f, blob.data(), x.data(), ref.data(), &probe_total);
        const double r = rel(out.data(), ref.data(), H);
        if (std::isfinite(f.swiglu_limit)) {
            // how far the UNCLAMPED answer is from the clamped reference: a missing or wrong clamp in the kernels
            // would put the tier about this far off, so `r` far below it shows the clamp is applied
            cpu::NativeFmt fu = f;
            fu.swiglu_limit = std::numeric_limits<float>::infinity();
            std::vector<float> ref_u((size_t) H);
            if (cpu_ref) ref_expert(fu, blob.data(), x.data(), ref_u.data(), nullptr, true);
            else ref_expert_gpu(fu, blob.data(), x.data(), ref_u.data(), nullptr);
            std::printf("  %-20s       unclamped answer vs clamped reference: rel %.3e (tier: %.3e)\n", "", rel(ref_u.data(), ref.data(), H), r);
        }
        const Ds4MoeStats& st = tier.stats();
        const bool ok = r < kLimit && st.hits == h && st.prefetched_useful == p && st.pcie == pc && st.cpu == c &&
                        st.file_tier == ft;
        std::printf("  %-20s e%03d: hits %lld pf %lld pcie %lld cpu %lld file %lld (want %lld/%lld/%lld/%lld/%lld)"
                    "  rel %.3e  %s\n", what, e, (long long) st.hits, (long long) st.prefetched_useful,
                    (long long) st.pcie, (long long) st.cpu, (long long) st.file_tier, (long long) h, (long long) p,
                    (long long) pc, (long long) c, (long long) ft, r, ok ? "ok" : "FAIL");
        ++g_checks;
        if (!ok) ++g_fail;
    };
    one(1, "hit (VRAM slot)", false, false, K, 0, 0, 0, 0);
    one(5, "prefetch staging", true, false, K, K, 0, 0, 0);
    one(9, "PCIe DMA", false, false, 0, 0, K, 0, 0);
    one(200, "file tier (CPU)", false, true, 0, 0, 0, K, K);

    // The mixed arm: all four sources in one layer, router weights, the real weighted sum - and the same six
    // experts through run_dev (x on the device, the engine's own call shape) on a second activation.
    std::vector<uint8_t> blob2((size_t) blobs.blob_bytes());
    std::vector<float> ref_sum((size_t) H), w6((size_t) K);
    void* d_x = nullptr;
    if (cudaMalloc(&d_x, (size_t) H * 4) != cudaSuccess) d_x = nullptr;
    const int32_t ids_a[6] = {1, 5, 9, 200, 2, 10};
    const int32_t ids_b[6] = {2, 6, 10, 201, 3, 11};
    for (int s = 0; s < std::min(sets, 2); ++s) {
        const int32_t* ids = s == 0 ? ids_a : ids_b;
        std::vector<int32_t> p16(cfg.kPredW, 30 + s);
        p16[0] = ids[1];   // pf-b 1.0 stages exactly one expert: the set's second id
        for (int64_t i = 0; i < H; ++i) x[(size_t) i] = nw[(size_t) i] * xn(rng);
        router_weights(rng, K, w6.data());
        // expected: two resident ids, one prefetched, two DMA'd (both arena-resident misses), one file-tier miss
        const int64_t want_h = 3, want_pf = 1, want_pc = 2, want_c = 1, want_ft = 1;
        for (int dev = 0; dev < (d_x ? 2 : 1); ++dev) {
            bool ran;
            if (dev) {
                if (cudaMemcpy(d_x, x.data(), (size_t) H * 4, cudaMemcpyHostToDevice) != cudaSuccess) break;
                tier.prefetch(L, p16.data());
                tier.reset_stats();
                ran = tier.run_dev(L, ids, w6.data(), d_x, out.data());
            } else {
                tier.prefetch(L, p16.data());
                tier.reset_stats();
                ran = tier.run(L, ids, w6.data(), x.data(), out.data());
            }
            if (!ran) {
                std::printf("  gpu: %s refused\n", dev ? "run_dev" : "run");
                ++g_fail;
                ++g_checks;
                continue;
            }
            std::fill(ref_sum.begin(), ref_sum.end(), 0.0f);
            for (int64_t k = 0; k < K; ++k) {
                blobs.read(L, ids[k], blob2.data());
                // an id outside the arena is the file tier: the pool computed it with Q8_K, the rest on the card
                if (ids[k] >= kArena) ref_expert(f, blob2.data(), x.data(), ref.data(), nullptr, true);
                else ref_expert_gpu(f, blob2.data(), x.data(), ref.data(), nullptr);
                for (int64_t i = 0; i < H; ++i) ref_sum[(size_t) i] += w6[(size_t) k] * ref[(size_t) i];
            }
            const double r = rel(out.data(), ref_sum.data(), H);
            const Ds4MoeStats& st = tier.stats();
            const bool ok = r < kLimit && st.hits == want_h && st.prefetched_useful == want_pf &&
                            st.pcie == want_pc && st.cpu == want_c && st.file_tier == want_ft;
            std::printf("  set %d %-15s: hits %lld pf %lld pcie %lld cpu %lld file %lld (want %lld/%lld/%lld/%lld/"
                        "%lld)  rel %.3e  %s\n", s, dev ? "run_dev" : "run (host x)", (long long) st.hits,
                        (long long) st.prefetched_useful, (long long) st.pcie, (long long) st.cpu,
                        (long long) st.file_tier, (long long) want_h, (long long) want_pf, (long long) want_pc,
                        (long long) want_c, (long long) want_ft, r, ok ? "ok" : "FAIL");
            ++g_checks;
            if (!ok) ++g_fail;
        }
    }
    if (d_x) cudaFree(d_x);
    tier.close();
    return 0;
}

// ================================ arm 4: the real model's timing ================================
//
// The tier's own run() on the real GGUF, the real routes and the real prediction file - the replay's measured arm
// (FINDINGS s12) driven through the library instead of the harness.  This is a FULL-MODEL run: the caller wraps it
// in `tools/ds4/memguard.sh 64 62 -- ...` (see the packet's safety rules), and `--gap-ms` puts the layer's dense
// work between prefetch and run as a sleep - the replay's `--gap-wall` stand-in on the host - so the wall has the
// same shape as s12's serial chain and the comparable number is the replay's own implied full-token.

constexpr int64_t kUbatch = 512;   ///< a `routes.bin` ubatch, in tokens (what route_probe dumps)
constexpr double kNonMoeMs = 17.6; ///< Phase 0's non-MoE GPU time, ms/token (the replay adds it to the MoE wall)

/// `routes.bin`: [512-token ubatch][layer][token][top_k] (layer, expert) u16 pairs (moe_replay's layout).
struct RouteFile {
    std::vector<uint16_t> all;
    int64_t n_tokens = 0, n_layers = 0, top_k = 0;
    /// The `top_k` (layer, expert) pairs of one token's one layer.
    const uint16_t* rec(int64_t t, int64_t l) const {
        const int64_t per_batch = kUbatch * n_layers * top_k;   // pairs in one ubatch
        const int64_t i = (t / kUbatch) * per_batch + l * (kUbatch * top_k) + (t % kUbatch) * top_k;
        return all.data() + (size_t) i * 2;
    }
};

/// Reads the u16 prediction table `[n_tokens][n_layers][w]` (sim_prefetch.py --export).
bool load_pred(const std::string& path, int64_t n_tokens, int64_t n_layers, int64_t w, std::vector<uint16_t>& out,
               std::string& err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        err = "cannot open " + path;
        return false;
    }
    const int64_t bytes = (int64_t) in.tellg();
    in.seekg(0);
    out.assign((size_t) (bytes / 2), 0);
    in.read((char*) out.data(), bytes);
    if (!in) {
        err = "short read on " + path;
        return false;
    }
    if ((int64_t) out.size() != n_tokens * n_layers * w) {
        err = std::to_string(out.size()) + " u16 entries, the routes have " + std::to_string(n_tokens) +
              " tokens x " + std::to_string(n_layers) + " layers x " + std::to_string(w);
        return false;
    }
    return true;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int arm_timing(const std::string& gguf, const Args& a) {
    std::printf("== ds4_moe tier timing: the real model through Ds4MoeTier ==\n");
    std::string err;
    Ds4MoeGeom g;
    if (!strata::ds4::ds4_moe_geom_from_gguf(gguf, g, err)) {
        std::printf("timing: %s\n", err.c_str());
        return 1;
    }
    cpu::NativeFmt f;
    if (!strata::ds4::ds4_moe_blob_layout(g, f, err)) {
        std::printf("timing: %s\n", err.c_str());
        return 1;
    }
    RouteFile r;
    {
        std::ifstream in(a.routes, std::ios::binary | std::ios::ate);
        if (!in) {
            std::printf("timing: cannot open %s\n", a.routes.c_str());
            return 1;
        }
        const int64_t bytes = (int64_t) in.tellg();
        in.seekg(0);
        r.all.assign((size_t) (bytes / 2), 0);
        in.read((char*) r.all.data(), bytes);
        if (!in) {
            std::printf("timing: short read on %s\n", a.routes.c_str());
            return 1;
        }
    }
    r.n_layers = g.n_layers;
    r.top_k = g.top_k;
    const int64_t per_tok = r.n_layers * r.top_k;   // pairs per token
    if (r.all.empty() || (int64_t) r.all.size() / 2 % per_tok != 0) {
        std::printf("timing: %s is not a whole number of tokens\n", a.routes.c_str());
        return 1;
    }
    r.n_tokens = (int64_t) r.all.size() / 2 / per_tok;
    if (a.offset < 0 || a.tokens <= 0 || a.offset + a.tokens > r.n_tokens) {
        std::printf("timing: token range [%lld, %lld) is outside 0..%lld\n", (long long) a.offset,
                    (long long) (a.offset + a.tokens), (long long) r.n_tokens);
        return 1;
    }
    std::vector<uint16_t> pred;
    if (!a.pred.empty() && !load_pred(a.pred, r.n_tokens, g.n_layers, Ds4MoeConfig::kPredW, pred, err)) {
        std::printf("timing: pred: %s\n", err.c_str());
        return 1;
    }

    Ds4GgufBlobs blobs(g, gguf);
    if (!blobs.open(err)) {
        std::printf("timing: %s\n", err.c_str());
        return 1;
    }
    Ds4MoeConfig cfg;
    cfg.slots = a.slots;
    cfg.pf_b = a.pf_b;
    cfg.pcie_frac = a.pcie_frac;
    cfg.threads = a.threads;
    cfg.arena_gib = a.arena_gib;
    cfg.max_arena_gib = a.arena_gib;
    cfg.pin = true;
    Ds4MoeTier tier;
    if (!tier.init(g, &blobs, cfg, err)) {
        std::printf("timing: init: %s\n", err.c_str());
        return 1;
    }
    const double t_seed = now_ms();
    if (!tier.seed_from_routes(a.profile.empty() ? a.routes : a.profile, err)) {
        std::printf("timing: seed: %s\n", err.c_str());
        return 1;
    }
    std::printf("seeded in %.1f s: arena %lld experts (%.1f GiB), file tier %lld, cache %lld of %lld slots, "
                "pool %d workers\n", (now_ms() - t_seed) / 1000.0, (long long) tier.arena_experts(),
                tier.arena_gib(), (long long) tier.file_tier(), (long long) tier.resident(), (long long) a.slots,
                tier.pool().workers());
    std::printf("routes %s: %lld tokens, replaying [%lld, %lld); pred %s\n", a.routes.c_str(),
                (long long) r.n_tokens, (long long) a.offset, (long long) (a.offset + a.tokens),
                a.pred.empty() ? "--" : a.pred.c_str());

    std::mt19937 rng(20261006);
    std::normal_distribution<float> nd(0.f, 1.f);
    const double inv = 1.0 / std::sqrt((double) f.n_embd);
    const int64_t H = f.n_embd, K = g.top_k;
    std::vector<float> x((size_t) g.n_layers * (size_t) H), out((size_t) H), w6((size_t) K, 1.0f / (float) K);
    std::vector<int32_t> ids((size_t) K), p16(Ds4MoeConfig::kPredW);
    const double t0 = now_ms();
    for (int64_t t = a.offset; t < a.offset + a.tokens; ++t) {
        for (auto& v : x) v = (float) (nd(rng) * inv);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const uint16_t* rec = r.rec(t, l);
            for (int64_t k = 0; k < K; ++k) ids[(size_t) k] = (int32_t) rec[k * 2 + 1];
            // The engine calls this right after the router's prediction and before the layer's dense work; the
            // predictions are the probe dump's, ranked (tools/ds4/predict_experts.py).
            if (!pred.empty()) {
                const uint16_t* pr = pred.data() + (size_t) ((t * g.n_layers + l) * Ds4MoeConfig::kPredW);
                for (int i = 0; i < Ds4MoeConfig::kPredW; ++i) p16[(size_t) i] = (int32_t) pr[i];
                tier.prefetch(l, p16.data());
            }
            if (a.gap_ms > 0)
                std::this_thread::sleep_for(std::chrono::microseconds((int64_t) (a.gap_ms * 1000.0)));
            if (!tier.run(l, ids.data(), w6.data(), x.data() + (size_t) l * H, out.data())) {
                std::printf("timing: run refused at token %lld layer %lld\n", (long long) t, (long long) l);
                return 1;
            }
        }
    }
    const double wall = now_ms() - t0;
    const Ds4MoeStats& st = tier.stats();
    const int64_t lookups = a.tokens * g.n_layers * K;
    const double tok_ms = wall / (double) a.tokens;
    const double gap_ms = a.gap_ms * (double) g.n_layers;
    std::printf("\n%d tokens x %lld layers, gap %.2f ms/layer (%.2f ms/token)\n", (int) a.tokens,
                (long long) g.n_layers, a.gap_ms, gap_ms);
    std::printf("lookups/token %lld: hit %.1f%%, miss-CPU %.1f%%, miss-PCIe %.1f%%\n", (long long) (g.n_layers * K),
                100.0 * (double) st.hits / (double) lookups, 100.0 * (double) st.cpu / (double) lookups,
                100.0 * (double) st.pcie / (double) lookups);
    std::printf("prefetch: issued %.2f/layer, useful %.2f/layer (%.1f%%)\n",
                (double) st.prefetch_issued / (double) (a.tokens * g.n_layers),
                (double) st.prefetched_useful / (double) (a.tokens * g.n_layers),
                100.0 * (double) st.prefetched_useful / (double) std::max<int64_t>(1, st.prefetch_issued));
    std::printf("ms/token: MoE wall %.2f = run %.2f + gap %.2f; gpu hit %.2f + pcie %.2f, cpu busy %.2f\n", tok_ms,
                st.wall_ms / (double) a.tokens, gap_ms, st.hit_ms / (double) a.tokens,
                st.pcie_ms / (double) a.tokens, st.cpu_ms / (double) a.tokens);
    std::printf("implied full-token = MoE %.2f + non-MoE GPU %.2f = %.2f ms -> %.2f tok/s (FINDINGS s12 arm: "
                "20.57)\n", tok_ms, kNonMoeMs, tok_ms + kNonMoeMs, 1000.0 / (tok_ms + kNonMoeMs));
    tier.close();
    return 0;
}

#endif  // DS4_MOE_CUDA

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        const bool has = i + 1 < argc;
        if (k == "--gguf" && has) a.gguf = argv[++i];
        else if (k == "--require-gguf") a.require_gguf = true;
        else if (k == "--no-synthetic") a.no_synthetic = true;
        else if (k == "--no-real") a.no_real = true;
        else if (k == "--sets" && has) a.sets = std::atoi(argv[++i]);
        else if (k == "--gpu") a.gpu = true;
        else if (k == "--timing") a.timing = true;
        else if (k == "--routes" && has) a.routes = argv[++i];
        else if (k == "--pred" && has) a.pred = argv[++i];
        else if (k == "--profile" && has) a.profile = argv[++i];
        else if (k == "--tokens" && has) a.tokens = std::atoll(argv[++i]);
        else if (k == "--offset" && has) a.offset = std::atoll(argv[++i]);
        else if (k == "--slots" && has) a.slots = std::atoll(argv[++i]);
        else if (k == "--threads" && has) a.threads = std::atoi(argv[++i]);
        else if (k == "--arena-gib" && has) a.arena_gib = std::atof(argv[++i]);
        else if (k == "--pcie-frac" && has) a.pcie_frac = std::atof(argv[++i]);
        else if (k == "--pf-b" && has) a.pf_b = std::atof(argv[++i]);
        else if (k == "--gap-ms" && has) a.gap_ms = std::atof(argv[++i]);
        else {
            std::printf(
                "usage: test_ds4_moe [--gguf PATH] [--require-gguf] [--sets N] [--no-synthetic] [--no-real]\n"
                "       test_ds4_moe_gpu [--gpu] [same]                    (the CUDA flavour: build-ds4-cuda)\n"
                "       test_ds4_moe_gpu --timing --routes R [--pred P] [--profile Q] [--tokens N] [--offset O]\n"
                "                        [--slots S] [--arena-gib G] [--pcie-frac F] [--pf-b B] [--gap-ms X]\n"
                "                        [--threads T]    (full-model run: wrap in tools/ds4/memguard.sh)\n");
            return 2;
        }
    }
    if (a.sets < 1) a.sets = 1;

#if !defined(DS4_MOE_CUDA)
    if (a.gpu || a.timing) {
        std::printf("test_ds4_moe: --gpu/--timing are not built into this binary.  It links ds4_moe_cpu, which\n"
                    "  is ds4_moe.cpp compiled without DS4_MOE_CUDA: no CUDA call in it, no CUDA library on its\n"
                    "  link line, so it cannot initialise the driver - that property is what makes the CPU gate\n"
                    "  safe to run on a busy machine.  The CUDA flavour is build-ds4-cuda/test_ds4_moe_gpu\n"
                    "  (the same source, compiled with DS4_MOE_CUDA).\n");
        return 2;
    }
#endif

#if defined(DS4_MOE_CUDA)
    if (a.timing) {
        if (a.routes.empty()) {
            std::printf("--timing needs --routes FILE\n");
            return 2;
        }
        return arm_timing(a.gguf, a);
    }
#endif

    std::printf("== test_ds4_moe: the routed-expert tier vs dequant+F32 (limit %.1e), %s flavour ==\n", kLimit,
#if defined(DS4_MOE_CUDA)
                "CUDA"
#else
                "CPU-only"
#endif
    );
    int rc = 0;
    if (!a.no_synthetic) rc |= arm_synthetic(a.sets);
    if (!a.no_real) rc |= arm_real(a.gguf, a.sets, a.require_gguf);
#if defined(DS4_MOE_CUDA)
    if (a.gpu) rc |= arm_gpu(a.gguf, a.sets);
#endif
    const Probe at_model_limit = probe_total;
    // the clamp path itself: the same real arms with a limit that binds on a large share of the elements
    probe_total = Probe{};
    if (!a.no_real) rc |= arm_real(a.gguf, a.sets, a.require_gguf, kForcedClamp);
#if defined(DS4_MOE_CUDA)
    if (a.gpu) rc |= arm_gpu(a.gguf, a.sets, kForcedClamp);
#endif
    if (!a.no_real || a.gpu) {
        const double frac = probe_total.total ? (double) probe_total.bound / (double) probe_total.total : 0.0;
        std::printf("\nforced clamp %.2f: the reference clamped %lld of %lld pre-activations (%.2f%%)\n", kForcedClamp,
                    (long long) probe_total.bound, (long long) probe_total.total, 100.0 * frac);
        check(frac > 0.05, "forced clamp actually binds (share of clamped pre-activations > 5%)", frac);
    }
    probe_total = at_model_limit;

    std::printf("\nclamp probe (max pre-activation over every expert sampled above; the model's "
                "swiglu_clamp_exp is %.1f)\n  max |gate| %.4f, max |up| %.4f -> %s\n", kClampExp,
                probe_total.gate_abs, probe_total.up_abs,
                (probe_total.gate_abs < kClampExp && probe_total.up_abs < kClampExp)
                    ? "the clamp is inactive on these activations"
                    : "the clamp binds on these activations (the kernels apply it, as the reference does)");

    std::printf("\ntest_ds4_moe: %d checks, %d failures, %d arm errors\n", g_checks, g_fail, rc);
    return (g_fail || rc) ? 1 : 0;
}
