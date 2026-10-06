// tools/ds4/test_ds4_dense.cpp - the ds4-dense gate.
//
// INVARIANT: token-by-token decode through Ds4Dense (+ a plain reference routed-expert computation, ggml
// mul_mat_id, right here in the test) reproduces ds4_ref's full-sequence forward on the same tokens.
//
//   test_ds4_dense <mini.gguf> [--tokens N] [--work DIR] [--ref-dir DIR] [--ref-bin PATH] [--cuda]
//
// It writes a deterministic token file, runs `ds4_ref --all-pos` into <work>/ref-allpos, decodes the same
// tokens one at a time, and checks:
//   * at the LAST position, per layer: cosine >= 0.99999 on the pre-LoRA attention output (attn_raw /
//     attn_csa_lid / attn_hca), on attn_out, on ffn_norm and on l_last (hc streams)
//   * logits: top-1 identical at the last position and cosine >= 0.99999
//   * per-token logits cosine for ALL positions, all >= 0.99999
//
// The decoder runs on the CPU backend by default; --cuda puts Ds4Dense (and the test-side expert reference) on
// ggml's GPU backend (needs a tree built with -DSTRATA_GGML_CUDA=ON, i.e. build-ds4-gpu). ds4_ref, the oracle,
// always runs on the CPU.

#include "ds4_dense.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr double COS_GATE = 0.99999;

// ------------------------------------------------------------------ small helpers

static std::vector<float> read_f32(const std::string & path, size_t expect) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    if (n < 0 || (size_t) n != expect * 4) return {};
    std::vector<float> v(expect);
    in.read((char *) v.data(), n);
    return v;
}

static double cosine(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    if (na == 0 && nb == 0) return 1.0;
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
}

// ------------------------------------------------------------------ the test-side routed-expert reference

struct MoeRef {
    ggml_context *  ctx  = nullptr;
    ggml_backend_t  be   = nullptr;
    ggml_gallocr_t  allo = nullptr;
    ggml_cgraph *   gf   = nullptr;
    ggml_tensor *   t_fn = nullptr;
    ggml_tensor *   t_id = nullptr;
    ggml_tensor *   t_w  = nullptr;
    ggml_tensor *   out  = nullptr;
    int64_t D = 0, topk = 0;
    std::vector<float> host;

    bool init(Ds4Dense & d, int il) {
        const strata::Ds4Geometry & g = d.geom();
        D = g.n_embd;
        topk = g.n_expert_used;
        be = d.backend();

        ggml_init_params ip = { /*mem_size*/ 64ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        ggml_set_no_alloc(ctx, true);
        allo = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!allo) return false;

        t_fn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);       ggml_set_input(t_fn);
        t_id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, 1);    ggml_set_input(t_id);
        t_w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, 1); ggml_set_input(t_w);

        const std::string p = "blk." + std::to_string(il) + ".";
        ggml_tensor * cur3 = ggml_reshape_3d(ctx, t_fn, D, 1, 1);
        ggml_tensor * up   = ggml_mul_mat_id(ctx, d.weight(p + "ffn_up_exps.weight"), cur3, t_id);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, d.weight(p + "ffn_gate_exps.weight"), cur3, t_id);
        const float lim = (float) g.swiglu_clamp_exp[(size_t) il];
        up   = ggml_clamp(ctx, up, -lim, lim);
        gate = ggml_clamp(ctx, gate, -INFINITY, lim);
        ggml_tensor * h    = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat_id(ctx, d.weight(p + "ffn_down_exps.weight"), h, t_id);
        down = ggml_mul(ctx, down, t_w);

        ggml_tensor * acc = nullptr;
        for (int64_t k = 0; k < topk; ++k) {
            ggml_tensor * v = ggml_view_2d(ctx, down, D, 1, down->nb[2], (size_t) k * down->nb[1]);
            acc = acc ? ggml_add(ctx, acc, v) : v;
        }
        out = acc;
        ggml_set_output(out);
        gf = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(gf, out);
        host.resize((size_t) D);
        return true;
    }

    bool run(const float * fn, const int * ids, const float * w, const float ** sum) {
        if (!ggml_gallocr_alloc_graph(allo, gf)) return false;
        ggml_backend_tensor_set(t_fn, fn, 0, (size_t) D * 4);
        std::vector<int32_t> iv((size_t) topk);
        for (int64_t i = 0; i < topk; ++i) iv[(size_t) i] = ids[i];
        ggml_backend_tensor_set(t_id, iv.data(), 0, iv.size() * 4);
        ggml_backend_tensor_set(t_w, w, 0, (size_t) topk * 4);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(out, host.data(), 0, host.size() * 4);
        *sum = host.data();
        return true;
    }
};

// ------------------------------------------------------------------ main

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep the log order under a crash / when piped
    setvbuf(stderr, nullptr, _IONBF, 0);
    std::string model, work = "bench/ds4-2026-10-06/dense", refbin, refdir;
    int nt_tokens = 0;
    bool use_cuda = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--tokens")   nt_tokens = std::atoi(next().c_str());
        else if (a == "--work")     work = next();
        else if (a == "--ref-dir")  refdir = next();
        else if (a == "--ref-bin")  refbin = next();
        else if (a == "--cuda")     use_cuda = true;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: test_ds4_dense <mini.gguf> [--tokens N] [--work DIR] [--ref-dir DIR] [--ref-bin PATH] [--cuda]\n");
            return 0;
        }
        else if (a[0] != '-' && model.empty()) model = a;   // positional: test_ds4_dense <mini.gguf>
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: test_ds4_dense <mini.gguf> [--tokens N] [--work DIR] [--ref-dir DIR] [--ref-bin PATH] [--cuda]\n");
        return 1;
    }
    if (refbin.empty()) {
        const fs::path self = fs::absolute(argv[0]);
        refbin = (self.parent_path() / "ds4_ref").string();
    }

    std::string err;
    Ds4DenseConfig cfg;
    ggml_backend_t gpu = nullptr;
    if (use_cuda) {
        gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!gpu) { std::fprintf(stderr, "[gate] --cuda: no GPU backend in this build (use build-ds4-gpu)\n"); return 2; }
    }
    cfg.backend   = gpu;       // nullptr = CPU
    cfg.n_threads = 8;
    cfg.gate_taps = true;
    std::printf("[gate] decoder backend: %s\n", gpu ? ggml_backend_name(gpu) : "CPU");

    Ds4Dense d;
    if (!d.init(model, cfg, err)) { std::fprintf(stderr, "[gate] init failed: %s\n", err.c_str()); return 1; }

    const strata::Ds4Geometry & g = d.geom();
    const int64_t D = g.n_embd, HC = g.hc, NH = g.n_head, DH = g.d_head, V = g.vocab_size;
    const int64_t SWA = g.n_swa;
    const int64_t nl  = d.n_layer();

    // the token count must exercise every path the packet names, for THIS geometry
    int64_t csa_ratio = 0, hca_ratio = 0;
    for (int64_t r : g.compress_ratios) {
        if (r == 4)   csa_ratio = 4;
        if (r == 128) hca_ratio = 128;
    }
    int64_t need = SWA + 1;                                                        // raw window wrap
    if (csa_ratio) need = std::max<int64_t>(need, csa_ratio + 1);                    // >= 1 CSA block
    if (hca_ratio) need = std::max<int64_t>(need, hca_ratio + 1);                    // >= 1 HCA block
    if (csa_ratio && g.indexer_top_k) need = std::max<int64_t>(need, csa_ratio * (g.indexer_top_k + 1) + 1);
    need = std::max<int64_t>(need, 160);                                             // comfortable margin
    const int64_t nt = nt_tokens > 0 ? nt_tokens : need;

    const int64_t n_vis_last_csa = csa_ratio ? (nt / csa_ratio) : 0;
    const int64_t n_vis_last_hca = hca_ratio ? (nt / hca_ratio) : 0;
    std::printf("[gate] model=%s D=%lld hc=%lld heads=%lld d_head=%lld V=%lld swa=%lld layers=%lld\n",
                model.c_str(), (long long) D, (long long) HC, (long long) NH, (long long) DH,
                (long long) V, (long long) SWA, (long long) nl);
    std::printf("[gate] tokens=%lld  paths: raw wrap=%s  csa blocks@last=%lld  hca blocks@last=%lld  "
                "indexer top_k=%lld\n",
                (long long) nt, nt > SWA ? "yes" : "NO", (long long) n_vis_last_csa, (long long) n_vis_last_hca,
                (long long) g.indexer_top_k);
    if (nt <= SWA || (hca_ratio && n_vis_last_hca < 1) || (csa_ratio && g.indexer_top_k && n_vis_last_csa <= g.indexer_top_k))
        std::fprintf(stderr, "[gate] WARNING: the chosen token count does not exercise every path\n");

    // ---- token file -------------------------------------------------------------------
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) { std::fprintf(stderr, "[gate] cannot create %s\n", work.c_str()); return 1; }
    const std::string tokfile = work + "/tokens.i32";
    std::vector<int32_t> tokens((size_t) nt);
    for (int64_t i = 0; i < nt; ++i) tokens[(size_t) i] = (int32_t) ((i * 37 + 11) % V);
    {
        std::ofstream out(tokfile, std::ios::binary);
        out.write((const char *) tokens.data(), (std::streamoff) tokens.size() * 4);
        if (!out) { std::fprintf(stderr, "[gate] cannot write %s\n", tokfile.c_str()); return 1; }
    }

    // ---- reference: ds4_ref over the whole sequence ------------------------------------
    if (refdir.empty()) refdir = work + "/ref-allpos";
    if (!fs::exists(refdir + "/manifest.json")) {
        if (!fs::exists(refbin)) {
            std::fprintf(stderr, "[gate] ds4_ref not found at %s (pass --ref-bin)\n", refbin.c_str());
            return 1;
        }
        const std::string cmd = "\"" + refbin + "\" -m \"" + model + "\" --tokens \"" + tokfile +
                                "\" --out \"" + refdir + "\" --all-pos -t 8 2>&1 | tee \"" + work + "/ref.log\"";
        std::printf("[gate] %s\n", cmd.c_str());
        const int rc = std::system(cmd.c_str());
        if (rc != 0) { std::fprintf(stderr, "[gate] ds4_ref failed (rc=%d)\n", rc); return 1; }
    } else {
        std::printf("[gate] reusing reference dump %s\n", refdir.c_str());
    }

    std::vector<std::vector<float>> ref_tensor;
    ref_tensor.reserve(64);
    auto load_ref = [&](const std::string & name, size_t per_tok) -> const float * {
        std::vector<float> v = read_f32(refdir + "/" + name + ".f32", per_tok * (size_t) nt);
        if (v.empty()) {
            std::fprintf(stderr, "[gate] missing/short reference tensor %s (want %zu floats)\n",
                         name.c_str(), per_tok * (size_t) nt);
            std::exit(1);
        }
        ref_tensor.push_back(std::move(v));
        return ref_tensor.back().data();
    };

    const float * r_logits = load_ref("result_output", (size_t) V);

    // ---- decode ------------------------------------------------------------------------
    std::vector<MoeRef> moe((size_t) nl);
    for (int64_t il = 0; il < nl; ++il)
        if (!moe[(size_t) il].init(d, (int) il)) { std::fprintf(stderr, "[gate] MoE ref init failed\n"); return 1; }

    std::vector<std::vector<float>> my_logits((size_t) nt);
    std::vector<std::vector<float>> my_raw, my_out, my_fn, my_ll;   // last position, per layer
    std::vector<int> ids((size_t) g.n_expert_used), ids2((size_t) g.n_expert_used);
    std::vector<float> wts((size_t) g.n_expert_used), w2((size_t) g.n_expert_used);
    std::vector<int64_t> pred_hits((size_t) nl, 0), pred_tot((size_t) nl, 0);

    const int64_t t_last = nt - 1;
    for (int64_t t = 0; t < nt; ++t) {
        if (!d.begin_token(tokens[(size_t) t])) { std::fprintf(stderr, "[gate] begin_token: %s\n", d.last_error().c_str()); return 1; }
        for (int64_t il = 0; il < nl; ++il) {
            int top16[16];
            if (!d.predict((int) il, top16, 16)) { std::fprintf(stderr, "[gate] predict: %s\n", d.last_error().c_str()); return 1; }

            const float * fn = nullptr;
            ggml_tensor * fn_dev = nullptr;
            if (!d.attn_router((int) il, (int) t, tokens[(size_t) t],
                               (int *) ids.data(), wts.data(), &fn, &fn_dev)) {
                std::fprintf(stderr, "[gate] attn_router(l=%lld,pos=%lld): %s\n",
                             (long long) il, (long long) t, d.last_error().c_str());
                return 1;
            }
            if (il >= (int) g.hash_layer_count) {
                int hit = 0;
                for (int64_t k = 0; k < g.n_expert_used; ++k)
                    for (int j = 0; j < 16; ++j) if (top16[j] == ids[(size_t) k]) { hit++; break; }
                pred_hits[(size_t) il] += hit;
                pred_tot[(size_t) il]  += g.n_expert_used;
            }

            const float * rsum = nullptr;
            if (!moe[(size_t) il].run(fn, (const int *) ids.data(), wts.data(), &rsum)) {
                std::fprintf(stderr, "[gate] MoE reference run failed\n");
                return 1;
            }
            if (!d.finish_layer((int) il, rsum)) { std::fprintf(stderr, "[gate] finish_layer: %s\n", d.last_error().c_str()); return 1; }

            if (t == t_last) {
                const float * a = d.tap_attn_raw((int) il);
                const float * o = d.tap_attn_out((int) il);
                const float * l = d.tap_l_last((int) il);
                if (!a || !o || !l || !fn) { std::fprintf(stderr, "[gate] missing tap at the last position\n"); return 1; }
                my_raw.push_back(std::vector<float>(a, a + NH * DH));
                my_out.push_back(std::vector<float>(o, o + D));
                my_fn.push_back(std::vector<float>(fn, fn + D));
                my_ll.push_back(std::vector<float>(l, l + D * HC));
            }
        }
        const float * lg = nullptr;
        int nv = 0;
        if (!d.logits(&lg, &nv)) { std::fprintf(stderr, "[gate] logits: %s\n", d.last_error().c_str()); return 1; }
        my_logits[(size_t) t].assign(lg, lg + nv);
    }

    // ---- comparison --------------------------------------------------------------------
    int fails = 0;
    std::printf("\n[gate] per-layer, last position %lld, cosine vs ds4_ref --all-pos\n", (long long) t_last);
    std::printf("  %-5s %-6s %-12s %-12s %-12s %-12s\n", "layer", "ratio", "attn_raw", "attn_out", "ffn_norm", "l_last");
    for (int64_t il = 0; il < nl; ++il) {
        const int64_t ratio = (size_t) il < g.compress_ratios.size() ? g.compress_ratios[(size_t) il] : 0;
        const std::string raw_name = ratio == 4 ? "attn_csa_lid" : (ratio == 128 ? "attn_hca" : "attn_raw");
        const float * r_raw = load_ref(raw_name + "-" + std::to_string(il), (size_t) (NH * DH));
        const float * r_out = load_ref("attn_out-"  + std::to_string(il), (size_t) D);
        const float * r_fn  = load_ref("ffn_norm-"  + std::to_string(il), (size_t) D);
        const float * r_ll  = load_ref("l_last-"    + std::to_string(il), (size_t) (D * HC));

        const size_t o_raw = (size_t) t_last * (size_t) (NH * DH);
        const size_t o_out = (size_t) t_last * (size_t) D;
        const size_t o_ll  = (size_t) t_last * (size_t) (D * HC);

        const double c_raw = cosine(my_raw[(size_t) il].data(), r_raw + o_raw, (size_t) (NH * DH));
        const double c_out = cosine(my_out[(size_t) il].data(), r_out + o_out, (size_t) D);
        const double c_fn  = cosine(my_fn[(size_t) il].data(),  r_fn  + o_out, (size_t) D);
        const double c_ll  = cosine(my_ll[(size_t) il].data(),  r_ll  + o_ll,  (size_t) (D * HC));

        std::printf("  %-5lld %-6lld %-12.7f %-12.7f %-12.7f %-12.7f\n",
                    (long long) il, (long long) ratio, c_raw, c_out, c_fn, c_ll);
        if (c_raw < COS_GATE) { std::printf("      ^ %s FAIL\n", raw_name.c_str()); fails++; }
        if (c_out < COS_GATE) { std::printf("      ^ attn_out FAIL\n"); fails++; }
        if (c_fn  < COS_GATE) { std::printf("      ^ ffn_norm FAIL\n"); fails++; }
        if (c_ll  < COS_GATE) { std::printf("      ^ l_last FAIL\n"); fails++; }
    }

    std::printf("\n[gate] per-token logits cosine, all %lld positions, vs ds4_ref --all-pos\n", (long long) nt);
    double worst = 2.0;
    int64_t worst_t = -1;
    int64_t n_bad = 0;
    for (int64_t t = 0; t < nt; ++t) {
        const double c = cosine(my_logits[(size_t) t].data(), r_logits + (size_t) t * (size_t) V, (size_t) V);
        if (c < worst) { worst = c; worst_t = t; }
        if (c < COS_GATE) {
            n_bad++;
            if (n_bad <= 10) std::printf("  pos %lld: cos %.8f FAIL\n", (long long) t, c);
        }
    }
    auto argmax = [](const float * v, int64_t n) { return (int) (std::max_element(v, v + n) - v); };
    const int top1_mine = argmax(my_logits[(size_t) t_last].data(), V);
    const int top1_ref  = argmax(r_logits + (size_t) t_last * (size_t) V, V);
    std::printf("  worst position %lld: cos %.8f   (gate >= %.5f), %lld positions below the gate\n",
                (long long) worst_t, worst, COS_GATE, (long long) n_bad);
    std::printf("  logits top-1 last position: Ds4Dense %d vs ds4_ref %d -> %s\n",
                top1_mine, top1_ref, top1_mine == top1_ref ? "identical" : "DIFFERENT");
    if (n_bad) fails += (int) n_bad;
    if (top1_mine != top1_ref) fails++;

    std::printf("\n[gate] predictor (prefetch hint) recall@16 on the true routed ids, non-hash layers\n");
    for (int64_t il = 0; il < nl; ++il) {
        if (il < (int) g.hash_layer_count || pred_tot[(size_t) il] == 0) continue;
        std::printf("  layer %-2lld %.3f\n", (long long) il,
                    (double) pred_hits[(size_t) il] / (double) pred_tot[(size_t) il]);
    }

    std::printf("\n[gate] %s (%d failed check(s))\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
