// tools/ds4/ds4_generate.cpp - the Strata DeepSeek-V4 decode engine end to end: Ds4Dense (attention + router +
// shared expert, on the CUDA or CPU ggml backend) + Ds4MoeTier (routed experts: VRAM cache, predicted prefetch,
// CPU pool, PCIe share) + greedy/temperature sampling.  Token ids in, token ids out (one per line on stdout,
// flushed as they are produced); text <-> ids is tools/ds4/ds4_chat.py's job (the tokenizer is Python, as in serve/).
//
//   ds4_generate -m model.gguf --ids 1,2,3 -n 64 [--backend cuda|cpu] [--experts gpu|cpu]
//                [--slots 2150] [--pcie 0.25] [--pf-b 1.43] [--profile ds4routes.bin] [--arena-gib 60]
//                [--threads 7] [--temp 0] [--seed 1] [--stop 1,2] [--dump-logits f.f32]
//
// Per token, per layer (the order FINDINGS s11/s12 measured):
//     predict(l) -> tier.prefetch(l)   the prefetch DMAs start before layer l's attention
//     attn_router(l)                   attention + router + shared expert (the DMAs overlap it)
//     tier.run(l)                      routed experts: hits + prefetched on the GPU, misses CPU/PCIe
//     finish_layer(l)                  shared + routed, hc_post
// The prompt is fed through the same one-token path (prefill = decode loop for now; Phase 5 owns batched prefill).
//
// Load the real model ONLY through tools/ds4/memguard.sh (it froze the box once without it):
//   bash tools/ds4/memguard.sh 80 82 -- build-ds4-gpu/ds4_generate -m <gguf> --ids ... -n 64
#include "ds4_dense.hpp"
#include "ds4_moe.hpp"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model, ids_csv, ids_file, profile, dump_logits;
    int n_predict = 64;
    std::string backend = "cuda";   // dense half
    std::string experts = "gpu";    // routed tier: gpu (cache + prefetch + CPU/PCIe) | cpu (pool only)
    int64_t slots = 2150;   // <= 0: auto (free VRAM after the dense half, minus --vram-margin)
    double vram_margin_gib = 0.9;
    double pcie = 0.25, pf_b = 1.43, arena_gib = 60.0;
    int threads = 0;
    int64_t ctx = 0;   // context cap for the compressed-KV state (0 = prompt + n_predict + 64)
    float temp = 0.0f;
    uint64_t seed = 1;
    std::vector<int> stop;
    bool quiet = false;
    bool ppl = false;
    float route_bias = 0.0f;
    std::string dense_requant;   // q4_k | q5_k | q6_k: requantize the big Q8_0 dense matrices at load   // cache-aware routing: + this to the selection score of VRAM-resident experts   // score every prompt position: mean NLL / perplexity of the prompt (quality check)
};

std::vector<int> parse_csv(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

void usage() {
    std::fprintf(stderr,
        "usage: ds4_generate -m model.gguf (--ids 1,2,3 | --ids-file f.i32) [-n 64] [--backend cuda|cpu]\n"
        "       [--experts gpu|cpu] [--slots 2150] [--pcie 0.25] [--pf-b 1.43] [--profile ds4routes.bin]\n"
        "       [--arena-gib 60] [--threads N] [--ctx N] [--temp 0] [--seed 1] [--stop id,id] [--dump-logits f.f32] [--quiet]\n");
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (k == "-m" || k == "--model") a.model = next();
        else if (k == "--ids") a.ids_csv = next();
        else if (k == "--ids-file") a.ids_file = next();
        else if (k == "-n" || k == "--n-predict") a.n_predict = std::atoi(next().c_str());
        else if (k == "--backend") a.backend = next();
        else if (k == "--experts") a.experts = next();
        else if (k == "--slots") { const std::string v = next(); a.slots = v == "auto" ? 0 : std::atoll(v.c_str()); }
        else if (k == "--vram-margin") a.vram_margin_gib = std::atof(next().c_str());
        else if (k == "--pcie") a.pcie = std::atof(next().c_str());
        else if (k == "--pf-b") a.pf_b = std::atof(next().c_str());
        else if (k == "--profile") a.profile = next();
        else if (k == "--arena-gib") a.arena_gib = std::atof(next().c_str());
        else if (k == "--threads") a.threads = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoll(next().c_str());
        else if (k == "--temp") a.temp = (float) std::atof(next().c_str());
        else if (k == "--seed") a.seed = (uint64_t) std::atoll(next().c_str());
        else if (k == "--stop") a.stop = parse_csv(next());
        else if (k == "--dump-logits") a.dump_logits = next();
        else if (k == "--quiet") a.quiet = true;
        else if (k == "--ppl") a.ppl = true;
        else if (k == "--dense-requant") a.dense_requant = next();
        else if (k == "--route-bias") a.route_bias = (float) std::atof(next().c_str());
        else { std::fprintf(stderr, "ds4_generate: unknown argument %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && (!a.ids_csv.empty() || !a.ids_file.empty());
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int sample(const float* logits, int n_vocab, float temp, std::mt19937_64& rng) {
    if (temp <= 0.0f) return (int) (std::max_element(logits, logits + n_vocab) - logits);
    const float mx = *std::max_element(logits, logits + n_vocab);
    std::vector<double> p((size_t) n_vocab);
    double z = 0;
    for (int i = 0; i < n_vocab; ++i) z += (p[(size_t) i] = std::exp((double) (logits[i] - mx) / temp));
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng);
    for (int i = 0; i < n_vocab; ++i) if ((r -= p[(size_t) i]) <= 0) return i;
    return n_vocab - 1;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) { usage(); return 2; }

    std::vector<int> prompt = a.ids_csv.empty() ? std::vector<int>() : parse_csv(a.ids_csv);
    if (!a.ids_file.empty()) {
        std::ifstream f(a.ids_file, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "ds4_generate: cannot open %s\n", a.ids_file.c_str()); return 2; }
        const size_t n = (size_t) f.tellg() / 4;
        f.seekg(0);
        prompt.resize(n);
        f.read((char*) prompt.data(), (std::streamsize) (n * 4));
    }
    if (prompt.empty()) { std::fprintf(stderr, "ds4_generate: empty prompt\n"); return 2; }

    // ---- dense half on the chosen backend
    ggml_backend_t be = nullptr;
    if (a.backend == "cuda") be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    else be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { std::fprintf(stderr, "ds4_generate: no %s backend in this build\n", a.backend.c_str()); return 2; }
    std::fprintf(stderr, "dense backend: %s\n", ggml_backend_name(be));

    const double t_load0 = now_ms();
    Ds4Dense dense;
    Ds4DenseConfig dc;
    dc.backend = be;
    dc.n_threads = a.threads > 0 ? a.threads : 8;
    dc.skip_routed_experts = true;
    if (!a.dense_requant.empty()) {
        const std::string q = a.dense_requant;
        dc.requant_type = q == "q4_k" ? GGML_TYPE_Q4_K : q == "q5_k" ? GGML_TYPE_Q5_K : q == "q6_k" ? GGML_TYPE_Q6_K : -1;
        if (dc.requant_type < 0) { std::fprintf(stderr, "ds4_generate: --dense-requant q4_k|q5_k|q6_k\n"); return 2; }
    }   // the tier owns them (on the GPU backend they would be a 72 GiB upload)
    // the compressed-KV state is sized for the context actually used, not the model's context_length: the
    // smallest compress ratio is 4, so ctx/4 rows (+2 for the partial block) cover every layer
    const int64_t ctx = a.ctx > 0 ? a.ctx : (int64_t) prompt.size() + a.n_predict + 64;
    dc.comp_cap_max = ctx / 4 + 2;
    std::fprintf(stderr, "context cap %lld tokens\n", (long long) ctx);
    std::string err;
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "ds4_generate: dense init: %s\n", err.c_str()); return 1; }
    if (!dense.reserve_graphs()) { std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str()); return 1; }
    if (a.slots <= 0) {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        const double blob = 6.75 * 1048576.0;   // one DS4 expert (IQ2_XXS gate/up + Q2_K down)
        a.slots = std::max<int64_t>(0, (int64_t) (((double) fr - a.vram_margin_gib * 1073741824.0) / blob));
        std::fprintf(stderr, "slots auto: %.2f GiB free after the dense half -> %lld slots (margin %.2f GiB)\n",
                     fr / 1073741824.0, (long long) a.slots, a.vram_margin_gib);
    }

    // ---- routed-expert tier
    namespace ds4 = strata::ds4;
    ds4::Ds4MoeTier tier;
    ds4::Ds4MoeConfig mc;
    mc.slots = a.slots;
    mc.pcie_frac = a.pcie;
    mc.pf_b = a.pf_b;
    mc.threads = a.threads;
    mc.arena_gib = a.arena_gib;
    mc.max_arena_gib = a.arena_gib;   // the explicit flag is the ceiling; mem_floor_gib + memguard still guard RAM
    mc.cpu_only = a.experts == "cpu";
    if (!tier.init(a.model, mc, err)) { std::fprintf(stderr, "ds4_generate: tier init: %s\n", err.c_str()); return 1; }
    if (!a.profile.empty() && !mc.cpu_only && !tier.seed_from_routes(a.profile, err)) {
        std::fprintf(stderr, "ds4_generate: profile: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "tier: %s, %lld resident slots, arena %.1f GiB (%lld experts), file tier %lld; load %.1f s\n",
                 tier.mode(), (long long) tier.resident(), tier.arena_gib(), (long long) tier.arena_experts(),
                 (long long) tier.file_tier(), (now_ms() - t_load0) / 1000.0);

    const int n_layer = (int) dense.n_layer();
    const int top_k = (int) tier.geom().top_k;
    const int kPredW = ds4::Ds4MoeConfig::kPredW;
    const int64_t n_embd = tier.geom().n_embd;
    std::vector<int> ids_i((size_t) top_k), pred_i((size_t) kPredW);
    std::vector<int32_t> ids32((size_t) top_k), pred32((size_t) kPredW);
    std::vector<float> w((size_t) top_k), routed((size_t) n_embd);
    std::mt19937_64 rng(a.seed);

    // per-phase wall time (ms, summed over the decode): predict+prefetch, attention+router, experts, finish, head
    double t_ph[5] = {0, 0, 0, 0, 0};
    bool timing = false;
    // one token through every layer; returns false on any engine error
    std::vector<float> rb((size_t) tier.geom().n_experts);
    auto step = [&](int tid, int pos) -> bool {
        if (a.route_bias != 0.0f && !mc.cpu_only)
            for (int l = 0; l < n_layer; ++l) {   // the residency the cache has NOW (admissions move it per token)
                for (int64_t e = 0; e < (int64_t) rb.size(); ++e) rb[(size_t) e] = tier.resident(l, e) ? a.route_bias : 0.0f;
                dense.set_route_bias(l, rb.data());
            }
        if (!dense.begin_token(tid)) return false;
        for (int l = 0; l < n_layer; ++l) {
            double t0 = timing ? now_ms() : 0, t1;
            if (!mc.cpu_only && mc.pf_b > 0 && dense.predict(l, pred_i.data(), kPredW)) {
                int n = 0;
                for (int i = 0; i < kPredW && pred_i[(size_t) i] >= 0; ++i) pred32[(size_t) n++] = pred_i[(size_t) i];
                if (n > 0) tier.prefetch(l, pred32.data(), n);
            }
            if (timing) { t1 = now_ms(); t_ph[0] += t1 - t0; t0 = t1; }
            const float* x = nullptr;
            ggml_tensor* x_dev = nullptr;
            if (!dense.attn_router(l, pos, tid, ids_i.data(), w.data(), &x, &x_dev)) return false;
            if (timing) { t1 = now_ms(); t_ph[1] += t1 - t0; t0 = t1; }
            for (int k = 0; k < top_k; ++k) ids32[(size_t) k] = ids_i[(size_t) k];
            if (!tier.run(l, ids32.data(), w.data(), x, routed.data())) {
                std::fprintf(stderr, "ds4_generate: tier.run refused at layer %d (x %s)\n", l, x ? "set" : "NULL");
                return false;
            }
            if (timing) { t1 = now_ms(); t_ph[2] += t1 - t0; t0 = t1; }
            if (!dense.finish_layer(l, routed.data())) return false;
            if (timing) { t1 = now_ms(); t_ph[3] += t1 - t0; }
        }
        return true;
    };

    // ---- prefill (decode loop over the prompt)
    tier.reset_stats();
    const double t_pf0 = now_ms();
    double nll = 0;
    int n_scored = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (!step(prompt[i], (int) i)) {
            std::fprintf(stderr, "ds4_generate: prefill failed at %zu: %s\n", i, dense.last_error().c_str());
            return 1;
        }
        if (a.ppl && i + 1 < prompt.size()) {
            const float* l = nullptr;
            int nv = 0;
            if (!dense.logits(&l, &nv)) return 1;
            const float mx = *std::max_element(l, l + nv);
            double z = 0;
            for (int v = 0; v < nv; ++v) z += std::exp((double) (l[v] - mx));
            nll += (std::log(z) + mx) - l[prompt[i + 1]];
            ++n_scored;
        }
    }
    if (a.ppl && n_scored)
        std::fprintf(stderr, "ppl: %d positions, mean NLL %.5f, perplexity %.4f\n", n_scored, nll / n_scored,
                     std::exp(nll / n_scored));
    const double pf_ms = now_ms() - t_pf0;
    const ds4::Ds4MoeStats pf_stats = tier.stats();

    // ---- decode
    tier.reset_stats();
    int pos = (int) prompt.size();
    int n_gen = 0;
    const float* lg = nullptr;
    int n_vocab = 0;
    const double t_dec0 = now_ms();
    timing = true;
    for (; n_gen < a.n_predict; ++n_gen) {
        const double th0 = now_ms();
        const bool lok = dense.logits(&lg, &n_vocab);
        t_ph[4] += now_ms() - th0;
        if (!lok) { std::fprintf(stderr, "ds4_generate: logits: %s\n", dense.last_error().c_str()); return 1; }
        if (n_gen == 0 && !a.dump_logits.empty()) {
            if (std::FILE* f = std::fopen(a.dump_logits.c_str(), "wb")) { std::fwrite(lg, 4, (size_t) n_vocab, f); std::fclose(f); }
        }
        const int tok = sample(lg, n_vocab, a.temp, rng);
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end()) { ++n_gen; break; }
        if (n_gen + 1 == a.n_predict) { ++n_gen; break; }   // the last token needs no forward pass
        if (!step(tok, pos++)) {
            std::fprintf(stderr, "ds4_generate: decode failed at pos %d: %s\n", pos - 1, dense.last_error().c_str());
            return 1;
        }
    }
    const double dec_ms = now_ms() - t_dec0;
    const ds4::Ds4MoeStats st = tier.stats();
    const int dec_steps = std::max(1, n_gen - 1);   // forward passes actually run during decode

    std::fprintf(stderr, "\nprefill: %zu tokens in %.2f s = %.2f tok/s (decode-loop prefill)\n", prompt.size(),
                 pf_ms / 1000.0, 1000.0 * (double) prompt.size() / pf_ms);
    std::fprintf(stderr, "decode : %d tokens, %d forward passes in %.2f s = %.2f tok/s\n", n_gen, dec_steps,
                 dec_ms / 1000.0, 1000.0 * dec_steps / dec_ms);
    const double look = (double) std::max<int64_t>(1, st.lookups());
    std::fprintf(stderr, "experts/token: hit %.1f%% (prefetched-useful %.1f/token of %.1f issued), cpu %.1f%%, "
                         "pcie %.1f%%, file tier %lld\n",
                 100.0 * (double) st.hits / look, (double) st.prefetched_useful / dec_steps,
                 (double) st.prefetch_issued / dec_steps, 100.0 * (double) st.cpu / look,
                 100.0 * (double) st.pcie / look, (long long) st.file_tier);
    std::fprintf(stderr, "decode ms/token: predict+prefetch %.2f, attention+router %.2f, experts %.2f, finish %.2f, "
                         "head %.2f\n", t_ph[0] / dec_steps, t_ph[1] / dec_steps, t_ph[2] / dec_steps,
                 t_ph[3] / dec_steps, t_ph[4] / dec_steps);
    std::fprintf(stderr, "tier ms/token: wall %.2f (gpu hits %.2f, pcie %.2f, cpu pool %.2f)\n", st.wall_ms / dec_steps,
                 st.hit_ms / dec_steps, st.pcie_ms / dec_steps, st.cpu_ms / dec_steps);
    (void) pf_stats;
    tier.close();
    ggml_backend_free(be);
    return 0;
}
