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
//
// --mtp FILE (the MTP head's GGUF) drafts with it alongside plain decoding and reports the acceptance rate - the
// output is unchanged (step 1 of speculative decoding: measure before building the verify loop).
// --mtp FILE --verify: speculative decoding, greedy only.  Each pass runs [token, MTP draft] through the trunk as one
// 2-token pass (experts of the two tokens read once, Ds4MoeTier::run_multi); the draft is kept when it equals the
// trunk's argmax, which also yields the next token for free.  A rejected draft's position is simply decoded again
// (all state is position-indexed).  Same tokens as plain greedy decoding, up to GPU rounding.
// --mtp-resident: copy the MTP experts (3.2 GB MXFP4) into anonymous RAM at load instead of reading them off the
// file's mmap: warm, a draft's experts take ~3.5 ms (ds4_mtp_bench, 2026-10-07); with the file's pages evicted
// ~10-18 ms.  Costs 3.2 GB of the memguard cap (the arena may have to shrink).
#include "ds4_dense.hpp"
#include "ds4_moe.hpp"
#include "mtp_experts.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
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
    std::string mtp;             // the MTP head's GGUF: draft + acceptance stats
    bool verify = false;         // --verify: speculative decoding with the MTP draft
    bool mtp_resident = false;   // --mtp-resident: the MTP experts copied into RAM at load (not the file's mmap)
    bool vram_lru = false;       // --vram-lru: misses / used prefetches take the layer's LRU VRAM slot (MiMo s16)
    bool arena_adapt = false;    // --arena-adapt: a file-tier read replaces the layer's LRU arena expert
    bool arena_skip = false;     // --arena-skip-resident: the VRAM seed's experts stay out of the host arena
    int prefill_chunk = 0;       // --prefill-chunk N: the prompt in chunks of N tokens (0 = through the decode loop)
    bool chunk_mmq = false;      // --chunk-mmq: prompt chunks' streamed experts through MMQ (int8 tensor cores)
    bool chunk_prestage = false; // --chunk-prestage: stream layer l+1's experts while layer l computes
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
        "       [--arena-gib 60] [--threads N] [--ctx N] [--temp 0] [--seed 1] [--stop id,id] [--dump-logits f.f32] [--quiet]\n"
        "       [--mtp FILE [--verify] [--mtp-resident]]\n");
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
        else if (k == "--mtp") a.mtp = next();
        else if (k == "--verify") a.verify = true;
        else if (k == "--mtp-resident") a.mtp_resident = true;
        else if (k == "--vram-lru") a.vram_lru = true;
        else if (k == "--arena-adapt") a.arena_adapt = true;
        else if (k == "--arena-skip-resident") a.arena_skip = true;
        else if (k == "--prefill-chunk") a.prefill_chunk = std::atoi(next().c_str());
        else if (k == "--chunk-mmq") a.chunk_mmq = true;
        else if (k == "--chunk-prestage") a.chunk_prestage = true;
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
    // ggml's DEBUG lines ("CUDA Graph id N reused", ~90 per token once graphs are reused) cost a stderr write each on
    // the decode path; drop them unless DS4_GGML_DEBUG=1
    if (!std::getenv("DS4_GGML_DEBUG"))
        ggml_log_set([](ggml_log_level lv, const char* text, void*) {
            if (lv != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
        }, nullptr);
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
    if (a.verify && (a.mtp.empty() || a.temp > 0.0f)) {
        std::fprintf(stderr, "ds4_generate: --verify needs --mtp and greedy decoding (--temp 0)\n");
        return 2;
    }

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
    dc.mtp_path = a.mtp;
    dc.prefill_chunk = a.prefill_chunk;
    if (a.prefill_chunk > 0 && !a.mtp.empty()) {
        std::fprintf(stderr, "ds4_generate: --prefill-chunk does not fill the MTP window yet; drop --mtp or the chunk\n");
        return 2;
    }
    std::fprintf(stderr, "context cap %lld tokens\n", (long long) ctx);
    std::string err;
    if (!dense.init(a.model, dc, err)) { std::fprintf(stderr, "ds4_generate: dense init: %s\n", err.c_str()); return 1; }
    if (!dense.reserve_graphs(a.verify ? 2 : 1)) { std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str()); return 1; }
    if (a.slots <= 0) {
        // CUDA-graph instances take VRAM when first captured (after this point): the MTP block and the 2-token
        // verify graphs add many (a 0.9 GiB margin OOMed in cudaGraphInstantiate with --mtp, 2026-10-07)
        if (!a.mtp.empty()) a.vram_margin_gib += 0.25;
        if (a.verify) a.vram_margin_gib += 0.75;
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
    mc.vram_lru = a.vram_lru;
    mc.arena_adapt = a.arena_adapt;
    mc.arena_skip_resident = a.arena_skip;
    // chunked prefill: the arena now, the VRAM cache after the prompt (the chunks use that VRAM first)
    const bool defer_seed = a.prefill_chunk > 0 && !mc.cpu_only && !a.profile.empty();
    mc.defer_cache = defer_seed;
    mc.chunk_mmq = a.chunk_mmq;
    mc.chunk_prestage = a.chunk_prestage;
    if (!tier.init(a.model, mc, err)) { std::fprintf(stderr, "ds4_generate: tier init: %s\n", err.c_str()); return 1; }
    if (!a.profile.empty() && !mc.cpu_only &&
        !(defer_seed ? tier.build_arena_from_routes(a.profile, 512, err) : tier.seed_from_routes(a.profile, err))) {
        std::fprintf(stderr, "ds4_generate: profile: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "tier: %s, %lld resident slots, arena %.1f GiB (%lld experts), file tier %lld; load %.1f s\n",
                 tier.mode(), (long long) tier.resident(), tier.arena_gib(), (long long) tier.arena_experts(),
                 (long long) tier.file_tier(), (now_ms() - t_load0) / 1000.0);

    const int n_layer = (int) dense.n_layer();
    // ---- MTP head (draft + acceptance stats)
    const int il_mtp = dense.mtp_layer();
    ds4::MtpExperts mtp_x;
    if (il_mtp >= 0) {
        if (!mtp_x.init(a.mtp, il_mtp, dense.geom().n_embd, dense.geom().n_expert_used,
                        (float) dense.geom().swiglu_clamp_exp[(size_t) il_mtp], a.threads > 0 ? a.threads : 8, err,
                        a.mtp_resident)) {
            std::fprintf(stderr, "ds4_generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "MTP head: layer %d from %s (experts on the CPU, MXFP4)\n", il_mtp, a.mtp.c_str());
    }
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

    // the MTP block at position `pos` with the token that follows it -> its argmax (the draft for pos + 2)
    double t_mtp = 0, t_mtp_x = 0;   // per draft: total, of which the routed experts
    int n_mtp = 0;
    const int NT = dense.max_tokens();
    std::vector<int> mids((size_t) (top_k * NT));
    std::vector<float> mw((size_t) (top_k * NT)), mrouted((size_t) (n_embd * NT));
    const float* mtp_last = nullptr;   // the last draft's MTP logits row (valid until the next Ds4Dense call)
    int mtp_nv = 0;
    // DS4_DUMP_DRAFTLOGITS=<file>: per draft, int32 k (the index of the generated token it guesses) + the MTP logits
    // row - plain --mtp and --verify must agree wherever both drafted token k (the MTP block's own KV across passes)
    std::FILE* drl = nullptr;
    if (const char* f = std::getenv("DS4_DUMP_DRAFTLOGITS")) drl = std::fopen(f, "wb");
    auto dump_draft = [&](int k) {
        if (!drl || !mtp_last) return;
        const int32_t k32 = k;
        std::fwrite(&k32, 4, 1, drl);
        std::fwrite(mtp_last, 4, (size_t) mtp_nv, drl);
    };
    // the MTP block at positions pos0..pos0+n-1 with their following tokens -> the last position's argmax
    auto mtp_draft_n = [&](const int* next_toks, int n, int pos0, int* draft) -> bool {
        const double t0 = now_ms();
        const float* x = nullptr;
        if (!dense.mtp_begin(next_toks, n)) return false;
        if (!dense.attn_router_n(il_mtp, pos0, n, mids.data(), mw.data(), &x)) return false;
        const double tx0 = now_ms();
        for (int t = 0; t < n; ++t)
            if (!mtp_x.run(x + (size_t) t * n_embd, mids.data() + t * top_k, mw.data() + t * top_k,
                           mrouted.data() + (size_t) t * n_embd)) return false;
        t_mtp_x += now_ms() - tx0;
        if (!dense.finish_layer_n(il_mtp, n, mrouted.data())) return false;
        const float* lg = nullptr;
        int nv = 0;
        if (!dense.mtp_logits_n(n, &lg, &nv)) return false;
        const float* last = lg + (size_t) (n - 1) * nv;
        mtp_last = last; mtp_nv = nv;
        if (draft) *draft = (int) (std::max_element(last, last + nv) - last);
        t_mtp += now_ms() - t0;
        ++n_mtp;
        return true;
    };
    auto mtp_draft = [&](int next_tok, int pos, int* draft) { return mtp_draft_n(&next_tok, 1, pos, draft); };

    // n consecutive tokens (positions pos0..) through every trunk layer in one pass: the verify step
    std::vector<int> ids_n((size_t) (top_k * NT));
    std::vector<int32_t> ids32_n((size_t) (top_k * NT));
    std::vector<float> w_n((size_t) (top_k * NT)), routed_n((size_t) (n_embd * NT));
    auto step_n = [&](const int* toks, int n, int pos0) -> bool {
        if (a.route_bias != 0.0f && !mc.cpu_only)
            for (int l = 0; l < n_layer; ++l) {
                for (int64_t e = 0; e < (int64_t) rb.size(); ++e) rb[(size_t) e] = tier.resident(l, e) ? a.route_bias : 0.0f;
                dense.set_route_bias(l, rb.data());
            }
        if (!dense.begin_tokens(toks, n)) return false;
        for (int l = 0; l < n_layer; ++l) {
            double t0 = timing ? now_ms() : 0, t1;
            if (!mc.cpu_only && mc.pf_b > 0 && dense.predict(l, pred_i.data(), kPredW)) {   // token 0's hint
                int m = 0;
                for (int i = 0; i < kPredW && pred_i[(size_t) i] >= 0; ++i) pred32[(size_t) m++] = pred_i[(size_t) i];
                if (m > 0) tier.prefetch(l, pred32.data(), m);
            }
            if (timing) { t1 = now_ms(); t_ph[0] += t1 - t0; t0 = t1; }
            const float* x = nullptr;
            if (!dense.attn_router_n(l, pos0, n, ids_n.data(), w_n.data(), &x)) return false;
            if (timing) { t1 = now_ms(); t_ph[1] += t1 - t0; t0 = t1; }
            for (int k = 0; k < top_k * n; ++k) ids32_n[(size_t) k] = ids_n[(size_t) k];
            if (!tier.run_multi(l, n, ids32_n.data(), w_n.data(), x, routed_n.data())) {
                std::fprintf(stderr, "ds4_generate: tier.run_multi refused at layer %d\n", l);
                return false;
            }
            if (timing) { t1 = now_ms(); t_ph[2] += t1 - t0; t0 = t1; }
            if (!dense.finish_layer_n(l, n, routed_n.data())) return false;
            if (timing) { t1 = now_ms(); t_ph[3] += t1 - t0; }
        }
        return true;
    };

    // ---- prefill (decode loop over the prompt)
    tier.reset_stats();
    const double t_pf0 = now_ms();
    double nll = 0;
    int n_scored = 0;
    std::vector<float> first_logits;   // chunked prefill: the prompt's last row (the decode state holds no token yet)
    if (a.prefill_chunk > 0) {
        const int NP = a.prefill_chunk;
        std::vector<int> cids((size_t) (NP * top_k));
        std::vector<int32_t> cids32((size_t) (NP * top_k));
        std::vector<float> cw((size_t) (NP * top_k));
        float* crouted = dense.prefill_routed_buffer();   // pinned on CUDA
        void* croute_dev = mc.cpu_only ? nullptr : dense.prefill_routed_device();   // the tier writes the sums there
        if (!crouted) { std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str()); return 1; }
        const int LR = 16;   // logits rows per head call (ppl)
        std::vector<float> lrows;
        double t_dense = 0, t_exp = 0, t_attn = 0, t_head = 0;
        for (size_t c0 = 0; c0 < prompt.size(); c0 += (size_t) NP) {
            const int n = (int) std::min<size_t>((size_t) NP, prompt.size() - c0);
            if (!dense.prefill_begin(prompt.data() + c0, n, (int) c0)) {
                std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str());
                return 1;
            }
            for (int l = 0; l < n_layer; ++l) {
                double t0 = now_ms();
                const float* fn = nullptr;
                if (!dense.prefill_attn(l, cids.data(), cw.data(), &fn)) {
                    std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str());
                    return 1;
                }
                double t1 = now_ms(); t_dense += t1 - t0; t_attn += t1 - t0; t0 = t1;
                for (int k = 0; k < n * top_k; ++k) cids32[(size_t) k] = cids[(size_t) k];
                if (!(croute_dev ? tier.run_chunk_dev(l, n, cids32.data(), cw.data(), fn, croute_dev)
                                 : tier.run_chunk(l, n, cids32.data(), cw.data(), fn, crouted))) {
                    std::fprintf(stderr, "ds4_generate: tier.run_chunk refused at layer %d\n", l);
                    return 1;
                }
                t1 = now_ms(); t_exp += t1 - t0; t0 = t1;
                if (!dense.prefill_finish(l, croute_dev ? nullptr : crouted)) {
                    std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str());
                    return 1;
                }
                t_dense += now_ms() - t0;
            }
            int nv = (int) dense.geom().vocab_size;
            const double th = now_ms();
            if (a.ppl) {   // rows whose next token is in the prompt
                const int last = (int) std::min<size_t>((size_t) n, prompt.size() - 1 - c0);
                for (int r0 = 0; r0 < last; r0 += LR) {
                    const int nr = std::min(LR, last - r0);
                    lrows.resize((size_t) nr * (size_t) nv);
                    if (!dense.prefill_logits(r0, nr, lrows.data())) {
                        std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str());
                        return 1;
                    }
                    for (int r = 0; r < nr; ++r) {
                        const float* l = lrows.data() + (size_t) r * nv;
                        const float mx = *std::max_element(l, l + nv);
                        double z = 0;
                        for (int v = 0; v < nv; ++v) z += std::exp((double) (l[v] - mx));
                        nll += (std::log(z) + mx) - l[prompt[c0 + (size_t) (r0 + r) + 1]];
                        ++n_scored;
                    }
                }
            }
            if (c0 + (size_t) n == prompt.size()) {
                first_logits.resize((size_t) nv);
                if (!dense.prefill_logits(n - 1, 1, first_logits.data())) {
                    std::fprintf(stderr, "ds4_generate: %s\n", dense.last_error().c_str());
                    return 1;
                }
            }
            t_head += now_ms() - th;
            dense.prefill_end();
        }
        tier.release_chunk();
        dense.prefill_release();
        const double t_seed = now_ms();
        if (defer_seed && !tier.seed_from_routes(a.profile, err)) {
            std::fprintf(stderr, "ds4_generate: profile: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "prefill chunks of %d: dense %.0f ms (attention+router %.0f, finish %.0f), experts %.0f ms, "
                             "head/ppl %.0f ms, cache seed %.0f ms -> %lld resident slots\n", NP, t_dense, t_attn,
                     t_dense - t_attn, t_exp, t_head, now_ms() - t_seed, (long long) tier.resident());
    }
    for (size_t i = 0; a.prefill_chunk <= 0 && i < prompt.size(); ++i) {
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
        static const bool vtrace = std::getenv("DS4_VRAM_TRACE") != nullptr;   // free VRAM per prefill step
        auto vfree = [&]() { size_t fr = 0, tot = 0; ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot); return fr / 1048576.0; };
        if (vtrace) std::fprintf(stderr, "[vram] pos %zu after trunk: %.0f MiB free\n", i, vfree());
        // the MTP block's own sliding-window KV is filled over the prompt as well (it reads the trunk state, so after
        // the trunk's logits above)
        if (il_mtp >= 0 && i + 1 < prompt.size() && !mtp_draft(prompt[i + 1], (int) i, nullptr)) {
            std::fprintf(stderr, "ds4_generate: MTP prefill failed at %zu: %s\n", i, dense.last_error().c_str());
            return 1;
        }
        if (vtrace && il_mtp >= 0) std::fprintf(stderr, "[vram] pos %zu after MTP:   %.0f MiB free\n", i, vfree());
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
    t_mtp = 0; t_mtp_x = 0; n_mtp = 0;
    int draft = -1, n_drafts = 0, n_accept = 0;
    int n_passes = 0;
    // DS4_DUMP_TOKLOGITS=<file>: the logits row behind every emitted token, appended (f32 x n_vocab each) - the verify
    // gate: on the CPU, multi-token passes are bit-identical to one-token decoding, so these rows must match exactly
    // (greedy tokens alone miss e.g. a skipped position on the mini fixtures)
    std::FILE* tokl = nullptr;
    if (const char* f = std::getenv("DS4_DUMP_TOKLOGITS")) tokl = std::fopen(f, "wb");
    auto dump_row = [&](const float* row, int nv) { if (tokl) std::fwrite(row, 4, (size_t) nv, tokl); };
    if (a.verify) {
        // pos = the position of `cur`, the newest token (emitted, not yet forwarded); draft = MTP's guess for pos + 1
        auto argmax = [](const float* v, int n) { return (int) (std::max_element(v, v + n) - v); };
        bool done = false;
        auto emit = [&](int tok) {
            std::printf("%d\n", tok);
            std::fflush(stdout);
            ++n_gen;
            if (n_gen >= a.n_predict || std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end()) done = true;
        };
        // test hooks (the mini MTP's random weights never draft right, so only the reject path would run):
        // DS4_VERIFY_ORACLE=<file> - drafts from a plain greedy run's stdout (one id per line; draft for the token
        // emitted n-th = line n); DS4_VERIFY_CORRUPT=k - then spoil every k-th draft (mixed accept/reject)
        std::vector<int> oracle;
        int corrupt_every = 0, n_fixed = 0;
        if (const char* f = std::getenv("DS4_VERIFY_ORACLE")) {
            std::FILE* fp = std::fopen(f, "r");
            if (!fp) { std::fprintf(stderr, "ds4_generate: cannot read DS4_VERIFY_ORACLE %s\n", f); return 1; }
            int v = 0;
            while (std::fscanf(fp, "%d", &v) == 1) oracle.push_back(v);
            std::fclose(fp);
            if (const char* c = std::getenv("DS4_VERIFY_CORRUPT")) corrupt_every = std::atoi(c);
            std::fprintf(stderr, "verify : TEST drafts from %s (%zu ids), corrupt every %d\n", f, oracle.size(), corrupt_every);
        }
        auto fix_draft = [&]() {   // after each MTP draft: n_gen = the index of the token the draft guesses
            dump_draft(n_gen);
            if (oracle.empty() || n_gen >= (int) oracle.size()) return;
            draft = oracle[(size_t) n_gen];
            if (corrupt_every > 0 && ++n_fixed % corrupt_every == 0) draft = (draft + 1) % n_vocab;
        };
        const double th0 = now_ms();
        if (!dense.logits(&lg, &n_vocab)) return 1;
        t_ph[4] += now_ms() - th0;
        int cur = argmax(lg, n_vocab);   // (--verify needs --mtp: never after chunked prefill)
        dump_row(lg, n_vocab);
        emit(cur);
        if (!done && !mtp_draft(cur, pos - 1, &draft)) {
            std::fprintf(stderr, "ds4_generate: MTP draft failed: %s\n", dense.last_error().c_str());
            return 1;
        }
        if (!done) fix_draft();
        while (!done) {
            const int toks[2] = { cur, draft };
            if (!step_n(toks, 2, pos)) {
                std::fprintf(stderr, "ds4_generate: verify pass failed at pos %d: %s\n", pos, dense.last_error().c_str());
                return 1;
            }
            ++n_passes;
            const double tl0 = now_ms();
            if (!dense.logits_n(2, &lg, &n_vocab)) return 1;
            t_ph[4] += now_ms() - tl0;
            const int t1 = argmax(lg, n_vocab);                       // the trunk's token at pos + 1
            ++n_drafts;
            if (t1 == draft) {
                ++n_accept;
                const int t2 = argmax(lg + n_vocab, n_vocab);         // ... and, for free, at pos + 2
                dump_row(lg, n_vocab);
                emit(t1);
                if (done) break;
                dump_row(lg + n_vocab, n_vocab);
                emit(t2);
                if (done) break;
                const int nxt[2] = { t1, t2 };
                if (!mtp_draft_n(nxt, 2, pos, &draft)) return 1;      // MTP at pos, pos+1 -> guess for pos + 3
                fix_draft();
                cur = t2;
                pos += 2;
            } else {
                dump_row(lg, n_vocab);
                emit(t1);
                if (done) break;
                if (!mtp_draft(t1, pos, &draft)) return 1;            // MTP at pos -> guess for pos + 2
                fix_draft();
                cur = t1;
                pos += 1;
            }
        }
    }
    for (; !a.verify && n_gen < a.n_predict; ++n_gen) {
        const double th0 = now_ms();
        bool lok = true;
        if (n_gen == 0 && !first_logits.empty()) { lg = first_logits.data(); n_vocab = (int) first_logits.size(); }
        else lok = dense.logits(&lg, &n_vocab);
        t_ph[4] += now_ms() - th0;
        if (!lok) { std::fprintf(stderr, "ds4_generate: logits: %s\n", dense.last_error().c_str()); return 1; }
        if (n_gen == 0 && !a.dump_logits.empty()) {
            if (std::FILE* f = std::fopen(a.dump_logits.c_str(), "wb")) { std::fwrite(lg, 4, (size_t) n_vocab, f); std::fclose(f); }
        }
        const int tok = sample(lg, n_vocab, a.temp, rng);
        dump_row(lg, n_vocab);
        if (draft >= 0) { ++n_drafts; n_accept += draft == tok; }
        draft = -1;
        if (il_mtp >= 0 && !mtp_draft(tok, pos - 1, &draft)) {   // MTP at the position whose logits gave `tok`
            std::fprintf(stderr, "ds4_generate: MTP draft failed at pos %d: %s\n", pos - 1, dense.last_error().c_str());
            return 1;
        }
        if (il_mtp >= 0) dump_draft(n_gen + 1);
        std::printf("%d\n", tok);
        std::fflush(stdout);
        if (std::find(a.stop.begin(), a.stop.end(), tok) != a.stop.end()) { ++n_gen; break; }
        if (n_gen + 1 == a.n_predict) { ++n_gen; break; }   // the last token needs no forward pass
        if (!step(tok, pos++)) {
            std::fprintf(stderr, "ds4_generate: decode failed at pos %d: %s\n", pos - 1, dense.last_error().c_str());
            return 1;
        }
    }
    if (tokl) std::fclose(tokl);
    if (drl) std::fclose(drl);
    const double dec_ms = now_ms() - t_dec0;
    const ds4::Ds4MoeStats st = tier.stats();
    const int dec_steps = a.verify ? std::max(1, n_passes) : std::max(1, n_gen - 1);   // forward passes run in decode

    std::fprintf(stderr, "\nprefill: %zu tokens in %.2f s = %.2f tok/s (%s)\n", prompt.size(),
                 pf_ms / 1000.0, 1000.0 * (double) prompt.size() / pf_ms,
                 a.prefill_chunk > 0 ? "chunked" : "decode-loop prefill");
    std::fprintf(stderr, "decode : %d tokens, %d forward passes in %.2f s = %.2f tok/s\n", n_gen, dec_steps,
                 dec_ms / 1000.0, 1000.0 * dec_steps / dec_ms);
    if (a.verify)   // the first token comes from the prefill's logits: the passes produced the other n_gen - 1
        std::fprintf(stderr, "verify : %d passes, %.2f tokens/pass, %.2f tok/s (tokens after the first / decode time)\n",
                     n_passes, (double) (n_gen - 1) / std::max(1, n_passes), 1000.0 * (n_gen - 1) / dec_ms);
    const double look = (double) std::max<int64_t>(1, st.lookups());
    std::fprintf(stderr, "experts/token: hit %.1f%% (prefetched-useful %.1f/token of %.1f issued), cpu %.1f%%, "
                         "pcie %.1f%%, file tier %lld\n",
                 100.0 * (double) st.hits / look, (double) st.prefetched_useful / dec_steps,
                 (double) st.prefetch_issued / dec_steps, 100.0 * (double) st.cpu / look,
                 100.0 * (double) st.pcie / look, (long long) st.file_tier);
    if (a.vram_lru || a.arena_adapt || st.file_tier > 0)
        std::fprintf(stderr, "tier moves: vram_lru swaps %.2f/pass (demoted %lld), arena swaps %lld, file reads %.2f "
                             "ms/pass\n", (double) st.vram_swaps / dec_steps, (long long) st.vram_demotes,
                     (long long) st.arena_swaps, st.file_ms / dec_steps);
    std::fprintf(stderr, "decode ms/token: predict+prefetch %.2f, attention+router %.2f, experts %.2f, finish %.2f, "
                         "head %.2f\n", t_ph[0] / dec_steps, t_ph[1] / dec_steps, t_ph[2] / dec_steps,
                 t_ph[3] / dec_steps, t_ph[4] / dec_steps);
    if (il_mtp >= 0)
        std::fprintf(stderr, "MTP: %d drafts, %d accepted = %.1f%% (greedy: draft == the next decoded token); "
                             "%.2f ms/draft (routed experts %.2f, rest %.2f); expert pages in RAM %.0f%%%s\n",
                     n_drafts, n_accept, n_drafts ? 100.0 * n_accept / n_drafts : 0.0, n_mtp ? t_mtp / n_mtp : 0.0,
                     n_mtp ? t_mtp_x / n_mtp : 0.0, n_mtp ? (t_mtp - t_mtp_x) / n_mtp : 0.0,
                     100.0 * mtp_x.resident_frac(), a.mtp_resident ? " (--mtp-resident copy)" : " (file mmap)");
    std::fprintf(stderr, "tier ms/token: wall %.2f (gpu hits %.2f, pcie %.2f, cpu pool %.2f)\n", st.wall_ms / dec_steps,
                 st.hit_ms / dec_steps, st.pcie_ms / dec_steps, st.cpu_ms / dec_steps);
    {
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(be), &fr, &tot);
        std::fprintf(stderr, "VRAM free at the end: %.2f GiB (calibrates --vram-margin)\n", fr / 1073741824.0);
    }
    (void) pf_stats;
    tier.close();
    ggml_backend_free(be);
    return 0;
}
