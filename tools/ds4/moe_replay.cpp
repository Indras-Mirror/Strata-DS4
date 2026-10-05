// tools/ds4/moe_replay.cpp - Phase 4a: Strata's MoE expert engine on DeepSeek-V4-Flash, decoupled from attention.
//
//     moe_replay <DeepSeek-V4-Flash.gguf> --routes ds4routes.bin --tokens 320 [options]
//
// The question this answers: with the expert tier built the Strata way - a VRAM cache of 6.75 MiB slots, the CPU
// pool on the misses, a PCIe share of the misses computed on the GPU *in parallel*, overlap, a pinned host arena -
// how many ms per token does the MoE half of a DeepSeek-V4 decode cost, and does MoE ms + the measured non-MoE GPU
// ms (Phase 0: 17.6) beat llama.cpp's config B (16.34 tok/s) and the go line (19.6 tok/s)?
//
// WHAT IT IS AND IS NOT.  It is the *engine*: the real ExpertCache slots, the real ExpertPool (ggml-cpu IQ2_XXS /
// Q2_K vec_dot with row splitting), the real PCIe miss split, the real pinned arena, driven by a replay of
// RECORDED routes (`ds4routes.bin`: 43 layers x top-6 expert ids per token) instead of a real model's router.  It
// is NOT a model: activations are random vectors of the right shape and the measured GPU "gap" is a synthetic
// busy kernel standing in for the dense+attention work the concurrent Phase-3 slice owns (Phase 0 measured
// ~0.4 ms/layer).  Separating the expert half is the whole point of the slice.
//
// TWO MODES.
//   * TIMING (default): replay --tokens tokens x 43 layers, report ms/token and the split across hit (VRAM
//     cache) / miss-CPU / miss-PCIe / gap / overlap-idle.  Sweep --slots / --pcie-frac / --threads.
//   * CORRECTNESS (--correctness K): for a few tokens, run every expert of every layer through the engine and
//     against a dequant+F32 reference of the same expert blobs (weights by ggml's to_float, activation by the
//     engine's own quantized buffer), per expert and for the whole layer's MoE output; rel err < 1e-3.  The pure
//     float error (the 8-bit activation rounding) is printed alongside, and the GPU grouped path is checked too.
//
// BLOB SOURCE.  Real blobs, read from the model's GGUF in place (three role slices per expert, assembled into the
// [gate|up|down] native layout) into a host arena.  --pin builds the arena with strata::core::PinnedArena
// (cudaHostRegister) so the PCIe share DMAs straight out of it and the PCIe fraction means something; without it
// the arena is ordinary memory and cudaMemcpyAsync stages through the driver, which is reported, not hidden.
//
// SHARED-RESOURCE PROTOCOL.  --arena-gib > 8 must not run while the Phase-3 gate holds the full 80 GB model in
// RAM.  The tool refuses a pin larger than --max-arena-gib and aborts when MemAvailable would fall below
// --mem-floor-gib.  It is otherwise read-only.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

// ---- the DeepSeek-V4-Flash expert geometry (docs/ds4/DSV4_ARCH_SPEC.md s1.3) -------------------------------
constexpr int64_t H = 4096;       // n_embd
constexpr int64_t FF = 2048;      // n_ff of one routed expert
constexpr int GU = 16;            // IQ2_XXS (gate and up)
constexpr int DN = 10;            // Q2_K (down)
constexpr int kLayers = 43;
constexpr int kExperts = 256;
constexpr int kTopK = 6;
constexpr int64_t kBatch = 512;   // route_probe tokenized in 512-token ubatches; see Routes
constexpr int kMaxParts = 8;      // the pool's MAXT is 8; K=6 needs no more
constexpr const char* kVerifyGguf = "blk.%d.ffn_%s_exps.weight";

int g_fail = 0;

void check(bool ok, const char* what, double got, double limit) {
    std::printf("  %-46s rel %.3e (limit %.1e)  %s\n", what, got, limit, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

double rel(const float* a, const float* b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        num += std::fabs((double) a[i] - (double) b[i]);
        den += std::fabs((double) b[i]);
    }
    return num / (den + 1e-30);
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_replay: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

float silu(float g) { return g / (1.0f + std::exp(-g)); }

// Q8_K (GGML_TYPE_Q8_K) is IQ2_XXS's and Q2_K's vec_dot_type, so it is the activation the CPU path quantizes to.
void deq_q8_k(const uint8_t* q, int64_t n, float* out) {
    struct Block { float d; int8_t qs[256]; int16_t bsums[16]; };
    static_assert(sizeof(Block) == 292, "block_q8_K layout");
    for (int64_t b = 0; b < n / 256; ++b) {
        const Block* blk = reinterpret_cast<const Block*>(q + b * sizeof(Block));
        for (int i = 0; i < 256; ++i) out[b * 256 + i] = blk->d * blk->qs[i];
    }
}

int64_t mem_available_gib() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    std::fclose(f);
    return kb < 0 ? -1 : (int64_t) (kb / (1024 * 1024));
}

// ---- arguments --------------------------------------------------------------------------------------------
struct Args {
    std::string gguf, routes, out_csv;
    int64_t tokens = 320, offset = 0;
    int64_t slots = 1820;        // 12.0 GB of 6.75 MiB slots: the measured ~50% held-out hit point (FINDINGS s2)
    double pcie_frac = 0.55;     // Strata's --pcie-frac for this PC
    int threads = 0;             // 0 = every physical core but the first (ExpertPool's default)
    double gap_ms = 0.4;         // Phase 0: ~0.4 ms/layer of dense+attention GPU work
    double arena_gib = 40.0;     // host arena budget for the replay's working set
    double max_arena_gib = 60.0; // the packet's pin ceiling
    int64_t mem_floor_gib = 3;   // abort below this MemAvailable
    int64_t correctness = 0, correctness_experts = 12;
    bool pin = false, cpu_only = false, no_cache = false, no_pcie = false, quiet = false;
    uint64_t seed = 20261005;
};

bool parse_args(int argc, char** argv, Args& a, std::string& err) {
    if (argc < 2) { err = "no model path"; return false; }
    a.gguf = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        const bool has = i + 1 < argc;
        auto next_s = [&]() { return has ? std::string(argv[++i]) : std::string(); };
        auto next_i = [&]() { return has ? (int64_t) std::atoll(argv[++i]) : 0; };
        auto next_d = [&]() { return has ? std::atof(argv[++i]) : 0.0; };
        if (k == "--routes") a.routes = next_s();
        else if (k == "--tokens") a.tokens = next_i();
        else if (k == "--offset") a.offset = next_i();
        else if (k == "--slots") a.slots = next_i();
        else if (k == "--pcie-frac") a.pcie_frac = next_d();
        else if (k == "--threads") a.threads = (int) next_i();
        else if (k == "--gap-ms") a.gap_ms = next_d();
        else if (k == "--arena-gib") a.arena_gib = next_d();
        else if (k == "--max-arena-gib") a.max_arena_gib = next_d();
        else if (k == "--mem-floor-gib") a.mem_floor_gib = next_i();
        else if (k == "--correctness") a.correctness = next_i();
        else if (k == "--correctness-experts") a.correctness_experts = next_i();
        else if (k == "--out") a.out_csv = next_s();
        else if (k == "--seed") a.seed = (uint64_t) next_i();
        else if (k == "--pin") a.pin = true;
        else if (k == "--cpu-only") a.cpu_only = true;
        else if (k == "--no-cache") a.no_cache = true;
        else if (k == "--no-pcie") a.no_pcie = true;
        else if (k == "--quiet") a.quiet = true;
        else { err = "unknown argument " + k; return false; }
    }
    if (a.routes.empty()) { err = "--routes is required"; return false; }
    return true;
}

void usage() {
    std::printf(
        "usage: moe_replay <model.gguf> --routes ds4routes.bin [--tokens N] [--offset O]\n"
        "       [--slots S] [--pcie-frac F] [--threads T] [--gap-ms X] [--arena-gib G] [--pin]\n"
        "       [--correctness K] [--cpu-only] [--no-cache] [--no-pcie] [--out csv] [--quiet]\n");
}

// ================================ route replay ================================
//
// route_probe.cpp dumps [layer:u16][expert:u16] from llama.cpp's eval callback, so the file is grouped by UBATCH:
// for each 512-token batch, layer 0's whole batch (512 tokens x 6), then layer 1's, ...  Within a layer the tokens
// are in order and the six experts of a token are contiguous.  Verified on the real file: 3,170,304 records =
// 12,288 tokens x 43 x 6, and every layer run inside every 512-token block is constant.
struct Routes {
    int64_t n_tokens = 0;
    std::vector<uint16_t> ids;   // [count * kLayers * kTopK] pairs (layer, expert), token-major
    int64_t offset = 0, count = 0;
    const uint16_t* tok(int64_t t) const { return ids.data() + (t - offset) * (kLayers * kTopK * 2); }
};

bool load_routes(const std::string& path, int64_t offset, int64_t want, Routes& r, std::string& err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { err = "cannot open " + path; return false; }
    const int64_t bytes = (int64_t) in.tellg();
    in.seekg(0);
    std::vector<uint16_t> all((size_t) (bytes / 2));
    in.read((char*) all.data(), bytes);
    if (!in) { err = "short read on " + path; return false; }
    const int64_t per_tok = (int64_t) kLayers * kTopK * 2;
    if (all.empty() || (int64_t) all.size() % per_tok != 0) { err = "not a whole number of tokens"; return false; }
    r.n_tokens = (int64_t) all.size() / per_tok;
    if (offset < 0 || want <= 0 || offset + want > r.n_tokens) {
        err = "token range outside 0.." + std::to_string(r.n_tokens);
        return false;
    }
    r.offset = offset;
    r.count = want;
    r.ids.resize((size_t) (want * per_tok));
    const int64_t recs_per_batch = kBatch * per_tok;
    for (int64_t t = offset; t < offset + want; ++t) {
        const int64_t b = t / kBatch, tb = t % kBatch;
        for (int64_t l = 0; l < kLayers; ++l) {
            const int64_t src = b * recs_per_batch + l * (kBatch * kTopK * 2) + tb * (kTopK * 2);
            std::memcpy(&r.ids[(size_t) ((t - offset) * per_tok + l * kTopK * 2)], &all[(size_t) src],
                        sizeof(uint16_t) * kTopK * 2);
        }
    }
    return true;
}

// The static profile: every (layer, expert) ranked by routing frequency over the TRAIN half of the bin
// (alternating 512-token blocks, exactly the split route_skew.py scores with).  Training on half and replaying a
// held-out block is the only version of this number that means anything: a profile built from every token scores
// its own training tokens and reads ~10 points better.  Filling the cache with the top `slots` of this ranking is
// what "seed the profile from ds4routes.bin" means in the phase-4 plan.
std::vector<std::pair<int32_t, int32_t>> rank_profile(const std::string& path, std::string& err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { err = "cannot open " + path; return {}; }
    const int64_t bytes = (int64_t) in.tellg();
    in.seekg(0);
    std::vector<uint16_t> all((size_t) (bytes / 2));
    in.read((char*) all.data(), bytes);
    const int64_t per_batch = kBatch * kLayers * kTopK;   // records in one 512-token block
    std::map<std::pair<int, int>, int64_t> cnt;
    for (size_t i = 0; i + 1 < all.size(); i += 2) {
        const int64_t rec = (int64_t) i / 2;
        const int64_t token = (rec / per_batch) * kBatch + (rec % (kBatch * kTopK)) / kTopK;
        if ((token / kBatch) % 2 != 0) continue;   // the held-out half
        cnt[{all[i], all[i + 1]}]++;
    }
    std::vector<std::pair<int32_t, int32_t>> ranked;
    ranked.reserve(cnt.size());
    for (auto& le : cnt) ranked.push_back({le.first.first, le.first.second});
    std::sort(ranked.begin(), ranked.end(), [&](const auto& x, const auto& y) {
        const int64_t cx = cnt[{x.first, x.second}], cy = cnt[{y.first, y.second}];
        return cx != cy ? cx > cy : x < y;
    });
    return ranked;
}

// ================================ blob source ================================
//
// One routed expert = the three GGUF role slices back to back, [gate rows | up rows | down rows] (the
// native_expert.hpp layout).  The model is mmap'd by GgufModel, so assembling a blob is three memcpys of pages
// the OS already has.
class BlobSource {
public:
    bool open(const std::string& gguf, const cpu::NativeFmt& f, std::string& err) {
        model_.reset(new strata::GgufModel(strata::gguf_split_paths(gguf)));
        gu_stride_ = (size_t) f.up_off;
        d_stride_ = (size_t) f.bytes - f.down_off;
        blob_ = (size_t) f.bytes;
        const char* roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) src_[r].assign(kLayers, nullptr);
        for (int l = 0; l < kLayers; ++l)
            for (int r = 0; r < 3; ++r) {
                char name[64];
                std::snprintf(name, sizeof name, kVerifyGguf, l, roles[r]);
                size_t at = 0;
                const strata::TensorInfo* ti = model_->find(name, &at);
                if (!ti) { err = std::string("no tensor ") + name; return false; }
                if (ti->type != (r == 2 ? DN : GU)) {
                    err = std::string(name) + " has type " + ti->type_name();
                    return false;
                }
                src_[r][l] = model_->shard(at).tensor_data(*ti);
            }
        return true;
    }
    size_t blob_bytes() const { return blob_; }
    void read(int layer, int expert, uint8_t* dst) const {
        std::memcpy(dst, src_[0][layer] + (size_t) expert * gu_stride_, gu_stride_);
        std::memcpy(dst + gu_stride_, src_[1][layer] + (size_t) expert * gu_stride_, gu_stride_);
        std::memcpy(dst + 2 * gu_stride_, src_[2][layer] + (size_t) expert * d_stride_, d_stride_);
    }

private:
    std::unique_ptr<strata::GgufModel> model_;
    std::vector<const uint8_t*> src_[3];
    size_t gu_stride_ = 0, d_stride_ = 0, blob_ = 0;
};

// A host arena of the replay's working set: every (layer, expert) the token range routes, in first-use order, up
// to the byte budget; anything past it is read straight from the GGUF on demand (the file tier).
struct Arena {
    void* base = nullptr;
    uint64_t bytes = 0;
    bool pinned = false;
    std::string backing;
    std::unique_ptr<strata::core::PinnedArena> owner;
    std::vector<int32_t> slot_of;    // kLayers*kExperts -> arena index, or -1
    int64_t used = 0, blob = 0;
    int64_t file_tier = 0;

    const uint8_t* ptr(int layer, int expert) const {
        const int32_t s = slot_of[(size_t) layer * kExperts + expert];
        return s < 0 ? nullptr : (const uint8_t*) base + (size_t) s * blob;
    }
    void close() { owner.reset(); base = nullptr; }
};

bool build_arena(Arena& ar, const Args& a, BlobSource& src, const Routes& r, std::string& err) {
    ar.blob = (int64_t) src.blob_bytes();
    ar.slot_of.assign((size_t) kLayers * kExperts, -1);
    const double budget_gib = std::min(a.arena_gib, a.max_arena_gib);
    const int64_t cap = (int64_t) (budget_gib * 1073741824.0 / (double) ar.blob);
    std::vector<std::pair<uint8_t, uint8_t>> first_use;
    for (int64_t t = 0; t < r.count; ++t) {
        const uint16_t* ids = r.tok(r.offset + t);
        for (int i = 0; i < kLayers * kTopK; ++i) {
            const int l = ids[i * 2], e = ids[i * 2 + 1];
            if (l >= kLayers || e >= kExperts) { err = "route out of range"; return false; }
            if (ar.slot_of[(size_t) l * kExperts + e] >= 0) continue;
            if ((int64_t) first_use.size() >= cap) { ++ar.file_tier; continue; }
            ar.slot_of[(size_t) l * kExperts + e] = (int32_t) first_use.size();
            first_use.push_back({(uint8_t) l, (uint8_t) e});
        }
    }
    ar.used = (int64_t) first_use.size();
    ar.bytes = (uint64_t) ar.used * (uint64_t) ar.blob;
    const int64_t avail = mem_available_gib();
    if (avail >= 0 && (int64_t) (ar.bytes / 1073741824ull) + a.mem_floor_gib > avail) {
        err = "an arena of " + std::to_string(ar.bytes / 1073741824ull) + " GiB would leave less than " +
              std::to_string(a.mem_floor_gib) + " GiB (MemAvailable " + std::to_string(avail) + " GiB)";
        return false;
    }
    if (ar.bytes == 0) { ar.backing = "empty"; return true; }
    if (a.pin) {
        ar.owner.reset(new strata::core::PinnedArena(ar.bytes));
        if (!ar.owner->valid()) { err = "pinned arena allocation failed"; return false; }
        ar.base = ar.owner->base;
        ar.pinned = true;
        ar.backing = "pinned: " + ar.owner->note;
    } else {
        void* p = nullptr;
        if (posix_memalign(&p, 4096, ar.bytes) != 0 || !p) { err = "arena allocation failed"; return false; }
        ar.base = p;
        ar.backing = "anonymous";
    }
    int nt = a.threads > 0 ? a.threads : 8;
    nt = std::min<int>(nt, 16);
    const double t0 = now_ms();
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int c = 0; c < nt; ++c)
        th.emplace_back([&]() {
            for (;;) {
                const int64_t i = next.fetch_add(1);
                if (i >= ar.used) break;
                src.read(first_use[(size_t) i].first, first_use[(size_t) i].second,
                         (uint8_t*) ar.base + (size_t) i * ar.blob);
            }
        });
    for (auto& t : th) t.join();
    const double s = (now_ms() - t0) / 1000.0;
    std::printf("arena: %lld experts (%.2f GiB, %s) read in %.1f s (%.2f GiB/s); %lld on the file tier\n",
                (long long) ar.used, (double) ar.bytes / 1073741824.0, ar.backing.c_str(), s,
                s > 0 ? (double) ar.bytes / 1073741824.0 / s : 0.0, (long long) ar.file_tier);
    return true;
}

// ================================ the synthetic GPU gap ================================
//
// The dense+attention work the concurrent Phase-3 slice owns, as a spin of the measured length (moe_gap.cu).
extern "C" void moe_gap_launch(unsigned long long cycles, void* stream);

struct GpuGap {
    bool armed = false;
    unsigned long long cycles = 0;
    void calibrate(double ms_target, void* s) {
        if (ms_target <= 0) return;
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "gap e0");
        ck(cudaEventCreate(&e1), "gap e1");
        const unsigned long long probe = 20000000ull;
        ck(cudaEventRecord(e0, (cudaStream_t) s), "gap rec0");
        moe_gap_launch(probe, s);
        ck(cudaEventRecord(e1, (cudaStream_t) s), "gap rec1");
        ck(cudaEventSynchronize(e1), "gap sync");
        float t = 0;
        ck(cudaEventElapsedTime(&t, e0, e1), "gap elapsed");
        const double per_ms = (double) probe / (double) t;
        cycles = (unsigned long long) (ms_target * per_ms);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        armed = true;
    }
    void launch(void* s) const {
        if (armed) moe_gap_launch(cycles, s);
    }
};

struct Stats {
    int64_t hits = 0, cpu = 0, pcie = 0;
    int64_t hit_bytes = 0, cpu_bytes = 0, pcie_bytes = 0;
    double gap_ms = 0, hit_ms = 0, pcie_ms = 0, cpu_ms = 0, wall_ms = 0;
    void add(const Stats& o) {
        hits += o.hits; cpu += o.cpu; pcie += o.pcie;
        hit_bytes += o.hit_bytes; cpu_bytes += o.cpu_bytes; pcie_bytes += o.pcie_bytes;
        gap_ms += o.gap_ms; hit_ms += o.hit_ms; pcie_ms += o.pcie_ms; cpu_ms += o.cpu_ms; wall_ms += o.wall_ms;
    }
};

// ================================ the engine ================================
//
// The four tiers from the plan in one object: the VRAM cache (strata::core::ExpertCache, 6.75 MiB slots,
// open_sized), the CPU pool on the misses, the GpuPlanSink PCIe share computed on the GPU in parallel, and the
// host arena the misses are read from.  Admission is compulsory-miss, as in Strata: a miss is computed on the CPU
// this token and its slot filled for the next one.
class Engine {
public:
    bool init(const Args& a, Arena& arena, BlobSource& src, const cpu::NativeFmt& f, std::string& err);
    void prepare_buffers();
    Stats run_token(const Routes& r, int64_t t, const std::vector<float>& x_tok, bool force_cpu);
    bool prefill(const std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t slots, std::string& err,
                 int64_t& resident);
    bool correctness(const Args& a, const Routes& r, BlobSource& src, std::string& err);
    void close();

    cpu::ExpertPool& pool() { return *pool_; }
    int64_t admitted() const { return admitted_; }

private:
    Stats layer(int l, const uint16_t* ids, const float* x, bool force_cpu);

    Args a_;
    Arena* arena_ = nullptr;
    BlobSource* src_ = nullptr;
    cpu::NativeFmt f_;
    int64_t blob_ = 0;
    std::unique_ptr<cpu::ExpertPool> pool_;
    strata::core::ExpertCache cache_;
    GpuGap gap_;
    strata::kernels::NativeExpertLayout gl_;
    void* s_ = nullptr;
    void *d_x_ = nullptr, *d_xq_ = nullptr, *d_parts_ = nullptr, *d_grp_ptr_ = nullptr, *d_grp_start_ = nullptr,
         *d_ngroups_ = nullptr, *d_ent_dst_ = nullptr, *d_ent_tok_ = nullptr, *d_scratch_ = nullptr,
         *d_stage_ = nullptr, *d_corr_ = nullptr;
    cudaEvent_t ev_[4] = {};
    std::vector<uint8_t> nact_, hq_;
    std::vector<float> parts_;
    std::vector<cpu::ExpertJobMulti> jobs_;
    std::vector<uint8_t> ftmp_;
    int64_t admitted_ = 0;
};

bool Engine::init(const Args& a, Arena& arena, BlobSource& src, const cpu::NativeFmt& f, std::string& err) {
    a_ = a;
    arena_ = &arena;
    src_ = &src;
    f_ = f;
    blob_ = (int64_t) f.bytes;
    pool_.reset(new cpu::ExpertPool(a.threads > 0 ? a.threads : 0));
    std::printf("pool: %d workers (host %s)\n", pool_->workers(), pool_->host_works() ? "works" : "waits");
    if (!a.cpu_only) {
        gl_ = strata::kernels::native_expert_layout(GU, DN, H, FF);
        if (!strata::kernels::native_expert_supported(GU, DN, H, FF)) {
            err = "native_expert_grouped has no kernel for IQ2_XXS/Q2_K at 4096/2048";
            return false;
        }
        cudaStream_t stream = nullptr;
        ck(cudaStreamCreate(&stream), "stream");
        s_ = stream;
        ck(cudaMalloc(&d_x_, (size_t) H * 4), "d_x");
        ck(cudaMalloc(&d_xq_, strata::kernels::native_q8_1_bytes((int) H, 1)), "d_xq");
        ck(cudaMalloc(&d_parts_, (size_t) kMaxParts * H * 4), "d_parts");
        ck(cudaMalloc(&d_grp_ptr_, sizeof(unsigned long long) * kMaxParts), "d_grp_ptr");
        ck(cudaMalloc(&d_grp_start_, sizeof(int32_t) * (kMaxParts + 1)), "d_grp_start");
        ck(cudaMalloc(&d_ngroups_, sizeof(int32_t)), "d_ngroups");
        ck(cudaMalloc(&d_ent_dst_, sizeof(int32_t) * kMaxParts), "d_ent_dst");
        ck(cudaMalloc(&d_ent_tok_, sizeof(int32_t) * kMaxParts), "d_ent_tok");
        ck(cudaMalloc(&d_scratch_, strata::kernels::native_expert_scratch_bytes(kMaxParts, FF)), "d_scratch");
        ck(cudaMalloc(&d_stage_, (size_t) kMaxParts * blob_), "d_stage");
        ck(cudaMalloc(&d_corr_, (size_t) kTopK * blob_), "d_corr");
        for (int i = 0; i < 4; ++i) ck(cudaEventCreate(&ev_[i]), "event");
        gap_.calibrate(a.gap_ms, s_);
        std::printf("gpu: gap %.3f ms/layer (%llu cycles), PCIe staging %d x %.2f MiB\n", a.gap_ms,
                    (unsigned long long) gap_.cycles, kMaxParts, (double) blob_ / 1048576.0);
    }
    if (!a.no_cache && !a.cpu_only) {
        std::vector<int64_t> sizes((size_t) a.slots, blob_);
        if (!cache_.open_sized(sizes, kLayers, kExperts, err)) return false;
        std::printf("cache: %lld slots of %.2f MiB = %.2f GiB\n", (long long) cache_.full_slots(),
                    (double) blob_ / 1048576.0, (double) cache_.full_bytes() / 1073741824.0);
    }
    return true;
}

void Engine::prepare_buffers() {
    nact_.resize(cpu::kNativeActBytes);
    hq_.resize(cpu::kNativeHBytes);
    parts_.assign((size_t) kMaxParts * H, 0.0f);
    jobs_.resize(kTopK);
}

bool Engine::prefill(const std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t slots, std::string& err,
                     int64_t& resident) {
    resident = 0;
    if (!cache_.valid()) return true;
    const int64_t want = std::min<int64_t>(slots, cache_.full_slots());
    std::vector<uint8_t> tmp;
    for (const auto& le : ranked) {
        if (resident >= want) break;
        const int32_t s = cache_.admit(le.first, le.second);
        if (s < 0) break;
        const uint8_t* p = arena_->ptr(le.first, le.second);
        if (!p) { tmp.resize((size_t) blob_); src_->read(le.first, le.second, tmp.data()); p = tmp.data(); }
        if (!cache_.fill_slot_blocking(s, p, err, blob_)) return false;
        ++resident;
    }
    return true;
}

Stats Engine::layer(int l, const uint16_t* ids, const float* x, bool force_cpu) {
    Stats st;
    const double t0 = now_ms();
    const bool gpu = !a_.cpu_only;

    // `ids` points at this layer's six (layer, expert) pairs; the expert of router index k is ids[2k+1].
    int32_t hit_i[kMaxParts], cpu_i[kMaxParts], pcie_i[kMaxParts], slot_i[kMaxParts] = {};
    int nh = 0, nc = 0, np = 0;
    for (int k = 0; k < kTopK; ++k) {
        const int32_t s = (!force_cpu && cache_.valid()) ? cache_.slot_of(l, ids[2 * k + 1]) : -1;
        slot_i[k] = s;
        if (s >= 0) hit_i[nh++] = k;
        else cpu_i[nc++] = k;
    }
    if (!force_cpu && !a_.no_pcie && cache_.valid()) {
        const int budget = (int) std::lround((double) nc * a_.pcie_frac);
        for (int i = 0; i < nc && np < budget; ++i) {
            const int k = cpu_i[nc - 1 - i];
            if (arena_->ptr(l, ids[2 * k + 1])) pcie_i[np++] = k;   // the last misses in routing order
        }
    }
    int cpu_keep[kMaxParts], nk = 0;
    for (int i = 0; i < nc; ++i) {
        bool in_pcie = false;
        for (int j = 0; j < np; ++j) in_pcie = in_pcie || pcie_i[j] == cpu_i[i];
        if (!in_pcie) cpu_keep[nk++] = cpu_i[i];
    }

    cpu::native_quant_act(f_, x, nact_.data());
    if (gpu) {
        ck(cudaMemcpyAsync(d_x_, x, (size_t) H * 4, cudaMemcpyHostToDevice, (cudaStream_t) s_), "h2d x");
        strata::kernels::native_quantize_q8_1((const float*) d_x_, d_xq_, (int) H, 1, s_);
        ck(cudaEventRecord(ev_[0], (cudaStream_t) s_), "rec gap0");
        gap_.launch(s_);
        ck(cudaEventRecord(ev_[1], (cudaStream_t) s_), "rec gap1");
    }

    int ev_hit = -1, ev_pcie = -1;
    if (nh > 0 && gpu) {
        std::vector<unsigned long long> ptr((size_t) nh);
        std::vector<int32_t> start((size_t) nh + 1), dst((size_t) nh, 0), tok((size_t) nh, 0);
        for (int i = 0; i < nh; ++i) {
            ptr[i] = (unsigned long long) cache_.device_slot(slot_i[hit_i[i]]);
            start[i] = i;
            dst[i] = hit_i[i];
        }
        start[nh] = nh;
        const int32_t one = nh;
        ck(cudaMemcpyAsync(d_grp_ptr_, ptr.data(), sizeof(unsigned long long) * nh, cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "grp_ptr");
        ck(cudaMemcpyAsync(d_grp_start_, start.data(), sizeof(int32_t) * (nh + 1), cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "grp_start");
        ck(cudaMemcpyAsync(d_ngroups_, &one, sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) s_), "ngroups");
        ck(cudaMemcpyAsync(d_ent_dst_, dst.data(), sizeof(int32_t) * nh, cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "ent_dst");
        ck(cudaMemcpyAsync(d_ent_tok_, tok.data(), sizeof(int32_t) * nh, cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "ent_tok");
        strata::kernels::native_expert_grouped(gl_, (const unsigned long long*) d_grp_ptr_,
                                               (const int32_t*) d_grp_start_, (const int32_t*) d_ngroups_,
                                               (const int32_t*) d_ent_dst_, (const int32_t*) d_ent_tok_,
                                               kMaxParts, nh, d_xq_, d_scratch_, (float*) d_parts_, s_, nh);
        ck(cudaEventRecord(ev_[2], (cudaStream_t) s_), "rec hit1");
        ev_hit = 2;
    }
    if (np > 0 && gpu) {
        std::vector<unsigned long long> ptr((size_t) np);
        std::vector<int32_t> start((size_t) np + 1), dst((size_t) np, 0), tok((size_t) np, 0);
        for (int i = 0; i < np; ++i) {
            const uint8_t* p = arena_->ptr(l, ids[2 * pcie_i[i] + 1]);
            ck(cudaMemcpyAsync((uint8_t*) d_stage_ + (size_t) i * blob_, p, (size_t) blob_,
                               cudaMemcpyHostToDevice, (cudaStream_t) s_), "pcie dma");
            ptr[i] = (unsigned long long) d_stage_ + (size_t) i * blob_;
            start[i] = i;
            dst[i] = pcie_i[i];
        }
        start[np] = np;
        const int32_t one = np;
        ck(cudaMemcpyAsync(d_grp_ptr_, ptr.data(), sizeof(unsigned long long) * np, cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "pcie grp_ptr");
        ck(cudaMemcpyAsync(d_grp_start_, start.data(), sizeof(int32_t) * (np + 1), cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "pcie grp_start");
        ck(cudaMemcpyAsync(d_ngroups_, &one, sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) s_),
           "pcie ngroups");
        ck(cudaMemcpyAsync(d_ent_dst_, dst.data(), sizeof(int32_t) * np, cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "pcie ent_dst");
        ck(cudaMemcpyAsync(d_ent_tok_, tok.data(), sizeof(int32_t) * np, cudaMemcpyHostToDevice,
                           (cudaStream_t) s_), "pcie ent_tok");
        strata::kernels::native_expert_grouped(gl_, (const unsigned long long*) d_grp_ptr_,
                                               (const int32_t*) d_grp_start_, (const int32_t*) d_ngroups_,
                                               (const int32_t*) d_ent_dst_, (const int32_t*) d_ent_tok_,
                                               kMaxParts, np, d_xq_, d_scratch_, (float*) d_parts_, s_, np);
        ck(cudaEventRecord(ev_[3], (cudaStream_t) s_), "rec pcie1");
        ev_pcie = 3;
    }

    // the CPU pool on what is left, WHILE the GPU works
    if (nk > 0) {
        std::vector<uint8_t> fbuf;
        for (int i = 0; i < nk; ++i) {
            const uint8_t* p = arena_->ptr(l, ids[2 * cpu_keep[i] + 1]);
            if (!p) {
                if (fbuf.empty()) fbuf.resize((size_t) blob_ * (size_t) nk);
                src_->read(l, ids[2 * cpu_keep[i] + 1], fbuf.data() + (size_t) i * blob_);
                p = fbuf.data() + (size_t) i * blob_;
            }
            jobs_[i].blob = p;
            jobs_[i].nt = 1;
            for (int t = 0; t < cpu::MAXT; ++t) { jobs_[i].nact[t] = nullptr; jobs_[i].out[t] = nullptr; }
            jobs_[i].nact[0] = nact_.data();
            jobs_[i].out[0] = parts_.data() + (size_t) cpu_keep[i] * H;
        }
        const double c0 = now_ms();
        pool_->run_split_multi_native(f_, jobs_.data(), nk);
        st.cpu_ms = now_ms() - c0;
    }

    if (gpu) {
        ck(cudaStreamSynchronize((cudaStream_t) s_), "layer sync");
        float ms = 0;
        if (gap_.armed) {
            ck(cudaEventElapsedTime(&ms, ev_[0], ev_[1]), "gap elapsed");
            st.gap_ms = ms;
        }
        if (ev_hit > 0) {
            ck(cudaEventElapsedTime(&ms, ev_[1], ev_[ev_hit]), "hit elapsed");
            st.hit_ms = ms;
        }
        if (ev_pcie > 0) {
            const int from = ev_hit > 0 ? ev_hit : 1;
            ck(cudaEventElapsedTime(&ms, ev_[from], ev_[ev_pcie]), "pcie elapsed");
            st.pcie_ms = ms;
        }
        ck(cudaMemcpyAsync(parts_.data(), d_parts_, (size_t) kMaxParts * H * 4, cudaMemcpyDeviceToHost,
                           (cudaStream_t) s_), "d2h parts");
        ck(cudaStreamSynchronize((cudaStream_t) s_), "parts sync");
    }

    // admission: a miss takes a free slot and is filled for the NEXT token (compulsory-miss, no eviction)
    if (!force_cpu && cache_.valid()) {
        for (int i = 0; i < nc; ++i) {
            const uint8_t* p = arena_->ptr(l, ids[2 * cpu_i[i] + 1]);
            if (!p) continue;
            const int32_t s = cache_.admit(l, ids[2 * cpu_i[i] + 1]);
            if (s < 0) continue;
            std::string e;
            if (!cache_.fill_slot_blocking(s, p, e, blob_)) {
                std::fprintf(stderr, "moe_replay: fill_slot: %s\n", e.c_str());
                std::exit(2);
            }
            ++admitted_;
        }
    }

    st.hits = nh;
    st.pcie = np;
    st.cpu = nk;
    st.hit_bytes = (int64_t) nh * blob_;
    st.pcie_bytes = (int64_t) np * blob_;
    st.cpu_bytes = (int64_t) nk * blob_;
    st.wall_ms = now_ms() - t0;
    return st;
}

Stats Engine::run_token(const Routes& r, int64_t t, const std::vector<float>& x_tok, bool force_cpu) {
    Stats sum;
    const uint16_t* ids = r.tok(t);
    for (int l = 0; l < kLayers; ++l) {
        const uint16_t* dst = ids + (size_t) l * kTopK * 2;
        sum.add(layer(l, dst, x_tok.data() + (size_t) l * H, force_cpu));
    }
    return sum;
}

// ================================ correctness ================================
//
// The engine's arithmetic against a dequant+F32 reference of the SAME blob.  The reference dequantizes the expert
// weights with ggml's own to_float and the engine's own quantized activation (Q8_K for both roles), so what is
// measured is the kernels' arithmetic and not the 8-bit activation rounding.  The pure-float error is printed
// alongside (Phase 2 measured 1.9e-2 on the fused expert), because that is what a user would see against a
// non-quantized engine.
struct RefWeights {
    std::vector<float> G, U, D;
};

void dequant_weights(const uint8_t* blob, const cpu::NativeFmt& f, RefWeights& w) {
    w.G.resize((size_t) FF * H);
    w.U.resize((size_t) FF * H);
    w.D.resize((size_t) H * FF);
    ggml_get_type_traits((ggml_type) GU)->to_float(blob, w.G.data(), FF * H);
    ggml_get_type_traits((ggml_type) GU)->to_float(blob + f.up_off, w.U.data(), FF * H);
    ggml_get_type_traits((ggml_type) DN)->to_float(blob + f.down_off, w.D.data(), H * FF);
}

void matvec(const std::vector<float>& w, int64_t rows, int64_t cols, const float* x, float* y) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* wr = w.data() + (size_t) r * cols;
        double s = 0;
        for (int64_t i = 0; i < cols; ++i) s += (double) wr[i] * x[i];
        y[r] = (float) s;
    }
}

void ref_expert(const RefWeights& w, const float* x, float* out) {
    std::vector<float> g((size_t) FF), u((size_t) FF), h((size_t) FF);
    matvec(w.G, FF, H, x, g.data());
    matvec(w.U, FF, H, x, u.data());
    for (int64_t r = 0; r < FF; ++r) h[(size_t) r] = silu(g[(size_t) r]) * u[(size_t) r];
    matvec(w.D, H, FF, h.data(), out);
}

bool Engine::correctness(const Args& a, const Routes& r, BlobSource& src, std::string& err) {
    (void) err;
    prepare_buffers();
    const int64_t ntok = std::max<int64_t>(1, std::min<int64_t>(a.correctness, 1));
    std::vector<float> x((size_t) H), xd((size_t) H);
    std::vector<uint8_t> act(cpu::kNativeActBytes);
    std::vector<float> e_eng((size_t) kTopK * H), e_ref((size_t) kTopK * H), e_flo((size_t) kTopK * H);
    std::vector<float> ref_moe((size_t) H), eng_moe((size_t) H), flo_moe((size_t) H);
    std::mt19937 rng((uint32_t) a.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const double inv = 1.0 / std::sqrt((double) H);
    double worst_expert = 0, worst_moe = 0, worst_flo_expert = 0, worst_flo_moe = 0, worst_gpu = 0;
    int64_t printed = 0;

    for (int64_t t = 0; t < ntok; ++t) {
        for (int l = 0; l < kLayers; ++l) {
            const uint16_t* ids = r.tok(r.offset + t) + (size_t) l * kTopK * 2;
            for (int64_t i = 0; i < H; ++i) x[(size_t) i] = (float) (nd(rng) * inv);
            cpu::native_quant_act(f_, x.data(), act.data());
            deq_q8_k(act.data(), H, xd.data());
            std::fill(ref_moe.begin(), ref_moe.end(), 0.0f);
            std::fill(eng_moe.begin(), eng_moe.end(), 0.0f);
            std::fill(flo_moe.begin(), flo_moe.end(), 0.0f);
            std::vector<std::vector<uint8_t>> blobs((size_t) kTopK, std::vector<uint8_t>((size_t) blob_));
            for (int k = 0; k < kTopK; ++k) {
                const int e = ids[k * 2 + 1];
                src.read(l, e, blobs[(size_t) k].data());
                RefWeights w;
                dequant_weights(blobs[(size_t) k].data(), f_, w);
                // the reference: dequantize the engine's OWN activation, then the F32 matmul
                {
                    std::vector<float> g((size_t) FF), u((size_t) FF), h((size_t) FF);
                    matvec(w.G, FF, H, xd.data(), g.data());
                    matvec(w.U, FF, H, xd.data(), u.data());
                    for (int64_t j = 0; j < FF; ++j) h[(size_t) j] = silu(g[(size_t) j]) * u[(size_t) j];
                    cpu::native_quant_h(f_, h.data(), hq_.data());
                    deq_q8_k(hq_.data(), FF, h.data());
                    matvec(w.D, H, FF, h.data(), &e_ref[(size_t) k * H]);
                }
                // the engine, one expert through the pool's own native path
                jobs_[0].blob = blobs[(size_t) k].data();
                jobs_[0].nt = 1;
                for (int j = 0; j < cpu::MAXT; ++j) { jobs_[0].nact[j] = nullptr; jobs_[0].out[j] = nullptr; }
                jobs_[0].nact[0] = act.data();
                jobs_[0].out[0] = &e_eng[(size_t) k * H];
                pool_->run_split_multi_native(f_, jobs_.data(), 1);
                // and the pure-float reference, for the record
                ref_expert(w, x.data(), &e_flo[(size_t) k * H]);
                const double rex = rel(&e_eng[(size_t) k * H], &e_ref[(size_t) k * H], H);
                const double rfl = rel(&e_eng[(size_t) k * H], &e_flo[(size_t) k * H], H);
                worst_expert = std::max(worst_expert, rex);
                worst_flo_expert = std::max(worst_flo_expert, rfl);
                if (printed < a.correctness_experts) {
                    std::printf("  L%-2d e%-3d expert rel %.3e (pure-float %.3e)\n", l, e, rex, rfl);
                    ++printed;
                }
                for (int64_t i = 0; i < H; ++i) {
                    const float wgt = 1.0f / (float) kTopK;
                    ref_moe[(size_t) i] += wgt * e_ref[(size_t) k * H + i];
                    eng_moe[(size_t) i] += wgt * e_eng[(size_t) k * H + i];
                    flo_moe[(size_t) i] += wgt * e_flo[(size_t) k * H + i];
                }
            }
            worst_moe = std::max(worst_moe, rel(eng_moe.data(), ref_moe.data(), H));
            worst_flo_moe = std::max(worst_flo_moe, rel(eng_moe.data(), flo_moe.data(), H));
            // the GPU grouped path on a sample of layers: same six experts, uploaded to the card
            if (!a.cpu_only && (l % 11 == 0)) {
                for (int k = 0; k < kTopK; ++k)
                    ck(cudaMemcpy((uint8_t*) d_corr_ + (size_t) k * blob_, blobs[(size_t) k].data(),
                                  (size_t) blob_, cudaMemcpyHostToDevice), "corr h2d");
                ck(cudaMemcpyAsync(d_x_, x.data(), (size_t) H * 4, cudaMemcpyHostToDevice, (cudaStream_t) s_),
                   "corr x");
                strata::kernels::native_quantize_q8_1((const float*) d_x_, d_xq_, (int) H, 1, s_);
                std::vector<unsigned long long> ptr((size_t) kTopK);
                std::vector<int32_t> start((size_t) kTopK + 1), dst((size_t) kTopK), tok((size_t) kTopK, 0);
                for (int k = 0; k < kTopK; ++k) {
                    ptr[(size_t) k] = (unsigned long long) d_corr_ + (size_t) k * blob_;
                    start[(size_t) k] = k;
                    dst[(size_t) k] = k;
                }
                start[kTopK] = kTopK;
                const int32_t one = kTopK;
                ck(cudaMemcpyAsync(d_grp_ptr_, ptr.data(), sizeof(unsigned long long) * kTopK,
                                   cudaMemcpyHostToDevice, (cudaStream_t) s_), "corr grp_ptr");
                ck(cudaMemcpyAsync(d_grp_start_, start.data(), sizeof(int32_t) * (kTopK + 1),
                                   cudaMemcpyHostToDevice, (cudaStream_t) s_), "corr grp_start");
                ck(cudaMemcpyAsync(d_ngroups_, &one, sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) s_),
                   "corr ngroups");
                ck(cudaMemcpyAsync(d_ent_dst_, dst.data(), sizeof(int32_t) * kTopK, cudaMemcpyHostToDevice,
                                   (cudaStream_t) s_), "corr ent_dst");
                ck(cudaMemcpyAsync(d_ent_tok_, tok.data(), sizeof(int32_t) * kTopK, cudaMemcpyHostToDevice,
                                   (cudaStream_t) s_), "corr ent_tok");
                strata::kernels::native_expert_grouped(gl_, (const unsigned long long*) d_grp_ptr_,
                                                       (const int32_t*) d_grp_start_, (const int32_t*) d_ngroups_,
                                                       (const int32_t*) d_ent_dst_, (const int32_t*) d_ent_tok_,
                                                       kMaxParts, kTopK, d_xq_, d_scratch_, (float*) d_parts_, s_,
                                                       kTopK);
                ck(cudaStreamSynchronize((cudaStream_t) s_), "corr sync");
                ck(cudaMemcpyAsync(parts_.data(), d_parts_, (size_t) kMaxParts * H * 4,
                                   cudaMemcpyDeviceToHost, (cudaStream_t) s_), "corr d2h");
                for (int k = 0; k < kTopK; ++k)
                    worst_gpu = std::max(worst_gpu, rel(&parts_[(size_t) dst[(size_t) k] * H],
                                                        &e_ref[(size_t) k * H], H));
            }
        }
    }
    check(worst_expert < 1e-3, "CPU expert output vs dequant+F32 (worst)", worst_expert, 1e-3);
    check(worst_moe < 1e-3, "per-layer MoE output vs dequant+F32 (worst)", worst_moe, 1e-3);
    if (!a.cpu_only) check(worst_gpu < 1e-3, "GPU grouped expert output vs dequant+F32", worst_gpu, 1e-3);
    std::printf("  pure-float (8-bit activation rounding, not kernel error): expert %.3e, MoE %.3e\n",
                worst_flo_expert, worst_flo_moe);
    return true;
}

void Engine::close() {
    if (!a_.cpu_only) {
        for (auto& e : ev_) if (e) cudaEventDestroy(e);
        void* bufs[] = {d_x_, d_xq_, d_parts_, d_grp_ptr_, d_grp_start_, d_ngroups_,
                        d_ent_dst_, d_ent_tok_, d_scratch_, d_stage_, d_corr_};
        for (void* p : bufs) if (p) cudaFree(p);
        if (s_) cudaStreamDestroy((cudaStream_t) s_);
        d_x_ = d_xq_ = d_parts_ = d_grp_ptr_ = d_grp_start_ = d_ngroups_ = nullptr;
        d_ent_dst_ = d_ent_tok_ = d_scratch_ = d_stage_ = d_corr_ = nullptr;
        s_ = nullptr;
    }
}

}  // namespace

// ================================ main ================================
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Args a;
    std::string err;
    if (!parse_args(argc, argv, a, err)) { std::printf("args: %s\n", err.c_str()); usage(); return 2; }

    cpu::NativeFmt f;
    if (!cpu::native_fmt(GU, DN, H, FF, f, err)) { std::printf("native_fmt: %s\n", err.c_str()); return 1; }
    std::printf("expert: %lld/%lld %s/%s, blob %zu B (%.2f MiB)\n", (long long) H, (long long) FF,
                ggml_type_name((ggml_type) GU), ggml_type_name((ggml_type) DN), f.bytes,
                (double) f.bytes / 1048576.0);

    Routes r;
    if (!load_routes(a.routes, a.offset, a.tokens, r, err)) { std::printf("routes: %s\n", err.c_str()); return 1; }
    std::printf("routes: %lld tokens in file, replaying [%lld, %lld)\n", (long long) r.n_tokens,
                (long long) r.offset, (long long) (r.offset + r.count));

    BlobSource src;
    if (!src.open(a.gguf, f, err)) { std::printf("gguf: %s\n", err.c_str()); return 1; }

    Arena arena;
    if (a.arena_gib > 0) {
        if (!build_arena(arena, a, src, r, err)) { std::printf("arena: %s\n", err.c_str()); return 1; }
    } else {
        arena.blob = (int64_t) f.bytes;
        arena.slot_of.assign((size_t) kLayers * kExperts, -1);
        std::printf("arena: disabled (every blob read from the GGUF)\n");
    }

    Engine eng;
    if (!eng.init(a, arena, src, f, err)) { std::printf("engine: %s\n", err.c_str()); return 1; }
    eng.prepare_buffers();

    if (a.correctness > 0) {
        if (!eng.correctness(a, r, src, err)) { std::printf("correctness: %s\n", err.c_str()); return 1; }
        eng.close();
        arena.close();
        std::printf("moe_replay: %d failures\n", g_fail);
        return g_fail ? 1 : 0;
    }

    if (!a.no_cache && !a.cpu_only) {
        const std::vector<std::pair<int32_t, int32_t>> ranked = rank_profile(a.routes, err);
        if (ranked.empty()) { std::printf("profile: %s\n", err.c_str()); return 1; }
        int64_t resident = 0;
        if (!eng.prefill(ranked, a.slots, err, resident)) { std::printf("prefill: %s\n", err.c_str()); return 1; }
        std::printf("cache prefilled: %lld of %lld slots from the routing profile\n", (long long) resident,
                    (long long) a.slots);
    }

    std::mt19937 rng((uint32_t) a.seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const double inv = 1.0 / std::sqrt((double) H);
    std::vector<float> x_tok((size_t) kLayers * H);

    Stats total;
    std::vector<Stats> per_tok;
    per_tok.reserve((size_t) r.count);
    const double t_start = now_ms();
    for (int64_t t = 0; t < r.count; ++t) {
        for (auto& v : x_tok) v = (float) (nd(rng) * inv);
        const Stats s = eng.run_token(r, r.offset + t, x_tok, false);
        total.add(s);
        per_tok.push_back(s);
    }
    const double wall = now_ms() - t_start;

    const double tok_ms = wall / (double) r.count;
    const double gpu_busy = (total.gap_ms + total.hit_ms + total.pcie_ms) / (double) r.count;
    const double cpu_busy = total.cpu_ms / (double) r.count;
    const double lookups = (double) r.count * kLayers * kTopK;
    const double non_moe = 17.6;   // Phase 0 config-B GPU kernels incl dense/attn, ms/token
    std::printf("\n== moe_replay: %lld tokens, %lld slots, pcie-frac %.2f, threads %d, gap %.2f ms/layer ==\n",
                (long long) r.count, (long long) a.slots, a.pcie_frac, eng.pool().workers(), a.gap_ms);
    if (a.cpu_only) std::printf("(CPU-ONLY: no cache, no PCIe share)\n");
    std::printf("lookups/token %d: hit %.1f%%, miss-CPU %.1f%%, miss-PCIe %.1f%%\n", kLayers * kTopK,
                100.0 * (double) total.hits / lookups, 100.0 * (double) total.cpu / lookups,
                100.0 * (double) total.pcie / lookups);
    std::printf("bytes/token: hit %.2f GiB, miss-CPU %.2f GiB, miss-PCIe %.2f GiB (total %.2f GiB)\n",
                (double) total.hit_bytes / r.count / 1073741824.0,
                (double) total.cpu_bytes / r.count / 1073741824.0,
                (double) total.pcie_bytes / r.count / 1073741824.0,
                (double) (total.hit_bytes + total.cpu_bytes + total.pcie_bytes) / r.count / 1073741824.0);
    std::printf("ms/token: MoE wall %.2f = gap %.2f + hit %.2f + miss-PCIe %.2f + miss-CPU %.2f\n", tok_ms,
                total.gap_ms / r.count, total.hit_ms / r.count, total.pcie_ms / r.count, cpu_busy);
    std::printf("          gpu busy %.2f, cpu busy %.2f, overlap idle (wall - max) %.2f\n", gpu_busy, cpu_busy,
                std::max(0.0, tok_ms - std::max(gpu_busy, cpu_busy)));
    // The gate's formula is MoE ms + the non-MoE GPU ms of the Phase-0 profile, so the arms run with --gap-ms 0:
    // `wall` IS the MoE engine's own time and adding 17.6 does not double count.  With a gap the injected GPU
    // work is already inside `wall`, so the same sum would count it twice; the with-gap arm is printed as an
    // overlap check instead (its wall against the coarse `max(gpu, cpu)` bound rather than a sum).
    if (a.gap_ms > 0) {
        std::printf("implied full-token (overlap): MoE wall %.2f already contains the injected %.2f ms/token of\n"
                    "          dense+attention; the serial rest of Phase 0's 17.6 (attn 1.6 + other 2.1) = 3.7\n",
                    tok_ms, total.gap_ms / r.count);
    } else {
        std::printf("implied full-token = MoE %.2f + non-MoE GPU %.2f = %.2f ms -> %.2f tok/s (bar 16.34, go 19.6)\n",
                    tok_ms, non_moe, tok_ms + non_moe, 1000.0 / (tok_ms + non_moe));
        std::printf("                    (MoE-only.  With the layer's GPU work overlapped as the doorbell does\n"
                    "                    it, add only Phase 0's serial attn+other, ~3.7 ms.)\n");
    }
    std::printf("admissions during the run: %lld\n", (long long) eng.admitted());

    if (!a.out_csv.empty()) {
        std::FILE* fcsv = std::fopen(a.out_csv.c_str(), "w");
        if (fcsv) {
            std::fprintf(fcsv, "token,hits,cpu,pcie,gap_ms,hit_ms,pcie_ms,cpu_ms,wall_ms\n");
            for (size_t i = 0; i < per_tok.size(); ++i) {
                const Stats& s = per_tok[i];
                std::fprintf(fcsv, "%zu,%lld,%lld,%lld,%.4f,%.4f,%.4f,%.4f,%.4f\n", i, (long long) s.hits,
                             (long long) s.cpu, (long long) s.pcie, s.gap_ms, s.hit_ms, s.pcie_ms, s.cpu_ms,
                             s.wall_ms);
            }
            std::fclose(fcsv);
            std::printf("wrote %s\n", a.out_csv.c_str());
        }
    }

    eng.close();
    arena.close();
    std::printf("moe_replay: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
