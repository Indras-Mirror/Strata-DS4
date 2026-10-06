// tools/ds4/ds4_moe.cpp - the DeepSeek-V4 routed-expert tier (slice ds4-moe).
//
// The body of `Ds4MoeTier`, extracted from `tools/ds4/moe_replay.cpp`'s `Engine` (the measured harness, FINDINGS
// s10-s12) so the real engine can use it once per layer.  Read the header for the contract, the invariant and the
// SwiGLU-clamp note; this file is the implementation and keeps the replay's per-layer shape, minus the replay's
// synthetic dense gap and its recorded-route table.
//
// TWO TU FLAVOURS OF ONE SOURCE.  Compiled without `DS4_MOE_CUDA` it contains no CUDA call at all and references
// no symbol of `strata_kernels` / `strata_engine`, so a binary that links it (this slice's gate) cannot
// initialise the driver - accidental or otherwise.  Compiled with `DS4_MOE_CUDA` it is the full three-tier
// implementation.  `tools/ds4/cmake/ds4_moe.cmake` builds both.
#include "ds4_moe.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>
#include <utility>

#include "strata/artifact/ds4_geometry.hpp"

#if defined(DS4_MOE_CUDA)
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#endif

namespace cpu = strata::kernels::cpu;

namespace strata::ds4 {

namespace {

constexpr int kMaxParts = 8;   ///< the grouped kernel's group/entry cap; top-k 6 needs no more
constexpr int kMaxTok = 4;     ///< tokens one run_multi call takes (draft verify: last token + up to 3 drafts)
constexpr int kMaxEnt = kMaxTok * 8;   ///< (expert, token) entries and distinct experts of one run_multi call
constexpr int kBatch = 512;    ///< a `ds4routes.bin` ubatch, in tokens (what route_probe tokenises)

[[maybe_unused]] double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[maybe_unused]] int64_t mem_available_gib() {
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    std::fclose(f);
    return kb < 0 ? -1 : (int64_t) (kb / (1024 * 1024));
}

/// One metadata integer.  The `deepseek4.*` geometry keys are one-element arrays in the converter's output, but a
/// scalar is accepted too rather than assumed away.
bool meta_i64(const GgufFile& f, const std::string& key, int64_t& out) {
    const MetaValue* v = f.get(key);
    if (!v) return false;
    if (v->type == MetaType::ARRAY) {
        if (v->items.empty()) return false;
        out = (int64_t) v->items[0].num();
        return true;
    }
    if (!v->is_num()) return false;
    out = (int64_t) v->num();
    return true;
}

#if defined(DS4_MOE_CUDA)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ds4_moe: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

/// The host arena: every expert the budget allows, in the order it was ranked, pinned when asked.  Whatever the
/// budget cannot hold is the FILE TIER and is read from the blob source on demand - the same fallback the replay
/// has, kept because "the code only ever runs with the whole model in RAM" is not a property to rely on.
struct Arena {
    void* base = nullptr;
    uint64_t bytes = 0;
    bool pinned = false;
    bool built = false;
    std::string backing;
    std::unique_ptr<strata::core::PinnedArena> owner;
    std::vector<int32_t> slot_of;   ///< (layer, expert) -> arena index, or -1
    int64_t used = 0, blob = 0, file_tier = 0;
    int64_t n_layers = 0, n_experts = 0;
    bool from_ranked = false;       ///< built from a routing profile (not just index order)
    double seconds = 0.0;           ///< how long the fill took (the startup report wants this number)

    const uint8_t* ptr(int64_t layer, int64_t expert) const {
        if (!base || layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) return nullptr;
        const int32_t s = slot_of[(size_t) (layer * n_experts + expert)];
        return s < 0 ? nullptr : (const uint8_t*) base + (size_t) s * (size_t) blob;
    }
    void close() {
        owner.reset();
        base = nullptr;
        bytes = 0;
        built = false;
        from_ranked = false;
        slot_of.clear();
        used = file_tier = 0;
    }
};

/// All the device state, so the CPU-only flavour has none of it to leak.
struct Gpu {
    Arena arena;
    std::unique_ptr<strata::core::ExpertCache> cache;
    strata::kernels::NativeExpertLayout gl;
    cudaStream_t s = nullptr;
    void *d_x = nullptr, *d_xq = nullptr, *d_parts = nullptr, *d_grp_ptr = nullptr, *d_grp_start = nullptr,
         *d_ngroups = nullptr, *d_ent_dst = nullptr, *d_ent_tok = nullptr, *d_scratch = nullptr, *d_stage = nullptr;
    cudaEvent_t ev[4] = {};
    void* d_pf = nullptr;         ///< kMaxPf staging slots, one expert blob each
    cudaStream_t s_pf = nullptr;  ///< the prefetch stream (never the main one: the DMAs run under the dense work)
    cudaEvent_t ev_pf = nullptr;
    uint8_t* h_dummy = nullptr;   ///< pinned zero blob: a guess outside the arena still costs its DMA
    float* h_x = nullptr;         ///< pinned host copy of x (the H2D source and the CPU tier's input)
    void* h_parts = nullptr;      ///< pinned D2H landing pad for the expert outputs

    void close() {
        for (auto& e : ev)
            if (e) cudaEventDestroy(e);
        void* bufs[] = {d_x, d_xq, d_parts, d_grp_ptr, d_grp_start, d_ngroups, d_ent_dst, d_ent_tok, d_scratch,
                        d_stage, d_pf};
        for (void* p : bufs)
            if (p) cudaFree(p);
        if (ev_pf) cudaEventDestroy(ev_pf);
        if (s_pf) cudaStreamDestroy(s_pf);
        if (s) cudaStreamDestroy(s);
        if (h_dummy) cudaFreeHost(h_dummy);
        if (h_x) cudaFreeHost(h_x);
        if (h_parts) cudaFreeHost(h_parts);
        cache.reset();
        arena.close();
        *this = Gpu();
    }
};

#else

/// No CUDA in this flavour; the type exists only so `Ds4MoeImpl` compiles unchanged.
struct Gpu {
    void close() {}
};

#endif  // DS4_MOE_CUDA

/// Every (layer, expert) by descending routing frequency over the TRAIN half of a `ds4routes.bin` (alternating
/// 512-token blocks - the split `tools/ds4/route_skew.py` scores with).  The file's layout, verified on the real
/// one (3,170,304 records = 12,288 tokens x 43 x 6): for each 512-token ubatch, layer 0's whole batch (512 tokens
/// x 6 experts, token-major), then layer 1's, ...
std::vector<std::pair<int32_t, int32_t>> rank_routes(const std::string& path, int64_t n_layers, int64_t top_k,
                                                     std::string& err) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        err = "cannot open " + path;
        return {};
    }
    const int64_t bytes = (int64_t) in.tellg();
    in.seekg(0);
    std::vector<uint16_t> all((size_t) (bytes / 2));
    in.read((char*) all.data(), bytes);
    if (!in) {
        err = "short read on " + path;
        return {};
    }
    const int64_t records = (int64_t) all.size() / 2;
    const int64_t per_token = n_layers * top_k;
    if (records == 0 || records % per_token != 0) {
        err = "not a whole number of tokens (records " + std::to_string(records) + ", per token " +
              std::to_string(per_token) + ")";
        return {};
    }
    const int64_t per_batch = kBatch * per_token;   // records in one 512-token block
    std::map<std::pair<int32_t, int32_t>, int64_t> cnt;
    for (int64_t i = 0; i + 1 < (int64_t) all.size(); i += 2) {
        const int64_t rec = i / 2;
        const int64_t token = (rec / per_batch) * kBatch + (rec % (kBatch * top_k)) / top_k;
        if ((token / kBatch) % 2 != 0) continue;    // the held-out half
        cnt[{all[(size_t) i], all[(size_t) i + 1]}]++;
    }
    std::vector<std::pair<int32_t, int32_t>> ranked;
    ranked.reserve(cnt.size());
    for (auto& le : cnt) ranked.push_back(le.first);
    std::sort(ranked.begin(), ranked.end(), [&](const auto& x, const auto& y) {
        const int64_t cx = cnt[x], cy = cnt[y];
        return cx != cy ? cx > cy : x < y;
    });
    return ranked;
}

}  // namespace

// ================================ Ds4MoeImpl ================================

struct Ds4MoeImpl {
    Ds4MoeGeom g;
    Ds4MoeConfig cfg;
    Ds4BlobSource* blobs = nullptr;
    std::unique_ptr<Ds4BlobSource> owned_blobs;
    cpu::NativeFmt f;
    int64_t blob = 0;
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<uint8_t> nact;        ///< the Q8_K activation the CPU kernel consumes
    std::vector<uint8_t> nact_n;      ///< run_multi: kMaxTok of them, f.act_bytes apart
    std::vector<float> parts_n;       ///< run_multi: the pool's outputs, one row per (token, k) entry
    std::vector<float> parts;         ///< top_k * n_embd, one expert output per routing index
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<uint8_t> ftmp;        ///< the file tier's read buffer, top_k blobs
    Ds4MoeStats st;
    bool inited = false;
    int64_t admitted = 0;
    double pcie_carry = 0.0, pf_carry = 0.0;
    int32_t pf_e[Ds4MoeConfig::kMaxPf] = {};
    int pf_n = 0;                     ///< staging slots the last prefetch() filled
    std::unique_ptr<Gpu> gpu;         ///< null in a CPU-only tier
};


// ---- the one place a blob's address is decided: the arena first, the blob source second ----------------

/// Returns `(layer, expert)`'s blob and sets `*from_file` when it had to be read into slot `scratch` of `ftmp`.
/// The pointer is valid until the next call with the same `scratch` slot, which is all `run` needs.
const uint8_t* acquire(Ds4MoeImpl& im, int64_t layer, int64_t expert, int scratch, bool* from_file) {
#if defined(DS4_MOE_CUDA)
    if (im.gpu) {
        if (const uint8_t* p = im.gpu->arena.ptr(layer, expert)) {
            *from_file = false;
            return p;
        }
    }
#endif
    *from_file = true;
    ++im.st.file_tier;
    if (im.ftmp.size() < (size_t) (scratch + 1) * (size_t) im.blob)
        im.ftmp.resize((size_t) std::max<int64_t>(im.g.top_k, scratch + 1) * (size_t) im.blob);
    uint8_t* dst = im.ftmp.data() + (size_t) scratch * (size_t) im.blob;
    im.blobs->read(layer, expert, dst);
    return dst;
}

// ================================ geometry ================================

bool ds4_moe_blob_layout(const Ds4MoeGeom& g, cpu::NativeFmt& f, std::string& err) {
    if (g.n_embd <= 0 || g.n_ff <= 0 || g.n_layers <= 0 || g.n_experts <= 0) {
        err = "geometry has a non-positive dimension";
        return false;
    }
    if (g.top_k < 1 || g.top_k > cpu::MAXT) {
        err = "top_k " + std::to_string(g.top_k) + " is outside 1.." + std::to_string((int) cpu::MAXT);
        return false;
    }
    std::string e;
    if (!cpu::native_fmt(g.gu_type, g.d_type, g.n_embd, g.n_ff, f, e)) {
        err = "native_fmt(" + std::to_string(g.gu_type) + "/" + std::to_string(g.d_type) + ") at " +
              std::to_string(g.n_embd) + "/" + std::to_string(g.n_ff) + ": " + e;
        return false;
    }
    f.swiglu_limit = g.swiglu_limit;
    return true;
}

bool ds4_moe_geom_from_gguf(const std::string& gguf, Ds4MoeGeom& g, std::string& err) {
    try {
        GgufFile f(gguf);
        int64_t v = 0;
        const std::pair<const char*, int64_t*> keys[] = {
            {"deepseek4.block_count", &g.n_layers},
            {"deepseek4.embedding_length", &g.n_embd},
            {"deepseek4.expert_count", &g.n_experts},
            {"deepseek4.expert_used_count", &g.top_k},
            {"deepseek4.expert_feed_forward_length", &g.n_ff},
        };
        for (const auto& k : keys) {
            if (!meta_i64(f, k.first, v)) {
                err = std::string("no metadata ") + k.first + " (is this a deepseek4 GGUF?)";
                return false;
            }
            *k.second = v;
        }
        const char* roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            char name[64];
            std::snprintf(name, sizeof name, "blk.0.ffn_%s_exps.weight", roles[r]);
            const TensorInfo* ti = f.find(name);
            if (!ti) {
                err = std::string("no tensor ") + name;
                return false;
            }
            // ne0 is the input width: gate/up are [n_embd, n_ff, n_experts], down is [n_ff, n_embd, n_experts].
            if ((int64_t) ti->shape.at(0) != (r == 2 ? g.n_ff : g.n_embd)) {
                err = std::string(name) + " has ne0 " + std::to_string(ti->shape.at(0));
                return false;
            }
            (r == 2 ? g.d_type : g.gu_type) = (int) ti->type;
        }
        // the SwiGLU clamp: one limit for the whole tier (it has one NativeFmt / layout), so every layer must agree
        std::vector<double> lim;
        const std::string le = strata::ds4_key_f64_arr(f, "deepseek4.swiglu_clamp_exp", lim);
        if (le.empty() && !lim.empty()) {
            for (double x : lim)
                if (x != lim[0]) {
                    err = "deepseek4.swiglu_clamp_exp differs between layers; the tier supports one limit";
                    return false;
                }
            if (lim[0] > 0.0) g.swiglu_limit = (float) lim[0];   // <= 0 means "no clamp" in llama.cpp
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

// ================================ blob sources ================================

struct Ds4GgufBlobs::Impl {
    std::unique_ptr<GgufModel> model;
    std::vector<const uint8_t*> src[3];   // per role, per layer
    size_t gu_stride = 0, d_stride = 0;
};

Ds4GgufBlobs::Ds4GgufBlobs(Ds4MoeGeom g, std::string gguf) : im_(new Impl), g_(g), path_(std::move(gguf)) {}
Ds4GgufBlobs::~Ds4GgufBlobs() = default;

bool Ds4GgufBlobs::open(std::string& err) {
    cpu::NativeFmt f;
    if (!ds4_moe_blob_layout(g_, f, err)) return false;
    blob_ = f.bytes;
    im_->gu_stride = (size_t) f.up_off;
    im_->d_stride = (size_t) f.bytes - f.down_off;
    try {
        im_->model.reset(new GgufModel(gguf_split_paths(path_)));
        const char* roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) im_->src[r].assign((size_t) g_.n_layers, nullptr);
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            for (int r = 0; r < 3; ++r) {
                char name[64];
                std::snprintf(name, sizeof name, "blk.%lld.ffn_%s_exps.weight", (long long) l, roles[r]);
                size_t at = 0;
                const TensorInfo* ti = im_->model->find(name, &at);
                if (!ti) {
                    err = std::string("no tensor ") + name;
                    return false;
                }
                if (ti->type != (uint32_t) (r == 2 ? g_.d_type : g_.gu_type)) {
                    err = std::string(name) + " has type " + ti->type_name();
                    return false;
                }
                im_->src[r][(size_t) l] = im_->model->shard(at).tensor_data(*ti);
            }
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

void Ds4GgufBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    // The three role slices back to back, [gate rows | up rows | down rows] (native_expert.hpp's layout), out of
    // the mmap: three memcpys of pages the OS already has.
    std::memcpy(dst, im_->src[0][(size_t) layer] + (size_t) expert * im_->gu_stride, im_->gu_stride);
    std::memcpy(dst + im_->gu_stride, im_->src[1][(size_t) layer] + (size_t) expert * im_->gu_stride,
                im_->gu_stride);
    std::memcpy(dst + 2 * im_->gu_stride, im_->src[2][(size_t) layer] + (size_t) expert * im_->d_stride,
                im_->d_stride);
}

bool Ds4MemoryBlobs::open(std::string& err) {
    cpu::NativeFmt f;
    if (!ds4_moe_blob_layout(g_, f, err)) return false;
    if (f.bytes != blob_) {
        err = "blob size " + std::to_string(blob_) + " != the geometry's " + std::to_string(f.bytes);
        return false;
    }
    return true;
}

void Ds4MemoryBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    const auto it = blobs_.find({layer, expert});
    if (it != blobs_.end()) std::memcpy(dst, it->second.data(), blob_);
    else std::memset(dst, 0, blob_);   // see the header: `has()` is the gate's assertion, this is the fallback
}

void Ds4MemoryBlobs::set(int64_t layer, int64_t expert, const uint8_t* blob) {
    if (layer < 0 || layer >= g_.n_layers || expert < 0 || expert >= g_.n_experts) return;
    blobs_[{layer, expert}].assign(blob, blob + blob_);
}

bool Ds4MemoryBlobs::has(int64_t layer, int64_t expert) const {
    return blobs_.find({layer, expert}) != blobs_.end();
}

// ================================ the tier ================================

Ds4MoeTier::Ds4MoeTier() : im_(new Ds4MoeImpl) {}
Ds4MoeTier::~Ds4MoeTier() { close(); }

bool Ds4MoeTier::init(const std::string& gguf, const Ds4MoeConfig& cfg, std::string& err) {
    Ds4MoeGeom g;
    if (!ds4_moe_geom_from_gguf(gguf, g, err)) return false;
    std::unique_ptr<Ds4BlobSource> src(new Ds4GgufBlobs(g, gguf));
    if (!src->open(err)) return false;
    if (!init(g, src.get(), cfg, err)) return false;
    im_->owned_blobs = std::move(src);
    return true;
}

bool Ds4MoeTier::init(const Ds4MoeGeom& geom, Ds4BlobSource* blobs, const Ds4MoeConfig& cfg, std::string& err) {
    close();
    im_->g = geom;
    im_->cfg = cfg;
    im_->blobs = blobs;
    if (!ds4_moe_blob_layout(geom, im_->f, err)) return false;
    im_->blob = (int64_t) im_->f.bytes;
    if (!blobs->open(err)) return false;
    if ((int64_t) blobs->blob_bytes() != im_->blob) {
        err = "blob source is " + std::to_string(blobs->blob_bytes()) + " bytes, the geometry's is " +
              std::to_string(im_->blob);
        return false;
    }
    im_->pool.reset(new cpu::ExpertPool(cfg.threads > 0 ? cfg.threads : 0));
    im_->jobs.resize((size_t) geom.top_k);
    im_->parts.assign((size_t) geom.top_k * (size_t) geom.n_embd, 0.0f);
    im_->nact.assign(cpu::kNativeActBytes, 0);
    im_->nact_n.assign((size_t) kMaxTok * cpu::kNativeActBytes, 0);
    im_->parts_n.assign((size_t) kMaxEnt * (size_t) geom.n_embd, 0.0f);

    if (cfg.cpu_only) {
        im_->inited = true;
        return true;
    }

#if !defined(DS4_MOE_CUDA)
    err = "this build has no CUDA tier (compiled without DS4_MOE_CUDA); set cfg.cpu_only";
    return false;
#else
    Gpu& gp = *(im_->gpu = std::unique_ptr<Gpu>(new Gpu()));
    gp.arena.blob = im_->blob;
    gp.arena.n_layers = geom.n_layers;
    gp.arena.n_experts = geom.n_experts;
    gp.gl = strata::kernels::native_expert_layout(geom.gu_type, geom.d_type, geom.n_embd, geom.n_ff);
    gp.gl.swiglu_limit = geom.swiglu_limit;
    if (!strata::kernels::native_expert_supported(geom.gu_type, geom.d_type, geom.n_embd, geom.n_ff)) {
        err = "native_expert_grouped has no kernel for these types at this geometry";
        return false;
    }
    ck(cudaStreamCreate(&gp.s), "stream");
    const size_t H4 = (size_t) geom.n_embd * 4;
    ck(cudaMalloc(&gp.d_x, (size_t) kMaxTok * H4), "d_x");
    ck(cudaMalloc(&gp.d_xq, strata::kernels::native_q8_1_bytes((int) geom.n_embd, kMaxTok)), "d_xq");
    ck(cudaMalloc(&gp.d_parts, (size_t) kMaxEnt * H4), "d_parts");
    ck(cudaMalloc(&gp.d_grp_ptr, sizeof(unsigned long long) * kMaxEnt), "d_grp_ptr");
    ck(cudaMalloc(&gp.d_grp_start, sizeof(int32_t) * (kMaxEnt + 1)), "d_grp_start");
    ck(cudaMalloc(&gp.d_ngroups, sizeof(int32_t)), "d_ngroups");
    ck(cudaMalloc(&gp.d_ent_dst, sizeof(int32_t) * kMaxEnt), "d_ent_dst");
    ck(cudaMalloc(&gp.d_ent_tok, sizeof(int32_t) * kMaxEnt), "d_ent_tok");
    ck(cudaMalloc(&gp.d_scratch, strata::kernels::native_expert_scratch_bytes(kMaxEnt, geom.n_ff)), "d_scratch");
    ck(cudaMalloc(&gp.d_stage, (size_t) kMaxParts * (size_t) im_->blob), "d_stage");
    for (int i = 0; i < 4; ++i) ck(cudaEventCreate(&gp.ev[i]), "event");
    ck(cudaMallocHost((void**) &gp.h_x, (size_t) kMaxTok * H4), "h_x");
    ck(cudaMallocHost(&gp.h_parts, (size_t) kMaxEnt * H4), "h_parts");
    if (cfg.pf_b > 0) {
        ck(cudaStreamCreateWithFlags(&gp.s_pf, cudaStreamNonBlocking), "pf stream");
        ck(cudaMalloc(&gp.d_pf, (size_t) Ds4MoeConfig::kMaxPf * (size_t) im_->blob), "d_pf");
        ck(cudaEventCreateWithFlags(&gp.ev_pf, cudaEventDisableTiming), "ev_pf");
        ck(cudaMallocHost((void**) &gp.h_dummy, (size_t) im_->blob), "h_dummy");
        std::memset(gp.h_dummy, 0, (size_t) im_->blob);
    }
    if (!cfg.no_cache && cfg.slots > 0) {
        std::vector<int64_t> sizes((size_t) cfg.slots, im_->blob);
        gp.cache.reset(new strata::core::ExpertCache());
        if (!gp.cache->open_sized(sizes, geom.n_layers, geom.n_experts, err)) return false;
    }
    // NO ARENA HERE.  The arena is 52-73 GiB of the real model and it is ORDERED (build_arena): building it twice -
    // once in index order here, once from the profile when the caller seeds - would read the file twice and page
    // the second read through the first's cache.  The first build wins (build_arena), and the first build is the
    // seed's; a caller that never seeds gets the file tier for every expert, which is slower but not wrong.
    im_->inited = true;
    return true;
#endif
}

bool Ds4MoeTier::build_arena(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
#if !defined(DS4_MOE_CUDA)
    (void) ranked;
    (void) err;
    return true;   // no arena in a CPU-only tier: every blob comes from the source
#else
    if (!im_->gpu || im_->cfg.cpu_only) return true;
    Gpu& gp = *im_->gpu;
    const int64_t total = im_->g.n_layers * im_->g.n_experts;
    // An arena that already holds every expert has nothing to gain from another ordering, and a rebuild would
    // re-read up to 72 GiB of the model for no change in behaviour.  Neither has a re-seed with a ranking when the
    // arena was already built from one: the first ranking wins, exactly as it does for the cache, where `admit`
    // hands out the free slots to the first seed and never evicts.
    if (gp.arena.built && (gp.arena.used >= total || (gp.arena.from_ranked && !ranked.empty()))) return true;
    gp.arena.close();

    const double budget_gib = std::min(im_->cfg.arena_gib, im_->cfg.max_arena_gib);
    if (budget_gib <= 0) {
        gp.arena.built = true;
        return true;
    }
    const int64_t cap = (int64_t) (budget_gib * 1073741824.0 / (double) im_->blob);
    std::vector<std::pair<int32_t, int32_t>> order = ranked;
    if (order.empty()) {
        order.reserve((size_t) total);
        for (int64_t l = 0; l < im_->g.n_layers; ++l)
            for (int64_t e = 0; e < im_->g.n_experts; ++e) order.push_back({(int32_t) l, (int32_t) e});
    }
    gp.arena.slot_of.assign((size_t) total, -1);
    std::vector<std::pair<int32_t, int32_t>> first;
    first.reserve(order.size());
    for (const auto& le : order) {
        const int64_t l = le.first, e = le.second;
        if (l < 0 || l >= im_->g.n_layers || e < 0 || e >= im_->g.n_experts) continue;
        if (gp.arena.slot_of[(size_t) (l * im_->g.n_experts + e)] >= 0) continue;
        if ((int64_t) first.size() >= cap) {
            ++gp.arena.file_tier;
            continue;
        }
        gp.arena.slot_of[(size_t) (l * im_->g.n_experts + e)] = (int32_t) first.size();
        first.push_back({(int32_t) l, (int32_t) e});
    }
    gp.arena.used = (int64_t) first.size();
    gp.arena.bytes = (uint64_t) gp.arena.used * (uint64_t) im_->blob;
    gp.arena.built = true;
    gp.arena.from_ranked = !ranked.empty();
    if (gp.arena.bytes == 0) {
        gp.arena.backing = "empty";
        return true;
    }
    const int64_t avail = mem_available_gib();
    const int64_t gib = (int64_t) (gp.arena.bytes / 1073741824ull);
    if (avail >= 0 && gib + (int64_t) im_->cfg.mem_floor_gib > avail) {
        err = "an arena of " + std::to_string(gib) + " GiB would leave less than " +
              std::to_string((int64_t) im_->cfg.mem_floor_gib) + " GiB free (MemAvailable " +
              std::to_string(avail) + " GiB)";
        gp.arena.close();
        return false;
    }
    if (im_->cfg.pin) {
        gp.arena.owner.reset(new strata::core::PinnedArena(gp.arena.bytes));
        if (!gp.arena.owner->valid()) {
            err = "pinned arena allocation of " + std::to_string(gib) + " GiB failed";
            gp.arena.close();
            return false;
        }
        gp.arena.base = gp.arena.owner->base;
        gp.arena.pinned = true;
        gp.arena.backing = "pinned (" + gp.arena.owner->note + ")";
    } else {
        void* p = nullptr;
        if (posix_memalign(&p, 4096, gp.arena.bytes) != 0 || !p) {
            err = "anonymous arena allocation failed";
            gp.arena.close();
            return false;
        }
        gp.arena.base = p;
        gp.arena.backing = "anonymous (pageable: the PCIe DMAs stage through the driver)";
    }
    int nt = im_->cfg.threads > 0 ? im_->cfg.threads : 8;
    nt = std::min<int>(nt, 16);
    const double t0 = now_ms();
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    for (int c = 0; c < nt; ++c)
        th.emplace_back([&]() {
            for (;;) {
                const int64_t i = next.fetch_add(1);
                if (i >= gp.arena.used) break;
                im_->blobs->read(first[(size_t) i].first, first[(size_t) i].second,
                                 (uint8_t*) gp.arena.base + (size_t) i * (size_t) im_->blob);
            }
        });
    for (auto& t : th) t.join();
    gp.arena.seconds = (now_ms() - t0) / 1000.0;
    return true;
#endif
}

bool Ds4MoeTier::seed_from_routes(const std::string& routes_bin, std::string& err) {
    std::string e;
    const std::vector<std::pair<int32_t, int32_t>> ranked =
        rank_routes(routes_bin, im_->g.n_layers, im_->g.top_k, e);
    if (ranked.empty()) {
        err = e.empty() ? "empty routing profile" : e;
        return false;
    }
    return seed_from_ranked(ranked, err);
}

bool Ds4MoeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (im_->cfg.cpu_only) return true;   // a CPU-only tier has neither a cache nor an arena to seed
    if (!build_arena(ranked, err)) return false;
#if !defined(DS4_MOE_CUDA)
    (void) ranked;
    err = "the VRAM cache needs the CUDA tier (compiled without DS4_MOE_CUDA)";
    return false;
#else
    if (!im_->gpu || !im_->gpu->cache) return true;   // no cache: nothing to seed
    const int64_t want = std::min<int64_t>(im_->cfg.slots, im_->gpu->cache->full_slots());
    int64_t filled = 0;
    for (const auto& le : ranked) {
        if (filled >= want) break;
        const int32_t s = im_->gpu->cache->admit(le.first, le.second);
        if (s < 0) break;
        bool from_file = false;
        const uint8_t* p = acquire(*im_, le.first, le.second, 0, &from_file);
        if (!im_->gpu->cache->fill_slot_blocking(s, p, err, im_->blob)) return false;
        ++filled;
    }
    im_->admitted = filled;
    return true;
#endif
}

void Ds4MoeTier::prefetch(int64_t layer, const int32_t* top16, int n) {
    im_->pf_n = 0;
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu || im_->cfg.cpu_only || im_->cfg.pf_b <= 0 || im_->cfg.no_cache || !im_->gpu->cache) return;
    if (!top16 || n <= 0) return;
    Gpu& gp = *im_->gpu;
    n = std::min(n, (int) Ds4MoeConfig::kPredW);
    int nb;
    if (im_->cfg.dither) {
        const double want = im_->cfg.pf_b + im_->pf_carry;
        nb = (int) std::floor(want);
        im_->pf_carry = want - std::floor(want);
    } else {
        nb = (int) std::lround(im_->cfg.pf_b);
    }
    nb = std::min(nb, (int) Ds4MoeConfig::kMaxPf);
    for (int i = 0; i < n && im_->pf_n < nb; ++i) {
        const int32_t e = top16[i];
        if (e < 0 || e >= im_->g.n_experts) continue;
        if (gp.cache->slot_of(layer, e) >= 0) continue;
        bool dup = false;
        for (int j = 0; j < im_->pf_n; ++j) dup = dup || im_->pf_e[j] == e;
        if (dup) continue;
        const uint8_t* src = gp.arena.ptr(layer, e);
        if (!src) {
            src = gp.h_dummy;   // a guess outside the arena still costs its DMA (the replay's own rule)
            ++im_->st.prefetch_dummy;
        }
        ck(cudaMemcpyAsync((uint8_t*) gp.d_pf + (size_t) im_->pf_n * (size_t) im_->blob, src, (size_t) im_->blob,
                           cudaMemcpyHostToDevice, gp.s_pf), "prefetch dma");
        im_->pf_e[im_->pf_n++] = e;
    }
    if (im_->pf_n > 0) ck(cudaEventRecord(gp.ev_pf, gp.s_pf), "record prefetch");
    im_->st.prefetch_issued += im_->pf_n;
#else
    (void) layer;
    (void) top16;
    (void) n;
#endif
}

// ---- the CPU path: the whole expert half, and the fallback for every miss the GPU does not take ----------

bool cpu_run(Ds4MoeImpl& im, int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out) {
    const int64_t H = im.g.n_embd;
    const int64_t K = im.g.top_k;
    const double t0 = now_ms();
    cpu::native_quant_act(im.f, x, im.nact.data());
    std::vector<const uint8_t*> held((size_t) K, nullptr);
    for (int64_t k = 0; k < K; ++k) {
        bool from_file = false;
        held[(size_t) k] = acquire(im, layer, ids6[k], (int) k, &from_file);
        im.jobs[(size_t) k].blob = held[(size_t) k];
        im.jobs[(size_t) k].nt = 1;
        for (int t = 0; t < cpu::MAXT; ++t) {
            im.jobs[(size_t) k].nact[t] = nullptr;
            im.jobs[(size_t) k].out[t] = nullptr;
        }
        im.jobs[(size_t) k].nact[0] = im.nact.data();
        im.jobs[(size_t) k].out[0] = im.parts.data() + (size_t) k * H;
    }
    const double c0 = now_ms();
    im.pool->run_split_multi_native(im.f, im.jobs.data(), (int) K);
    im.st.cpu_ms += now_ms() - c0;
    for (int64_t i = 0; i < H; ++i) {
        double s = 0;
        for (int64_t k = 0; k < K; ++k) s += (double) w6[k] * (double) im.parts[(size_t) (k * H + i)];
        out[i] = (float) s;
    }
    im.st.cpu += K;
    im.st.wall_ms += now_ms() - t0;
    return true;
}

#if defined(DS4_MOE_CUDA)

/// One layer of the three-tier path.  The shape is `moe_replay.cpp`'s `Engine::layer`, with the replay's
/// synthetic dense gap and `--serial` wait removed: `run` is called AFTER the layer's dense work by contract, so
/// there is nothing to overlap the CPU tier against except the GPU tier, which is what runs beside it.
// DS4_TIER_PROF=1: where a layer's wall time goes inside gpu_run (summed, printed by close())
double g_prof[9] = {};
int64_t g_prof_n = 0;
const char* const g_prof_name[9] = {"split", "activation", "gpu launch", "cpu pool", "layer sync", "d2h parts",
                                    "sum", "admission", "pcie budget"};
bool gpu_run(Ds4MoeImpl& im, int64_t layer, const int32_t* ids6, const float* w6, const float* x_host,
             const void* x_dev, float* out) {
    static const bool prof = std::getenv("DS4_TIER_PROF") != nullptr;
    double tp = prof ? now_ms() : 0;
    auto mark = [&](int i) { if (prof) { const double t = now_ms(); g_prof[i] += t - tp; tp = t; } };
    Gpu& gp = *im.gpu;
    const int64_t H = im.g.n_embd;
    const int64_t K = im.g.top_k;
    const int64_t B = im.blob;
    const double t0 = now_ms();
    Ds4MoeStats st;

    // ---- split the routed experts: resident, prefetched (also computed on the GPU), CPU, PCIe ----
    int32_t hit_i[kMaxParts], cpu_i[kMaxParts], pcie_i[kMaxParts], slot_i[kMaxParts] = {};
    unsigned long long pf_ptr[kMaxParts] = {};
    int nh = 0, nc = 0, np = 0, npf_hit = 0;
    for (int64_t k = 0; k < K; ++k) {
        const int32_t e = ids6[k];
        const int32_t s = (gp.cache && !im.cfg.no_cache) ? gp.cache->slot_of(layer, e) : -1;
        slot_i[k] = s;
        int pf_at = -1;
        for (int j = 0; j < im.pf_n && s < 0; ++j)
            if (im.pf_e[j] == e) pf_at = j;
        if (s >= 0) {
            hit_i[nh++] = (int32_t) k;
        } else if (pf_at >= 0) {
            pf_ptr[k] = (unsigned long long) gp.d_pf + (size_t) pf_at * (size_t) B;
            hit_i[nh++] = (int32_t) k;
            ++npf_hit;
        } else {
            cpu_i[nc++] = (int32_t) k;
        }
    }
    st.prefetched_useful = npf_hit;
    mark(0);

    // The PCIe share of the misses, with the replay's error-diffused budget AND its eligibility rule: the DMA
    // reads straight out of the pinned host arena, so a blob the arena does not hold (the file tier) stays on the
    // CPU rather than being staged through the driver - which is what the replay measured, and the reason its
    // PCIe percentage is a percentage of what the arena held.
    {
        int eligible = 0;
        for (int i = 0; i < nc; ++i)
            if (gp.arena.ptr(layer, ids6[cpu_i[i]])) ++eligible;
        int budget;
        if (im.cfg.dither) {
            const double want = (double) eligible * im.cfg.pcie_frac + im.pcie_carry;
            budget = (int) std::floor(want);
            im.pcie_carry = want - budget;
        } else {
            budget = (int) std::lround((double) eligible * im.cfg.pcie_frac);
        }
        for (int i = 0; i < nc && np < budget; ++i) {
            const int k = cpu_i[nc - 1 - i];
            if (gp.arena.ptr(layer, ids6[k])) pcie_i[np++] = (int32_t) k;
        }
    }
    mark(8);
    int32_t cpu_keep[kMaxParts], nk = 0;
    bool cpu_k[kMaxParts] = {};   ///< routing index -> the pool computed it (the sum's source selector)
    for (int i = 0; i < nc; ++i) {
        bool in_pcie = false;
        for (int j = 0; j < np; ++j) in_pcie = in_pcie || pcie_i[j] == cpu_i[i];
        if (!in_pcie) {
            cpu_keep[nk++] = cpu_i[i];
            cpu_k[cpu_i[i]] = true;
        }
    }

    // ---- the activation, on both sides: Q8_K for the pool, q8_1 for the card ----
    if (x_dev) {
        ck(cudaMemcpyAsync(gp.h_x, x_dev, (size_t) H * 4, cudaMemcpyDeviceToHost, gp.s), "x d2h");
        ck(cudaStreamSynchronize(gp.s), "x sync");   // the pool needs it on the host, now
    } else {
        std::memcpy(gp.h_x, x_host, (size_t) H * 4);
    }
    cpu::native_quant_act(im.f, gp.h_x, im.nact.data());
    ck(cudaMemcpyAsync(gp.d_x, gp.h_x, (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "x h2d");
    strata::kernels::native_quantize_q8_1((const float*) gp.d_x, gp.d_xq, (int) H, 1, gp.s);
    ck(cudaEventRecord(gp.ev[0], gp.s), "record activation");
    mark(1);

    // ---- one group's metadata, then the grouped kernel: w 0 = the GPU's hits, 1 = the PCIe share ----
    auto launch_group = [&](int w, int n, const std::vector<unsigned long long>& ptr,
                            const std::vector<int32_t>& start, const std::vector<int32_t>& dst,
                            const std::vector<int32_t>& tok) {
        const int32_t one = n;
        ck(cudaMemcpyAsync(gp.d_grp_ptr, ptr.data(), sizeof(unsigned long long) * n, cudaMemcpyHostToDevice, gp.s),
           "grp_ptr");
        ck(cudaMemcpyAsync(gp.d_grp_start, start.data(), sizeof(int32_t) * (n + 1), cudaMemcpyHostToDevice, gp.s),
           "grp_start");
        ck(cudaMemcpyAsync(gp.d_ngroups, &one, sizeof(int32_t), cudaMemcpyHostToDevice, gp.s), "ngroups");
        ck(cudaMemcpyAsync(gp.d_ent_dst, dst.data(), sizeof(int32_t) * n, cudaMemcpyHostToDevice, gp.s), "ent_dst");
        ck(cudaMemcpyAsync(gp.d_ent_tok, tok.data(), sizeof(int32_t) * n, cudaMemcpyHostToDevice, gp.s), "ent_tok");
        strata::kernels::native_expert_grouped(gp.gl, (const unsigned long long*) gp.d_grp_ptr,
                                              (const int32_t*) gp.d_grp_start, (const int32_t*) gp.d_ngroups,
                                              (const int32_t*) gp.d_ent_dst, (const int32_t*) gp.d_ent_tok,
                                              kMaxParts, n, gp.d_xq, gp.d_scratch, (float*) gp.d_parts, gp.s, n);
        (void) w;
    };

    int ev_hit = -1, ev_pcie = -1;
    if (nh > 0) {
        std::vector<unsigned long long> ptr((size_t) nh);
        std::vector<int32_t> start((size_t) nh + 1), dst((size_t) nh, 0), tok((size_t) nh, 0);
        for (int i = 0; i < nh; ++i) {
            ptr[(size_t) i] = pf_ptr[hit_i[i]] ? pf_ptr[hit_i[i]]
                                               : (unsigned long long) gp.cache->device_slot(slot_i[hit_i[i]]);
            start[(size_t) i] = i;
            dst[(size_t) i] = hit_i[i];
        }
        start[(size_t) nh] = nh;
        if (npf_hit > 0) ck(cudaStreamWaitEvent(gp.s, gp.ev_pf, 0), "wait prefetch");
        launch_group(0, nh, ptr, start, dst, tok);
        ck(cudaEventRecord(gp.ev[2], gp.s), "record hits");
        ev_hit = 2;
    }
    if (np > 0) {
        std::vector<unsigned long long> ptr((size_t) np);
        std::vector<int32_t> start((size_t) np + 1), dst((size_t) np, 0), tok((size_t) np, 0);
        for (int i = 0; i < np; ++i) {
            const uint8_t* p = gp.arena.ptr(layer, ids6[pcie_i[i]]);
            ck(cudaMemcpyAsync((uint8_t*) gp.d_stage + (size_t) i * (size_t) B, p, (size_t) B,
                               cudaMemcpyHostToDevice, gp.s), "pcie dma");
            ptr[(size_t) i] = (unsigned long long) gp.d_stage + (size_t) i * (size_t) B;
            start[(size_t) i] = i;
            dst[(size_t) i] = pcie_i[i];
        }
        start[(size_t) np] = np;
        launch_group(1, np, ptr, start, dst, tok);
        ck(cudaEventRecord(gp.ev[3], gp.s), "record pcie");
        ev_pcie = 3;
    }

    mark(2);
    // ---- the CPU tier on what is left, WHILE the card works ----
    if (nk > 0) {
        std::vector<const uint8_t*> held((size_t) nk, nullptr);
        for (int i = 0; i < nk; ++i) {
            bool from_file = false;
            held[(size_t) i] = acquire(im, layer, ids6[cpu_keep[i]], i, &from_file);
            cpu::ExpertJobMulti& j = im.jobs[(size_t) i];
            j.blob = held[(size_t) i];
            j.nt = 1;
            for (int t = 0; t < cpu::MAXT; ++t) {
                j.nact[t] = nullptr;
                j.out[t] = nullptr;
            }
            j.nact[0] = im.nact.data();
            j.out[0] = im.parts.data() + (size_t) cpu_keep[i] * H;
        }
        const double c0 = now_ms();
        im.pool->run_split_multi_native(im.f, im.jobs.data(), nk);
        st.cpu_ms = now_ms() - c0;
    }

    mark(3);
    ck(cudaStreamSynchronize(gp.s), "layer sync");
    mark(4);
    float ms = 0;
    if (ev_hit > 0) {
        ck(cudaEventElapsedTime(&ms, gp.ev[0], gp.ev[ev_hit]), "hits elapsed");
        st.hit_ms = ms;
    }
    if (ev_pcie > 0) {
        ck(cudaEventElapsedTime(&ms, gp.ev[ev_hit > 0 ? ev_hit : 0], gp.ev[ev_pcie]), "pcie elapsed");
        st.pcie_ms = ms;
    }
    ck(cudaMemcpyAsync(gp.h_parts, gp.d_parts, (size_t) std::max<int64_t>(kMaxParts, K) * (size_t) H * 4,
                       cudaMemcpyDeviceToHost, gp.s), "d2h parts");
    ck(cudaStreamSynchronize(gp.s), "parts sync");
    const float* gpu_parts = (const float*) gp.h_parts;
    mark(5);
    // DS4_CHECK_GPU=1: every card-computed expert recomputed on the CPU from the arena's bytes (diagnostic, slow)
    static const bool check_gpu = std::getenv("DS4_CHECK_GPU") != nullptr;
    if (check_gpu) {
        std::vector<float> ref((size_t) H);
        for (int64_t k = 0; k < K; ++k) {
            if (cpu_k[k]) continue;
            const uint8_t* p = gp.arena.ptr(layer, ids6[k]);
            if (!p) continue;
            ds4_moe_cpu_expert(im.f, H, im.g.n_ff, p, gp.h_x, ref.data(), *im.pool);
            double num = 0, den = 0;
            const float* g = gpu_parts + (size_t) k * (size_t) H;
            for (int64_t i = 0; i < H; ++i) { const double d = g[i] - ref[(size_t) i]; num += d * d; den += (double) ref[(size_t) i] * ref[(size_t) i]; }
            bool is_pcie = false;
            for (int j = 0; j < np; ++j) is_pcie = is_pcie || pcie_i[j] == k;
            std::fprintf(stderr, "[chk] L%lld e%d %s rel %.3e\n", (long long) layer, ids6[k], is_pcie ? "pcie" : "hit ",
                         std::sqrt(num / std::max(den, 1e-30)));
        }
    }

    // ---- the weighted sum.  Routing-indexed, so all three tiers write into one layout without a scatter ----
    // The pool's results are `im.parts` (host) and the card's are the readback above: the D2H is one bulk copy, so
    // it cannot be told to leave the pool's indices alone - the index's own source selects which buffer to read.
    // Reading `gpu_parts` for a CPU-computed index would sum uninitialized device memory, which is why this is not
    // a detail (the gate's file-tier arm caught it: the replay is speed-only and never sums a layer).
    for (int64_t i = 0; i < H; ++i) {
        double s = 0;
        for (int64_t k = 0; k < K; ++k) {
            const float* p = cpu_k[k] ? im.parts.data() + (size_t) k * (size_t) H
                                      : gpu_parts + (size_t) k * (size_t) H;
            s += (double) w6[k] * (double) p[(size_t) i];
        }
        out[i] = (float) s;
    }

    mark(6);
    // ---- admission: a compulsory miss takes a free slot and is filled for the NEXT token ----
    if (gp.cache && !im.cfg.no_cache) {
        for (int i = 0; i < nc; ++i) {
            const int32_t e = ids6[cpu_i[i]];
            const uint8_t* p = gp.arena.ptr(layer, e);
            if (!p) continue;
            const int32_t s = gp.cache->admit(layer, e);
            if (s < 0) continue;
            std::string e2;
            if (!gp.cache->fill_slot_blocking(s, p, e2, B)) {
                std::fprintf(stderr, "ds4_moe: fill_slot: %s\n", e2.c_str());
                std::abort();
            }
            ++im.admitted;
        }
    }

    mark(7);
    if (prof) ++g_prof_n;
    st.hits = nh;
    st.pcie = np;
    st.cpu = nk;
    st.wall_ms = now_ms() - t0;
    im.st.add(st);
    im.pf_n = 0;   // the staging is per layer; the caller must run(l) before prefetch(l+1)
    return true;
}


/// `nt` tokens of one layer in one pass (draft verification): the distinct experts across the tokens are each
/// fetched/read ONCE - a group with one entry per token that routed to it - so the second token's experts cost only
/// what the first did not already pay.  Entry j = t*K + k writes output row j; token t's sum is taken over its own k
/// in order, exactly as gpu_run does for one token.
bool gpu_run_n(Ds4MoeImpl& im, int64_t layer, int nt, const int32_t* ids, const float* w, const float* x_host,
               float* out) {
    Gpu& gp = *im.gpu;
    const int64_t H = im.g.n_embd;
    const int64_t K = im.g.top_k;
    const int64_t B = im.blob;
    const int NE = nt * (int) K;
    const double t0 = now_ms();
    Ds4MoeStats st;

    // ---- distinct experts, first-seen order ----
    int32_t ue[kMaxEnt], ent_u[kMaxEnt];
    int nu = 0;
    for (int j = 0; j < NE; ++j) {
        int u = -1;
        for (int i = 0; i < nu && u < 0; ++i)
            if (ue[i] == ids[j]) u = i;
        if (u < 0) { u = nu; ue[nu++] = ids[j]; }
        ent_u[j] = u;
    }
    // ---- classify: resident slot / prefetch staging / miss ----
    enum { HIT = 0, PCIE = 1, CPU = 2 };
    int cls[kMaxEnt];
    unsigned long long dptr[kMaxEnt] = {};
    int32_t miss[kMaxEnt];
    int nmiss = 0, npf = 0;
    for (int u = 0; u < nu; ++u) {
        const int32_t e = ue[u];
        const int32_t sl = (gp.cache && !im.cfg.no_cache) ? gp.cache->slot_of(layer, e) : -1;
        int pf_at = -1;
        for (int j = 0; j < im.pf_n && sl < 0; ++j)
            if (im.pf_e[j] == e) pf_at = j;
        if (sl >= 0) {
            cls[u] = HIT;
            dptr[u] = (unsigned long long) gp.cache->device_slot(sl);
        } else if (pf_at >= 0) {
            cls[u] = HIT;
            dptr[u] = (unsigned long long) gp.d_pf + (size_t) pf_at * (size_t) B;
            ++npf;
        } else {
            cls[u] = CPU;
            miss[nmiss++] = u;
        }
    }
    // ---- the PCIe share of the distinct misses (same budget rule as gpu_run), at most kMaxParts staged ----
    int np = 0;
    {
        int eligible = 0;
        for (int i = 0; i < nmiss; ++i)
            if (gp.arena.ptr(layer, ue[miss[i]])) ++eligible;
        int budget;
        if (im.cfg.dither) {
            const double want = (double) eligible * im.cfg.pcie_frac + im.pcie_carry;
            budget = (int) std::floor(want);
            im.pcie_carry = want - budget;
        } else {
            budget = (int) std::lround((double) eligible * im.cfg.pcie_frac);
        }
        budget = std::min(budget, kMaxParts);
        for (int i = 0; i < nmiss && np < budget; ++i) {
            const int u = miss[nmiss - 1 - i];
            if (gp.arena.ptr(layer, ue[u])) { cls[u] = PCIE; ++np; }
        }
    }

    // ---- activations: Q8_K per token for the pool, q8_1 rows for the card ----
    std::memcpy(gp.h_x, x_host, (size_t) nt * (size_t) H * 4);
    for (int t = 0; t < nt; ++t)
        cpu::native_quant_act(im.f, gp.h_x + (size_t) t * H, im.nact_n.data() + (size_t) t * im.f.act_bytes);
    ck(cudaMemcpyAsync(gp.d_x, gp.h_x, (size_t) nt * (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "x h2d");
    strata::kernels::native_quantize_q8_1((const float*) gp.d_x, gp.d_xq, (int) H, nt, gp.s);

    // ---- one grouped launch per GPU class (hits, PCIe), groups = distinct experts, entries = their tokens ----
    auto launch = [&](int want) {
        std::vector<unsigned long long> ptr;
        std::vector<int32_t> start, dst, tok;
        int si = 0;
        for (int u = 0; u < nu; ++u) {
            if (cls[u] != want) continue;
            if (want == PCIE) {
                ck(cudaMemcpyAsync((uint8_t*) gp.d_stage + (size_t) si * (size_t) B, gp.arena.ptr(layer, ue[u]),
                                   (size_t) B, cudaMemcpyHostToDevice, gp.s), "pcie dma");
                dptr[u] = (unsigned long long) gp.d_stage + (size_t) si * (size_t) B;
                ++si;
            }
            ptr.push_back(dptr[u]);
            start.push_back((int32_t) dst.size());
            for (int j = 0; j < NE; ++j)
                if (ent_u[j] == u) { dst.push_back(j); tok.push_back(j / (int) K); }
        }
        const int ng = (int) ptr.size();
        if (ng == 0) return;
        start.push_back((int32_t) dst.size());
        const int32_t ngv = ng;
        ck(cudaMemcpyAsync(gp.d_grp_ptr, ptr.data(), sizeof(unsigned long long) * ng, cudaMemcpyHostToDevice, gp.s), "grp_ptr");
        ck(cudaMemcpyAsync(gp.d_grp_start, start.data(), sizeof(int32_t) * (ng + 1), cudaMemcpyHostToDevice, gp.s), "grp_start");
        ck(cudaMemcpyAsync(gp.d_ngroups, &ngv, sizeof(int32_t), cudaMemcpyHostToDevice, gp.s), "ngroups");
        ck(cudaMemcpyAsync(gp.d_ent_dst, dst.data(), sizeof(int32_t) * dst.size(), cudaMemcpyHostToDevice, gp.s), "ent_dst");
        ck(cudaMemcpyAsync(gp.d_ent_tok, tok.data(), sizeof(int32_t) * tok.size(), cudaMemcpyHostToDevice, gp.s), "ent_tok");
        strata::kernels::native_expert_grouped(gp.gl, (const unsigned long long*) gp.d_grp_ptr,
                                              (const int32_t*) gp.d_grp_start, (const int32_t*) gp.d_ngroups,
                                              (const int32_t*) gp.d_ent_dst, (const int32_t*) gp.d_ent_tok, ng,
                                              (int64_t) dst.size(), gp.d_xq, gp.d_scratch, (float*) gp.d_parts, gp.s, ng);
    };
    if (npf > 0) ck(cudaStreamWaitEvent(gp.s, gp.ev_pf, 0), "wait prefetch");
    launch(HIT);
    launch(PCIE);

    // ---- the pool on the CPU misses, while the card works: one job per distinct expert, one row per token ----
    int nj = 0;
    if ((int) im.jobs.size() < nu) im.jobs.resize((size_t) nu);
    for (int u = 0; u < nu; ++u) {
        if (cls[u] != CPU) continue;
        bool from_file = false;
        cpu::ExpertJobMulti& jb = im.jobs[(size_t) nj];
        jb.blob = acquire(im, layer, ue[u], nj, &from_file);
        jb.nt = 0;
        for (int t = 0; t < cpu::MAXT; ++t) { jb.nact[t] = nullptr; jb.out[t] = nullptr; }
        for (int j = 0; j < NE; ++j)
            if (ent_u[j] == u) {
                jb.nact[jb.nt] = im.nact_n.data() + (size_t) (j / (int) K) * im.f.act_bytes;
                jb.out[jb.nt] = im.parts_n.data() + (size_t) j * (size_t) H;
                ++jb.nt;
            }
        ++nj;
    }
    if (nj > 0) {
        const double c0 = now_ms();
        im.pool->run_split_multi_native(im.f, im.jobs.data(), nj);
        st.cpu_ms = now_ms() - c0;
    }
    ck(cudaMemcpyAsync(gp.h_parts, gp.d_parts, (size_t) NE * (size_t) H * 4, cudaMemcpyDeviceToHost, gp.s), "d2h parts");
    ck(cudaStreamSynchronize(gp.s), "layer sync");
    const float* gpu_parts = (const float*) gp.h_parts;

    // ---- per-token weighted sums, k in order ----
    for (int t = 0; t < nt; ++t)
        for (int64_t i = 0; i < H; ++i) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) {
                const int j = t * (int) K + (int) k;
                const float* p = cls[ent_u[j]] == CPU ? im.parts_n.data() + (size_t) j * H : gpu_parts + (size_t) j * H;
                acc += (double) w[j] * (double) p[(size_t) i];
            }
            out[(size_t) t * H + i] = (float) acc;
        }

    // ---- admission of the distinct misses (as gpu_run) ----
    if (gp.cache && !im.cfg.no_cache) {
        for (int u = 0; u < nu; ++u) {
            if (cls[u] == HIT) continue;
            const uint8_t* p = gp.arena.ptr(layer, ue[u]);
            if (!p) continue;
            const int32_t sl = gp.cache->admit(layer, ue[u]);
            if (sl < 0) continue;
            std::string e2;
            if (!gp.cache->fill_slot_blocking(sl, p, e2, B)) {
                std::fprintf(stderr, "ds4_moe: fill_slot: %s\n", e2.c_str());
                std::abort();
            }
            ++im.admitted;
        }
    }
    // stats per (token, expert) lookup, so hit rates stay comparable with single-token runs
    for (int j = 0; j < NE; ++j) {
        const int c = cls[ent_u[j]];
        if (c == HIT) ++st.hits; else if (c == PCIE) ++st.pcie; else ++st.cpu;
    }
    st.prefetched_useful = npf;
    st.wall_ms = now_ms() - t0;
    im.st.add(st);
    im.pf_n = 0;
    return true;
}
#endif  // DS4_MOE_CUDA

bool Ds4MoeTier::run(int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out) {
    if (!im_->inited || !im_->pool) return false;
    if (!ids6 || !w6 || !x || !out) return false;
    if (layer < 0 || layer >= im_->g.n_layers) return false;
    if (im_->cfg.cpu_only) return cpu_run(*im_, layer, ids6, w6, x, out);
#if defined(DS4_MOE_CUDA)
    return gpu_run(*im_, layer, ids6, w6, x, nullptr, out);
#else
    return false;
#endif
}

bool Ds4MoeTier::run_multi(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, float* out) {
    if (!im_->inited || !im_->pool) return false;
    if (!ids || !w || !x || !out || nt < 1 || nt > kMaxTok) return false;
    if (layer < 0 || layer >= im_->g.n_layers) return false;
    if (im_->cfg.cpu_only) {   // the CPU tier: token by token (no sharing; it is not the verify path)
        const int64_t K = im_->g.top_k, H = im_->g.n_embd;
        for (int t = 0; t < nt; ++t)
            if (!cpu_run(*im_, layer, ids + t * K, w + t * K, x + (size_t) t * H, out + (size_t) t * H)) return false;
        return true;
    }
#if defined(DS4_MOE_CUDA)
    return gpu_run_n(*im_, layer, nt, ids, w, x, out);
#else
    return false;
#endif
}

bool Ds4MoeTier::run_dev(int64_t layer, const int32_t* ids6, const float* w6, const void* x_dev, float* out) {
    if (!im_->inited) return false;
    if (!ids6 || !w6 || !x_dev || !out) return false;
    if (layer < 0 || layer >= im_->g.n_layers) return false;
    if (im_->cfg.cpu_only) return false;   // there is no device input without a device tier
#if defined(DS4_MOE_CUDA)
    return gpu_run(*im_, layer, ids6, w6, nullptr, x_dev, out);
#else
    return false;
#endif
}

bool ds4_moe_cpu_expert(const cpu::NativeFmt& f, int64_t n_embd, int64_t n_ff, const uint8_t* blob,
                        const float* x, float* out, cpu::ExpertPool& pool) {
    (void) n_ff;
    std::vector<uint8_t> nact(cpu::kNativeActBytes);
    std::vector<float> parts((size_t) n_embd);
    cpu::ExpertJobMulti job;
    job.blob = blob;
    job.nt = 1;
    for (int t = 0; t < cpu::MAXT; ++t) {
        job.nact[t] = nullptr;
        job.out[t] = nullptr;
    }
    cpu::native_quant_act(f, x, nact.data());
    job.nact[0] = nact.data();
    job.out[0] = parts.data();
    pool.run_split_multi_native(f, &job, 1);
    std::memcpy(out, parts.data(), (size_t) n_embd * sizeof(float));
    return true;
}

void Ds4MoeTier::close() {
#if defined(DS4_MOE_CUDA)
    if (g_prof_n > 0) {
        double tot = 0;
        for (double v : g_prof) tot += v;
        std::fprintf(stderr, "tier profile (%lld layer calls, %.3f ms/call):", (long long) g_prof_n, tot / g_prof_n);
        for (int i = 0; i < 9; ++i) std::fprintf(stderr, " %s %.3f", g_prof_name[i], g_prof[i] / g_prof_n);
        std::fprintf(stderr, "\n");
    }
#endif
    if (!im_) return;
    if (im_->gpu) im_->gpu->close();
    im_->gpu.reset();
    im_->pool.reset();
    im_->owned_blobs.reset();
    im_->blobs = nullptr;
    im_->inited = false;
    im_->pf_n = 0;
}

const Ds4MoeStats& Ds4MoeTier::stats() const { return im_->st; }
void Ds4MoeTier::reset_stats() { im_->st = Ds4MoeStats(); }
void Ds4MoeTier::add_stats(const Ds4MoeStats& s) { im_->st.add(s); }
const Ds4MoeGeom& Ds4MoeTier::geom() const { return im_->g; }
const cpu::NativeFmt& Ds4MoeTier::fmt() const { return im_->f; }
const Ds4MoeConfig& Ds4MoeTier::config() const { return im_->cfg; }
cpu::ExpertPool& Ds4MoeTier::pool() { return *im_->pool; }

const char* Ds4MoeTier::mode() const {
    if (!im_->inited) return "uninitialised";
    if (im_->cfg.cpu_only) return "cpu-only";
    return "gpu";
}

int64_t Ds4MoeTier::resident() const { return im_->admitted; }
int64_t Ds4MoeTier::file_tier() const {
#if defined(DS4_MOE_CUDA)
    return im_->gpu ? im_->gpu->arena.file_tier : 0;
#else
    return 0;
#endif
}

int64_t Ds4MoeTier::arena_experts() const {
#if defined(DS4_MOE_CUDA)
    return im_->gpu ? im_->gpu->arena.used : 0;
#else
    return 0;
#endif
}

bool Ds4MoeTier::resident(int64_t layer, int64_t expert) const {
#if defined(DS4_MOE_CUDA)
    if (im_ && im_->gpu && im_->gpu->cache) return im_->gpu->cache->slot_of(layer, expert) >= 0;
#endif
    (void) layer;
    (void) expert;
    return false;
}

double Ds4MoeTier::arena_gib() const {
#if defined(DS4_MOE_CUDA)
    return im_->gpu ? (double) im_->gpu->arena.bytes / 1073741824.0 : 0.0;
#else
    return 0.0;
#endif
}

void Ds4MoeStats::add(const Ds4MoeStats& o) {
    hits += o.hits;
    prefetched_useful += o.prefetched_useful;
    prefetch_issued += o.prefetch_issued;
    prefetch_dummy += o.prefetch_dummy;
    cpu += o.cpu;
    pcie += o.pcie;
    file_tier += o.file_tier;
    gap_ms += o.gap_ms;
    hit_ms += o.hit_ms;
    pcie_ms += o.pcie_ms;
    cpu_ms += o.cpu_ms;
    wall_ms += o.wall_ms;
}

}  // namespace strata::ds4
