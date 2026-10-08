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
#include <mutex>
#include <array>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

#include "strata/artifact/ds4_geometry.hpp"

#if defined(DS4_MOE_CUDA)
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#if defined(DS4_MOE_MMQ)
#include "strata/prefill/moe_mmq.hpp"
#endif
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
    std::vector<uint64_t> off;      ///< arena index -> byte offset (blobs differ in size per layer; used + 1 entries)
    int64_t used = 0, blob = 0, file_tier = 0;
    int64_t n_layers = 0, n_experts = 0;
    bool from_ranked = false;       ///< built from a routing profile (not just index order)
    double seconds = 0.0;           ///< how long the fill took (the startup report wants this number)

    const uint8_t* ptr(int64_t layer, int64_t expert) const {
        if (!base || layer < 0 || layer >= n_layers || expert < 0 || expert >= n_experts) return nullptr;
        const int32_t s = slot_of[(size_t) (layer * n_experts + expert)];
        return s < 0 ? nullptr : (const uint8_t*) base + (size_t) off[(size_t) s];
    }
    void close() {
        owner.reset();
        base = nullptr;
        bytes = 0;
        built = false;
        from_ranked = false;
        slot_of.clear();
        off.clear();
        used = file_tier = 0;
    }
};

/// All the device state, so the CPU-only flavour has none of it to leak.
struct Gpu {
    Arena arena;
    std::unique_ptr<strata::core::ExpertCache> cache;
    strata::kernels::NativeExpertLayout gl;
    std::vector<strata::kernels::NativeExpertLayout> gll;   ///< per model layer (all == gl for a uniform geometry)
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

    /// run_chunk (prompt chunks, MiMo prefill): allocated on first use, for up to `c_tok` tokens.  The streamed
    /// experts go through two device halves of `c_half` bytes (copy stream `s_cp` fills one while `s` computes
    /// the other); file-tier blobs are read into the matching pinned bounce half first.
    int64_t c_tok = 0, c_half = 0;
    int64_t c_bhalf = 0;          ///< bytes of one pinned bounce half (file-tier reads); c_half may be larger (prestage)
    /// Ds4MoeConfig::chunk_prestage: layer `c_pre_layer`'s arena experts `c_pre_e` (ascending ids, not VRAM-resident)
    /// are DMA'd / being DMA'd into c_stage[c_pre_half] at stride `c_pre_sl`, event c_copied[c_pre_half]
    int c_pre_layer = -1, c_pre_half = -1;
    int64_t c_pre_sl = 0;
    std::vector<int32_t> c_pre_e;
    void *c_w = nullptr, *c_out = nullptr;   ///< run_chunk: the top-k weights and the summed rows on the card
    void *c_x = nullptr, *c_xq = nullptr, *c_parts = nullptr, *c_scratch = nullptr, *c_ptr = nullptr,
         *c_start = nullptr, *c_ng = nullptr, *c_dst = nullptr, *c_tokv = nullptr, *c_stage[2] = {};
    uint8_t* c_bounce[2] = {};
    float* c_hparts = nullptr;
    cudaStream_t s_cp = nullptr;
    cudaEvent_t c_copied[2] = {}, c_used[2] = {};
    /// run_chunk's MMQ products (Ds4MoeConfig::chunk_mmq): up to `m_rows` sorted entries per launch.  `m_ok[l]`:
    /// layer l's types are covered (-1 = not checked yet).
    int64_t m_rows = 0;
    void *m_gu = nullptr, *m_h = nullptr, *m_xg = nullptr, *m_xu = nullptr, *m_xd = nullptr, *m_bnd = nullptr,
         *m_iota = nullptr;
    std::vector<int8_t> m_ok;
#if defined(DS4_MOE_MMQ)
    /// Created once and never destroyed: a binary that also links ggml-cuda (mimo_generate's dense backend) resolves
    /// ~ggml_backend_cuda_context to ggml-cuda's real one, which segfaults on the context strata_mmq built.
    strata::prefill::mmq::Context* m_ctx = nullptr;
#endif
    void free_mmq() {
        for (void* p : {m_gu, m_h, m_xg, m_xu, m_xd, m_bnd, m_iota})
            if (p) cudaFree(p);
        m_gu = m_h = m_xg = m_xu = m_xd = m_bnd = m_iota = nullptr;
        m_rows = 0;
    }

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
        for (void* p : {c_x, c_xq, c_parts, c_scratch, c_ptr, c_start, c_ng, c_dst, c_tokv, c_stage[0], c_stage[1], c_w, c_out})
            if (p) cudaFree(p);
        for (uint8_t* p : c_bounce)
            if (p) cudaFreeHost(p);
        if (c_hparts) cudaFreeHost(c_hparts);
        for (int i = 0; i < 2; ++i) {
            if (c_copied[i]) cudaEventDestroy(c_copied[i]);
            if (c_used[i]) cudaEventDestroy(c_used[i]);
        }
        if (s_cp) cudaStreamDestroy(s_cp);
        free_mmq();
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
                                                     std::string& err, int64_t batch = kBatch) {
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
    // `n_layers` here = layers IN THE FILE (the routed ones); `batch` = its ubatch in tokens.  A final short ubatch
    // (a file's tail) is laid out the same way with fewer tokens; it only shifts which half its tokens count in.
    const int64_t per_batch = batch * per_token;    // records in one ubatch
    std::map<std::pair<int32_t, int32_t>, int64_t> cnt;
    for (int64_t i = 0; i + 1 < (int64_t) all.size(); i += 2) {
        const int64_t rec = i / 2;
        const int64_t token = (rec / per_batch) * batch + (rec % (batch * top_k)) / top_k;
        if ((token / kBatch) % 2 != 0) continue;    // the held-out half (alternate 512-token blocks, as route_skew)
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
    cpu::NativeFmt f;                 ///< the largest routed layer's (buffer sizing; the uniform geometry's only one)
    int64_t blob = 0;                 ///< the largest blob (buffer strides)
    std::vector<cpu::NativeFmt> fl;   ///< per MODEL layer; fl[l].bytes == 0 for a layer without routed experts
    std::vector<int64_t> bl;          ///< per layer blob bytes
    bool mixed_sizes = false;         ///< blob size differs between routed layers (MiMo): cache opened at seed
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
    std::vector<int64_t> last_use;    ///< arena_adapt: (layer, expert) -> the run() call that last used it
    int64_t tick = 0;
    std::vector<int64_t> vlast;       ///< vram_lru: (layer, expert) -> the run() call that last used its VRAM copy
    std::vector<int64_t> vmiss;       ///< vram_lru: (layer, expert) -> the layer's call count at its last miss
    std::vector<int64_t> lcalls;      ///< vram_lru: run() calls per layer (= decode tokens)
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
    uint8_t* dst = im.ftmp.data() + (size_t) scratch * (size_t) im.blob;   // max-blob stride, layer's bytes
    const double t0 = now_ms();
    im.blobs->read(layer, expert, dst);
    im.st.file_ms += now_ms() - t0;
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
    if (g.per_layer()) {   // the largest routed layer's layout: what every per-call buffer is sized by
        bool any = false;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!g.routed(l)) continue;
            cpu::NativeFmt fl;
            if (!ds4_moe_layer_layout(g, l, fl, err)) return false;
            if (!any || fl.bytes > f.bytes) f = fl;
            any = true;
        }
        if (!any) err = "per-layer geometry with no routed layer";
        return any;
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

bool ds4_moe_layer_layout(const Ds4MoeGeom& g, int64_t l, cpu::NativeFmt& f, std::string& err) {
    if (!g.per_layer()) return ds4_moe_blob_layout(g, f, err);
    if (!g.routed(l)) {
        err = "layer " + std::to_string(l) + " has no routed experts";
        return false;
    }
    if (g.top_k < 1 || g.top_k > cpu::MAXT) {
        err = "top_k " + std::to_string(g.top_k) + " is outside 1.." + std::to_string((int) cpu::MAXT);
        return false;
    }
    std::string e;
    if (!cpu::native_fmt3(g.gate_type(l), g.up_type(l), g.down_type(l), g.n_embd, g.n_ff, f, e)) {
        err = "layer " + std::to_string(l) + ": native_fmt3(" + std::to_string(g.gate_type(l)) + "/" +
              std::to_string(g.up_type(l)) + "/" + std::to_string(g.down_type(l)) + "): " + e;
        return false;
    }
    f.swiglu_limit = g.swiglu_limit;
    return true;
}

bool ds4_moe_geom_from_gguf(const std::string& gguf, Ds4MoeGeom& g, std::string& err) {
    try {
        GgufFile f(gguf);
        // the architecture's own key prefix: deepseek4.*, mimo2.*, ... (the key names are llama.cpp's, shared)
        std::string arch = "deepseek4";
        if (const MetaValue* a = f.get("general.architecture"); a && a->type == MetaType::STRING) arch = a->s;
        int64_t v = 0;
        const std::pair<std::string, int64_t*> keys[] = {
            {arch + ".block_count", &g.n_layers},
            {arch + ".embedding_length", &g.n_embd},
            {arch + ".expert_count", &g.n_experts},
            {arch + ".expert_used_count", &g.top_k},
            {arch + ".expert_feed_forward_length", &g.n_ff},
        };
        for (const auto& k : keys) {
            if (!meta_i64(f, k.first, v)) {
                err = "no metadata " + k.first + " (is this an MoE GGUF?)";
                return false;
            }
            *k.second = v;
        }
        // every layer's three expert types; a layer with none of the tensors has no routed experts (MiMo's layer 0)
        const char* roles[3] = {"gate", "up", "down"};
        std::vector<int> ty[3];
        for (int64_t l = 0; l < g.n_layers; ++l) {
            int have = 0;
            int t3[3] = {-1, -1, -1};
            for (int r = 0; r < 3; ++r) {
                char name[64];
                std::snprintf(name, sizeof name, "blk.%lld.ffn_%s_exps.weight", (long long) l, roles[r]);
                const TensorInfo* ti = f.find(name);
                if (!ti) continue;
                ++have;
                // ne0 is the input width: gate/up are [n_embd, n_ff, n_experts], down is [n_ff, n_embd, n_experts].
                if ((int64_t) ti->shape.at(0) != (r == 2 ? g.n_ff : g.n_embd)) {
                    err = std::string(name) + " has ne0 " + std::to_string(ti->shape.at(0));
                    return false;
                }
                t3[r] = (int) ti->type;
            }
            if (have != 0 && have != 3) {
                err = "layer " + std::to_string(l) + " has " + std::to_string(have) + " of the 3 expert tensors";
                return false;
            }
            for (int r = 0; r < 3; ++r) ty[r].push_back(t3[r]);
        }
        int first = -1;
        bool uniform = true;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (ty[0][(size_t) l] < 0) { uniform = false; continue; }
            if (first < 0) first = (int) l;
            uniform = uniform && ty[0][(size_t) l] == ty[0][(size_t) first] &&
                      ty[1][(size_t) l] == ty[0][(size_t) first] && ty[2][(size_t) l] == ty[2][(size_t) first];
        }
        if (first < 0) {
            err = "no layer has routed expert tensors";
            return false;
        }
        g.gu_type = ty[0][(size_t) first];
        g.d_type = ty[2][(size_t) first];
        g.gate_types.clear();
        g.up_types.clear();
        g.down_types.clear();
        if (!uniform) {   // per-layer formats (or layers without experts): the vectors carry them
            g.gate_types = ty[0];
            g.up_types = ty[1];
            g.down_types = ty[2];
        }
        // the SwiGLU clamp: one limit for the whole tier (it has one NativeFmt / layout), so every layer must agree
        std::vector<double> lim;
        const std::string le = strata::ds4_key_f64_arr(f, (arch + ".swiglu_clamp_exp").c_str(), lim);
        if (le.empty() && !lim.empty()) {
            for (double x : lim)
                if (x != lim[0]) {
                    err = arch + ".swiglu_clamp_exp differs between layers; the tier supports one limit";
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
    std::vector<const uint8_t*> src[3];   // per role, per layer (null: no routed experts)
    std::vector<size_t> stride[3];        // per role, per layer: bytes of one expert's slice
    // pread() path: per role, per layer, the shard's fd and the slice array's file offset.  A cold mmap memcpy
    // faults page by page (MiMo on ntfs3: ~1.1 GB/s even with 8 readers, FINDINGS s10); pread asks for each slice
    // in one request.  Same bytes.  STRATA_BLOB_MMAP=1 keeps the memcpy path.
    std::vector<int> fd[3];
    std::vector<uint64_t> foff[3];
    std::vector<int> fds;
    // O_DIRECT twins of `fds` (-1 where the filesystem refused it): 4 KiB-aligned spans into a per-thread bounce,
    // bypassing the page cache (MiMo's NVMe: 5.5 GB/s direct vs 2.6 buffered sequential).  STRATA_BLOB_DIRECT=0 off.
    std::vector<int> dfds;
    std::vector<int> dfd[3];
    ~Impl() {
        for (int f : fds) if (f >= 0) ::close(f);
        for (int f : dfds) if (f >= 0) ::close(f);
    }
};

Ds4GgufBlobs::Ds4GgufBlobs(Ds4MoeGeom g, std::string gguf) : im_(new Impl), g_(g), path_(std::move(gguf)) {}
Ds4GgufBlobs::~Ds4GgufBlobs() = default;

bool Ds4GgufBlobs::open(std::string& err) {
    cpu::NativeFmt f;
    if (!ds4_moe_blob_layout(g_, f, err)) return false;
    blob_ = f.bytes;   // the largest layer's
    try {
        im_->model.reset(new GgufModel(gguf_split_paths(path_)));
        const char* roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            im_->src[r].assign((size_t) g_.n_layers, nullptr);
            im_->stride[r].assign((size_t) g_.n_layers, 0);
        }
        for (int64_t l = 0; l < g_.n_layers; ++l) {
            if (!g_.routed(l)) continue;
            cpu::NativeFmt fl;
            if (!ds4_moe_layer_layout(g_, l, fl, err)) return false;
            im_->stride[0][(size_t) l] = fl.up_off;
            im_->stride[1][(size_t) l] = fl.down_off - fl.up_off;
            im_->stride[2][(size_t) l] = fl.bytes - fl.down_off;
            const int want[3] = {g_.gate_type(l), g_.up_type(l), g_.down_type(l)};
            for (int r = 0; r < 3; ++r) {
                char name[64];
                std::snprintf(name, sizeof name, "blk.%lld.ffn_%s_exps.weight", (long long) l, roles[r]);
                size_t at = 0;
                const TensorInfo* ti = im_->model->find(name, &at);
                if (!ti) {
                    err = std::string("no tensor ") + name;
                    return false;
                }
                if (ti->type != (uint32_t) want[r]) {
                    err = std::string(name) + " has type " + ti->type_name();
                    return false;
                }
                im_->src[r][(size_t) l] = im_->model->shard(at).tensor_data(*ti);
                if (!std::getenv("STRATA_BLOB_MMAP")) {
                    if (im_->fds.size() < im_->model->size()) im_->fds.assign(im_->model->size(), -1);
                    int& fdv = im_->fds[at];
                    if (fdv < 0) fdv = ::open(im_->model->shard(at).path().c_str(), O_RDONLY | O_CLOEXEC);
                    if (im_->fd[r].empty()) { im_->fd[r].assign((size_t) g_.n_layers, -1); im_->foff[r].assign((size_t) g_.n_layers, 0); }
                    im_->fd[r][(size_t) l] = fdv;
                    const char* dv = std::getenv("STRATA_BLOB_DIRECT");
                    if (!(dv && dv[0] == '0')) {
                        if (im_->dfds.size() < im_->model->size()) im_->dfds.assign(im_->model->size(), -2);
                        int& dfv = im_->dfds[at];
                        if (dfv == -2) dfv = ::open(im_->model->shard(at).path().c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
                        if (im_->dfd[r].empty()) im_->dfd[r].assign((size_t) g_.n_layers, -1);
                        im_->dfd[r][(size_t) l] = dfv;
                    }
                    im_->foff[r][(size_t) l] = im_->model->shard(at).data_start() + ti->offset;
                }
            }
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

size_t Ds4GgufBlobs::blob_bytes(int64_t l) const {
    if (l < 0 || l >= g_.n_layers) return 0;
    return im_->stride[0][(size_t) l] + im_->stride[1][(size_t) l] + im_->stride[2][(size_t) l];
}

void Ds4GgufBlobs::read_span(int r, int64_t layer, int64_t expert, size_t lo, size_t hi, uint8_t* dst) const {
    const size_t sz = im_->stride[r][(size_t) layer];
    if (hi > sz) hi = sz;
    if (lo >= hi) return;
    const size_t n = hi - lo;
    const int fdv = im_->fd[r].empty() ? -1 : im_->fd[r][(size_t) layer];
    const int dfv = im_->dfd[r].empty() ? -1 : im_->dfd[r][(size_t) layer];
    const uint64_t off = (fdv >= 0 || dfv >= 0) ? im_->foff[r][(size_t) layer] + (uint64_t) expert * sz + lo : 0;
    if (dfv >= 0) {   // O_DIRECT: the aligned span around the range, then the range out of it
        constexpr uint64_t A = 4096;
        const uint64_t a0 = off & ~(A - 1), a1 = (off + n + A - 1) & ~(A - 1);
        struct Bounce { uint8_t* p = nullptr; size_t n = 0; ~Bounce() { std::free(p); } };
        thread_local Bounce bb;
        if (bb.n < a1 - a0) {
            std::free(bb.p);
            bb.p = nullptr;
            bb.n = 0;
            void* q = nullptr;
            if (posix_memalign(&q, A, (size_t) (a1 - a0)) == 0) { bb.p = (uint8_t*) q; bb.n = (size_t) (a1 - a0); }
        }
        if (bb.p) {
            size_t got = 0;
            while (got < a1 - a0) {
                const ssize_t k = ::pread(dfv, bb.p + got, (size_t) (a1 - a0 - got), (off_t) (a0 + got));
                if (k <= 0) break;
                got += (size_t) k;
            }
            if (got >= off + n - a0) {   // the file's last span may be short of a1; the range is all we need
                std::memcpy(dst, bb.p + (off - a0), n);
                return;
            }
        }
    }
    if (fdv >= 0) {   // pread the range; fall back to the mmap on any short read / error
        size_t got = 0;
        while (got < n) {
            const ssize_t k = ::pread(fdv, dst + got, n - got, (off_t) (off + got));
            if (k <= 0) break;
            got += (size_t) k;
        }
        if (got == n) return;
    }
    std::memcpy(dst, im_->src[r][(size_t) layer] + (size_t) expert * sz + lo, n);
}

void Ds4GgufBlobs::read(int64_t layer, int64_t expert, uint8_t* dst) const {
    // The three role slices back to back, [gate rows | up rows | down rows] (native_expert.hpp's layout).
    // Per-layer strides (gate and up may differ: MiMo).
    size_t at = 0;
    for (int r = 0; r < 3; ++r) {
        const size_t sz = im_->stride[r][(size_t) layer];
        read_span(r, layer, expert, 0, sz, dst + at);
        at += sz;
    }
}

void Ds4GgufBlobs::read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const {
    if (nparts <= 1) {
        if (part == 0) read(layer, expert, dst);
        return;
    }
    const size_t bl = blob_bytes(layer);
    auto cut = [&](int q) { return q >= nparts ? bl : std::min(bl, (bl * (size_t) q / (size_t) nparts) & ~(size_t) 4095); };
    const size_t p0 = cut(part), p1 = cut(part + 1);
    size_t at = 0;
    for (int r = 0; r < 3 && at < p1; ++r) {
        const size_t sz = im_->stride[r][(size_t) layer];
        const size_t lo = std::max(p0, at), hi = std::min(p1, at + sz);
        if (lo < hi) read_span(r, layer, expert, lo - at, hi - at, dst + lo);
        at += sz;
    }
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
    // per-layer formats and blob sizes (all equal to f / blob for a uniform geometry)
    im_->fl.assign((size_t) geom.n_layers, cpu::NativeFmt());
    im_->bl.assign((size_t) geom.n_layers, 0);
    im_->mixed_sizes = false;
    for (int64_t l = 0; l < geom.n_layers; ++l) {
        if (!geom.routed(l)) continue;
        if (!geom.per_layer()) {
            im_->fl[(size_t) l] = im_->f;
        } else if (!ds4_moe_layer_layout(geom, l, im_->fl[(size_t) l], err)) {
            return false;
        }
        im_->bl[(size_t) l] = (int64_t) im_->fl[(size_t) l].bytes;
        im_->mixed_sizes = im_->mixed_sizes || im_->bl[(size_t) l] != im_->blob;
    }
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
    gp.gll.assign((size_t) geom.n_layers, gp.gl);
    for (int64_t l = 0; l < geom.n_layers; ++l) {
        if (!geom.routed(l)) continue;
        const int gt = geom.gate_type(l), ut = geom.up_type(l), dt = geom.down_type(l);
        if (!strata::kernels::native_expert_supported3(gt, ut, dt, geom.n_embd, geom.n_ff)) {
            err = "native_expert_grouped has no kernel for layer " + std::to_string(l) + "'s types " +
                  std::to_string(gt) + "/" + std::to_string(ut) + "/" + std::to_string(dt);
            return false;
        }
        strata::kernels::NativeExpertLayout L = strata::kernels::native_expert_layout3(gt, ut, dt, geom.n_embd, geom.n_ff);
        L.swiglu_limit = geom.swiglu_limit;
        if ((int64_t) L.bytes != im_->bl[(size_t) l]) {
            err = "layer " + std::to_string(l) + ": GPU blob " + std::to_string(L.bytes) + " != CPU blob " +
                  std::to_string(im_->bl[(size_t) l]);
            return false;
        }
        gp.gll[(size_t) l] = L;
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
    // a geometry whose blob size varies per layer opens its cache at the seed, sized to the ranked experts
    if (!cfg.no_cache && cfg.slots > 0 && !im_->mixed_sizes && !cfg.defer_cache) {   // defer_cache: opened at the seed
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


/// The arena's ranking: with `arena_skip_resident`, minus the head the VRAM seed takes
/// (the same walk as the cache seed: ranked order, up to `slots` experts or the `slot_gib` bytes).
std::vector<std::pair<int32_t, int32_t>> arena_order(const Ds4MoeImpl& im, const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    if (!(im.cfg.arena_skip_resident && !im.cfg.no_cache && im.cfg.slots > 0)) return ranked;
    const double budget = im.cfg.slot_gib > 0 ? im.cfg.slot_gib * 1073741824.0 : (double) im.cfg.slots * (double) im.blob;
    std::vector<char> res((size_t) (im.g.n_layers * im.g.n_experts), 0);
    double used = 0;
    int64_t n = 0;
    for (const auto& le : ranked) {
        if (n >= im.cfg.slots) break;
        if (!im.g.routed(le.first) || le.second < 0 || le.second >= im.g.n_experts) continue;
        char& r = res[(size_t) (le.first * im.g.n_experts + le.second)];
        if (r) continue;
        const int64_t b = im.bl[(size_t) le.first];
        if (used + (double) b > budget) break;
        r = 1;
        used += (double) b;
        ++n;
    }
    std::vector<std::pair<int32_t, int32_t>> rest;
    rest.reserve(ranked.size());
    for (const auto& le : ranked)
        if (le.first < 0 || le.first >= im.g.n_layers || le.second < 0 || le.second >= im.g.n_experts ||
            !res[(size_t) (le.first * im.g.n_experts + le.second)])
            rest.push_back(le);
    return rest;
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
    const uint64_t cap_bytes = (uint64_t) (budget_gib * 1073741824.0);
    std::vector<std::pair<int32_t, int32_t>> order = ranked;
    if (order.empty()) {
        order.reserve((size_t) total);
        for (int64_t l = 0; l < im_->g.n_layers; ++l)
            for (int64_t e = 0; e < im_->g.n_experts; ++e)
                if (im_->g.routed(l)) order.push_back({(int32_t) l, (int32_t) e});
    }
    gp.arena.slot_of.assign((size_t) total, -1);
    gp.arena.off.assign(1, 0);
    std::vector<std::pair<int32_t, int32_t>> first;
    first.reserve(order.size());
    for (const auto& le : order) {
        const int64_t l = le.first, e = le.second;
        if (l < 0 || l >= im_->g.n_layers || e < 0 || e >= im_->g.n_experts) continue;
        if (!im_->g.routed(l)) continue;
        if (gp.arena.slot_of[(size_t) (l * im_->g.n_experts + e)] >= 0) continue;
        const uint64_t b = (uint64_t) im_->bl[(size_t) l];
        if (gp.arena.off.back() + b > cap_bytes) {   // the budget in BYTES: blobs differ per layer (MiMo)
            ++gp.arena.file_tier;
            continue;
        }
        gp.arena.slot_of[(size_t) (l * im_->g.n_experts + e)] = (int32_t) first.size();
        first.push_back({(int32_t) l, (int32_t) e});
        gp.arena.off.push_back(gp.arena.off.back() + b);
    }
    gp.arena.used = (int64_t) first.size();
    gp.arena.bytes = gp.arena.off.back();
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
                                 (uint8_t*) gp.arena.base + (size_t) gp.arena.off[(size_t) i]);
            }
        });
    for (auto& t : th) t.join();
    gp.arena.seconds = (now_ms() - t0) / 1000.0;
    return true;
#endif
}

bool Ds4MoeTier::seed_from_routes(const std::string& routes_bin, std::string& err) {
    return seed_from_routes(routes_bin, kBatch, err);
}

bool Ds4MoeTier::seed_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err) {
    std::string e;
    const std::vector<std::pair<int32_t, int32_t>> ranked =
        rank_routes(routes_bin, im_->g.n_routed(), im_->g.top_k, e, batch_tokens);
    if (ranked.empty()) {
        err = e.empty() ? "empty routing profile" : e;
        return false;
    }
    return seed_from_ranked(ranked, err);
}

bool Ds4MoeTier::seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (im_->cfg.cpu_only) return true;   // a CPU-only tier has neither a cache nor an arena to seed
    if (!build_arena(arena_order(*im_, ranked), err)) return false;
#if !defined(DS4_MOE_CUDA)
    (void) ranked;
    err = "the VRAM cache needs the CUDA tier (compiled without DS4_MOE_CUDA)";
    return false;
#else
    // A per-layer-sized geometry (MiMo) opens its cache now: one slot per ranked expert, each the size of ITS
    // layer's blob, until `slots` experts or `slot_gib` bytes.  The seed below admits in the same order, so slot i
    // is ranked[i]'s and fits it exactly; the cache never evicts, so a slot never meets another layer's blob.
    if (im_->mixed_sizes && im_->gpu && !im_->gpu->cache && !im_->cfg.no_cache && im_->cfg.slots > 0) {
        const double budget = im_->cfg.slot_gib > 0 ? im_->cfg.slot_gib * 1073741824.0
                                                     : (double) im_->cfg.slots * (double) im_->blob;
        std::vector<int64_t> sizes;
        std::vector<char> seen((size_t) (im_->g.n_layers * im_->g.n_experts), 0);
        double used = 0;
        for (const auto& le : ranked) {
            if ((int64_t) sizes.size() >= im_->cfg.slots) break;
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= im_->g.n_experts) continue;
            char& sn = seen[(size_t) (le.first * im_->g.n_experts + le.second)];
            if (sn) continue;
            const int64_t b = im_->bl[(size_t) le.first];
            if (used + (double) b > budget) break;
            sn = 1;
            sizes.push_back(b);
            used += (double) b;
        }
        if (!sizes.empty()) {
            im_->gpu->cache.reset(new strata::core::ExpertCache());
            if (!im_->gpu->cache->open_sized(sizes, im_->g.n_layers, im_->g.n_experts, err)) return false;
        }
    }
    // uniform blobs with defer_cache: the cache opens now (after the prompt chunks gave their VRAM back)
    if (!im_->mixed_sizes && im_->cfg.defer_cache && im_->gpu && !im_->gpu->cache && !im_->cfg.no_cache &&
        im_->cfg.slots > 0) {
        std::vector<int64_t> sizes((size_t) im_->cfg.slots, im_->blob);
        im_->gpu->cache.reset(new strata::core::ExpertCache());
        if (!im_->gpu->cache->open_sized(sizes, im_->g.n_layers, im_->g.n_experts, err)) return false;
    }
    if (!im_->gpu || !im_->gpu->cache) return true;   // no cache: nothing to seed
    const int64_t want = std::min<int64_t>(im_->cfg.slots, im_->gpu->cache->full_slots());
    int64_t filled = 0;
    // vram_lru: the LRU's starting order = the profile's rank: the coldest seeded expert goes first
    auto lru_seed_order = [&](int64_t n_seeded) {
        if (!im_->cfg.vram_lru) return;
        const int64_t NX = im_->g.n_experts;
        im_->vlast.assign((size_t) (im_->g.n_layers * NX), -1);
        int64_t r = 0;
        for (const auto& le : ranked) {
            if (r >= n_seeded) break;
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= NX) continue;
            int64_t& v = im_->vlast[(size_t) (le.first * NX + le.second)];
            if (v != -1 || im_->gpu->cache->slot_of(le.first, le.second) < 0) continue;
            v = -2 - r++;   // rank 0 -> -2 (most recent of the seeds), the last seeded -> the oldest
        }
    };
    if (im_->mixed_sizes || !std::getenv("DS4_SERIAL_SEED")) {
        // MiMo (and DS4 since 2026-10-08: the serial seed below took 8 s for 2226 slots after chunked prefill): the
        // same admissions in the same order, but the copies overlap - arena blobs queued straight from the
        // pinned arena, file-tier blobs read by 8 threads into pinned buffers (the serial seed read one blob at a time:
        // ~7 s for ~1850 experts after a prompt).  Same slots, same bytes.
        strata::core::ExpertCache& cache = *im_->gpu->cache;
        std::vector<std::array<int64_t, 3>> fjobs;   // slot, layer, expert
        for (const auto& le : ranked) {
            if (filled >= want) break;
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= im_->g.n_experts) continue;
            if (cache.slot_of(le.first, le.second) >= 0) continue;
            const int32_t s = cache.admit(le.first, le.second);
            if (s < 0) break;
            if (const uint8_t* p = im_->gpu->arena.ptr(le.first, le.second)) {
                if (!cache.fill_slot_queued(s, p, err, im_->bl[(size_t) le.first])) return false;
            } else {
                fjobs.push_back({s, le.first, le.second});
            }
            ++filled;
        }
        std::atomic<size_t> nx{0};
        std::atomic<bool> bad{false};
        std::mutex mu;
        std::string terr;
        std::vector<std::thread> th;
        const double t0 = now_ms();
        const int nth = (int) std::min<size_t>(8, fjobs.size());
        for (int k = 0; k < nth; ++k)
            th.emplace_back([&]() {
                uint8_t* buf = nullptr;
                if (cudaMallocHost((void**) &buf, (size_t) im_->blob) != cudaSuccess) {
                    bad = true;
                    return;
                }
                for (size_t q; !bad && (q = nx.fetch_add(1)) < fjobs.size();) {
                    const auto& j = fjobs[q];
                    im_->blobs->read(j[1], j[2], buf);
                    std::lock_guard<std::mutex> lk(mu);   // the cache's counters; the copy is ~0.4 ms of a ~2 ms read
                    std::string e;
                    if (!cache.fill_slot_blocking((int32_t) j[0], buf, e, im_->bl[(size_t) j[1]])) {
                        terr = e;
                        bad = true;
                    }
                }
                cudaFreeHost(buf);
            });
        for (auto& t : th) t.join();
        if (cudaStreamSynchronize((cudaStream_t) 0) != cudaSuccess && !bad) {
            terr = "seed: queued fills failed";
            bad = true;
        }
        im_->st.file_tier += (int64_t) fjobs.size();
        im_->st.file_ms += now_ms() - t0;
        lru_seed_order(filled);
        if (bad) {
            err = terr.empty() ? "seed: pinned buffer allocation failed" : terr;
            return false;
        }
        im_->admitted = filled;
        return true;
    }
    for (const auto& le : ranked) {
        if (filled >= want) break;
        if (im_->mixed_sizes) {   // keep slot i == the i-th distinct routed entry (the sizes list above)
            if (!im_->g.routed(le.first) || le.second < 0 || le.second >= im_->g.n_experts) continue;
            if (im_->gpu->cache->slot_of(le.first, le.second) >= 0) continue;
        }
        const int32_t s = im_->gpu->cache->admit(le.first, le.second);
        if (s < 0) break;
        bool from_file = false;
        const uint8_t* p = acquire(*im_, le.first, le.second, 0, &from_file);
        if (!im_->gpu->cache->fill_slot_blocking(s, p, err, im_->bl[(size_t) le.first])) return false;
        ++filled;
    }
    lru_seed_order(filled);
    im_->admitted = filled;
    return true;
#endif
}

void Ds4MoeTier::prefetch(int64_t layer, const int32_t* top16, int n) {
    im_->pf_n = 0;
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu || im_->cfg.cpu_only || im_->cfg.pf_b <= 0 || im_->cfg.no_cache || !im_->gpu->cache) return;
    if (!top16 || n <= 0 || !im_->g.routed(layer)) return;
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
        bool dummy = false;
        if (!src) {
            src = gp.h_dummy;   // a guess outside the arena still costs its DMA (the replay's own rule)
            ++im_->st.prefetch_dummy;
            dummy = true;
        }
        ck(cudaMemcpyAsync((uint8_t*) gp.d_pf + (size_t) im_->pf_n * (size_t) im_->blob, src,
                           (size_t) im_->bl[(size_t) layer], cudaMemcpyHostToDevice, gp.s_pf), "prefetch dma");
        // A dummy slot holds ZEROS, not the expert: it must never be matched by run() (found by
        // tools/mimo/test_mimo_moe: prefetched experts outside the arena were computed from the zero blob, rel 0.2-0.5).
        // -1 matches no expert id; the DMA above still happens, so the replay-measured timing is unchanged.
        im_->pf_e[im_->pf_n++] = dummy ? -1 : e;
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
    const cpu::NativeFmt& f = im.fl[(size_t) layer];
    cpu::native_quant_act(f, x, im.nact.data());
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
    im.pool->run_split_multi_native(f, im.jobs.data(), (int) K);
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
    const int64_t B = im.blob;                        // staging STRIDE (the largest blob)
    const int64_t BL = im.bl[(size_t) layer];         // this layer's blob (the bytes a DMA moves)
    const bool lru = im.cfg.vram_lru && gp.cache && !im.cfg.no_cache;   // uniform (DS4) too: every slot fits any blob
    if (lru && !im.cfg.arena_adapt) ++im.tick;   // (arena_adapt bumps it below)
    if (im.cfg.arena_adapt) {   // every lookup of this layer is a use (the LRU the arena swaps below evict by)
        const int64_t NX = im.g.n_experts;
        if (im.last_use.empty()) im.last_use.assign((size_t) (im.g.n_layers * NX), -1);
        ++im.tick;
        for (int64_t k = 0; k < K; ++k) im.last_use[(size_t) (layer * NX + ids6[k])] = im.tick;
    }
    const cpu::NativeFmt& f = im.fl[(size_t) layer];
    const strata::kernels::NativeExpertLayout& GL = gp.gll[(size_t) layer];
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
    // skip_miss: a miss with a small gate weight is dropped - its 6.75 MB never moves (wloc = the weights the sum uses)
    float wloc[kMaxParts];
    for (int64_t k = 0; k < K; ++k) wloc[k] = w6[k];
    if (im.cfg.skip_miss > 0.0f && nc > 0) {
        double ws = 0;
        for (int64_t k = 0; k < K; ++k) ws += (double) w6[k];
        int keep = 0;
        for (int i = 0; i < nc; ++i) {
            const int32_t k = cpu_i[i];
            if ((double) w6[k] < (double) im.cfg.skip_miss * ws) { wloc[k] = 0.0f; ++st.skipped; }
            else cpu_i[keep++] = k;
        }
        nc = keep;
    }
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
    cpu::native_quant_act(f, gp.h_x, im.nact.data());
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
        strata::kernels::native_expert_grouped(GL, (const unsigned long long*) gp.d_grp_ptr,
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
    // ---- vram_lru: the layer's least-recently-used resident expert not routed to now gives up its slot ----
    const int64_t NXL = im.g.n_experts;
    int32_t lru_new[2 * kMaxParts], lru_old[2 * kMaxParts], lru_dem[2 * kMaxParts];   // dem: arena slot or -1
    int n_lru = 0;
    // Exclusive tiers: a victim with no arena copy (the seed leaves VRAM experts out of the arena) is DEMOTED into the
    // arena slot the incoming expert vacates - else its next use would be a file read (FINDINGS s16: 5.9 -> 16.6
    // file reads/token without this).  Both copies run on gp.s: the D2H after the incoming blob left the arena slot,
    // the slot write after the D2H.
    auto arena_slot = [&](int32_t e) { return gp.arena.base ? gp.arena.slot_of[(size_t) (layer * NXL + e)] : -1; };
    auto demote = [&](int32_t victim_slot, int32_t into_arena) {
        ck(cudaMemcpyAsync((uint8_t*) gp.arena.base + (size_t) gp.arena.off[(size_t) into_arena],
                           gp.cache->device_slot(victim_slot), (size_t) BL, cudaMemcpyDeviceToHost, gp.s), "lru demote");
    };
    auto lru_victim = [&]() -> int32_t {
        int32_t v = -1;
        int64_t best = INT64_MAX;
        for (int64_t e = 0; e < NXL; ++e) {
            if (gp.cache->slot_of(layer, e) < 0) continue;
            const int64_t u = im.vlast[(size_t) (layer * NXL + e)];
            if (u >= im.tick) continue;   // used by this call (a hit) or already taken by it
            if (u < best) { best = u; v = (int32_t) e; }
        }
        return v;
    };
    // admission filter (cache_sim: admit-all 61.6% hit at 144 swaps/token; "second miss within 4 tokens" 63.0% at
    // 24.6): an expert takes a slot only if it also missed in one of the layer's previous kLruWin calls
    constexpr int64_t kLruWin = 4;
    int64_t lc = 0;
    auto admit_ok = [&](int32_t e) {
        const int64_t m = im.vmiss[(size_t) (layer * NXL + e)];
        return m >= 0 && lc - m <= kLruWin;
    };
    if (lru) {
        if (im.vlast.empty()) im.vlast.assign((size_t) (im.g.n_layers * NXL), -1);
        if (im.vmiss.empty()) im.vmiss.assign((size_t) (im.g.n_layers * NXL), -1);
        if (im.lcalls.empty()) im.lcalls.assign((size_t) im.g.n_layers, 0);
        lc = ++im.lcalls[(size_t) layer];
        for (int64_t k = 0; k < K; ++k) im.vlast[(size_t) (layer * NXL + ids6[k])] = im.tick;
        // prefetched experts the routing used: device-to-device into a victim slot (the hits kernel above read the
        // staging copy; this copy only reads it too, and the next prefetch waits for the layer sync)
        for (int i = 0; i < nh; ++i) {
            const int k = hit_i[i];
            if (!pf_ptr[k] || !admit_ok(ids6[k])) continue;
            const int32_t v = lru_victim();
            if (v < 0) break;
            const int32_t sl = gp.cache->slot_of(layer, v);
            int32_t dem = -1;
            if (arena_slot(v) < 0) {
                dem = arena_slot(ids6[k]);   // prefetches come from the arena (never the dummy: pf_e = -1 then)
                if (dem < 0) continue;
                demote(sl, dem);
            }
            ck(cudaMemcpyAsync(gp.cache->device_slot(sl), (const void*) pf_ptr[k], (size_t) BL, cudaMemcpyDeviceToDevice,
                               gp.s), "lru pf copy");
            im.vlast[(size_t) (layer * NXL + v)] = im.tick;   // not again in this call
            lru_old[n_lru] = v;
            lru_dem[n_lru] = dem;
            lru_new[n_lru++] = ids6[k];
        }
    }
    if (np > 0) {
        std::vector<unsigned long long> ptr((size_t) np);
        std::vector<int32_t> start((size_t) np + 1), dst((size_t) np, 0), tok((size_t) np, 0);
        for (int i = 0; i < np; ++i) {
            const uint8_t* p = gp.arena.ptr(layer, ids6[pcie_i[i]]);
            uint8_t* to = (uint8_t*) gp.d_stage + (size_t) i * (size_t) B;
            int32_t v = lru && admit_ok(ids6[pcie_i[i]]) ? lru_victim() : -1;
            if (v >= 0 && arena_slot(v) < 0 && arena_slot(ids6[pcie_i[i]]) < 0) v = -1;   // nowhere to demote to
            if (v >= 0) {
                uint8_t* vs = gp.cache->device_slot(gp.cache->slot_of(layer, v));
                im.vlast[(size_t) (layer * NXL + v)] = im.tick;
                int32_t dem = -1;
                if (arena_slot(v) < 0) {   // demote: arena -> stage, victim -> arena slot, stage -> victim slot
                    dem = arena_slot(ids6[pcie_i[i]]);
                    ck(cudaMemcpyAsync(to, p, (size_t) BL, cudaMemcpyHostToDevice, gp.s), "pcie dma");
                    demote(gp.cache->slot_of(layer, v), dem);
                    ck(cudaMemcpyAsync(vs, to, (size_t) BL, cudaMemcpyDeviceToDevice, gp.s), "lru stage copy");
                } else {   // the DMA lands in the victim slot: the kernel below reads it there, and it stays
                    to = vs;
                    ck(cudaMemcpyAsync(to, p, (size_t) BL, cudaMemcpyHostToDevice, gp.s), "pcie dma");
                }
                lru_old[n_lru] = v;
                lru_dem[n_lru] = dem;
                lru_new[n_lru++] = ids6[pcie_i[i]];
            } else {
                ck(cudaMemcpyAsync(to, p, (size_t) BL, cudaMemcpyHostToDevice, gp.s), "pcie dma");
            }
            ptr[(size_t) i] = (unsigned long long) to;
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
        // arena blobs by address; file-tier blobs read IN PARALLEL (an NVMe needs several reads in flight - serial
        // 9 MB reads measured 1.6-5.3 ms each on MiMo, FINDINGS s9/s10) into their own ftmp slots - same bytes as
        // acquire() reads one by one
        std::vector<int> fi;
        for (int i = 0; i < nk; ++i) {
            held[(size_t) i] = gp.arena.ptr(layer, ids6[cpu_keep[i]]);
            if (!held[(size_t) i]) fi.push_back(i);
        }
        // the file reads run WHILE the pool computes the arena-held misses (they are mostly NVMe wait; FINDINGS s10:
        // ~36 ms/token serial before the pool); then the pool computes the file-tier ones
        auto fill_job = [&](int slot, int i) {
            cpu::ExpertJobMulti& j = im.jobs[(size_t) slot];
            j.blob = held[(size_t) i];
            j.nt = 1;
            for (int t = 0; t < cpu::MAXT; ++t) {
                j.nact[t] = nullptr;
                j.out[t] = nullptr;
            }
            j.nact[0] = im.nact.data();
            j.out[0] = im.parts.data() + (size_t) cpu_keep[i] * H;
        };
        std::vector<std::thread> rd;
        std::atomic<size_t> nx{0};
        double tf = 0;
        // File-tier blobs: with arena_adapt each is read STRAIGHT into the arena slot it will take (the layer's
        // least-recently-used arena expert not routed now) - no scratch copy - and one blob is split over up to 8
        // threads (read_part), since one 9 MB read alone ran ~3.3 ms (FINDINGS s16).  Same bytes as acquire().
        const int64_t NX = im.g.n_experts;
        const bool direct = im.cfg.arena_adapt && gp.arena.base && !im.last_use.empty();
        std::vector<uint8_t*> fdst(fi.size(), nullptr);
        std::vector<int32_t> fvic(fi.size(), -1);
        if (!fi.empty()) {
            if (im.ftmp.size() < (size_t) nk * (size_t) im.blob) im.ftmp.resize((size_t) nk * (size_t) im.blob);
            if (direct && gp.s_pf) ck(cudaStreamSynchronize(gp.s_pf), "pf sync");   // no prefetch DMA reads a victim
            for (size_t q = 0; q < fi.size(); ++q) {
                fdst[q] = im.ftmp.data() + (size_t) fi[q] * (size_t) im.blob;
                if (!direct) continue;
                int64_t victim = -1, best = INT64_MAX;
                for (int64_t v = 0; v < NX; ++v) {
                    if (gp.arena.slot_of[(size_t) (layer * NX + v)] < 0) continue;
                    const int64_t u = im.last_use[(size_t) (layer * NX + v)];
                    if (u >= im.tick) continue;   // routed now, or already this call's victim
                    bool taken = false;
                    for (size_t r = 0; r < q && !taken; ++r) taken = fvic[r] == (int32_t) v;
                    if (taken) continue;
                    if (u < best) { best = u; victim = v; }
                }
                if (victim < 0) continue;
                fvic[q] = (int32_t) victim;
                fdst[q] = (uint8_t*) gp.arena.base +
                          (size_t) gp.arena.off[(size_t) gp.arena.slot_of[(size_t) (layer * NX + victim)]];
            }
            tf = now_ms();
            const int parts = (int) std::max<size_t>(1, std::min<size_t>(8, 8 / fi.size()));
            const size_t tasks = fi.size() * (size_t) parts;
            const int nth = (int) std::min<size_t>(8, tasks);
            for (int q = 0; q < nth; ++q)
                rd.emplace_back([&, parts, tasks]() {
                    for (size_t r; (r = nx.fetch_add(1)) < tasks;) {
                        const size_t b = r / (size_t) parts;
                        im.blobs->read_part(layer, ids6[cpu_keep[fi[b]]], (int) (r % (size_t) parts), parts, fdst[b]);
                    }
                });
        }
        int na = 0;
        for (int i = 0; i < nk; ++i)
            if (held[(size_t) i]) fill_job(na++, i);
        const double c0 = now_ms();
        if (na > 0) im.pool->run_split_multi_native(f, im.jobs.data(), na);
        if (!fi.empty()) {
            for (auto& t : rd) t.join();
            im.st.file_tier += (int64_t) fi.size();
            im.st.file_ms += now_ms() - tf;   // read wall time, now overlapped with the pool above
            static const bool check_read = std::getenv("DS4_CHECK_READ") != nullptr;
            int nf = 0;
            for (size_t q = 0; q < fi.size(); ++q) {
                const int i = fi[q];
                held[(size_t) i] = fdst[q];
                if (check_read) {   // DS4_CHECK_READ=1: the split read against one whole read
                    std::vector<uint8_t> ref((size_t) im.blob);
                    im.blobs->read(layer, ids6[cpu_keep[i]], ref.data());
                    if (std::memcmp(ref.data(), fdst[q], (size_t) BL) != 0)
                        std::fprintf(stderr, "[read-chk] L%lld e%d: split read DIFFERS\n", (long long) layer,
                                     ids6[cpu_keep[i]]);
                    static int64_t n_chk = 0;
                    if (++n_chk % 200 == 0) std::fprintf(stderr, "[read-chk] %lld blobs checked\n", (long long) n_chk);
                }
                fill_job(na + nf++, i);
            }
            im.pool->run_split_multi_native(f, im.jobs.data() + na, nf);
        }
        st.cpu_ms = now_ms() - c0;
        // arena_adapt: the blobs read above already sit in their victims' slots - publish the swaps
        for (size_t q = 0; q < fi.size(); ++q) {
            if (fvic[q] < 0) continue;
            const int32_t e = ids6[cpu_keep[fi[q]]];
            const int32_t slot = gp.arena.slot_of[(size_t) (layer * NX + fvic[q])];
            gp.arena.slot_of[(size_t) (layer * NX + fvic[q])] = -1;
            gp.arena.slot_of[(size_t) (layer * NX + e)] = slot;
            ++im.st.arena_swaps;
        }
    }

    mark(3);
    ck(cudaStreamSynchronize(gp.s), "layer sync");
    if (lru)   // every lookup that was not a VRAM hit is a miss for the admission filter
        for (int64_t k = 0; k < K; ++k)
            if (slot_i[k] < 0) im.vmiss[(size_t) (layer * NXL + ids6[k])] = lc;
    static const bool check_lru = std::getenv("DS4_CHECK_LRU") != nullptr;
    for (int i = 0; i < n_lru; ++i) {   // vram_lru: the copies are done - the slots now hold the new experts
        gp.cache->replace(layer, lru_old[i], lru_new[i]);
        if (lru_dem[i] >= 0) {   // the arena slot changed hands: the demoted expert in, the promoted one out
            gp.arena.slot_of[(size_t) (layer * NXL + lru_new[i])] = -1;
            gp.arena.slot_of[(size_t) (layer * NXL + lru_old[i])] = lru_dem[i];
            if (!im.last_use.empty()) im.last_use[(size_t) (layer * NXL + lru_old[i])] = im.tick;   // warm, keep it
            ++st.vram_demotes;
        }
        ++st.vram_swaps;
        if (check_lru) {   // DS4_CHECK_LRU=1: the slot and the demoted arena copy against the file, byte for byte
            std::vector<uint8_t> ref((size_t) im.blob);
            std::string e;
            im.blobs->read(layer, lru_new[i], ref.data());
            const bool ok_v = gp.cache->verify_slot(gp.cache->slot_of(layer, lru_new[i]), ref.data(), e, BL);
            bool ok_a = true;
            if (lru_dem[i] >= 0) {
                im.blobs->read(layer, lru_old[i], ref.data());
                ok_a = std::memcmp(gp.arena.ptr(layer, lru_old[i]), ref.data(), (size_t) BL) == 0;
            }
            static int64_t n_ok = 0, n_bad = 0;
            (ok_v && ok_a ? n_ok : n_bad) += 1;
            if (!(ok_v && ok_a) || (n_ok + n_bad) % 500 == 0)
                std::fprintf(stderr, "[lru-chk] L%lld new e%d slot %s, demoted e%d arena %s (%lld ok, %lld bad)\n",
                             (long long) layer, lru_new[i], ok_v ? "ok" : "BAD", lru_old[i],
                             lru_dem[i] < 0 ? "-" : ok_a ? "ok" : "BAD", (long long) n_ok, (long long) n_bad);
        }
    }
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
            ds4_moe_cpu_expert(f, H, im.g.n_ff, p, gp.h_x, ref.data(), *im.pool);
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
            if (wloc[k] == 0.0f) continue;   // a dropped miss: its parts row was never computed (0 * NaN is NaN)
            const float* p = cpu_k[k] ? im.parts.data() + (size_t) k * (size_t) H
                                      : gpu_parts + (size_t) k * (size_t) H;
            s += (double) wloc[k] * (double) p[(size_t) i];
        }
        out[i] = (float) s;
    }

    mark(6);
    // ---- admission: a compulsory miss takes a free slot and is filled for the NEXT token ----
    if (gp.cache && !im.cfg.no_cache && !im.mixed_sizes) {   // mixed: the seed sized every slot to its expert
        for (int i = 0; i < nc; ++i) {
            const int32_t e = ids6[cpu_i[i]];
            if (gp.cache->slot_of(layer, e) >= 0) continue;   // vram_lru already gave it a slot this call
            const uint8_t* p = gp.arena.ptr(layer, e);
            if (!p) continue;
            const int32_t s = gp.cache->admit(layer, e);
            if (s < 0) continue;
            std::string e2;
            if (!gp.cache->fill_slot_blocking(s, p, e2, BL)) {
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
    const int64_t B = im.blob;                        // staging stride
    const int64_t BL = im.bl[(size_t) layer];         // this layer's blob
    const cpu::NativeFmt& f = im.fl[(size_t) layer];
    const strata::kernels::NativeExpertLayout& GL = gp.gll[(size_t) layer];
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
        cpu::native_quant_act(f, gp.h_x + (size_t) t * H, im.nact_n.data() + (size_t) t * im.f.act_bytes);
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
                                   (size_t) BL, cudaMemcpyHostToDevice, gp.s), "pcie dma");
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
        strata::kernels::native_expert_grouped(GL, (const unsigned long long*) gp.d_grp_ptr,
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
        im.pool->run_split_multi_native(f, im.jobs.data(), nj);
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
    if (gp.cache && !im.cfg.no_cache && !im.mixed_sizes) {
        for (int u = 0; u < nu; ++u) {
            if (cls[u] == HIT) continue;
            const uint8_t* p = gp.arena.ptr(layer, ue[u]);
            if (!p) continue;
            const int32_t sl = gp.cache->admit(layer, ue[u]);
            if (sl < 0) continue;
            std::string e2;
            if (!gp.cache->fill_slot_blocking(sl, p, e2, BL)) {
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


#if defined(DS4_MOE_CUDA)
/// Prompt chunk: `n` tokens of one layer, every expert on the GPU.  Distinct experts are groups; a resident one runs
/// from its slot (one launch for all of them), the rest are streamed - arena blobs by DMA straight from the pinned
/// arena, file-tier blobs read (8 threads) into a pinned bounce half and DMA'd from there - into two device halves,
/// batch k+1 copying while batch k computes.  Every entry (token, k) writes its own row of c_parts; the weighted sum
/// over k is done on the host.
// MIMO_CHUNK_PROF=1: run_chunk's time split (host waits on the copy events / final sync, GPU time of the grouped
// launches by event pairs), printed by release_chunk.  Timing only - the same work runs.
struct ChunkProf { double file_gb = 0, rd_busy = 0; double wall = 0, file = 0, file_rd = 0, wait_copied = 0, sync = 0, kern = 0, resident_kern = 0; int64_t calls = 0, launches = 0; };
ChunkProf g_cprof;
bool chunk_prof() { static const bool on = [] { const char* e = std::getenv("MIMO_CHUNK_PROF"); return e && *e == '1'; }(); return on; }

// The stride run_chunk gives a streamed blob in a device half (BL, or BL rounded to whole blocks of every type when
// the layer runs on MMQ) - the same rule as in gpu_run_chunk, for chunk_prestage
int64_t chunk_stride(Ds4MoeImpl& im, int64_t layer) {
    const int64_t BL = im.bl[(size_t) layer];
#if defined(DS4_MOE_MMQ)
    Gpu& gp = *im.gpu;
    namespace mmq = strata::prefill::mmq;
    const strata::kernels::NativeExpertLayout& GL = gp.gll[(size_t) layer];
    const int gt = GL.gu_type, ut = GL.up_type >= 0 ? GL.up_type : GL.gu_type, dt = GL.d_type;
    if (!im.cfg.chunk_mmq || !mmq::built()) return BL;
    if (!(mmq::fits(gt, GL.n_ff) && mmq::fits(ut, GL.n_ff) && mmq::fits(dt, GL.n_embd) && GL.up_off % 2 == 0 &&
          GL.down_off % 2 == 0)) return BL;
    int64_t lcm = 16;
    auto gcd = [](int64_t a, int64_t b) { while (b) { const int64_t t = a % b; a = b; b = t; } return a; };
    for (int t : {gt, ut, dt}) {
        const int64_t tsz = (int64_t) mmq::matrix_bytes(t, 1, 256);
        lcm = lcm / gcd(lcm, tsz) * tsz;
    }
    const int64_t SL = (BL + lcm - 1) / lcm * lcm;
    return SL - BL > 1048576 - 4096 ? BL : SL;
#else
    return BL;
#endif
}

// chunk_prestage: DMA `layer`'s arena experts that are not VRAM-resident, ascending ids, into device half `h` at the
// run_chunk stride; run_chunk(layer) then only waits for c_copied[h].  The half must hold them all (else: skip).
void chunk_prestage(Ds4MoeImpl& im, int64_t layer, int h) {
    Gpu& gp = *im.gpu;
    gp.c_pre_layer = -1;
    if (!im.g.routed(layer) || !gp.s_cp) return;
    const int64_t BL = im.bl[(size_t) layer], SL = chunk_stride(im, layer), NX = im.g.n_experts;
    std::vector<int32_t> es;
    for (int64_t e = 0; e < NX; ++e) {
        if (gp.cache && !im.cfg.no_cache && gp.cache->slot_of(layer, e) >= 0) continue;
        if (gp.arena.ptr(layer, e)) es.push_back((int32_t) e);
    }
    if (es.empty() || (int64_t) es.size() * SL > gp.c_half) return;
    ck(cudaStreamWaitEvent(gp.s_cp, gp.c_used[h], 0), "prestage wait used");
    for (size_t i = 0; i < es.size(); ++i)
        ck(cudaMemcpyAsync((uint8_t*) gp.c_stage[h] + i * (size_t) SL, gp.arena.ptr(layer, es[i]), (size_t) BL,
                           cudaMemcpyHostToDevice, gp.s_cp), "prestage dma");
    ck(cudaEventRecord(gp.c_copied[h], gp.s_cp), "prestage copied");
    gp.c_pre_layer = (int) layer;
    gp.c_pre_half = h;
    gp.c_pre_sl = SL;
    gp.c_pre_e = std::move(es);
}

bool gpu_run_chunk(Ds4MoeImpl& im, int64_t layer, int n, const int32_t* ids, const float* w, const float* x_host,
                   float* out, void* out_dev = nullptr) {
    Gpu& gp = *im.gpu;
    const bool prof = chunk_prof();
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> kev;   // prof: one pair per grouped launch
    bool in_resident = true;
    std::vector<char> kev_res;
    const int64_t H = im.g.n_embd, K = im.g.top_k, NE = (int64_t) n * K;
    const int64_t BL = im.bl[(size_t) layer];
    const strata::kernels::NativeExpertLayout& GL = gp.gll[(size_t) layer];
    const double t0 = now_ms();
    Ds4MoeStats st;

    if (n > gp.c_tok) {   // (re)allocate for n tokens
        for (void* p : {gp.c_x, gp.c_xq, gp.c_parts, gp.c_scratch, gp.c_dst, gp.c_tokv, gp.c_w, gp.c_out}) if (p) cudaFree(p);
        if (gp.c_hparts) cudaFreeHost(gp.c_hparts);
        gp.c_tok = n;
        const int64_t ne = (int64_t) n * K;
        ck(cudaMalloc(&gp.c_x, (size_t) n * (size_t) H * 4), "c_x");
        ck(cudaMalloc(&gp.c_xq, (size_t) n * strata::kernels::native_q8_1_bytes((int) H, 1)), "c_xq");   // its ncols <= 8
        ck(cudaMalloc(&gp.c_parts, (size_t) ne * (size_t) H * 4), "c_parts");
        ck(cudaMalloc(&gp.c_scratch, strata::kernels::native_expert_scratch_bytes(ne, im.g.n_ff)), "c_scratch");
        ck(cudaMalloc(&gp.c_dst, sizeof(int32_t) * (size_t) ne), "c_dst");
        ck(cudaMalloc(&gp.c_tokv, sizeof(int32_t) * (size_t) ne), "c_tok");
        ck(cudaMalloc(&gp.c_w, sizeof(float) * (size_t) ne), "c_w");
        ck(cudaMalloc(&gp.c_out, (size_t) n * (size_t) H * 4), "c_out");
        ck(cudaMallocHost((void**) &gp.c_hparts, (size_t) n * (size_t) H * 4), "c_hparts");   // the summed rows
    }
    if (!gp.s_cp) {
        const char* mib = std::getenv("MIMO_CHUNK_STAGE_MIB");
        gp.c_half = (int64_t) (mib ? std::atoll(mib) : 1024) * 1048576;
        gp.c_half = std::max<int64_t>(gp.c_half, im.blob);
        gp.c_bhalf = gp.c_half;   // the pinned bounce halves keep this size
        // prestage: one device half holds a whole layer (every expert at the MMQ stride, <= blob + 1 MiB each)
        if (im.cfg.chunk_prestage)
            gp.c_half = std::max<int64_t>(gp.c_half, im.g.n_experts * (im.blob + 1048576));
        const int64_t ng = im.g.n_experts;
        ck(cudaMalloc(&gp.c_ptr, sizeof(unsigned long long) * (size_t) ng), "c_ptr");
        ck(cudaMalloc(&gp.c_start, sizeof(int32_t) * (size_t) (ng + 1)), "c_start");
        ck(cudaMalloc(&gp.c_ng, sizeof(int32_t)), "c_ng");
        for (int h = 0; h < 2; ++h) {
            // + 1 MiB: the MMQ slot stride rounds a blob up (<= ~0.3 MiB) and MMQ reads up to 4 KiB past a matrix;
            // zeroed once, so whatever MMQ reads past a blob decodes to finite numbers (llama.cpp zero-pads too)
            ck(cudaMalloc(&gp.c_stage[h], (size_t) gp.c_half + 1048576), "c_stage");
            ck(cudaMemset(gp.c_stage[h], 0, (size_t) gp.c_half + 1048576), "c_stage zero");
            ck(cudaMallocHost((void**) &gp.c_bounce[h], (size_t) gp.c_bhalf), "c_bounce");
            ck(cudaEventCreateWithFlags(&gp.c_copied[h], cudaEventDisableTiming), "ev");
            ck(cudaEventCreateWithFlags(&gp.c_used[h], cudaEventDisableTiming), "ev");
            ck(cudaEventRecord(gp.c_used[h], gp.s), "ev");
        }
        ck(cudaStreamCreateWithFlags(&gp.s_cp, cudaStreamNonBlocking), "s_cp");
    }
    im.pf_n = 0;   // a chunk does not use the decode prefetch staging

    // ---- MMQ (Ds4MoeConfig::chunk_mmq): this layer's types covered -> streamed blobs sit `SL` apart in a half (BL
    // rounded up to whole blocks of every type, so each matrix is one MMQ "expert stride" away from the next)
    bool use_mmq = false;
    int64_t SL = BL;
#if defined(DS4_MOE_MMQ)
    namespace mmq = strata::prefill::mmq;
    const int gt = GL.gu_type, ut = GL.up_type >= 0 ? GL.up_type : GL.gu_type, dt = GL.d_type;
    if (im.cfg.chunk_mmq && mmq::built()) {
        if (gp.m_ok.size() != gp.gll.size()) gp.m_ok.assign(gp.gll.size(), -1);
        int8_t& ok = gp.m_ok[(size_t) layer];
        if (ok < 0) {
            ok = mmq::fits(gt, GL.n_ff) && mmq::fits(ut, GL.n_ff) && mmq::fits(dt, GL.n_embd) &&
                 GL.up_off % 2 == 0 && GL.down_off % 2 == 0;   // a finite swiglu_limit: mmq::swiglu_clamp
            if (!ok) std::fprintf(stderr, "ds4_moe: run_chunk: layer %lld stays on MMVQ (types %d/%d/%d)\n",
                                  (long long) layer, gt, ut, dt);
        }
        use_mmq = ok > 0;
    }
    if (use_mmq) {
        // 256 values = one block of the K-quants, 8 of MXFP4 (a multiple of its 17-byte block either way)
        int64_t lcm = 16;
        auto gcd = [](int64_t a, int64_t b) { while (b) { const int64_t t = a % b; a = b; b = t; } return a; };
        for (int t : {gt, ut, dt}) {
            const int64_t tsz = (int64_t) mmq::matrix_bytes(t, 1, 256);
            lcm = lcm / gcd(lcm, tsz) * tsz;
        }
        SL = (BL + lcm - 1) / lcm * lcm;
        if (SL - BL > 1048576 - 4096) use_mmq = false, SL = BL;
        const int64_t cap = std::max<int64_t>(16384, n);
        if (use_mmq && gp.m_rows < cap) {
            gp.free_mmq();
            gp.m_rows = cap;
            const int64_t FF = GL.n_ff;
            ck(cudaMalloc(&gp.m_gu, (size_t) cap * (size_t) (2 * FF) * 4), "m_gu");
            ck(cudaMalloc(&gp.m_h, (size_t) cap * (size_t) FF * 4), "m_h");
            ck(cudaMalloc(&gp.m_xg, mmq::q8_bytes(cap, H)), "m_xg");
            ck(cudaMalloc(&gp.m_xu, mmq::q8_bytes(cap, H)), "m_xu");
            ck(cudaMalloc(&gp.m_xd, mmq::q8_bytes(cap, FF)), "m_xd");
            ck(cudaMalloc(&gp.m_bnd, sizeof(int32_t) * (size_t) (im.g.n_experts + 1)), "m_bnd");
            ck(cudaMalloc(&gp.m_iota, sizeof(int32_t) * (size_t) cap), "m_iota");
            mmq::iota((int32_t*) gp.m_iota, cap, gp.s);
            if (!gp.m_ctx) gp.m_ctx = new mmq::Context();   // never deleted (see Gpu::m_ctx)
        }
    }
#endif

    // ---- groups: distinct experts with their entries (token-major entry j = t*K + k writes parts row j) ----
    const int64_t NX = im.g.n_experts;
    std::vector<int32_t> cnt((size_t) NX, 0);
    for (int64_t j = 0; j < NE; ++j) {
        if (ids[j] < 0 || ids[j] >= NX) return false;
        ++cnt[(size_t) ids[j]];
    }
    // processing order: VRAM-resident experts, then arena (DMA straight away), then file tier (read in the background
    // from the start of the call, so the arena DMAs and the reads overlap); entries are laid out in this order, so
    // consecutive experts of one class are adjacent in the sorted entry arrays (one launch per run)
    std::vector<int32_t> order[3];
    std::vector<int32_t> slot_e((size_t) NX, -1);
    for (int64_t e = 0; e < NX; ++e) {
        if (!cnt[(size_t) e]) continue;
        const int32_t sl = (gp.cache && !im.cfg.no_cache) ? gp.cache->slot_of(layer, e) : -1;
        slot_e[(size_t) e] = sl;
        order[sl >= 0 ? 0 : gp.arena.ptr(layer, e) ? 1 : 2].push_back((int32_t) e);
    }
    std::vector<int32_t> first((size_t) NX + 1, 0);
    {
        int32_t at = 0;
        for (const auto& o : order)
            for (int32_t e : o) {
                first[(size_t) e] = at;
                at += cnt[(size_t) e];
            }
    }
    std::vector<int32_t> dst((size_t) NE), tok((size_t) NE), fill(first.begin(), first.end() - 1);
    for (int64_t j = 0; j < NE; ++j) {
        const int32_t p = fill[(size_t) ids[j]]++;
        dst[(size_t) p] = (int32_t) j;
        tok[(size_t) p] = (int32_t) (j / K);
    }
    // entry arrays sorted by expert: uploaded once; each launch passes its groups' slices via grp_start offsets
    ck(cudaMemcpyAsync(gp.c_dst, dst.data(), sizeof(int32_t) * (size_t) NE, cudaMemcpyHostToDevice, gp.s), "c_dst");
    ck(cudaMemcpyAsync(gp.c_tokv, tok.data(), sizeof(int32_t) * (size_t) NE, cudaMemcpyHostToDevice, gp.s), "c_tok");

    // ---- activations ----
    ck(cudaMemcpyAsync(gp.c_x, x_host, (size_t) n * (size_t) H * 4, cudaMemcpyHostToDevice, gp.s), "c_x h2d");
    {   // the MMVQ quantizer takes <= 8 columns per call; columns are contiguous per token, so step through them
        const size_t col = strata::kernels::native_q8_1_bytes((int) H, 1);
        if (strata::kernels::native_q8_1_bytes((int) H, 8) != 8 * col) {
            std::fprintf(stderr, "ds4_moe: run_chunk: q8_1 layout is not per-column contiguous\n");
            return false;
        }
        for (int c0 = 0; c0 < n; c0 += 8)
            strata::kernels::native_quantize_q8_1((const float*) gp.c_x + (size_t) c0 * (size_t) H,
                                                  (uint8_t*) gp.c_xq + (size_t) c0 * col, (int) H, std::min(8, n - c0), gp.s);
    }

    // a launch over groups `es` (expert ids) whose blobs are at `ptrs`: grp_start are offsets into the sorted entries
    std::vector<int32_t> lstart;
#if defined(DS4_MOE_MMQ)
    // MMQ over experts es[q0, q1) at ptrs[q0] + (q - q0) * SL: three products (gate, up -> SwiGLU -> down) over their
    // sorted entries [r0, r1); the down product writes each entry's parts row through c_dst
    std::vector<int32_t> mbnd;
    auto mmq_launch = [&](const std::vector<int32_t>& es, const std::vector<unsigned long long>& ptrs, size_t q0,
                          size_t q1) {
        const int64_t r0 = first[(size_t) es[q0]], r1 = first[(size_t) es[q1 - 1]] + cnt[(size_t) es[q1 - 1]];
        const int64_t R = r1 - r0, FF = GL.n_ff;
        int64_t maxr = 0;
        mbnd.clear();
        for (size_t q = q0; q < q1; ++q) {
            mbnd.push_back(first[(size_t) es[q]] - (int32_t) r0);
            maxr = std::max<int64_t>(maxr, cnt[(size_t) es[q]]);
        }
        mbnd.push_back((int32_t) R);
        ck(cudaMemcpyAsync(gp.m_bnd, mbnd.data(), sizeof(int32_t) * mbnd.size(), cudaMemcpyHostToDevice, gp.s), "m_bnd");
        const int32_t* tokr = (const int32_t*) gp.c_tokv + r0;
        mmq::quantize((const float*) gp.c_x, tokr, gp.m_xg, gt, H, H, R, gp.s);
        if (ut != gt) mmq::quantize((const float*) gp.c_x, tokr, gp.m_xu, ut, H, H, R, gp.s);
        const uint8_t* base = (const uint8_t*) ptrs[q0];
        mmq::Product p;
        p.n = (int) (q1 - q0);
        p.expert_bytes = (size_t) SL;
        p.bounds = (const int32_t*) gp.m_bnd;
        p.total_rows = R;
        p.max_rows = maxr;
        p.w = base; p.type = gt; p.w_rows = FF; p.w_cols = H; p.xq = gp.m_xg;
        p.ids = (const int32_t*) gp.m_iota; p.dst = (float*) gp.m_gu; p.ld_dst = 2 * FF;
        gp.m_ctx->run(p, gp.s);
        p.w = base + GL.up_off; p.type = ut; p.xq = ut != gt ? gp.m_xu : gp.m_xg; p.dst = (float*) gp.m_gu + FF;
        gp.m_ctx->run(p, gp.s);
        if (std::isfinite(GL.swiglu_limit))   // DeepSeek-V4 (every layer clamps at 10)
            mmq::swiglu_clamp((const float*) gp.m_gu, (float*) gp.m_h, R, FF, false, GL.swiglu_limit, gp.s);
        else
            mmq::swiglu((const float*) gp.m_gu, (float*) gp.m_h, R, FF, false, gp.s);
        mmq::quantize((const float*) gp.m_h, nullptr, gp.m_xd, dt, FF, FF, R, gp.s);
        p.w = base + GL.down_off; p.type = dt; p.w_rows = H; p.w_cols = FF; p.xq = gp.m_xd;
        p.ids = (const int32_t*) gp.c_dst + r0; p.dst = (float*) gp.c_parts; p.ld_dst = H;
        gp.m_ctx->run(p, gp.s);
    };
#endif
    auto launch = [&](const std::vector<int32_t>& es, const std::vector<unsigned long long>& ptrs) {
        if (es.empty()) return;
#if defined(DS4_MOE_MMQ)
        bool strided = use_mmq;
        for (size_t q = 1; strided && q < ptrs.size(); ++q) strided = ptrs[q] == ptrs[0] + q * (unsigned long long) SL;
        if (strided) {   // sub-runs of at most m_rows entries (one expert has <= n <= m_rows)
            cudaEvent_t e0 = nullptr, e1 = nullptr;
            if (prof) {
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, gp.s);
            }
            size_t q0 = 0;
            int64_t rows = 0;
            for (size_t q = 0; q < es.size(); ++q) {
                if (rows + cnt[(size_t) es[q]] > gp.m_rows) {
                    mmq_launch(es, ptrs, q0, q);
                    q0 = q;
                    rows = 0;
                }
                rows += cnt[(size_t) es[q]];
            }
            mmq_launch(es, ptrs, q0, es.size());
            if (prof) {
                cudaEventRecord(e1, gp.s);
                kev.emplace_back(e0, e1);
                kev_res.push_back(in_resident ? 1 : 0);
            }
            return;
        }
#endif
        lstart.clear();
        // cap_entries = NE, not this launch's count: the kernel's scratch regions are laid out by it and entries are
        // indexed absolutely (grp_start); its SwiGLU pass visits only [grp_start[0], grp_start[ng]).
        for (int32_t e : es) lstart.push_back(first[(size_t) e]);
        // the kernel walks entries [grp_start[g], grp_start[g+1]); groups here are not adjacent in the sorted
        // arrays, so each launch covers ONE contiguous run: callers pass experts in ascending id order and runs are
        // split where they are not adjacent (see below)
        lstart.push_back(first[(size_t) es.back()] + cnt[(size_t) es.back()]);
        const int32_t ng = (int32_t) es.size();
        ck(cudaMemcpyAsync(gp.c_ptr, ptrs.data(), sizeof(unsigned long long) * es.size(), cudaMemcpyHostToDevice, gp.s), "c_ptr");
        ck(cudaMemcpyAsync(gp.c_start, lstart.data(), sizeof(int32_t) * lstart.size(), cudaMemcpyHostToDevice, gp.s), "c_start");
        ck(cudaMemcpyAsync(gp.c_ng, &ng, sizeof(int32_t), cudaMemcpyHostToDevice, gp.s), "c_ng");
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        if (prof) {
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, gp.s);
        }
        strata::kernels::native_expert_grouped(GL, (const unsigned long long*) gp.c_ptr, (const int32_t*) gp.c_start,
                                              (const int32_t*) gp.c_ng, (const int32_t*) gp.c_dst,
                                              (const int32_t*) gp.c_tokv, ng, NE, gp.c_xq, gp.c_scratch,
                                              (float*) gp.c_parts, gp.s, ng);
        if (prof) {
            cudaEventRecord(e1, gp.s);
            kev.emplace_back(e0, e1);
            kev_res.push_back(in_resident ? 1 : 0);
        }
        // the host vectors above are pageable: cudaMemcpyAsync has staged them before returning
    };
    // ---- resident experts: adjacent in the entry order -> one launch
    std::vector<int32_t> run_e;
    std::vector<unsigned long long> run_p;
    int64_t ent_hit = 0, ent_str = 0;
    for (int32_t e : order[0]) {
        run_e.push_back(e);
        run_p.push_back((unsigned long long) gp.cache->device_slot(slot_e[(size_t) e]));
        ent_hit += cnt[(size_t) e];
    }
    launch(run_e, run_p);
    in_resident = false;

    // ---- streamed experts: arena ones, then the file tier, in batches that fit one device half.  File-tier blobs
    // are read by a background pool into pinned slots (both bounce halves, `cap` blobs; more = waves) from here on;
    // a batch waits only for the blobs it DMAs.
    // chunk_prestage: this layer's arena experts may already sit in a device half (issued by the previous call)
    std::vector<int32_t> arena_e(order[1]);
    int h0 = 0;   // the first half the batch loop below uses
    bool used_pre = false;
    if (gp.c_pre_layer == (int) layer && gp.c_pre_sl == SL) {
        std::vector<int32_t> pidx((size_t) NX, -1);
        for (size_t i = 0; i < gp.c_pre_e.size(); ++i) pidx[(size_t) gp.c_pre_e[i]] = (int32_t) i;
        std::vector<int32_t> es, rest;
        std::vector<unsigned long long> ps;
        for (int32_t e : order[1]) {
            if (pidx[(size_t) e] < 0) { rest.push_back(e); continue; }
            es.push_back(e);
            ps.push_back((unsigned long long) ((uint8_t*) gp.c_stage[gp.c_pre_half] + (size_t) pidx[(size_t) e] * (size_t) SL));
            ent_str += cnt[(size_t) e];
        }
        if (!es.empty()) {
            ck(cudaStreamWaitEvent(gp.s, gp.c_copied[gp.c_pre_half], 0), "wait prestaged");
            // runs adjacent in the entry order AND in the half (strided: one MMQ launch per run)
            std::vector<int32_t> re;
            std::vector<unsigned long long> rp;
            for (size_t q = 0; q < es.size(); ++q) {
                if (!re.empty() && (first[(size_t) re.back()] + cnt[(size_t) re.back()] != first[(size_t) es[q]] ||
                                    rp.back() + (unsigned long long) SL != ps[q])) {
                    launch(re, rp);
                    re.clear();
                    rp.clear();
                }
                re.push_back(es[q]);
                rp.push_back(ps[q]);
            }
            launch(re, rp);
            ck(cudaEventRecord(gp.c_used[gp.c_pre_half], gp.s), "used prestaged");
            used_pre = true;
        }
        arena_e = std::move(rest);
        h0 = 1 - gp.c_pre_half;
    }
    gp.c_pre_layer = -1;
    std::vector<int32_t> stream_e(arena_e);
    stream_e.insert(stream_e.end(), order[2].begin(), order[2].end());
    for (int32_t e : stream_e) ent_str += cnt[(size_t) e];
    const size_t nA = arena_e.size(), nF = order[2].size();
    const int64_t per_half = std::max<int64_t>(1, gp.c_half / SL);
    const size_t slots_half = (size_t) std::max<int64_t>(1, gp.c_bhalf / BL), cap = 2 * slots_half;
    auto slot_ptr = [&](size_t k) { return gp.c_bounce[(k % cap) / slots_half] + ((k % cap) % slots_half) * (size_t) BL; };
    std::vector<std::atomic<int>> ready(nF);
    for (auto& r : ready) r.store(0, std::memory_order_relaxed);
    std::atomic<size_t> rnext{0}, rdone{0};
    std::atomic<double> t_rd_last{0};
    size_t wave_end = 0;   // reads issued for file blobs [0, wave_end)
    std::vector<std::thread> readers;
    double t_file = 0, t_rd0 = 0;
    auto start_wave = [&]() {   // the next `cap` file blobs (their slots must be free: the caller made sure)
        for (auto& t : readers) t.join();
        readers.clear();
        const size_t w0 = wave_end, w1 = std::min(nF, wave_end + cap);
        wave_end = w1;
        rnext.store(w0);
        const int nth = (int) std::min<size_t>(8, w1 - w0);
        for (int k = 0; k < nth; ++k)
            readers.emplace_back([&, w1]() {
                for (size_t q; (q = rnext.fetch_add(1)) < w1;) {
                    im.blobs->read(layer, order[2][q], slot_ptr(q));
                    ready[q].store(1, std::memory_order_release);
                    ready[q].notify_one();
                    if (prof && rdone.fetch_add(1) + 1 == nF) t_rd_last.store(now_ms());
                }
            });
    };
    if (nF) {
        t_rd0 = now_ms();
        start_wave();
    }
    int h_last = used_pre ? 1 - h0 : -1;
    for (size_t b0 = 0, bi = 0; b0 < stream_e.size(); b0 += (size_t) per_half, ++bi) {
        const size_t b1 = std::min(stream_e.size(), b0 + (size_t) per_half);
        const int h = (int) ((h0 + bi) & 1);
        h_last = h;
        // the half is free once the kernel that last read it is done
        ck(cudaStreamWaitEvent(gp.s_cp, gp.c_used[h], 0), "wait used");
        std::vector<int32_t> es;
        std::vector<unsigned long long> ps;
        for (size_t i = b0; i < b1; ++i) {
            const uint8_t* src;
            if (i < nA) {
                src = gp.arena.ptr(layer, stream_e[i]);   // (arena_e: not prestaged)
            } else {
                const size_t k = i - nA;
                if (k >= wave_end) {   // a new wave reuses the slots: every DMA out of them must be done first
                    const double tw = now_ms();
                    ck(cudaStreamSynchronize(gp.s_cp), "wave");
                    if (prof) g_cprof.wait_copied += now_ms() - tw;
                    start_wave();
                }
                if (!ready[k].load(std::memory_order_acquire)) {
                    const double tw = now_ms();
                    ready[k].wait(0, std::memory_order_acquire);
                    t_file += now_ms() - tw;
                }
                src = slot_ptr(k);
            }
            uint8_t* d = (uint8_t*) gp.c_stage[h] + (i - b0) * (size_t) SL;
            ck(cudaMemcpyAsync(d, src, (size_t) BL, cudaMemcpyHostToDevice, gp.s_cp), "stream dma");
            es.push_back(stream_e[i]);
            ps.push_back((unsigned long long) d);
        }
        ck(cudaEventRecord(gp.c_copied[h], gp.s_cp), "copied");
        ck(cudaStreamWaitEvent(gp.s, gp.c_copied[h], 0), "wait copied");
        // one launch per adjacent run inside the batch (the arena / file boundary can split one)
        std::vector<int32_t> re;
        std::vector<unsigned long long> rp;
        for (size_t q = 0; q < es.size(); ++q) {
            if (!re.empty() && first[(size_t) re.back()] + cnt[(size_t) re.back()] != first[(size_t) es[q]]) {
                launch(re, rp);
                re.clear();
                rp.clear();
            }
            re.push_back(es[q]);
            rp.push_back(ps[q]);
        }
        launch(re, rp);
        ck(cudaEventRecord(gp.c_used[h], gp.s), "used");
    }
    for (auto& t : readers) t.join();
    const int64_t n_file = (int64_t) nF;
    if (prof && nF) {
        g_cprof.file_rd += now_ms() - t_rd0;
        g_cprof.rd_busy += t_rd_last.load() - t_rd0;
        g_cprof.file_gb += (double) nF * (double) BL / 1e9;
    }

    // ---- chunk_prestage: the next routed layer's arena experts stream in while this layer computes and the caller
    // runs the next dense half (the half this call did not use last; its kernels finish before the DMA starts) ----
    if (im.cfg.chunk_prestage && !im.cfg.arena_adapt && gp.arena.base) {
        int64_t nl = layer;
        for (int64_t k = 0; k < im.g.n_layers; ++k) {
            nl = (nl + 1) % im.g.n_layers;
            if (im.g.routed(nl)) break;
        }
        chunk_prestage(im, nl, h_last >= 0 ? 1 - h_last : 0);
    }

    // ---- outputs: the top-k mix on the card, n rows back ----
    ck(cudaMemcpyAsync(gp.c_w, w, sizeof(float) * (size_t) NE, cudaMemcpyHostToDevice, gp.s), "c_w");
    strata::kernels::weighted_rows_sum((const float*) gp.c_parts, (const float*) gp.c_w, (int) K, H, n,
                                       (float*) gp.c_out, gp.s);
    if (out_dev)   // straight into the caller's device tensor (no host round trip)
        ck(cudaMemcpyAsync(out_dev, gp.c_out, (size_t) n * (size_t) H * 4, cudaMemcpyDeviceToDevice, gp.s), "out d2d");
    else
        ck(cudaMemcpyAsync(gp.c_hparts, gp.c_out, (size_t) n * (size_t) H * 4, cudaMemcpyDeviceToHost, gp.s), "out d2h");
    const double ts = now_ms();
    ck(cudaStreamSynchronize(gp.s), "chunk sync");
    if (!out_dev) std::memcpy(out, gp.c_hparts, (size_t) n * (size_t) H * 4);
    if (prof) {
        g_cprof.sync += now_ms() - ts;
        for (size_t i = 0; i < kev.size(); ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, kev[i].first, kev[i].second);
            g_cprof.kern += ms;
            if (kev_res[i]) g_cprof.resident_kern += ms;
            cudaEventDestroy(kev[i].first);
            cudaEventDestroy(kev[i].second);
        }
        g_cprof.launches += (int64_t) kev.size();
        g_cprof.calls += 1;
        g_cprof.file += t_file;
        g_cprof.wall += now_ms() - t0;
    }
    st.hits = ent_hit;
    st.pcie = ent_str;
    st.file_tier = n_file;
    st.file_ms = t_file;
    st.wall_ms = now_ms() - t0;
    im.st.add(st);
    return true;
}
#endif

bool Ds4MoeTier::run(int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out) {
    if (!im_->inited || !im_->pool) return false;
    if (!ids6 || !w6 || !x || !out) return false;
    if (!im_->g.routed(layer)) return false;
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
    if (!im_->g.routed(layer)) return false;
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

bool Ds4MoeTier::build_arena_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err) {
    if (im_->cfg.cpu_only) return true;
    std::string e;
    const std::vector<std::pair<int32_t, int32_t>> ranked =
        rank_routes(routes_bin, im_->g.n_routed(), im_->g.top_k, e, batch_tokens);
    if (ranked.empty()) {
        err = e.empty() ? "empty routing profile" : e;
        return false;
    }
    return build_arena(arena_order(*im_, ranked), err);
}

void Ds4MoeTier::release_chunk() {
#if defined(DS4_MOE_CUDA)
    if (!im_->gpu) return;
    if (chunk_prof() && g_cprof.calls) {
        const ChunkProf& c = g_cprof;
        std::fprintf(stderr, "chunk prof: %lld calls, %lld launches: wall %.0f ms = waits on reads %.0f + wave/copy waits "
                             "%.0f + final sync %.0f + other host %.0f; reader wall %.0f ms; GPU grouped kernels %.0f ms "
                             "(resident %.0f); file tier %.1f GB read in %.0f ms of reader time = %.2f GB/s\n",
                     (long long) c.calls, (long long) c.launches, c.wall, c.file, c.wait_copied, c.sync,
                     c.wall - c.file - c.wait_copied - c.sync, c.file_rd, c.kern, c.resident_kern, c.file_gb, c.rd_busy,
                     c.rd_busy > 0 ? c.file_gb / (c.rd_busy / 1000.0) : 0.0);
        g_cprof = ChunkProf{};
    }
    Gpu& gp = *im_->gpu;
    if (gp.s) cudaStreamSynchronize(gp.s);
    if (gp.s_cp) cudaStreamSynchronize(gp.s_cp);
    for (void* p : {gp.c_x, gp.c_xq, gp.c_parts, gp.c_scratch, gp.c_ptr, gp.c_start, gp.c_ng, gp.c_dst, gp.c_tokv,
                    gp.c_stage[0], gp.c_stage[1], gp.c_w, gp.c_out})
        if (p) cudaFree(p);
    for (uint8_t* p : gp.c_bounce)
        if (p) cudaFreeHost(p);
    if (gp.c_hparts) cudaFreeHost(gp.c_hparts);
    for (int i = 0; i < 2; ++i) {
        if (gp.c_copied[i]) cudaEventDestroy(gp.c_copied[i]);
        if (gp.c_used[i]) cudaEventDestroy(gp.c_used[i]);
    }
    if (gp.s_cp) cudaStreamDestroy(gp.s_cp);
    gp.c_x = gp.c_xq = gp.c_parts = gp.c_scratch = gp.c_ptr = gp.c_start = gp.c_ng = gp.c_dst = gp.c_tokv = nullptr;
    gp.c_w = gp.c_out = nullptr;
    gp.c_stage[0] = gp.c_stage[1] = nullptr;
    gp.c_bounce[0] = gp.c_bounce[1] = nullptr;
    gp.c_hparts = nullptr;
    for (int i = 0; i < 2; ++i) gp.c_copied[i] = gp.c_used[i] = nullptr;
    gp.s_cp = nullptr;
    gp.c_tok = gp.c_half = gp.c_bhalf = 0;
    gp.c_pre_layer = gp.c_pre_half = -1;
    gp.c_pre_e.clear();
    gp.free_mmq();
#endif
}

bool Ds4MoeTier::run_chunk(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, float* out) {
    if (!im_->inited || !ids || !w || !x || !out || n < 1) return false;
    if (!im_->g.routed(layer)) return false;
#if defined(DS4_MOE_CUDA)
    if (im_->gpu && !im_->cfg.cpu_only) return gpu_run_chunk(*im_, layer, n, ids, w, x, out);
#endif
    // CPU tier: token by token
    const int64_t K = im_->g.top_k, H = im_->g.n_embd;
    for (int t = 0; t < n; ++t)
        if (!run(layer, ids + t * K, w + t * K, x + (size_t) t * H, out + (size_t) t * H)) return false;
    return true;
}

bool Ds4MoeTier::run_chunk_dev(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, void* out_dev) {
    if (!im_->inited || !ids || !w || !x || !out_dev || n < 1) return false;
    if (!im_->g.routed(layer)) return false;
#if defined(DS4_MOE_CUDA)
    if (im_->gpu && !im_->cfg.cpu_only) return gpu_run_chunk(*im_, layer, n, ids, w, x, nullptr, out_dev);
#endif
    return false;
}

bool Ds4MoeTier::run_dev(int64_t layer, const int32_t* ids6, const float* w6, const void* x_dev, float* out) {
    if (!im_->inited) return false;
    if (!ids6 || !w6 || !x_dev || !out) return false;
    if (!im_->g.routed(layer)) return false;
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

bool Ds4MoeTier::in_arena(int64_t layer, int64_t expert) const {
#if defined(DS4_MOE_CUDA)
    if (im_ && im_->gpu) return im_->gpu->arena.ptr(layer, expert) != nullptr;
#endif
    (void) layer;
    (void) expert;
    return false;
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
    file_ms += o.file_ms;
    arena_swaps += o.arena_swaps;
    vram_swaps += o.vram_swaps;
    vram_demotes += o.vram_demotes;
    skipped += o.skipped;
    gap_ms += o.gap_ms;
    hit_ms += o.hit_ms;
    pcie_ms += o.pcie_ms;
    cpu_ms += o.cpu_ms;
    wall_ms += o.wall_ms;
}

}  // namespace strata::ds4
