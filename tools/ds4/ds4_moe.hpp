// tools/ds4/ds4_moe.hpp - the DeepSeek-V4 routed-expert TIER as a reusable library (slice ds4-moe).
//
// WHAT THIS IS.  `tools/ds4/moe_replay.cpp` proved the expert half of a DeepSeek-V4 decode on this machine
// (FINDINGS s10-s12): a VRAM cache of 6.75 MiB expert slots seeded from a routing profile, the CPU ExpertPool on
// the misses, a PCIe share of the misses computed on the GPU in parallel, a pinned host arena holding every
// expert, and predicted expert prefetch on a second stream under the dense work - measured 20.57 tok/s at 2150
// slots, pf-b 1.43, pcie 0.25 dithered.  That file is a HARNESS: it drives the pieces itself, per token, with a
// synthetic stand-in for the dense half and recorded routes instead of a router.
//
// `Ds4MoeTier` is the same machinery as one object, used once per layer by the real engine:
//
//     tier.init(gguf, cfg, err);                     // once, at startup
//     tier.seed_from_routes(routes_bin, err);        // the static profile, once
//     for each layer l of each token:
//         tier.prefetch(l, top16);                   // right after predict(l), before attention(l)
//         tier.run(l, ids6, w6, x, out);             // after router(l); out = sum_k w_k * expert_k(x)
//
// `run` is synchronous: the engine calls it per layer and needs `out` before the next one.  `prefetch` is not:
// it issues its DMAs on a second stream and returns immediately, so they overlap the dense work the caller is
// about to do between `prefetch` and `run`.
//
// TWO MODES.  `cfg.cpu_only` runs the whole expert half on the CPU pool: no ExpertCache, no PCIe share, no CUDA
// call of any kind, so the tier can be linked into a binary that must never touch the driver (this slice's gate,
// and machines without a GPU).  Otherwise the three tiers of the plan run concurrently and the weighted sum is
// assembled from all of them.
//
// THE INVARIANT, and the one thing to read before changing anything here:
//
//     for any (layer, ids, weights, x), run()'s output equals the weighted sum of the dequantised experts
//     computed in F32 - the only permitted difference being the expert kernels' own 8-bit activation rounding.
//
// It is tested by `tools/ds4/test_ds4_moe.cpp` against a dequant+F32 reference (weights through ggml's
// `to_float`), CPU-only, at rel < 1e-3 per layer and per expert.
//
// THE SWIGLU CLAMP.  DeepSeek-V4's graph clamps the expert pre-activations before the SwiGLU - gate -> min(gate, L),
// up -> clamp(up, -L, L), L = `swiglu_clamp_exp` = 10.0 (`tools/ds4/ds4_ref.cpp`, mirroring llama.cpp's DEEPSEEK4
// branch).  The native expert kernels apply it through `NativeFmt::swiglu_limit` (CPU: `native_gu_rows` and the
// AVX2/AVX-512 multi-token kernels) and `NativeExpertLayout::swiglu_limit` (CUDA: the grouped kernels' SwiGLU
// passes); `ds4_moe_geom_from_gguf` reads L into `Ds4MoeGeom::swiglu_limit` and the tier passes it to both.  Every
// other model leaves the limit at +inf, which is the unclamped arithmetic bit for bit.  On real activations L
// binds rarely (FINDINGS s13), so `test_ds4_moe` also reruns its real arms with L forced to 1.0 to check the path.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds4 {

namespace cpu = strata::kernels::cpu;

// ================================ geometry ================================

/// The routed-expert half of one DeepSeek-V4-Flash layer.  The defaults are the real model's
/// (docs/ds4/GGUF_INVENTORY.txt): 43 layers x 256 experts, top-6, gate/up IQ2_XXS, down Q2_K, n_embd 4096,
/// n_ff 2048.  `ds4_moe_geom_from_gguf` reads them from metadata instead of trusting the defaults.
struct Ds4MoeGeom {
    int64_t n_embd = 4096;      ///< n_embd
    int64_t n_ff = 2048;        ///< n_ff of one routed expert
    int64_t n_layers = 43;      ///< layers that route
    int64_t n_experts = 256;    ///< experts per layer
    int64_t top_k = 6;          ///< experts routed per token
    int gu_type = 16;           ///< GGML_TYPE_IQ2_XXS (gate and up)
    int d_type = 10;            ///< GGML_TYPE_Q2_K (down)
    /// `deepseek4.swiglu_clamp_exp` (10.0 on every layer of the 0731 GGUF; one value - the tier has one format):
    /// gate -> min(gate, lim), up -> clamp(up, -lim, lim) before the SwiGLU.  +inf = no clamp.
    float swiglu_limit = std::numeric_limits<float>::infinity();
    /// PER-LAYER FORMATS (MiMo-V2.6-Flash GSQ-RCO picks a format per tensor: gate, up and down independently).
    /// Empty = every layer is `gu_type`/`gu_type`/`d_type` (DeepSeek-V4: the uniform path, unchanged).  Otherwise
    /// `n_layers` entries each, indexed by the MODEL's layer number; -1 = that layer has no routed experts (MiMo's
    /// dense layer 0).  `gu_type`/`d_type` then hold the first routed layer's types (reports only).
    std::vector<int> gate_types, up_types, down_types;

    bool per_layer() const { return !gate_types.empty(); }
    bool routed(int64_t l) const { return l >= 0 && l < n_layers && (!per_layer() || gate_types[(size_t) l] >= 0); }
    int gate_type(int64_t l) const { return per_layer() ? gate_types[(size_t) l] : gu_type; }
    int up_type(int64_t l) const { return per_layer() ? up_types[(size_t) l] : gu_type; }
    int down_type(int64_t l) const { return per_layer() ? down_types[(size_t) l] : d_type; }
    /// Layers that route (n_layers for a uniform geometry).
    int64_t n_routed() const {
        if (!per_layer()) return n_layers;
        int64_t n = 0;
        for (int64_t l = 0; l < n_layers; ++l) n += routed(l);
        return n;
    }
};

/// Reads `n_embd / n_ff / n_layers / n_experts / top_k` from the GGUF's `<arch>.*` metadata (`general.architecture`:
/// deepseek4, mimo2, ...) and the expert quant types of every layer's `blk.L.ffn_{gate,up,down}_exps.weight`.  All
/// layers present and alike -> the uniform geometry (per-layer vectors empty), exactly as before; otherwise the
/// per-layer vectors.  Only the header region of the file is read; no tensor data is touched.
bool ds4_moe_geom_from_gguf(const std::string& gguf, Ds4MoeGeom& g, std::string& err);

/// The native blob layout for `g` (the three role slices back to back) and its byte size.  False with `err` when
/// ggml-cpu has no dot product for the pair at this geometry, i.e. when the CPU tier could not run at all.  For a
/// per-layer geometry this is the LARGEST routed layer's layout (what buffers are sized by).
bool ds4_moe_blob_layout(const Ds4MoeGeom& g, cpu::NativeFmt& f, std::string& err);
/// Layer `l`'s own layout (any geometry).  False for a layer without routed experts.
bool ds4_moe_layer_layout(const Ds4MoeGeom& g, int64_t l, cpu::NativeFmt& f, std::string& err);

// ================================ blob sources ================================

/// Where one expert's bytes come from.  `read` is synchronous and const: the tier reads under a lock-free
/// per-thread split, and a source that is not reentrant is the caller's problem (both shipped ones are).
class Ds4BlobSource {
public:
    virtual ~Ds4BlobSource() = default;
    /// A short label for the startup report ("gguf mmap", "memory").
    virtual const char* kind() const = 0;
    /// Validates the source against the geometry (the roles exist, their types and strides are right).
    virtual bool open(std::string& err) = 0;
    /// Bytes of one expert blob: [gate rows | up rows | down rows].  The LARGEST layer's for a per-layer geometry.
    virtual size_t blob_bytes() const = 0;
    /// Bytes of one expert blob of layer `l` (`blob_bytes()` unless the formats differ per layer).
    virtual size_t blob_bytes(int64_t l) const { (void) l; return blob_bytes(); }
    /// Writes one expert's blob into `dst` (blob_bytes() large).
    virtual void read(int64_t layer, int64_t expert, uint8_t* dst) const = 0;
    /// Part `part` of `nparts` of the same blob: bytes [p0, p1) of it into dst + p0 (p0 = blob * part / nparts, 4 KiB
    /// rounded; the parts tile the blob), so several threads can read ONE blob.  Default: part 0 reads it all.
    virtual void read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const {
        if (part == 0) read(layer, expert, dst);
        (void) nparts;
    }
};

/// The production source: the three per-role tensors of a `deepseek4` GGUF, mmap'd in place, so assembling one
/// blob is three memcpys of pages the OS already has.  Reading `N` experts touches `N * 6.75 MiB` of the file,
/// not the model.
class Ds4GgufBlobs : public Ds4BlobSource {
public:
    Ds4GgufBlobs(Ds4MoeGeom g, std::string gguf);
    ~Ds4GgufBlobs() override;
    Ds4GgufBlobs(const Ds4GgufBlobs&) = delete;
    Ds4GgufBlobs& operator=(const Ds4GgufBlobs&) = delete;

    const char* kind() const override { return "gguf mmap"; }
    bool open(std::string& err) override;
    size_t blob_bytes() const override { return blob_; }
    size_t blob_bytes(int64_t l) const override;
    void read(int64_t layer, int64_t expert, uint8_t* dst) const override;
    void read_part(int64_t layer, int64_t expert, int part, int nparts, uint8_t* dst) const override;

private:
    /// Bytes [lo, hi) of role r's slice of (layer, expert) into dst (O_DIRECT, pread, then the mmap as fallbacks).
    void read_span(int r, int64_t layer, int64_t expert, size_t lo, size_t hi, uint8_t* dst) const;
    struct Impl;
    std::unique_ptr<Impl> im_;
    Ds4MoeGeom g_;
    std::string path_;
    size_t blob_ = 0;
};

/// An in-memory source: the gate's synthetic arms and anything that builds expert bytes itself.  `set` copies, so
/// the caller may free its buffer.  Sparse by (layer, expert) - a test materializes a handful of experts of an
/// 11,008-slot model, so the storage must not be sized for the whole table.  A pair never set reads as zeros;
/// `has()` is what a test asserts with, so reaching that fallback is a visible mistake rather than a silent one.
class Ds4MemoryBlobs : public Ds4BlobSource {
public:
    Ds4MemoryBlobs(Ds4MoeGeom g, size_t blob_bytes) : g_(g), blob_(blob_bytes) {}
    const char* kind() const override { return "memory"; }
    bool open(std::string& err) override;
    size_t blob_bytes() const override { return blob_; }
    void read(int64_t layer, int64_t expert, uint8_t* dst) const override;
    void set(int64_t layer, int64_t expert, const uint8_t* blob);
    bool has(int64_t layer, int64_t expert) const;
    int64_t count() const { return (int64_t) blobs_.size(); }

private:
    Ds4MoeGeom g_;
    size_t blob_ = 0;
    std::map<std::pair<int64_t, int64_t>, std::vector<uint8_t>> blobs_;
};

// ================================ configuration ================================

/// The measured best arm of FINDINGS s12 as the defaults: 2150 slots (the 24 GiB card's real expert budget),
/// pcie 0.25 dithered, pf-b 1.43 experts/layer, and a host arena budgeted for every expert of the real model
/// (72.6 GiB).  `arena_gib` is a budget, not a promise: the arena fills in profile order up to it and everything
/// past it is read from the blob source on demand (the file tier).
struct Ds4MoeConfig {
    int64_t slots = 2150;         ///< VRAM expert slots (6.75 MiB each); 0 or `no_cache` = none
    double pcie_frac = 0.25;      ///< share of the (non-prefetched) misses computed on the GPU
    int threads = 0;              ///< CPU pool workers; 0 = every physical core but the first
    bool pin = true;              ///< register the host arena with CUDA so the PCIe share DMAs out of it
    double arena_gib = 72.6;      ///< host arena budget; the file tier takes the rest
    double pf_b = 1.43;           ///< predicted prefetch budget, experts per layer
    bool dither = true;           ///< error-diffused rounding for both the PCIe share and the prefetch budget
    bool cpu_only = false;        ///< no CUDA at all: the pool computes every expert
    bool no_cache = false;        ///< GPU tier without a VRAM cache (every miss is CPU or PCIe)
    double mem_floor_gib = 3.0;   ///< refuse an arena that would leave less than this much MemAvailable
    double max_arena_gib = 60.0;  ///< hard ceiling on the arena, whatever `arena_gib` says
    /// VRAM budget for the expert cache in GiB, for geometries whose blob size varies per layer (MiMo): the seed
    /// takes the ranked profile's head until `slots` experts OR this many bytes, sizing each slot to its expert.
    /// 0 = `slots` x the largest blob.  Ignored for uniform geometries (DS4: `slots` alone, as before).
    double slot_gib = 0.0;
    /// Per-layer-sized geometries (MiMo) only: leave the experts the VRAM seed takes OUT of the host arena.  That
    /// cache never evicts or admits after the seed, so their arena copies are never read again; the bytes go to the
    /// next-ranked experts instead (fewer file-tier reads).  Off = the arena holds the ranking's head as before.
    bool arena_skip_resident = false;
    /// Decode: a file-tier expert that `run` had to read replaces the least-recently-used arena expert of its layer
    /// (same blob size), so the experts a conversation uses stay in RAM after their first use.  Off = static arena.
    bool arena_adapt = false;
    /// Prompt chunks (`run_chunk`): the streamed experts' products through llama.cpp's MMQ (int8 tensor cores; the
    /// activations rounded to MMQ's q8_1) instead of the grouped MMVQ kernel, on the layers whose gate/up/down types
    /// MMQ covers and that have no SwiGLU clamp.  Needs a build with the prompt MMQ path (`strata_mmq`, with
    /// STRATA_MMQ_KQUANTS for K-quants / MXFP4); otherwise, and for VRAM-resident experts, the MMVQ kernel runs.
    bool chunk_mmq = false;
    /// Decode, per-layer-sized geometries (MiMo): a PCIe-share miss (and a prefetched expert the routing used) takes
    /// the least-recently-used VRAM slot of its layer instead of a staging buffer, so the cache follows the
    /// conversation (route_probe sim, 1800 slots: static 40.5% held-out hit, LRU 61.6%; FINDINGS s16).  Same math per
    /// expert; only where the blob sits changes.  Off = the static seed, as before.
    bool vram_lru = false;
    uint64_t seed = 20261005;     ///< the synthetic arm's generator (unused by the tier itself)

    static constexpr int kPredW = 16;   ///< ranked predictions the caller passes to `prefetch`
    static constexpr int kMaxPf = 6;    ///< prefetch staging slots (the DMA window of one layer)
};

/// Per-token counters, split exactly as FINDINGS s12 reports them.  `hits` includes the prefetched misses that
/// the caller then routed to (they are computed from the staged blob on the GPU, like a resident slot).
struct Ds4MoeStats {
    int64_t hits = 0;               ///< served from a resident VRAM slot
    int64_t prefetched_useful = 0;  ///< routed misses whose blob the prefetch had already staged
    int64_t prefetch_issued = 0;    ///< DMAs the prefetch started (useful or not)
    int64_t prefetch_dummy = 0;     ///< guesses outside the arena (the DMA happens; the bytes are zeros)
    int64_t cpu = 0;                ///< computed by the CPU pool
    int64_t pcie = 0;               ///< misses DMA'd to the GPU and computed there
    int64_t file_tier = 0;          ///< reads served by the blob source instead of the arena
    double gap_ms = 0, hit_ms = 0, pcie_ms = 0, cpu_ms = 0, wall_ms = 0;
    double file_ms = 0;             ///< time spent reading file-tier blobs (inside run / prefetch)
    int64_t arena_swaps = 0;        ///< arena_adapt: file-tier experts moved into the arena
    int64_t vram_swaps = 0;         ///< vram_lru: misses / prefetches moved into a VRAM slot
    int64_t vram_demotes = 0;       ///< ...of which the victim went back to the arena (it had no arena copy)

    void add(const Ds4MoeStats& o);
    int64_t lookups() const { return hits + cpu + pcie; }
};

/// The tier's private state, defined in ds4_moe.cpp.  It carries the CUDA members in a CUDA build and none in a
/// CPU-only one, which is what keeps this header - and every binary linking it - free of a CUDA dependency.
struct Ds4MoeImpl;

/// The tier.  One instance serves every layer of a model (its cache is `n_layers x n_experts`), so the engine
/// holds one and calls `run` 43 times per token.
class Ds4MoeTier {
public:
    Ds4MoeTier();
    ~Ds4MoeTier();
    Ds4MoeTier(const Ds4MoeTier&) = delete;
    Ds4MoeTier& operator=(const Ds4MoeTier&) = delete;

    /// `gguf` convenience: builds a `Ds4GgufBlobs` over the file and takes ownership of it.
    bool init(const std::string& gguf, const Ds4MoeConfig& cfg, std::string& err);
    /// The general form.  `blobs` must outlive the tier.  In CUDA mode this allocates the device buffers and the
    /// VRAM cache; in `cpu_only` it allocates nothing but the CPU pool.  It does NOT build the host arena (52-73
    /// GiB of the real model): that is `build_arena`, which `seed_from_routes`/`seed_from_ranked` call, because
    /// the arena is ORDERED and building it twice would read the file twice.  Seed before the first `run`, or the
    /// tier runs the file tier for every expert (correct, slower).
    bool init(const Ds4MoeGeom& geom, Ds4BlobSource* blobs, const Ds4MoeConfig& cfg, std::string& err);

    /// Builds the host arena, in `ranked` order, up to the configured budget.  Called by the seed functions with
    /// the routing profile, or with every (layer, expert) in index order by a caller that has none; public so a
    /// test can force it.  The first build wins: a second call with a ranking after a ranked build is a no-op
    /// (the same rule the cache follows, where the first seed takes the free slots), so a re-seed re-fills the
    /// cache and never re-reads the arena.
    bool build_arena(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);

    /// Ranks every (layer, expert) by routing frequency over the TRAIN half of a `ds4routes.bin` file
    /// (alternating 512-token blocks - the split `tools/ds4/route_skew.py` scores with), builds the host arena
    /// from it and fills the VRAM cache with the top `slots` of it.  The format is `moe_replay.cpp`'s:
    /// [512-token ubatch][layer][token][6] u16.
    bool seed_from_routes(const std::string& routes_bin, std::string& err);
    /// The same for route files with `batch_tokens`-token ubatches (tools/mimo/route_probe writes 2048) and only the
    /// routed layers in them (`Ds4MoeGeom::n_routed()` per token).
    bool seed_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err);
    /// The same fill from an already-ranked list (the arena and the gate both have one to hand).
    bool seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);

    /// Issues, asynchronously, the DMAs for up to `pf_b` of `top16`'s experts that are not resident, into this
    /// layer's staging slots.  Call it after the router prediction for `layer` and BEFORE the layer's dense GPU
    /// work; `run(layer, ...)` consumes what it staged and clears the staging.  No-op in `cpu_only` and when
    /// `pf_b <= 0`.  `n` may be smaller than the caller's list; never larger than kPredW.
    void prefetch(int64_t layer, const int32_t* top16, int n = Ds4MoeConfig::kPredW);

    /// out[i] = sum_k w6[k] * expert_{ids6[k]}(x)[i], with `x` on the host.  Synchronous.
    bool run(int64_t layer, const int32_t* ids6, const float* w6, const float* x, float* out);
    /// The same with `x` already on the device (the engine's attention output).  CUDA modes only.
    bool run_dev(int64_t layer, const int32_t* ids6, const float* w6, const void* x_dev, float* out);

    void close();

    const Ds4MoeStats& stats() const;
    void reset_stats();
    void add_stats(const Ds4MoeStats& s);

    const Ds4MoeGeom& geom() const;
    const cpu::NativeFmt& fmt() const;
    const Ds4MoeConfig& config() const;
    const char* mode() const;          ///< "cpu-only" | "gpu" | "uninitialised"
    int64_t resident() const;          ///< experts admitted to the VRAM cache
    int64_t arena_experts() const;     ///< blobs the host arena holds
    int64_t file_tier() const;         ///< blobs the arena could not take
    double arena_gib() const;
    /// Whether `expert` of `layer` is resident in a VRAM slot right now (false in CPU-only mode).
    bool resident(int64_t layer, int64_t expert) const;
    /// Whether `expert` of `layer` is in the host arena (false in CPU-only mode and for the file tier).
    bool in_arena(int64_t layer, int64_t expert) const;
    /// `nt` (1..4) tokens of one layer at once - draft verification: `ids`/`w` are nt*top_k (token-major), `x` and
    /// `out` nt*n_embd.  Each distinct expert is read/fetched once and applied to every token that routed to it.
    bool run_multi(int64_t layer, int nt, const int32_t* ids, const float* w, const float* x, float* out);
    /// A prompt chunk: `n` tokens (any count) of one layer, `ids`/`w` n*top_k token-major, `x`/`out` n*n_embd.  GPU
    /// tier: every expert on the card - resident from its slot, the rest streamed once per chunk (MiMo prefill;
    /// MIMO_CHUNK_STAGE_MIB sizes the two streaming halves, default 1024).  CPU tier: token by token.
    bool run_chunk(int64_t layer, int n, const int32_t* ids, const float* w, const float* x, float* out);
    /// The arena alone from a routing profile (the order `seed_from_routes` would give it, incl. arena_skip_resident),
    /// WITHOUT opening the VRAM cache - so prompt chunks can run first and `release_chunk` can give their buffers back
    /// before the cache takes the VRAM.  A later `seed_from_routes` keeps this arena and only seeds the cache.
    bool build_arena_from_routes(const std::string& routes_bin, int batch_tokens, std::string& err);
    /// Frees run_chunk's device and pinned buffers (the next run_chunk allocates them again).
    void release_chunk();
    cpu::ExpertPool& pool();           ///< the CPU tier's worker pool (the gate's single-expert check uses it)

private:
    std::unique_ptr<Ds4MoeImpl> im_;
};

/// One expert through the CPU pool, in native format: `out` (n_embd floats) from `x`, using the tier's own
/// kernels.  Exposed because the gate needs the single-expert number as well as the weighted sum, and because it
/// is the only part of the CPU path that is worth testing on its own.
///
/// The SwiGLU is clamped per `f.swiglu_limit` (see the file header).
bool ds4_moe_cpu_expert(const cpu::NativeFmt& f, int64_t n_embd, int64_t n_ff, const uint8_t* blob,
                        const float* x, float* out, cpu::ExpertPool& pool);

}  // namespace strata::ds4
