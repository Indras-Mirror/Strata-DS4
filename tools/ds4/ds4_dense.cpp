// tools/ds4/ds4_dense.cpp - see ds4_dense.hpp for the API and docs/ds4/ENGINE_DENSE.md for the state layout.
//
// The op sequence mirrors tools/ds4/ds4_ref.cpp (the Phase-3 verified reference forward) one token at a time.
// Everything the reference does over the whole sequence is done here over the *visible* keys only: the raw
// sliding window is the last n_swa tokens, the compressed source is the completed blocks, and the reference's
// visibility masks become the shape of the decode graph.  ggml-cpu's flash-attn skips a masked key outright
// (`if (mv == -INFINITY) continue;`, ops.cpp) and the visible keys keep their relative order, so the attention
// numerics are the reference's rather than an approximation of them.

#include "ds4_dense.hpp"

#include "ggml-alloc.h"
#include "ggml-impl.h"   // ggml_cgraph::uid + ggml_graph_next_uid (graph reuse, see alloc_graph)
#include "ggml-cpu.h"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace strata;

namespace {

constexpr float NEG_INF  = -INFINITY;
constexpr float SENTINEL = -1.0e30f;   // "compressor state row not written yet" score: exp() underflows to 0

// Tokens one verify pass carries (the main token + up to 3 drafts).  Every ring is this many - 1 slots longer than
// its gather window, so a later token of a pass never overwrites a row an earlier token of the same pass reads.
constexpr int64_t kNtMax = 4;

// Every graph here has its own gallocr and never changes after it is built, so it needs allocating once.  Stamping it
// with a uid then lets ggml-cuda skip its per-call node-property memcpy/memcmp (ggml_cuda_graph_update_required) and
// reuse the captured CUDA graph as is: ~90 graph launches per token, 165-243 nodes each, all CPU time the GPU waits
// on (the layers are serial).  The inputs are uploaded into fixed tensors, so nothing a reused graph reads moves.
// DS4_GRAPH_REUSE=0: allocate and compare every call, as before (A/B).
// With one allocator shared by every decode graph (Impl::make_allo), a graph that needs more than the buffer holds
// makes ggml-alloc reallocate it, which leaves every graph allocated earlier pointing into freed memory: the
// allocator's epoch then advances and each graph re-allocates (and ggml-cuda re-captures it) on its next use.
// reserve_graphs reserves every variant first, so in practice the buffer reaches its size before any graph runs.
bool alloc_graph(ggml_gallocr_t a, ggml_cgraph * g) {
    static const bool reuse = [] { const char * e = std::getenv("DS4_GRAPH_REUSE"); return !e || std::atoi(e) != 0; }();
    static std::map<const void *, uint64_t> epoch;                  // per allocator
    static std::map<const ggml_cgraph *, uint64_t> graph_epoch;     // the epoch each graph was allocated in
    uint64_t & ep = epoch[a];
    auto it = graph_epoch.find(g);
    if (reuse && g->uid != 0 && it != graph_epoch.end() && it->second == ep) return true;
    if (it != graph_epoch.end() && it->second != ep) {
        // allocated before the shared buffer moved: ggml-alloc never re-binds a tensor that has data, so unbind this
        // graph's compute-buffer tensors (and views of them) first.  Persistent state, inputs and weights live in
        // other buffers and keep theirs.
        auto computed = [](const ggml_tensor * t) {
            return t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE;
        };
        auto unbind = [&](ggml_tensor * t) {
            if (computed(t) || (t->view_src && computed(t->view_src))) { t->data = nullptr; t->buffer = nullptr; }
        };
        const int nn = ggml_graph_n_nodes(g);
        for (int i = 0; i < nn; ++i) {
            ggml_tensor * t = ggml_graph_node(g, i);
            for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) unbind(t->src[j]);
            unbind(t);
        }
        for (int i = 0; i < g->n_leafs; ++i) unbind(g->leafs[i]);
    }
    const size_t before = ggml_gallocr_get_buffer_size(a, 0);
    if (!ggml_gallocr_alloc_graph(a, g)) return false;
    if (ggml_gallocr_get_buffer_size(a, 0) != before && before != 0) ++ep;   // the buffer moved
    graph_epoch[g] = ep;
    if (reuse) g->uid = ggml_graph_next_uid();
    return true;
}

int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// ------------------------------------------------------------------ weights (zero-copy on CPU, uploaded off-CPU)

struct WStore {
    ggml_context * ctx = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    std::map<std::string, ggml_tensor *> t;

    ~WStore() {
        for (ggml_backend_buffer_t b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "[ds4_dense] missing weight tensor %s\n", n.c_str());
            std::abort();
        }
        return it->second;
    }
};

bool is_routed_expert(const std::string & n) {
    return n.size() > 12 && n.compare(n.size() - 12, 12, "_exps.weight") == 0;
}

// Off-CPU, `token_embd` stays on the host (begin_token looks the row up there: 1 GiB of VRAM the expert cache gets
// instead) and, with `skip_experts`, the routed experts are not loaded at all.
// --dense-requant: the big Q8_0 dense matrices, by name
bool requant_target(const std::string & n) {
    for (const char * s : { "attn_q_b.weight", "attn_output_a.weight", "attn_output_b.weight", "ffn_gate_shexp.weight",
                            "ffn_up_shexp.weight", "ffn_down_shexp.weight" })
        if (n.size() > std::strlen(s) && n.compare(n.size() - std::strlen(s), std::string::npos, s) == 0 &&
            n.rfind("blk.", 0) == 0 && n.find("indexer") == std::string::npos)
            return true;
    if (n.size() > 21 && n.compare(n.size() - 21, 21, ".nextn.eh_proj.weight") == 0) return true;   // MTP block (BF16)
    return n == "output.weight";
}

// `prefix` non-empty: only the tensors whose name starts with it (the MTP file: its own block - its token_embd and
// output duplicate the trunk's), plus those named in `alias` (file name -> store name: the MTP file's own hc_head
// weights, which differ from the trunk's).  A second call appends to the same store.
bool load_weights(const GgufModel & model, ggml_backend_t backend, WStore & w, bool skip_experts, int requant,
                  std::string & err, const std::string & prefix = "",
                  const std::map<std::string, std::string> & alias = {}) {
    if (!w.ctx) {
        ggml_init_params ip = { /*mem_size*/ 256ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        w.ctx = ggml_init(ip);
        if (!w.ctx) { err = "ggml_init(weights) failed"; return false; }
    }
    ggml_set_no_alloc(w.ctx, true);
    auto wanted = [&](const std::string & n) { return prefix.empty() || n.rfind(prefix, 0) == 0 || alias.count(n); };
    auto sname = [&](const std::string & n) { auto it = alias.find(n); return it == alias.end() ? n : it->second; };

    const bool cpu = ggml_backend_get_default_buffer_type(backend) == ggml_backend_cpu_buffer_type();

    if (cpu) {
        // bind every tensor straight onto the GGUF mmap (ds4_ref's loader)
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile & sh = model.shard(s);
            const auto & tensors = sh.tensors();
            if (tensors.empty()) continue;
            const uint8_t * any = sh.tensor_data(tensors[0]);
            uint8_t * base = const_cast<uint8_t *>(any) - sh.data_start() - tensors[0].offset;
            ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
            if (!buf) { err = "cannot wrap " + sh.path(); return false; }
            w.bufs.push_back(buf);
            for (const auto & ti : tensors) {
                if (!wanted(ti.name)) continue;
                if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
                int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
                for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
                ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
                if (!t) { err = "cannot create " + ti.name; return false; }
                ggml_set_name(t, sname(ti.name).c_str());
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[sname(ti.name)] = t;
            }
        }
        return true;
    }

    // non-CPU: create the tensor set, allocate one backend buffer for it, upload the payloads
    auto skip = [&](const std::string & n) {
        return !wanted(n) || n == "token_embd.weight" || (skip_experts && is_routed_expert(n));
    };
    for (size_t s = 0; s < model.size(); ++s) {
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip(ti.name)) continue;
            if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
            int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
            for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
            ggml_type ty = (ggml_type) ti.type;
            if (requant >= 0 && (ty == GGML_TYPE_Q8_0 || ty == GGML_TYPE_BF16) && requant_target(ti.name) &&
                ne[0] % ggml_blck_size((ggml_type) requant) == 0)
                ty = (ggml_type) requant;
            // every other BF16 matrix (the MTP file's attn_kv, attn_q_a) -> Q8_0, like the trunk's: a BF16 mul_mat
            // goes through cuBLAS on CUDA, whose first use loads its kernels into VRAM after --slots auto has sized the
            // expert cache (cudaGraphInstantiate OOM with --mtp, 2026-10-07)
            else if (ty == GGML_TYPE_BF16 && ti.shape.size() >= 2 && ne[0] % 32 == 0)
                ty = GGML_TYPE_Q8_0;
            ggml_tensor * t = ggml_new_tensor(w.ctx, ty, (int) ti.shape.size(), ne);
            if (!t) { err = "cannot create " + ti.name; return false; }
            ggml_set_name(t, sname(ti.name).c_str());
            w.t[sname(ti.name)] = t;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buf) { err = "cannot allocate the weight buffer on the backend"; return false; }
    w.bufs.push_back(buf);
    for (size_t s = 0; s < model.size(); ++s) {
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip(ti.name)) continue;
            ggml_tensor * t = w.t[sname(ti.name)];
            if ((int) t->type == (int) ti.type) {
                ggml_backend_tensor_set(t, model.shard(s).tensor_data(ti), 0, ggml_nbytes(t));
                continue;
            }
            // requantize: rows dequantized with the file type's to_float, re-quantized to t->type, in parallel
            const int64_t n0 = t->ne[0], nr = ggml_nrows(t);
            const size_t src_row = ggml_row_size((ggml_type) ti.type, n0), dst_row = ggml_row_size(t->type, n0);
            const uint8_t * src = model.shard(s).tensor_data(ti);
            std::vector<uint8_t> dst((size_t) nr * dst_row);
            const int nth = std::max(1u, std::thread::hardware_concurrency());
            std::vector<std::thread> th;
            for (int k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> f((size_t) n0);
                    for (int64_t r = k; r < nr; r += nth) {
                        ggml_get_type_traits((ggml_type) ti.type)->to_float(src + (size_t) r * src_row, f.data(), n0);
                        ggml_quantize_chunk(t->type, f.data(), dst.data() + (size_t) r * dst_row, 0, 1, n0, nullptr);
                    }
                });
            for (auto & x : th) x.join();
            ggml_backend_tensor_set(t, dst.data(), 0, dst.size());
        }
    }
    return true;
}

}  // namespace

// ================================================================== Impl

struct Ds4Dense::Impl {
    std::string  path;
    Ds4Geometry  g;
    std::unique_ptr<GgufModel> model;
    ggml_backend_t backend = nullptr;
    bool own_backend = false;
    int  n_threads = 8;

    WStore          w;
    std::unique_ptr<GgufModel> mtp_model; // the MTP (nextn) head's own GGUF, when loaded
    int64_t         il_mtp = -1;          // its block index (= n_trunk), or -1
    int64_t         n_trunk = 0;          // trunk layers (ly holds n_trunk + 1 entries with the MTP block)
    ggml_context *  sctx = nullptr;     // persistent decode state
    // Per-step inputs and per-layer router outputs live at fixed offsets in two small buffers so a token costs ONE
    // upload of every layer's inputs and one readback per layer (was ~11 tensor_set + 3 tensor_get per layer, each a
    // stream sync on CUDA: 1283 copies + 905 syncs per token measured by nsys).
    ggml_context *  ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr, obuf = nullptr;
    ggml_tensor *   i_span = nullptr;   // I8 over the whole input buffer
    std::vector<uint8_t> in_host;       // its host mirror
    uint8_t *       in_base = nullptr;  // device base of ibuf (offsets are t->data - in_base)
    int             in_pos = -1;        // the first position whose inputs are uploaded
    int             in_n   = 0;         // ... and the pass's token count
    int             cur_tid = -1;       // the token id in i_tid
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_context *  gctx = nullptr;     // graph node structs (data comes from the gallocrs)
    std::string     err;

    // ---- prompt chunks (prefill_*).  While `pf` is set the builders build a chunk graph: the rings are read as
    // [ring | the chunk's own rows] and written back with the chunk's last rows, the gate taps are skipped and the
    // graph is not registered.  pf_swap() points the decode hand-off / input tensors at the chunk-sized ones in P for
    // the duration of one build (the graph keeps the pointers it was built with).
    bool    pf = false;
    int64_t pf_cap = 0;                  // Ds4DenseConfig::prefill_chunk
    int     pf_pos0 = 0, pf_n = 0;       // the chunk begun by prefill_begin
    int64_t pf_row0 = 0;                 // build_head in pf mode: first row of the chunk
    ggml_context * pf_ctx = nullptr;     // the chunk tensors (P)
    ggml_backend_buffer_t pf_buf = nullptr;
    ggml_backend_buffer_t pf_obuf = nullptr;   // the chunk's router output blocks (o_f / o_i alias one buffer)
    ggml_gallocr_t pf_allo = nullptr;    // one allocator for every chunk graph: each is built, run once and dropped
    std::vector<uint8_t> pf_meta;        // the chunk graph's node structs (one reused buffer)
    ggml_tensor * pf_logits = nullptr;   // the last chunk head graph's output
    struct PfRatio {                     // per compress ratio (4 = CSA, 128 = HCA): the layers of a ratio share inputs
        ggml_tensor * i_slot_comp = nullptr, * i_slot_state = nullptr, * i_comp_pos = nullptr, * i_state_pos = nullptr,
                    * i_idx_state = nullptr, * vis_full = nullptr;
    };
    struct PfSet {
        ggml_tensor * x_state = nullptr, * routed_sum = nullptr, * i_tid = nullptr, * i_emb = nullptr, * i_pos = nullptr;
        ggml_tensor * hap = nullptr, * post_f = nullptr, * comb_f = nullptr, * shexp = nullptr, * o_f = nullptr,
                    * o_i = nullptr;
        ggml_tensor * i_slot_raw = nullptr, * i_idx_raw = nullptr, * i_mask_raw = nullptr;
        PfRatio r4, r128;
    } P;
    // host side of a chunk: router output blocks [fn | ids | wts] x n, the ffn_norm rows the expert tier reads and the
    // routed sums it writes - pinned when the backend has a host buffer type (CUDA): pageable copies are staged by
    // the driver in pieces that queue behind the tier's expert DMAs on the one host-to-device copy engine
    ggml_backend_buffer_t pf_hbuf = nullptr;
    std::vector<uint8_t> pf_hvec;        // the same, pageable (CPU backend)
    uint8_t * pf_out_host = nullptr;
    float *   pf_fn = nullptr;
    float *   pf_routed = nullptr;
    void pf_free() {
        if (pf_allo) ggml_gallocr_free(pf_allo);
        if (pf_buf) ggml_backend_buffer_free(pf_buf);
        if (pf_obuf) ggml_backend_buffer_free(pf_obuf);
        if (pf_ctx) ggml_free(pf_ctx);
        pf_allo = nullptr; pf_buf = nullptr; pf_obuf = nullptr; pf_ctx = nullptr; P = PfSet();
        std::vector<uint8_t>().swap(pf_meta);
        if (pf_hbuf) ggml_backend_buffer_free(pf_hbuf);
        pf_hbuf = nullptr;
        std::vector<uint8_t>().swap(pf_hvec);
        pf_out_host = nullptr; pf_fn = nullptr; pf_routed = nullptr;
    }

    // One graph allocator per graph, never a shared one.  ggml-alloc binds a tensor's data pointer once
    // and skips re-binding it on later allocations, while a reserve triggered by a *different* graph
    // frees and reallocates the chunks it had bound - a shared allocator therefore leaves the tensors of
    // every other graph dangling into freed memory as soon as one graph grows.  Per-graph allocators also
    // give each decode graph the stable, dedicated buffer ggml-cuda needs to capture it as a CUDA graph.
    bool ensure_n(int64_t n);           // init/head/finish graphs for n-token passes (defined after the builders)

    // DS4_SHARED_ALLO (default on): every decode graph on ONE allocator - they run one at a time and everything that
    // outlives a graph lives in sbuf / ibuf / obuf, so their intermediates may share memory (was ~850 MiB of VRAM for
    // ~500 per-graph arenas at p600).  Needs alloc_graph's epoch (a growth reallocates the shared buffer).
    // DS4_SHARED_ALLO=0: one allocator per graph, as before.
    mutable ggml_gallocr_t shared_allo = nullptr;
    static bool shared_on() {
        static const bool on = [] { const char * e = std::getenv("DS4_SHARED_ALLO"); return !e || std::atoi(e) != 0; }();
        return on;
    }
    ggml_gallocr_t make_allo() const {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        if (!shared_on()) return ggml_gallocr_new(buft);
        if (!shared_allo) shared_allo = ggml_gallocr_new(buft);
        return shared_allo;
    }
    void free_allo(ggml_gallocr_t a) { if (a && a != shared_allo) ggml_gallocr_free(a); }

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, NHPG = 0, RQ = 0, SWA = 0;
    int64_t RAW = 0;   // raw window ring storage = SWA + kNtMax - 1
    int64_t IDXH = 0, IDXK = 0, IDXTOPK = 0;
    int64_t NEXP = 0, NUSED = 0, NFF = 0, NSHEXP = 0;

    // A pass carries n <= kNtMax tokens (one per column); every per-token tensor below has a kNtMax axis and the
    // graphs for n tokens view its first n columns, so the n = 1 graphs are the single-token graphs exactly.
    ggml_tensor * x_state    = nullptr;   // [D, hc, kNtMax]  layer input / output state
    ggml_tensor * routed_sum = nullptr;   // [D, kNtMax]      routed-expert sums fed to finish_layer
    ggml_tensor * i_tid      = nullptr;   // I32[kNtMax]
    ggml_tensor * i_emb      = nullptr;   // F32[D, kNtMax]   embd[tid], looked up on the host (begin_tokens)
    const uint8_t * embd_data = nullptr;  // token_embd on the GGUF mmap
    int           embd_type  = -1;
    size_t        embd_row   = 0;         // bytes per row
    std::vector<float> host_emb;
    ggml_tensor * i_pos      = nullptr;   // I32[kNtMax]
    ggml_tensor * rot        = nullptr;   // [idx_k, idx_k] F32 Walsh-Hadamard (lightning indexer)
    std::vector<float> host_logits;       // [V, kNtMax]
    int64_t       o_blk = 0;              // bytes per token in a layer's router output block [fn | ids | wts]
    int64_t       o_ids_off = 0, o_wts_off = 0;

    struct Var {
        int64_t cap = 0;
        int64_t n = 1;                    // tokens per pass
        // false: the pass completes no compressed block (one token, (pos+1) % ratio != 0) - the compressor and
        // indexer ring states are written, but the pooled row (invisible until its block completes) is not built
        bool emit = true;
        ggml_tensor * vis = nullptr;      // F32 [cap, n] 0 visible / -inf hidden, per query
        ggml_tensor * dbg_isc = nullptr;  // F32 [cap, n] indexer scores + visibility (DS4_DBG_INDEXER=1)
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t allo = nullptr;
    };
    struct Layer {
        int64_t ratio = 0;
        bool    csa = false;
        int64_t ring = 0, sdim = 0, comp_max = 0;
        int64_t ring_sz = 0;              // compressor state ring storage = ring + kNtMax - 1

        ggml_tensor * raw = nullptr;      // [DH, RAW]              raw sliding-window K (=V, post-RoPE)
        // compressed caches have one spare row at comp_max: when two tokens of a pass fall in the same block, the
        // earlier one writes its (partial, never visible) row there, so set_rows never sees a duplicate index
        ggml_tensor * comp = nullptr;     // [DH, comp_max + 1]     compressed K (post-RoPE)

        ggml_tensor * st_kv = nullptr;    // [sdim, ring]           compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [sdim, ring]           compressor state (scores, SENTINEL unwritten)
        ggml_tensor * icomp = nullptr;    // [IDXK, comp_max + 1]   lightning-indexer compressed K (post-WHT)
        ggml_tensor * ist_kv = nullptr;   // [2*IDXK, ring_sz]      indexer compressor state (values)
        ggml_tensor * ist_sc = nullptr;   // [2*IDXK, ring_sz]      indexer compressor state (scores)

        ggml_tensor * hap = nullptr;      // [D, HC, kNtMax]        attn hc_post result (finish residual)
        ggml_tensor * post_f = nullptr;   // [HC, kNtMax]
        ggml_tensor * comb_f = nullptr;   // [HC, HC, kNtMax]
        ggml_tensor * shexp = nullptr;    // [D, kNtMax]            shared expert output
        ggml_tensor * fn = nullptr;       // [D]                    ffn_norm activation of token 0 (device)
        ggml_tensor * t_attn_raw = nullptr;  // [NH*DH, kNtMax]     flash-attn output, pre output-LoRA
        ggml_tensor * t_attn = nullptr;      // [D, kNtMax]         attention output, post output-LoRA
        ggml_tensor * t_llast = nullptr;     // [D, HC, kNtMax]     l_last

        // per-step inputs, laid out for the pass's n (the host fills the first entries)
        ggml_tensor * i_slot_raw = nullptr;   // I32[kNtMax]               raw ring slot per token
        ggml_tensor * i_idx_raw = nullptr;    // I32[SWA + kNtMax - 1]     union window, oldest -> newest
        ggml_tensor * i_mask_raw = nullptr;   // F16[(SWA + kNtMax - 1) * kNtMax]  [n_kv, n] per-query window mask
        ggml_tensor * i_slot_comp = nullptr;  // I32[kNtMax]
        ggml_tensor * i_slot_state = nullptr; // I32[kNtMax]
        ggml_tensor * i_comp_pos = nullptr;   // I32[kNtMax]
        ggml_tensor * i_state_pos = nullptr;  // I32[kNtMax]
        ggml_tensor * i_idx_state = nullptr;  // I32[ring * kNtMax]        [ring, n] per-token state gather

        std::vector<Var> vars;
        ggml_cgraph * gf_predict = nullptr;
        ggml_tensor * predict_out = nullptr;
        ggml_gallocr_t allo_predict = nullptr;
        ggml_cgraph * gf_finish[kNtMax + 1] = {};      // by n
        ggml_gallocr_t allo_finish[kNtMax + 1] = {};
        std::vector<float> host_fn, host_attn_raw, host_attn, host_llast;   // [.., kNtMax]
        int  tap_n = 0;                   // the n of the pass the taps / host_fn hold
        bool have_attn = false, have_finish = false;
        ggml_tensor * vis_full = nullptr; // F32 [comp_max * kNtMax]  compressed-row visibility (graphs view [cap, n])
        ggml_tensor * rbias = nullptr;    // F32 [NEXP]       cache-aware routing: added to the SELECTION score only
        // router outputs: per token a block of o_blk bytes [fn | ids | wts], kNtMax blocks, one readback per layer
        ggml_tensor * o_ids = nullptr;    // I32 [NUSED]      token 0's router ids
        ggml_tensor * o_wts = nullptr;    // F32 [NUSED]      token 0's router weights
        ggml_tensor * o_f = nullptr;      // F32 alias of the whole layer block (graphs view fn / wts columns)
        ggml_tensor * o_i = nullptr;      // I32 alias of the same bytes (graphs view the ids columns)
        ggml_tensor * o_span = nullptr;   // I8 over the whole layer block
        std::vector<uint8_t> host_out;
    };
    std::vector<Layer> ly;

    ggml_cgraph * gf_init[kNtMax + 1] = {};       // by n
    ggml_cgraph * gf_mtp_in[kNtMax + 1] = {};     // MTP input: eh_proj(enorm(embd) | hnorm(trunk state)), by n
    ggml_cgraph * gf_mtp_head[kNtMax + 1] = {};   // MTP head: hc_head -> shared_head_norm -> output, by n
    ggml_tensor * mtp_logits_t[kNtMax + 1] = {};
    ggml_gallocr_t allo_mtp_in[kNtMax + 1] = {};
    ggml_gallocr_t allo_mtp_head[kNtMax + 1] = {};
    ggml_cgraph * gf_head[kNtMax + 1] = {};
    ggml_tensor * logits_t[kNtMax + 1] = {};      // [V, n] head graph outputs
    ggml_gallocr_t allo_init[kNtMax + 1] = {};
    ggml_gallocr_t allo_head[kNtMax + 1] = {};

    ~Impl() {
        pf_free();
        for (Var & v : all_vars()) free_allo(v.allo);
        for (Layer & L : ly) {
            free_allo(L.allo_predict);
            for (ggml_gallocr_t a : L.allo_finish) free_allo(a);
        }
        for (ggml_gallocr_t a : allo_init) free_allo(a);
        for (ggml_gallocr_t a : allo_head) free_allo(a);
        for (ggml_gallocr_t a : allo_mtp_in) free_allo(a);
        for (ggml_gallocr_t a : allo_mtp_head) free_allo(a);
        if (shared_allo) ggml_gallocr_free(shared_allo);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (ibuf) ggml_backend_buffer_free(ibuf);
        if (obuf) ggml_backend_buffer_free(obuf);
        if (ictx) ggml_free(ictx);
        if (sctx) ggml_free(sctx);
        if (gctx) ggml_free(gctx);
        if (own_backend && backend) ggml_backend_free(backend);
    }

    // every attn graph variant across every layer, for the destructor above
    std::vector<Var> all_vars() {
        std::vector<Var> r;
        for (Layer & L : ly) for (Var & v : L.vars) r.push_back(v);
        return r;
    }
};

// ================================================================== graph builder helpers (ds4_ref's Graph)

namespace {

struct B {
    Ds4Dense::Impl * im = nullptr;
    ggml_context *   c  = nullptr;
    const Ds4Geometry * g = nullptr;

    ggml_tensor * W(const std::string & n) const { return im->w.get(n); }
    ggml_tensor * BL(int il, const std::string & s) const {
        return im->w.get("blk." + std::to_string(il) + "." + s);
    }
    ggml_tensor * fill_f32(std::initializer_list<int64_t> ne, float v) const {
        ggml_tensor * tmpl = nullptr;
        auto it = ne.begin();
        switch (ne.size()) {
            case 1: tmpl = ggml_new_tensor_1d(c, GGML_TYPE_F32, it[0]); break;
            case 2: tmpl = ggml_new_tensor_2d(c, GGML_TYPE_F32, it[0], it[1]); break;
            case 3: tmpl = ggml_new_tensor_3d(c, GGML_TYPE_F32, it[0], it[1], it[2]); break;
            default: std::abort();
        }
        return ggml_fill(c, tmpl, v);
    }
    ggml_tensor * v1(ggml_tensor * a, int64_t n0, size_t off) const { return ggml_view_1d(c, a, n0, off); }
    ggml_tensor * v2(ggml_tensor * a, int64_t n0, int64_t n1, size_t nb1, size_t off) const {
        return ggml_view_2d(c, a, n0, n1, nb1, off);
    }

    // hc affine: x*scale + base  (deepseek4.cpp:279-285)
    ggml_tensor * hc_affine(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) const {
        return ggml_add(c, ggml_mul(c, x, scale), base);
    }
    // deepseek4.cpp:314-349
    ggml_tensor * sinkhorn(ggml_tensor * comb) const {
        comb = ggml_soft_max(c, comb);
        ggml_tensor * eps = fill_f32({ 1 }, (float) g->hc_eps);
        comb = ggml_add(c, comb, eps);
        auto norm_cols = [&]() {
            ggml_tensor * cd = ggml_cont(c, ggml_permute(c, comb, 1, 0, 2, 3));
            ggml_tensor * cs = ggml_add(c, ggml_sum_rows(c, cd), eps);
            cs = ggml_permute(c, cs, 1, 0, 2, 3);
            comb = ggml_div(c, comb, cs);
        };
        norm_cols();
        for (int64_t i = 1; i < g->hc_sinkhorn_iters; ++i) {
            ggml_tensor * rs = ggml_add(c, ggml_sum_rows(c, comb), eps);
            comb = ggml_div(c, comb, rs);
            norm_cols();
        }
        return comb;
    }
    // deepseek4.cpp:287-312
    // The fused hyper-connection ops (ggml_dsv4_hc_comb/pre/post, llama.cpp #25585): one kernel each instead of the
    // Sinkhorn loop's ~140 tiny ops per call - on CUDA that loop alone was ~12k kernels/token.  DS4_HC_UNFUSED=1 keeps
    // the op-by-op form (the original deepseek4.cpp graph) for A/B.
    static bool unfused() { static const bool u = std::getenv("DS4_HC_UNFUSED") != nullptr; return u; }
    ggml_tensor * hc_mean(ggml_tensor * x, ggml_tensor * pre) const {
        if (!unfused()) return ggml_dsv4_hc_pre(c, x, pre);
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        ggml_tensor * acc = nullptr;
        for (int64_t h = 0; h < hc; ++h) {
            ggml_tensor * xh = v2(x, D, nt, x->nb[2], (size_t) h * x->nb[1]);
            ggml_tensor * wh = v2(pre, 1, nt, pre->nb[1], (size_t) h * pre->nb[0]);
            ggml_tensor * cur = ggml_mul(c, xh, wh);
            acc = acc ? ggml_add(c, acc, cur) : cur;
        }
        return acc;
    }
    // deepseek4.cpp:351-407
    ggml_tensor * hc_pre(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                         ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        ggml_tensor * flat = ggml_reshape_2d(c, x, hc * D, nt);
        ggml_tensor * flat_norm = ggml_rms_norm(c, flat, (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat_norm);

        ggml_tensor * scale_pre  = v1(sc, 1, 0);
        ggml_tensor * scale_post = v1(sc, 1, 1 * sizeof(float));
        ggml_tensor * base_pre   = v1(bs, hc, 0);
        ggml_tensor * base_post  = v1(bs, hc, (size_t) hc * 4);

        ggml_tensor * pre = v2(mixes, hc, nt, mixes->nb[1], 0);
        pre = ggml_sigmoid(c, hc_affine(pre, scale_pre, base_pre));
        pre = ggml_scale_bias(c, pre, 1.0f, (float) g->hc_eps);

        *post = v2(mixes, hc, nt, mixes->nb[1], (size_t) hc * mixes->nb[0]);
        *post = ggml_sigmoid(c, hc_affine(*post, scale_post, base_post));
        *post = ggml_scale(c, *post, 2.0f);

        ggml_tensor * scale_comb = v1(sc, 1, 2 * sizeof(float));
        ggml_tensor * base_comb  = v1(bs, hc * hc, (size_t) 2 * hc * 4);
        if (!unfused()) {
            *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) g->hc_eps, (int32_t) g->hc_sinkhorn_iters);
        } else {
            *comb = v2(mixes, hc * hc, nt, mixes->nb[1], (size_t) 2 * hc * mixes->nb[0]);
            *comb = hc_affine(*comb, scale_comb, base_comb);
            *comb = ggml_reshape_3d(c, *comb, hc, hc, nt);
            *comb = sinkhorn(*comb);
        }
        return hc_mean(x, pre);
    }
    // deepseek4.cpp:409-444
    ggml_tensor * hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb) const {
        if (!unfused()) return ggml_dsv4_hc_post(c, x, residual, post, comb);
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[1];
        ggml_tensor * out = nullptr;
        for (int64_t dst = 0; dst < hc; ++dst) {
            ggml_tensor * post_dst = v2(post, 1, nt, post->nb[1], (size_t) dst * post->nb[0]);
            ggml_tensor * cur = ggml_mul(c, x, post_dst);
            for (int64_t src = 0; src < hc; ++src) {
                ggml_tensor * res_src = v2(residual, D, nt, residual->nb[2], (size_t) src * residual->nb[1]);
                ggml_tensor * cs = v2(comb, 1, nt, comb->nb[2],
                                      (size_t) dst * comb->nb[0] + (size_t) src * comb->nb[1]);
                cur = ggml_add(c, cur, ggml_mul(c, res_src, cs));
            }
            cur = ggml_reshape_3d(c, cur, D, 1, nt);
            out = out ? ggml_concat(c, out, cur, 1) : cur;
        }
        return out;
    }
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wgt) const {
        return ggml_mul(c, ggml_rms_norm(c, x, (float) g->rms_eps), wgt);
    }
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int n_ctx,
                       float base, float fscale, float ext, float attn, float bf, float bs) const {
        a = ggml_rope_ext(c, a, pos, nullptr, (int) n_dims, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bs);
        return ggml_rope_set_offset(a, (int) (g->d_head - n_dims));
    }
    ggml_tensor * rope_back(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int n_ctx,
                            float base, float fscale, float ext, float attn, float bf, float bs) const {
        a = ggml_rope_ext_back(c, a, pos, nullptr, (int) n_dims, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bs);
        return ggml_rope_set_offset(a, (int) (g->d_head - n_dims));
    }
    ggml_tensor * rope_at(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int64_t off, int n_ctx,
                          float base, float fscale, float ext, float attn, float bf, float bs) const {
        a = ggml_rope_ext(c, a, pos, nullptr, (int) n_dims, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bs);
        return ggml_rope_set_offset(a, (int) off);
    }
    // orthonormal Walsh-Hadamard rotation over contiguous `n`-blocks (llama-impl.h:57-75 / llama-kv-cache.cpp:24)
    ggml_tensor * hadamard(ggml_tensor * cur, ggml_tensor * rot_m) const {
        const int64_t n = rot_m->ne[0];
        ggml_tensor * res = ggml_is_contiguous(cur)
            ? ggml_reshape_2d(c, cur, n, ggml_nelements(cur) / n)
            : ggml_cont_2d(c, cur, n, ggml_nelements(cur) / n);
        res = ggml_mul_mat(c, rot_m, res);
        return ggml_reshape_4d(c, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    }
    // the compress-RoPE / raw-RoPE parameter set ds4_ref derives per layer
    void rope_cfg(int64_t ratio, float & freq_base, float & freq_scale, float & ext_factor, float & attn_factor,
                  float & beta_fast, float & beta_slow, int & n_ctx) const {
        const bool comp = ratio != 0;
        freq_base   = comp ? (float) g->compress_rope_base : (float) g->rope_freq_base;
        freq_scale  = comp ? (float) (1.0 / g->rope_factor) : 1.0f;
        ext_factor  = comp ? 1.0f : 0.0f;
        attn_factor = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * std::log(1.0f / freq_scale));
        beta_fast   = comp ? (float) g->yarn_beta_fast : 0.0f;
        beta_slow   = comp ? (float) g->yarn_beta_slow : 0.0f;
        n_ctx       = comp ? (int) g->rope_orig_ctx : 0;
    }
};

}  // namespace

// ================================================================== graph construction

// Attention + router for one compressed-row capacity `cap` (0 = nothing compressed yet) and `n` tokens per pass
// (positions pos0 .. pos0+n-1, one per column).  All tensors are created once; per-step values arrive through the
// layer's input leaves, so the graph is stable.  n = 1 is the single-token decode graph.
//
// The n tokens share one flash-attn: the raw keys are the union window (SWA + n - 1 keys, oldest -> newest) with a
// per-query window mask, the compressed keys are the cache after every token of the pass wrote its row, with a
// per-query visibility mask (a block completed by token t is visible to tokens > t, never the reverse).  The
// compressor rings take every token's state before any gather; each token gathers its own `ring` slots.
static ggml_cgraph * build_attn(Ds4Dense::Impl & im, int il, int64_t cap, int64_t n, Ds4Dense::Impl::Layer & L,
                               bool emit = true) {
    Ds4Dense::Impl::Var var;
    var.cap = cap;
    var.n = n;
    var.emit = emit = emit || im.pf || n != 1;   // only one-token decode skips

    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 4096, false);
    B b { &im, gc, &im.g };

    const int64_t D = im.D, HC = im.HC, DH = im.DH, DHR = im.DHR, NH = im.NH, OGD = im.OGD, OL = im.OL, OG = im.OG;
    const int64_t SWA = im.SWA;
    const int64_t n_raw = SWA + n - 1;            // union of the n raw windows
    const int64_t ratio = L.ratio;
    const bool csa = L.csa;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int n_ctx;
    b.rope_cfg(ratio, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_ctx);
    const float crb = (float) im.g.compress_rope_base;

    auto first = [&](ggml_tensor * t, int64_t ne) { return ggml_view_1d(gc, t, ne, 0); };
    ggml_tensor * pos = first(im.i_pos, n);

    // ---- attention hyper-connection -------------------------------------------------
    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * attn_pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                      b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * xn = b.rms_w(attn_pre, b.BL(il, "attn_norm.weight"));

    // ---- q latent (kept for the indexer's query) --------------------------------------
    ggml_tensor * qr2 = ggml_mul_mat(gc, b.BL(il, "attn_q_a.weight"), xn);
    qr2 = b.rms_w(qr2, b.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_mul_mat(gc, b.BL(il, "attn_q_b.weight"), qr2);
    q = ggml_reshape_3d(gc, q, DH, NH, n);
    q = ggml_rms_norm(gc, q, (float) im.g.rms_eps);
    q = b.rope(q, pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- kv latent (K == V) -----------------------------------------------------------
    ggml_tensor * kv = ggml_mul_mat(gc, b.BL(il, "attn_kv.weight"), xn);
    kv = b.rms_w(kv, b.BL(il, "attn_kv_a_norm.weight"));
    kv = ggml_reshape_3d(gc, kv, DH, 1, n);
    kv = b.rope(kv, pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- raw sliding-window ring ------------------------------------------------------
    // chunk (im.pf): the window gathers from [ring | the chunk's keys]; the ring gets the chunk's last keys after
    // everything else has read it (`wb`, expanded last - ggml runs the nodes in this order)
    std::vector<ggml_tensor *> wb;
    auto tail_into = [&](ggml_tensor * ring, ggml_tensor * rows2, int64_t ring_sz, ggml_tensor * slots) {
        const int64_t k = std::min<int64_t>(n, ring_sz);
        ggml_tensor * last = ggml_view_2d(gc, rows2, rows2->ne[0], k, rows2->nb[1], (size_t) (n - k) * rows2->nb[1]);
        wb.push_back(ggml_set_rows(gc, ring, last, first(slots, k)));
    };
    ggml_tensor * raw_src = nullptr;
    if (!im.pf) {
        ggml_tensor * raw2 = ggml_set_rows(gc, L.raw, ggml_cont_2d(gc, kv, DH, n), first(L.i_slot_raw, n));
        raw_src = ggml_get_rows(gc, raw2, first(L.i_idx_raw, n_raw));   // [DH, n_raw] oldest -> newest
    } else {
        ggml_tensor * kv2 = ggml_cont_2d(gc, kv, DH, n);
        raw_src = ggml_get_rows(gc, ggml_concat(gc, L.raw, kv2, 1), first(L.i_idx_raw, n_raw));
        tail_into(L.raw, kv2, im.RAW, L.i_slot_raw);
    }
    ggml_tensor * k_all = ggml_reshape_3d(gc, raw_src, DH, 1, n_raw);             // token axis on ne2 (ds4_ref layout)
    ggml_tensor * mask = ggml_reshape_2d(gc, first(L.i_mask_raw, n_raw * n), n_raw, n);

    // ---- compressed keys -------------------------------------------------------------
    if (cap > 0) {
        ggml_tensor * comp_k = nullptr;
        ggml_tensor * cmask = nullptr;
        ggml_tensor * state_pos = first(L.i_state_pos, n);
        ggml_tensor * comp_pos  = first(L.i_comp_pos, n);
        ggml_tensor * slot_st   = im.pf ? nullptr : first(L.i_slot_state, n);   // chunk: tail_into uses its own
        ggml_tensor * idx_st    = first(L.i_idx_state, L.ring * n);   // [ring] per token, flat (get_rows takes 1-D)
        var.vis = ggml_view_2d(gc, L.vis_full, cap, n, (size_t) cap * sizeof(float), 0);

        if (csa) {
            // overlap compressor (ds4_ref attention():514-556): each token state is [ prev-half | cur-half ]
            auto overlap = [&](const std::string & pfx, int64_t head_dim, ggml_tensor * ring_kv, ggml_tensor * ring_sc)
                    -> ggml_tensor * {
                ggml_tensor * st_kv = ggml_mul_mat(gc, b.BL(il, pfx + "_kv.weight"), xn);
                ggml_tensor * st_sc = ggml_mul_mat(gc, b.BL(il, pfx + "_gate.weight"), xn);
                ggml_tensor * ape = ggml_get_rows(gc, b.BL(il, pfx + "_ape.weight"), state_pos);
                st_sc = ggml_add(gc, st_sc, ape);

                ggml_tensor * skv = ggml_cont_2d(gc, st_kv, 2 * head_dim, n);
                ggml_tensor * ssc = ggml_cont_2d(gc, st_sc, 2 * head_dim, n);
                ggml_tensor * rkv, * rsc;
                if (!im.pf) {
                    rkv = ggml_set_rows(gc, ring_kv, skv, slot_st);
                    rsc = ggml_set_rows(gc, ring_sc, ssc, slot_st);
                    if (!emit) {   // the states only: no block completes at this position
                        wb.push_back(rkv);
                        wb.push_back(rsc);
                        return nullptr;
                    }
                } else {   // [ring | the chunk's states]; the ring gets the chunk's last states afterwards
                    rkv = ggml_concat(gc, ring_kv, skv, 1);
                    rsc = ggml_concat(gc, ring_sc, ssc, 1);
                    tail_into(ring_kv, skv, L.ring_sz, L.i_slot_state);
                    tail_into(ring_sc, ssc, L.ring_sz, L.i_slot_state);
                }
                ggml_tensor * rows_kv = ggml_reshape_3d(gc, ggml_get_rows(gc, rkv, idx_st), 2 * head_dim, 2 * ratio, n);
                ggml_tensor * rows_sc = ggml_reshape_3d(gc, ggml_get_rows(gc, rsc, idx_st), 2 * head_dim, 2 * ratio, n);

                const size_t nb1 = rows_kv->nb[1], nb2 = rows_kv->nb[2];
                const size_t cur_off = (size_t) ratio * nb1 + (size_t) head_dim * 4;
                ggml_tensor * pkv = ggml_cont(gc, ggml_view_3d(gc, rows_kv, head_dim, ratio, n, nb1, nb2, 0));
                ggml_tensor * ckv = ggml_cont(gc, ggml_view_3d(gc, rows_kv, head_dim, ratio, n, nb1, nb2, cur_off));
                ggml_tensor * psc = ggml_cont(gc, ggml_view_3d(gc, rows_sc, head_dim, ratio, n, nb1, nb2, 0));
                ggml_tensor * csc = ggml_cont(gc, ggml_view_3d(gc, rows_sc, head_dim, ratio, n, nb1, nb2, cur_off));
                ggml_tensor * values = ggml_concat(gc, pkv, ckv, 1);          // [hd, 2*ratio, n]
                ggml_tensor * scores = ggml_concat(gc, psc, csc, 1);
                values = ggml_cont(gc, ggml_permute(gc, values, 1, 0, 2, 3)); // [2*ratio, hd, n]
                scores = ggml_cont(gc, ggml_permute(gc, scores, 1, 0, 2, 3));
                ggml_tensor * wts = ggml_soft_max(gc, scores);
                ggml_tensor * cc = ggml_sum_rows(gc, ggml_mul(gc, values, wts));
                cc = ggml_cont(gc, ggml_permute(gc, cc, 1, 0, 2, 3));         // [hd, 1, n]
                cc = b.rms_w(cc, b.BL(il, pfx + "_norm.weight"));
                return b.rope_at(cc, comp_pos, DHR, head_dim - DHR, n_ctx, crb,
                                 freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            };

            comp_k = overlap("attn_compressor", DH, L.st_kv, L.st_sc);

            // lightning indexer: compressed keys (post-WHT) + query, then top-k over the visible blocks, per query
            ggml_tensor * icomp2 = L.icomp;   // !emit: the block keys as they are (the current block is invisible)
            if (ggml_tensor * lk = overlap("indexer_compressor", im.IDXK, L.ist_kv, L.ist_sc)) {
                ggml_tensor * lid_k = b.hadamard(ggml_cont(gc, lk), im.rot);
                icomp2 = ggml_set_rows(gc, L.icomp, ggml_cont_2d(gc, lid_k, im.IDXK, n), first(L.i_slot_comp, n));
            }
            // ds4_ref's lid_k is 3-D [idx_k, 1, n_blocks]; this 3-D view of the ring's first `cap`
            // columns has the same shape, so the permute below lands `cap` on ne[1] exactly like the
            // reference's kp = permute(lid_k, 0,2,1,3).  A 2-D view would put `cap` on ne[2] and the
            // mul_mat would then drop the block axis (the result would be [1,1,idx_h] instead of
            // [cap,1,idx_h]) - the decoded indexer scores would be wrong and the graph layout invalid.
            ggml_tensor * lid_src = ggml_view_3d(gc, icomp2, im.IDXK, 1, cap,
                                                 icomp2->nb[1], icomp2->nb[1], 0);

            ggml_tensor * iq = ggml_mul_mat(gc, b.BL(il, "indexer.attn_q_b.weight"), qr2);
            iq = ggml_reshape_3d(gc, iq, im.IDXK, im.IDXH, n);
            iq = b.rope_at(iq, pos, DHR, im.IDXK - DHR, n_ctx, crb,
                           freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            iq = b.hadamard(iq, im.rot);
            ggml_tensor * iw = ggml_mul_mat(gc, b.BL(il, "indexer.proj.weight"), xn);
            iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.IDXK * im.IDXH)));

            ggml_tensor * qp = ggml_permute(gc, iq, 0, 2, 1, 3);            // [idx_k, n, idx_h]
            ggml_tensor * kp = ggml_permute(gc, lid_src, 0, 2, 1, 3);       // [idx_k, cap, 1]
            ggml_tensor * kq = ggml_mul_mat(gc, kp, qp);                    // [cap, n, idx_h]
            kq = ggml_cont(gc, ggml_permute(gc, kq, 2, 1, 0, 3));           // [idx_h, n, cap]
            ggml_tensor * sc = ggml_relu(gc, kq);
            sc = ggml_mul(gc, sc, ggml_reshape_4d(gc, iw, im.IDXH, n, 1, 1));
            sc = ggml_sum_rows(gc, sc);                                     // [1, n, cap]
            sc = ggml_cont(gc, ggml_permute(gc, sc, 2, 1, 0, 3));           // [cap, n, 1]
            sc = ggml_reshape_2d(gc, sc, cap, n);

            sc = ggml_add(gc, sc, var.vis);
            var.dbg_isc = sc;
            ggml_set_output(sc);

            // mask = -inf everywhere, 0 at each query's top-k rows, then ANDed with visibility
            const int64_t ntk = std::min<int64_t>(cap, im.IDXTOPK);
            // chunk off-CPU: one batched argsort for every query row (ggml-cuda's top_k launches cub once per row: 3
            // kernels x n rows x 21 CSA layers, ~63k launches for a 3K-token chunk).  The same top-k set except among
            // exact ties (ReLU zeros), which cub breaks arbitrarily too; the CPU keeps top_k (no launch cost) so its
            // chunk gate stays bit-exact against the decode loop
            const bool batched_tk = im.pf && !ggml_backend_is_cpu(im.backend);
            ggml_tensor * tk = ggml_cont(gc, batched_tk ? ggml_argsort_top_k(gc, sc, (int) ntk)
                                                        : ggml_top_k(gc, sc, (int) ntk));   // [ntk, n]
            ggml_tensor * a = ggml_fill(gc, var.vis, NEG_INF);
            a = ggml_view_4d(gc, a, 1, cap, n, 1, a->nb[0], a->nb[1], a->nb[2], 0);
            ggml_tensor * zv = b.fill_f32({ 1, ntk, n }, 0.0f);
            ggml_tensor * m = ggml_set_rows(gc, a, zv, tk);                     // query j's indices into slice j
            m = ggml_view_4d(gc, m, m->ne[1], m->ne[2], 1, 1, m->nb[2], m->nb[3], m->nb[3], 0);
            m = ggml_add(gc, m, var.vis);
            cmask = ggml_cast(gc, m, GGML_TYPE_F16);
        } else {
            // HCA: one block == `ratio` consecutive tokens, no overlap (ds4_ref attention():603-622)
            ggml_tensor * st_kv = ggml_mul_mat(gc, b.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * st_sc = ggml_mul_mat(gc, b.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_tensor * ape = ggml_get_rows(gc, b.BL(il, "attn_compressor_ape.weight"), state_pos);
            st_sc = ggml_add(gc, st_sc, ape);

            ggml_tensor * skv = ggml_cont_2d(gc, st_kv, DH, n);
            ggml_tensor * ssc = ggml_cont_2d(gc, st_sc, DH, n);
            ggml_tensor * rkv, * rsc;
            if (!im.pf) {
                rkv = ggml_set_rows(gc, L.st_kv, skv, slot_st);
                rsc = ggml_set_rows(gc, L.st_sc, ssc, slot_st);
                if (!emit) { wb.push_back(rkv); wb.push_back(rsc); }
            } else {
                rkv = ggml_concat(gc, L.st_kv, skv, 1);
                rsc = ggml_concat(gc, L.st_sc, ssc, 1);
                tail_into(L.st_kv, skv, L.ring_sz, L.i_slot_state);
                tail_into(L.st_sc, ssc, L.ring_sz, L.i_slot_state);
            }
            if (emit) {
                ggml_tensor * rows_kv = ggml_reshape_3d(gc, ggml_get_rows(gc, rkv, idx_st), DH, ratio, n);
                ggml_tensor * rows_sc = ggml_reshape_3d(gc, ggml_get_rows(gc, rsc, idx_st), DH, ratio, n);
                ggml_tensor * values = ggml_cont(gc, ggml_permute(gc, rows_kv, 1, 0, 2, 3));   // [ratio, DH, n]
                ggml_tensor * scores = ggml_cont(gc, ggml_permute(gc, rows_sc, 1, 0, 2, 3));
                ggml_tensor * wts = ggml_soft_max(gc, scores);
                ggml_tensor * cc = ggml_sum_rows(gc, ggml_mul(gc, values, wts));  // [1, DH, n]
                cc = ggml_cont(gc, ggml_permute(gc, cc, 1, 0, 2, 3));             // [DH, 1, n]
                cc = b.rms_w(cc, b.BL(il, "attn_compressor_norm.weight"));
                comp_k = b.rope_at(cc, comp_pos, DHR, DH - DHR, n_ctx, crb,
                                   freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cmask = ggml_cast(gc, var.vis, GGML_TYPE_F16);
        }

        ggml_tensor * comp2 = comp_k ? ggml_set_rows(gc, L.comp, ggml_cont_2d(gc, comp_k, DH, n), first(L.i_slot_comp, n))
                                     : L.comp;   // !emit: no block completes here
        ggml_tensor * comp_src = ggml_view_2d(gc, comp2, DH, cap, comp2->nb[1], 0);
        k_all = ggml_concat(gc, k_all, ggml_reshape_3d(gc, comp_src, DH, 1, cap), 2);
        mask = ggml_concat(gc, mask, cmask, 0);                              // [n_raw + cap, n]
    }

    // ---- attention --------------------------------------------------------------------
    // ggml-cuda has a flash-attn kernel for d_head 512 only when the key count is a multiple of FATTN_KQ_STRIDE
    // (256) - llama.cpp pads its KV cache to that.  Off-CPU, pad with zero keys masked -inf: they get exactly zero
    // weight.  The CPU path (the one verified bit-exact against ds4_ref) is left as it was.
    if (!ggml_backend_is_cpu(im.backend)) {
        const int64_t n_kv = k_all->ne[2], pad = (256 - n_kv % 256) % 256;
        if (pad) {
            k_all = ggml_pad(gc, k_all, 0, 0, (int) pad, 0);
            mask = ggml_concat(gc, mask, ggml_cast(gc, b.fill_f32({ pad, n }, NEG_INF), GGML_TYPE_F16), 0);
        }
    }
    ggml_tensor * qp = ggml_permute(gc, q, 0, 2, 1, 3);                      // [DH, n, NH]
    ggml_tensor * kp = ggml_permute(gc, k_all, 0, 2, 1, 3);
    ggml_tensor * kf = ggml_cast(gc, kp, GGML_TYPE_F16);
    ggml_tensor * vf = ggml_cast(gc, kp, GGML_TYPE_F16);
    ggml_tensor * out = ggml_flash_attn_ext(gc, qp, kf, vf, mask, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, b.BL(il, "attn_sinks.weight"));     // out [DH, NH, n]

    // the gate's attn_raw tap: ds4_ref dumps the raw flash-attn output *before* the de-RoPE
    // (its attn_csa_lid / attn_hca probe is named at the flash-attn result, not after rope_back)
    ggml_tensor * attn_raw = ggml_cont_2d(gc, out, NH * DH, n);

    // de-RoPE then grouped output LoRA
    out = ggml_reshape_3d(gc, out, DH, NH, n);
    out = b.rope_back(out, pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    out = ggml_reshape_3d(gc, out, OGD, OG, n);
    out = ggml_permute(gc, out, 0, 2, 1, 3);                                 // [OGD, n, OG]
    ggml_tensor * wo_a = b.BL(il, "attn_output_a.weight");
    if (ggml_n_dims(wo_a) == 2) wo_a = ggml_reshape_3d(gc, wo_a, OGD, OL, OG);
    ggml_tensor * oa = ggml_mul_mat(gc, wo_a, out);                          // [OL, n, OG]
    oa = ggml_permute(gc, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(gc, oa, OL * OG, n);
    ggml_tensor * attn_out = ggml_mul_mat(gc, b.BL(il, "attn_output_b.weight"), oa);   // [D, n]

    // ---- feed-forward hyper-connection -------------------------------------------------
    ggml_tensor * hap = b.hc_post(attn_out, xin, post_a, comb_a);
    ggml_tensor * post_f = nullptr, * comb_f = nullptr;
    ggml_tensor * ffn_pre = b.hc_pre(hap, b.BL(il, "hc_ffn_fn.weight"), b.BL(il, "hc_ffn_scale.weight"),
                                     b.BL(il, "hc_ffn_base.weight"), &post_f, &comb_f);
    ggml_tensor * fn = b.rms_w(ffn_pre, b.BL(il, "ffn_norm.weight"));

    // ---- MoE router (ds4_ref main():904-928) -------------------------------------------
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(gc, ggml_softplus(gc, rlog));             // [NEXP, n]

    ggml_tensor * selected = nullptr;
    if (il < (int) im.g.hash_layer_count) {
        selected = ggml_get_rows(gc, b.BL(il, "ffn_gate_tid2eid.weight"), first(im.i_tid, n));
    } else {
        ggml_tensor * sel = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
        // cache-aware routing (set_route_bias): a selection-only bias like exp_probs_b; the mixing weights below still
        // come from the unbiased probs.  All zeros (the default) = the model's own routing, bit for bit.
        sel = ggml_add(gc, sel, ggml_reshape_2d(gc, L.rbias, im.NEXP, 1));
        selected = ggml_argsort_top_k(gc, sel, (int) im.NUSED);
    }
    ggml_set_output(selected);                                               // [NUSED, n]
    ggml_tensor * probs3 = ggml_reshape_3d(gc, probs, 1, im.NEXP, n);
    ggml_tensor * wts = ggml_get_rows(gc, probs3, selected);              // [1, topk, n]
    wts = ggml_reshape_2d(gc, wts, im.NUSED, n);
    ggml_tensor * wsum = ggml_add(gc, ggml_sum_rows(gc, wts), b.fill_f32({ 1, 1 }, 0.0f));
    wsum = ggml_clamp(gc, wsum, 6.103515625e-5f, INFINITY);
    wts = ggml_div(gc, wts, wsum);
    wts = ggml_reshape_3d(gc, wts, 1, im.NUSED, n);
    wts = ggml_scale(gc, wts, (float) im.g.weights_scale);
    ggml_set_output(wts);

    // ---- shared expert (ds4_ref main():946-956) ----------------------------------------
    ggml_tensor * up_s = ggml_mul_mat(gc, b.BL(il, "ffn_up_shexp.weight"), fn);
    ggml_tensor * gate_s = ggml_mul_mat(gc, b.BL(il, "ffn_gate_shexp.weight"), fn);
    const float lim = (float) im.g.swiglu_clamp_shexp[(size_t) il];
    up_s = ggml_clamp(gc, up_s, -lim, lim);
    gate_s = ggml_clamp(gc, gate_s, NEG_INF, lim);
    ggml_tensor * z = ggml_swiglu_split(gc, gate_s, up_s);
    ggml_tensor * shexp = ggml_mul_mat(gc, b.BL(il, "ffn_down_shexp.weight"), z);

    // ---- persisted hand-offs (finish_layer + the gate's taps), first n columns ----------
    auto cols2 = [&](ggml_tensor * t, int64_t n0) { return ggml_view_2d(gc, t, n0, n, t->nb[1], 0); };
    auto cols3 = [&](ggml_tensor * t, int64_t n0, int64_t n1) {
        return ggml_view_3d(gc, t, n0, n1, n, t->nb[1], t->nb[2], 0);
    };
    const size_t ob = (size_t) im.o_blk;
    ggml_tensor * c_fn  = ggml_cpy(gc, ggml_reshape_2d(gc, fn, D, n), ggml_view_2d(gc, L.o_f, D, n, ob, 0));
    ggml_tensor * c_hap = ggml_cpy(gc, ggml_reshape_3d(gc, hap, D, HC, n), cols3(L.hap, D, HC));
    ggml_tensor * c_pf  = ggml_cpy(gc, post_f, cols2(L.post_f, HC));   // [HC, n], a strided view of mixes
    ggml_tensor * c_cf  = ggml_cpy(gc, ggml_reshape_3d(gc, comb_f, HC, HC, n), cols3(L.comb_f, HC, HC));
    ggml_tensor * c_sh  = ggml_cpy(gc, ggml_reshape_2d(gc, shexp, D, n), cols2(L.shexp, D));
    ggml_tensor * c_ar  = im.pf ? nullptr : ggml_cpy(gc, attn_raw, cols2(L.t_attn_raw, NH * DH));
    ggml_tensor * c_at  = im.pf ? nullptr : ggml_cpy(gc, ggml_reshape_2d(gc, attn_out, D, n), cols2(L.t_attn, D));

    for (ggml_tensor * t : { c_fn, c_hap, c_pf, c_cf, c_sh }) ggml_build_forward_expand(gf, t);
    if (!im.pf) for (ggml_tensor * t : { c_ar, c_at }) ggml_build_forward_expand(gf, t);   // gate taps (decode only)
    ggml_build_forward_expand(gf, selected);
    ggml_build_forward_expand(gf, wts);
    ggml_build_forward_expand(gf, ggml_cpy(gc, selected,   // [NUSED, n], a top-k view: strided for n > 1
                                           ggml_view_2d(gc, L.o_i, im.NUSED, n, ob, (size_t) im.o_ids_off)));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_2d(gc, wts, im.NUSED, n),
                                           ggml_view_2d(gc, L.o_f, im.NUSED, n, ob, (size_t) im.o_wts_off)));

    for (ggml_tensor * t : wb) ggml_build_forward_expand(gf, t);   // chunk ring write-backs, after every read
    if (im.pf) return gf;   // a chunk graph: run once by the caller, not registered
    var.gf = gf;
    var.allo = im.make_allo();   // one arena per graph variant; see Impl::make_allo
    L.vars.push_back(var);
    return gf;
}

// finish_layer: ffn_out = routed_sum + shexp ; l_last = hc_post(ffn_out, hap, post_f, comb_f), per token
static ggml_cgraph * build_finish(Ds4Dense::Impl & im, int64_t n, Ds4Dense::Impl::Layer & L) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 512, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * shex2 = ggml_view_2d(gc, L.shexp, D, n, L.shexp->nb[1], 0);
    ggml_tensor * rs    = ggml_view_2d(gc, im.routed_sum, D, n, im.routed_sum->nb[1], 0);
    ggml_tensor * ffn_out = ggml_add(gc, shex2, rs);

    ggml_tensor * hap3 = ggml_view_3d(gc, L.hap, D, HC, n, L.hap->nb[1], L.hap->nb[2], 0);
    ggml_tensor * pf2  = ggml_view_2d(gc, L.post_f, HC, n, L.post_f->nb[1], 0);
    ggml_tensor * cf3  = ggml_view_3d(gc, L.comb_f, HC, HC, n, L.comb_f->nb[1], L.comb_f->nb[2], 0);
    ggml_tensor * llast = b.hc_post(ffn_out, hap3, pf2, cf3);
    ggml_tensor * ll3 = ggml_reshape_3d(gc, llast, D, HC, n);

    ggml_build_forward_expand(gf, ggml_cpy(gc, ll3, ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1],
                                                                 im.x_state->nb[2], 0)));
    if (im.pf) return gf;   // a chunk graph: no l_last tap, not registered
    ggml_build_forward_expand(gf, ggml_cpy(gc, ll3, ggml_view_3d(gc, L.t_llast, D, HC, n, L.t_llast->nb[1],
                                                                 L.t_llast->nb[2], 0)));
    L.gf_finish[n] = gf;
    return gf;
}

// predict: the layer's router on rms_norm(hc_attn_pre) * ffn_norm.weight  (prefetch hint "A"), token 0 of the pass
static ggml_cgraph * build_predict(Ds4Dense::Impl & im, int il, Ds4Dense::Impl::Layer & L) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 512, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, 1, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * attn_pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                      b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * fnp = b.rms_w(attn_pre, b.BL(il, "ffn_norm.weight"));
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fnp);
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(gc, ggml_softplus(gc, rlog));

    ggml_tensor * out = nullptr;
    if (il < (int) im.g.hash_layer_count) {
        out = ggml_cast(gc, ggml_get_rows(gc, b.BL(il, "ffn_gate_tid2eid.weight"), ggml_view_1d(gc, im.i_tid, 1, 0)),
                        GGML_TYPE_F32);
    } else {
        out = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    L.predict_out = out;
    return gf;
}

// begin_tokens: hc_init = repeat(embd[tid], hc), per token
static ggml_cgraph * build_init(Ds4Dense::Impl & im, int64_t n) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_view_3d(gc, im.i_emb, D, 1, n, im.i_emb->nb[1], im.i_emb->nb[1], 0);   // embd rows
    x = ggml_repeat_4d(gc, x, D, HC, n, 1);
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1],
                                                               im.x_state->nb[2], 0)));
    return gf;
}

// logits: hc_head -> output_norm -> output, per token.  `mtp`: the MTP head - its own hc_head (`mtp.output_hc_*`, from
// the MTP file: llama.cpp loads that file as its own model, so graph_mtp's hc_head_* are the file's) and
// shared_head_norm, the trunk's output (the file's is a BF16 copy of the original V4's; ours is 0731's).
// DS4_MTP_TRUNK_HC=1: the trunk's hc_head instead (the pre-2026-10-07 behaviour, for A/B and mutation checks)
static ggml_cgraph * build_head(Ds4Dense::Impl & im, int64_t n, bool mtp = false) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * xin = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2],
                                     im.pf ? (size_t) im.pf_row0 * im.x_state->nb[2] : 0);   // chunk: rows row0..
    ggml_tensor * flat = ggml_reshape_2d(gc, xin, HC * D, n);
    ggml_tensor * flat_norm = ggml_rms_norm(gc, flat, (float) im.g.rms_eps);
    static const bool trunk_hc = [] { const char * e = std::getenv("DS4_MTP_TRUNK_HC"); return e && *e && *e != '0'; }();
    const std::string hp = mtp && !trunk_hc ? "mtp." : "";
    ggml_tensor * mixes = ggml_mul_mat(gc, im.w.get(hp + "output_hc_fn.weight"), flat_norm);
    ggml_tensor * scale = im.w.get(hp + "output_hc_scale.weight");
    ggml_tensor * base  = im.w.get(hp + "output_hc_base.weight");
    ggml_tensor * pre = ggml_sigmoid(gc, b.hc_affine(mixes, b.v1(scale, 1, 0), b.v1(base, HC, 0)));
    pre = ggml_scale_bias(gc, pre, 1.0f, (float) im.g.hc_eps);
    ggml_tensor * hc_head = b.hc_mean(xin, pre);
    ggml_tensor * rn = b.rms_w(hc_head, mtp ? b.BL((int) im.il_mtp, "nextn.shared_head_norm.weight")
                                            : im.w.get("output_norm.weight"));
    ggml_tensor * lg = ggml_mul_mat(gc, im.w.get("output.weight"), rn);
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);
    if (im.pf) im.pf_logits = lg;
    else (mtp ? im.mtp_logits_t : im.logits_t)[n] = lg;
    return gf;
}

// MTP input: x_state <- eh_proj(concat(enorm(embd[next]) repeated over the hc streams, hnorm(x_state))), per token.
// x_state holds the trunk's final hc streams (after the last finish_layer); i_emb the next tokens' embeddings.
static ggml_cgraph * build_mtp_in(Ds4Dense::Impl & im, int64_t n) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;
    const int il = (int) im.il_mtp;
    ggml_tensor * xs = ggml_view_3d(gc, im.x_state, D, HC, n, im.x_state->nb[1], im.x_state->nb[2], 0);
    ggml_tensor * emb = ggml_view_2d(gc, im.i_emb, D, n, im.i_emb->nb[1], 0);
    ggml_tensor * en = b.rms_w(emb, b.BL(il, "nextn.enorm.weight"));
    en = ggml_repeat_4d(gc, ggml_reshape_3d(gc, en, D, 1, n), D, HC, n, 1);
    ggml_tensor * hn = b.rms_w(xs, b.BL(il, "nextn.hnorm.weight"));
    ggml_tensor * x = ggml_mul_mat(gc, b.BL(il, "nextn.eh_proj.weight"), ggml_concat(gc, en, hn, 0));   // [D, HC, n]
    ggml_build_forward_expand(gf, ggml_cpy(gc, x, xs));
    return gf;
}

// ================================================================== public API

Ds4Dense::Ds4Dense() : p_(new Impl()) {}
Ds4Dense::~Ds4Dense() = default;

bool Ds4Dense::init(const std::string & model_path, const Ds4DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.path = model_path;
    im.n_threads = cfg.n_threads;
    im.pf_cap = std::max<int64_t>(0, cfg.prefill_chunk);

    im.backend = cfg.backend;
    if (!im.backend) {
        im.backend = ggml_backend_cpu_init();
        im.own_backend = true;
        if (!im.backend) { err = "cannot create a CPU backend"; return false; }
    }
    if (ggml_backend_is_cpu(im.backend)) ggml_backend_cpu_set_n_threads(im.backend, im.n_threads);
    // DS4_CPU_FA_REF=1: ggml-cpu's reference flash-attn for any query count (its tiled kernel takes >= 64 queries and
    // rounds differently) - the chunk gate uses it to show 64+-token chunks are the decode math exactly
    if (ggml_backend_is_cpu(im.backend) && std::getenv("DS4_CPU_FA_REF")) ggml_backend_cpu_set_use_ref(im.backend, true);

    im.model = std::make_unique<GgufModel>(GgufModel::open(model_path));
    if (im.model->size() == 0) { err = "cannot open " + model_path; return false; }
    if (!(err = ds4_read_geometry(*im.model, im.g)).empty()) return false;
    if (!(err = check_ds4_model(*im.model, im.g)).empty()) return false;
    if (!load_weights(*im.model, im.backend, im.w, cfg.skip_routed_experts,
                      ggml_backend_is_cpu(im.backend) ? -1 : cfg.requant_type, err)) return false;
    im.n_trunk = im.g.n_layer();
    if (!cfg.mtp_path.empty()) {   // the MTP head: its own GGUF, block blk.<n_trunk>
        im.mtp_model = std::make_unique<GgufModel>(GgufModel::open(cfg.mtp_path));
        if (im.mtp_model->size() == 0) { err = "cannot open " + cfg.mtp_path; return false; }
        if (!(err = ds4_attach_mtp(*im.mtp_model, im.g, im.il_mtp)).empty()) return false;
        if (!load_weights(*im.mtp_model, im.backend, im.w, cfg.skip_routed_experts,
                          ggml_backend_is_cpu(im.backend) ? -1 : cfg.requant_type, err,
                          "blk." + std::to_string(im.il_mtp) + ".", ds4_mtp_head_alias())) return false;
    }
    {
        size_t sh = 0;
        const TensorInfo * te = im.model->find("token_embd.weight", &sh);
        if (!te) { err = "no token_embd.weight"; return false; }
        im.embd_data = im.model->shard(sh).tensor_data(*te);
        im.embd_type = (int) te->type;
        im.embd_row  = ggml_row_size((ggml_type) te->type, (int64_t) te->shape.at(0));
        if (!ggml_get_type_traits((ggml_type) te->type)->to_float && te->type != GGML_TYPE_F32) {
            err = "token_embd type has no to_float"; return false;
        }
    }

    const Ds4Geometry & g = im.g;
    im.D = g.n_embd; im.HC = g.hc; im.DH = g.d_head; im.DHR = g.d_rope; im.NH = g.n_head;
    im.OG = g.o_groups; im.OL = g.o_lora; im.OGD = g.o_group_dim(); im.NHPG = g.n_head / g.o_groups;
    im.RQ = g.r_q; im.SWA = g.n_swa; im.RAW = g.n_swa + kNtMax - 1;
    im.IDXH = g.indexer_n_head; im.IDXK = g.indexer_key_dim; im.IDXTOPK = g.indexer_top_k;
    im.NEXP = g.n_expert; im.NUSED = g.n_expert_used; im.NFF = g.n_ff; im.NSHEXP = g.n_expert_shared;

    const int64_t nl = im.n_trunk + (im.il_mtp >= 0 ? 1 : 0);   // + the MTP block
    if (nl <= 0 || im.D <= 0 || im.HC <= 0 || im.SWA <= 0) { err = "degenerate geometry"; return false; }
    if (im.NUSED <= 0) { err = "expert_used_count is zero"; return false; }

    bool any_csa = false;
    // ---- persistent decode state -------------------------------------------------------
    {
        ggml_init_params ip = { /*mem_size*/ 128ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        im.sctx = ggml_init(ip);
        if (!im.sctx) { err = "ggml_init(state) failed"; return false; }
        ggml_set_no_alloc(im.sctx, true);

        auto nt3 = [&](ggml_type ty, int64_t n0, int64_t n1, int64_t n2) { return ggml_new_tensor_3d(im.sctx, ty, n0, n1, n2); };
        auto nt2 = [&](ggml_type ty, int64_t n0, int64_t n1) { return ggml_new_tensor_2d(im.sctx, ty, n0, n1); };
        auto nt1 = [&](ggml_type ty, int64_t n0) { return ggml_new_tensor_1d(im.sctx, ty, n0); };
        ggml_init_params ipi = { /*mem_size*/ 16ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        im.ictx = ggml_init(ipi);
        if (!im.ictx) { err = "ggml_init(inputs) failed"; return false; }
        std::vector<ggml_tensor *> in_list;                    // placed back to back in ibuf
        auto in1 = [&](ggml_type ty, int64_t n0) {
            ggml_tensor * t = ggml_new_tensor_1d(im.ictx, ty, n0);
            in_list.push_back(t);
            return t;
        };
        auto on1 = [&](ggml_type ty, int64_t n0) { return ggml_new_tensor_1d(im.ictx, ty, n0); };
        const int64_t NT = kNtMax;

        im.x_state    = nt3(GGML_TYPE_F32, im.D, im.HC, NT);
        im.routed_sum = nt2(GGML_TYPE_F32, im.D, NT);
        im.i_tid      = nt1(GGML_TYPE_I32, NT);
        im.i_emb      = nt2(GGML_TYPE_F32, im.D, NT);
        im.i_pos      = in1(GGML_TYPE_I32, NT);

        im.ly.resize((size_t) nl);
        for (int64_t il = 0; il < nl; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            L.ratio = (size_t) il < g.compress_ratios.size() ? g.compress_ratios[(size_t) il] : 0;
            L.csa = (L.ratio == 4);
            if (L.ratio != 0 && L.ratio != 4 && L.ratio != 128) { err = "unsupported compress ratio"; return false; }
            any_csa = any_csa || L.csa;
            L.ring = L.csa ? 2 * L.ratio : L.ratio;
            L.ring_sz = L.ring + kNtMax - 1;
            L.sdim = L.csa ? 2 * im.DH : im.DH;
            const int64_t by_ctx = L.ratio != 0 ? std::max<int64_t>(1, g.context_length / L.ratio) : 0;
            L.comp_max = cfg.comp_cap_max > 0 ? std::min<int64_t>(cfg.comp_cap_max, by_ctx) : by_ctx;

            L.raw     = nt2(GGML_TYPE_F32, im.DH, im.RAW);
            L.hap     = nt3(GGML_TYPE_F32, im.D, im.HC, NT);
            L.post_f  = nt2(GGML_TYPE_F32, im.HC, NT);
            L.comb_f  = nt3(GGML_TYPE_F32, im.HC, im.HC, NT);
            L.shexp   = nt2(GGML_TYPE_F32, im.D, NT);
            L.fn      = on1(GGML_TYPE_F32, im.D);
            L.o_ids   = on1(GGML_TYPE_I32, im.NUSED);
            L.o_wts   = on1(GGML_TYPE_F32, im.NUSED);
            L.t_attn_raw = nt2(GGML_TYPE_F32, im.NH * im.DH, NT);
            L.t_attn  = nt2(GGML_TYPE_F32, im.D, NT);
            L.t_llast = nt3(GGML_TYPE_F32, im.D, im.HC, NT);

            L.i_slot_raw  = in1(GGML_TYPE_I32, NT);
            L.i_idx_raw   = in1(GGML_TYPE_I32, im.SWA + NT - 1);
            L.i_mask_raw  = in1(GGML_TYPE_F16, (im.SWA + NT - 1) * NT);
            L.i_slot_comp = in1(GGML_TYPE_I32, NT);
            L.i_slot_state = in1(GGML_TYPE_I32, NT);
            L.i_comp_pos  = in1(GGML_TYPE_I32, NT);
            L.i_state_pos = in1(GGML_TYPE_I32, NT);
            L.i_idx_state = in1(GGML_TYPE_I32, std::max<int64_t>(1, L.ring) * NT);
            if (L.ratio != 0) L.vis_full = in1(GGML_TYPE_F32, L.comp_max * NT);
            L.rbias = in1(GGML_TYPE_F32, im.NEXP);

            if (L.ratio != 0) {
                L.comp  = nt2(GGML_TYPE_F32, im.DH, L.comp_max + 1);
                L.st_kv = nt2(GGML_TYPE_F32, L.sdim, L.ring_sz);
                L.st_sc = nt2(GGML_TYPE_F32, L.sdim, L.ring_sz);
                if (L.csa) {
                    L.icomp  = nt2(GGML_TYPE_F32, im.IDXK, L.comp_max + 1);
                    L.ist_kv = nt2(GGML_TYPE_F32, 2 * im.IDXK, L.ring_sz);   // CSA: ring == 2 * ratio
                    L.ist_sc = nt2(GGML_TYPE_F32, 2 * im.IDXK, L.ring_sz);
                }
            }
            // every layer (ratio-0 sliding-window layers too): ffn_norm is the expert tier's input
            L.host_fn.resize((size_t) (im.D * NT));
            if (cfg.gate_taps) {   // the gate's taps: ~10 MB of device->host copies per token, off by default
                L.host_attn_raw.resize((size_t) (im.NH * im.DH * NT));
                L.host_attn.resize((size_t) (im.D * NT));
                L.host_llast.resize((size_t) (im.D * im.HC * NT));
            }
        }
        if (any_csa) im.rot = nt2(GGML_TYPE_F32, im.IDXK, im.IDXK);

        im.sbuf = ggml_backend_alloc_ctx_tensors_from_buft(im.sctx, ggml_backend_get_default_buffer_type(im.backend));
        if (!im.sbuf) { err = "cannot allocate the persistent decode state"; return false; }
        ggml_backend_buffer_clear(im.sbuf, 0);

        // ---- the input span and the per-layer output blocks: tensors placed at fixed offsets ----
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(im.backend);
        const size_t al = std::max<size_t>(64, ggml_backend_buft_get_alignment(buft));
        auto up = [&](size_t n) { return (n + al - 1) / al * al; };
        size_t in_bytes = 0;
        for (ggml_tensor * t : in_list) in_bytes += up(ggml_nbytes(t));
        im.ibuf = ggml_backend_buft_alloc_buffer(buft, in_bytes + al);
        if (!im.ibuf) { err = "cannot allocate the input span"; return false; }
        im.in_base = (uint8_t *) ggml_backend_buffer_get_base(im.ibuf);
        {
            size_t off = 0;
            for (ggml_tensor * t : in_list) {
                if (ggml_backend_tensor_alloc(im.ibuf, t, im.in_base + off) != GGML_STATUS_SUCCESS) {
                    err = "cannot place an input tensor"; return false;
                }
                off += up(ggml_nbytes(t));
            }
            im.i_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) in_bytes);
            if (ggml_backend_tensor_alloc(im.ibuf, im.i_span, im.in_base) != GGML_STATUS_SUCCESS) {
                err = "cannot place the input span"; return false;
            }
            im.in_host.assign(in_bytes, 0);
        }
        // router outputs: per layer kNtMax token blocks [fn | ids | wts] of o_blk bytes; a pass of n tokens reads back
        // the first n blocks of its layer in one tensor_get
        const size_t o1 = up(ggml_row_size(GGML_TYPE_F32, im.D)), o2 = o1 + up(4 * im.NUSED);
        const size_t blk = o2 + up(4 * im.NUSED);
        im.o_blk = (int64_t) blk; im.o_ids_off = (int64_t) o1; im.o_wts_off = (int64_t) o2;
        const size_t lblk = blk * (size_t) NT;
        im.obuf = ggml_backend_buft_alloc_buffer(buft, lblk * (size_t) nl + al);
        if (!im.obuf) { err = "cannot allocate the router output blocks"; return false; }
        {
            uint8_t * ob = (uint8_t *) ggml_backend_buffer_get_base(im.obuf);
            for (int64_t il = 0; il < nl; ++il) {
                Impl::Layer & L = im.ly[(size_t) il];
                uint8_t * b0 = ob + (size_t) il * lblk;
                L.o_f    = ggml_new_tensor_1d(im.ictx, GGML_TYPE_F32, (int64_t) (lblk / 4));
                L.o_i    = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I32, (int64_t) (lblk / 4));
                L.o_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) lblk);
                for (auto [t, o] : { std::pair<ggml_tensor *, size_t>{ L.fn, 0 }, { L.o_ids, o1 }, { L.o_wts, o2 },
                                     { L.o_f, 0 }, { L.o_i, 0 }, { L.o_span, 0 } })
                    if (ggml_backend_tensor_alloc(im.obuf, t, b0 + o) != GGML_STATUS_SUCCESS) {
                        err = "cannot place a router output block"; return false;
                    }
                L.host_out.assign(lblk, 0);
            }
        }

        // compressor score rings start at SENTINEL: a token that was never written gets weight 0, exactly
        // like ds4_ref's zero/-inf row appended for the first block's missing previous half
        std::vector<float> sent;
        for (Impl::Layer & L : im.ly) {
            if (L.ratio == 0) continue;
            sent.assign((size_t) (L.sdim * L.ring_sz), SENTINEL);
            ggml_backend_tensor_set(L.st_sc, sent.data(), 0, sent.size() * 4);
            if (L.csa) {
                sent.assign((size_t) (2 * im.IDXK * L.ring_sz), SENTINEL);
                ggml_backend_tensor_set(L.ist_sc, sent.data(), 0, sent.size() * 4);
            }
        }
        if (im.rot) {
            const int n = (int) im.IDXK;
            std::vector<float> rd((size_t) n * n, 0.0f);
            rd[0] = 1.0f / std::sqrt((float) n);
            for (int s = 1; s < n; s *= 2)
                for (int i = 0; i < s; ++i)
                    for (int j = 0; j < s; ++j) {
                        const float v = rd[(size_t) i * n + j];
                        rd[(size_t) (i + s) * n + j] = v;
                        rd[(size_t) i * n + (j + s)] = v;
                        rd[(size_t) (i + s) * n + (j + s)] = -v;
                    }
            ggml_backend_tensor_set(im.rot, rd.data(), 0, rd.size() * 4);
        }
    }

    // ---- graph scaffolding -------------------------------------------------------------
    {
        ggml_init_params ip = { /*mem_size*/ 256ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        im.gctx = ggml_init(ip);
        if (!im.gctx) { err = "ggml_init(graph) failed"; return false; }
        ggml_set_no_alloc(im.gctx, true);

        if (!im.ensure_n(1)) { err = im.err; return false; }
        for (int64_t il = 0; il < nl; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            L.gf_predict = build_predict(im, (int) il, L);
            L.allo_predict = im.make_allo();
            if (!L.allo_predict) { err = "ggml_gallocr_new failed"; return false; }
            build_attn(im, (int) il, 0, 1, L);   // cap 0; larger capacities (and n > 1) are built on demand
        }
    }
    im.host_logits.resize((size_t) (g.vocab_size * kNtMax));
    return true;
}

// init / head / finish graphs for passes of n tokens, built on first use
bool Ds4Dense::Impl::ensure_n(int64_t n) {
    if (n < 1 || n > kNtMax) { err = "pass size out of range (1.." + std::to_string(kNtMax) + ")"; return false; }
    if (!gf_init[n]) {
        gf_init[n] = build_init(*this, n);
        allo_init[n] = make_allo();
        gf_head[n] = build_head(*this, n);
        allo_head[n] = make_allo();
        if (!allo_init[n] || !allo_head[n]) { err = "ggml_gallocr_new failed"; return false; }
    }
    if (il_mtp >= 0 && !gf_mtp_in[n]) {
        gf_mtp_in[n] = build_mtp_in(*this, n);
        allo_mtp_in[n] = make_allo();
        gf_mtp_head[n] = build_head(*this, n, true);
        allo_mtp_head[n] = make_allo();
        if (!allo_mtp_in[n] || !allo_mtp_head[n]) { err = "ggml_gallocr_new failed"; return false; }
    }
    for (Layer & L : ly) {
        if (L.gf_finish[n]) continue;
        build_finish(*this, n, L);
        L.allo_finish[n] = make_allo();
        if (!L.allo_finish[n]) { err = "ggml_gallocr_new failed"; return false; }
    }
    return true;
}

// the attention variant for (cap, n), built on first use
static Ds4Dense::Impl::Var * attn_var(Ds4Dense::Impl & im, int il, int64_t cap, int64_t n, bool emit = true) {
    Ds4Dense::Impl::Layer & L = im.ly[(size_t) il];
    emit = emit || n != 1 || cap == 0;
    for (Ds4Dense::Impl::Var & v : L.vars)
        if (v.cap == cap && v.n == n && v.emit == emit) return &v;
    build_attn(im, il, cap, n, L, emit);
    return &L.vars.back();
}

// DS4_COMP_SKIP (default on): a one-token pass that completes no compressed block skips the pooled row (FINDINGS s21)
static bool comp_skip_on() {
    static const bool on = [] { const char * e = std::getenv("DS4_COMP_SKIP"); return !e || std::atoi(e) != 0; }();
    return on;
}

// compressed-row capacity a pass whose last token sits at `pos_last` needs (0 for a ratio-0 layer)
static int64_t attn_cap(const Ds4Dense::Impl::Layer & L, int64_t pos_last) {
    if (L.ratio == 0) return 0;
    return std::min<int64_t>(L.comp_max, next_pow2(pos_last / L.ratio + 1));
}

const strata::Ds4Geometry & Ds4Dense::geom()    const { return p_->g; }
int64_t                     Ds4Dense::n_layer() const { return p_->n_trunk; }
int                         Ds4Dense::mtp_layer() const { return (int) p_->il_mtp; }
int                         Ds4Dense::max_tokens() const { return (int) kNtMax; }
ggml_backend_t              Ds4Dense::backend() const { return p_->backend; }
const std::string &         Ds4Dense::last_error() const { return p_->err; }
ggml_tensor *               Ds4Dense::weight(const std::string & name) const { return p_->w.get(name); }

bool Ds4Dense::begin_token(int tid) { return begin_tokens(&tid, 1); }

bool Ds4Dense::begin_tokens(const int * tids, int n) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_init[n], im.gf_init[n])) { im.err = "gallocr(init) failed"; return false; }
    std::vector<int32_t> t32((size_t) n);
    // embd[tid] on the host, exactly what get_rows + cast computes (F16 -> F32 is exact; quantized rows dequantize
    // with the same to_float)
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        const int tid = tids[t];
        if (tid < 0 || tid >= im.g.vocab_size) { im.err = "token id out of range"; return false; }
        t32[(size_t) t] = tid;
        const uint8_t * row = im.embd_data + (size_t) tid * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_tensor_set(im.i_tid, t32.data(), 0, t32.size() * 4);
    im.cur_tid = n == 1 ? tids[0] : -1;
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (ggml_backend_graph_compute(im.backend, im.gf_init[n]) != GGML_STATUS_SUCCESS) {
        im.err = "init graph compute failed";
        return false;
    }
    return true;
}

bool Ds4Dense::predict(int il, int * top_ids, int n_top) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || n_top <= 0) { im.err = "predict: bad layer/args"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    for (int i = 0; i < n_top; ++i) top_ids[i] = -1;

    if (!alloc_graph(L.allo_predict, L.gf_predict)) { im.err = "gallocr(predict) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, L.gf_predict) != GGML_STATUS_SUCCESS) {
        im.err = "predict graph compute failed";
        return false;
    }
    ggml_tensor * out = L.predict_out;
    if (il < (int) im.g.hash_layer_count) {
        std::vector<int32_t> ids((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, ids.data(), 0, ids.size() * 4);
        const int n = std::min<int>(n_top, (int) ids.size());
        for (int i = 0; i < n; ++i) top_ids[i] = ids[(size_t) i];
        return true;
    }
    std::vector<float> sel((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, sel.data(), 0, sel.size() * 4);
    std::vector<int> idx(sel.size());
    for (size_t i = 0; i < sel.size(); ++i) idx[i] = (int) i;
    const int n = std::min<int>(n_top, (int) sel.size());
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
    for (int i = 0; i < n; ++i) top_ids[i] = idx[(size_t) i];
    return true;
}

bool Ds4Dense::attn_router(int il, int pos, int tid, int * routed_ids, float * routed_w,
                           const float ** ffn_norm_host, ggml_tensor ** ffn_norm_dev) {
    Impl & im = *p_;
    if (tid >= 0 && tid != im.cur_tid) {
        const int32_t t32 = tid;
        ggml_backend_tensor_set(im.i_tid, &t32, 0, sizeof t32);
        im.cur_tid = tid;
    }
    if (!attn_router_n(il, pos, 1, routed_ids, routed_w, ffn_norm_host)) return false;
    if (ffn_norm_dev) *ffn_norm_dev = im.ly[(size_t) il].fn;
    return true;
}

bool Ds4Dense::attn_router_n(int il, int pos0, int n, int * routed_ids, float * routed_w,
                             const float ** ffn_norm_host) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "attn_router: bad layer"; return false; }
    if (n < 1 || n > kNtMax) { im.err = "attn_router: pass size out of range"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    if (pos0 < 0) { im.err = "attn_router: negative position"; return false; }
    const int64_t pos_last = (int64_t) pos0 + n - 1;

    // pick the graph variant for the compressed-row capacity the pass needs, and allocate its arena
    if (L.ratio != 0 && pos_last / L.ratio >= L.comp_max) {
        im.err = "attn_router: position beyond the compressed cache capacity";
        return false;
    }
    const bool emit = !(comp_skip_on() && n == 1 && L.ratio != 0 && (pos_last + 1) % L.ratio != 0);
    Impl::Var * var = attn_var(im, il, attn_cap(L, pos_last), n, emit);
    if (!alloc_graph(var->allo, var->gf)) { im.err = "gallocr(attn) failed"; return false; }

    // ---- per-step inputs: every layer's, for this pass, in ONE upload per pass ------------------
    if (pos0 != im.in_pos || n != im.in_n) {
        auto put = [&](ggml_tensor * t, const void * src, size_t nbytes) {
            std::memcpy(im.in_host.data() + ((const uint8_t *) t->data - im.in_base), src, nbytes);
        };
        std::vector<int32_t> p32((size_t) n), s32((size_t) n);
        for (int t = 0; t < n; ++t) {
            p32[(size_t) t] = pos0 + t;
            s32[(size_t) t] = (int32_t) ((pos0 + t) % im.RAW);
        }
        put(im.i_pos, p32.data(), p32.size() * 4);
        // raw keys: the union of the n windows, oldest -> newest; query t sees [pos_t - SWA + 1, pos_t] and tok >= 0
        const int64_t n_raw = im.SWA + n - 1;
        std::vector<int32_t> idx((size_t) n_raw);
        std::vector<ggml_fp16_t> msk((size_t) (n_raw * n));
        const ggml_fp16_t h0 = ggml_fp32_to_fp16(0.0f), hinf = ggml_fp32_to_fp16(NEG_INF);
        for (int64_t i = 0; i < n_raw; ++i) {
            const int64_t tok = (int64_t) pos0 - im.SWA + 1 + i;
            idx[(size_t) i] = (int32_t) (((tok % im.RAW) + im.RAW) % im.RAW);
            for (int t = 0; t < n; ++t) {
                const int64_t pt = (int64_t) pos0 + t;
                msk[(size_t) (t * n_raw + i)] = tok >= 0 && tok <= pt && tok > pt - im.SWA ? h0 : hinf;
            }
        }
        std::vector<int32_t> sp(n), cpp(n), sc(n), ss(n), sidx;
        std::vector<float> vis;
        for (Impl::Layer & Ly : im.ly) {
            put(Ly.i_slot_raw, s32.data(), s32.size() * 4);
            put(Ly.i_idx_raw, idx.data(), idx.size() * 4);
            put(Ly.i_mask_raw, msk.data(), msk.size() * 2);
            if (Ly.ratio == 0) continue;
            if (pos_last / Ly.ratio >= Ly.comp_max) continue;   // attn_router refuses this layer above
            const int64_t cp_ = attn_cap(Ly, pos_last);
            sidx.resize((size_t) (Ly.ring * n));
            vis.assign((size_t) (cp_ * n), NEG_INF);
            for (int t = 0; t < n; ++t) {
                const int64_t pt = (int64_t) pos0 + t, bl = pt / Ly.ratio;
                sp[(size_t) t]  = (int32_t) (pt % Ly.ratio);      // compressor `ape` row
                cpp[(size_t) t] = (int32_t) (Ly.ratio * bl);      // block start, for the compressed-key RoPE
                // block slot in the compressed caches; a later token of the pass in the same block owns the row,
                // this one writes its partial row to the spare slot
                const bool later_same = t + 1 < n && (pt + 1) / Ly.ratio == bl;
                sc[(size_t) t]  = (int32_t) (later_same ? Ly.comp_max : bl);
                ss[(size_t) t]  = (int32_t) (pt % Ly.ring_sz);    // token slot in the compressor state ring
                for (int64_t i = 0; i < Ly.ring; ++i) {
                    const int64_t tok = pt - Ly.ring + 1 + i;
                    sidx[(size_t) (t * Ly.ring + i)] = (int32_t) (((tok % Ly.ring_sz) + Ly.ring_sz) % Ly.ring_sz);
                }
                const int64_t nv = (pt + 1) / Ly.ratio;           // complete blocks query t may see
                for (int64_t i = 0; i < nv && i < cp_; ++i) vis[(size_t) (t * cp_ + i)] = 0.0f;
            }
            put(Ly.i_state_pos, sp.data(), (size_t) n * 4);
            put(Ly.i_comp_pos, cpp.data(), (size_t) n * 4);
            put(Ly.i_slot_comp, sc.data(), (size_t) n * 4);
            put(Ly.i_slot_state, ss.data(), (size_t) n * 4);
            put(Ly.i_idx_state, sidx.data(), sidx.size() * 4);
            put(Ly.vis_full, vis.data(), vis.size() * 4);
        }
        ggml_backend_tensor_set(im.i_span, im.in_host.data(), 0, im.in_host.size());
        im.in_pos = pos0;
        im.in_n = n;
    }

    // DS4_DBG_GRAPH=1: each attention graph variant's node count and op histogram, once (A/B of graph shapes)
    static const bool dbg_graph = std::getenv("DS4_DBG_GRAPH") != nullptr;
    if (dbg_graph) {
        static std::map<const void *, int> seen;
        if (!seen.count(var->gf)) {
            seen[var->gf] = 1;
            std::map<std::string, int> hist;
            const int nn = ggml_graph_n_nodes(var->gf);
            for (int i = 0; i < nn; ++i) {
                ggml_tensor * t = ggml_graph_node(var->gf, i);
                std::string k = ggml_op_desc(t);
                hist[k]++;
            }
            std::fprintf(stderr, "[graph] attn layer %d pos %lld: %d nodes |", il, (long long) pos0, nn);
            for (auto & kv : hist) std::fprintf(stderr, " %s:%d", kv.first.c_str(), kv.second);
            std::fprintf(stderr, "\n");
        }
    }
    if (ggml_backend_graph_compute(im.backend, var->gf) != GGML_STATUS_SUCCESS) {
        im.err = "attn graph compute failed";
        return false;
    }

    static const bool dbg_idx = std::getenv("DS4_DBG_INDEXER") != nullptr;
    if (dbg_idx && var->dbg_isc) {
        std::vector<float> v((size_t) (var->cap * n));
        ggml_backend_tensor_get(var->dbg_isc, v.data(), 0, v.size() * 4);
        for (int t = 0; t < n; ++t) {
            std::fprintf(stderr, "[idx] L%d pos %d:", il, pos0 + t);
            for (int64_t i = 0; i < var->cap; ++i)
                if (std::isfinite(v[(size_t) (t * var->cap + i)])) std::fprintf(stderr, " %.6g", v[(size_t) (t * var->cap + i)]);
            std::fprintf(stderr, "\n");
        }
    }
    // ---- router outputs + hand-offs: n token blocks [fn | ids | wts], one readback ----------------
    {
        const size_t ob = (size_t) im.o_blk;
        ggml_backend_tensor_get(L.o_span, L.host_out.data(), 0, ob * (size_t) n);
        for (int t = 0; t < n; ++t) {
            const uint8_t * b0 = L.host_out.data() + ob * (size_t) t;
            const int32_t * ids = (const int32_t *) (b0 + im.o_ids_off);
            const float * wv = (const float *) (b0 + im.o_wts_off);
            for (int64_t i = 0; i < im.NUSED; ++i) {
                routed_ids[t * im.NUSED + i] = ids[i];
                routed_w[t * im.NUSED + i] = wv[i];
            }
            std::memcpy(L.host_fn.data() + (size_t) t * im.D, b0, (size_t) im.D * 4);
        }
    }
    if (!L.host_attn_raw.empty())
        ggml_backend_tensor_get(L.t_attn_raw, L.host_attn_raw.data(), 0, (size_t) (im.NH * im.DH * n) * 4);
    if (!L.host_attn.empty()) ggml_backend_tensor_get(L.t_attn, L.host_attn.data(), 0, (size_t) (im.D * n) * 4);

    if (ffn_norm_host) *ffn_norm_host = L.host_fn.data();
    L.tap_n = n;
    L.have_attn = true;
    return true;
}

bool Ds4Dense::reserve_graphs(int n_max) {
    Impl & im = *p_;
    n_max = std::max(1, std::min<int>(n_max, (int) kNtMax));
    for (int n = 1; n <= n_max; ++n)
        if (!im.ensure_n(n)) return false;
    for (int il = 0; il < (int) im.ly.size(); ++il) {
        Impl::Layer & L = im.ly[(size_t) il];
        std::vector<int64_t> caps;
        if (L.ratio == 0) caps.push_back(0);
        else {
            for (int64_t c = 1; c < L.comp_max; c *= 2) caps.push_back(c);
            caps.push_back(L.comp_max);
        }
        for (int n = 1; n <= n_max; ++n) {
            for (int64_t cap : caps) {
                Impl::Var * var = attn_var(im, il, cap, n);
                if (!ggml_gallocr_reserve(var->allo, var->gf)) { im.err = "reserve: attention graph"; return false; }
                if (n == 1 && cap > 0 && comp_skip_on()) {   // the no-block-completes variant too
                    var = attn_var(im, il, cap, n, false);
                    if (!ggml_gallocr_reserve(var->allo, var->gf)) { im.err = "reserve: attention graph"; return false; }
                }
            }
            if (!ggml_gallocr_reserve(L.allo_finish[n], L.gf_finish[n])) { im.err = "reserve: finish"; return false; }
        }
        if (L.gf_predict && !ggml_gallocr_reserve(L.allo_predict, L.gf_predict)) { im.err = "reserve: predict"; return false; }
    }
    for (int n = 1; n <= n_max; ++n) {
        if (!ggml_gallocr_reserve(im.allo_init[n], im.gf_init[n])) { im.err = "reserve: init"; return false; }
        if (!ggml_gallocr_reserve(im.allo_head[n], im.gf_head[n])) { im.err = "reserve: head"; return false; }
        if (im.il_mtp >= 0 && (!ggml_gallocr_reserve(im.allo_mtp_in[n], im.gf_mtp_in[n]) ||
                               !ggml_gallocr_reserve(im.allo_mtp_head[n], im.gf_mtp_head[n]))) {
            im.err = "reserve: mtp"; return false;
        }
    }
    if (Impl::shared_on() && im.shared_allo) {
        std::fprintf(stderr, "dense compute buffers: one shared arena %.1f MiB for every decode graph\n",
                     ggml_gallocr_get_buffer_size(im.shared_allo, 0) / 1048576.0);
    } else {   // where the dense half's compute-buffer VRAM goes: one gallocr arena per graph
        auto sz = [](ggml_gallocr_t a) { return a ? (double) ggml_gallocr_get_buffer_size(a, 0) : 0.0; };
        double attn = 0, attn_max = 0, fin = 0, pred = 0, other = 0;
        int64_t n_attn = 0;
        for (Impl::Layer & L : im.ly) {
            for (Impl::Var & v : L.vars) { const double b = sz(v.allo); attn += b; attn_max = std::max(attn_max, b); ++n_attn; }
            for (ggml_gallocr_t a : L.allo_finish) fin += sz(a);
            pred += sz(L.allo_predict);
        }
        for (int n = 0; n <= (int) kNtMax; ++n) other += sz(im.allo_init[n]) + sz(im.allo_head[n]);
        for (int n = 0; n <= (int) kNtMax; ++n) other += sz(im.allo_mtp_in[n]) + sz(im.allo_mtp_head[n]);
        const double M = 1048576.0;
        std::fprintf(stderr, "dense compute buffers: attention %.0f MiB (%lld graphs, largest %.1f MiB), finish %.0f, "
                             "predict %.0f, init/head/mtp %.0f MiB\n", attn / M, (long long) n_attn, attn_max / M,
                     fin / M, pred / M, other / M);
    }
    return true;
}

void Ds4Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    Impl::Layer & L = im.ly[(size_t) il];
    std::memcpy(im.in_host.data() + ((const uint8_t *) L.rbias->data - im.in_base), bias, (size_t) im.NEXP * 4);
    im.in_pos = -1;   // re-upload the span before the next attention graph
}

bool Ds4Dense::finish_layer(int il, const float * routed_sum) { return finish_layer_n(il, 1, routed_sum); }

bool Ds4Dense::finish_layer_n(int il, int n, const float * routed_sum) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "finish_layer: bad layer"; return false; }
    if (!im.ensure_n(n)) return false;
    Impl::Layer & L = im.ly[(size_t) il];
    if (!alloc_graph(L.allo_finish[n], L.gf_finish[n])) { im.err = "gallocr(finish) failed"; return false; }
    ggml_backend_tensor_set(im.routed_sum, routed_sum, 0, (size_t) (im.D * n) * 4);
    if (ggml_backend_graph_compute(im.backend, L.gf_finish[n]) != GGML_STATUS_SUCCESS) {
        im.err = "finish graph compute failed";
        return false;
    }
    if (!L.host_llast.empty()) ggml_backend_tensor_get(L.t_llast, L.host_llast.data(), 0, (size_t) (im.D * im.HC * n) * 4);
    L.have_finish = true;
    return true;
}

bool Ds4Dense::logits(const float ** out, int * n_vocab) { return logits_n(1, out, n_vocab); }

bool Ds4Dense::logits_n(int n, const float ** out, int * n_vocab) {
    Impl & im = *p_;
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_head[n], im.gf_head[n])) { im.err = "gallocr(head) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_head[n]) != GGML_STATUS_SUCCESS) {
        im.err = "head graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(im.logits_t[n], im.host_logits.data(), 0, (size_t) (im.g.vocab_size * n) * 4);
    if (out) *out = im.host_logits.data();
    if (n_vocab) *n_vocab = (int) im.g.vocab_size;
    return true;
}

bool Ds4Dense::mtp_begin(const int * next_tids, int n) {
    Impl & im = *p_;
    if (im.il_mtp < 0) { im.err = "mtp_begin: no MTP head loaded"; return false; }
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_mtp_in[n], im.gf_mtp_in[n])) { im.err = "gallocr(mtp_in) failed"; return false; }
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        const int tid = next_tids[t];
        if (tid < 0 || tid >= im.g.vocab_size) { im.err = "token id out of range"; return false; }
        const uint8_t * row = im.embd_data + (size_t) tid * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);
    if (const char * dbg = std::getenv("DS4_DBG_MTP_H")) {    // append the trunk state the MTP reads (D*HC*n f32)
        std::vector<float> v((size_t) (im.D * im.HC * n));
        ggml_backend_tensor_get(im.x_state, v.data(), 0, v.size() * 4);
        if (std::FILE * f = std::fopen(dbg, "ab")) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
    }
    if (ggml_backend_graph_compute(im.backend, im.gf_mtp_in[n]) != GGML_STATUS_SUCCESS) {
        im.err = "mtp input graph compute failed";
        return false;
    }
    if (const char * dbg = std::getenv("DS4_DBG_MTP_IN")) {   // append the MTP input state (D*HC*n f32) to a file
        std::vector<float> v((size_t) (im.D * im.HC * n));
        ggml_backend_tensor_get(im.x_state, v.data(), 0, v.size() * 4);
        if (std::FILE * f = std::fopen(dbg, "ab")) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
    }
    return true;
}

bool Ds4Dense::mtp_logits_n(int n, const float ** out, int * n_vocab) {
    Impl & im = *p_;
    if (im.il_mtp < 0) { im.err = "mtp_logits_n: no MTP head loaded"; return false; }
    if (!im.ensure_n(n)) return false;
    if (!alloc_graph(im.allo_mtp_head[n], im.gf_mtp_head[n])) { im.err = "gallocr(mtp head) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_mtp_head[n]) != GGML_STATUS_SUCCESS) {
        im.err = "mtp head graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(im.mtp_logits_t[n], im.host_logits.data(), 0, (size_t) (im.g.vocab_size * n) * 4);
    if (out) *out = im.host_logits.data();
    if (n_vocab) *n_vocab = (int) im.g.vocab_size;
    return true;
}

// ================================================================== prompt chunks (prefill_*)

static const Ds4Dense::Impl::Layer * pf_layer_of_ratio(const Ds4Dense::Impl & im, int64_t ratio) {
    for (const auto & L : im.ly) if (L.ratio == ratio) return &L;
    return nullptr;
}

static Ds4Dense::Impl::PfRatio * pf_ratio_set(Ds4Dense::Impl & im, int64_t ratio) {
    return ratio == 4 ? &im.P.r4 : ratio == 128 ? &im.P.r128 : nullptr;
}

// the chunk-sized hand-off and input tensors (P) + the shared graph allocator, at the first prefill_begin
static bool pf_alloc(Ds4Dense::Impl & im) {
    const int64_t NP = im.pf_cap, D = im.D, HC = im.HC;
    ggml_init_params ip = { /*mem_size*/ 64ull * ggml_tensor_overhead(), /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    im.pf_ctx = ggml_init(ip);
    if (!im.pf_ctx) { im.err = "prefill: ggml_init"; return false; }
    ggml_context * c = im.pf_ctx;
    auto& P = im.P;
    P.x_state    = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NP);
    P.routed_sum = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NP);
    P.i_tid      = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
    P.i_emb      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NP);
    P.i_pos      = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
    P.hap        = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, HC, NP);
    P.post_f     = ggml_new_tensor_2d(c, GGML_TYPE_F32, HC, NP);
    P.comb_f     = ggml_new_tensor_3d(c, GGML_TYPE_F32, HC, HC, NP);
    P.shexp      = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, NP);
    P.i_slot_raw = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.RAW);
    P.i_idx_raw  = ggml_new_tensor_1d(c, GGML_TYPE_I32, im.SWA - 1 + NP);
    P.i_mask_raw = ggml_new_tensor_1d(c, GGML_TYPE_F16, (im.SWA - 1 + NP) * NP);
    for (int64_t r : { (int64_t) 4, (int64_t) 128 }) {
        const Ds4Dense::Impl::Layer * L = pf_layer_of_ratio(im, r);
        if (!L) continue;
        Ds4Dense::Impl::PfRatio & R = *pf_ratio_set(im, r);
        R.i_slot_comp  = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
        R.i_slot_state = ggml_new_tensor_1d(c, GGML_TYPE_I32, L->ring_sz);
        R.i_comp_pos   = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
        R.i_state_pos  = ggml_new_tensor_1d(c, GGML_TYPE_I32, NP);
        R.i_idx_state  = ggml_new_tensor_1d(c, GGML_TYPE_I32, L->ring * NP);
        R.vis_full     = ggml_new_tensor_1d(c, GGML_TYPE_F32, L->comp_max * NP);
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(im.backend);
    im.pf_buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
    if (!im.pf_buf) { im.err = "prefill: cannot allocate the chunk tensors"; return false; }
    // router output blocks: the decode layout (o_blk bytes per token), o_f and o_i alias the same bytes
    const size_t ob = (size_t) im.o_blk * (size_t) NP;
    im.pf_obuf = ggml_backend_buft_alloc_buffer(buft, ob + 256);
    if (!im.pf_obuf) { im.err = "prefill: cannot allocate the router output blocks"; return false; }
    P.o_f = ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t) (ob / 4));
    P.o_i = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) (ob / 4));
    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(im.pf_obuf);
    if (ggml_backend_tensor_alloc(im.pf_obuf, P.o_f, base) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(im.pf_obuf, P.o_i, base) != GGML_STATUS_SUCCESS) {
        im.err = "prefill: cannot place the router output blocks"; return false;
    }
    {
        const size_t rows = (size_t) (D * NP) * sizeof(float), al = 256;
        const size_t o_sz = (ob + al - 1) / al * al, r_sz = (rows + al - 1) / al * al, tot = o_sz + 2 * r_sz;
        uint8_t * base = nullptr;
        ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(im.backend));
        if (hbt && (im.pf_hbuf = ggml_backend_buft_alloc_buffer(hbt, tot)))
            base = (uint8_t *) ggml_backend_buffer_get_base(im.pf_hbuf);
        else {
            im.pf_hvec.assign(tot, 0);
            base = im.pf_hvec.data();
        }
        im.pf_out_host = base;
        im.pf_fn = (float *) (base + o_sz);
        im.pf_routed = (float *) (base + o_sz + r_sz);
    }
    im.pf_allo = ggml_gallocr_new(ggml_backend_get_default_buffer_type(im.backend));   // its own (freed by pf_free)
    if (!im.pf_allo) { im.err = "prefill: ggml_gallocr_new"; return false; }
    im.pf_meta.assign(64ull * 1024 * 1024, 0);
    return true;
}

// point the decode tensors the builders use at the chunk's (and back: swapping twice restores)
static void pf_swap(Ds4Dense::Impl & im, Ds4Dense::Impl::Layer * L) {
    auto& P = im.P;
    std::swap(im.x_state, P.x_state); std::swap(im.routed_sum, P.routed_sum);
    std::swap(im.i_tid, P.i_tid); std::swap(im.i_emb, P.i_emb); std::swap(im.i_pos, P.i_pos);
    if (!L) return;
    std::swap(L->hap, P.hap); std::swap(L->post_f, P.post_f); std::swap(L->comb_f, P.comb_f);
    std::swap(L->shexp, P.shexp); std::swap(L->o_f, P.o_f); std::swap(L->o_i, P.o_i);
    std::swap(L->i_slot_raw, P.i_slot_raw); std::swap(L->i_idx_raw, P.i_idx_raw); std::swap(L->i_mask_raw, P.i_mask_raw);
    if (Ds4Dense::Impl::PfRatio * R = pf_ratio_set(im, L->ratio)) {
        std::swap(L->i_slot_comp, R->i_slot_comp); std::swap(L->i_slot_state, R->i_slot_state);
        std::swap(L->i_comp_pos, R->i_comp_pos); std::swap(L->i_state_pos, R->i_state_pos);
        std::swap(L->i_idx_state, R->i_idx_state); std::swap(L->vis_full, R->vis_full);
    }
}

// build one chunk graph (pf mode), run it once on the shared allocator, `after` reads its outputs, drop it.  Built
// fresh each time (uid 0): ggml-cuda runs a graph whose nodes changed directly, without capturing it.
template <class Build, class After>
static bool pf_run(Ds4Dense::Impl & im, Ds4Dense::Impl::Layer * L, Build build, After after, const char * what) {
    ggml_init_params ip = { im.pf_meta.size(), im.pf_meta.data(), /*no_alloc*/ true };
    ggml_context * tc = ggml_init(ip);
    if (!tc) { im.err = std::string(what) + ": ggml_init"; return false; }
    ggml_context * keep = im.gctx;
    im.gctx = tc;
    im.pf = true;
    pf_swap(im, L);
    ggml_cgraph * gf = build();
    pf_swap(im, L);
    im.pf = false;
    im.gctx = keep;
    bool ok = gf && ggml_gallocr_alloc_graph(im.pf_allo, gf) &&
              ggml_backend_graph_compute(im.backend, gf) == GGML_STATUS_SUCCESS;
    if (!ok) im.err = std::string(what) + ": chunk graph failed";
    else ok = after();
    ggml_free(tc);
    return ok;
}

bool Ds4Dense::prefill_begin(const int * tids, int n, int pos0) {
    Impl & im = *p_;
    if (im.pf_cap <= 0) { im.err = "prefill: Ds4DenseConfig::prefill_chunk is 0"; return false; }
    if (im.il_mtp >= 0) { im.err = "prefill: chunked prefill does not fill the MTP window (not implemented)"; return false; }
    if (n < 1 || n > im.pf_cap || pos0 < 0) { im.err = "prefill: chunk size/position out of range"; return false; }
    if (!im.pf_ctx && !pf_alloc(im)) return false;
    auto& P = im.P;
    const int64_t pos_last = (int64_t) pos0 + n - 1;
    for (const Impl::Layer & L : im.ly)
        if (L.ratio != 0 && pos_last / L.ratio >= L.comp_max) { im.err = "prefill: beyond the compressed cache capacity"; return false; }

    // the decode input span on the device (the chunk graphs read the route bias from it); decode re-uploads after
    ggml_backend_tensor_set(im.i_span, im.in_host.data(), 0, im.in_host.size());
    im.in_pos = -1;

    std::vector<int32_t> t32((size_t) n), p32((size_t) n);
    im.host_emb.resize((size_t) (im.D * n));
    for (int t = 0; t < n; ++t) {
        const int tid = tids[t];
        if (tid < 0 || tid >= im.g.vocab_size) { im.err = "token id out of range"; return false; }
        t32[(size_t) t] = tid;
        p32[(size_t) t] = pos0 + t;
        const uint8_t * row = im.embd_data + (size_t) tid * im.embd_row;
        float * dst = im.host_emb.data() + (size_t) t * im.D;
        if (im.embd_type == GGML_TYPE_F32) std::memcpy(dst, row, (size_t) im.D * 4);
        else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, dst, im.D);
    }
    ggml_backend_tensor_set(P.i_tid, t32.data(), 0, t32.size() * 4);
    ggml_backend_tensor_set(P.i_pos, p32.data(), 0, p32.size() * 4);
    ggml_backend_tensor_set(P.i_emb, im.host_emb.data(), 0, im.host_emb.size() * 4);

    // a ring row of position tok: before the chunk -> its slot in the ring, inside it -> ring_sz + (tok - pos0)
    auto ext = [&](int64_t tok, int64_t ring_sz) -> int32_t {
        return (int32_t) (tok < pos0 ? ((tok % ring_sz) + ring_sz) % ring_sz : ring_sz + (tok - pos0));
    };
    auto tail = [&](int64_t ring_sz) {   // ring slots of the chunk's last min(n, ring_sz) positions
        const int64_t k = std::min<int64_t>(n, ring_sz);
        std::vector<int32_t> v((size_t) k);
        for (int64_t j = 0; j < k; ++j) v[(size_t) j] = (int32_t) ((pos0 + n - k + j) % ring_sz);
        return v;
    };
    {   // raw window (the same for every layer)
        const int64_t n_raw = im.SWA + n - 1;
        std::vector<int32_t> idx((size_t) n_raw);
        std::vector<ggml_fp16_t> msk((size_t) (n_raw * n));
        const ggml_fp16_t h0 = ggml_fp32_to_fp16(0.0f), hinf = ggml_fp32_to_fp16(NEG_INF);
        for (int64_t i = 0; i < n_raw; ++i) {
            const int64_t tok = (int64_t) pos0 - im.SWA + 1 + i;
            idx[(size_t) i] = ext(tok, im.RAW);
            for (int t = 0; t < n; ++t) {
                const int64_t pt = (int64_t) pos0 + t;
                msk[(size_t) (t * n_raw + i)] = tok >= 0 && tok <= pt && tok > pt - im.SWA ? h0 : hinf;
            }
        }
        const std::vector<int32_t> sl = tail(im.RAW);
        ggml_backend_tensor_set(P.i_idx_raw, idx.data(), 0, idx.size() * 4);
        ggml_backend_tensor_set(P.i_mask_raw, msk.data(), 0, msk.size() * 2);
        ggml_backend_tensor_set(P.i_slot_raw, sl.data(), 0, sl.size() * 4);
    }
    for (int64_t r : { (int64_t) 4, (int64_t) 128 }) {   // compressed layers: the same per ratio
        const Impl::Layer * L = pf_layer_of_ratio(im, r);
        if (!L) continue;
        Impl::PfRatio & R = *pf_ratio_set(im, r);
        const int64_t cap = attn_cap(*L, pos_last);
        std::vector<int32_t> sp((size_t) n), cpp((size_t) n), sc((size_t) n), sidx((size_t) (L->ring * n));
        std::vector<float> vis((size_t) (cap * n), NEG_INF);
        for (int t = 0; t < n; ++t) {
            const int64_t pt = (int64_t) pos0 + t, bl = pt / r;
            sp[(size_t) t]  = (int32_t) (pt % r);
            cpp[(size_t) t] = (int32_t) (r * bl);
            const bool later_same = t + 1 < n && (pt + 1) / r == bl;   // a later token of the chunk owns the row
            sc[(size_t) t]  = (int32_t) (later_same ? L->comp_max : bl);
            for (int64_t i = 0; i < L->ring; ++i) sidx[(size_t) (t * L->ring + i)] = ext(pt - L->ring + 1 + i, L->ring_sz);
            const int64_t nv = (pt + 1) / r;
            for (int64_t i = 0; i < nv && i < cap; ++i) vis[(size_t) (t * cap + i)] = 0.0f;
        }
        const std::vector<int32_t> ss = tail(L->ring_sz);
        ggml_backend_tensor_set(R.i_state_pos, sp.data(), 0, sp.size() * 4);
        ggml_backend_tensor_set(R.i_comp_pos, cpp.data(), 0, cpp.size() * 4);
        ggml_backend_tensor_set(R.i_slot_comp, sc.data(), 0, sc.size() * 4);
        ggml_backend_tensor_set(R.i_idx_state, sidx.data(), 0, sidx.size() * 4);
        ggml_backend_tensor_set(R.i_slot_state, ss.data(), 0, ss.size() * 4);
        ggml_backend_tensor_set(R.vis_full, vis.data(), 0, vis.size() * 4);
    }
    im.pf_pos0 = pos0;
    im.pf_n = n;
    return pf_run(im, nullptr, [&] { return build_init(im, n); }, [] { return true; }, "prefill_begin");
}

bool Ds4Dense::prefill_attn(int il, int * routed_ids, float * routed_w, const float ** ffn_norm_host) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || im.pf_n <= 0) { im.err = "prefill_attn: bad layer / no chunk"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    const int n = im.pf_n;
    const int64_t cap = attn_cap(L, (int64_t) im.pf_pos0 + n - 1);
    return pf_run(im, &L, [&] { return build_attn(im, il, cap, n, L); }, [&] {
        const size_t ob = (size_t) im.o_blk;
        ggml_backend_tensor_get(im.P.o_f, im.pf_out_host, 0, ob * (size_t) n);
        for (int t = 0; t < n; ++t) {
            const uint8_t * b0 = im.pf_out_host + ob * (size_t) t;
            const int32_t * ids = (const int32_t *) (b0 + im.o_ids_off);
            const float * wv = (const float *) (b0 + im.o_wts_off);
            for (int64_t i = 0; i < im.NUSED; ++i) {
                routed_ids[t * im.NUSED + i] = ids[i];
                routed_w[t * im.NUSED + i] = wv[i];
            }
            std::memcpy(im.pf_fn + (size_t) t * im.D, b0, (size_t) im.D * 4);
        }
        if (ffn_norm_host) *ffn_norm_host = im.pf_fn;
        return true;
    }, "prefill_attn");
}

bool Ds4Dense::prefill_finish(int il, const float * routed_sum) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || im.pf_n <= 0) { im.err = "prefill_finish: bad layer / no chunk"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    const int n = im.pf_n;
    if (routed_sum) ggml_backend_tensor_set(im.P.routed_sum, routed_sum, 0, (size_t) (im.D * n) * 4);
    // (nullptr: the sums are already in P.routed_sum - see prefill_routed_device)
    return pf_run(im, &L, [&] { return build_finish(im, n, L); }, [] { return true; }, "prefill_finish");
}

bool Ds4Dense::prefill_logits(int row0, int nrows, float * out) {
    Impl & im = *p_;
    if (im.pf_n <= 0 || row0 < 0 || nrows < 1 || row0 + nrows > im.pf_n) { im.err = "prefill_logits: rows out of range"; return false; }
    im.pf_row0 = row0;
    return pf_run(im, nullptr, [&] { return build_head(im, nrows); }, [&] {
        ggml_backend_tensor_get(im.pf_logits, out, 0, (size_t) (im.g.vocab_size * nrows) * 4);
        return true;
    }, "prefill_logits");
}

bool Ds4Dense::prefill_end() {
    Impl & im = *p_;
    im.pf_n = 0;
    im.in_pos = -1;      // the decode inputs are uploaded again by the next attn_router
    im.cur_tid = -1;
    return true;
}

void Ds4Dense::prefill_release() { p_->pf_free(); }

void * Ds4Dense::prefill_routed_device() {
    Impl & im = *p_;
    if (ggml_backend_is_cpu(im.backend)) return nullptr;
    if (!im.pf_ctx && im.pf_cap > 0 && !pf_alloc(im)) return nullptr;
    return im.P.routed_sum ? im.P.routed_sum->data : nullptr;
}

float * Ds4Dense::prefill_routed_buffer() {
    Impl & im = *p_;
    if (!im.pf_ctx && im.pf_cap > 0 && !pf_alloc(im)) return nullptr;
    return im.pf_routed;
}

// taps: the LAST token of the most recent pass (for n = 1, the token)
const float * Ds4Dense::tap_attn_out(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_attn ? L.host_attn.data() + (size_t) (L.tap_n - 1) * im.D : nullptr;
}

const float * Ds4Dense::tap_attn_raw(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_attn ? L.host_attn_raw.data() + (size_t) (L.tap_n - 1) * im.NH * im.DH : nullptr;
}

const float * Ds4Dense::tap_l_last(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_finish ? L.host_llast.data() + (size_t) (L.tap_n - 1) * im.D * im.HC : nullptr;
}
