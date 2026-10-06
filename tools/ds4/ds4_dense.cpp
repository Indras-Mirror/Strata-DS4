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
#include <vector>

using namespace strata;

namespace {

constexpr float NEG_INF  = -INFINITY;
constexpr float SENTINEL = -1.0e30f;   // "compressor state row not written yet" score: exp() underflows to 0

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
bool load_weights(const GgufModel & model, ggml_backend_t backend, WStore & w, bool skip_experts, std::string & err) {
    ggml_init_params ip = { /*mem_size*/ 256ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    w.ctx = ggml_init(ip);
    if (!w.ctx) { err = "ggml_init(weights) failed"; return false; }
    ggml_set_no_alloc(w.ctx, true);

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
                if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
                int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
                for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
                ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
                if (!t) { err = "cannot create " + ti.name; return false; }
                ggml_set_name(t, ti.name.c_str());
                if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                    err = "cannot bind " + ti.name;
                    return false;
                }
                w.t[ti.name] = t;
            }
        }
        return true;
    }

    // non-CPU: create the tensor set, allocate one backend buffer for it, upload the payloads
    auto skip = [&](const std::string & n) {
        return n == "token_embd.weight" || (skip_experts && is_routed_expert(n));
    };
    for (size_t s = 0; s < model.size(); ++s) {
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip(ti.name)) continue;
            if (!model.in_bounds(ti, s)) { err = ti.name + " out of bounds"; return false; }
            int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
            for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
            ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
            if (!t) { err = "cannot create " + ti.name; return false; }
            ggml_set_name(t, ti.name.c_str());
            w.t[ti.name] = t;
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buf) { err = "cannot allocate the weight buffer on the backend"; return false; }
    w.bufs.push_back(buf);
    for (size_t s = 0; s < model.size(); ++s) {
        for (const auto & ti : model.shard(s).tensors()) {
            if (skip(ti.name)) continue;
            ggml_tensor * t = w.t[ti.name];
            ggml_backend_tensor_set(t, model.shard(s).tensor_data(ti), 0, ggml_nbytes(t));
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
    ggml_context *  sctx = nullptr;     // persistent decode state
    // Per-step inputs and per-layer router outputs live at fixed offsets in two small buffers so a token costs ONE
    // upload of every layer's inputs and one readback per layer (was ~11 tensor_set + 3 tensor_get per layer, each a
    // stream sync on CUDA: 1283 copies + 905 syncs per token measured by nsys).
    ggml_context *  ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr, obuf = nullptr;
    ggml_tensor *   i_span = nullptr;   // I8 over the whole input buffer
    std::vector<uint8_t> in_host;       // its host mirror
    uint8_t *       in_base = nullptr;  // device base of ibuf (offsets are t->data - in_base)
    int             in_pos = -1;        // the position whose inputs are uploaded
    int             cur_tid = -1;       // the token id in i_tid
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_context *  gctx = nullptr;     // graph node structs (data comes from the gallocrs)
    std::string     err;

    // One graph allocator per graph, never a shared one.  ggml-alloc binds a tensor's data pointer once
    // and skips re-binding it on later allocations, while a reserve triggered by a *different* graph
    // frees and reallocates the chunks it had bound - a shared allocator therefore leaves the tensors of
    // every other graph dangling into freed memory as soon as one graph grows.  Per-graph allocators also
    // give each decode graph the stable, dedicated buffer ggml-cuda needs to capture it as a CUDA graph.
    ggml_gallocr_t make_allo() const {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        return ggml_gallocr_new(buft);
    }

    int64_t D = 0, HC = 0, DH = 0, DHR = 0, NH = 0, OG = 0, OL = 0, OGD = 0, NHPG = 0, RQ = 0, SWA = 0;
    int64_t IDXH = 0, IDXK = 0, IDXTOPK = 0;
    int64_t NEXP = 0, NUSED = 0, NFF = 0, NSHEXP = 0;

    ggml_tensor * x_state    = nullptr;   // [D, hc]  layer input / output state
    ggml_tensor * routed_sum = nullptr;   // [D]      routed-expert sum fed to finish_layer
    ggml_tensor * i_tid      = nullptr;   // I32[1]
    ggml_tensor * i_emb      = nullptr;   // F32[D]   embd[tid], looked up on the host (begin_token)
    const uint8_t * embd_data = nullptr;  // token_embd on the GGUF mmap
    int           embd_type  = -1;
    size_t        embd_row   = 0;         // bytes per row
    std::vector<float> host_emb;
    ggml_tensor * i_pos      = nullptr;   // I32[1]
    ggml_tensor * rot        = nullptr;   // [idx_k, idx_k] F32 Walsh-Hadamard (lightning indexer)
    ggml_tensor * logits_t   = nullptr;   // [V, 1]   head graph output
    std::vector<float> host_logits;

    struct Var {
        int64_t cap = 0;
        ggml_tensor * vis = nullptr;      // F32 [cap, 1] 0 visible / -inf hidden
        ggml_tensor * out_ids = nullptr;  // I32 [NUSED, 1]
        ggml_tensor * out_wts = nullptr;  // F32 [1, NUSED, 1]
        ggml_tensor * dbg_isc = nullptr;  // F32 [cap, 1] indexer scores + visibility (DS4_DBG_INDEXER=1)
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t allo = nullptr;
    };
    struct Layer {
        int64_t ratio = 0;
        bool    csa = false;
        int64_t ring = 0, sdim = 0, comp_max = 0;

        ggml_tensor * raw = nullptr;      // [DH, SWA]              raw sliding-window K (=V, post-RoPE)
        ggml_tensor * comp = nullptr;     // [DH, comp_max]         compressed K (post-RoPE)

        ggml_tensor * st_kv = nullptr;    // [sdim, ring]           compressor state (values)
        ggml_tensor * st_sc = nullptr;    // [sdim, ring]           compressor state (scores, SENTINEL unwritten)
        ggml_tensor * icomp = nullptr;    // [IDXK, comp_max]       lightning-indexer compressed K (post-WHT)
        ggml_tensor * ist_kv = nullptr;   // [2*IDXK, 2*ratio]      indexer compressor state (values)
        ggml_tensor * ist_sc = nullptr;   // [2*IDXK, 2*ratio]      indexer compressor state (scores)

        ggml_tensor * hap = nullptr;      // [D, HC]                attn hc_post result (finish residual)
        ggml_tensor * post_f = nullptr;   // [HC]
        ggml_tensor * comb_f = nullptr;   // [HC, HC]
        ggml_tensor * shexp = nullptr;    // [D]                    shared expert output
        ggml_tensor * fn = nullptr;       // [D]                    ffn_norm activation (device)
        ggml_tensor * t_attn_raw = nullptr;  // [NH*DH]             flash-attn output, pre output-LoRA
        ggml_tensor * t_attn = nullptr;      // [D]                 attention output, post output-LoRA
        ggml_tensor * t_llast = nullptr;     // [D, HC]             l_last

        ggml_tensor * i_slot_raw = nullptr;   // I32[1]
        ggml_tensor * i_idx_raw = nullptr;    // I32[SWA]
        ggml_tensor * i_mask_raw = nullptr;   // F16[SWA]
        ggml_tensor * i_slot_comp = nullptr;  // I32[1]
        ggml_tensor * i_slot_state = nullptr; // I32[1]
        ggml_tensor * i_comp_pos = nullptr;   // I32[1]
        ggml_tensor * i_state_pos = nullptr;  // I32[1]
        ggml_tensor * i_idx_state = nullptr;  // I32[ring]

        std::vector<Var> vars;
        ggml_cgraph * gf_predict = nullptr;
        ggml_tensor * predict_out = nullptr;
        ggml_gallocr_t allo_predict = nullptr;
        ggml_cgraph * gf_finish = nullptr;
        ggml_gallocr_t allo_finish = nullptr;
        std::vector<float> host_fn, host_attn_raw, host_attn, host_llast;
        bool have_attn = false, have_finish = false;
        ggml_tensor * vis_full = nullptr; // F32 [comp_max]   compressed-row visibility (graphs view [cap])
        ggml_tensor * rbias = nullptr;    // F32 [NEXP]       cache-aware routing: added to the SELECTION score only
        ggml_tensor * o_ids = nullptr;    // I32 [NUSED]      router ids    } one contiguous block with fn,
        ggml_tensor * o_wts = nullptr;    // F32 [NUSED]      router weights} read back by one tensor_get
        ggml_tensor * o_span = nullptr;   // I8 over [fn | ids | wts]
        std::vector<uint8_t> host_out;
    };
    std::vector<Layer> ly;

    ggml_cgraph * gf_init = nullptr;
    ggml_cgraph * gf_head = nullptr;
    ggml_gallocr_t allo_init = nullptr;
    ggml_gallocr_t allo_head = nullptr;

    ~Impl() {
        for (Var & v : all_vars()) if (v.allo) ggml_gallocr_free(v.allo);
        for (Layer & L : ly) {
            if (L.allo_predict) ggml_gallocr_free(L.allo_predict);
            if (L.allo_finish)  ggml_gallocr_free(L.allo_finish);
        }
        if (allo_init) ggml_gallocr_free(allo_init);
        if (allo_head) ggml_gallocr_free(allo_head);
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

// Attention + router for one compressed-row capacity `cap` (0 = nothing compressed yet).  All tensors are
// created once; per-step values arrive through the layer's input leaves, so the graph is stable.
static ggml_cgraph * build_attn(Ds4Dense::Impl & im, int il, int64_t cap, Ds4Dense::Impl::Layer & L) {
    Ds4Dense::Impl::Var var;
    var.cap = cap;

    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 4096, false);
    B b { &im, gc, &im.g };

    const int64_t D = im.D, HC = im.HC, DH = im.DH, DHR = im.DHR, NH = im.NH, OGD = im.OGD, OL = im.OL, OG = im.OG;
    const int64_t SWA = im.SWA;
    const int64_t ratio = L.ratio;
    const bool csa = L.csa;

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int n_ctx;
    b.rope_cfg(ratio, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, n_ctx);
    const float crb = (float) im.g.compress_rope_base;

    // ---- attention hyper-connection -------------------------------------------------
    ggml_tensor * xin = ggml_reshape_3d(gc, im.x_state, D, HC, 1);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * attn_pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                      b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * xn = b.rms_w(attn_pre, b.BL(il, "attn_norm.weight"));

    // ---- q latent (kept for the indexer's query) --------------------------------------
    ggml_tensor * qr2 = ggml_mul_mat(gc, b.BL(il, "attn_q_a.weight"), xn);
    qr2 = b.rms_w(qr2, b.BL(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_mul_mat(gc, b.BL(il, "attn_q_b.weight"), qr2);
    q = ggml_reshape_3d(gc, q, DH, NH, 1);
    q = ggml_rms_norm(gc, q, (float) im.g.rms_eps);
    q = b.rope(q, im.i_pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- kv latent (K == V) -----------------------------------------------------------
    ggml_tensor * kv = ggml_mul_mat(gc, b.BL(il, "attn_kv.weight"), xn);
    kv = b.rms_w(kv, b.BL(il, "attn_kv_a_norm.weight"));
    kv = ggml_reshape_3d(gc, kv, DH, 1, 1);
    kv = b.rope(kv, im.i_pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // ---- raw sliding-window ring ------------------------------------------------------
    ggml_tensor * raw2 = ggml_set_rows(gc, L.raw, ggml_cont_2d(gc, kv, DH, 1), L.i_slot_raw);
    ggml_tensor * raw_src = ggml_get_rows(gc, raw2, L.i_idx_raw);      // [DH, SWA] oldest -> newest
    ggml_tensor * k_all = ggml_reshape_3d(gc, raw_src, DH, 1, SWA);   // token axis on ne2 (ds4_ref layout)
    ggml_tensor * mask = ggml_reshape_2d(gc, L.i_mask_raw, SWA, 1);

    // ---- compressed keys -------------------------------------------------------------
    if (cap > 0) {
        ggml_tensor * comp_k = nullptr;
        ggml_tensor * cmask = nullptr;

        if (csa) {
            // overlap compressor (ds4_ref attention():514-556): each token state is [ prev-half | cur-half ]
            auto overlap = [&](const std::string & pfx, int64_t head_dim, ggml_tensor * ring_kv, ggml_tensor * ring_sc,
                               ggml_tensor * idx_state) -> ggml_tensor * {
                ggml_tensor * st_kv = ggml_mul_mat(gc, b.BL(il, pfx + "_kv.weight"), xn);
                ggml_tensor * st_sc = ggml_mul_mat(gc, b.BL(il, pfx + "_gate.weight"), xn);
                ggml_tensor * ape = ggml_get_rows(gc, b.BL(il, pfx + "_ape.weight"), L.i_state_pos);
                st_sc = ggml_add(gc, st_sc, ape);

                ggml_tensor * rkv = ggml_set_rows(gc, ring_kv, ggml_cont_2d(gc, st_kv, 2 * head_dim, 1), L.i_slot_state);
                ggml_tensor * rsc = ggml_set_rows(gc, ring_sc, ggml_cont_2d(gc, st_sc, 2 * head_dim, 1), L.i_slot_state);
                ggml_tensor * rows_kv = ggml_get_rows(gc, rkv, idx_state);   // [2*hd, 2*ratio]
                ggml_tensor * rows_sc = ggml_get_rows(gc, rsc, idx_state);

                const size_t nb1 = rows_kv->nb[1];
                ggml_tensor * pkv = ggml_cont(gc, ggml_view_2d(gc, rows_kv, head_dim, ratio, nb1, 0));
                ggml_tensor * ckv = ggml_cont(gc, ggml_view_2d(gc, rows_kv, head_dim, ratio, nb1,
                                                              (size_t) ratio * nb1 + (size_t) head_dim * 4));
                ggml_tensor * psc = ggml_cont(gc, ggml_view_2d(gc, rows_sc, head_dim, ratio, nb1, 0));
                ggml_tensor * csc = ggml_cont(gc, ggml_view_2d(gc, rows_sc, head_dim, ratio, nb1,
                                                               (size_t) ratio * nb1 + (size_t) head_dim * 4));
                ggml_tensor * values = ggml_concat(gc, pkv, ckv, 1);          // [hd, 2*ratio]
                ggml_tensor * scores = ggml_concat(gc, psc, csc, 1);
                values = ggml_cont(gc, ggml_permute(gc, values, 1, 0, 2, 3)); // [2*ratio, hd]
                scores = ggml_cont(gc, ggml_permute(gc, scores, 1, 0, 2, 3));
                ggml_tensor * wts = ggml_soft_max(gc, scores);
                ggml_tensor * cc = ggml_sum_rows(gc, ggml_mul(gc, values, wts));
                cc = ggml_cont(gc, ggml_permute(gc, cc, 1, 0, 2, 3));         // [hd, 1]
                cc = b.rms_w(cc, b.BL(il, pfx + "_norm.weight"));
                return b.rope_at(cc, L.i_comp_pos, DHR, head_dim - DHR, n_ctx, crb,
                                 freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            };

            comp_k = overlap("attn_compressor", DH, L.st_kv, L.st_sc, L.i_idx_state);

            // lightning indexer: compressed keys (post-WHT) + query, then top-k over the visible blocks
            ggml_tensor * lid_k = ggml_cont(gc, overlap("indexer_compressor", im.IDXK, L.ist_kv, L.ist_sc, L.i_idx_state));
            lid_k = b.hadamard(lid_k, im.rot);
            ggml_tensor * icomp2 = ggml_set_rows(gc, L.icomp, ggml_cont_2d(gc, lid_k, im.IDXK, 1), L.i_slot_comp);
            // ds4_ref's lid_k is 3-D [idx_k, 1, n_blocks]; this 3-D view of the ring's first `cap`
            // columns has the same shape, so the permute below lands `cap` on ne[1] exactly like the
            // reference's kp = permute(lid_k, 0,2,1,3).  A 2-D view would put `cap` on ne[2] and the
            // mul_mat would then drop the block axis (the result would be [1,1,idx_h] instead of
            // [cap,1,idx_h]) - the decoded indexer scores would be wrong and the graph layout invalid.
            ggml_tensor * lid_src = ggml_view_3d(gc, icomp2, im.IDXK, 1, cap,
                                                 icomp2->nb[1], icomp2->nb[1], 0);

            ggml_tensor * iq = ggml_mul_mat(gc, b.BL(il, "indexer.attn_q_b.weight"), qr2);
            iq = ggml_reshape_3d(gc, iq, im.IDXK, im.IDXH, 1);
            iq = b.rope_at(iq, im.i_pos, DHR, im.IDXK - DHR, n_ctx, crb,
                           freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            iq = b.hadamard(iq, im.rot);
            ggml_tensor * iw = ggml_mul_mat(gc, b.BL(il, "indexer.proj.weight"), xn);
            iw = ggml_scale(gc, iw, 1.0f / std::sqrt((float) (im.IDXK * im.IDXH)));

            ggml_tensor * qp = ggml_permute(gc, iq, 0, 2, 1, 3);            // [idx_k, 1, idx_h]
            ggml_tensor * kp = ggml_permute(gc, lid_src, 0, 2, 1, 3);       // [idx_k, cap, 1]
            ggml_tensor * kq = ggml_mul_mat(gc, kp, qp);                    // [cap, 1, idx_h]
            kq = ggml_cont(gc, ggml_permute(gc, kq, 2, 1, 0, 3));           // [idx_h, 1, cap]
            ggml_tensor * sc = ggml_relu(gc, kq);
            sc = ggml_mul(gc, sc, ggml_reshape_4d(gc, iw, im.IDXH, 1, 1, 1));
            sc = ggml_sum_rows(gc, sc);                                     // [1, 1, cap]
            sc = ggml_cont(gc, ggml_permute(gc, sc, 2, 1, 0, 3));           // [cap, 1, 1]
            sc = ggml_reshape_2d(gc, sc, cap, 1);

            var.vis = ggml_view_2d(gc, L.vis_full, cap, 1, (size_t) cap * sizeof(float), 0);
            sc = ggml_add(gc, sc, var.vis);
            var.dbg_isc = sc;
            ggml_set_output(sc);

            // mask = -inf everywhere, 0 at the top-k rows, then ANDed with visibility
            const int64_t ntk = std::min<int64_t>(cap, im.IDXTOPK);
            ggml_tensor * tk = ggml_cont(gc, ggml_top_k(gc, sc, (int) ntk));    // [ntk, 1]
            ggml_tensor * a = ggml_fill(gc, var.vis, NEG_INF);
            a = ggml_view_4d(gc, a, 1, cap, 1, 1, a->nb[0], a->nb[1], a->nb[2], 0);
            ggml_tensor * zv = b.fill_f32({ 1, ntk, 1 }, 0.0f);
            ggml_tensor * m = ggml_set_rows(gc, a, zv, tk);
            m = ggml_view_4d(gc, m, m->ne[1], m->ne[2], 1, 1, m->nb[2], m->nb[3], m->nb[3], 0);
            m = ggml_add(gc, m, var.vis);
            cmask = ggml_cast(gc, m, GGML_TYPE_F16);
        } else {
            // HCA: one block == `ratio` consecutive tokens, no overlap (ds4_ref attention():603-622)
            ggml_tensor * st_kv = ggml_mul_mat(gc, b.BL(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * st_sc = ggml_mul_mat(gc, b.BL(il, "attn_compressor_gate.weight"), xn);
            ggml_tensor * ape = ggml_get_rows(gc, b.BL(il, "attn_compressor_ape.weight"), L.i_state_pos);
            st_sc = ggml_add(gc, st_sc, ape);

            ggml_tensor * rkv = ggml_set_rows(gc, L.st_kv, ggml_cont_2d(gc, st_kv, DH, 1), L.i_slot_state);
            ggml_tensor * rsc = ggml_set_rows(gc, L.st_sc, ggml_cont_2d(gc, st_sc, DH, 1), L.i_slot_state);
            ggml_tensor * rows_kv = ggml_get_rows(gc, rkv, L.i_idx_state);   // [DH, ratio]
            ggml_tensor * rows_sc = ggml_get_rows(gc, rsc, L.i_idx_state);
            ggml_tensor * values = ggml_cont(gc, ggml_permute(gc, rows_kv, 1, 0, 2, 3));   // [ratio, DH]
            ggml_tensor * scores = ggml_cont(gc, ggml_permute(gc, rows_sc, 1, 0, 2, 3));
            ggml_tensor * wts = ggml_soft_max(gc, scores);
            ggml_tensor * cc = ggml_sum_rows(gc, ggml_mul(gc, values, wts));  // [1, DH]
            cc = ggml_cont(gc, ggml_permute(gc, cc, 1, 0, 2, 3));             // [DH, 1]
            cc = b.rms_w(cc, b.BL(il, "attn_compressor_norm.weight"));
            comp_k = b.rope_at(cc, L.i_comp_pos, DHR, DH - DHR, n_ctx, crb,
                               freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            var.vis = ggml_view_2d(gc, L.vis_full, cap, 1, (size_t) cap * sizeof(float), 0);
            cmask = ggml_cast(gc, var.vis, GGML_TYPE_F16);
        }

        ggml_tensor * comp2 = ggml_set_rows(gc, L.comp, ggml_cont_2d(gc, comp_k, DH, 1), L.i_slot_comp);
        ggml_tensor * comp_src = ggml_view_2d(gc, comp2, DH, cap, comp2->nb[1], 0);
        k_all = ggml_concat(gc, k_all, ggml_reshape_3d(gc, comp_src, DH, 1, cap), 2);
        mask = ggml_concat(gc, mask, cmask, 0);
    }

    // ---- attention --------------------------------------------------------------------
    // ggml-cuda has a flash-attn kernel for d_head 512 only when the key count is a multiple of FATTN_KQ_STRIDE
    // (256) - llama.cpp pads its KV cache to that.  Off-CPU, pad with zero keys masked -inf: they get exactly zero
    // weight.  The CPU path (the one verified bit-exact against ds4_ref) is left as it was.
    if (!ggml_backend_is_cpu(im.backend)) {
        const int64_t n_kv = k_all->ne[2], pad = (256 - n_kv % 256) % 256;
        if (pad) {
            k_all = ggml_pad(gc, k_all, 0, 0, (int) pad, 0);
            mask = ggml_concat(gc, mask, ggml_cast(gc, b.fill_f32({ pad, 1 }, NEG_INF), GGML_TYPE_F16), 0);
        }
    }
    ggml_tensor * qp = ggml_permute(gc, q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(gc, k_all, 0, 2, 1, 3);
    ggml_tensor * kf = ggml_cast(gc, kp, GGML_TYPE_F16);
    ggml_tensor * vf = ggml_cast(gc, kp, GGML_TYPE_F16);
    ggml_tensor * out = ggml_flash_attn_ext(gc, qp, kf, vf, mask, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, b.BL(il, "attn_sinks.weight"));

    // the gate's attn_raw tap: ds4_ref dumps the raw flash-attn output *before* the de-RoPE
    // (its attn_csa_lid / attn_hca probe is named at the flash-attn result, not after rope_back)
    ggml_tensor * attn_raw = ggml_cont_2d(gc, out, NH * DH, 1);

    // de-RoPE then grouped output LoRA
    out = ggml_reshape_3d(gc, out, DH, NH, 1);
    out = b.rope_back(out, im.i_pos, DHR, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    out = ggml_reshape_3d(gc, out, OGD, OG, 1);
    out = ggml_permute(gc, out, 0, 2, 1, 3);
    ggml_tensor * wo_a = b.BL(il, "attn_output_a.weight");
    if (ggml_n_dims(wo_a) == 2) wo_a = ggml_reshape_3d(gc, wo_a, OGD, OL, OG);
    ggml_tensor * oa = ggml_mul_mat(gc, wo_a, out);
    oa = ggml_permute(gc, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(gc, oa, OL * OG, 1);
    ggml_tensor * attn_out = ggml_mul_mat(gc, b.BL(il, "attn_output_b.weight"), oa);

    // ---- feed-forward hyper-connection -------------------------------------------------
    ggml_tensor * hap = b.hc_post(attn_out, xin, post_a, comb_a);
    ggml_tensor * post_f = nullptr, * comb_f = nullptr;
    ggml_tensor * ffn_pre = b.hc_pre(hap, b.BL(il, "hc_ffn_fn.weight"), b.BL(il, "hc_ffn_scale.weight"),
                                     b.BL(il, "hc_ffn_base.weight"), &post_f, &comb_f);
    ggml_tensor * fn = b.rms_w(ffn_pre, b.BL(il, "ffn_norm.weight"));

    // ---- MoE router (ds4_ref main():904-928) -------------------------------------------
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fn);
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(gc, ggml_softplus(gc, rlog));

    ggml_tensor * selected = nullptr;
    if (il < (int) im.g.hash_layer_count) {
        selected = ggml_get_rows(gc, b.BL(il, "ffn_gate_tid2eid.weight"), im.i_tid);
    } else {
        ggml_tensor * sel = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
        // cache-aware routing (set_route_bias): a selection-only bias like exp_probs_b; the mixing weights below still
        // come from the unbiased probs.  All zeros (the default) = the model's own routing, bit for bit.
        sel = ggml_add(gc, sel, ggml_reshape_2d(gc, L.rbias, im.NEXP, 1));
        selected = ggml_argsort_top_k(gc, sel, (int) im.NUSED);
    }
    ggml_set_output(selected);
    ggml_tensor * probs3 = ggml_reshape_3d(gc, probs, 1, im.NEXP, 1);
    ggml_tensor * wts = ggml_get_rows(gc, probs3, selected);              // [1, topk, 1]
    wts = ggml_reshape_2d(gc, wts, im.NUSED, 1);
    ggml_tensor * wsum = ggml_add(gc, ggml_sum_rows(gc, wts), b.fill_f32({ 1, 1 }, 0.0f));
    wsum = ggml_clamp(gc, wsum, 6.103515625e-5f, INFINITY);
    wts = ggml_div(gc, wts, wsum);
    wts = ggml_reshape_3d(gc, wts, 1, im.NUSED, 1);
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

    // ---- persisted hand-offs (finish_layer + the gate's taps) ---------------------------
    ggml_tensor * c_fn  = ggml_cpy(gc, ggml_reshape_2d(gc, fn, D, 1), L.fn);
    ggml_tensor * c_hap = ggml_cpy(gc, ggml_reshape_2d(gc, hap, D, HC), L.hap);
    ggml_tensor * c_pf  = ggml_cpy(gc, ggml_reshape_2d(gc, post_f, HC, 1), L.post_f);
    ggml_tensor * c_cf  = ggml_cpy(gc, ggml_reshape_2d(gc, comb_f, HC, HC), L.comb_f);
    ggml_tensor * c_sh  = ggml_cpy(gc, ggml_reshape_2d(gc, shexp, D, 1), L.shexp);
    ggml_tensor * c_ar  = ggml_cpy(gc, attn_raw, L.t_attn_raw);
    ggml_tensor * c_at  = ggml_cpy(gc, ggml_reshape_2d(gc, attn_out, D, 1), L.t_attn);

    for (ggml_tensor * t : { c_fn, c_hap, c_pf, c_cf, c_sh, c_ar, c_at }) ggml_build_forward_expand(gf, t);
    ggml_build_forward_expand(gf, selected);
    ggml_build_forward_expand(gf, wts);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_1d(gc, selected, im.NUSED), L.o_ids));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_1d(gc, wts, im.NUSED), L.o_wts));

    var.gf = gf;
    var.out_ids = selected;
    var.out_wts = wts;
    var.allo = im.make_allo();   // one arena per graph variant; see Impl::make_allo
    L.vars.push_back(var);
    return gf;
}

// finish_layer: ffn_out = routed_sum + shexp ; l_last = hc_post(ffn_out, hap, post_f, comb_f)
static ggml_cgraph * build_finish(Ds4Dense::Impl & im, int il, Ds4Dense::Impl::Layer & L) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 512, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * shex2 = ggml_reshape_2d(gc, L.shexp, D, 1);
    ggml_tensor * rs    = ggml_reshape_2d(gc, im.routed_sum, D, 1);
    ggml_tensor * ffn_out = ggml_add(gc, shex2, rs);

    ggml_tensor * hap3 = ggml_reshape_3d(gc, L.hap, D, HC, 1);
    ggml_tensor * pf2  = ggml_reshape_2d(gc, L.post_f, HC, 1);
    ggml_tensor * cf3  = ggml_reshape_3d(gc, L.comb_f, HC, HC, 1);
    ggml_tensor * llast = b.hc_post(ffn_out, hap3, pf2, cf3);
    ggml_tensor * ll2 = ggml_reshape_2d(gc, llast, D, HC);

    ggml_build_forward_expand(gf, ggml_cpy(gc, ll2, im.x_state));
    ggml_build_forward_expand(gf, ggml_cpy(gc, ll2, L.t_llast));

    (void) il;
    L.gf_finish = gf;
    return gf;
}

// predict: the layer's router on rms_norm(hc_attn_pre) * ffn_norm.weight  (prefetch hint "A")
static ggml_cgraph * build_predict(Ds4Dense::Impl & im, int il, Ds4Dense::Impl::Layer & L) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 512, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * xin = ggml_reshape_3d(gc, im.x_state, D, HC, 1);
    ggml_tensor * post_a = nullptr, * comb_a = nullptr;
    ggml_tensor * attn_pre = b.hc_pre(xin, b.BL(il, "hc_attn_fn.weight"), b.BL(il, "hc_attn_scale.weight"),
                                      b.BL(il, "hc_attn_base.weight"), &post_a, &comb_a);
    ggml_tensor * fnp = b.rms_w(attn_pre, b.BL(il, "ffn_norm.weight"));
    ggml_tensor * rlog = ggml_mul_mat(gc, b.BL(il, "ffn_gate_inp.weight"), fnp);
    ggml_mul_mat_set_prec(rlog, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(gc, ggml_softplus(gc, rlog));

    ggml_tensor * out = nullptr;
    if (il < (int) im.g.hash_layer_count) {
        out = ggml_cast(gc, ggml_get_rows(gc, b.BL(il, "ffn_gate_tid2eid.weight"), im.i_tid), GGML_TYPE_F32);
    } else {
        out = ggml_add(gc, probs, b.BL(il, "exp_probs_b.bias"));
    }
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    L.predict_out = out;
    return gf;
}

// begin_token: hc_init = repeat(embd[tid], hc)
static ggml_cgraph * build_init(Ds4Dense::Impl & im) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 64, false);
    const int64_t D = im.D, HC = im.HC;
    ggml_tensor * x = ggml_reshape_3d(gc, im.i_emb, D, 1, 1);   // embd[tid], from the host (begin_token)
    x = ggml_repeat_4d(gc, x, D, HC, 1, 1);
    ggml_build_forward_expand(gf, ggml_cpy(gc, ggml_reshape_2d(gc, x, D, HC), im.x_state));
    return gf;
}

// logits: hc_head -> output_norm -> output
static ggml_cgraph * build_head(Ds4Dense::Impl & im) {
    ggml_context * gc = im.gctx;
    ggml_cgraph * gf = ggml_new_graph_custom(gc, 256, false);
    B b { &im, gc, &im.g };
    const int64_t D = im.D, HC = im.HC;

    ggml_tensor * xin = ggml_reshape_3d(gc, im.x_state, D, HC, 1);
    ggml_tensor * flat = ggml_reshape_2d(gc, xin, HC * D, 1);
    ggml_tensor * flat_norm = ggml_rms_norm(gc, flat, (float) im.g.rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(gc, im.w.get("output_hc_fn.weight"), flat_norm);
    ggml_tensor * scale = im.w.get("output_hc_scale.weight");
    ggml_tensor * base  = im.w.get("output_hc_base.weight");
    ggml_tensor * pre = ggml_sigmoid(gc, b.hc_affine(mixes, b.v1(scale, 1, 0), b.v1(base, HC, 0)));
    pre = ggml_scale_bias(gc, pre, 1.0f, (float) im.g.hc_eps);
    ggml_tensor * hc_head = b.hc_mean(xin, pre);
    ggml_tensor * rn = b.rms_w(hc_head, im.w.get("output_norm.weight"));
    ggml_tensor * lg = ggml_mul_mat(gc, im.w.get("output.weight"), rn);
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);
    im.logits_t = lg;
    return gf;
}

// ================================================================== public API

Ds4Dense::Ds4Dense() : p_(new Impl()) {}
Ds4Dense::~Ds4Dense() = default;

bool Ds4Dense::init(const std::string & model_path, const Ds4DenseConfig & cfg, std::string & err) {
    Impl & im = *p_;
    im.path = model_path;
    im.n_threads = cfg.n_threads;

    im.backend = cfg.backend;
    if (!im.backend) {
        im.backend = ggml_backend_cpu_init();
        im.own_backend = true;
        if (!im.backend) { err = "cannot create a CPU backend"; return false; }
    }
    if (ggml_backend_is_cpu(im.backend)) ggml_backend_cpu_set_n_threads(im.backend, im.n_threads);

    im.model = std::make_unique<GgufModel>(GgufModel::open(model_path));
    if (im.model->size() == 0) { err = "cannot open " + model_path; return false; }
    if (!(err = ds4_read_geometry(*im.model, im.g)).empty()) return false;
    if (!(err = check_ds4_model(*im.model, im.g)).empty()) return false;
    if (!load_weights(*im.model, im.backend, im.w, cfg.skip_routed_experts, err)) return false;
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
    im.RQ = g.r_q; im.SWA = g.n_swa;
    im.IDXH = g.indexer_n_head; im.IDXK = g.indexer_key_dim; im.IDXTOPK = g.indexer_top_k;
    im.NEXP = g.n_expert; im.NUSED = g.n_expert_used; im.NFF = g.n_ff; im.NSHEXP = g.n_expert_shared;

    const int64_t nl = g.n_layer();
    if (nl <= 0 || im.D <= 0 || im.HC <= 0 || im.SWA <= 0) { err = "degenerate geometry"; return false; }
    if (im.NUSED <= 0) { err = "expert_used_count is zero"; return false; }

    bool any_csa = false;
    // ---- persistent decode state -------------------------------------------------------
    {
        ggml_init_params ip = { /*mem_size*/ 128ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
        im.sctx = ggml_init(ip);
        if (!im.sctx) { err = "ggml_init(state) failed"; return false; }
        ggml_set_no_alloc(im.sctx, true);

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

        im.x_state    = nt2(GGML_TYPE_F32, im.D, im.HC);
        im.routed_sum = nt1(GGML_TYPE_F32, im.D);
        im.i_tid      = nt1(GGML_TYPE_I32, 1);
        im.i_emb      = nt1(GGML_TYPE_F32, im.D);
        im.i_pos      = in1(GGML_TYPE_I32, 1);

        im.ly.resize((size_t) nl);
        for (int64_t il = 0; il < nl; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            L.ratio = (size_t) il < g.compress_ratios.size() ? g.compress_ratios[(size_t) il] : 0;
            L.csa = (L.ratio == 4);
            if (L.ratio != 0 && L.ratio != 4 && L.ratio != 128) { err = "unsupported compress ratio"; return false; }
            any_csa = any_csa || L.csa;
            L.ring = L.csa ? 2 * L.ratio : L.ratio;
            L.sdim = L.csa ? 2 * im.DH : im.DH;
            const int64_t by_ctx = L.ratio != 0 ? std::max<int64_t>(1, g.context_length / L.ratio) : 0;
            L.comp_max = cfg.comp_cap_max > 0 ? std::min<int64_t>(cfg.comp_cap_max, by_ctx) : by_ctx;

            L.raw     = nt2(GGML_TYPE_F32, im.DH, im.SWA);
            L.hap     = nt2(GGML_TYPE_F32, im.D, im.HC);
            L.post_f  = nt1(GGML_TYPE_F32, im.HC);
            L.comb_f  = nt2(GGML_TYPE_F32, im.HC, im.HC);
            L.shexp   = nt1(GGML_TYPE_F32, im.D);
            L.fn      = on1(GGML_TYPE_F32, im.D);
            L.o_ids   = on1(GGML_TYPE_I32, im.NUSED);
            L.o_wts   = on1(GGML_TYPE_F32, im.NUSED);
            L.t_attn_raw = nt1(GGML_TYPE_F32, im.NH * im.DH);
            L.t_attn  = nt1(GGML_TYPE_F32, im.D);
            L.t_llast = nt2(GGML_TYPE_F32, im.D, im.HC);

            L.i_slot_raw  = in1(GGML_TYPE_I32, 1);
            L.i_idx_raw   = in1(GGML_TYPE_I32, im.SWA);
            L.i_mask_raw  = in1(GGML_TYPE_F16, im.SWA);
            L.i_slot_comp = in1(GGML_TYPE_I32, 1);
            L.i_slot_state = in1(GGML_TYPE_I32, 1);
            L.i_comp_pos  = in1(GGML_TYPE_I32, 1);
            L.i_state_pos = in1(GGML_TYPE_I32, 1);
            L.i_idx_state = in1(GGML_TYPE_I32, std::max<int64_t>(1, L.ring));
            if (L.ratio != 0) L.vis_full = in1(GGML_TYPE_F32, L.comp_max);
            L.rbias = in1(GGML_TYPE_F32, im.NEXP);

            if (L.ratio != 0) {
                L.comp  = nt2(GGML_TYPE_F32, im.DH, L.comp_max);
                L.st_kv = nt2(GGML_TYPE_F32, L.sdim, L.ring);
                L.st_sc = nt2(GGML_TYPE_F32, L.sdim, L.ring);
                if (L.csa) {
                    L.icomp  = nt2(GGML_TYPE_F32, im.IDXK, L.comp_max);
                    L.ist_kv = nt2(GGML_TYPE_F32, 2 * im.IDXK, 2 * L.ratio);
                    L.ist_sc = nt2(GGML_TYPE_F32, 2 * im.IDXK, 2 * L.ratio);
                }
            }
            // every layer (ratio-0 sliding-window layers too): ffn_norm is the expert tier's input
            L.host_fn.resize((size_t) im.D);
            if (cfg.gate_taps) {   // the gate's taps: ~10 MB of device->host copies per token, off by default
                L.host_attn_raw.resize((size_t) (im.NH * im.DH));
                L.host_attn.resize((size_t) im.D);
                L.host_llast.resize((size_t) (im.D * im.HC));
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
        size_t out_bytes = 0;
        const size_t blk = up(ggml_row_size(GGML_TYPE_F32, im.D)) + up(4 * im.NUSED) + up(4 * im.NUSED);
        out_bytes = blk * (size_t) nl;
        im.obuf = ggml_backend_buft_alloc_buffer(buft, out_bytes + al);
        if (!im.obuf) { err = "cannot allocate the router output blocks"; return false; }
        {
            uint8_t * ob = (uint8_t *) ggml_backend_buffer_get_base(im.obuf);
            for (int64_t il = 0; il < nl; ++il) {
                Impl::Layer & L = im.ly[(size_t) il];
                uint8_t * b0 = ob + (size_t) il * blk;
                const size_t o1 = up(ggml_nbytes(L.fn)), o2 = o1 + up(ggml_nbytes(L.o_ids));
                if (ggml_backend_tensor_alloc(im.obuf, L.fn, b0) != GGML_STATUS_SUCCESS ||
                    ggml_backend_tensor_alloc(im.obuf, L.o_ids, b0 + o1) != GGML_STATUS_SUCCESS ||
                    ggml_backend_tensor_alloc(im.obuf, L.o_wts, b0 + o2) != GGML_STATUS_SUCCESS) {
                    err = "cannot place a router output block"; return false;
                }
                L.o_span = ggml_new_tensor_1d(im.ictx, GGML_TYPE_I8, (int64_t) (o2 + ggml_nbytes(L.o_wts)));
                if (ggml_backend_tensor_alloc(im.obuf, L.o_span, b0) != GGML_STATUS_SUCCESS) {
                    err = "cannot place a router output span"; return false;
                }
                L.host_out.assign((size_t) ggml_nbytes(L.o_span), 0);
            }
        }

        // compressor score rings start at SENTINEL: a token that was never written gets weight 0, exactly
        // like ds4_ref's zero/-inf row appended for the first block's missing previous half
        std::vector<float> sent;
        for (Impl::Layer & L : im.ly) {
            if (L.ratio == 0) continue;
            sent.assign((size_t) (L.sdim * L.ring), SENTINEL);
            ggml_backend_tensor_set(L.st_sc, sent.data(), 0, sent.size() * 4);
            if (L.csa) {
                sent.assign((size_t) (2 * im.IDXK * 2 * L.ratio), SENTINEL);
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

        im.gf_init = build_init(im);
        im.allo_init = im.make_allo();
        im.gf_head = build_head(im);
        im.allo_head = im.make_allo();
        if (!im.allo_init || !im.allo_head) { err = "ggml_gallocr_new failed"; return false; }
        for (int64_t il = 0; il < nl; ++il) {
            Impl::Layer & L = im.ly[(size_t) il];
            L.gf_predict = build_predict(im, (int) il, L);
            L.allo_predict = im.make_allo();
            L.gf_finish  = build_finish(im, (int) il, L);
            L.allo_finish = im.make_allo();
            if (!L.allo_predict || !L.allo_finish) { err = "ggml_gallocr_new failed"; return false; }
            build_attn(im, (int) il, 0, L);   // cap 0; larger capacities are built on demand
        }
    }
    im.host_logits.resize((size_t) g.vocab_size);
    return true;
}

const strata::Ds4Geometry & Ds4Dense::geom()    const { return p_->g; }
int64_t                     Ds4Dense::n_layer() const { return (int64_t) p_->ly.size(); }
ggml_backend_t              Ds4Dense::backend() const { return p_->backend; }
const std::string &         Ds4Dense::last_error() const { return p_->err; }
ggml_tensor *               Ds4Dense::weight(const std::string & name) const { return p_->w.get(name); }

bool Ds4Dense::begin_token(int tid) {
    Impl & im = *p_;
    if (!ggml_gallocr_alloc_graph(im.allo_init, im.gf_init)) { im.err = "gallocr(init) failed"; return false; }
    const int32_t t = tid;
    ggml_backend_tensor_set(im.i_tid, &t, 0, sizeof t);
    im.cur_tid = tid;
    // embd[tid] on the host, exactly what get_rows + cast computes (F16 -> F32 is exact; quantized rows dequantize
    // with the same to_float)
    if (tid < 0 || tid >= im.g.vocab_size) { im.err = "token id out of range"; return false; }
    im.host_emb.resize((size_t) im.D);
    const uint8_t * row = im.embd_data + (size_t) tid * im.embd_row;
    if (im.embd_type == GGML_TYPE_F32) std::memcpy(im.host_emb.data(), row, (size_t) im.D * 4);
    else ggml_get_type_traits((ggml_type) im.embd_type)->to_float(row, im.host_emb.data(), im.D);
    ggml_backend_tensor_set(im.i_emb, im.host_emb.data(), 0, (size_t) im.D * 4);
    if (ggml_backend_graph_compute(im.backend, im.gf_init) != GGML_STATUS_SUCCESS) {
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

    if (!ggml_gallocr_alloc_graph(L.allo_predict, L.gf_predict)) { im.err = "gallocr(predict) failed"; return false; }
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
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "attn_router: bad layer"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    if (pos < 0) { im.err = "attn_router: negative position"; return false; }

    // pick the graph variant for the compressed-row capacity this position needs, and allocate its arena
    Impl::Var * var = nullptr;
    int64_t b = 0, cap = 0, n_vis = 0;
    if (L.ratio != 0) {
        b = pos / L.ratio;
        if (b >= L.comp_max) { im.err = "attn_router: position beyond the compressed cache capacity"; return false; }
        cap = std::min<int64_t>(L.comp_max, next_pow2(b + 1));
        n_vis = (pos + 1) / L.ratio;
        for (Impl::Var & v : L.vars)
            if (v.cap == cap) { var = &v; break; }
        if (!var) { build_attn(im, il, cap, L); var = &L.vars.back(); }
    } else {
        for (Impl::Var & v : L.vars)
            if (v.cap == 0) { var = &v; break; }
        if (!var) { build_attn(im, il, 0, L); var = &L.vars.back(); }
    }
    if (!ggml_gallocr_alloc_graph(var->allo, var->gf)) { im.err = "gallocr(attn) failed"; return false; }

    // ---- per-step inputs: every layer's, for this position, in ONE upload per token ------------
    if (tid >= 0 && tid != im.cur_tid) {
        const int32_t t32 = tid;
        ggml_backend_tensor_set(im.i_tid, &t32, 0, sizeof t32);
        im.cur_tid = tid;
    }
    if (pos != im.in_pos) {
        auto put = [&](ggml_tensor * t, const void * src, size_t n) {
            std::memcpy(im.in_host.data() + ((const uint8_t *) t->data - im.in_base), src, n);
        };
        const int32_t p32 = pos, s32 = (int32_t) (pos % im.SWA);
        put(im.i_pos, &p32, sizeof p32);
        std::vector<int32_t> idx((size_t) im.SWA);
        std::vector<ggml_fp16_t> msk((size_t) im.SWA);
        for (int64_t i = 0; i < im.SWA; ++i) {
            const int64_t tok = (int64_t) pos - im.SWA + 1 + i;
            idx[(size_t) i] = (int32_t) (((tok % im.SWA) + im.SWA) % im.SWA);
            msk[(size_t) i] = ggml_fp32_to_fp16(tok >= 0 ? 0.0f : NEG_INF);
        }
        std::vector<int32_t> sidx;
        std::vector<float> vis;
        for (Impl::Layer & Ly : im.ly) {
            put(Ly.i_slot_raw, &s32, sizeof s32);
            put(Ly.i_idx_raw, idx.data(), idx.size() * 4);
            put(Ly.i_mask_raw, msk.data(), msk.size() * 2);
            if (Ly.ratio == 0) continue;
            const int64_t bl = pos / Ly.ratio;
            if (bl >= Ly.comp_max) continue;   // attn_router refuses this layer below
            const int64_t cp_ = std::min<int64_t>(Ly.comp_max, next_pow2(bl + 1));
            const int64_t nv = (pos + 1) / Ly.ratio;
            const int32_t sp = (int32_t) (pos % Ly.ratio);      // compressor `ape` row
            const int32_t cpp = (int32_t) (Ly.ratio * bl);      // block start, for the compressed-key RoPE
            const int32_t sc = (int32_t) bl;                    // block slot in the compressed caches
            const int32_t ss = (int32_t) (pos % Ly.ring);       // token slot in the compressor state ring
            put(Ly.i_state_pos, &sp, sizeof sp);
            put(Ly.i_comp_pos, &cpp, sizeof cpp);
            put(Ly.i_slot_comp, &sc, sizeof sc);
            put(Ly.i_slot_state, &ss, sizeof ss);
            sidx.resize((size_t) Ly.ring);
            for (int64_t i = 0; i < Ly.ring; ++i) {
                const int64_t tok = (int64_t) pos - Ly.ring + 1 + i;
                sidx[(size_t) i] = (int32_t) (((tok % Ly.ring) + Ly.ring) % Ly.ring);
            }
            put(Ly.i_idx_state, sidx.data(), sidx.size() * 4);
            vis.assign((size_t) cp_, NEG_INF);
            for (int64_t i = 0; i < nv && i < cp_; ++i) vis[(size_t) i] = 0.0f;
            put(Ly.vis_full, vis.data(), vis.size() * 4);
        }
        ggml_backend_tensor_set(im.i_span, im.in_host.data(), 0, im.in_host.size());
        im.in_pos = pos;
    }

    if (ggml_backend_graph_compute(im.backend, var->gf) != GGML_STATUS_SUCCESS) {
        im.err = "attn graph compute failed";
        return false;
    }

    static const bool dbg_idx = std::getenv("DS4_DBG_INDEXER") != nullptr;
    if (dbg_idx && var->dbg_isc) {
        std::vector<float> v((size_t) var->cap);
        ggml_backend_tensor_get(var->dbg_isc, v.data(), 0, v.size() * 4);
        std::fprintf(stderr, "[idx] L%d pos %d:", il, pos);
        for (float x : v) if (std::isfinite(x)) std::fprintf(stderr, " %.6g", x);
        std::fprintf(stderr, "\n");
    }
    // ---- router outputs + hand-offs -----------------------------------------------------
    {
        ggml_backend_tensor_get(L.o_span, L.host_out.data(), 0, L.host_out.size());   // [fn | ids | wts]
        const uint8_t * b0 = (const uint8_t *) L.fn->data;
        const int32_t * ids = (const int32_t *) (L.host_out.data() + ((const uint8_t *) L.o_ids->data - b0));
        const float * wv = (const float *) (L.host_out.data() + ((const uint8_t *) L.o_wts->data - b0));
        for (int64_t i = 0; i < im.NUSED; ++i) { routed_ids[i] = ids[i]; routed_w[i] = wv[i]; }
        std::memcpy(L.host_fn.data(), L.host_out.data(), (size_t) im.D * 4);
    }
    if (!L.host_attn_raw.empty()) ggml_backend_tensor_get(L.t_attn_raw, L.host_attn_raw.data(), 0, L.host_attn_raw.size() * 4);
    if (!L.host_attn.empty()) ggml_backend_tensor_get(L.t_attn, L.host_attn.data(), 0, L.host_attn.size() * 4);

    if (ffn_norm_dev)  *ffn_norm_dev  = L.fn;
    if (ffn_norm_host) *ffn_norm_host = L.host_fn.empty() ? nullptr : L.host_fn.data();
    L.have_attn = true;
    return true;
}

void Ds4Dense::set_route_bias(int il, const float * bias) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size() || !im.ly[(size_t) il].rbias) return;
    Impl::Layer & L = im.ly[(size_t) il];
    std::memcpy(im.in_host.data() + ((const uint8_t *) L.rbias->data - im.in_base), bias, (size_t) im.NEXP * 4);
    im.in_pos = -1;   // re-upload the span before the next attention graph
}

bool Ds4Dense::finish_layer(int il, const float * routed_sum) {
    Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) { im.err = "finish_layer: bad layer"; return false; }
    Impl::Layer & L = im.ly[(size_t) il];
    if (!ggml_gallocr_alloc_graph(L.allo_finish, L.gf_finish)) { im.err = "gallocr(finish) failed"; return false; }
    ggml_backend_tensor_set(im.routed_sum, routed_sum, 0, (size_t) im.D * 4);
    if (ggml_backend_graph_compute(im.backend, L.gf_finish) != GGML_STATUS_SUCCESS) {
        im.err = "finish graph compute failed";
        return false;
    }
    if (!L.host_llast.empty()) ggml_backend_tensor_get(L.t_llast, L.host_llast.data(), 0, L.host_llast.size() * 4);
    L.have_finish = true;
    return true;
}

bool Ds4Dense::logits(const float ** out, int * n_vocab) {
    Impl & im = *p_;
    if (!ggml_gallocr_alloc_graph(im.allo_head, im.gf_head)) { im.err = "gallocr(head) failed"; return false; }
    if (ggml_backend_graph_compute(im.backend, im.gf_head) != GGML_STATUS_SUCCESS) {
        im.err = "head graph compute failed";
        return false;
    }
    ggml_backend_tensor_get(im.logits_t, im.host_logits.data(), 0, im.host_logits.size() * 4);
    if (out) *out = im.host_logits.data();
    if (n_vocab) *n_vocab = (int) im.host_logits.size();
    return true;
}

const float * Ds4Dense::tap_attn_out(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_attn ? L.host_attn.data() : nullptr;
}

const float * Ds4Dense::tap_attn_raw(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_attn ? L.host_attn_raw.data() : nullptr;
}

const float * Ds4Dense::tap_l_last(int il) const {
    const Impl & im = *p_;
    if (il < 0 || il >= (int) im.ly.size()) return nullptr;
    const Impl::Layer & L = im.ly[(size_t) il];
    return L.have_finish ? L.host_llast.data() : nullptr;
}
