// tools/ds4/ds4_dense.hpp - DeepSeek-V4 (`deepseek4`) single-token DECODE of the dense (non-expert) half.
//
// Phase 4b of the Strata DS4 engine: one new token per call, at position `pos`, against persistent decode
// state (raw sliding-window ring, CSA/HCA compressor rings + compressed K caches, lightning-indexer cache).
// The routed-expert tier is NOT here (slice ds4-moe); this class hands out the ffn_norm activation the
// experts consume and takes back their weighted sum in `finish_layer`.
//
//   begin_token(tid)                     -> hc_init, the layer-0 input state
//   predict(l, top_ids, n_top)           -> prefetch hint: the layer's router on rms_norm(hc_attn_pre)*ffn_norm.w
//   attn_router(l, pos, tid, ...)        -> attention (raw + visible compressed + indexer top-k), MoE router
//                                           ids/weights, the shared expert, and ffn_norm as host f32 + device ptr
//   finish_layer(l, routed_sum)          -> shared expert + routed sum, hc_post -> next layer state
//   logits(...)                          -> hc_head, output_norm, output
//
// Multi-token passes (speculative verify): the *_n calls run n <= max_tokens() consecutive positions pos0..pos0+n-1
// through the same steps in one graph per step; each token attends exactly what it would one token at a time (the
// raw window and compressed blocks up to its own position, including blocks completed earlier in the pass).
// All decode state is position-indexed, so a rejected draft is undone by decoding its position again.
//
// The invariant (and the gate in test_ds4_dense.cpp): token-by-token decode through Ds4Dense plus a plain
// reference routed-expert computation reproduces ds4_ref's full-sequence forward on the same tokens.
//
// State layout, the fixed-shape graph variants and the API are documented in docs/ds4/ENGINE_DENSE.md.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "strata/artifact/ds4_geometry.hpp"

#include <memory>
#include <string>

/// Construction parameters.  `backend` selects the compute backend (CPU or CUDA - the same code path);
/// NULL means "create and own a CPU backend".  `comp_cap_max` bounds the compressed-row capacity per layer
/// (0 = `deepseek4.context_length / ratio`, the model's own maximum).
struct Ds4DenseConfig {
    ggml_backend_t backend   = nullptr;
    int            n_threads = 8;
    int64_t        comp_cap_max = 0;           ///< compressed-KV rows per layer (0 = context_length / ratio)
    /// Do not load ffn_{gate,up,down}_exps (the routed-expert tier owns them): the real model on the GPU, where
    /// uploading them would ask for ~72 GiB.  `weight()` of one of them then aborts.
    bool           skip_routed_experts = false;
    /// Keep host copies of attn_raw / attn_out / l_last per layer for tap_*() (test_ds4_dense); off = no copies.
    bool           gate_taps = false;
    /// Off-CPU only: requantize the big Q8_0 dense matrices (attn_q_b, attn_output_a/b, shared expert, output) to this
    /// ggml type at load (e.g. GGML_TYPE_Q4_K) - ~3 GiB of VRAM back for the expert cache.  -1 = keep the file's types.
    int            requant_type = -1;
    /// The MTP (nextn) head's own GGUF (DeepSeek-V4-Flash-MTP-*.gguf): loads its block as layer n_layer() (an extra
    /// ratio-0 sliding-window layer, see mtp_begin).  Empty = no MTP head.
    std::string    mtp_path;
    /// Prompt chunks: the most tokens one prefill_* pass may carry (0 = no chunked prefill).  The chunk's hand-off
    /// tensors and graph buffers are allocated at the first prefill_begin and freed by prefill_release.
    int64_t        prefill_chunk = 0;
    /// Off-CPU: the compressor / indexer F16 matrices as Q8_0 (router and hyper-connection mixers stay F16).
    bool           f16_q8 = false;
};

/// One-token decode of the DS4 dense half.  Not thread-safe; one instance decodes one sequence.
class Ds4Dense {
public:
    struct Impl;   // opaque, defined in the .cpp (public so the file-local graph builders can see it)

    Ds4Dense();
    ~Ds4Dense();

    Ds4Dense(const Ds4Dense &)            = delete;
    Ds4Dense & operator=(const Ds4Dense &) = delete;

    /// Open the GGUF (metadata + tensor map, zero-copy weight bindings), allocate the persistent decode
    /// state and build the per-layer graphs.  Returns false and fills `err` on failure.
    bool init(const std::string & model_path, const Ds4DenseConfig & cfg, std::string & err);

    const strata::Ds4Geometry & geom()     const;
    int64_t                     n_layer()  const;
    int                         max_tokens() const;   ///< tokens a multi-token pass may carry
    /// The MTP block's layer index (== n_layer()) when an MTP head is loaded, else -1.  Drafting a token at position
    /// p: after the trunk's last finish_layer and logits for p, mtp_begin(&tok[p+1], 1), then the usual
    /// attn_router(_n) / routed experts / finish_layer(_n) with il = mtp_layer() at position p, then mtp_logits_n()
    /// predicts token p+2.  The MTP block keeps its own sliding-window KV, written at position p.
    int                         mtp_layer() const;
    ggml_backend_t              backend()  const;
    /// Borrowed weight tensor (zero-copy on the CPU backend).  Aborts on an unknown name.
    ggml_tensor *               weight(const std::string & name) const;

    /// Load the embedding for `tid` into the layer-0 input state (hc_init).  Also records the token id for
    /// the hash-routed layers.  Call once per token, before `predict(0, ...)`.
    bool begin_token(int tid);
    /// begin_token for a pass of `n` tokens (consecutive positions).  predict() then hints for tokens[0].
    bool begin_tokens(const int * tids, int n);

    /// Expert-prefetch predictor ("A" in tools/ds4/predict_experts.py): the layer's router applied to
    /// rms_norm(hc_attn_pre_l) * ffn_norm.weight_l.  Hash layers (l < hash_layer_count) return their exact
    /// tid2eid ids in the first n_expert_used slots.  `top_ids` holds `n_top` entries (n_top <= 16);
    /// unused slots are set to -1.  A hint only: it does not change the decode result.
    bool predict(int il, int * top_ids, int n_top);

    /// Attention for layer `il` at position `pos` + the layer's MoE router + the shared expert.
    /// `routed_ids`/`routed_w` must hold n_expert_used entries; `routed_w` is the final normalised,
    /// x weights_scale weight as llama.cpp/ds4_ref use it.  `ffn_norm_host` points at the [n_embd] f32
    /// activation the routed experts consume (`ffn_norm_dev` is the same tensor on the backend).
    bool attn_router(int il, int pos, int tid,
                     int * routed_ids, float * routed_w,
                     const float ** ffn_norm_host, ggml_tensor ** ffn_norm_dev);
    /// attn_router for the pass begun by begin_tokens(tids, n), positions pos0..pos0+n-1.  `routed_ids`/`routed_w`
    /// hold n * n_expert_used entries (token-major); `*ffn_norm_host` points at n * n_embd floats (token-major).
    bool attn_router_n(int il, int pos0, int n, int * routed_ids, float * routed_w, const float ** ffn_norm_host);

    /// Add the routed-expert sum (n_embd f32, host) to the shared expert held from `attn_router` and run the
    /// ffn hyper-connection into the next layer state.
    bool finish_layer(int il, const float * routed_sum);
    /// finish_layer for an n-token pass: `routed_sum` is n * n_embd floats (token-major).
    bool finish_layer_n(int il, int n, const float * routed_sum);

    /// Head for the current state: hc_head -> output_norm -> output.  Valid after the last layer's
    /// `finish_layer`.  `*out` is a borrowed [n_vocab] f32 host buffer.
    bool logits(const float ** out, int * n_vocab);
    /// Head for every token of an n-token pass: `*out` is n * n_vocab floats (token-major).
    bool logits_n(int n, const float ** out, int * n_vocab);
    /// MTP input for n tokens: replaces the state (the trunk's final hc streams) with eh_proj(enorm(embd[next_tids]),
    /// hnorm(state)).  Call after the trunk's logits - they read the same state.
    bool mtp_begin(const int * next_tids, int n);
    /// MTP head (hc_head -> nextn.shared_head_norm -> output) after the MTP block's finish_layer(_n).
    bool mtp_logits_n(int n, const float ** out, int * n_vocab);

    /// Diagnostics for the gate, all as computed on the last `attn_router`/`finish_layer` call (host copies):
    /// the flash-attn output before the output LoRA [n_head*d_head], the attention output after it [n_embd],
    /// and l_last [n_embd*hc], of the last token of the pass.  NULL when unavailable.
    /// Cache-aware routing: `bias` (n_expert floats) is added to layer `il`'s expert SELECTION score (not to the mixing
    /// weights) from the next attention graph on.  Zeros = the model's routing.  Hash layers ignore it.
    void set_route_bias(int il, const float * bias);
    /// Build and allocate every attention-graph variant up to the context cap (and the per-layer predict/finish graphs)
    /// now, so the VRAM they need is taken at load - not when the context first reaches a new capacity mid-run.
    /// `n_max` > 1 also reserves the multi-token variants up to that pass size.
    bool reserve_graphs(int n_max = 1);
    // ---- prompt chunks: n <= Ds4DenseConfig::prefill_chunk tokens at positions pos0 .. pos0+n-1 ----
    //   prefill_begin(tids, n, pos0)
    //   for each layer il: prefill_attn(il, ids, w, &fn)   [ids/w n*n_expert_used token-major, fn n*n_embd]
    //                      prefill_finish(il, routed)       [routed n*n_embd]
    //   prefill_logits(row0, nrows, out) for any rows of the chunk, then prefill_end(); decoding continues at pos0+n.
    // Same math as n one-token passes (the decode graphs with n columns); each graph is built for the chunk, run
    // once on a shared allocator and dropped.  The rings keep their decode size: a chunk reads [ring | its own rows]
    // and writes back its last rows.  Not with an MTP head (its window is not filled here).
    bool prefill_begin(const int * tids, int n, int pos0);
    bool prefill_attn(int il, int * routed_ids, float * routed_w, const float ** ffn_norm_host);
    bool prefill_finish(int il, const float * routed_sum);
    /// Logits of rows row0 .. row0+nrows-1 of the chunk into `out` (nrows * n_vocab floats, row-major).
    bool prefill_logits(int row0, int nrows, float * out);
    bool prefill_end();
    /// Frees the chunk tensors and the shared graph buffer (VRAM back for the expert cache).
    void prefill_release();
    /// A host buffer of prefill_chunk * n_embd floats for the routed sums prefill_finish takes - pinned on CUDA, so
    /// the expert tier writes and prefill_finish uploads without pageable staging.  Valid until prefill_release.
    float * prefill_routed_buffer();
    /// The chunk's routed-sum tensor on the device (prefill_chunk * n_embd floats, token-major), or NULL on the CPU
    /// backend.  An expert tier on the same device can write the sums there; prefill_finish(il, nullptr) then skips
    /// the host upload (which would queue behind the tier's expert DMAs on the copy engine).
    void * prefill_routed_device();
    const float * tap_attn_raw(int il) const;
    const float * tap_attn_out(int il) const;
    const float * tap_l_last  (int il) const;

    const std::string & last_error() const;

private:
    std::unique_ptr<Impl> p_;
};
