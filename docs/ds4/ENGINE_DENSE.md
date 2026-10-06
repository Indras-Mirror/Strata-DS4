# Ds4Dense - the DS4 dense (non-expert) half, one token at a time

`tools/ds4/ds4_dense.{hpp,cpp}` implements the Phase-4b decode engine for the non-expert part of
DeepSeek-V4-Flash: the hyper-connection streams, attention (raw sliding window + CSA/HCA compressed
keys + the lightning indexer), the MoE router, the shared expert and the head.  One instance decodes
one sequence, one new token per call, against persistent state - the same way a KV-cache decode works
in llama.cpp.  The routed experts live in a *separate* tier (slice `ds4-moe`); this class hands out the
`ffn_norm` activation they consume and takes their weighted sum back.

The math is `tools/ds4/ds4_ref.cpp` (the Phase-3-verified CPU reference forward, which matches
llama.cpp's logits - `FINDINGS.md` s9) with the reference's whole-sequence tensors narrowed to the
*visible* keys at the current position.  The backend is a `ggml_backend_t` chosen at `init` (CPU or
CUDA, identical code path); every per-layer graph kind is built once and reused, which is what lets
ggml-cuda capture the decode graphs as CUDA graphs.

## API (`ds4_dense.hpp`)

| call | what it does |
| --- | --- |
| `init(model_path, cfg, err)` | open the GGUF, bind the weights zero-copy on CPU (upload off-CPU), allocate the persistent state, build the graphs |
| `begin_token(tid)` | `hc_init` = `embd[tid]` repeated over the `hc` streams, into the layer-0 input state |
| `predict(il, top_ids, n_top)` | prefetch hint "A": the layer router applied to `rms_norm(hc_attn_pre_l) * ffn_norm.weight_l`, `sqrt(softplus(.)) + exp_probs_b`, ranked top-`n_top`; hash layers return their exact `tid2eid[tid]` |
| `attn_router(il, pos, tid, ids, wts, &fn_host, &fn_dev)` | attention + hc_post(attn) + hc_pre(ffn) + ffn_norm + router (`n_expert_used` ids and final weights) + the shared expert |
| `finish_layer(il, routed_sum)` | `ffn_out = shexp + routed_sum`, then `hc_post` -> the next layer state |
| `logits(&out, &n_vocab)` | `hc_head` -> `output_norm` -> `output` |
| `tap_attn_raw/attn_out/l_last(il)` | host copies for the gate (pre output-LoRA attention, post-LoRA attention, `l_last`) |

`predict` is a hint only: it does not change the decode result.  `attn_router`'s `routed_w` is already
normalised, `x weights_scale`, the way llama.cpp and `ds4_ref` use it.

## Persistent state (`ggml_context` `sctx`, one backend buffer, zeroed, then seeded)

Per layer `l`, ratio `r = compress_ratios[l]`, `csa = (r == 4)`:

* **raw sliding window** `raw [d_head, n_swa]` - post-RoPE K (= V).  Token `p` writes slot `p % n_swa`;
  the gather list is the window oldest -> newest, so the visible keys keep `ds4_ref`'s order; tokens
  `< 0` land on unwritten slots and are masked `-inf`.
* **compressed block cache** `comp [d_head, comp_max]` - slot `p / r`.  The row is rewritten on every
  position of the block and only becomes *visible* once the block is complete
  (`(p + 1) / r` blocks visible), so a partial row is never attended to.
* **compressor state ring** `st_kv/st_sc [sdim, ring]`, `ring = 2*r` (CSA, overlap: half the state is
  the "prev half" of the block, half the "cur half") or `r` (HCA); `sdim = 2*d_head` / `d_head`.
  Token `p` writes slot **`p % ring`** (llama-kv-cache-dsv4: `stream_off + pos%state_size`), and the
  gather list `i_idx_state` lists the last `ring` slots oldest -> newest, which is exactly the
  `2*r` states of block `b-1` followed by block `b` when block `b` completes.  `st_sc` starts at
  `SENTINEL = -1e30` so rows never written get softmax weight 0 - the same as the zero/`-inf` row the
  reference appends for the first block's missing prev half.
* **lightning indexer** (CSA only) `icomp [idx_k, comp_max]` (compressed keys, post-Walsh-Hadamard),
  `ist_kv/ist_sc [2*idx_k, 2*r]`.  The indexer queries `cap` block rows, `relu`s and weights by
  `indexer.proj`, and `top_k` the visible ones; `min(cap, top_k)` rows are unmasked, always `&`-ed with
  the visibility mask so a top-k hit on an invisible block stays hidden.
* **hand-offs to `finish_layer`** `hap`, `post_f`, `comb_f`, `shexp`, `fn`, and the gate taps
  `t_attn_raw`/`t_attn`/`t_llast`.  `x_state [D, hc]` carries the hc stream between layers and between
  tokens; `routed_sum [D]` is the expert tier's input.

`comp_max = context_length / r` (or the `comp_cap_max` override).

## Graphs

`init`, `head`, `predict` and `finish` are one graph each per layer; attention is built **per capacity**
`cap = min(comp_max, next_pow2(p/r + 1))` and the matching variant is picked per position (cap 0 for
non-compressed layers).  Larger caps are built on demand and cached in `Layer::vars`.

**Every graph owns its own `ggml_gallocr`.**  `ggml-alloc` binds a tensor's data pointer once and skips
re-binding it later, while a reserve triggered by a *different* graph frees and reallocates the chunks
it had bound - one shared allocator therefore leaves every other graph's tensors dangling into freed
memory as soon as one graph grows.  Per-graph arenas also give each decode graph the stable, dedicated
buffer CUDA graph capture needs.

Per-step values arrive through input leaves (`i_pos`, `i_tid`, `i_slot_raw`, `i_idx_raw`, `i_mask_raw`,
`i_slot_state`, `i_slot_comp`, `i_comp_pos`, `i_state_pos`, `i_idx_state`, the per-variant `vis`), so the
graphs themselves never change shape.

## Verification status

`test_ds4_dense <mini.gguf> [--tokens N] [--work DIR] [--ref-dir DIR] [--ref-bin PATH]` (CPU only) runs
`ds4_ref --all-pos` on the same tokens, decodes them one at a time, and compares per-layer
`attn_raw`/`attn_out`/`ffn_norm`/`l_last` at the last position plus the logits at every position.

Measured against `ds4_ref` on the packet's mini fixture (`bench/ds4-2026-10-06/dense/mini-ds4.gguf`):

* **`--tokens 63` -> PASS, every cosine exactly 1.00000000** (all 63 positions' logits, both layers'
  taps).  The dense's op sequence *is* the reference's: CSA blocks 0..14, the lightning-indexer top-k
  (`8 of 15`), the raw window, the hyper-connections, the shared expert and the head are all bit-equal.
* **`DS4_SWA=16` variant, `--tokens 40` -> PASS, every cosine exactly 1.00000000** - the same gate,
  but `sliding_window = 16` (metadata only, no tensor shape depends on it), so the raw-window *wrap*
  happens from position 16 instead of 129 and the whole run stays inside the reference's stable token
  range.  This is the run that gates the ring wrap, the negative-position masking, 10 CSA blocks and
  the indexer top-k bit-exactly.
* `--tokens >= 64` fails - **because the reference itself moves**: with `q`, `kv` and the query-0 mask row
  dumped and compared (`DS4_PROBE=1`), they are *bit-identical* between a 63 and a 64 token run, while
  the flash-attn output for query 0 changes by 1.8e-4 relative.  ggml-cpu's `flash_attn_ext` is not
  invariant to the query count (`ne01 >= 64` takes a different kernel path); a batch-1 decoder cannot
  reproduce that, and `-t 1` vs `-t 8` gives byte-identical reference dumps, so it is not threading.
  The stock fixture's unit-scale random weights give the net a ~1e3-1e4 gain (sigmoid/softmax
  saturation, then the sinkhorn `hc_eps` dominating), which turns that 1.8e-4 into 1.6e-2 at the head -
  cos 0.9999 at position 0 and 0.75 by position 61.
* `bench/ds4-2026-10-06/dense/make_mini_tame.py` writes the same geometry/tensor set with the real
  model's `1/sqrt(fan_in)` weight convention.  On it the amplification is gone (`attn_raw`,
  `ffn_norm`, `l_last` >= 0.99999 at the last position, all logits cos 0.99987..0.99995, top-1
  identical, predictor recall@16 1.000): the residual is the reference-side query-count artifact
  propagated, uniform over positions, and still ~3x above the gate's 0.99999 on `attn_out`/logits.

Residuals: the same gate cannot be met for `--tokens >= 129` (which the packet requires, for the raw
wrap, the HCA block and the indexer top-k) on *any* token-by-token decoder, because the raw-window wrap
and the ratio-128 HCA block only become reachable at `>= 128` tokens, i.e. inside the unstable regime.
The ratio-128 HCA *visible* block is the one path left un-gated: it only becomes visible at position
127, so its value (compressed-key rope/norm, softmax over the 128 state rows) is exercised with
`cap = 1` but output-masked, and is validated by construction (shared code with the CSA path) only.

## Backend / CUDA tree

`Ds4DenseConfig::backend` takes any `ggml_backend_t`; the graph builders are backend agnostic and take
no CPU-specific path (weights upload instead of binding onto the mmap when the backend is not CPU).
`build-ds4-cuda` compiles and links `libds4_dense.a` + `test_ds4_dense` - but note `GGML_CUDA=OFF`
there (that tree builds Strata's own `strata_kernels`/`strata_core` `.cu` files, not ggml's CUDA
backend), so the CUDA tree currently links no `ggml-cuda` and the test, by construction, only ever
creates a CPU backend.  To actually *run* the dense on the GPU the GPU round needs: a ggml tree
configured with `-DGGML_CUDA=ON` (and `libggml-cuda` linked, which `tools/ds4/cmake/ds4_dense.cmake`
already does when a `ggml-cuda` target exists), plus a backend selector in `test_ds4_dense` (today it
hardcodes CPU on purpose - the safety rules for this machine forbid GPU runs).
