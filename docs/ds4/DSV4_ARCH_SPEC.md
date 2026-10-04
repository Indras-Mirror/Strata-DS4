# DeepSeek-V4 (GGUF arch `deepseek4`) — Exact Architecture Spec

Sources:
- `/home/mal/AI/llama.cpp-new/src/models/deepseek4.cpp` (graph + tensor loader)
- `/home/mal/AI/llama.cpp-new/src/llama-graph.cpp` (shared FFN/MoE/attention builders)
- `/home/mal/AI/llama.cpp-new/src/llama-hparams.{h,cpp}` (hyper-params + getters)
- `/home/mal/AI/llama.cpp-new/src/llama-arch.{h,cpp}` (GGUF key map + tensor name map)
- `/home/mal/AI/llama.cpp-new/src/llama-kv-cache-dsv4.{h,cpp}` (KV/compressor caches)
- `/home/mal/AI/llama.cpp-new/src/llama-kv-cache.cpp` (Hadamard rotation machinery)
- `/home/mal/AI/llama.cpp-new/src/llama-impl.h` (`llama_mul_mat_hadamard`)
- `/home/mal/AI/llama.cpp-new/conversion/deepseek.py` (HF→GGUF converter, `DeepseekV4Model`)
- `/home/mal/AI/llama.cpp-new/conversion/base.py` (generic hparam writers)
- `/home/mal/AI/llama.cpp-new/models/templates/deepseek-ai-DeepSeek-V4*.jinja` (chat template)
- `/home/mal/AI/llama.cpp-master-rebase/src/llama-kv-cache-dsv4.{h,cpp}` (identical to the `-new` version at the time of writing; header is byte-for-byte the same — see §8)

This spec is written to support writing CUDA kernels and verifying logit parity. File:line cites are given as
`path:line`. All "per-layer" tensors use index `i ∈ [0, n_layer_all)` where `n_layer_all = num_hidden_layers +
num_nextn_predict_layers`; the first `n_layer()` (= `n_layer_all - n_layer_nextn`) blocks are the trunk, the
remaining `n_layer_nextn` blocks are MTP/NextN blocks (see §7).

Notation used throughout:

- `D = n_embd` — hidden size (per hyper-connection stream)
- `H = dsv4_hc_mult` — hyper-connection multiplicity (always **4**; asserted in code)
- `n_heads = n_head()` — number of query heads
- `d_h = n_embd_head_k()` — full q/k (and value) head dim = `qk_nope_head_dim + qk_rope_head_dim`
- `d_rope = n_rot()` — RoPE dim = `qk_rope_head_dim`
- `d_nope = d_h - d_rope` — non-positional part = `qk_nope_head_dim`
- `r_q = n_lora_q` — q low-rank dim = `q_lora_rank`

---

## 0. High-level summary

DeepSeek-V4 is a decoder-only **Transformer with hyper-connections** (a.k.a. `hc_mult = 4` residual streams).
Each transformer block is run **twice** inside one layer index: once for attention (`hc_attn_*`) and once for FFN
(`hc_ffn_*`). Between the two passes a full hyper-connection pre/post gate runs (not a plain residual add).

Attention is **MLA-style MQA with a single fused key/value latent**:

- `qr = RMSNorm(Wq_a · x)` (low-rank query latent)
- `q = Wq_b · qr`, then an **RMSNorm over the head** — note `K == V` (there is no separate value)
- `kv = Wkv · x`, then RMSNorm — this single `d_h`-vector is used as **both K and V**
- RoPE is applied only to the last `d_rope` dims of `q` and `kv` (`d_nope` dims get no positional encoding)
- attention is `softmax(q·k / √d_h + mask + sinks) · k` (the "value" is the key)
- output is **de-RoPE'd** (`rope_ext_back` on the `d_rope` slice) then projected through a grouped LoRA
  (`Wo_a` over `o_groups` groups, then `Wo_b`)

MoE FFN (every trunk layer) is **DeepSeek-V3-style top-k** with a DeepSeek-V4 twist: the first
`num_hash_layers` layers replace learned top-k **selection** with a per-token **hash table** (`tid2eid`), while
the remaining layers use the learned `gate` + `exp_probs_b` bias. The gating score is `sqrt(softplus(logits))`.
Experts are combined with normalized, scaled weights plus a parallel shared-expert FFN added afterwards.

The unique parts are (§8): (a) **hyper-connections** (4 residual streams mixed through a learned Sinkhorn
coupling matrix), (b) a **lightning indexer** (a tiny second attention head that picks top-k compressed blocks),
(c) **compressed KV** at 4:1 and 128:1 ratios with learned **APE** (absolute positional embeddings), and
(d) **Hadamard rotation** of q/k before indexer/compressed attention for score decorrelation.

---

## 1. Hyper-parameters (`llama_hparams`)

Struct: `src/llama-hparams.h:40`. DSV4-specific fields at `src/llama-hparams.h:242-250`.

### 1.1 DSV4-specific fields

| C++ field | type | meaning |
|---|---|---|
| `dsv4_o_group_count` | uint32 | `o_groups` — number of output groups for the `Wo_a` LoRA (`deepseek4.cpp:86,119`) |
| `dsv4_o_lora_rank` | uint32 | `o_lora_rank` — rank of the `Wo_b` output LoRA (`deepseek4.cpp:87,120`) |
| `dsv4_hc_mult` | uint32 | hyper-connection multiplicity `hc` (`deepseek4.cpp:89`) |
| `dsv4_hc_sinkhorn_iters` | uint32 | number of Sinkhorn row/col normalization iterations (`deepseek4.cpp:341`) |
| `dsv4_hash_layer_count` | uint32 | `num_hash_layers` — first N trunk layers use hash routing (`deepseek4.cpp:52,154`) |
| `dsv4_compress_rope_base` | float | `compress_rope_theta` — RoPE base for compressed layers (`deepseek4.cpp:48,514`) |
| `dsv4_hc_eps` | float | `hc_eps` — epsilon added in Sinkhorn / gates (`deepseek4.cpp:51,322`) |
| `dsv4_compress_ratios[]` | array[512] | per-layer compression ratio: `0` (plain SWA), `4` (CSA), `128` (HCA) (`deepseek4.cpp:61,129`) |

Plus the reused DeepSeek-V2/V3 MLA/MoE fields:

| C++ field | meaning (DSV4 use) |
|---|---|
| `n_lora_q` | `q_lora_rank` — low-rank dim of Q (`deepseek4.cpp:30,82`) |
| `n_swa` | `sliding_window` — raw-attention sliding window (`deepseek4.cpp:31`) |
| `n_ff_exp` | `moe_intermediate_size` — routed expert FFN hidden (`deepseek4.cpp:33,161`) |
| `n_expert_shared` | `n_shared_experts` — shared experts (`deepseek4.cpp:34,165`) |
| `expert_weights_scale` | `routed_scaling_factor` — MoE weight scale (`deepseek4.cpp:35`) |
| `expert_weights_norm` | `norm_topk_prob` — normalize expert weights (`deepseek4.cpp:36`) |
| `expert_gating_func` | gating op; **must be SQRT_SOFTPLUS** (`deepseek4.cpp:63-66`) |
| `swiglu_clamp_exp[]` | `swiglu_limit` per routed-expert FFN (`deepseek4.cpp:37`) |
| `swiglu_clamp_shexp[]` | `swiglu_limit` per shared-expert FFN (`deepseek4.cpp:38-40`) |
| `indexer_n_head` | `index_n_heads` (`deepseek4.cpp:42`) |
| `indexer_head_size` | `index_head_dim` (`deepseek4.cpp:43`) |
| `indexer_top_k` | `index_topk` (`deepseek4.cpp:44`) |
| `n_layer_nextn` | `num_nextn_predict_layers` (MTP depth) (`deepseek4.cpp:19`) |

### 1.2 Generic fields that matter, and how DSV4 sets them

| Field | DSV4 value / source |
|---|---|
| `n_embd` | `hidden_size` (GGUF `embedding_length`) |
| `n_embd_out()` | `= dsv4_hc_mult * n_embd` — set explicitly at `deepseek4.cpp:54` (`n_embd_out_impl = hc*n_embd`); written by converter `deepseek.py:678` (`add_embedding_length_out(hidden_size * hc_mult)`) |
| `n_embd_inp()` | `= n_embd` (no deepstack) |
| `n_layer_all` | `block_count` = `num_hidden_layers + num_nextn_predict_layers` (MTP case) |
| `n_layer()` | `n_layer_all - n_layer_nextn` (`llama-hparams.h:381`, effective trunk layers) |
| `n_head()` | `num_attention_heads` |
| `n_head_kv()` | single shared KV head — the graph reshapes `kv` to `[d_h, 1, nt]` at `deepseek4.cpp:964`, so attention is MQA (`n_head_kv = 1`) |
| `n_embd_head_k()` / `n_embd_head_v()` | `d_h = qk_nope_head_dim + qk_rope_head_dim` (both equal; asserted `deepseek4.cpp:925`) |
| `n_rot()` | `d_rope = qk_rope_head_dim` (GGUF `rope.dimension_count`, converter `deepseek.py:654`) |
| `swa_type` | `LLAMA_SWA_TYPE_STANDARD` (`deepseek4.cpp:67`) |
| `is_swa_impl[]` | **all trunk layers SWA** (`set_swa_pattern(0)` at `deepseek4.cpp:68`); MTP blocks also SWA (`deepseek4.cpp:69-71`) |
| `n_swa` | `sliding_window` (`deepseek4.cpp:31`) |

Because `n_rot(il)` and `n_embd_head_k(il)` select the SWA variants when a layer is SWA
(`llama-hparams.cpp:85-91,115-121`), **all** DSV4 layers read `n_rot_swa`/`n_embd_head_k_swa`, which default to the
full values `n_rot_full`/`n_embd_head_k_full` (`llama-model.cpp:1228-1237`).

### 1.3 GGUF metadata key names (llama.cpp side)

`src/llama-arch.h:198-288` (enum) ↔ `src/llama-arch.cpp:193-301` (string). DSV4-relevant keys:

| GGUF key (`%s` = `deepseek4`) | enum | loaded into |
|---|---|---|
| `deepseek4.attention.q_lora_rank` | `ATTENTION_Q_LORA_RANK` | `n_lora_q` |
| `deepseek4.attention.sliding_window` | `ATTENTION_SLIDING_WINDOW` | `n_swa` |
| `deepseek4.attention.layer_norm_rms_epsilon` | `ATTENTION_LAYERNORM_RMS_EPS` | `f_norm_rms_eps` (overwrites global, `deepseek4.cpp:29`) |
| `deepseek4.attention.output_group_count` | `ATTENTION_OUTPUT_GROUP_COUNT` | `dsv4_o_group_count` |
| `deepseek4.attention.output_lora_rank` | `ATTENTION_OUTPUT_LORA_RANK` | `dsv4_o_lora_rank` |
| `deepseek4.attention.compress_rope_freq_base` | `ATTENTION_COMPRESS_ROPE_FREQ_BASE` | `dsv4_compress_rope_base` |
| `deepseek4.attention.compress_ratios` | `ATTENTION_COMPRESS_RATIOS` | `dsv4_compress_ratios[]` |
| `deepseek4.attention.indexer.head_count` | `ATTENTION_INDEXER_HEAD_COUNT` | `indexer_n_head` |
| `deepseek4.attention.indexer.key_length` | `ATTENTION_INDEXER_KEY_LENGTH` | `indexer_head_size` |
| `deepseek4.attention.indexer.top_k` | `ATTENTION_INDEXER_TOP_K` | `indexer_top_k` |
| `deepseek4.expert_feed_forward_length` | `EXPERT_FEED_FORWARD_LENGTH` | `n_ff_exp` |
| `deepseek4.expert_shared_count` | `EXPERT_SHARED_COUNT` | `n_expert_shared` |
| `deepseek4.expert_weights_scale` | `EXPERT_WEIGHTS_SCALE` | `expert_weights_scale` |
| `deepseek4.expert_weights_norm` | `EXPERT_WEIGHTS_NORM` | `expert_weights_norm` |
| `deepseek4.expert_gating_func` | `EXPERT_GATING_FUNC` | `expert_gating_func` (== SQRT_SOFTPLUS) |
| `deepseek4.swiglu_clamp_exp` / `_shexp` | `SWIGLU_CLAMP_EXP` / `_SHEXP` | `swiglu_clamp_exp[]` / `_shexp[]` |
| `deepseek4.hyper_connection.count` | `HYPER_CONNECTION_COUNT` | `dsv4_hc_mult` |
| `deepseek4.hyper_connection.sinkhorn_iterations` | `HYPER_CONNECTION_SINKHORN_ITERATIONS` | `dsv4_hc_sinkhorn_iters` |
| `deepseek4.hyper_connection.epsilon` | `HYPER_CONNECTION_EPSILON` | `dsv4_hc_eps` |
| `deepseek4.hash_layer_count` | `HASH_LAYER_COUNT` | `dsv4_hash_layer_count` |
| `deepseek4.nextn_predict_layers` | `NEXTN_PREDICT_LAYERS` | `n_layer_nextn` |

Converter side: `conversion/deepseek.py:650-681` (`DeepseekV4Model.set_gguf_parameters`). `n_expert`,
`n_expert_used`, and the gating func are written by the generic base from `n_routed_experts`,
`num_experts_per_tok`, and `score_function="sqrtsoftplus"` (`conversion/base.py:1306-1328`). The DSV4 loader
asserts `score_function` resolved to `SQRT_SOFTPLUS` (`deepseek4.cpp:64-66`).

> Known fixed numerics (from the converter + code): `n_expert = 256` (routed),
> `n_expert_used = 6` (top-k), `dsv4_hc_mult = 4` (asserted `deepseek4.cpp:362`),
> `DSV4_CSA_RATIO = 4`, `DSV4_HCA_RATIO = 128` (`llama-kv-cache-dsv4.cpp:18-19`).

---

## 2. Tensors per layer (name, shape, role)

All tensor creation is in `llama_model_deepseek4::load_arch_tensors` (`deepseek4.cpp:79-178`). GGUF tensor-name
strings in `src/llama-arch.cpp:415-519`; converter name map `conversion/deepseek.py:825-893`.

Shapes are given as `create_tensor(..., {ne0, ne1, ...})` i.e. `ne[0]` is the first dimension (row/contiguous dim in
ggml `mul_mat` terms).

### 2.1 Root / non-layer tensors

| GGUF name | C++ field | shape | role |
|---|---|---|---|
| `token_embd.weight` | `tok_embd` | `[D, V]` | token embedding (`deepseek4.cpp:97`) |
| `output_norm.weight` | `output_norm` | `[D]` | final RMSNorm before LM head (`deepseek4.cpp:99,1395`) |
| `output.weight` | `output` | `[D, V]` | LM head (untied; see §6) (`deepseek4.cpp:100,1399`) |
| `output_hc_fn` | `hc_head_fn` | `[hc*D, hc]` | head hyper-connection mix weight (`deepseek4.cpp:102`) |
| `output_hc_base` | `hc_head_base` | `[hc]` | head HC bias (`deepseek4.cpp:103`) |
| `output_hc_scale` | `hc_head_scale` | `[1]` | head HC scale (`deepseek4.cpp:104`) |

### 2.2 Per-layer tensors (`i` = layer index)

| GGUF name | C++ field | shape | role |
|---|---|---|---|
| `blk.i.attn_norm.weight` | `attn_norm` | `[D]` | RMSNorm before attention (`deepseek4.cpp:110,1308`) |
| `blk.i.attn_sinks.weight` | `attn_sinks` | `[n_heads]` | per-head attention sink bias (`deepseek4.cpp:111`) |
| `blk.i.attn_q_a.weight` | `wq_a` | `[D, r_q]` | Q low-rank down-proj (`deepseek4.cpp:112,937`) |
| `blk.i.attn_q_a_norm.weight` | `attn_q_a_norm` | `[r_q]` | RMSNorm on `qr` (`deepseek4.cpp:113,940`) |
| `blk.i.attn_q_b.weight` | `wq_b` | `[r_q, n_heads*d_h]` | Q up-proj (`deepseek4.cpp:114,943`) |
| `blk.i.attn_kv.weight` | `wkv` | `[D, d_h]` | fused K/V latent proj (`deepseek4.cpp:115,962`) |
| `blk.i.attn_kv_a_norm.weight` | `attn_kv_norm` | `[d_h]` | RMSNorm on `kv` (`deepseek4.cpp:116,963`) |
| `blk.i.attn_output_a.weight` | `wo_a` | `[n_heads*d_h/o_groups, o_lora_rank, o_groups]` | grouped output LoRA A (`deepseek4.cpp:119,1263`) |
| `blk.i.attn_output_b.weight` | `wo_b` | `[o_groups*o_lora_rank, D]` | output LoRA B (`deepseek4.cpp:120,1268`) |
| `blk.i.hc_attn_fn.weight` | `hc_attn_fn` | `[hc*D, hc_mix_dim]` | attn HC mix weight (`deepseek4.cpp:122`) |
| `blk.i.hc_attn_base.weight` | `hc_attn_base` | `[hc_mix_dim]` | attn HC bias (`deepseek4.cpp:123`) |
| `blk.i.hc_attn_scale.weight` | `hc_attn_scale` | `[3]` | attn HC scales (`deepseek4.cpp:124`) |
| `blk.i.hc_ffn_fn.weight` | `hc_ffn_fn` | `[hc*D, hc_mix_dim]` | FFN HC mix weight (`deepseek4.cpp:125`) |
| `blk.i.hc_ffn_base.weight` | `hc_ffn_base` | `[hc_mix_dim]` | FFN HC bias (`deepseek4.cpp:126`) |
| `blk.i.hc_ffn_scale.weight` | `hc_ffn_scale` | `[3]` | FFN HC scales (`deepseek4.cpp:127`) |
| `blk.i.ffn_gate.weight` | `ffn_gate_inp` | `[D, n_expert]` | router weight (`deepseek4.cpp:153`) |
| `blk.i.ffn_gate_tid2eid.weight` | `ffn_gate_tid2eid` | `[n_expert_used, V]` | hash routing table (hash layers only) (`deepseek4.cpp:155`) |
| `blk.i.exp_probs_b.bias` | `ffn_exp_probs_b` | `[n_expert]` | router selection bias (non-hash layers only) (`deepseek4.cpp:157`) |
| `blk.i.ffn_norm.weight` | `ffn_norm` | `[D]` | RMSNorm before FFN (`deepseek4.cpp:159,1328`) |
| `blk.i.ffn_gate_exp.weight` | `ffn_gate_exps` | `[D, n_ff_exp, n_expert]` | routed expert gate (w1) (`deepseek4.cpp:161`) |
| `blk.i.ffn_down_exp.weight` | `ffn_down_exps` | `[n_ff_exp, D, n_expert]` | routed expert down (w2) (`deepseek4.cpp:162`) |
| `blk.i.ffn_up_exp.weight` | `ffn_up_exps` | `[D, n_ff_exp, n_expert]` | routed expert up (w3) (`deepseek4.cpp:163`) |
| `blk.i.ffn_gate_shexp.weight` | `ffn_gate_shexp` | `[D, n_ff_exp*n_expert_shared]` | shared expert gate (`deepseek4.cpp:165`) |
| `blk.i.ffn_down_shexp.weight` | `ffn_down_shexp` | `[n_ff_exp*n_expert_shared, D]` | shared expert down (`deepseek4.cpp:166`) |
| `blk.i.ffn_up_shexp.weight` | `ffn_up_shexp` | `[D, n_ff_exp*n_expert_shared]` | shared expert up (`deepseek4.cpp:167`) |

### 2.3 Compressor + indexer tensors (only when `dsv4_compress_ratios[i] != 0`)

`coff = (ratio == 4) ? 2 : 1` (`deepseek4.cpp:131`).

| GGUF name | C++ field | shape | present when | role |
|---|---|---|---|---|
| `blk.i.attn_compressor_kv.weight` | `attn_comp_wkv` | `[D, coff*d_h]` | ratio∈{4,128} | compressor value proj (`deepseek4.cpp:133`) |
| `blk.i.attn_compressor_gate.weight` | `attn_comp_wgate` | `[D, coff*d_h]` | ratio∈{4,128} | compressor score proj (`deepseek4.cpp:134`) |
| `blk.i.attn_compressor_ape.weight` | `attn_comp_ape` | `[coff*d_h, ratio]` | ratio∈{4,128} | APE (abs pos embed) table (`deepseek4.cpp:135`) |
| `blk.i.attn_compressor_norm.weight` | `attn_comp_norm` | `[d_h]` | ratio∈{4,128} | RMSNorm on compressed block (`deepseek4.cpp:136`) |
| `blk.i.indexer.proj.weight` | `indexer_proj` | `[D, indexer_n_head]` | ratio==4 | indexer weight proj (`deepseek4.cpp:141`) |
| `blk.i.indexer.attn_q_b.weight` | `indexer_attn_q_b` | `[r_q, indexer_n_head*indexer_head_size]` | ratio==4 | indexer Q proj (`deepseek4.cpp:142`) |
| `blk.i.indexer.compressor_kv.weight` | `indexer_comp_wkv` | `[D, 2*indexer_head_size]` | ratio==4 | indexer compressor value (`deepseek4.cpp:144`) |
| `blk.i.indexer.compressor_gate.weight` | `indexer_comp_wgate` | `[D, 2*indexer_head_size]` | ratio==4 | indexer compressor score (`deepseek4.cpp:145`) |
| `blk.i.indexer.compressor_ape.weight` | `indexer_comp_ape` | `[2*indexer_head_size, ratio]` | ratio==4 | indexer APE (`deepseek4.cpp:146`) |
| `blk.i.indexer.compressor_norm.weight` | `indexer_comp_norm` | `[indexer_head_size]` | ratio==4 | RMSNorm on indexer compressed block (`deepseek4.cpp:147`) |

`ratio` is validated to be exactly `0`, `4`, or `128` (`deepseek4.cpp:148-150`).

### 2.4 NextN/MTP tensors (per MTP layer, `i >= n_layer()`)

| GGUF name | C++ field | shape | role |
|---|---|---|---|
| `blk.i.nextn.eh_proj.weight` | `nextn.eh_proj` | `[2*D, D]` | fused [e|h] input proj (`deepseek4.cpp:170,1462`) |
| `blk.i.nextn.enorm.weight` | `nextn.enorm` | `[D]` | RMSNorm on token emb (`deepseek4.cpp:171,1454`) |
| `blk.i.nextn.hnorm.weight` | `nextn.hnorm` | `[D]` | RMSNorm on hidden state (`deepseek4.cpp:172,1451`) |
| `blk.i.nextn.embed_tokens.weight` | `nextn.embed_tokens` | `[D, V]` | optional MTP token emb (falls back to `tok_embd`) (`deepseek4.cpp:173`) |
| `blk.i.nextn.shared_head_head.weight` | `nextn.shared_head_head` | `[D, V]` | optional MTP LM head (falls back to `output`) (`deepseek4.cpp:174`) |
| `blk.i.nextn.shared_head_norm.weight` | `nextn.shared_head_norm` | `[D]` | optional MTP head norm (falls back to `output_norm`) (`deepseek4.cpp:175`) |

Note: `eh_proj` is fused from `e_proj` and `h_proj` by concatenation at convert time
(`conversion/deepseek.py:811-814`). The MTP block reuses the trunk's `hc_attn_*`, `hc_ffn_*`, `attn_norm`,
`ffn_norm`, and MoE weights of that layer index.

### 2.5 Dense-FFN vs MoE layers

There is **no dense (non-MoE) FFN** in the trunk: every trunk layer uses the MoE path
(`deepseek4.cpp:1339-1356`), followed by a parallel shared-expert FFN (`deepseek4.cpp:1358-1362`). The
`n_layer_dense_lead` / `first_k_dense_replace` concept from V2/V3 does **not** apply. DSpark (`DFLASH` arch) is a
separate draft arch that wraps DSV4 blocks (`src/models/dflash.cpp:7-60`); it is out of scope except as noted in §9.

---

## 3. Attention mechanism (exact computation)

Entry point `llama_model_deepseek4::graph::build_attention_impl` (`deepseek4.cpp:904-1272`).

### 3.1 Q latent (MLA low-rank)

```
qr      = Wq_a · x                        [r_q, nt]            deepseek4.cpp:937  (build_lora_mm)
qr      = RMSNorm_eps(qr) * q_a_norm      [r_q, nt]            deepseek4.cpp:940
q       = Wq_b · qr                       [d_h, n_heads, nt]   deepseek4.cpp:943
q       = RMSNorm_eps(q)  (per head)      [d_h, n_heads, nt]   deepseek4.cpp:945
```
`build_lora_mm` is a plain `mul_mat` (LoRA hooks are no-ops without adapters) — `llama-graph.cpp:1487-1516`.
`RMSNorm_eps` uses `f_norm_rms_eps` (`llama-graph.cpp:1564`); the graph-level `norm_rms_eps` is the same value
(overwritten from `attention.layer_norm_rms_epsilon`, `deepseek4.cpp:29`).

### 3.2 KV latent (single fused K = V)

```
kv      = Wkv · x                         [d_h, nt]            deepseek4.cpp:962
kv      = RMSNorm_eps(kv) * kv_norm       [d_h, nt]            deepseek4.cpp:963
kv      = reshape -> [d_h, 1, nt]         (1 KV head, MQA)     deepseek4.cpp:964
```

### 3.3 RoPE split

`d_nope = d_h - d_rope`, `d_rope = n_rot()` (`deepseek4.cpp:917-918`). Both `q` and `kv` are sliced into a
non-positional prefix `[0, d_nope)` and a positional suffix `[d_nope, d_h)`; RoPE is applied **only** to the suffix:

```
q_nope, q_pe   = split(q,  d_nope)                        deepseek4.cpp:948-955
q_pe           = RoPE(q_pe, pos, d_rope, rope_params)     deepseek4.cpp:956
q              = concat(q_nope, q_pe)                     deepseek4.cpp:959
kv_nope, kv_pe = split(kv, d_nope)                        deepseek4.cpp:967-974
kv_pe          = RoPE(kv_pe, pos, d_rope, rope_params)    deepseek4.cpp:975
kv             = concat(kv_nope, kv_pe)                   deepseek4.cpp:978
```

RoPE parameters are selected per-layer by compression ratio (`deepseek4.cpp:928-935`):

- `use_compress_rope = (compress_ratios[il] != 0)`
- uncompressed: `freq_base = rope_theta`, `freq_scale = 1`, `ext_factor = 0`
- compressed:   `freq_base = compress_rope_theta`, `freq_scale`, `ext_factor`, `beta_fast/slow`, `n_ctx_orig`
  from the model's YaRN rope parameters.

### 3.4 Attention variants (by ratio)

Selected at `deepseek4.cpp:1224-1245`:

1. **MTP / `inp_mtp`** → `build_attn(inp_mtp, ...)` = plain SWA MQA over raw K (`deepseek4.cpp:1225-1231`).
2. **`ratio == 4` (CSA)** with indexer inputs → `build_csa_lid_attention` (`deepseek4.cpp:1232-1237,734-793`).
3. **`ratio == 128` (HCA)** → `build_hca_attention` (`deepseek4.cpp:1238-1241,795-848`).
4. **`ratio == 0`** → `build_raw_attention` = plain SWA MQA (`deepseek4.cpp:1242-1244,850-884`).

Common to all variants: `q` (and `kv`) are optionally Hadamard-rotated when `k_rot` is present
(`deepseek4.cpp:752-755,807-810,861-864`), and the attention is always called with
**`v = k`** (see `deepseek4.cpp:786,841,877` and `deepseek4.cpp:1226-1230`):

```
out = softmax( (q·k)/√d_h + mask + sinks ) · k        // K is also V
```

`build_attn_mha` (`llama-graph.cpp:2500-2633`): `kq = kᵀq`, F32 accumulator (`llama-graph.cpp:2571`), then
`soft_max_ext(kq, mask, 1/√d_h, alibi=0)` with `soft_max_add_sinks(kq, sinks)`, then `kqv = v·softmax`
(`llama-graph.cpp:2600-2610`). There is **no** MLA `v_mla` decompression here (`v_mla = nullptr`). Flash attention
is used when `cparams.flash_attn` (`llama-graph.cpp:2523-2545`), with `soft_max_ext_add_sinks`/`flash_attn_ext_add_sinks`.

### 3.5 Raw (uncompressed, ratio 0) attention

`build_raw_attention` (`deepseek4.cpp:850-884`): stores `kv` into the raw SWA KV cache, then
`build_attn_mha(q, raw_k, raw_k, ..., 1/√d_h, sinks)`. It is **sliding-window** — the mask is the SWA mask and
`get_k` returns only the SWA window (`llama_kv_cache_iswa`). The window size is `n_swa`.

### 3.6 CSA (ratio 4) attention — raw SWA + compressed + lightning indexer

`build_csa_lid_attention` (`deepseek4.cpp:734-793`):

1. `top_k = build_lid_top_k(...)` (§8.2) selects `indexer_top_k` compressed blocks.
2. `q`, `kv` are Hadamard-rotated if `k_rot` set (`deepseek4.cpp:752-755`).
3. `raw_k = raw KV cache K` (SWA window), `csa_k = compressed KV cache K` (`deepseek4.cpp:767-775`).
4. `k_all = concat(raw_k, csa_k)` (`deepseek4.cpp:777`).
5. mask = `concat(raw_mask, top_k_mask(csa_mask, top_k))` (`deepseek4.cpp:780-784`) — only the top-k compressed
   blocks survive; the rest are `-∞` (`build_top_k_mask`, `deepseek4.cpp:705-732`).
6. `out = build_attn_mha(q, k_all, k_all, ..., 1/√d_h, sinks)` (`deepseek4.cpp:786`).

### 3.7 HCA (ratio 128) attention — raw SWA + heavily compressed

`build_hca_attention` (`deepseek4.cpp:795-848`): identical structure but **all** compressed blocks attend (no
indexer top-k); `hca_mask` is the full compressed mask (`deepseek4.cpp:836`).

### 3.8 Output projection (de-RoPE + grouped LoRA)

`deepseek4.cpp:1247-1271`:

```
out        = reshape(attn_out, d_h, n_heads, nt)
out_nope, out_pe = split(out, d_nope)
out_pe     = RoPE_back(out_pe, pos, d_rope, rope_params)   // inverse RoPE  deepseek4.cpp:1256
out        = concat(out_nope, out_pe)                      // "de-rope"      deepseek4.cpp:1258
out        = reshape(out, o_group_dim, o_groups, nt)       // o_group_dim = (n_heads/o_groups)*d_h
out        = permute -> mul_mat(Wo_a, out)                 // grouped LoRA A deepseek4.cpp:1262-1263
out        = permute/cont -> [o_lora_rank*o_groups, nt]
out        = Wo_b · out                                    // LoRA B         deepseek4.cpp:1268
```

`Wo_a` has shape `[n_heads*d_h/o_groups, o_lora_rank, o_groups]` (`deepseek4.cpp:119`) and acts on the
`o_groups`-grouped attention output.

### 3.9 KV cache contents and per-token-per-layer size

The DSV4 cache is a composite (`llama-kv-cache-dsv4.cpp:1158-1284`):

| cache | stores | dim per entry | entries (relative to ctx) |
|---|---|---|---|
| `kv_raw` (ISWA) | raw K (SWA window) | `d_h` (K-only, `n_embd_v_gqa` is zeroed via MLA trick) | full context (SWA ring) |
| `kv_csa` | 4:1 compressed K blocks | `d_h` | `⌈ctx/4⌉` (`dsv4_comp_size`, `:28-30`) |
| `kv_hca` | 128:1 compressed K blocks | `d_h` | `⌈ctx/128⌉` |
| `kv_lid` | 4:1 compressed indexer K | `indexer_head_size` | `⌈ctx/4⌉` |
| `csa_state` | compressor ring state (kv+score) | `2*d_h` each | `state_size = 8` per stream |
| `hca_state` | compressor ring state (kv+score) | `d_h` each | `state_size = 128` per stream |
| `lid_state` | indexer compressor ring state | `2*indexer_head_size` each | `state_size = 8` per stream |

State dims/sizes: `llama-kv-cache-dsv4.cpp:1263-1277`. K-only is achieved by setting MLA impl dims so
`is_mla()` is true (`dsv4_make_k_only`, `llama-kv-cache-dsv4.cpp:831-835`), which makes `llama_kv_cache` allocate
K storage only. Compressor state tensors are **F32**, shape `[n_embd_state, state_size, n_planes]` with
`n_planes = n_stream*(1+n_rs_seq)` (`llama-kv-cache-dsv4.cpp:912-914`).

**Per token per layer** the raw cache stores exactly `d_h` elements (one fused K=V vector). No separate V is
stored. The compressed caches store `d_h` (or `indexer_head_size`) per *block* (one block per `ratio` tokens).

---

## 4. MoE router (exact)

`build_moe_ffn` (`llama-graph.cpp:1915-2264`); called for the trunk at `deepseek4.cpp:1339-1356` and for MTP at
`deepseek4.cpp:1496-1506`.

### 4.1 Scoring function

```
logits = W_gate · x                         [n_expert, nt]      llama-graph.cpp:1947
        (F32 accumulate for sqrt-softplus)                       llama-graph.cpp:1948-1950
probs  = sqrt(softplus(logits))                                  llama-graph.cpp:1975-1978
```

`softplus(z) = log(1 + exp(z))`, then `sqrt`. Gating op is `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`
(enum `llama-hparams.h:17`).

### 4.2 Selection bias (DeepSeek-V3 style)

```
selection_probs = probs + exp_probs_b      // bias added only for selection   llama-graph.cpp:1986-1990
```
`exp_probs_b` is `ffn_exp_probs_b` (`[n_expert]` bias), present only on non-hash layers.

### 4.3 Group-limited routing

`llama-graph.cpp:2003-2026` only triggers when `hparams.n_expert_groups > 1`. **DeepSeek-V4 does not set
`n_group`/`topk_group`** (`conversion/base.py:1312-1317` reads them only if present), so `n_expert_groups == 0`
and group-limited routing is **skipped**. There is no `n_group`/`topk_group` for DSV4.

### 4.4 Top-k selection

Two modes:

- **Non-hash layers** (`i >= num_hash_layers`): `selected_experts = argsort_top_k(selection_probs, n_expert_used)`
  (`llama-graph.cpp:2031`). `n_expert_used = 6`.
- **Hash layers** (`i < num_hash_layers`): the caller precomputes
  `selected_experts = get_rows(ffn_gate_tid2eid, token_ids)` (`deepseek4.cpp:1334-1337`), i.e. a direct
  vocab-indexed lookup (`[n_expert_used, V]` table). `exp_probs_b` is `nullptr` for these layers. The router
  weight `ffn_gate_inp` **is still used** to compute `probs`/weights; only the *selection* is hashed.

### 4.5 Weight extraction, normalization, scaling

```
weights = get_rows(probs, selected_experts)      // UNBIASED probs (not selection_probs)  llama-graph.cpp:2045
if norm_w:                                        // expert_weights_norm (norm_topk_prob)
    weights = weights / sum(weights)  (clamped ≥ 6.1e-5)                                  llama-graph.cpp:2056-2070
if w_scale != 0 and != 1:                         // expert_weights_scale (routed_scaling_factor)
    weights = weights * w_scale                                                          llama-graph.cpp:2071-2074
```

### 4.6 Expert FFN + combination

Per selected expert `e` (SwiGLU with limited activation, `llama-graph.cpp:2143-2176`; DSV4-specific clamp path
`llama-graph.cpp:2153-2156`):

```
up   = W_up_e · x                      clamp(up, -limit, +limit)
gate = W_gate_e · x                    clamp(gate, -inf, +limit)
h    = swiglu_split(gate, up)          // gate*silu(up) with the split applied after clamps
y_e  = W_down_e · h
y    = Σ_e weights_e · y_e             // weighted sum over top-k    llama-graph.cpp:2226-2263
```
`limit = swiglu_clamp_exp[il]` (`llama-graph.cpp:2147`).

Shared experts (parallel, `LLM_FFN_PAR`), `deepseek4.cpp:1358-1362`, `build_ffn` at `llama-graph.cpp:1669-1868`:

```
up_s   = W_up_shexp · x                clamp(up_s, -limit_s, +limit_s)
gate_s = W_gate_shexp · x              clamp(gate_s, -inf, +limit_s)
z      = swiglu_split(gate_s, up_s)
z      = W_down_shexp · z
```
`limit_s = swiglu_clamp_shexp[il]` (`llama-graph.cpp:1752`).

Final FFN output (`deepseek4.cpp:1365`):

```
ffn_out = moe_out + ffn_shexp
```
(`n_expert_shared` experts are stacked along the hidden dim of a single parallel "shared expert" FFN, i.e. the
shared-expert hidden width is `n_ff_exp * n_expert_shared`.)

---

## 5. RoPE / YaRN

- **Type/params come from config** via the generic rope loader (`llama-model.cpp:1146-1237`): `rope_theta`
  (`freq_base`), `rope.scaling.type` (YaRN → `LLAMA_ROPE_SCALING_TYPE_YARN`), `freq_scale`, `yarn_ext_factor`,
  `yarn_attn_factor`, `yarn_beta_fast/slow`, `original_context_length`.
- `d_rope = rope.dimension_count = qk_rope_head_dim` (`conversion/deepseek.py:654`).
- The DSV4 loader reads a **separate** `compress_rope_theta` (`deepseek4.cpp:48`) used **only** by compressed
  layers (§3.3). The `rope_type` used is `LLAMA_ROPE_TYPE_NEOX` for the indexer cache (`llama-kv-cache-dsv4.cpp:1216`)
  and whatever the model's `rope_type` resolves to for the trunk.
- Custom YaRN attention factor for compressed RoPE (`deepseek4.cpp:10-16`):

  ```
  dsv4_rope_attn_factor(freq_scale, ext_factor) =
      1.0                                  if ext_factor == 0
      1 / (1 + 0.1 * log(1/freq_scale))    otherwise
  ```
  Applied to `rope_ext` in the compressed-K builders (`deepseek4.cpp:513-515,597-599`) and the main q/kv RoPE when
  the layer is compressed (`deepseek4.cpp:932,956,975`).

- RoPE is applied with `ggml_rope_ext(pos, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)`
  and removed with `ggml_rope_ext_back` on the output (§3.8).

---

## 6. Normalizations, embedding/output tying

- **RMSNorm** everywhere (`LLM_NORM_RMS`, eps = `f_norm_rms_eps`, `llama-graph.cpp:1564`). The DSV4 loader
  overwrites `f_norm_rms_eps` from `attention.layer_norm_rms_epsilon` (`deepseek4.cpp:29`).
- Placement (trunk, per layer, `deepseek4.cpp:1308-1329`):
  - `attn_norm` after the attention hyper-connection pre-gate, before `build_attention`.
  - `ffn_norm` after the FFN hyper-connection pre-gate, before `build_moe_ffn` + shared FFN.
- **Extra norms on the Q/KV latents** (the "extra norms" asked about):
  - `attn_q_a_norm`: RMSNorm on `qr` (`deepseek4.cpp:940`).
  - head RMSNorm on `q` after `wq_b` (`deepseek4.cpp:945`).
  - `attn_kv_norm`: RMSNorm on `kv` after `wkv` (`deepseek4.cpp:963`).
  - `attn_comp_norm` / `indexer_comp_norm`: RMSNorm on compressed K blocks (`deepseek4.cpp:501,585`).
  - MTP: `hnorm`/`enorm` on hidden/token embeddings (`deepseek4.cpp:1451-1454`).
- **Embedding/output tying**: `tok_embd` (`token_embd.weight`, `[D,V]`) and `output` (`output.weight`, `[D,V]`)
  are **separate** tensors (`deepseek4.cpp:97,100`) — **untied**. The converter does **not** skip `lm_head`
  (the `tie_word_embeddings` skip at `deepseek.py:387-390` is in the V2 model, not V4).
- Final pipeline (`deepseek4.cpp:1379-1401`): flatten HC streams → `hc_head` (hyper-connection head gate) →
  `output_norm` (RMSNorm) → `output` (LM head).

---

## 7. Hyper-connections (exact)

DeepSeek-V4 keeps `hc = 4` parallel residual streams of width `D` (`deepseek4.cpp:1285-1288` repeats the input
embedding to `[D, hc, nt]`).

### 7.1 Pre-gate (`build_hc_pre`, full form, `deepseek4.cpp:349-405`)

```
hc_dim     = hc * D
hc_mix_dim = (2 + hc) * hc                        // = 24 for hc=4
flat       = RMSNorm_eps(x flattened to [hc_dim, nt])
mixes      = hc_fn · flat                         // [hc_mix_dim, nt]
pre        = sigmoid( mixes[0:hc] * scale_pre + base_pre ) + eps     // [hc, nt]
post       = 2 * sigmoid( mixes[hc:2hc] * scale_post + base_post )   // [hc, nt]
comb_raw   = mixes[2hc:2hc+hc*hc] * scale_comb + base_comb
comb       = Sinkhorn( reshape(comb_raw, hc, hc, nt) )               // [hc, hc, nt]
result     = Σ_h pre_h · x_h                      // weighted mean of streams  (build_hc_pre, :285-310)
```
`scale_pre/base_pre` from `hc_scale[0]`/`hc_base[0:hc]`; `scale_post`/`base_post` from `hc_scale[1]`/
`hc_base[hc:2hc]`; `scale_comb`/`base_comb` from `hc_scale[2]`/`hc_base[2hc:]` (`deepseek4.cpp:370-397`).

### 7.2 Sinkhorn normalization (`build_hc_sinkhorn`, `deepseek4.cpp:312-347`)

The `hc×hc` coupling matrix is normalized to a doubly-stochastic-ish matrix: softmax → add `eps` → normalize
columns → then `sinkhorn_iters-1` rounds of normalize-rows/normalize-cols. `eps = dsv4_hc_eps`.

### 7.3 Post-gate (`build_hc_post`, `deepseek4.cpp:407-442`)

For each destination stream `dst`:
```
out_dst = post_dst · x + Σ_src comb[dst,src] · residual_src
```
where `x` is the block output, `residual` is the pre-block stream tensor.

### 7.4 Head gate (`build_hc_head`, `deepseek4.cpp:444-464`)

Same pre-gate structure but with a single head weight (`hc_head_fn` shape `[hc*D, hc]`, so `mix` → `hc` pre
gates, no post/comb), used to collapse the `hc` streams to one vector before `output_norm`.

### 7.5 Fused variants

`cparams.fused_dsv4_hc_pre/comb/post` swap these into fused kernels `ggml_dsv4_hc_pre/comb/post`
(`deepseek4.cpp:295-299,388-391,416-419`); semantics identical. Reference implementations live in
`ggml/src/ggml-sycl/dsv4-hc.cpp` and are covered by `tests/test-backend-ops.cpp:3766-3920`.

---

## 8. Lightning indexer, compressed KV, Hadamard rotation, hash routing

### 8.1 Compressed-K block generation

Two builders:

- **Overlap (CSA/LID, ratio 4)** — `build_overlap_compressed_kv_from_state` (`deepseek4.cpp:524-606`).
- **Non-overlap (HCA, ratio 128)** — `build_hca_compressed_kv_from_state` (`deepseek4.cpp:466-522`).

Both follow:

```
state_kv    = Wkv_comp · x                                   // [coff*d_h, nt]
state_score = Wgate_comp · x  +  APE[pos % ratio]            // [coff*d_h, nt]  (+ learned abs pos)
gather  (overlap: 2*ratio rows from ring state [prev|cur];  hca: ratio rows)
values, scores = split/permute to [d_h, R, n_blocks]
weights = softmax(scores)  over the R axis
comp    = Σ_R values * weights                                // [d_h, n_blocks]
comp    = RMSNorm(comp) * comp_norm
split comp into nope/pe; RoPE(pe) with compress_rope_base; concat
```
`R = ratio` (HCA) or `2*ratio` (CSA overlap, previous + current blocks). APE = `attn_comp_ape`, a learned
`[coff*d_h, ratio]` table indexed by `pos % ratio` (`deepseek4.cpp:997-998,1010-1013`). For the indexer the same
compression runs at `indexer_head_size` with `indexer_comp_*` (`deepseek4.cpp:1073-1083`).

### 8.2 Lightning indexer top-k (`build_lid_top_k`, `deepseek4.cpp:608-703`)

```
indexer_q       = Windexer_q_b · qr                [idx_h, idx_heads, nt]   :627-628
indexer_q       = RoPE(split) then Hadamard-rotate                    :631-647
indexer_weights = Windexer_proj · x / sqrt(idx_h * idx_heads)         :649-651
indexer_k       = lid compressed-K cache                              :653-661
score           = relu(k · q) · weights, sum over heads               :682-692
score           = score + mask                                        :694
top_k           = top_k(score, indexer_top_k)                         :698-699
```
This is the **lightning indexer**: a small learned attention over compressed indexer keys that yields per-query
top-k compressed-block ids. There is a fused `ggml_lightning_indexer` op when `cparams.fused_lid`
(`deepseek4.cpp:672-675`).

### 8.3 Hadamard rotation

`llama_mul_mat_hadamard` (`llama-impl.h:57-75`) multiplies contiguous `n`-sized blocks of a tensor by a
pre-generated Hadamard matrix `rot`. Matrices are generated for sizes `64,128,...` up to the max head size
(`llama-kv-cache.cpp:343-359`, `ggml_gen_hadamard`). `attn_rot_k` is enabled when K is quantized and head dim
`% 64 == 0`, and **unconditionally for DeepSeek lightning indexers** (DEEPSEEK32/4, GLM_DSA) when
`n_embd_head_k_full == indexer_head_size` (`llama-kv-cache.cpp:319-329`). Rotation is applied to `q`/`kv` before
attention and re-applied to the output after (Hadamard is symmetric/unitary) — see §3.4/§3.8. Rot size is the
largest power-of-2 divisor of the head dim starting at 64 (`llama-kv-cache.cpp:1421-1429`).

### 8.4 Hash routing

`num_hash_layers` (hparam `dsv4_hash_layer_count`) trunk layers replace learned top-k selection with a
`[n_expert_used, V]` table `ffn_gate_tid2eid` mapping token id → expert ids (`deepseek4.cpp:153-158,1334-1337`).
Converted to I32 at `conversion/deepseek.py:764-779`.

---

## 9. MTP / NextN layers

- `n_layer_nextn` is validated `< n_layer_all` (`deepseek4.cpp:27`); the loader verifies MTP tensors exist and
  otherwise zeroes it out (`deepseek4.cpp:20-26`).
- MTP is **single-block only** in the current implementation: `GGML_ASSERT(n_layer_nextn == 1)`
  (`deepseek4.cpp:1410`).
- The MTP graph (`graph_mtp`, `deepseek4.cpp:1407-1546`) takes the last trunk hidden state `h` (the
  `hc`-stream tensor, or its masked `flat` form) plus the current token embedding:

  ```
  h_norm = RMSNorm(h) * hnorm                       :1451
  e_norm = RMSNorm(token_embd) * enorm, repeated hc  :1454-1457
  concat(e_norm, h_norm) -> [2D, nt]
  inpL = eh_proj · concat                           :1462   (fused e_proj|h_proj)
  → same hyper-connection attention block + MoE FFN block as trunk :1469-1520
  → hc_head → shared_head_norm → shared_head_head → logits        :1530-1544
  ```
- `embed_tokens`/`shared_head_head`/`shared_head_norm` fall back to the trunk `tok_embd`/`output`/`output_norm`
  when absent (`deepseek4.cpp:1438,1533,1539`).
- MTP uses an ISWA KV cache filtered to `il >= n_layer()` (`llama-model.cpp:2184-2203`). Hash routing is not
  supported in MTP blocks (`deepseek4.cpp:1495`).

---

## 10. Tokenizer and chat template

- **Tokenizer**: GPT2 / byte-level BPE. `DeepseekV4Model` does not override `set_vocab`, so it inherits
  `TextModel.set_vocab → _set_vocab_gpt2()` (`conversion/base.py:1192-1193,1741`). The actual pre-tokenizer string
  (`deepseek-llm` / `deepseek-v4`) is recorded in GGUF `tokenizer.*` metadata at convert time.
- **Chat template**: embedded at conversion (`conversion/deepseek.py:557-564`), choosing
  `models/templates/deepseek-ai-DeepSeek-V4.jinja` (or `...-V4-Flash-0731.jinja` for `0731` model ids). The
  template uses the DSML tool-call dialect, `｜User｜` / `｜Assistant｜` role markers,
  `<think>`/`</think>` thinking tags, a `Reasoning Effort:` system preamble for high/max effort, and
  `｜end▁of▁sentence｜` as the assistant EOS marker. Both templates are structurally identical except the
  `reasoning_effort` preamble text and the `high` vs `max` gating
  (`models/templates/deepseek-ai-DeepSeek-V4.jinja:1-138`).

---

## 11. Unusual / notable points (checklist)

1. **Hyper-connections** (`hc_mult = 4`): 4 residual streams with learned Sinkhorn coupling — not a plain residual
   (`deepseek4.cpp:349-442`).
2. **Fused K=V** ("kv" latent): no separate value; `attn_mha(q, k, k)` (`deepseek4.cpp:786,841,877`).
3. **RoPE split** (`nope`/`rope`) with **inverse RoPE** on the attention output (`deepseek4.cpp:1247-1258`).
4. **Extra RMSNorms** on `qr`, `q` (per head), and `kv` (§6).
5. **Compressed KV** at 4:1 and 128:1 with learned **APE** and `compress_rope_theta` (§8.1).
6. **Lightning indexer** (sparse top-k token/block selection) with fused `ggml_lightning_indexer` (§8.2).
7. **Hadamard rotation** of q/k for indexer/compressed attention (§8.3).
8. **Hash routing** for the first `num_hash_layers` layers (§8.4).
9. **Attention sinks** (`attn_sinks`, per-head scalar added inside softmax) (`deepseek4.cpp:111`, `llama-graph.cpp:2601`).
10. **Limited SwiGLU** (`swiglu_limit`) with asymmetric clamp (up clamped to ±limit, gate to -inf..limit)
    (`llama-graph.cpp:2149-2163`, `1755-1768`).
11. **`n_embd_out = hc_mult * n_embd`** (output embedding width is the hyper-connection width) (`deepseek4.cpp:54`).
12. **`sqrt(softplus)` MoE gating** (not softmax/sigmoid) (`llama-graph.cpp:1975-1978`).
13. **Single MTP block** only (`deepseek4.cpp:1410`).
14. **`llm_arch_is_hybrid(DEEPSEEK4) == true`** (`llama-arch.cpp:980`) and it supports recurrent-state rollback
    (`llama-arch.cpp:1003`) and SM tensor (`llama-arch.cpp:1025`), but the SWA rollback API returns 0
    (`llama-model.cpp:2536-2541`).

---

## 12. Notes for kernel writers / parity verification

- Attention uses **F32 accumulation** for both `kq` and `mul_mat` Q/KV (`llama-graph.cpp:2571`; the Q/KV/FFN
  `build_lora_mm` path does not force F32, but `kq` is forced to F32). MoE `logits` are forced F32 for
  sqrt-softplus (`llama-graph.cpp:1948-1950`).
- The compressed-K builders and compressor states are **F32** (`llama-kv-cache-dsv4.cpp:913-914`); `kv_raw` uses
  `type_k` from the model (quantized). Hadamard rotation is F32 matrix × quantized K.
- `ggml_swiglu_split` (limited SwiGLU) is a fused gate/up op: it computes `gate * silu(up)` where both are clamped
  *before* the nonlinearity for DSV4 (see `llama-graph.cpp:2153-2163`).
- `ggml_rope_ext` / `ggml_rope_ext_back` with `GGML_PREC_F32` are used on the `d_rope` slices; the `nope` slices
  are untouched views.
- Every DSV4 tensor name is prefixed `blk.%d.` (root tensors are unprefixed `output_*`), matching
  `llama-arch.cpp:415-519`.

---

## 13. Repo difference note

`/home/mal/AI/llama.cpp-master-rebase/src/llama-kv-cache-dsv4.h` is **identical** to the `-new` header
(2279-line `.cpp` in `-master-rebase` vs the `-new` copy; the header is byte-for-byte the same structure). No
architectural divergence was found in the DSV4 cache between the two trees.
