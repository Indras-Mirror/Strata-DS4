# Strata → DeepSeek-V4-Flash architecture map

Read-only scoping report. Maps the existing Qwen3.8-Flash-Next (`qwen4exp`) engine onto
DeepSeek-V4-Flash (`deepseek4`). Every claim carries a `file:line`. The target GGUF's
tensor inventory is already captured in `docs/ds4/GGUF_INVENTORY.txt` (1328 tensors, 80.76 GiB).

Conventions below: **(a)** reusable as-is, **(b)** parameterise (change a constant / read a metadata
key), **(c)** must be replaced / newly written.

---

## 0. The two architectures at a glance

| | Qwen3.8-Flash-Next (`qwen4exp`) | DeepSeek-V4-Flash (`deepseek4`) |
|---|---|---|
| Layers | 48 (`layout.hpp:29`) | **43** (`GGUF_INVENTORY.txt:3`) |
| Hidden / `n_embd` | 2560 (`layout.hpp:28`) | **4096** (`:6`) |
| Expert width `n_ff` | 640 (`layout.hpp:53`) | **2048** (`:26`) |
| Routed experts | 512 (`layout.hpp:52`) | **256** (`:27`) |
| Shared expert | 1 (per-layer scalar gate) | 1 (`expert_shared_count=1`, `:28`) |
| Top-k | 10 (K; `expert.hpp:36` NE=512, K from CLI) | **6** (`expert_used_count=6`, `:15`) |
| Attention heads | 24 q / 2 kv, head_dim 256 (`layout.hpp:41-43`) | **64** q / **1** kv, MLA: q_lora_rank 1024, output_lora_rank 1024, 8 groups (`:6,7,21,22,23`) |
| Attention type | 36 GDN (SSM/linear) + 12 QSA sparse-full every 4th | sliding-window + NSA indexer (indexer.head_count 64, key_length 128, top_k 512) (`:34,35,36`) |
| RoPE | NeoX partial, freq_base 1e7 (`rope.hpp:1-4`, `qsa.hpp:125`) | YaRN (factor 16, orig 65536), dim 64 (`:9,10,11,20`) |
| Vocab | 248320, pre `qwen35` (`strata_tokenizer.py:3-6`) | **129280**, pre `joyai-llm` (`:19,42`) |
| PLE n-gram table | `blk.1.ple_*` + `per_layer_token_embd.weight` (28.8 GB) | **none** |
| MTP | 1 layer, 512 experts (`mtp.hpp:3-19`) | 1 layer (`nextn_predict_layers=1`, `:37`) |
| Residual | GR hyper-connections hc=4 (`layout.hpp:48`) | hyper-connections count 4 (`:38`) |

---

## 1. End-to-end paths

### 1.1 Decode path (single token)

Driver is `src/program/generate.cpp::main` (`generate.cpp:1228`). The plain decode loop is the
`for (int64_t pos = pos_start;; ++pos)` loop at **`generate.cpp:7577`** and runs this cycle:

1. **PLE window advance + issue/collect** — `ss.ple_token = tok`, then
   `ple_issue_token`/`ple_finish_token` (`generate.cpp:7595-7621`). Host-side n-gram hash + SSD row reads,
   deliberately OUTSIDE the captured graphs (`layer.hpp:565-575`).
2. **Embed + residual broadcast** — `put_input` (`generate.cpp:3894-3910`) → `embed_row`
   (`layer.cpp:1033`) does a column dequant of `token_embd.weight`; `session_zero` on pos 0 or a
   `cudaMemcpyAsync` broadcast to all `hc` streams (`generate.cpp:3899-3908`).
3. **The 48-layer pass.** Three dispatch choices:
   - `--no-capture`: `session_token` (`session.cpp:781-823`) → `block_layer` per layer
     (`layer.cpp:1344`).
   - token graph (default, single graph for all 48 layers): `session_run_token`
     (`session.cpp:903-963`), captured by `session_capture_token` (`session.cpp:829-901`).
   - two-graphs-per-layer: `session_loop` (`session.cpp:568-779`).
   Dispatch site: `generate.cpp:7636-7655`.
4. **Layer body** (the mixers + MoE are composed in `block_layer_pre`/`block_layer_post`,
   `layer.cpp:1170` / `1319`). Per layer, in stream order:
   - `gr_read (attn) → mixer → gr_write (attn)` — mixer is `gdn_layer` (`layer.cpp:223`) on
     36 layers or `qsa_layer` (declared `layer.hpp:382`) on the 12 full-attention layers.
   - `gr_read (ffn) → moe_route → [doorbell ring]` — routing half, `layer.cpp:393`.
   - host runs the CPU pool on the published ids/weights.
   - `moe_finish` (shared expert + combine) — `layer.cpp:470`; split into `moe_shared` +
     `moe_combine_parts` (`layer.cpp:453`).
   - `gr_write (ffn)`.
5. **LM head** — `run_head` (`generate.cpp:3868-3873`) → `lm_head`/`lm_head_mix` (`layer.cpp:1088/1109`)
   or `NativeHead::run` for `--native`.
6. **Logits readback + NaN scan** — `generate.cpp:7701-7713`.
7. **Sampling** — `sample_tokens` (`generate.cpp:7742`, `include/strata/kernels/sampler.hpp`),
   Philox(seed, position) draw (`generate.cpp:7739-7742`).
8. **Next token** — teacher-forced from the prompt while in the prompt, else the sampled token
   (`generate.cpp:7794`).

The doorbell handoff that lets the CPU pool overlap the GPU: `Doorbell` struct (`layer.hpp:405-422`),
published by `doorbell_publish` in `moe_route` (`layer.cpp:380`), polled with a per-iteration driver
query in `session_loop` (`session.cpp:672-694`).

### 1.2 Speculative decode path (MTP draft + verify window)

Entered from the plain loop once the first generated token is produced (`generate.cpp:7796`); the loop is
the `while ((int64_t) produced.size() < o.max_new)` at **`generate.cpp:8000`**.

1. **Draft** — `MtpDrafter::draft` / `draft_first` (`generate.cpp:7981, 8093`; class `include/strata/core/mtp.hpp:36`).
   One MTP layer (QSA attention over its own K/V + 512-expert MoE + shared expert, `mtp.hpp:8-19`).
   Optional suffix/prompt-lookup drafter `SuffixDrafter` + `DraftPolicy` (`generate.cpp:7990-8017`,
   `include/strata/spec/{suffix_drafter,draft_policy}.hpp`).
2. **Verify window** — `Verifier::run` (`generate.cpp:8039`; class `include/strata/core/verify.hpp:58`).
   T tokens (last accepted + T-1 drafts) through all 48 layers in ONE captured graph
   (`verify.cpp`, `record_window`), producing an argmax per row. Pool is the multi-token
   `drive_pool_multi` → `expert_pool_dispatch_multi` (`generate.cpp:783-790`, `expert_source.hpp:388`).
3. **Acceptance** — `a` = number of leading drafts matching (`generate.cpp:8049-8050`).
4. **Commit** — `Verifier::commit(a+1)` (`generate.cpp:8071`, `verify.hpp:129`) replays the accepted rows to
   advance GDN state / indexer tail / PLE history.
5. **Adaptive tier** runs on a side thread between windows (`generate.cpp:8067-8070, 7898-7971`).

`--spec N` window size, `--spec-min-p`, `--mtp-max-t`, `--suffix-draft` are the tunables
(`generate.cpp:449-528, 1436-1437`).

### 1.3 Prefill path

`strata::prefill::Prefill` (`include/strata/prefill/prefill.hpp:44`), invoked at `generate.cpp:7570` via
`prefill.stats()` after `Prefill::run`. Position `[pos0, pos0+n)` in chunks of `--prefill CHUNK`
(`generate.cpp:1380-1388`).

Per chunk (`prefill.hpp:1-10`): tensor-core GEMMs for the projections (`src/prefill/gemm.cu`,
`src/prefill/kernels.cu`), recurrences walked in-kernel, routed experts grouped by expert — resident ones
from the VRAM tier, the rest streamed from the host arena through a pinned ring on a copy stream. Two expert
arithmetic paths: MMQ (`include/strata/prefill/moe_mmq.hpp`, ggml-cuda MMQ int8 tensor cores) and the opt-in
fused Q2_0 path (`include/strata/prefill/moe_fused.hpp`, Strata's own int8 TC kernels). The `Prefill` object
borrows the expert cache's top slots for its buffers (`Prefill::init` `borrow` param,
`generate.cpp:4411-4422` lend/refill logic).

---

## 2. Expert cache / adaptive tier / expert profile

### 2.1 The three tiers (dispatch is in `ExpertDispatch`, `include/strata/core/expert_source.hpp:235`)

1. **GPU cache** — `ExpertCache` (`include/strata/core/expert_cache.hpp:72`): `n_slots` device slots +
   a `n_layers × n_expert` residency table (`slot_of`, `admit`, `fill_slot`). **No eviction** by design
   (`expert_cache.hpp:133-136`). Slot sizes are uniform (canonical Q2_0 `BLOB`) or per-layer
   (`open_sized`, `expert_cache.hpp:86`). Optional CUDA VMM segmentation for hot shrink/grow (`#533`,
   `expert_cache.hpp:105-124`).
2. **CPU pool** — `ExpertPool` (`include/strata/kernels/cpu/pool.hpp`); adapter `expert_pool_dispatch` /
   `expert_pool_dispatch_multi` (`expert_source.hpp:382-389`). Reads a routed expert's blob from an
   `ExpertSource`.
3. **Source tiers** — `FileExpertSource` (`expert_source.hpp:414`) mmaps `experts.bin` (or, in
   `--native` without `experts.bin`, assembles blobs from the GGUF shards in place); `ArenaExpertSource`
   (`expert_source.hpp:653`) loads the arena into a `PinnedArena` at startup; `pin_cache_complement`
   (`expert_source.hpp:456`) builds a "resident RAM" compact copy of the GPU-cache complement
   (`--resident-experts`), page-locked so the prompt path DMA-copies and the verify window reads a PCIe
   share (`expert_source.hpp:210-232`, `GpuPlanSink`).

The verify window's GPU plan (`GpuPlanSink`) splits a layer's routed set into **VRAM-resident groups**
(computed on-GPU by `moe_hit_grouped_s2_dev` / `native_expert_grouped`) and a **PCIe share** of the misses
(`pcie_mode` 0 = DMA into staging, 1 = grouped kernel reads mapped arena, 2 = copy kernel stages in-graph;
`expert_source.hpp:230-231`). `--pcie-frac` sets `pcie_num/256` (`generate.cpp:7850-7852`).

### 2.2 Adaptive tier

`generate.cpp:7898-7971`. Usage counts per (layer, expert) are decayed (`drive.d.usage`). Every
`--adapt-every` rounds, missing experts routed ≥2× are ranked against each layer's least-routed resident
expert and swapped when the gain exceeds +1.5 (`generate.cpp:7920-7932`); copies run on a side stream while
the GPU drafts (`generate.cpp:7952-7963`). `--expert-profile-save` writes the learned ranking back
(`rank_learned_profile`/`write_expert_profile`, `expert_cache.hpp:54-70`).

### 2.3 Expert profile (static residency)

`read_expert_profile` (`expert_cache.hpp:51`) reads `profile.bin` (`STRP` + counts + ranked
(layer,expert) pairs + a `n_layers × n_expert` frequency table) written by `tools/make_profile.py`.
The device-side residency table `d_res` then makes the hit decision fully on-device in the token graph
(`generate.cpp:4267-4331`).

### 2.4 What is architecture-agnostic (reusable as-is or with a constant)

- `ExpertCache` slot storage, residency table, `open_sized`, `verify_slot`, segmentation — agnostic to
  expert width/type, only to blob bytes. **(a)**
- `FileExpertSource` / `ArenaExpertSource` tiering, `GpuPlanSink` VRAM/PCIe split, `RouterLookahead`
  (`expert_source.hpp:174`) — agnostic; only the *blob assembly* (3 GGUF role slices) and the router matrix
  width change. **(b)**
- The profile file format and the adaptive ranking/swap logic — agnostic; `n_layers`/`n_expert` are already
  parameters (`read_expert_profile(..., n_layers, n_expert, ...)`). **(b)** (top-k, `expert_used_count`,
  and routing *mechanism* change.)
- `ExpertPool` worker/pinning scaffolding and the doorbell `session_loop` overlap — agnostic to the expert
  kernel it calls. **(a)**

---

## 3. Pack format and the GGUF loader (`--native`)

### 3.1 Canonical Q2_0 pack (`tools/strata_pack.py`, `docs/pack-format.md`)

- `index.txt` — flat text manifest, one tensor per line (columns documented in
  `tools/pack_index.py:42-66`: `name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits
  code_bias group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16`).
  `kind` 0 Verbatim / 1 BF16-in-F32 / 2 F32 / 3 F16-in-F32 (`weights.hpp:41-46`).
- `dense.bin` — the non-expert tensors in engine form (bf16/f16/f32; fp16 scales widened to f32 at load,
  `weights.hpp:11-15`).
- `experts.bin` — 48 × 512 expert blobs, each exactly `BLOB = 1,382,400` B, layout `[gate codes | up codes |
  down codes | gate scales | up scales | down scales]` (`expert.hpp:45-51`).

### 3.2 Native pack (`tools/iq_pack.py`) — what `--native` loads

- `native_experts.txt` — one line per layer:
  `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]` (v3), with a per-role
  `gate,up,down` shard column in v4 (`iq_pack.py:13, 464-470`; reader cap
  `kExpertLayoutVersion = 4`, `expert_layout.hpp:80`). `offset` is absolute within the --native shard (or the
  named shard).
- `index.txt` — dense quantized tensors, `token_embd`, `output` are served **natively** from the GGUF, so
  their rows carry shape only (`iq_pack.py:15-16`).
- `dense.bin` — the float tensors (kinds 4/5/2, i.e. BF16/F16/F32) in engine form; the FORM table
  (`iq_pack.py:73-86`) names every BF16/F16/F32 tensor.
- `experts.bin` (optional) — per-layer blobs of raw GGUF slices `[gate rows | up rows | down rows]`; blob
  size per layer because the files mix IQ1_M…IQ3_S gate/up and Q2_0/IQ4_NL down
  (`iq_pack.py:9-12`, `native_expert.hpp:3-8`). Without it, the engine reads the GGUF shards in place
  (`FileExpertSource::set_gguf`, `expert_source.hpp:433`).

### 3.3 The GGUF loader / architecture guard

- Reader: `include/strata/artifact/gguf_reader.hpp` (`GgufFile`, `GgufModel`). v3 only
  (`gguf_reader.hpp:426-427`), split-shard aware (`gguf_reader.hpp:501-571`), `block_geometry` for ggml
  type ids (`gguf_reader.hpp:122-220`).
- **Architecture guard**: `Qwen4ExpGuard` + `check_architecture` (`gguf_reader.hpp:575-604`) hard-codes
  `general.architecture == "qwen4exp"` and asserts `qwen4exp.block_count` (48), `embedding_length` (2560),
  `expert_count`, `expert_used_count`, `attention.head_count` (24), `attention.head_count_kv` (2).
  Called from `NativeDense::load` (`native_dense.cpp:103`), `NativeHead::load` (`native_head.cpp:29`),
  `NativeEmbed::load` (`native_head.cpp:113`).
- Tensor names the native dense loader serves: `blk.<l>.{attn_qkv,attn_gate,ssm_out,attn_q,attn_k,attn_v,
  attn_output,ffn_gate_shexp,ffn_up_shexp,ffn_down_shexp}.weight` plus `blk.1.ple_key.weight`
  (`native_dense.cpp:31-33`); eligibility requires `native_mmvq_supported(type)` (`native_dense.cpp:53`).
- `NativeHead::load` reads `output.weight` (must be `native_mmvq_supported`, 2-D, canonical dims,
  `native_head.cpp:32-63`); `NativeEmbed::load` reads `token_embd.weight` (`native_head.cpp:116-124`).

**For DeepSeek**: the guard must become `deepseek4`; the served-name list is entirely different
(`attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_a`, `attn_output_b`, `ffn_gate/up/down_exps`,
`ffn_gate/up/down_shexp`, `ffn_gate_inp`, `ffn_gate_tid2eid`, `indexer.*`, `hc_*`); and the native dense
"attach GGUF block to canonical WeightRef" mechanism only makes sense where a canonical tensor still exists.

---

## 4. Hard-coded Qwen-specific assumptions

| # | Assumption | file:line | DeepSeek impact |
|---|---|---|---|
| 1 | Architecture guard string `qwen4exp` + `qwen4exp.*` keys (block_count 48, hidden 2560, head 24/kv 2) | `gguf_reader.hpp:575-604` | **(c)** → `deepseek4`; keys change to `deepseek4.*` (block_count 43, embedding 4096, head 64/kv 1) |
| 2 | `ModelGeometry`: n_embd 2560, n_layers 48, qsa_interval 4, ssm_state_size 128, ssm_k_heads 16, ssm_v_heads 48, ssm_d_conv 4, ssm_conv_channels 10240, ssm_value_dim 6144, n_head 24, n_head_kv 2, head_dim 256, idx_q_heads 4, idx_key_dim 128, hc 4, hc_lr 320, n_expert 512, n_ff 640 | `layout.hpp:27-59` | **(b)/(c)** — n_embd/hc/n_expert/n_ff parameterise; the entire SSM block + QSA head/idx block is replaced (DeepSeek has no SSM; MLA + NSA indexer) |
| 3 | Planner `Geometry` (n_layers 48, n_qsa_layers 12, head_dim 256, indexer 4/1/128, ssm_*, `vram_pool_bytes()` 5.943 GB) | `plan/plan.hpp:22-59,125-127` | **(c)** — rewrite for MLA/NSA geometry; KV-cost derivation (`kv_bytes_per_token`) is wrong for MLA compressed KV |
| 4 | Expert geometry `H=2560, FF=640, NE=512, QK=64, QKA=32, BLOB=1382400`, blob plane offsets | `expert.hpp:34-51` | **(b)** — H→4096, FF→2048, NE→256, BLOB recomputed; plane layout changes for GGUF-native blobs |
| 5 | `is_qsa_layer`: `layer % 4 == 3` selects full-attention layers | `layout.hpp:63-65` | **(c)** — DeepSeek attention is per-layer sliding-window + NSA indexer, not a GDN/QSA alternation |
| 6 | Tensor names: `attn_qkv/attn_gate/ssm_out/attn_q/k/v/output`, `ffn_gate/up/down_exps`, `ffn_gate_inp`, `ffn_gate/up/down_shexp`, `ffn_gate_inp_shexp`, `indexer.{k_proj,q_proj,k_norm,q_norm}`, `hc_*`, `output_hc_*`, `ple_*`, `per_layer_token_embd` | `native_dense.cpp:31-33`, `layer.cpp:240,319,328,356,407,425`, `iq_pack.py:73-86` | **(c)** — DeepSeek: `attn_q_a/q_b/kv/output_a/output_b`, `ffn_gate_tid2eid`, no `attn_qkv`, no `ssm_*`, no `ple_*`, no `ffn_gate_inp_shexp` |
| 7 | GDN linear-attention/SSM kernels (conv, l2-norm, gate, recurrent step, out-norm) | `layer.cpp:223-333`; `src/kernels/cuda/{gdn,native_gdn,fused_gdn,gr}.cu`; `include/strata/kernels/{gdn,native_gdn,gr}.hpp` | **(c)** — remove; DeepSeek has no SSM mixer |
| 8 | QSA sparse-full attention + indexer (top-k cell selection, pooled keys, RoPE-rotated indexer) | `layer.cpp:803-1017` (qsa path); `src/kernels/cuda/{qsa,qsa_decode_attn,qsa_prompt_attn,qsa_select,native_qsa*}.cu`; `include/strata/kernels/qsa.hpp` | **(c)** — DeepSeek's NSA indexer differs (indexer.top_k 512 vs 2048, sliding_window 128, compress_ratios); the *concept* of a pooled/selected KV is shared but the kernels are new |
| 9 | PLE n-gram table (per-layer embedding, `blk.1.ple_*`, 28.8 GB `per_layer_token_embd.weight`) | `layer.cpp:1173-1227,1290-1317`; `src/kernels/cuda/{ple,native_ple_postops}.cu`; `src/ngram/`; `include/strata/kernels/{ple,ngram}.hpp` | **(c)** — remove; DeepSeek has no PLE/n-gram |
| 10 | RoPE: NeoX partial (n_rot 64 of head_dim 256), freq_base 1e7, scaling folded into table | `rope.hpp:1-14,42-63`, `qsa.hpp:95,121-125`, `src/kernels/cuda/rope.cu` | **(c)** — DeepSeek: YaRN (factor 16, orig 65536), `rope.dimension_count=64`, plus MLA `compress_rope_freq_base` 160000 for compressed keys |
| 11 | Tokenizer: gpt2 byte-BPE, `pre=qwen35`, vocab 248320; `serve/chat_template.jinja` | `strata_tokenizer.py:3-6,46-49`; `serve/chat_template.jinja` | **(c)** — DeepSeek: `pre=joyai-llm`, vocab 129280; chat template differs |
| 12 | MTP draft layer geometry (512-expert MoE, QSA attention, `tools/mtp_rt.py` quant) | `mtp.hpp:3-19`; `generate.cpp:3013` (`draft_geometry{}`); `tools/mtp_*.py` | **(c)** — DeepSeek MTP is 1 layer but 256 experts + MLA + hyper-connections; `tools/mtp_pack.py`/`mtp_fetch.py`/`mtp_rt.py` must be re-derived |
| 13 | Shared expert scalar gate `ffn_gate_inp_shexp.weight` (BF16, 1-D) | `layer.cpp:421-425` | **(c)** — DeepSeek shared expert is `ffn_gate/up/down_shexp` (Q8_0) with no separate gate-inp; adds `expert_weights_scale` 1.5 and `expert_weights_norm` |
| 14 | Routed-expert gating: BF16 `ffn_gate_inp` → top-10 → renormalise | `layer.cpp:356-373`; `src/kernels/cuda/{router_top10,native_router}.cu` | **(c)** — DeepSeek routes via `ffn_gate_tid2eid` (I32 hash map `[6, 129280]`), `hash_layer_count=3`, `expert_weights_norm`, `swiglu_clamp_exp`; `expert_gating_func=4` |
| 15 | GR hyper-connections (hc=4, hc_lr=320, gated residual read/write) | `layout.hpp:48-49`; `src/kernels/cuda/gr.cu` + `fused_gr.cu` | **(b)/(c)** — DeepSeek also has `hyper_connection.count=4`, but the residual path is different (`hc_attn_fn`/`hc_ffn_fn` `[16384,24]`, sinkhorn); reuse the fused GR idea, rewrite the arithmetic |
| 16 | SwiGLU without clamp (shared/routed experts) | `shared_expert.hpp:1-7`; `native_expert.hpp` | **(b)/(c)** — add `swiglu_clamp_exp` clamping (8 values) |
| 17 | KV-cache format INT8/Q4_0/K8V4 sized for head_dim 256 / 2 kv heads | `layer.hpp:206-306`; `src/kernels/cuda/{kv_q4,kv_q8,kv_stream}.cu` | **(c)** — MLA compresses K/V to 512-dim latent (`attn_kv.weight` `[4096,512]`, `attn_kv_a_norm` `[512]`), single kv head; cache stores the latent, not per-head K/V |
| 18 | `sampler`/logits paths assume vocab from `output.weight` `ne1` | `layer.cpp:1111-1124` | **(b)** — vocab 129280; `output.weight` Q8_0 |
| 19 | `--native` served projection type set (`native_mmvq_supported`) | `native_mmvq.cu:1565-1569` | **(b)** — must add **Q2_K** (type 10) for `ffn_down_exps`; see §5 |

---

## 5. CUDA/CPU kernel inventory for quant types

### 5.1 CPU experts

| Path | Quant types | file:line |
|---|---|---|
| Canonical Q2_0 expert kernel (AVX-512 VNNI+VBMI) | Q2_0 (type 42) only | `expert.hpp:33-45,114-124`; `src/kernels/cpu/expert.cpp` |
| Canonical Q2_0 AVX2 fallback | Q2_0 | `expert.hpp:172-175`; `src/kernels/cpu/q2_avx2.cpp` |
| Scalar oracle | Q2_0 | `expert.hpp:177` (`s2_expert_scalar`) |
| Native i-quant experts (ggml-cpu `vec_dot`) | gate/up: IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; down: Q2_0, IQ4_NL | `native_expert.hpp:3-8,32-35`; `src/kernels/cpu/native_expert.cpp` |
| Router dot (BF16) | `ffn_gate_inp` BF16 | `src/kernels/cpu/kq_avx1.cpp`, `kq_avx2.cpp` |

### 5.2 GPU experts — decode window (grouped kernels)

| Path | Quant types | file:line |
|---|---|---|
| `native_mmvq` (pinned CUDA MMVQ, Q8_1 activation) | Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, Q2_0, **and** IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/IQ2_S/IQ1_M (delegated to `iq_mmvq`) | `native_mmvq.cu:1565-1569`, dispatch `1601-1614` |
| `iq_kernels` (llama.cpp i-quants + Q2_0 + UD-Q4_K_XL K-quants) | IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0; Q4_K/Q5_K/Q5_1/Q8_0 | `iq_kernels.hpp:1-7`; `iq_kernels.cu:1777` (`iq_supported`) |
| `native_expert_grouped` (fused grouped gate/up+down) | gu_type 18/21/23 (IQ3_XXS/IQ3_S/IQ4_XS) seen in the dispatch; gated by `native_expert_supported` | `iq_kernels.cu:2474-2493`; `iq_kernels.hpp:49-51` |
| Canonical Q2_0 grouped hit kernel | Q2_0 (Strata S-form) | `include/strata/kernels/s2_expert_grouped.hpp`; `src/kernels/cuda/s2_expert_grouped.cu` |
| Shared expert | canonical S-form (Q2_0/K-quants/IQ4_NL…) + native MMVQ override | `include/strata/kernels/shared_expert.hpp`; `shared_expert.cu` |

### 5.3 GPU prefill

| Path | Quant types | file:line |
|---|---|---|
| MMQ (ggml-cuda int8 TC) | i-quants + Q2_0 + Q8_0; `STRATA_MMQ_KQUANTS`: Q4_K/Q5_K/Q5_1; IQ1_M not covered | `moe_mmq.hpp:15-17` |
| Fused (Strata int8 TC, sm_80+) | Q2_0 (native IQ opt-in) | `moe_fused.hpp:1-15,30-34` |

### 5.4 DeepSeek V4 Flash tensors vs. what exists

From `docs/ds4/GGUF_INVENTORY.txt`:

| DeepSeek tensor | GGUF type | supported today? |
|---|---|---|
| `ffn_gate_exps.weight`, `ffn_up_exps.weight` | IQ2_XXS (16) | **yes** — `iq_mmvq`/`native_mmvq_supported` |
| `ffn_down_exps.weight` | **Q2_K (10)** | **NO** — not in `native_mmvq_supported` (`native_mmvq.cu:1565`) nor `iq_supported`; new Q2_K MMVQ (GPU) and ggml-cpu Q2_K dot (CPU) required |
| `ffn_gate/down/up_shexp.weight` | Q8_0 (8) | yes |
| `ffn_gate_inp.weight` | F16 | router path replaced (see §4 #14) |
| `ffn_gate_tid2eid.weight` | I32 `[6,129280]` | new (hash-router) |
| `attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_a`, `attn_output_b` | Q8_0 | yes (Q8_0 MMVQ), but the MLA projection *graph* is new |
| `token_embd.weight` | F16 | yes (`embed_type_supported` F16/BF16, `iq_kernels.hpp:18`) |
| `output.weight` | Q8_0 | yes |
| `indexer.*`, `attn_compressor_*`, `hc_*`, `exp_probs_b`, `attn_sinks` | F16/F32/I32 | float tensors; new kernels for the compressor/NSA and hyper-connection paths |

**Bottom line**: the expert *cache/streaming/adaptive* machinery (§2) and the *Q8_0/IQ2_XXS/IQ3_XXS/IQ4_XS*
quant kernels carry over almost unchanged; the whole **attention stack (GDN+QSA → MLA+NSA)**, the **PLE**,
the **router** (`ffn_gate_inp`→`ffn_gate_tid2eid`), the **RoPE variant**, the **tokenizer**, and the
**Q2_K down-projection kernel** are the required new work.
