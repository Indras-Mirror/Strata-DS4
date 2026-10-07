# DeepSeek-V4-Flash on Strata: feasibility findings

Status: **investigation**, not a working engine. This branch holds the scoping, the measurements and the tools
behind them, so anyone looking at "Strata for DeepSeek" starts from data instead of from zero.

Test machine: RTX 4090 24 GB, Ryzen 7 5700X (8 cores, AVX2), 90 GB DDR4, model on NVMe.
Model: DeepSeek-V4-Flash 0731, Q2 GGUF (huihui abliterated), 80.76 GiB, GGUF arch `deepseek4`.

## 1. Does the model fit a Strata-style split?

Yes. From `GGUF_INVENTORY.txt`:

| Part | Size | Quant | Placement in a Strata-style engine |
| --- | ---: | --- | --- |
| Routed experts, 43 layers x 256 | 72.6 GiB (6.75 MiB per expert) | gate/up IQ2_XXS, down Q2_K | hot ones in VRAM, the rest pinned in RAM, computed on the CPU |
| Everything else (attention, shared expert, router, embeddings) | ~8 GiB | Q8_0 / F16 | VRAM |

Strata already has IQ2_XXS and Q8_0 kernels; Q2_K (the expert down projection) is new work.

## 2. Is the routing skewed enough for an expert cache? (the deciding number)

Strata's speed on Qwen3.8-Flash-Next comes from routing skew: ~40% of the experts cached in VRAM serve 84-96% of
the lookups. We measured DeepSeek-V4's routing with `tools/ds4/route_probe.cpp` (a llama.cpp eval callback that
records every token's top-6 experts per layer) over 12,288 tokens of C++, CUDA, Python, docs, JavaScript and an
agentic-coding transcript, then built a static cache from alternating 512-token blocks and scored it on the
held-out blocks (`tools/ds4/route_skew.py`):

| VRAM for experts | Experts cached | Held-out hit rate | Oracle (best possible) | Uniform (no skew) |
| ---: | ---: | ---: | ---: | ---: |
| 4 GB | 5.5% | 26.6% | 28.0% | 5.5% |
| 8 GB | 11.0% | 39.9% | 42.0% | 11.0% |
| **12 GB** | **16.5%** | **49.7%** | 52.1% | 16.5% |
| 16 GB | 22.0% | 57.4% | 60.0% | 22.0% |

- 50% of lookups need 12.2 GB of experts, 80% need 32.6 GB, 90% need 44.1 GB.
- Usage perplexity per layer (out of 256): the three hash-routed layers (0-2, experts chosen by token id) are nearly
  flat at ~237; the router layers concentrate on 119-174.
- A profile learned on half the text predicts the other half within ~2.5 points of the oracle, so a learned
  profile (Strata's `--expert-profile-save`) would work.

**Reading:** the measured +51% decode (section 3) matches this prediction of ~2x at a ~50% hit rate. DeepSeek-V4's router is ~3x more concentrated than uniform but far flatter than Qwen3.8-Flash-Next's.
A 24 GB card serves about half the expert lookups from VRAM, not ~90%.

## 3. Baseline: llama.cpp on the same machine

`llama.cpp` (master + DSv4 KV cache + MoE expert cache, CUDA 13.4), 32K context, q4_0 KV in RAM, `--load-mode none`,
4 coding prompts x 320 tokens, thinking off:

| Configuration | Decode | Prefill (7.8K-token prompt) |
| --- | ---: | ---: |
Clean runs (2026-10-05 09:38-09:58, quiet machine, NVIDIA 595.91; `clean-ab.results.txt`):

| Configuration | Decode | Prefill (7.8K-token prompt) |
| --- | ---: | ---: |
| A: automatic placement (whole layers of experts on the GPU) | **10.85 tok/s** (10.5-11.3) | **175.2 tok/s** |
| B: all experts in RAM + `--moe-expert-cache 40` (~11 GB of hot experts) | **16.34 tok/s** (15.7-17.1), **+51%** | 153.8 tok/s (-12%) |

Earlier runs the same morning gave A 9.2 and B 16.2 tok/s (A during a stalled-load/download period, B during an apt
upgrade): B is stable, A was depressed in the first run. B's prefill is a little slower: with every expert
host-resident, the large prompt batches stream experts over PCIe.

(B is llama.cpp's own hot-expert cache - the Strata idea inside llama.cpp - and answers "how much of the gain is
available without a new engine".)

## 4. What a Strata port would take

See `STRATA_ARCH_MAP.md` (what in Strata is Qwen-specific, with file:line) and `DSV4_ARCH_SPEC.md` (DeepSeek-V4's
exact computation, cited against llama.cpp). Summary:

- **Reusable:** the expert cache, tiering, PCIe streaming, adaptive tier, expert profile, CPU expert pool, server.
- **New:** the whole attention stack (MLA latent + compressed attention at ratios 4/128 + lightning indexer top-512 +
  128-token sliding window + attention sinks), hyper-connections with Sinkhorn mixing, the sqrt-softplus router with
  selection bias and hash-routed layers, YaRN RoPE, the tokenizer/template, the MTP head, a Q2_K expert kernel.

Estimated decode with a 12 GB cache at ~50% hits: ~0.87 GB of expert weights per token from RAM instead of ~1.46 GB,
so roughly 2x llama.cpp's automatic placement (~18-25 tok/s), before speculative decoding. MTP helps less than on
Qwen because each verify window pulls in more cold experts (MTP and DSpark drafts were slower than no drafts in
earlier llama.cpp tests on this machine).

## 5. Reproduce

```
g++ -O2 -std=c++17 -I <llama.cpp>/include -I <llama.cpp>/ggml/include tools/ds4/route_probe.cpp \
    -L <llama.cpp build>/bin -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,<llama.cpp build>/bin -o route_probe
./route_probe DeepSeek-V4-Flash.gguf routes.bin corpus/*.txt        # experts on the CPU, attention on the GPU
python tools/ds4/route_skew.py routes.bin
```

## 6. Phase 2 - Q2_K experts on both paths

The expert arithmetic the port needs: gate/up IQ2_XXS (already served), down Q2_K (new).

- **CPU.** ggml-cpu has `ggml_vec_dot_q2_K_q8_K` (AVX2), so `native_fmt` / the pool's i-quant path already
  routes Q2_K through ggml's own vec_dot; the one change was the pool's activation buffers, which were sized for
  Qwen (n_embd 2560) and too small for DSV4's 4096 (Q8_K 4672 B) and n_ff 2048 (Q8_K 2336 B):
  `kNativeActBytes` 4096 -> 4672, `kNativeHBytes` 1024 -> 2336 (`native_expert.hpp`).
- **GPU.** `native_mmvq.cu` had no Q2_K; it now has `native_q2_k_mmvq` / `native_q2_k_f32` (port of llama.cpp's
  `vec_dot_q2_K_q8_1`, q8_1 activation), in `native_mmvq_supported`, `native_mmvq_weight_bytes` (84 B / 256) and
  the `native_mmvq()` dispatch. `iq_kernels.cu` gets the same dot as `Fmt<10>` with `dq_q2_k`, so the grouped
  native-expert path (`native_expert_supported`/`native_expert_grouped`, the verify-window VRAM slots and the
  PCIe share) accepts the DSV4 pair (IQ2_XXS gate/up, Q2_K down) too.

Gate: `build-ds4-cuda/ds4_expert_parity <gguf>` - one real expert, gate/up IQ2_XXS and down Q2_K, random input,
each role against ggml's dequantizer + an F32 matmul.  The activation (Q8_K on the CPU, q8_1 on the GPU) is
dequantized in the reference, so the measured error is the kernel's arithmetic, not the 8-bit activation
rounding (reported alongside).

| path | gate | up | down | fused |
| --- | ---: | ---: | ---: | ---: |
| CPU (ggml-cpu vec_dot) | 8.7e-8 | 9.3e-8 | 1.8e-7 | 1.8e-7 |
| GPU (native MMVQ) | 2.9e-5 | 3.0e-5 | 9.5e-8 | 9.6e-8 |

Pure-float reference (including activation rounding): CPU/Q8_K gate 7.0e-3, up 7.2e-3, fused 1.9e-2;
GPU/q8_1 gate 5.5e-3, up 5.8e-3, fused 1.2e-2. GPU Q2_K dequant vs ggml `to_float`: rel 0.000e+00 (bit-exact).

Micro-benchmark, ms per expert matvec (gate+up+down, one token; RTX 4090 + 5700X):

| CPU 1 thread | CPU 8 threads | GPU |
| ---: | ---: | ---: |
| 1.55 | 0.71 | 0.017 (30 iters) |

**GPU parity run 2026-10-05 21:38 (ds4-p2-gpu-parity) - PASS.** Both commands above were run under
`flock ~/.quetza-data/conductor/ds4-gpu.lock`, one GPU process at a time. GPU max rel err 3.0e-5 << 1e-3 gate;
`mmvq_multi_parity` exit 0 (with `multi_exact` on, every column T=1..8 of every type incl Q2_K is bitwise equal
to a single-column call, 0 bit diffs; with it off, the control finds diffs at T>4, so the comparison has power).
No kernel changes were needed. Phase 2 is now fully verified on both paths.

## 7. Phase 0 measurements (2026-10-05) - raw logs in `bench/ds4-2026-10-05/phase0/`

**Goldens:** `bench/ds4-2026-10-05/goldens/{p64,p600,p3000}/` (62 MB, git-ignored; regenerate with
`tools/ds4/make_goldens.sh`). Two llama.cpp p64 runs are **bit-identical** (cosine 1.0, KL 0), so any Phase 3
mismatch is a real bug, not noise.

**Decode profile, config B** (`profile-table.txt`; nsys slows it to 79.7 ms/token vs ~61 ms un-profiled):

| phase | ms/token | share |
|---|---|---|
| GPU kernels (attention 1.6, dense 13.9, other 2.1) | 17.6 | 22% |
| PCIe H2D copies (**415 MB/token**) | 22.1 | 28% |
| CPU experts + host + sync/idle (residual) | 40.0 | 50% |

Verdict: CPU expert time is ~half the token, so the plan stands. New lever: the 415 MB/token of H2D copies
(28%) - overlap them and cut them with the Phase 4 engine.

**DSpark draft (huihui abliterated Q8_0) in llama.cpp-master-rebase - NOT worth it:**

| run | result |
|---|---|
| B (hot-expert cache) + draft | OOM in system RAM (80.8 GiB model + 10.9 GB draft) |
| A (default placement) + draft | CUDA OOM allocating the draft's 1.2 GB compute buffer |
| A36 control (`-ncmoe 36`, no draft) | 11.24 tok/s decode, 175 tok/s prefill |
| A36 + draft, n=5 | **6.3-8.0 tok/s** (2 of 4 prompts; stopped early), acceptance 49% (mean len 3.4) |
| B CPU control (same session) | 15.50 tok/s decode, 150 tok/s prefill |

Making VRAM room for the draft costs ~4.7 tok/s vs B, and the draft then slows A36 down further. Drop DSpark
from Phase 6 unless a Strata-side draft can run without evicting expert slots.

**antirez ds4 on the 4090:** CUDA build (sm_89) done in `~/AI/ds4-ref` (`./ds4`), **not yet run** - todo.

## 8. Status at pause (2026-10-05 ~11:50, paused for the day by Mal)

| phase | state |
|---|---|
| 0 | done except the ds4 run (above) |
| 1 loader/geometry/tokenizer/pack | done, gates verified (4febf29, 64692d1, 7689b05) |
| 2 Q2_K experts | **done both paths** (CPU rel err < 2e-7 / GPU 3.0e-5 << 1e-3; 0.71 / 0.017 ms per expert) |
| 3 reference forward (`tools/ds4/ds4_ref.cpp`, CPU ggml) | **DONE 2026-10-06** - gate = p600/p3000 logits (top1 1.0, KL 0.0086/0.0079, last position only) PASS; per-tensor cosine is a diagnostic (residual proven to be CPU-vs-CUDA noise x Q8 rounding, s9) |
| 4a MoE engine replay | **done** - best implied 17.28 tok/s (slots 2300, uncaptured loop); ~22.3 with the loop's ~13 ms/token harness cost removed, above the go line 19.6; see s10 |
| 4b step 1 (real layer order) | **gate NOT met** - measured 18.70 tok/s best (slots 2300, at the VRAM edge), 16.72 at 1820; zero-overhead ceiling 20.1-21.2; kill criterion fires, decision with Mal; see s11 |
| 4b step 2 (predicted prefetch) | **GO in the replay** - 20.57 tok/s at 2150 slots (speed only, no output check; the real engine is not built yet); see s12 |
| 4b graph-5-7 | not started |

## 9. Phase 3 gate: ds4_ref vs the llama.cpp oracle on p64 (2026-10-05, gate RED)

`bash tools/ds4/run_ref_gate.sh p64` compares 477 tensors (`--cosine 0.9999 --top1 0.99 --kl 0.01`).
Result: **456 FAIL / 21 pass**. (The gate log's table looks like "57 failed" because
`compare_golden.py` prints only its first 60 rows; the breach list is truncated after 100 with
"... 850 more". Count the breaches, not the table.)

The ref is exact through layer 1 and through layer 2's attention *input*: the 21 passes are
`hc_init`, everything of layers 0-1, plus `attn_norm-2` (0.9999219) and `hc_attn_pre-2`
(0.9999203). First divergence is layer 2 - the first CSA (ratio-4) layer - inside the attention
module: `attn_csa_lid-2` cos 0.9033, `attn_out-2` 0.8245, and every later tensor is then poisoned
downstream (including every `attn_hca-*`: on p64 HCA has `n_blocks = 75/128 = 0`, so the ref
computes a raw-window-only attention there and the failures are downstream, not HCA-specific).

**Step 1 answer - the goldens' `attn_csa_lid`/`attn_hca` are NOT post-Hadamard with `-ctk f16`.**
`build_input_k_rot()` returns a rotation only when `attn_rot_k` holds, and for DSV4 `attn_rot_k`
is forced true only by the lightning-indexer clause `n_embd_head_k_full == indexer_head_size`
(`llama-kv-cache.cpp:322-335`). The raw/csa/hca caches keep `n_embd_head_k_full = 512`
(`attention.key_length`) vs `indexer_head_size = 128`, and `type_k = f16` is not quantized, so
`self_k_rot`/`csa.k_rot`/`hca.k_rot` are all null and no Hadamard touches q, K, the compressed
cache or the attention output. Only `kv_lid` is forced to `indexer_head_size`
(`llama-kv-cache-dsv4.cpp:1290`), which is exactly the indexer rotation `ds4_ref` already applies.
Verified numerically: rotating `ds4_ref`'s dumped attn output by the oracle-shaped Hadamard
(n=128 blocks) gives cos ~0.0 vs the golden, not ~1.0, for `attn_raw-*`, `attn_csa_lid-*`,
`attn_hca-*`. So there is no like-for-like rotation to add and no comparison to fix.

Structural re-read of the whole `overlap_compress` path against `build_overlap_compressed_kv_from_state`
+ `dsv4_build_comp_plans` found no arithmetic difference (gather index order = all prev-window reads
then all cur-window reads; prev = plane 0 / cur = plane 1; `comp_pos = ratio*b = state_write_pos =
source_start`; softmax over 2*ratio; the compress RoPE params - base 160000, freq_scale 1/16,
ext_factor 1, attn_factor 1/(1+0.1*ln16), beta 32/1, n_ctx_orig 65536 - and the raw mask all match).
**RESOLVED (same session): the compressed-attention mask was filled transposed.** `make_comp` wrote the
per-block visibility mask as `m16[b*nt + t]`, but the mask tensor is `{n_blocks, nt}` and ggml indexes it as
`mask[key + query*ne0]` (`ggml_flash_attn_ext`: `mp = mask->data + iq1*mask->nb[1]`, then `mp[ic]`), so element
`(block b, query t)` actually lives at `t*n_blocks + b`. Every query therefore saw only the *last few* compressed
blocks instead of all visible ones (for query 74 of p64: blocks 14-17 instead of 0-17), which quietly corrupted
the compressed half of every CSA/HCA attention. Proven by reconstructing both variants in numpy from ds4_ref's own
dumped `q-2` / `k_all-2`: the transposed mask reproduces ds4_ref's output at cos 1.000000, the correct mask
reproduces the golden at 0.999979. Fix (both the f16 and the f32 mask, same loop): fill `t*nb + b`.
Gates after the fix (all three prompts, `run_ref_gate.sh`): p64 **35/477** pass (was 21), p600 **42/477**,
p3000 **38/477**; `attn_csa_lid-2` 0.999979/0.999992 and `attn_hca-3` 0.999975/0.999982, i.e. the compressed half of
CSA *and* HCA (p600 has 4 HCA blocks, p3000 has 23) now matches. Final logits: p600 top1=1.0 top5=1.0 KL=0.0086,
p3000 top1=1.0 top5=1.0 KL=0.0079 (both under the 0.01 KL gate); p64 top1=1.0 top5=0.4 KL=1.39.

The remaining breach is a slow, compounding numeric drift, not another structural bug I could find. It enters at
layer 0: `attn_raw-0` matches to 2.2e-8 max_abs, but `attn_out-0` (de-rope + grouped output LoRA, Q8_0) is already
0.9999892 and `ffn_moe_out-0` 0.999657 (rel 2.6%); no scale offset (best-fit scale 0.998-1.001). It grows ~1e-3
per layer until MoE routing flips discrete top-6 expert selections (l_last-16/17 -> 0.95/0.91 on p64). The likeliest
source is CPU-vs-CUDA matmul numerics: ggml-cpu quantizes the *activations* to Q8_0 for the Q8_0/Q2_K weight
matmuls, while the CUDA oracle does not, and every layer's output LoRA is two Q8_0 matmuls. INFERRED, not proven:
the confirming test is to re-run the oracle with the Q8_0/Q2_K tensors promoted to F16 (or on the CPU backend) and
see whether the residual collapses. Do NOT relax the 0.9999 per-tensor gate before that test.

**Confirming test DONE (2026-10-06, `tools/ds4/check_act_quant.py`, numpy, ~100 MB, no model load) - the residual is
NOT a ds4_ref bug, and the "CUDA does not quantize activations" inference was WRONG.** Rebuilding layer 0's output
projection (de-RoPE + grouped Wo_a + Wo_b, real Q8_0 weights read via mmap) from each side's own `attn_raw-0`:
- with Q8_0-quantized activations, numpy from the golden's `attn_raw-0` reproduces the golden `attn_out-0` at cos
  0.99999885, and from ds4_ref's `attn_raw-0` reproduces ds4_ref's `attn_out-0` at **cos 1.000000000**. F32 or F16
  activations only reach 0.99995 vs either. So **both** backends quantize activations (CUDA MMQ q8_1 rounds the same
  as ggml-cpu q8_0) and ds4_ref's output path is exact.
- The difference is all in the input: `attn_raw-0` ref vs golden is cos 0.9999999062 (rms diff 1.5e-4 on |x| rms 0.34,
  max_abs 3.4e-3 - the earlier "2.2e-8" was 1-cos-scale, not max_abs), i.e. CPU-vs-CUDA flash-attention accumulation
  noise. Q8 activation rounding amplifies it ~100x: through the F32 path the same input gap gives 0.99999986, through
  the Q8 path 0.99998932 - and **random Gaussian noise of the same rms on the golden input gives 0.999988-0.999991**,
  the observed value. Downstream, more Q8 rounding plus discrete MoE top-6 flips compound it to the l_last-16/17 drop.
**Verdict:** the per-tensor cos > 0.9999 at every layer of p64 is below the noise floor of *any* CPU-vs-CUDA pair on
this model (one layer of flash-attn noise already costs 1e-5 after one Q8 matmul). Logit gates (p600/p3000 top1 1.0,
KL < 0.01) pass. Gate change is Mal's call and is NOT made here: options are (a) gate per-tensor against a measured
noise floor (oracle vs oracle with an input perturbation of the observed rms), or (b) treat the logit gates as the
Phase 3 gate and keep the per-tensor table as a diagnostic.

**Gate decision (Mal, 2026-10-06): option (b).** `run_ref_gate.sh` now passes `--tensors-diag`: per-tensor metric
breaches are reported, not failing; logits, token ids, missing tensors and shapes still fail. Results: p600 PASS
(top1 1, top5 1, KL 0.00858), p3000 PASS (KL 0.00787), p64 top1 1 / top5 1 / KL 0.106 (exit 1 - p64 is not a logits
prompt in the plan, diagnostic only). Caveat: the goldens were dumped without `--all-pos`, so each "mean KL" is
over 1 position; an all-positions teacher-forced KL needs regenerated goldens (a full-model GPU run).

The probe taps that localized this are committed:
`tools/ds4/golden_dump.cpp` captures `q`, `kv`, `csa_state_kv`, `csa_state_score_ape`,
`csa_state_compress` under `DS4_GOLDEN_PROBE=1`, and `tools/ds4/ds4_ref.cpp` dumps the same names
under `DS4_PROBE=1`, so the compressed-K path can be diffed tensor by tensor.

## 10. Phase 4a: MoE engine replay (2026-10-05, ds4-p4a-moe)

The expert half of the engine (ExpertCache 6.75 MiB slots via `open_sized` + profile seed, CPU ExpertPool,
the PCIe share of the misses computed on the GPU in parallel, 49 GiB PinnedArena) replayed over
`bench/ds4-2026-10-05/route-probe/ds4routes.bin` (43 layers x top-6), 320 tokens of a held-out block,
random activations; attention is NOT modelled, the MoE is measured alone. Code `tools/ds4/moe_replay.cpp`
+ `moe_gap.cu` (target `ds4_moe_replay`); logs/CSVs in `bench/ds4-2026-10-05/moe-replay/`.

**Gate 1 (correctness): PASS on the CPU path.** CPU expert vs dequant+F32 worst 3.18e-7; per-layer MoE
2.58e-7 (both << 1e-3). The GPU grouped path's 6.2e-4..2.0e-3 residual is reference fidelity, not the
kernel (its reference rebuilds the down activation from a host F32 swiglu output while the card quantizes
its own device intermediate; `ds4_expert_parity` avoids that and reads 3.0e-5).

**Gate 2 (implied = MoE + Phase 0's 17.6 ms non-MoE GPU), `--gap-ms 0`:**

| arm | hit | MoE ms/tok | implied tok/s |
| --- | ---: | ---: | ---: |
| base 1820 slots, pcie 0.55, 7 thr | 50.9% | 48.94 | 15.03 |
| **slots 2300 (15.1 GiB)** | 57.0% | 40.28 | **17.28** |
| pcie 0.75 / 0.35 | 50.9% | 43.96 / 44.67 | 16.25 / 16.06 |
| threads 8 | 50.9% | 41.34 | 16.97 |

**Verdict: the kill criterion fires as written (17.28 < go line 19.6), but the gap is harness overhead,
not architecture.** The un-captured loop costs ~13 ms/token (a sync per layer + ~10 launches + a 128 KB
D2H) overlapped with neither engine; slots-2300 minus that is 27.3 + 17.6 = 44.9 ms = **22.3 tok/s**,
above the go line. The captured token graph is Phase 4b's job. Do NOT invoke the s7 llama.cpp fallback yet.

**Plan-arithmetic changes (measured):** the CPU miss path is **10-16 GB/s, not the plan's ~25-30** - the
5700X has no AVX-512, so Strata's native IQ2_XXS/Q2_K path runs ggml-cpu AVX2 dots; Qwen's 36 GB/s is the
AVX-512 Q2_0 kernel and does not transfer. PCIe measured 15.8 GB/s. Held-out hit at 1820 slots 50.9%
(matches s2). **Harness-found engine bug (fixed, `173e535`):** `ExpertPool::run_split_multi_native` sized
its row split with the compile-time Qwen H/FF (2560/640) instead of the model's n_embd/n_ff - a silent
~1/3 of every expert (rel 9.1e-1 -> 2.6e-7); `SplitBufMulti::ff` overflowed for the same reason. Phase 2
missed it (it drives the single-token kernels directly); canonical Qwen is bit-unchanged.

**Residuals:** the adaptive tier (learned swap) is not implemented - static profile seed + compulsory-miss
admission, no eviction; the shared expert is not computed (inside the 17.6 ms constant); `--gap-ms`
under-delivers (0.4 ms calibrated -> 6.6 ms/token; clock64 vs run boost clocks); run-to-run timing +-4 ms/tok.

Commits (not pushed): `173e535`, `f6c4066`, `805981d`, `dbc4704`, `f035baa`, `3ef564c`.

## 11. Phase 4b step 1: the MoE replay in the real layer order (2026-10-06, gate NOT met)

s10's 22.3 tok/s projection was the claim most likely to be wrong, and it was. It subtracted the un-captured
overhead from the **CPU** side (27.3 ms), but in the same run the **GPU** side was longer (hit 2.3 + PCIe 28.2 =
30.4 ms), and it assumed the 17.6 ms of non-MoE GPU work overlaps the experts. In a real decode it cannot:
experts(l) need router(l), which comes after attention(l), and attention(l+1) needs experts(l).

New harness flags (`tools/ds4/moe_replay.cpp`, sweep `tools/ds4/p4b_sweep.sh`, logs/CSVs
`bench/ds4-2026-10-06/p4b/`): `--serial` (the CPU waits for the layer's dense GPU work before quantizing and starting
its misses), `--gap-wall` (the dense stand-in spins on `%globaltimer`, so 0.41 ms/layer delivers 17.8 ms/token
instead of 4a's clock64 under-delivery), `--pinned-meta` (one pinned upload per group instead of 5 pageable
copies), `--dither` (error-diffused PCIe budget: `lround` on 1-4 misses snapped pcie-frac 0.50 and 0.55 to the
identical split; the real share was ~0.59). Gap = the whole Phase-0 non-MoE GPU time (17.6/43), so `wall` IS a
measured token time with harness syncs included. 320 held-out tokens, threads 7, pinned 48 GiB arena; every run under
`tools/ds4/memguard.sh` (64 GiB cap, swap off, 4 GiB watchdog), ComfyUI idle alongside.

| arm | hit | gpu busy | cpu busy | wall ms | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4a reference re-run (`--gap-ms 0`, implied +17.6) | 57.0% | 29.8 | 24.4 | 37.7 (+17.6) | 18.08 implied |
| **serial, slots 2300, pcie 0.55** | 57.0% | 47.2 | 23.2 | **53.5** | **18.70** |
| serial + pinned-meta | 57.0% | 48.0 | 24.7 | 54.8 | 18.23 |
| serial + pinned, pcie 0.50 (same split as 0.55, rounding) | 57.0% | 48.6 | 24.7 | 55.5 | 18.02 |
| serial + pinned, pcie 0.60 | 57.0% | 48.6 | 23.8 | 54.6 | 18.31 |
| serial + dither 0.53 | 57.0% | 45.6 | 25.6 | 55.3 | 18.09 |
| serial + dither 0.47 | 57.0% | 43.5 | 28.7 | 56.3 | 17.78 |
| serial + dither 0.53, **slots 1820 (12 GiB)** | 50.9% | 49.7 | 29.9 | 59.8 | **16.72** |

Readings:
- **Measured best 18.7 tok/s, under the go line 19.6** (bar 16.34). The kill criterion fires on a measured number now,
  not an implied one. At 1820 slots it is 16.7, about the bar.
- **VRAM:** ~7.2 GiB of non-expert weights must sit in VRAM (8.2 minus the 1 GiB F16 token_embd), plus context/KV/
  scratch, so ~14.8 GiB is left for slots. 2300 slots (15.2 GiB) is at or just over the edge; 1820 is safe.
- **Pinned copies changed nothing** (54.8 vs 53.5, inside the +-1-2 ms run noise). The per-layer copy cost is not
  where the time goes.
- **Balancing the per-token sums does not help:** wall - max(gpu, gap+cpu) grows from 6.3 to 9.7-12.7 ms as the
  split moves toward the CPU. What remains is per-layer imbalance (which side is long changes layer by layer) plus a
  sync per layer. A captured graph + doorbell can cut the sync part but not the imbalance.
- **Zero-overhead bound** (wall = gpu busy): 47.2 ms = 21.2 tok/s at 2300 slots, 49.7 ms = 20.1 at 1820. So even a
  perfect Phase 4b lands around 19.5-21 depending on the slot count it can afford: at best marginal over the go line,
  ~+20-30% over llama.cpp, with the full DS4 GPU forward (Phases 4b-5) still to be written.
- The critical path is the GPU: dense 17.8 + PCIe DMA ~26-28 (0.44 GiB/token at ~17 GB/s). PCIe bandwidth and the
  per-layer dependency, not the CPU, set the ceiling on this box.

**Verdict for Mal (decision not taken here):** per PORT_PLAN s1 / s4 the kill criterion fires (measured 18.7 < 19.6).
Options: (a) stop the Strata port and take s7's fallback - put the PCIe-share split and profile-seeded cache into
llama.cpp's MoE expert cache, where the same mechanism lifts the 16.34 bar without new attention code; (b) continue
to a real captured token graph to measure how much of the ~6 ms/token overhead it removes, accepting that the
ceiling is ~20-21 tok/s.

## 12. Phase 4b step 2: predicted expert prefetch (2026-10-06) - 20.57 tok/s measured in the replay

**Why the plan's PCIe split could not reach 20:** `tools/ds4/bw_contend.cu` - CPU alone 23.3 GB/s, PCIe alone 23.7,
**both at once CPU 12.2 + PCIe 17.7 = 29.9 GB/s**. DMA out of pinned RAM and the CPU's expert reads share the DDR4
bus, so splitting misses buys ~+28% over the CPU alone, not 2x (4a: pcie-0 48.1 ms vs pcie-0.55 48.9 ms MoE).

**The idle window:** during each layer's dense work (~0.41 ms) the PCIe link and the CPU are idle (the experts are
not known yet). `tools/ds4/hidden_probe.cpp` dumped 1024 tokens' router inputs (`bench/ds4-2026-10-06/hidden-probe/`,
run under memguard, mmap); `tools/ds4/predict_experts.py`: layer l's router on an earlier state recalls the true
top-6 at 62% (layer l-1's ffn_norm) / 63% (layer l's pre-attention stream, re-normed with ffn_norm) @6, 70% @8,
78% @12; sanity (true input) 1.000 on all 40 layers; hash layers 0-2 are exact (tid2eid). `sim_prefetch.py`:
1.43 experts/layer (what 0.41 ms x 23.7 GB/s moves) cuts misses 27-28%.

**Measured in the replay** (`--pred --pf-b`, `tools/ds4/p4b_prefetch.sh`, `bench/ds4-2026-10-06/p4b-prefetch/`;
held-out 512 probe tokens, cache seeded from ds4routes' train half, `--serial --gap-wall`, prefetch DMA on a second
stream under the gap, a per-layer prediction readback charged, guesses outside the arena still DMA a dummy blob):

| slots | arm | hit (incl. prefetched) | tok/s |
| ---: | --- | ---: | ---: |
| 1820 | base | 53.8% | 17.84 |
| 1820 | prefetch 1.43 | 66.1% | 19.32 |
| 1820 | prefetch 1.43, pcie 0.35 dithered | | 19.34 |
| 1820 | prefetch 2.0 / 3.0 | | 19.01 / 15.93 (DMA crowds the link) |
| **2150 (the real VRAM budget)** | base | | 19.42 |
| **2150** | **prefetch 1.43, pcie 0.25 dithered** | | **20.57** |
| 2150 | prefetch 1.43, pcie 0.35 / 0.15 | | 20.37 / 20.27 |
| 2300 (over budget) | prefetch 1.43, pcie 0.55 | 71.8% | 21.18 |

Useful prefetches 31.9/token at 1820 (sim said 32): the simulation holds. VRAM budget for slots: 24 GiB - ~7.2
non-expert - ~0.8 scratch - ~1 ctx/KV - ~0.5 other = ~14.5 GiB = ~2150 slots.

**Verdict: GO** - 20.57 tok/s at the realistic slot count vs llama.cpp 16.34 (+26%), over the go line 19.6.

**What this does NOT show (read before quoting the number):** the replay measures SPEED only - random activations,
a timed stand-in for attention (Phase 0's 17.6 ms of llama.cpp non-MoE GPU time), real routes and real expert
kernels. It does not generate text and does not check outputs. Coherence rests on Phase 2 (expert kernels exact) and
Phase 3 (reference forward = llama.cpp logits); prefetch cannot change results by construction (the true router
still selects; a wrong guess wastes a DMA) but no end-to-end output check has run. The real Strata DS4 decode engine
(GPU attention + CSA/HCA + indexer + the expert tier + this prefetch) does not exist yet: it must pass the logits gate
vs llama.cpp and a real generation check before any tok/s counts. Predictions came from llama.cpp's hidden states on
2 corpus files; recall on other text is untested. Next: build that engine (Phase 4b proper), predictor on the GPU.
(a) llama.cpp fallback: same prefetch idea, but its cache only updates between graph runs - not started.

## 13. The real engine on the GPU (2026-10-06 evening / 2026-10-07)

The decode engine (`tools/ds4/ds4_generate.cpp` = `Ds4Dense` + `Ds4MoeTier`) runs the real 80.8 GiB model end to end:
coherent text, first-token top-1 = llama.cpp's.  Speed ladder (p600 prompt, 128-token greedy decode): 5.6 tok/s ->
CUDA graphs 9.7 -> fused `ggml_dsv4_hc_*` ops 11.2 -> 2150 slots 12.8 -> 70 GiB arena (no NVMe file-tier reads)
16.25 -> pcie 0.55 + one input upload per token 16.97 -> `--dense-requant q6_k` 19.35 at 2350 hand-set slots.

**Re-measured 2026-10-07 (the honest table - README corrected in `3a02f9c`):**

| config | tok/s | hit | ppl |
| --- | ---: | ---: | ---: |
| q6_k, `--slots auto` (~2220-2230 slots), pre-multi-token binary | 17.21 | 68.4% | 10.47 |
| q6_k, `--slots auto`, current binary (two runs) | 17.25 / 16.75 | 68.2% / 65.8% | 10.64 / 10.62 |
| q6_k, `--slots auto --route-bias 0.05` (`combo`, 2026-10-06) | 19.08 | 75.1% | 10.77 |
| q6_k, 2350 hand-set slots (at the VRAM edge; OOMed at pos 512 before `reserve_graphs`) | 19.35 | 74.6% | 10.55 |

- ~120 fewer VRAM slots cost ~6 points of hit rate and ~2 tok/s: **VRAM is the scarcest resource** for this model.
- **Perplexity moves by up to ~0.2 between identical runs** (10.47 vs 10.64): which experts the CPU vs the GPU computes
  depends on timing and the two paths round differently (Q8_K vs q8_1 activations). Generated tokens diverge too
  (`ab-old.ids` vs `ab-new.ids` differ from token 3). Compare configs on several runs, not one.
- Attention+router is 22.24 ms on the multi-token binary vs 21.53 on the old one (+0.7 ms, reproduced on 2 runs) -
  not explained (the larger input span is ~150 KB more per token: microseconds). Profile with nsys before guessing.

**Multi-token passes (`c076a86`)** - `Ds4Dense::begin_tokens / attn_router_n / finish_layer_n / logits_n`, n <= 4
(ENGINE_DENSE.md "Multi-token passes").  CPU: bit-identical to one-token decoding at every position, passes of 1-4
tokens, both mini fixtures; n = 1 byte-identical to the pre-change build; mutation-tested (3 deliberate bugs, all
caught).  CUDA: ggml-cuda picks other kernels for n > 1 columns (position 0 of a 2-token pass already differs 1.7%
on the amplifying mini fixtures), so the CUDA gate is "as close to ds4_ref as one-token decoding": swa16 worst cos
0.99678 (multi) vs 0.99687 (one-token), tame 0.99916 vs 0.99912.

**MTP head (`c0cf7fa`, `a6e984b`, `cabe73f`)** - `/media/mal/NVME1TB/Models/DS4-MTP/DeepSeek-V4-Flash-MTP-bf16.gguf`
(5.6 GB: 3.2 GB MXFP4 experts, ~0.45 GB BF16 dense, 2 GB duplicate token_embd/output that we skip).  Loads into
`Ds4Dense` as layer 43 (`mtp_path`); `ds4_ref --mtp` is the oracle; `tools/ds4/make_mini_mtp.py` the mini fixture
(real random MXFP4 experts - the mini trunk's expert blobs are all zeros).  CPU gate: fed the decoder's own trunk
state (`DS4_DBG_MTP_H` -> `DS4_REF_MTP_H`), MTP logits cos 1.00000000 at all 62 positions, top-1 62/62.  Against the
oracle's own trunk: cos 0.99996 - the trunk's final state differs from ds4_ref's by ~1 ulp (rel 1.6e-7) at every
position, invisible in the trunk gate (zero experts) but amplified by the MTP's MXFP4 experts' Q8_0 activation rounding.

**MTP acceptance on the real model (p600, q6_k, slots auto, `bench/ds4-2026-10-07/mtp-trace.log`):
127 drafts, 76 accepted = 59.8%** (greedy: MTP's argmax at p == the trunk's next token).  Draft cost 16.2 ms
(MTP experts = CPU ggml `mul_mat_id` straight off the MXFP4 file - v1).  That run decoded at 12.22 tok/s because each
step also paid the draft and the hit rate was 62.5% (VRAM margin grew); ppl 10.56.
Estimate (NOT measured): 1.6 tokens per verify pass; with a 16 ms draft ~22 tok/s, with a ~3 ms draft ~26-27.

**OOM with `--mtp` (twice, `cudaGraphInstantiate`)**: fixed by `--slots auto` margin +0.25 GiB with `--mtp` (+0.75 with
`--verify`) and the MTP file's remaining BF16 matrices -> Q8_0 at load (which of the two mattered is not isolated).
`DS4_VRAM_TRACE=1` logs free VRAM per prefill step: ~44 MiB drops at positions 1, 5, 9, 17, 33, 65, 129 (the
attention-graph capacity variants being captured as the compressed context doubles), finishing with 0.54 GiB free.
After a CUDA abort, ggml's gdb backtrace on the 75 GB process drove MemAvailable under memguard's floor (watchdog
kill): `memguard.sh` now exports `GGML_NO_BACKTRACE=1`.

**`--verify` (`cabe73f`)**: greedy speculative loop written - each pass = [token, MTP draft] as one 2-token trunk pass
(experts shared through `Ds4MoeTier::run_multi`), an accepted draft also yields the next token.  **Compiled, never run.**

## 14. MTP head uses its own hc_head (2026-10-07, CPU-only session)

The MTP head now finishes with the MTP file's own `output_hc_fn/base/scale` (bound as `mtp.output_hc_*`) instead of
the trunk's - in `Ds4Dense`, the `ds4_ref --mtp` oracle and `make_mini_mtp.py` (whose fixture now carries its own
hc_head, scale 2.0, fn/base from its own rng, appended last so every earlier tensor is unchanged).  llama.cpp
(`src/models/deepseek4.cpp:95-106`) loads the MTP file as its own model with `hc_head_*` required, so this matches it.
`DS4_MTP_TRUNK_HC=1` restores the old head (GPU A/B later).  `token_embd`/`output` stay the trunk's (0731).

CPU gates (`bench/ds4-2026-10-06/dense`): trunk swa16 40/40 + tame 63/63 multi bit-identical, PASS; MTP exact form
(decoder's own trunk state): worst cos 0.99999996, top-1 62/62, MTP in multi passes bit-identical 63/63, PASS.
Mutation: `DS4_MTP_TRUNK_HC=1` -> worst cos 0.986, 62/62 positions below the gate, top-1 55/62, FAIL (caught).
The old fixture (`mini-ds4-mtp.pre-hchead.gguf`, no own hc_head) is refused at attach.  `ds4_generate` on the CPU
(`CUDA_VISIBLE_DEVICES=` - no GPU touched): output byte-identical with and without `--mtp`.
**NOT measured: real-model acceptance with the fix** (GPU queue item 1).

## 15. `--verify` smoke-tested on the CPU (2026-10-07, CPU-only, `bench/ds4-2026-10-07/verify-cpu/`)

First run of the speculative loop.  `run.sh` (mini tame fixture, `CUDA_VISIBLE_DEVICES=` so no GPU is touched)
checks every `--verify` run against plain greedy on three things: the tokens, the logits row behind every token
(`DS4_DUMP_TOKLOGITS`; bit-identical on the CPU), and the MTP draft logits wherever both drafted the same token
index (`DS4_DUMP_DRAFTLOGITS` + `tools/ds4/cmp_draft_logits.py`; this checks the MTP block's own KV across 1- and
2-token MTP passes).  Test hooks drive the accept path the random mini MTP never reaches: `DS4_VERIFY_ORACLE=<plain
stdout>` (drafts = the right tokens) and `DS4_VERIFY_CORRUPT=k` (spoil every k-th).
Result: **26/26 identical** - 3 prompts x {MTP drafts (0% accept), oracle 100%, 69.6%, 50%}, `-n` 1/2/3/7/8, and
`--stop` at a token first seen at index 1, 5, 9 (with and without corruption).  Pass counts are as expected
(40 tokens at 100% = 20 passes, 1.95 tokens/pass).
Mutation tests: a skipped position on one reject was NOT caught by token equality (the mini model's greedy tokens
survived it) - hence the logits-row gate, which catches it (3 failures); reusing a position on an accept: 7
failures; MTP on accept run for the last position only: 15 failures (draft gate).  The draft gate also caught a
stale dump in the new test hook itself (fixed).
No loop bug was found (29/29 after task 3 added `--mtp-resident` cases).  **NOT tested: CUDA** (n=2 kernels round differently there, so equality vs plain greedy will
only be approximate - GPU queue item 2) **and real-model acceptance/speed.**

## 16. The 16 ms MTP draft is page faults, not the MXFP4 kernel (2026-10-07, CPU, `tools/ds4/ds4_mtp_bench.cpp`)

`ds4_mtp_bench` times `MtpExperts::run` (now in `tools/ds4/mtp_experts.hpp`, shared with ds4_generate) on the REAL
MTP file with random routed ids (6 experts = 80.2 MB MXFP4 per draft), under an 8 GB cgroup:

| state of the MTP file's pages | threads | ms/draft (experts) | effective GB/s |
|---|---|---|---|
| warm (page cache) | 8 / 6 / 4 / 2 | 3.55 / **3.29** / 3.47 / 5.50 | 22.6 / 24.4 / 23.1 / 14.6 |
| `--resident` copy (anonymous RAM) | 6 | 3.34 | 24.0 |
| starting uncached (first touch, 60 runs) | 8 | 17.73 mean (min 3.2, max 32.8) | 4.5 |
| `MADV_PAGEOUT` before every run | 8 | 9.90 mean (max 18.6) | 8.1 |

The ggml MXFP4 CPU path is bandwidth-bound and fine (~23 GB/s, the same as the tier's CPU pool).  The 16.2 ms/draft
of the real run matches the cold rows: under memguard's 80 GB cap (which counts page cache) next to a 70 GiB pinned
arena, the MTP file's pages are evicted.  So "Option A" (the tier's CPU pool on MXFP4) would not help; residency does.
- `ds4_generate --mtp-resident`: copies the 3.2 GB of MTP experts into anonymous RAM at load.  CPU gate: drafts
  bit-identical to the mmap path (verify-cpu run.sh, 3 prompts).  Costs 3.2 GB of the 80 GB cap - the GPU run must
  check whether the arena has to shrink (file-tier reads cost ~5 ms each: trading 3 GiB of arena may cost more than
  it saves).  Alternatives if RAM is too tight: keep only the hot MTP experts resident (needs the MTP routing skew),
  or a smaller expert format (quality cost).
- The MTP report line now splits the draft (`routed experts X, rest Y` ms) and prints how much of the MTP expert
  bytes are in RAM at the end (`mincore`; 100% with the copy) - the next GPU run shows directly whether eviction
  happened.
**NOT measured: any of this on the real model run** (GPU queue).
