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
| 3 reference forward (`tools/ds4/ds4_ref.cpp`, CPU ggml) | **gate RED** - exact through layer 1, first divergence at layer 2 (CSA ratio-4), a real arithmetic bug not yet isolated; probe taps added (`a7318e2`); see s9 |
| 4a MoE engine replay | **done** - best implied 17.28 tok/s (slots 2300, uncaptured loop); ~22.3 with the loop's ~13 ms/token harness cost removed, above the go line 19.6; see s10 |
| 4b-7 | not started (4b = captured token graph) |

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
The bug is real arithmetic that reading has not yet isolated, so probe taps were added:
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
