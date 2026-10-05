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
| GPU (native MMVQ) | not run (see below) | not run | not run | not run |

Pure-float reference (including Q8_K activation rounding): gate 7.0e-3, up 7.2e-3, fused 1.9e-2.

Micro-benchmark, ms per expert matvec (gate+up+down, one token; RTX 4090 + 5700X):

| CPU 1 thread | CPU 8 threads | GPU |
| ---: | ---: | ---: |
| 1.55 | 0.71 | not run (see below) |

The GPU half is a residual: `p0-gpu-runs` still holds the GPU when this slice ends, and the shared protocol
forbids a GPU run before `.done-p0-gpu-runs`.  Run, once it appears (and under `flock ~/.quetza-data/conductor/ds4-gpu.lock`):

```
flock ~/.quetza-data/conductor/ds4-gpu.lock build-ds4-cuda/ds4_expert_parity \
    /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf --bench 30
flock ~/.quetza-data/conductor/ds4-gpu.lock build-ds4-cuda/mmvq_multi_parity
```
