<h1 align="center">Strata-DS4</h1>

<p align="center"><b>DeepSeek-V4-Flash (284B MoE) on one gaming PC, faster than llama.cpp</b><br>
A DeepSeek-V4 engine built on <a href="https://github.com/Niko1221/Strata">Strata</a> · Linux · NVIDIA · MIT</p>

> **Status: experimental, single-developer research code.** It generates correct, coherent text and beats
> llama.cpp's best configuration on the machine below, but it has been measured on exactly one PC, has no server or
> chat UI of its own yet, and serves one sequence at a time. Expect rough edges. The upstream Strata README (the
> Qwen3.8-Flash engine this is built on) is in [README.strata.md](README.strata.md).

## What this is

[Strata](https://github.com/Niko1221/Strata) is an inference engine specialised for one MoE model
(Qwen3.8-Flash-Next): the hot experts live in a VRAM cache, the rest in pinned RAM, and each layer's misses are
split between the CPU and PCIe at the same time. This fork applies the same idea to **DeepSeek-V4-Flash-0731**, an
architecture llama.cpp supports but which is unusually hard to run fast on a consumer PC:

- 43 layers × 256 experts (top-6), ~73 GiB of experts at 2-bit - far more than VRAM, most of RAM;
- **hyper-connections** (4 residual streams mixed by a 20-iteration Sinkhorn normalisation every layer);
- **compressed sparse attention** (raw 128-token window + 4× compressed blocks chosen by a "lightning indexer" + 128×
  compressed blocks), attention sinks, a grouped output LoRA, hash routing on the first 3 layers, a SwiGLU clamp.

## How fast is it?

Measured on: **RTX 4090 24 GB · Ryzen 7 5700X (8 cores, AVX2) · 90 GB DDR4-3200 · NVMe**, model
`DeepSeek-V4-Flash-Q2-0731.gguf` (80.8 GiB: experts IQ2_XXS gate/up + Q2_K down, dense Q8_0). Decode = greedy
generation, 128 tokens after a 600-token prompt; quality = perplexity over that prompt (lower is better).

**Decode**

| Engine / config | Decode | Perplexity |
| --- | ---: | ---: |
| llama.cpp, best config (`-ngl 99 -cmoe --moe-expert-cache 40 -fa on`) | 16.3 tok/s | - |
| Strata-DS4 2026-10-07 (`--dense-requant q6_k --slots auto`) | 16.8-17.3 tok/s | 10.47-10.64 |
| **Strata-DS4 now** (`--vram-lru --pf-b 0.7 --arena-skip-resident --arena-gib 58`, recommended) | **23.3-24.0 tok/s** | 10.37-10.55 |
| + `--route-bias 0.05` (lossy, see below; measured on the older config: +9% / +1% perplexity) | - | - |

The 2026-10-08 numbers were measured with a video player and other background processes using about a core of the
CPU (the CPU computes part of the expert misses, so a quiet machine should do a little better). Perplexity moves by
up to ~0.2 between identical runs: which experts the CPU and which the GPU computes depends on timing, and the two
paths round differently (q8_1 vs Q8_K activations).

**Prompt processing (prefill)**

| Prompt | decode-loop prefill (before) | `--prefill-chunk 4096 --chunk-mmq --chunk-prestage` |
| --- | ---: | ---: |
| 3,000 tokens | ~18-21 tok/s | 184-259 tok/s |
| 6,000 tokens | ~18-21 tok/s | **359-392 tok/s** |

Chunked prefill is longer-prompt-faster: each chunk streams every expert of every layer over PCIe once (~74 GB), so
the cost per token falls as the chunk fills. It uses MMQ (int8 tensor cores) for the expert products, so perplexity
moves by ~0.3% (6.38 -> 6.39-6.41 on the 3K prompt); without `--chunk-mmq` it is the decode math.

**Long context** (2026-10-08/09)

Measured up to a 128K-token real document (chunked prefill 4096); 256K loads and decodes. DeepSeek-V4's
compressed sparse attention makes long context cheap in principle - each CSA layer attends to a fixed 512 compressed
blocks chosen by its lightning indexer - but the indexer itself scores every block, and that was the cost that grew:

| position | prefill | attention+router per token, before -> now | decode before | decode now |
| --- | ---: | ---: | ---: | ---: |
| 600 | (decode loop) | 15.5 ms -> 15.9 ms | 25.0 tok/s | 23.7-25.3 tok/s |
| 16,384 | 466 tok/s | 28.7 ms -> - | 12.7 tok/s | not re-measured yet |
| 131,072 | 248 tok/s | 48.1 / 38.2 ms -> **17.1-18.7 ms** | 10.4 tok/s | **19.0-22.4 tok/s** (probe, upper bound) |
| 262,144 | - | -> 18.5 ms | - | **21.8 tok/s** (probe, upper bound) |

- "Now" = the fused lightning-indexer kernel (one CUDA kernel for score + ReLU + head weights + mask instead of a
  ~9-op chain with two `[blocks x 64]` copies, on by default off-CPU) and an O(512) instead of O(blocks) sparse row
  selection. The attention column is real; the "now" decode totals come from a `--pos-offset` probe (decoding at a deep
  position over empty caches), whose expert hit rate is a short prompt's (~80%) rather than a long document's
  (53-63% measured), so a real 128K document will land between the two columns. A real-document re-run is next.
- Prefill at 128K was 73% dense-half time; query-blocked raw-window attention (`DS4_RAW_BLOCK`) is in but its long
  prefill timing is not measured yet.

**Compressed KV cache options** (the CSA/HCA compressed-key caches; quality on the 600-token prompt):

| `--comp-type` | bytes vs F32 | perplexity |
| --- | ---: | ---: |
| `f32` (default) | 1x | 10.4468 |
| `q8_0` + `--icomp-q8` | ~0.27x | 10.5031 (+0.54%) |
| **`iq4_nl` + `--icomp-q8`** (recommended for long context) | ~0.14x | **10.5090 (+0.60%)** |
| `q4_0` + `--icomp-q8` | ~0.14x | 10.5358 (+0.85%) |

Below 8 bits the rows are Walsh-Hadamard rotated (128-blocks) before rounding and after the gather (`--comp-rot`
forces it either way); that is why 4-bit costs barely more than 8-bit here. At 256K the 4-bit caches give back a few
GB of VRAM, i.e. a few hundred more expert slots. `--comp-host` keeps the cache in pinned RAM (zero-copy reads); it
works, but at 256K it costs ~3 ms/token for +62 slots - only worth it beyond ~512K.

The first token's logits match llama.cpp's (same top-1; KL 0.03-0.10, which is the difference between the GPU's and
the CPU's 8-bit activation rounding).

## How it works

Per token, per layer (`tools/ds4/ds4_generate.cpp`):

1. **Predict** (`Ds4Dense::predict`): the layer's router applied to an earlier hidden state guesses the experts
   (~62-70% recall) and the predicted misses start DMA-ing into VRAM on a second CUDA stream **under** the attention.
2. **Attention + router** (`Ds4Dense`, ggml on CUDA, CUDA graphs, fused hyper-connection ops, one input upload per
   token, one readback per layer).
3. **Experts** (`Ds4MoeTier`): hits from the VRAM slot cache (seeded from a routing profile), prefetched experts, a
   share of the misses DMA'd over PCIe and computed on the GPU, the rest computed by the CPU pool straight from a
   pinned RAM arena - all three at once.
4. **Finish**: shared expert + routed sum, hyper-connection post-mix.

What moved the needle on the real model (each step measured, in order): CUDA graphs 5.6 → 9.7 tok/s; llama.cpp's
fused DeepSeek-V4 hyper-connection kernels 11.2; enough VRAM back for 2150 expert slots 12.8; a 70 GiB RAM arena
(no NVMe reads mid-token) 16.3; PCIe share + one upload per token 17.0; then (2026-10-08) reusing the captured CUDA
graphs without ggml's per-call node compare (attention 22 → 17 ms/token) 18.8; an exclusive VRAM LRU (a miss takes the
least-recently-used slot of its layer; hit rate 70 → 84%) 22.3; one shared compute arena for all ~500 attention graph
variants (~850 MiB of VRAM → +120 expert slots) and skipping the compressed-row pooling on tokens that complete no
block (3 of 4 tokens in CSA layers, 127 of 128 in HCA) → 23-24.

**Prefill** (`--prefill-chunk N`): the same per-layer graphs for N tokens at once, built per chunk on a shared
allocator; the expert side streams each layer's experts into VRAM once per chunk and runs them through MMQ, the next
layer's experts DMA in while the current one computes, and the VRAM expert cache is seeded after the prompt.

## Running it

Requirements: Linux, CUDA 13.x toolkit, an NVIDIA GPU with 24 GB, **~80 GB of free RAM**, the GGUF above.

```bash
git clone https://github.com/Indras-Mirror/Strata-DS4 && cd Strata-DS4
cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-ds4-gpu ds4_generate          # ggml-cuda takes ~10 minutes the first time

# token ids in, token ids out (tools/ds4/ds4_chat.py does text <-> ids with the GGUF's tokenizer)
# short prompts, fastest decode
bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m DeepSeek-V4-Flash-Q2-0731.gguf \
     --ids 0,1,2 -n 128 --slots auto --pcie 0.55 --dense-requant q6_k --vram-lru --pf-b 0.7 \
     --arena-skip-resident --arena-gib 58 --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
# long prompts: chunked prefill (needs the full arena - do NOT combine with --arena-skip-resident)
bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m DeepSeek-V4-Flash-Q2-0731.gguf \
     --ids-file prompt.i32 -n 128 --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k --vram-lru --pf-b 0.7 \
     --prefill-chunk 4096 --chunk-mmq --chunk-prestage --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
python3 tools/ds4/ds4_chat.py "Explain RoPE in two sentences." -n 128 --model DeepSeek-V4-Flash-Q2-0731.gguf -- \
     --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
```

`memguard.sh` runs the engine in a cgroup with a hard RAM cap and swap off, so a misconfiguration kills the process
instead of freezing the desktop - use it. Startup takes ~1.5-2 minutes (the 70 GiB arena is read and pinned).

Useful flags: `--slots N|auto` (VRAM expert cache), `--arena-gib` (pinned RAM arena; the rest is read from the file),
`--pcie F` (share of misses sent over PCIe), `--pf-b` (predicted prefetch per layer), `--dense-requant q4_k|q5_k|q6_k`,
`--vram-lru` (misses take the layer's least-recently-used VRAM slot), `--arena-skip-resident` (the VRAM-seeded
experts stay out of the RAM arena), `--prefill-chunk N` / `--chunk-mmq` / `--chunk-prestage` (prefill),
`--comp-type f32|q8_0|q5_0|q4_0|iq4_nl` / `--icomp-q8` / `--comp-rot 0|1` / `--comp-host` (compressed KV cache,
see above), `--pos-offset N` (decode-timing probe at a deep position, no prefill), `--route-bias D` and
`--skip-miss T` (lossy, see below), `--mtp FILE [--verify]` (DeepSeek-V4's MTP draft head:
63% acceptance measured; `--verify` is CPU-tested, not yet tuned on CUDA, and chunked prefill does not fill the MTP
window yet), `--ppl` (perplexity of the prompt), `--dump-logits`, `--temp`, `--ctx`.

### Cache-aware routing (`--route-bias`, lossy, off by default)
Adds `D` to the router's *selection* score of experts already in VRAM (the mixing weights stay exact), so near-ties
resolve toward the cache. It trades quality for speed - measured: 0.05 → +9% decode / +1.1% perplexity, 0.2 → +29% /
+9.6%. Leave it at 0 unless you have measured it on your own workload.
`--skip-miss T` drops a routed expert that is a VRAM miss and weighs less than T of the token's routing weight
(0.05: ~0.4% of lookups, within run-to-run noise; 0.10: 23.8 tok/s vs 22.3-23.4 for the reference runs, +5% perplexity
- not recommended).

## Verifying it

- `test_ds4_dense` - the dense half vs a CPU reference forward (`ds4_ref`) on small synthetic models: bit-exact on the
  CPU backend; `--cuda` agrees to ~1e-4 (CUDA vs CPU arithmetic).
- `test_ds4_moe` / `test_ds4_moe_gpu --gpu` - the expert tier on real expert slices vs a dequantised F32 reference,
  every source (VRAM slot, prefetch, PCIe, CPU, file), the SwiGLU clamp forced on, and multi-token runs.
- `tools/ds4/compare_logits.py` - the engine's logits vs llama.cpp goldens.
- `DS4_CHECK_GPU=1` recomputes every GPU-computed expert on the CPU and logs the difference; `DS4_CHECK_LRU=1`
  byte-checks every VRAM-LRU swap against the file.
- Chunked prefill is bit-identical to the one-token decode loop on the CPU backend (logits of every prompt position,
  chunks of 1-70 tokens; `DS4_CPU_FA_REF=1` for >= 64-token chunks, whose flash-attention otherwise tiles), and every
  speed change above was checked the same way and mutation-tested (break it on purpose, the gate must fail).

## Roadmap

- **MTP verification on CUDA**: the head drafts at 63% acceptance here (74-93% reported elsewhere); at that rate
  depth 1 is worth ~+10%, deeper drafts lose (each costs ~9 ms and more expert reads).
- Long-context decode on a real document (the probe numbers above are an upper bound): the expert hit rate falls to
  53-63% on long documents; a cross-layer expert cache is the candidate fix.
- The dense half's ~4,500 small kernels per token (attention+router is ~15.5 ms against a ~6 ms weight-read floor).
  In progress: 2026-10-09 cut a CSA layer from 121 to ~107 kernels per token (161 -> 130 when a block completes;
  attention+router 16.5 -> 16.0 ms in the first A/B), all bit-exact on the CPU gates; an nsys profile decides what
  is kernel time and what is launch gaps.
- DSpark (0731's native 5-token drafter) and n-gram drafting on the same verification path.
- A server (OpenAI-compatible) and a merge back into a single multi-model Strata; a MiMo-V2.6-Flash engine is next.

## Credits

Built on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT). Uses
[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), including its DeepSeek-V4 hyper-connection kernels,
as the reference implementation. DeepSeek-V4-Flash by [DeepSeek](https://huggingface.co/deepseek-ai). The engine's
design notes and every measurement behind the numbers above are in [docs/ds4/](docs/ds4/) (`FINDINGS.md`).

License: MIT (see [LICENSE](LICENSE)).
