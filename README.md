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
generation, 128 tokens after a 75-token prompt; quality = perplexity over a 600-token technical text (lower is better).

| Engine / config | Decode | Perplexity |
| --- | ---: | ---: |
| llama.cpp, best config (`-ngl 99 -cmoe --moe-expert-cache 40 -fa on`) | 16.3 tok/s | - |
| **Strata-DS4**, file weights as shipped | **17.0 tok/s** | 10.49 |
| **Strata-DS4**, `--dense-requant q6_k --slots auto` (recommended) | **19.1-19.4 tok/s** | 10.55 |
| Strata-DS4, + `--route-bias 0.2` (lossy, see below) | 22.0 tok/s | 11.50 |

The first token's logits match llama.cpp's (same top-1; KL 0.03-0.10, which is the difference between the GPU's and
the CPU's 8-bit activation rounding - perplexity is the same whichever path computes the experts). Prompt processing
currently runs through the decode path (~11-14 tok/s): batched prefill is not written yet.

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
(no NVMe reads mid-token) 16.3; PCIe share + one upload per token 17.0; Q6_K dense matrices (VRAM → +200 slots) 19.4.

## Running it

Requirements: Linux, CUDA 13.x toolkit, an NVIDIA GPU with 24 GB, **~80 GB of free RAM**, the GGUF above.

```bash
git clone https://github.com/Indras-Mirror/Strata-DS4 && cd Strata-DS4
cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-ds4-gpu ds4_generate          # ggml-cuda takes ~10 minutes the first time

# token ids in, token ids out (tools/ds4/ds4_chat.py does text <-> ids with the GGUF's tokenizer)
bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m DeepSeek-V4-Flash-Q2-0731.gguf \
     --ids 0,1,2 -n 128 --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k \
     --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
python3 tools/ds4/ds4_chat.py "Explain RoPE in two sentences." -n 128 --model DeepSeek-V4-Flash-Q2-0731.gguf -- \
     --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
```

`memguard.sh` runs the engine in a cgroup with a hard RAM cap and swap off, so a misconfiguration kills the process
instead of freezing the desktop - use it. Startup takes ~1.5-2 minutes (the 70 GiB arena is read and pinned).

Useful flags: `--slots N|auto` (VRAM expert cache), `--arena-gib` (pinned RAM arena; the rest is read from the file),
`--pcie F` (share of misses sent over PCIe), `--pf-b` (predicted prefetch per layer), `--dense-requant q4_k|q5_k|q6_k`,
`--route-bias D` (see below), `--ppl` (perplexity of the prompt), `--dump-logits`, `--temp`, `--ctx`.

### Cache-aware routing (`--route-bias`, lossy, off by default)
Adds `D` to the router's *selection* score of experts already in VRAM (the mixing weights stay exact), so near-ties
resolve toward the cache. It trades quality for speed - measured: 0.05 → +9% decode / +1.1% perplexity, 0.2 → +29% /
+9.6%. Leave it at 0 unless you have measured it on your own workload.

## Verifying it

- `test_ds4_dense` - the dense half vs a CPU reference forward (`ds4_ref`) on small synthetic models: bit-exact on the
  CPU backend; `--cuda` agrees to ~1e-4 (CUDA vs CPU arithmetic).
- `test_ds4_moe` / `test_ds4_moe_gpu --gpu` - the expert tier on real expert slices vs a dequantised F32 reference,
  every source (VRAM slot, prefetch, PCIe, CPU, file), the SwiGLU clamp forced on, and multi-token runs.
- `tools/ds4/compare_logits.py` - the engine's logits vs llama.cpp goldens.
- `DS4_CHECK_GPU=1` recomputes every GPU-computed expert on the CPU and logs the difference.

## Roadmap

- **Multi-token verification + the MTP draft head** (DeepSeek-V4-Flash's own next-token predictor; 74-93% acceptance
  reported on 0731): the expert reads are shared between the verified tokens, expected +20-30%. In progress.
- DSpark (0731's native 5-token drafter) and n-gram drafting on the same verification path.
- Batched prefill.
- A server (OpenAI-compatible) and a merge back into a single multi-model Strata; a MiMo-V2.6-Flash engine is next.

## Credits

Built on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT). Uses
[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), including its DeepSeek-V4 hyper-connection kernels,
as the reference implementation. DeepSeek-V4-Flash by [DeepSeek](https://huggingface.co/deepseek-ai). The engine's
design notes and every measurement behind the numbers above are in [docs/ds4/](docs/ds4/) (`FINDINGS.md`).

License: MIT (see [LICENSE](LICENSE)).
