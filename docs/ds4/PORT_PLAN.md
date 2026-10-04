# Strata for DeepSeek-V4-Flash: port plan

Status 2026-10-05: scoping and feasibility done (`FINDINGS.md`); no engine code written yet. This is the plan to
work from. Read with `DSV4_ARCH_SPEC.md` (what DeepSeek-V4 computes, cited to llama.cpp) and `STRATA_ARCH_MAP.md`
(what in Strata is Qwen-specific, cited to Strata's source).

## 1. Goal and the number to beat

Run DeepSeek-V4-Flash 0731 Q2 (`/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf`, 80.76 GiB, huihui
abliterated, arch `deepseek4`) on one RTX 4090 + Ryzen 7 5700X + 90 GB DDR4 faster than llama.cpp, with output that
matches llama.cpp.

| Reference (llama.cpp-master-rebase 4cd3d353d, CUDA 13.4) | Decode | Prefill 7.8K |
| --- | ---: | ---: |
| A: automatic placement | 9.2 tok/s | 175 tok/s |
| **B: `-cmoe --moe-expert-cache 40 --load-mode none`** (the bar) | **16.2 tok/s** | 153 tok/s |

(B measured during an apt upgrade; the clean re-run is in `bench/ds4-2026-10-05/llamacpp-ab/clean-ab.results.txt`
- use those numbers if present.)

**Target: >= 20 tok/s decode (+25% over B), prefill >= 175 tok/s, logits matching llama.cpp.**
**Kill criterion:** if phase 4 (the MoE engine on real weights) cannot beat B by 20%, stop the native port and put
the same ideas into llama.cpp's expert cache instead (section 7).

## 2. Where the speed can come from (and where it cannot)

Per decoded token: 43 layers x 6 routed experts x 6.75 MiB = ~1.74 GB of expert weights. Measured routing
(`FINDINGS.md` s2): a 12 GB VRAM cache serves ~48-50% of lookups.

- **Not from smarter allocation.** Global (Strata-style) ranking beats llama.cpp's equal slots per layer by only
  +0.3 points (48.2% vs 47.9% at 11.3 GB), measured on the probe data. Do not spend time here.
- **From splitting the misses.** llama.cpp computes every miss on the CPU while the GPU waits (GPU 20-40% busy in
  B). Strata's `GpuPlanSink` sends a share of the misses over PCIe to the GPU (`--pcie-frac`, 0.55 on this PC for
  Qwen) *while* the CPU computes the rest (`expert_source.hpp:210-232`). ~0.87 GB of misses per token: CPU at
  ~25-30 GB/s effective and PCIe 4.0 x16 at ~20-25 GB/s working in parallel is the whole case for the port.
- **From overlap.** Strata's doorbell (`layer.hpp:405-422`, `session.cpp:672-694`) runs the CPU pool while the GPU
  does attention and the shared expert; CUDA graphs per token cut launch overhead (43 layers x many small ops).
- **From pinned memory.** `ArenaExpertSource` pins the expert arena (faster DMA, no page faults). With 90 GB RAM,
  pin ~60 GB and leave the rest in the file tier (Strata already has `--resident-budget-gib` / file tier).
- **Maybe from speculative decoding.** The GGUF has **no MTP layer** (metadata says 1, no `nextn` tensors). MTP
  and DSpark drafts were *slower* than no drafts in llama.cpp on this PC. Treat as optional, last.

Phase 0 measures where B's ~62 ms/token actually goes before any of this is built.

## 3. Architecture decisions

1. **New model path inside Strata, not a rewrite of the Qwen path.** Add `deepseek4` beside `qwen4exp`: own
   geometry struct, own layer body, own pack tool; share the expert machinery, CPU pool, server. Keep Qwen working
   (run its tests after every phase).
2. **Correctness before speed.** Phase 3 is a plain, uncaptured, F32-accumulating reference path checked against
   llama.cpp golden dumps layer by layer. Only then CUDA graphs, fusion, quantized KV.
3. **llama.cpp is the oracle.** `~/AI/llama.cpp-master-rebase` (CUDA 13 build in `build/bin`) - do NOT use
   `llama.cpp-new` / `-dspark` (CUDA 12 builds, no longer run without `LD_LIBRARY_PATH=~/cuda13/lib64`). For
   parity runs use F16 KV in llama.cpp: a quantized K turns on the Hadamard rotation of the main attention
   (`DSV4_ARCH_SPEC.md` s8.3), which a first port should not have to match.
4. **Stage the attention by context length**, so each piece is verified alone:
   - <= 128 tokens: raw sliding-window attention (window 128) + CSA compressor (ratio 4). The indexer picks
     top-512 blocks, and with <= 2048 tokens there are <= 512 CSA blocks, so **the indexer selects everything and
     can be skipped**. HCA (ratio 128) has no complete block yet.
   - <= 2048 tokens: add the HCA compressor (one block per 128 tokens).
   - > 2048 tokens: add the lightning indexer (and its Hadamard rotation, which is unconditional for indexers).

## 4. Phases

Each phase ends with a measurable gate. Commit after each gate on branch `deepseek4`; push only with the gate's
numbers in `FINDINGS.md`.

### Phase 0 - oracle and profile (no engine code)
- **First, check the existing `ds4` engine.** huihui's model card runs this exact file with `./ds4 -m
  DeepSeek-V4-Flash-Q2-0731.gguf --ctx 32768` (a DeepSeek-V4-specific engine; also mentioned by a Mac user on
  r/Qwen_AI). Find its repo, what it does (expert caching? CPU/GPU split? CUDA?) and its speed here. If it already
  does what phase 4 plans, build on it or borrow from it instead of starting from Strata alone.
- `tools/ds4/golden_dump.cpp`: like `route_probe.cpp` (eval callback), but dump per-layer tensors for 3 fixed
  prompts (64, 600, 3000 tokens): `attn_norm`, attention output, `ffn_norm`, MoE output, `l_out` (the hc streams),
  final logits; plus the token ids. Names from `deepseek4.cpp` `cb(...)` calls. Store under
  `bench/ds4-golden/` (large: keep the 64-token set in git, others outside).
- Profile run B: where do the ~62 ms/token go (expert CPU time, GPU attention, sync)? `GGML_SCHED_DEBUG`, `perf`,
  or timing callbacks. **Gate:** a table of the split; if expert-CPU time is < 50% of the token, re-think the case
  in s2 before phase 1.

### Phase 1 - loader, geometry, tokenizer, pack
- `gguf_reader.hpp:575-604`: architecture guard for `deepseek4` beside `Qwen4ExpGuard`; read all `deepseek4.*`
  keys listed in `DSV4_ARCH_SPEC.md` s1.3 into a `Ds4Geometry` (D=4096, L=43, heads 64, kv 1, d_h 512, d_rope 64,
  r_q 1024, o_groups 8, o_lora 1024, E=256, K=6, n_ff 2048, hc 4, sinkhorn 20, window 128, ratios[], indexer
  64/128/512, hash layers 3, scale 1.5, swiglu clamp 10, YaRN factor 16 orig 65536, compress rope base 160000).
- Tokenizer: `tools/strata_tokenizer.py` handles gpt2 byte-BPE; add the `joyai-llm` pre-tokenizer regex,
  transcribed from llama.cpp's `llama-vocab.cpp` (the Qwen comment at `strata_tokenizer.py:46-53` explains why it
  must come from the oracle, not by resemblance). Vocab 129280.
- Pack tool `tools/ds4_pack.py` (model on `tools/iq_pack.py`): `native_experts.txt` per layer (gate/up IQ2_XXS,
  down Q2_K, 6.75 MiB blobs); `dense.bin` for the F16/F32 tensors (router, compressor, indexer, hc, norms, sinks,
  APE, `tid2eid` I32); Q8_0 projections served natively from the GGUF.
- **Gate:** tokenizer round-trips 10k lines identical to llama.cpp's `llama-tokenize`; pack loads; every tensor
  in `GGUF_INVENTORY.txt` accounted for.

### Phase 2 - Q2_K experts
- CPU: the pool's i-quant path calls ggml dots; add Q2_K (ggml-cpu has `ggml_vec_dot_q2_K_q8_K`, AVX2).
- GPU: Q2_K MMVQ for the cache slots and the PCIe share (ggml-cuda `mmvq` has it; `native_mmvq_supported`
  at `native_mmvq.cu:1565-1569` lists what Strata serves).
- **Gate:** per-expert parity test (like `tools/test_iq_pack.py`): one real expert, random input, CPU and GPU
  outputs vs ggml dequant + F32 matmul, rel err < 1e-3.

### Phase 3 - reference forward pass (slow, correct)
Plain CUDA (or even CPU) path, no graphs, F32 accumulation, F16/F32 KV:
- embed -> repeat to 4 hc streams; per layer: `hc_pre` (RMSNorm of the flattened streams, `hc_fn` mix, sigmoid
  gates, 20-iteration Sinkhorn on the 4x4 coupling, weighted stream mean) -> `attn_norm` -> q: `Wq_a`, RMSNorm,
  `Wq_b`, per-head RMSNorm; kv: `Wkv`, RMSNorm (K = V, one head) -> RoPE on the last 64 dims (compressed layers use
  YaRN with base 160000 and the custom attn factor, `DSV4_ARCH_SPEC.md` s5) -> attention with sinks over raw
  window (+ compressed blocks) -> de-RoPE the output -> grouped LoRA `Wo_a` (8 groups) -> `Wo_b` -> `hc_post`;
  then `hc_pre` (ffn) -> `ffn_norm` -> router (F32 logits, sqrt(softplus), +bias for selection only, hash layers
  0-2 take ids from `tid2eid[token]` but still use the router's probs as weights, normalise, x1.5) -> 6 experts
  with clamped SwiGLU (up to +-10, gate to <= 10) + shared expert -> `hc_post`; end: `hc_head` -> `output_norm`
  -> `output`.
- Order of work: <= 128 tokens (raw + CSA compressor, no indexer, no HCA) -> <= 2048 (HCA) -> > 2048 (indexer).
- **Gate:** against the golden dumps, per layer: cosine > 0.9999 on hc streams at 64 tokens; final logits top-1
  agreement > 99% and mean KL < 0.01 teacher-forced over the 600- and 3000-token prompts.

### Phase 4 - Strata's MoE engine on DeepSeek (the go/no-go)
- Expert cache (`expert_cache.hpp:72`) with 6.75 MiB slots (`open_sized`), profile with n_layers 43 / n_expert
  256 / K 6 (seed it from `bench/ds4-2026-10-05/route-probe/ds4routes.bin`), adaptive tier, CPU pool with the
  doorbell overlap, `GpuPlanSink` PCIe share, pinned arena within a RAM budget (pin <= ~60 GB).
- Decode loop as a token graph once the reference path matches.
- **Gate:** decode tok/s with the same 4 prompts x 320 tokens as `bench.sh`, against B. >= 20 tok/s = continue;
  < 19.4 (B + 20%) = stop (s1 kill criterion). Re-check parity (top-1 > 99% vs golden).

### Phase 5 - KV formats, long context, prefill
- Raw window ring (d_h 512, int8 like Strata's KV), CSA/HCA/indexer compressed caches, F32 compressor ring states
  (`DSV4_ARCH_SPEC.md` s3.9), the lightning indexer kernel; chunked prefill with tensor-core GEMMs and streamed
  experts (`prefill.hpp:44`).
- **Gate:** needle test at 64K and 128K; prefill >= 175 tok/s at 7.8K.

### Phase 6 - speculation (optional)
- No MTP in this GGUF. **Best option: huihui's matching abliterated DSpark draft,**
  `huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/dspark-abliterated/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`
  (10.1 GB; DSpark = llama.cpp arch `DFLASH`, `src/models/dflash.cpp`). **Downloaded (2026-10-05) to
  `/media/mal/SSD NVME/Models/huihui-DeepSeek-V4-Flash-0731-dspark-abliterated/`** (`fetch.sh` there resumes +
  sha256-verifies; `fetch.log` says VERIFIED OK when done). Quick test before any port work: llama.cpp-master-rebase
  with it as the draft (`-md`) on top of config B - if it beats 16.2 tok/s, that is a free win today. Other options: a GGUF with the `nextn` block (the converter writes it when
  present), or a separate draft (DSpark), or Strata's suffix/prompt-lookup drafter alone (`SuffixDrafter`,
  works without a draft model; helps code edits).
- **Gate:** decode faster with drafts than without, on the same bench.

### Phase 7 - server and release
- `serve/`: tokenizer + DeepSeek chat template (DSML tool calls, `<think>`, `Reasoning Effort:` preamble;
  `models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja` in llama.cpp), config, `setup.py` model entry.
- Point `~/.local/bin/deepseek-local-quetza` at it (today it is broken: CUDA-12 llama.cpp build).
- README section with measured numbers; push branch; consider a PR or a separately named repo.

## 5. Machine notes (things that already bit us)

- Load DeepSeek with `--load-mode none` in llama.cpp: mmap on the exFAT NVMe + 90 GB RAM swapped and stalled the
  load for 10+ min with the GPU idle. In Strata, prefer explicit reads into a pinned arena.
- RAM: 90 GB, a known bad byte lane at ~56.6 GB with soft-offline mitigations (memory note `ram-fault-2026-09-18`).
  Keep a MemAvailable watchdog (kill < 3 GB) on every big run - `bench.sh` has one.
- `/tmp` is wiped on reboot: keep tools/results in the repo (`bench/`, `tools/ds4/`).
- Never `pgrep -f`/`pkill -f` a pattern that also appears in your own command line (it killed the shell 3x).
- Build: CUDA 13.4 at `/usr/local/cuda`, sm_89. Strata's build: `setup.build_engine(...)` or the cmake line in
  `setup.py` `build_engine` (Release, Ninja, `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
  -DSTRATA_GGML_DIR=third_party/llama.cpp`). `third_party/llama.cpp` is not in git: copy it from
  `~/AI/Strata/third_party/llama.cpp` (same pinned commit).
- The production Strata (`~/AI/Strata`, Qwen, serving Mal's daily wrappers) is separate - never develop there.

## 6. Effort and risk

| Phase | Size | Main risk |
| --- | --- | --- |
| 0 | small | none |
| 1 | medium | tokenizer pre-split details |
| 2 | small | - |
| 3 | **large** | hyper-connections + compressor + indexer exactness; RoPE/YaRN details |
| 4 | large | the go/no-go: PCIe + CPU split may not reach 20 tok/s |
| 5 | large | long-context memory layout |
| 6 | medium | no draft head in this GGUF |
| 7 | medium | - |

## 7. Fallback if the kill criterion fires

Put the two ideas that matter into `llama.cpp-master-rebase`'s MoE expert cache instead: (1) send a share of the
misses to the GPU over PCIe while the CPU computes the rest, (2) persist/learn the cache contents (profile save).
Same measured gain mechanism, no new attention code, and it upstreams to every llama.cpp user.

## 8. Later: DeepSeek-V4.1-Flash

`pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF` (arch `deepseek41`, 552B backbone + 196B Engram, 415.8 GB) does not
fit this PC (207 GB backbone vs ~104 GB RAM+VRAM). Its Engram tables are Strata's PLE idea (SSD rows per token),
so a V4 port is the stepping stone for 256 GB+ machines.
