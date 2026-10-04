# The existing `ds4` engine: DwarfStar by antirez

Phase 0 of `PORT_PLAN.md` asked what huihui's model card means by
`./ds4 -m DeepSeek-V4-Flash-Q2-0731.gguf --ctx 32768`. It is **DwarfStar**, the
`antirez/ds4` project. This is a survey of it as of 2026-10-05, with the
concrete verdict for our port: **do not build the native engine on it; borrow
three specific mechanisms into `llama.cpp-master-rebase`'s expert cache.**

## 1. What it is, where it lives, how alive it is

- Repo: <https://github.com/antirez/ds4> (model card links the branch
  <https://github.com/antirez/ds4/tree/ds4f-mxfp4>).
- Author: Salvatore Sanfilippo (antirez) plus 29 other contributors (30 total).
  `LICENSE` is MIT (`Copyright (c) 2026 The ds4.c authors`, plus the GGML and
  DeepSeek notices because quant layouts/tables/CPU dot logic are adapted from
  llama.cpp).
- Self-contained C (plus one Objective-C file for Metal and `.cu` for CUDA);
  `ds4.c` is a single ~2.9 MB / 71k-line file. It deliberately **does not link
  GGML** and is **not a general GGUF runner** — it accepts only DeepSeek V4
  Flash/PRO and GLM 5.2/5.3 GGUFs with the expected tensor layout (`README.md`
  "Model Weights"; `AGENT.md`).
- Activity: created 2026-05-06, **23,429 stars**, 2,251 forks, last push
  2026-09-20 (`main` @ `0aaea5a238fb41a35106a551e73c8409dfb751ac`). Very active.
- **The branch the model card points at is stale.** `ds4f-mxfp4` @
  `80df56af4070d0fc62f6f9682b1854f8e5be8b00` (2026-08-01) is seven weeks behind
  `main`, and it predates the CUDA expert cache: in that branch
  `ds4_gpu_stream_expert_cache_budget_for_expert_size` returns 0,
  `ds4_gpu_set_streaming_expert_cache_budget` is a no-op and
  `configured_count` returns 0 (`ds4_cuda.cu:27260-27395` @ branch), and
  `tests/test_cuda_ssd_cache.c` / `tests/test_cuda_ssd_batch.c` do not exist.
  "Use the latest ds4" therefore means `main`, which is what this survey cites
  (line numbers marked `main` are `ds4.c`/`ds4_cuda.cu` at `0aaea5a`).

## 2. Backends

`make cpu`, `make cuda CUDA_ARCH=sm_89`, `make cuda-spark`, `make cuda-generic`,
`make strix-halo`. The CPU backend is explicitly a reference/debug path only
(`AGENT.md:12`, `README.md` "Backends"); production is Metal (primary, 96 GB+
Macs), CUDA (Linux, multi-GPU tensor/expert parallel and DGX Spark), and ROCm
(Strix Halo). We confirmed the Linux CPU-only build compiles cleanly on this
machine: `make cpu -j8` ⇒ exit 0, five binaries produced (no GPU touched, model
not run).

## 3. How it puts 43×256 routed experts on one GPU (the part we care about)

Two distinct CUDA modes:

**Resident placement.** Model tensors are placed across the devices in
`--gpu-devices` with per-device `--gpu-vram` budgets, and the process
**refuses to start if layers would spill to the CPU** (`README.md` "Tensor
Parallelism across CUDA GPUs"). A 24 GB card cannot hold the 80.76 GiB Q2 model
this way.

**SSD streaming** (`--ssd-streaming`, plus `--ssd-streaming-cache-experts
N|NGB`). This is the only path open to us, and it is a real expert cache:

- A **global VRAM expert cache** sized in *experts*, not bytes: budget resolved
  through `ds4_gpu_stream_expert_cache_budget_for_expert_size()` with the
  8 GiB headroom reserve and automatic shrink to free VRAM
  (`ds4_cuda.cu:27181-27220` main). Eviction is LRU by a monotonically
  increasing use stamp over a flat slot array indexed by the expert's
  **weight-file offset** (`g_stream_expert_by_gate`, `ds4_cuda.cu:27223-27282`).
  Hits are re-stamped before misses are loaded so a miss cannot evict a hit
  (`:27231-27246`). This is a global cache, matching our measured finding that
  global ranking beats per-layer slots by only ~0.3 points.
- **Misses are file reads plus PCIe copies, never CPU compute.**
  `cuda_model_copy_to_device_streamed()` reads the expert blob (O_DIRECT
  `pread` by default, `DS4_CUDA_NO_DIRECT_IO=1` to disable) into 8 MiB staged
  buffers and issues `cudaMemcpyAsync` H2D on a dedicated upload stream, with a
  ring of four chunks and CUDA events for flow control; pages are dropped with
  `posix_fadvise(POSIX_FADV_DONTNEED)` after the copy
  (`ds4_cuda.cu:2105-2230`, `:22953-23087`, staging pool at `:1640-1730`).
- **Overlap via a next-layer read-ahead thread.** A single reader thread
  (`cuda_stream_prefetch_read`, `ds4_cuda.cu:27349-27395`) walks the *next
  layer's* selected experts while the current layer's graph runs, using two
  8 MiB staging buffers double-buffered with events. Reserved slots are
  invisible until every copy finishes and are then published with `used = 0`
  so "look-ahead is not evidence of reuse" and unused read-ahead is evicted
  before demand-hot experts (`:27395-27400` and the comment there).
- **Pinned / host-registered memory.** The mmap'd model is registered with
  `cudaHostRegister(..., cudaHostRegisterMapped | cudaHostRegisterReadOnly)`
  (`ds4_cuda.cu:3864`, `:4016`), and TP bulk staging uses `cudaHostAlloc`
  (`:33680`). On integrated/ATS GPUs (DGX Spark) it also uses
  `cudaMemPrefetchAsync` over the whole mapping (`cuda_model_prefetch_range`,
  `:1985-2066`).
- **Hot-expert preload** exists but is **Metal-gated**: per-model hot lists are
  compiled into the binary (`ds4_streaming_hotlist.inc`, 13,334 lines;
  `ds4_default_streaming_hotlist_flash` / `_pro` / `_glm52`) and a runtime
  profiler can write a hotlist file (`ds4.c:1580-1900`); the preload call site
  is `ds4.c:33877` inside the `metal_graph_streaming_expert_hotlist_*` family.
  On CUDA the cache warms only from actual routing.

**What it does not do:** there is no CPU side to the expert math in a GPU build.
`DS4_NO_GPU` swaps in a completely separate CPU path (`ds4_cpu.o`,
`ds4_cli_cpu.o`) used only for reference; the CUDA graph computes every routed
expert on the GPU. So of our four Phase-4 mechanisms, DwarfStar implements the
hot-expert cache, the overlap/prefetch, and pinned/staged memory — **but not
the parallel CPU+GPU split of misses (`GpuPlanSink`, `--pcie-frac`), which our
`PORT_PLAN.md` §2 calls the whole case for the port.**

## 4. Quants, attention/KV, speculation

The CUDA kernels are exactly our mix: `IQ2_XXS` gate/up and `Q2_K` down
(`cuda_block_iq2_xxs` / `cuda_block_q2_K` at `ds4_cuda.cu:63,81`; device dots at
`:19615`, `:19925`), plus Q4_K/Q5_K/Q6_K and MXFP4 for other repos. The 2-bit
quant is asymmetrical by design (only routed experts quantized), same as ours.

The DSV4-specific attention is fully implemented, because this is a
model-specific engine: raw 128-token sliding-window KV ring, CSA (ratio 4) and
HCA (ratio 128) compressor tensors, the lightning indexer (64 heads × 128 dim,
top-512), attention sinks, hyper-connections with **20-iteration Sinkhorn**
(`n_hc_sinkhorn_iter = 20`, `ds4.c:627,664,782`), compressor/indexer tensors
(`ds4.c:4639-4650`), YaRN RoPE on the compressed layers with base 160000
(`DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE`), and a 48-byte-header disk KV payload
that stores raw ring rows, compressed rows and compressor/indexer frontiers
(`README.md` "Disk KV Cache"). The shapes match `DSV4_ARCH_SPEC.md`.

Speculation: legacy MTP and DSpark are opt-in and "at most a slight speedup";
DSpark needs a separate ~5.6 GiB support GGUF, is not supported for PRO, and the
runtime is described for Metal (resident or streaming) — not advertised for
CUDA. This agrees with our plan treating Phase 6 as optional and last.

## 5. Claimed speeds and hardware (no discrete consumer GPU data)

From `README.md` (Metal CLI, `--ctx 32768`, greedy, `-n 256`) and the QA
reference table:

| Machine | Quant | Prefill | Decode |
| --- | --- | ---: | ---: |
| M3 Max 128 GB, long prompt | q2 | 250 t/s | 21.5 t/s |
| M5 Max 128 GB, short | q2 | 87 t/s | 34.3 t/s |
| M3 Ultra 512 GB, short | q2 | 84 t/s | 36.9 t/s |
| **DGX Spark GB10 128 GB, 7047 tok** | **q2** | **343.8 t/s** | **13.75 t/s** |
| V4.1 Q2 Spark, 64 GiB cache, 32K ctx | q2 | ~100 t/s | ~9.3 t/s |
| 8×L40S server, resident Q4 | q4 | 2000 t/s | 120 t/s agg |

The only single-GPU CUDA streaming reference is the **DGX Spark**, a 128 GB
unified-memory part — not a discrete 24 GB card over PCIe. There is no RTX 4090
or any discrete-consumer-GPU number for DeepSeek streaming. `cuda-generic`
builds exist, but the documented CUDA SSD-streaming target is Spark-class.

## 6. Fit to our machine and plan

1. **Capacity.** On the 4090 the model must stream. `ds4_backend_supports_ssd_streaming`
   allows CUDA on Linux (`ds4.c:433` main), but the *automatic* cache budget is
   Metal/ROCm-only (`ds4.c:445`, `DS4_ROCM_BUILD`), so we would have to pass an
   explicit budget; single-GPU is enforced (`ds4_cuda.cu:27149`).
2. **Same hit rate, worse miss path.** ~12 GB of VRAM left for experts is ~16%
   of 11,008 slots — by our own `route_skew.py` measurements that is a ~50% hit
   rate, i.e. the same cache llama.cpp's `-cmoe` already runs. But ds4's misses
   are NVMe O_DIRECT reads + PCIe H2D onto a GPU that also computes everything;
   llama.cpp B instead computes every miss on the CPU while the mostly-idle GPU
   waits. ds4 is **strictly more bandwidth-limited** in the 24 GB discrete case,
   and has no CPU-compute path to hide it. There is no evidence it beats the
   16.34 tok/s bar here, and the design of the miss path suggests it will be
   below it. (We cannot measure — the GPU is off-limits for this scout.)
3. **RAM.** The 80.76 GiB model does not sit comfortably in 90 GB RAM with the
   known bad byte lane, and ds4 (by design) streams from the GGUF and drops
   pages, leaning on the NVMe; llama.cpp B keeps all experts host-resident.
4. **Parity oracle.** ds4 is validated against the **official DeepSeek API**
   continuation/logprob vectors (`tests/test-vectors`, `--dump-logprobs`), not
   against llama.cpp. Our Phase 3 gate requires matching llama.cpp golden
   dumps. Adopting ds4 means adopting a different oracle and discarding our
   Phase 1–3 work.
5. **Scope.** ds4 is a whole replacement engine (tokenizer, template, agent,
   server, Metal/CUDA/ROCm). "Building on it" means abandoning the Strata port
   plan, not extending it.

## 7. Verdict

**Do not build the native port on ds4, and do not adopt it as the engine.**
It does not implement the one differentiated idea our plan rests on (split the
expert misses between CPU compute and PCIe to the GPU), its supported streaming
target is a unified-memory Spark, not a 24 GB discrete card, and its parity
target is a different oracle. On this PC the likely result is at or below
llama.cpp B's 16.34 tok/s — the exact outcome our kill criterion would reject.

**Borrow these pieces into `llama.cpp-master-rebase`'s expert cache instead**
(plan §7 fallback), from `main` @ `0aaea5a`:

1. **Next-layer read-ahead thread with reserved-slot publish and
   "look-ahead is not a hit" accounting** — `ds4_cuda.cu:27309-27400`. This is
   the cleanest statement of the overlap idea and directly portable to
   llama.cpp's cache as a background prefetcher for layer L+1.
2. **Staged multi-chunk async H2D upload ring (4 chunks / CUDA events), with
   O_DIRECT pread and page dropping** — `ds4_cuda.cu:1640-1730`, `:2105-2230`.
3. **Global LRU cache by weight-file offset, protected prefetch slots, 8 GiB
   VRAM reserve + auto-shrink** — `ds4_cuda.cu:27171-27282`. A concrete design
   for making `--moe-expert-cache` a global (rather than per-layer) LRU with
   prefetch protection.
4. **Precomputed hotlist format + profiler** — `ds4.c:1580-1900` and
   `ds4_streaming_hotlist.inc`. A ready-made profile-file schema for our
   `--expert-profile-save` / cache warm-start.

**Use it as a correctness cross-reference, not a base.** It is an independent,
API-vector-validated implementation of the same DSV4 attention/compressor/
indexer/hyper-connection math; `./ds4 --dump-logits` gives a second opinion
when our Phase-3 reference path disagrees with llama.cpp. If a 128 GB-class
unified-memory GPU ever enters the picture, ds4 is a proven path (13.75 tok/s
Flash Q2 on a Spark) and would be the sensible engine — but that is not this
machine.

## 8. Reproduce this survey

```sh
git clone --depth 1 https://github.com/antirez/ds4.git ~/AI/ds4-ref   # main @0aaea5a
cd ~/AI/ds4-ref && make cpu -j8        # Linux CPU-only compile check, exit 0
# Key files: ds4.c, ds4_cuda.cu, ds4_streaming_hotlist.inc,
#            tests/test_cuda_ssd_cache.c, QA_BEFORE_RELEASES.md ("CUDA SSD Streaming")
```

Source links: repo <https://github.com/antirez/ds4>; card-linked branch
<https://github.com/antirez/ds4/tree/ds4f-mxfp4>; weights
<https://huggingface.co/antirez/deepseek-v4-gguf>; the huihui card we started
from <https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF>.
