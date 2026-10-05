# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're porting the Strata inference engine (built for Qwen3.8-Flash-Next) to DeepSeek-V4-Flash, so DeepSeek runs
faster on my RTX 4090 + Ryzen 7 5700X + 90 GB DDR4 than llama.cpp does. **Phases 0, 1, 2 and 4a are done;
Phase 3 (the oracle gate) is RED at layer 2.** Use /conductor. **Status table: `docs/ds4/FINDINGS.md` s8.**
(Last session ended at a scheduled poweroff, 2026-10-05 23:07 - work was committed, nothing is in flight.)

**Read first, in this order** (all in `~/AI/Strata-DS4`, branch `deepseek4`, fork github.com/Indras-Mirror/Strata):
1. `docs/ds4/PORT_PLAN.md` - the plan: goal, the number to beat, phases with gates, kill criterion, machine notes.
2. `docs/ds4/FINDINGS.md` - s8 = status; **s9 = the Phase 3 gate (RED, layer-2 CSA)**; **s10 = the Phase 4a MoE
   verdict**. Also s2 (routing skew) and s3 (llama.cpp baseline).
3. `docs/ds4/DSV4_ARCH_SPEC.md` - DeepSeek-V4's exact computation, cited to llama.cpp source.
4. `docs/ds4/STRATA_ARCH_MAP.md` - what in Strata is Qwen-specific vs reusable, cited to Strata source.
5. `docs/ds4/GGUF_INVENTORY.txt` - every tensor in the model file.
6. Memory notes `strata-ds4-port` and `strata-0139-lazy-vision`.

**Where the work stands (2026-10-05)**
- **Phase 0** done except the antirez `ds4` engine run (still todo).
- **Phase 1** done, gates verified. **Phase 2** done on BOTH paths (CPU 1.8e-7 / GPU 3.0e-5 << 1e-3; 0.71 / 0.017 ms
  per expert) - FINDINGS s6, commit `4d3463d`.
- **Phase 3** (reference forward, `tools/ds4/ds4_ref.cpp`): **gate RED.** `bash tools/ds4/run_ref_gate.sh p64` =
  456 FAIL / 21 pass. Exact through layer 1; **first divergence is layer 2 - the first CSA (ratio-4) layer**,
  inside the attention module (`attn_csa_lid-2` cos 0.9033, `attn_out-2` 0.8245), poisoning everything downstream.
  Two earlier real bugs were fixed on the way (`dcecd11`: `hc_scale` byte offsets, `attn_out` dump aliasing).
  **Step 1 is already answered**: the goldens' `attn_csa_lid`/`attn_hca` are **NOT** post-Hadamard with `-ctk f16`
  (no `k_rot` touch the raw/csa/hca caches; only `kv_lid` is rotated, which the ref already does) - so this is a
  real arithmetic bug, not a comparison artifact. A structural re-read of `overlap_compress` found no difference,
  so **probe taps were added** (`a7318e2`): `tools/ds4/golden_dump.cpp` captures `q`, `kv`, `csa_state_kv`,
  `csa_state_score_ape`, `csa_state_compress` under `DS4_GOLDEN_PROBE=1`, and `tools/ds4/ds4_ref.cpp` dumps the
  same names under `DS4_PROBE=1` - diff them tensor by tensor to isolate the layer-2 CSA bug.
- **Phase 4a** (MoE engine replay on real routes) **DONE** - FINDINGS s10. Best arm **17.28 tok/s** implied
  (slots 2300 / 15.1 GiB, un-captured loop), which beats the bar 16.34 but not the go line 19.6, so **the kill
  criterion fires as written** - BUT ~13 ms/token of the wall is an un-captured per-layer sync + ~10 launches +
  a 128 KB D2H, overlapped with neither engine; slots-2300 minus that = 44.9 ms = **22.3 tok/s, above the go
  line**. **Do NOT invoke the s7 llama.cpp fallback** - the missing piece is Phase 4b's captured token graph.
  Also: the harness **found and fixed a real engine bug** (`173e535`) - `ExpertPool::run_split_multi_native`
  sized its row split with the compile-time Qwen H/FF (2560/640) instead of the model's, computing ~1/3 of every
  expert.

**Key facts**
- Model: `/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf` (80.76 GiB, arch `deepseek4`, 43 layers,
  256 experts top-6, experts IQ2_XXS gate/up + Q2_K down; **no MTP layer in the file**).
- Oracle: `~/AI/llama.cpp-master-rebase/build/bin` (CUDA 13.4, has deepseek4 + `--moe-expert-cache`). Use F16 KV
  for parity runs. `llama.cpp-new` and `-dspark` are CUDA-12 builds and do not run as-is.
- Bar to beat: llama.cpp `-cmoe --moe-expert-cache 40 --load-mode none` = 16.34 tok/s decode, 154 tok/s prefill
  (clean run; llama.cpp's default placement = 10.85). Target >= 20 tok/s, go line 19.6.
- **Measured corrections to the plan's arithmetic (s10):** the CPU miss path is **10-16 GB/s, not ~25-30** - the
  5700X has no AVX-512, so Strata's native IQ2_XXS/Q2_K path runs ggml-cpu AVX2 dots; Qwen's 36 GB/s is the
  AVX-512 Q2_0 kernel and does not transfer. PCIe measured **15.8 GB/s**. Held-out hit rate 50.9% at 12 GB.
- The win must come from splitting cache misses between CPU and a PCIe/GPU share in parallel, overlap and pinned
  memory - NOT from smarter cache allocation (measured: +0.3 points only).
- Route probe: `tools/ds4/route_probe.cpp` + `route_skew.py`; routes `bench/ds4-2026-10-05/route-probe/ds4routes.bin`.
- DSpark is dropped (it made decode slower - FINDINGS s7).

**Rules**
- Never develop in `~/AI/Strata` (production Qwen engine serving my wrappers). Work only in `~/AI/Strata-DS4`.
- Correctness before speed: each phase ends at its gate; record numbers in FINDINGS.md and commit before moving on.
  Measure before concluding; reproduce before diagnosing.
- One full-model process at a time (80 GB model, 90 GB RAM); wrap every GPU run / full-model load / large pin in
  `flock ~/.quetza-data/conductor/ds4-gpu.lock`. `free -g` before pinning (abort if MemAvailable < 3 GB). Ask
  before pushing to GitHub.
- Keep tools and results in the repo, not `/tmp` (wiped on reboot).

**Next order:**
1. **Finish Phase 3**: use the `DS4_GOLDEN_PROBE` / `DS4_PROBE` taps to diff the layer-2 CSA path tensor by tensor
   and fix the arithmetic in `ds4_ref.cpp`. Gate: `bash tools/ds4/run_ref_gate.sh p64` to green (cos > 0.9999),
   then p600/p3000 (top-1 > 0.99, KL < 0.01, teacher-forced). Do NOT weaken thresholds.
2. **Phase 4b**: the captured token graph (CUDA graph per token) over the s10 engine - the ~13 ms/token of
   un-overlapped loop cost is the whole 17.28 -> 22.3 gap. Gate: >= 19.6 tok/s vs 16.34, top-1 > 99% vs golden.
3. **antirez ds4** run (`~/AI/ds4-ref/ds4`, built for CUDA) on the same prompts with `--ssd-streaming`.
4. Residuals to pick up: the adaptive tier (learned swap) is not implemented (static profile seed, no eviction);
   the shared expert is not computed in the replay (folded into the 17.6 ms constant).
