# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're porting the Strata inference engine (built for Qwen3.8-Flash-Next) to DeepSeek-V4-Flash, so DeepSeek runs
faster on my RTX 4090 + Ryzen 7 5700X + 90 GB DDR4 than llama.cpp does. **Phases 0, 1, 2 and 4a are done;
Phase 3's structural bug is fixed (p600/p3000 logits PASS) but its strict per-tensor gate is still red on a
numeric drift.** Use /conductor. **Status table: `docs/ds4/FINDINGS.md` s8.**
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
- **Phase 3** (reference forward, `tools/ds4/ds4_ref.cpp`): **structural bug FIXED, strict gate still red on numerics.**
  The compressed-attention **mask was filled transposed** (`7fab6eb`): `make_comp` wrote `m16[b*nt + t]`, but ggml
  indexes the mask as `mask[key + query*ne0]`, so element `(block b, query t)` lives at `t*n_blocks + b` - every
  query saw only the last few compressed blocks. Fixed in both the f16 and f32 masks. Proven by rebuilding the
  attention in numpy from ds4_ref's own dumped `q-2`/`k_all-2`: the transposed mask reproduces ds4_ref at cos
  1.000000, the correct mask reproduces the golden at 0.999979. After the fix (`dec40d8`): p64 **35/477** (was 21),
  p600 42/477, p3000 38/477; `attn_csa_lid-2` 0.999979, `attn_hca-3` 0.999975; **p600 logits top1 1.0 KL 0.0086,
  p3000 top1 1.0 KL 0.0079 - both PASS the < 0.01 KL gate.** Earlier real bugs fixed on the way: `dcecd11`
  (`hc_scale` byte offsets, `attn_out` dump aliasing). Step 1 answered: the goldens are **NOT** post-Hadamard under
  `-ctk f16` (only `kv_lid` rotates), so this was arithmetic, not a comparison artifact.
  **Remaining (the only thing red): a slow compounding numeric drift**, not structural - `attn_raw-0` matches to
  2.2e-8 but `attn_out-0` (de-rope + two grouped Q8_0 output-LoRA matmuls) is 0.9999892 and `ffn_moe_out-0` 0.999657,
  growing ~1e-3/layer until MoE routing flips discrete top-6 picks. **INFERRED cause: ggml-cpu quantizes the
  *activations* to Q8_0 for Q8_0/Q2_K matmuls while the CUDA oracle does not.** Confirming test: re-run the oracle
  with those tensors promoted to F16 (or on the CPU backend) and see whether the residual collapses. Do NOT relax
  the 0.9999 per-tensor gate before that test. Probe taps: `DS4_GOLDEN_PROBE=1` (golden_dump) / `DS4_PROBE=1` (ds4_ref).
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
1. **Finish Phase 3**: run the confirming test for the residual numeric drift - re-run the oracle with the
   Q8_0/Q2_K tensors promoted to F16, or on the CPU backend, and see whether `attn_out-*`/`ffn_moe_out-*` collapse
   toward the golden. If it does, the drift is CPU-vs-CUDA activation quantization and the per-tensor 0.9999 gate
   needs a CPU-vs-CUDA-justified tolerance (document the evidence); if not, the residual is still in `ds4_ref.cpp`.
   p600/p3000 logits already pass; the goal is p64 green. Do NOT weaken thresholds before that test.
2. **Phase 4b**: the captured token graph (CUDA graph per token) over the s10 engine - the ~13 ms/token of
   un-overlapped loop cost is the whole 17.28 -> 22.3 gap. Gate: >= 19.6 tok/s vs 16.34, top-1 > 99% vs golden.
3. **antirez ds4** run (`~/AI/ds4-ref/ds4`, built for CUDA) on the same prompts with `--ssd-streaming`.
4. Residuals to pick up: the adaptive tier (learned swap) is not implemented (static profile seed, no eviction);
   the shared expert is not computed in the replay (folded into the 17.6 ms constant).
