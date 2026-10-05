# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're porting the Strata inference engine (built for Qwen3.8-Flash-Next) to DeepSeek-V4-Flash, so DeepSeek runs
faster on my RTX 4090 + Ryzen 7 5700X + 90 GB DDR4 than llama.cpp does. Phases 0-2 are mostly done and Phase 3
(reference forward pass) is written but not yet checked against the goldens. **Status table: `docs/ds4/FINDINGS.md`
section 8.** This is an overnight run: the GPU and RAM are yours until I'm back. Use /conductor.

**Read first, in this order** (all in `~/AI/Strata-DS4`, branch `deepseek4`, fork github.com/Indras-Mirror/Strata):
1. `docs/ds4/PORT_PLAN.md` - the plan: goal, the number to beat, phases with gates, kill criterion, machine notes.
2. `docs/ds4/FINDINGS.md` - what was measured: routing skew (12 GB cache = ~50% hits) and the llama.cpp baseline.
3. `docs/ds4/DSV4_ARCH_SPEC.md` - DeepSeek-V4's exact computation, cited to llama.cpp source.
4. `docs/ds4/STRATA_ARCH_MAP.md` - what in Strata is Qwen-specific vs reusable, cited to Strata source.
5. `docs/ds4/GGUF_INVENTORY.txt` - every tensor in the model file.
6. Memory notes `strata-ds4-port` and `strata-0139-lazy-vision`.

**Key facts**
- Model: `/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf` (80.76 GiB, arch `deepseek4`, 43 layers,
  256 experts top-6, experts IQ2_XXS gate/up + Q2_K down; **no MTP layer in the file**).
- Oracle: `~/AI/llama.cpp-master-rebase/build/bin` (CUDA 13.4, has deepseek4 + `--moe-expert-cache`). Use F16 KV
  for parity runs. `llama.cpp-new` and `-dspark` are CUDA-12 builds and do not run as-is.
- Bar to beat: llama.cpp `-cmoe --moe-expert-cache 40 --load-mode none` = 16.34 tok/s decode, 154 tok/s prefill
  (clean run; llama.cpp's default placement = 10.85). Target >= 20 tok/s.
- Kill criterion: if Phase 4 can't beat the bar by 20%, stop and move the ideas into llama.cpp's expert cache.
- The win must come from splitting cache misses between CPU and a PCIe/GPU share in parallel, overlap and pinned
  memory - NOT from smarter cache allocation (measured: +0.3 points only).
- Bench script with a RAM watchdog: `bench/ds4-2026-10-05/llamacpp-ab/bench.sh` (copy to /tmp to run; it needs
  ~75 GB RAM and the GPU to itself). Route probe: `tools/ds4/route_probe.cpp` + `route_skew.py`.

- Draft model for speculation (no MTP in the main GGUF): huihui's abliterated DSpark Q8_0 at
  `/media/mal/SSD NVME/Models/huihui-DeepSeek-V4-Flash-0731-dspark-abliterated/` (check `fetch.log` for
  VERIFIED OK; re-run `fetch.sh` to resume). Worth a quick llama.cpp test on top of config B early on.

**Rules**
- Never develop in `~/AI/Strata` (production Qwen engine serving my wrappers). Work only in `~/AI/Strata-DS4`.
- Correctness before speed: each phase ends at its gate in PORT_PLAN.md; record the measured numbers in
  FINDINGS.md and commit before moving on. Measure before concluding; reproduce before diagnosing.
- Overnight: the GPU is approved. One full-model process at a time (80 GB model, 90 GB RAM); wrap every GPU run or
  full-model load in `flock ~/.quetza-data/conductor/ds4-gpu.lock`. Still ask before pushing to GitHub.
- Keep tools and results in the repo, not `/tmp` (wiped on reboot).
- Delegate broad code reading to `deepseek-quetza pro --skip "..."` workers; use the conductor skill for
  multi-slice implementation if it fits.

**Overnight order:**
1. Phase 3 oracle gate: `tools/ds4/run_ref_gate.sh` (ds4_ref vs goldens; plan gate: hc-stream cosine > 0.9999 at p64,
   top-1 > 99% and KL < 0.01 on p600/p3000). Fix and re-run until it passes. Goldens are bit-exact run to run.
2. Phase 2 GPU parity: the `ds4_expert_parity` GPU command in FINDINGS s6.
3. Phase 4a: dispatch the ready packet `~/.quetza-data/conductor/packets/p4a-moe-engine.md` (MoE engine replayed on real
   routes; drop its "wait for .done-p0-gpu-runs" lines, the GPU is free). Then 4b: full decode loop, and the gate (>= 19.6
   tok/s vs 16.34, top-1 > 99%).
4. Quick: run antirez ds4 (`~/AI/ds4-ref/ds4`, already built for CUDA) on the same prompts with `--ssd-streaming`, for its
   4090 speed.
DSpark is dropped: it made decode slower (FINDINGS s7). Record numbers in FINDINGS and commit after every step.
