# MiMo-V2.6-Flash in Strata - start prompt for a second session (written 2026-10-07)

Paste everything below the line into a fresh Claude Code session started in `~/AI` (NOT in `~/AI/Strata-DS4`).

---

We're porting the Strata inference engine to **MiMo-V2.6-Flash** (arch `mimo2`) so it decodes faster than llama.cpp
on my RTX 4090 24 GB + Ryzen 7 5700X (8c, AVX2 only, no AVX-512) + 90 GB DDR4. Another Claude session is working on the
DeepSeek-V4-Flash port in `~/AI/Strata-DS4` at the same time - **do not edit files there**. Work solo (no /conductor,
no worker dispatch - they burn tokens). Measure before concluding; correctness before speed; say what was NOT tested.

## Setup (do this first)
1. Create your own worktree from the DS4 branch (it has the reusable expert tier, CUDA build fixes, clamp, memguard):
   `cd ~/AI/Strata-DS4 && git worktree add ~/AI/Strata-MiMo -b mimo deepseek4` - then work ONLY in `~/AI/Strata-MiMo`.
   Don't push; commit your own paths only.
2. Build tree: `cmake -S . -B build-mimo-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
   -DSTRATA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp` (STRATA_GGML_CUDA
   also turns on GGML_CUDA_GRAPHS - without graphs, decode is launch-bound). ggml-cuda takes ~10 min to compile.

## Read first
- `docs/mimo/MIMO_V26_FEASIBILITY.md` - desk study (fit, speed estimates, uncensored options). Nothing in it is measured.
- `docs/ds4/FINDINGS.md` s10-s12 (machine physics: CPU+PCIe share DDR4, ~30 GB/s total; predicted prefetch) and
  `docs/ds4/ENGINE_MOE.md` (the expert tier).
- The DS4 engine you will mirror: `tools/ds4/ds4_dense.{hpp,cpp}` (dense half on a ggml backend, one graph per layer,
  CUDA graphs, per-token input span, fused ops), `tools/ds4/ds4_moe.{hpp,cpp}` (Ds4MoeTier: VRAM slot cache seeded
  from a routing profile, pinned RAM arena, CPU pool, PCIe share, predicted prefetch, `run_multi`),
  `tools/ds4/ds4_generate.cpp` (decode loop, `--slots auto`, `--ppl`, `--dense-requant`, `--route-bias`).

## Files (downloaded + sha256-verified 2026-10-06)
`/media/mal/SSD NVME/Models/MiMo-V2.6-Flash-RL-GSQ-RCO/` (NTFS; if missing after a reboot: `udisksctl mount -b /dev/nvme0n1p3`):
- `MiMo-V2.6-Flash-RL-GSQ-RCO-3bit.gguf` (115.7 GB; experts Q2_K/Q3_K/MXFP4 chosen per tensor - see quantization.json)
- `MiMo-V2.6-Flash-RL-Uncensored-Heretic-lora.gguf` (70 MB rank-1 LoRA, llama.cpp `--lora`)
- llama.cpp oracle: `~/AI/llama.cpp-master-rebase/build/bin` (CUDA 13.4, has mimo2).

## Safety rules (non-negotiable - this box has frozen before)
- **Every full-model load goes through `tools/ds4/memguard.sh <cap_gib> <need_gib> -- <cmd>`** (cgroup RAM cap, swap
  off, MemAvailable watchdog, refuses while ComfyUI runs). It also takes the shared GPU lock
  `~/.quetza-data/conductor/ds4-gpu.lock` - the DS4 session uses the same lock, so your full-model runs will WAIT for
  theirs and vice versa. Never bypass it; never start a second full-model process.
- NEVER a CPU-only full-model run (`-ngl 0` froze the box). The 3-bit file does NOT fit RAM+VRAM (~8 GiB must page
  from NVMe) - llama.cpp must use mmap for it (not `--load-mode none`), under memguard.
- Small tests: `nice -n 10 systemd-run --user --scope -q -p MemoryMax=<n>G -p MemorySwapMax=0 <cmd>`.
- Before any GPU run check `ss -ltn | grep 8188` (ComfyUI) and `nvidia-smi`; if ComfyUI is up, stop and tell me.
- Never `pkill -f`/`pgrep -f` a pattern in your own command line; kill only PIDs you started. Keep results in the repo
  under `bench/mimo-<date>/`, not /tmp.

## Plan (cheapest first; report numbers after each step)
1. **llama.cpp baseline** on the 3-bit file: `-ngl 99 -cmoe -fa on -t 8` (+ try `--moe-expert-cache N` if this build
   has it), decode tok/s over >= 256 tokens and prefill tok/s, with and without `--lora` (Heretic). This is the bar.
2. **Routing skew** (decides the VRAM hit rate): generalise `tools/ds4/route_probe.cpp` (DS4 = top-6, MiMo = top-8, 47
   routed layers, layer 0 is dense) and run `tools/ds4/route_skew.py` - hit rate vs slot count on held-out tokens.
3. **Expert kernels**: MiMo's experts are Q2_K / Q3_K / MXFP4, varying per tensor. Check which the native kernels
   support (`STRATA_GU_FMTS` / `STRATA_D_FMTS` in `src/kernels/cuda/iq_kernels.cu`; CPU `native_fmt` covers anything
   ggml-cpu has). Missing ones: add CUDA grouped-kernel support (mirror an existing type) or requantise at load.
   Ds4MoeTier currently assumes ONE format for all layers - it needs a per-layer (per-tensor) format.
   Note: ggml-cpu's swiglu has no clamp for MiMo; DS4's swiglu_limit defaults to +inf so it is a no-op for you.
4. **Dense half** (`Mimo2Dense`, like Ds4Dense but much simpler: hybrid SWA(128)/global attention, GQA 4/8 kv heads,
   head 192/v 128, partial RoPE 0.334, SWA sink bias, value scale 0.707, layer 0 dense FFN). Gate = logits vs llama.cpp
   (top-1 match, KL small) on a few prompts, then coherent text, then tok/s vs step 1.
5. **Fit**: 3-bit spills ~8 GiB to NVMe. Options: a "VRAM-only resident" arena mode (don't duplicate cached experts in
   RAM), `--dense-requant`-style VRAM savings, or a ~2.3 bpw requant (fits fully, est. 16-20 tok/s). Measure first.
6. LoRA: if the Heretic LoRA touches expert tensors, the tier needs a rank-1 correction per expert (cheap) - check
   which tensors it has before designing.

Write findings to `docs/mimo/FINDINGS.md` as you go (every gate + number, what failed), commit after each step.
