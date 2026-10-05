# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're porting the Strata inference engine to DeepSeek-V4-Flash so it decodes faster than llama.cpp on my RTX 4090 +
Ryzen 7 5700X + 90 GB DDR4. Use /conductor. **Status table: `docs/ds4/FINDINGS.md` s8. Read s9, s11, s12 first.**
(Last session: 2026-10-06, ended cleanly - no workers, no GPU jobs, nothing in flight.)

## Where it stands (2026-10-06)
- **Phase 3 DONE** (`cb6af7b`): gate = p600/p3000 logits (top1 1.0, KL < 0.01); per-tensor cosine is a diagnostic
  (`compare_golden.py --tensors-diag`) - the residual was proven to be CPU-vs-CUDA flash-attn noise x Q8 activation
  rounding (`tools/ds4/check_act_quant.py`, s9). Small coherency check (`4725894`): ds4_ref and llama.cpp give the same
  next-token top-5 on all three prompts.
- **Phase 4b speed verdict: GO in the replay** (s11, s12). Real layer order alone = 18.7 tok/s (kill line 19.6) -
  the PCIe split can't help because CPU + PCIe share DDR4 (both at once = 29.9 GB/s total, `bw_contend.cu`).
  **Predicted expert prefetch** (layer l's router on the pre-attention state, ~63% recall@6; 1.43 experts/layer DMA'd
  under the dense work) measured **20.57 tok/s at 2150 slots** (the real VRAM budget) vs llama.cpp 16.34.
  SPEED ONLY: random activations, attention is a timed stand-in. No real generation yet.
- **Engine build started, NOT verified** - three conductor workers were stopped mid-slice (Mal left for work):
  - `ds4-dense` + `ds4-moe` WIP saved on branch **`ds4-engine-wip`** (`e0b1aa4`, 3233 lines: ds4_dense.{hpp,cpp},
    ds4_moe.{hpp,cpp}, tests, tools/ds4/cmake/*.cmake). Unknown whether it builds or passes. Test outputs from the
    dense worker are untracked in `bench/ds4-2026-10-06/dense/` (35 MB).
  - `llama-prefetch` (track "a": `--moe-prefetch N` in llama.cpp's expert cache) WIP on branch **`ds4-moe-prefetch`**
    in worktree **`~/AI/llama.cpp-ds4-prefetch`** (`583f5d173`, 1129 lines). Unverified.
  - Packets (the full specs + gates): `~/.quetza-data/conductor/packets/{ds4-dense,ds4-moe,llama-prefetch}.md`
    (+ `_ds4_rules.md` appended). The CMake hooks are already on deepseek4 (`6de177d`, OPTIONAL includes of
    `tools/ds4/cmake/ds4_{dense,moe,engine}.cmake`).

## Next
1. Re-dispatch the three slices with a **resume addendum**: "start from the WIP on branch ds4-engine-wip /
   ds4-moe-prefetch; first make it build, then drive to the packet's gate". Use NEW slice names (e.g. `ds4-dense-2`) -
   see memory `qc-redispatch-footguns`. For ds4-dense/ds4-moe: either cherry-pick the WIP files into the working
   tree first (`git checkout ds4-engine-wip -- <their files>`), or have them work on the wip branch in a worktree.
2. Then a glue slice `ds4-engine` (owns `tools/ds4/cmake/ds4_engine.cmake`): tokenizer + Ds4Dense + Ds4MoeTier +
   prefetch -> `ds4_generate` CLI.
3. **GPU steps only when Mal says the GPU is free** (he runs ComfyUI): engine logits gate vs llama.cpp goldens
   (p600/p3000), a real generation coherence check (greedy paragraphs vs llama.cpp), then decode tok/s for real;
   llama.cpp `--moe-prefetch 1.4` bench + exactness check (command in `docs/ds4-moe-prefetch.md` of that worktree).

## Rules (non-negotiable)
- Never develop in `~/AI/Strata` (production). Work in `~/AI/Strata-DS4` (+ the llama.cpp-ds4-prefetch worktree).
- **Every big run through `tools/ds4/memguard.sh`** (MemoryMax cap, swap off, 4 GiB watchdog, refuses while ComfyUI
  is up unless `MEMGUARD_ALLOW_COMFY=1` with Mal's OK). On 2026-10-06 a CPU-only full-model llama.cpp run
  (`-ngl 0 --load-mode none`, ~81 GB anon) + ComfyUI froze the box - **never do CPU-only full-model runs**.
- One full-model process at a time; flock `~/.quetza-data/conductor/ds4-gpu.lock`.
- Measure before concluding; correctness gates before speed claims; record numbers in FINDINGS.md and commit.
- Don't weaken gates. Ask before pushing to GitHub. Keep tools/results in the repo, not /tmp.
- `bench/ds4-2026-10-06/hidden-probe/h.bin` (1.4 GB hidden-state dump, untracked) is only needed to re-derive
  `pred.bin`/`routes.bin` (committed) - safe to delete.
