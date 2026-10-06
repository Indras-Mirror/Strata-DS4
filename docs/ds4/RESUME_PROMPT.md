# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're porting the Strata inference engine to DeepSeek-V4-Flash so it decodes faster than llama.cpp on my RTX 4090 +
Ryzen 7 5700X + 90 GB DDR4. **Work solo - do NOT use /conductor or dispatch workers** (they burned too many tokens);
I'll tell you when the GPU is free (I run ComfyUI). Status table: `docs/ds4/FINDINGS.md` s8; read s9, s11, s12 first,
then `docs/ds4/ENGINE_DENSE.md` and `docs/ds4/ENGINE_MOE.md`. (Written 2026-10-06 17:20; nothing in flight.)

## Where it stands
- Phase 3 DONE (reference forward `tools/ds4/ds4_ref.cpp` matches llama.cpp logits; coherency check `4725894`).
- Speed verdict GO in the replay: predicted expert prefetch = **20.57 tok/s** at 2150 slots vs llama.cpp 16.34
  (FINDINGS s12; speed only, random activations, attention a timed stand-in).
- **The real engine is written and CPU-verified, never run on the GPU, never generated text:**
  - `6eda16d` **Ds4Dense** (`tools/ds4/ds4_dense.{hpp,cpp}`): one-token decode of attention/compressor/indexer/router/
    shared expert + the expert predictor. CPU mini gate PASS, every cosine 1.00000000 (`DS4_SWA=16` variant, 40
    tokens, and the stock mini at 63 tokens). The 160-token stock-mini FAIL is a reference-side ggml-cpu flash-attn
    artifact (not query-count invariant above 64 queries), documented in the report/ENGINE_DENSE.md.
  - `71c265c` **Ds4MoeTier** (`tools/ds4/ds4_moe.{hpp,cpp}`): VRAM cache + prefetch + CPU pool + PCIe split as a
    library. CPU gate PASS (rel 2.3e-7 vs dequant+F32 on real expert slices; I re-ran it).
  - `565c6cc` **ds4_generate** (`tools/ds4/ds4_generate.cpp`, `cmake/ds4_engine.cmake`) + **ds4_chat.py** (tokenizer +
    the official DeepSeek-V4-Flash-0731 Jinja template, verified to render/round-trip). Written by me, NOT BUILT yet.
  - llama.cpp track: worktree `~/AI/llama.cpp-ds4-prefetch`, branch `ds4-moe-prefetch` (`38330da56`):
    `--moe-prefetch N` built (llama-server/bench), CPU unit test 156 checks 0 failures (I re-ran it). Predicts from the
    PREVIOUS layer's router input (graph split per layer via cb_eval). Bench harness committed `b04de46`
    (`bench/ds4-2026-10-06/prefetch/run-config.sh` + `probe.py`).

## Blockers found (fix first)
1. **`build-ds4-cuda` has `GGML_CUDA=OFF`** - its ggml has no CUDA backend, so Ds4Dense can't run on the GPU there.
   Configure a separate tree: `cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
   -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=third_party/llama.cpp` (check the flags against
   build-ds4-cuda's CMakeCache), then build `test_ds4_dense ds4_ref test_ds4_moe_gpu ds4_generate`.
2. `test_ds4_dense` hard-codes a CPU backend: add `--cuda` (ggml_backend_init_by_type GPU), keep ds4_ref on CPU.
3. **SwiGLU clamp - likely a real correctness bug.** DS4 clamps expert `up` to +-10 and `gate` to <= 10 before SwiGLU;
   Strata's native kernels (CPU `native_gu_rows`, CUDA grouped kernel `src/kernels/cuda/iq_kernels.cu`) don't clamp.
   `tools/ds4/check_clamp_real.py` (real activations from `bench/ds4-2026-10-06/hidden-probe/h.bin` + the routed
   experts) reached **10.088 at layer 11** before I stopped it (partial log `bench/ds4-2026-10-06/clamp_real.txt`).
   It measures |x| for both, so fix it to the real rule (gate only from above), run all 43 layers, and if it binds,
   add the clamp to both kernels + re-run Phase 2 parity (`ds4_expert_parity`) and `test_ds4_moe`.

## Next, in order (GPU steps only when I say it's free)
1. Blockers 1-3 (CPU work, can start now).
2. GPU, smallest first: `test_ds4_dense --cuda` mini gate (same thresholds) -> `test_ds4_moe_gpu --gpu` ->
   `bash tools/ds4/memguard.sh 64 62 -- build-ds4-gpu/test_ds4_moe_gpu --timing --routes
   bench/ds4-2026-10-06/hidden-probe/routes.bin --pred bench/ds4-2026-10-06/hidden-probe/pred.bin --profile
   bench/ds4-2026-10-05/route-probe/ds4routes.bin` (expect ~s12's numbers).
3. **Real engine:** `ds4_generate` on the real model (through memguard, cap 80 / need 82): first the logits gate -
   feed p600's tokens (`bench/ds4-2026-10-05/goldens/p600/tokens.i32`) with `--ids-file ... -n 1 --dump-logits`, compare
   top-1/KL with the golden `result_output`; then coherence: `python3 tools/ds4/ds4_chat.py --raw "def fibonacci(n):"
   -n 64 --bin build-ds4-gpu/ds4_generate` and a chat prompt; greedy text vs llama.cpp on identical ids; then tok/s.
4. llama.cpp prefetch bench: for pf in 0 1.0 1.4 2.0: `bash tools/ds4/memguard.sh 80 82 -- bash
   bench/ds4-2026-10-06/prefetch/run-config.sh $pf pf$pf` (exactness = identical greedy tokens pf0 vs pf1.4).
5. Record every number in FINDINGS (new s13), commit, update memory note `strata-ds4-port`.

## Next workload after DeepSeek-V4: MiMo-V2.6-Flash in Strata (Mal confirmed 2026-10-06)
- Plan/estimates: `docs/mimo/MIMO_V26_FEASIBILITY.md`. Same 4096x2048 expert shape as DS4 (Ds4MoeTier reusable, top-8),
  simpler attention (hybrid SWA/global), 256 experts x 47 layers.
- **Files** (downloading overnight via `"/media/mal/SSD NVME/Models/MiMo-V2.6-Flash-RL-GSQ-RCO/fetch.sh"`, check its
  `fetch.log` for `ALL VERIFIED OK`): GSQ-RCO 3-bit GGUF (115.7 GB) + Heretic rank-1 uncensor LoRA (70 MB).
- Order: (1) llama.cpp baseline: 3-bit `-cmoe` +/- `--lora` under memguard (it does NOT fit RAM+VRAM: ~8 GiB pages
  from NVMe; mmap, not --load-mode none); (2) route-probe MiMo's routing skew (route_probe.cpp generalised to K=8);
  (3) Strata port: mimo2 dense half (like Ds4Dense), Q3_K + MXFP4 expert kernels (CPU AVX2 + CUDA), a "VRAM-only
  resident" arena mode, LoRA support in the expert tier if the Heretic LoRA touches expert tensors; (4) consider a
  ~2.3 bpw build (fits fully; est. 16-20 tok/s) if the 3-bit spill hurts.

## Rules (non-negotiable)
- **Every full-model load through `tools/ds4/memguard.sh`** (cap + swap off + 4 GiB watchdog + shared lock; refuses
  while ComfyUI is up). NEVER a CPU-only full-model run (froze the box 2026-10-06). One full-model process at a time.
- Never develop in `~/AI/Strata` (production). Don't push. Don't weaken gates. Measure before concluding.
- Never `pgrep -f`/`pkill -f` a pattern that's in your own command line (it self-matches).
- Untracked, deletable: `bench/ds4-2026-10-06/hidden-probe/h.bin` (1.4 GB; needed by check_clamp_real.py - keep until
  blocker 3 is done), `bench/ds4-2026-10-05/{goldens-probe*,ref-probe*,ref}`.
