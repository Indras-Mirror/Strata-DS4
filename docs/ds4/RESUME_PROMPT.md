# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're porting the Strata inference engine to DeepSeek-V4-Flash so it decodes faster than llama.cpp on my RTX 4090 24 GB
+ Ryzen 7 5700X (8c, AVX2 only) + 90 GB DDR4. **Work solo - do NOT use /conductor or dispatch workers** (they burned
too many tokens). **The GPU is free now** (ComfyUI is off) - confirm with `ss -ltn | grep 8188` (nothing = off) and
`nvidia-smi` before each GPU run; if ComfyUI comes back, stop GPU work and tell me.
Read first: `docs/ds4/FINDINGS.md` s8 (status table), s9, s11, s12; then `docs/ds4/ENGINE_DENSE.md` and
`docs/ds4/ENGINE_MOE.md`. Branch `deepseek4`, tip `0edfea4` or later, nothing pushed, nothing in flight.
(Written 2026-10-06.)

## Key facts
- Model: `/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf` (80.76 GiB; 43 layers, 256 experts top-6, experts
  IQ2_XXS gate/up + Q2_K down = 6.75 MiB each; no MTP layer). Non-expert weights ~7.2 GiB (must sit in VRAM).
- Oracle: `~/AI/llama.cpp-master-rebase/build/bin` (CUDA 13.4). Goldens: `bench/ds4-2026-10-05/goldens/{p64,p600,p3000}`
  (last position only; `tools/ds4/compare_golden.py --tensors-diag` = the Phase-3 gate: p600/p3000 top-1 >= 0.99, KL < 0.01).
- Bar: llama.cpp config B (`-ngl 99 -cmoe --moe-expert-cache 40 --load-mode none -fa on -t 8`) = **16.34 tok/s**;
  go line 19.6; target 20.
- Machine physics (measured, FINDINGS s11/s12): CPU and PCIe share DDR4 - CPU alone 23.3 GB/s, PCIe alone 23.7, both at
  once 29.9 total. The win is predicted prefetch into VRAM under the dense work, not the CPU/PCIe split.
- VRAM budget for expert slots ~14.5 GiB = **2150 slots** (24 - 7.2 non-expert - ~0.8 scratch - ~1 ctx/KV - ~0.5).

## Where it stands
- Phase 3 DONE (reference forward `tools/ds4/ds4_ref.cpp` = llama.cpp logits; coherency check `4725894`).
- Speed verdict GO in the replay (`tools/ds4/moe_replay.cpp`): predicted prefetch = **20.57 tok/s** at 2150 slots,
  pf-b 1.43, pcie 0.25 dithered (FINDINGS s12; speed only - random activations, attention a timed stand-in).
- **The real engine is written and CPU-verified, but has never run on the GPU and never generated text:**
  - `6eda16d` **Ds4Dense** (`tools/ds4/ds4_dense.{hpp,cpp}`): one-token decode of attention/compressor/indexer/router/
    shared expert + the expert predictor (`predict(l)`). CPU mini gate PASS, every cosine 1.00000000 (`DS4_SWA=16`
    mini variant at 40 tokens; stock mini at 63). The 160-token stock-mini FAIL is a reference-side ggml-cpu
    flash-attn artifact (not query-count invariant above 64 queries) - see ENGINE_DENSE.md.
  - `71c265c` **Ds4MoeTier** (`tools/ds4/ds4_moe.{hpp,cpp}`): VRAM cache + predicted prefetch + CPU pool + PCIe split as a
    library (CPU-only and CUDA flavours). CPU gate PASS (rel 2.3e-7 vs dequant+F32 on real expert slices, re-verified).
  - `565c6cc` **ds4_generate** (`tools/ds4/ds4_generate.cpp`, `tools/ds4/cmake/ds4_engine.cmake`): per token, per layer
    predict -> tier.prefetch -> attn_router -> tier.run -> finish_layer; prefill = decode loop; ids in / ids out
    (stdout, one per line), timing + expert stats on stderr. Flags: `--backend cuda|cpu --experts gpu|cpu --slots
    --pcie --pf-b --profile --arena-gib --temp --stop --dump-logits`. **ds4_chat.py** = text in/out (tokenizer from
    the GGUF + the official DeepSeek-V4-Flash-0731 Jinja template; renders/round-trips verified; runs the engine
    through memguard). Both written by me, **NOT BUILT yet**.
  - llama.cpp track: worktree `~/AI/llama.cpp-ds4-prefetch`, branch `ds4-moe-prefetch` (`38330da56`): `--moe-prefetch N`
    built (llama-server/bench in its `build/`), CPU unit test 156 checks 0 failures (re-verified). Predicts from the
    PREVIOUS layer's router input; per-layer graph split via cb_eval (breaks CUDA graphs - measure the cost). Bench
    harness `bench/ds4-2026-10-06/prefetch/run-config.sh` + `probe.py` (`b04de46`).

## Blockers (fix first; 1-3 are CPU work)
1. **`build-ds4-cuda` has `GGML_CUDA=OFF`** (verified in its CMakeCache; it has STRATA_ENABLE_CUDA=ON,
   STRATA_GGML_DIR=third_party/llama.cpp, Ninja) - its ggml has no CUDA backend, so Ds4Dense can't use the GPU there.
   Configure a new tree, copying build-ds4-cuda's cache flags and adding GGML_CUDA:
   `cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DGGML_CUDA=ON
   -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp` then
   `nice -n 10 ninja -C build-ds4-gpu -j6 test_ds4_dense ds4_ref test_ds4_moe_gpu ds4_generate`. Watch for two ggml
   copies colliding (strata's kernels vs ggml-cuda) at link time.
2. `test_ds4_dense` hard-codes a CPU backend: add `--cuda` (ggml_backend_init_by_type GPU for Ds4Dense; keep ds4_ref on
   CPU as the oracle). ds4_generate already selects the backend with `--backend`.
3. **SwiGLU clamp - likely a real correctness bug.** DS4 clamps expert `up` to [-10, 10] and `gate` to <= 10 before
   SwiGLU (check the exact rule in ds4_ref.cpp:933-936 / llama.cpp's DEEPSEEK4 branch). Strata's native kernels (CPU
   `native_gu_rows` in include/strata/kernels/cpu/native_expert.hpp:49, CUDA grouped kernel src/kernels/cuda/
   iq_kernels.cu:1190,1412,2474) don't clamp. `tools/ds4/check_clamp_real.py` (real ffn_norm activations from
   `bench/ds4-2026-10-06/hidden-probe/h.bin` x the actually-routed experts) hit **10.088 at layer 11** before I
   stopped it (partial `bench/ds4-2026-10-06/clamp_real.txt`). It uses |x| for both: fix it to the real rule, run all
   43 layers (~10 min, CPU, MemoryMax=6G), and if the clamp binds, add it to BOTH kernels, then re-run Phase 2 parity
   (`ds4_expert_parity`) + `test_ds4_moe` (+ `test_ds4_moe_gpu --gpu`). Note `ds4_moe_replay` shares these kernels.

## Then, GPU (smallest first; record every number)
1. `test_ds4_dense --cuda` mini gate, same thresholds as the CPU gate (cos >= 0.99999, top-1 identical):
   `nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G ./build-ds4-gpu/test_ds4_dense
   bench/ds4-2026-10-06/dense/mini-ds4-swa16.gguf --tokens 40 --cuda --work bench/ds4-2026-10-06/dense/gpu-swa16
   --ref-dir bench/ds4-2026-10-06/dense/gpu-swa16/ref-allpos` (fixtures: mini-ds4.gguf, mini-ds4-swa16.gguf, mini-ds4-tame.gguf).
2. `nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 build-ds4-gpu/test_ds4_moe_gpu --gpu`
   (4 sources x 24 real slices, rel < 1e-3 each).
3. `bash tools/ds4/memguard.sh 64 62 -- build-ds4-gpu/test_ds4_moe_gpu --timing --routes
   bench/ds4-2026-10-06/hidden-probe/routes.bin --pred bench/ds4-2026-10-06/hidden-probe/pred.bin --profile
   bench/ds4-2026-10-05/route-probe/ds4routes.bin` - should land near s12's numbers.
4. **Real engine, real model** (memguard cap 80 / need 82; one full-model process at a time):
   a. logits gate: `--ids-file bench/ds4-2026-10-05/goldens/p600/tokens.i32 -n 1 --dump-logits <f>` -> compare top-1 /
      KL with the golden `result_output` (tools/ds4/compare_golden.py's logits metrics or a small numpy script);
      same for p3000. Gate: top-1 match, KL < 0.01.
   b. coherence: `python3 tools/ds4/ds4_chat.py --raw "def fibonacci(n):" -n 64 --bin build-ds4-gpu/ds4_generate
      --print-ids` and a chat prompt (`python3 tools/ds4/ds4_chat.py "Explain RoPE in two sentences." -n 128 ...`);
      compare greedy tokens with llama.cpp on the identical prompt ids (llama-cli/server, same greedy settings).
   c. speed: decode tok/s over >= 256 tokens on a code prompt at 2150 slots / pf-b 1.43 / pcie 0.25, plus
      `--pf-b 0` as the no-prefetch control. Compare with 16.34 (llama.cpp) and 20.57 (replay).
5. llama.cpp prefetch bench: for pf in 0 1.0 1.4 2.0: `bash tools/ds4/memguard.sh 80 82 -- bash
   bench/ds4-2026-10-06/prefetch/run-config.sh $pf pf$pf` (exactness = identical greedy tokens pf0 vs pf1.4; speed =
   decode mean vs the in-session pf0).
6. Write FINDINGS **s13** (every gate + number, what failed), update s8, commit, update memory note `strata-ds4-port`
   and this prompt.

## Next workload after DeepSeek-V4: MiMo-V2.6-Flash in Strata (Mal confirmed 2026-10-06)
- Plan/estimates: `docs/mimo/MIMO_V26_FEASIBILITY.md`. Same 4096x2048 expert shape as DS4 (Ds4MoeTier reusable, top-8),
  simpler attention (hybrid SWA/global), 256 experts x 47 layers.
- **Files** (downloading via `"/media/mal/SSD NVME/Models/MiMo-V2.6-Flash-RL-GSQ-RCO/fetch.sh"`, check its `fetch.log`
  for `ALL VERIFIED OK`): GSQ-RCO 3-bit GGUF (115.7 GB) + Heretic rank-1 uncensor LoRA (70 MB).
- Order: (1) llama.cpp baseline: 3-bit `-cmoe` +/- `--lora` under memguard (it does NOT fit RAM+VRAM: ~8 GiB pages
  from NVMe; mmap, not --load-mode none); (2) route-probe MiMo's routing skew (route_probe.cpp generalised to K=8);
  (3) Strata port: mimo2 dense half (like Ds4Dense), Q3_K + MXFP4 expert kernels (CPU AVX2 + CUDA), a "VRAM-only
  resident" arena mode, LoRA support in the expert tier if the Heretic LoRA touches expert tensors; (4) consider a
  ~2.3 bpw build (fits fully; est. 16-20 tok/s) if the 3-bit spill hurts.

## Rules (non-negotiable)
- **Every full-model load through `tools/ds4/memguard.sh <cap_gib> <need_gib> -- <cmd>`** (cgroup RAM cap + swap off +
  4 GiB MemAvailable watchdog + the shared ds4 lock; refuses while ComfyUI is up unless `MEMGUARD_ALLOW_COMFY=1` with
  my OK). NEVER a CPU-only full-model run (`-ngl 0` etc. froze the box 2026-10-06). One full-model process at a time.
  Small tests: `nice -n 10 systemd-run --user --scope -q -p MemoryMax=<n>G -p MemorySwapMax=0 <cmd>`.
- Never develop in `~/AI/Strata` (production). Don't push (ask first). Don't weaken gates. Measure before concluding;
  correctness gates before speed claims; say what was NOT tested.
- Never `pgrep -f`/`pkill -f` a pattern that's in your own command line (it self-matches). Kill only PIDs you started.
- Keep tools/results in the repo, not /tmp. Commit only your own paths.
- Untracked, deletable later: `bench/ds4-2026-10-06/hidden-probe/h.bin` (1.4 GB - keep until blocker 3 is done),
  `bench/ds4-2026-10-05/{goldens-probe*,ref-probe*,ref}`, `bench/ds4-2026-10-06/dense/diag*` test outputs.
  **KEEP** `bench/ds4-2026-10-06/dense/mini-ds4{,-swa16,-tame}.gguf` (the GPU mini gate's fixtures; untracked).
