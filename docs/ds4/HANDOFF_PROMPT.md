# Handoff prompt — finish the Strata DeepSeek-V4-Flash port

Copy the block below the line into a fresh session started in `~/AI/Strata-DS4` (best run under `/conductor`;
phase work is token-heavy and GPU-serial). Written 2026-10-06, branch tip `e9933cc`.

---

## Mission
Finish porting the **Strata** inference engine (built for Qwen3.8-Flash-Next) to **DeepSeek-V4-Flash**, so
DeepSeek decodes faster than llama.cpp on this box: RTX 4090 24 GB, Ryzen 7 5700X (8c, **AVX2 only**), 90 GB DDR4.
Repo: `~/AI/Strata-DS4`, branch `deepseek4` (fork `github.com/Indras-Mirror/Strata`).
**NEVER develop in `~/AI/Strata`** — that is the production Qwen engine serving Mal's daily wrappers.

Read first (in this order), all under `~/AI/Strata-DS4`:
1. `docs/ds4/PORT_PLAN.md` — the plan, the bar, phases with gates, kill criterion, machine notes.
2. `docs/ds4/FINDINGS.md` — **s8** = status table, **s9** = the Phase 3 gate, **s10** = the Phase 4a MoE verdict;
   s2 = routing skew, s3 = the llama.cpp baseline, s6 = Phase 2.
3. `docs/ds4/DSV4_ARCH_SPEC.md` — DeepSeek-V4's exact computation, cited to llama.cpp source.
4. `docs/ds4/STRATA_ARCH_MAP.md` — what in Strata is Qwen-specific vs reusable, cited to Strata source.
5. `docs/ds4/GGUF_INVENTORY.txt` — every tensor in the model.
6. Memory notes `strata-ds4-port`, `strata-0139-lazy-vision`.

## Current state (2026-10-06, tip `e9933cc`, **nothing pushed to GitHub**)

| Phase | State |
| --- | --- |
| 0 oracle / profile | done except the antirez `ds4` engine run (low priority, `~/AI/ds4-ref/ds4`) |
| 1 loader / geometry / tokenizer / pack | done, gates verified |
| 2 Q2_K experts | **done both paths** — CPU 1.8e-7 / GPU 3.0e-5 ≪ 1e-3; 0.71 / 0.017 ms per expert |
| 3 reference forward (`tools/ds4/ds4_ref.cpp`) | **structural bug fixed**; strict per-tensor gate still RED on a numeric drift → **the first thing to finish** |
| 4a MoE engine replay | **done** — verdict GO (best 17.28 tok/s implied; ~22.3 with the harness cost removed) |
| 4b captured token graph | **not started** → **the second thing to finish** |
| 5 KV formats / long context / prefill | not started |
| 6 speculation | skipped (no MTP layer in the GGUF; DSpark was slower) |
| 7 server / release | not started |

Model: `/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf` (80.76 GiB, arch `deepseek4`, 43 layers,
256 experts top-6, experts IQ2_XXS gate/up + Q2_K down, **no MTP layer**).
Oracle: `~/AI/llama.cpp-master-rebase/build/bin` (CUDA 13.4, has `deepseek4` + `--moe-expert-cache`). Use F16 KV for
parity runs. `llama.cpp-new` / `-dspark` are CUDA-12 builds and do not run as-is.
Bar to beat: llama.cpp `-cmoe --moe-expert-cache 40 --load-mode none` = **16.34 tok/s decode, 154 tok/s prefill**
(clean run; llama.cpp default placement = 10.85). Target **>= 20 tok/s**, go line **19.6**.

## Task A — close the Phase 3 gate (do this first)

`bash tools/ds4/run_ref_gate.sh p64` → `compare_golden --cosine 0.9999 --top1 0.99 --kl 0.01`.
Gate: p64 hc-stream tensors cosine > 0.9999; p600/p3000 teacher-forced top-1 > 0.99 and mean KL < 0.01.

State: the compressed-attention **mask was filled transposed** (`7fab6eb`) — `make_comp` wrote `m16[b*nt + t]` but
ggml indexes the mask as `mask[key + query*ne0]`, so element `(block b, query t)` lives at `t*n_blocks + b`; every
query attended only the last few compressed blocks. Fixed (both the f16 and f32 masks). After the fix (**p600 and
p3000 logits PASS**: top-1 1.0, KL 0.0086/0.0079), p64 = 35/477; `attn_csa_lid-2` 0.999979, `attn_hca-3` 0.999975.

Remaining red = **a slow compounding numeric drift**, not structural: `attn_raw-0` matches to 2.2e-8, but
`attn_out-0` (de-rope + two grouped Q8_0 output-LoRA matmuls) is 0.9999892 and `ffn_moe_out-0` 0.999657, growing
~1e-3/layer until MoE routing flips discrete top-6 picks. **INFERRED cause:** ggml-cpu quantizes the *activations*
to Q8_0 for Q8_0/Q2_K weight matmuls while the CUDA oracle does not.
**Confirming test to run before anything else:** re-run the oracle with those tensors promoted to F16 (llama.cpp
`-ot` / `--override-tensor`, e.g. the `attn_output`/`ffn_*` Q8_0 tensors) **or** run the oracle on the CPU backend,
and see whether `attn_out-*` / `ffn_moe_out-*` collapse toward the golden.
- If it collapses → the drift is CPU-vs-CUDA activation quantization. The per-tensor 0.9999 gate then needs a
  CPU-vs-CUDA-justified tolerance — **document the evidence in FINDINGS.md**, do not silently relax it.
- If it does not collapse → the residual is still in `ds4_ref.cpp`; probe taps are in place
  (`DS4_GOLDEN_PROBE=1` in golden_dump captures `q`, `kv`, `csa_state_kv`, `csa_state_score_ape`, `csa_comp_k`,
  `k_all`; `DS4_PROBE=1` in ds4_ref dumps the same names) — diff tensor by tensor.
Do **not** weaken thresholds before the confirming test. Step 1 is already answered: the goldens are **NOT**
post-Hadamard under `-ctk f16` (only `kv_lid` rotates), so this is arithmetic, not a comparison artifact.

## Task B — Phase 4b, the captured token graph (the speed verdict)

Phase 4a measured the real expert engine (ExpertCache 6.75 MiB slots + profile seed, CPU ExpertPool, the PCIe share
of misses computed on the GPU in parallel, 49 GiB pinned arena) replaying real routes via
`tools/ds4/moe_replay.cpp` + `moe_gap.cu` (FINDINGS s10). Best arm **17.28 tok/s** implied (slots 2300 / 15.1 GiB).
The kill criterion fires as written (17.28 < 19.6) — **but ~13 ms/token of the wall is an un-captured per-layer
sync + ~10 launches + a 128 KB D2H, overlapped with *neither* engine; minus that = 44.9 ms = 22.3 tok/s, above the
go line.** So the missing piece is the captured token graph, **not** the architecture.
**Do NOT invoke the s7 llama.cpp fallback.**
Build the full decode loop as a CUDA graph per token (Strata's doorbell overlap doing what 4a's `--gap-ms` only
approximated), and gate: **decode >= 19.6 tok/s vs 16.34**, plus re-check parity (top-1 > 99% vs golden).
Residuals to pick up: the adaptive tier (learned swap) is not implemented (static profile seed, no eviction), and
the shared expert is not computed in the replay (folded into the 17.6 ms constant — the prompt graph must add it).

## Then — phases 5-7 (see `PORT_PLAN.md`)
5. KV formats + long context + prefill (raw int8 window ring, CSA/HCA/indexer caches, chunked prefill with
   tensor-core GEMMs). Gate: needle at 64K/128K, prefill >= 175 tok/s at 7.8K.
6. Speculation — skipped by default (no MTP in the GGUF; DSpark was slower). Optional last.
7. Server + release (`serve/`, the DeepSeek chat template incl. DSML tool calls, config, README with measured
   numbers). Optionally the antirez `ds4` run for comparison.

## Key measured facts (use these, they override the plan's earlier estimates)
- CPU expert miss path is **10-16 GB/s, not the plan's ~25-30** — the 5700X has no AVX-512, so the native
  IQ2_XXS/Q2_K path runs ggml-cpu AVX2 dots; Qwen's 36 GB/s is the AVX-512 Q2_0 kernel and does not transfer.
- PCIe measured **15.8 GB/s**. Held-out hit rate **50.9%** at a 12 GB cache (whole-bin profile leaks to 65.5%).
- A real engine bug was found and fixed: `ExpertPool::run_split_multi_native` sized its row split with the
  compile-time Qwen H/FF instead of the model's `n_embd`/`n_ff` (only ~1/3 of every expert — `173e535`).
- `tools/ds4/route_probe.cpp` + `route_skew.py`; routes at `bench/ds4-2026-10-05/route-probe/ds4routes.bin`.
- Bench script with a RAM watchdog: `bench/ds4-2026-10-05/llamacpp-ab/bench.sh`.

## Rules (non-negotiable)
- **Never** develop in `~/AI/Strata`. Work only in `~/AI/Strata-DS4`.
- Correctness before speed: each phase ends at its gate; record the measured numbers in `FINDINGS.md` and commit
  before moving on. **Measure before concluding; reproduce before diagnosing.**
- Load DeepSeek with `--load-mode none` (mmap on the exFAT NVMe + 90 GB RAM swaps and stalls the load with the GPU idle).
- One full-model process at a time. Wrap **every** GPU run / full-model load / large pin in
  `flock ~/.quetza-data/conductor/ds4-gpu.lock <cmd>`. `free -g` before pinning; abort if MemAvailable < 3 GB
  (the box has a known bad byte lane at ~56.6 GB).
- Keep tools and results in the repo, not `/tmp` (wiped on reboot).
- Never `tmux kill-server`, bare `pkill`, or `killall` — kill only PIDs you started. Never `pkill -f` a pattern
  that appears in your own command line. Do not weaken gates/tests. Do not merge.
- **Ask before pushing to GitHub.**

## Runbook order
1. Task A: the Phase 3 confirming test → fix → `run_ref_gate.sh p64` green → re-run p600/p3000 → commit + FINDINGS.
2. Task B: Phase 4b captured token graph → the >= 19.6 tok/s gate + parity → commit + FINDINGS s10.
3. Phase 5 → 6 (optional) → 7. Optional antirez `ds4` run.
4. Keep `docs/ds4/RESUME_PROMPT.md` and the `strata-ds4-port` memory note current as you go.
