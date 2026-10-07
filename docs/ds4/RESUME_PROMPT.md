# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're building Strata-DS4: the Strata inference engine for DeepSeek-V4-Flash-0731, decoding faster than llama.cpp on my
RTX 4090 24 GB + Ryzen 7 5700X (8c, AVX2 only) + 90 GB DDR4. Public repo: https://github.com/Indras-Mirror/Strata-DS4
(local branch `deepseek4` tracks remote `ds4/main`; remote `origin` is the old Strata fork - don't push there).
**Work solo** - no /conductor, no workers, no subagents. Speed first, quality measured alongside.
Another session works on MiMo in `~/AI/Strata-MiMo` (branch `mimo`) - don't touch it; you share the GPU lock.
Written 2026-10-07 at commit `83c5324` (end of a CPU-only session that did the whole CPU queue - FINDINGS s14-18).

## GPU first: ask me whether it is free
The CPU work queue is done. What is left needs the GPU (MiMo session may still have it): **ask me before any GPU
run**, and if it is busy, only CPU work (see "If the GPU is still busy" at the end). Compiling `build-ds4-gpu` is fine.
Even CPU runs of the GPU-tree binary: `env CUDA_VISIBLE_DEVICES=` so it cannot touch the GPU.

Read first, in this order: this file; `docs/ds4/FINDINGS.md` s13-18 (the latest measurements); `docs/ds4/ENGINE_DENSE.md`
(state layout, multi-token passes, MTP head); `README.md`; then `git log --oneline -15`.

## Where it stands (all measured, p600 prompt + 128-token greedy decode, real model)
- Engine: `build-ds4-gpu/ds4_generate` = `Ds4Dense` (attention + router + shared expert + hyper-connections on the
  CUDA ggml backend, CUDA graphs) + `Ds4MoeTier` (routed experts: VRAM slot cache seeded from a routing profile,
  predicted prefetch, CPU pool, PCIe share, 70 GiB pinned arena). Coherent text, first-token top-1 = llama.cpp.
- Honest numbers (README corrected): `--dense-requant q6_k --slots auto` **16.8-17.3 tok/s**, ppl 10.47-10.64;
  `+ --route-bias 0.05` **19.1 tok/s**, ppl 10.77; 2350 hand-set slots 19.35 (VRAM edge). llama.cpp best: 16.34.
  Per token: attention+router ~22 ms, experts ~25-31 ms (bandwidth-bound: CPU pool + PCIe DMA share ~30 GB/s of
  DDR4), finish ~1.4, head ~0.6. **VRAM slots are the scarcest resource**: ~120 fewer slots = -6 points hit rate.
- ppl and even generated tokens vary run to run (~0.2 ppl): the CPU vs GPU expert split is timing-dependent and the
  two paths round differently. Judge configs on several runs.
- Multi-token passes (n <= 4) in `Ds4Dense`: done, CPU bit-identical to one-token decoding, CUDA within the same
  distance of ds4_ref as one-token decoding.
- MTP head: loads into `Ds4Dense` as layer 43 from
  `/media/mal/NVME1TB/Models/DS4-MTP/DeepSeek-V4-Flash-MTP-bf16.gguf`; CPU gate exact vs `ds4_ref --mtp`.
  **Fixed 2026-10-07 (`e54b5d9`): it now uses the MTP file's OWN hc_head** (`mtp.output_hc_*`); the 59.8% real
  acceptance (76/127) was measured with the trunk's. `DS4_MTP_TRUNK_HC=1` = the old head, for A/B. philpax measured
  0.74-0.93 for this head on 0731 (README-philpax.md). **New acceptance not measured yet.**
- MTP draft cost 16.2 ms measured; `ds4_mtp_bench` (CPU, real file) shows the experts take **3.3 ms warm** and
  10-18 ms when the file's pages are not in RAM -> the 16 ms is page-cache eviction under memguard's cap (80 GB,
  counts page cache, next to the 70 GiB arena), not the MXFP4 kernel. `--mtp-resident` copies the 3.2 GB of MTP
  experts into RAM (CPU gate: identical drafts). The MTP line now prints `routed experts X, rest Y` ms and the share
  of MTP expert bytes in RAM at the end.
- `--verify`: **CPU-tested** (`2868913`): 29/29 runs bit-identical to plain greedy in tokens, the logits row behind
  every token and the MTP draft logits (`bench/ds4-2026-10-07/verify-cpu/run.sh`; test hooks `DS4_VERIFY_ORACLE`,
  `DS4_VERIFY_CORRUPT`, `DS4_DUMP_TOKLOGITS`, `DS4_DUMP_DRAFTLOGITS`), 3 mutations caught. Never run on CUDA.
- 0731-aligned MTP sidecar applied: `/media/mal/NVME1TB/Models/DS4-MTP/DeepSeek-V4-Flash-MTP-bf16-aligned0731.gguf`
  (`tools/ds4/mtp_sidecar.py`; eh_proj = [e_proj | h_proj], checked; 0 bytes differ outside the 21 tensors). Its
  authors report chained (depth 2-3) gains; depth 1 (ours) not reported. Untested end to end.
- +0.7 ms attention+router since multi-token: CPU A/B shows only +~20 VIEW nodes per layer, no new compute op.
- Estimate, NOT measured: verify at ~75% acceptance + ~4 ms drafts ~26-27 tok/s.

## CPU work queue: DONE 2026-10-07 (FINDINGS s14-18)
1 hc_head fix, 2 `--verify` CPU smoke test, 3 draft-cost diagnosis + `--mtp-resident`, 4 sidecar GGUF, 5 graph A/B.
Not done: the long-context `vis` upload item (comp_max*kNtMax floats per compressed layer, ~0.35 ms/token at 32K;
build visibility on the device from i_pos) and the README update (do it after the GPU numbers).

## GPU queue (only when I say the GPU is free; each through memguard, one at a time)
Base command (p600, the comparison config):
`bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf --ids-file bench/ds4-2026-10-05/goldens/p600/tokens.i32 -n 128 --slots auto --arena-gib 70 --pcie 0.55 --ppl --dense-requant q6_k --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin`
(the prompt is the BINARY `goldens/p600/tokens.i32`; `bench/ds4-2026-10-06/real/p600.ids` is an old run's text
output, not a prompt). Logs to `bench/ds4-2026-10-0X/`. M=/media/mal/NVME1TB/Models/DS4-MTP
1. `--mtp $M/DeepSeek-V4-Flash-MTP-bf16.gguf`: acceptance with the MTP's own hc_head (was 59.8%); read the new
   MTP line (expert ms vs rest, expert pages in RAM %). Same with `DS4_MTP_TRUNK_HC=1` once to confirm the A/B.
2. Same with `--mtp $M/DeepSeek-V4-Flash-MTP-bf16-aligned0731.gguf`: depth-1 acceptance, original vs aligned.
3. `--mtp-resident`: does it fit? Watch memguard (80 GB cap) - may need `--arena-gib 67`; compare the file-tier
   reads/token (each ~5 ms) against the saved draft ms. If RAM is too tight: keep only hot MTP experts resident
   (log the MTP routing skew first).
4. `--mtp <best> --verify [--mtp-resident]`: tok/s, tokens/pass, token equality vs plain greedy of the SAME binary
   (expect rare divergence on CUDA: n=2 kernels round differently; several runs).
5. `test_ds4_dense --cuda` with `--mtp` + `--multi` on the mini fixtures (the MTP block on CUDA).
6. nsys for the +0.7 ms (CPU A/B: only +~20 VIEW nodes/layer - look at CUDA-graph update checks; or it is noise).
`DS4_VRAM_TRACE=1` logs free VRAM per prefill step; the run prints "VRAM free at the end" (0.54 GiB with --mtp).

## Gates (CPU; must stay PASS)
Build: `nice -n 10 ninja -C build-ds4 test_ds4_dense ds4_ref` (CPU tree) and `nice -n 10 ninja -C build-ds4-gpu
ds4_generate test_ds4_dense` (compiles the GPU tree; running it on the CPU is fine with `--backend cpu`).
Small runs: `nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 <cmd>`.
Fixtures (untracked, keep): `bench/ds4-2026-10-06/dense/mini-ds4{,-swa16,-tame}.gguf`, `mini-ds4-mtp.gguf`
(regenerate the last with `python3 tools/ds4/make_mini_mtp.py <out>`; deterministic, seed 11).
```
D=bench/ds4-2026-10-06/dense
# trunk + multi-token (each reuses its own ref dir; the test runs ds4_ref itself when the dir is missing)
./build-ds4/test_ds4_dense $D/mini-ds4-swa16.gguf --tokens 40 --work $D/cpu-x --ref-dir $D/cpu-x/ref-allpos-mini-ds4-swa16 --multi 2,3,1,4
./build-ds4/test_ds4_dense $D/mini-ds4-tame.gguf  --tokens 63 --work $D/cpu-x --ref-dir $D/cpu-x/ref-allpos-mini-ds4-tame  --multi 2,3,1,4
#   expect: worst cos 1.00000000 (swa16) / 0.99999996 (tame), multi bit-identical 40/40 and 63/63, PASS
# MTP block, exact form: feed the oracle the decoder's own trunk state (see ENGINE_DENSE.md "MTP head" for why)
W=/tmp/mtpw; rm -rf $W; mkdir -p $W
DS4_DBG_MTP_H=$W/h.f32 ./build-ds4/test_ds4_dense $D/mini-ds4-tame.gguf --tokens 63 --work $W --mtp $D/mini-ds4-mtp.gguf
DS4_REF_MTP_H=$W/h.f32 ./build-ds4/ds4_ref -m $D/mini-ds4-tame.gguf --tokens $W/tokens.i32 --out $W/refh --all-pos -t 8 --mtp $D/mini-ds4-mtp.gguf
./build-ds4/test_ds4_dense $D/mini-ds4-tame.gguf --tokens 63 --work $W --ref-dir $W/refh --mtp $D/mini-ds4-mtp.gguf --multi 2,3,1,4
#   expect: MTP logits worst cos 0.99999996, top-1 62/62, MTP in multi passes bit-identical 63/63, PASS
#   (mutation check: DS4_MTP_TRUNK_HC=1 on the last command must FAIL - cos ~0.986)
#   (rm h.f32 before re-running: the dump APPENDS; the first command's own MTP check fails ~0.99996 by design)
# generator end to end on the CPU (output must not change with --mtp; --verify must match plain greedy on the CPU)
env CUDA_VISIBLE_DEVICES= ./build-ds4-gpu/ds4_generate -m $D/mini-ds4-tame.gguf --ids 11,48,21,58,31,4,41,14 -n 24 --backend cpu --experts cpu --arena-gib 1 [--mtp $D/mini-ds4-mtp.gguf [--verify]]
# the verify gate (29 runs: tokens + per-token logits + MTP draft logits vs plain greedy)
bench/ds4-2026-10-07/verify-cpu/run.sh      # expect "FAILURES: 0"
# MTP experts microbenchmark on the real file (CPU only)
./build-ds4/ds4_mtp_bench /media/mal/NVME1TB/Models/DS4-MTP/DeepSeek-V4-Flash-MTP-bf16.gguf --threads 6 [--cold|--resident]
```
The mini MTP fixture now carries its own hc_head (regenerate: `PYTHONPATH=~/AI/llama.cpp-master-rebase/gguf-py python3
tools/ds4/make_mini_mtp.py $D/mini-ds4-mtp.gguf`); the old one is kept as `mini-ds4-mtp.pre-hchead.gguf` and is
now refused at attach.
Mutation-test new gates (break the code on purpose, the gate must fail, restore). Don't trust a perfect result untested.

## Lessons (don't re-learn these)
- Without CUDA graphs the dense half is launch-bound; fused `ggml_dsv4_hc_*` ops replaced the Sinkhorn op loop.
- Each `ggml_backend_tensor_set/get` on CUDA is a stream sync: ONE input-span upload per pass (`in_host`/`i_span`),
  ONE router readback per layer per pass (`o_span`, per-token blocks [fn | ids | wts]). Keep it that way.
- Graph variants (capacity doubling x pass size) are all reserved at load (`reserve_graphs(n_max)`); CUDA-graph
  instances still take VRAM when first captured, AFTER `--slots auto` sized the cache: the auto margin grows +0.25 GiB
  with `--mtp`, +0.75 with `--verify`. Two OOMs in `cudaGraphInstantiate` taught this.
- A BF16 matrix on CUDA may route through cuBLAS: load-time requant every BF16 matrix (now Q8_0 by default, the
  `--dense-requant` type for the big ones).
- ggml-cuda picks different kernels for 1 vs n columns: multi-token results differ from one-token by rounding on
  CUDA (not on the CPU). The CPU is where bit-exactness is gated.
- The mini trunk's expert blobs are all zeros: trunk gates never exercise expert numerics. The MTP fixture has real
  MXFP4 experts, which amplify 1-ulp trunk differences (rel 1.6e-7) to cos ~0.99996 - hence the DS4_REF_MTP_H gate.
- ggml-cpu flash-attn: split-KV path for 1 query with >=512 keys, tiled for >= Q-tile queries - different rounding;
  the mini fixtures stay below both.
- `get_rows` does not broadcast a 2-D index (flatten); `set_rows` broadcasts its index over dim 2 (per-query top-k);
  duplicate `set_rows` indices have no defined order (two tokens of a pass in one compressed block: the earlier one
  writes a spare row `comp_max`).
- Strata's gguf_reader needed MXFP4 (type 39: 32 elements / 17 bytes) - added.
- After a CUDA abort, ggml attaches gdb to the 75 GB process and can eat the last RAM: memguard exports
  `GGML_NO_BACKTRACE=1`.
- memguard's 80 GB cap counts page cache; decode is unaffected (the arena is pinned), load may be slower than an
  uncapped llama.cpp. Load-speed investigation: Mal said not now.
- Experts are DDR4-bandwidth-bound (CPU pool + PCIe DMA share ~30 GB/s): the levers are hit rate (VRAM, routing)
  and sharing reads across tokens (verify).
- Sweeps: dense requant q6_k best (q5_k kernel slower, q4_k +6% ppl); route-bias 0.05 +1% ppl / +9% speed,
  0.2 +9.6% / +29%. SwiGLU clamp binds on real data (L39/L40) - kernels take `swiglu_limit`.
- DSpark (3 MoE layers) as a llama.cpp draft was slower than no draft on this box (6-8 vs 11.2 tok/s, 2026-10-05):
  each draft step reads more experts. Retry only on our own verify loop, after MTP.
- History rewriting: RANGED `git filter-branch base..branch`; `git filter-repo` strips upstream GPG signatures.
- The built-in WebSearch tool failed before; HuggingFace API via curl and `gh` work.
- Greedy-token equality is a WEAK gate on the mini fixtures (a deliberately skipped position kept the tokens): gate
  on logits rows (DS4_DUMP_TOKLOGITS) - on the CPU they must be bit-identical.
- A test harness that compares empty outputs "passes": require non-empty references.
- MAP_PRIVATE mmap of the MTP file is evicted under the memguard cap; mincore is the check (`resident_frac`).

## If the GPU is still busy (CPU-only options)
- The `vis` upload item above (device-side visibility from i_pos), gated bit-exact on the CPU fixtures.
- A CPU routing-skew probe for the MTP is NOT possible without the real trunk (its router input is the trunk state).

## Rules (non-negotiable)
- **Every full-model load through `tools/ds4/memguard.sh 80 78 -- <cmd>`** (cgroup RAM cap, swap off, watchdog,
  shared GPU lock `~/.quetza-data/conductor/ds4-gpu.lock`, refuses while ComfyUI is up). NEVER a CPU-only full-model
  run. One full-model process at a time. Check `ss -ltn | grep 8188` + `nvidia-smi` first; if ComfyUI is up, ask me.
- If `/media/mal/SSD NVME` is missing after a reboot: `udisksctl mount -b /dev/nvme0n1p3`.
- Push to `ds4` only when I say so (local is 14 commits ahead); nothing private or third-party in the tree (two corpus
  files were stripped from history; private copies in `~/AI/Strata-DS4-private/`). Shared kernels
  (`src/kernels/`, `include/strata/kernels/`): backward-compatible additions only, or ask me.
- Never `pkill -f` a pattern in your own command line. Keep results in `bench/`, commit your own paths only, update
  `docs/ds4/FINDINGS.md` and this prompt at milestones. Say what was NOT tested.
