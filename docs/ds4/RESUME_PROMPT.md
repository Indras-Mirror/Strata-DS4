# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're building Strata-DS4: the Strata inference engine for DeepSeek-V4-Flash-0731, decoding faster than llama.cpp on my
RTX 4090 24 GB + Ryzen 7 5700X (8c, AVX2 only) + 90 GB DDR4. Public repo: https://github.com/Indras-Mirror/Strata-DS4
(local branch `deepseek4` tracks remote `ds4/main`; remote `origin` is the old Strata fork - don't push there).
**Work solo** - no /conductor, no workers, no subagents. Speed first, quality measured alongside.
Another session works on MiMo in `~/AI/Strata-MiMo` (branch `mimo`) - don't touch it; you share the GPU lock.
Written 2026-10-07 at commit `cabe73f` + the docs commit after it.

## THIS SESSION IS CPU-ONLY
The MiMo session has the GPU. **Do not run anything on the GPU** (no `build-ds4-gpu` binary execution, no memguard
full-model loads, no `--cuda` tests) until I say the GPU is free. Compiling `build-ds4-gpu` is fine (CPU work,
`nice -n 10`). Everything in "CPU work queue" below is designed to be done and verified on the CPU with the mini
fixtures. Keep CPU load polite (`nice -n 10`, the MiMo session may be timing things): small tests only.

Read first, in this order: this file; `docs/ds4/FINDINGS.md` s13 (the latest measurements); `docs/ds4/ENGINE_DENSE.md`
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
  **Real-model acceptance 59.8%** (76/127, greedy, n_max=1), draft cost 16.2 ms (MTP experts on the CPU via ggml
  `mul_mat_id` straight off the MXFP4 file). philpax measured **0.74-0.93** acceptance for this same head on 0731
  with ik_llama.cpp (README-philpax.md next to the file) - we are leaving acceptance on the table, see task 1.
- `ds4_generate --mtp FILE --verify` (greedy speculative decoding, 2-token verify passes with shared expert reads via
  `Ds4MoeTier::run_multi`): **written and compiled, never run** (not even on the CPU).
- Estimate, NOT measured: verify at 60% acceptance + 16 ms drafts ~22 tok/s; at ~75% + ~3 ms drafts ~27 tok/s.

## CPU work queue (do in order; each step: implement, gate on the CPU, commit, note in FINDINGS)

### 1. Fix the MTP head's hc_head (likely the acceptance gap)
`Ds4Dense` uses the TRUNK's `output_hc_fn/base/scale` for the MTP head (`build_head(im, n, true)` in
`tools/ds4/ds4_dense.cpp`; `ds4_ref.cpp`'s `do_head` too). That is wrong: the MTP file's own `output_hc_*` differ
substantially from the trunk's (scale 1.65 vs 0.79; base/fn differ throughout - checked 2026-10-07 with gguf-py), and
llama.cpp loads the MTP file as its OWN model (`mtp_only` in `src/models/deepseek4.cpp:95`), so its `graph_mtp` uses
the MTP file's `hc_head_*` (`model.hc_head_fn`, line ~1573) - and also that file's `token_embd` and `output` (BF16
copies of the ORIGINAL V4's, not 0731's). The avlp12 sidecar also lists `hc_head.*` as an MTP-own parameter.
- Load the MTP file's `output_hc_fn/base/scale` (prefix-filtered load currently takes only `blk.43.*`: add these three
  under new names, e.g. `mtp.output_hc_*`, so they don't collide with the trunk's) and use them in the MTP head.
  Same change in `ds4_ref --mtp` (the oracle) and in `tools/ds4/make_mini_mtp.py` (write `output_hc_*` too, with values
  different from the trunk's, so the gate would catch using the wrong ones).
- Decide token_embd/output for the MTP: the MTP file's are BF16 (1 GB each) of the original V4. v1 = keep the trunk's
  (0731) embedding/output; note it. A later GPU A/B can measure acceptance with the MTP file's output head
  (requantized to q6_k, ~0.4 GB VRAM = ~60 slots) vs the trunk's.
- Gate: `test_ds4_dense <fixture> --mtp <mini mtp> --multi 2,3,1,4` (see "Gates" below), plus a mutation check
  (swap back to the trunk's hc_head -> the MTP gate must fail).

### 2. Smoke-test `--verify` on the CPU (it has never run)
`ds4_generate` runs on the CPU with the mini fixtures: `--backend cpu --experts cpu --arena-gib 1` (see Gates).
- On the CPU the multi-token kernels are bit-identical to one-token, so **`--verify` must produce byte-identical
  tokens to plain greedy decoding**. Compare stdout of both (several prompts, `-n 40`).
- The mini MTP has random weights (~0% acceptance), so only the reject path runs. Add a test hook to force the accept
  path: e.g. env `DS4_VERIFY_ORACLE=<ids file>` = take drafts from a plain-greedy run's output (100% acceptance), and
  a variant that corrupts every 3rd draft (mixed accept/reject). Check: identical tokens, correct pass counts,
  `n_predict` and `--stop` honoured exactly (emit() can produce 2 tokens per pass - the stop/limit must cut at the
  right one), position bookkeeping (pos += 2 on accept, += 1 on reject).
- Review the loop in `tools/ds4/ds4_generate.cpp` (search `if (a.verify)`): after a REJECT the trunk ran position
  pos+1 with the wrong draft - that position is decoded again next pass (state is position-indexed: raw ring slot,
  compressor ring slots, compressed rows are all rewritten). The MTP after an accept runs n=2 at [pos, pos+1] with
  next tokens [t1, t2]; after a reject n=1 at [pos] with next t1. Double-check the MTP block's own KV gets every
  position exactly once in final form.

### 3. Make the MTP draft cheap (16.2 ms -> a few ms)
Measured 16.2 ms/draft; the MTP dense part is one layer on the GPU (~0.5 ms expected), so the CPU ggml MXFP4
`mul_mat_id` (`MtpExperts` in ds4_generate.cpp, 6 experts x ~13 MB) is the suspect - but **measure first**:
- CPU microbenchmark (no GPU needed): time `MtpExperts::run` alone on the REAL MTP file with random routed ids (mmap,
  ~3.2 GB of experts; run under `systemd-run ... MemoryMax=8G`, warm the page cache first). Expect ~80 MB/draft at
  ~15 GB/s = ~5 ms if bandwidth-bound; if it is far slower, ggml's MXFP4 CPU path is the problem.
- Option A: the tier's own CPU pool on MXFP4: `Ds4MoeTier` with `cpu_only` over the MTP file. Its kernels are generic
  ggml-cpu dot products (`native_fmt` in `src/kernels/cpu/native_expert.cpp` accepts any type with a vec_dot - MXFP4
  has one, activations Q8_0) but the tier enumerates experts as `blk.0 .. blk.<block_count-1>` (`ds4_moe.cpp` ~350):
  the MTP file has block_count 44 and experts only in `blk.43`. Add an opt-in "only layer N" config field
  (default = old behaviour; this is our own tool code, not a shared kernel). Gate with `test_ds4_moe`-style check vs
  a dequant+F32 reference on a few MTP experts.
- Option B (GPU, later): the MTP's routing may be skewed enough for a small VRAM cache of MTP experts; ggml-cuda can
  run MXFP4 `mul_mat_id`. Costs slots - measure the MTP routing skew first (log its routed ids during a run).
- Also add a timing split to `mtp_draft_n` (dense vs experts ms) so the next GPU run shows where the 16 ms goes.

### 4. Apply the 0731-aligned sidecar (optional, after 1-3)
`/media/mal/NVME1TB/Models/DS4-MTP/mtp_aligned_r6c_step5000.safetensors` (avlp12, README-avlp12.md): the MTP's 22
non-expert tensors (74.3M params, bf16 + f32) re-trained on 0731's own greedy outputs; they report chained-draft
acceptance d2 43.2->51.1%, d3 9.8->22.2%. Names: `block.attn.{wq_a,wq_b,wkv,q_norm,kv_norm,attn_sink}`,
`block.{attn,ffn}_hc.{fn,base,scale}`, `block.{attn,ffn}_norm`, `e_proj`, `h_proj` (separate! our GGUF has them fused
as `nextn.eh_proj` [8192 -> 4096]: check the concat order against llama.cpp's converter - our graph concatenates
[enorm(embd) | hnorm(h)], so eh_proj = [e_proj | h_proj] along the input dim if the converter matches), `enorm`,
`hnorm`, `norm` (= shared_head_norm), `hc_head.{fn,base,scale}`. Note the safetensors shapes are [out, in] (torch),
GGUF ne is [in, out]. Write a converter that produces an "aligned" MTP GGUF (copy the original, replace these tensors;
experts unchanged), keep it outside the repo (`/media/mal/NVME1TB/Models/DS4-MTP/`), and gate it on the CPU by
loading it in `test_ds4_dense --mtp` (runs, finite) - real acceptance needs the GPU (queue it).

### 5. Smaller items
- +0.7 ms attention+router since the multi-token change (22.24 vs 21.53 ms, 2 runs each) - unexplained. CPU-side
  investigation: compare the n=1 graphs' node counts / ops old (`3e39323`) vs new (`ggml_graph_n_nodes`, op names) on
  the mini fixture; look for extra copies/conts the views introduced. Fix only with evidence; the real test is GPU.
- Long contexts: the per-pass input upload carries `vis` as comp_max*kNtMax floats per compressed layer
  (~0.35 ms/token at 32K). Building visibility on the device from i_pos would remove it.
- Update `README.md` (roadmap/status: multi-token, MTP, acceptance) at the next milestone; FINDINGS + this prompt too.

## GPU queue (only when I say the GPU is free; each through memguard, one at a time)
Base command (p600, the comparison config):
`bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf --ids-file bench/ds4-2026-10-05/goldens/p600/tokens.i32 -n 128 --slots auto --arena-gib 70 --pcie 0.55 --ppl --dense-requant q6_k --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin`
(the prompt is the BINARY `goldens/p600/tokens.i32`; `bench/ds4-2026-10-06/real/p600.ids` is an old run's text
output, not a prompt). Logs to `bench/ds4-2026-10-0X/`.
1. `--mtp <file>` after task 1: acceptance with the MTP's own hc_head (was 59.8%).
2. `--mtp <file> --verify`: tok/s, tokens/pass, and token equality vs plain greedy (expect rare divergence: CUDA
   kernels differ for 2 columns; compare against plain greedy of the SAME binary, several runs).
3. `test_ds4_dense --cuda` with `--mtp` + `--multi` on the mini fixtures (the MTP block on CUDA).
4. nsys profile for the +0.7 ms.
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
#   expect: MTP logits worst cos 1.00000000, top-1 62/62, MTP in multi passes bit-identical 63/63, PASS
#   (rm h.f32 before re-running: the dump APPENDS; the first command's own MTP check fails ~0.99996 by design)
# generator end to end on the CPU (output must not change with --mtp; --verify must match plain greedy on the CPU)
./build-ds4-gpu/ds4_generate -m $D/mini-ds4-tame.gguf --ids 11,48,21,58,31,4,41,14 -n 24 --backend cpu --experts cpu --arena-gib 1 [--mtp $D/mini-ds4-mtp.gguf [--verify]]
```
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

## Rules (non-negotiable)
- **Every full-model load through `tools/ds4/memguard.sh 80 78 -- <cmd>`** (cgroup RAM cap, swap off, watchdog,
  shared GPU lock `~/.quetza-data/conductor/ds4-gpu.lock`, refuses while ComfyUI is up). NEVER a CPU-only full-model
  run. One full-model process at a time. Check `ss -ltn | grep 8188` + `nvidia-smi` first; if ComfyUI is up, ask me.
- If `/media/mal/SSD NVME` is missing after a reboot: `udisksctl mount -b /dev/nvme0n1p3`.
- Push to `ds4` only when I say so (local is 9+ commits ahead); nothing private or third-party in the tree (two corpus
  files were stripped from history; private copies in `~/AI/Strata-DS4-private/`). Shared kernels
  (`src/kernels/`, `include/strata/kernels/`): backward-compatible additions only, or ask me.
- Never `pkill -f` a pattern in your own command line. Keep results in `bench/`, commit your own paths only, update
  `docs/ds4/FINDINGS.md` and this prompt at milestones. Say what was NOT tested.
