# Resume prompt: Strata for DeepSeek-V4-Flash

Paste everything below the line into a fresh Claude Code session started in `~/AI/Strata-DS4`.

---

We're building Strata-DS4: the Strata inference engine for DeepSeek-V4-Flash-0731, decoding faster than llama.cpp on my
RTX 4090 24 GB + Ryzen 7 5700X (8c, AVX2 only) + 90 GB DDR4. Public repo: https://github.com/Indras-Mirror/Strata-DS4
(local branch `deepseek4` tracks remote `ds4/main`; remote `origin` is the old Strata fork - don't push there).
**Work solo** - no /conductor, no workers. Speed first (Qwen-Flash-level is the aim), quality measured alongside.
Another session works on MiMo in `~/AI/Strata-MiMo` (branch `mimo`) - don't touch it; you share the GPU lock.
Read first: `README.md` (status, numbers, flags), `docs/ds4/FINDINGS.md`, `docs/ds4/ENGINE_DENSE.md`,
`docs/ds4/ENGINE_MOE.md`, then `git log --oneline -25`. (Written 2026-10-07.)

## Where it stands
- The real engine runs on the GPU end to end: `build-ds4-gpu/ds4_generate` (build tree configured with
  `-DSTRATA_GGML_CUDA=ON`, which also turns on CUDA graphs). Coherent text; first-token logits = llama.cpp's top-1.
- Best measured (p600 prompt, 128-token decode): **~19.1-19.4 tok/s, ppl 10.55** with
  `--slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin`
  (llama.cpp best config: 16.34). Unmodified weights: 17.0 tok/s, ppl 10.49. `--route-bias 0.2`: 22 tok/s, ppl 11.50.
- Per-token cost at the best config: attention+router ~21 ms, experts ~25 ms (CPU pool + PCIe share; bandwidth-bound,
  DDR4 shared by CPU and DMA), finish ~1.5, head ~0.6.
- `Ds4MoeTier::run_multi` (n<=4 tokens per layer, each distinct expert read once) is done and tested (arm 5).

## In progress: multi-token verify + the MTP draft head
1. DONE (`97ae1b9`): rings sized for multi-token passes (raw window storage SWA+3, state rings ring+3); n=1 unchanged
   (CPU mini gate bit-exact).
2. NEXT: the n-token attention graph in `tools/ds4/ds4_dense.cpp` - design:
   - per-step inputs and persistent hand-offs (`x_state`, `routed_sum`, per-layer `hap/post_f/comb_f/shexp/fn`,
     router outputs) get an n axis (kNtMax = 4); graph variants keyed by (cap, n); `reserve_graphs` covers n=1,2;
   - raw keys gathered oldest->newest over the union window (SWA+n-1 keys) with a per-query mask [n_kv, n] (n=1 =
     today's graph exactly); one flash_attn for all n queries;
   - compressor/indexer state updates unrolled per token, comp-row writes chained (a block completed by token 0 is
     visible to token 1, never the reverse); indexer scores/top-k/mask batched per query;
   - projections, fused hc ops, router, shared expert, head all take n columns.
   Gate: CPU mini - a 2-token pass must match two 1-token passes (~1e-6); n=1 still bit-exact vs ds4_ref.
3. Then the MTP head as a 44th layer: `/media/mal/NVME1TB/Models/DS4-MTP/DeepSeek-V4-Flash-MTP-bf16.gguf` (philpax,
   sha verified). Layer 43 is a ratio-0 sliding-window layer: inputs h_t (main final hc streams, before the head) and
   embd(t+1) -> hnorm/enorm -> concat -> nextn.eh_proj -> normal DS4 layer (own SWA KV) -> hc_head ->
   nextn.shared_head_norm -> main `output`. Mirror llama.cpp `src/models/deepseek4.cpp` graph_mtp
   (~/AI/llama.cpp-master-rebase). Its experts are MXFP4 (the CUDA expert kernels don't do MXFP4): v1 = a CPU-only
   second Ds4MoeTier straight from the file (needs a layer offset: blobs are blk.43); v2 = requant to IQ4_XS + VRAM
   slots. Skip its duplicate token_embd/output. Optional: avlp12 0731-aligned sidecar (`mtp_aligned_r6c_step5000.safetensors`).
   MTP KV must be filled for the last <=128 prompt positions during prefill.
4. Verify loop in ds4_generate (n_max=1 first): draft d from MTP; main pass on [a, d]; accept if argmax at a == d.
   Rollback is free: all decode state is position-indexed and rewritten when a position is decoded again.
   Measure acceptance, tok/s, and greedy-token equality vs plain decoding.
5. After: DSpark (0731 native, 3 MoE layers, block 5; local Q8_0 at `/media/mal/SSD NVME/Models/
   huihui-DeepSeek-V4-Flash-0731-dspark-abliterated/`), n-gram drafting, batched prefill, an OpenAI server.

## Rules (non-negotiable)
- **Every full-model load through `tools/ds4/memguard.sh 80 78 -- <cmd>`** (cgroup RAM cap, swap off, watchdog, shared
  GPU lock `~/.quetza-data/conductor/ds4-gpu.lock`, refuses while ComfyUI is up). NEVER a CPU-only full-model run.
  One full-model process at a time. Check `ss -ltn | grep 8188` + `nvidia-smi` first; if ComfyUI is up, ask me.
- Small tests: `nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 <cmd>`.
  Mini fixtures (untracked, keep): `bench/ds4-2026-10-06/dense/mini-ds4{,-swa16,-tame}.gguf`.
- If `/media/mal/SSD NVME` is missing after a reboot: `udisksctl mount -b /dev/nvme0n1p3`.
- Push to `ds4` only when I say so; nothing private or third-party in the tree (two corpus files were stripped from
  history; private copies in `~/AI/Strata-DS4-private/`). Shared kernels: backward-compatible additions only.
- Never `pkill -f` a pattern in your own command line. Keep results in `bench/`, commit your own paths, update
  `docs/ds4/FINDINGS.md` and this prompt at milestones.
