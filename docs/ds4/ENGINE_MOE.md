# The routed-expert tier (`Ds4MoeTier`)

Slice `ds4-moe` of the Phase-4b decode engine. `tools/ds4/ds4_moe.{hpp,cpp}` turn the machinery that
`tools/ds4/moe_replay.cpp` proved (FINDINGS s10-s12) into one object the engine holds per model and calls once per
layer: a VRAM expert cache seeded from a routing profile, a pinned host arena, a CPU expert pool for the misses, a
PCIe share of the misses computed on the GPU, and predicted prefetch on a second stream. The harness stays as it
is - it is the measurement, and this library is what the engine gets.

```
tier.init(gguf, cfg, err);                  // once, at startup
tier.seed_from_routes(routes_bin, err);     // the static profile, once
for each layer l of each token:
    tier.prefetch(l, top16);                // right after predict(l), before attention(l) - async
    tier.run(l, ids6, w6, x, out);          // after router(l) - synchronous: out = sum_k w_k * expert_k(x)
```

`run` is synchronous because the engine needs `out` before the next layer; `prefetch` is not, so its DMAs overlap
the dense work the caller does between the two calls. `run_dev` is the same with `x` already in device memory.

## The invariant

For any `(layer, ids, weights, x)` the tier's output equals the weighted sum of the dequantised experts computed in
F32; the only permitted difference is the expert kernels' own 8-bit activation rounding. `test_ds4_moe` measures
both numbers separately - the kernel arithmetic against a `to_float` reference on the ENGINE's own quantized
activation (rel ~2.5e-07, the invariant proper) and the pure-float rounding beside it (~1.9e-02). Measured, not
argued.

## Three decisions worth knowing

**The SwiGLU clamp is NOT applied, on evidence.** DeepSeek-V4's graph clamps the expert gate/up pre-activations to
`swiglu_clamp_exp` = 10.0 before the SwiGLU (`tools/ds4/ds4_ref.cpp:933`, mirroring llama.cpp's DEEPSEEK4 branch);
the native kernels this tier is built on do not (`include/strata/kernels/cpu/native_expert.hpp:49`,
`src/kernels/cuda/iq_kernels.cu:1190`), and both existing parity gates pass with unclamped references
(Phase-2 `ds4_expert_parity` 3.0e-5, `moe_replay --correctness` 3.18e-7). So the tier reproduces the kernels
exactly, and the gate PRINTS the largest pre-activation it saw. On the real model's slices that is |gate| 5.06,
|up| 4.03 against a limit of 10.0 - the clamp is inactive there, so unclamped equals clamped and the tier's output
is the model's. Moving the limit into the tier would need a second, unfused gate/up path on both backends; if a
future activation ever exceeds 10.0 the printed probe is the trigger to build it.

**The arena is a budget, not a promise.** `arena_gib` (72.6 GiB - every expert of the real model) is a ceiling, and
`max_arena_gib`/`mem_floor_gib` cap it further so the tier can never be the process that swaps the box. The arena
fills in ROUTING-PROFILE order, so the bytes most likely to be asked for are the ones kept; everything past the
budget is the file tier - read from the blob source on demand and counted separately (`stats().file_tier`). It is
built once, by the first `seed_*` call; a re-seed refills the cache and never re-reads the file.

**CPU-only is a real mode, not a debug flag.** `cfg.cpu_only` runs every expert on the pool: no ExpertCache, no
PCIe share, no CUDA call, and - because `ds4_moe.cpp` is compiled without `DS4_MOE_CUDA` for the `ds4_moe_cpu`
target - no CUDA symbol on the link line either. That property is what makes the gate safe to run while another
process owns the GPU, and it is what serves machines with no GPU. The CUDA flavour is the SAME source with
`DS4_MOE_CUDA=1`; the gate's GPU arm is a separate binary so the CPU gate's safety property is never weakened.

Blob sources are pluggable (`Ds4BlobSource`): `Ds4GgufBlobs` mmaps the three per-role tensors of a `deepseek4`
GGUF and assembles one 6.75 MiB blob with three memcpys, so reading N experts touches N x 6.75 MiB of the file -
a slice of a tensor, not a model load. `Ds4MemoryBlobs` is the gate's synthetic arm.

## Gate

`test_ds4_moe` (CPU-only flavour) vs the dequant+F32 reference, limit 1e-3, several layers and expert sets:

```
nice -n 10 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 build-ds4/test_ds4_moe
```

Measured 2026-10-06 (round 3, this tree): 10 checks, 0 failures, 0 arm errors, exit 0. Weighted sum rel
2.27e-07 - 2.63e-07 and single expert 2.45e-07 - 2.91e-07 on synthetic L0/L1/L7 and real L0/L20 slices; the 8-bit
activation rounding printed beside them is 1.8e-02 - 2.1e-02. Clamp probe 5.06 / 4.03 (inactive).

The real arm reads <= 12 expert slices of the 0731 GGUF through its mmap (81 MiB touched). `make_mini_gguf.py` is
NOT used: it writes its routed experts as zero blocks on purpose, so a numeric gate against it would be vacuous.

The CUDA half COMPILES in `build-ds4-cuda` (`ds4_moe_cuda`, `test_ds4_moe_gpu`); `ds4_moe_replay` still builds
unchanged. The GPU arms are not run in this round - the machine's GPU is busy.

## READY TO RUN (GPU - blocked only by the GPU being busy)

Arm 3, the four sources a routed expert can come from (resident slot / prefetch staging / PCIe DMA out of the
pinned arena / the file tier) each against the dequant+F32 reference on 24 real expert slices (~162 MiB of the
GGUF through its mmap; the reference quantizes with the card's q8_1 for the card's experts and Q8_K for the pool's,
because the tier really uses both), split counted as well as the value:

```
nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 build-ds4-cuda/test_ds4_moe_gpu --gpu
```

Arm 4, the tier's own `run` on the real model, the real routes and the real prediction file - the replay's
measured arm through the library. FULL MODEL, so it must go through the memory guard (cap 64 GiB, needs 62 GiB
available) and it must not start while ComfyUI is up:

```
bash tools/ds4/memguard.sh 64 62 -- build-ds4-cuda/test_ds4_moe_gpu --timing \
  --routes bench/ds4-2026-10-06/hidden-probe/routes.bin \
  --pred   bench/ds4-2026-10-06/hidden-probe/pred.bin \
  --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin
```

`--gap-ms` (default 0.41) stands in for the layer's dense work between `prefetch` and `run`, as the replay's
`--gap-wall` did, so the wall has s12's shape and the comparable number is the replay's implied full-token
(20.57 tok/s at 2150 slots, pf-b 1.43, pcie 0.25 dithered - FINDINGS s12). Expect the tier to land near it;
that comparison is the point of arm 4.

## Residuals

- The clamp question above is left to its measured probe rather than to a second gate/up path on both backends.
- `prefetch` currently stages into `kMaxPf` (6) slots per layer; a deeper DMA window would need the replay's
  queueing, which s12 showed does not pay at this pf_b.
- `run_dev` has no CPU-only counterpart: with `x` on the device there is nothing to test on the CPU arm.
