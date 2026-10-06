# MiMo-V2.6-Flash on this box (RTX 4090 24 GB + 90 GB DDR4) - feasibility (2026-10-06)

Status: **desk study only** - nothing downloaded, nothing run. Numbers marked *card* come from the model card /
`quantization.json` of `pfeifferj/MiMo-V2.6-Flash-RL-GSQ-RCO-GGUF` and the base model's `config.json`; everything
else is *estimated* from them plus the DeepSeek-V4 measurements in `docs/ds4/FINDINGS.md` s10-s12.

## 1. The model (card)
- **MoE, yes.** `mimo2` arch, 308.8B stored params, ~15B active. 48 layers: layer 0 dense FFN (16384), layers 1-47
  routed, **256 experts, top-8, no shared expert**. Expert shape n_embd 4096 x n_ff 2048 x 3 matrices - **the same
  25.2M-param expert as DeepSeek-V4-Flash**, so Strata's DS4 expert tier (Ds4MoeTier, slots, prefetch) fits it as is
  (kMaxParts 8 = top-8 exactly).
- Attention: hybrid sliding-window (128) + global layers (`hybrid_layer_pattern`), GQA (4 kv heads global, 8 SWA),
  head_dim 192 / v 128, partial RoPE (0.334), sink bias on SWA layers, `attention_value_scale` 0.707. Much simpler than
  DS4's CSA/HCA compressor + indexer + hyper-connections.
- Experts: 47 x 256 = 12,032. Non-expert: 5.93B params.

## 2. The GSQ-RCO files (card)
| file | size | expert formats | non-expert | PPL | MMLU-Pro |
| --- | ---: | --- | --- | ---: | ---: |
| source reference | 172.9 GB | MXFP4 | BF16 | 3.253 | 57.70 |
| **GSQ-RCO 3-bit** | **115.7 GB** | Q2_K 212.6B + Q3_K 83.8B + MXFP4 6.4B params | Q8_0 (6.3 GB) | 3.375 | 57.35 |
| GSQ-RCO 3.5-bit | 135.0 GB | Q2_K/Q3_K/MXFP4 | BF16 | 3.356 | 57.25 |
Standard ggml types only - runs in stock llama.cpp (tested at `58367713`). RCO picks one format per stacked
projection tensor (141 groups), so formats vary per layer (see `quantization.json`).

## 3. Does it fit? (estimated)
3-bit: experts = 69.8 (Q2_K) + 36.0 (Q3_K) + 3.4 (MXFP4) = **109.2 GB = 101.7 GiB**, avg **9.07 MB/expert**
(DS4: 7.08 MB). Non-expert 5.9 GiB (VRAM).
Capacity for experts: VRAM ~16 GiB (24 - 5.9 non-expert - ~2 ctx/KV/scratch/CUDA) + a RAM arena of ~78 GiB (90 total
minus desktop/OS headroom; never pin near the 56.6 GB bad byte lane without the existing soft-offline mitigation)
= **~94 GiB < 101.7 GiB** - even with VRAM-resident experts dropped from the arena (Ds4MoeTier currently keeps the
arena a superset of the cache; a "VRAM-only resident" mode is needed), **~8 GiB (~900 coldest experts) live on the
NVMe**. If those still draw ~2-3% of lookups that is ~80 MB/token from disk = **~30-40 ms/token** - the dominant
cost. **3-bit: runs, but spills to SSD; ~10-12 tok/s at best in Strata, less in llama.cpp.** 3.5-bit: does not fit.

A **~2.25-2.5 bpw** variant fits cleanly: DS4's recipe (IQ2_XXS gate/up + Q2_K down) = 85.2 GB experts (79.4 GiB,
7.08 MB/expert - byte-identical expert size to DS4) + 5.9 GiB non-expert -> VRAM 16 GiB of slots (~2400) + a ~63 GiB
arena, nothing on disk.

## 4. Speed (estimated from DS4's measured machine numbers)
Per token: 8 x 47 = 376 expert lookups (DS4: 258).
- 2-bit recipe: 376 x 7.08 MB = 2.66 GB/token (1.45x DS4). ~20% of experts resident (DS4: 19.5% -> 57% hit, ~70% with
  prefetch - MiMo's routing skew is unmeasured). Misses ~0.8 GB/token over the ~30 GB/s shared CPU+PCIe DDR4 budget
  (FINDINGS s12) = ~27 ms + non-expert ~8-12 ms (6.3 GB/token of VRAM reads, cheap SWA attention) + ~6 ms overhead
  = **~42-50 ms -> ~16-20 tok/s** with predicted prefetch.
- 3-bit GSQ-RCO: 3.41 GB/token, fewer slots (~1890), + the NVMe spill -> **~10-12 tok/s**.
First thing to measure if pursued: MiMo's routing skew (route_probe + route_skew.py on MiMo) - it sets the hit rate.

## 5. Uncensored options (card / HF search 2026-10-06)
- **MorinoNushi/MiMo-V2.6-Flash-RL-Uncensored-Heretic-LoRA-GGUF**: rank-1 "Heretic" directional-ablation LoRA,
  ~70-100 MB, refusals 95.7% -> 3.6% (140 harmful prompts, thinking suppressed; higher with thinking on), harmless KL
  0.057 vs the MXFP4 base. Applies to **any** GGUF quant of MiMo-V2.6-Flash-RL at runtime (llama.cpp `--lora`), base
  weights untouched -> **GSQ-RCO 3-bit + this LoRA = uncensored 3-bit with no requantisation**.
  Open: which tensors it touches - if it includes the routed experts' projections, Strata's expert tier needs a
  rank-1 correction (out += B(A.x) per expert, cheap) or the edited tensors must be merged + requantised.
- Merged versions: MorinoNushi ...-Heretic-GGUF (MXFP4, 167 GB), AliceThirty ...-UNCENSORED-gguf (MXFP4, 168 GB),
  dealignai ...-UNCENSORED (safetensors). None is a 2-3 bit quant.
- **A 2-bit RCO uncensored does not exist.** Paths: (a) cheapest - llama.cpp `llama-quantize` with an imatrix to the
  IQ2_XXS/Q2_K recipe (the card's "native imatrix" control is this class: PPL 3.47 at 3-bit, so expect clearly worse
  at ~2.3 bpw) + the Heretic LoRA; (b) best - run GSQ + RCO (IST-DASLab code) for a ~2.3 bpw budget from the 173 GB
  source: needs the source on disk, expert-input capture over 1M tokens through a 309B model offloaded on this box
  (likely days of GPU), then the LoRA. RCO's own 3-bit result (KL 0.152) says a 2.3 bpw RCO is plausible but untested.

## 6. Port effort in Strata (estimated)
Reusable as is: Ds4MoeTier (expert cache, CPU pool, PCIe split, predicted prefetch - top-8 fits), memguard, the
replay harness, golden/compare tooling. New: the `mimo2` dense half (hybrid SWA/global attention with sinks,
partial RoPE, GQA, dense layer 0, router - check MiMo's gating func + bias) - **well under DS4's dense effort**
(no compressor, indexer or hyper-connections). Kernels: the 2-bit recipe reuses IQ2_XXS/Q2_K; the GSQ-RCO 3-bit needs
Q3_K and MXFP4 expert kernels (CPU AVX2 + CUDA grouped) added to the native expert path. Plus "VRAM-only resident"
arena mode for models larger than RAM.

## 7. Recommendation
Finish DeepSeek-V4 first (it is days from a real tok/s number). Then: (1) download nothing big yet - route-probe
MiMo's skew from the 3-bit GGUF in llama.cpp under memguard (mmap; it will page from NVMe, slow but safe) - that one
number decides 2-bit vs 3-bit; (2) try GSQ-RCO 3-bit + Heretic LoRA in stock llama.cpp `-cmoe` as the baseline;
(3) port only if the skew makes >= ~15 tok/s plausible.
