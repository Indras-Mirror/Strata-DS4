#!/bin/bash
# batch 16: MTP verify on CUDA (first time). Best decode config + --f16-q8; MTP's own hc_head, experts resident.
#   base (no MTP) / --mtp (drafts only, acceptance) / --mtp --verify, interleaved x2, no ppl (verify rows differ)
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.gctx1g Q_PPL=
B=bench/ds4-2026-10-08; M=/media/mal/NVME1TB/Models/DS4-MTP/DeepSeek-V4-Flash-MTP-bf16.gguf
F="--arena-skip-resident --arena-gib 58 --vram-lru --pf-b 0.7 --f16-q8"
for r in 1 2; do
  $B/q.sh $B/b17-base-$r.log $F
  $B/q.sh $B/b17-verify-$r.log $F --mtp $M --mtp-resident --verify
done

echo batch16-done
