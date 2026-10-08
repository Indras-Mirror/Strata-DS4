#!/bin/bash
# batch 18: MTP draft with only its K heaviest routed experts (--mtp-keep 1/2/3; all 6 = b16-draft: 67.7%, 11.2 ms)
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.mtpkeep Q_PPL=
B=bench/ds4-2026-10-08; M=/media/mal/NVME1TB/Models/DS4-MTP/DeepSeek-V4-Flash-MTP-bf16.gguf
F="--arena-skip-resident --arena-gib 58 --vram-lru --pf-b 0.7 --f16-q8 --mtp $M --mtp-resident"
for k in 1 2 3; do $B/q.sh $B/b18-keep$k.log $F --mtp-keep $k; done
echo batch18-done
