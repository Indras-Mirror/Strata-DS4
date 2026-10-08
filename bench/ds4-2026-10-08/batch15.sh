#!/bin/bash
# batch 15: --f16-q8 (compressor/indexer F16 -> Q8_0), interleaved off/on x2, ppl in every run; box quiet now
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.f16q8
B=bench/ds4-2026-10-08; F="--arena-skip-resident --arena-gib 58 --vram-lru --pf-b 0.7"
for r in 1 2; do
  $B/q.sh $B/b15-off-$r.log $F
  $B/q.sh $B/b15-on-$r.log $F --f16-q8
done
echo batch15-done
