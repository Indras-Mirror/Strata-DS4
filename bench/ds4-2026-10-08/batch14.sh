#!/bin/bash
# batch 14: interleaved A/B DS4_COMP_SKIP=0/1 (same binary), p600 decode config, no ppl; then one ppl run with skip
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.compskip
B=bench/ds4-2026-10-08; F="--arena-skip-resident --arena-gib 58 --vram-lru --pf-b 0.7"
for r in 1 2; do
  DS4_COMP_SKIP=0 Q_PPL= $B/q.sh $B/b14-off-$r.log $F
  DS4_COMP_SKIP=1 Q_PPL= $B/q.sh $B/b14-on-$r.log $F
done
$B/q.sh $B/b14-on-ppl.log $F
echo batch14-done
