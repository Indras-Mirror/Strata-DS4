#!/bin/bash
# batch 13: interleaved A/B, same binary: DS4_SHARED_ALLO=0 vs 1, twice (box busy: VLC + wakeword ~1 core)
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.shared
B=bench/ds4-2026-10-08; F="--arena-skip-resident --arena-gib 58 --vram-lru --pf-b 0.7"
for r in 1 2; do
  DS4_SHARED_ALLO=0 Q_PPL= $B/q.sh $B/b13-per-$r.log $F
  DS4_SHARED_ALLO=1 Q_PPL= $B/q.sh $B/b13-shared-$r.log $F
done
echo batch13-done
