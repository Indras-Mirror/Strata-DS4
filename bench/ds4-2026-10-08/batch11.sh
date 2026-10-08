#!/bin/bash
# batch 11: decode tuning with --vram-lru (p600, 128 tokens, ppl): prefetch budget (wasted DMAs block the copy
# engine at 85% hit) and skip-miss. Reference: b2-lru.log 22.29 tok/s ppl 10.4568 (pf-b 1.43 default)
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.skip
B=bench/ds4-2026-10-08; F="--arena-skip-resident --arena-gib 58 --vram-lru"
$B/q.sh $B/b11-pfb07.log $F --pf-b 0.7
$B/q.sh $B/b11-pfb0.log $F --pf-b 0
$B/q.sh $B/b11-skip05.log $F --skip-miss 0.05
$B/q.sh $B/b11-skip10.log $F --skip-miss 0.10
echo batch11-done
