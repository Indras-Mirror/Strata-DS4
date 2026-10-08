#!/bin/bash
# batch 12: shared decode graph allocator (VRAM -> slots). p600 decode config w/ ppl; p6000 chunked prefill + decode
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.shared
B=bench/ds4-2026-10-08
$B/q.sh $B/b12-p600.log --arena-skip-resident --arena-gib 58 --vram-lru --pf-b 0.7
Q_PROMPT=$B/p6000-concat.i32 Q_N=128 Q_PPL= $B/q.sh $B/b12-p6000.log --vram-lru --pf-b 0.7 --prefill-chunk 4096 --chunk-mmq --chunk-prestage
echo batch12-done
