#!/bin/bash
# batch 8: --chunk-prestage (layer l+1's experts stream while layer l computes); MMQ, chunk 4096, full arena
#   p3000 with ppl (must equal b6 MMQ ppl 6.4052: same math), p6000 speed
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.prestage MIMO_CHUNK_PROF=1
B=bench/ds4-2026-10-08; F="--vram-lru --prefill-chunk 4096 --chunk-mmq --chunk-prestage"
Q_PROMPT=bench/ds4-2026-10-05/goldens/p3000/tokens.i32 Q_N=64 $B/q.sh $B/b8-p3000-pre-ppl.log $F
Q_PROMPT=$B/p6000-concat.i32 Q_N=128 Q_PPL= $B/q.sh $B/b8-p6000-pre.log $F
echo batch8-done
