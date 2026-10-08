#!/bin/bash
# batch 4: p3000 chunk 2048 profiled (MIMO_CHUNK_PROF=1), MMVQ vs --chunk-mmq (clamped swiglu), with ppl
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.mmq MIMO_CHUNK_PROF=1
export Q_PROMPT=bench/ds4-2026-10-05/goldens/p3000/tokens.i32 Q_N=64
Q=bench/ds4-2026-10-08/q.sh; B=bench/ds4-2026-10-08; F="--arena-skip-resident --arena-gib 58 --vram-lru --prefill-chunk 2048"
$Q $B/b4-p3000-mmvq.log $F
$Q $B/b4-p3000-mmq.log $F --chunk-mmq
echo batch4-done
