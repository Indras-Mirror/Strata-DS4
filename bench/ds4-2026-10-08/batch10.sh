#!/bin/bash
# batch 10: routed sums device-to-device (no H2D between layers). p6000 MMQ +/- prestage; p3000 MMVQ+prestage ppl
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.d2d MIMO_CHUNK_PROF=1
B=bench/ds4-2026-10-08; F="--vram-lru --prefill-chunk 4096"
Q_PROMPT=$B/p6000-concat.i32 Q_N=64 Q_PPL= $B/q.sh $B/b10-p6000-mmq-pre.log $F --chunk-mmq --chunk-prestage
Q_PROMPT=$B/p6000-concat.i32 Q_N=64 Q_PPL= $B/q.sh $B/b10-p6000-mmq.log $F --chunk-mmq
Q_PROMPT=bench/ds4-2026-10-05/goldens/p3000/tokens.i32 Q_N=32 $B/q.sh $B/b10-p3000-mmvq-pre-ppl.log $F --chunk-prestage
echo batch10-done
