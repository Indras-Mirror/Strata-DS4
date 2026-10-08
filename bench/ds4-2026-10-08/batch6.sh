#!/bin/bash
# batch 6: batched indexer top-k in chunks (argsort off-CPU); p3000 chunk 4096, full arena, MMQ vs MMVQ, ppl
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.argsort MIMO_CHUNK_PROF=1
export Q_PROMPT=bench/ds4-2026-10-05/goldens/p3000/tokens.i32 Q_N=128
Q=bench/ds4-2026-10-08/q.sh; B=bench/ds4-2026-10-08; F="--vram-lru --prefill-chunk 4096"
$Q $B/b6-p3000-mmq.log $F --chunk-mmq
$Q $B/b6-p3000-mmvq.log $F
echo batch6-done
