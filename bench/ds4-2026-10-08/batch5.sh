#!/bin/bash
# batch 5: p3000 chunk 4096, full 70 GiB arena (no skip-resident: the VRAM-seed experts stay in RAM for prefill),
# profiled; MMVQ vs --chunk-mmq (clamped swiglu); ppl for both
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.mmq MIMO_CHUNK_PROF=1
export Q_PROMPT=bench/ds4-2026-10-05/goldens/p3000/tokens.i32 Q_N=128
Q=bench/ds4-2026-10-08/q.sh; B=bench/ds4-2026-10-08; F="--vram-lru --prefill-chunk 4096"
$Q $B/b5-p3000-full-mmvq.log $F
$Q $B/b5-p3000-full-mmq.log $F --chunk-mmq
echo batch5-done
