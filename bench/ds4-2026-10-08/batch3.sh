#!/bin/bash
# batch 3: chunked prefill on the 2998-token prompt (fast seed), chunk 2048 / 4096, no ppl (speed); then 4096 + ppl
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.fastseed
export Q_PROMPT=bench/ds4-2026-10-05/goldens/p3000/tokens.i32 Q_N=256 Q_PPL=
Q=bench/ds4-2026-10-08/q.sh; B=bench/ds4-2026-10-08; F="--arena-skip-resident --arena-gib 58 --vram-lru"
$Q $B/b3-p3000-c2048.log $F --prefill-chunk 2048
$Q $B/b3-p3000-c4096.log $F --prefill-chunk 4096
Q_PPL=--ppl $Q $B/b3-p3000-c4096-ppl.log $F --prefill-chunk 4096
echo batch3-done
