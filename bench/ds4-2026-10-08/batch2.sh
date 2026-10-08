#!/bin/bash
# batch 2: quiet ggml log (skip-resident repeat), vram-lru byte check, vram-lru speed. Snapshot binary.
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.quietlog
Q=bench/ds4-2026-10-08/q.sh; B=bench/ds4-2026-10-08
$Q $B/b2-skipres-quiet.log --arena-skip-resident --arena-gib 58
DS4_CHECK_LRU=1 $Q $B/b2-lru-check.log --arena-skip-resident --arena-gib 58 --vram-lru -n 32
$Q $B/b2-lru.log --arena-skip-resident --arena-gib 58 --vram-lru
echo batch2-done
