#!/bin/bash
# 2026-10-08 batch 1: graph-reuse A/B, arena-skip-resident, vram-lru (byte check, then speed). One at a time via q.sh.
cd /home/mal/AI/Strata-DS4
Q=bench/ds4-2026-10-08/q.sh; B=bench/ds4-2026-10-08
DS4_GRAPH_REUSE=0 $Q $B/b1-base-reuse0.log
DS4_GRAPH_REUSE=1 $Q $B/b1-base-reuse1.log
$Q $B/b1-skipres.log --arena-skip-resident --arena-gib 58
DS4_CHECK_LRU=1 $Q $B/b1-lru-check.log --arena-skip-resident --arena-gib 58 --vram-lru -n 32
$Q $B/b1-lru.log --arena-skip-resident --arena-gib 58 --vram-lru
echo batch1-done
