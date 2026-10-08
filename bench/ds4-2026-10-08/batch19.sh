#!/bin/bash
# batch 19: what context fits today - load at --ctx 65536 / 131072 / 262144 (short prompt, 32 tokens): VRAM, slots
cd /home/mal/AI/Strata-DS4
export DS4_BIN=bench/ds4-2026-10-08/bin/ds4_generate.mtpkeep Q_PPL= Q_N=32
B=bench/ds4-2026-10-08; F="--arena-skip-resident --arena-gib 58 --vram-lru --pf-b 0.7 --f16-q8"
for c in 65536 131072 262144; do $B/q.sh $B/b19-ctx$c.log $F --ctx $c; done
echo batch19-done
