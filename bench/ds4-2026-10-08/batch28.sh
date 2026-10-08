#!/bin/bash
# b28 (~8 min): dec2 (pre-cuts) vs dec4 (op cuts + K==V single cast + CUDA FWHT hint), the standard flag set
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08
F="--vram-lru --pf-b 0.7 --f16-q8 --arena-skip-resident --arena-gib 58 --comp-type iq4_nl --icomp-q8"
for v in dec2 dec4; do DS4_BIN=$B/bin/ds4_generate.$v $B/q.sh $B/b28-$v.log $F; done
for v in dec2 dec4; do printf '%s: ' $v; grep -h '^decode :\|^ppl:' $B/b28-$v.log | sed 's/.*= //;s/.*perplexity /ppl /' | tr '\n' ' '; grep -h 'decode ms/token' $B/b28-$v.log; done
