#!/bin/bash
# b27 (~8 min): dense-half op cuts (flat2 reshapes, DS4_OVL_GATHER half-row gather, gate taps only with gate_taps)
# p600 + 128 --ppl, old (dec2) vs new (dec3); ppl must match bit-for-bit-ish, tok/s is the point
B=bench/ds4-2026-10-08; cd /home/mal/AI/Strata-DS4
for v in dec2 dec3; do DS4_BIN=$B/bin/ds4_generate.$v bash $B/q.sh $B/b27-$v.log --comp-type iq4_nl --icomp-q8; done
echo done > $B/b27.done
