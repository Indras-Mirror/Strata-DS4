#!/bin/bash
# b25 (short GPU slot, ~20 min): today's decode/KV changes on CUDA.
#  0. CUDA mini-fixture gate (seconds): fused indexer + O(ntk) sparse select + raw blocking vs ds4_ref
#  1. p600 + 128 decode, OLD (cq8 snapshot) vs NEW (dec1), --ppl - same config as b23
#  2. --pos-offset probes (decode over empty caches at a deep position, no prefill): 128K fused-idx off/on,
#     + --icomp-q8 --comp-type q8_0, then 256K with --icomp-q8 --comp-type q4_0
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08; D=bench/ds4-2026-10-06/dense; L=~/.quetza-data/conductor/ds4-gpu.lock
{
for F in swa16:40 tame:63; do f=${F%:*}; t=${F#*:}
  flock "$L" ./build-ds4-gpu/test_ds4_dense $D/mini-ds4-$f.gguf --tokens $t --work $D/cpu-x --ref-dir $D/cpu-x/ref-allpos-mini-ds4-$f --cuda --multi 2,3,1,4 2>&1 | grep -E 'logits cosine|min cos|PASS|FAIL' | tail -4
done; } > $B/b25-cuda-gate.log 2>&1
F="--vram-lru --pf-b 0.7 --f16-q8 --arena-skip-resident --arena-gib 58"
DS4_BIN=$B/bin/ds4_generate.cq8  $B/q.sh $B/b25-p600-old.log $F
DS4_BIN=$B/bin/ds4_generate.dec1 $B/q.sh $B/b25-p600-new.log $F
export DS4_BIN=$B/bin/ds4_generate.dec1 Q_PROMPT=$B/ctx/probe16.i32 Q_N=48 Q_PPL=
DS4_FUSED_IDX=0 $B/q.sh $B/b25-p128k-chain.log $F --pos-offset 131072 --vram-margin 2.0
$B/q.sh $B/b25-p128k-fused.log $F --pos-offset 131072 --vram-margin 2.0
$B/q.sh $B/b25-p128k-q8q8.log  $F --pos-offset 131072 --vram-margin 2.0 --icomp-q8 --comp-type q8_0
$B/q.sh $B/b25-p256k-q8q4.log  $F --pos-offset 262144 --vram-margin 2.0 --icomp-q8 --comp-type q4_0
echo "=== b25 summary ==="; cat $B/b25-cuda-gate.log
for f in $B/b25-p600-old.log $B/b25-p600-new.log $B/b25-p128k-chain.log $B/b25-p128k-fused.log $B/b25-p128k-q8q8.log $B/b25-p256k-q8q4.log; do
  printf '%-22s %s | %s | %s\n' "$(basename $f)" "$(grep -h '^decode :' $f | sed 's/.*= //')" "$(grep -h '^ppl:' $f | sed 's/.*perplexity //')" "$(grep -h 'slots auto' $f | grep -o '[0-9]* slots')"
  grep -h 'decode ms/token\|experts/token\|rc=' $f | sed 's/^/    /'
done
