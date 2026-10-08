#!/bin/bash
# b26 (~15 min): the KV ladder's real-model quality (p600 + 128, --ppl, same binary) and the pinned-RAM probe.
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08; export DS4_BIN=$B/bin/ds4_generate.dec2
F="--vram-lru --pf-b 0.7 --f16-q8 --arena-skip-resident --arena-gib 58"
$B/q.sh $B/b26-f32.log    $F
$B/q.sh $B/b26-q8i8.log   $F --comp-type q8_0 --icomp-q8
$B/q.sh $B/b26-q4.log     $F --comp-type q4_0 --icomp-q8
$B/q.sh $B/b26-iq4.log    $F --comp-type iq4_nl --icomp-q8
Q_PROMPT=$B/ctx/probe16.i32 Q_N=48 Q_PPL= $B/q.sh $B/b26-p256k-host.log $F --pos-offset 262144 --vram-margin 2.0 --icomp-q8 --comp-type q8_0 --comp-host
echo "=== b26 summary ==="
for f in $B/b26-f32.log $B/b26-q8i8.log $B/b26-q4.log $B/b26-iq4.log $B/b26-p256k-host.log; do
  printf '%-20s %s | ppl %s | %s\n' "$(basename $f)" "$(grep -h '^decode :' $f | sed 's/.*= //')" "$(grep -h '^ppl:' $f | sed 's/.*perplexity //')" "$(grep -h 'slots auto' $f | grep -o '[0-9]* slots')"
  grep -h 'decode ms/token\|rc=' $f | sed 's/^/    /'
done
