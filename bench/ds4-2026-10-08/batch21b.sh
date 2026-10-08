#!/bin/bash
# b21b: the 32K and 128K curve points again, with a bigger VRAM margin.
# b21-ctx32768.log died in cudaGraphInstantiate ("CUDA error: out of memory" at decode start) AFTER a good prefill:
# --slots auto's default margin is 0.9 GiB, but the decode graph variants' capture cost grows with cap
# (cap = next_pow2(pos/4+1): 8192 at 32K, 32768 at 128K), so the margin has to cover it at long context.
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08
F="--vram-lru --pf-b 0.7 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --vram-margin 2.0"
for n in 32768 131072; do
    case $n in 32768) C=36000;; 131072) C=135000;; esac
    Q_PROMPT=$B/ctx/ctx$n.i32 Q_N=16 Q_PPL= $B/q.sh $B/b21-ctx$n-m2.log $F --ctx $C
done
echo "=== b21b summary ==="
for f in $B/b21-ctx32768-m2.log $B/b21-ctx131072-m2.log; do
    printf '%-26s %s | %s\n' "$(basename "$f")" "$(grep -h '^prefill:' "$f")" "$(grep -h '^decode :' "$f")"
    grep -h 'decode ms/token\|experts/token\|slots auto\|rc=' "$f" | sed 's/^/    /'
done
