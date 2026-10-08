#!/bin/bash
# b24: (1) --comp-q8's long-context payoff - 128K with and without it, same binary (the slot gain only shows at long
# context; the ppl cost was settled at p600 in b23); (2) the llama.cpp 64K bar (never obtained, s32); (3) the first
# 256K end-to-end run, with --comp-q8 and a 2048 prefill chunk (halves the [cap, n] score/mask intermediates,
# 1 GiB each at cap 65536 and n 4096, which were never blocked).
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08
export DS4_BIN=$B/bin/ds4_generate.cq8
F="--vram-lru --pf-b 0.7 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --vram-margin 2.0"
Q_PROMPT=$B/ctx/ctx131072.i32 Q_N=32 Q_PPL= $B/q.sh $B/b24-128k-q8.log  $F --ctx 135000 --comp-q8
Q_PROMPT=$B/ctx/ctx131072.i32 Q_N=32 Q_PPL= $B/q.sh $B/b24-128k-f32.log $F --ctx 135000
bash $B/llamacpp-ctx.sh
F2="--vram-lru --pf-b 0.7 --prefill-chunk 2048 --chunk-mmq --chunk-prestage --vram-margin 2.0"
Q_PROMPT=$B/ctx/ctx262144.i32 Q_N=32 Q_PPL= $B/q.sh $B/b24-256k-q8.log $F2 --ctx 266000 --comp-q8
echo "=== b24 summary ==="
for f in $B/b24-128k-q8.log $B/b24-128k-f32.log $B/b24-256k-q8.log; do
    printf '%-20s %s | %s\n' "$(basename "$f")" "$(grep -h '^prefill:' "$f")" "$(grep -h '^decode :' "$f")"
    grep -h 'decode ms/token\|experts/token\|slots auto\|rc=' "$f" | sed 's/^/    /'
done
grep -h 'llama.cpp 64K' $B/b22-llamacpp-ctx65536.log
