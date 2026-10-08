#!/bin/bash
# b24b: the rest of b24, queued BEHIND strata-glm's build + baseline run (Mal: GLM gets priority).  Waits for the
# marker GLM touches when it is done with the GPU, then the usual lock + MemAvailable gate in q.sh.
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08
M=~/.quetza-data/conductor/glm-gpu-done
until [ -e "$M" ]; do sleep 30; done
export DS4_BIN=$B/bin/ds4_generate.cq8
F="--vram-lru --pf-b 0.7 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --vram-margin 2.0"
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
