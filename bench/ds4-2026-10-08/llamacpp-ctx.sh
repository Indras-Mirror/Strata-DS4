#!/bin/bash
# b22: llama.cpp at 64K - the comparison number for the DS4 long-context decode (FINDINGS s28).
# Same document (ctx/ctx65536.txt), same context capacity, llama.cpp's own best configuration (config B, s3):
# every expert in RAM + the hot-expert cache, F16/Q4_0 KV, flash attention, 8 threads, no mmap.
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08; LOG=$B/b22-llamacpp-ctx65536.log
BIN=/home/mal/AI/llama.cpp-master-rebase/build-cu134/bin/llama-cli
M=/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf
L=~/.quetza-data/conductor/ds4-gpu.lock
until flock -n "$L" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge 70 ]; do sleep 30; done
ss -ltn | grep -q 8188 && { echo "ComfyUI up - refusing" > "$LOG"; exit 9; }
bash tools/ds4/memguard.sh 80 78 -- "$BIN" -m "$M" -f "$B/ctx/ctx65536.txt" -n 128 -c 70000 \
    -ngl 99 -cmoe --moe-expert-cache 40 --load-mode none -fa on -t 8 -ctk q4_0 -ctv q4_0 --no-kv-offload \
    --no-warmup > "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
grep -hE 'prompt eval time|llama_perf_context_print: +eval time|llama_perf_context_print: +load time' "$LOG" | tail -4
