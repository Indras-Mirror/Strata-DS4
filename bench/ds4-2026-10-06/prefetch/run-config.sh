#!/bin/bash
# run-config.sh <prefetch> <label> [--verbose]
# One full-model llama-server (config B) with --moe-prefetch <prefetch>, then probe.py.
# MUST be invoked through tools/ds4/memguard.sh (it loads the real model).
set -u
pf="$1"; label="$2"; shift 2
VERB="${1:-}"

BIN=/home/mal/AI/llama.cpp-ds4-prefetch/build/bin/llama-server
M=/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf
OUT=/home/mal/AI/Strata-DS4/bench/ds4-2026-10-06/prefetch
PORT=8140
LOG="$OUT/$label.log"

echo "$label: start $(date +%T) pf=$pf ${VERB}"
$BIN -m $M --port $PORT -c 32768 -fa on -t 8 -np 1 --jinja -ctk q4_0 -ctv q4_0 \
     --no-kv-offload --load-mode none -ngl 99 -cmoe --moe-expert-cache 40 \
     --moe-prefetch "$pf" $VERB > "$LOG" 2>&1 &
SP=$!

( while kill -0 $SP 2>/dev/null; do
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt 3 ]; then echo "$label WATCHDOG: MemAvailable ${a}G - killing"; kill -9 $SP; fi
    sleep 2
  done ) &
WD=$!

until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $SP 2>/dev/null || { echo "$label: server died during load"; tail -25 "$LOG"; kill $WD 2>/dev/null; exit 1; }
    sleep 3
done
echo "$label: up $(date +%T); VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"
grep -E "MoE expert (cache|prefetch) (enabled|init)" "$LOG" | sed "s/^/$label LOG /"

python3 "$OUT/probe.py" "$label" "$pf"
rc=$?

kill $WD 2>/dev/null
kill $SP 2>/dev/null
for i in $(seq 20); do kill -0 $SP 2>/dev/null || break; sleep 1; done
kill -9 $SP 2>/dev/null
echo "$label: done $(date +%T) rc=$rc"
exit $rc
