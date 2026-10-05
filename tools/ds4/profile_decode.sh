#!/usr/bin/env bash
# profile_decode.sh - profile llama.cpp config B (all experts host-resident +
# hot-expert cache) to split one decode token into GPU attention, GPU dense,
# PCIe/copies, and the host-side residual (CPU expert matmuls + sync/idle).
#
# NOT run as part of phase 0. Needs the 80 GB model, the GPU and ~75 GB RAM.
# Exact server flags and the RAM watchdog follow
# bench/ds4-2026-10-05/llamacpp-ab/bench.sh. See docs/ds4/PORT_PLAN.md phase 0.
#
#   tools/ds4/profile_decode.sh
#   PROMPT="..." DECODE_TOKENS=128 tools/ds4/profile_decode.sh
#
# Uses nsys when present, else perf, else reports that no profiler is available.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${LLAMA_SERVER:-/home/mal/AI/llama.cpp-master-rebase/build/bin/llama-server}
MODEL=${DS4_MODEL:-/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf}
OUT=${OUTDIR:-$ROOT/bench/ds4-2026-10-05/profile-decode}
PORT=${PORT:-8142}
DECODE_TOKENS=${DECODE_TOKENS:-256}
PROMPT=${PROMPT:-"Write a Python class implementing an LRU cache with O(1) get and put, with tests."}
PARSER="$ROOT/tools/ds4/parse_profile.py"

# Config B, copied from bench/ds4-2026-10-05/llamacpp-ab/bench.sh.
B_FLAGS=(-c 32768 -fa on -t 8 -np 1 --jinja -ctk q4_0 -ctv q4_0 --no-kv-offload \
         --load-mode none -cmoe --moe-expert-cache 40)

mkdir -p "$OUT"

watchdog_start() {
    local pid=$1
    # NB: the subshell's stdout must be detached, otherwise `wd=$(watchdog_start ...)`
    # never sees EOF (the background loop holds the command-substitution pipe open).
    ( while kill -0 "$pid" 2>/dev/null; do
          a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
          if [ "$a" -lt 3 ]; then
              echo "WATCHDOG: MemAvailable ${a}G - killing $pid"
              kill -9 "$pid" 2>/dev/null
          fi
          sleep 2
      done ) >/dev/null 2>&1 &
    echo $!
}

wait_health() {
    local tries=0
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
        tries=$((tries + 1))
        if [ "$tries" -gt 600 ]; then
            echo "profile_decode: server never became healthy" >&2
            return 1
        fi
        sleep 3
    done
}

# Drive one decode-heavy request and record the server's timings object.
drive() {
    python3 - "$PORT" "$DECODE_TOKENS" "$PROMPT" "$OUT/timings.json" <<'PY'
import json, sys, urllib.request
port, n, prompt, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]

def post(body):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
      "chat_template_kwargs": {"thinking": False}})
r = post({"messages": [{"role": "user", "content": prompt}], "max_tokens": n,
          "temperature": 0.6, "chat_template_kwargs": {"thinking": False},
          "reasoning_format": "none"})
timings = r.get("timings", {})
with open(out, "w") as f:
    json.dump(timings, f, indent=2)
print("timings:", timings)
PY
}

run_under_nsys() {
    local rep="$OUT/nsys-B"
    echo "[profile] nsys capture -> $rep.nsys-rep"
    # --cuda-graph-trace=node: llama.cpp replays a per-token CUDA graph; without this
    # nsys reports the whole graph as one activity and the kernel summary misses every
    # decode kernel (it only sees the un-graphed warmup/prefill work).
    nsys profile --force-overwrite=true -o "$rep" -t cuda,nvtx,osrt --cuda-graph-trace=node \
        "$BIN" -m "$MODEL" --port "$PORT" "${B_FLAGS[@]}" >"$OUT/server.log" 2>&1 &
    local prof=$!
    local wd; wd=$(watchdog_start "$prof")
    wait_health
    drive
    kill -INT "$prof" 2>/dev/null || true
    wait "$prof" 2>/dev/null || true
    kill "$wd" 2>/dev/null || true

    if [ ! -f "$rep.nsys-rep" ]; then
        echo "profile_decode: nsys produced no report; see $OUT/server.log" >&2
        return 1
    fi
    nsys stats --force-export=true --report cuda_gpu_kern_sum   --format csv "$rep.nsys-rep" >"$OUT/kern_sum.csv" 2>/dev/null || true
    nsys stats --force-export=true --report cuda_gpu_mem_time_sum --format csv "$rep.nsys-rep" >"$OUT/mem_time.csv" 2>/dev/null || true
    # prefer the sqlite: it can cut the exact decode window, the CSV path can only
    # average the whole process lifetime (model load + warmup included).
    if [ -f "$rep.sqlite" ]; then
        python3 "$PARSER" --nsys-sqlite "$rep.sqlite" \
            --timings "$OUT/timings.json" --decode-tokens "$DECODE_TOKENS"
    else
        python3 "$PARSER" --kern-csv "$OUT/kern_sum.csv" --mem-csv "$OUT/mem_time.csv" \
            --timings "$OUT/timings.json" --decode-tokens "$DECODE_TOKENS"
    fi
}

run_under_perf() {
    echo "[profile] perf fallback (CPU only; no GPU trace)"
    "$BIN" -m "$MODEL" --port "$PORT" "${B_FLAGS[@]}" >"$OUT/server.log" 2>&1 &
    local pid=$!
    local wd; wd=$(watchdog_start "$pid")
    wait_health

    perf record -o "$OUT/perf.data" -F 999 -g -p "$pid" >"$OUT/perf.log" 2>&1 &
    local perf_pid=$!
    sleep 1
    drive
    kill -INT "$perf_pid" 2>/dev/null || true
    wait "$perf_pid" 2>/dev/null || true
    kill -INT "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    kill "$wd" 2>/dev/null || true

    perf report --stdio -i "$OUT/perf.data" --sort symbol >"$OUT/perf.txt" 2>/dev/null || true
    python3 "$PARSER" --perf-text "$OUT/perf.txt" --timings "$OUT/timings.json" \
        --decode-tokens "$DECODE_TOKENS"
}

if [ ! -x "$BIN" ]; then
    echo "profile_decode: server not found: $BIN" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "profile_decode: model not found: $MODEL" >&2
    exit 1
fi

if command -v nsys >/dev/null 2>&1; then
    run_under_nsys
elif command -v perf >/dev/null 2>&1; then
    run_under_perf
else
    echo "profile_decode: neither nsys nor perf is installed." >&2
    echo "  Run config B with GGML_SCHED_DEBUG=1 for a scheduler dump, or install nsys." >&2
    exit 1
fi
