#!/usr/bin/env bash
# make_goldens.sh - run tools/ds4/golden_dump for the three fixed prompts.
#
# NOT run as part of phase 0: it needs the 80 GB model and the GPU. Run it on a
# quiet machine (Mal must not be running anything else; the model needs ~75 GB
# RAM). See docs/ds4/PORT_PLAN.md phase 0.
#
#   tools/ds4/make_goldens.sh              # p64, p600, p3000 into bench/ds4-2026-10-05/goldens/
#   GOLDENS_OUT=/data/goldens tools/ds4/make_goldens.sh --all-pos
#
# Extra arguments are passed straight to golden_dump (e.g. --all-pos).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DUMP=${GOLDEN_DUMP_BIN:-$ROOT/tools/ds4/golden_dump}
MODEL=${DS4_MODEL:-/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf}
OUT=${GOLDENS_OUT:-$ROOT/bench/ds4-2026-10-05/goldens}

# Config B from bench/ds4-2026-10-05/llamacpp-ab/bench.sh, with F16 KV so the
# dump matches the parity setup in PORT_PLAN.md s3 (a quantised K turns on the
# Hadamard rotation and a first port should not have to match it).
FLAGS=(-ngl 99 -cmoe --moe-expert-cache 40 --load-mode none -ctk f16 -ctv f16 -fa on -t 8)

if [ ! -x "$DUMP" ]; then
    echo "make_goldens: $DUMP not found - build tools/ds4/golden_dump.cpp first" >&2
    echo "  (build command is in the file header)" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "make_goldens: model not found: $MODEL" >&2
    exit 1
fi

mkdir -p "$OUT"
for name in p64 p600 p3000; do
    prompt="$ROOT/tools/ds4/golden_prompts/$name.txt"
    echo "=== $name -> $OUT/$name ==="
    "$DUMP" -m "$MODEL" -f "$prompt" --out "$OUT/$name" "${FLAGS[@]}" "$@" 2>&1 | tee "$OUT/$name.log"
done

echo "goldens written under $OUT"
