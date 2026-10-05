#!/usr/bin/env bash
# run_ref_gate.sh - run ds4_ref on a golden prompt under the shared GPU/whole-model lock and gate it.
#
#   bash tools/ds4/run_ref_gate.sh p64
#
# Loads the whole 80 GiB model (mmap + full pass), so it obeys the shared-resource protocol: it refuses to
# start below 78 GB of available RAM and wraps the run in flock on the common ds4 lock.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF="${DS4_GGUF:-/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf}"
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"
PROMPT="${1:-p64}"
BIN="$REPO/build-ds4/ds4_ref"
GOLD="$REPO/bench/ds4-2026-10-05/goldens/$PROMPT"
OUT="$REPO/bench/ds4-2026-10-05/ref/$PROMPT"

for f in "$BIN" "$GGUF" "$GOLD/tokens.i32"; do
    [ -e "$f" ] || { echo "missing $f" >&2; exit 2; }
done

avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 78 ]; then
    echo "refusing to load the model: only ${avail} GB available (< 78)" >&2
    exit 3
fi

mkdir -p "$(dirname "$OUT")"
echo "== ds4_ref $PROMPT (avail ${avail} GB) =="
flock "$LOCK" "$BIN" -m "$GGUF" --tokens "$GOLD/tokens.i32" --out "$OUT" -t 8

echo "== compare $PROMPT =="
python3 "$REPO/tools/ds4/compare_golden.py" "$GOLD" "$OUT" \
    --cosine 0.9999 --top1 0.99 --kl 0.01 --tensors-diag
