#!/usr/bin/env bash
# moe_sweep.sh - the Phase-4a gate runs of tools/ds4/moe_replay, one GPU process at a time.
#
#   bash tools/ds4/moe_sweep.sh [n_tokens]
#
# Obeys the shared-resource protocol: refuses to start until the concurrent Phase-3 slice has released the box
# (~/.quetza-data/conductor/.done-ds4-phase3-gate), refuses below 45 GB of MemAvailable, and wraps every run in
# flock on the common ds4 lock.  Results land in bench/ds4-2026-10-05/moe-replay/.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF="${DS4_GGUF:-/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf}"
ROUTES="$REPO/bench/ds4-2026-10-05/route-probe/ds4routes.bin"
OUT="$REPO/bench/ds4-2026-10-05/moe-replay"
BIN="$REPO/build-ds4-cuda/ds4_moe_replay"
DONE="$HOME/.quetza-data/conductor/.done-ds4-phase3-gate"
LOCK="$HOME/.quetza-data/conductor/ds4-gpu.lock"
TOKENS="${1:-320}"
OFFSET="${2:-1024}"   # a held-out 512-token block (the profile trains on the even ones)

for f in "$BIN" "$GGUF" "$ROUTES"; do
    [ -e "$f" ] || { echo "missing $f" >&2; exit 2; }
done
if [ ! -e "$DONE" ]; then
    echo "refusing: $DONE does not exist (the Phase-3 slice still holds the box)" >&2
    exit 3
fi
avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 58 ]; then
    echo "refusing: only ${avail} GB available (< 58 needed for a 49 GiB arena + the GPU cache)" >&2
    exit 3
fi
mkdir -p "$OUT"
echo "== moe_sweep: tokens $TOKENS, avail ${avail} GB, offset ${OFFSET}, $(date -Is) =="

run() {   # run <name> <args...>
    local name="$1"; shift
    echo "---- $name: $*"
    flock "$LOCK" "$BIN" "$GGUF" --routes "$ROUTES" --tokens "$TOKENS" --offset "$OFFSET" --out "$OUT/$name.csv" "$@" \
        2>&1 | tee "$OUT/$name.log"
    echo "---- $name exit ${PIPESTATUS[0]}"
}

# 1. the engine's arithmetic on the GPU grouped path (no big arena, no cache: tensors only)
run gate-correctness --correctness 1 --correctness-experts 12 --arena-gib 0 --no-cache

# 2. the plan's default: 12 GB of slots, 55% of the misses over PCIe, pinned arena
run base --slots 1820 --pcie-frac 0.55 --threads 7 --pin --arena-gib 52

# 3. tuning points
run slots-2300   --slots 2300 --pcie-frac 0.55 --threads 7 --pin --arena-gib 52
run pcie-75      --slots 1820 --pcie-frac 0.75 --threads 7 --pin --arena-gib 52
run pcie-35      --slots 1820 --pcie-frac 0.35 --threads 7 --pin --arena-gib 52
run pcie-0       --slots 1820 --pcie-frac 0.00 --threads 7 --pin --arena-gib 52
run threads-8    --slots 1820 --pcie-frac 0.55 --threads 8 --pin --arena-gib 52

echo "== moe_sweep done $(date -Is) =="
