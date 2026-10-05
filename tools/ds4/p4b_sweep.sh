#!/usr/bin/env bash
# p4b_sweep.sh - Phase 4b step 1: the MoE replay in the REAL layer order (--serial: experts(l) wait for layer l's
# dense GPU work, here the whole Phase-0 non-MoE time 17.6/43 = 0.41 ms/layer on the wall-clock gap kernel), with
# and without pinned per-layer copies.  `wall` is then a measured token time, harness syncs included.
# Every run goes through tools/ds4/memguard.sh (64 GiB cap, swap off, 4 GiB watchdog, ds4 lock).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF="${DS4_GGUF:-/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf}"
ROUTES="$REPO/bench/ds4-2026-10-05/route-probe/ds4routes.bin"
OUT="$REPO/bench/ds4-2026-10-06/p4b"
BIN="$REPO/build-ds4-cuda/ds4_moe_replay"
TOKENS="${TOKENS:-320}"; OFFSET="${OFFSET:-1024}"
mkdir -p "$OUT"
echo "== p4b_sweep tokens $TOKENS offset $OFFSET $(date -Is) =="
run() {
    local name="$1"; shift
    if [ "${SKIP_DONE:-0}" = 1 ] && [ -s "$OUT/$name.csv" ]; then echo "---- $name: skipped (done)"; return; fi
    echo "---- $name: $*"
    bash "$REPO/tools/ds4/memguard.sh" 64 62 -- "$BIN" "$GGUF" --routes "$ROUTES" --tokens "$TOKENS" \
        --offset "$OFFSET" --out "$OUT/$name.csv" --pin --arena-gib 52 --threads 7 "$@" 2>&1 | tee "$OUT/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "---- $name exit $rc"
    [ "$rc" -eq 0 ] || { echo "stopping sweep on failure" >&2; exit "$rc"; }
}
# reference: 4a's best arm re-run today (same flags as 4a) so the day-to-day noise is visible
run ref-4a-slots2300      --slots 2300 --pcie-frac 0.55 --gap-ms 0
# the real chain, 4a's copies
run serial                --slots 2300 --pcie-frac 0.55 --gap-ms 0.41 --gap-wall --serial
# the real chain, pinned copies
run serial-pinned         --slots 2300 --pcie-frac 0.55 --gap-ms 0.41 --gap-wall --serial --pinned-meta
# re-balance: per-token GPU 30.4 vs CPU 27.3 in 4a -> shift misses toward the CPU, and the other way
run serial-pinned-pcie50  --slots 2300 --pcie-frac 0.50 --gap-ms 0.41 --gap-wall --serial --pinned-meta
run serial-pinned-pcie60  --slots 2300 --pcie-frac 0.60 --gap-ms 0.41 --gap-wall --serial --pinned-meta
# step 2 (after the first five): honour the fraction with error diffusion; balance point from the serial runs
# rates (CPU 13.9 GB/s, PCIe 17.1 GB/s + 1.7 ms hits) is a PCIe share of ~0.53 of the misses.
# SKIP_DONE=1 skips arms whose csv already exists.
run dither-pcie53  --slots 2300 --pcie-frac 0.53 --gap-ms 0.41 --gap-wall --serial --dither
run dither-pcie47  --slots 2300 --pcie-frac 0.47 --gap-ms 0.41 --gap-wall --serial --dither
run dither-pcie53-slots1820  --slots 1820 --pcie-frac 0.53 --gap-ms 0.41 --gap-wall --serial --dither
echo "== p4b_sweep done $(date -Is) =="
