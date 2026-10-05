#!/usr/bin/env bash
# p4b_prefetch.sh - Phase 4b step 2: predicted prefetch during the dense work, measured in the replay (FINDINGS s12).
# Tokens = the held-out second half of the hidden_probe dump (its own routes), cache seeded from ds4routes.bin's train
# half, predictions from tools/ds4/sim_prefetch.py --export. Real layer order (--serial) and the wall-clock gap.
# Every run under memguard.sh.
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GGUF="${DS4_GGUF:-/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf}"
HP="$REPO/bench/ds4-2026-10-06/hidden-probe"
PROFILE="$REPO/bench/ds4-2026-10-05/route-probe/ds4routes.bin"
OUT="$REPO/bench/ds4-2026-10-06/p4b-prefetch"
BIN="$REPO/build-ds4-cuda/ds4_moe_replay"
mkdir -p "$OUT"
run() {
    local name="$1"; shift
    if [ "${SKIP_DONE:-0}" = 1 ] && [ -s "$OUT/$name.csv" ]; then echo "---- $name: skipped (done)"; return; fi
    echo "---- $name: $*"
    bash "$REPO/tools/ds4/memguard.sh" 64 62 -- "$BIN" "$GGUF" --routes "$HP/routes.bin" --profile "$PROFILE" \
        --tokens 512 --offset 512 --out "$OUT/$name.csv" --pin --arena-gib 52 --threads 7 \
        --gap-ms 0.41 --gap-wall --serial --pcie-frac 0.55 "$@" 2>&1 | tee "$OUT/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "---- $name exit $rc"
    [ "$rc" -eq 0 ] || { echo "stopping on failure" >&2; exit "$rc"; }
}
run s1820-base  --slots 1820
run s1820-pf143 --slots 1820 --pred "$HP/pred.bin" --pf-b 1.43
run s2300-base  --slots 2300
run s2300-pf143 --slots 2300 --pred "$HP/pred.bin" --pf-b 1.43
run s1820-pf2   --slots 1820 --pred "$HP/pred.bin" --pf-b 2.0
run s1820-pf3   --slots 1820 --pred "$HP/pred.bin" --pf-b 3.0
# step 3: with prefetch the CPU idles (17.7 vs GPU 44.9 busy) -> move misses off PCIe; 2150 = the real VRAM budget
# (24 GiB - 7.2 non-expert - ~0.8 scratch - ~1 ctx/KV - ~0.5 other ~= 14.5 GiB of 6.75 MiB slots)
PF=(--pred "$HP/pred.bin" --pf-b 1.43 --dither)
run s1820-pf143-p35 --slots 1820 "${PF[@]}" --pcie-frac 0.35
run s1820-pf143-p25 --slots 1820 "${PF[@]}" --pcie-frac 0.25
run s2150-base      --slots 2150
run s2150-pf143-p35 --slots 2150 "${PF[@]}" --pcie-frac 0.35
run s2150-pf143-p25 --slots 2150 "${PF[@]}" --pcie-frac 0.25
run s2150-pf143-p15 --slots 2150 "${PF[@]}" --pcie-frac 0.15
echo "== done $(date -Is) =="
