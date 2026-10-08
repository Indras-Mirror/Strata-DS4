#!/bin/bash
# b20: real-model A/B of the device-side visibility mask (DS4_VIS_DEV=1) against the host mask (DS4_VIS_DEV=0),
# interleaved (single runs swing ~1 tok/s), p600 + 128 greedy, the recommended decode config.  Binary snapshotted,
# so a rebuild cannot change a queued run.  The wakeword daemon is stopped for the measurements and restarted after.
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08
export DS4_BIN=$B/bin/ds4_generate.visdev
F="--vram-lru --pf-b 0.7 --f16-q8 --arena-skip-resident --arena-gib 58"
systemctl --user stop wakeword-daemon 2>/dev/null
trap 'systemctl --user start wakeword-daemon 2>/dev/null' EXIT
for rep in 1 2; do
    DS4_VIS_DEV=1 $B/q.sh $B/b20-visdev1-$rep.log $F
    DS4_VIS_DEV=0 $B/q.sh $B/b20-visdev0-$rep.log $F
done
echo "=== b20 summary ==="
for f in $B/b20-visdev1-1.log $B/b20-visdev0-1.log $B/b20-visdev1-2.log $B/b20-visdev0-2.log; do
    printf '%-24s %s | %s | %s\n' "$(basename "$f")" \
        "$(grep -h '^decode :' "$f" | sed 's/.*= //')" \
        "$(grep -hiE 'ppl|perplexity' "$f" | head -1 | sed 's/^ *//')" \
        "$(grep -h 'VRAM free at the end' "$f")"
done
