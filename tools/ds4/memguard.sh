#!/usr/bin/env bash
# memguard.sh - run one big ds4 process so it can never freeze the desktop.
#
#   bash tools/ds4/memguard.sh <max_gib> <need_avail_gib> -- <cmd...>
#
# Why: on 2026-10-06 a CPU-only full-model oracle (~81 GB anonymous RAM) plus ComfyUI starting up pushed the box
# into swap thrash and a hard freeze (journal: minutes of "Under memory pressure", no OOM kill). So every big run:
#   1. refuses to start (unless MEMGUARD_ALLOW_COMFY=1, Mal's call for an IDLE ComfyUI) if ComfyUI (or another python main.py on :8188) is up, or MemAvailable < need_avail_gib;
#   2. runs in a systemd user scope with MemoryMax=<max_gib>G and MemorySwapMax=0, so the kernel OOM-kills THIS
#      process (exit 137) instead of swapping the desktop (verified: a 400 MB alloc under a 200M cap -> killed);
#   3. a watchdog kills the scope if system MemAvailable falls below MEMGUARD_FLOOR_GIB (default 4) - covers
#      something else growing after the start;
#   4. holds the shared ds4 GPU/whole-model lock.
set -u
[ $# -ge 4 ] && [ "$3" = "--" ] || { echo "usage: memguard.sh <max_gib> <need_avail_gib> -- cmd..." >&2; exit 2; }
MAX="$1"; NEED="$2"; shift 3
FLOOR="${MEMGUARD_FLOOR_GIB:-4}"
LOCK="${DS4_LOCK:-$HOME/.quetza-data/conductor/ds4-gpu.lock}"

avail_gib() { awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }

if [ "${MEMGUARD_ALLOW_COMFY:-0}" != 1 ] && { pgrep -f '[C]omfyUI.*main.py|[c]omfyui/main.py' >/dev/null || ss -ltn 2>/dev/null | grep -q ':8188 '; }; then
    echo "memguard: refusing - ComfyUI is running (it takes system RAM too; that combination froze the box)" >&2
    exit 3
fi
a=$(avail_gib)
if [ "$a" -lt "$NEED" ]; then
    echo "memguard: refusing - MemAvailable ${a} GiB < ${NEED} GiB needed" >&2
    exit 3
fi

# a ggml abort attaches gdb for a backtrace - on a 75 GB process that ate the last free RAM (2026-10-07, after a
# CUDA OOM); the error message is enough
export GGML_NO_BACKTRACE="${GGML_NO_BACKTRACE:-1}"
unit="ds4-guard-$$"
echo "memguard: scope $unit MemoryMax=${MAX}G swap=0 floor=${FLOOR} GiB (avail ${a} GiB)"
flock "$LOCK" systemd-run --user --scope -q --unit="$unit" -p MemoryMax="${MAX}G" -p MemorySwapMax=0 "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    if [ "$(avail_gib)" -lt "$FLOOR" ]; then
        echo "memguard: MemAvailable below ${FLOOR} GiB - killing $unit" >&2
        systemctl --user kill -s KILL "$unit.scope" 2>/dev/null
        break
    fi
    sleep 1
done
wait "$pid"
rc=$?
[ $rc -eq 137 ] && echo "memguard: process was killed (exit 137) - memory cap or watchdog" >&2
exit $rc
