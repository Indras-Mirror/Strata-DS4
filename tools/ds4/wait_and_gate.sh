#!/usr/bin/env bash
# wait_and_gate.sh - wait for p0-gpu-runs to release the shared model, then run the Phase-3 gate.
#
# The packet forbids loading the 80 GB model before ~/.quetza-data/conductor/.done-p0-gpu-runs exists.
# This waits for it and then runs p64 -> p600 -> p3000 through run_ref_gate.sh, logging everything.
set -u
REPO=/home/mal/AI/Strata-DS4
DONE=$HOME/.quetza-data/conductor/.done-p0-gpu-runs
LOG=$REPO/bench/ds4-2026-10-05/ref/gate.log
mkdir -p "$REPO/bench/ds4-2026-10-05/ref"
echo "[wait_and_gate] $(date -Is) waiting for $DONE" >>"$LOG"
for i in $(seq 1 240); do
    [ -e "$DONE" ] && break
    sleep 30
done
if [ ! -e "$DONE" ]; then echo "[wait_and_gate] $(date -Is) gave up: lock never released" >>"$LOG"; exit 1; fi
echo "[wait_and_gate] $(date -Is) lock released, starting" >>"$LOG"
for p in p64 p600 p3000; do
    echo "[wait_and_gate] ===== $p $(date -Is) =====" >>"$LOG"
    bash "$REPO/tools/ds4/run_ref_gate.sh" "$p" >>"$LOG" 2>&1
    echo "[wait_and_gate] $p exit=$?" >>"$LOG"
done
echo "[wait_and_gate] $(date -Is) done" >>"$LOG"
