#!/usr/bin/env bash
# run the remaining goldens one at a time (one model process at a time)
set -uo pipefail
cd /home/mal/AI/Strata-DS4
DUMP=tools/ds4/golden_dump
MODEL=/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf
OUT=bench/ds4-2026-10-05/goldens
FLAGS=(-ngl 99 -cmoe --moe-expert-cache 40 --load-mode none -ctk f16 -ctv f16 -fa on -t 8)

wait_ready() {
  for i in $(seq 20); do
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$g" -lt 2048 ] && [ "$a" -ge 78 ]; then echo "READY gpu=${g}MiB avail=${a}G"; return 0; fi
    echo "WAIT gpu=${g}MiB avail=${a}G"; sleep 60
  done
  echo "NOT READY after 20min"; return 1
}

run_one() {  # $1=name $2=outdir
  wait_ready || exit 1
  echo "=== $1 -> $2  $(date -Is) ==="
  rm -rf "$2"
  "$DUMP" -m "$MODEL" -f "tools/ds4/golden_prompts/$1.txt" --out "$2" "${FLAGS[@]}" \
      > "bench/ds4-2026-10-05/phase0/$1.run.log" 2>&1
  echo "EXIT $1 = $?"
}

run_one p64   "$OUT/p64-run2"
run_one p600  "$OUT/p600"
run_one p3000 "$OUT/p3000"
echo "ALL DONE $(date -Is)"
