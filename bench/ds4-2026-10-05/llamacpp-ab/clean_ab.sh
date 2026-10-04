#!/bin/bash
# wait for 120 s of continuous GPU idle (no compute apps, <2 GB used), then run B2 and A2
idle=0; end=$((SECONDS+10800))
while [ $idle -lt 120 ]; do
  [ $SECONDS -gt $end ] && { echo "GAVE UP: GPU never idle for 2 min within 3 h"; exit 1; }
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$apps" -eq 0 ] && [ "$used" -lt 2000 ]; then idle=$((idle+10)); else idle=0; fi
  sleep 10
done
echo "GPU idle since $(date +%T); load: $(cut -d' ' -f1-3 /proc/loadavg)"
cd /tmp/claude-1000/dsab
for r in "B2-cache40 -cmoe --moe-expert-cache 40" "A2-autofit"; do set -- $r; ./bench.sh "$@" 2>&1 | grep -E "$1|died|WATCHDOG"; done
D=~/AI/Strata-DS4/bench/ds4-2026-10-05/llamacpp-ab; cp B2-cache40.log A2-autofit.log $D/ 2>/dev/null; echo DONE
