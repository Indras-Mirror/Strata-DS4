#!/bin/bash
# queue one full-model run behind the shared GPU lock: wait for the lock AND 70 GiB available, then memguard
# usage: q.sh <log> <extra ds4_generate args...>   (env vars pass through)
cd /home/mal/AI/Strata-DS4
LOG=$1; shift
L=~/.quetza-data/conductor/ds4-gpu.lock
until flock -n "$L" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge 70 ]; do sleep 30; done
if ss -ltn | grep -q 8188; then echo "ComfyUI up - refusing" > "$LOG"; exit 9; fi
bash tools/ds4/memguard.sh 80 78 -- ${DS4_BIN:-build-ds4-gpu/ds4_generate} -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf \
  --ids-file bench/ds4-2026-10-05/goldens/p600/tokens.i32 -n 128 --slots auto --arena-gib 70 --pcie 0.55 --ppl \
  --dense-requant q6_k --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin "$@" > "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
