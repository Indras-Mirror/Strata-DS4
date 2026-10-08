#!/bin/bash
# batch 7: nsys of the MMQ chunk path (p3000, no ppl); speed on a 5996-token prompt (p3000 twice), chunk 4096, no ppl
cd /home/mal/AI/Strata-DS4
B=bench/ds4-2026-10-08; BIN=$B/bin/ds4_generate.argsort
L=~/.quetza-data/conductor/ds4-gpu.lock
until flock -n "$L" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge 70 ]; do sleep 30; done
bash tools/ds4/memguard.sh 80 78 -- /usr/local/cuda/bin/nsys profile -o $B/nsys/pf-mmq-argsort --force-overwrite true \
  --trace cuda --sample none --cpuctxsw none $BIN -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf \
  --ids-file bench/ds4-2026-10-05/goldens/p3000/tokens.i32 -n 4 --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k \
  --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin --vram-lru --prefill-chunk 4096 --chunk-mmq > $B/nsys/pf-mmq-argsort.log 2>&1
export DS4_BIN=$BIN Q_PROMPT=$B/p6000-concat.i32 Q_N=128 Q_PPL=
$B/q.sh $B/b7-p6000-mmq.log --vram-lru --prefill-chunk 4096 --chunk-mmq
echo batch7-done
