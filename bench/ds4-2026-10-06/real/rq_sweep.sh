#!/bin/bash
cd ~/AI/Strata-DS4; R=bench/ds4-2026-10-06/real
for cfg in "q6_k 2350" "q5_k 2450"; do set -- $cfg
  bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf \
    --ids-file bench/ds4-2026-10-05/goldens/p600/tokens.i32 -n 128 --slots $2 --arena-gib 70 --pcie 0.55 --ppl \
    --dense-requant $1 --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin > $R/rq-$1.ids 2> $R/rq-$1.log
  echo "$1 slots $2 rc=$? | $(grep ppl: $R/rq-$1.log) | $(grep '^decode' $R/rq-$1.log) | $(grep experts/ $R/rq-$1.log | cut -c1-30)"
done
