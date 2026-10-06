#!/bin/bash
# cache-aware routing sweep: route-bias delta vs perplexity (p600 prompt) and decode speed (128 tokens)
cd ~/AI/Strata-DS4; R=bench/ds4-2026-10-06/real
for d in 0 0.05 0.1 0.2; do
  bash tools/ds4/memguard.sh 80 78 -- build-ds4-gpu/ds4_generate -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf \
    --ids-file bench/ds4-2026-10-05/goldens/p600/tokens.i32 -n 128 --slots 2150 --arena-gib 70 --pcie 0.55 --ppl \
    --route-bias $d --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin > $R/rb$d.ids 2> $R/rb$d.log
  echo "delta $d rc=$? | $(grep ppl: $R/rb$d.log) | $(grep '^decode' $R/rb$d.log) | $(grep experts/ $R/rb$d.log | cut -c1-40)"
done
