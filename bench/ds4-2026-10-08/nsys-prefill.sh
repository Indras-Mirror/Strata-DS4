#!/bin/bash
# nsys on the chunked prefill: s31's open question is whether the dense half (384.9 s of a 128K prefill, 15x the
# indexer's flop estimate) is the [cap,n,64] permute/cont copies or the small-K batched matmul.  This profiles a
# 32K prefill - same cap-dependent work, 1/4 the wall time - and prints the top kernels by time.
# Usage: bash bench/ds4-2026-10-08/nsys-prefill.sh [ctx_tokens]   (default 32768; use 131072 when there is time)
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08; N=${1:-32768}
case $N in 32768) C=36000;; 131072) C=135000;; *) C=$((N + 3000));; esac
mkdir -p $B/nsys
L=~/.quetza-data/conductor/ds4-gpu.lock
until flock -n "$L" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge 70 ]; do sleep 30; done
ss -ltn | grep -q 8188 && { echo "ComfyUI up - refusing"; exit 9; }
bash tools/ds4/memguard.sh 80 78 -- nsys profile --stats=true --force-overwrite=true \
    --cuda-graph-trace=node -o "$B/nsys/pf-ctx$N" \
    "${DS4_BIN:-build-ds4-gpu/ds4_generate}" -m /media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf \
    --ids-file "$B/ctx/ctx$N.i32" -n 4 --slots auto --arena-gib 70 --pcie 0.55 --dense-requant q6_k \
    --vram-lru --pf-b 0.7 --prefill-chunk 4096 --chunk-mmq --chunk-prestage --vram-margin 2.0 --ctx $C \
    > "$B/nsys/pf-ctx$N.log" 2>&1
echo "rc=$?" >> "$B/nsys/pf-ctx$N.log"
echo "=== top kernels (from the nsys stats) ==="
grep -A 22 -iE 'gpukernsum|CUDA Kernel Statistics' "$B/nsys/pf-ctx$N.log" | head -26
