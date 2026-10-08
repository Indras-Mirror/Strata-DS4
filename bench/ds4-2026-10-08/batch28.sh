#!/bin/bash
# b28 (~12 min): dec2 (pre-cuts) vs dec4 (op cuts + K==V single cast + CUDA FWHT hint + 3-op vis mask), the standard flag set
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08
F="--vram-lru --pf-b 0.7 --f16-q8 --arena-skip-resident --arena-gib 58 --comp-type iq4_nl --icomp-q8"
for v in dec2 dec4; do DS4_BIN=$B/bin/ds4_generate.$v $B/q.sh $B/b28-$v.log $F; done
# nsys: is the dense half's attention+router kernel time or launch/sync gaps? (16-token prompt, 48 tokens, no ppl)
Q_PROMPT=$B/ctx/probe16.i32 Q_N=48 Q_PPL= DS4_BIN="/usr/local/cuda/bin/nsys profile --cuda-graph-trace=node -t cuda,nvtx -o $B/b28-nsys -f true $B/bin/ds4_generate.dec4" \
  $B/q.sh $B/b28-nsys.log $F
/usr/local/cuda/bin/nsys stats -r cuda_gpu_kern_sum,cuda_api_sum --format csv -o $B/b28-nsys $B/b28-nsys.nsys-rep > /dev/null 2>&1
for v in dec2 dec4; do printf '%s: ' $v; grep -h '^decode :\|^ppl:' $B/b28-$v.log | sed 's/.*= //;s/.*perplexity /ppl /' | tr '\n' ' '; grep -h 'decode ms/token' $B/b28-$v.log; done
