#!/bin/bash
# bench.sh <label> <extra llama-server args...>
L=$1; shift
BIN=/home/mal/AI/llama.cpp-master-rebase/build/bin/llama-server
M=/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf
LOG=/tmp/claude-1000/dsab/$L.log
$BIN -m $M --port 8140 -c 32768 -fa on -t 8 -np 1 --jinja -ctk q4_0 -ctv q4_0 --no-kv-offload --load-mode none "$@" > $LOG 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 3 ]; then echo "$L WATCHDOG: MemAvailable ${a}G - killing" ; kill -9 $SP; fi; sleep 2; done ) &
until curl -sf http://127.0.0.1:8140/health >/dev/null; do kill -0 $SP 2>/dev/null || { echo "$L: server died"; grep -iE "error|fail" $LOG | tail -5; exit 1; }; sleep 3; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/$L VRAM /"
python3 - "$L" <<'PY'
import json,sys,urllib.request
L=sys.argv[1]
prompts=["Write a Python class implementing an LRU cache with get/put and O(1) operations, with docstrings and tests.",
 "Explain how a CUDA warp executes a reduction with shuffle instructions, with a code example.",
 "Write a bash script that watches a directory and rsyncs changed files to a remote host, with error handling.",
 "Refactor this into idiomatic Rust: a function that parses a CSV line with quoted fields into a Vec<String>."]
def run(p,n):
    body={"messages":[{"role":"user","content":p}],"max_tokens":n,"temperature":0.6,"chat_template_kwargs":{"thinking":False},"reasoning_format":"none"}
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8140/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1200))
    t=r.get("timings",{}); return t.get("predicted_per_second",0), t.get("predicted_n",0), r["choices"][0]["message"].get("content","")[:80]
run("Say hi.",16)  # warmup
xs=[]
for p in prompts:
    tps,n,txt=run(p,320); xs.append(tps); print(f"{L} {tps:6.2f} tok/s ({n} tok) :: {txt!r}",flush=True)
print(f"{L} DECODE MEAN {sum(xs)/len(xs):.2f} tok/s")
doc=open('/home/mal/AI/Strata/serve/server.py').read()[:30000]
body={"messages":[{"role":"user","content":doc+"\n\nSummarise this file in one line."}],"max_tokens":8,"chat_template_kwargs":{"thinking":False}}
r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8140/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1800))
t=r["timings"]; print(f"{L} PREFILL {t['prompt_n']} tok at {t['prompt_per_second']:.1f} tok/s")
PY
kill $SP; for i in $(seq 20); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 3000 ]; do sleep 2; done
