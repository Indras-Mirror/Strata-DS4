#!/bin/bash
# dspark_bench.sh <label> <extra llama-server args...>
# Same 4 prompts x 320 tokens as llamacpp-ab/bench.sh, plus the full timings
# object (draft acceptance) per prompt. RAM watchdog as in bench.sh.
L=$1; shift
BIN=/home/mal/AI/llama.cpp-master-rebase/build/bin/llama-server
M=/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf
OUT=/home/mal/AI/Strata-DS4/bench/ds4-2026-10-05/phase0
LOG=$OUT/$L.log
PORT=${PORT:-8143}
if [ "${BASE:-B}" = "A" ]; then
    BASEARGS=()                      # config A: automatic placement, frees host RAM
else
    BASEARGS=(-cmoe --moe-expert-cache 40)
fi
$BIN -m $M --port $PORT -c 32768 -fa on -t 8 -np 1 --jinja -ctk q4_0 -ctv q4_0 --no-kv-offload --load-mode none \
     "${BASEARGS[@]}" "$@" > $LOG 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); if [ "$a" -lt 3 ]; then echo "$L WATCHDOG: MemAvailable ${a}G - killing"; kill -9 $SP; fi; sleep 2; done ) >/dev/null 2>&1 &
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do kill -0 $SP 2>/dev/null || { echo "$L: server died"; grep -iE "error|fail|assert" $LOG | tail -8; exit 1; }; sleep 3; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed "s/^/$L VRAM /"
python3 - "$L" "$PORT" "$OUT" "$SP" <<'PY'
import json,sys,urllib.request
L,port,out,pid=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
def cpu_ticks():
    f=open(f"/proc/{pid}/stat").read().rsplit(")",1)[1].split()
    return int(f[11])+int(f[12])   # utime+stime (fields 14,15), MIN_NUM after comm
HZ=100.0
prompts=["Write a Python class implementing an LRU cache with get/put and O(1) operations, with docstrings and tests.",
 "Explain how a CUDA warp executes a reduction with shuffle instructions, with a code example.",
 "Write a bash script that watches a directory and rsyncs changed files to a remote host, with error handling.",
 "Refactor this into idiomatic Rust: a function that parses a CSV line with quoted fields into a Vec<String>."]
res={"label":L,"prompts":[]}
def run(p,n,collect=True):
    body={"messages":[{"role":"user","content":p}],"max_tokens":n,"temperature":0.6,"chat_template_kwargs":{"thinking":False},"reasoning_format":"none"}
    r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1800))
    return r
run("Say hi.",16,False)  # warmup
xs=[]
for p in prompts:
    c0=cpu_ticks()
    r=run(p,320)
    c1=cpu_ticks()
    t=r.get("timings",{}); xs.append(t.get("predicted_per_second",0))
    d={k:t[k] for k in ("predicted_n","predicted_per_second","prompt_n","prompt_per_second","draft_n","draft_n_accepted") if k in t}
    if d.get("predicted_n"): d["cpu_ms_per_tok"]=(c1-c0)/HZ*1000.0/d["predicted_n"]
    res["prompts"].append(d)
    print(f"{L} {t.get('predicted_per_second',0):6.2f} tok/s ({t.get('predicted_n')} tok) "
          f"cpu={d.get('cpu_ms_per_tok',0):5.1f} ms/tok draft={d.get('draft_n')} acc={d.get('draft_n_accepted')} "
          f":: {r['choices'][0]['message'].get('content','')[:50]!r}",flush=True)
print(f"{L} DECODE MEAN {sum(xs)/len(xs):.2f} tok/s")
doc=open('/home/mal/AI/Strata/serve/server.py').read()[:30000]
r=run(doc+"\n\nSummarise this file in one line.",8)
t=r["timings"]; res["prefill"]=t
print(f"{L} PREFILL {t['prompt_n']} tok at {t['prompt_per_second']:.1f} tok/s")
json.dump(res,open(f"{out}/{L}.json","w"),indent=2)
PY
kill $SP; for i in $(seq 20); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -lt 3000 ]; do sleep 2; done
