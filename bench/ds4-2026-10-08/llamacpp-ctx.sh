#!/bin/bash
# b22: llama.cpp at 64K - the comparison number for the DS4 long-context decode (FINDINGS s28/s31).
# Uses llama-server + one raw /completion request, which is how the 16.34 tok/s bar was measured (s3) - and it
# avoids this build's llama-cli, which spins in its interactive loop when stdin is EOF (234M "> " lines, 671 MB log,
# 0% of the prefill done; s32).
cd /home/mal/AI/Strata-DS4 || exit 1
B=bench/ds4-2026-10-08; LOG=$B/b22-llamacpp-ctx65536.log
BIN=/home/mal/AI/llama.cpp-master-rebase/build-cu134/bin/llama-server
M=/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf
L=~/.quetza-data/conductor/ds4-gpu.lock
until flock -n "$L" true && [ "$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)" -ge 70 ]; do sleep 30; done
ss -ltn | grep -q 8188 && { echo "ComfyUI up - refusing" > "$LOG"; exit 9; }
python3 -c 'import json,sys; print(json.dumps({"prompt":open(sys.argv[1],encoding="utf-8",errors="ignore").read(),"n_predict":128,"temperature":0.0,"top_k":1,"cache_prompt":False,"stream":False}))' \
    "$B/ctx/ctx65536.txt" > /tmp/b22-req.json || exit 1
bash tools/ds4/memguard.sh 80 78 -- "$BIN" -m "$M" --port 8140 -c 70000 -fa on -t 8 -np 1 \
    -ngl 99 -cmoe --moe-expert-cache 40 --load-mode none -ctk q4_0 -ctv q4_0 --no-kv-offload > "$LOG" 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -lt 3 ] && kill -9 $SP; sleep 5; done ) &
for i in $(seq 120); do curl -sf http://127.0.0.1:8140/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || { echo "server died during load" >> "$LOG"; exit 1; }; sleep 5; done
echo "server up after $((i*5)) s" >> "$LOG"
curl -s http://127.0.0.1:8140/completion -H 'Content-Type: application/json' --data-binary @/tmp/b22-req.json > /tmp/b22-resp.json
python3 - <<'PY' | tee -a "$LOG"
import json
r = json.load(open('/tmp/b22-resp.json'))
t = r.get('timings', {})
print(f"llama.cpp 64K: prompt {t.get('prompt_n')} tok at {t.get('prompt_per_second',0):.1f} tok/s | "
      f"decode {t.get('predicted_n')} tok at {t.get('predicted_per_second',0):.2f} tok/s")
PY
kill $SP 2>/dev/null; for i in $(seq 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -9 $SP 2>/dev/null
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 3000 ]; do sleep 3; done
