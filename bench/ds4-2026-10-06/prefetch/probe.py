#!/usr/bin/env python3
# probe.py <label> <prefetch> -- drives one llama-server (config B + --moe-prefetch N):
#   * exactness: greedy (temp 0), fixed seed, ~200 tokens, return_tokens -> the raw token ids
#   * speed: the 4 code prompts x 320 tokens at temp 0.6, mean predicted_per_second (the 16.34 bar)
# writes <label>.json into this directory and prints one summary line per prompt.
import json
import os
import sys
import urllib.request

LABEL = sys.argv[1]
PF    = sys.argv[2]
PORT  = 8140
BASE  = f"http://127.0.0.1:{PORT}"
OUT   = os.path.dirname(os.path.abspath(__file__))

EXACT_PROMPT = ("Write a Python class implementing an LRU cache with get/put and O(1) operations, "
                "with docstrings and tests.")
PROMPTS = [
    "Write a Python class implementing an LRU cache with get/put and O(1) operations, with docstrings and tests.",
    "Explain how a CUDA warp executes a reduction with shuffle instructions, with a code example.",
    "Write a bash script that watches a directory and rsyncs changed files to a remote host, with error handling.",
    "Refactor this into idiomatic Rust: a function that parses a CSV line with quoted fields into a Vec<String>.",
]

def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

res = {"label": LABEL, "prefetch": PF}

# --- exactness: greedy, fixed seed, raw token ids ---------------------------
body = {"prompt": EXACT_PROMPT, "n_predict": 200, "temperature": 0.0, "seed": 1234,
        "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
        "cache_prompt": False, "return_tokens": True}
r = post("/completion", body)
res["exact"] = {"tokens": r.get("tokens", []), "n": r.get("tokens_predicted"),
                "content": r.get("content", ""), "timings": r.get("timings", {})}
print(f"{LABEL} exact n={res['exact']['n']} tokens={res['exact']['tokens']}", flush=True)

# --- speed: the config-B bar (mean of the 4 code prompts, 320 tokens each) ---
post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}],
     "max_tokens": 16, "chat_template_kwargs": {"thinking": False}})  # warmup
xs = []
for p in PROMPTS:
    b = {"messages": [{"role": "user", "content": p}], "max_tokens": 320,
         "temperature": 0.6, "seed": 1234, "chat_template_kwargs": {"thinking": False},
         "reasoning_format": "none"}
    r = post("/v1/chat/completions", b)
    t = r.get("timings", {})
    tps = t.get("predicted_per_second", 0.0)
    xs.append(tps)
    print(f"{LABEL} {tps:6.2f} tok/s ({t.get('predicted_n')} tok) :: "
          f"{r['choices'][0]['message'].get('content','')[:60]!r}", flush=True)
res["speed"] = {"per_prompt": xs, "mean": sum(xs)/len(xs)}
print(f"{LABEL} DECODE MEAN {res['speed']['mean']:.2f} tok/s", flush=True)

with open(f"{OUT}/{LABEL}.json", "w") as f:
    json.dump(res, f, indent=1)
