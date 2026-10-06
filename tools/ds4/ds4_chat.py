#!/usr/bin/env python3
"""ds4_chat.py - text in, text out for the Strata DS4 engine (tools/ds4/ds4_generate does ids in, ids out).

  # plain continuation (the coherence check), greedy:
  python3 tools/ds4/ds4_chat.py --raw "def fibonacci(n):" -n 64
  # chat, rendered with the model's own Jinja template (llama.cpp's DeepSeek-V4-Flash-0731 template):
  python3 tools/ds4/ds4_chat.py "Explain RoPE in two sentences." -n 128
  # compare against llama.cpp on the same ids (greedy, identical prompt ids):
  python3 tools/ds4/ds4_chat.py --raw "..." -n 64 --print-ids

Loads the real model, so the engine is started through tools/ds4/memguard.sh (cap/need adjustable). Extra
arguments after `--` go to ds4_generate (e.g. -- --experts cpu --slots 1820).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

MODEL = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf"
TEMPLATE = Path.home() / "AI/llama.cpp-master-rebase/models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja"


def safe_decode(tok, ids):
    """decode, passing special tokens (not byte-level, e.g. <｜end▁of▁sentence｜>) through as their literal text"""
    out, run = [], []
    for i in ids:
        try:
            tok.token_bytes(i)
            run.append(i)
        except KeyError:
            if run:
                out.append(tok.decode(run)); run = []
            out.append(tok.tokens[i])
    if run:
        out.append(tok.decode(run))
    return "".join(out)


def render_chat(tok, prompt, thinking):
    import jinja2
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tpl = env.from_string(TEMPLATE.read_text())
    bos_id = tok.special_ids.get("tokenizer.ggml.bos_token_id")
    bos = tok.tokens[bos_id] if bos_id is not None else ""
    return tpl.render(messages=[{"role": "user", "content": prompt}], add_generation_prompt=True,
                      thinking=thinking, bos_token=bos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--raw", action="store_true", help="plain continuation, no chat template (BOS + text)")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--bin", default=str(REPO / "build-ds4-gpu/ds4_generate"))
    ap.add_argument("--cap", default="80")
    ap.add_argument("--need", default="78")
    ap.add_argument("--print-ids", action="store_true")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    tok = Tokenizer.from_gguf(a.model)
    bos_id = tok.special_ids.get("tokenizer.ggml.bos_token_id")
    eos_id = tok.special_ids.get("tokenizer.ggml.eos_token_id")
    if a.raw:
        ids = ([bos_id] if bos_id is not None else []) + tok.encode(a.prompt)
    else:
        ids = tok.encode(render_chat(tok, a.prompt, a.thinking), parse_special=True)
    if a.print_ids:
        print("prompt ids:", ",".join(map(str, ids)), file=sys.stderr)
    stop = [i for i in (eos_id,) if i is not None]
    rest = [x for x in a.rest if x != "--"]
    cmd = ["bash", str(REPO / "tools/ds4/memguard.sh"), a.cap, a.need, "--", a.bin, "-m", a.model,
           "--ids", ",".join(map(str, ids)), "-n", str(a.n), "--temp", str(a.temp)]
    if stop:
        cmd += ["--stop", ",".join(map(str, stop))]
    cmd += rest
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    out_ids = []
    printed = ""
    for line in proc.stdout:
        line = line.strip()
        if not line.lstrip("-").isdigit():
            continue
        out_ids.append(int(line))
        text = safe_decode(tok, out_ids)
        sys.stdout.write(text[len(printed):])   # stream, re-decoding so multi-byte UTF-8 joins up
        sys.stdout.flush()
        printed = text
    rc = proc.wait()
    print()
    if a.print_ids:
        print("generated ids:", ",".join(map(str, out_ids)), file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
