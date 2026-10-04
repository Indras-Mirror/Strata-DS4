#!/usr/bin/env python3
"""tools/ds4/make_tokenizer_goldens.py - freeze llama.cpp's token ids for the DSV4 tokenizer gate.

The oracle is llama.cpp-master-rebase's `llama-tokenize`.  It loads ONLY the vocab (381 MB RSS, ~0.7 s on this
PC; the 80 GB of weights are never touched), so this runs while the GPU is busy.  The binary links CUDA 12.9
runtime libs that are no longer installed system-wide, so the two pip nvidia packages are put on
LD_LIBRARY_PATH; tokenization never initialises CUDA.

Output: tests/ds4/tokenizer_golden.json - the exact ids for the three golden prompts plus a diverse corpus, and
tests/ds4/corpus.txt - the corpus split into lines, so the tokenizer test can check line by line as well as in
one blob (a line boundary is where a pre-tokenizer mistake shows up).

Run:  python tools/ds4/make_tokenizer_goldens.py [--model PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_MODEL = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf"
DEFAULT_ORACLE = pathlib.Path.home() / "AI/llama.cpp-master-rebase/build/bin/llama-tokenize"
# the pip CUDA 12 runtime + cublas the master-rebase binary was linked against (system /usr/local/cuda-12.9 is gone)
CUDA12_LIBS = [
    "/home/mal/anaconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
    "/home/mal/anaconda3/lib/python3.12/site-packages/nvidia/cublas/lib",
]

# A corpus chosen to hit the pre-tokenizer's boundaries: multi-byte UTF-8, emoji, combining marks, a Han run, a
# kana run, digit runs of length 1..10 (the `\p{N}{1,3}` pre-split), contractions, punctuation runs glued to
# letters (the `[punct][A-Za-z]+` alternative), leading spaces, CRLF, tabs, and control bytes.
CORPUS = [
    "Hello, world!",
    "  leading and trailing  ",
    "a\n\n\nb",
    "def f(x):\n\treturn x  # comment\n",
    "\u4f60\u597d\uff0c\u4e16\u754c",
    "\u0645\u0631\u062d\u0628\u0627",
    "\U0001f600\U0001f680\U0001f1fa\U0001f1f8",
    "e\u0301\u0301 combining",
    "\x00\x01\x7f control",
    "\u00a0non-breaking\u00a0space",
    "1234567890",
    "0 12 345 6789 01234 567890",
    "MixedCASE_and-dashes",
    "\r\n\r\n",
    "\u2028\u2029",
    "x" * 5000,
    "it's they're we've I'm you'll he'd",
    "don't isn't won't can't",
    "24/7, 3.14159; a-b_c? (yes!) [no] {maybe} <tag>",
    "\u3042\u3044\u3046\u3048\u304a\u30a2\u30a4\u30a6\u30a8\u30aa",
    "\u4e00\u4e8c\u4e09\u56db\u4e94 \u6f22\u5b57\u30c6\u30b9\u30c8",
    "mixed \u6f22\u5b57abc123def\u3042\u3044",
    "emoji \U0001f4a9\U0001f44d\U0001f3fd between words",
    "\u00e9\u00e8\u00ea\u00eb r\u00e9sum\u00e9 na\u00efve",
    "tabs\tand\tspaces   runs",
    "Punct!At#Start$Of%Word^Runs*various",
    "http://example.com/path?query=1&x=2#frag",
    "The quick brown fox jumps over the lazy dog.",
    "1,234,567.89 USD",
    "\uff08\uff09\u3001\u3002\uff01\uff1f",
    "  " * 3 + "deep_indent",
    "line with a trailing space \nnext",
    "\U0001d400\U0001d401\U0001d402 math alphanumerics",
]


def tokenize(model: str, oracle: pathlib.Path, text_file: pathlib.Path) -> list[int]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(CUDA12_LIBS + [env.get("LD_LIBRARY_PATH", "")])
    p = subprocess.run([str(oracle), "-m", model, "-f", str(text_file), "--ids", "--log-disable"],
                       capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit("llama-tokenize failed (%d):\n%s" % (p.returncode, p.stderr[-2000:]))
    line = [ln for ln in p.stdout.splitlines() if ln.strip().startswith("[")]
    if not line:
        sys.exit("llama-tokenize printed no id list:\n%s" % p.stdout[-2000:])
    return json.loads(line[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    ap.add_argument("--out", default=str(REPO / "tests/ds4/tokenizer_golden.json"))
    args = ap.parse_args()
    oracle = pathlib.Path(args.oracle)
    if not oracle.exists():
        sys.exit("oracle not found: %s" % oracle)
    if not pathlib.Path(args.model).exists():
        sys.exit("model not found: %s" % args.model)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    prompts = {}
    for name in ("p64", "p600", "p3000"):
        f = REPO / "tools/ds4/golden_prompts" / (name + ".txt")
        prompts[name] = tokenize(args.model, oracle, f)

    # corpus: the text is written once, tokenized as one blob and line by line (skipping an empty last line).
    corpus = REPO / "tests/ds4/corpus.txt"
    corpus.write_text("\n".join(CORPUS) + "\n", encoding="utf-8")
    corpus_ids = tokenize(args.model, oracle, corpus)
    per_line = [tokenize(args.model, oracle, _line_file(REPO, i, s)) for i, s in enumerate(CORPUS)]

    doc = {
        "source": str(oracle),
        "model": args.model,
        "note": "ids from llama.cpp-master-rebase llama-tokenize --ids (vocab-only load); oracle for the "
                "DSV4 tokenizer parity gate",
        "prompts": prompts,
        "corpus": CORPUS,
        "corpus_ids": corpus_ids,
        "corpus_line_ids": per_line,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote %s: p64=%d p600=%d p3000=%d corpus=%d ids"
          % (out, len(prompts["p64"]), len(prompts["p600"]), len(prompts["p3000"]), len(corpus_ids)))
    return 0


def _line_file(repo: pathlib.Path, i: int, s: str) -> pathlib.Path:
    d = repo / "tests/ds4/.corpus_lines"
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("%03d.txt" % i)
    f.write_text(s, encoding="utf-8")
    return f


if __name__ == "__main__":
    raise SystemExit(main())
