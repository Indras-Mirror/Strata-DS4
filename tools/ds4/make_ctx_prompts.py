#!/usr/bin/env python3
"""make_ctx_prompts.py - token files for the long-context measurements (FINDINGS s24).

Tokenizes the repo's OWN text (docs, tools, src, include - no third_party, nothing private) with the GGUF's
tokenizer and writes the first N ids of each size, so a long prompt is 64K/128K/256K tokens of real prose and code
instead of a repeated fixture.  ds4_generate takes them with --ids-file; llama.cpp parity runs with --file.

  python3 tools/ds4/make_ctx_prompts.py                       # 64K / 128K / 256K into bench/ds4-2026-10-08/ctx/
  python3 tools/ds4/make_ctx_prompts.py --sizes 1048576 --out /data/ctx
  DS4_TOKENIZER_MODEL=<other.gguf> python3 tools/ds4/make_ctx_prompts.py
"""
import argparse
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

MODEL = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf"
PATTERNS = ["docs/**/*.md", "tools/**/*.py", "tools/**/*.cpp", "tools/**/*.hpp",
            "include/strata/**/*.hpp", "src/**/*.cpp", "src/**/*.cu", "CMakeLists.txt"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[65536, 131072, 262144])
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "bench/ds4-2026-10-08/ctx")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    files = sorted({f for p in PATTERNS for f in REPO.glob(p)})
    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    tok = Tokenizer.from_gguf(a.model)
    ids = tok.encode(text)
    print(f"corpus: {len(files)} files, {len(text)} chars, {len(ids)} tokens")
    a.out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for n in a.sizes:
        if len(ids) < n:
            print(f"corpus too short for {n} tokens - not written", file=sys.stderr)
            rc = 1
            continue
        p = a.out / f"ctx{n}.i32"
        p.write_bytes(struct.pack(f"<{n}i", *ids[:n]))
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
