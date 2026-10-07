#!/usr/bin/env python3
"""tools/ds4/cmp_draft_logits.py REF OUT N_VOCAB - compare ds4_generate DS4_DUMP_DRAFTLOGITS dumps.

Each record: int32 k (the generated-token index the draft guesses) + the MTP logits row (f32 x n_vocab).  Every k
in OUT must also be in REF with a byte-identical row (on the CPU, multi-token passes are bit-identical to one-token
decoding).  Exit 1 on any mismatch, or if OUT has no records (unless --allow-empty: a 1-token run drafts nothing)."""
import sys

import numpy as np


def load(path, nv):
    raw = np.fromfile(path, dtype=np.uint8)
    rec = 4 * (1 + nv)
    if raw.size % rec: raise SystemExit(f"{path}: size {raw.size} not a multiple of {rec}")
    r = raw.reshape(-1, rec)
    return {int(x[:4].view(np.int32)[0]): x[4:].tobytes() for x in r}, r.shape[0]


def main():
    allow_empty = "--allow-empty" in sys.argv
    ref_p, out_p, nv = [x for x in sys.argv[1:] if x != "--allow-empty"]
    nv = int(nv)
    ref, _ = load(ref_p, nv)
    out, n = load(out_p, nv)
    if n == 0: print("no drafts"); return 0 if allow_empty else 1
    bad = [k for k in out if k not in ref or ref[k] != out[k]]
    print(f"{n} drafts, {len(bad)} differ from the reference" + (f": token indices {sorted(bad)[:10]}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
