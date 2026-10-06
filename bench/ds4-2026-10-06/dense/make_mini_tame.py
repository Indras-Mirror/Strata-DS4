#!/usr/bin/env python3
"""bench/ds4-2026-10-06/dense/make_mini_tame.py - the numerically tame variant of the mini fixture.

Same geometry, same tensor set, same exercised code paths as tools/ds4/make_mini_gguf.py; only the
*values* differ: every floating tensor is drawn N(0,1) and then scaled by 1/sqrt(fan_in), which is
the per-matrix convention a real transformer checkpoint follows.

Why this is needed for the ds4-dense gate.  The stock generator draws *unit-scale* weights for
everything, including `hc_*_fn` (fan-in 1024).  The hyper-connection mixes then land at +-32, so
sigmoid/softmax saturate, the sinkhorn `hc_eps` (1e-6) takes over the normalisation and the two
layer net ends up with a ~1e3-1e4 gain (hc_head |max| 4576 against a 13.75 embedding).

The ds4-dense gate compares a *batch-1* decoder against ds4_ref's batched full-sequence graph, and
ggml-cpu's flash_attn_ext is not invariant to the query count: with q, kv and the mask row all
bit-identical (dumped via DS4_PROBE), the reference's t=0 attention output moves by 1.8e-4 relative
between a 63 and a 64 token run.  The hot fixture amplifies that to 1.6e-2 at the head, so no
token-by-token decoder can meet the gate's cos >= 0.99999 - the fixture, not the engine, is what
fails.  Scaling the weights removes the amplification without touching the gate's thresholds, the
token count, or the code paths under test.

    python3 bench/ds4-2026-10-06/dense/make_mini_tame.py <out.gguf>
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "ds4"))

import make_mini_gguf as M           # noqa: E402  (the stock geometry + tensor set)
from gguf import GGUFWriter          # noqa: E402


def tame_scale(tensor) -> np.float32:
    return np.float32(1.0 / math.sqrt(float(tensor.shape[-1])))


_add_tensor = GGUFWriter.add_tensor


def add_tensor(self, name, tensor, raw_shape=None, raw_dtype=None):
    # float tensors only: I32 (tid2eid) must keep its ids, and the raw quantised expert blobs are
    # already all zeros.
    if raw_dtype is None and tensor.dtype.kind == "f":
        tensor = (tensor.astype(np.float32) * tame_scale(tensor)).astype(tensor.dtype)
    return _add_tensor(self, name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype)


GGUFWriter.add_tensor = add_tensor

# DS4_SWA=<n> shrinks the sliding window (metadata only - no tensor shape depends on it).  With
# SWA=16 the raw-window wrap is reachable at 17 tokens instead of 129, i.e. inside the token range
# where ds4_ref's own attention numerics are stable, so the wrap path can be gated bit-exactly.
_u32 = GGUFWriter.add_uint32


def add_uint32(self, key, value):
    if key == "deepseek4.attention.sliding_window" and os.environ.get("DS4_SWA"):
        value = int(os.environ["DS4_SWA"])
    return _u32(self, key, value)


GGUFWriter.add_uint32 = add_uint32

# Q8_0 is scaled before quantisation: the block scales have to see the scaled values.
_quantize = M.quants.quantize


def quantize(arr, *args, **kwargs):
    return _quantize(arr * tame_scale(arr), *args, **kwargs)


M.quants.quantize = quantize

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "mini-ds4-tame.gguf"))
    M.write_model(out)
    print("wrote", out, out.stat().st_size, "bytes")
