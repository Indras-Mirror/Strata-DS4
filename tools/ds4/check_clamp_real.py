#!/usr/bin/env python3
"""check_clamp_real.py - does DeepSeek-V4's SwiGLU clamp (swiglu_clamp_exp = 10) ever bind on REAL activations?

Strata's native expert kernels (CPU + CUDA) compute silu(gate)*up with NO clamp; the model clamps gate/up to 10
first (docs/ds4/ENGINE_MOE.md). test_ds4_moe's probe used synthetic activations; this one uses the real ffn_norm
activations and the actually-routed experts from hidden_probe's dump (bench/ds4-2026-10-06/hidden-probe/h.bin).
Expert weights are dequantised from the GGUF via gguf-py (mmap; only the sampled experts are touched).
  python3 tools/ds4/check_clamp_real.py [n_tokens_per_layer]
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import predict_experts as P
from gguf import GGUFReader
from gguf.quants import dequantize

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
d = P.load_dump(str(Path(__file__).resolve().parents[2] / "bench/ds4-2026-10-06/hidden-probe/h.bin"))
r = GGUFReader(P.GGUF)
t = {x.name: x for x in r.tensors if "_exps.weight" in x.name and ("gate" in x.name or "up" in x.name)}
rng = np.random.default_rng(0)
n_tok = d["K"][0].shape[0]
LIM = 10.0   # deepseek4.swiglu_clamp_exp, 10.0 on all 43 layers of the 0731 GGUF
# the model's rule (ds4_ref.cpp / llama.cpp DEEPSEEK4): up -> clamp(up, -LIM, LIM), gate -> min(gate, LIM)
silu = lambda v: v / (1.0 + np.exp(-v))
worst_g = (-1e30, None); worst_u = (0.0, None)
n_g = n_u = total = 0; worst_rel = (0.0, None)
for l in range(43):
    gt, ut = t[f"blk.{l}.ffn_gate_exps.weight"], t[f"blk.{l}.ffn_up_exps.weight"]
    toks = rng.choice(n_tok, size=N, replace=False)
    need = defaultdict(list)
    for j in toks:
        for e in d["K"][l][j]:
            need[int(e)].append(int(j))
    lg, lu, lrel, lbind = -1e30, 0.0, 0.0, 0
    for e, js in need.items():
        # expert e's slice of the [n_ff, n_embd, n_expert] tensor (ggml ne order) = rows of the raw block data
        def mat(tt):
            raw = np.asarray(tt.data)            # [n_expert, n_ff, bytes_per_row]
            return dequantize(raw[e], tt.tensor_type).reshape(-1, 4096)   # [n_ff, n_embd]
        G, U = mat(gt), mat(ut)
        X = d["F"][l][js].astype(np.float32)     # [m, 4096] the router input = the experts' input
        g, u = X @ G.T, X @ U.T
        bg = int((g > LIM).sum()); bu = int((np.abs(u) > LIM).sum())
        n_g += bg; n_u += bu; lbind += bg + bu; total += g.size
        if g.max() > lg: lg = float(g.max())
        if np.abs(u).max() > lu: lu = float(np.abs(u).max())
        if g.max() > worst_g[0]: worst_g = (float(g.max()), (l, e))
        if np.abs(u).max() > worst_u[0]: worst_u = (float(np.abs(u).max()), (l, e))
        if bg or bu:
            # per-token relative change of the SwiGLU output h (the down projection's input)
            h0 = silu(g) * u
            h1 = silu(np.minimum(g, LIM)) * np.clip(u, -LIM, LIM)
            rel = np.linalg.norm(h1 - h0, axis=1) / np.maximum(np.linalg.norm(h0, axis=1), 1e-30)
            lrel = max(lrel, float(rel.max()))
            if rel.max() > worst_rel[0]: worst_rel = (float(rel.max()), (l, e))
    print(f"layer {l:2d}: max gate {lg:7.3f}  max |up| {lu:7.3f}  binding elems {lbind:4d}  "
          f"max rel dh {lrel:.3e}  ({len(need)} experts, {N} tokens)", flush=True)
print(f"\nmax gate {worst_g[0]:.3f} at {worst_g[1]}; max |up| {worst_u[0]:.3f} at {worst_u[1]}")
print(f"binding: gate > {LIM}: {n_g}, |up| > {LIM}: {n_u} of {total} elements each "
      f"({100.0*(n_g+n_u)/max(2*total,1):.5f}%); worst per-token rel change of h {worst_rel[0]:.3e} at {worst_rel[1]}")
