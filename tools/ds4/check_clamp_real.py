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
worst = (0.0, None); over = 0; total = 0; per_layer = {}
for l in range(43):
    gt, ut = t[f"blk.{l}.ffn_gate_exps.weight"], t[f"blk.{l}.ffn_up_exps.weight"]
    toks = rng.choice(n_tok, size=N, replace=False)
    need = defaultdict(list)
    for j in toks:
        for e in d["K"][l][j]:
            need[int(e)].append(int(j))
    lmax = 0.0
    for e, js in need.items():
        # expert e's slice of the [n_ff, n_embd, n_expert] tensor (ggml ne order) = rows of the raw block data
        def mat(tt):
            raw = np.asarray(tt.data)            # [n_expert, n_ff, bytes_per_row]
            return dequantize(raw[e], tt.tensor_type).reshape(-1, 4096)   # [n_ff, n_embd]
        G, U = mat(gt), mat(ut)
        X = d["F"][l][js].astype(np.float32)     # [m, 4096] the router input = the experts' input
        g, u = X @ G.T, X @ U.T
        m = max(np.abs(g).max(), np.abs(u).max())
        over += int((np.abs(g) > 10).sum() + (np.abs(u) > 10).sum()); total += g.size + u.size
        lmax = max(lmax, m)
        if m > worst[0]:
            worst = (float(m), (l, e))
    per_layer[l] = lmax
    print(f"layer {l:2d}: max |pre-act| {lmax:7.3f}  ({len(need)} experts, {N} tokens)", flush=True)
print(f"\nworst |gate|/|up| pre-activation {worst[0]:.3f} at (layer, expert) {worst[1]}; "
      f"elements beyond the clamp (|x| > 10): {over} of {total} ({100.0*over/max(total,1):.4f}%)")
