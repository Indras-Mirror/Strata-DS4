#!/usr/bin/env python3
"""sim_prefetch.py - offline: how many cache misses per token does predicted prefetch during attention remove?

Cache = top-S of the routing profile from ds4routes.bin's TRAIN half (as moe_replay seeds it); tokens = the
hidden_probe dump (held out). Per layer the prefetcher ranks the predictor's experts, skips cached ones, and loads
the first B (fractional B is error-diffused across layers). Hash layers (0-2) are predicted exactly (tid2eid).
  python3 tools/ds4/sim_prefetch.py bench/ds4-2026-10-06/hidden-probe/h.bin
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import predict_experts as P
from gguf import GGUFReader

REPO = Path(__file__).resolve().parents[2]
ROUTES = REPO / "bench/ds4-2026-10-05/route-probe/ds4routes.bin"
L, E, K, BATCH = 43, 256, 6, 512

def profile():
    a = np.fromfile(ROUTES, dtype=np.uint16).reshape(-1, 2)
    rec = np.arange(len(a)); per_batch = BATCH * L * K
    tok = (rec // per_batch) * BATCH + (rec % (BATCH * K)) // K
    keep = (tok // BATCH) % 2 == 0
    c = Counter(map(tuple, a[keep].tolist()))
    return sorted(c, key=lambda le: (-c[le], le))

def main():
    d = P.load_dump(sys.argv[1])
    r = GGUFReader(P.GGUF)
    t = {x.name: x for x in r.tensors if x.name.startswith("blk.") and x.name.split(".")[2] in ("ffn_gate_inp", "exp_probs_b", "ffn_norm")}
    n = d["K"][0].shape[0]
    preds = {}   # layer -> {name: [n, 12] ranked}
    for l in range(L):
        bt = t.get(f"blk.{l}.exp_probs_b.bias")
        if bt is None:
            preds[l] = {"A": d["K"][l], "F": d["K"][l]}; continue
        W = np.asarray(t[f"blk.{l}.ffn_gate_inp.weight"].data).reshape(256, 4096).astype(np.float32)
        b = np.asarray(bt.data).astype(np.float32).reshape(256)
        nw = np.asarray(t[f"blk.{l}.ffn_norm.weight"].data).astype(np.float32).reshape(4096)
        a = d["A"][l]; an = a / np.sqrt((a.astype(np.float64) ** 2).mean(1, keepdims=True) + P.EPS) * nw
        preds[l] = {"A": P.select(W, b, an, 16), "F": P.select(W, b, d["F"][l - 1], 16)}
    # per-layer predictor choice: calibrated on the first half of the tokens, scored on the second half
    half = n // 2
    choice = {}
    for l in range(L):
        sc = {k: P.recall(v[:half, :6], d["K"][l][:half]) for k, v in preds[l].items()}
        choice[l] = max(sc, key=sc.get)
    ranked = profile()
    print(f"{n} tokens (scoring the second half: {n - half}); per-layer predictor picked on the first half: "
          f"A on {sum(c == 'A' for c in choice.values())} layers, F on {sum(c == 'F' for c in choice.values())}")
    print(f"{'slots':>5} {'B/layer':>7} {'pred':>5} {'hit%':>6} {'miss/tok':>8} {'after':>7} {'cut':>6} {'prefetch/tok':>12} {'useful%':>8}")
    for S in (1820, 2300):
        cached = set(ranked[:S])
        for B in (0.0, 1.0, 1.43, 2.0, 3.0):
            for pname in (("best",) if B > 0 else ("-",)):
                miss0 = miss1 = pf = useful = hits = 0; carry = 0.0
                for j in range(half, n):
                    for l in range(L):
                        act = d["K"][l][j]
                        m = [e for e in act if (l, int(e)) not in cached]
                        hits += K - len(m); miss0 += len(m)
                        want = B + carry; nb = int(np.floor(want)); carry = want - nb
                        p = preds[l][choice[l]][j] if pname == "best" else []
                        got = []
                        for e in p:
                            if len(got) >= nb: break
                            if (l, int(e)) not in cached: got.append(int(e))
                        pf += len(got); u = len(set(got) & set(map(int, m))); useful += u
                        miss1 += len(m) - u
                nt = n - half
                print(f"{S:5d} {B:7.2f} {pname:>5} {100*hits/(nt*L*K):6.1f} {miss0/nt:8.1f} {miss1/nt:7.1f} "
                      f"{100*(1-miss1/max(miss0,1)):5.1f}% {pf/nt:12.1f} {100*useful/max(pf,1):7.1f}%")

if __name__ == "__main__" and "--export" not in sys.argv:
    main()


def export(dump_path, out_dir):
    """write the replay inputs: routes.bin (route_probe format, from the dump's own selections) and pred.bin
    ([n_tok][43][16] u16, the per-layer predictor picked on the first half of the tokens)."""
    d = P.load_dump(dump_path)
    r = GGUFReader(P.GGUF)
    t = {x.name: x for x in r.tensors if x.name.startswith("blk.") and x.name.split(".")[2] in ("ffn_gate_inp", "exp_probs_b", "ffn_norm")}
    n = d["K"][0].shape[0]; half = n // 2
    out = np.zeros((n, L, 16), dtype=np.uint16)
    for l in range(L):
        bt = t.get(f"blk.{l}.exp_probs_b.bias")
        if bt is None:
            out[:, l, :6] = d["K"][l]; out[:, l, 6:] = d["K"][l][:, :1]; continue
        W = np.asarray(t[f"blk.{l}.ffn_gate_inp.weight"].data).reshape(256, 4096).astype(np.float32)
        b = np.asarray(bt.data).astype(np.float32).reshape(256)
        nw = np.asarray(t[f"blk.{l}.ffn_norm.weight"].data).astype(np.float32).reshape(4096)
        a = d["A"][l]; an = a / np.sqrt((a.astype(np.float64) ** 2).mean(1, keepdims=True) + P.EPS) * nw
        pa, pf = P.select(W, b, an, 16), P.select(W, b, d["F"][l - 1], 16)
        use_a = P.recall(pa[:half, :6], d["K"][l][:half]) >= P.recall(pf[:half, :6], d["K"][l][:half])
        out[:, l] = pa if use_a else pf
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out.tofile(out_dir / "pred.bin")
    # routes in route_probe order: per 512-token ubatch, layer-major, token, k
    recs = []
    for u in range(0, n, BATCH):
        for l in range(L):
            k = d["K"][l][u:u + BATCH]
            lay = np.full_like(k, l)
            recs.append(np.stack([lay, k], axis=-1).reshape(-1, 2))
    np.concatenate(recs).astype(np.uint16).tofile(out_dir / "routes.bin")
    print(f"wrote {out_dir}/pred.bin ({n} x {L} x 16) and routes.bin")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--export":
    export(sys.argv[1], sys.argv[3])
