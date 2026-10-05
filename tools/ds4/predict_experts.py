#!/usr/bin/env python3
"""predict_experts.py - can layer l's routed experts be predicted before layer l's attention finishes?

Reads hidden_probe's dump and the router weights from the GGUF (mmap, ~90 MB touched), and for every non-hash layer
scores three predictors by recall: the fraction of the 6 actually-selected experts that appear in the predictor's
top-k (k = 6, 8, 10, 12 - a bigger k prefetches more bytes).
  true   router_l(ffn_norm_l)                         - sanity: must reproduce the selection (~100%)
  F      router_l(ffn_norm_{l-1})                     - known after layer l-1's attention (a whole layer early)
  A      router_l(rmsnorm(hc_attn_pre_l) * ffn_norm_w_l) - known before layer l's attention
Selection as llama.cpp deepseek4: top-6 of sqrt(softplus(W x)) + exp_probs_b. Hash layers (0-2) take experts from
tid2eid[token]: 100% predictable from the token id alone.

  python3 tools/ds4/predict_experts.py bench/ds4-2026-10-06/hidden-probe/h.bin
"""
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "AI/llama.cpp-master-rebase/gguf-py"))
from gguf import GGUFReader  # noqa: E402

GGUF = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf"
KS = (6, 8, 10, 12)
EPS = 1e-6


def load_dump(path):
    data = {"A": defaultdict(list), "F": defaultdict(list), "K": defaultdict(list)}
    with open(path, "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            kind, _, layer, n, w = struct.unpack("<cBHII", h)
            kind = kind.decode()
            dt = np.int32 if kind in "KT" else np.float32
            arr = np.frombuffer(f.read(4 * n * w), dtype=dt).reshape(n, w)
            if kind != "T":
                data[kind][layer].append(arr)
    return {k: {l: np.concatenate(v) for l, v in d.items()} for k, d in data.items()}


def select(W, b, x, k):
    logits = x.astype(np.float32) @ W.T                     # [n, 256]
    s = np.sqrt(np.logaddexp(0, logits))                    # sqrt(softplus)
    if b is not None:
        s = s + b
    return np.argsort(-s, axis=1)[:, :k]


def recall(pred, actual):
    hit = (pred[:, :, None] == actual[:, None, :]).any(axis=1)  # [n, 6]
    return hit.mean()


def main():
    d = load_dump(sys.argv[1])
    r = GGUFReader(GGUF)
    t = {x.name: x for x in r.tensors if x.name.startswith("blk.") and
         x.name.split(".")[2] in ("ffn_gate_inp", "exp_probs_b", "ffn_norm")}
    layers = sorted(d["K"])
    n_tok = d["K"][layers[0]].shape[0]
    print(f"{n_tok} tokens, {len(layers)} layers")
    print(f"{'layer':>5} {'true@6':>7} " + " ".join(f"F@{k:<3d}" for k in KS) + "  " + " ".join(f"A@{k:<3d}" for k in KS))
    agg = defaultdict(list)
    for l in layers:
        W = np.asarray(t[f"blk.{l}.ffn_gate_inp.weight"].data).reshape(256, 4096).astype(np.float32)
        bt = t.get(f"blk.{l}.exp_probs_b.bias")
        if bt is None:
            print(f"{l:5d}  hash layer (tid2eid): 100% predictable from the token id")
            continue
        b = np.asarray(bt.data).astype(np.float32).reshape(256)
        nw = np.asarray(t[f"blk.{l}.ffn_norm.weight"].data).astype(np.float32).reshape(4096)
        actual = d["K"][l]
        row = [recall(select(W, b, d["F"][l], 6), actual)]
        agg["true"].append(row[0])
        fr = []
        if l - 1 in d["F"]:
            for k in KS:
                fr.append(recall(select(W, b, d["F"][l - 1], k), actual)); agg[f"F{k}"].append(fr[-1])
        a = d["A"][l]
        an = a / np.sqrt((a.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + EPS) * nw
        ar = []
        for k in KS:
            ar.append(recall(select(W, b, an, k), actual)); agg[f"A{k}"].append(ar[-1])
        print(f"{l:5d} {row[0]:7.3f} " + " ".join(f"{x:5.3f}" for x in fr) + "  " + " ".join(f"{x:5.3f}" for x in ar))
    print("mean  " + f"{np.mean(agg['true']):7.3f} " + " ".join(f"{np.mean(agg[f'F{k}']):5.3f}" for k in KS) + "  "
          + " ".join(f"{np.mean(agg[f'A{k}']):5.3f}" for k in KS))


if __name__ == "__main__":
    main()
