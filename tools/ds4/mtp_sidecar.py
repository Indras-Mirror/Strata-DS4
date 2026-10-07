#!/usr/bin/env python3
"""tools/ds4/mtp_sidecar.py - apply avlp12's 0731-aligned MTP sidecar (mtp_aligned_*.safetensors: the MTP block's
non-expert tensors re-trained on DeepSeek-V4-Flash-0731's own greedy outputs) to the MTP GGUF.

    python3 tools/ds4/mtp_sidecar.py check SIDECAR MTP.gguf          # name map + similarity to the originals
    python3 tools/ds4/mtp_sidecar.py write SIDECAR MTP.gguf OUT.gguf # copy MTP.gguf, overwrite those tensors in place

The output keeps the GGUF's header and every tensor's type/shape: each sidecar tensor is converted to the GGUF
tensor's existing type (BF16/F32) and written over its bytes, so experts and everything not in the sidecar are
byte-identical.  e_proj and h_proj are fused into nextn.eh_proj along the input dim; `check` decides the order by
correlating them with the GGUF's halves (the sidecar is a fine-tune of the originals).
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_mini_gguf  # noqa: E402,F401  (puts gguf-py on the path)
from gguf import GGUFReader  # noqa: E402

IL = 43
P = f"blk.{IL}."
# sidecar name -> GGUF name (None = special-cased)
MAP = {
    "block.attn.attn_sink": P + "attn_sinks.weight",
    "block.attn.kv_norm.weight": P + "attn_kv_a_norm.weight",
    "block.attn.q_norm.weight": P + "attn_q_a_norm.weight",
    "block.attn.wkv.weight": P + "attn_kv.weight",
    "block.attn.wq_a.weight": P + "attn_q_a.weight",
    "block.attn.wq_b.weight": P + "attn_q_b.weight",
    "block.attn_hc.base": P + "hc_attn_base.weight",
    "block.attn_hc.fn": P + "hc_attn_fn.weight",
    "block.attn_hc.scale": P + "hc_attn_scale.weight",
    "block.attn_norm.weight": P + "attn_norm.weight",
    "block.ffn_hc.base": P + "hc_ffn_base.weight",
    "block.ffn_hc.fn": P + "hc_ffn_fn.weight",
    "block.ffn_hc.scale": P + "hc_ffn_scale.weight",
    "block.ffn_norm.weight": P + "ffn_norm.weight",
    "enorm.weight": P + "nextn.enorm.weight",
    "hnorm.weight": P + "nextn.hnorm.weight",
    "norm.weight": P + "nextn.shared_head_norm.weight",
    "hc_head.base": "output_hc_base.weight",
    "hc_head.fn": "output_hc_fn.weight",
    "hc_head.scale": "output_hc_scale.weight",
    "e_proj.weight": None,
    "h_proj.weight": None,
}
EH = P + "nextn.eh_proj.weight"


def read_sidecar(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        raw = mm[base + a: base + b]
        if v["dtype"] == "BF16":
            x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] == "F32":
            x = raw.view(np.float32)
        else:
            raise SystemExit(f"{k}: dtype {v['dtype']}")
        out[k] = (x.reshape(v["shape"]), v["dtype"])
    return out


def gguf_f32(t):
    """a GGUF tensor as float32 in numpy (torch) order [out, in]"""
    if t.tensor_type.name == "BF16":
        raw = np.asarray(t.data).view(np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(tuple(int(d) for d in reversed(t.shape)))
    if t.tensor_type.name == "F32":
        return np.asarray(t.data, dtype=np.float32).reshape(tuple(int(d) for d in reversed(t.shape)))
    raise SystemExit(f"{t.name}: type {t.tensor_type.name} not handled")


def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def plan(side, rd):
    """[(gguf tensor, new float32 data in [out, in] order)], printing the similarity of each to the original"""
    tens = {t.name: t for t in rd.tensors}
    todo = []
    for k, (x, dt) in sorted(side.items()):
        g = MAP.get(k, "?")
        if g == "?":
            raise SystemExit(f"unmapped sidecar tensor {k}")
        if g is None:
            continue
        t = tens[g]
        old = gguf_f32(t)
        if old.size != x.size:
            raise SystemExit(f"{k} -> {g}: {x.shape} vs GGUF {old.shape}")
        x = x.reshape(old.shape)
        print(f"  {k:28s} -> {g:34s} {t.tensor_type.name:4s} {str(list(old.shape)):14s} cos {cos(old, x):.5f}"
              f"  rel|d| {np.linalg.norm(x - old) / (np.linalg.norm(old) + 1e-30):.4f}")
        todo.append((t, x))
    # eh_proj [out 4096, in 8192]: which half is e_proj?
    t = tens[EH]
    old = gguf_f32(t)
    e, h = side["e_proj.weight"][0], side["h_proj.weight"][0]
    D = e.shape[1]
    c = {"[e|h]": (cos(old[:, :D], e), cos(old[:, D:], h)), "[h|e]": (cos(old[:, :D], h), cos(old[:, D:], e))}
    for k, v in c.items():
        print(f"  eh_proj as {k}: cos first half {v[0]:.5f}, second half {v[1]:.5f}")
    order = max(c, key=lambda k: min(c[k]))
    if min(c[order]) < 0.5:
        raise SystemExit("eh_proj: neither concat order matches the originals")
    print(f"  -> eh_proj = {order} along the input dim")
    new = np.concatenate([e, h] if order == "[e|h]" else [h, e], axis=1)
    todo.append((t, new))
    return todo


def to_bytes(t, x):
    if t.tensor_type.name == "BF16":   # round to nearest even
        u = x.astype(np.float32).view(np.uint32)
        u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return u.tobytes()
    return x.astype(np.float32).tobytes()


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("check", "write"):
        print(__doc__); return 2
    side = read_sidecar(sys.argv[2])
    rd = GGUFReader(sys.argv[3])
    todo = plan(side, rd)
    if sys.argv[1] == "check":
        return 0
    out = Path(sys.argv[4])
    shutil.copyfile(sys.argv[3], out)
    mm = np.memmap(out, dtype=np.uint8, mode="r+")
    for t, x in todo:
        b = to_bytes(t, x)
        if len(b) != t.n_bytes:
            raise SystemExit(f"{t.name}: {len(b)} bytes vs {t.n_bytes}")
        mm[t.data_offset: t.data_offset + t.n_bytes] = np.frombuffer(b, np.uint8)
    mm.flush()
    del mm
    # verify: re-read and compare every replaced tensor
    rd2 = GGUFReader(out)
    tens = {t.name: t for t in rd2.tensors}
    for t, x in todo:
        y = gguf_f32(tens[t.name])
        tol = 0 if t.tensor_type.name == "F32" else 2 ** -8
        err = float(np.max(np.abs(y - x.reshape(y.shape)) / (np.abs(x.reshape(y.shape)) + 1e-30)))
        if err > tol:
            raise SystemExit(f"verify {t.name}: max rel err {err}")
    print(f"wrote {out} ({len(todo)} tensors replaced, verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
