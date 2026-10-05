#!/usr/bin/env python3
"""check_act_quant.py - does activation quantization explain the Phase 3 attn_out-0 residual?

Rebuilds layer 0's attention output projection (de-RoPE + grouped Wo_a + Wo_b, both Q8_0) in numpy
from the golden's own attn_raw-0 (last token), reading only those two weight tensors from the GGUF
via mmap (~70 MB touched, no model load, no GPU). Each matmul is run with the activations as:
  f32  - exact (dequantized weights x float activations)
  q8   - activations quantized to Q8_0 blocks of 32 (what ggml-cpu does for Q8_0 weights; CUDA MMQ's
         q8_1 uses the same rounding)
  f16  - activations rounded to half (a cuBLAS-style path)
and each result is compared with the golden (CUDA oracle) and ds4_ref (ggml-cpu) attn_out-0.

  nice python3 tools/ds4/check_act_quant.py [p64]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "AI/llama.cpp-master-rebase/gguf-py"))
from gguf import GGUFReader  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GGUF = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf"
P = sys.argv[1] if len(sys.argv) > 1 else "p64"
GOLD = REPO / f"bench/ds4-2026-10-05/goldens/{P}"
REF = REPO / f"bench/ds4-2026-10-05/ref/{P}"


def load(d, name):
    return np.fromfile(d / f"{name}.f32", dtype=np.float32).astype(np.float64)


def q8_0_dequant(raw, rows, cols):
    # Q8_0 block: f16 d + 32 x int8
    b = raw.reshape(rows, cols // 32, 34)
    d = b[:, :, :2].copy().view(np.float16).astype(np.float64)
    q = b[:, :, 2:].copy().view(np.int8).astype(np.float64)
    return (q * d).reshape(rows, cols)


def act(x, mode):
    if mode == "f32":
        return x
    if mode == "f16":
        return x.astype(np.float16).astype(np.float64)
    # Q8_0 activation quantization as in ggml quantize_row_q8_0_ref
    b = x.reshape(-1, 32)
    amax = np.abs(b).max(axis=1, keepdims=True)
    d = amax / 127.0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(d > 0, np.round(b / d), 0.0)
    d16 = d.astype(np.float16).astype(np.float64)
    return (q * d16).reshape(x.shape)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    r = GGUFReader(GGUF)
    kv = {f.name: f for f in r.fields.values()}

    def meta(key):
        f = kv[key]
        return f.parts[f.data[0]][0].item()

    base = float(meta("deepseek4.rope.freq_base"))
    n_head = int(meta("deepseek4.attention.head_count"))
    d_h = int(meta("deepseek4.attention.key_length"))
    d_rope = int(meta("deepseek4.rope.dimension_count"))
    t = {x.name: x for x in r.tensors if x.name.startswith("blk.0.attn_output_")}
    wa_t, wb_t = t["blk.0.attn_output_a.weight"], t["blk.0.attn_output_b.weight"]
    print(f"rope base {base} n_head {n_head} d_h {d_h} d_rope {d_rope}")
    print("wo_a", wa_t.tensor_type.name, list(wa_t.shape), " wo_b", wb_t.tensor_type.name, list(wb_t.shape))

    # ggml shape [ne0=in, ne1=out]
    a_in, a_out = int(wa_t.shape[0]), int(wa_t.shape[1])
    b_in, b_out = int(wb_t.shape[0]), int(wb_t.shape[1])
    Wa = q8_0_dequant(np.asarray(wa_t.data).reshape(-1), a_out, a_in)
    Wb = q8_0_dequant(np.asarray(wb_t.data).reshape(-1), b_out, b_in)

    man = json.loads((GOLD / "manifest.json").read_text())
    pos = man["n_tokens"] - 1
    x = load(GOLD, "attn_raw-0").reshape(n_head, d_h)

    # de-RoPE: NORMAL mode (adjacent pairs) on the last d_rope dims, rotate by -theta
    i = np.arange(d_rope // 2)
    theta = pos * base ** (-2.0 * i / d_rope)
    c, s = np.cos(theta), np.sin(theta)
    r0 = x[:, d_h - d_rope::2].copy()
    r1 = x[:, d_h - d_rope + 1::2].copy()
    x[:, d_h - d_rope::2] = r0 * c + r1 * s
    x[:, d_h - d_rope + 1::2] = -r0 * s + r1 * c

    n_groups = n_head * d_h // a_in
    o_lora = a_out // n_groups
    xg = x.reshape(n_groups, a_in)
    gold = load(GOLD, "attn_out-0")
    ref = load(REF, "attn_out-0")
    print(f"pos {pos} groups {n_groups} o_lora {o_lora}")
    print(f"ds4_ref vs golden: cos {cos(ref, gold):.9f}  max_abs {np.abs(ref - gold).max():.3e}")
    print(f"{'act mode':10s} {'cos vs golden':>16s} {'cos vs ds4_ref':>16s} {'maxabs golden':>14s} {'maxabs ref':>12s}")
    for mode in ("f32", "q8", "f16"):
        oa = np.concatenate([Wa[g * o_lora:(g + 1) * o_lora] @ act(xg[g], mode) for g in range(n_groups)])
        y = Wb @ act(oa, mode)
        print(f"{mode:10s} {cos(y, gold):16.9f} {cos(y, ref):16.9f} "
              f"{np.abs(y - gold).max():14.3e} {np.abs(y - ref).max():12.3e}")


if __name__ == "__main__":
    main()
