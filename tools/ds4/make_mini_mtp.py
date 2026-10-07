#!/usr/bin/env python3
"""tools/ds4/make_mini_mtp.py - a miniature DeepSeek-V4 MTP (nextn) block matching tools/ds4/make_mini_gguf.py.

The real MTP head ships as its own GGUF (DeepSeek-V4-Flash-MTP-bf16.gguf): block `blk.<n_layer>` = one ratio-0
sliding-window layer plus the nextn tensors (eh_proj, enorm, hnorm, shared_head_norm), MXFP4 routed experts.  This
writes the same tensor set at the mini geometry (layer index L = 2), with tame 1/sqrt(fan_in) weights like
bench/ds4-2026-10-06/dense/make_mini_tame.py and - unlike the mini trunk, whose expert blobs are all zeros - real
random MXFP4 experts, so the gate exercises the MXFP4 expert path too.  The MTP head's own hc_head (`output_hc_*`,
like the real file's) is written with values unlike the trunk's, so a head that used the trunk's would fail the gate.
token_embd / output are not written: the MTP head uses the trunk's (the real file's are original-V4 copies).

    python3 tools/ds4/make_mini_mtp.py <out.gguf>

"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

import make_mini_gguf as M  # noqa: E402  (geometry constants; also puts gguf-py on the path)
from gguf import GGMLQuantizationType as Q, GGUFWriter, quants  # noqa: E402

IL = M.L   # the MTP block follows the trunk


def tensors():
    p = "blk.%d." % IL
    D, HC = M.D, M.HC
    return [
        (p + "nextn.eh_proj.weight", [2 * D, D], "F16"),
        (p + "nextn.enorm.weight", [D], "F32"),
        (p + "nextn.hnorm.weight", [D], "F32"),
        (p + "nextn.shared_head_norm.weight", [D], "F32"),
        (p + "attn_norm.weight", [D], "F32"),
        (p + "attn_sinks.weight", [M.HEADS], "F32"),
        (p + "attn_q_a.weight", [D, M.RQ], "F16"),
        (p + "attn_q_a_norm.weight", [M.RQ], "F32"),
        (p + "attn_q_b.weight", [M.RQ, M.HEAD_DIM], "F16"),
        (p + "attn_kv.weight", [D, M.DH], "F16"),
        (p + "attn_kv_a_norm.weight", [M.DH], "F32"),
        (p + "attn_output_a.weight", [M.O_GROUP_DIM, M.OL * M.OG], "F16"),
        (p + "attn_output_b.weight", [M.OG * M.OL, D], "F16"),
        (p + "hc_attn_fn.weight", [HC * D, (2 + HC) * HC], "F32"),
        (p + "hc_attn_base.weight", [(2 + HC) * HC], "F32"),
        (p + "hc_attn_scale.weight", [3], "F32"),
        (p + "hc_ffn_fn.weight", [HC * D, (2 + HC) * HC], "F32"),
        (p + "hc_ffn_base.weight", [(2 + HC) * HC], "F32"),
        (p + "hc_ffn_scale.weight", [3], "F32"),
        (p + "ffn_gate_inp.weight", [D, M.NE], "F32"),
        (p + "exp_probs_b.bias", [M.NE], "F32"),
        (p + "ffn_norm.weight", [D], "F32"),
        (p + "ffn_gate_exps.weight", [D, M.NFF, M.NE], "MXFP4"),
        (p + "ffn_up_exps.weight", [D, M.NFF, M.NE], "MXFP4"),
        (p + "ffn_down_exps.weight", [M.NFF, D, M.NE], "MXFP4"),
        (p + "ffn_gate_shexp.weight", [D, M.NFF * M.NSHEXP], "F16"),
        (p + "ffn_up_shexp.weight", [D, M.NFF * M.NSHEXP], "F16"),
        (p + "ffn_down_shexp.weight", [M.NFF * M.NSHEXP, D], "F16"),
        # appended last so the rng stream (and every tensor above) stays what the seed-11 fixture always had
        ("output_hc_fn.weight", [HC * D, HC], "F16"),
        ("output_hc_base.weight", [HC], "F32"),
        ("output_hc_scale.weight", [1], "F32"),
    ]


def write_model(path: Path) -> None:
    w = GGUFWriter(str(path), "deepseek4")
    w.add_uint32("deepseek4.block_count", M.L + 1)
    w.add_uint32("deepseek4.nextn_predict_layers", 1)
    w.add_uint32("deepseek4.embedding_length", M.D)
    w.add_array("deepseek4.attention.compress_ratios", [float(r) for r in M.RATIOS] + [0.0])
    w.add_array("deepseek4.swiglu_clamp_exp", [10.0] * (M.L + 1))
    w.add_array("deepseek4.swiglu_clamp_shexp", [10.0] * (M.L + 1))

    rng = np.random.default_rng(11)
    for name, ne, ty in tensors():
        n = int(np.prod(ne))
        np_shape = tuple(reversed(ne))          # gguf-py writes dims reversed
        x = rng.standard_normal(n).astype(np.float32).reshape(np_shape)
        if name == "output_hc_scale.weight":
            x = np.full(np_shape, 2.0, np.float32)   # the real MTP file's is ~2x the trunk's
        elif name.endswith("norm.weight"):
            x = 1.0 + 0.1 * x                   # norm gains near 1, like a trained checkpoint
        elif len(ne) >= 2 or name.endswith("_fn.weight"):
            x = x / math.sqrt(float(ne[0]))     # 1/sqrt(fan_in)
        if ty == "F32":
            w.add_tensor(name, x)
        elif ty == "F16":
            w.add_tensor(name, x.astype(np.float16))
        else:
            w.add_tensor(name, quants.quantize(x, Q.MXFP4), raw_dtype=Q.MXFP4)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mini-ds4-mtp.gguf")
    write_model(out)
    print("wrote", out, out.stat().st_size, "bytes")
