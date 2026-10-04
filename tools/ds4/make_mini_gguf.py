#!/usr/bin/env python3
"""tools/ds4/make_mini_gguf.py - write a persistent miniature `deepseek4` GGUF for ds4_ref smoke tests.

The real artifact is 80 GiB; this is a 2-layer model with the same architecture (layer 0 = ratio-4 CSA with
the indexer and hash routing, layer 1 = ratio-128 HCA) but tiny dims, so ds4_ref can be exercised end to
end (loader, hyper-connections, raw/CSA/HCA attention, MoE, head, dumps) without the model or the GPU.

Tensor shapes here are given in GGUF order (ne0 first, as ds4_expected_tensors does); gguf-py stores them
reversed, so the numpy arrays are built with the shape reversed.

    python3 tools/ds4/make_mini_gguf.py /tmp/mini-ds4.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))

from _paths import add_gguf_py  # noqa: E402

add_gguf_py()
from gguf import GGMLQuantizationType as Q, GGUFWriter, quants  # noqa: E402

# ---- miniature geometry (same ratios/structure as the real model, tiny widths)
D, L, HEADS, DH, DROPE, RQ = 256, 2, 2, 64, 16, 32
OG, OL = 2, 32
NE, TOPK, NFF, NSHEXP = 256, 2, 256, 1
HC, HASH = 4, 1
IDXH, IDXK = 2, 64
V = 64
RATIOS = [4, 128]
HEAD_DIM = HEADS * DH
O_GROUP_DIM = HEAD_DIM // OG


def tensors():
    """(name, gguf ne, ggml type) for every tensor the miniature model holds."""
    t = [
        ("token_embd.weight", [D, V], "F16"),
        ("output_norm.weight", [D], "F32"),
        ("output.weight", [D, V], "Q8_0"),
        ("output_hc_fn.weight", [HC * D, HC], "F16"),
        ("output_hc_base.weight", [HC], "F32"),
        ("output_hc_scale.weight", [1], "F32"),
    ]
    for i in range(L):
        p = "blk.%d." % i
        t += [
            (p + "attn_norm.weight", [D], "F32"),
            (p + "attn_sinks.weight", [HEADS], "F32"),
            (p + "attn_q_a.weight", [D, RQ], "Q8_0"),
            (p + "attn_q_a_norm.weight", [RQ], "F32"),
            (p + "attn_q_b.weight", [RQ, HEAD_DIM], "Q8_0"),
            (p + "attn_kv.weight", [D, DH], "Q8_0"),
            (p + "attn_kv_a_norm.weight", [DH], "F32"),
            (p + "attn_output_a.weight", [O_GROUP_DIM, OL * OG], "Q8_0"),
            (p + "attn_output_b.weight", [OG * OL, D], "Q8_0"),
            (p + "hc_attn_fn.weight", [HC * D, (2 + HC) * HC], "F16"),
            (p + "hc_attn_base.weight", [(2 + HC) * HC], "F32"),
            (p + "hc_attn_scale.weight", [3], "F32"),
            (p + "hc_ffn_fn.weight", [HC * D, (2 + HC) * HC], "F16"),
            (p + "hc_ffn_base.weight", [(2 + HC) * HC], "F32"),
            (p + "hc_ffn_scale.weight", [3], "F32"),
        ]
        ratio = RATIOS[i]
        if ratio != 0:
            coff = 2 if ratio == 4 else 1
            t += [
                (p + "attn_compressor_kv.weight", [D, coff * DH], "F16"),
                (p + "attn_compressor_gate.weight", [D, coff * DH], "F16"),
                (p + "attn_compressor_ape.weight", [coff * DH, ratio], "F16"),
                (p + "attn_compressor_norm.weight", [DH], "F32"),
            ]
            if ratio == 4:
                t += [
                    (p + "indexer.proj.weight", [D, IDXH], "F16"),
                    (p + "indexer.attn_q_b.weight", [RQ, IDXH * IDXK], "F16"),
                    (p + "indexer_compressor_kv.weight", [D, 2 * IDXK], "F16"),
                    (p + "indexer_compressor_gate.weight", [D, 2 * IDXK], "F16"),
                    (p + "indexer_compressor_ape.weight", [2 * IDXK, ratio], "F16"),
                    (p + "indexer_compressor_norm.weight", [IDXK], "F32"),
                ]
        t += [
            (p + "ffn_gate_inp.weight", [D, NE], "F16"),
            (p + "ffn_norm.weight", [D], "F32"),
            (p + "ffn_gate_exps.weight", [D, NFF, NE], "IQ2_XXS"),
            (p + "ffn_down_exps.weight", [NFF, D, NE], "Q2_K"),
            (p + "ffn_up_exps.weight", [D, NFF, NE], "IQ2_XXS"),
            (p + "ffn_down_shexp.weight", [NFF * NSHEXP, D], "Q8_0"),
            (p + "ffn_gate_shexp.weight", [D, NFF * NSHEXP], "Q8_0"),
            (p + "ffn_up_shexp.weight", [D, NFF * NSHEXP], "Q8_0"),
        ]
        if i < HASH:
            t.append((p + "ffn_gate_tid2eid.weight", [TOPK, V], "I32"))
        else:
            t.append((p + "exp_probs_b.bias", [NE], "F32"))
    return t


BLK = {"IQ2_XXS": 256, "Q2_K": 256}


def write_model(path: Path) -> None:
    w = GGUFWriter(str(path), "deepseek4")
    u32 = {"block_count": L, "embedding_length": D, "attention.head_count": HEADS,
           "attention.head_count_kv": 1, "attention.key_length": DH, "attention.value_length": DH,
           "rope.dimension_count": DROPE, "attention.q_lora_rank": RQ,
           "attention.output_group_count": OG, "attention.output_lora_rank": OL,
           "attention.sliding_window": 128, "expert_count": NE, "expert_used_count": TOPK,
           "expert_feed_forward_length": NFF, "expert_shared_count": NSHEXP,
           "expert_gating_func": 4, "hash_layer_count": HASH,
           "attention.indexer.head_count": IDXH, "attention.indexer.key_length": IDXK,
           "attention.indexer.top_k": 8, "vocab_size": V, "context_length": 4096,
           "nextn_predict_layers": 1, "hyper_connection.count": HC,
           "hyper_connection.sinkhorn_iterations": 20}
    for k, v in u32.items():
        w.add_uint32("deepseek4." + k, v)
    for k, v in {"expert_weights_scale": 1.5, "attention.compress_rope_freq_base": 160000.0,
                 "rope.freq_base": 10000.0, "rope.scaling.factor": 16.0,
                 "rope.scaling.original_context_length": 65536.0,
                 "rope.scaling.yarn_beta_fast": 32.0, "rope.scaling.yarn_beta_slow": 1.0,
                 "attention.layer_norm_rms_epsilon": 1e-6, "hyper_connection.epsilon": 1e-6}.items():
        w.add_float32("deepseek4." + k, v)
    w.add_bool("deepseek4.expert_weights_norm", True)
    w.add_string("deepseek4.rope.scaling.type", "yarn")
    w.add_string("tokenizer.ggml.pre", "joyai-llm")
    w.add_array("deepseek4.attention.compress_ratios", [float(r) for r in RATIOS])
    w.add_array("deepseek4.swiglu_clamp_exp", [10.0] * L)

    rng = np.random.default_rng(7)
    for name, ne, ty in tensors():
        n = int(np.prod(ne))
        np_shape = tuple(reversed(ne))          # gguf-py writes dims reversed
        if ty == "F16":
            w.add_tensor(name, rng.standard_normal(n).astype(np.float16).reshape(np_shape))
        elif ty == "F32":
            w.add_tensor(name, rng.standard_normal(n).astype(np.float32).reshape(np_shape))
        elif ty == "I32":
            w.add_tensor(name, rng.integers(0, NE, n, dtype=np.int32).reshape(np_shape))
        elif ty == "Q8_0":
            assert ne[0] % 32 == 0, (name, ne)
            arr = quants.quantize(rng.standard_normal(n).astype(np.float32).reshape(np_shape), Q.Q8_0)
            w.add_tensor(name, arr, raw_dtype=Q.Q8_0)
        else:
            e = BLK[ty]
            assert ne[0] % e == 0, (name, ne)
            byte_shape = np_shape[:-1] + (ne[0] // e * (66 if ty == "IQ2_XXS" else 84),)
            # random quant bytes dequantize to NaN (random f16 block scales), so keep the routed experts
            # at zero: the smoke test is about wiring and finiteness, not about expert values
            arr = np.zeros(int(np.prod(byte_shape)), dtype=np.uint8).reshape(byte_shape)
            w.add_tensor(name, arr, raw_shape=byte_shape, raw_dtype=Q[ty])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mini-ds4.gguf")
    write_model(out)
    print("wrote", out, out.stat().st_size, "bytes")
