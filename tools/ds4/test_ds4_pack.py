#!/usr/bin/env python3
"""tools/ds4/test_ds4_pack.py - exercise tools/ds4_pack.py end to end on a small synthetic `deepseek4` GGUF.

The real artifact is 80 GiB and the pack run is a Phase-1 residual; this builds a miniature model with the same
architecture shape (2 layers, ratio-4 CSA with the indexer on layer 0, ratio-128 HCA on layer 1, hash routing on
layer 0, IQ2_XXS gate/up + Q2_K down experts) and checks the pack's files: dense.bin, index.txt (dense rows and
native Q8_0 rows), native_experts.txt (per-layer blob and types), ds4_pack.json.

Run:  python tools/ds4/test_ds4_pack.py
"""
from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))

from _paths import add_gguf_py  # noqa: E402

add_gguf_py()
from gguf import GGMLQuantizationType as Q, GGUFWriter, quants  # noqa: E402

import ds4_pack  # noqa: E402

# ---- the miniature geometry
D, L, HEADS, DH, DROPE, RQ = 256, 2, 2, 64, 16, 32
OG, OL = 2, 32
NE, TOPK, NFF, NSHEXP = 256, 2, 256, 1
HC, HASH = 4, 1
IDXH, IDXK = 2, 64
V = 64
RATIOS = [4, 128]


def tensor_list():
    """(name, shape, ggml-type-name) for every tensor the miniature model holds."""
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
            (p + "attn_q_b.weight", [RQ, HEADS * DH], "Q8_0"),
            (p + "attn_kv.weight", [D, DH], "Q8_0"),
            (p + "attn_kv_a_norm.weight", [DH], "F32"),
            (p + "attn_output_a.weight", [HEADS * DH // OG, OL * OG], "Q8_0"),
            (p + "attn_output_b.weight", [OL * OG, D], "Q8_0"),
            (p + "hc_attn_fn.weight", [HC * D, (2 + HC) * HC], "F16"),
            (p + "hc_attn_base.weight", [(2 + HC) * HC], "F32"),
            (p + "hc_attn_scale.weight", [3], "F32"),
            (p + "hc_ffn_fn.weight", [HC * D, (2 + HC) * HC], "F16"),
            (p + "hc_ffn_base.weight", [(2 + HC) * HC], "F32"),
            (p + "hc_ffn_scale.weight", [3], "F32"),
        ]
        r = RATIOS[i]
        coff = 2 if r == 4 else 1
        t += [
            (p + "attn_compressor_kv.weight", [D, coff * DH], "F16"),
            (p + "attn_compressor_gate.weight", [D, coff * DH], "F16"),
            (p + "attn_compressor_ape.weight", [coff * DH, r], "F16"),
            (p + "attn_compressor_norm.weight", [DH], "F32"),
        ]
        if r == 4:
            t += [
                (p + "indexer.proj.weight", [D, IDXH], "F16"),
                (p + "indexer.attn_q_b.weight", [RQ, IDXH * IDXK], "F16"),
                (p + "indexer_compressor_kv.weight", [D, 2 * IDXK], "F16"),
                (p + "indexer_compressor_gate.weight", [D, 2 * IDXK], "F16"),
                (p + "indexer_compressor_ape.weight", [2 * IDXK, r], "F16"),
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
        t.append((p + "ffn_gate_tid2eid.weight", [TOPK, V], "I32") if i < HASH
                 else (p + "exp_probs_b.bias", [NE], "F32"))
    return t


def write_model(path: Path) -> None:
    w = GGUFWriter(str(path), "deepseek4")
    u32 = {"block_count": L, "embedding_length": D, "attention.head_count": HEADS,
           "attention.head_count_kv": 1, "attention.key_length": DH, "attention.value_length": DH,
           "rope.dimension_count": DROPE, "attention.q_lora_rank": RQ,
           "attention.output_group_count": OG, "attention.output_lora_rank": OL,
           "attention.sliding_window": 128, "expert_count": NE, "expert_used_count": TOPK,
           "expert_feed_forward_length": NFF, "expert_shared_count": NSHEXP, "expert_gating_func": 4,
           "hash_layer_count": HASH, "attention.indexer.head_count": IDXH,
           "attention.indexer.key_length": IDXK, "attention.indexer.top_k": 8,
           "vocab_size": V, "context_length": 4096, "nextn_predict_layers": 1,
           "hyper_connection.count": HC, "hyper_connection.sinkhorn_iterations": 20}
    for k, v in u32.items():
        w.add_uint32("deepseek4." + k, v)
    for k, v in {"expert_weights_scale": 1.5, "attention.compress_rope_freq_base": 160000.0,
                 "rope.freq_base": 10000.0, "rope.scaling.factor": 16.0,
                 "rope.scaling.original_context_length": 65536.0, "attention.layer_norm_rms_epsilon": 1e-6,
                 "hyper_connection.epsilon": 1e-6}.items():
        w.add_float32("deepseek4." + k, v)
    w.add_bool("deepseek4.expert_weights_norm", True)
    w.add_string("deepseek4.rope.scaling.type", "yarn")
    w.add_string("tokenizer.ggml.pre", "joyai-llm")
    w.add_array("deepseek4.attention.compress_ratios", [float(r) for r in RATIOS])
    w.add_array("deepseek4.swiglu_clamp_exp", [10.0] * L)

    rng = np.random.default_rng(7)
    geom = {"F16": (np.float16, 2), "F32": (np.float32, 4), "I32": (np.int32, 4)}
    for name, shape, ty in tensor_list():
        n = int(np.prod(shape))
        if ty in geom:
            dt, _ = geom[ty]
            arr = rng.standard_normal(n).astype(dt) if dt != np.int32 else rng.integers(0, NE, n, dtype=np.int32)
            w.add_tensor(name, arr.reshape(shape))
        elif ty == "Q8_0":
            arr = quants.quantize(rng.standard_normal(shape).astype(np.float32), Q.Q8_0)
            w.add_tensor(name, arr, raw_dtype=Q.Q8_0)
        else:
            e, b = {"IQ2_XXS": (256, 66), "Q2_K": (256, 84)}[ty]
            assert n % e == 0 and shape[-1] % e == 0
            arr = rng.integers(0, 256, n // e * b, dtype=np.uint8)
            w.add_tensor(name, arr, raw_shape=tuple(shape[:-1]) + (shape[-1] // e * b,), raw_dtype=Q[ty])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class Ds4Pack(unittest.TestCase):
    def test_pack(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src, out = td / "mini.gguf", td / "pack"
            out.mkdir()
            write_model(src)
            # pretokenize to skip the (separately tested) GGUF tokenizer extraction
            (out / "tokenizer").mkdir()
            (out / "tokenizer" / "vocab.json").write_text("{}", encoding="utf-8")
            (out / "tokenizer" / "chat_template.jinja").write_text("", encoding="utf-8")

            from gguf import GGUFReader
            md = {f.name: f.contents() for f in GGUFReader(str(src)).fields.values()}
            geom = {k: md.get(k) for k in ds4_pack.GEOM_KEYS if k in md}
            self.assertEqual(ds4_pack.pack(src, out, geom), 0)

            self.assertTrue((out / "ds4_pack.json").exists())
            self.assertTrue((out / "dense.bin").exists())

            # native_experts.txt: one line per layer, gate/up IQ2_XXS(16), down Q2_K(10), blob = per-expert bytes
            lines = [l for l in (out / "native_experts.txt").read_text().splitlines() if not l.startswith("#")]
            self.assertEqual(len(lines), L, "one expert line per layer")
            for l in lines:
                f = l.split()
                self.assertEqual((f[1], f[2]), ("16", "10"), "gate/up IQ2_XXS, down Q2_K")
            gate_bytes = D * NFF * NE // 256 * 66 // NE
            down_bytes = NFF * D * NE // 256 * 84 // NE
            self.assertEqual(int(lines[0].split()[4]), gate_bytes * 2 + down_bytes, "per-expert blob size")
            self.assertEqual(int(lines[1].split()[3]), int(lines[0].split()[4]) * NE, "layer 1 offset")

            # index.txt: header + one row per non-expert tensor; native rows are Q8_0 (code_bits 8)
            head, body = (out / "index.txt").read_text().splitlines()[0:2], \
                [l for l in (out / "index.txt").read_text().splitlines() if not l.startswith("#")]
            m = re.match(r"# align (\d+) pool (\d+) tensors (\d+)", head[1])
            pool, ntensors = int(m.group(2)), int(m.group(3))
            self.assertEqual(ntensors, len(body))
            self.assertEqual((out / "dense.bin").stat().st_size, pool, "dense.bin is the arena's pool bytes")
            names = {r.split()[0] for r in body}
            self.assertNotIn("blk.0.ffn_gate_exps.weight", names, "experts are not index rows")
            self.assertIn("token_embd.weight", names)
            natives = [r for r in body if r.split()[9] == "8"]        # code_bits 8 == served natively (Q8_0)
            self.assertTrue(all(r.split()[0] in names for r in natives))
            self.assertIn("blk.1.attn_q_b.weight", {r.split()[0] for r in natives})
            self.assertIn("blk.0.ffn_gate_tid2eid.weight", names, "the I32 hash table is in dense.bin")
            for r in body:
                self.assertEqual(len(r.split()), 19, "every index row is 19 fields")


if __name__ == "__main__":
    unittest.main(verbosity=2)
