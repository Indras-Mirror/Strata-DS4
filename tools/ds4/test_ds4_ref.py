#!/usr/bin/env python3
"""tools/ds4/test_ds4_ref.py - structural regression test for ds4_ref on the miniature model.

The real 80 GiB artifact needs the whole machine and the shared lock; this drives ds4_ref over the
miniature `deepseek4` GGUF from make_mini_gguf.py (2 layers: ratio-4 CSA + indexer + hash routing, ratio-128
HCA) and checks the reference path end to end: every layer runs, every captured tensor is emitted with the
golden_dump manifest shape convention and is finite, and the attention output names follow the layer ratio.

It is a wiring/regression gate, not a numeric-oracle gate - the numeric gate is compare_golden.py against
bench/ds4-2026-10-05/goldens/p*, which needs the real weights.

    python3 tools/ds4/test_ds4_ref.py            # needs build-ds4/ds4_ref
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / "build-ds4" / "ds4_ref"
sys.path.insert(0, str(HERE))

import make_mini_gguf  # noqa: E402


def run_ref(td: Path, n_tokens: int) -> dict:
    gguf = td / "mini.gguf"
    make_mini_gguf.write_model(gguf)
    toks = td / "tokens.i32"
    toks.write_bytes(struct.pack("<%di" % n_tokens, *[i % make_mini_gguf.V for i in range(n_tokens)]))
    out = td / "ref"
    p = subprocess.run([str(BIN), "-m", str(gguf), "--tokens", str(toks), "--out", str(out), "-t", "4"],
                       capture_output=True, text=True)
    assert p.returncode == 0, "ds4_ref failed:\n%s\n%s" % (p.stdout[-2000:], p.stderr[-2000:])
    return json.loads((out / "manifest.json").read_text()), out


class Ds4Ref(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BIN.exists():
            raise unittest.SkipTest("build-ds4/ds4_ref not built")

    def check(self, manifest: dict, out: Path, n_tokens: int, expect_indexer: bool):
        D, HC, V = make_mini_gguf.D, make_mini_gguf.HC, make_mini_gguf.V
        HEADS, DH = make_mini_gguf.HEADS, make_mini_gguf.DH
        by_name = {t["name"]: t for t in manifest["tensors"]}

        # every global and per-layer capture the golden set has, for both layers
        for base in ("hc_init", "hc_head", "result_norm", "result_output"):
            self.assertIn(base, by_name, base)
        for il in range(make_mini_gguf.L):
            for base in ("hc_attn_pre", "attn_norm", "attn_out", "hc_attn_post", "hc_ffn_pre",
                         "ffn_norm", "ffn_moe_out", "ffn_shexp", "ffn_out", "l_last"):
                self.assertIn("%s-%d" % (base, il), by_name, base)

        # the attention output name is chosen by the layer's compression ratio
        self.assertIn("attn_csa_lid-0", by_name)          # layer 0 is ratio 4
        self.assertIn("attn_hca-1", by_name)              # layer 1 is ratio 128
        self.assertNotIn("attn_raw-0", by_name)

        # golden shape convention: feat dims + one row (last token) unless --all-pos
        self.assertEqual(by_name["l_last-0"]["shape"], [D, HC, 1])
        self.assertEqual(by_name["hc_init"]["shape"], [D, HC, 1])
        self.assertEqual(by_name["attn_csa_lid-0"]["shape"], [HEADS * DH, 1])
        self.assertEqual(by_name["result_output"]["shape"], [V, 1])

        # every tensor is the declared size and finite
        for t in manifest["tensors"]:
            a = np.fromfile(out / t["file"], dtype="<f4")
            self.assertEqual(a.size, int(np.prod(t["shape"])), t["name"])
            self.assertTrue(np.isfinite(a).all(), "%s has non-finite values" % t["name"])

        # the prompt actually reached the model
        self.assertEqual(manifest["n_tokens"], n_tokens)
        if expect_indexer:
            self.assertGreater(n_tokens // 4, make_mini_gguf.IDXK and 8)   # > top_k blocks -> indexer built

    def test_short_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            m, out = run_ref(td, 12)
            self.check(m, out, 12, expect_indexer=False)

    def test_long_prompt_exercises_hca_and_indexer(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            m, out = run_ref(td, 300)     # 75 CSA blocks (> top_k 8 -> indexer) and 2 HCA blocks
            self.check(m, out, 300, expect_indexer=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
