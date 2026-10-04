#!/usr/bin/env python3
"""Unit tests for tools/ds4/compare_golden.py using synthetic numpy dirs.

    python3 tools/ds4/test_compare_golden.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_golden  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "compare_golden.py")

DTYPE_TO_NP = {"f32": "<f4", "i32": "<i4", "f16": "<f2"}


def write_golden(directory, tensors, tokens=None, extra=None):
    """tensors: {name: (np.ndarray, dtype)}; writes files + manifest.json."""
    os.makedirs(directory, exist_ok=True)
    entries = []
    for name, (arr, dtype) in tensors.items():
        suffix = ".f32" if dtype == "f32" else (".i32" if dtype == "i32" else ".f16")
        fname = name + suffix
        arr.astype(DTYPE_TO_NP[dtype]).tofile(os.path.join(directory, fname))
        entries.append({
            "name": name, "layer": 0, "shape": list(arr.shape),
            "dtype": dtype, "file": fname, "sha256": "test",
        })
    if tokens is not None:
        tokens.astype("<i4").tofile(os.path.join(directory, "tokens.i32"))
        entries.append({
            "name": "tokens", "layer": -1, "shape": list(tokens.shape),
            "dtype": "i32", "file": "tokens.i32", "sha256": "test",
        })
    manifest = {"tool": "test", "n_tokens": 8, "all_pos": False, "tensors": entries}
    if extra:
        manifest.update(extra)
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def tiny_logits(top_index=0, vocab=32, positions=4):
    logits = np.zeros((vocab, positions), dtype=np.float32)
    logits[top_index, :] = 10.0
    return logits


class CompareGoldenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = os.path.join(self.tmp.name, "a")
        self.b = os.path.join(self.tmp.name, "b")

    def _base(self):
        return {
            "attn_norm-0": (np.linspace(-1, 1, 64, dtype=np.float32), "f32"),
            "l_last-0": (np.random.RandomState(0).randn(16, 4, 8).astype(np.float32), "f32"),
            "result_output": (tiny_logits(0), "f32"),
        }

    def test_identical_passes(self):
        tensors = self._base()
        write_golden(self.a, tensors, tokens=np.arange(8, dtype=np.int32))
        write_golden(self.b, tensors, tokens=np.arange(8, dtype=np.int32))
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertTrue(ok, report["breaches"])
        self.assertEqual(report["breaches"], [])
        self.assertAlmostEqual(report["logits"]["top1"], 1.0)
        self.assertLess(report["logits"]["kl"], 1e-9)

    def test_cosine_breach_fails(self):
        tensors = self._base()
        write_golden(self.a, tensors)
        flipped = dict(tensors)
        flipped["attn_norm-0"] = (-tensors["attn_norm-0"][0], "f32")
        write_golden(self.b, flipped)
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertFalse(ok)
        self.assertTrue(any("cosine" in b for b in report["breaches"]), report["breaches"])

    def test_rel_l2_and_max_abs_thresholds(self):
        tensors = self._base()
        write_golden(self.a, tensors)
        noisy = dict(tensors)
        noisy["attn_norm-0"] = (tensors["attn_norm-0"][0] * 1.5, "f32")
        write_golden(self.b, noisy)
        # scaled by 1.5: cosine is still 1, but rel_l2 = 0.5 > default 0.05
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertFalse(ok)
        self.assertTrue(any("rel_l2" in x for x in report["breaches"]), report["breaches"])
        # relaxing rel_l2 lets it through
        ok2, _ = compare_golden.compare_dirs(self.a, self.b, {"rel_l2": 1.0})
        self.assertTrue(ok2)
        ok3, _ = compare_golden.compare_dirs(self.a, self.b, {"max_abs": 1e-9})
        self.assertFalse(ok3)

    def test_logits_top1_breach(self):
        tensors = self._base()
        write_golden(self.a, tensors)
        bad = dict(tensors)
        bad["result_output"] = (tiny_logits(7), "f32")
        write_golden(self.b, bad)
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertFalse(ok)
        self.assertEqual(report["logits"]["top1"], 0.0)
        self.assertTrue(any("top1" in x for x in report["breaches"]), report["breaches"])

    def test_missing_tensor_fails(self):
        tensors = self._base()
        write_golden(self.a, tensors)
        write_golden(self.b, {k: v for k, v in tensors.items() if k != "l_last-0"})
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertFalse(ok)
        self.assertTrue(any("missing in candidate" in x for x in report["breaches"]))

    def test_token_ids_compared_exactly(self):
        tensors = self._base()
        write_golden(self.a, tensors, tokens=np.arange(8, dtype=np.int32))
        write_golden(self.b, tensors, tokens=np.array([0, 1, 2, 3, 4, 5, 6, 99], dtype=np.int32))
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertFalse(ok)
        self.assertTrue(any("token" in x for x in report["breaches"]), report["breaches"])

    def test_shape_mismatch_fails(self):
        tensors = self._base()
        write_golden(self.a, tensors)
        bad = dict(tensors)
        bad["attn_norm-0"] = (np.zeros(32, dtype=np.float32), "f32")
        write_golden(self.b, bad)
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertFalse(ok)
        self.assertTrue(any("shape" in x for x in report["breaches"]))

    def test_cli_exit_codes(self):
        tensors = self._base()
        write_golden(self.a, tensors, tokens=np.arange(8, dtype=np.int32))
        write_golden(self.b, tensors, tokens=np.arange(8, dtype=np.int32))
        same = subprocess.run([sys.executable, SCRIPT, self.a, self.b],
                              capture_output=True, text=True)
        self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
        self.assertIn("PASS", same.stdout)

        bad = dict(tensors)
        bad["result_output"] = (tiny_logits(9), "f32")
        write_golden(self.b, bad, tokens=np.arange(8, dtype=np.int32))
        diff = subprocess.run([sys.executable, SCRIPT, self.a, self.b, "--top1", "0.99"],
                              capture_output=True, text=True)
        self.assertEqual(diff.returncode, 1, diff.stdout + diff.stderr)
        self.assertIn("FAIL", diff.stdout)

    def test_tokens_as_sibling_object(self):
        # golden_dump writes "tokens" as a sibling of "tensors"; make sure it is compared
        tensors = self._base()
        write_golden(self.a, tensors)
        write_golden(self.b, tensors)
        for d, ids in ((self.a, np.arange(8, dtype=np.int32)),
                       (self.b, np.array([0, 1, 2, 3, 4, 5, 6, 99], dtype=np.int32))):
            ids.astype("<i4").tofile(os.path.join(d, "tokens.i32"))
            with open(os.path.join(d, "manifest.json")) as f:
                manifest = json.load(f)
            manifest["tensors"] = [e for e in manifest["tensors"] if e["name"] != "tokens"]
            manifest["tokens"] = {"name": "tokens", "layer": -1, "shape": [8],
                                  "dtype": "i32", "file": "tokens.i32", "sha256": "test"}
            with open(os.path.join(d, "manifest.json"), "w") as f:
                json.dump(manifest, f)
        ok, report = compare_golden.compare_dirs(self.a, self.b)
        self.assertFalse(ok)
        self.assertTrue(any("token" in x for x in report["breaches"]), report["breaches"])

    def test_ignore_option(self):
        tensors = self._base()
        write_golden(self.a, tensors)
        bad = dict(tensors)
        bad["attn_norm-0"] = (-tensors["attn_norm-0"][0], "f32")
        write_golden(self.b, bad)
        ok, _ = compare_golden.compare_dirs(self.a, self.b, ignore=["attn_norm-0"])
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
