#!/usr/bin/env python3
"""tools/ds4/test_tokenizer_golden.py - the Phase-1 tokenizer gate.

Checks the joyai-llm pre-tokenizer (tools/strata_tokenizer.py) against ids frozen from
llama.cpp-master-rebase `llama-tokenize --ids` in tests/ds4/tokenizer_golden.json.  The three golden prompts are
the constraint from the packet: their ids must match llama.cpp exactly.  The corpus adds the boundaries that a
prompt made of English prose never reaches (Han/kana runs, digit runs, contractions, punctuation-glued words,
control bytes, emoji, non-ASCII whitespace).

Run:  python tools/ds4/test_tokenizer_golden.py
      python -m unittest tools/ds4/test_tokenizer_golden.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))

import strata_tokenizer as ST  # noqa: E402

GOLDEN = REPO / "tests/ds4/tokenizer_golden.json"
MODEL = os.environ.get("DS4_GGUF", "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf")


def _golden() -> dict:
    if not GOLDEN.exists():
        raise unittest.SkipTest("no %s; run tools/ds4/make_tokenizer_goldens.py" % GOLDEN)
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@unittest.skipUnless(pathlib.Path(MODEL).exists(), "GGUF %s not present" % MODEL)
class TokenizerGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = _golden()
        cls.tk = ST.Tokenizer.from_gguf(MODEL)
        assert cls.tk.pre == "joyai-llm", cls.tk.pre

    def test_golden_prompt_ids(self):
        """The three prompts must tokenize to exactly llama.cpp's ids."""
        for name, want in self.g["prompts"].items():
            got = self.tk.encode((REPO / "tools/ds4/golden_prompts" / (name + ".txt"))
                                 .read_text(encoding="utf-8"))
            self.assertEqual(got, want, "%s: %d ids, want %d" % (name, len(got), len(want)))

    def test_corpus_blob_and_lines(self):
        """A diverse corpus as one blob and line by line - the pre-split boundaries."""
        blob = "\n".join(self.g["corpus"]) + "\n"
        self.assertEqual(self.tk.encode(blob), self.g["corpus_ids"], "corpus blob")
        for i, (line, want) in enumerate(zip(self.g["corpus"], self.g["corpus_line_ids"])):
            self.assertEqual(self.tk.encode(line), want, "corpus line %d: %r" % (i, line[:40]))

    def test_round_trip(self):
        for s in self.g["corpus"]:
            self.assertEqual(self.tk.decode(self.tk.encode(s)), s, "round trip %r" % s[:40])


if __name__ == "__main__":
    unittest.main(verbosity=2)
