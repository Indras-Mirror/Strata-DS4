#!/usr/bin/env python3
"""compare_logits.py - the real engine's logits gate: ds4_generate --dump-logits vs a llama.cpp golden.

    python3 tools/ds4/compare_logits.py bench/ds4-2026-10-05/goldens/p600 engine_p600.f32 [--top1 1] [--kl 0.01]

The golden's result_output.f32 holds the LAST prompt position's logits (one vector); ds4_generate -n 1
--dump-logits writes the logits it samples the first token from - the same position.  Metrics are
compare_golden.py's (top-1 agreement, top-5 overlap, KL(golden || engine)) plus the cosine and the top-10 ids.
Exit 1 when top-1 differs or KL >= the limit.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_golden as CG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden_dir")
    ap.add_argument("engine_f32")
    ap.add_argument("--kl", type=float, default=0.01)
    a = ap.parse_args()
    ref = np.fromfile(Path(a.golden_dir) / "result_output.f32", dtype=np.float32)
    cand = np.fromfile(a.engine_f32, dtype=np.float32)
    if ref.shape != cand.shape:
        # the golden may hold several positions; the last vocab-sized row is the last position
        if ref.size % cand.size == 0:
            ref = ref[-cand.size:]
        else:
            print(f"shape mismatch: golden {ref.shape} vs engine {cand.shape}")
            return 1
    m = CG.logits_metrics(ref[:, None], cand[:, None]) if hasattr(CG, "logits_metrics") else None
    if m is None:
        p = np.exp(ref - ref.max()); p /= p.sum()
        q = np.exp(cand - cand.max()); q /= q.sum()
        m = {"top1": float(ref.argmax() == cand.argmax()),
             "top5": len(set(np.argsort(-ref)[:5]) & set(np.argsort(-cand)[:5])) / 5,
             "kl": float(np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300))))}
    cos = float(ref @ cand / (np.linalg.norm(ref) * np.linalg.norm(cand)))
    print(f"golden {a.golden_dir}: top1 {'match' if m['top1'] == 1.0 else 'DIFFERENT'} "
          f"(golden {int(ref.argmax())}, engine {int(cand.argmax())}), top5 overlap {m['top5']:.2f}, "
          f"KL {m['kl']:.6f}, cosine {cos:.6f}")
    print("  golden top10:", list(map(int, np.argsort(-ref)[:10])))
    print("  engine top10:", list(map(int, np.argsort(-cand)[:10])))
    ok = m["top1"] == 1.0 and m["kl"] < a.kl
    print("GATE", "PASS" if ok else "FAIL", f"(top-1 match, KL < {a.kl})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
