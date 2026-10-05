#!/usr/bin/env python3
"""compare_golden.py - compare two directories written by tools/ds4/golden_dump.

Per tensor it reports max-abs difference, relative L2, and cosine similarity.
For the final logits ("result_output") it additionally reports top-1 agreement,
top-5 overlap, and the mean KL divergence of the softmax distributions, treating
directory A as the reference (the llama.cpp oracle). It exits 1 if any metric
breaches its threshold, which makes it usable directly as a phase-3 gate.

    python3 tools/ds4/compare_golden.py bench/.../goldens/p600 ref/.../p600 \
        --cosine 0.9999 --top1 0.99 --kl 0.01

Only the thresholds explicitly requested are enforced; the rest use defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

DTYPES = {"f32": "<f4", "i32": "<i4", "f16": "<f2"}

# defaults: cosine is the tight gate, the rest are sanity bounds
DEFAULT_THRESHOLDS = {
    "max_abs": math.inf,
    "rel_l2": 0.05,
    "cosine": 0.999,
    "top1": 0.99,
    "top5": 0.99,
    "kl": 0.01,
}

LOGIT_NAMES = ("result_output",)
TOKEN_NAMES = ("tokens",)


class GoldenError(Exception):
    pass


def load_manifest(directory):
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        raise GoldenError("no manifest.json in %s" % directory)
    with open(path) as f:
        manifest = json.load(f)
    if isinstance(manifest, dict):
        tensors = manifest.get("tensors", [])
    elif isinstance(manifest, list):
        tensors = manifest
    else:
        raise GoldenError("manifest.json in %s is neither object nor list" % directory)
    by_name = {}
    for entry in tensors:
        by_name[entry["name"]] = entry
    # golden_dump writes the token ids as a sibling "tokens" object, not inside "tensors"
    if isinstance(manifest, dict) and isinstance(manifest.get("tokens"), dict):
        by_name[manifest["tokens"]["name"]] = manifest["tokens"]
    return manifest, by_name


def load_tensor(directory, entry):
    dtype = entry.get("dtype", "f32")
    if dtype not in DTYPES:
        raise GoldenError("unsupported dtype %r for %s" % (dtype, entry["name"]))
    path = os.path.join(directory, entry["file"])
    if not os.path.isfile(path):
        raise GoldenError("missing tensor file %s" % path)
    arr = np.fromfile(path, dtype=DTYPES[dtype])
    shape = tuple(int(d) for d in entry.get("shape", []))
    expected = 1
    for d in shape:
        expected *= d
    if expected != arr.size:
        raise GoldenError(
            "shape %s of %s does not match file size %d" % (list(shape), entry["name"], arr.size)
        )
    return arr.reshape(shape)


def _cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def tensor_metrics(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise GoldenError("size mismatch: %d vs %d" % (a.size, b.size))
    diff = a - b
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff) / (np.linalg.norm(b) + 1e-12)),
        "cosine": _cosine(a, b),
    }


def _softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-300)


def logits_metrics(ref, cand):
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    if cand.ndim == 1:
        cand = cand[:, None]
    if ref.shape != cand.shape:
        raise GoldenError("logits shape mismatch: %s vs %s" % (ref.shape, cand.shape))

    ref1 = ref.argmax(axis=0)
    cand1 = cand.argmax(axis=0)
    top1 = float(np.mean(ref1 == cand1))

    k = min(5, ref.shape[0])
    ref5 = np.argpartition(-ref, k - 1, axis=0)[:k, :]
    cand5 = np.argpartition(-cand, k - 1, axis=0)[:k, :]
    overlaps = [len(set(ref5[:, t]) & set(cand5[:, t])) / k for t in range(ref.shape[1])]
    top5 = float(np.mean(overlaps))

    p = _softmax(ref, axis=0)
    q = _softmax(cand, axis=0)
    kl = float(np.mean(np.sum(p * (np.log(p + 1e-300) - np.log(q + 1e-300)), axis=0)))

    return {"top1": top1, "top5": top5, "kl": kl, "n_positions": int(ref.shape[1])}


def _breaches(name, metrics, thresholds):
    out = []
    if metrics["max_abs"] > thresholds["max_abs"]:
        out.append("%s: max_abs %.6g > %.6g" % (name, metrics["max_abs"], thresholds["max_abs"]))
    if metrics["rel_l2"] > thresholds["rel_l2"]:
        out.append("%s: rel_l2 %.6g > %.6g" % (name, metrics["rel_l2"], thresholds["rel_l2"]))
    if metrics["cosine"] < thresholds["cosine"]:
        out.append("%s: cosine %.6g < %.6g" % (name, metrics["cosine"], thresholds["cosine"]))
    return out


def compare_dirs(dir_a, dir_b, thresholds=None, ignore=(), tensors_diag=False):
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({k: v for k, v in thresholds.items() if v is not None})
    ignore = set(ignore)

    meta_a, ta = load_manifest(dir_a)
    meta_b, tb = load_manifest(dir_b)

    rows = []
    breaches = []
    diagnostics = []  # per-tensor metric breaches when tensors_diag (reported, not failing)
    names = sorted((set(ta) | set(tb)) - ignore)
    for name in names:
        if name in TOKEN_NAMES:
            continue
        if name not in ta:
            breaches.append("%s: missing in reference" % name)
            rows.append({"name": name, "shape": None, "status": "missing in A"})
            continue
        if name not in tb:
            breaches.append("%s: missing in candidate" % name)
            rows.append({"name": name, "shape": None, "status": "missing in B"})
            continue
        a = load_tensor(dir_a, ta[name])
        b = load_tensor(dir_b, tb[name])
        if a.shape != b.shape:
            breaches.append("%s: shape %s != %s" % (name, list(a.shape), list(b.shape)))
            rows.append({"name": name, "shape": list(a.shape), "status": "shape mismatch"})
            continue
        m = tensor_metrics(a, b)
        bad = _breaches(name, m, th)
        (diagnostics if tensors_diag else breaches).extend(bad)
        rows.append({
            "name": name,
            "shape": list(a.shape),
            "max_abs": m["max_abs"],
            "rel_l2": m["rel_l2"],
            "cosine": m["cosine"],
            "status": "FAIL" if bad else "ok",
        })

    # token ids must match exactly
    for name in TOKEN_NAMES:
        if name in ta and name in tb:
            a = load_tensor(dir_a, ta[name])
            b = load_tensor(dir_b, tb[name])
            n_bad = int(np.sum(a.astype(np.int64) != b.astype(np.int64))) if a.shape == b.shape else -1
            if n_bad != 0:
                breaches.append("%s: %d mismatched token ids" % (name, n_bad))
            rows.append({"name": name, "shape": list(a.shape),
                         "status": "ok" if n_bad == 0 else "FAIL"})

    logits = None
    for name in LOGIT_NAMES:
        if name in ta and name in tb:
            a = load_tensor(dir_a, ta[name])
            b = load_tensor(dir_b, tb[name])
            if a.shape != b.shape:
                breaches.append("%s: shape %s != %s" % (name, list(a.shape), list(b.shape)))
                continue
            logits = logits_metrics(a, b)
            logits["name"] = name
            if logits["top1"] < th["top1"]:
                breaches.append("%s: top1 %.6g < %.6g" % (name, logits["top1"], th["top1"]))
            if logits["top5"] < th["top5"]:
                breaches.append("%s: top5 %.6g < %.6g" % (name, logits["top5"], th["top5"]))
            if logits["kl"] > th["kl"]:
                breaches.append("%s: KL %.6g > %.6g" % (name, logits["kl"], th["kl"]))

    report = {
        "ok": not breaches,
        "breaches": breaches,
        "diagnostics": diagnostics,
        "tensors": rows,
        "logits": logits,
        "thresholds": th,
        "meta_a": {k: meta_a[k] for k in ("n_tokens", "all_pos") if isinstance(meta_a, dict) and k in meta_a},
        "meta_b": {k: meta_b[k] for k in ("n_tokens", "all_pos") if isinstance(meta_b, dict) and k in meta_b},
    }
    return report["ok"], report


def format_report(report, limit=60):
    lines = []
    lines.append("%-34s %-16s %12s %12s %12s %6s" %
                 ("tensor", "shape", "max_abs", "rel_l2", "cosine", "status"))
    lines.append("-" * 98)
    for row in report["tensors"][:limit]:
        shape = "x".join(str(d) for d in row.get("shape") or []) or "-"
        if "cosine" in row:
            lines.append("%-34s %-16s %12.4g %12.4g %12.7f %6s" %
                         (row["name"], shape, row["max_abs"], row["rel_l2"], row["cosine"], row["status"]))
        else:
            lines.append("%-34s %-16s %12s %12s %12s %6s" %
                         (row["name"], shape, "-", "-", "-", row.get("status", "-")))
    if len(report["tensors"]) > limit:
        lines.append("... %d more tensors" % (len(report["tensors"]) - limit))

    lg = report["logits"]
    if lg is not None:
        lines.append("")
        lines.append("logits %s: top1=%.6g top5=%.6g KL=%.6g over %d positions" %
                     (lg["name"], lg["top1"], lg["top5"], lg["kl"], lg["n_positions"]))

    lines.append("")
    if report["ok"]:
        lines.append("PASS: all tensors within thresholds" if not report.get("diagnostics")
                     else "PASS: logits/tokens/shapes gate (per-tensor metrics are diagnostics)")
    if report.get("diagnostics"):
        lines.append("diagnostic (not gating): %d per-tensor breach(es)" % len(report["diagnostics"]))
    else:
        lines.append("FAIL: %d breach(es)" % len(report["breaches"]))
        for b in report["breaches"][:limit]:
            lines.append("  - " + b)
        if len(report["breaches"]) > limit:
            lines.append("  ... %d more" % (len(report["breaches"]) - limit))
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description="compare two golden_dump directories")
    p.add_argument("reference", help="reference (oracle) directory")
    p.add_argument("candidate", help="candidate directory")
    p.add_argument("--max-abs", type=float, default=None, help="max allowed |a-b| (default: inf)")
    p.add_argument("--rel-l2", type=float, default=None, help="max allowed relative L2 (default: 0.05)")
    p.add_argument("--cosine", type=float, default=None, help="min allowed cosine (default: 0.999)")
    p.add_argument("--top1", type=float, default=None, help="min logits top-1 agreement (default: 0.99)")
    p.add_argument("--top5", type=float, default=None, help="min logits top-5 overlap (default: 0.99)")
    p.add_argument("--kl", type=float, default=None, help="max mean KL (default: 0.01)")
    p.add_argument("--ignore", action="append", default=[], help="tensor name to skip (repeatable)")
    p.add_argument("--tensors-diag", action="store_true",
                   help="per-tensor max_abs/rel_l2/cosine breaches are reported but do not fail; the gate is the "
                        "logits (top1/top5/KL), token ids, missing tensors and shapes (FINDINGS s9)")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    thresholds = {
        "max_abs": args.max_abs,
        "rel_l2": args.rel_l2,
        "cosine": args.cosine,
        "top1": args.top1,
        "top5": args.top5,
        "kl": args.kl,
    }
    try:
        ok, report = compare_dirs(args.reference, args.candidate, thresholds, args.ignore, args.tensors_diag)
    except GoldenError as exc:
        print("compare_golden: %s" % exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, default=float))
    else:
        print(format_report(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
