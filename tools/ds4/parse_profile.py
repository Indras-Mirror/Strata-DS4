#!/usr/bin/env python3
"""parse_profile.py - turn profiler output for llama.cpp config B into a
per-decode-token split: GPU attention, GPU dense/device-MoE, PCIe copies, and
the host-side residual (CPU expert matmuls + sync/idle).

Input comes from profile_decode.sh:
  --kern-csv   nsys stats --report cuda_gpu_kern_sum --format csv
  --mem-csv    nsys stats --report cuda_gpu_mem_time_sum --format csv
  --timings    the server's /v1/chat/completions "timings" object (JSON)
  --perf-text  perf report --stdio (fallback when nsys is unavailable)

Assumptions, stated because they bound the claim: the trace covers the whole
server lifetime (warmup + prefill + decode); kernel totals are divided by the
number of generated tokens, so the per-token GPU figures are a mild
overestimate. The host residual is total - GPU - copies and lumps CPU expert
matmuls together with sync/idle; the CPU-only split is only available from the
perf fallback.
"""

import argparse
import csv
import json
import re
import sys

ATTN_RE = re.compile(r"flash_attn|soft_max|softmax|rope|indexer|lightning|dsv4_hc|lid_", re.I)
MOE_RE = re.compile(r"expert|moe", re.I)
DENSE_RE = re.compile(
    r"mul_mat|mmq|gemm|gemv|cublas|rms_norm|norm|silu|swiglu|gelu|relu|"
    r"argmax|topk|top_k|scale|concat|get_rows|set_rows|cpy|cont|dup|sum|add|mul|"
    r"pad|repeat|permute|view|reshape",
    re.I,
)


def _field(row, *needles):
    for key, value in row.items():
        if key is None:
            continue
        for needle in needles:
            if needle.lower() in key.lower():
                return value
    return None


def _read_nsys_csv(path, header_keyword):
    rows = []
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    start = None
    for i, line in enumerate(lines):
        if header_keyword.lower() in line.lower():
            start = i
            break
    if start is None:
        return rows
    reader = csv.DictReader(lines[start:])
    for row in reader:
        if any(v is None or str(v).strip() == "" for v in row.values()):
            continue
        rows.append(row)
    return rows


def load_kernels(path):
    out = []
    for row in _read_nsys_csv(path, "Kernel Name"):
        name = _field(row, "Kernel Name")
        total = _field(row, "Total Time")
        if name is None or total is None:
            continue
        try:
            ns = float(total)
        except ValueError:
            continue
        out.append((name, ns))
    return out


def load_copies(path):
    out = []
    for row in _read_nsys_csv(path, "Operation"):
        op = _field(row, "Operation")
        total = _field(row, "Total Time")
        if op is None or total is None:
            continue
        try:
            ns = float(total)
        except ValueError:
            continue
        out.append((op, ns))
    return out


def classify(name):
    if MOE_RE.search(name):
        return "moe"
    if ATTN_RE.search(name):
        return "attention"
    if DENSE_RE.search(name):
        return "dense"
    return "other"


def parse_timings(path):
    with open(path) as f:
        t = json.load(f)
    pps = t.get("predicted_per_second")
    n = t.get("predicted_n")
    if not pps:
        raise SystemExit("parse_profile: timings has no predicted_per_second (%s)" % path)
    n = int(n) if n else None
    return float(pps), n, t


def print_gpu_table(kernels, copies, total_ms, decode_tokens):
    buckets = {"attention": 0.0, "dense": 0.0, "moe": 0.0, "other": 0.0}
    for name, ns in kernels:
        buckets[classify(name)] += ns

    def mpt(ns):
        return ns / 1e6 / decode_tokens

    gpu_ms = mpt(sum(buckets.values()))
    copy_ms = mpt(sum(ns for _, ns in copies))
    residual_ms = max(0.0, total_ms - gpu_ms - copy_ms)

    def pct(ms):
        return 100.0 * ms / total_ms if total_ms else 0.0

    print("config B decode profile (%d decode tokens, %.2f ms/token measured)"
          % (decode_tokens, total_ms))
    print("%-30s %10s %8s" % ("phase", "ms/token", "% token"))
    print("-" * 52)
    for label, ms in [
        ("GPU attention", mpt(buckets["attention"])),
        ("GPU dense", mpt(buckets["dense"])),
        ("GPU device MoE", mpt(buckets["moe"])),
        ("GPU other", mpt(buckets["other"])),
        ("GPU kernels total", gpu_ms),
        ("GPU<->host copies (PCIe)", copy_ms),
        ("CPU experts + host (residual)", residual_ms),
    ]:
        print("%-30s %10.3f %7.1f%%" % (label, ms, pct(ms)))
    print("-" * 52)
    print("%-30s %10.3f %7.1f%%" % ("total (measured)", total_ms, 100.0))
    print("%-30s %10.3f %7.1f%%" % ("GPU busy", gpu_ms, pct(gpu_ms)))
    print("")
    print("note: 'CPU experts + host' = total - GPU - copies; it includes sync/idle and")
    print("      host-side launch overhead. Use the perf fallback for the CPU split.")
    if copies:
        print("")
        print("copies by operation:")
        for op, ns in sorted(copies, key=lambda x: -x[1]):
            print("  %-34s %10.3f ms/token" % (op.strip("[]"), mpt(ns)))


def print_perf_table(path, total_ms, decode_tokens):
    rows = []
    line_re = re.compile(r"^\s*([\d.]+)%\s+(\S+)\s+(\S+)\s+(.*)$")
    with open(path) as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            pct = float(m.group(1))
            symbol = m.group(4).strip()
            rows.append((pct, symbol))
    if not rows:
        print("parse_profile: no perf rows found in %s" % path)
        return
    total_pct = sum(p for p, _ in rows) or 1.0
    expert = sum(p for p, s in rows if re.search(r"vec_dot|gemm|iq2|q2_k|q8", s, re.I))
    print("perf fallback: CPU symbol attribution (no GPU trace available)")
    print("total sample weight %.1f%%; expert-ish quant kernels %.1f%% of samples"
          % (total_pct, expert))
    print("%-8s %-10s %s" % ("%samples", "ms/token", "symbol"))
    for pct, symbol in rows[:25]:
        print("%7.2f%% %9.3f  %s" % (pct, total_ms * pct / 100.0, symbol[:90]))


def main(argv=None):
    p = argparse.ArgumentParser(description="summarise a decode profile")
    p.add_argument("--kern-csv")
    p.add_argument("--mem-csv")
    p.add_argument("--timings", required=True)
    p.add_argument("--perf-text")
    p.add_argument("--decode-tokens", type=int, default=None)
    args = p.parse_args(argv)

    pps, n, _ = parse_timings(args.timings)
    total_ms = 1000.0 / pps
    decode_tokens = args.decode_tokens or n or 1

    if args.kern_csv:
        kernels = load_kernels(args.kern_csv)
        copies = load_copies(args.mem_csv) if args.mem_csv else []
        if not kernels:
            print("parse_profile: no kernels parsed from %s" % args.kern_csv, file=sys.stderr)
            return 2
        print_gpu_table(kernels, copies, total_ms, decode_tokens)
        return 0
    if args.perf_text:
        print_perf_table(args.perf_text, total_ms, decode_tokens)
        return 0
    print("parse_profile: pass --kern-csv (nsys) or --perf-text (perf)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
