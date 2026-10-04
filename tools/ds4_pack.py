#!/usr/bin/env python3
"""tools/ds4_pack.py - the DeepSeek-V4-Flash pack (Phase 1 of the port plan).

    python tools/ds4_pack.py --gguf DeepSeek-V4-Flash-Q2-0731.gguf --out pack/ds4

Produces the same pack layout the engine already loads (docs/pack-format.md, tools/iq_pack.py), so a Phase-2/3
`deepseek4` loader reads it the way the Qwen loader reads an iq pack:

  native_experts.txt   one line per layer: `layer gu_type d_type offset blob_bytes gate_off up_off down_off
                       [shard]` (v3).  DSV4 has a uniform expert shape - gate/up IQ2_XXS, down Q2_K, one blob
                       per expert - so the v3 (single-shard-per-file) form always applies; a layer split across
                       shards names its per-role shard in the v4 `gate,up,down` column, exactly as iq_pack.
  index.txt            one row per non-expert tensor, the format tools/pack_index.py defines.  Float tensors
                       (F16/F32/I32) are in dense.bin; the Q8_0 projections and `token_embd`/`output` are served
                       NATIVELY from the GGUF (the engine reads their rows from the shard), marked with the
                       canonical form the loader expects.
  dense.bin            the float tensors as stored (F16 2 B, F32 4 B, I32 4 B verbatim), aligned to 64 B.
  tokenizer/           exported from the GGUF by tools/strata_tokenizer.py (joyai-llm pre-tokenizer + the
                       model's chat template), never needing the weight shards at run time.
  ds4_pack.json        the artifact the pack was cut from: shards (name/size), the DSV4 geometry keys, the
                       native_experts.txt hash.  A pack without it is not finished.

Unlike the Qwen path there is no `--base`/`--compat-bf16` reuse yet: DSV4 stores every tensor it needs in the
form the engine reads it, so a straight pass is enough.  The expert blobs are NOT materialised (no --experts-bin)
- the 80 GB file lives on an external NVMe and the machine is RAM-bound - so the engine reads the experts from
the GGUF in place (`FileExpertSource::set_gguf`), which is what `--native` without experts.bin already does.

This is the code path for the full pack; running it on the 80 GiB artifact is a Phase-1 residual (the machine
had no free RAM for an 80 GB read at the time).  tools/ds4/test_ds4_pack.py exercises it end to end on a small
synthetic `deepseek4` GGUF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import iq_pack as IQ  # noqa: E402  (Model, tensor_bytes, check_split, converters - all model-agnostic)

ARCH = "deepseek4"
FLOAT_KIND = {"F32": "2", "F16": "5", "I32": "0"}   # dense.bin forms: F32 copy, raw F16 (kind 5), verbatim I32
ROLES = ("gate", "up", "down")
ALIGN = IQ.ALIGN


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith("_exps.weight")


def classify(model: IQ.Model):
    """(dense, native, experts): the tensors of dense.bin, the tensors served from the GGUF, and the expert
    families.  A quantized tensor is native; a float (F16/F32/I32) one is dense.  Anything else is refused,
    so a new type in a future GGUF fails here rather than at token 4000."""
    dense, native, experts, bad = [], [], [], []
    for name, (g, t, mm, p) in model.where.items():
        if is_expert(name):
            experts.append(name)
        elif t.type_name in FLOAT_KIND:
            dense.append(name)
        elif t.type_name == "Q8_0":
            native.append(name)
        else:
            bad.append("%s is %s (no dense form and not Q8_0)" % (name, t.type_name))
    return dense, native, experts, bad


def expert_layout(model: IQ.Model, src: pathlib.Path, n_expert: int):
    """(lines, total_bytes, problems): the native_experts.txt body.  Per layer the blob is
    [gate | up | down] slices, sized from IQ2_XXS gate/up and Q2_K down."""
    T = {n: w[1] for n, w in model.where.items()}
    exps = sorted({int(n.split(".")[1]) for n in T if is_expert(n)})
    if not exps:
        return [], 0, ["the model has no expert tensors"]
    if exps != list(range(len(exps))):
        return [], 0, ["expert layers are not contiguous: %s" % (exps[:8],)]
    lines, offset, problems = [], 0, []
    for l in exps:
        names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for r in ROLES]
        miss = [n for n in names if n not in T]
        if miss:
            problems.append("layer %d: missing %s" % (l, ", ".join(miss)))
            continue
        ts = [T[n] for n in names]
        if any(t.expected_bytes() is None or len(t.shape) != 3 or int(t.shape[2]) != n_expert for t in ts):
            problems.append("layer %d: an expert tensor is not [*, *, %d] of whole blocks" % (l, n_expert))
            continue
        if ts[0].type_name != ts[1].type_name:
            problems.append("layer %d: gate and up are %s / %s" % (l, ts[0].type_name, ts[1].type_name))
            continue
        per = [t.expected_bytes() // n_expert for t in ts]
        blob = sum(per)
        ws = [model.where[n] for n in names]
        files = ["" if w[3] == src else w[3].name for w in ws]
        column = files[0] if len(set(files)) == 1 else ",".join(files)
        line = "%d %d %d %d %d %d %d %d" % (l, ts[0].type_id, ts[2].type_id, offset, blob,
                                            *[w[0].data_start + w[1].offset for w in ws])
        lines.append(line + ("" if not column else " " + column))
        offset += blob * n_expert
    return lines, offset, problems


def write_index(out: pathlib.Path, rows: list, src: pathlib.Path, served: int) -> int:
    at = 0
    for r in rows:                       # r[5] is dst_off
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt.tmp", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/ds4_pack.py (deepseek4) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    (out / "index.txt.tmp").replace(out / "index.txt")
    print("index.txt: %d tensors, %d served natively, arena %.2f GiB" % (len(rows), served, at / 2**30))
    return at


def native_row(name: str, t) -> list:
    """A tensor served from the GGUF: shape only, canonical Q8_0 form (code_bits 8, group 32), like iq_pack."""
    ne0 = int(t.shape[0])
    ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
    return [name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7


def pack(src: pathlib.Path, out: pathlib.Path, geom: dict) -> int:
    model = IQ.Model(src)
    if len(model.paths) > 1:
        print("model shards: " + ", ".join(p.name for p in model.paths))
    dense, native, experts, bad = classify(model)
    if bad:
        for m in bad[:8]:
            print(m)
        return 1
    if not experts:
        print("the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)")
        return 1

    T = {n: w[1] for n, w in model.where.items()}
    n_expert = int(T["blk.0.ffn_gate_inp.weight"].shape[1])
    lines, total, problems = expert_layout(model, src, n_expert)
    if problems:
        for m in problems[:8]:
            print(m)
        return 1

    # ---- dense.bin and the index.  Written to temporaries; the pack is published only when every row is in.
    rows, at = [], 0
    with open(out / "dense.bin.tmp", "wb") as fo:
        for name in dense:
            g, t, mm, p = model.where[name]
            raw = IQ.tensor_bytes(mm, g, t)
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0
            rows.append([name, "0", FLOAT_KIND[t.type_name], str(at), str(len(raw)), "0", str(len(raw)),
                         str(ne0), str(ne1), "0", "0", "1"] + ["0"] * 7)
            fo.write(raw.tobytes())
            pad = (-len(raw)) % ALIGN
            fo.write(b"\0" * pad)
            at += len(raw) + pad
    for name in native:
        rows.append(native_row(name, model.where[name][1]))
    write_index(out, rows, src, len(native))

    (out / "native_experts.txt.tmp").write_text(
        "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
        "(n_expert %d, total %d; absolute offsets in %s, or in the named shard beside it)\n"
        % (n_expert, total, src.name) + "".join(l + "\n" for l in lines),
        encoding="utf-8", newline="\n")
    (out / "dense.bin.tmp").replace(out / "dense.bin")
    (out / "native_experts.txt.tmp").replace(out / "native_experts.txt")

    side = {"schema": 1, "tool": "tools/ds4_pack.py", "architecture": ARCH,
            "shards": [{"name": q.name, "size": s} for q, s in zip(model.paths, model.sizes)],
            "n_expert": n_expert, "expert_bytes": total,
            "native_experts_sha256": hashlib.sha256((out / "native_experts.txt").read_bytes()).hexdigest(),
            "geometry": geom}
    (out / "ds4_pack.json.tmp").write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8")
    (out / "ds4_pack.json.tmp").replace(out / "ds4_pack.json")
    print("dense.bin: %d tensors, %d native, %d expert families, expert arena %.2f GiB"
          % (len(dense), len(native), len(lines), total / 2**30))
    return 0


GEOM_KEYS = [
    "deepseek4.block_count", "deepseek4.embedding_length", "deepseek4.attention.head_count",
    "deepseek4.attention.head_count_kv", "deepseek4.attention.key_length", "deepseek4.rope.dimension_count",
    "deepseek4.attention.q_lora_rank", "deepseek4.attention.output_group_count",
    "deepseek4.attention.output_lora_rank", "deepseek4.attention.sliding_window", "deepseek4.expert_count",
    "deepseek4.expert_used_count", "deepseek4.expert_feed_forward_length", "deepseek4.expert_shared_count",
    "deepseek4.expert_weights_scale", "deepseek4.expert_weights_norm", "deepseek4.expert_gating_func",
    "deepseek4.hash_layer_count", "deepseek4.attention.indexer.head_count",
    "deepseek4.attention.indexer.key_length", "deepseek4.attention.indexer.top_k",
    "deepseek4.attention.compress_ratios", "deepseek4.attention.compress_rope_freq_base",
    "deepseek4.rope.scaling.type", "deepseek4.rope.scaling.factor",
    "deepseek4.rope.scaling.original_context_length", "deepseek4.nextn_predict_layers",
    "deepseek4.hyper_connection.count", "deepseek4.hyper_connection.sinkhorn_iterations",
    "tokenizer.ggml.pre",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = pathlib.Path(a.gguf).absolute()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = IQ.G.GGUFFile(src)
    md = g.metadata
    if md.get("general.architecture") != ARCH:
        print("architecture is '%s', this packer requires '%s'" % (md.get("general.architecture"), ARCH))
        return 1
    geom = {k: md.get(k) for k in GEOM_KEYS if k in md}

    # ---- experts first: a model that cannot be packed is refused before any pack file changes
    rc = pack(src, out, geom)
    if rc:
        return rc
    tok = out / "tokenizer"
    if not (tok / "vocab.json").exists():
        import subprocess
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)
    print("pack complete: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
