// ds4_ref: the Phase-3 *reference* forward pass for DeepSeek-V4 (GGUF arch `deepseek4`).
//
// Slow and correct, on purpose.  It is a plain ggml graph executed on the CPU backend: no CUDA, no
// captured graphs, no fused kernels.  Weights are read straight out of the GGUF mmap (zero copy) and the
// quantized experts go through ggml-cpu's own IQ2_XXS / Q2_K matmul, so the numerics are the closest
// thing to the llama.cpp oracle that Strata can produce without being llama.cpp.
//
// It emits activations in exactly the golden_dump format (manifest.json + <name>.f32 + tokens.i32, last
// token position by default, every position with --all-pos), so tools/ds4/compare_golden.py compares a
// candidate directory against bench/ds4-2026-10-05/goldens/p* directly:
//
//   tools/ds4/ds4_ref -m $DS4_GGUF --tokens bench/ds4-2026-10-05/goldens/p64/tokens.i32 --out /tmp/ref-p64
//   python3 tools/ds4/compare_golden.py bench/ds4-2026-10-05/goldens/p64 /tmp/ref-p64 --cosine 0.9999
//
// Tensor names and shapes match golden_dump's captured node set (docs/ds4/DSV4_ARCH_SPEC.md s2):
//   per layer: hc_attn_pre, attn_norm, attn_raw|attn_csa_lid|attn_hca, attn_out, hc_attn_post,
//              hc_ffn_pre, ffn_norm, ffn_moe_out, ffn_shexp, ffn_out, l_last
//   global   : hc_init, hc_head, result_norm, result_output
// The semantics of each node follow llama.cpp-master-rebase src/models/deepseek4.cpp exactly:
//   * build_hc_pre / build_hc_sinkhorn / build_hc_post / build_hc_head (deepseek4.cpp:287-466)
//   * the raw / CSA / HCA attention sites (deepseek4.cpp:753-953) and the output LoRA (deepseek4.cpp:1298-1314)
//   * the overlap / HCA compressor (deepseek4.cpp:468-586), fed from the token sequence directly rather
//     than through the cache's ring state (a full prefill makes every compressor source a local token)
//   * the MoE router / experts (llama-graph.cpp:1943-2373, DEEPSEEK4 swiglu-clamp branch)
//
// Build: CMake target `ds4_ref` (links ggml + ggml-cpu + ggml-base), see tools/ds4/build_ds4_ref.sh.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "strata/artifact/ds4_geometry.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata;

static constexpr float NEG_INF = -INFINITY;

// ------------------------------------------------------------------ small utilities

static std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char) c;
        }
    }
    return o;
}

static std::string sha256_file(const std::string & path) {
    std::string cmd = "sha256sum \"" + path + "\" 2>/dev/null";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) return "unavailable";
    char buf[128] = {0};
    const char * r = std::fgets(buf, sizeof buf, p);
    pclose(p);
    if (!r) return "unavailable";
    std::string s(buf);
    const size_t sp = s.find(' ');
    return sp == std::string::npos ? std::string("unavailable") : s.substr(0, sp);
}

static bool read_i32_file(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    if (n <= 0 || n % 4 != 0) return false;
    out.resize((size_t) n / 4);
    in.read((char *) out.data(), n);
    return (bool) in;
}

// ------------------------------------------------------------------ weights (zero-copy ggml tensors over the GGUF mmap)

struct Weights {
    ggml_context * ctx = nullptr;                 // no_alloc: tensor structs only
    std::vector<ggml_backend_buffer_t> bufs;      // one per shard, wrapping the mmap
    std::map<std::string, ggml_tensor *> t;
    int64_t n_tensors = 0;

    ggml_tensor * get(const std::string & name) const {
        auto it = t.find(name);
        if (it == t.end()) { std::fprintf(stderr, "[ds4_ref] missing tensor %s\n", name.c_str()); std::exit(2); }
        return it->second;
    }
    ~Weights() {
        for (auto * b : bufs) ggml_backend_buffer_free(b);
        if (ctx) ggml_free(ctx);
    }
};

static bool load_weights(const GgufModel & model, Weights & w) {
    ggml_init_params ip = { /*mem_size*/ 1024ull * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    w.ctx = ggml_init(ip);
    if (!w.ctx) return false;
    ggml_set_no_alloc(w.ctx, true);

    for (size_t s = 0; s < model.size(); ++s) {
        const GgufFile & sh = model.shard(s);
        const auto & tensors = sh.tensors();
        if (tensors.empty()) continue;
        const uint8_t * any = sh.tensor_data(tensors[0]);
        uint8_t * base = const_cast<uint8_t *>(any) - sh.data_start() - tensors[0].offset;
        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, (size_t) sh.file_size());
        if (!buf) { std::fprintf(stderr, "[ds4_ref] cannot wrap %s\n", sh.path().c_str()); return false; }
        w.bufs.push_back(buf);
        for (const auto & ti : tensors) {
            if (!model.in_bounds(ti, s)) { std::fprintf(stderr, "[ds4_ref] %s out of bounds\n", ti.name.c_str()); return false; }
            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (size_t d = 0; d < ti.shape.size() && d < GGML_MAX_DIMS; ++d) ne[d] = (int64_t) ti.shape[d];
            ggml_tensor * t = ggml_new_tensor(w.ctx, (ggml_type) ti.type, (int) ti.shape.size(), ne);
            if (!t) { std::fprintf(stderr, "[ds4_ref] cannot create %s\n", ti.name.c_str()); return false; }
            ggml_set_name(t, ti.name.c_str());
            if (ggml_backend_tensor_alloc(buf, t, const_cast<uint8_t *>(sh.tensor_data(ti))) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[ds4_ref] cannot bind %s\n", ti.name.c_str());
                return false;
            }
            if (w.t.count(ti.name)) { std::fprintf(stderr, "[ds4_ref] duplicate tensor %s\n", ti.name.c_str()); return false; }
            w.t[ti.name] = t;
            w.n_tensors++;
        }
    }
    return true;
}

// ------------------------------------------------------------------ dumped tensors

struct Dumped {
    std::string name;
    int il = -1;
    std::vector<int64_t> feat;   // dims before the token axis
    std::vector<float> data;     // feat * rows, row-major (token slowest)
    int64_t rows = 0;
};

struct DumpSet {
    std::string dir;
    bool all_pos = false;
    std::map<std::string, Dumped> by_name;
};

// `p` holds feat*nt floats (C-order, token slowest).  Copies the requested rows with the golden shape.
static void dump_put(DumpSet & ds, const std::string & base, int il, const std::vector<int64_t> & feat,
                     const float * p, int64_t nt) {
    int64_t nfeat = 1;
    for (int64_t d : feat) nfeat *= d;
    Dumped d;
    d.name = (il >= 0) ? base + "-" + std::to_string(il) : base;
    d.il = il;
    d.feat = feat;
    if (ds.all_pos) {
        d.rows = nt;
        d.data.assign(p, p + (size_t) nfeat * nt);
    } else {
        d.rows = 1;
        d.data.assign(p + (size_t) nfeat * (nt - 1), p + (size_t) nfeat * nt);
    }
    ds.by_name[d.name] = std::move(d);
}

static bool write_dumps(DumpSet & ds, const std::vector<int32_t> & tokens) {
    std::error_code ec;
    fs::create_directories(ds.dir, ec);
    if (ec) { std::fprintf(stderr, "[ds4_ref] cannot create %s\n", ds.dir.c_str()); return false; }

    {
        const std::string p = ds.dir + "/tokens.i32";
        FILE * f = std::fopen(p.c_str(), "wb");
        if (!f) return false;
        for (int32_t v : tokens) std::fwrite(&v, sizeof v, 1, f);
        std::fclose(f);
    }

    std::vector<const Dumped *> ordered;
    for (auto & kv : ds.by_name) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(), [](const Dumped * a, const Dumped * b) {
        if (a->il != b->il) return a->il < b->il;
        return a->name < b->name;
    });

    for (const Dumped * t : ordered) {
        const std::string p = ds.dir + "/" + t->name + ".f32";
        FILE * f = std::fopen(p.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "[ds4_ref] cannot write %s\n", p.c_str()); return false; }
        if (std::fwrite(t->data.data(), sizeof(float), t->data.size(), f) != t->data.size()) return false;
        std::fclose(f);
    }

    const std::string mp = ds.dir + "/manifest.json";
    FILE * mf = std::fopen(mp.c_str(), "wb");
    if (!mf) return false;
    std::fprintf(mf, "{\n");
    std::fprintf(mf, "  \"tool\": \"ds4_ref\",\n");
    std::fprintf(mf, "  \"model\": \"\",\n");
    std::fprintf(mf, "  \"prompt_file\": \"\",\n");
    std::fprintf(mf, "  \"n_tokens\": %zu,\n", tokens.size());
    std::fprintf(mf, "  \"add_bos\": false,\n");
    std::fprintf(mf, "  \"all_pos\": %s,\n", ds.all_pos ? "true" : "false");
    std::fprintf(mf, "  \"n_gpu_layers\": 0,\n");
    std::fprintf(mf, "  \"moe_expert_cache_slots\": 0,\n");
    std::fprintf(mf, "  \"kv_type_k\": \"f32\",\n");
    std::fprintf(mf, "  \"kv_type_v\": \"f32\",\n");
    std::fprintf(mf, "  \"tensors\": [\n");
    bool first = true;
    for (const Dumped * t : ordered) {
        std::string shape;
        for (int64_t d : t->feat) shape += std::to_string(d) + ", ";
        shape += std::to_string(t->rows);
        const std::string file = t->name + ".f32";
        std::fprintf(mf, "%s    { \"name\": \"%s\", \"layer\": %d, \"shape\": [%s], \"dtype\": \"f32\", "
                         "\"file\": \"%s\", \"sha256\": \"%s\" }",
                     first ? "" : ",\n", json_escape(t->name).c_str(), t->il, shape.c_str(),
                     json_escape(file).c_str(), sha256_file(ds.dir + "/" + file).c_str());
        first = false;
    }
    std::fprintf(mf, "\n  ],\n");
    std::fprintf(mf, "  \"tokens\": { \"name\": \"tokens\", \"layer\": -1, \"shape\": [%zu], "
                     "\"dtype\": \"i32\", \"file\": \"tokens.i32\", \"sha256\": \"%s\" }\n",
                 tokens.size(), sha256_file(ds.dir + "/tokens.i32").c_str());
    std::fprintf(mf, "}\n");
    std::fclose(mf);
    std::printf("[ds4_ref] wrote %zu tensors to %s\n", ordered.size(), ds.dir.c_str());
    return true;
}

// ------------------------------------------------------------------ graph scaffolding

using Bytes = std::shared_ptr<std::vector<uint8_t>>;

struct InputFill { ggml_tensor * t; Bytes data; };

// the per-compression-ratio tensors every layer of that ratio shares
struct CompInputs {
    int64_t ratio = 0;
    int64_t n_blocks = 0;
    std::shared_ptr<std::vector<uint8_t>> mask_f16;   // [n_blocks, nt] F16, -inf where not visible
    std::shared_ptr<std::vector<uint8_t>> mask_f32;   // [n_blocks, nt] F32, same
    std::shared_ptr<std::vector<uint8_t>> idx;        // I32 gather indices (overlap: prev-half ++ cur-half)
};

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_gallocr_t allo = nullptr;
    Ds4Geometry * g = nullptr;
    Weights * w = nullptr;
    mutable std::vector<InputFill> fills;

    ggml_tensor * W(const std::string & name) const { return w->get(name); }
    ggml_tensor * L(int il, const std::string & suffix) const {
        return w->get("blk." + std::to_string(il) + "." + suffix);
    }

    ggml_tensor * input_type(ggml_type ty, std::initializer_list<int64_t> ne, const void * data, size_t bytes, const char * nm = "input") const {
        ggml_tensor * t = nullptr;
        switch (ne.size()) {
            case 1: t = ggml_new_tensor_1d(ctx, ty, *ne.begin()); break;
            case 2: { auto it = ne.begin(); t = ggml_new_tensor_2d(ctx, ty, it[0], it[1]); } break;
            case 3: { auto it = ne.begin(); t = ggml_new_tensor_3d(ctx, ty, it[0], it[1], it[2]); } break;
            case 4: { auto it = ne.begin(); t = ggml_new_tensor_4d(ctx, ty, it[0], it[1], it[2], it[3]); } break;
            default: std::abort();
        }
        ggml_set_input(t);
        ggml_set_name(t, nm);
        auto buf = std::make_shared<std::vector<uint8_t>>(bytes);
        std::memcpy(buf->data(), data, bytes);
        fills.push_back({t, buf});
        return t;
    }
    ggml_tensor * input(std::initializer_list<int64_t> ne, const void * data, size_t bytes, const char * nm = "input") const {
        return input_type(GGML_TYPE_F32, ne, data, bytes, nm);
    }
    ggml_tensor * input_i32(std::initializer_list<int64_t> ne, const void * data, size_t bytes, const char * nm = "input") const {
        return input_type(GGML_TYPE_I32, ne, data, bytes, nm);
    }
    ggml_tensor * input_f16(std::initializer_list<int64_t> ne, const void * data, size_t bytes, const char * nm = "input") const {
        return input_type(GGML_TYPE_F16, ne, data, bytes, nm);
    }

    ggml_tensor * view1(ggml_tensor * a, int64_t n0, size_t off) const {
        return ggml_view_1d(ctx, a, n0, off);
    }
    ggml_tensor * view2(ggml_tensor * a, int64_t n0, int64_t n1, size_t nb1, size_t off) const {
        return ggml_view_2d(ctx, a, n0, n1, nb1, off);
    }

    // hc affine: x*scale + base  (deepseek4.cpp:279-285)
    ggml_tensor * hc_affine(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) const {
        return ggml_add(ctx, ggml_mul(ctx, x, scale), base);
    }

    // deepseek4.cpp:314-349
    ggml_tensor * sinkhorn(ggml_tensor * comb) const {
        const int64_t hc = g->hc;
        comb = ggml_soft_max(ctx, comb);
        ggml_tensor * eps = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), (float) g->hc_eps);
        comb = ggml_add(ctx, comb, eps);
        auto norm_cols = [&]() {
            ggml_tensor * cd = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));
            ggml_tensor * cs = ggml_add(ctx, ggml_sum_rows(ctx, cd), eps);
            cs = ggml_permute(ctx, cs, 1, 0, 2, 3);
            comb = ggml_div(ctx, comb, cs);
        };
        norm_cols();
        for (int64_t i = 1; i < g->hc_sinkhorn_iters; ++i) {
            ggml_tensor * rs = ggml_add(ctx, ggml_sum_rows(ctx, comb), eps);
            comb = ggml_div(ctx, comb, rs);
            norm_cols();
        }
        (void) hc;
        return comb;
    }

    // deepseek4.cpp:287-312: weighted mean of the hc streams under `pre`
    ggml_tensor * hc_mean(ggml_tensor * x, ggml_tensor * pre) const {
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        ggml_tensor * acc = nullptr;
        for (int64_t h = 0; h < hc; ++h) {
            ggml_tensor * xh = view2(x, D, nt, x->nb[2], (size_t) h * x->nb[1]);
            ggml_tensor * wh = view2(pre, 1, nt, pre->nb[1], (size_t) h * pre->nb[0]);
            ggml_tensor * cur = ggml_mul(ctx, xh, wh);
            acc = acc ? ggml_add(ctx, acc, cur) : cur;
        }
        return acc;
    }

    // deepseek4.cpp:351-407 (unfused): returns the stream mean, sets pre/post/comb
    ggml_tensor * hc_pre(ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
                         ggml_tensor ** post, ggml_tensor ** comb) const {
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[2];
        const int64_t mix_dim = g->hc_mix_dim();
        ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc * D, nt);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, (float) g->rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_norm);   // [mix_dim, nt]

        ggml_tensor * scale_pre  = view1(hc_scale, 1, 0);
        ggml_tensor * scale_post = view1(hc_scale, 1, 1);
        ggml_tensor * base_pre   = view1(hc_base, hc, 0);
        ggml_tensor * base_post  = view1(hc_base, hc, (size_t) hc * 4);

        ggml_tensor * pre = view2(mixes, hc, nt, mixes->nb[1], 0);
        pre = ggml_sigmoid(ctx, hc_affine(pre, scale_pre, base_pre));
        pre = ggml_scale_bias(ctx, pre, 1.0f, (float) g->hc_eps);

        *post = view2(mixes, hc, nt, mixes->nb[1], (size_t) hc * mixes->nb[0]);
        *post = ggml_sigmoid(ctx, hc_affine(*post, scale_post, base_post));
        *post = ggml_scale(ctx, *post, 2.0f);

        ggml_tensor * scale_comb = view1(hc_scale, 1, 2);
        ggml_tensor * base_comb  = view1(hc_base, hc * hc, (size_t) 2 * hc * 4);
        *comb = view2(mixes, hc * hc, nt, mixes->nb[1], (size_t) 2 * hc * mixes->nb[0]);
        *comb = hc_affine(*comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx, *comb, hc, hc, nt);
        *comb = sinkhorn(*comb);
        (void) mix_dim;

        return hc_mean(x, pre);
    }

    // deepseek4.cpp:409-444
    ggml_tensor * hc_post(ggml_tensor * x, ggml_tensor * residual, ggml_tensor * post, ggml_tensor * comb) const {
        const int64_t D = g->n_embd, hc = g->hc, nt = x->ne[1];
        ggml_tensor * out = nullptr;
        for (int64_t dst = 0; dst < hc; ++dst) {
            ggml_tensor * post_dst = view2(post, 1, nt, post->nb[1], (size_t) dst * post->nb[0]);
            ggml_tensor * cur = ggml_mul(ctx, x, post_dst);
            for (int64_t src = 0; src < hc; ++src) {
                ggml_tensor * res_src = view2(residual, D, nt, residual->nb[2], (size_t) src * residual->nb[1]);
                ggml_tensor * cs = view2(comb, 1, nt, comb->nb[2],
                        (size_t) dst * comb->nb[0] + (size_t) src * comb->nb[1]);
                cur = ggml_add(ctx, cur, ggml_mul(ctx, res_src, cs));
            }
            cur = ggml_reshape_3d(ctx, cur, D, 1, nt);
            out = out ? ggml_concat(ctx, out, cur, 1) : cur;
        }
        return out;
    }

    // generic RMSNorm-with-weight (build_norm, LLM_NORM_RMS)
    ggml_tensor * rms_w(ggml_tensor * x, ggml_tensor * wgt) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, (float) g->rms_eps), wgt);
    }

    // ggml_rope_ext + the nope/rope offset used throughout deepseek4.cpp
    ggml_tensor * rope(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int mode, int n_ctx,
                       float base, float fscale, float ext, float attn, float bfast, float bslow) const {
        a = ggml_rope_ext(ctx, a, pos, nullptr, (int) n_dims, mode, n_ctx, base, fscale, ext, attn, bfast, bslow);
        return ggml_rope_set_offset(a, (int) (g->d_head - n_dims));
    }
    ggml_tensor * rope_back(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int mode, int n_ctx,
                            float base, float fscale, float ext, float attn, float bfast, float bslow) const {
        a = ggml_rope_ext_back(ctx, a, pos, nullptr, (int) n_dims, mode, n_ctx, base, fscale, ext, attn, bfast, bslow);
        return ggml_rope_set_offset(a, (int) (g->d_head - n_dims));
    }

    // orthonormal Walsh-Hadamard rotation over contiguous `n`-blocks (llama-impl.h:57-75, llama-kv-cache.cpp:24)
    ggml_tensor * hadamard(ggml_tensor * cur, ggml_tensor * rot) const {
        const int64_t n = rot->ne[0];
        ggml_tensor * res = ggml_is_contiguous(cur)
            ? ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n)
            : ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
        res = ggml_mul_mat(ctx, rot, res);
        return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    }
    ggml_tensor * rope_at(ggml_tensor * a, ggml_tensor * pos, int64_t n_dims, int64_t off, int n_ctx,
                          float base, float fscale, float ext, float attn, float bf, float bs) const {
        a = ggml_rope_ext(ctx, a, pos, nullptr, (int) n_dims, GGML_ROPE_TYPE_NORMAL, n_ctx, base, fscale, ext, attn, bf, bs);
        return ggml_rope_set_offset(a, (int) off);
    }

    // one attention block (raw / CSA + lightning indexer / HCA) at layer il; returns the attention output
    // before the output projection
    ggml_tensor * attention(int il, ggml_tensor * xn, ggml_tensor * qr, ggml_tensor * inp_pos,
                            const CompInputs & ci, int64_t ratio,
                            const std::shared_ptr<std::vector<uint8_t>> & raw_mask,
                            ggml_tensor * rot_idx, int64_t top_k, ggml_tensor ** attn_out_name) const;
};

// ------------------------------------------------------------------ the attention block

ggml_tensor * Graph::attention(int il, ggml_tensor * xn, ggml_tensor * qr, ggml_tensor * inp_pos,
                               const CompInputs & ci, int64_t ratio,
                               const std::shared_ptr<std::vector<uint8_t>> & raw_mask,
                               ggml_tensor * rot_idx, int64_t top_k, ggml_tensor ** attn_out_name) const {
    const int64_t D = g->n_embd, nt = xn->ne[1];
    const int64_t d_h = g->d_head, dh_rope = g->d_rope;
    const int64_t n_head = g->n_head, n_groups = g->o_groups;
    const int64_t n_heads_group = n_head / n_groups;
    const int64_t o_group_dim = n_heads_group * d_h;
    const int64_t o_lora = g->o_lora;
    const int64_t n_blocks = ci.n_blocks;
    (void) qr;

    const bool comp = ratio != 0;
    const float freq_base  = comp ? (float) g->compress_rope_base : (float) g->rope_freq_base;
    const float freq_scale = comp ? (float) (1.0 / g->rope_factor) : 1.0f;
    const float ext_factor = comp ? 1.0f : 0.0f;
    const float attn_factor = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * std::log(1.0f / freq_scale));
    const float beta_fast  = comp ? (float) g->yarn_beta_fast : 0.0f;
    const float beta_slow  = comp ? (float) g->yarn_beta_slow : 0.0f;
    const int   n_ctx      = comp ? (int) g->rope_orig_ctx : 0;

    // q latent
    ggml_tensor * qr2 = ggml_mul_mat(ctx, L(il, "attn_q_a.weight"), xn);
    qr2 = rms_w(qr2, L(il, "attn_q_a_norm.weight"));
    ggml_tensor * q = ggml_mul_mat(ctx, L(il, "attn_q_b.weight"), qr2);
    q = ggml_reshape_3d(ctx, q, d_h, n_head, nt);
    q = ggml_rms_norm(ctx, q, (float) g->rms_eps);
    q = rope(q, inp_pos, dh_rope, GGML_ROPE_TYPE_NORMAL, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    // kv latent (K == V)
    ggml_tensor * kv = ggml_mul_mat(ctx, L(il, "attn_kv.weight"), xn);
    kv = rms_w(kv, L(il, "attn_kv_a_norm.weight"));
    kv = ggml_reshape_3d(ctx, kv, d_h, 1, nt);
    kv = rope(kv, inp_pos, dh_rope, GGML_ROPE_TYPE_NORMAL, n_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    ggml_tensor * k_all = kv;
    ggml_tensor * mask = input_f16({nt, nt}, raw_mask->data(), raw_mask->size(), "raw_mask");

    if (comp && n_blocks > 0) {
        std::vector<int32_t> spos((size_t) nt);
        for (int64_t t = 0; t < nt; ++t) spos[(size_t) t] = (int32_t) (t % ratio);
        ggml_tensor * state_pos = input_i32({nt}, spos.data(), spos.size() * 4, "state_pos");

        const int64_t coff = (ratio == 4) ? 2 : 1;
        const int64_t state_dim = coff * d_h;

        // overlap (ratio 4) compressor shared by the CSA value keys and the indexer keys
        auto overlap_compress = [&](const char * prefix, int64_t head_dim) -> ggml_tensor * {
            ggml_tensor * st_kv = ggml_mul_mat(ctx, L(il, std::string(prefix) + "_kv.weight"), xn);
            ggml_tensor * st_sc = ggml_mul_mat(ctx, L(il, std::string(prefix) + "_gate.weight"), xn);
            ggml_tensor * ape_rows = ggml_get_rows(ctx, L(il, std::string(prefix) + "_ape.weight"), state_pos);
            st_sc = ggml_add(ctx, st_sc, ape_rows);
            ggml_tensor * zrow = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * head_dim, 1), 0.0f);
            ggml_tensor * nrow = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * head_dim, 1), NEG_INF);
            ggml_tensor * src_kv = ggml_concat(ctx, st_kv, zrow, 1);
            ggml_tensor * src_sc = ggml_concat(ctx, st_sc, nrow, 1);

            const int64_t n_read = ratio * n_blocks;
            ggml_tensor * idx_all = input_i32({2 * ratio * n_blocks}, ci.idx->data(), ci.idx->size(), "overlap_idx");
            ggml_tensor * kv_rows = ggml_get_rows(ctx, src_kv, idx_all);
            ggml_tensor * sc_rows = ggml_get_rows(ctx, src_sc, idx_all);
            ggml_tensor * kv_prev = ggml_cont(ctx, view2(kv_rows, head_dim, n_read, kv_rows->nb[1], 0));
            kv_prev = ggml_reshape_3d(ctx, kv_prev, head_dim, ratio, n_blocks);
            ggml_tensor * sc_prev = ggml_cont(ctx, view2(sc_rows, head_dim, n_read, sc_rows->nb[1], 0));
            sc_prev = ggml_reshape_3d(ctx, sc_prev, head_dim, ratio, n_blocks);
            size_t off_k = (size_t) n_read * kv_rows->nb[1] + ggml_row_size(kv_rows->type, head_dim);
            ggml_tensor * kv_cur = ggml_cont(ctx, view2(kv_rows, head_dim, n_read, kv_rows->nb[1], off_k));
            kv_cur = ggml_reshape_3d(ctx, kv_cur, head_dim, ratio, n_blocks);
            size_t off_s = (size_t) n_read * sc_rows->nb[1] + ggml_row_size(sc_rows->type, head_dim);
            ggml_tensor * sc_cur = ggml_cont(ctx, view2(sc_rows, head_dim, n_read, sc_rows->nb[1], off_s));
            sc_cur = ggml_reshape_3d(ctx, sc_cur, head_dim, ratio, n_blocks);

            ggml_tensor * values = ggml_concat(ctx, kv_prev, kv_cur, 1);
            ggml_tensor * scores = ggml_concat(ctx, sc_prev, sc_cur, 1);
            values = ggml_cont(ctx, ggml_permute(ctx, values, 1, 0, 2, 3));
            scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
            ggml_tensor * wts = ggml_soft_max(ctx, scores);
            ggml_tensor * c = ggml_sum_rows(ctx, ggml_mul(ctx, values, wts));
            c = ggml_cont(ctx, ggml_permute(ctx, c, 1, 0, 2, 3));           // [head_dim, 1, n_blocks]
            c = rms_w(c, L(il, std::string(prefix) + "_norm.weight"));
            std::vector<int32_t> cpos((size_t) n_blocks);
            for (int64_t b = 0; b < n_blocks; ++b) cpos[(size_t) b] = (int32_t) (ratio * b);
            ggml_tensor * cpos_t = input_i32({n_blocks}, cpos.data(), cpos.size() * 4, "overlap_pos");
            return rope_at(c, cpos_t, dh_rope, head_dim - dh_rope, n_ctx, (float) g->compress_rope_base,
                           freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        };

        ggml_tensor * comp_k = nullptr;
        ggml_tensor * cmask_f16 = input_f16({n_blocks, nt}, ci.mask_f16->data(), ci.mask_f16->size(), "comp_mask_f16");

        if (ratio == 4) {
            comp_k = overlap_compress("attn_compressor", d_h);

            // lightning indexer: only needed when there are more visible blocks than top_k
            if (n_blocks > top_k) {
                const int64_t idx_k = g->indexer_key_dim, idx_h = g->indexer_n_head;
                ggml_tensor * lid_k = ggml_cont(ctx, overlap_compress("indexer_compressor", idx_k));
                if (rot_idx) lid_k = hadamard(lid_k, rot_idx);

                ggml_tensor * iq = ggml_mul_mat(ctx, L(il, "indexer.attn_q_b.weight"), qr2);   // [idx_h*idx_k, nt]
                iq = ggml_reshape_3d(ctx, iq, idx_k, idx_h, nt);
                iq = rope_at(iq, inp_pos, dh_rope, idx_k - dh_rope, n_ctx, (float) g->compress_rope_base,
                             freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                iq = hadamard(iq, rot_idx);
                ggml_tensor * iw = ggml_mul_mat(ctx, L(il, "indexer.proj.weight"), xn);       // [idx_h, nt]
                iw = ggml_scale(ctx, iw, 1.0f / std::sqrt((float) (idx_k * idx_h)));

                ggml_tensor * qp = ggml_permute(ctx, iq, 0, 2, 1, 3);   // [idx_k, nt, idx_h]
                ggml_tensor * kp = ggml_permute(ctx, lid_k, 0, 2, 1, 3); // [idx_k, n_blocks, 1]
                ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);            // [n_blocks, nt, idx_h]
                kq = ggml_cont(ctx, ggml_permute(ctx, kq, 2, 1, 0, 3));  // [idx_h, nt, n_blocks]
                ggml_tensor * sc = ggml_relu(ctx, kq);
                sc = ggml_mul(ctx, sc, ggml_reshape_4d(ctx, iw, idx_h, nt, 1, 1));
                sc = ggml_sum_rows(ctx, sc);                             // [1, nt, n_blocks]
                sc = ggml_cont(ctx, ggml_permute(ctx, sc, 2, 1, 0, 3));  // [n_blocks, nt]

                ggml_tensor * vis_f = input({n_blocks, nt}, ci.mask_f32->data(), ci.mask_f32->size(), "comp_mask_f32");
                sc = ggml_add(ctx, sc, vis_f);
                const int64_t ntk = std::min(n_blocks, top_k);
                ggml_tensor * tk = ggml_cont(ctx, ggml_top_k(ctx, sc, (int) ntk));   // [ntk, nt]

                // mask = -inf, zeroed at the top-k rows, then ANDed with visibility (deepseek4.cpp:724-751)
                ggml_tensor * a = ggml_fill(ctx, vis_f, NEG_INF);
                a = ggml_view_4d(ctx, a, 1, n_blocks, nt, 1, a->nb[0], a->nb[1], a->nb[2], 0);
                ggml_tensor * zv = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, ntk, nt), 0.0f);
                ggml_tensor * m = ggml_set_rows(ctx, a, zv, tk);
                m = ggml_view_4d(ctx, m, m->ne[1], m->ne[2], 1, 1, m->nb[2], m->nb[3], m->nb[3], 0);
                m = ggml_add(ctx, m, vis_f);
                cmask_f16 = ggml_cast(ctx, m, GGML_TYPE_F16);
            }
        } else {
            // HCA: one block == `ratio` consecutive tokens, no overlap
            ggml_tensor * st_kv = ggml_mul_mat(ctx, L(il, "attn_compressor_kv.weight"), xn);
            ggml_tensor * st_sc = ggml_mul_mat(ctx, L(il, "attn_compressor_gate.weight"), xn);
            ggml_tensor * ape_rows = ggml_get_rows(ctx, L(il, "attn_compressor_ape.weight"), state_pos);
            st_sc = ggml_add(ctx, st_sc, ape_rows);
            ggml_tensor * idx_all = input_i32({ratio * n_blocks}, ci.idx->data(), ci.idx->size(), "hca_idx");
            ggml_tensor * kv_rows = ggml_get_rows(ctx, st_kv, idx_all);
            ggml_tensor * sc_rows = ggml_get_rows(ctx, st_sc, idx_all);
            ggml_tensor * kv3 = ggml_reshape_3d(ctx, kv_rows, d_h, ratio, n_blocks);
            ggml_tensor * sc3 = ggml_reshape_3d(ctx, sc_rows, d_h, ratio, n_blocks);
            ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, kv3, 1, 0, 2, 3));
            ggml_tensor * scores = ggml_cont(ctx, ggml_permute(ctx, sc3, 1, 0, 2, 3));
            ggml_tensor * wts = ggml_soft_max(ctx, scores);
            ggml_tensor * c = ggml_sum_rows(ctx, ggml_mul(ctx, values, wts));
            c = ggml_cont(ctx, ggml_permute(ctx, c, 1, 0, 2, 3));
            c = rms_w(c, L(il, "attn_compressor_norm.weight"));
            std::vector<int32_t> cpos((size_t) n_blocks);
            for (int64_t b = 0; b < n_blocks; ++b) cpos[(size_t) b] = (int32_t) (ratio * b);
            ggml_tensor * cpos_t = input_i32({n_blocks}, cpos.data(), cpos.size() * 4, "hca_pos");
            comp_k = rope_at(c, cpos_t, dh_rope, d_h - dh_rope, n_ctx, (float) g->compress_rope_base,
                             freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        }

        (void) state_dim;
        k_all = ggml_concat(ctx, kv, comp_k, 2);
        mask = ggml_concat(ctx, mask, cmask_f16, 0);
    }

    // attention: K == V, per-head sink bias.  flash-attn wants q = [d_h, n_tokens, n_head],
    // k/v = [d_h, n_kv, n_head_kv] (llama-graph.cpp:2687-2713) and an F16 mask (llama-graph.cpp:2808).
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k_all, 0, 2, 1, 3);
    ggml_tensor * kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
    ggml_tensor * vf = ggml_cast(ctx, kp, GGML_TYPE_F16);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kf, vf, mask, 1.0f / std::sqrt((float) d_h), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, L(il, "attn_sinks.weight"));
    *attn_out_name = out;

    // de-RoPE then grouped output LoRA
    out = ggml_reshape_3d(ctx, out, d_h, n_head, nt);
    out = rope_back(out, inp_pos, dh_rope, GGML_ROPE_TYPE_NORMAL, n_ctx, freq_base, freq_scale, ext_factor,
                    attn_factor, beta_fast, beta_slow);
    out = ggml_reshape_3d(ctx, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    // the GGUF stores Wo_a 2-D [o_group_dim, o_lora*o_groups]; llama.cpp creates it 3-D
    // [o_group_dim, o_lora, n_groups] (deepseek4.cpp:119), which is what the grouped mul_mat needs
    ggml_tensor * wo_a = L(il, "attn_output_a.weight");
    if (ggml_n_dims(wo_a) == 2) wo_a = ggml_reshape_3d(ctx, wo_a, o_group_dim, o_lora, n_groups);
    ggml_tensor * oa = ggml_mul_mat(ctx, wo_a, out);
    oa = ggml_permute(ctx, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx, oa, o_lora * n_groups, nt);
    return ggml_mul_mat(ctx, L(il, "attn_output_b.weight"), oa);
}

// ------------------------------------------------------------------ main

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path, tokens_path, out_dir;
    bool all_pos = false;
    int  n_threads = 8;
    int  max_layer = 1 << 20;
    bool do_hca = true;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", a.c_str()); std::exit(1); } return argv[++i]; };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--all-pos") all_pos = true;
        else if (a == "-t" || a == "--threads") n_threads = std::atoi(next().c_str());
        else if (a == "--max-layer") max_layer = std::atoi(next().c_str());
        else if (a == "--no-hca") do_hca = false;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || tokens_path.empty() || out_dir.empty()) {
        std::fprintf(stderr, "usage: ds4_ref -m model.gguf --tokens tokens.i32 --out dir [--all-pos] [-t N] [--max-layer N]\n");
        return 1;
    }

    std::vector<int32_t> tokens;
    if (!read_i32_file(tokens_path, tokens) || tokens.empty()) {
        std::fprintf(stderr, "error: cannot read token ids from %s\n", tokens_path.c_str());
        return 1;
    }
    const int64_t nt = (int64_t) tokens.size();
    std::printf("[ds4_ref] model=%s tokens=%lld threads=%d all_pos=%d max_layer=%d\n",
                model_path.c_str(), (long long) nt, n_threads, (int) all_pos, max_layer);

    GgufModel model = GgufModel::open(model_path);
    Ds4Geometry g;
    std::string err = ds4_read_geometry(model, g);
    if (!err.empty()) { std::fprintf(stderr, "error: geometry: %s\n", err.c_str()); return 1; }
    if (!(err = check_ds4_model(model, g)).empty()) {
        std::fprintf(stderr, "error: tensor map: %s\n", err.c_str()); return 1;
    }
    std::printf("[ds4_ref] geometry D=%lld L=%lld hc=%lld experts=%lld top%lld d_h=%lld rope=%lld groups=%lld\n",
                (long long) g.n_embd, (long long) g.n_layer_all, (long long) g.hc, (long long) g.n_expert,
                (long long) g.n_expert_used, (long long) g.d_head, (long long) g.d_rope, (long long) g.o_groups);

    Weights w;
    if (!load_weights(model, w)) { std::fprintf(stderr, "error: weight load failed\n"); return 1; }
    std::printf("[ds4_ref] bound %lld tensors zero-copy from %zu shard(s)\n",
                (long long) w.n_tensors, model.size());

    const int64_t D = g.n_embd, hc = g.hc, n_head = g.n_head, d_h = g.d_head, dh_rope = g.d_rope;
    (void) n_head;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { std::fprintf(stderr, "error: no CPU backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu, n_threads);
    ggml_gallocr_t allo = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    Graph G;
    G.ctx = w.ctx; G.cpu = cpu; G.allo = allo; G.g = &g; G.w = &w;

    DumpSet ds; ds.dir = out_dir; ds.all_pos = all_pos;

    auto run = [&](ggml_cgraph * gf) -> bool {
        if (!ggml_gallocr_alloc_graph(allo, gf)) { std::fprintf(stderr, "[ds4_ref] gallocr failed\n"); return false; }
        for (auto & f : G.fills) {
            if (f.t->buffer == nullptr || f.t->data == nullptr) {
                std::fprintf(stderr, "[ds4_ref] note: input %s was not placed in the graph, skipped\n",
                             f.t->name[0] ? f.t->name : "(unnamed)");
                continue;
            }
            ggml_backend_tensor_set(f.t, f.data->data(), 0, std::min(f.data->size(), ggml_nbytes(f.t)));
        }
        if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[ds4_ref] compute failed\n"); return false;
        }
        return true;
    };

    // ---------------- embedded host state + per-layer dumps
    std::vector<float> hc_state((size_t) D * hc * nt, 0.0f);

    // initial embedding in its own tiny graph (dump hc_init)
    {
        G.fills.clear();
        ggml_tensor * ids = ggml_new_tensor_1d(w.ctx, GGML_TYPE_I32, nt);
        ggml_set_input(ids); ggml_set_name(ids, "tokens");
        auto tbuf = std::make_shared<std::vector<uint8_t>>((size_t) nt * 4);
        std::memcpy(tbuf->data(), tokens.data(), (size_t) nt * 4);
        G.fills.push_back({ids, tbuf});

        ggml_tensor * emb = ggml_get_rows(w.ctx, w.get("token_embd.weight"), ids);   // [D, nt] f16
        emb = ggml_cast(w.ctx, emb, GGML_TYPE_F32);
        ggml_tensor * x = ggml_reshape_3d(w.ctx, emb, D, 1, nt);
        x = ggml_repeat_4d(w.ctx, x, D, hc, nt, 1);
        ggml_set_output(x);
        ggml_set_name(x, "hc_init");

        ggml_cgraph * gf = ggml_new_graph_custom(w.ctx, 256, false);
        ggml_build_forward_expand(gf, x);
        if (!run(gf)) return 1;
        std::memcpy(hc_state.data(), x->data, hc_state.size() * 4);
        dump_put(ds, "hc_init", -1, {D, hc}, (const float *) x->data, nt);
    }

    const int64_t n_layer = std::min<int64_t>(g.n_layer(), max_layer);
    const int64_t n_swa = g.n_swa;
    const int mode = GGML_ROPE_TYPE_NORMAL;

    // raw SWA mask, shared by every layer
    std::shared_ptr<std::vector<uint8_t>> raw_mask = std::make_shared<std::vector<uint8_t>>((size_t) nt * nt * 2);
    {
        ggml_fp16_t * m = (ggml_fp16_t *) raw_mask->data();
        for (int64_t t = 0; t < nt; ++t)
            for (int64_t s = 0; s < nt; ++s)
                m[t * nt + s] = ggml_fp32_to_fp16((s <= t && t - s < n_swa) ? 0.0f : NEG_INF);
    }

    // per-ratio compressor inputs (masks + gather indices), shared by every layer of that ratio
    auto make_comp = [&](int64_t ratio, CompInputs & ci) {
        ci.ratio = ratio;
        ci.n_blocks = nt / ratio;
        if (ci.n_blocks <= 0) return;
        const int64_t nb = ci.n_blocks;
        ci.mask_f16 = std::make_shared<std::vector<uint8_t>>((size_t) nb * nt * 2);
        ci.mask_f32 = std::make_shared<std::vector<uint8_t>>((size_t) nb * nt * 4);
        ggml_fp16_t * m16 = (ggml_fp16_t *) ci.mask_f16->data();
        float * m32 = (float *) ci.mask_f32->data();
        for (int64_t t = 0; t < nt; ++t)
            for (int64_t b = 0; b < nb; ++b) {
                const float v = (b < (t + 1) / ratio) ? 0.0f : NEG_INF;
                m16[b * nt + t] = ggml_fp32_to_fp16(v);
                m32[b * nt + t] = v;
            }
        const int64_t per = (ratio == 4) ? 2 * ratio : ratio;
        ci.idx = std::make_shared<std::vector<uint8_t>>((size_t) per * nb * 4);
        int32_t * p = (int32_t *) ci.idx->data();
        if (ratio == 4) {
            // [ all prev halves | all cur halves ], prev/cur are `ratio` tokens each
            int64_t k = 0;
            for (int64_t b = 0; b < nb; ++b)
                for (int64_t j = 0; j < ratio; ++j) {
                    const int64_t t = ratio * b - ratio + j;
                    p[k++] = (t < 0) ? (int32_t) nt : (int32_t) t;   // nt == the appended zero row
                }
            for (int64_t b = 0; b < nb; ++b)
                for (int64_t j = 0; j < ratio; ++j) p[k++] = (int32_t) (ratio * b + j);
        } else {
            int64_t k = 0;
            for (int64_t b = 0; b < nb; ++b)
                for (int64_t j = 0; j < ratio; ++j) p[k++] = (int32_t) (ratio * b + j);
        }
    };

    CompInputs ci4, ci128;
    make_comp(4, ci4);
    make_comp(128, ci128);
    if (!do_hca) ci128.n_blocks = 0;
    std::printf("[ds4_ref] blocks: csa(4)=%lld hca(128)=%lld\n",
                (long long) ci4.n_blocks, (long long) ci128.n_blocks);

    // orthonormal Walsh-Hadamard matrix for the lightning indexer (llama-kv-cache.cpp:24-60)
    std::vector<float> rot_data;
    if (ci4.n_blocks > g.indexer_top_k) {
        const int n = (int) g.indexer_key_dim;
        rot_data.assign((size_t) n * n, 0.0f);
        rot_data[0] = 1.0f / std::sqrt((float) n);
        for (int s = 1; s < n; s *= 2)
            for (int i = 0; i < s; ++i)
                for (int j = 0; j < s; ++j) {
                    const float v = rot_data[i * n + j];
                    rot_data[(i + s) * n + j] = v;
                    rot_data[i * n + (j + s)] = v;
                    rot_data[(i + s) * n + (j + s)] = -v;
                }
    }

    // ---------------- layers
    ggml_set_no_alloc(w.ctx, true);
    for (int64_t il = 0; il < n_layer; ++il) {
        G.fills.clear();
        const int64_t ratio = (size_t) il < g.compress_ratios.size() ? g.compress_ratios[(size_t) il] : 0;

        // inputs
        ggml_tensor * xin = G.input({D, hc, nt}, hc_state.data(), hc_state.size() * 4);
        std::vector<int32_t> posv((size_t) nt);
        for (int64_t t = 0; t < nt; ++t) posv[(size_t) t] = (int32_t) t;
        ggml_tensor * inp_pos = G.input_i32({nt}, posv.data(), posv.size() * 4);

        // ---- attention hyper-connection
        ggml_tensor * post_a = nullptr, * comb_a = nullptr;
        ggml_tensor * attn_pre = G.hc_pre(xin, G.L(il, "hc_attn_fn.weight"),
                                          G.L(il, "hc_attn_scale.weight"), G.L(il, "hc_attn_base.weight"),
                                          &post_a, &comb_a);
        ggml_set_output(attn_pre); ggml_set_name(attn_pre, "hc_attn_pre");
        ggml_tensor * xn = G.rms_w(attn_pre, G.L(il, "attn_norm.weight"));
        ggml_set_output(xn); ggml_set_name(xn, "attn_norm");

        const CompInputs & ci = (ratio == 4) ? ci4 : ci128;
        const bool use_indexer = ratio == 4 && ci.n_blocks > g.indexer_top_k;
        ggml_tensor * rot_t = use_indexer && !rot_data.empty()
            ? G.input({(int64_t) g.indexer_key_dim, (int64_t) g.indexer_key_dim}, rot_data.data(), rot_data.size() * 4)
            : nullptr;
        ggml_tensor * attn_raw = nullptr;
        ggml_tensor * attn_out = G.attention(il, xn, nullptr, inp_pos, ci, ratio, raw_mask, rot_t,
                                             g.indexer_top_k, &attn_raw);
        const char * attn_name = ratio == 4 ? "attn_csa_lid" : (ratio == 128 ? "attn_hca" : "attn_raw");
        ggml_set_output(attn_raw);
        ggml_set_name(attn_raw, attn_name);

        ggml_tensor * residual = xin;
        ggml_tensor * hap = G.hc_post(attn_out, residual, post_a, comb_a);
        ggml_set_output(hap); ggml_set_name(hap, "hc_attn_post");

        // ---- feed-forward hyper-connection
        ggml_tensor * post_f = nullptr, * comb_f = nullptr;
        ggml_tensor * ffn_pre = G.hc_pre(hap, G.L(il, "hc_ffn_fn.weight"),
                                         G.L(il, "hc_ffn_scale.weight"), G.L(il, "hc_ffn_base.weight"),
                                         &post_f, &comb_f);
        ggml_set_output(ffn_pre); ggml_set_name(ffn_pre, "hc_ffn_pre");
        ggml_tensor * fn = G.rms_w(ffn_pre, G.L(il, "ffn_norm.weight"));
        ggml_set_output(fn); ggml_set_name(fn, "ffn_norm");

        // ---- MoE router
        const int64_t n_expert = g.n_expert, topk = g.n_expert_used, n_ff = g.n_ff;
        ggml_tensor * logits = ggml_mul_mat(w.ctx, G.L(il, "ffn_gate_inp.weight"), fn);
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        ggml_tensor * probs = ggml_sqrt(w.ctx, ggml_softplus(w.ctx, logits));

        ggml_tensor * selected = nullptr;
        if (il < g.hash_layer_count) {
            ggml_tensor * tid = G.input_i32({nt}, tokens.data(), (size_t) nt * 4);
            selected = ggml_get_rows(w.ctx, G.L(il, "ffn_gate_tid2eid.weight"), tid);
        } else {
            ggml_tensor * sel = ggml_add(w.ctx, probs, G.L(il, "exp_probs_b.bias"));
            selected = ggml_argsort_top_k(w.ctx, sel, (int) topk);
        }

        ggml_tensor * probs3 = ggml_reshape_3d(w.ctx, probs, 1, n_expert, nt);
        ggml_tensor * weights = ggml_get_rows(w.ctx, probs3, selected);          // [1, topk, nt]
        weights = ggml_reshape_2d(w.ctx, weights, topk, nt);
        ggml_tensor * wsum = ggml_add(w.ctx,
                ggml_sum_rows(w.ctx, weights),
                ggml_fill(w.ctx, ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, 1, 1), 0.0f));
        wsum = ggml_clamp(w.ctx, wsum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(w.ctx, weights, wsum);
        weights = ggml_reshape_3d(w.ctx, weights, 1, topk, nt);
        weights = ggml_scale(w.ctx, weights, (float) g.weights_scale);

        ggml_tensor * cur3 = ggml_reshape_3d(w.ctx, fn, D, 1, nt);
        ggml_tensor * up = ggml_mul_mat_id(w.ctx, G.L(il, "ffn_up_exps.weight"), cur3, selected);
        ggml_tensor * gate = ggml_mul_mat_id(w.ctx, G.L(il, "ffn_gate_exps.weight"), cur3, selected);
        const float limit = (float) g.swiglu_clamp_exp[(size_t) il];
        up = ggml_clamp(w.ctx, up, -limit, limit);
        gate = ggml_clamp(w.ctx, gate, NEG_INF, limit);
        ggml_tensor * h = ggml_swiglu_split(w.ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat_id(w.ctx, G.L(il, "ffn_down_exps.weight"), h, selected);
        down = ggml_mul(w.ctx, down, weights);                                    // [D, topk, nt]
        ggml_tensor * moe_out = nullptr;
        for (int64_t k = 0; k < topk; ++k) {
            ggml_tensor * v = G.view2(down, D, nt, down->nb[2], (size_t) k * down->nb[1]);
            moe_out = moe_out ? ggml_add(w.ctx, moe_out, v) : v;
        }
        ggml_set_output(moe_out); ggml_set_name(moe_out, "ffn_moe_out");

        // ---- shared expert (parallel single FFN, width n_ff*n_expert_shared)
        const int64_t n_ff_s = n_ff * g.n_expert_shared;
        ggml_tensor * up_s = ggml_mul_mat(w.ctx, G.L(il, "ffn_up_shexp.weight"), fn);
        ggml_tensor * gate_s = ggml_mul_mat(w.ctx, G.L(il, "ffn_gate_shexp.weight"), fn);
        const float limit_s = (float) g.swiglu_clamp_shexp[(size_t) il];
        up_s = ggml_clamp(w.ctx, up_s, -limit_s, limit_s);
        gate_s = ggml_clamp(w.ctx, gate_s, NEG_INF, limit_s);
        ggml_tensor * z = ggml_swiglu_split(w.ctx, gate_s, up_s);
        ggml_tensor * shexp = ggml_mul_mat(w.ctx, G.L(il, "ffn_down_shexp.weight"), z);
        ggml_set_output(shexp); ggml_set_name(shexp, "ffn_shexp");
        (void) n_ff_s;

        ggml_tensor * ffn_out = ggml_add(w.ctx, moe_out, shexp);
        ggml_set_output(ffn_out); ggml_set_name(ffn_out, "ffn_out");

        ggml_tensor * l_last = G.hc_post(ffn_out, hap, post_f, comb_f);
        ggml_set_output(l_last); ggml_set_name(l_last, "l_last");

        // compute
        ggml_cgraph * gf = ggml_new_graph_custom(w.ctx, 2048, false);
        ggml_build_forward_expand(gf, l_last);
        if (!run(gf)) return 1;

        // read dumps
        dump_put(ds, attn_name, il, {n_head * d_h}, (const float *) attn_raw->data, nt);
        dump_put(ds, "hc_attn_pre", il, {D}, (const float *) attn_pre->data, nt);
        dump_put(ds, "attn_norm", il, {D}, (const float *) xn->data, nt);
        dump_put(ds, "attn_out", il, {D}, (const float *) attn_out->data, nt);
        dump_put(ds, "hc_attn_post", il, {D, hc}, (const float *) hap->data, nt);
        dump_put(ds, "hc_ffn_pre", il, {D}, (const float *) ffn_pre->data, nt);
        dump_put(ds, "ffn_norm", il, {D}, (const float *) fn->data, nt);
        dump_put(ds, "ffn_moe_out", il, {D}, (const float *) moe_out->data, nt);
        dump_put(ds, "ffn_shexp", il, {D}, (const float *) shexp->data, nt);
        dump_put(ds, "ffn_out", il, {D}, (const float *) ffn_out->data, nt);
        dump_put(ds, "l_last", il, {D, hc}, (const float *) l_last->data, nt);

        std::memcpy(hc_state.data(), l_last->data, hc_state.size() * 4);
        std::printf("[ds4_ref] layer %lld/%lld done (ratio=%lld)\n", (long long) (il + 1), (long long) n_layer, (long long) ratio);
        std::fflush(stdout);
    }

    // ---------------- head
    if (n_layer == g.n_layer()) {
        G.fills.clear();
        ggml_tensor * xin = G.input({D, hc, nt}, hc_state.data(), hc_state.size() * 4);
        const int64_t mix_dim = g.hc_mix_dim();
        (void) mix_dim;
        ggml_tensor * flat = ggml_reshape_2d(w.ctx, xin, hc * D, nt);
        ggml_tensor * flat_norm = ggml_rms_norm(w.ctx, flat, (float) g.rms_eps);
        ggml_tensor * mixes = ggml_mul_mat(w.ctx, w.get("output_hc_fn.weight"), flat_norm);   // [hc, nt]
        ggml_tensor * scale = w.get("output_hc_scale.weight");
        ggml_tensor * base  = w.get("output_hc_base.weight");
        ggml_tensor * pre = ggml_sigmoid(w.ctx, G.hc_affine(mixes, G.view1(scale, 1, 0), G.view1(base, hc, 0)));
        pre = ggml_scale_bias(w.ctx, pre, 1.0f, (float) g.hc_eps);
        ggml_tensor * hc_head = G.hc_mean(xin, pre);
        ggml_set_output(hc_head); ggml_set_name(hc_head, "hc_head");
        ggml_tensor * rn2 = G.rms_w(hc_head, w.get("output_norm.weight"));
        ggml_set_output(rn2); ggml_set_name(rn2, "result_norm");
        ggml_tensor * logits = ggml_mul_mat(w.ctx, w.get("output.weight"), rn2);
        ggml_set_output(logits); ggml_set_name(logits, "result_output");

        ggml_cgraph * gf = ggml_new_graph_custom(w.ctx, 256, false);
        ggml_build_forward_expand(gf, logits);
        if (!run(gf)) return 1;
        dump_put(ds, "hc_head", -1, {D}, (const float *) hc_head->data, nt);
        dump_put(ds, "result_norm", -1, {D}, (const float *) rn2->data, nt);
        dump_put(ds, "result_output", -1, {g.vocab_size}, (const float *) logits->data, nt);
    }

    if (!write_dumps(ds, tokens)) return 1;
    ggml_gallocr_free(allo);
    ggml_backend_free(cpu);
    return 0;
}
