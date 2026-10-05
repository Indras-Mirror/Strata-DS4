// golden_dump: dump per-layer DeepSeek-V4 (GGUF arch `deepseek4`) activations for a prompt file so a
// from-scratch engine can be verified layer by layer against llama.cpp (the oracle).
//
// For each captured tensor we write raw little-endian F32: last token position by default, every
// position with --all-pos. Output is a directory:
//   <out>/manifest.json   { tensor name, layer, shape, dtype, file, sha256 }
//   <out>/<name>.f32      raw float32
//   <out>/tokens.i32      the tokenized prompt (little-endian int32)
//
// Tensor names are the llama.cpp graph-builder names from src/models/deepseek4.cpp's cb(...) calls,
// suffixed with "-<layer>" by llama_context::graph_get_cb(). We match by base name:
//   per layer : attn_norm, attn_out, attn_raw, attn_csa_lid, attn_hca, ffn_norm, ffn_moe_out,
//               ffn_shexp, ffn_out, l_last, hc_attn_pre, hc_attn_post, hc_ffn_pre
//               (there is no hc_ffn_post: ffn_out is the post-FFN hc stream)
//   global    : hc_init, hc_head, result_norm, result_output
// "attention output" = attn_out (after the grouped output LoRA); the attention-module output before
// the output projection is attn_raw / attn_csa_lid / attn_hca (one of the three per layer).
// "layer output (hc streams)" = l_last ([n_embd, hc, n_tokens]); "final logits" = result_output.
//
// Build (static llama.cpp build in ~/AI/llama.cpp-master-rebase/build, CUDA 13.4; mirrors how
// llama-eval-callback is linked there):
//   L=~/AI/llama.cpp-master-rebase
//   g++ -O3 -DNDEBUG -std=c++17 -fopenmp \
//       -I $L/common -I $L/include -I $L/ggml/include -I $L/src \
//       tools/ds4/golden_dump.cpp \
//       $L/build/common/libllama-common.a $L/build/src/libllama.a \
//       $L/build/ggml/src/libggml.a $L/build/ggml/src/libggml-cpu.a \
//       $L/build/ggml/src/ggml-cuda/libggml-cuda.a $L/build/ggml/src/libggml-base.a \
//       $L/build/common/libllama-common-base.a $L/build/vendor/cpp-httplib/libcpp-httplib.a \
//       -lgomp -pthread -lm -ldl -lrt -lssl -lcrypto \
//       -L/usr/local/cuda-13.4/targets/x86_64-linux/lib -lcudart -lcublas -lcublasLt -lnccl -lcuda \
//       -Wl,-rpath,/usr/local/cuda-13.4/targets/x86_64-linux/lib \
//       -o tools/ds4/golden_dump
//
// Usage:
//   tools/ds4/golden_dump -m model.gguf -f prompt.txt --out bench/.../goldens/p64 \
//       -ngl 99 -cmoe --moe-expert-cache 40 --load-mode none -ctk f16 -ctv f16
//   (all llama.cpp common flags are accepted; --out and --all-pos are this tool's own.)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------- name matching

static const std::set<std::string> & layer_bases() {
    static const std::set<std::string> s = {
        "attn_norm", "attn_out", "attn_raw", "attn_csa_lid", "attn_hca",
        "ffn_norm", "ffn_moe_out", "ffn_shexp", "ffn_out",
        "l_last", "hc_attn_pre", "hc_attn_post", "hc_ffn_pre",
        // probe taps for the compressed-attention (CSA/HCA) path
        "q", "kv", "csa_state_kv", "csa_state_score_ape", "csa_state_compress", "csa_comp_k",
    };
    return s;
}

// node whose dump axis is the compressed-block axis (ne[2]) rather than tokens: we want
// every block, not one token-indexed row.
static bool block_indexed(const std::string & base) {
    return base == "csa_state_compress" || base == "csa_comp_k";
}

// extra intermediate taps used to diff the compressed-attention path. They are only
// captured when DS4_GOLDEN_PROBE=1 so a normal golden regeneration produces exactly the
// gate tensor set (compare_golden flags extra golden-only names as breaches).
static bool probe_enabled() {
    static const bool on = []() {
        const char * e = getenv("DS4_GOLDEN_PROBE");
        return e != nullptr && e[0] != '0';
    }();
    return on;
}

static bool probe_base(const std::string & base) {
    return base == "q" || base == "kv" || base == "csa_state_kv" ||
           base == "csa_state_score_ape" || base == "csa_state_compress" ||
           base == "csa_comp_k";
}

static const std::set<std::string> & global_bases() {
    static const std::set<std::string> s = { "hc_init", "hc_head", "result_norm", "result_output" };
    return s;
}

// tensors that are indexed by *output* position (the tokens selected by inp_out_ids), not by
// token position: with a normal batch only the last prompt token is an output, so these have
// one column regardless of how many tokens the ubatch holds.
static bool output_indexed(const std::string & base) {
    return base == "hc_head" || base == "result_norm" || base == "result_output";
}

// returns true if the tensor is one we capture; fills base name and layer (-1 for globals)
static bool match_name(const char * raw, std::string & base, int & il) {
    std::string s = raw;
    il = -1;
    const size_t dash = s.rfind('-');
    if (dash != std::string::npos) {
        const std::string tail = s.substr(dash + 1);
        if (!tail.empty() && tail.find_first_not_of("0123456789") == std::string::npos) {
            il = std::atoi(tail.c_str());
            s  = s.substr(0, dash);
        }
    }
    base = s;
    if (il >= 0) {
        return layer_bases().count(base) != 0;
    }
    return global_bases().count(base) != 0;
}

// ---------------------------------------------------------------- output state

struct DumpState {
    std::string out_dir;
    bool all_pos = false;
    bool capture = false;
    bool is_final = false;
    int64_t row_begin = 0;
    int64_t row_end = 0;

    struct Tensor {
        std::string name;
        std::string base;
        int il = -1;
        std::vector<int64_t> feat_shape; // dims excluding the token axis
        size_t per_tok = 0;              // bytes per token row
        size_t rows = 0;                 // token rows written
        FILE * f = nullptr;
    };
    std::map<std::string, Tensor> tensors;
    std::set<std::string> warned;
};

// one file per tensor, written once when the tensor is first seen and appended for
// every later row: "wb" (not "ab") so re-running into an existing output directory
// truncates instead of silently doubling the file.
static FILE * open_append(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "wb");
    return f;
}

// the token axis is the last logical axis. ggml_tensor has no n_dims field and ggml_n_dims()
// trims trailing 1s, so the rank has to come from the name; these are the shapes the
// deepseek4.cpp graph builder produces (see DSV4_ARCH_SPEC.md):
//   [d_h, n_heads, nt] : attn_raw, attn_csa_lid, attn_hca   (build_attn_mha, non-flash path)
//   [d_h*n_heads, nt]  : the same three with -fa on (build_attn_mha reshapes to 2-D)
//   [D, hc, nt]        : hc_init, hc_attn_post, l_last
//   [D/..., nt]        : everything else
// note: there is no per-layer "hc_ffn_post" in the graph (the FFN hc stream is ffn_out).
static int base_rank(const std::string & base) {
    if (base == "attn_raw" || base == "attn_csa_lid" || base == "attn_hca" ||
        base == "hc_init" || base == "hc_attn_post" || base == "l_last") {
        return 3;
    }
    if (block_indexed(base)) {
        return 3;
    }
    return 2;
}

// ggml trims trailing 1s, so the name-based rank hint can be wrong (it is for the
// flash-attention 2-D attention outputs). Resolve the real token axis from the buffer
// layout: the stride times the token count must cover exactly nbytes, and the axis must
// hold the requested rows. Prefer the hint; otherwise try ranks largest-first so a 3-D
// tensor is never mistaken for a 2-D one (which would lose features).
static int resolve_token_axis(const ggml_tensor * t, int64_t rows_needed, int hint) {
    const size_t nbytes = ggml_nbytes(t);
    auto ok = [&](int r) -> bool {
        if (r < 2 || r > 4) return false;
        const int64_t nl = t->ne[r - 1];
        return nl > 0 && rows_needed <= nl && (size_t) t->nb[r - 1] * (size_t) nl == nbytes;
    };
    if (ok(hint)) return hint;
    for (int r = 4; r >= 2; --r) if (ok(r)) return r;
    return hint;
}

static bool write_f32_rows(DumpState & st, ggml_tensor * t, const std::string & name,
                           const std::string & base, int il, int64_t rb, int64_t re) {
    const int nd = resolve_token_axis(t, re, base_rank(base));
    if (!ggml_is_contiguous(t)) {
        if (st.warned.insert(name).second) {
            std::fprintf(stderr, "[golden_dump] skip non-contiguous tensor %s\n", name.c_str());
        }
        return false;
    }
    const int64_t ne_last = t->ne[nd - 1];
    if (rb < 0 || re > ne_last || rb >= re) {
        if (st.warned.insert(name).second) {
            std::fprintf(stderr, "[golden_dump] skip tensor %s: rows [%lld,%lld) outside token axis %d (ne=%lld)\n",
                         name.c_str(), (long long) rb, (long long) re, nd - 1, (long long) ne_last);
        }
        return false;
    }
    const size_t nbytes = ggml_nbytes(t);
    const size_t per_tok = t->nb[nd - 1];
    if (nbytes == 0 || per_tok == 0 || nbytes != per_tok * (size_t) ne_last) {
        if (st.warned.insert(name).second) {
            std::fprintf(stderr, "[golden_dump] skip tensor %s: layout mismatch (nbytes=%zu per_tok=%zu ne_last=%lld)\n",
                         name.c_str(), nbytes, per_tok, (long long) ne_last);
        }
        return false;
    }

    const size_t ts = ggml_type_size(t->type);
    if (ts == 0 || per_tok % ts != 0) return false;
    const size_t n_feat = per_tok / ts;

    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        if (st.warned.insert(name).second) {
            std::fprintf(stderr, "[golden_dump] skip tensor %s: unsupported dtype %s\n",
                         name.c_str(), ggml_type_name(t->type));
        }
        return false;
    }

    auto it = st.tensors.find(name);
    if (it == st.tensors.end()) {
        DumpState::Tensor tn;
        tn.name = name;
        tn.base = base;
        tn.il   = il;
        for (int i = 0; i < nd - 1; ++i) tn.feat_shape.push_back(t->ne[i]);
        tn.per_tok = per_tok;
        const std::string path = st.out_dir + "/" + name + ".f32";
        tn.f = open_append(path);
        if (tn.f == nullptr) {
            std::fprintf(stderr, "[golden_dump] cannot open %s\n", path.c_str());
            return false;
        }
        it = st.tensors.emplace(name, tn).first;
    }
    DumpState::Tensor & tn = it->second;

    std::vector<uint8_t> raw(per_tok);
    std::vector<float>   dst(n_feat);
    const bool host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);

    for (int64_t r = rb; r < re; ++r) {
        const size_t off = (size_t) r * per_tok;
        if (host) {
            std::memcpy(raw.data(), (const uint8_t *) t->data + off, per_tok);
        } else {
            ggml_backend_tensor_get(t, raw.data(), off, per_tok);
        }
        const void * src = raw.data();
        if (t->type == GGML_TYPE_F32) {
            std::memcpy(dst.data(), src, n_feat * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * s = (const ggml_fp16_t *) src;
            for (size_t i = 0; i < n_feat; ++i) dst[i] = ggml_fp16_to_fp32(s[i]);
        } else { // BF16
            const ggml_bf16_t * s = (const ggml_bf16_t *) src;
            for (size_t i = 0; i < n_feat; ++i) dst[i] = ggml_bf16_to_fp32(s[i]);
        }
        if (std::fwrite(dst.data(), sizeof(float), n_feat, tn.f) != n_feat) {
            std::fprintf(stderr, "[golden_dump] short write for %s\n", name.c_str());
            return false;
        }
        tn.rows++;
    }
    return true;
}

// ---------------------------------------------------------------- eval callback

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    DumpState * st = (DumpState *) user_data;
    std::string base;
    int il = -1;
    if (!match_name(t->name, base, il)) {
        return ask ? false : true;
    }
    if (probe_base(base) && !probe_enabled()) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    if (!st->capture) {
        return true;
    }
    // the output-indexed tensors only exist for the token(s) selected as outputs (the last
    // token of each ubatch), so in --all-pos mode capture them on the final ubatch only.
    if (st->all_pos && output_indexed(base) && !st->is_final) {
        return true;
    }
    int64_t rb = st->row_begin;
    int64_t re = st->row_end;
    if (output_indexed(base)) {
        rb = 0;
        re = t->ne[base_rank(base) - 1];
    }
    if (block_indexed(base)) {
        rb = 0;
        re = t->ne[2];
    }
    write_f32_rows(*st, t, t->name, base, il, rb, re);
    return true;
}

// ---------------------------------------------------------------- json / sha

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); o += buf; }
                else o += (char) c;
        }
    }
    return o;
}

static std::string sha256_file(const std::string & path) {
    std::string cmd = "sha256sum \"" + path + "\" 2>/dev/null";
    FILE * p = popen(cmd.c_str(), "r");
    if (p == nullptr) return "unavailable";
    char buf[128] = {0};
    const char * r = std::fgets(buf, sizeof buf, p);
    pclose(p);
    if (r == nullptr) return "unavailable";
    std::string s(buf);
    const size_t sp = s.find(' ');
    return sp == std::string::npos ? std::string("unavailable") : s.substr(0, sp);
}

// ---------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // pull our own args out before llama.cpp common parsing sees them
    std::string out_dir;
    bool all_pos = false;
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out") {
            if (i + 1 >= argc) { std::fprintf(stderr, "error: --out needs a directory\n"); return 1; }
            out_dir = argv[++i];
        } else if (a.rfind("--out=", 0) == 0) {
            out_dir = a.substr(6);
        } else if (a == "--all-pos") {
            all_pos = true;
        } else {
            rest.push_back(argv[i]);
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (out_dir.empty()) {
        std::fprintf(stderr, "error: --out <dir> is required\n");
        return 1;
    }
    if (params.model.path.empty()) {
        std::fprintf(stderr, "error: -m <model.gguf> is required\n");
        return 1;
    }

    std::string text;
    if (!params.prompt_file.empty()) {
        std::ifstream in(params.prompt_file, std::ios::binary);
        if (!in) { std::fprintf(stderr, "error: cannot read prompt file %s\n", params.prompt_file.c_str()); return 1; }
        std::stringstream ss; ss << in.rdbuf(); text = ss.str();
    } else if (!params.prompt.empty()) {
        text = params.prompt;
    } else {
        std::fprintf(stderr, "error: provide a prompt with -f <file> (raw text) or -p <string>\n");
        return 1;
    }

    DumpState st;
    st.out_dir = out_dir;
    st.all_pos = all_pos;
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) { std::fprintf(stderr, "error: cannot create %s: %s\n", out_dir.c_str(), ec.message().c_str()); return 1; }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) { std::fprintf(stderr, "error: failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(vocab, text, add_bos, false);
    if (tokens.empty()) { std::fprintf(stderr, "error: prompt tokenized to zero tokens\n"); return 1; }

    params.n_ctx = std::max<int>(params.n_ctx, (int) GGML_PAD((int64_t) tokens.size() + 64, 64));
    llama_context_params cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) { std::fprintf(stderr, "error: failed to create context\n"); return 1; }

    std::printf("[golden_dump] model      : %s\n", params.model.path.c_str());
    std::printf("[golden_dump] prompt     : %zu tokens (add_bos=%d, all_pos=%d)\n",
                tokens.size(), (int) add_bos, (int) all_pos);
    std::printf("[golden_dump] output     : %s\n", out_dir.c_str());
    std::printf("[golden_dump] n_ctx=%d n_batch=%d n_ubatch=%d ngl=%d moe_cache=%d kv=%s/%s\n",
                cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, params.n_gpu_layers,
                params.n_moe_cache_slots, ggml_type_name(cparams.type_k), ggml_type_name(cparams.type_v));

    // token ids
    {
        const std::string p = out_dir + "/tokens.i32";
        FILE * f = std::fopen(p.c_str(), "wb");
        if (f == nullptr) { std::fprintf(stderr, "error: cannot write %s\n", p.c_str()); return 1; }
        for (llama_token id : tokens) {
            const int32_t v = (int32_t) id;
            std::fwrite(&v, sizeof v, 1, f);
        }
        std::fclose(f);
    }

    const size_t n = tokens.size();
    const int chunk = std::max(1, std::min(params.n_ubatch > 0 ? params.n_ubatch : 512,
                                           params.n_batch   > 0 ? params.n_batch   : 512));
    bool ok = true;
    for (size_t off = 0; off < n; off += chunk) {
        const int m = (int) std::min<size_t>(chunk, n - off);
        st.is_final = (off + m == n);
        if (all_pos) {
            st.capture = true; st.row_begin = 0; st.row_end = m;
        } else if (st.is_final) {
            st.capture = true; st.row_begin = m - 1; st.row_end = m;
        } else {
            st.capture = false;
        }
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + off, m)) != 0) {
            std::fprintf(stderr, "error: decode failed at offset %zu\n", off);
            ok = false;
            break;
        }
    }
    st.capture = false;

    for (auto & kv : st.tensors) {
        if (kv.second.f != nullptr) std::fclose(kv.second.f);
    }

    if (!ok) { llama_free(ctx); llama_model_free(model); llama_backend_free(); return 1; }

    // manifest.json
    std::vector<const DumpState::Tensor *> ordered;
    ordered.reserve(st.tensors.size());
    for (const auto & kv : st.tensors) ordered.push_back(&kv.second);
    std::sort(ordered.begin(), ordered.end(), [](const DumpState::Tensor * a, const DumpState::Tensor * b) {
        if (a->il != b->il) return a->il < b->il;
        return a->name < b->name;
    });

    const std::string mp = out_dir + "/manifest.json";
    FILE * mf = std::fopen(mp.c_str(), "wb");
    if (mf == nullptr) { std::fprintf(stderr, "error: cannot write %s\n", mp.c_str()); return 1; }

    std::fprintf(mf, "{\n");
    std::fprintf(mf, "  \"tool\": \"ds4-golden_dump\",\n");
    std::fprintf(mf, "  \"model\": \"%s\",\n", json_escape(params.model.path).c_str());
    std::fprintf(mf, "  \"prompt_file\": \"%s\",\n", json_escape(params.prompt_file).c_str());
    std::fprintf(mf, "  \"n_tokens\": %zu,\n", n);
    std::fprintf(mf, "  \"add_bos\": %s,\n", add_bos ? "true" : "false");
    std::fprintf(mf, "  \"all_pos\": %s,\n", all_pos ? "true" : "false");
    std::fprintf(mf, "  \"n_gpu_layers\": %d,\n", params.n_gpu_layers);
    std::fprintf(mf, "  \"moe_expert_cache_slots\": %d,\n", params.n_moe_cache_slots);
    std::fprintf(mf, "  \"kv_type_k\": \"%s\",\n", ggml_type_name(cparams.type_k));
    std::fprintf(mf, "  \"kv_type_v\": \"%s\",\n", ggml_type_name(cparams.type_v));
    std::fprintf(mf, "  \"tensors\": [\n");

    bool first = true;
    auto emit = [&](const std::string & name, int il, const std::string & shape, const char * dtype,
                    const std::string & file, const std::string & sha) {
        std::fprintf(mf, "%s    { \"name\": \"%s\", \"layer\": %d, \"shape\": [%s], \"dtype\": \"%s\", "
                         "\"file\": \"%s\", \"sha256\": \"%s\" }",
                     first ? "" : ",\n", json_escape(name).c_str(), il, shape.c_str(), dtype,
                     json_escape(file).c_str(), sha.c_str());
        first = false;
    };

    for (const DumpState::Tensor * t : ordered) {
        std::string shape;
        for (int64_t d : t->feat_shape) shape += std::to_string(d) + ", ";
        shape += std::to_string(t->rows);
        emit(t->name, t->il, shape, "f32", t->name + ".f32", sha256_file(out_dir + "/" + t->name + ".f32"));
    }

    std::printf("[golden_dump] wrote %zu tensors\n", st.tensors.size());

    std::fprintf(mf, "\n  ],\n");
    std::fprintf(mf, "  \"tokens\": { \"name\": \"tokens\", \"layer\": -1, \"shape\": [%zu], "
                     "\"dtype\": \"i32\", \"file\": \"tokens.i32\", \"sha256\": \"%s\" }\n",
                 n, sha256_file(out_dir + "/tokens.i32").c_str());
    std::fprintf(mf, "}\n");
    std::fclose(mf);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
