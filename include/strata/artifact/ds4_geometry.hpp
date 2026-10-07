// include/strata/artifact/ds4_geometry.hpp - the DeepSeek-V4-Flash (`deepseek4`) geometry and tensor map.
//
// Two jobs, both from docs/ds4/PORT_PLAN.md Phase 1 and docs/ds4/DSV4_ARCH_SPEC.md:
//
//   1. `Ds4Geometry` is every number a Phase-2/3 kernel will depend on, taken from the GGUF's own
//      `deepseek4.*` keys - D=4096, 43 layers, 64 q / 1 kv heads, d_h 512, d_rope 64, q_lora 1024,
//      o_groups 8, o_lora 1024, 256 experts top-6, n_ff 2048, hc 4 (Sinkhorn 20), window 128, the
//      per-layer `compress_ratios`, the indexer 64/128/512, 3 hash layers, weights scale 1.5, swiglu
//      clamp 10, YaRN factor 16 / orig 65536, compress rope base 160000.  Nothing here is a compiled-in
//      constant where the GGUF carries a key; a key that is absent is an error, not a default.
//
//   2. `check_ds4_model` resolves the FULL tensor map: every name the file holds must be one this model
//      predicts, and every tensor this model predicts must be in the file.  A single unknown or missing
//      tensor fails the load with its name.  The per-layer set is conditional exactly as the architecture
//      is: compressor tensors appear only where `compress_ratios[i] != 0`, the six indexer tensors only
//      where the ratio is 4, and hash layers carry `ffn_gate_tid2eid` where the rest carry `exp_probs_b`.
//      That conditional structure is what makes "zero unknowns" a real check instead of a name-less match.
//
// It does NOT load tensor data: `GgufModel` mmaps the header, so this runs on a machine that is already
// using the RAM (see the packet's NO-GPU / no-full-load constraint).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata {

// ggml type ids this model uses (the ids are `ggml_type`'s; see gguf_reader.hpp's name table).
namespace ds4_type {
constexpr uint32_t F32 = 0;
constexpr uint32_t F16 = 1;
constexpr uint32_t Q8_0 = 8;
constexpr uint32_t Q2_K = 10;
constexpr uint32_t IQ2_XXS = 16;
constexpr uint32_t I32 = 26;
}  // namespace ds4_type

/// The DSV4 geometry.  Field names track `llama-hparams`' DSV4 fields (docs/ds4/DSV4_ARCH_SPEC.md s1.1) so a
/// value can be traced both ways; every one is read from a key by `ds4_read_geometry`.
struct Ds4Geometry {
    int64_t n_embd = 0;              ///< deepseek4.embedding_length
    int64_t n_layer_all = 0;         ///< deepseek4.block_count (trunk + nextn blocks)
    int64_t n_layer_nextn_requested = 0;  ///< deepseek4.nextn_predict_layers as declared
    int64_t n_layer_nextn = 0;       ///< effective: zeroed when the MTP tensors are absent (llama.cpp deepseek4.cpp:20-26)

    int64_t n_head = 0;              ///< attention.head_count
    int64_t n_head_kv = 0;           ///< attention.head_count_kv (always 1: fused K=V MQA)
    int64_t d_head = 0;              ///< attention.key_length (== value_length); qk_nope + qk_rope
    int64_t d_rope = 0;              ///< rope.dimension_count
    int64_t r_q = 0;                 ///< attention.q_lora_rank
    int64_t o_groups = 0;            ///< attention.output_group_count
    int64_t o_lora = 0;              ///< attention.output_lora_rank
    int64_t n_swa = 0;               ///< attention.sliding_window

    int64_t n_expert = 0;            ///< expert_count
    int64_t n_expert_used = 0;       ///< expert_used_count (top-k)
    int64_t n_ff = 0;                ///< expert_feed_forward_length
    int64_t n_expert_shared = 0;     ///< expert_shared_count
    double weights_scale = 0;        ///< expert_weights_scale (routed_scaling_factor)
    bool weights_norm = false;       ///< expert_weights_norm (norm_topk_prob)
    int64_t gating_func = 0;         ///< expert_gating_func (== 4, SQRT_SOFTPLUS)

    int64_t hc = 0;                  ///< hyper_connection.count (multiplicity)
    int64_t hc_sinkhorn_iters = 0;   ///< hyper_connection.sinkhorn_iterations
    double hc_eps = 0;               ///< hyper_connection.epsilon

    int64_t hash_layer_count = 0;    ///< hash_layer_count

    int64_t indexer_n_head = 0;      ///< attention.indexer.head_count
    int64_t indexer_key_dim = 0;     ///< attention.indexer.key_length
    int64_t indexer_top_k = 0;       ///< attention.indexer.top_k

    double rms_eps = 0;              ///< attention.layer_norm_rms_epsilon
    std::string rope_scaling_type;   ///< rope.scaling.type ("yarn")
    double rope_freq_base = 0;       ///< rope.freq_base
    double rope_factor = 0;          ///< rope.scaling.factor
    double rope_orig_ctx = 0;        ///< rope.scaling.original_context_length
    double yarn_beta_fast = 0;       ///< rope.scaling.yarn_beta_fast
    double yarn_beta_slow = 0;       ///< rope.scaling.yarn_beta_slow
    double compress_rope_base = 0;   ///< attention.compress_rope_freq_base

    int64_t vocab_size = 0;          ///< vocab_size
    int64_t context_length = 0;      ///< context_length
    std::string tokenizer_pre;       ///< tokenizer.ggml.pre ("joyai-llm")

    std::vector<int64_t> compress_ratios;   ///< attention.compress_ratios, per block (0 / 4 / 128)
    std::vector<double> swiglu_clamp_exp;   ///< swiglu_clamp_exp, per block
    std::vector<double> swiglu_clamp_shexp; ///< swiglu_clamp_shexp, per block (falls back to _exp when absent)

    int64_t n_layer() const { return n_layer_all - n_layer_nextn; }        ///< trunk layers
    int64_t hc_dim() const { return hc * n_embd; }                         ///< hc*D
    int64_t hc_mix_dim() const { return (2 + hc) * hc; }                   ///< (2+hc)*hc = 24
    int64_t heads_dim() const { return n_head * d_head; }                  ///< n_heads*d_h
    int64_t o_group_dim() const { return n_head * d_head / o_groups; }     ///< (n_heads*d_h)/o_groups
};

// ------------------------------------------------------------------ key readers (missing key == error)
inline std::string ds4_want_u64(const GgufFile& g, const char* key, uint64_t& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (!v->is_num()) return std::string(key) + " is not a number";
    out = (uint64_t)v->num();
    return {};
}
inline std::string ds4_want_f64(const GgufFile& g, const char* key, double& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (!v->is_num()) return std::string(key) + " is not a number";
    out = v->num();
    return {};
}
inline std::string ds4_want_bool(const GgufFile& g, const char* key, bool& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    out = v->u != 0;
    return {};
}
inline std::string ds4_key_f64_arr(const GgufFile& g, const char* key, std::vector<double>& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (v->type != MetaType::ARRAY) return std::string(key) + " is not an array";
    if (v->count > v->items.size())  // the reader samples arrays at 64 (see read_value)
        return std::string(key) + " has " + std::to_string(v->count) + " items, beyond the reader's 64-item cap";
    out.clear();
    out.reserve(v->items.size());
    for (const auto& e : v->items) out.push_back(e.num());
    return {};
}
inline std::string ds4_key_i64_arr(const GgufFile& g, const char* key, std::vector<int64_t>& out) {
    const MetaValue* v = g.get(key);
    if (!v) return std::string("missing ") + key;
    if (v->type != MetaType::ARRAY) return std::string(key) + " is not an array";
    if (v->count > v->items.size())
        return std::string(key) + " has " + std::to_string(v->count) + " items, beyond the reader's 64-item cap";
    out.clear();
    out.reserve(v->items.size());
    for (const auto& e : v->items) out.push_back((int64_t)e.num());
    return {};
}

/// Read every `deepseek4.*` key into `out`.  On the first missing/mistyped key returns the reason; the
/// geometry is then unreliable and must not be used.
inline std::string ds4_read_geometry(const GgufModel& model, Ds4Geometry& out) {
    const GgufFile& g = model.meta();
    std::string err;
    uint64_t u = 0;
    if (!(err = ds4_want_u64(g, "deepseek4.embedding_length", u)).empty()) return err;
    out.n_embd = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.block_count", u)).empty()) return err;
    out.n_layer_all = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.head_count", u)).empty()) return err;
    out.n_head = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.head_count_kv", u)).empty()) return err;
    out.n_head_kv = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.key_length", u)).empty()) return err;
    out.d_head = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.rope.dimension_count", u)).empty()) return err;
    out.d_rope = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.q_lora_rank", u)).empty()) return err;
    out.r_q = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.output_group_count", u)).empty()) return err;
    out.o_groups = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.output_lora_rank", u)).empty()) return err;
    out.o_lora = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.sliding_window", u)).empty()) return err;
    out.n_swa = (int64_t)u;

    if (!(err = ds4_want_u64(g, "deepseek4.expert_count", u)).empty()) return err;
    out.n_expert = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.expert_used_count", u)).empty()) return err;
    out.n_expert_used = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.expert_feed_forward_length", u)).empty()) return err;
    out.n_ff = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.expert_shared_count", u)).empty()) return err;
    out.n_expert_shared = (int64_t)u;
    if (!(err = ds4_want_f64(g, "deepseek4.expert_weights_scale", out.weights_scale)).empty()) return err;
    if (!(err = ds4_want_bool(g, "deepseek4.expert_weights_norm", out.weights_norm)).empty()) return err;
    if (!(err = ds4_want_u64(g, "deepseek4.expert_gating_func", u)).empty()) return err;
    out.gating_func = (int64_t)u;

    if (!(err = ds4_want_u64(g, "deepseek4.hyper_connection.count", u)).empty()) return err;
    out.hc = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.hyper_connection.sinkhorn_iterations", u)).empty()) return err;
    out.hc_sinkhorn_iters = (int64_t)u;
    if (!(err = ds4_want_f64(g, "deepseek4.hyper_connection.epsilon", out.hc_eps)).empty()) return err;

    if (!(err = ds4_want_u64(g, "deepseek4.hash_layer_count", u)).empty()) return err;
    out.hash_layer_count = (int64_t)u;

    if (!(err = ds4_want_u64(g, "deepseek4.attention.indexer.head_count", u)).empty()) return err;
    out.indexer_n_head = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.indexer.key_length", u)).empty()) return err;
    out.indexer_key_dim = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.attention.indexer.top_k", u)).empty()) return err;
    out.indexer_top_k = (int64_t)u;

    if (!(err = ds4_want_f64(g, "deepseek4.attention.layer_norm_rms_epsilon", out.rms_eps)).empty()) return err;
    {
        const MetaValue* v = g.get("deepseek4.rope.scaling.type");
        if (!v) return "missing deepseek4.rope.scaling.type";
        out.rope_scaling_type = v->s;
    }
    if (!(err = ds4_want_f64(g, "deepseek4.rope.freq_base", out.rope_freq_base)).empty()) return err;
    if (!(err = ds4_want_f64(g, "deepseek4.rope.scaling.factor", out.rope_factor)).empty()) return err;
    if (!(err = ds4_want_f64(g, "deepseek4.rope.scaling.original_context_length", out.rope_orig_ctx)).empty()) return err;
    if (!(err = ds4_want_f64(g, "deepseek4.rope.scaling.yarn_beta_fast", out.yarn_beta_fast)).empty()) return err;
    if (!(err = ds4_want_f64(g, "deepseek4.rope.scaling.yarn_beta_slow", out.yarn_beta_slow)).empty()) return err;
    if (!(err = ds4_want_f64(g, "deepseek4.attention.compress_rope_freq_base", out.compress_rope_base)).empty()) return err;

    if (!(err = ds4_want_u64(g, "deepseek4.vocab_size", u)).empty()) return err;
    out.vocab_size = (int64_t)u;
    if (!(err = ds4_want_u64(g, "deepseek4.context_length", u)).empty()) return err;
    out.context_length = (int64_t)u;
    {
        const MetaValue* v = g.get("tokenizer.ggml.pre");
        if (!v) return "missing tokenizer.ggml.pre";
        out.tokenizer_pre = v->s;
    }

    if (!(err = ds4_key_i64_arr(g, "deepseek4.attention.compress_ratios", out.compress_ratios)).empty()) return err;
    if ((int64_t)out.compress_ratios.size() < out.n_layer_all)
        return "deepseek4.attention.compress_ratios has " + std::to_string(out.compress_ratios.size()) +
               " entries, need at least block_count (" + std::to_string(out.n_layer_all) + ")";
    for (int64_t r : out.compress_ratios)
        if (r != 0 && r != 4 && r != 128) return "compress_ratios contains " + std::to_string(r) + " (only 0, 4, 128)";

    if (!(err = ds4_key_f64_arr(g, "deepseek4.swiglu_clamp_exp", out.swiglu_clamp_exp)).empty()) return err;
    if ((int64_t)out.swiglu_clamp_exp.size() < out.n_layer_all)
        return "deepseek4.swiglu_clamp_exp is shorter than block_count";
    if (!(err = ds4_key_f64_arr(g, "deepseek4.swiglu_clamp_shexp", out.swiglu_clamp_shexp)).empty()) {
        // llama.cpp: swiglu_clamp_shexp falls back to swiglu_clamp_exp when the key is absent
        if (err.rfind("missing ", 0) == 0) out.swiglu_clamp_shexp = out.swiglu_clamp_exp;
        else return err;
    }
    if (out.swiglu_clamp_shexp.empty()) out.swiglu_clamp_shexp = out.swiglu_clamp_exp;

    if (!(err = ds4_want_u64(g, "deepseek4.nextn_predict_layers", u)).empty()) return err;
    out.n_layer_nextn_requested = (int64_t)u;
    out.n_layer_nextn = 0;
    if (out.n_layer_nextn_requested > 0 && out.n_layer_nextn_requested < out.n_layer_all) {
        // llama.cpp zeroes it when `blk.<n_layer_all - nextn>.nextn.eh_proj.weight` is absent (the 0731 artifact
        // has the key but no MTP tensors) -- deepseek4.cpp:20-26.
        const std::string probe = "blk." + std::to_string(out.n_layer_all - out.n_layer_nextn_requested) +
                                  ".nextn.eh_proj.weight";
        if (model.find(probe) == nullptr) out.n_layer_nextn = 0;
        else out.n_layer_nextn = out.n_layer_nextn_requested;
    }
    return {};
}

/// The MTP (nextn) head shipped as its own GGUF (DeepSeek-V4-Flash-MTP-*.gguf): its block is `blk.<n_layer()>`,
/// one index past the trunk.  Checks the file against the trunk geometry and extends the per-block arrays (SwiGLU
/// clamps, compress ratio) to that index from the MTP file's own metadata.  `il_mtp` = the block index.
inline std::string ds4_attach_mtp(const GgufModel& mtp, Ds4Geometry& g, int64_t& il_mtp) {
    il_mtp = g.n_layer();
    const std::string p = "blk." + std::to_string(il_mtp) + ".";
    for (const char* n : { "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight", "attn_norm.weight",
                           "ffn_gate_inp.weight", "ffn_gate_exps.weight" })
        if (mtp.find(p + n) == nullptr) return "MTP file has no " + p + n;
    for (const char* n : { "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight" })
        if (mtp.find(n) == nullptr) return std::string("MTP file has no ") + n + " (its own hc_head)";
    const GgufFile& f = mtp.meta();
    uint64_t u = 0;
    std::string err;
    if (!(err = ds4_want_u64(f, "deepseek4.embedding_length", u)).empty()) return "MTP file: " + err;
    if ((int64_t)u != g.n_embd) return "MTP file: embedding_length differs from the trunk's";
    std::vector<double> ce, cs, cr;
    if (!(err = ds4_key_f64_arr(f, "deepseek4.swiglu_clamp_exp", ce)).empty()) return "MTP file: " + err;
    if (!ds4_key_f64_arr(f, "deepseek4.swiglu_clamp_shexp", cs).empty()) cs = ce;
    if ((int64_t)ce.size() <= il_mtp || (int64_t)cs.size() <= il_mtp) return "MTP file: clamp arrays too short";
    g.swiglu_clamp_exp.resize((size_t)il_mtp + 1, ce[(size_t)il_mtp]);
    g.swiglu_clamp_exp[(size_t)il_mtp] = ce[(size_t)il_mtp];
    g.swiglu_clamp_shexp.resize((size_t)il_mtp + 1, cs[(size_t)il_mtp]);
    g.swiglu_clamp_shexp[(size_t)il_mtp] = cs[(size_t)il_mtp];
    if ((int64_t)g.compress_ratios.size() <= il_mtp) g.compress_ratios.resize((size_t)il_mtp + 1, 0);
    if (g.compress_ratios[(size_t)il_mtp] != 0) return "MTP block is not a ratio-0 (sliding-window) layer";
    return {};
}

/// The MTP head's own hc_head weights: loaded from the MTP file next to the trunk's under these names.  They are not
/// the trunk's (real file: output_hc_scale 1.65 vs the 0731 trunk's 0.79) - llama.cpp loads the MTP file as its own
/// model, so its graph_mtp uses the file's hc_head_*.
inline const std::map<std::string, std::string>& ds4_mtp_head_alias() {
    static const std::map<std::string, std::string> m = {
        { "output_hc_fn.weight", "mtp.output_hc_fn.weight" },
        { "output_hc_base.weight", "mtp.output_hc_base.weight" },
        { "output_hc_scale.weight", "mtp.output_hc_scale.weight" },
    };
    return m;
}

// ------------------------------------------------------------------ the tensor map
struct Ds4TensorSpec {
    std::string name;
    std::vector<uint64_t> shape;  // GGUF order: dim 0 varies fastest
    uint32_t type = 0;
};

namespace ds4_detail {
inline Ds4TensorSpec spec(const std::string& n, std::initializer_list<uint64_t> ne, uint32_t t) {
    Ds4TensorSpec s;
    s.name = n;
    s.shape.assign(ne.begin(), ne.end());
    s.type = t;
    return s;
}
}  // namespace ds4_detail

/// Every tensor the architecture predicts, by name, with the shape and ggml type a kernel will assume.
/// The per-layer conditionals mirror `llama_model_deepseek4::load_arch_tensors` (`deepseek4.cpp:79-178`)
/// and `GGUF_INVENTORY.txt`.
inline std::vector<Ds4TensorSpec> ds4_expected_tensors(const Ds4Geometry& g) {
    using namespace ds4_detail;
    std::vector<Ds4TensorSpec> v;
    const uint64_t D = (uint64_t)g.n_embd, hc = (uint64_t)g.hc, V = (uint64_t)g.vocab_size;
    const uint64_t hd = (uint64_t)g.d_head, rq = (uint64_t)g.r_q;

    // root
    v.push_back(spec("token_embd.weight", {D, V}, ds4_type::F16));
    v.push_back(spec("output_norm.weight", {D}, ds4_type::F32));
    v.push_back(spec("output.weight", {D, V}, ds4_type::Q8_0));
    v.push_back(spec("output_hc_fn.weight", {hc * D, hc}, ds4_type::F16));
    v.push_back(spec("output_hc_base.weight", {hc}, ds4_type::F32));
    v.push_back(spec("output_hc_scale.weight", {1}, ds4_type::F32));

    for (int64_t i = 0; i < g.n_layer_all; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        const bool is_mtp = i >= g.n_layer();
        v.push_back(spec(p + "attn_norm.weight", {D}, ds4_type::F32));
        v.push_back(spec(p + "attn_sinks.weight", {(uint64_t)g.n_head}, ds4_type::F32));
        v.push_back(spec(p + "attn_q_a.weight", {D, rq}, ds4_type::Q8_0));
        v.push_back(spec(p + "attn_q_a_norm.weight", {rq}, ds4_type::F32));
        v.push_back(spec(p + "attn_q_b.weight", {rq, (uint64_t)g.heads_dim()}, ds4_type::Q8_0));
        v.push_back(spec(p + "attn_kv.weight", {D, hd}, ds4_type::Q8_0));
        v.push_back(spec(p + "attn_kv_a_norm.weight", {hd}, ds4_type::F32));
        v.push_back(spec(p + "attn_output_a.weight", {(uint64_t)g.o_group_dim(), (uint64_t)(g.o_lora * g.o_groups)},
                         ds4_type::Q8_0));
        v.push_back(spec(p + "attn_output_b.weight", {(uint64_t)(g.o_groups * g.o_lora), D}, ds4_type::Q8_0));
        v.push_back(spec(p + "hc_attn_fn.weight", {hc * D, (uint64_t)g.hc_mix_dim()}, ds4_type::F16));
        v.push_back(spec(p + "hc_attn_base.weight", {(uint64_t)g.hc_mix_dim()}, ds4_type::F32));
        v.push_back(spec(p + "hc_attn_scale.weight", {3}, ds4_type::F32));
        v.push_back(spec(p + "hc_ffn_fn.weight", {hc * D, (uint64_t)g.hc_mix_dim()}, ds4_type::F16));
        v.push_back(spec(p + "hc_ffn_base.weight", {(uint64_t)g.hc_mix_dim()}, ds4_type::F32));
        v.push_back(spec(p + "hc_ffn_scale.weight", {3}, ds4_type::F32));

        const int64_t ratio = i < (int64_t)g.compress_ratios.size() ? g.compress_ratios[(size_t)i] : 0;
        if (ratio != 0) {
            const uint64_t coff = ratio == 4 ? 2 : 1;
            v.push_back(spec(p + "attn_compressor_kv.weight", {D, coff * hd}, ds4_type::F16));
            v.push_back(spec(p + "attn_compressor_gate.weight", {D, coff * hd}, ds4_type::F16));
            v.push_back(spec(p + "attn_compressor_ape.weight", {coff * hd, (uint64_t)ratio}, ds4_type::F16));
            v.push_back(spec(p + "attn_compressor_norm.weight", {hd}, ds4_type::F32));
            if (ratio == 4) {
                const uint64_t ik = (uint64_t)g.indexer_key_dim;
                v.push_back(spec(p + "indexer.proj.weight", {D, (uint64_t)g.indexer_n_head}, ds4_type::F16));
                v.push_back(spec(p + "indexer.attn_q_b.weight", {rq, (uint64_t)g.indexer_n_head * ik}, ds4_type::F16));
                v.push_back(spec(p + "indexer_compressor_kv.weight", {D, 2 * ik}, ds4_type::F16));
                v.push_back(spec(p + "indexer_compressor_gate.weight", {D, 2 * ik}, ds4_type::F16));
                v.push_back(spec(p + "indexer_compressor_ape.weight", {2 * ik, (uint64_t)ratio}, ds4_type::F16));
                v.push_back(spec(p + "indexer_compressor_norm.weight", {ik}, ds4_type::F32));
            }
        }

        v.push_back(spec(p + "ffn_gate_inp.weight", {D, (uint64_t)g.n_expert}, ds4_type::F16));
        v.push_back(spec(p + "ffn_norm.weight", {D}, ds4_type::F32));
        v.push_back(spec(p + "ffn_gate_exps.weight", {D, (uint64_t)g.n_ff, (uint64_t)g.n_expert}, ds4_type::IQ2_XXS));
        v.push_back(spec(p + "ffn_down_exps.weight", {(uint64_t)g.n_ff, D, (uint64_t)g.n_expert}, ds4_type::Q2_K));
        v.push_back(spec(p + "ffn_up_exps.weight", {D, (uint64_t)g.n_ff, (uint64_t)g.n_expert}, ds4_type::IQ2_XXS));
        v.push_back(spec(p + "ffn_down_shexp.weight", {(uint64_t)(g.n_ff * g.n_expert_shared), D}, ds4_type::Q8_0));
        v.push_back(spec(p + "ffn_gate_shexp.weight", {D, (uint64_t)(g.n_ff * g.n_expert_shared)}, ds4_type::Q8_0));
        v.push_back(spec(p + "ffn_up_shexp.weight", {D, (uint64_t)(g.n_ff * g.n_expert_shared)}, ds4_type::Q8_0));

        if (is_mtp) {
            // NextN block: llama.cpp asserts a single MTP layer and these three are required; the three
            // embed_tokens / shared_head_* tensors are optional (they fall back to the trunk tensors).
            v.push_back(spec(p + "nextn.eh_proj.weight", {2 * D, D}, ds4_type::F16));
            v.push_back(spec(p + "nextn.enorm.weight", {D}, ds4_type::F32));
            v.push_back(spec(p + "nextn.hnorm.weight", {D}, ds4_type::F32));
        } else if (i < g.hash_layer_count) {
            v.push_back(spec(p + "ffn_gate_tid2eid.weight", {(uint64_t)g.n_expert_used, V}, ds4_type::I32));
        } else {
            v.push_back(spec(p + "exp_probs_b.bias", {(uint64_t)g.n_expert}, ds4_type::F32));
        }
    }
    return v;
}

/// Resolve the full tensor map against a model.  Returns "" when every tensor in the file is expected AND
/// every expected tensor is present with the shape and type the kernels assume; otherwise the first failure,
/// naming the tensor (and, for a shape/type mismatch, what it has vs what is required).
inline std::string check_ds4_model(const GgufModel& model, const Ds4Geometry& g) {
    std::vector<Ds4TensorSpec> want = ds4_expected_tensors(g);
    std::map<std::string, const Ds4TensorSpec*> by_name;
    for (const auto& s : want) by_name[s.name] = &s;

    for (size_t sh = 0; sh < model.size(); ++sh) {
        for (const auto& t : model.shard(sh).tensors()) {
            auto it = by_name.find(t.name);
            if (it == by_name.end())
                return "unknown tensor '" + t.name + "' in " + model.shard(sh).path() +
                       " (not in the deepseek4 tensor map)";
            const Ds4TensorSpec& w = *it->second;
            if (t.shape.size() != w.shape.size()) {
                std::string got;
                for (auto d : t.shape) got += (got.empty() ? "" : ",") + std::to_string(d);
                return "tensor '" + t.name + "' has " + std::to_string(t.shape.size()) + " dims (" + got +
                       "), expected " + std::to_string(w.shape.size());
            }
            for (size_t d = 0; d < w.shape.size(); ++d)
                if (t.shape[d] != w.shape[d])
                    return "tensor '" + t.name + "' dim " + std::to_string(d) + " = " + std::to_string(t.shape[d]) +
                           ", expected " + std::to_string(w.shape[d]);
            if (t.type != w.type)
                return "tensor '" + t.name + "' is " + ggml_type_name(t.type) + ", expected " +
                       ggml_type_name(w.type);
            if (!model.in_bounds(t, sh))
                return "tensor '" + t.name + "' payload is outside " + model.shard(sh).path();
        }
    }
    for (const auto& s : want)
        if (model.find(s.name) == nullptr) return "missing tensor '" + s.name + "'";
    return {};
}

}  // namespace strata
