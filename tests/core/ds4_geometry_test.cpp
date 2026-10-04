// tests/core/ds4_geometry_test.cpp - the Phase-1 geometry + tensor-map gate for DeepSeek-V4-Flash.
//
// Two parts:
//
//   * A pure unit test of the conditional tensor map, on a synthesized geometry with the 0731 artifact's
//     numbers.  No file, no GPU: it checks the count (1328) and that the compressor / indexer / hash-router
//     tensors appear on exactly the layers the architecture says.
//
//   * The real-file gate: open the GGUF (header mmap only, no tensor data), read every `deepseek4.*` key into
//     `Ds4Geometry`, assert the geometry, and resolve the full tensor map with ZERO unknowns and zero missing.
//
// The GGUF path is argv[1] or $DS4_GGUF, default the 0731 artifact.  When it is absent the real-file part is
// skipped with a printed reason (so a CI box without the 80 GB file still builds and runs the unit part); the
// Phase-1 gate is the run WITH the path, whose command is in the report.
#include "strata/artifact/ds4_geometry.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// The 0731 artifact's geometry, filled here the way ds4_read_geometry fills it.
strata::Ds4Geometry artifact_geometry() {
    strata::Ds4Geometry g;
    g.n_embd = 4096;
    g.n_layer_all = 43;
    g.n_layer_nextn_requested = 1;
    g.n_layer_nextn = 0;  // the artifact has the key but no nextn tensors
    g.n_head = 64;
    g.n_head_kv = 1;
    g.d_head = 512;
    g.d_rope = 64;
    g.r_q = 1024;
    g.o_groups = 8;
    g.o_lora = 1024;
    g.n_swa = 128;
    g.n_expert = 256;
    g.n_expert_used = 6;
    g.n_ff = 2048;
    g.n_expert_shared = 1;
    g.weights_scale = 1.5;
    g.weights_norm = true;
    g.gating_func = 4;
    g.hc = 4;
    g.hc_sinkhorn_iters = 20;
    g.hc_eps = 1e-6;
    g.hash_layer_count = 3;
    g.indexer_n_head = 64;
    g.indexer_key_dim = 128;
    g.indexer_top_k = 512;
    g.rms_eps = 1e-6;
    g.rope_scaling_type = "yarn";
    g.rope_freq_base = 10000;
    g.rope_factor = 16;
    g.rope_orig_ctx = 65536;
    g.yarn_beta_fast = 32;
    g.yarn_beta_slow = 1;
    g.compress_rope_base = 160000;
    g.vocab_size = 129280;
    g.context_length = 1048576;
    g.tokenizer_pre = "joyai-llm";
    g.compress_ratios.assign(g.n_layer_all + 1, 0);
    for (int i = 2; i <= 42; ++i) g.compress_ratios[i] = (i % 2 == 0) ? 4 : 128;
    g.swiglu_clamp_exp.assign(g.n_layer_all, 10.0);
    g.swiglu_clamp_shexp = g.swiglu_clamp_exp;
    return g;
}

bool has_name(const std::vector<strata::Ds4TensorSpec>& v, const std::string& n) {
    for (const auto& s : v) if (s.name == n) return true;
    return false;
}

void unit_tests() {
    const strata::Ds4Geometry g = artifact_geometry();
    const auto t = strata::ds4_expected_tensors(g);
    check(t.size() == 1328, "synthesized map has 1328 tensors");

    check(!has_name(t, "blk.0.attn_compressor_kv.weight") && !has_name(t, "blk.0.indexer.proj.weight"),
          "layer 0 (ratio 0): no compressor / indexer");
    check(has_name(t, "blk.3.attn_compressor_kv.weight") && !has_name(t, "blk.3.indexer.proj.weight"),
          "layer 3 (ratio 128, HCA): compressor but no indexer");
    check(has_name(t, "blk.42.indexer_compressor_norm.weight") && has_name(t, "blk.42.indexer.proj.weight"),
          "layer 42 (ratio 4, CSA): indexer present");
    check(has_name(t, "blk.0.ffn_gate_tid2eid.weight") && has_name(t, "blk.2.ffn_gate_tid2eid.weight"),
          "hash layers 0-2 carry ffn_gate_tid2eid");
    check(!has_name(t, "blk.0.exp_probs_b.bias") && has_name(t, "blk.3.exp_probs_b.bias"),
          "non-hash layers carry exp_probs_b");
    check(!has_name(t, "blk.42.nextn.eh_proj.weight"), "no MTP block when nextn is zeroed");
    check(has_name(t, "output_hc_fn.weight") && has_name(t, "output_hc_scale.weight"),
          "root hc head tensors present");

    // shape spot checks against the real artifact (GGUF order)
    for (const auto& s : t) {
        if (s.name == "blk.42.attn_output_a.weight")
            check(s.shape == std::vector<uint64_t>({4096, 8192}), "attn_output_a is [n_heads*d_h/8, o_lora*8]");
        if (s.name == "blk.42.indexer.attn_q_b.weight")
            check(s.shape == std::vector<uint64_t>({1024, 8192}), "indexer.attn_q_b is [r_q, idx_heads*idx_key]");
        if (s.name == "blk.3.attn_compressor_ape.weight")
            check(s.shape == std::vector<uint64_t>({512, 128}), "HCA compressor APE is [d_h, 128]");
    }
}

void real_file(const std::string& path) {
    strata::GgufModel model = strata::GgufModel::open(path);
    std::printf("  opened %s: %zu shard(s), %zu tensors\n", path.c_str(), model.size(),
                model.meta().tensors().size());
    std::string err = strata::check_architecture_ds4(model.meta());
    check(err.empty(), err.empty() ? "architecture is deepseek4 with the expected constants" : err.c_str());
    if (!err.empty()) return;

    strata::Ds4Geometry g;
    err = strata::ds4_read_geometry(model, g);
    check(err.empty(), err.empty() ? "all deepseek4.* keys read into Ds4Geometry" : err.c_str());
    if (!err.empty()) return;

    check(g.n_embd == 4096 && g.n_layer_all == 43, "D=4096, layers=43");
    check(g.n_head == 64 && g.n_head_kv == 1 && g.d_head == 512 && g.d_rope == 64, "64 q heads / 1 kv head / d_h 512 / rope 64");
    check(g.r_q == 1024 && g.o_groups == 8 && g.o_lora == 1024, "q_lora 1024, 8 output groups, o_lora 1024");
    check(g.n_expert == 256 && g.n_expert_used == 6 && g.n_ff == 2048 && g.n_expert_shared == 1,
          "256 experts, top-6, n_ff 2048, 1 shared expert");
    check(g.hc == 4 && g.hc_sinkhorn_iters == 20 && g.hash_layer_count == 3, "hc 4, sinkhorn 20, 3 hash layers");
    check(g.hc_mix_dim() == 24 && g.hc_dim() == 16384 && g.heads_dim() == 32768 && g.o_group_dim() == 4096,
          "derived dims: hc_mix 24, hc_dim 16384, heads_dim 32768, o_group_dim 4096");
    check(g.indexer_n_head == 64 && g.indexer_key_dim == 128 && g.indexer_top_k == 512, "indexer 64/128/512");
    check(g.weights_scale == 1.5 && g.weights_norm, "routed scale 1.5, weights normalized");
    check(g.rope_scaling_type == "yarn" && g.rope_factor == 16 && g.rope_orig_ctx == 65536,
          "YaRN factor 16, orig ctx 65536");
    check(g.compress_rope_base == 160000, "compress rope base 160000");
    check(g.n_swa == 128, "sliding window 128");
    check(g.n_layer_nextn_requested == 1 && g.n_layer_nextn == 0 && g.n_layer() == 43,
          "nextn requested 1, zeroed to 0 (no MTP tensors), 43 trunk layers");
    check(g.tokenizer_pre == "joyai-llm", "tokenizer pre is joyai-llm");
    check(g.compress_ratios.size() >= (size_t)g.n_layer_all, "compress_ratios covers every block");

    const auto want = strata::ds4_expected_tensors(g);
    const size_t nfile = model.meta().tensors().size();
    check(want.size() == nfile, "expected tensor count equals the file's");
    err = strata::check_ds4_model(model, g);
    check(err.empty(), err.empty() ? "full tensor map resolves: zero unknowns, zero missing" : err.c_str());
    if (!err.empty()) std::printf("    %s\n", err.c_str());
}
}  // namespace

int main(int argc, char** argv) {
    unit_tests();
    std::string path = argc > 1 ? argv[1]
                                : (std::getenv("DS4_GGUF") ? std::getenv("DS4_GGUF")
                                                           : "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf");
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("  SKIP real-file gate: %s not readable (pass argv[1] or $DS4_GGUF)\n", path.c_str());
    } else {
        std::fclose(f);
        real_file(path);
    }
    std::printf("%s (%d failure%s)\n", g_fail ? "FAIL" : "PASS", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
