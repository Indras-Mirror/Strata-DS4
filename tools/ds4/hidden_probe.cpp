// hidden_probe: run DeepSeek-V4 text through llama.cpp and dump, per layer and token, the router's selected experts
// plus the hidden states an EARLY expert predictor could use - to measure whether layer l's experts can be predicted
// before layer l's attention finishes (prefetch them over the idle PCIe link during attention; FINDINGS s12).
//
//   hidden_probe model.gguf out.bin n_tokens_per_file file1 [file2...]
//
// Records, back to back: struct { char kind; uint8 pad; uint16 layer; uint32 n_tok; uint32 width; } + payload
//   kind 'A' = hc_attn_pre-l  (f32 [4096, n_tok]: layer l's pre-attention mixed stream, known BEFORE attention l)
//   kind 'F' = ffn_norm-l     (f32 [4096, n_tok]: the router's actual input)
//   kind 'K' = ffn_moe_topk-l (i32 [6, n_tok]: the selected experts)
//   kind 'T' = tokens         (i32 [n_tok], layer 0xffff)
// Same placement as route_probe (experts on the CPU via mmap, the rest on the GPU) - run it under memguard.sh.
//
// Build: as golden_dump.cpp's header, with tools/ds4/hidden_probe.cpp -o build-ds4-cuda/hidden_probe
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static FILE * g_out = nullptr;

#pragma pack(push, 1)
struct Rec { char kind; uint8_t pad; uint16_t layer; uint32_t n_tok; uint32_t width; };
#pragma pack(pop)

static void put(char kind, int layer, const ggml_tensor * t) {
    static std::vector<char> buf, packed;
    const size_t nb = ggml_nbytes(t);
    buf.resize(nb);
    const char * base = (const char *) t->data;
    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) { ggml_backend_tensor_get(t, buf.data(), 0, nb); base = buf.data(); }
    const int64_t w = t->ne[0], n = t->ne[1];
    packed.resize((size_t) w * n * 4);
    for (int64_t j = 0; j < n; j++)
        for (int64_t i = 0; i < w; i++)
            std::memcpy(packed.data() + (j * w + i) * 4, base + i * t->nb[0] + j * t->nb[1], 4);
    Rec r{kind, 0, (uint16_t) layer, (uint32_t) n, (uint32_t) w};
    fwrite(&r, sizeof r, 1, g_out);
    fwrite(packed.data(), 1, packed.size(), g_out);
}

static bool cb(struct ggml_tensor * t, bool ask, void *) {
    char kind = 0; size_t plen = 0;
    if (!strncmp(t->name, "hc_attn_pre-", 12)) { kind = 'A'; plen = 12; }
    else if (!strncmp(t->name, "ffn_norm-", 9)) { kind = 'F'; plen = 9; }
    else if (!strncmp(t->name, "ffn_moe_topk-", 13)) { kind = 'K'; plen = 13; }
    if (!kind) return ask ? false : true;
    if (ask) return true;
    if (t->type != (kind == 'K' ? GGML_TYPE_I32 : GGML_TYPE_F32) || ggml_n_dims(t) > 2) {
        fprintf(stderr, "hidden_probe: unexpected %s type %d dims %d\n", t->name, t->type, ggml_n_dims(t));
        return true;
    }
    put(kind, atoi(t->name + plen), t);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: hidden_probe model.gguf out.bin n_tok file1 [file2...]\n"); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    static llama_model_tensor_buft_override ov[] = {{"_exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    g_out = fopen(argv[2], "wb");
    const int n_max = atoi(argv[3]);
    for (int f = 4; f < argc; f++) {
        std::ifstream in(argv[f]); std::stringstream ss; ss << in.rdbuf(); std::string text = ss.str();
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 3; }
        n = std::min(n, n_max); toks.resize(n);
        Rec r{'T', 0, 0xffff, (uint32_t) n, 1};
        fwrite(&r, sizeof r, 1, g_out); fwrite(toks.data(), 4, n, g_out);
        auto cp = llama_context_default_params(); cp.n_ctx = n + 256; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_threads = 8; cp.n_threads_batch = 8; cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
        llama_context * ctx = llama_init_from_model(model, cp);
        for (int i = 0; i < n; i += 512) {
            int m = std::min(512, n - i);
            llama_batch b = llama_batch_get_one(toks.data() + i, m);
            if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 4; }
            fprintf(stderr, "FILE %s ubatch at %d done\n", argv[f], i);
        }
        fflush(g_out);
        llama_free(ctx);
    }
    fclose(g_out); llama_model_free(model); return 0;
}
