// route_probe: run DeepSeek-V4 prompts through llama.cpp (CPU) and dump every token's routed experts per layer.
// out: binary file of records [layer:u16][expert:u16] per (token,layer,k), plus a header line in stderr.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
static FILE * g_out = nullptr;
static long g_recs = 0;
static bool cb(struct ggml_tensor * t, bool ask, void *) {
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) return ask ? false : true;
    if (ask) return true;
    int layer = atoi(t->name + 13);
    // [k, n_tokens] int32, possibly a strided view, possibly in VRAM: copy its bytes out first
    static std::vector<char> buf;
    size_t nb = ggml_nbytes(t);
    buf.resize(nb);
    const char * base = (const char *) t->data;
    if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) { ggml_backend_tensor_get(t, buf.data(), 0, nb); base = buf.data(); }
    for (int64_t j = 0; j < t->ne[1]; j++)
        for (int64_t i = 0; i < t->ne[0]; i++) {
            int32_t e = *(const int32_t *)(base + i * t->nb[0] + j * t->nb[1]);
            uint16_t rec[2] = {(uint16_t)layer, (uint16_t)e};
            fwrite(rec, sizeof rec, 1, g_out); g_recs++;
        }
    return true;
}
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: route_probe model.gguf out.bin file1 [file2...]\n"); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    static llama_model_tensor_buft_override ov[] = {{"_exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    g_out = fopen(argv[2], "wb");
    for (int f = 3; f < argc; f++) {
        std::ifstream in(argv[f]); std::stringstream ss; ss << in.rdbuf(); std::string text = ss.str();
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 3; }
        if (n > 2048) n = 2048; toks.resize(n);
        auto cp = llama_context_default_params(); cp.n_ctx = 2304; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_threads = 16; cp.n_threads_batch = 16; cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
        llama_context * ctx = llama_init_from_model(model, cp);
        for (int i = 0; i < n; i += 512) {
            int m = std::min(512, n - i);
            llama_batch b = llama_batch_get_one(toks.data() + i, m);
            if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 4; }
        }
        fprintf(stderr, "FILE %s tokens %d records %ld\n", argv[f], n, g_recs); fflush(g_out);
        llama_free(ctx);
    }
    fclose(g_out); llama_model_free(model); return 0;
}
