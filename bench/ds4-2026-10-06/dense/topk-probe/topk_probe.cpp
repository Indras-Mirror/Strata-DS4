// topk_probe.cpp - does the indexer mask chain (top_k -> fill -inf -> set_rows 0 -> + vis) give the same mask on
// ggml-cpu and ggml-cuda?  cap 16, 9 visible rows (pos 36 of the mini fixture), top_k 8.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <vector>
static std::vector<float> run(ggml_backend_t be, const std::vector<float>& sc_h, const std::vector<float>& vis_h,
                              std::vector<int>& tk_h) {
    const int cap = (int) sc_h.size(), ntk = 8;
    ggml_init_params ip = {64 << 20, nullptr, true};
    ggml_context* gc = ggml_init(ip);
    ggml_tensor* sc = ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, 1); ggml_set_input(sc);
    ggml_tensor* vis = ggml_new_tensor_2d(gc, GGML_TYPE_F32, cap, 1); ggml_set_input(vis);
    ggml_tensor* s2 = ggml_add(gc, sc, vis);
    ggml_tensor* tk = ggml_cont(gc, ggml_top_k(gc, s2, ntk));
    ggml_set_output(tk);
    ggml_tensor* a = ggml_fill(gc, vis, -INFINITY);
    a = ggml_view_4d(gc, a, 1, cap, 1, 1, a->nb[0], a->nb[1], a->nb[2], 0);
    ggml_tensor* zv = ggml_fill(gc, ggml_new_tensor_3d(gc, GGML_TYPE_F32, 1, ntk, 1), 0.0f);
    ggml_tensor* m = ggml_set_rows(gc, a, zv, tk);
    m = ggml_view_4d(gc, m, m->ne[1], m->ne[2], 1, 1, m->nb[2], m->nb[3], m->nb[3], 0);
    m = ggml_add(gc, m, vis);
    ggml_set_output(m);
    ggml_cgraph* gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, tk);
    ggml_build_forward_expand(gf, m);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(al, gf);
    ggml_backend_tensor_set(sc, sc_h.data(), 0, cap * 4);
    ggml_backend_tensor_set(vis, vis_h.data(), 0, cap * 4);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> out(cap);
    ggml_backend_tensor_get(m, out.data(), 0, cap * 4);
    tk_h.resize(ntk);
    ggml_backend_tensor_get(tk, tk_h.data(), 0, ntk * 4);
    ggml_gallocr_free(al);
    ggml_free(gc);
    return out;
}
int main() {
    const int cap = 16;
    std::vector<float> sc(cap), vis(cap);
    for (int i = 0; i < cap; ++i) { sc[i] = 0.1f * ((i * 7) % 11) + 0.01f * i; vis[i] = i < 9 ? 0.0f : -INFINITY; }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    std::vector<int> tc, tg;
    auto mc = run(cpu, sc, vis, tc), mg = run(gpu, sc, vis, tg);
    std::printf("scores:"); for (int i = 0; i < cap; ++i) std::printf(" %.2f", vis[i] == 0 ? sc[i] : -INFINITY); std::printf("\n");
    std::printf("cpu topk:"); for (int v : tc) std::printf(" %d", v); std::printf("\n");
    std::printf("gpu topk:"); for (int v : tg) std::printf(" %d", v); std::printf("\n");
    std::printf("cpu mask:"); for (float v : mc) std::printf(" %s", std::isinf(v) ? "-" : "0"); std::printf("\n");
    std::printf("gpu mask:"); for (float v : mg) std::printf(" %s", std::isinf(v) ? "-" : "0"); std::printf("\n");
}
