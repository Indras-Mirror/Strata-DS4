// tools/ds4/ds4_mtp_bench.cpp - times ds4::MtpExperts::run (the MTP block's routed experts, CPU ggml mul_mat_id
// straight off the MTP file's mmap) with random routed ids: is the 16 ms/draft measured in ds4_generate the
// kernel, or page faults on a file whose pages were evicted?  CPU only.
//
//   ds4_mtp_bench FILE [--iters 50] [--threads 8] [--d 4096] [--k 6] [--il 43] [--nexp 256] [--cold] [--resident]
//
// --cold: before every run, MADV_PAGEOUT the whole mapping (the file's pages leave the page cache), so each run
// reads its experts from disk - the worst case of a run whose page cache is under pressure (memguard's cgroup cap
// counts page cache).  Residency (mincore) is printed before/after.  --resident: the experts copied into RAM at init
// (ds4_generate --mtp-resident).

#include "mtp_experts.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ds4_mtp_bench FILE [--iters N] [--threads N] [--d D] [--k K] [--il L] [--nexp E] [--cold]\n");
        return 2;
    }
    const std::string path = argv[1];
    int iters = 50, threads = 8, il = 43, nexp = 256;
    int64_t D = 4096, K = 6;
    bool cold = false, resident = false;
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::atoll(argv[++i]) : 0LL; };
        if (k == "--iters") iters = (int) next();
        else if (k == "--threads") threads = (int) next();
        else if (k == "--d") D = next();
        else if (k == "--k") K = next();
        else if (k == "--il") il = (int) next();
        else if (k == "--nexp") nexp = (int) next();
        else if (k == "--cold") cold = true;
        else if (k == "--resident") resident = true;
        else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    strata::ds4::MtpExperts mx;
    std::string err;
    if (!mx.init(path, il, D, K, 10.0f, threads, err, resident)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }

    const strata::GgufFile& sh = mx.model->shard(0);
    const auto& ts = sh.tensors();
    uint8_t* base = const_cast<uint8_t*>(sh.tensor_data(ts[0])) - sh.data_start() - ts[0].offset;
    const size_t len = (size_t) sh.file_size();
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    auto resident_gib = [&]() {
        std::vector<unsigned char> v((len + pg - 1) / pg);
        if (mincore(base, len, v.data()) != 0) return -1.0;
        size_t n = 0;
        for (unsigned char c : v) n += c & 1;
        return (double) n * pg / (1u << 30);
    };
    std::printf("file %.2f GiB, resident %.2f GiB at start, %d threads, %s\n", len / double(1u << 30), resident_gib(),
                threads, cold ? "COLD (MADV_PAGEOUT before each run)" : "warm");

    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) D), o((size_t) D), w((size_t) K, 1.0f / (float) K);
    std::vector<int> all(nexp), ids((size_t) K);
    std::iota(all.begin(), all.end(), 0);
    std::vector<double> ms;
    for (int it = -2; it < iters; ++it) {   // 2 untimed warm-up runs
        for (auto& v : x) v = nd(rng);
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + K, ids.begin());
        if (cold) madvise(base, len, MADV_PAGEOUT);
        const double t0 = now_ms();
        if (!mx.run(x.data(), ids.data(), w.data(), o.data())) { std::fprintf(stderr, "run failed\n"); return 1; }
        if (it >= 0) ms.push_back(now_ms() - t0);
    }
    std::sort(ms.begin(), ms.end());
    const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
    // bytes per run: K experts x (gate + up + down) MXFP4 (17 bytes / 32 weights); n_ff from the gate tensor
    const strata::TensorInfo* g = mx.model->find("blk." + std::to_string(il) + ".ffn_gate_exps.weight");
    const double nff = g ? (double) g->shape.at(1) : 0;
    const double mb = K * 3.0 * D * nff * 17.0 / 32.0 / 1e6;
    std::printf("run: mean %.2f ms, median %.2f, min %.2f, max %.2f over %d; %.1f MB of experts/run = %.1f GB/s (mean)\n",
                mean, ms[ms.size() / 2], ms.front(), ms.back(), (int) ms.size(), mb, mb / mean);
    std::printf("resident %.2f GiB at the end; expert spans resident %.1f%% (MtpExperts::resident_frac)\n", resident_gib(),
                100.0 * mx.resident_frac());
    return 0;
}
