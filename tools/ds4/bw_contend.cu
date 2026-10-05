// bw_contend.cu - does PCIe DMA out of pinned host RAM steal DDR4 bandwidth from CPU expert reads?
//   nvcc -O3 -Xcompiler "-O3 -march=native -fopenmp" tools/ds4/bw_contend.cu -o build-ds4-cuda/bw_contend
// CPU: 7 threads sum a 2 GiB buffer (bandwidth bound). GPU: H2D copies from a 1 GiB pinned buffer.
// Reports CPU GB/s alone, PCIe GB/s alone, and both concurrently.
#include <cuda_runtime.h>
#include <omp.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double cpu_read(const float* a, size_t n, int reps, int thr) {
    double t0 = now(); volatile float sink = 0;
    for (int r = 0; r < reps; ++r) {
        float tot = 0;
#pragma omp parallel for num_threads(thr) reduction(+:tot) schedule(static)
        for (size_t b = 0; b < n / 64; ++b) {
            __m256 s = _mm256_setzero_ps();
            for (int k = 0; k < 64; k += 8) s = _mm256_add_ps(s, _mm256_loadu_ps(a + b * 64 + k));
            float o[8]; _mm256_storeu_ps(o, s); tot += o[0] + o[7];
        }
        sink = sink + tot;
    }
    return (double) n * 4 * reps / (now() - t0) / 1e9;
}
int main() {
    const size_t cpu_bytes = 2ull << 30, pin_bytes = 1ull << 30;
    float* a = (float*) aligned_alloc(64, cpu_bytes);
    memset(a, 1, cpu_bytes);
    void *h, *d; cudaStream_t s;
    cudaMallocHost(&h, pin_bytes); memset(h, 2, pin_bytes);
    cudaMalloc(&d, pin_bytes); cudaStreamCreate(&s);
    auto dma = [&](int reps) { double t0 = now(); for (int i = 0; i < reps; ++i) cudaMemcpyAsync(d, h, pin_bytes, cudaMemcpyHostToDevice, s);
                               cudaStreamSynchronize(s); return (double) pin_bytes * reps / (now() - t0) / 1e9; };
    cpu_read(a, cpu_bytes / 4, 2, 7); dma(2);   // warm
    for (int thr : {7, 8}) std::printf("CPU alone (%d thr): %.1f GB/s\n", thr, cpu_read(a, cpu_bytes / 4, 10, thr));
    std::printf("PCIe alone: %.1f GB/s\n", dma(10));
    std::atomic<bool> stop{false}; double pcie_bw = 0; long nrep = 0;
    std::thread g([&] { double t0 = now(); while (!stop) { cudaMemcpyAsync(d, h, pin_bytes, cudaMemcpyHostToDevice, s); cudaStreamSynchronize(s); ++nrep; }
                        pcie_bw = (double) pin_bytes * nrep / (now() - t0) / 1e9; });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    double c = cpu_read(a, cpu_bytes / 4, 10, 7);
    stop = true; g.join();
    std::printf("CONCURRENT: CPU %.1f GB/s + PCIe %.1f GB/s = %.1f GB/s total\n", c, pcie_bw, c + pcie_bw);
    return 0;
}
