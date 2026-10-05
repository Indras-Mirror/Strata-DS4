// tools/ds4/moe_gap.cu - the synthetic per-layer GPU gap of tools/ds4/moe_replay.
//
// The Phase-3 slice owns the real deepseek4 attention; Phase 4a only needs the GPU to be BUSY for the measured
// length of a layer's dense+attention work (Phase 0: ~0.4 ms) so that the MoE engine's overlap is measured
// against something.  clock64() rather than an FMA loop, so the duration does not depend on the boost clock the
// driver picks.  One tiny kernel in its own translation unit keeps moe_replay.cpp plain C++.
#include <cuda_runtime.h>

namespace {

__global__ void gap_kernel(unsigned long long cycles) {
    const unsigned long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

}  // namespace

extern "C" void moe_gap_launch(unsigned long long cycles, void* stream) {
    gap_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(cycles);
}

namespace {

// wall-clock version for Phase 4b: %globaltimer is nanoseconds, independent of the SM clock, so a target of
// 0.41 ms is 0.41 ms whatever boost bin the driver picks (the clock64 version under-delivered ~2.6x in 4a).
__global__ void gap_kernel_ns(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}

}  // namespace

extern "C" void moe_gap_launch_ns(unsigned long long ns, void* stream) {
    gap_kernel_ns<<<1, 32, 0, (cudaStream_t) stream>>>(ns);
}
