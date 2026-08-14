// Machine-ceiling probe for the roofline baseline.
//
// Measures, on the Metal device of this machine, the two numbers a roofline
// needs: achieved streaming memory bandwidth and achieved fused-multiply-add
// throughput. Both are measured, not read off a spec sheet, because the
// spec figures are ALU-count times boost clock and no kernel sustains that.
//
// Timing uses the command buffer's own GPU timestamps, so host-side
// submission cost is excluded. Every configuration is run REPS times and the
// best pass is reported: the best pass is the one least polluted by other
// work on a shared laptop GPU.
//
// Build (bench/roofline.py does this for you):
//   clang++ -std=c++17 -fobjc-arc -O2 -framework Metal -framework Foundation \
//       bench/metal/roofline_probe.mm -o bench/.cache/roofline_probe
//
// Output: one JSON object on stdout.

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static const char *kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// ---- bandwidth ----------------------------------------------------------
// One float4 read + one float4 written per thread: 32 bytes of traffic.
kernel void stream_copy(device const float4 *in  [[buffer(0)]],
                        device       float4 *out [[buffer(1)]],
                        constant     float  &a   [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    out[gid] = in[gid] * a;
}

// Read-only: a grid-stride sum, so the write-back path contributes no
// traffic. The store is guarded by a condition that never holds, which the
// compiler cannot prove, so the loads survive dead-code elimination.
//
// Four independent accumulators, four loads in flight per iteration: a single
// accumulator serialises on its own dependency chain and measures latency
// rather than bandwidth. This is the kernel that sets the ceiling every other
// number in the baseline is divided by, so it is worth the unroll.
kernel void stream_read(device const float4 *in    [[buffer(0)]],
                        device       float  *out   [[buffer(1)]],
                        constant     uint   &n4    [[buffer(2)]],
                        uint gid   [[thread_position_in_grid]],
                        uint gsize [[threads_per_grid]]) {
    float4 a0 = float4(0.0f), a1 = a0, a2 = a0, a3 = a0;
    uint i = gid;
    for (; i + 3 * gsize < n4; i += 4 * gsize) {
        a0 += in[i];
        a1 += in[i + gsize];
        a2 += in[i + 2 * gsize];
        a3 += in[i + 3 * gsize];
    }
    for (; i < n4; i += gsize) a0 += in[i];
    float4 acc = a0 + a1 + a2 + a3;
    if (acc.x == 1.0e38f) out[gid] = acc.x + acc.y + acc.z + acc.w;
}

// Fills a private buffer without a host copy.
kernel void fill_buf(device float4 *buf [[buffer(0)]],
                     uint gid [[thread_position_in_grid]]) {
    buf[gid] = float4(1.0f, 1.0f, 1.0f, 1.0f);
}

// ---- FMA throughput -----------------------------------------------------
// Eight independent accumulator chains give the scheduler enough
// instruction-level parallelism to cover FMA latency; the loop body is
// unrolled four deep so loop overhead is not what is being measured.
// 32 FMAs per iteration = 64 flops per iteration per thread.
#define FMA_BODY(T)                                                        \
    T a0 = T(gid & 7), a1 = a0 + T(1), a2 = a0 + T(2), a3 = a0 + T(3);     \
    T a4 = a0 + T(4), a5 = a0 + T(5), a6 = a0 + T(6), a7 = a0 + T(7);      \
    const T b = T(1.0009765625), c = T(0.0009765625);                      \
    for (uint i = 0; i < iters; ++i) {                                     \
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);                            \
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);                            \
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);                            \
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);                            \
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);                            \
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);                            \
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);                            \
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);                            \
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);                            \
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);                            \
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);                            \
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);                            \
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);                            \
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);                            \
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);                            \
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);                            \
    }                                                                      \
    out[gid] = float(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7);

kernel void fma_f32(device float *out   [[buffer(0)]],
                    constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    FMA_BODY(float)
}

kernel void fma_f16(device float *out   [[buffer(0)]],
                    constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    FMA_BODY(half)
}

// Same chains on half4: Apple GPUs can issue packed 16-bit arithmetic, and
// whether that doubles throughput is the difference between a 6 TFLOP and a
// 12 TFLOP ceiling for an fp16 kernel, so it is measured rather than assumed.
kernel void fma_f16x4(device float *out   [[buffer(0)]],
                      constant uint &iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    half4 a0 = half4(gid & 7), a1 = a0 + half4(1), a2 = a0 + half4(2), a3 = a0 + half4(3);
    half4 a4 = a0 + half4(4), a5 = a0 + half4(5), a6 = a0 + half4(6), a7 = a0 + half4(7);
    const half4 b = half4(1.0009765625h), c = half4(0.0009765625h);
    for (uint i = 0; i < iters; ++i) {
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);
        a0 = fma(a0, b, c); a1 = fma(a1, b, c);
        a2 = fma(a2, b, c); a3 = fma(a3, b, c);
        a4 = fma(a4, b, c); a5 = fma(a5, b, c);
        a6 = fma(a6, b, c); a7 = fma(a7, b, c);
    }
    half4 s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    out[gid] = float(s.x + s.y + s.z + s.w);
}
)METAL";

static const int kReps = 12;

struct Ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
};

// ---- CPU ceilings -------------------------------------------------------
// The CPU shares the same memory system as the GPU but reaches a different
// fraction of it, and llama.cpp's CPU path is a real backend on this machine,
// so scoring its results against the GPU's ceilings would be meaningless.

static double cpu_triad_gbs(size_t bytes, int threads, int reps) {
    const size_t n = bytes / sizeof(float);
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f);
    const float s = 3.0f;
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                const size_t lo = n * t / threads, hi = n * (t + 1) / threads;
                for (size_t i = lo; i < hi; ++i) c[i] = a[i] + s * b[i];
            });
        }
        for (auto &th : pool) th.join();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt < best) best = dt;
    }
    // two streams read, one written
    return (double)(3 * bytes) / best / 1e9;
}

static double cpu_sgemm_gflops(int n, int reps) {
    std::vector<float> A((size_t)n * n, 1.0f), B((size_t)n * n, 1.0f), C((size_t)n * n, 0.0f);
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
                    A.data(), n, B.data(), n, 0.0f, C.data(), n);
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt < best) best = dt;
    }
    return 2.0 * (double)n * n * n / best / 1e9;
}

static id<MTLComputePipelineState> pipeline(Ctx &c, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [c.lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { fprintf(stderr, "no kernel %s\n", name); exit(1); }
    id<MTLComputePipelineState> ps = [c.dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) { fprintf(stderr, "pipeline %s: %s\n", name, err.localizedDescription.UTF8String); exit(1); }
    return ps;
}

// Runs `body` on a fresh command buffer and returns GPU-side seconds.
template <typename F>
static double timed(Ctx &c, F body) {
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    body(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { fprintf(stderr, "gpu error: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
    return cb.GPUEndTime - cb.GPUStartTime;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        Ctx c;
        c.dev = MTLCreateSystemDefaultDevice();
        if (!c.dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        c.queue = [c.dev newCommandQueue];

        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeFast;
        c.lib = [c.dev newLibraryWithSource:[NSString stringWithUTF8String:kSource]
                                    options:opts
                                      error:&err];
        if (!c.lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }

        id<MTLComputePipelineState> psCopy = pipeline(c, "stream_copy");
        id<MTLComputePipelineState> psRead = pipeline(c, "stream_read");
        id<MTLComputePipelineState> psFill = pipeline(c, "fill_buf");

        // 512 MiB per buffer: far past any cache, small enough to leave the
        // machine usable.
        const size_t bytes = 512ull << 20;
        const size_t n4 = bytes / sizeof(float) / 4;   // float4 elements

        std::string out = "{\n";
        char buf[512];

        snprintf(buf, sizeof buf,
                 "  \"device\": \"%s\",\n  \"max_threadgroup\": %llu,\n"
                 "  \"buffer_bytes\": %llu,\n  \"reps\": %d,\n",
                 c.dev.name.UTF8String,
                 (unsigned long long)psCopy.maxTotalThreadsPerThreadgroup,
                 (unsigned long long)bytes, kReps);
        out += buf;

        // ---- bandwidth, both storage modes ------------------------------
        out += "  \"bandwidth_gbs\": {\n";
        const MTLResourceOptions modes[2] = {MTLResourceStorageModePrivate,
                                             MTLResourceStorageModeShared};
        const char *modeName[2] = {"private", "shared"};
        for (int m = 0; m < 2; ++m) {
            id<MTLBuffer> a = [c.dev newBufferWithLength:bytes options:modes[m]];
            id<MTLBuffer> b = [c.dev newBufferWithLength:bytes options:modes[m]];
            float scale = 1.0f;
            uint32_t n4u = (uint32_t)n4;

            timed(c, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:psFill];
                [enc setBuffer:a offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(n4, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            });

            double bestCopy = 1e18, bestRead = 1e18;
            for (int r = 0; r < kReps; ++r) {
                double t = timed(c, [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setComputePipelineState:psCopy];
                    [enc setBuffer:a offset:0 atIndex:0];
                    [enc setBuffer:b offset:0 atIndex:1];
                    [enc setBytes:&scale length:sizeof scale atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(n4, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });
                if (t < bestCopy) bestCopy = t;
            }
            // A grid large enough to fill every core with resident threads,
            // each striding the buffer rather than owning one element.
            const size_t readGrid = 1u << 22;
            for (int r = 0; r < kReps; ++r) {
                double t = timed(c, [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setComputePipelineState:psRead];
                    [enc setBuffer:a offset:0 atIndex:0];
                    [enc setBuffer:b offset:0 atIndex:1];
                    [enc setBytes:&n4u length:sizeof n4u atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(readGrid, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });
                if (t < bestRead) bestRead = t;
            }
            snprintf(buf, sizeof buf,
                     "    \"copy_%s\": %.2f,\n    \"read_%s\": %.2f%s\n",
                     modeName[m], (double)(2 * bytes) / bestCopy / 1e9,
                     modeName[m], (double)bytes / bestRead / 1e9,
                     m == 0 ? "," : "");
            out += buf;
        }
        out += "  },\n";

        // ---- FMA throughput ---------------------------------------------
        out += "  \"gflops\": {\n";
        const char *fmaKernels[3] = {"fma_f32", "fma_f16", "fma_f16x4"};
        const int lanes[3] = {1, 1, 4};
        const size_t threads = 1u << 20;
        const uint32_t iters = 512;
        id<MTLBuffer> sink = [c.dev newBufferWithLength:threads * sizeof(float)
                                                options:MTLResourceStorageModePrivate];
        for (int k = 0; k < 3; ++k) {
            id<MTLComputePipelineState> ps = pipeline(c, fmaKernels[k]);
            double best = 1e18;
            for (int r = 0; r < kReps; ++r) {
                double t = timed(c, [&](id<MTLComputeCommandEncoder> enc) {
                    [enc setComputePipelineState:ps];
                    [enc setBuffer:sink offset:0 atIndex:0];
                    [enc setBytes:&iters length:sizeof iters atIndex:1];
                    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                });
                if (t < best) best = t;
            }
            // 32 FMAs per iteration, 2 flops per FMA, `lanes` values per FMA.
            double flops = (double)threads * iters * 32.0 * 2.0 * lanes[k];
            snprintf(buf, sizeof buf, "    \"%s\": %.1f%s\n",
                     fmaKernels[k], flops / best / 1e9, k == 2 ? "" : ",");
            out += buf;
        }
        out += "  },\n";

        // ---- CPU ceilings, same memory system, different reach ----------
        // A CPU rep costs milliseconds, and best-of-5 was still moving by 39%
        // between invocations, so these run many more times than the GPU probes
        // do. Best-of-N converges on the ceiling; the mean would measure the
        // machine's background load instead.
        const int perf_cores = 6;   // P-cores; llama.cpp defaults its threads to these
        const int all_cores = (int)std::thread::hardware_concurrency();
        double triad_p = cpu_triad_gbs(256ull << 20, perf_cores, 50);
        double triad_all = cpu_triad_gbs(256ull << 20, all_cores, 50);
        double sgemm = cpu_sgemm_gflops(2048, 20);
        snprintf(buf, sizeof buf,
                 "  \"cpu\": {\n    \"triad_gbs_p_cores\": %.1f,\n"
                 "    \"triad_gbs_all_cores\": %.1f,\n"
                 "    \"accelerate_sgemm_gflops\": %.1f,\n"
                 "    \"p_cores\": %d,\n    \"all_cores\": %d\n  }\n}\n",
                 triad_p, triad_all, sgemm, perf_cores, all_cores);
        out += buf;

        fputs(out.c_str(), stdout);
        return 0;
    }
}
