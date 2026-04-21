// gpubench - CUDA performance benchmarking tool
//
// Measures:
//   - Device info
//   - Host<->device and device<->device memory bandwidth
//   - FP32 / FP16 FMA compute throughput
//   - SGEMM (cuBLAS) throughput
//   - NVFP4 GEMM (cuBLASLt, Blackwell-class GPUs only)
//   - Kernel launch latency

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#if defined(GPUBENCH_ENABLE_FP4) && CUDART_VERSION >= 12080
#include <cublasLt.h>
#define GPUBENCH_FP4_BUILT 1
#else
#define GPUBENCH_FP4_BUILT 0
#endif

#define CUDA_CHECK(expr) do {                                              \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                               \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                     #expr, __FILE__, __LINE__, cudaGetErrorString(_e));   \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

#define CUBLAS_CHECK(expr) do {                                            \
    cublasStatus_t _s = (expr);                                            \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
        std::fprintf(stderr, "cuBLAS error %s at %s:%d (status %d)\n",     \
                     #expr, __FILE__, __LINE__, (int)_s);                  \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

// Check for a kernel launch error (configuration) AND any async error that
// surfaced since last sync. Without this, a failed launch is silent and
// timings look impossibly fast.
#define KERNEL_CHECK() do {                                                \
    cudaError_t _e1 = cudaGetLastError();                                  \
    if (_e1 != cudaSuccess) {                                              \
        std::fprintf(stderr, "kernel launch error at %s:%d: %s\n",         \
                     __FILE__, __LINE__, cudaGetErrorString(_e1));         \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

struct Args {
    int device = 0;
    size_t bw_bytes = 512ull * 1024 * 1024;   // 512 MiB per copy
    int bw_iters = 10;
    int compute_iters = 50;
    int sgemm_n = 4096;
    int sgemm_iters = 20;
    int fp4_n = 4096;
    int fp4_iters = 20;
    int launch_iters = 10000;
    bool run_info = true;
    bool run_bw = true;
    bool run_compute = true;
    bool run_sgemm = true;
    bool run_fp4 = true;
    bool run_launch = true;
};

static void print_usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --device N          CUDA device to use (default 0)\n"
        "  --bw-mib N          Bytes per bandwidth copy, in MiB (default 512)\n"
        "  --bw-iters N        Bandwidth iterations (default 10)\n"
        "  --compute-iters N   Compute iterations (default 50)\n"
        "  --sgemm-n N         Square SGEMM dimension (default 4096)\n"
        "  --sgemm-iters N     SGEMM iterations (default 20)\n"
        "  --fp4-n N           Square NVFP4 GEMM dimension (default 4096, multiple of 128)\n"
        "  --fp4-iters N       NVFP4 GEMM iterations (default 20)\n"
        "  --launch-iters N    Kernel launch iterations (default 10000)\n"
        "  --only SECTION      One of: info,bw,compute,sgemm,fp4,launch (can repeat)\n"
        "  --skip SECTION      Skip a section (can repeat)\n"
        "  -h, --help          Show this help\n",
        prog);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    bool only_mode = false;
    // first pass: see if --only is used, in which case default-off everything
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--only") == 0) { only_mode = true; break; }
    }
    if (only_mode) {
        a.run_info = a.run_bw = a.run_compute = a.run_sgemm = a.run_fp4 = a.run_launch = false;
    }

    auto set_section = [&](const char* s, bool v) {
        if      (!std::strcmp(s, "info"))    a.run_info = v;
        else if (!std::strcmp(s, "bw"))      a.run_bw = v;
        else if (!std::strcmp(s, "compute")) a.run_compute = v;
        else if (!std::strcmp(s, "sgemm"))   a.run_sgemm = v;
        else if (!std::strcmp(s, "fp4"))     a.run_fp4 = v;
        else if (!std::strcmp(s, "launch"))  a.run_launch = v;
        else { std::fprintf(stderr, "unknown section: %s\n", s); std::exit(2); }
    };

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(arg, "-h") || !std::strcmp(arg, "--help")) {
            print_usage(argv[0]); std::exit(0);
        } else if (!std::strcmp(arg, "--device"))         a.device = std::atoi(need("--device"));
        else if (!std::strcmp(arg, "--bw-mib"))           a.bw_bytes = (size_t)std::atoll(need("--bw-mib")) * 1024ull * 1024ull;
        else if (!std::strcmp(arg, "--bw-iters"))         a.bw_iters = std::atoi(need("--bw-iters"));
        else if (!std::strcmp(arg, "--compute-iters"))    a.compute_iters = std::atoi(need("--compute-iters"));
        else if (!std::strcmp(arg, "--sgemm-n"))          a.sgemm_n = std::atoi(need("--sgemm-n"));
        else if (!std::strcmp(arg, "--sgemm-iters"))      a.sgemm_iters = std::atoi(need("--sgemm-iters"));
        else if (!std::strcmp(arg, "--fp4-n"))            a.fp4_n = std::atoi(need("--fp4-n"));
        else if (!std::strcmp(arg, "--fp4-iters"))        a.fp4_iters = std::atoi(need("--fp4-iters"));
        else if (!std::strcmp(arg, "--launch-iters"))     a.launch_iters = std::atoi(need("--launch-iters"));
        else if (!std::strcmp(arg, "--only"))             set_section(need("--only"), true);
        else if (!std::strcmp(arg, "--skip"))             set_section(need("--skip"), false);
        else {
            std::fprintf(stderr, "unknown arg: %s\n", arg);
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    return a;
}

// ---------------- timing helpers ----------------

class EventTimer {
public:
    EventTimer() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~EventTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start_, s)); }
    float stop_ms(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, s));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// ---------------- kernels ----------------

// FP32 FMA-heavy kernel. Each thread does ITERS FMAs on registers.
template <int ITERS>
__global__ void fma_fp32_kernel(float* out, float a, float b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)tid * 1.0e-3f + 1.0f;
    float y = (float)tid * 2.0e-3f + 1.0f;
    #pragma unroll 32
    for (int i = 0; i < ITERS; ++i) {
        x = x * a + b;
        y = y * a + b;
    }
    if (tid == n) out[0] = x + y;  // sink; branch almost never taken
}

// FP16x2 FMA-heavy kernel using __half2 intrinsics.
template <int ITERS>
__global__ void fma_fp16_kernel(__half* out, __half2 a, __half2 b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __half2 x = __floats2half2_rn((float)tid * 1.0e-3f + 1.0f, (float)tid * 1.5e-3f + 1.0f);
    __half2 y = __floats2half2_rn((float)tid * 2.0e-3f + 1.0f, (float)tid * 2.5e-3f + 1.0f);
    #pragma unroll 32
    for (int i = 0; i < ITERS; ++i) {
        x = __hfma2(x, a, b);
        y = __hfma2(y, a, b);
    }
    if (tid == n) {
        __half2 s = __hadd2(x, y);
        out[0] = __low2half(s);
    }
}

__global__ void empty_kernel() {}

// ---------------- sections ----------------

static void section_info(const Args& a) {
    std::printf("== Device Info ==\n");
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    std::printf("CUDA devices visible: %d\n", ndev);

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, a.device));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);

    std::printf("Using device %d: %s\n", a.device, p.name);
    std::printf("  Compute capability : %d.%d\n", p.major, p.minor);
    std::printf("  SMs                : %d\n", p.multiProcessorCount);
    std::printf("  Max threads/SM     : %d\n", p.maxThreadsPerMultiProcessor);
    std::printf("  Warp size          : %d\n", p.warpSize);
    std::printf("  Global memory      : %.2f GiB\n", (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Memory/core clocks and bus width moved from cudaDeviceProp to attributes in CUDA 13.
    int memClockKHz = 0, coreClockKHz = 0, busWidth = 0;
    cudaDeviceGetAttribute(&memClockKHz,  cudaDevAttrMemoryClockRate,  a.device);
    cudaDeviceGetAttribute(&coreClockKHz, cudaDevAttrClockRate,        a.device);
    cudaDeviceGetAttribute(&busWidth,     cudaDevAttrGlobalMemoryBusWidth, a.device);
    std::printf("  Memory clock       : %.0f MHz\n", memClockKHz / 1.0e3);
    std::printf("  Memory bus width   : %d bits\n", busWidth);
    double peak_bw = 2.0 * (double)memClockKHz * 1.0e3 * (busWidth / 8.0) / 1.0e9;
    std::printf("  Theoretical BW     : %.1f GB/s\n", peak_bw);
    std::printf("  L2 cache           : %d KiB\n", p.l2CacheSize / 1024);
    std::printf("  Core clock         : %.0f MHz\n", coreClockKHz / 1.0e3);
    std::printf("  ECC enabled        : %s\n", p.ECCEnabled ? "yes" : "no");
    std::printf("  CUDA runtime/driver: %d.%d / %d.%d\n",
                rt / 1000, (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10);
    std::printf("\n");
}

static void section_bandwidth(const Args& a) {
    std::printf("== Memory Bandwidth ==\n");
    size_t bytes = a.bw_bytes;
    std::printf("Transfer size: %.1f MiB, iters: %d\n\n",
                (double)bytes / (1024.0 * 1024.0), a.bw_iters);

    // Allocate buffers
    void* h_pageable = std::malloc(bytes);
    if (!h_pageable) { std::fprintf(stderr, "malloc failed\n"); std::exit(1); }
    std::memset(h_pageable, 0xA5, bytes);

    void* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    std::memset(h_pinned, 0x5A, bytes);

    void *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    EventTimer t;
    auto measure = [&](const char* name, cudaMemcpyKind kind, void* dst, const void* src) {
        // warmup
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
        std::vector<double> gbps;
        gbps.reserve(a.bw_iters);
        for (int i = 0; i < a.bw_iters; ++i) {
            t.start();
            CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
            float ms = t.stop_ms();
            double gb = (double)bytes / 1.0e9;
            gbps.push_back(gb / (ms / 1.0e3));
        }
        double best = *std::max_element(gbps.begin(), gbps.end());
        double med  = median(gbps);
        std::printf("  %-22s  median %7.2f GB/s   best %7.2f GB/s\n", name, med, best);
    };

    measure("H2D pageable",   cudaMemcpyHostToDevice,   d_a, h_pageable);
    measure("H2D pinned",     cudaMemcpyHostToDevice,   d_a, h_pinned);
    measure("D2H pageable",   cudaMemcpyDeviceToHost,   h_pageable, d_a);
    measure("D2H pinned",     cudaMemcpyDeviceToHost,   h_pinned, d_a);
    measure("D2D",            cudaMemcpyDeviceToDevice, d_b, d_a);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    std::free(h_pageable);
    std::printf("\n");
}

static void section_compute(const Args& a) {
    std::printf("== Compute Throughput (FMA microbenchmark) ==\n");

    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, a.device));

    // Size the grid to saturate the GPU.
    const int block = 256;
    int threads_per_sm = p.maxThreadsPerMultiProcessor;
    int total_threads = p.multiProcessorCount * threads_per_sm;
    int grid = (total_threads + block - 1) / block;

    constexpr int ITERS = 2048;  // inner-loop FMAs per "lane"
    // Two independent FMA chains per thread → 2 FMAs per ITER.
    // fp32 FMA = 2 flops; fp16x2 FMA = 4 flops (2 lanes * 2 flops).
    double fp32_flops_per_thread = (double)ITERS * 2.0 * 2.0;
    double fp16_flops_per_thread = (double)ITERS * 2.0 * 4.0;
    double fp32_total_flops = fp32_flops_per_thread * (double)grid * (double)block;
    double fp16_total_flops = fp16_flops_per_thread * (double)grid * (double)block;

    std::printf("Grid: %d blocks x %d threads = %lld threads, inner iters=%d, runs=%d\n",
                grid, block, (long long)grid * block, ITERS, a.compute_iters);

    float *d_out_f = nullptr;
    __half *d_out_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out_f, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_h, sizeof(__half)));

    EventTimer t;

    // FP32
    {
        // warmup
        fma_fp32_kernel<ITERS><<<grid, block>>>(d_out_f, 1.0001f, 0.9999f, -1);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> tflops;
        tflops.reserve(a.compute_iters);
        for (int i = 0; i < a.compute_iters; ++i) {
            t.start();
            fma_fp32_kernel<ITERS><<<grid, block>>>(d_out_f, 1.0001f, 0.9999f, -1);
            float ms = t.stop_ms();
            tflops.push_back(fp32_total_flops / (ms / 1.0e3) / 1.0e12);
        }
        std::printf("  FP32 FMA    median %7.2f TFLOP/s   best %7.2f TFLOP/s\n",
                    median(tflops), *std::max_element(tflops.begin(), tflops.end()));
    }

    // FP16
    {
        __half2 a2 = __floats2half2_rn(1.0001f, 1.0001f);
        __half2 b2 = __floats2half2_rn(0.9999f, 0.9999f);
        fma_fp16_kernel<ITERS><<<grid, block>>>(d_out_h, a2, b2, -1);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> tflops;
        tflops.reserve(a.compute_iters);
        for (int i = 0; i < a.compute_iters; ++i) {
            t.start();
            fma_fp16_kernel<ITERS><<<grid, block>>>(d_out_h, a2, b2, -1);
            float ms = t.stop_ms();
            tflops.push_back(fp16_total_flops / (ms / 1.0e3) / 1.0e12);
        }
        std::printf("  FP16x2 FMA  median %7.2f TFLOP/s   best %7.2f TFLOP/s\n",
                    median(tflops), *std::max_element(tflops.begin(), tflops.end()));
    }

    CUDA_CHECK(cudaFree(d_out_f));
    CUDA_CHECK(cudaFree(d_out_h));
    std::printf("\n");
}

static void section_sgemm(const Args& a) {
    std::printf("== SGEMM (cuBLAS) ==\n");
    int N = a.sgemm_n;
    size_t elems = (size_t)N * N;
    size_t bytes = elems * sizeof(float);
    std::printf("Matrix: %d x %d (fp32), iters: %d\n", N, N, a.sgemm_iters);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Fill with something nonzero. A simple kernel would do, but a memset to
    // a bit pattern yields valid fp32 values; enough to exercise the math.
    CUDA_CHECK(cudaMemset(dA, 0x3f, bytes));  // ~0.7something
    CUDA_CHECK(cudaMemset(dB, 0x3f, bytes));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Allow TF32 on Ampere+; this is what most real workloads get.
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    const float alpha = 1.0f, beta = 0.0f;
    // warmup
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, dA, N, dB, N, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    EventTimer t;
    double flops_per_call = 2.0 * (double)N * (double)N * (double)N;
    std::vector<double> tflops;
    tflops.reserve(a.sgemm_iters);
    for (int i = 0; i < a.sgemm_iters; ++i) {
        t.start();
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                 &alpha, dA, N, dB, N, &beta, dC, N));
        float ms = t.stop_ms();
        tflops.push_back(flops_per_call / (ms / 1.0e3) / 1.0e12);
    }
    std::printf("  SGEMM (TF32 allowed)  median %7.2f TFLOP/s   best %7.2f TFLOP/s\n",
                median(tflops), *std::max_element(tflops.begin(), tflops.end()));

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    std::printf("\n");
}

// NVFP4 block-scaled GEMM via cuBLASLt. Blackwell-class GPUs (CC >= 10.0) only.
//
// Layout rationale: FP4 tensor cores on Blackwell require TN operands (A^T * B).
// We set opA = T, opB = N and describe operands as col-major K-by-{M,N} tiles.
// NVFP4 uses one UE4M3 (8-bit) scale per 16-element block along K, laid out in
// the cuBLASLt-expected tile pattern. For throughput we don't care about scale
// values, just that the buffer is large enough and readable; we fill with 0x38
// (UE4M3 encoding of 1.0) and size conservatively to the canonical
// round_up(rows,128) x round_up(K/16, 4) allocation.
static void section_fp4(const Args& a) {
    std::printf("== NVFP4 GEMM (cuBLASLt) ==\n");

#if !GPUBENCH_FP4_BUILT
    (void)a;
    std::printf("  Skipped: built against CUDA < 12.8 (no FP4 API).\n"
                "  Rebuild with CUDA 12.8+ toolkit to enable.\n\n");
    return;
#else
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, a.device));
    if (p.major < 10) {
        std::printf("  Skipped: device compute capability %d.%d lacks FP4 tensor cores (need >= 10.0).\n\n",
                    p.major, p.minor);
        return;
    }

    int N = a.fp4_n;
    if (N % 128 != 0 || N <= 0) {
        std::printf("  --fp4-n must be a positive multiple of 128 (got %d). Skipping.\n\n", N);
        return;
    }
    int M = N, K = N;
    std::printf("Matrix: %dx%dx%d (fp4 e2m1, bf16 out, fp32 accum), iters: %d\n",
                M, N, K, a.fp4_iters);

    // Operand buffers. fp4 is 4 bits -> M*K/2 bytes.
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytesA = (size_t)M * K / 2;
    size_t bytesB = (size_t)K * N / 2;
    size_t bytesC = (size_t)M * N * sizeof(uint16_t);  // bf16
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    // Fill operands with a non-degenerate bit pattern. Throughput is independent
    // of operand values since the tensor cores don't short-circuit zeros.
    CUDA_CHECK(cudaMemset(dA, 0x42, bytesA));
    CUDA_CHECK(cudaMemset(dB, 0x24, bytesB));
    CUDA_CHECK(cudaMemset(dC, 0,    bytesC));

    // Scale tensors: one UE4M3 byte per 16-element block along K.
    // Shape for A-scale: [M, K/16]; for B-scale: [N, K/16]. Pad generously to
    // the canonical 128x4 tile alignment so cuBLASLt reads stay in-bounds.
    auto round_up = [](int x, int m) { return ((x + m - 1) / m) * m; };
    size_t aScaleRows = (size_t)round_up(M, 128);
    size_t aScaleCols = (size_t)round_up(K / 16, 4);
    size_t bScaleRows = (size_t)round_up(N, 128);
    size_t bScaleCols = (size_t)round_up(K / 16, 4);
    size_t bytesAScale = aScaleRows * aScaleCols;  // 1 byte per scale
    size_t bytesBScale = bScaleRows * bScaleCols;
    void *dAScale = nullptr, *dBScale = nullptr;
    CUDA_CHECK(cudaMalloc(&dAScale, bytesAScale));
    CUDA_CHECK(cudaMalloc(&dBScale, bytesBScale));
    CUDA_CHECK(cudaMemset(dAScale, 0x38, bytesAScale));  // UE4M3 1.0
    CUDA_CHECK(cudaMemset(dBScale, 0x38, bytesBScale));

    // cuBLASLt setup.
    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAScale, sizeof(dAScale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBScale, sizeof(dBScale)));

    int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Layouts: operand tensors stored col-major, leading dim = K.
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF,    M, N, M));

    // Workspace + heuristic.
    size_t workspaceBytes = 128ull * 1024 * 1024;
    void* dWorkspace = nullptr;
    CUDA_CHECK(cudaMalloc(&dWorkspace, workspaceBytes));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));

    cublasLtMatmulHeuristicResult_t heur{};
    int nResults = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        lt, desc, layoutA, layoutB, layoutC, layoutC, pref, 1, &heur, &nResults);
    if (hs != CUBLAS_STATUS_SUCCESS || nResults == 0) {
        std::printf("  No cuBLASLt algorithm found for NVFP4 GEMM (status=%d, results=%d).\n"
                    "  This usually means the driver/toolkit combo doesn't support FP4 on this GPU yet.\n\n",
                    (int)hs, nResults);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(desc);
        cublasLtDestroy(lt);
        cudaFree(dWorkspace);
        cudaFree(dAScale); cudaFree(dBScale);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return;
    }

    const float alpha = 1.0f, beta = 0.0f;
    auto run_once = [&]() {
        return cublasLtMatmul(lt, desc, &alpha, dA, layoutA, dB, layoutB,
                              &beta, dC, layoutC, dC, layoutC,
                              &heur.algo, dWorkspace, workspaceBytes, 0);
    };

    // Warmup.
    CUBLAS_CHECK(run_once());
    CUDA_CHECK(cudaDeviceSynchronize());

    EventTimer t;
    double flopsPerCall = 2.0 * (double)M * (double)N * (double)K;
    std::vector<double> tflops;
    tflops.reserve(a.fp4_iters);
    for (int i = 0; i < a.fp4_iters; ++i) {
        t.start();
        CUBLAS_CHECK(run_once());
        float ms = t.stop_ms();
        tflops.push_back(flopsPerCall / (ms / 1.0e3) / 1.0e12);
    }
    std::printf("  NVFP4 GEMM   median %7.2f TFLOP/s   best %7.2f TFLOP/s\n",
                median(tflops), *std::max_element(tflops.begin(), tflops.end()));

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(lt);
    CUDA_CHECK(cudaFree(dWorkspace));
    CUDA_CHECK(cudaFree(dAScale));
    CUDA_CHECK(cudaFree(dBScale));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    std::printf("\n");
#endif
}

static void section_launch(const Args& a) {
    std::printf("== Kernel Launch Latency ==\n");
    // warmup
    for (int i = 0; i < 100; ++i) empty_kernel<<<1, 1>>>();
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    EventTimer t;
    t.start();
    for (int i = 0; i < a.launch_iters; ++i) empty_kernel<<<1, 1>>>();
    float ms = t.stop_ms();
    double per_us = (ms * 1.0e3) / (double)a.launch_iters;
    std::printf("  empty kernel  iters=%d  total=%.2f ms  per-launch=%.2f us\n",
                a.launch_iters, ms, per_us);
    std::printf("\n");
}

// ---------------- main ----------------

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { std::fprintf(stderr, "no CUDA devices\n"); return 1; }
    if (a.device < 0 || a.device >= ndev) {
        std::fprintf(stderr, "device %d out of range (have %d)\n", a.device, ndev);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(a.device));
    CUDA_CHECK(cudaFree(0));  // force context init

    std::printf("gpubench - CUDA performance benchmark\n\n");

    if (a.run_info)    section_info(a);
    if (a.run_bw)      section_bandwidth(a);
    if (a.run_compute) section_compute(a);
    if (a.run_sgemm)   section_sgemm(a);
    if (a.run_fp4)     section_fp4(a);
    if (a.run_launch)  section_launch(a);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
