// Fused NVFP4×bf16 gemv (M=1): y[N] = x[K] @ W^T
//
// Reads packed FP4 weights + UE4M3 per-16 block scales + bf16 activations
// directly and never materializes bf16 weights in DRAM. This is the decode
// fast path — for M>1 cuBLASLt gemvx/GEMM on dequanted weights is still
// preferred because dequant amortizes across M.
//
// Layout (matches fp4_dequant.cu):
//   W_packed : u8 [N, K/2]    (two FP4 E2M1 per byte; low nibble = even k)
//   W_scales : u8 [N, K/16]   (one UE4M3 scale per 16 k-elements)
//   global   : f32 scalar     (per-tensor weight scale from modelopt)
//   x        : bf16 [K]
//   y        : bf16 [N]
//
// K must be a positive multiple of 16.
//
// Launch: gridDim = (N,), blockDim = (128,). One block per output row; the
// block walks K in 16-element chunks (= one scale block's worth) and runs
// a warp-shuffle + shmem reduction to produce y[n].

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __constant__ float XK_FP4_GEMV_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float xk_gemv_ue4m3_to_f32(uint8_t b) {
    uint32_t x   = b & 0x7Fu;
    uint32_t exp = (x >> 3) & 0xFu;
    uint32_t man = x & 0x7u;
    if (exp == 0) {
        return (float)man * (1.0f / 512.0f);
    }
    return ldexpf((float)(8u + man), (int)exp - 10);
}

// Block-wide sum reduction across 128 threads (= 4 warps). Result lives in
// thread 0 of the block; other threads' return values are undefined.
__device__ __forceinline__ float xk_block_reduce_sum(float v) {
    // Warp reduce.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    __shared__ float warp_sums[4];
    const int lane   = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float w = (lane < 4) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            w += __shfl_down_sync(0xffffffffu, w, off);
        }
        return w;
    }
    return 0.0f;
}

__global__ void xk_fp4_gemv_bf16_kernel(
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ w_packed,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    int N,
    int K)
{
    const int n  = blockIdx.x;
    if (n >= N) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int num_blocks = K >> 4;              // K / 16
    const size_t row_packed = (size_t)n * (K >> 1);   // K/2 bytes per row
    const size_t row_scales = (size_t)n * (K >> 4);   // K/16 bytes per row

    // Stage the 16-entry FP4 LUT into shared memory at block entry. In the
    // inner loop each thread indexes with its own 4-bit `code`, so reading
    // from __constant__ would serialize through the MIO pipe (broadcast only
    // fires when all warp lanes share the same address). Shmem has 32 banks,
    // so 16 distinct codes across a warp land in 16 distinct banks and are
    // served in parallel. Confirmed via ncu: pre-shmem kernel spent ~68% of
    // stall cycles on MIO throttle.
    __shared__ float s_lut[16];
    if (tid < 16) s_lut[tid] = XK_FP4_GEMV_LUT[tid];
    __syncthreads();

    float acc = 0.0f;

    // Each thread strides through K/16-sized blocks. Within a warp, threads
    // 0..31 read consecutive 8-byte chunks -> 256-byte coalesced load.
    for (int b = tid; b < num_blocks; b += nthreads) {
        // 16 FP4 weights = 8 packed bytes = one uint64.
        uint64_t w_pack = *reinterpret_cast<const uint64_t*>(
            w_packed + row_packed + (size_t)b * 8);

        const uint8_t sb = w_scales[row_scales + b];
        const float   bs = xk_gemv_ue4m3_to_f32(sb) * global_scale;

        // 16 bf16 activations = 32 bytes = two uint4 loads.
        const __nv_bfloat16* xb = x + (size_t)b * 16;
        uint4 xa0 = *reinterpret_cast<const uint4*>(xb);
        uint4 xa1 = *reinterpret_cast<const uint4*>(xb + 8);
        const __nv_bfloat16* xa0p = reinterpret_cast<const __nv_bfloat16*>(&xa0);
        const __nv_bfloat16* xa1p = reinterpret_cast<const __nv_bfloat16*>(&xa1);

        // LOOP2: manual 4-way unroll to give ILP and schedule ahead.
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            uint8_t b0 = (uint8_t)(w_pack >> ((i >> 1) * 8));
            uint8_t b1 = (uint8_t)(w_pack >> (((i + 1) >> 1) * 8));
            uint8_t b2 = (uint8_t)(w_pack >> (((i + 2) >> 1) * 8));
            uint8_t b3 = (uint8_t)(w_pack >> (((i + 3) >> 1) * 8));
            uint8_t c0 = (i & 1)     ? (b0 >> 4) : (b0 & 0xF);
            uint8_t c1 = ((i+1) & 1) ? (b1 >> 4) : (b1 & 0xF);
            uint8_t c2 = ((i+2) & 1) ? (b2 >> 4) : (b2 & 0xF);
            uint8_t c3 = ((i+3) & 1) ? (b3 >> 4) : (b3 & 0xF);
            float w0 = s_lut[c0] * bs;
            float w1 = s_lut[c1] * bs;
            float w2 = s_lut[c2] * bs;
            float w3 = s_lut[c3] * bs;
            float x0 = __bfloat162float((i     < 8) ? xa0p[i]     : xa1p[i - 8]);
            float x1 = __bfloat162float((i + 1 < 8) ? xa0p[i + 1] : xa1p[i - 7]);
            float x2 = __bfloat162float((i + 2 < 8) ? xa0p[i + 2] : xa1p[i - 6]);
            float x3 = __bfloat162float((i + 3 < 8) ? xa0p[i + 3] : xa1p[i - 5]);
            acc0 = fmaf(w0, x0, acc0);
            acc1 = fmaf(w1, x1, acc1);
            acc2 = fmaf(w2, x2, acc2);
            acc3 = fmaf(w3, x3, acc3);
        }
        acc = acc0 + acc1 + acc2 + acc3;
    }

    float sum = xk_block_reduce_sum(acc);
    if (tid == 0) {
        y[n] = __float2bfloat16_rn(sum);
    }
}

// Returns 0 on success, negative cudaError_t on failure, -1 if K is invalid.
extern "C" int xk_fp4_gemv_bf16(
    void* y,
    const void* x,
    const void* w_packed,
    const void* w_scales,
    float global_scale,
    int N,
    int K,
    void* stream)
{
    if (N <= 0 || K <= 0) return 0;
    if ((K & 15) != 0) return -1;

    dim3 block(128);
    dim3 grid((unsigned)N);
    cudaStream_t s = (cudaStream_t)stream;
    xk_fp4_gemv_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)y,
        (const __nv_bfloat16*)x,
        (const uint8_t*)w_packed,
        (const uint8_t*)w_scales,
        global_scale,
        N, K);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : -(int)e;
}
