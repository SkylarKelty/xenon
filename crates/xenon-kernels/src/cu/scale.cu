// Element-wise scalar multiply: out[i] = in[i] * scale, bf16 I/O, fp32 math.
// In-place safe (out == in).
//
// Used for the Gemma 4 `layer_scalar` broadcast at the tail of each decoder
// layer, and for any other scalar-broadcast multiplication that comes up.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int BLOCK_THREADS = 256;

__global__ void xk_scale_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int n,
    float scale)
{
    const int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (tid >= n) return;
    out[tid] = __float2bfloat16_rn(__bfloat162float(in[tid]) * scale);
}

extern "C" int xk_scale_bf16(
    void* out,
    const void* in,
    int n,
    float scale,
    void* stream)
{
    if (n <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
    xk_scale_bf16_kernel<<<blocks, BLOCK_THREADS, 0, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)in,
        n, scale);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
