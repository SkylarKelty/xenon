// Element-wise (a + b) * scale, bf16 I/O with fp32 arithmetic.
// Used for the PLE combine `(ctx + raw) * 1/sqrt(2)` and for residual adds
// with a multiplier elsewhere in the forward pass.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int BLOCK_THREADS = 256;

__global__ void xk_add_scale_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int n,
    float scale)
{
    const int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (tid >= n) return;
    float va = __bfloat162float(a[tid]);
    float vb = __bfloat162float(b[tid]);
    out[tid] = __float2bfloat16_rn((va + vb) * scale);
}

extern "C" int xk_add_scale_bf16(
    void* out,
    const void* a,
    const void* b,
    int n,
    float scale,
    void* stream)
{
    if (n <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
    xk_add_scale_bf16_kernel<<<blocks, BLOCK_THREADS, 0, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b,
        n, scale);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
