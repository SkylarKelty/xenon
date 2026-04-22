// Final-logit softcap: out[i] = tanh(in[i] / cap) * cap, bf16 I/O.
// Gemma uses cap = 30.0 (final_logit_softcapping in the config).
//
// Fused into one kernel since each element is independent and bf16 writes
// should be done once.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int BLOCK_THREADS = 256;

__global__ void xk_softcap_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int n,
    float inv_cap,
    float cap)
{
    const int i = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(in[i]);
    out[i] = __float2bfloat16_rn(tanhf(v * inv_cap) * cap);
}

extern "C" int xk_softcap_bf16(
    void* out,
    const void* in,
    int n,
    float cap,
    void* stream)
{
    if (n <= 0) return 0;
    if (cap <= 0.0f) return -(int)cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t)stream;
    int blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;
    xk_softcap_bf16_kernel<<<blocks, BLOCK_THREADS, 0, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)in,
        n, 1.0f / cap, cap);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
