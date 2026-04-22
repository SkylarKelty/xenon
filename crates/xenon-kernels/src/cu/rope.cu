// Rotary Position Embedding (RoPE), in-place on [T, H, D] bf16 tensor.
//
// HF "rotate_half" convention: pair (x[i], x[i + rotary_dim/2]) rotates by
// angle (pos * theta^(-2i/rotary_dim)). Dims [rotary_dim, head_dim) pass
// through untouched — this covers Gemma 4's partial_rotary_factor=0.25 on
// full-attention layers (rotary_dim=128, head_dim=512).
//
// Grid: (tokens, heads). Block: 128 threads, stride loop over pairs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int BLOCK_THREADS = 128;

__global__ void xk_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const int* __restrict__ positions,
    int heads,
    int head_dim,
    int rotary_dim,
    float theta)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int half = rotary_dim / 2;
    const int pos = positions[t];

    __nv_bfloat16* xh = x + ((size_t)t * heads + h) * head_dim;

    for (int i = tid; i < half; i += BLOCK_THREADS) {
        float inv_freq = __powf(theta, -2.0f * (float)i / (float)rotary_dim);
        float angle = (float)pos * inv_freq;
        float c, s;
        __sincosf(angle, &s, &c);
        float x0 = __bfloat162float(xh[i]);
        float x1 = __bfloat162float(xh[i + half]);
        xh[i]        = __float2bfloat16_rn(x0 * c - x1 * s);
        xh[i + half] = __float2bfloat16_rn(x1 * c + x0 * s);
    }
}

extern "C" int xk_rope_bf16(
    void* x,
    const void* positions,
    int tokens,
    int heads,
    int head_dim,
    int rotary_dim,
    float theta,
    void* stream)
{
    if (tokens <= 0 || heads <= 0 || head_dim <= 0) return 0;
    if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1) != 0) return -(int)cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(tokens, heads);
    dim3 block(BLOCK_THREADS);
    xk_rope_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)x,
        (const int*)positions,
        heads,
        head_dim,
        rotary_dim,
        theta);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
