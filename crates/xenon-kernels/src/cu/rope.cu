// Rotary Position Embedding (RoPE), in-place on [T, H, D] bf16 tensor.
//
// HF "rotate_half" convention: pair (x[i], x[i + head_dim/2]) rotates by
// angle (pos * theta^(-2i/head_dim)) for i in 0..rotary_pairs. The
// unrotated pair angles (i >= rotary_pairs) in [rotary_pairs, head_dim/2)
// and their partners in [head_dim/2 + rotary_pairs, head_dim) are
// passed through. This matches HF's proportional-rope layout for Gemma 4
// full-attn layers (head_dim=512, partial_rotary_factor=0.25 ⇒
// rotary_pairs = int(0.25 * 512 / 2) = 64).
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
    int rotary_pairs,
    float theta)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int head_half = head_dim / 2;
    const int pos = positions[t];

    __nv_bfloat16* xh = x + ((size_t)t * heads + h) * head_dim;

    for (int i = tid; i < rotary_pairs; i += BLOCK_THREADS) {
        float inv_freq = __powf(theta, -2.0f * (float)i / (float)head_dim);
        float angle = (float)pos * inv_freq;
        float c, s;
        __sincosf(angle, &s, &c);
        float x0 = __bfloat162float(xh[i]);
        float x1 = __bfloat162float(xh[i + head_half]);
        xh[i]             = __float2bfloat16_rn(x0 * c - x1 * s);
        xh[i + head_half] = __float2bfloat16_rn(x1 * c + x0 * s);
    }
}

// `rotary_dim` is the number of actually rotated dimensions (2 * rotary_pairs).
// It must be even and <= head_dim. Internally we only need `rotary_pairs =
// rotary_dim / 2` to know how many pair-angles rotate.
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
    if ((head_dim & 1) != 0) return -(int)cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(tokens, heads);
    dim3 block(BLOCK_THREADS);
    xk_rope_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)x,
        (const int*)positions,
        heads,
        head_dim,
        rotary_dim / 2,
        theta);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
