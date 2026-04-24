// Emotion probe scoring: out[t, n] = dot(x[t] - mean, vectors[n])
//
// x:       [T, H] bf16 (row-major residual stream)
// mean:    [H]    bf16 (per-layer global mean, subtracted before projection)
// vectors: [N, H] bf16 (row-major, each row is a unit-normed emotion direction)
// out:     [T, N] fp32
//
// One block per output scalar (t, n); block reduces across the H dim with
// BLOCK_THREADS = 256 threads. H=2560, N=171 in the Gemma 4 E4B setup, so
// grid is (N, T) and the full call is 0.88 MFLOP per token — negligible vs
// one decode step.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int BLOCK_THREADS = 256;

__global__ void xk_emotion_score_bf16_kernel(
    float* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ mean,
    const __nv_bfloat16* __restrict__ vectors,
    int t, int h, int n,
    int accumulate)
{
    extern __shared__ float smem[];
    const int n_idx = blockIdx.x;
    const int t_idx = blockIdx.y;
    const int tid   = threadIdx.x;

    if (t_idx >= t || n_idx >= n) return;

    const __nv_bfloat16* x_row = x       + (size_t)t_idx * h;
    const __nv_bfloat16* v_row = vectors + (size_t)n_idx * h;

    float acc = 0.0f;
    for (int i = tid; i < h; i += BLOCK_THREADS) {
        float xi = __bfloat162float(x_row[i]) - __bfloat162float(mean[i]);
        float vi = __bfloat162float(v_row[i]);
        acc += xi * vi;
    }
    smem[tid] = acc;
    __syncthreads();

    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        const size_t o = (size_t)t_idx * n + n_idx;
        if (accumulate) {
            // Each (t_idx, n_idx) slot has exactly one writer per call, so
            // atomicAdd is contention-free within this launch; its point is
            // preserving the pre-existing value across repeated accumulating
            // calls.
            atomicAdd(&out[o], smem[0]);
        } else {
            out[o] = smem[0];
        }
    }
}

extern "C" int xk_emotion_score_bf16(
    void* out,
    const void* x,
    const void* mean,
    const void* vectors,
    int t,
    int h,
    int n,
    int accumulate,
    void* stream)
{
    if (t <= 0 || h <= 0 || n <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)n, (unsigned)t);
    dim3 block(BLOCK_THREADS);
    size_t shmem = BLOCK_THREADS * sizeof(float);
    xk_emotion_score_bf16_kernel<<<grid, block, shmem, s>>>(
        (float*)out,
        (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)mean,
        (const __nv_bfloat16*)vectors,
        t, h, n, accumulate);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
