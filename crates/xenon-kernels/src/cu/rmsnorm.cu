// RMSNorm forward: y[r, i] = x[r, i] * rsqrt(mean_r(x^2) + eps) * weight[i].
//
// One block per row. Strided load, shared-memory reduction, bf16 I/O with
// fp32 accumulation. Tuned for hidden sizes 2560 / 10240 (Gemma 4 residual
// and MLP).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int BLOCK_THREADS = 256;

__global__ void xk_rmsnorm_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,  // nullable: null = no-scale RMS
    int hidden,
    float eps)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* row_in  = x   + (size_t)row * hidden;
    __nv_bfloat16*       row_out = out + (size_t)row * hidden;

    // Sum of squares, accumulated in fp32.
    float local = 0.0f;
    for (int i = tid; i < hidden; i += BLOCK_THREADS) {
        float v = __bfloat162float(row_in[i]);
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();

    // Pairwise reduction in shared memory.
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float mean_sq = smem[0] / (float)hidden;
    const float scale   = rsqrtf(mean_sq + eps);

    // Apply: fp32 arithmetic, bf16 writeback. If weight is null, treat as 1.
    for (int i = tid; i < hidden; i += BLOCK_THREADS) {
        float v = __bfloat162float(row_in[i]);
        float w = weight ? __bfloat162float(weight[i]) : 1.0f;
        row_out[i] = __float2bfloat16_rn(v * scale * w);
    }
}

// C entry point. Returns 0 on success, negative cudaError_t on failure.
extern "C" int xk_rmsnorm_bf16(
    void* out,
    const void* x,
    const void* weight,
    int rows,
    int hidden,
    float eps,
    void* stream)
{
    if (rows <= 0 || hidden <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(rows);
    dim3 block(BLOCK_THREADS);
    size_t shmem = BLOCK_THREADS * sizeof(float);
    xk_rmsnorm_bf16_kernel<<<grid, block, shmem, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)weight,
        hidden,
        eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
