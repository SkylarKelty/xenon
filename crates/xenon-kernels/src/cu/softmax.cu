// Attention softmax with causal + optional sliding-window mask.
//
// scores layout: [rows, T_kv] row-major, one row per (batch*head, q_token).
// For each row: multiply by `scale`, mask positions kv_pos > q_pos (causal);
// when window > 0, additionally mask kv_pos < q_pos - window + 1.
// q_pos for row r is `q_pos_base + (r % T_q)`.
//
// fp32 reductions, bf16 I/O.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

static constexpr int BLOCK_THREADS = 256;

__global__ void xk_softmax_attn_bf16_kernel(
    __nv_bfloat16* __restrict__ scores,
    int T_q,
    int T_kv,
    float scale,
    int q_pos_base,
    int window)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int q_local = row % T_q;
    const int q_pos = q_pos_base + q_local;
    const int kv_min = window > 0 ? max(0, q_pos - window + 1) : 0;
    const int kv_max = q_pos; // inclusive

    __nv_bfloat16* row_ptr = scores + (size_t)row * T_kv;

    // Pass 1: find masked max.
    float local_max = -FLT_MAX;
    for (int j = tid; j < T_kv; j += BLOCK_THREADS) {
        if (j >= kv_min && j <= kv_max) {
            float v = __bfloat162float(row_ptr[j]) * scale;
            if (v > local_max) local_max = v;
        }
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = smem[tid];
            float b = smem[tid + s];
            smem[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float row_max = smem[0];

    // Pass 2: exp & sum.
    float local_sum = 0.0f;
    for (int j = tid; j < T_kv; j += BLOCK_THREADS) {
        if (j >= kv_min && j <= kv_max) {
            float v = __bfloat162float(row_ptr[j]) * scale;
            local_sum += __expf(v - row_max);
        }
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / smem[0];

    // Pass 3: write normalized probs; masked positions become 0.
    for (int j = tid; j < T_kv; j += BLOCK_THREADS) {
        if (j >= kv_min && j <= kv_max) {
            float v = __bfloat162float(row_ptr[j]) * scale;
            row_ptr[j] = __float2bfloat16_rn(__expf(v - row_max) * inv_sum);
        } else {
            row_ptr[j] = __float2bfloat16_rn(0.0f);
        }
    }
}

extern "C" int xk_softmax_attn_bf16(
    void* scores,
    int rows,
    int T_q,
    int T_kv,
    float scale,
    int q_pos_base,
    int window,
    void* stream)
{
    if (rows <= 0 || T_q <= 0 || T_kv <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(rows);
    dim3 block(BLOCK_THREADS);
    size_t shmem = BLOCK_THREADS * sizeof(float);
    xk_softmax_attn_bf16_kernel<<<grid, block, shmem, s>>>(
        (__nv_bfloat16*)scores,
        T_q, T_kv, scale, q_pos_base, window);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
