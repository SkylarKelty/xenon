// Naive multi-head attention (GQA-aware) for correctness baselines.
//
// Layouts (all row-major):
//   q: [T_q,  H,     D]
//   k: [T_kv, H_kv,  D]
//   v: [T_kv, H_kv,  D]
//   out: [T_q, H, D]
// GQA: kv_head = q_head / (H / H_kv).
// Masking: causal (kv_pos <= q_pos) + optional sliding window.
//
// One block per (q_tok, q_head). Keeps entire row of scores in shared memory,
// so T_kv is bounded by shmem (per-block dynamic shmem ~48 KiB default, so
// T_kv <= ~11K fp32 entries). That's plenty for correctness runs; swap to a
// tiled FlashAttention-style kernel for production.
//
// fp32 accumulation throughout, bf16 I/O.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

static constexpr int BLOCK_THREADS = 128;

__global__ void xk_attn_naive_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window,
    int gqa_group)
{
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce = smem + T_kv;

    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int q_tok  = blk / H;
    const int q_head = blk % H;
    const int kv_head = q_head / gqa_group;
    const int q_pos = q_pos_base + q_tok;
    const int kv_min = window > 0 ? max(0, q_pos - window + 1) : 0;
    const int kv_max = q_pos;

    const __nv_bfloat16* q_vec = q + ((size_t)q_tok * H + q_head) * D;

    // Phase 1: scaled dot products for kv positions in valid window; -inf else.
    for (int j = tid; j < T_kv; j += BLOCK_THREADS) {
        if (j >= kv_min && j <= kv_max) {
            const __nv_bfloat16* k_vec = k + ((size_t)j * H_kv + kv_head) * D;
            float acc = 0.0f;
            for (int i = 0; i < D; ++i) {
                acc += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
            }
            scores[j] = acc * scale;
        } else {
            scores[j] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Phase 2: max.
    float local_max = -FLT_MAX;
    for (int j = tid; j < T_kv; j += BLOCK_THREADS) {
        if (scores[j] > local_max) local_max = scores[j];
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = reduce[tid];
            float b = reduce[tid + s];
            reduce[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    const float mx = reduce[0];

    // Phase 3: exp & sum.
    float local_sum = 0.0f;
    for (int j = tid; j < T_kv; j += BLOCK_THREADS) {
        if (j >= kv_min && j <= kv_max) {
            float e = __expf(scores[j] - mx);
            scores[j] = e;
            local_sum += e;
        } else {
            scores[j] = 0.0f;
        }
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduce[0];

    // Phase 4: normalized probs * V, projected across D.
    __nv_bfloat16* out_vec = out + ((size_t)q_tok * H + q_head) * D;
    for (int d = tid; d < D; d += BLOCK_THREADS) {
        float acc = 0.0f;
        for (int j = kv_min; j <= kv_max; ++j) {
            acc += scores[j] * __bfloat162float(v[((size_t)j * H_kv + kv_head) * D + d]);
        }
        out_vec[d] = __float2bfloat16_rn(acc * inv_sum);
    }
}

extern "C" int xk_attn_naive_bf16(
    void* out,
    const void* q,
    const void* k,
    const void* v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window,
    void* stream)
{
    if (T_q <= 0 || T_kv <= 0 || H <= 0 || H_kv <= 0 || D <= 0) return 0;
    if (H % H_kv != 0) return -(int)cudaErrorInvalidValue;
    const int gqa_group = H / H_kv;

    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(T_q * H);
    dim3 block(BLOCK_THREADS);
    size_t shmem = (T_kv + BLOCK_THREADS) * sizeof(float);
    xk_attn_naive_bf16_kernel<<<grid, block, shmem, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v,
        T_q, T_kv, H, H_kv, D, scale, q_pos_base, window, gqa_group);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
