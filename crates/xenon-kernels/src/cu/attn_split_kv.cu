// Split-KV multi-head attention (GQA-aware).
//
// Same math as attn_naive, but with a third grid dimension over T_kv chunks
// so decode (where T_q * H alone only yields 8 blocks) can saturate the GPU.
//
// Two-kernel design:
//   1. partial: grid = [T_q * H * n_chunks], each block owns a T_kv slice,
//      emits (local_max, local_sum_exp, numerator[D]) for its chunk.
//   2. merge:   grid = [T_q * H], each block combines n_chunks partials via
//      standard online-softmax and writes final bf16 out[D].
//
// Per-chunk work mirrors attn_naive_bf16: threads partition the T_kv slice
// and each thread does a full-D dot product independently — no intra-block
// reduction for the QK dot. Only cross-thread reductions are the local
// max/sum (single block-wide reduce each).
//
// Layouts (row-major):
//   q: [T_q,  H,     D]
//   k: [T_kv, H_kv,  D]
//   v: [T_kv, H_kv,  D]
//   out: [T_q, H, D]
//   partial_max: [T_q, H, n_chunks]              (fp32)
//   partial_sum: [T_q, H, n_chunks]              (fp32)
//   partial_num: [T_q, H, n_chunks, D]           (fp32)
// GQA: kv_head = q_head / (H / H_kv).
// Masking: causal (kv_pos <= q_pos) + optional sliding window.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

static constexpr int BLOCK_THREADS = 128;

__global__ void xk_attn_split_kv_partial_bf16_kernel(
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_num,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window,
    int gqa_group, int chunk_size, int n_chunks)
{
    extern __shared__ float smem[];
    float* scores = smem;                        // [chunk_size]
    float* reduce = smem + chunk_size;           // [BLOCK_THREADS]

    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int chunk_idx = blk % n_chunks;
    const int q_head    = (blk / n_chunks) % H;
    const int q_tok     = blk / (n_chunks * H);
    const int kv_head   = q_head / gqa_group;
    const int q_pos     = q_pos_base + q_tok;
    const int kv_min_v  = window > 0 ? max(0, q_pos - window + 1) : 0;
    const int kv_max_v  = q_pos;
    const int chunk_start = chunk_idx * chunk_size;
    const int chunk_end   = min(chunk_start + chunk_size, T_kv);
    const int chunk_len   = chunk_end - chunk_start;

    const __nv_bfloat16* q_vec = q + ((size_t)q_tok * H + q_head) * D;

    // Phase 1: scaled dot products for this chunk's slice.
    // Threads partition T_kv; each thread computes a full D-wide dot — no
    // intra-block reduction. Out-of-chunk and masked slots go to -FLT_MAX.
    for (int j_local = tid; j_local < chunk_size; j_local += BLOCK_THREADS) {
        int j = chunk_start + j_local;
        if (j_local < chunk_len && j >= kv_min_v && j <= kv_max_v) {
            const __nv_bfloat16* k_vec = k + ((size_t)j * H_kv + kv_head) * D;
            float acc = 0.0f;
            for (int i = 0; i < D; ++i) {
                acc += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
            }
            scores[j_local] = acc * scale;
        } else {
            scores[j_local] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Phase 2: local max over chunk.
    // LOOP3: warp-shuffle reduction instead of shared-memory tree.
    float local_max = -FLT_MAX;
    for (int j_local = tid; j_local < chunk_size; j_local += BLOCK_THREADS) {
        if (scores[j_local] > local_max) local_max = scores[j_local];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    __shared__ float warp_max[4];
    const int lane = tid & 31; const int wid = tid >> 5;
    if (lane == 0) warp_max[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float b = (lane < 4) ? warp_max[lane] : -FLT_MAX;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFFu, b, off);
            if (other > b) b = other;
        }
        if (lane == 0) warp_max[0] = b;
    }
    __syncthreads();
    const float mx = warp_max[0];

    // Phase 3: exp(score - mx) -> scores[]; zero masked slots; local sum.
    // LOOP3: warp-shuffle reduction for sum too.
    float local_sum = 0.0f;
    for (int j_local = tid; j_local < chunk_size; j_local += BLOCK_THREADS) {
        if (scores[j_local] > -FLT_MAX * 0.5f) {
            float e = __expf(scores[j_local] - mx);
            scores[j_local] = e;
            local_sum += e;
        } else {
            scores[j_local] = 0.0f;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, off);
    }
    __shared__ float warp_sum[4];
    if (lane == 0) warp_sum[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        float b = (lane < 4) ? warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            b += __shfl_xor_sync(0xFFFFFFFFu, b, off);
        }
        if (lane == 0) warp_sum[0] = b;
    }
    __syncthreads();
    const float sum_exp = warp_sum[0];

    // Phase 4: numerator[d] = Σ_j scores[j] * V[j, d] across this chunk.
    // Each thread owns a strided set of d's and iterates over the chunk.
    // Masked slots have scores[j_local] == 0 so they drop out naturally.
    const size_t pidx = ((size_t)q_tok * H + q_head) * n_chunks + chunk_idx;
    float* num_row = partial_num + pidx * D;
    for (int d = tid; d < D; d += BLOCK_THREADS) {
        float acc = 0.0f;
        for (int j_local = 0; j_local < chunk_len; ++j_local) {
            int j = chunk_start + j_local;
            acc += scores[j_local]
                 * __bfloat162float(v[((size_t)j * H_kv + kv_head) * D + d]);
        }
        num_row[d] = acc;
    }

    if (tid == 0) {
        partial_max[pidx] = mx;
        partial_sum[pidx] = sum_exp;
    }
}

__global__ void xk_attn_split_kv_merge_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_num,
    int T_q, int H, int D, int n_chunks)
{
    extern __shared__ float smem[];
    float* chunk_max = smem;                          // [n_chunks]
    float* chunk_sum = chunk_max + n_chunks;          // [n_chunks]
    float* reduce    = chunk_sum + n_chunks;          // [BLOCK_THREADS]

    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int q_tok  = blk / H;
    const int q_head = blk % H;
    const size_t base = ((size_t)q_tok * H + q_head) * n_chunks;

    // Load chunk_max / chunk_sum.
    for (int c = tid; c < n_chunks; c += BLOCK_THREADS) {
        chunk_max[c] = partial_max[base + c];
        chunk_sum[c] = partial_sum[base + c];
    }
    __syncthreads();

    // Global max across non-empty chunks (empty chunks have chunk_sum == 0).
    float local_max = -FLT_MAX;
    for (int c = tid; c < n_chunks; c += BLOCK_THREADS) {
        if (chunk_sum[c] > 0.0f && chunk_max[c] > local_max) local_max = chunk_max[c];
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
    const float gmx = reduce[0];

    // Global sum = Σ_c chunk_sum[c] * exp(chunk_max[c] - gmx).
    float local_sum = 0.0f;
    for (int c = tid; c < n_chunks; c += BLOCK_THREADS) {
        if (chunk_sum[c] > 0.0f) {
            local_sum += chunk_sum[c] * __expf(chunk_max[c] - gmx);
        }
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float gsum = reduce[0];
    const float inv_gsum = gsum > 0.0f ? 1.0f / gsum : 0.0f;

    // Output: out[d] = (Σ_c num[c, d] * exp(max[c] - gmx)) / gsum.
    __nv_bfloat16* out_vec = out + ((size_t)q_tok * H + q_head) * D;
    for (int d = tid; d < D; d += BLOCK_THREADS) {
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            if (chunk_sum[c] > 0.0f) {
                acc += partial_num[(base + c) * D + d]
                     * __expf(chunk_max[c] - gmx);
            }
        }
        out_vec[d] = __float2bfloat16_rn(acc * inv_gsum);
    }
}

extern "C" int xk_attn_split_kv_bf16(
    void* out,
    void* partial_max,
    void* partial_sum,
    void* partial_num,
    const void* q,
    const void* k,
    const void* v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window,
    int chunk_size,
    void* stream)
{
    if (T_q <= 0 || T_kv <= 0 || H <= 0 || H_kv <= 0 || D <= 0) return 0;
    if (H % H_kv != 0) return -(int)cudaErrorInvalidValue;
    if (chunk_size <= 0) return -(int)cudaErrorInvalidValue;
    const int gqa_group = H / H_kv;
    const int n_chunks = (T_kv + chunk_size - 1) / chunk_size;
    cudaStream_t s = (cudaStream_t)stream;

    // Partial kernel.
    {
        dim3 grid(T_q * H * n_chunks);
        dim3 block(BLOCK_THREADS);
        size_t shmem = (size_t)(chunk_size + BLOCK_THREADS) * sizeof(float);
        xk_attn_split_kv_partial_bf16_kernel<<<grid, block, shmem, s>>>(
            (float*)partial_max, (float*)partial_sum, (float*)partial_num,
            (const __nv_bfloat16*)q,
            (const __nv_bfloat16*)k,
            (const __nv_bfloat16*)v,
            T_q, T_kv, H, H_kv, D, scale, q_pos_base, window,
            gqa_group, chunk_size, n_chunks);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return -(int)e;
    }

    // Merge kernel.
    {
        dim3 grid(T_q * H);
        dim3 block(BLOCK_THREADS);
        size_t shmem = (size_t)(2 * n_chunks + BLOCK_THREADS) * sizeof(float);
        xk_attn_split_kv_merge_bf16_kernel<<<grid, block, shmem, s>>>(
            (__nv_bfloat16*)out,
            (const float*)partial_max,
            (const float*)partial_sum,
            (const float*)partial_num,
            T_q, H, D, n_chunks);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return -(int)e;
    }

    return 0;
}

// Device-resident variant of the partial kernel: `cur_pos` is read through
// a device pointer so the same launch config (grid, block, args) works at
// every decode step while cur_pos grows. Internally sets q_pos_base = cur_pos
// and T_kv = cur_pos + T_q (the post-append total valid K/V length).
// Grid size is fixed at capture time via `n_chunks` — chunks past the live
// T_kv mask all scores to -FLT_MAX and drop out of the merge kernel via
// the existing chunk_sum == 0 check.
__global__ void xk_attn_split_kv_partial_bf16_device_kernel(
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_num,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int T_q,
    const int* __restrict__ cur_pos_ptr,
    int H, int H_kv, int D,
    float scale,
    int window,
    int gqa_group, int chunk_size, int n_chunks)
{
    const int q_pos_base = *cur_pos_ptr;
    const int T_kv       = q_pos_base + T_q;

    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce = smem + chunk_size;

    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int chunk_idx = blk % n_chunks;
    const int q_head    = (blk / n_chunks) % H;
    const int q_tok     = blk / (n_chunks * H);
    const int kv_head   = q_head / gqa_group;
    const int q_pos     = q_pos_base + q_tok;
    const int kv_min_v  = window > 0 ? max(0, q_pos - window + 1) : 0;
    const int kv_max_v  = q_pos;
    const int chunk_start = chunk_idx * chunk_size;
    const int chunk_end   = min(chunk_start + chunk_size, T_kv);
    const int chunk_len   = chunk_end - chunk_start;

    const __nv_bfloat16* q_vec = q + ((size_t)q_tok * H + q_head) * D;

    for (int j_local = tid; j_local < chunk_size; j_local += BLOCK_THREADS) {
        int j = chunk_start + j_local;
        if (j_local < chunk_len && j >= kv_min_v && j <= kv_max_v) {
            const __nv_bfloat16* k_vec = k + ((size_t)j * H_kv + kv_head) * D;
            float acc = 0.0f;
            for (int i = 0; i < D; ++i) {
                acc += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
            }
            scores[j_local] = acc * scale;
        } else {
            scores[j_local] = -FLT_MAX;
        }
    }
    __syncthreads();

    float local_max = -FLT_MAX;
    for (int j_local = tid; j_local < chunk_size; j_local += BLOCK_THREADS) {
        if (scores[j_local] > local_max) local_max = scores[j_local];
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

    float local_sum = 0.0f;
    for (int j_local = tid; j_local < chunk_size; j_local += BLOCK_THREADS) {
        if (scores[j_local] > -FLT_MAX * 0.5f) {
            float e = __expf(scores[j_local] - mx);
            scores[j_local] = e;
            local_sum += e;
        } else {
            scores[j_local] = 0.0f;
        }
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float sum_exp = reduce[0];

    const size_t pidx = ((size_t)q_tok * H + q_head) * n_chunks + chunk_idx;
    float* num_row = partial_num + pidx * D;
    for (int d = tid; d < D; d += BLOCK_THREADS) {
        float acc = 0.0f;
        for (int j_local = 0; j_local < chunk_len; ++j_local) {
            int j = chunk_start + j_local;
            acc += scores[j_local]
                 * __bfloat162float(v[((size_t)j * H_kv + kv_head) * D + d]);
        }
        num_row[d] = acc;
    }

    if (tid == 0) {
        partial_max[pidx] = mx;
        partial_sum[pidx] = sum_exp;
    }
}

// Device-resident split-KV launch: `cur_pos` is a `[1] i32` device buffer,
// read by the kernel at launch time. `n_chunks_fixed` fixes the grid at
// capture time. Caller sizes the partial buffers for at least
// `T_q * H * n_chunks_fixed`.
extern "C" int xk_attn_split_kv_bf16_device(
    void* out,
    void* partial_max,
    void* partial_sum,
    void* partial_num,
    const void* q,
    const void* k,
    const void* v,
    int T_q,
    const void* cur_pos_ptr,
    int H, int H_kv, int D,
    float scale,
    int window,
    int chunk_size,
    int n_chunks_fixed,
    void* stream)
{
    if (T_q <= 0 || H <= 0 || H_kv <= 0 || D <= 0) return 0;
    if (H % H_kv != 0) return -(int)cudaErrorInvalidValue;
    if (chunk_size <= 0 || n_chunks_fixed <= 0) return -(int)cudaErrorInvalidValue;
    const int gqa_group = H / H_kv;
    cudaStream_t s = (cudaStream_t)stream;

    {
        dim3 grid(T_q * H * n_chunks_fixed);
        dim3 block(BLOCK_THREADS);
        size_t shmem = (size_t)(chunk_size + BLOCK_THREADS) * sizeof(float);
        xk_attn_split_kv_partial_bf16_device_kernel<<<grid, block, shmem, s>>>(
            (float*)partial_max, (float*)partial_sum, (float*)partial_num,
            (const __nv_bfloat16*)q,
            (const __nv_bfloat16*)k,
            (const __nv_bfloat16*)v,
            T_q, (const int*)cur_pos_ptr,
            H, H_kv, D, scale, window,
            gqa_group, chunk_size, n_chunks_fixed);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return -(int)e;
    }
    {
        dim3 grid(T_q * H);
        dim3 block(BLOCK_THREADS);
        size_t shmem = (size_t)(2 * n_chunks_fixed + BLOCK_THREADS) * sizeof(float);
        xk_attn_split_kv_merge_bf16_kernel<<<grid, block, shmem, s>>>(
            (__nv_bfloat16*)out,
            (const float*)partial_max,
            (const float*)partial_sum,
            (const float*)partial_num,
            T_q, H, D, n_chunks_fixed);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return -(int)e;
    }
    return 0;
}
