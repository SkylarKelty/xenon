// FlashAttention-2 style tiled attention.
//
// Processes K/V in tiles of BR rows, keeps running (max, sum, O) across tiles
// via the standard online-softmax update. Shared memory footprint is
// O(BR * D) instead of O(T_kv), so this handles arbitrary context length.
//
// Layouts match attn_naive.cu: q [T_q, H, D], k/v [T_kv, H_kv, D].
// Each block handles one (q_tok, q_head) pair. Thread i owns
// D_PER_THREAD = D / BLOCK_THREADS dims of the Q vector and O accumulator.
//
// Constraints:
// - D % BLOCK_THREADS == 0  (D = 256 or 512 with BLOCK_THREADS = 128 works).
// - MAX_D_PER_THREAD >= D_PER_THREAD  (static bound for register arrays).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

static constexpr int BR = 16;
static constexpr int BLOCK_THREADS = 128;
static constexpr int MAX_D_PER_THREAD = 8;  // D up to 512 * 2 = 1024 headroom

__global__ void xk_attn_flash_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window, int gqa_group)
{
    extern __shared__ unsigned char smem_buf[];
    __nv_bfloat16* K_tile = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* V_tile = K_tile + BR * D;
    float* scores = (float*)(V_tile + BR * D);  // [BR]
    float* reduce_scratch = scores + BR;        // [BLOCK_THREADS]

    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int q_tok = blk / H;
    const int q_head = blk % H;
    const int kv_head = q_head / gqa_group;
    const int q_pos = q_pos_base + q_tok;
    const int kv_min = window > 0 ? max(0, q_pos - window + 1) : 0;
    const int kv_max = q_pos;

    const int D_PER_THREAD = D / BLOCK_THREADS;

    // Load Q row into thread-local registers.
    float q_reg[MAX_D_PER_THREAD];
    const __nv_bfloat16* q_vec = q + ((size_t)q_tok * H + q_head) * D;
    #pragma unroll
    for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
        if (i < D_PER_THREAD) {
            int d = tid + i * BLOCK_THREADS;
            q_reg[i] = d < D ? __bfloat162float(q_vec[d]) : 0.0f;
        }
    }

    // O accumulator, running (m, l).
    float o_reg[MAX_D_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < MAX_D_PER_THREAD; ++i) o_reg[i] = 0.0f;
    float m = -FLT_MAX;
    float l = 0.0f;

    // Iterate over K/V tiles.
    for (int tile_start = 0; tile_start < T_kv; tile_start += BR) {
        const int tile_len = min(BR, T_kv - tile_start);

        // Cooperatively load K_tile and V_tile into shared memory.
        const int total = tile_len * D;
        for (int e = tid; e < total; e += BLOCK_THREADS) {
            int jj = e / D;
            int dd = e % D;
            int kv_pos = tile_start + jj;
            size_t src = ((size_t)kv_pos * H_kv + kv_head) * D + dd;
            K_tile[jj * D + dd] = k[src];
            V_tile[jj * D + dd] = v[src];
        }
        __syncthreads();

        // Compute scores[j] = scale * (Q · K_tile[j]) for each row in tile.
        // Each thread contributes a partial dot; reduce across the block.
        for (int j = 0; j < tile_len; ++j) {
            int kv_pos = tile_start + j;
            bool masked = (kv_pos < kv_min) || (kv_pos > kv_max);
            float local = 0.0f;
            if (!masked) {
                #pragma unroll
                for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
                    if (i < D_PER_THREAD) {
                        int d = tid + i * BLOCK_THREADS;
                        if (d < D) local += q_reg[i] * __bfloat162float(K_tile[j * D + d]);
                    }
                }
            }
            reduce_scratch[tid] = local;
            __syncthreads();
            for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_scratch[tid] += reduce_scratch[tid + s];
                __syncthreads();
            }
            if (tid == 0) {
                scores[j] = masked ? -FLT_MAX : reduce_scratch[0] * scale;
            }
            __syncthreads();
        }

        // All threads compute identical m_tile, m_new, alpha from scores.
        float m_tile = -FLT_MAX;
        for (int j = 0; j < tile_len; ++j) if (scores[j] > m_tile) m_tile = scores[j];
        float m_new = fmaxf(m, m_tile);
        float alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - m_new);

        // Compute per-row P and tile_sum in thread-local registers.
        float p_local[BR];
        float tile_sum = 0.0f;
        for (int j = 0; j < tile_len; ++j) {
            float p = (scores[j] > -FLT_MAX / 2.0f) ? __expf(scores[j] - m_new) : 0.0f;
            p_local[j] = p;
            tile_sum += p;
        }

        // Update O: O = alpha * O + P @ V_tile (owned dims only).
        #pragma unroll
        for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
            if (i < D_PER_THREAD) {
                int d = tid + i * BLOCK_THREADS;
                if (d < D) {
                    float acc = alpha * o_reg[i];
                    for (int j = 0; j < tile_len; ++j) {
                        acc += p_local[j] * __bfloat162float(V_tile[j * D + d]);
                    }
                    o_reg[i] = acc;
                }
            }
        }

        l = alpha * l + tile_sum;
        m = m_new;
        __syncthreads();
    }

    // Finalize and write O / l.
    __nv_bfloat16* out_vec = out + ((size_t)q_tok * H + q_head) * D;
    float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
        if (i < D_PER_THREAD) {
            int d = tid + i * BLOCK_THREADS;
            if (d < D) out_vec[d] = __float2bfloat16_rn(o_reg[i] * inv_l);
        }
    }
}

extern "C" int xk_attn_flash_bf16(
    void* out, const void* q, const void* k, const void* v,
    int T_q, int T_kv, int H, int H_kv, int D,
    float scale, int q_pos_base, int window, void* stream)
{
    if (T_q <= 0 || T_kv <= 0 || H <= 0 || H_kv <= 0 || D <= 0) return 0;
    if (H % H_kv != 0) return -(int)cudaErrorInvalidValue;
    if (D % BLOCK_THREADS != 0) return -(int)cudaErrorInvalidValue;
    if (D / BLOCK_THREADS > MAX_D_PER_THREAD) return -(int)cudaErrorInvalidValue;
    const int gqa_group = H / H_kv;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(T_q * H);
    dim3 block(BLOCK_THREADS);
    size_t shmem = (size_t)BR * D * 2 * sizeof(__nv_bfloat16)
                 + (size_t)BR * sizeof(float)
                 + (size_t)BLOCK_THREADS * sizeof(float);
    xk_attn_flash_bf16_kernel<<<grid, block, shmem, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v,
        T_q, T_kv, H, H_kv, D, scale, q_pos_base, window, gqa_group);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
