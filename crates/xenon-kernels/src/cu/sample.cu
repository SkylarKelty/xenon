// Sampling kernel: temperature-scaled softmax + block-wide top-K extraction.
//
// Single block, 256 threads. One scan of the full vocab for max+scratch-fill,
// one scan for the softmax denominator, then K passes of block-wide argmax
// against the f32 scratch (with each pick stamped as -FLT_MAX so later passes
// skip it). Output: [K] f32 softmax probs (normalized over the full vocab)
// and [K] u32 token ids, in descending-prob order. Host applies top-P + CDF
// sample on the compact K-sized slice.
//
// Greedy fast path (`greedy=1`): skips phase 2/3, writes only the argmax.
// No scratch buffer required in that case. Greedy is used by the default
// decode loop; the full path only runs when temperature > 0.
//
// Numerical model: fp32 throughout. Read bf16 logits, multiply by 1/T, store
// to f32 scratch; all reductions use fp32. Drift vs a reference softmax
// implementation is within a few ULPs on the K returned probs.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

static constexpr int BLOCK_THREADS = 256;
static constexpr int WARP = 32;
static constexpr int N_WARPS = BLOCK_THREADS / WARP;  // 8

// Warp-wide argmax reduce with index tie-break toward the lower idx. Matches
// the host `argmax_bf16` in xenon-engine (first-wins on equal values) so the
// greedy fast path is bit-identical to the pre-kernel behaviour.
__device__ inline void warp_argmax_f32i(float& v, int& i) {
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFF, v, off);
        int   oi = __shfl_xor_sync(0xFFFFFFFF, i, off);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
}

__device__ inline float warp_sum_f32(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

__global__ void xk_sample_topk_bf16_kernel(
    float* __restrict__ out_probs,     // [K]
    uint32_t* __restrict__ out_ids,    // [K]
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ scratch,       // [vocab], unused when greedy=1
    int vocab, float inv_T, int top_k, int greedy)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid >> 5;

    __shared__ float s_wmax[N_WARPS];
    __shared__ int   s_widx[N_WARPS];
    __shared__ float s_wsum[N_WARPS];
    __shared__ float s_max;
    __shared__ float s_sum_inv;

    // --- Phase 1: scan logits once. Find global max. If not greedy, also
    // populate the scratch f32 buffer with (logit * 1/T) so later phases can
    // mutate it (phase 3 stamps -FLT_MAX on each pick). ---
    float lm = -FLT_MAX;
    int   li = INT_MAX;
    if (greedy) {
        for (int i = tid; i < vocab; i += BLOCK_THREADS) {
            float v = __bfloat162float(logits[i]);
            if (v > lm || (v == lm && i < li)) { lm = v; li = i; }
        }
    } else {
        for (int i = tid; i < vocab; i += BLOCK_THREADS) {
            float v = __bfloat162float(logits[i]) * inv_T;
            scratch[i] = v;
            if (v > lm || (v == lm && i < li)) { lm = v; li = i; }
        }
    }
    warp_argmax_f32i(lm, li);
    if (lane == 0) { s_wmax[warp] = lm; s_widx[warp] = li; }
    __syncthreads();
    if (warp == 0) {
        float v = (lane < N_WARPS) ? s_wmax[lane] : -FLT_MAX;
        int   i = (lane < N_WARPS) ? s_widx[lane] : INT_MAX;
        warp_argmax_f32i(v, i);
        if (lane == 0) {
            s_max = v;
            if (greedy) {
                out_probs[0] = 1.0f;
                out_ids[0]   = (uint32_t)i;
            }
        }
    }
    __syncthreads();
    if (greedy) return;

    // --- Phase 2: denominator = sum of exp(scratch[i] - max). ---
    const float maxv = s_max;
    float ls = 0.0f;
    for (int i = tid; i < vocab; i += BLOCK_THREADS) {
        ls += __expf(scratch[i] - maxv);
    }
    ls = warp_sum_f32(ls);
    if (lane == 0) s_wsum[warp] = ls;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < N_WARPS) ? s_wsum[lane] : 0.0f;
        v = warp_sum_f32(v);
        if (lane == 0) s_sum_inv = 1.0f / v;
    }
    __syncthreads();

    // --- Phase 3: iterative top-K on scratch. After each pick, overwrite the
    // winner's scratch slot with -FLT_MAX so the next pass skips it. Reading
    // scratch across K iterations hits L2 after the first sweep, so per-iter
    // cost is well under a µs for our vocab. ---
    const float sum_inv = s_sum_inv;
    for (int k = 0; k < top_k; ++k) {
        float lm_k = -FLT_MAX;
        int   li_k = INT_MAX;
        for (int i = tid; i < vocab; i += BLOCK_THREADS) {
            float v = scratch[i];
            if (v > lm_k || (v == lm_k && i < li_k)) { lm_k = v; li_k = i; }
        }
        warp_argmax_f32i(lm_k, li_k);
        if (lane == 0) { s_wmax[warp] = lm_k; s_widx[warp] = li_k; }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < N_WARPS) ? s_wmax[lane] : -FLT_MAX;
            int   i = (lane < N_WARPS) ? s_widx[lane] : INT_MAX;
            warp_argmax_f32i(v, i);
            if (lane == 0) {
                out_probs[k] = __expf(v - maxv) * sum_inv;
                out_ids[k]   = (uint32_t)i;
                scratch[i]   = -FLT_MAX;
            }
        }
        __syncthreads();
    }
}

extern "C" int xk_sample_topk_bf16(
    void* out_probs, void* out_ids,
    const void* logits, void* scratch,
    int vocab, float temperature, int top_k, int greedy,
    void* stream)
{
    if (vocab <= 0 || top_k < 1) return -(int)cudaErrorInvalidValue;
    if (!greedy) {
        if (temperature <= 0.0f) return -(int)cudaErrorInvalidValue;
        if (scratch == nullptr) return -(int)cudaErrorInvalidValue;
    }
    cudaStream_t s = (cudaStream_t)stream;
    const float inv_T = greedy ? 1.0f : (1.0f / temperature);
    xk_sample_topk_bf16_kernel<<<1, BLOCK_THREADS, 0, s>>>(
        (float*)out_probs, (uint32_t*)out_ids,
        (const __nv_bfloat16*)logits, (float*)scratch,
        vocab, inv_T, top_k, greedy);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
