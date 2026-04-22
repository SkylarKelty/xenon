// Token embedding gather with optional scalar multiplier.
//
// Given token IDs (int32 [T]) and embedding table (bf16 [V, H]), produce
// embedded tokens (bf16 [T, H]). If `scale != 1.0`, each gathered element
// is multiplied by `scale` in fp32 before bf16 writeback. Used to apply
// Gemma's `embed_scale = sqrt(hidden)` that the Python `ScaledWordEmbedding`
// applies inside its forward pass.

#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int BLOCK_THREADS = 256;

__global__ void xk_embed_gather_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ table,
    const int* __restrict__ ids,
    int vocab,
    int hidden,
    float scale)
{
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    int id = ids[t];
    if (id < 0 || id >= vocab) id = 0;
    const __nv_bfloat16* row = table + (size_t)id * hidden;
    __nv_bfloat16* dst = out + (size_t)t * hidden;
    const bool unity = (scale == 1.0f);
    for (int i = tid; i < hidden; i += BLOCK_THREADS) {
        if (unity) {
            dst[i] = row[i];
        } else {
            dst[i] = __float2bfloat16_rn(__bfloat162float(row[i]) * scale);
        }
    }
}

extern "C" int xk_embed_gather_bf16(
    void* out,
    const void* table,
    const void* ids,
    int tokens,
    int vocab,
    int hidden,
    float scale,
    void* stream)
{
    if (tokens <= 0 || hidden <= 0 || vocab <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(tokens);
    dim3 block(BLOCK_THREADS);
    xk_embed_gather_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)table,
        (const int*)ids,
        vocab, hidden, scale);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
