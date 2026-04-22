// Extract one layer's slice from a [tokens, num_layers, per_layer_dim] bf16
// tensor into a contiguous [tokens, per_layer_dim] buffer.
//
// Source layout: row-major with strides (L*Hl, Hl, 1).
// Destination: row-major [T, Hl], stride (Hl, 1).
//
// Grid: (T). Block: 128 threads, strided copy of Hl elements per row.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int BLOCK_THREADS = 128;

__global__ void xk_per_layer_slice_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ src,
    int tokens, int num_layers, int per_layer_dim, int layer_idx)
{
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_in = src + ((size_t)t * num_layers + layer_idx) * per_layer_dim;
    __nv_bfloat16* row_out = out + (size_t)t * per_layer_dim;
    for (int i = tid; i < per_layer_dim; i += BLOCK_THREADS) {
        row_out[i] = row_in[i];
    }
}

extern "C" int xk_per_layer_slice_bf16(
    void* out,
    const void* src,
    int tokens,
    int num_layers,
    int per_layer_dim,
    int layer_idx,
    void* stream)
{
    if (tokens <= 0 || per_layer_dim <= 0) return 0;
    if (layer_idx < 0 || layer_idx >= num_layers) return -(int)cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(tokens);
    dim3 block(BLOCK_THREADS);
    xk_per_layer_slice_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)src,
        tokens, num_layers, per_layer_dim, layer_idx);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
