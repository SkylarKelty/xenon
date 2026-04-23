// Device-resident KV cache append + cur_len increment.
//
// The host-memcpy append path (`KvCache::append_from_offset`) bakes the
// destination offset (= cur_len * row_elts) into the memcpy node at graph
// capture time, so replays would write every step at the same slot and
// silently corrupt the cache. These two kernels do the same work but read
// cur_len indirectly through a device pointer, so the captured graph adapts
// to whatever cur_len holds at replay time.
//
// `xk_kv_append_bf16` copies `n_tokens * row_elts` bf16 elements from `src`
// (at `src_offset`) into `kv_buf` at offset `(*cur_len_ptr) * row_elts`.
// `xk_inc_i32` increments `*counter_ptr` by `n_tokens`.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

static constexpr int APPEND_BLOCK = 256;

__global__ void xk_kv_append_bf16_kernel(
    __nv_bfloat16* __restrict__ kv_buf,
    const __nv_bfloat16* __restrict__ src,
    int src_offset,
    const int* __restrict__ cur_len_ptr,
    int row_elts,
    int n_tokens)
{
    const int tid = blockIdx.x * APPEND_BLOCK + threadIdx.x;
    const int total = n_tokens * row_elts;
    if (tid >= total) return;
    const int cur_len = *cur_len_ptr;
    const size_t dst_off = (size_t)cur_len * (size_t)row_elts + (size_t)tid;
    const size_t src_off = (size_t)src_offset + (size_t)tid;
    kv_buf[dst_off] = src[src_off];
}

__global__ void xk_inc_i32_kernel(int* counter_ptr, int delta) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *counter_ptr += delta;
    }
}

extern "C" int xk_kv_append_bf16(
    void* kv_buf,
    const void* src,
    int src_offset,
    const void* cur_len_ptr,
    int row_elts,
    int n_tokens,
    void* stream)
{
    if (row_elts <= 0 || n_tokens <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    const int total = n_tokens * row_elts;
    const int grid = (total + APPEND_BLOCK - 1) / APPEND_BLOCK;
    xk_kv_append_bf16_kernel<<<grid, APPEND_BLOCK, 0, s>>>(
        (__nv_bfloat16*)kv_buf,
        (const __nv_bfloat16*)src,
        src_offset,
        (const int*)cur_len_ptr,
        row_elts, n_tokens);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}

extern "C" int xk_inc_i32(
    void* counter_ptr,
    int delta,
    void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    xk_inc_i32_kernel<<<1, 1, 0, s>>>((int*)counter_ptr, delta);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
