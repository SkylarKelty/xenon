// Swizzle NVFP4 per-16 UE4M3 block scales from row-major [M, K_blocks]
// into the 128x4-tile-interleaved layout that cuBLASLt's
// CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 mode requires.
//
// Equivalent to vLLM's `swizzle_blockscale` (nvfp4_utils.py):
//   padded = zeros([Mp, Kp])                 # Mp = round_up(M, 128), Kp = round_up(Kb, 4)
//   padded[:M, :Kb] = src
//   padded = padded.reshape(1, Mp/128, 4, 32, Kp/4, 4)
//   out = padded.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(Mp, Kp)
//
// The permutation rearranges each 128-row × 4-k_block super-tile:
// within a tile, src row m = A*128 + D*32 + C (A=tile, D∈[0,4), C∈[0,32)),
// src k_block k = B*4 + E (B=k_tile, E∈[0,4)); in dst the order becomes
// (A, B, C, D, E), stride pattern 128*Kp / 512 / 16 / 4 / 1.
//
// Out-of-range lanes (m >= M or k >= Kb) write zero — modelopt safetensors
// store scales exactly-sized, so padding only happens when M isn't a
// multiple of 128 (e.g. per-step activations).
//
// Launch: gridDim = (Mp/128, Kp/4), blockDim = (4, 32). Every thread writes
// 4 dst bytes (one per E). Cheap relative to the GEMM it feeds.

#include <cuda_runtime.h>
#include <cstdint>

__global__ void xk_swizzle_blockscale_kernel(
    uint8_t* __restrict__ dst,         // [Mp, Kp]
    const uint8_t* __restrict__ src,   // [M, Kb]
    int M, int Kb,
    int Mp, int Kp)
{
    const int A = blockIdx.x;         // m-tile index
    const int B = blockIdx.y;         // k-tile index
    const int D = threadIdx.x;        // m_sub1 in [0,4)
    const int C = threadIdx.y;        // m_sub2 in [0,32)

    const int m_src = A * 128 + D * 32 + C;

    #pragma unroll
    for (int E = 0; E < 4; ++E) {
        const int k_src = B * 4 + E;
        uint8_t val = 0;
        if (m_src < M && k_src < Kb) {
            val = src[(size_t)m_src * Kb + k_src];
        }
        const size_t dst_off =
            (size_t)A * (size_t)Kp * 128
          + (size_t)B * 512
          + (size_t)C * 16
          + (size_t)D * 4
          + E;
        dst[dst_off] = val;
    }
}

// Returns 0 on success, negative cudaError_t on failure.
// `dst` must be `Mp * Kp` bytes where Mp = round_up(M, 128), Kp = round_up(Kb, 4).
extern "C" int xk_swizzle_blockscale(
    void* dst,
    const void* src,
    int M,
    int Kb,
    int Mp,
    int Kp,
    void* stream)
{
    if (M <= 0 || Kb <= 0) return 0;
    if (Mp <= 0 || Kp <= 0) return -1;
    if (Mp % 128 != 0) return -2;
    if (Kp % 4 != 0) return -3;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)(Mp / 128), (unsigned)(Kp / 4));
    dim3 block(4, 32);
    xk_swizzle_blockscale_kernel<<<grid, block, 0, s>>>(
        (uint8_t*)dst,
        (const uint8_t*)src,
        M, Kb, Mp, Kp);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : -(int)e;
}
