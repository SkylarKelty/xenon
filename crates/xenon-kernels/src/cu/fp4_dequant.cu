// NVFP4 dequantization to bf16.
//
// Input layout (modelopt modelopt=NVFP4, group_size=16):
//   packed   : u8 [rows, cols/2]   — two FP4 E2M1 values per byte, low nibble
//                                    is the even-index element.
//   scales   : u8 [rows, cols/16]  — one UE4M3 scale per 16-element K block.
//   scale2   : f32 scalar          — per-tensor global scale.
//
// Dequantized value for element (r, c):
//   fp4_e2m1_to_f32(nibble) * ue4m3_to_f32(scales[r, c/16]) * scale2.
//
// Expects cols to be a positive multiple of 16.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// FP4 E2M1 values for each 4-bit codepoint (signed; top bit = sign).
__device__ __constant__ float XK_FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Decode UE4M3 (FP8 E4M3, sign forced to +). 8-bit layout:
//   bit 7 : sign  (ignored — UE4M3 is non-negative)
//   bits 3-6 : exponent, bias 7
//   bits 0-2 : mantissa
__device__ __forceinline__ float xk_ue4m3_to_f32(uint8_t b) {
    uint32_t x   = b & 0x7Fu;
    uint32_t exp = (x >> 3) & 0xFu;
    uint32_t man = x & 0x7u;
    if (exp == 0) {
        // Subnormal: 2^-6 * (mantissa / 8).
        return (float)man * (1.0f / 512.0f);
    }
    // Normal: (8 + man) * 2^(exp - 10), which equals 2^(exp-7) * (1 + man/8).
    return ldexpf((float)(8u + man), (int)exp - 10);
}

__global__ void xk_fp4_dequant_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    float global_scale,
    int rows,
    int cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || col >= cols) return;

    const int byte_idx = col >> 1;
    const int nibble   = col & 1;
    const uint8_t byte = packed[(size_t)row * (cols >> 1) + byte_idx];
    const uint8_t code = nibble ? (byte >> 4) : (byte & 0xF);
    const float   fp4  = XK_FP4_LUT[code];

    const int block_idx = col >> 4;  // / 16
    const uint8_t sb    = scales[(size_t)row * (cols >> 4) + block_idx];
    const float   bs    = xk_ue4m3_to_f32(sb);

    const float v = fp4 * bs * global_scale;
    out[(size_t)row * cols + col] = __float2bfloat16_rn(v);
}

// Returns 0 on success; negative cudaError_t on failure; -1 if cols is invalid.
extern "C" int xk_fp4_dequant_bf16(
    void* out,
    const void* packed,
    const void* scales,
    float global_scale,
    int rows,
    int cols,
    void* stream)
{
    if (rows <= 0 || cols <= 0) return 0;
    if ((cols & 15) != 0) return -1;  // must be multiple of 16

    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    cudaStream_t s = (cudaStream_t)stream;
    xk_fp4_dequant_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)out,
        (const uint8_t*)packed,
        (const uint8_t*)scales,
        global_scale,
        rows,
        cols);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : -(int)e;
}
