// Activation quantizer: bf16 [rows, cols] -> NVFP4 packed [rows, cols/2] + UE4M3
// block scales [rows, cols/16]. Matches the storage format of modelopt-quantized
// weights so both operands use the same VEC16_UE4M3 scale mode in cuBLASLt.
//
// Per 16-element block along the inner (K) dim:
//   1. find max_abs
//   2. scale = max_abs / 6.0                     (6.0 is max |E2M1|)
//   3. round scale to UE4M3 (exp 0..15, mantissa 0..7)
//   4. for each element: quant = round(x / decoded_scale) to nearest E2M1
//   5. pack two FP4 nibbles per byte (low = even index).
//
// One row per block, 16 threads per warp × (cols/16) warps feels unnecessary —
// keep it simple: one thread block per (row, block-of-16), each thread handles
// one of the 16 values. A single warp per block (16 threads active out of 32)
// is fine at our T=1..32 scale.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

// FP4 E2M1 decode LUT — same as in fp4_dequant.cu.
__device__ __constant__ float d_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Round |v| to the nearest positive E2M1 value and return its 3-bit magnitude code.
__device__ __forceinline__ unsigned e2m1_mag_code(float v) {
    float a = fabsf(v);
    // Walk the 8 positive E2M1 magnitudes and pick nearest (round to even on ties).
    unsigned best = 0;
    float best_d = fabsf(a - d_e2m1_lut[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        float d = fabsf(a - d_e2m1_lut[i]);
        // Ties-to-even: lower bit of i as tiebreaker.
        if (d < best_d || (d == best_d && (i & 1) == 0)) {
            best_d = d;
            best = i;
        }
    }
    return best;
}

// Encode a non-negative float as UE4M3 (FP8 E4M3 with sign=0).
__device__ __forceinline__ uint8_t f32_to_ue4m3(float x) {
    // UE4M3: (1.m) * 2^(exp - 7) for exp in [1, 15] (bias 7). exp=0 is subnormal.
    // Max repr ~ 448, min normal ~ 2^-6. Clamp + round-to-nearest-even.
    if (!(x > 0.0f)) return 0;
    if (x >= 448.0f) return (15u << 3) | 7u;  // saturate
    int exp;
    float mant = frexpf(x, &exp);   // mant in [0.5, 1.0), value = mant * 2^exp
    // Shift to (1.0, 2.0) mantissa form.
    mant *= 2.0f;                    // now in [1.0, 2.0)
    int e = exp - 1;                 // biased exponent of that form
    int biased = e + 7;
    if (biased <= 0) {
        // Subnormal: value = m / 512 for m in [0, 7]. Encode m = round(x * 512).
        int m = (int)roundf(x * 512.0f);
        if (m > 7) m = 7;
        return (uint8_t)m;
    }
    if (biased > 15) return (15u << 3) | 7u;
    // Normal: mantissa bits = round((mant - 1.0) * 8).
    int m = (int)roundf((mant - 1.0f) * 8.0f);
    if (m == 8) {                    // mantissa rolled over → bump exponent
        m = 0;
        biased += 1;
        if (biased > 15) return (15u << 3) | 7u;
    }
    return (uint8_t)((biased << 3) | m);
}

__device__ __forceinline__ float decode_ue4m3(uint8_t b) {
    unsigned exp = (b >> 3) & 0xF;
    unsigned man = b & 0x7;
    if (exp == 0) return (float)man * (1.0f / 512.0f);
    return (float)(8 + man) * exp2f((float)exp - 10.0f);
}

// Per-block kernel: blockIdx.y = row, blockIdx.x = block-of-16 within row.
// blockDim.x = 16.
__global__ void xk_nvfp4_quantize_bf16_kernel(
    uint8_t* __restrict__ packed,        // [rows, cols/2]
    uint8_t* __restrict__ scales,        // [rows, cols/16]
    const __nv_bfloat16* __restrict__ input,  // [rows, cols]
    int cols)
{
    const int row = blockIdx.y;
    const int blk = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float block_vals[16];
    __shared__ float smax;

    const int base = row * cols + blk * 16;
    float v = __bfloat162float(input[base + tid]);
    block_vals[tid] = v;
    float a = fabsf(v);
    // Warp reduce max-abs across 16 lanes (one warp).
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xFFFF, a, off, 16);
        a = fmaxf(a, o);
    }
    if (tid == 0) smax = a;
    __syncthreads();

    // Scale = max_abs / 6.0, encoded as UE4M3.
    uint8_t s_code;
    float s_decoded;
    if (tid == 0) {
        float s = smax * (1.0f / 6.0f);
        s_code = f32_to_ue4m3(s);
        // Decode so we use the actual (quantized) scale for element quantization.
        s_decoded = decode_ue4m3(s_code);
        if (s_decoded == 0.0f) s_decoded = 1.0f; // avoid /0 on an all-zero block
        scales[row * (cols / 16) + blk] = s_code;
        // Stash in shared mem so all threads can read.
        block_vals[0] = s_decoded;
    }
    __syncthreads();
    float inv_s = 1.0f / block_vals[0];

    // Quantize this thread's element to a 4-bit E2M1 code.
    float nv = __bfloat162float(input[base + tid]) * inv_s;
    unsigned mag = e2m1_mag_code(nv);
    unsigned sign = (nv < 0.0f) ? 0x8u : 0x0u;
    uint8_t code = (uint8_t)(sign | mag);

    // Pack two nibbles per byte: even index = low nibble, odd = high.
    // __shfl_sync is warp-convergent — all 16 lanes must participate — so
    // the shuffle runs unconditionally, then only even lanes write.
    unsigned neighbor = __shfl_sync(0xFFFF, (unsigned)code, tid ^ 1, 16);
    if ((tid & 1) == 0) {
        uint8_t byte = (uint8_t)(((neighbor & 0xF) << 4) | ((unsigned)code & 0xF));
        int byte_idx = (blk * 16 + tid) / 2;
        packed[row * (cols / 2) + byte_idx] = byte;
    }
}

extern "C" int xk_nvfp4_quantize_bf16(
    void* packed,
    void* scales,
    const void* input,
    int rows,
    int cols,
    void* stream)
{
    if (rows <= 0 || cols <= 0) return 0;
    if (cols % 16 != 0) return -(int)cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(cols / 16, rows);
    dim3 block(16);
    xk_nvfp4_quantize_bf16_kernel<<<grid, block, 0, s>>>(
        (uint8_t*)packed,
        (uint8_t*)scales,
        (const __nv_bfloat16*)input,
        cols);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -(int)e;
    return 0;
}
