// GELU-tanh activation, as used by Gemma's MLP (hidden_activation =
// "gelu_pytorch_tanh"):
//   gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
//
// The _glu variant fuses the element-wise product with a second tensor (Gemma
// MLP is gate/up/down with `act(gate) * up` in the middle), saving one launch
// and one round-trip through HBM for the intermediate.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__device__ __forceinline__ float xk_gelu_tanh_f32(float x) {
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    const float inner = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void xk_gelu_tanh_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float x = __bfloat162float(in[tid]);
    out[tid] = __float2bfloat16_rn(xk_gelu_tanh_f32(x));
}

__global__ void xk_gelu_tanh_glu_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int n)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const float g = __bfloat162float(gate[tid]);
    const float u = __bfloat162float(up[tid]);
    out[tid] = __float2bfloat16_rn(xk_gelu_tanh_f32(g) * u);
}

extern "C" int xk_gelu_tanh_bf16(void* out, const void* in, int n, void* stream) {
    if (n <= 0) return 0;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    cudaStream_t s = (cudaStream_t)stream;
    xk_gelu_tanh_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)out, (const __nv_bfloat16*)in, n);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : -(int)e;
}

extern "C" int xk_gelu_tanh_glu_bf16(
    void* out, const void* gate, const void* up, int n, void* stream)
{
    if (n <= 0) return 0;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    cudaStream_t s = (cudaStream_t)stream;
    xk_gelu_tanh_glu_bf16_kernel<<<grid, block, 0, s>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)gate,
        (const __nv_bfloat16*)up,
        n);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : -(int)e;
}
