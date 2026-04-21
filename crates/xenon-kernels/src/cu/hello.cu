// Sanity kernel: validates the Rust -> nvcc -> static-lib -> CUDA runtime
// pipeline end-to-end. Real inference kernels (RMSNorm, GELU, RoPE, FMHA)
// land in phase 1+.

#include <cuda_runtime.h>
#include <cstdint>

__global__ void xk_hello_kernel(uint32_t* out, uint32_t n) {
    *out = n * 2u + 1u;
}

// Returns 0 on success; negative cudaError_t on failure. `result` is only
// written on success.
extern "C" int xk_hello(uint32_t n, uint32_t* result) {
    uint32_t* d = nullptr;
    cudaError_t e = cudaMalloc((void**)&d, sizeof(uint32_t));
    if (e != cudaSuccess) return -(int)e;

    xk_hello_kernel<<<1, 1>>>(d, n);
    e = cudaGetLastError();
    if (e != cudaSuccess) { cudaFree(d); return -(int)e; }

    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { cudaFree(d); return -(int)e; }

    uint32_t h = 0;
    e = cudaMemcpy(&h, d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d);
    if (e != cudaSuccess) return -(int)e;

    *result = h;
    return 0;
}
