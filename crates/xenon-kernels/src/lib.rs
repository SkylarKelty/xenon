//! xenon-kernels: hand-rolled CUDA kernels compiled by `build.rs` and linked
//! as a C ABI. Phase 0 contains only a sanity kernel; phase 1 adds RMSNorm,
//! GELU, RoPE, and FMHA.

unsafe extern "C" {
    fn xk_hello(n: u32, result: *mut u32) -> i32;
}

/// Errors from the C wrapper around a CUDA kernel. The inner code is the
/// negated `cudaError_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelError(pub i32);

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error (code {})", -self.0)
    }
}

impl std::error::Error for KernelError {}

/// Runs the hello kernel: returns `n * 2 + 1` computed on the GPU. Proves the
/// nvcc -> static lib -> CUDA runtime link path is working.
pub fn hello(n: u32) -> Result<u32, KernelError> {
    let mut out: u32 = 0;
    let code = unsafe { xk_hello(n, &mut out as *mut u32) };
    if code == 0 {
        Ok(out)
    } else {
        Err(KernelError(code))
    }
}
