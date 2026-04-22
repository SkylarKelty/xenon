//! xenon-kernels: CUDA device abstraction + hand-rolled kernels + cuBLASLt glue.
//!
//! Kernels in `src/cu/` are compiled by `build.rs` and exposed as a C ABI.
//! - `cuda`    — runtime FFI and safe wrappers (Device, Stream, DeviceBuffer).
//! - `kernels` — Rust wrappers around our `.cu` entry points.
//! - `cublas`  — cuBLASLt FFI + bf16 matmul wrapper.

pub mod cublas;
pub mod cuda;
pub mod kernels;
pub mod kv_cache;

pub use cublas::{
    linear_bf16_reference, matmul_bf16_reference, CublasError, CublasLt, GemmError,
};
pub use cuda::{device_synchronize, mem_info, CudaError, Device, DeviceBuffer, Stream};
pub use kv_cache::{KvCache, SlotSpec};
pub use kernels::{
    attn_naive_bf16, attn_naive_bf16_reference, embed_gather_bf16, fp4_dequant_bf16,
    fp4_dequant_bf16_reference, gelu_tanh_bf16, gelu_tanh_bf16_reference, gelu_tanh_glu_bf16,
    hello, rmsnorm_bf16, rmsnorm_bf16_reference, rope_bf16, rope_bf16_reference,
    softmax_attn_bf16, softmax_attn_bf16_reference, ue4m3_to_f32,
};
