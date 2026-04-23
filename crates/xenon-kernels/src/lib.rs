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
pub use cuda::{device_synchronize, mem_info, CudaError, Device, DeviceBuffer, PinnedBuffer, Stream};
pub use kv_cache::{KvCache, SlotSpec};
pub use kernels::{
    add_scale_bf16, attn_flash_bf16, attn_naive_bf16, attn_naive_bf16_reference,
    attn_flash_tc_bf16, attn_split_kv_auto_chunk_size, attn_split_kv_bf16,
    attn_split_kv_bf16_device,
    test_mma_bf16, embed_gather_bf16, fp4_dequant_bf16,
    fp4_dequant_bf16_reference, fp4_gemv_bf16, gelu_tanh_bf16, gelu_tanh_bf16_reference,
    gelu_tanh_glu_bf16, hello, inc_i32_device, kv_append_bf16, nvfp4_quantize_bf16,
    per_layer_slice_bf16, rmsnorm_bf16,
    rmsnorm_bf16_reference, rope_bf16, rope_bf16_reference, round_up,
    sample_topk_bf16, sample_topk_bf16_reference,
    scale_bf16, softcap_bf16,
    softmax_attn_bf16, softmax_attn_bf16_reference, swizzle_blockscale_ue4m3, ue4m3_to_f32,
};
