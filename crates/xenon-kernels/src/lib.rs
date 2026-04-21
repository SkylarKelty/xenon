//! xenon-kernels: CUDA device abstraction + hand-rolled kernels.
//!
//! Kernels are compiled by `build.rs` via `nvcc` into a static lib that the
//! Rust side links against. The `cuda` module provides the minimal runtime
//! FFI (device/stream/allocations); `kernels` provides safe wrappers around
//! each `.cu` entry point.

pub mod cuda;
pub mod kernels;

pub use cuda::{device_synchronize, mem_info, CudaError, Device, DeviceBuffer, Stream};
pub use kernels::{hello, rmsnorm_bf16, rmsnorm_bf16_reference};
