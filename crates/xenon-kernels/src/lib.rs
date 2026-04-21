//! xenon-kernels: hand-rolled CUDA kernels, compiled by `build.rs` and exposed
//! to Rust via a C ABI. Skeleton — kernels land in phase 1 (RMSNorm, GELU,
//! RoPE) and phase 2 (FMHA for sliding + full attention, dual head_dim).
