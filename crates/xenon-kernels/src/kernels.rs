//! Rust wrappers around the C-ABI kernel entry points.

use std::ffi::c_void;

use half::bf16;

use crate::cuda::{CudaError, DeviceBuffer, Stream};

unsafe extern "C" {
    fn xk_hello(n: u32, result: *mut u32) -> i32;
    fn xk_rmsnorm_bf16(
        out: *mut c_void,
        x: *const c_void,
        weight: *const c_void,
        rows: i32,
        hidden: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Runs the hello kernel. Proves the Rust -> nvcc -> CUDA runtime link works.
pub fn hello(n: u32) -> Result<u32, CudaError> {
    let mut out: u32 = 0;
    let code = unsafe { xk_hello(n, &mut out as *mut u32) };
    if code == 0 {
        Ok(out)
    } else {
        Err(CudaError(-code))
    }
}

/// RMSNorm forward on bf16 tensors. Input and output are row-major `[rows, hidden]`.
/// `weight` is a length-`hidden` gain vector. All buffers live on the device.
///
/// Launches on `stream` if provided, otherwise on the default stream. The
/// caller is responsible for synchronizing.
pub fn rmsnorm_bf16(
    out: &mut DeviceBuffer<bf16>,
    x: &DeviceBuffer<bf16>,
    weight: &DeviceBuffer<bf16>,
    rows: usize,
    hidden: usize,
    eps: f32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), rows * hidden, "rmsnorm: out length");
    assert_eq!(x.len(), rows * hidden, "rmsnorm: x length");
    assert_eq!(weight.len(), hidden, "rmsnorm: weight length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_rmsnorm_bf16(
            out.as_device_ptr(),
            x.as_device_ptr(),
            weight.as_device_ptr(),
            rows as i32,
            hidden as i32,
            eps,
            stream_ptr,
        )
    };
    if code == 0 {
        Ok(())
    } else {
        Err(CudaError(-code))
    }
}

/// Reference implementation for correctness checks. Runs on host in fp32.
pub fn rmsnorm_bf16_reference(
    x: &[bf16],
    weight: &[bf16],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<bf16> {
    assert_eq!(x.len(), rows * hidden);
    assert_eq!(weight.len(), hidden);
    let mut out = vec![bf16::ZERO; rows * hidden];
    for r in 0..rows {
        let row_in = &x[r * hidden..(r + 1) * hidden];
        let mean_sq: f32 = row_in.iter().map(|v| {
            let f = v.to_f32();
            f * f
        }).sum::<f32>() / hidden as f32;
        let scale = (mean_sq + eps).sqrt().recip();
        for i in 0..hidden {
            let v = row_in[i].to_f32();
            let w = weight[i].to_f32();
            out[r * hidden + i] = bf16::from_f32(v * scale * w);
        }
    }
    out
}
