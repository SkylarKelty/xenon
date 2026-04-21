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
    fn xk_fp4_dequant_bf16(
        out: *mut c_void,
        packed: *const c_void,
        scales: *const c_void,
        global_scale: f32,
        rows: i32,
        cols: i32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_gelu_tanh_bf16(
        out: *mut c_void,
        input: *const c_void,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_gelu_tanh_glu_bf16(
        out: *mut c_void,
        gate: *const c_void,
        up: *const c_void,
        n: i32,
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
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

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

/// NVFP4 dequant to bf16. `packed` is `[rows, cols/2]` U8; `scales` is
/// `[rows, cols/16]` UE4M3; `global_scale` is the per-tensor fp32 multiplier.
pub fn fp4_dequant_bf16(
    out: &mut DeviceBuffer<bf16>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<u8>,
    global_scale: f32,
    rows: usize,
    cols: usize,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), rows * cols);
    assert_eq!(packed.len(), rows * cols / 2);
    assert_eq!(scales.len(), rows * cols / 16);
    assert!(cols % 16 == 0, "cols must be a multiple of 16");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_fp4_dequant_bf16(
            out.as_device_ptr(),
            packed.as_device_ptr(),
            scales.as_device_ptr(),
            global_scale,
            rows as i32,
            cols as i32,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode a UE4M3 (FP8 E4M3 forced non-negative) byte to f32.
pub fn ue4m3_to_f32(b: u8) -> f32 {
    let x = (b & 0x7F) as u32;
    let exp = (x >> 3) & 0xF;
    let man = x & 0x7;
    if exp == 0 {
        // Subnormal: 2^-6 * (man / 8).
        (man as f32) * (1.0 / 512.0)
    } else {
        // Normal: (8 + man) * 2^(exp - 10).
        ((8 + man) as f32) * (2.0f32).powi(exp as i32 - 10)
    }
}

/// Reference host-side NVFP4 dequant matching the kernel's math bit-for-bit.
pub fn fp4_dequant_bf16_reference(
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    rows: usize,
    cols: usize,
) -> Vec<bf16> {
    assert_eq!(packed.len(), rows * cols / 2);
    assert_eq!(scales.len(), rows * cols / 16);
    let mut out = vec![bf16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let byte = packed[r * (cols / 2) + (c >> 1)];
            let code = if c & 1 == 0 { byte & 0xF } else { byte >> 4 };
            let fp4 = FP4_E2M1_LUT[code as usize];
            let sb = scales[r * (cols / 16) + (c >> 4)];
            let bs = ue4m3_to_f32(sb);
            out[r * cols + c] = bf16::from_f32(fp4 * bs * global_scale);
        }
    }
    out
}

/// GELU-tanh: `y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu_tanh_bf16(
    out: &mut DeviceBuffer<bf16>,
    input: &DeviceBuffer<bf16>,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), input.len());
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_gelu_tanh_bf16(
            out.as_device_ptr(),
            input.as_device_ptr(),
            input.len() as i32,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Fused gated activation: `out[i] = gelu_tanh(gate[i]) * up[i]`.
pub fn gelu_tanh_glu_bf16(
    out: &mut DeviceBuffer<bf16>,
    gate: &DeviceBuffer<bf16>,
    up: &DeviceBuffer<bf16>,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), gate.len());
    assert_eq!(out.len(), up.len());
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_gelu_tanh_glu_bf16(
            out.as_device_ptr(),
            gate.as_device_ptr(),
            up.as_device_ptr(),
            out.len() as i32,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

pub fn gelu_tanh_bf16_reference(input: &[bf16]) -> Vec<bf16> {
    const K0: f32 = 0.7978845608028654;
    const K1: f32 = 0.044715;
    input
        .iter()
        .map(|v| {
            let x = v.to_f32();
            let inner = K0 * (x + K1 * x * x * x);
            bf16::from_f32(0.5 * x * (1.0 + inner.tanh()))
        })
        .collect()
}
