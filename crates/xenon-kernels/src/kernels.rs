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
    fn xk_rope_bf16(
        x: *mut c_void,
        positions: *const c_void,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_softmax_attn_bf16(
        scores: *mut c_void,
        rows: i32,
        t_q: i32,
        t_kv: i32,
        scale: f32,
        q_pos_base: i32,
        window: i32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_embed_gather_bf16(
        out: *mut c_void,
        table: *const c_void,
        ids: *const c_void,
        tokens: i32,
        vocab: i32,
        hidden: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_attn_naive_bf16(
        out: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        t_q: i32,
        t_kv: i32,
        h: i32,
        h_kv: i32,
        d: i32,
        scale: f32,
        q_pos_base: i32,
        window: i32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_add_scale_bf16(
        out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        n: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_scale_bf16(
        out: *mut c_void,
        input: *const c_void,
        n: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_per_layer_slice_bf16(
        out: *mut c_void,
        src: *const c_void,
        tokens: i32,
        num_layers: i32,
        per_layer_dim: i32,
        layer_idx: i32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_softcap_bf16(
        out: *mut c_void,
        input: *const c_void,
        n: i32,
        cap: f32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_attn_flash_bf16(
        out: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        t_q: i32,
        t_kv: i32,
        h: i32,
        h_kv: i32,
        d: i32,
        scale: f32,
        q_pos_base: i32,
        window: i32,
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
/// Pass `weight = None` for the no-scale variant (`with_scale=False` in HF) —
/// the kernel uses weight 1.0 and produces pure RMS-normalized output.
pub fn rmsnorm_bf16(
    out: &mut DeviceBuffer<bf16>,
    x: &DeviceBuffer<bf16>,
    weight: Option<&DeviceBuffer<bf16>>,
    rows: usize,
    hidden: usize,
    eps: f32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), rows * hidden, "rmsnorm: out length");
    assert_eq!(x.len(), rows * hidden, "rmsnorm: x length");
    if let Some(w) = weight {
        assert_eq!(w.len(), hidden, "rmsnorm: weight length");
    }
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let weight_ptr = weight.map(|w| w.as_device_ptr() as *const std::ffi::c_void)
        .unwrap_or(std::ptr::null());
    let code = unsafe {
        xk_rmsnorm_bf16(
            out.as_device_ptr(),
            x.as_device_ptr(),
            weight_ptr,
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
    weight: Option<&[bf16]>,
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<bf16> {
    assert_eq!(x.len(), rows * hidden);
    if let Some(w) = weight {
        assert_eq!(w.len(), hidden);
    }
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
            let w = weight.map(|w| w[i].to_f32()).unwrap_or(1.0);
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

/// RoPE (rotate_half convention) in-place on `[tokens, heads, head_dim]`.
/// Only the first `rotary_dim` dims rotate (partial-rotary layers pass the
/// tail through unchanged). `positions` is `[tokens]` of i32.
pub fn rope_bf16(
    x: &mut DeviceBuffer<bf16>,
    positions: &DeviceBuffer<i32>,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(x.len(), tokens * heads * head_dim, "rope: x length");
    assert_eq!(positions.len(), tokens, "rope: positions length");
    assert!(rotary_dim <= head_dim, "rope: rotary_dim > head_dim");
    assert!(rotary_dim % 2 == 0, "rope: rotary_dim must be even");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_rope_bf16(
            x.as_device_ptr(),
            positions.as_device_ptr(),
            tokens as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            theta,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Host reference for RoPE matching the kernel's math. For partial rotary,
/// pairs `(i, i + head_dim/2)` are rotated for `i in 0..rotary_dim/2`; the
/// rest of the head_dim dimensions pass through.
pub fn rope_bf16_reference(
    x: &[bf16],
    positions: &[i32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
) -> Vec<bf16> {
    assert_eq!(x.len(), tokens * heads * head_dim);
    assert_eq!(positions.len(), tokens);
    assert!(head_dim % 2 == 0);
    assert!(rotary_dim % 2 == 0);
    assert!(rotary_dim <= head_dim);
    let mut out = x.to_vec();
    let head_half = head_dim / 2;
    let rotary_pairs = rotary_dim / 2;
    for t in 0..tokens {
        let pos = positions[t] as f32;
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for i in 0..rotary_pairs {
                let inv_freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = pos * inv_freq;
                let (s, c) = angle.sin_cos();
                let x0 = x[base + i].to_f32();
                let x1 = x[base + i + head_half].to_f32();
                out[base + i] = bf16::from_f32(x0 * c - x1 * s);
                out[base + i + head_half] = bf16::from_f32(x1 * c + x0 * s);
            }
        }
    }
    out
}

/// Attention softmax with causal + optional sliding-window mask, in-place.
/// `scores` is `[rows, t_kv]` where `rows = batch*heads*t_q`. For row `r`,
/// the query position is `q_pos_base + (r % t_q)`. `window == 0` = no window.
pub fn softmax_attn_bf16(
    scores: &mut DeviceBuffer<bf16>,
    rows: usize,
    t_q: usize,
    t_kv: usize,
    scale: f32,
    q_pos_base: i32,
    window: i32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(scores.len(), rows * t_kv, "softmax: scores length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_softmax_attn_bf16(
            scores.as_device_ptr(),
            rows as i32,
            t_q as i32,
            t_kv as i32,
            scale,
            q_pos_base,
            window,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Host reference for masked attention softmax.
pub fn softmax_attn_bf16_reference(
    scores: &[bf16],
    rows: usize,
    t_q: usize,
    t_kv: usize,
    scale: f32,
    q_pos_base: i32,
    window: i32,
) -> Vec<bf16> {
    assert_eq!(scores.len(), rows * t_kv);
    let mut out = vec![bf16::ZERO; rows * t_kv];
    for r in 0..rows {
        let q_local = (r % t_q) as i32;
        let q_pos = q_pos_base + q_local;
        let kv_min = if window > 0 { (q_pos - window + 1).max(0) } else { 0 };
        let kv_max = q_pos;
        let row = &scores[r * t_kv..(r + 1) * t_kv];
        let mut m = f32::NEG_INFINITY;
        for j in 0..t_kv {
            let jj = j as i32;
            if jj >= kv_min && jj <= kv_max {
                let v = row[j].to_f32() * scale;
                if v > m {
                    m = v;
                }
            }
        }
        let mut sum = 0.0f32;
        let mut buf = vec![0.0f32; t_kv];
        for j in 0..t_kv {
            let jj = j as i32;
            if jj >= kv_min && jj <= kv_max {
                let v = row[j].to_f32() * scale;
                let e = (v - m).exp();
                buf[j] = e;
                sum += e;
            }
        }
        let inv = 1.0 / sum;
        for j in 0..t_kv {
            let jj = j as i32;
            let v = if jj >= kv_min && jj <= kv_max { buf[j] * inv } else { 0.0 };
            out[r * t_kv + j] = bf16::from_f32(v);
        }
    }
    out
}

/// Naive multi-head attention (GQA-aware) with causal + optional sliding-window
/// mask. All tensors row-major bf16.
///
/// Shapes:
/// - `q`: `[t_q, h, d]`
/// - `k`: `[t_kv, h_kv, d]`
/// - `v`: `[t_kv, h_kv, d]`
/// - `out`: `[t_q, h, d]`
///
/// `h` must be a multiple of `h_kv`. `scale` is usually `1/sqrt(d)`.
#[allow(clippy::too_many_arguments)]
pub fn attn_naive_bf16(
    out: &mut DeviceBuffer<bf16>,
    q: &DeviceBuffer<bf16>,
    k: &DeviceBuffer<bf16>,
    v: &DeviceBuffer<bf16>,
    t_q: usize,
    t_kv: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    scale: f32,
    q_pos_base: i32,
    window: i32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(q.len(), t_q * h * d, "attn: q length");
    // K/V can be KvCache slots sized [max_len, h_kv, d] where max_len >= t_kv;
    // the kernel only reads the first t_kv rows.
    assert!(k.len() >= t_kv * h_kv * d, "attn: k length");
    assert!(v.len() >= t_kv * h_kv * d, "attn: v length");
    assert_eq!(out.len(), t_q * h * d, "attn: out length");
    assert!(h % h_kv == 0, "attn: h must be divisible by h_kv");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_attn_naive_bf16(
            out.as_device_ptr(),
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            t_q as i32,
            t_kv as i32,
            h as i32,
            h_kv as i32,
            d as i32,
            scale,
            q_pos_base,
            window,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Element-wise `out[i] = (a[i] + b[i]) * scale`, bf16 I/O with fp32 ops.
pub fn add_scale_bf16(
    out: &mut DeviceBuffer<bf16>,
    a: &DeviceBuffer<bf16>,
    b: &DeviceBuffer<bf16>,
    scale: f32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), a.len(), "add_scale: out/a length");
    assert_eq!(out.len(), b.len(), "add_scale: out/b length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_add_scale_bf16(
            out.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            out.len() as i32,
            scale,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Element-wise `out[i] = in[i] * scale`, bf16 I/O. In-place safe.
pub fn scale_bf16(
    out: &mut DeviceBuffer<bf16>,
    input: &DeviceBuffer<bf16>,
    scale: f32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), input.len(), "scale: out/in length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_scale_bf16(
            out.as_device_ptr(),
            input.as_device_ptr(),
            out.len() as i32,
            scale,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Extract layer `layer_idx`'s slice from a `[tokens, num_layers,
/// per_layer_dim]` bf16 tensor into a contiguous `[tokens, per_layer_dim]`
/// output. Used to split the PLE combined tensor for per-layer consumption.
pub fn per_layer_slice_bf16(
    out: &mut DeviceBuffer<bf16>,
    src: &DeviceBuffer<bf16>,
    tokens: usize,
    num_layers: usize,
    per_layer_dim: usize,
    layer_idx: usize,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), tokens * per_layer_dim, "slice: out length");
    assert_eq!(src.len(), tokens * num_layers * per_layer_dim, "slice: src length");
    assert!(layer_idx < num_layers, "slice: layer_idx out of range");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_per_layer_slice_bf16(
            out.as_device_ptr(),
            src.as_device_ptr(),
            tokens as i32,
            num_layers as i32,
            per_layer_dim as i32,
            layer_idx as i32,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Final-logit softcap: `out[i] = tanh(in[i] / cap) * cap`. Gemma uses
/// `cap = 30.0` via `final_logit_softcapping`.
pub fn softcap_bf16(
    out: &mut DeviceBuffer<bf16>,
    input: &DeviceBuffer<bf16>,
    cap: f32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), input.len(), "softcap: out/in length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_softcap_bf16(
            out.as_device_ptr(),
            input.as_device_ptr(),
            out.len() as i32,
            cap,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// FlashAttention-2 style tiled attention. Same signature and math as
/// `attn_naive_bf16`, but uses online softmax so shared memory is O(BR * D)
/// regardless of T_kv — handles arbitrary context length.
///
/// Requires `d % 128 == 0` (the block size is 128).
#[allow(clippy::too_many_arguments)]
pub fn attn_flash_bf16(
    out: &mut DeviceBuffer<bf16>,
    q: &DeviceBuffer<bf16>,
    k: &DeviceBuffer<bf16>,
    v: &DeviceBuffer<bf16>,
    t_q: usize,
    t_kv: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    scale: f32,
    q_pos_base: i32,
    window: i32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(q.len(), t_q * h * d, "attn_flash: q length");
    // Same "K/V may be a larger KvCache slot" exception as attn_naive.
    assert!(k.len() >= t_kv * h_kv * d, "attn_flash: k length");
    assert!(v.len() >= t_kv * h_kv * d, "attn_flash: v length");
    assert_eq!(out.len(), t_q * h * d, "attn_flash: out length");
    assert!(h % h_kv == 0);
    assert!(d % 128 == 0, "attn_flash: d must be a multiple of 128");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_attn_flash_bf16(
            out.as_device_ptr(),
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            t_q as i32, t_kv as i32,
            h as i32, h_kv as i32, d as i32,
            scale, q_pos_base, window, stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Host reference for naive attention. Same math as the kernel.
#[allow(clippy::too_many_arguments)]
pub fn attn_naive_bf16_reference(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    t_q: usize,
    t_kv: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    scale: f32,
    q_pos_base: i32,
    window: i32,
) -> Vec<bf16> {
    assert_eq!(q.len(), t_q * h * d);
    assert_eq!(k.len(), t_kv * h_kv * d);
    assert_eq!(v.len(), t_kv * h_kv * d);
    assert!(h % h_kv == 0);
    let gqa = h / h_kv;
    let mut out = vec![bf16::ZERO; t_q * h * d];
    for q_tok in 0..t_q {
        let q_pos = q_pos_base + q_tok as i32;
        let kv_min = if window > 0 { (q_pos - window + 1).max(0) } else { 0 };
        let kv_max = q_pos;
        for q_head in 0..h {
            let kv_head = q_head / gqa;
            let q_base = (q_tok * h + q_head) * d;
            let mut scores = vec![f32::NEG_INFINITY; t_kv];
            let mut max_v = f32::NEG_INFINITY;
            for j in 0..t_kv {
                if (j as i32) < kv_min || (j as i32) > kv_max {
                    continue;
                }
                let k_base = (j * h_kv + kv_head) * d;
                let mut acc = 0.0f32;
                for i in 0..d {
                    acc += q[q_base + i].to_f32() * k[k_base + i].to_f32();
                }
                let s = acc * scale;
                scores[j] = s;
                if s > max_v {
                    max_v = s;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..t_kv {
                if (j as i32) < kv_min || (j as i32) > kv_max {
                    continue;
                }
                let e = (scores[j] - max_v).exp();
                scores[j] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            let out_base = (q_tok * h + q_head) * d;
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..t_kv {
                    if (j as i32) < kv_min || (j as i32) > kv_max {
                        continue;
                    }
                    acc += scores[j] * v[(j * h_kv + kv_head) * d + dd].to_f32();
                }
                out[out_base + dd] = bf16::from_f32(acc * inv);
            }
        }
    }
    out
}

/// Gather embeddings: `out[t] = table[ids[t]] * scale` across the hidden dim.
/// Pass `scale = 1.0` for a plain gather; Gemma uses `sqrt(hidden_size)` for
/// `embed_tokens` and `sqrt(hidden_size_per_layer_input)` for the per-layer
/// embedding table.
#[allow(clippy::too_many_arguments)]
pub fn embed_gather_bf16(
    out: &mut DeviceBuffer<bf16>,
    table: &DeviceBuffer<bf16>,
    ids: &DeviceBuffer<i32>,
    tokens: usize,
    vocab: usize,
    hidden: usize,
    scale: f32,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(out.len(), tokens * hidden, "embed: out length");
    assert_eq!(table.len(), vocab * hidden, "embed: table length");
    assert_eq!(ids.len(), tokens, "embed: ids length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_embed_gather_bf16(
            out.as_device_ptr(),
            table.as_device_ptr(),
            ids.as_device_ptr(),
            tokens as i32,
            vocab as i32,
            hidden as i32,
            scale,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}
