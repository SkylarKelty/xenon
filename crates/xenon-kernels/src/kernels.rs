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
    fn xk_fp4_gemv_bf16(
        y: *mut c_void,
        x: *const c_void,
        w_packed: *const c_void,
        w_scales: *const c_void,
        global_scale: f32,
        n: i32,
        k: i32,
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
    fn xk_nvfp4_quantize_bf16(
        packed: *mut c_void,
        scales: *mut c_void,
        input: *const c_void,
        rows: i32,
        cols: i32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_swizzle_blockscale(
        dst: *mut c_void,
        src: *const c_void,
        m: i32,
        k_blocks: i32,
        m_padded: i32,
        k_padded: i32,
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
    fn xk_attn_split_kv_bf16(
        out: *mut c_void,
        partial_max: *mut c_void,
        partial_sum: *mut c_void,
        partial_num: *mut c_void,
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
        chunk_size: i32,
        stream: *mut c_void,
    ) -> i32;
    fn xk_test_mma_bf16(
        d: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    fn xk_attn_flash_tc_bf16(
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
    fn xk_sample_topk_bf16(
        out_probs: *mut c_void,
        out_ids: *mut c_void,
        logits: *const c_void,
        scratch: *mut c_void,
        vocab: i32,
        temperature: f32,
        top_k: i32,
        greedy: i32,
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
    // Buffers may be oversized (e.g. persistent scratch that serves multiple
    // shapes in the same decode step). The kernel reads/writes exactly
    // rows*hidden elements, so `>=` is sufficient. Weight stays exact — it's
    // a model parameter with a specific shape.
    assert!(out.len() >= rows * hidden, "rmsnorm: out length");
    assert!(x.len() >= rows * hidden, "rmsnorm: x length");
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
    // `out` may be oversized scratch shared across weights; packed/scales
    // are always the weight's own exact-sized buffers.
    assert!(out.len() >= rows * cols, "fp4_dequant: out length");
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

/// Fused FP4×bf16 gemv (M=1). `y = x @ W^T` where the FP4 weight is read
/// directly — no intermediate bf16 materialization. Decode fast path; for
/// M>1 the existing `fp4_dequant_bf16` + `linear_bf16` path is faster.
///
/// `w_packed` is `[n, k/2]` U8, `w_scales` is `[n, k/16]` UE4M3,
/// `global_scale` is the per-tensor f32 multiplier. `x` is `[k]` bf16,
/// `y` is `[n]` bf16. `k` must be a multiple of 16.
pub fn fp4_gemv_bf16(
    y: &mut DeviceBuffer<bf16>,
    x: &DeviceBuffer<bf16>,
    w_packed: &DeviceBuffer<u8>,
    w_scales: &DeviceBuffer<u8>,
    global_scale: f32,
    n: usize,
    k: usize,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert!(k % 16 == 0, "fp4_gemv: k must be a multiple of 16");
    assert!(y.len() >= n, "fp4_gemv: y length");
    assert!(x.len() >= k, "fp4_gemv: x length");
    assert_eq!(w_packed.len(), n * k / 2, "fp4_gemv: w_packed length");
    assert_eq!(w_scales.len(), n * k / 16, "fp4_gemv: w_scales length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_fp4_gemv_bf16(
            y.as_device_ptr(),
            x.as_device_ptr(),
            w_packed.as_device_ptr(),
            w_scales.as_device_ptr(),
            global_scale,
            n as i32,
            k as i32,
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
    // `x` may be persistent scratch sized for the largest head_dim variant
    // (e.g. 4096 for full-attn) while this call uses a smaller head_dim
    // (e.g. sliding's 256 → 2048). The kernel reads/writes exactly
    // tokens*heads*head_dim elements.
    assert!(x.len() >= tokens * heads * head_dim, "rope: x length");
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
    // q/out may be oversized persistent scratch sized for the largest
    // head_dim; the kernel reads/writes exactly t_q*h*d elements.
    assert!(q.len() >= t_q * h * d, "attn: q length");
    // K/V can be KvCache slots sized [max_len, h_kv, d] where max_len >= t_kv;
    // the kernel only reads the first t_kv rows.
    assert!(k.len() >= t_kv * h_kv * d, "attn: k length");
    assert!(v.len() >= t_kv * h_kv * d, "attn: v length");
    assert!(out.len() >= t_q * h * d, "attn: out length");
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

/// Block-scaled bf16 → NVFP4 quantization for activations.
///
/// Input: bf16 `[rows, cols]`. Output: packed FP4 `[rows, cols/2]` u8 and
/// UE4M3 block scales `[rows, cols/16]` u8. `cols` must be a multiple of 16.
///
/// Per 16-element block along the inner dim: max_abs → UE4M3 scale (= max/6),
/// elements quantized to nearest E2M1 via that scale. Output matches the
/// storage format of our weight tensors, so the cuBLASLt NVFP4 GEMM sees
/// both operands in the same VEC16_UE4M3 layout.
pub fn nvfp4_quantize_bf16(
    packed: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>,
    input: &DeviceBuffer<bf16>,
    rows: usize,
    cols: usize,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(packed.len(), rows * (cols / 2), "quant: packed length");
    assert_eq!(scales.len(), rows * (cols / 16), "quant: scales length");
    assert_eq!(input.len(), rows * cols, "quant: input length");
    assert!(cols % 16 == 0);
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_nvfp4_quantize_bf16(
            packed.as_device_ptr(),
            scales.as_device_ptr(),
            input.as_device_ptr(),
            rows as i32,
            cols as i32,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Smallest multiple of `align` that is >= `n`.
#[inline]
pub fn round_up(n: usize, align: usize) -> usize {
    ((n + align - 1) / align) * align
}

/// Swizzle UE4M3 per-16 block scales from row-major `[m, k_blocks]` into
/// cuBLASLt's 128×4-interleaved layout (the one `VEC16_UE4M3` mode expects).
/// `dst` must have `round_up(m, 128) * round_up(k_blocks, 4)` bytes.
///
/// Used for both weight scales (swizzled once at load) and activation
/// scales (swizzled each time `nvfp4_quantize_bf16` runs, which is per
/// prefill step in the M≥128 dispatch).
pub fn swizzle_blockscale_ue4m3(
    dst: &mut DeviceBuffer<u8>,
    src: &DeviceBuffer<u8>,
    m: usize,
    k_blocks: usize,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    let m_padded = round_up(m, 128);
    let k_padded = round_up(k_blocks, 4);
    assert!(src.len() >= m * k_blocks, "swizzle: src length");
    assert!(dst.len() >= m_padded * k_padded, "swizzle: dst length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_swizzle_blockscale(
            dst.as_device_ptr(),
            src.as_device_ptr(),
            m as i32,
            k_blocks as i32,
            m_padded as i32,
            k_padded as i32,
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

/// Pick a chunk_size for split-KV attention that aims to fill the GPU with
/// blocks. For decode shapes (T_q * H small) this returns a smaller chunk_size
/// so more blocks launch; for prefill (T_q * H already saturates), returns
/// T_kv so n_chunks collapses to 1 and the merge-kernel cost is negligible.
///
/// `sm_count` is the device SM count (e.g. 26 on RTX PRO 2000 Blackwell).
/// The min chunk size (32) avoids per-block overhead swamping useful work.
pub fn attn_split_kv_auto_chunk_size(
    t_q: usize,
    t_kv: usize,
    h: usize,
    sm_count: usize,
) -> usize {
    const MIN_CHUNK: usize = 32;
    let target_blocks = (sm_count * 2).max(1);
    let heads = (t_q * h).max(1);
    if heads >= target_blocks {
        // Already saturated without splitting.
        return t_kv.max(MIN_CHUNK);
    }
    let n_chunks_want = target_blocks.div_ceil(heads);
    let cs_from_target = t_kv.div_ceil(n_chunks_want);
    cs_from_target.max(MIN_CHUNK)
}

/// Split-KV attention: two-kernel decomposition that parallelises the T_kv
/// dimension across a third grid axis. Same math (and same answer within
/// fp32 round-off) as `attn_naive_bf16`, but at decode shapes (T_q=1, H=8)
/// it launches ~n_chunks× more blocks, saturating SMs that would otherwise
/// idle.
///
/// Caller supplies scratch for the per-chunk `(max, sum, numerator)` arrays;
/// sizes are `[T_q * H * n_chunks]` floats for max/sum and
/// `[T_q * H * n_chunks * D]` floats for numerator, where
/// `n_chunks = ceil(T_kv / chunk_size)`.
#[allow(clippy::too_many_arguments)]
pub fn attn_split_kv_bf16(
    out: &mut DeviceBuffer<bf16>,
    partial_max: &mut DeviceBuffer<f32>,
    partial_sum: &mut DeviceBuffer<f32>,
    partial_num: &mut DeviceBuffer<f32>,
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
    chunk_size: usize,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert!(q.len() >= t_q * h * d, "attn_split_kv: q length");
    assert!(k.len() >= t_kv * h_kv * d, "attn_split_kv: k length");
    assert!(v.len() >= t_kv * h_kv * d, "attn_split_kv: v length");
    assert!(out.len() >= t_q * h * d, "attn_split_kv: out length");
    assert!(h % h_kv == 0, "attn_split_kv: h divisible by h_kv");
    assert!(chunk_size > 0, "attn_split_kv: chunk_size > 0");
    let n_chunks = t_kv.div_ceil(chunk_size);
    let partials_needed = t_q * h * n_chunks;
    assert!(partial_max.len() >= partials_needed, "attn_split_kv: partial_max length");
    assert!(partial_sum.len() >= partials_needed, "attn_split_kv: partial_sum length");
    assert!(partial_num.len() >= partials_needed * d, "attn_split_kv: partial_num length");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_attn_split_kv_bf16(
            out.as_device_ptr(),
            partial_max.as_device_ptr(),
            partial_sum.as_device_ptr(),
            partial_num.as_device_ptr(),
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            t_q as i32, t_kv as i32,
            h as i32, h_kv as i32, d as i32,
            scale, q_pos_base, window,
            chunk_size as i32,
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Flash-attention-2 style prefill attention on tensor cores (mma.m16n8k16).
/// Same signature as `attn_naive_bf16`. BR=16, BC=16 tiling; requires
/// `D % 16 == 0`. Decode shapes (T_q=1) won't hit the tile minimum — use
/// `attn_split_kv_bf16` there. See `attn_flash_tc.cu` for the smem layout.
#[allow(clippy::too_many_arguments)]
pub fn attn_flash_tc_bf16(
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
    assert!(q.len() >= t_q * h * d, "attn_flash_tc: q length");
    assert!(k.len() >= t_kv * h_kv * d, "attn_flash_tc: k length");
    assert!(v.len() >= t_kv * h_kv * d, "attn_flash_tc: v length");
    assert!(out.len() >= t_q * h * d, "attn_flash_tc: out length");
    assert!(h % h_kv == 0, "attn_flash_tc: h divisible by h_kv");
    assert!(d % 16 == 0, "attn_flash_tc: d must be a multiple of 16 (mma k-dim)");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_attn_flash_tc_bf16(
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

/// Stage-1 validation for the tensor-core attention kernel. Runs a single
/// `mma.m16n8k16.row.col.f32.bf16.bf16.f32` on A[16,16] (row-major) and
/// B[16,8] (col-major, i.e. B[k,n] at offset n*16+k) into D[16,8] fp32.
/// Proves the PTX + sm_120a gencode combo works and that our fragment
/// gather maps correctly. Only one block × 32 threads.
pub fn test_mma_bf16(
    d: &mut DeviceBuffer<f32>,
    a: &DeviceBuffer<bf16>,
    b: &DeviceBuffer<bf16>,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert_eq!(a.len(), 16 * 16, "test_mma_bf16: A must be 16x16");
    assert_eq!(b.len(), 16 * 8, "test_mma_bf16: B must be 16x8");
    assert_eq!(d.len(), 16 * 8, "test_mma_bf16: D must be 16x8");
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_test_mma_bf16(
            d.as_device_ptr(),
            a.as_device_ptr(),
            b.as_device_ptr(),
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Temperature-scaled softmax + top-K extraction over a bf16 logit row.
///
/// Writes the K highest-probability `(prob, token_id)` pairs in descending-
/// prob order into `out_probs` (f32) and `out_ids` (u32). `out_probs[k]` is
/// the softmax probability of `out_ids[k]` normalized over the full vocab —
/// so the caller can apply top-P and renormalize on the compact K slice.
///
/// `greedy=true` is a dedicated fast path: argmax over bf16 logits, writes
/// just `out_probs[0] = 1.0` + `out_ids[0] = argmax`. No scratch needed;
/// equivalent to (and bit-identical with) the host `argmax_bf16`. Use this
/// when temperature <= 0 / default sampling.
///
/// Non-greedy requires `temperature > 0` and a `[vocab]` f32 scratch buffer
/// that the kernel mutates (scaled logits + iterative masking).
#[allow(clippy::too_many_arguments)]
pub fn sample_topk_bf16(
    out_probs: &mut DeviceBuffer<f32>,
    out_ids: &mut DeviceBuffer<u32>,
    logits: &DeviceBuffer<bf16>,
    scratch: Option<&mut DeviceBuffer<f32>>,
    vocab: usize,
    temperature: f32,
    top_k: usize,
    greedy: bool,
    stream: Option<&Stream>,
) -> Result<(), CudaError> {
    assert!(top_k >= 1, "sample_topk: top_k must be >= 1");
    assert!(out_probs.len() >= top_k, "sample_topk: out_probs length");
    assert!(out_ids.len() >= top_k, "sample_topk: out_ids length");
    assert!(logits.len() >= vocab, "sample_topk: logits length");
    if !greedy {
        assert!(temperature > 0.0, "sample_topk: temperature must be > 0 (non-greedy)");
        let s = scratch.as_ref().expect("sample_topk: scratch required when !greedy");
        assert!(s.len() >= vocab, "sample_topk: scratch length");
    }
    let stream_ptr = stream.map(|s| s.as_raw()).unwrap_or(std::ptr::null_mut());
    let scratch_ptr = scratch
        .map(|s| s.as_device_ptr())
        .unwrap_or(std::ptr::null_mut());
    let code = unsafe {
        xk_sample_topk_bf16(
            out_probs.as_device_ptr(),
            out_ids.as_device_ptr(),
            logits.as_device_ptr(),
            scratch_ptr,
            vocab as i32,
            temperature,
            top_k as i32,
            if greedy { 1 } else { 0 },
            stream_ptr,
        )
    };
    if code == 0 { Ok(()) } else { Err(CudaError(-code)) }
}

/// Host reference for `sample_topk_bf16`'s output: returns the top-K
/// `(prob, token_id)` pairs in descending-prob order, with probs being the
/// full-vocab softmax values. Used for kernel validation.
pub fn sample_topk_bf16_reference(
    logits: &[bf16],
    temperature: f32,
    top_k: usize,
    greedy: bool,
) -> (Vec<f32>, Vec<u32>) {
    if greedy {
        let mut best_i = 0usize;
        let mut best_v = logits[0].to_f32();
        for (i, v) in logits.iter().enumerate().skip(1) {
            let f = v.to_f32();
            if f > best_v { best_v = f; best_i = i; }
        }
        return (vec![1.0], vec![best_i as u32]);
    }
    let inv_t = 1.0 / temperature;
    let scaled: Vec<f32> = logits.iter().map(|v| v.to_f32() * inv_t).collect();
    let mut maxv = f32::NEG_INFINITY;
    for &s in &scaled { if s > maxv { maxv = s; } }
    let mut probs: Vec<f32> = scaled.iter().map(|s| (s - maxv).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() { *p *= inv_sum; }
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    // Sort desc by prob, tie-break by index asc (matches kernel).
    idx.sort_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let top: Vec<(f32, u32)> = idx.into_iter().take(top_k)
        .map(|i| (probs[i], i as u32)).collect();
    let out_probs = top.iter().map(|&(p, _)| p).collect();
    let out_ids = top.iter().map(|&(_, i)| i).collect();
    (out_probs, out_ids)
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
