//! Forward-pass orchestration for xenon.
//!
//! Loads Gemma 4 weights (NVFP4 packed resident, norms bf16), wires up the
//! decoder stack, and exposes a simple [`Engine`] type with `generate`
//! suitable for both CLI and the HTTP server.

#![allow(clippy::too_many_arguments)]

use std::path::Path;

use half::bf16;
use xenon_core::{GemmaConfig, LayerKind, MmapWeights, Tokenizer};
use xenon_kernels::{
    add_scale_bf16, attn_naive_bf16,
    cuda::{Device, DeviceBuffer, Stream},
    fp4_dequant_bf16, fp4_gemv_bf16, gelu_tanh_glu_bf16, per_layer_slice_bf16,
    rmsnorm_bf16, rope_bf16, scale_bf16, softcap_bf16, CublasLt, KvCache, SlotSpec,
};

// -------------------- Weight containers --------------------

/// NVFP4 linear weight resident on device in packed form. Dequant to bf16
/// scratch on demand (via [`QuantLinearDev::dequant_to`]).
pub struct QuantLinearDev {
    pub packed: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,
    pub global_scale: f32,
    pub out_features: usize,
    pub in_features: usize,
}

impl QuantLinearDev {
    pub fn load(q: &xenon_core::QuantLinearRef<'_>) -> anyhow::Result<Self> {
        let mut packed: DeviceBuffer<u8> = DeviceBuffer::new(q.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut scales: DeviceBuffer<u8> = DeviceBuffer::new(q.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        packed.copy_from_host_bytes(q.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
        scales.copy_from_host_bytes(q.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            packed, scales,
            global_scale: q.global_scale,
            out_features: q.out_features,
            in_features: q.in_features,
        })
    }

    pub fn dequant_to(&self, out: &mut DeviceBuffer<bf16>, stream: &Stream) -> anyhow::Result<()> {
        fp4_dequant_bf16(out, &self.packed, &self.scales, self.global_scale,
                          self.out_features, self.in_features, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// `y = x @ W^T` with FP4 weight. For `m == 1` this takes the fused
    /// FP4×bf16 gemv path (no intermediate bf16 weight); for `m > 1` it
    /// dequants the weight into transient scratch and calls cuBLASLt bf16
    /// linear, which amortizes the dequant cost across M.
    pub fn forward(
        &self,
        lt: &mut CublasLt,
        y: &mut DeviceBuffer<bf16>,
        x: &DeviceBuffer<bf16>,
        m: usize,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        if m == 1 {
            fp4_gemv_bf16(y, x, &self.packed, &self.scales, self.global_scale,
                           self.out_features, self.in_features, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))
        } else {
            let mut w: DeviceBuffer<bf16> = DeviceBuffer::new_async(
                self.out_features * self.in_features, stream)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            fp4_dequant_bf16(&mut w, &self.packed, &self.scales, self.global_scale,
                              self.out_features, self.in_features, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            lt.linear_bf16(y, x, &w, None, m, self.out_features, self.in_features,
                            1.0, 0.0, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))
        }
    }
}

pub struct LayerWeights {
    pub input_layernorm: DeviceBuffer<bf16>,
    pub post_attention_layernorm: DeviceBuffer<bf16>,
    pub pre_feedforward_layernorm: DeviceBuffer<bf16>,
    pub post_feedforward_layernorm: DeviceBuffer<bf16>,
    pub post_per_layer_input_norm: DeviceBuffer<bf16>,
    pub q_norm: DeviceBuffer<bf16>,
    pub k_norm: Option<DeviceBuffer<bf16>>,

    pub q_proj: QuantLinearDev,
    pub k_proj: Option<QuantLinearDev>,
    pub v_proj: Option<QuantLinearDev>,
    pub o_proj: QuantLinearDev,

    pub gate_proj: QuantLinearDev,
    pub up_proj: QuantLinearDev,
    pub down_proj: QuantLinearDev,

    pub per_layer_input_gate: QuantLinearDev,
    pub per_layer_projection: QuantLinearDev,

    pub layer_scalar: f32,
}

pub fn load_layer_weights(
    mm: &MmapWeights,
    cfg: &GemmaConfig,
    layer_idx: usize,
    stream: &Stream,
) -> anyhow::Result<LayerWeights> {
    let prefix = format!("model.language_model.layers.{layer_idx}");
    let tc = &cfg.text_config;
    let hidden = tc.hidden_size;
    let d = cfg.head_dim_for_layer(layer_idx);
    let owns_kv = mm.layer_owns_kv(layer_idx);

    let upload_bf16 = |bytes: &[u8], len: usize| -> anyhow::Result<DeviceBuffer<bf16>> {
        assert_eq!(bytes.len(), len * 2, "bf16 tensor size");
        let mut d: DeviceBuffer<bf16> = DeviceBuffer::new(len).map_err(|e| anyhow::anyhow!("{e}"))?;
        d.copy_from_host_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(d)
    };

    let input_layernorm = upload_bf16(mm.load_bf16(&format!("{prefix}.input_layernorm.weight"))?, hidden)?;
    let post_attention_layernorm = upload_bf16(mm.load_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?, hidden)?;
    let pre_feedforward_layernorm = upload_bf16(mm.load_bf16(&format!("{prefix}.pre_feedforward_layernorm.weight"))?, hidden)?;
    let post_feedforward_layernorm = upload_bf16(mm.load_bf16(&format!("{prefix}.post_feedforward_layernorm.weight"))?, hidden)?;
    let post_per_layer_input_norm = upload_bf16(mm.load_bf16(&format!("{prefix}.post_per_layer_input_norm.weight"))?, hidden)?;
    let q_norm = upload_bf16(mm.load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?, d)?;
    let k_norm = if owns_kv {
        Some(upload_bf16(mm.load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?, d)?)
    } else { None };

    let q_proj = QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.self_attn.q_proj"))?)?;
    let (k_proj, v_proj) = if owns_kv {
        (Some(QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.self_attn.k_proj"))?)?),
         Some(QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.self_attn.v_proj"))?)?))
    } else { (None, None) };
    let o_proj = QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.self_attn.o_proj"))?)?;

    let gate_proj = QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.mlp.gate_proj"))?)?;
    let up_proj = QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.mlp.up_proj"))?)?;
    let down_proj = QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.mlp.down_proj"))?)?;

    let per_layer_input_gate = QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.per_layer_input_gate"))?)?;
    let per_layer_projection = QuantLinearDev::load(&mm.load_quant_linear(&format!("{prefix}.per_layer_projection"))?)?;

    let layer_scalar_bytes = mm.load_bf16(&format!("{prefix}.layer_scalar"))?;
    let layer_scalar = bf16::from_bits(u16::from_le_bytes([layer_scalar_bytes[0], layer_scalar_bytes[1]])).to_f32();
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(LayerWeights {
        input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
        post_feedforward_layernorm, post_per_layer_input_norm,
        q_norm, k_norm,
        q_proj, k_proj, v_proj, o_proj,
        gate_proj, up_proj, down_proj,
        per_layer_input_gate, per_layer_projection,
        layer_scalar,
    })
}

pub struct TopLevelWeights {
    pub per_layer_model_projection: QuantLinearDev,
    pub per_layer_projection_norm: DeviceBuffer<bf16>,
    pub norm: DeviceBuffer<bf16>,
    /// lm_head (embed_tokens.weight) resident on device. Historically lived
    /// in pinned host memory and was copied per-step; now that the decoder
    /// stack is ~11 ms the 1.34 GB PCIe transfer can't hide behind it and
    /// becomes the dominant per-step cost. Resident costs 1.34 GB VRAM,
    /// worth it at the current budget.
    pub lm_head: DeviceBuffer<bf16>,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelShape {
    pub hidden: usize,
    pub inter: usize,
    pub vocab: usize,
    pub h_heads: usize,
    pub h_kv: usize,
    pub per_layer: usize,
    pub n_layers: usize,
    pub ple_width: usize,
    pub eps: f32,
    pub softcap: f32,
}

#[derive(Clone, Copy)]
pub struct LayerMeta {
    pub layer_idx: usize,
    pub t_q: usize,
    pub t_kv: usize,
    pub q_pos_base: i32,
    pub hidden: usize,
    pub inter: usize,
    pub h_heads: usize,
    pub h_kv: usize,
    pub head_dim: usize,
    pub per_layer: usize,
    pub eps: f32,
    pub window: i32,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub owns_kv: bool,
}

fn clone_buffer_async(src: &DeviceBuffer<bf16>, stream: &Stream) -> anyhow::Result<DeviceBuffer<bf16>> {
    let mut dst = DeviceBuffer::<bf16>::new_async(src.len(), stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    dst.copy_from_device(src).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(dst)
}

// -------------------- Decoder-layer forward --------------------

pub fn layer_forward(
    lw: &LayerWeights,
    meta: LayerMeta,
    h: &mut DeviceBuffer<bf16>,
    ple_layer: &DeviceBuffer<bf16>,
    positions: &DeviceBuffer<i32>,
    kv: &mut KvCache,
    lt: &mut CublasLt,
    stream: &Stream,
) -> anyhow::Result<()> {
    let LayerMeta {
        layer_idx, t_q, t_kv, q_pos_base,
        hidden, inter, h_heads, h_kv, head_dim: d, per_layer,
        eps, window, rope_theta, rotary_dim, owns_kv,
    } = meta;

    let mut d_residual: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_normed:   DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_tmp:      DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Attention block.
    d_residual.copy_from_device(h).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, h, Some(&lw.input_layernorm), t_q, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_q: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * h_heads * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.q_proj.forward(lt, &mut d_q, &d_normed, t_q, stream)?;
    {
        let q_tmp = clone_buffer_async(&d_q, stream)?;
        rmsnorm_bf16(&mut d_q, &q_tmp, Some(&lw.q_norm), t_q * h_heads, d, eps, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    rope_bf16(&mut d_q, positions, t_q, h_heads, d, rotary_dim, rope_theta, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if owns_kv {
        let kl = lw.k_proj.as_ref().expect("owner layer missing k_proj");
        let vl = lw.v_proj.as_ref().expect("owner layer missing v_proj");
        let knw = lw.k_norm.as_ref().expect("owner layer missing k_norm");
        let mut d_k: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * h_kv * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut d_v: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * h_kv * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        kl.forward(lt, &mut d_k, &d_normed, t_q, stream)?;
        vl.forward(lt, &mut d_v, &d_normed, t_q, stream)?;
        {
            let k_tmp = clone_buffer_async(&d_k, stream)?;
            rmsnorm_bf16(&mut d_k, &k_tmp, Some(knw), t_q * h_kv, d, eps, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let v_tmp = clone_buffer_async(&d_v, stream)?;
            rmsnorm_bf16(&mut d_v, &v_tmp, None, t_q * h_kv, d, eps, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        rope_bf16(&mut d_k, positions, t_q, h_kv, d, rotary_dim, rope_theta, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        kv.append(layer_idx, &d_k, &d_v, t_q).map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    let mut d_attn_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * h_heads * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    attn_naive_bf16(&mut d_attn_out, &d_q, kv.k_buf(layer_idx), kv.v_buf(layer_idx),
                     t_q, t_kv, h_heads, h_kv, d, 1.0, q_pos_base, window, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_attn_hidden: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.o_proj.forward(lt, &mut d_attn_hidden, &d_attn_out, t_q, stream)?;
    rmsnorm_bf16(&mut d_tmp, &d_attn_hidden, Some(&lw.post_attention_layernorm),
                  t_q, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(h, &d_residual, &d_tmp, 1.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // MLP block.
    d_residual.copy_from_device(h).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, h, Some(&lw.pre_feedforward_layernorm),
                  t_q, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_gate_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * inter, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_up_out:   DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * inter, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_act:      DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * inter, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_mlp_out:  DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.gate_proj.forward(lt, &mut d_gate_out, &d_normed, t_q, stream)?;
    lw.up_proj.forward(lt, &mut d_up_out, &d_normed, t_q, stream)?;
    gelu_tanh_glu_bf16(&mut d_act, &d_gate_out, &d_up_out, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.down_proj.forward(lt, &mut d_mlp_out, &d_act, t_q, stream)?;
    rmsnorm_bf16(&mut d_tmp, &d_mlp_out, Some(&lw.post_feedforward_layernorm),
                  t_q, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(h, &d_residual, &d_tmp, 1.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // PLE block.
    d_residual.copy_from_device(h).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_gate_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * per_layer, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_glu:      DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * per_layer, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_proj_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.per_layer_input_gate.forward(lt, &mut d_ple_gate_out, h, t_q, stream)?;
    gelu_tanh_glu_bf16(&mut d_ple_glu, &d_ple_gate_out, ple_layer, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.per_layer_projection.forward(lt, &mut d_ple_proj_out, &d_ple_glu, t_q, stream)?;
    rmsnorm_bf16(&mut d_tmp, &d_ple_proj_out, Some(&lw.post_per_layer_input_norm),
                  t_q, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(h, &d_residual, &d_tmp, 1.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // layer_scalar tail multiply.
    let h_in = clone_buffer_async(h, stream)?;
    scale_bf16(h, &h_in, lw.layer_scalar, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

// -------------------- KV cache assembly --------------------

pub fn build_kv_cache(mm: &MmapWeights, cfg: &GemmaConfig, max_len: usize) -> anyhow::Result<KvCache> {
    let tc = &cfg.text_config;
    let n_layers = tc.num_hidden_layers;
    let num_kv_shared = tc.num_kv_shared_layers.unwrap_or(0);
    let first_shared = n_layers - num_kv_shared;
    let h_kv = tc.num_key_value_heads;

    let mut slot_specs: Vec<SlotSpec> = Vec::new();
    let mut slot_idx_of_owner: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut slot_for_layer: Vec<usize> = vec![0; n_layers];
    for l in 0..n_layers {
        if mm.layer_owns_kv(l) {
            let idx = slot_specs.len();
            slot_specs.push(SlotSpec { h_kv, head_dim: cfg.head_dim_for_layer(l) });
            slot_idx_of_owner.insert(l, idx);
            slot_for_layer[l] = idx;
        }
    }
    for l in first_shared..n_layers {
        let kind = tc.layer_types[l];
        let owner = (0..first_shared).rev()
            .find(|&ol| tc.layer_types[ol] == kind)
            .ok_or_else(|| anyhow::anyhow!("no KV owner for layer {l}"))?;
        slot_for_layer[l] = *slot_idx_of_owner.get(&owner).unwrap();
    }
    Ok(KvCache::new(slot_specs, slot_for_layer, max_len).map_err(|e| anyhow::anyhow!("{e}"))?)
}

// -------------------- Forward step --------------------

pub fn forward_step(
    mm: &MmapWeights,
    cfg: &GemmaConfig,
    shape: ModelShape,
    layers: &[LayerWeights],
    top: &TopLevelWeights,
    kv: &mut KvCache,
    ids: &[i32],
    q_pos_base: i32,
    lt: &mut CublasLt,
    stream: &Stream,
    stream_lm: &Stream,
) -> anyhow::Result<Vec<bf16>> {
    let tc = &cfg.text_config;
    let t_q = ids.len();
    let t_kv = kv.len() + t_q;
    let ModelShape { hidden, inter, vocab, h_heads, h_kv, per_layer, n_layers, ple_width, eps, softcap } = shape;
    let _ = stream_lm; // kept for signature compatibility; lm_head is resident.

    let positions_host: Vec<i32> = (0..t_q as i32).map(|i| q_pos_base + i).collect();
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(t_q).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&positions_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    let ie_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens.weight", ids, hidden)?;
    let embed_scale = (hidden as f32).sqrt();
    let ie_scaled: Vec<bf16> = ie_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * embed_scale)).collect();
    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host(&ie_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    let raw_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens_per_layer.weight", ids, ple_width)?;
    let raw_scale = (per_layer as f32).sqrt();
    let raw_scaled: Vec<bf16> = raw_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * raw_scale)).collect();
    let mut d_raw: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_raw.copy_from_host(&raw_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_plmp_w: DeviceBuffer<bf16> = DeviceBuffer::new(ple_width * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    top.per_layer_model_projection.dequant_to(&mut d_plmp_w, stream)?;
    let mut d_ctx: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ctx_normed: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_combined: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_ctx, &d_h, &d_plmp_w, None,
                    t_q, ple_width, hidden, (hidden as f32).powf(-0.5), 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_ctx_normed, &d_ctx, Some(&top.per_layer_projection_norm),
                  t_q * n_layers, per_layer, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut d_combined, &d_ctx_normed, &d_raw, (0.5f32).sqrt(), Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    drop(d_plmp_w); drop(d_raw); drop(d_ctx); drop(d_ctx_normed);

    let mut d_ple_layer: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    for layer_idx in 0..n_layers {
        per_layer_slice_bf16(&mut d_ple_layer, &d_combined,
                              t_q, n_layers, per_layer, layer_idx, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (rope_theta, rotary_dim) = cfg.rope_for_layer(layer_idx);
        let d_layer = cfg.head_dim_for_layer(layer_idx);
        let window = match tc.layer_types[layer_idx] {
            LayerKind::SlidingAttention => tc.sliding_window.unwrap_or(0) as i32,
            LayerKind::FullAttention => 0,
        };
        let meta = LayerMeta {
            layer_idx,
            t_q, t_kv, q_pos_base,
            hidden, inter, h_heads, h_kv,
            head_dim: d_layer, per_layer, eps, window, rope_theta, rotary_dim,
            owns_kv: mm.layer_owns_kv(layer_idx),
        };
        layer_forward(&layers[layer_idx], meta, &mut d_h, &d_ple_layer,
                       &d_positions, kv, lt, stream)?;
    }
    kv.advance(t_q);

    let mut d_last_row: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_last_row.copy_region_from_device(0, &d_h, (t_q - 1) * hidden, hidden)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_normed: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, &d_last_row, Some(&top.norm), 1, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_logits: DeviceBuffer<bf16> = DeviceBuffer::new(vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_logits, &d_normed, &top.lm_head, None,
                    1, vocab, hidden, 1.0, 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_capped: DeviceBuffer<bf16> = DeviceBuffer::new(vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    softcap_bf16(&mut d_capped, &d_logits, softcap, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(d_capped.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?)
}

// -------------------- Batched forward (multi-request decode) --------------------

/// Per-request state for a batched decode step. One slot per concurrent
/// request. Each carries its own KV cache (independent storage) and
/// current absolute position.
pub struct BatchSlot {
    pub kv_cache: KvCache,
    pub cur_pos: i32,
    pub completed: bool,
}

impl BatchSlot {
    pub fn new(mm: &MmapWeights, cfg: &GemmaConfig, max_len: usize) -> anyhow::Result<Self> {
        Ok(Self { kv_cache: build_kv_cache(mm, cfg, max_len)?, cur_pos: 0, completed: false })
    }
}

/// One decode step for N requests at once. All GEMMs run with M=N (batched
/// over requests naturally); attention loops per request because each slot
/// has its own K/V buffer.
///
/// `new_tokens.len() == slots.len()`. Each `new_tokens[i]` is request `i`'s
/// newly sampled token, fed in at its own `slots[i].cur_pos`. This call
/// appends K/V into `slots[i].kv_cache` and bumps `cur_pos` by 1.
///
/// Returns `Vec<Vec<bf16>>` of length N, each inner vec is `[vocab]` softcap
/// logits — caller picks next tokens independently per request.
pub fn forward_step_batched(
    mm: &MmapWeights,
    cfg: &GemmaConfig,
    shape: ModelShape,
    layers: &[LayerWeights],
    top: &TopLevelWeights,
    slots: &mut [BatchSlot],
    new_tokens: &[i32],
    lt: &mut CublasLt,
    stream: &Stream,
    stream_lm: &Stream,
) -> anyhow::Result<Vec<Vec<bf16>>> {
    let n = slots.len();
    assert_eq!(new_tokens.len(), n, "new_tokens length must match slots");
    let tc = &cfg.text_config;
    let ModelShape { hidden, inter, vocab, h_heads, h_kv, per_layer, n_layers, ple_width, eps, softcap } = shape;
    let _ = stream_lm; // kept for signature compatibility; lm_head is resident.

    // Positions array: one per request row. rope_bf16 already accepts a
    // positions vector of length `tokens`, one position per row.
    let positions_host: Vec<i32> = slots.iter().map(|s| s.cur_pos).collect();
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(n).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&positions_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Input embedding: host-gather N rows of embed_tokens, scale, upload.
    let ie_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens.weight", new_tokens, hidden)?;
    let embed_scale = (hidden as f32).sqrt();
    let ie_scaled: Vec<bf16> = ie_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * embed_scale)).collect();
    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(n * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host(&ie_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    // PLE assembly across the N-row batch.
    let raw_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens_per_layer.weight", new_tokens, ple_width)?;
    let raw_scale = (per_layer as f32).sqrt();
    let raw_scaled: Vec<bf16> = raw_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * raw_scale)).collect();
    let mut d_raw: DeviceBuffer<bf16> = DeviceBuffer::new(n * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_raw.copy_from_host(&raw_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_plmp_w: DeviceBuffer<bf16> = DeviceBuffer::new(ple_width * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    top.per_layer_model_projection.dequant_to(&mut d_plmp_w, stream)?;
    let mut d_ctx: DeviceBuffer<bf16> = DeviceBuffer::new(n * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ctx_normed: DeviceBuffer<bf16> = DeviceBuffer::new(n * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_combined: DeviceBuffer<bf16> = DeviceBuffer::new(n * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_ctx, &d_h, &d_plmp_w, None,
                    n, ple_width, hidden, (hidden as f32).powf(-0.5), 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_ctx_normed, &d_ctx, Some(&top.per_layer_projection_norm),
                  n * n_layers, per_layer, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut d_combined, &d_ctx_normed, &d_raw, (0.5f32).sqrt(), Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    drop(d_plmp_w); drop(d_raw); drop(d_ctx); drop(d_ctx_normed);

    let mut d_ple_layer: DeviceBuffer<bf16> = DeviceBuffer::new(n * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    for layer_idx in 0..n_layers {
        per_layer_slice_bf16(&mut d_ple_layer, &d_combined,
                              n, n_layers, per_layer, layer_idx, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (rope_theta, rotary_dim) = cfg.rope_for_layer(layer_idx);
        let d_layer = cfg.head_dim_for_layer(layer_idx);
        let window = match tc.layer_types[layer_idx] {
            LayerKind::SlidingAttention => tc.sliding_window.unwrap_or(0) as i32,
            LayerKind::FullAttention => 0,
        };
        layer_forward_batched(&layers[layer_idx], BatchedLayerMeta {
            layer_idx,
            n, hidden, inter, h_heads, h_kv,
            head_dim: d_layer, per_layer, eps, window, rope_theta, rotary_dim,
            owns_kv: mm.layer_owns_kv(layer_idx),
        }, &mut d_h, &d_ple_layer, &d_positions, slots, lt, stream)?;
    }

    // Bump cur_pos AND advance each slot's KV cache. Missing the advance
    // would leave cur_len at 0 and every subsequent append() would rewrite
    // the same physical slot, which silently corrupts generation.
    for s in slots.iter_mut() {
        s.cur_pos += 1;
        s.kv_cache.advance(1);
    }

    // Final norm + lm_head + softcap for all N rows.
    let mut d_normed_all: DeviceBuffer<bf16> = DeviceBuffer::new(n * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed_all, &d_h, Some(&top.norm), n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_logits: DeviceBuffer<bf16> = DeviceBuffer::new(n * vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_logits, &d_normed_all, &top.lm_head, None,
                    n, vocab, hidden, 1.0, 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_capped: DeviceBuffer<bf16> = DeviceBuffer::new(n * vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    softcap_bf16(&mut d_capped, &d_logits, softcap, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let all = d_capped.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Split [N, vocab] into N per-request rows.
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(all[i * vocab .. (i + 1) * vocab].to_vec());
    }
    Ok(out)
}

/// Per-call meta for batched layer_forward.
#[derive(Clone, Copy)]
struct BatchedLayerMeta {
    layer_idx: usize,
    n: usize,
    hidden: usize,
    inter: usize,
    h_heads: usize,
    h_kv: usize,
    head_dim: usize,
    per_layer: usize,
    eps: f32,
    window: i32,
    rope_theta: f32,
    rotary_dim: usize,
    owns_kv: bool,
}

fn layer_forward_batched(
    lw: &LayerWeights,
    meta: BatchedLayerMeta,
    h: &mut DeviceBuffer<bf16>,
    ple_layer: &DeviceBuffer<bf16>,
    positions: &DeviceBuffer<i32>,
    slots: &mut [BatchSlot],
    lt: &mut CublasLt,
    stream: &Stream,
) -> anyhow::Result<()> {
    let BatchedLayerMeta {
        layer_idx, n, hidden, inter, h_heads, h_kv, head_dim: d, per_layer,
        eps, window, rope_theta, rotary_dim, owns_kv,
    } = meta;

    let mut d_residual: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_normed:   DeviceBuffer<bf16> = DeviceBuffer::new_async(n * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_tmp:      DeviceBuffer<bf16> = DeviceBuffer::new_async(n * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Attention block.
    d_residual.copy_from_device(h).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, h, Some(&lw.input_layernorm), n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_q: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * h_heads * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.q_proj.forward(lt, &mut d_q, &d_normed, n, stream)?;
    {
        let q_tmp = clone_buffer_async(&d_q, stream)?;
        rmsnorm_bf16(&mut d_q, &q_tmp, Some(&lw.q_norm), n * h_heads, d, eps, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    rope_bf16(&mut d_q, positions, n, h_heads, d, rotary_dim, rope_theta, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if owns_kv {
        let kl = lw.k_proj.as_ref().expect("owner layer missing k_proj");
        let vl = lw.v_proj.as_ref().expect("owner layer missing v_proj");
        let knw = lw.k_norm.as_ref().expect("owner layer missing k_norm");
        let mut d_k: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * h_kv * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut d_v: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * h_kv * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        kl.forward(lt, &mut d_k, &d_normed, n, stream)?;
        vl.forward(lt, &mut d_v, &d_normed, n, stream)?;
        {
            let k_tmp = clone_buffer_async(&d_k, stream)?;
            rmsnorm_bf16(&mut d_k, &k_tmp, Some(knw), n * h_kv, d, eps, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let v_tmp = clone_buffer_async(&d_v, stream)?;
            rmsnorm_bf16(&mut d_v, &v_tmp, None, n * h_kv, d, eps, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        rope_bf16(&mut d_k, positions, n, h_kv, d, rotary_dim, rope_theta, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Scatter each row into its slot's KV cache.
        let row_kv = h_kv * d;
        for i in 0..n {
            let mut d_k_row: DeviceBuffer<bf16> = DeviceBuffer::new_async(row_kv, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut d_v_row: DeviceBuffer<bf16> = DeviceBuffer::new_async(row_kv, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
            d_k_row.copy_region_from_device(0, &d_k, i * row_kv, row_kv).map_err(|e| anyhow::anyhow!("{e}"))?;
            d_v_row.copy_region_from_device(0, &d_v, i * row_kv, row_kv).map_err(|e| anyhow::anyhow!("{e}"))?;
            slots[i].kv_cache.append(layer_idx, &d_k_row, &d_v_row, 1)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
    }

    // Per-request attention (kv_len varies per slot, so we loop). GEMMs
    // elsewhere remain batched at M=N.
    let mut d_attn_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * h_heads * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let row_q = h_heads * d;
    for i in 0..n {
        // At this point we've appended 1 token but not advanced cur_len yet.
        // The newly-written token lives at offset cur_len (which is
        // slots[i].cur_pos). Attention should see kv_len = cur_pos + 1 so
        // the new token is included.
        let t_kv_i = slots[i].cur_pos as usize + 1;
        let q_pos_i = slots[i].cur_pos;
        let mut d_q_row: DeviceBuffer<bf16> = DeviceBuffer::new_async(row_q, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut d_out_row: DeviceBuffer<bf16> = DeviceBuffer::new_async(row_q, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        d_q_row.copy_region_from_device(0, &d_q, i * row_q, row_q).map_err(|e| anyhow::anyhow!("{e}"))?;
        attn_naive_bf16(&mut d_out_row, &d_q_row,
                         slots[i].kv_cache.k_buf(layer_idx),
                         slots[i].kv_cache.v_buf(layer_idx),
                         1, t_kv_i, h_heads, h_kv, d, 1.0, q_pos_i, window, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Scatter result back into d_attn_out[i].
        d_attn_out.copy_slice_from_device(i * row_q, &d_out_row).map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    let mut d_attn_hidden: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.o_proj.forward(lt, &mut d_attn_hidden, &d_attn_out, n, stream)?;
    rmsnorm_bf16(&mut d_tmp, &d_attn_hidden, Some(&lw.post_attention_layernorm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(h, &d_residual, &d_tmp, 1.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // MLP block — all batched at M=N.
    d_residual.copy_from_device(h).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, h, Some(&lw.pre_feedforward_layernorm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_gate_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * inter, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_up_out:   DeviceBuffer<bf16> = DeviceBuffer::new_async(n * inter, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_act:      DeviceBuffer<bf16> = DeviceBuffer::new_async(n * inter, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_mlp_out:  DeviceBuffer<bf16> = DeviceBuffer::new_async(n * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.gate_proj.forward(lt, &mut d_gate_out, &d_normed, n, stream)?;
    lw.up_proj.forward(lt, &mut d_up_out, &d_normed, n, stream)?;
    gelu_tanh_glu_bf16(&mut d_act, &d_gate_out, &d_up_out, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.down_proj.forward(lt, &mut d_mlp_out, &d_act, n, stream)?;
    rmsnorm_bf16(&mut d_tmp, &d_mlp_out, Some(&lw.post_feedforward_layernorm), n, hidden, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(h, &d_residual, &d_tmp, 1.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;

    // PLE block — batched.
    d_residual.copy_from_device(h).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_gate_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * per_layer, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_glu:      DeviceBuffer<bf16> = DeviceBuffer::new_async(n * per_layer, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_proj_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.per_layer_input_gate.forward(lt, &mut d_ple_gate_out, h, n, stream)?;
    gelu_tanh_glu_bf16(&mut d_ple_glu, &d_ple_gate_out, ple_layer, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.per_layer_projection.forward(lt, &mut d_ple_proj_out, &d_ple_glu, n, stream)?;
    rmsnorm_bf16(&mut d_tmp, &d_ple_proj_out, Some(&lw.post_per_layer_input_norm), n, hidden, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(h, &d_residual, &d_tmp, 1.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;

    // layer_scalar tail multiply (applies to all N rows equally).
    let h_in = clone_buffer_async(h, stream)?;
    scale_bf16(h, &h_in, lw.layer_scalar, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

pub fn argmax_bf16(row: &[bf16]) -> i32 {
    let mut best_i = 0;
    let mut best_v = row[0].to_f32();
    for (i, v) in row.iter().enumerate().skip(1) {
        let f = v.to_f32();
        if f > best_v {
            best_v = f;
            best_i = i;
        }
    }
    best_i as i32
}

// -------------------- Engine (high-level API) --------------------

pub struct Engine {
    pub cfg: GemmaConfig,
    pub shape: ModelShape,
    pub mm: MmapWeights,
    pub tokenizer: Tokenizer,
    pub layers: Vec<LayerWeights>,
    pub top: TopLevelWeights,
    pub kv: KvCache,
    pub lt: CublasLt,
    pub stream: Stream,
    pub stream_lm: Stream,
    /// Absolute position of the next token to be fed. Advances as KV is filled.
    pub cur_pos: i32,
    pub max_len: usize,
}

/// Gemma 4 EOS tokens: 1 (eos_token_id) and 106 (<turn|>).
pub const GEMMA4_EOS: &[u32] = &[1, 106];

/// Wrap a user prompt in Gemma 4's chat template.
pub fn wrap_chat_prompt(prompt: &str) -> String {
    format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt)
}

pub struct GenerateStats {
    pub prompt_len: usize,
    pub generated: usize,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
}

/// One logical request inside a batched generate call.
pub struct BatchRequest {
    pub prompt_ids: Vec<u32>,
    pub max_new: usize,
}

#[derive(Default, Clone)]
pub struct BatchGenResult {
    pub generated: Vec<u32>,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
}

impl Engine {
    /// Load the model. `max_len` sizes the KV cache (prompt + max_new).
    pub fn load(model_dir: &Path, max_len: usize) -> anyhow::Result<Self> {
        Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
        let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
        let stream_lm = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
        let lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

        let cfg = GemmaConfig::from_path(&model_dir.join("config.json"))?;
        let tc = &cfg.text_config;
        let shape = ModelShape {
            hidden: tc.hidden_size,
            inter: tc.intermediate_size,
            vocab: tc.vocab_size,
            h_heads: tc.num_attention_heads,
            h_kv: tc.num_key_value_heads,
            per_layer: tc.hidden_size_per_layer_input
                .ok_or_else(|| anyhow::anyhow!("hidden_size_per_layer_input missing"))?,
            n_layers: tc.num_hidden_layers,
            ple_width: tc.hidden_size_per_layer_input.unwrap_or(0) * tc.num_hidden_layers,
            eps: tc.rms_norm_eps as f32,
            softcap: tc.final_logit_softcapping.unwrap_or(30.0) as f32,
        };

        let tokenizer = Tokenizer::from_dir(model_dir)?;
        let mm = MmapWeights::open(&model_dir.join("model.safetensors"))?;

        let mut layers: Vec<LayerWeights> = Vec::with_capacity(shape.n_layers);
        for l in 0..shape.n_layers {
            layers.push(load_layer_weights(&mm, &cfg, l, &stream)?);
        }
        let top = TopLevelWeights {
            per_layer_model_projection: QuantLinearDev::load(
                &mm.load_quant_linear("model.language_model.per_layer_model_projection")?)?,
            per_layer_projection_norm: {
                let b = mm.load_bf16("model.language_model.per_layer_projection_norm.weight")?;
                let mut d: DeviceBuffer<bf16> = DeviceBuffer::new(shape.per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
                d.copy_from_host_bytes(b).map_err(|e| anyhow::anyhow!("{e}"))?;
                d
            },
            norm: {
                let b = mm.load_bf16("model.language_model.norm.weight")?;
                let mut d: DeviceBuffer<bf16> = DeviceBuffer::new(shape.hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
                d.copy_from_host_bytes(b).map_err(|e| anyhow::anyhow!("{e}"))?;
                d
            },
            lm_head: {
                let b = mm.load_bf16("model.language_model.embed_tokens.weight")?;
                let mut d: DeviceBuffer<bf16> = DeviceBuffer::new(shape.vocab * shape.hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
                d.copy_from_host_bytes(b).map_err(|e| anyhow::anyhow!("{e}"))?;
                d
            },
        };
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

        let kv = build_kv_cache(&mm, &cfg, max_len)?;

        Ok(Self { cfg, shape, mm, tokenizer, layers, top, kv, lt, stream, stream_lm, cur_pos: 0, max_len })
    }

    /// Reset KV cache + position tracking for a new request.
    pub fn reset(&mut self) {
        self.kv.reset();
        self.cur_pos = 0;
    }

    pub fn tokenize(&self, text: &str, add_specials: bool) -> anyhow::Result<Vec<u32>> {
        self.tokenizer.encode(text, add_specials).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn decode(&self, ids: &[u32], skip_specials: bool) -> anyhow::Result<String> {
        self.tokenizer.decode(ids, skip_specials).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Run prefill + decode. `on_token` receives each generated token and
    /// returns `true` to continue, `false` to stop early (e.g., client
    /// disconnected). Stops automatically on any token in `stop_tokens`
    /// (after at least one step) or after `max_new`.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        stop_tokens: &[u32],
        mut on_token: impl FnMut(u32) -> bool,
    ) -> anyhow::Result<GenerateStats> {
        use std::time::Instant;

        anyhow::ensure!(prompt_ids.len() + max_new <= self.max_len,
            "prompt ({}) + max_new ({}) exceeds max_len ({})",
            prompt_ids.len(), max_new, self.max_len);

        self.reset();

        let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&x| x as i32).collect();
        let t_prefill = Instant::now();
        let logits = forward_step(&self.mm, &self.cfg, self.shape,
                                    &self.layers, &self.top, &mut self.kv,
                                    &prompt_i32, self.cur_pos, &mut self.lt,
                                    &self.stream, &self.stream_lm)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        self.cur_pos += prompt_ids.len() as i32;
        let mut next: u32 = argmax_bf16(&logits) as u32;

        if !on_token(next) { return Ok(GenerateStats { prompt_len: prompt_ids.len(), generated: 1, prefill_ms, decode_ms: Vec::new() }); }

        let mut decode_ms: Vec<f64> = Vec::with_capacity(max_new);
        let mut generated = 1usize;
        for step in 0..max_new {
            if stop_tokens.contains(&next) && step > 0 { break; }
            let t_step = Instant::now();
            let logits = forward_step(&self.mm, &self.cfg, self.shape,
                                       &self.layers, &self.top, &mut self.kv,
                                       &[next as i32], self.cur_pos, &mut self.lt,
                                       &self.stream, &self.stream_lm)?;
            decode_ms.push(t_step.elapsed().as_secs_f64() * 1e3);
            self.cur_pos += 1;
            next = argmax_bf16(&logits) as u32;
            generated += 1;
            if !on_token(next) { break; }
        }

        Ok(GenerateStats { prompt_len: prompt_ids.len(), generated, prefill_ms, decode_ms })
    }

    /// Batched generate for up to N concurrent requests. GEMMs batch at M=N
    /// so weight bandwidth is amortized across requests; attention loops
    /// per request internally (each has its own KV cache).
    ///
    /// Prefill runs serially per request (v1 simplification — prefill
    /// batching across different prompt lengths would need ragged-batch
    /// attention). Decode is batched. Requests that hit EOS or max_new
    /// early keep their slot in the batch until all complete — wastes
    /// some compute but keeps the implementation simple.
    ///
    /// `on_token(request_idx, token) -> bool` is called for each newly
    /// sampled token. Return `false` to stop the entire batch.
    pub fn generate_batch(
        &mut self,
        requests: Vec<BatchRequest>,
        stop_tokens: &[u32],
        mut on_token: impl FnMut(usize, u32) -> bool,
    ) -> anyhow::Result<Vec<BatchGenResult>> {
        use std::time::Instant;
        let n = requests.len();
        anyhow::ensure!(n > 0, "generate_batch: empty request list");
        for (i, r) in requests.iter().enumerate() {
            anyhow::ensure!(r.prompt_ids.len() + r.max_new <= self.max_len,
                "request {i}: prompt ({}) + max_new ({}) exceeds max_len ({})",
                r.prompt_ids.len(), r.max_new, self.max_len);
        }

        // Per-request BatchSlots with independent KV caches.
        let mut slots: Vec<BatchSlot> = (0..n)
            .map(|_| BatchSlot::new(&self.mm, &self.cfg, self.max_len))
            .collect::<anyhow::Result<_>>()?;
        let mut results: Vec<BatchGenResult> = vec![BatchGenResult::default(); n];
        let mut last_tokens: Vec<i32> = vec![0; n];
        let mut done: Vec<bool> = vec![false; n];

        // --- Serial prefill ---
        for i in 0..n {
            let prompt_i32: Vec<i32> = requests[i].prompt_ids.iter().map(|&x| x as i32).collect();
            let q_pos_base = slots[i].cur_pos;
            let t0 = Instant::now();
            let logits = forward_step(&self.mm, &self.cfg, self.shape,
                                       &self.layers, &self.top, &mut slots[i].kv_cache,
                                       &prompt_i32, q_pos_base,
                                       &mut self.lt, &self.stream, &self.stream_lm)?;
            results[i].prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
            slots[i].cur_pos += requests[i].prompt_ids.len() as i32;
            let next = argmax_bf16(&logits) as u32;
            results[i].generated.push(next);
            last_tokens[i] = next as i32;
            if !on_token(i, next) { done[i] = true; }
            // EOS on the prefill-sampled token is not a stop for step 0
            // (must generate at least one non-EOS token); decode loop
            // checks at the start of each step.
        }

        // --- Batched decode ---
        let max_max_new = requests.iter().map(|r| r.max_new).max().unwrap_or(0);
        for step in 0..max_max_new {
            // Check which requests finished since the last step.
            for i in 0..n {
                if done[i] { continue; }
                let gen_len = results[i].generated.len();
                if gen_len >= requests[i].max_new {
                    done[i] = true;
                    continue;
                }
                if step > 0 && stop_tokens.contains(results[i].generated.last().unwrap()) {
                    done[i] = true;
                }
            }
            if done.iter().all(|&d| d) { break; }

            let t_step = Instant::now();
            let logits_per_req = forward_step_batched(
                &self.mm, &self.cfg, self.shape,
                &self.layers, &self.top, &mut slots, &last_tokens,
                &mut self.lt, &self.stream, &self.stream_lm,
            )?;
            let ms = t_step.elapsed().as_secs_f64() * 1e3;

            for i in 0..n {
                if done[i] { continue; }
                let next = argmax_bf16(&logits_per_req[i]) as u32;
                results[i].decode_ms.push(ms);
                results[i].generated.push(next);
                last_tokens[i] = next as i32;
                if !on_token(i, next) { done[i] = true; }
            }
        }

        Ok(results)
    }
}
