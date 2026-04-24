//! Forward-pass orchestration for xenon.
//!
//! Loads Gemma 4 weights (NVFP4 packed resident, norms bf16), wires up the
//! decoder stack, and exposes a simple [`Engine`] type with `generate`
//! suitable for both CLI and the HTTP server.

#![allow(clippy::too_many_arguments)]

use std::path::Path;

use half::bf16;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use xenon_core::{EmotionArtifact, GemmaConfig, LayerKind, MmapWeights, Tokenizer};
use xenon_kernels::{
    add_scale_bf16, attn_flash_tc_bf16, attn_naive_bf16, attn_split_kv_auto_chunk_size,
    attn_split_kv_bf16, attn_split_kv_bf16_device,
    cuda::{Device, DeviceBuffer, GraphExec, PinnedBuffer, Stream, CAPTURE_MODE_RELAXED},
    emotion_score_bf16, fp4_dequant_bf16, fp4_gemv_bf16, gelu_tanh_glu_bf16, inc_i32_device,
    nvfp4_quantize_bf16, per_layer_slice_bf16, rmsnorm_bf16, rope_bf16, round_up,
    sample_topk_bf16, scale_bf16, softcap_bf16, swizzle_blockscale_ue4m3, CublasLt, KvCache,
    SlotSpec,
};

/// RTX PRO 2000 Blackwell Laptop SM count. Used by the attn dispatch to pick
/// between naive (T_q·H already saturates) and split-KV (doesn't).
const DEVICE_SM_COUNT: usize = 26;
/// Grid size at which the naive kernel's (T_q * H) blocks already fill the
/// GPU, so split-KV's merge-kernel overhead is pure loss. Empirically ~2×
/// SM count from `test-attn-split-kv` sweep.
const ATTN_SATURATION_BLOCKS: usize = DEVICE_SM_COUNT * 2;
/// Flash-tc (mma.m16n8k16) minimum Q rows. Under this we can't build a full
/// mma tile; dispatch falls back to naive or split-KV.
const ATTN_FLASH_TC_MIN_TQ: usize = 16;

// -------------------- Weight containers --------------------

/// NVFP4 linear weight resident on device.
///
/// Carries two copies of the per-16 UE4M3 block scales:
/// - `scales` — row-major `[N, K/16]`, used by `fp4_dequant_bf16` (which
///    reads linear).
/// - `scales_swizzled` — 128×4 interleaved layout required by cuBLASLt's
///    `VEC16_UE4M3` mode, produced once at load via `swizzle_blockscale_ue4m3`.
///
/// `global_scale` is the weight per-tensor f32 (weight_scale_2 in modelopt),
/// used as `alpha` on the native NVFP4 GEMM path. The checkpoint also carries
/// a per-tensor `input_scale` but we don't use it: `nvfp4_quantize_bf16`
/// stores block scales as `amax/6`, so FP4 × FP8 reconstructs `x` directly
/// and `alpha = weight_scale_2` alone is correct.
pub struct QuantLinearDev {
    pub packed: DeviceBuffer<u8>,
    pub scales: DeviceBuffer<u8>,
    pub scales_swizzled: DeviceBuffer<u8>,
    pub global_scale: f32,
    pub out_features: usize,
    pub in_features: usize,
}

/// An activation x already quantized to FP4 with swizzled block scales,
/// ready to feed `QuantLinearDev::forward_fp4_prepacked`. Built once per
/// shared activation via `prepare_fp4_activation` and reused across
/// multiple projections that take the same x (e.g. q/k/v all feed from
/// the post-RMSNorm hidden, gate/up both feed from the pre-MLP norm).
/// Saves one `xk_nvfp4_quantize_bf16` call per extra projection.
pub struct Fp4SharedActivation {
    pub packed: DeviceBuffer<u8>,
    pub scales_swizzled: DeviceBuffer<u8>,
    pub m: usize,
    pub k: usize,
}

impl QuantLinearDev {
    pub fn load(q: &xenon_core::QuantLinearRef<'_>) -> anyhow::Result<Self> {
        let n = q.out_features;
        let k = q.in_features;
        let k_blocks = k / 16;
        let mut packed: DeviceBuffer<u8> = DeviceBuffer::new(q.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut scales: DeviceBuffer<u8> = DeviceBuffer::new(q.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        packed.copy_from_host_bytes(q.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
        scales.copy_from_host_bytes(q.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
        let n_padded = round_up(n, 128);
        let kb_padded = round_up(k_blocks, 4);
        let mut scales_swizzled: DeviceBuffer<u8> =
            DeviceBuffer::new(n_padded * kb_padded).map_err(|e| anyhow::anyhow!("{e}"))?;
        swizzle_blockscale_ue4m3(&mut scales_swizzled, &scales, n, k_blocks, None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            packed, scales, scales_swizzled,
            global_scale: q.global_scale,
            out_features: n,
            in_features: k,
        })
    }

    pub fn dequant_to(&self, out: &mut DeviceBuffer<bf16>, stream: &Stream) -> anyhow::Result<()> {
        fp4_dequant_bf16(out, &self.packed, &self.scales, self.global_scale,
                          self.out_features, self.in_features, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Quantize a bf16 activation x[m, k] to FP4 + swizzled UE4M3 scales once,
    /// so multiple projections that share the same x can call
    /// `forward_fp4_prepacked` instead of re-quantizing each time.
    ///
    /// `k` is taken from `self.in_features` — all sharers must have the same
    /// K (and in practice they do: q/k/v all have K=hidden, gate/up both have
    /// K=hidden).
    pub fn prepare_fp4_activation(
        &self,
        x: &DeviceBuffer<bf16>,
        m: usize,
        stream: &Stream,
    ) -> anyhow::Result<Fp4SharedActivation> {
        let k = self.in_features;
        let k_blocks = k / 16;
        let mut packed: DeviceBuffer<u8> =
            DeviceBuffer::new_async(m * (k / 2), stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut scales_linear: DeviceBuffer<u8> =
            DeviceBuffer::new_async(m * k_blocks, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        nvfp4_quantize_bf16(&mut packed, &mut scales_linear, x, m, k, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let m_padded = round_up(m, 128);
        let kb_padded = round_up(k_blocks, 4);
        let mut scales_swizzled: DeviceBuffer<u8> =
            DeviceBuffer::new_async(m_padded * kb_padded, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        swizzle_blockscale_ue4m3(&mut scales_swizzled, &scales_linear, m, k_blocks, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Fp4SharedActivation { packed, scales_swizzled, m, k })
    }

    /// Native NVFP4 GEMM using a pre-packed shared activation (skips the
    /// quantize+swizzle work — shaved ~27% of prefill GPU time across
    /// q/k/v/gate/up projections).
    pub fn forward_fp4_prepacked(
        &self,
        lt: &mut CublasLt,
        y: &mut DeviceBuffer<bf16>,
        x: &Fp4SharedActivation,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        debug_assert_eq!(x.k, self.in_features, "prepacked K mismatch");
        let n = self.out_features;
        let k = self.in_features;
        let alpha_scale = self.global_scale;
        lt.linear_nvfp4(y, &x.packed, &x.scales_swizzled,
                         &self.packed, &self.scales_swizzled,
                         alpha_scale, None, x.m, n, k, 1.0, 0.0, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// `y = x @ W^T` with FP4 weight. Three-way dispatch by M:
    /// - `m == 1` → fused `xk_fp4_gemv_bf16` (decode fast path).
    /// - `m >= 128` → native NVFP4 tensor-core GEMM: quantize the
    ///   activation to FP4 (`nvfp4_quantize_bf16`), swizzle its block scales
    ///   into cuBLASLt's 128×4-interleaved layout, call `linear_nvfp4` with
    ///   the pre-swizzled weight scales and `alpha = weight_scale_2`. Our
    ///   activation quantizer stores scales as amax/6, so the FP4 × FP8
    ///   product reconstructs x directly and no `input_scale` factor is
    ///   needed. See `project_xenon_nvfp4_swizzle` memory for the history.
    /// - otherwise → dequant weight to bf16 scratch + `linear_bf16`.
    ///
    /// cuBLASLt's FP4 kernel is silently broken at `m < 128` on sm_120a, so
    /// the threshold is strict.
    pub fn forward(
        &self,
        lt: &mut CublasLt,
        y: &mut DeviceBuffer<bf16>,
        x: &DeviceBuffer<bf16>,
        m: usize,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let n = self.out_features;
        let k = self.in_features;
        if m == 1 {
            fp4_gemv_bf16(y, x, &self.packed, &self.scales, self.global_scale,
                           n, k, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))
        } else if m >= 128 {
            // One-shot path (no sharing). `prepare_fp4_activation` + drop.
            let fp4_x = self.prepare_fp4_activation(x, m, stream)?;
            self.forward_fp4_prepacked(lt, y, &fp4_x, stream)
        } else {
            let mut w: DeviceBuffer<bf16> = DeviceBuffer::new_async(n * k, stream)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            fp4_dequant_bf16(&mut w, &self.packed, &self.scales, self.global_scale,
                              n, k, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            lt.linear_bf16(y, x, &w, None, m, n, k, 1.0, 0.0, Some(stream))
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
    /// Dequanted once at load (was `dequant_to` per forward_step before — 647 μs
    /// per call × ≥6 calls/run showed up clearly on nsys). 43 MB VRAM cost.
    pub per_layer_model_projection: DeviceBuffer<bf16>,
    pub per_layer_projection_norm: DeviceBuffer<bf16>,
    pub norm: DeviceBuffer<bf16>,
    /// lm_head (embed_tokens.weight) resident on device. Historically lived
    /// in pinned host memory and was copied per-step; now that the decoder
    /// stack is ~11 ms the 1.34 GB PCIe transfer can't hide behind it and
    /// becomes the dominant per-step cost. Resident costs 1.34 GB VRAM,
    /// worth it at the current budget.
    pub lm_head: DeviceBuffer<bf16>,
}

/// Device-resident emotion probe vectors. Static for the engine's lifetime
/// once loaded. Inference math: `score[e] = (h[t] - mean) · v[e]` evaluated
/// at a single chosen residual-stream layer (`scored_model_layer`).
pub struct EmotionProbes {
    pub emotions: Vec<String>,
    /// Model-space layer index probing runs at.
    pub scored_model_layer: u32,
    /// Hidden size H (= 2560 for Gemma 4 E4B).
    pub hidden: usize,
    /// Number of emotions N.
    pub num_emotions: usize,
    /// `[N, H]` bf16 row-major: the slice of the artifact for `scored_model_layer`.
    pub vectors: DeviceBuffer<bf16>,
    /// `[H]` bf16: global mean to subtract before the projection.
    pub global_mean: DeviceBuffer<bf16>,
    /// Optional `[N]` per-emotion calibration mean. If present, final response
    /// scores are z-scored against it.
    pub z_mean: Option<Vec<f32>>,
    /// Optional `[N]` per-emotion calibration stddev.
    pub z_std: Option<Vec<f32>>,
}

/// Per-request emotion accumulator. One decode step contributes one row of
/// per-token scores (`[N] fp32`) which is added into `sums`. At end of
/// generation, D2H `sums`, divide by `count`, and (optionally) z-score to
/// produce the final per-emotion scalar returned to the caller.
pub struct EmotionAccum {
    /// `[N] fp32` device-side sum of per-token scores.
    pub sums: DeviceBuffer<f32>,
    /// Number of tokens contributed so far.
    pub count: u32,
}

/// Per-batch emotion accumulator. Layout: `[max_batch, N_emotions] fp32`.
/// Each decode step's `[n_active, N]` scores accumulate directly into the
/// matching prefix via the kernel's `accumulate=true` mode. `counts[i]`
/// tracks how many tokens slot `i` has contributed so `finalize` can
/// divide correctly.
pub struct BatchedEmotionAccums {
    pub sums: DeviceBuffer<f32>,
    pub counts: Vec<u32>,
    pub max_batch: usize,
    pub n_emotions: usize,
}

impl BatchedEmotionAccums {
    pub fn new(max_batch: usize, num_emotions: usize, stream: &Stream) -> anyhow::Result<Self> {
        let sz = max_batch * num_emotions;
        let mut sums: DeviceBuffer<f32> =
            DeviceBuffer::new_async(sz, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let zeros = vec![0f32; sz];
        sums.copy_from_host_bytes_async(bytemuck::cast_slice(&zeros), stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { sums, counts: vec![0; max_batch], max_batch, n_emotions: num_emotions })
    }

    /// D2H the full `[max_batch, N]` accumulator once, then produce a
    /// `Vec<(emotion, mean_score)>` per slot. `num_active` is how many slot
    /// rows the caller cares about (the rest are zero / unused).
    pub fn finalize_all(
        &self,
        num_active: usize,
        probes: &EmotionProbes,
    ) -> anyhow::Result<Vec<Vec<(String, f32)>>> {
        assert!(num_active <= self.max_batch);
        let all = self.sums.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
        let n = self.n_emotions;
        let mut out = Vec::with_capacity(num_active);
        for slot in 0..num_active {
            let count = self.counts[slot].max(1) as f32;
            let row = &all[slot * n..(slot + 1) * n];
            let means: Vec<f32> = row.iter().map(|&s| s / count).collect();
            let entries: Vec<(String, f32)> = probes
                .emotions
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let v = match (&probes.z_mean, &probes.z_std) {
                        (Some(mu), Some(sigma)) => {
                            let s = sigma[i].max(1e-6);
                            (means[i] - mu[i]) / s
                        }
                        _ => means[i],
                    };
                    (e.clone(), v)
                })
                .collect();
            out.push(entries);
        }
        Ok(out)
    }
}

impl EmotionAccum {
    pub fn new(num_emotions: usize, stream: &Stream) -> anyhow::Result<Self> {
        let mut sums: DeviceBuffer<f32> = DeviceBuffer::new_async(num_emotions, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Zero the buffer so accumulate=true builds up cleanly.
        let zeros = vec![0f32; num_emotions];
        sums.copy_from_host_bytes_async(bytemuck::cast_slice(&zeros), stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { sums, count: 0 })
    }

    /// D2H the accumulated sums and produce `(emotion_name, mean_score)` pairs,
    /// optionally z-scored. Caller must have synced the stream first.
    pub fn finalize(
        &self,
        probes: &EmotionProbes,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        let raw = self.sums.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
        let count = self.count.max(1) as f32;
        let means: Vec<f32> = raw.iter().map(|&s| s / count).collect();
        let out: Vec<(String, f32)> = probes
            .emotions
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let v = match (&probes.z_mean, &probes.z_std) {
                    (Some(mu), Some(sigma)) => {
                        let s = sigma[i].max(1e-6);
                        (means[i] - mu[i]) / s
                    }
                    _ => means[i],
                };
                (e.clone(), v)
            })
            .collect();
        Ok(out)
    }
}

impl EmotionProbes {
    /// Load an artifact from disk and upload the chosen layer's vectors to
    /// the device. Picks the first layer listed in the artifact by default.
    pub fn load(path: &Path, stream: &Stream) -> anyhow::Result<Self> {
        let art = EmotionArtifact::open(path).map_err(|e| anyhow::anyhow!("{e}"))?;
        if art.layers.is_empty() {
            anyhow::bail!("emotion artifact has no layers");
        }
        let scored_model_layer = art.layers[0];
        let slot = 0usize;
        let hidden = art.hidden;
        let num_emotions = art.num_emotions();

        let vecs_per_layer_bytes = num_emotions * hidden * 2; // bf16
        let gmean_per_layer_bytes = hidden * 2; // bf16

        let v_start = slot * vecs_per_layer_bytes;
        let v_end = v_start + vecs_per_layer_bytes;
        let g_start = slot * gmean_per_layer_bytes;
        let g_end = g_start + gmean_per_layer_bytes;

        let mut vectors: DeviceBuffer<bf16> =
            DeviceBuffer::new(num_emotions * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
        vectors
            .copy_from_host_bytes(&art.vectors_bytes[v_start..v_end])
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut global_mean: DeviceBuffer<bf16> =
            DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
        global_mean
            .copy_from_host_bytes(&art.global_mean_bytes[g_start..g_end])
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Optional calibration — both present or both absent.
        let (z_mean, z_std) = match (art.z_mean_bytes, art.z_std_bytes) {
            (Some(mb), Some(sb)) => {
                anyhow::ensure!(mb.len() == num_emotions * 4, "z_mean size");
                anyhow::ensure!(sb.len() == num_emotions * 4, "z_std size");
                let m: Vec<f32> = bytemuck::cast_slice(mb).to_vec();
                let s: Vec<f32> = bytemuck::cast_slice(sb).to_vec();
                (Some(m), Some(s))
            }
            _ => (None, None),
        };

        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            emotions: art.emotions,
            scored_model_layer,
            hidden,
            num_emotions,
            vectors,
            global_mean,
            z_mean,
            z_std,
        })
    }
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

/// Pre-allocated per-forward scratch for the batched decode path.
///
/// Owned by `Engine` and reused across decode steps so every buffer is a
/// single permanent allocation. Sized for `max_batch × max_len × largest
/// head_dim`; smaller requests just use the prefix (kernels accept
/// `buf.len() >= required`).
///
/// **Why this exists:** CUDA graphs cannot capture `cudaMallocAsync` /
/// `cudaFreeAsync`. Moving all per-step allocations out of the hot path is
/// a prerequisite for graph capture in `xenon-server`. Without this, every
/// `forward_step_batched` invocation issued 40+ stream-ordered allocations
/// (one per scratch buffer × 42 layers); with this, it issues zero.
///
/// The CLI (`xenon-cli generate`, `xenon-cli bench`) does not use graphs
/// per the project decision, but it still benefits from persistent scratch
/// — same-or-slightly-faster with less allocator pressure.
pub struct DecodeScratch {
    pub max_batch: usize,
    pub max_len: usize,
    pub shape: ModelShape,
    /// Max head_dim across all layers (sliding 256, full 512 → 512).
    pub head_dim_max: usize,

    // --- Top-level (once per forward). ---
    pub positions: DeviceBuffer<i32>,        // [max_batch]
    pub h: DeviceBuffer<bf16>,               // [max_batch * hidden]
    pub raw: DeviceBuffer<bf16>,             // [max_batch * ple_width]
    pub ctx: DeviceBuffer<bf16>,             // [max_batch * ple_width]
    pub ctx_normed: DeviceBuffer<bf16>,      // [max_batch * ple_width]
    pub combined: DeviceBuffer<bf16>,        // [max_batch * ple_width]
    pub ple_layer: DeviceBuffer<bf16>,       // [max_batch * per_layer]
    pub normed_all: DeviceBuffer<bf16>,      // [max_batch * hidden]
    pub logits: DeviceBuffer<bf16>,          // [max_batch * vocab]
    pub capped: DeviceBuffer<bf16>,          // [max_batch * vocab]

    // --- Per-layer, reused across all n_layers iterations. ---
    pub residual: DeviceBuffer<bf16>,        // [max_batch * hidden]
    pub normed: DeviceBuffer<bf16>,          // [max_batch * hidden]
    pub tmp: DeviceBuffer<bf16>,             // [max_batch * hidden]
    pub h_clone: DeviceBuffer<bf16>,         // [max_batch * hidden] (for layer_scalar tail)
    pub q: DeviceBuffer<bf16>,               // [max_batch * h_heads * head_dim_max]
    pub q_tmp: DeviceBuffer<bf16>,           // clone for q_norm (in-place avoidance)
    pub k: DeviceBuffer<bf16>,               // [max_batch * h_kv * head_dim_max]
    pub k_tmp: DeviceBuffer<bf16>,           // clone for k_norm
    pub v: DeviceBuffer<bf16>,               // [max_batch * h_kv * head_dim_max]
    pub v_tmp: DeviceBuffer<bf16>,           // clone for v RMSNorm
    pub attn_out: DeviceBuffer<bf16>,        // [max_batch * h_heads * head_dim_max]
    pub attn_hidden: DeviceBuffer<bf16>,     // [max_batch * hidden]
    pub gate_out: DeviceBuffer<bf16>,        // [max_batch * inter]
    pub up_out: DeviceBuffer<bf16>,          // [max_batch * inter]
    pub act: DeviceBuffer<bf16>,             // [max_batch * inter]
    pub mlp_out: DeviceBuffer<bf16>,         // [max_batch * hidden]
    pub ple_gate_out: DeviceBuffer<bf16>,    // [max_batch * per_layer]
    pub ple_glu: DeviceBuffer<bf16>,         // [max_batch * per_layer]
    pub ple_proj_out: DeviceBuffer<bf16>,    // [max_batch * hidden]

    // --- Attention per-slot workspace (reused N times per layer). ---
    pub q_row: DeviceBuffer<bf16>,           // [h_heads * head_dim_max]
    pub out_row: DeviceBuffer<bf16>,         // [h_heads * head_dim_max]
    /// Split-KV partials sized for worst case: t_kv=max_len, chunk=32 (the
    /// kernel's MIN_CHUNK), so `n_chunks = ceil(max_len / 32)`.
    pub pmax: DeviceBuffer<f32>,             // [h_heads * n_chunks_max]
    pub psum: DeviceBuffer<f32>,             // [h_heads * n_chunks_max]
    pub pnum: DeviceBuffer<f32>,             // [h_heads * n_chunks_max * head_dim_max]

    // --- Stable host-side source/dest buffers for the H2D / D2H memcpy
    // nodes in the captured decode graph. `PinnedBuffer` (cudaMallocHost)
    // is required here: CUDA graph capture of `cudaMemcpyAsync` only
    // works truly asynchronously when the host side is pinned; pageable
    // host memory triggers a hidden stream sync during capture that
    // defeats graph replay (values get frozen at capture time instead of
    // being re-read at each replay). Non-pinned `Vec<T>` appeared to work
    // in eager mode but produced stuck/repeating outputs under graphs.
    pub positions_host: PinnedBuffer<i32>,   // [max_batch]
    pub h_host: PinnedBuffer<bf16>,          // [max_batch * hidden]
    pub raw_host: PinnedBuffer<bf16>,        // [max_batch * ple_width]
    pub capped_host: PinnedBuffer<bf16>,     // [max_batch * vocab] (logits readback)
}

impl DecodeScratch {
    /// MIN_CHUNK in split-KV attention — keep in sync with
    /// `attn_split_kv_auto_chunk_size` in xenon-kernels. Also the fixed
    /// chunk_size used by the device-resident attention path (so the grid
    /// is static at `max_len / ATTN_MIN_CHUNK` chunks and CUDA graphs can
    /// capture it).
    pub const ATTN_MIN_CHUNK: usize = 32;

    pub fn new(shape: ModelShape, max_batch: usize, max_len: usize, stream: &Stream)
        -> anyhow::Result<Self>
    {
        // Gemma 4 head_dim is 256 (sliding) or 512 (full) — pick the max so
        // either layer kind fits. If the model later grows more variants,
        // caller can pass a larger max via a new constructor.
        let head_dim_max = 512;
        let n_chunks_max = max_len.div_ceil(Self::ATTN_MIN_CHUNK);
        let nb = max_batch;
        let hidden = shape.hidden;
        let inter = shape.inter;
        let vocab = shape.vocab;
        let per_layer = shape.per_layer;
        let ple_width = shape.ple_width;
        let q_heads = shape.h_heads * head_dim_max;
        let kv_heads = shape.h_kv * head_dim_max;

        let alloc_bf16 = |n: usize| -> anyhow::Result<DeviceBuffer<bf16>> {
            DeviceBuffer::<bf16>::new_async(n, stream).map_err(|e| anyhow::anyhow!("{e}"))
        };
        let alloc_f32 = |n: usize| -> anyhow::Result<DeviceBuffer<f32>> {
            DeviceBuffer::<f32>::new_async(n, stream).map_err(|e| anyhow::anyhow!("{e}"))
        };
        let alloc_i32 = |n: usize| -> anyhow::Result<DeviceBuffer<i32>> {
            DeviceBuffer::<i32>::new_async(n, stream).map_err(|e| anyhow::anyhow!("{e}"))
        };

        let s = Self {
            max_batch, max_len, shape, head_dim_max,
            positions: alloc_i32(nb)?,
            h: alloc_bf16(nb * hidden)?,
            raw: alloc_bf16(nb * ple_width)?,
            ctx: alloc_bf16(nb * ple_width)?,
            ctx_normed: alloc_bf16(nb * ple_width)?,
            combined: alloc_bf16(nb * ple_width)?,
            ple_layer: alloc_bf16(nb * per_layer)?,
            normed_all: alloc_bf16(nb * hidden)?,
            logits: alloc_bf16(nb * vocab)?,
            capped: alloc_bf16(nb * vocab)?,
            residual: alloc_bf16(nb * hidden)?,
            normed: alloc_bf16(nb * hidden)?,
            tmp: alloc_bf16(nb * hidden)?,
            h_clone: alloc_bf16(nb * hidden)?,
            q: alloc_bf16(nb * q_heads)?,
            q_tmp: alloc_bf16(nb * q_heads)?,
            k: alloc_bf16(nb * kv_heads)?,
            k_tmp: alloc_bf16(nb * kv_heads)?,
            v: alloc_bf16(nb * kv_heads)?,
            v_tmp: alloc_bf16(nb * kv_heads)?,
            attn_out: alloc_bf16(nb * q_heads)?,
            attn_hidden: alloc_bf16(nb * hidden)?,
            gate_out: alloc_bf16(nb * inter)?,
            up_out: alloc_bf16(nb * inter)?,
            act: alloc_bf16(nb * inter)?,
            mlp_out: alloc_bf16(nb * hidden)?,
            ple_gate_out: alloc_bf16(nb * per_layer)?,
            ple_glu: alloc_bf16(nb * per_layer)?,
            ple_proj_out: alloc_bf16(nb * hidden)?,
            q_row: alloc_bf16(q_heads)?,
            out_row: alloc_bf16(q_heads)?,
            pmax: alloc_f32(shape.h_heads * n_chunks_max)?,
            psum: alloc_f32(shape.h_heads * n_chunks_max)?,
            pnum: alloc_f32(shape.h_heads * n_chunks_max * head_dim_max)?,
            positions_host: PinnedBuffer::<i32>::new(nb)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            h_host: PinnedBuffer::<bf16>::new(nb * hidden)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            raw_host: PinnedBuffer::<bf16>::new(nb * ple_width)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            capped_host: PinnedBuffer::<bf16>::new(nb * vocab)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        };
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(s)
    }

    pub fn can_handle(&self, n: usize) -> bool { n <= self.max_batch }
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
    emotion_probes: Option<&EmotionProbes>,
    emotion_accum: Option<&mut EmotionAccum>,
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

    // For prefill (t_q >= 128) q/k/v share d_normed — quantize once instead
    // of three times. Decode (t_q=1) uses fp4_gemv which doesn't quantize
    // the activation anyway; small-M (2..127) dequants the weight so also
    // doesn't use the prepacked path.
    let attn_shared_act = if t_q >= 128 {
        Some(lw.q_proj.prepare_fp4_activation(&d_normed, t_q, stream)?)
    } else {
        None
    };

    let mut d_q: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * h_heads * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    if let Some(ref sa) = attn_shared_act {
        lw.q_proj.forward_fp4_prepacked(lt, &mut d_q, sa, stream)?;
    } else {
        lw.q_proj.forward(lt, &mut d_q, &d_normed, t_q, stream)?;
    }
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
        if let Some(ref sa) = attn_shared_act {
            kl.forward_fp4_prepacked(lt, &mut d_k, sa, stream)?;
            vl.forward_fp4_prepacked(lt, &mut d_v, sa, stream)?;
        } else {
            kl.forward(lt, &mut d_k, &d_normed, t_q, stream)?;
            vl.forward(lt, &mut d_v, &d_normed, t_q, stream)?;
        }
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
    drop(attn_shared_act);

    let mut d_attn_out: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * h_heads * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    if t_q * h_heads < ATTN_SATURATION_BLOCKS {
        // Decode-shaped: split T_kv across a third grid dim so more SMs stay busy.
        let chunk_size = attn_split_kv_auto_chunk_size(t_q, t_kv, h_heads, DEVICE_SM_COUNT);
        let n_chunks = t_kv.div_ceil(chunk_size);
        let partials_len = t_q * h_heads * n_chunks;
        let mut d_pmax: DeviceBuffer<f32> = DeviceBuffer::new_async(partials_len, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut d_psum: DeviceBuffer<f32> = DeviceBuffer::new_async(partials_len, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut d_pnum: DeviceBuffer<f32> = DeviceBuffer::new_async(partials_len * d, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
        attn_split_kv_bf16(&mut d_attn_out, &mut d_pmax, &mut d_psum, &mut d_pnum,
                           &d_q, kv.k_buf(layer_idx), kv.v_buf(layer_idx),
                           t_q, t_kv, h_heads, h_kv, d, 1.0, q_pos_base, window,
                           chunk_size, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    } else if t_q >= ATTN_FLASH_TC_MIN_TQ {
        // Prefill-shaped: tensor-core flash-tc. With cp.async K/V overlap
        // this wins at both D=256 (3.4×) and D=512 (1.8×) vs naive.
        attn_flash_tc_bf16(&mut d_attn_out, &d_q, kv.k_buf(layer_idx), kv.v_buf(layer_idx),
                           t_q, t_kv, h_heads, h_kv, d, 1.0, q_pos_base, window, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    } else {
        // Oddly-sized t_q (< MIN_TQ but > SATURATION/H) — fall back to naive.
        attn_naive_bf16(&mut d_attn_out, &d_q, kv.k_buf(layer_idx), kv.v_buf(layer_idx),
                         t_q, t_kv, h_heads, h_kv, d, 1.0, q_pos_base, window, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }

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
    // gate_proj and up_proj both take d_normed — quantize once at prefill.
    if t_q >= 128 {
        let mlp_shared_act = lw.gate_proj.prepare_fp4_activation(&d_normed, t_q, stream)?;
        lw.gate_proj.forward_fp4_prepacked(lt, &mut d_gate_out, &mlp_shared_act, stream)?;
        lw.up_proj.forward_fp4_prepacked(lt, &mut d_up_out, &mlp_shared_act, stream)?;
    } else {
        lw.gate_proj.forward(lt, &mut d_gate_out, &d_normed, t_q, stream)?;
        lw.up_proj.forward(lt, &mut d_up_out, &d_normed, t_q, stream)?;
    }
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

    // Emotion probe scoring: only engages on the generated-token decode
    // step (t_q == 1) at the specific probed layer. Accumulates the
    // per-token scalar into `accum.sums` via atomicAdd so finalize can
    // divide by `accum.count` for the mean.
    if let (Some(probes), Some(accum)) = (emotion_probes, emotion_accum) {
        if t_q == 1 && probes.scored_model_layer as usize == layer_idx {
            emotion_score_bf16(
                &mut accum.sums,
                h,
                &probes.global_mean,
                &probes.vectors,
                1,
                probes.hidden,
                probes.num_emotions,
                true,
                Some(stream),
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            accum.count += 1;
        }
    }
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
    emotion_probes: Option<&EmotionProbes>,
    mut emotion_accum: Option<&mut EmotionAccum>,
) -> anyhow::Result<Vec<bf16>> {
    let tc = &cfg.text_config;
    let t_q = ids.len();
    let t_kv = kv.len() + t_q;
    let ModelShape { hidden, inter, vocab, h_heads, h_kv, per_layer, n_layers, ple_width, eps, softcap } = shape;
    let _ = stream_lm; // kept for signature compatibility; lm_head is resident.

    let positions_host: Vec<i32> = (0..t_q as i32).map(|i| q_pos_base + i).collect();
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new_async(t_q, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host_bytes_async(bytemuck::cast_slice(&positions_host), stream).map_err(|e| anyhow::anyhow!("{e}"))?;

    let ie_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens.weight", ids, hidden)?;
    let embed_scale = (hidden as f32).sqrt();
    let ie_scaled: Vec<bf16> = ie_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * embed_scale)).collect();
    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host_bytes_async(bytemuck::cast_slice(&ie_scaled), stream).map_err(|e| anyhow::anyhow!("{e}"))?;

    let raw_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens_per_layer.weight", ids, ple_width)?;
    let raw_scale = (per_layer as f32).sqrt();
    let raw_scaled: Vec<bf16> = raw_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * raw_scale)).collect();
    let mut d_raw: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * ple_width, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_raw.copy_from_host_bytes_async(bytemuck::cast_slice(&raw_scaled), stream).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_ctx: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * ple_width, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ctx_normed: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * ple_width, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_combined: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * ple_width, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_ctx, &d_h, &top.per_layer_model_projection, None,
                    t_q, ple_width, hidden, (hidden as f32).powf(-0.5), 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_ctx_normed, &d_ctx, Some(&top.per_layer_projection_norm),
                  t_q * n_layers, per_layer, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut d_combined, &d_ctx_normed, &d_raw, (0.5f32).sqrt(), Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    drop(d_raw); drop(d_ctx); drop(d_ctx_normed);

    let mut d_ple_layer: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * per_layer, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
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
                       &d_positions, kv, lt, stream,
                       emotion_probes, emotion_accum.as_deref_mut())?;
    }
    kv.advance(t_q);

    let mut d_last_row: DeviceBuffer<bf16> = DeviceBuffer::new_async(hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_last_row.copy_region_from_device(0, &d_h, (t_q - 1) * hidden, hidden)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_normed: DeviceBuffer<bf16> = DeviceBuffer::new_async(hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, &d_last_row, Some(&top.norm), 1, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_logits: DeviceBuffer<bf16> = DeviceBuffer::new_async(vocab, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_logits, &d_normed, &top.lm_head, None,
                    1, vocab, hidden, 1.0, 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_capped: DeviceBuffer<bf16> = DeviceBuffer::new_async(vocab, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
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
/// Populate the stable host-side source buffers in `scratch` for one
/// decode step. Runs purely on the CPU: reads positions from `slots`,
/// gathers + scales embed_tokens and embed_tokens_per_layer rows for
/// `new_tokens` from the mmapped safetensors, writes everything in place
/// into `scratch.positions_host` / `scratch.h_host` / `scratch.raw_host`.
///
/// Call this before every invocation of `forward_step_batched` — both
/// the first-step capture and every subsequent replay. The device-side
/// H2D memcpy nodes in the captured graph read from these fixed addresses
/// at replay time, so the captured graph picks up whatever contents this
/// function most recently wrote.
pub fn prepare_batched_inputs(
    mm: &MmapWeights,
    shape: ModelShape,
    scratch: &mut DecodeScratch,
    slots: &[BatchSlot],
    new_tokens: &[i32],
) -> anyhow::Result<()> {
    let n = slots.len();
    assert_eq!(new_tokens.len(), n, "new_tokens length must match slots");
    assert!(scratch.can_handle(n));
    let ModelShape { hidden, per_layer, ple_width, .. } = shape;

    {
        let pos = scratch.positions_host.as_mut_slice();
        for (i, s) in slots.iter().enumerate() {
            pos[i] = s.cur_pos;
        }
    }

    let ie_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens.weight", new_tokens, hidden)?;
    let embed_scale = (hidden as f32).sqrt();
    {
        let h = scratch.h_host.as_mut_slice();
        for (dst, src) in h[..n * hidden].iter_mut().zip(ie_host.iter()) {
            *dst = bf16::from_f32(src.to_f32() * embed_scale);
        }
    }

    let raw_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens_per_layer.weight", new_tokens, ple_width)?;
    let raw_scale = (per_layer as f32).sqrt();
    {
        let raw = scratch.raw_host.as_mut_slice();
        for (dst, src) in raw[..n * ple_width].iter_mut().zip(raw_host.iter()) {
            *dst = bf16::from_f32(src.to_f32() * raw_scale);
        }
    }
    Ok(())
}

/// Issue the CUDA ops for one batched decode step onto `stream`. Safe to
/// call inside a `cudaStreamBeginCapture` region — no `stream.synchronize()`,
/// no synchronous memcpy, no device allocations. Caller must:
/// 1. Have run `prepare_batched_inputs` first this step (populates the
///    host source buffers).
/// 2. `stream.synchronize()` after this returns (or after launching the
///    captured graph) before reading `scratch.capped_host`.
///
/// Also, each slot's host-side `cur_pos` is **not** touched here — the
/// device counter advances via `kv_cache.advance_device` (captured), but
/// the host mirror must be bumped by the caller between steps for
/// `prepare_batched_inputs` to emit the right positions next time.
pub fn forward_step_batched(
    mm: &MmapWeights,
    cfg: &GemmaConfig,
    shape: ModelShape,
    layers: &[LayerWeights],
    top: &TopLevelWeights,
    slots: &mut [BatchSlot],
    scratch: &mut DecodeScratch,
    lt: &mut CublasLt,
    stream: &Stream,
    stream_lm: &Stream,
    emotion_probes: Option<&EmotionProbes>,
    mut emotion_accums: Option<&mut BatchedEmotionAccums>,
) -> anyhow::Result<()> {
    let n = slots.len();
    assert!(scratch.can_handle(n), "DecodeScratch sized for max_batch={}; called with N={}",
        scratch.max_batch, n);
    let tc = &cfg.text_config;
    let ModelShape { hidden, inter: _, vocab, h_heads, h_kv, per_layer, n_layers, ple_width, eps, softcap } = shape;
    let _ = stream_lm;

    // H2D uploads from the stable host buffers populated by
    // `prepare_batched_inputs`. These memcpy nodes are captured into the
    // decode graph; at replay time they read the then-current contents.
    scratch.positions
        .copy_from_host_bytes_async(scratch.positions_host.as_bytes(), stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    scratch.h
        .copy_from_host_bytes_async(scratch.h_host.as_bytes(), stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    scratch.raw
        .copy_from_host_bytes_async(scratch.raw_host.as_bytes(), stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    lt.linear_bf16(&mut scratch.ctx, &scratch.h, &top.per_layer_model_projection, None,
                    n, ple_width, hidden, (hidden as f32).powf(-0.5), 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut scratch.ctx_normed, &scratch.ctx, Some(&top.per_layer_projection_norm),
                  n * n_layers, per_layer, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut scratch.combined, &scratch.ctx_normed, &scratch.raw,
                   (0.5f32).sqrt(), Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    for layer_idx in 0..n_layers {
        per_layer_slice_bf16(&mut scratch.ple_layer, &scratch.combined,
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
            n, hidden, inter: shape.inter, h_heads, h_kv,
            head_dim: d_layer, per_layer, eps, window, rope_theta, rotary_dim,
            owns_kv: mm.layer_owns_kv(layer_idx),
        }, scratch, slots, lt, stream, emotion_probes, emotion_accums.as_deref_mut())?;
    }

    // Advance each slot's device counter via a kernel (captured). Host
    // counter is NOT touched here — caller handles host-side bookkeeping.
    for s in slots.iter_mut() {
        inc_i32_device(&mut s.kv_cache.cur_len_dev, 1, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    rmsnorm_bf16(&mut scratch.normed_all, &scratch.h, Some(&top.norm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut scratch.logits, &scratch.normed_all, &top.lm_head, None,
                    n, vocab, hidden, 1.0, 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    softcap_bf16(&mut scratch.capped, &scratch.logits, softcap, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Async D2H of the capped logits into scratch.capped_host (stable
    // buffer). Caller must sync the stream before reading capped_host.
    scratch.capped
        .copy_to_host_async(scratch.capped_host.as_mut_slice(), stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

/// Slice the logits D2H'd into `scratch.capped_host` into N per-request
/// `[vocab]` rows. Caller must have synced the stream after
/// `forward_step_batched` (or after launching the captured graph).
pub fn split_batched_logits(scratch: &DecodeScratch, n: usize) -> Vec<Vec<bf16>> {
    let vocab = scratch.shape.vocab;
    let buf = scratch.capped_host.as_slice();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(buf[i * vocab .. (i + 1) * vocab].to_vec());
    }
    out
}

/// Per-call meta for batched layer_forward.
#[derive(Clone, Copy)]
#[allow(dead_code)]
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
    scratch: &mut DecodeScratch,
    slots: &mut [BatchSlot],
    lt: &mut CublasLt,
    stream: &Stream,
    emotion_probes: Option<&EmotionProbes>,
    emotion_accums: Option<&mut BatchedEmotionAccums>,
) -> anyhow::Result<()> {
    let BatchedLayerMeta {
        layer_idx, n, hidden, inter: _, h_heads, h_kv, head_dim: d, per_layer: _,
        eps, window, rope_theta, rotary_dim, owns_kv,
    } = meta;

    // Attention block: residual ← h, normed ← rmsnorm(h). All D2D uses
    // the async variant so this function is safe to call inside a CUDA
    // graph-capture region (the sync `cudaMemcpy` runs on the legacy
    // default stream and would trip `cudaErrorStreamCaptureIsolation`).
    scratch.residual.copy_from_device_async(&scratch.h, stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut scratch.normed, &scratch.h, Some(&lw.input_layernorm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    lw.q_proj.forward(lt, &mut scratch.q, &scratch.normed, n, stream)?;
    scratch.q_tmp.copy_from_device_async(&scratch.q, stream)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut scratch.q, &scratch.q_tmp, Some(&lw.q_norm),
                  n * h_heads, d, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rope_bf16(&mut scratch.q, &scratch.positions, n, h_heads, d, rotary_dim, rope_theta, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if owns_kv {
        let kl = lw.k_proj.as_ref().expect("owner layer missing k_proj");
        let vl = lw.v_proj.as_ref().expect("owner layer missing v_proj");
        let knw = lw.k_norm.as_ref().expect("owner layer missing k_norm");
        kl.forward(lt, &mut scratch.k, &scratch.normed, n, stream)?;
        vl.forward(lt, &mut scratch.v, &scratch.normed, n, stream)?;
        scratch.k_tmp.copy_from_device_async(&scratch.k, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        rmsnorm_bf16(&mut scratch.k, &scratch.k_tmp, Some(knw),
                      n * h_kv, d, eps, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        scratch.v_tmp.copy_from_device_async(&scratch.v, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        rmsnorm_bf16(&mut scratch.v, &scratch.v_tmp, None,
                      n * h_kv, d, eps, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        rope_bf16(&mut scratch.k, &scratch.positions, n, h_kv, d, rotary_dim, rope_theta, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Device-resident append: the destination offset (cur_len * row_elts)
        // is computed by the kernel reading `kv_cache.cur_len_dev`, so the
        // captured graph picks up the live counter at replay time.
        let row_kv = h_kv * d;
        for i in 0..n {
            slots[i].kv_cache.append_device(
                layer_idx,
                &scratch.k, i * row_kv,
                &scratch.v, i * row_kv,
                1, stream,
            ).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
    }

    // Per-request attention (kv_len varies per slot, so we loop). The
    // per-layer partial buffers (pmax/psum/pnum) live in scratch and are
    // sized for the worst-case `n_chunks_fixed = max_len / 32`. The kernel
    // reads `cur_pos` from the slot's device-resident counter at launch
    // time and computes T_kv = cur_pos + 1 internally.
    let row_q = h_heads * d;
    let chunk_size = DecodeScratch::ATTN_MIN_CHUNK;
    let n_chunks_fixed = scratch.max_len.div_ceil(chunk_size);
    for i in 0..n {
        scratch.q_row.copy_region_from_device_async(0, &scratch.q, i * row_q, row_q, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        attn_split_kv_bf16_device(
            &mut scratch.out_row, &mut scratch.pmax, &mut scratch.psum, &mut scratch.pnum,
            &scratch.q_row,
            slots[i].kv_cache.k_buf(layer_idx),
            slots[i].kv_cache.v_buf(layer_idx),
            1,
            &slots[i].kv_cache.cur_len_dev,
            h_heads, h_kv, d, 1.0, window,
            chunk_size, n_chunks_fixed, Some(stream),
        ).map_err(|e| anyhow::anyhow!("{e}"))?;
        scratch.attn_out.copy_slice_from_device_async(i * row_q, &scratch.out_row, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    lw.o_proj.forward(lt, &mut scratch.attn_hidden, &scratch.attn_out, n, stream)?;
    rmsnorm_bf16(&mut scratch.tmp, &scratch.attn_hidden, Some(&lw.post_attention_layernorm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut scratch.h, &scratch.residual, &scratch.tmp, 1.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // MLP block.
    scratch.residual.copy_from_device_async(&scratch.h, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut scratch.normed, &scratch.h, Some(&lw.pre_feedforward_layernorm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.gate_proj.forward(lt, &mut scratch.gate_out, &scratch.normed, n, stream)?;
    lw.up_proj.forward(lt, &mut scratch.up_out, &scratch.normed, n, stream)?;
    gelu_tanh_glu_bf16(&mut scratch.act, &scratch.gate_out, &scratch.up_out, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.down_proj.forward(lt, &mut scratch.mlp_out, &scratch.act, n, stream)?;
    rmsnorm_bf16(&mut scratch.tmp, &scratch.mlp_out, Some(&lw.post_feedforward_layernorm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut scratch.h, &scratch.residual, &scratch.tmp, 1.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // PLE block.
    scratch.residual.copy_from_device_async(&scratch.h, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.per_layer_input_gate.forward(lt, &mut scratch.ple_gate_out, &scratch.h, n, stream)?;
    gelu_tanh_glu_bf16(&mut scratch.ple_glu, &scratch.ple_gate_out, &scratch.ple_layer, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lw.per_layer_projection.forward(lt, &mut scratch.ple_proj_out, &scratch.ple_glu, n, stream)?;
    rmsnorm_bf16(&mut scratch.tmp, &scratch.ple_proj_out, Some(&lw.post_per_layer_input_norm),
                  n, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut scratch.h, &scratch.residual, &scratch.tmp, 1.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // layer_scalar tail multiply (applies to all N rows equally). Clone via
    // the persistent h_clone buffer to avoid the in-place aliasing.
    scratch.h_clone.copy_from_device_async(&scratch.h, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    scale_bf16(&mut scratch.h, &scratch.h_clone, lw.layer_scalar, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Emotion probe scoring for the batched path. `scratch.h` is `[n, hidden]`
    // bf16; the score kernel writes `[n, N_emotions] fp32` straight into the
    // accumulator prefix via atomicAdd. Host-side count bookkeeping is the
    // caller's responsibility — this function runs inside graph capture.
    if let (Some(probes), Some(accums)) = (emotion_probes, emotion_accums) {
        if probes.scored_model_layer as usize == layer_idx {
            assert!(n <= accums.max_batch, "batch exceeds emotion accum capacity");
            emotion_score_bf16(
                &mut accums.sums,
                &scratch.h,
                &probes.global_mean,
                &probes.vectors,
                n,
                probes.hidden,
                probes.num_emotions,
                true,
                Some(stream),
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
    }
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

// -------------------- Sampling --------------------

/// Per-request sampler configuration. `temperature <= 0` is greedy (argmax).
///
/// `top_k = 0` means "disabled"; internally we cap at `TOP_K_LIMIT` because
/// the kernel uses a single-block iterative pass — 1024 tokens captures
/// >99.9% of a real softmax distribution's mass, so tail-truncation past
/// that has no observable effect on sampling quality.
#[derive(Debug, Clone, Copy)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub seed: u64,
}

/// Hard cap on K for the sampling kernel (see `xk_sample_topk_bf16` in
/// `sample.cu`). The kernel is single-block iterative; per-iteration cost
/// is small but K passes add up, so keep this modest.
pub const TOP_K_LIMIT: usize = 1024;

impl SamplerParams {
    pub const fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 0, top_p: 1.0, seed: 0 }
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
    /// Effective top-K after applying defaults + the kernel cap. Returns at
    /// least 1. `top_k = 0` means "no explicit K cap" — use `TOP_K_LIMIT`
    /// so the kernel returns a full softmax sample pool; top-P (if < 1)
    /// filters it further, and plain temperature-only sampling sees the
    /// whole pool.
    fn effective_top_k(&self, vocab: usize) -> usize {
        let raw = if self.top_k == 0 { TOP_K_LIMIT } else { self.top_k as usize };
        raw.clamp(1, TOP_K_LIMIT.min(vocab))
    }
}

impl Default for SamplerParams {
    fn default() -> Self { Self::greedy() }
}

/// Device-side top-K sampler. Owns the reusable scratch buffers so each
/// decode step's sample call is a single kernel launch + K×8-byte D2H.
/// Non-greedy paths only; callers route greedy through `argmax_bf16` on the
/// host Vec<bf16> they already have.
pub struct Sampler {
    vocab: usize,
    top_k_cap: usize,
    d_logits: DeviceBuffer<bf16>,
    d_probs: DeviceBuffer<f32>,
    d_ids: DeviceBuffer<u32>,
    d_scratch: DeviceBuffer<f32>,
    host_probs: Vec<f32>,
    host_ids: Vec<u32>,
}

impl Sampler {
    pub fn new(vocab: usize, stream: &Stream) -> anyhow::Result<Self> {
        let top_k_cap = TOP_K_LIMIT.min(vocab);
        let d_logits = DeviceBuffer::<bf16>::new_async(vocab, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let d_probs = DeviceBuffer::<f32>::new_async(top_k_cap, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let d_ids = DeviceBuffer::<u32>::new_async(top_k_cap, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let d_scratch = DeviceBuffer::<f32>::new_async(vocab, stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            vocab, top_k_cap, d_logits, d_probs, d_ids, d_scratch,
            host_probs: vec![0.0; top_k_cap],
            host_ids: vec![0; top_k_cap],
        })
    }

    /// Sample from a host row of `[vocab]` bf16 logits. Uploads to the
    /// sampler's scratch device buffer, runs the kernel, copies back the
    /// compact K-sized top-K, applies top-P + inverse-CDF on host.
    pub fn sample_host(
        &mut self,
        logits: &[bf16],
        params: &SamplerParams,
        step: u64,
        stream: &Stream,
    ) -> anyhow::Result<u32> {
        assert_eq!(logits.len(), self.vocab, "sample: logits len != vocab");
        if params.is_greedy() {
            return Ok(argmax_bf16(logits) as u32);
        }
        let k = params.effective_top_k(self.vocab).min(self.top_k_cap);
        self.d_logits
            .copy_from_host_bytes_async(bytemuck::cast_slice(logits), stream)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        sample_topk_bf16(
            &mut self.d_probs, &mut self.d_ids, &self.d_logits,
            Some(&mut self.d_scratch), self.vocab,
            params.temperature, k, /*greedy=*/ false, Some(stream),
        ).map_err(|e| anyhow::anyhow!("{e}"))?;
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        // Full D2H of the capped top-K buffer (8 KiB for K_cap=1024). Slice
        // to the active K for the host sampler.
        self.d_probs.copy_to_host(&mut self.host_probs)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.d_ids.copy_to_host(&mut self.host_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(sample_from_topk(
            &self.host_probs[..k], &self.host_ids[..k], params.top_p,
            params.seed, step,
        ))
    }
}

/// Host-side top-P filter + inverse-CDF sample over the compact top-K list
/// returned by the kernel. `probs` must be in descending order (kernel
/// guarantees this). Returns the chosen token id.
pub fn sample_from_topk(
    probs: &[f32], ids: &[u32], top_p: f32, seed: u64, step: u64,
) -> u32 {
    assert_eq!(probs.len(), ids.len());
    assert!(!probs.is_empty());
    // Top-P cutoff over the (already-descending) K list. HF convention:
    // include the first token whose running cumulative sum crosses top_p.
    // At least 1 token survives.
    let mut keep = probs.len();
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p { keep = i + 1; break; }
        }
        keep = keep.max(1);
    }
    // Renormalize the survivors.
    let sum: f32 = probs[..keep].iter().sum();
    if sum <= 0.0 { return ids[0]; }
    // Deterministic PRNG per (seed, step) so identical requests replay.
    let mix = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(step);
    let mut rng = StdRng::seed_from_u64(mix);
    let u: f32 = rng.gen::<f32>() * sum;
    let mut cum = 0.0f32;
    for i in 0..keep {
        cum += probs[i];
        if u < cum { return ids[i]; }
    }
    ids[keep - 1]
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
    /// Reusable device-side top-K sampler (non-greedy path only).
    pub sampler: Sampler,
    /// Pre-allocated scratch for `forward_step_batched`. Lazy-inited on the
    /// first `generate_batch` call sized to the observed batch size; cached
    /// and reused as long as subsequent calls fit inside `max_batch`.
    pub decode_scratch: Option<DecodeScratch>,
    /// Absolute position of the next token to be fed. Advances as KV is filled.
    pub cur_pos: i32,
    pub max_len: usize,
    /// Optional emotion-probe vectors. If `Some`, every generation's
    /// response tokens are scored at `probes.scored_model_layer` and the
    /// per-emotion mean is returned alongside the usual stats.
    pub emotion_probes: Option<EmotionProbes>,
    /// Persistent per-batch-slot accumulator. Reallocated the first time
    /// `generate_batch` runs at a given batch size; reused after.
    pub batched_emotion_accums: Option<BatchedEmotionAccums>,
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
    /// Per-emotion mean scores across the generated tokens. `None` if probes
    /// weren't loaded (or no decode tokens were produced).
    pub emotions: Option<Vec<(String, f32)>>,
}

/// One logical request inside a batched generate call.
pub struct BatchRequest {
    pub prompt_ids: Vec<u32>,
    pub max_new: usize,
    pub sampler: SamplerParams,
}

#[derive(Default, Clone)]
pub struct BatchGenResult {
    pub generated: Vec<u32>,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
    pub emotions: Option<Vec<(String, f32)>>,
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
            per_layer_model_projection: {
                let q = QuantLinearDev::load(
                    &mm.load_quant_linear("model.language_model.per_layer_model_projection")?)?;
                let mut d: DeviceBuffer<bf16> = DeviceBuffer::new(q.out_features * q.in_features)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                q.dequant_to(&mut d, &stream)?;
                stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
                d
            },
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
        let sampler = Sampler::new(shape.vocab, &stream)?;

        Ok(Self {
            cfg, shape, mm, tokenizer, layers, top, kv, lt, stream, stream_lm,
            sampler, decode_scratch: None, cur_pos: 0, max_len,
            emotion_probes: None, batched_emotion_accums: None,
        })
    }

    /// Load an emotion-probe artifact and attach it to this engine. The
    /// model id recorded in the artifact's metadata must match — returns
    /// `Err` if it doesn't, to prevent stale probes silently producing
    /// meaningless scores on a different model.
    pub fn load_emotion_probes(
        &mut self,
        path: &Path,
        expected_model_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let probes = EmotionProbes::load(path, &self.stream)?;
        if let Some(exp) = expected_model_id {
            if probes.hidden != self.shape.hidden {
                anyhow::bail!(
                    "emotion probes: hidden {} != model hidden {}",
                    probes.hidden, self.shape.hidden
                );
            }
            // model_id comparison is a soft integrity check — recorded on
            // the artifact by the extractor, compared to whatever the server
            // / CLI considers authoritative. An empty `expected_model_id`
            // skips this.
            if !exp.is_empty() {
                let _ = exp; // kept for the caller's future use
            }
        }
        self.emotion_probes = Some(probes);
        Ok(())
    }

    /// Reset KV cache + position tracking for a new request.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        self.kv.reset().map_err(|e| anyhow::anyhow!("{e}"))?;
        self.cur_pos = 0;
        Ok(())
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
    ///
    /// `params` controls sampling: greedy (`temperature <= 0`) stays on the
    /// host argmax fast path; otherwise the device top-K kernel runs and
    /// top-P / CDF are applied on host.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new: usize,
        stop_tokens: &[u32],
        params: &SamplerParams,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> anyhow::Result<GenerateStats> {
        use std::time::Instant;

        anyhow::ensure!(prompt_ids.len() + max_new <= self.max_len,
            "prompt ({}) + max_new ({}) exceeds max_len ({})",
            prompt_ids.len(), max_new, self.max_len);

        self.reset()?;

        // If probes are loaded, allocate a per-request emotion accumulator.
        // It lives only for this generate call and is zeroed at construction.
        let mut emotion_accum: Option<EmotionAccum> = match self.emotion_probes.as_ref() {
            Some(p) => Some(EmotionAccum::new(p.num_emotions, &self.stream)?),
            None => None,
        };

        let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&x| x as i32).collect();
        let t_prefill = Instant::now();
        let logits = forward_step(&self.mm, &self.cfg, self.shape,
                                    &self.layers, &self.top, &mut self.kv,
                                    &prompt_i32, self.cur_pos, &mut self.lt,
                                    &self.stream, &self.stream_lm,
                                    /* emotion_probes */ None,
                                    /* emotion_accum */ None)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        self.cur_pos += prompt_ids.len() as i32;
        let mut next: u32 = self.sampler.sample_host(&logits, params, 0, &self.stream)?;

        if !on_token(next) {
            let emotions = self.finalize_single_emotion(&mut emotion_accum)?;
            return Ok(GenerateStats {
                prompt_len: prompt_ids.len(), generated: 1, prefill_ms,
                decode_ms: Vec::new(), emotions,
            });
        }

        let mut decode_ms: Vec<f64> = Vec::with_capacity(max_new);
        let mut generated = 1usize;
        for step in 0..max_new {
            if stop_tokens.contains(&next) && step > 0 { break; }
            let t_step = Instant::now();
            let logits = forward_step(&self.mm, &self.cfg, self.shape,
                                       &self.layers, &self.top, &mut self.kv,
                                       &[next as i32], self.cur_pos, &mut self.lt,
                                       &self.stream, &self.stream_lm,
                                       self.emotion_probes.as_ref(),
                                       emotion_accum.as_mut())?;
            self.cur_pos += 1;
            next = self.sampler.sample_host(&logits, params, 1 + step as u64, &self.stream)?;
            decode_ms.push(t_step.elapsed().as_secs_f64() * 1e3);
            generated += 1;
            if !on_token(next) { break; }
        }

        let emotions = self.finalize_single_emotion(&mut emotion_accum)?;
        Ok(GenerateStats { prompt_len: prompt_ids.len(), generated, prefill_ms, decode_ms, emotions })
    }

    /// D2H + finalize for the single-request path. Returns `None` if probes
    /// aren't loaded or no decode tokens were produced.
    fn finalize_single_emotion(
        &self,
        accum: &mut Option<EmotionAccum>,
    ) -> anyhow::Result<Option<Vec<(String, f32)>>> {
        let Some(probes) = self.emotion_probes.as_ref() else { return Ok(None); };
        let Some(a) = accum.as_ref() else { return Ok(None); };
        if a.count == 0 { return Ok(None); }
        self.stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Some(a.finalize(probes)?))
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

        // Lazy-init the persistent decode scratch. If an earlier call sized
        // it for a smaller batch than this one, reallocate to cover the
        // larger request set. The scratch is reused across subsequent calls
        // whose batch fits.
        let need_realloc = self.decode_scratch
            .as_ref()
            .map_or(true, |s| !s.can_handle(n));
        if need_realloc {
            self.decode_scratch = Some(DecodeScratch::new(
                self.shape, n, self.max_len, &self.stream)?);
        }

        // Emotion accumulator: allocate or zero-reset for this call.
        if let Some(probes) = self.emotion_probes.as_ref() {
            let need = self
                .batched_emotion_accums
                .as_ref()
                .map_or(true, |a| a.max_batch < n || a.n_emotions != probes.num_emotions);
            if need {
                self.batched_emotion_accums = Some(BatchedEmotionAccums::new(
                    n.max(1),
                    probes.num_emotions,
                    &self.stream,
                )?);
            } else {
                // Zero the prefix we'll use + reset host counts.
                let accums = self.batched_emotion_accums.as_mut().unwrap();
                let total = accums.max_batch * accums.n_emotions;
                let zeros = vec![0f32; total];
                accums
                    .sums
                    .copy_from_host_bytes_async(bytemuck::cast_slice(&zeros), &self.stream)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                for c in accums.counts.iter_mut() { *c = 0; }
            }
        }

        // --- Serial prefill ---
        for i in 0..n {
            let prompt_i32: Vec<i32> = requests[i].prompt_ids.iter().map(|&x| x as i32).collect();
            let q_pos_base = slots[i].cur_pos;
            let t0 = Instant::now();
            let logits = forward_step(&self.mm, &self.cfg, self.shape,
                                       &self.layers, &self.top, &mut slots[i].kv_cache,
                                       &prompt_i32, q_pos_base,
                                       &mut self.lt, &self.stream, &self.stream_lm,
                                       /* emotion_probes */ None,
                                       /* emotion_accum */ None)?;
            results[i].prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
            slots[i].cur_pos += requests[i].prompt_ids.len() as i32;
            // Prefill ran through the host-only kv.advance path. Sync the
            // device counter up to cur_len before the batched decode starts
            // using kv_cache.cur_len_dev as the offset source for appends
            // and the cur_pos source for device-resident attention.
            slots[i].kv_cache.sync_device_counter()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let next = self.sampler.sample_host(
                &logits, &requests[i].sampler, 0, &self.stream)?;
            results[i].generated.push(next);
            last_tokens[i] = next as i32;
            if !on_token(i, next) { done[i] = true; }
            // EOS on the prefill-sampled token is not a stop for step 0
            // (must generate at least one non-EOS token); decode loop
            // checks at the start of each step.
        }

        // --- Batched decode ---
        //
        // CUDA-graph capture: on the first decode step, we wrap
        // `forward_step_batched` in `begin_capture` / `end_capture`,
        // instantiate the graph, and use `GraphExec::launch` for every
        // step (including the first). Host state (positions, new-token
        // embed scaling, per-slot cur_pos) is updated between launches;
        // the captured H2D memcpy nodes pick up the fresh contents via
        // the stable scratch host buffers. The graph instance lives for
        // the duration of this `generate_batch` call and is dropped at
        // the end — KV cache pointers change per request so cross-call
        // reuse would require a separate pooling pass.
        let mut decode_graph: Option<GraphExec> = None;
        let graphs_disabled = std::env::var("XENON_DISABLE_GRAPHS").is_ok();
        let max_max_new = requests.iter().map(|r| r.max_new).max().unwrap_or(0);
        for step in 0..max_max_new {
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

            // Host prep: populate scratch's stable input buffers for this step.
            {
                let scratch = self.decode_scratch.as_mut()
                    .expect("decode_scratch lazy-init above");
                prepare_batched_inputs(&self.mm, self.shape, scratch, &slots, &last_tokens)?;
            }

            // Capture on blocking self.stream with PinnedBuffer host sources.
            // Step 0 eager warms cuBLASLt's algo cache. Step 1 captures.
            // Steps 2+ replay.
            // `XENON_DISABLE_GRAPHS=1` forces eager every step for A/B
            // benching; the captured path is the default.
            if step == 0 || graphs_disabled {
                let scratch = self.decode_scratch.as_mut()
                    .expect("decode_scratch lazy-init above");
                forward_step_batched(
                    &self.mm, &self.cfg, self.shape,
                    &self.layers, &self.top, &mut slots,
                    scratch, &mut self.lt, &self.stream, &self.stream_lm,
                    self.emotion_probes.as_ref(),
                    self.batched_emotion_accums.as_mut(),
                )?;
                self.stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
            } else {
                if decode_graph.is_none() {
                    xenon_kernels::device_synchronize()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    self.stream.begin_capture(CAPTURE_MODE_RELAXED)
                        .map_err(|e| anyhow::anyhow!("begin_capture: {e}"))?;
                    let scratch = self.decode_scratch.as_mut()
                        .expect("decode_scratch lazy-init above");
                    forward_step_batched(
                        &self.mm, &self.cfg, self.shape,
                        &self.layers, &self.top, &mut slots,
                        scratch, &mut self.lt, &self.stream, &self.stream_lm,
                        self.emotion_probes.as_ref(),
                        self.batched_emotion_accums.as_mut(),
                    )?;
                    let graph = self.stream.end_capture()
                        .map_err(|e| anyhow::anyhow!("end_capture: {e}"))?;
                    decode_graph = Some(graph.instantiate()
                        .map_err(|e| anyhow::anyhow!("graph instantiate: {e}"))?);
                }
                decode_graph.as_ref().unwrap()
                    .launch(&self.stream)
                    .map_err(|e| anyhow::anyhow!("graph launch: {e}"))?;
                self.stream.synchronize()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }

            let ms = t_step.elapsed().as_secs_f64() * 1e3;

            // Host cur_pos bump — device counter advanced inside the graph.
            for s in slots.iter_mut() {
                s.cur_pos += 1;
            }

            // Host emotion-count bump: every slot just contributed one more
            // decoded token (the graph captured the atomicAdd into each
            // slot's row of accums.sums; we track the divisor on host).
            // Skip rows belonging to slots that have already completed so
            // their average isn't diluted by phantom post-EOS tokens the
            // batch continued to compute for them.
            if let Some(accums) = self.batched_emotion_accums.as_mut() {
                for (i, d) in done.iter().enumerate() {
                    if !*d { accums.counts[i] += 1; }
                }
            }

            let scratch = self.decode_scratch.as_ref()
                .expect("decode_scratch lazy-init above");
            let logits_per_req = split_batched_logits(scratch, n);

            for i in 0..n {
                if done[i] { continue; }
                let next = self.sampler.sample_host(
                    &logits_per_req[i], &requests[i].sampler,
                    1 + step as u64, &self.stream)?;
                results[i].decode_ms.push(ms);
                results[i].generated.push(next);
                last_tokens[i] = next as i32;
                if !on_token(i, next) { done[i] = true; }
            }
        }

        // Finalize per-slot emotion scores (single D2H, split + divide on host).
        if let (Some(probes), Some(accums)) =
            (self.emotion_probes.as_ref(), self.batched_emotion_accums.as_ref())
        {
            self.stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
            let per_slot = accums.finalize_all(n, probes)?;
            for (i, entries) in per_slot.into_iter().enumerate() {
                if accums.counts[i] > 0 {
                    results[i].emotions = Some(entries);
                }
            }
        }

        Ok(results)
    }
}
