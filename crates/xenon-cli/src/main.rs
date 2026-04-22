use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use half::bf16;
use rand::{Rng, SeedableRng};
use xenon_core::{
    config::QuantConfig,
    weights::{is_excluded, MmapWeights, SafetensorsHeader, WeightBreakdown},
    GemmaConfig, LayerKind,
};
use xenon_kernels::{
    attn_naive_bf16, attn_naive_bf16_reference,
    cuda::{device_synchronize, mem_info, Device, DeviceBuffer, Stream},
    embed_gather_bf16, fp4_dequant_bf16, fp4_dequant_bf16_reference, gelu_tanh_bf16,
    gelu_tanh_bf16_reference, gelu_tanh_glu_bf16, linear_bf16_reference, matmul_bf16_reference,
    rmsnorm_bf16, rmsnorm_bf16_reference, rope_bf16, rope_bf16_reference, softmax_attn_bf16,
    softmax_attn_bf16_reference, CublasLt, KvCache, SlotSpec,
};

/// Parse a little-endian bf16 byte slice into a `Vec<bf16>`. Avoids
/// alignment assumptions `bytemuck::cast_slice` would impose on mmap'd data.
fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<bf16> {
    assert!(bytes.len() % 2 == 0, "bf16 bytes must be even-length");
    bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

#[derive(Parser, Debug)]
#[command(name = "xenon-cli", about = "xenon command-line tool")]
struct Args {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print model shape, layer layout, and quantization details.
    Info { model: PathBuf },
    /// Walk the safetensors header and categorize tensors (NVFP4 pairs vs
    /// plain); validate the modelopt exclude invariant.
    Load { model: PathBuf },
    /// Print tensor entries (name, dtype, shape, bytes) matching an optional
    /// glob-ish pattern (trailing `*` supported).
    List {
        model: PathBuf,
        /// Glob pattern; if omitted, prints all.
        #[arg(long)]
        pattern: Option<String>,
        /// Limit on number of lines printed. Set to 0 for unlimited.
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Run the hello kernel to verify the CUDA build pipeline.
    Sanity,
    /// Print CUDA device count and free/total VRAM on device 0.
    Vram,
    /// Run the RMSNorm kernel on random input and compare to a CPU reference.
    TestRmsnorm {
        #[arg(long, default_value_t = 42)]
        rows: usize,
        #[arg(long, default_value_t = 2560)]
        hidden: usize,
        #[arg(long, default_value_t = 1e-6)]
        eps: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Dequantize synthetic NVFP4 bytes on the GPU, compare to a CPU reference.
    TestDequant {
        #[arg(long, default_value_t = 32)]
        rows: usize,
        #[arg(long, default_value_t = 128)]
        cols: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run a bf16 matmul via cuBLASLt and compare to a CPU fp32 reference.
    TestGemm {
        #[arg(long, default_value_t = 128)]
        m: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 128)]
        k: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run a bf16 Linear (y = x @ W^T) via cuBLASLt and compare to a CPU reference.
    TestLinear {
        #[arg(long, default_value_t = 128)]
        m: usize,
        #[arg(long, default_value_t = 128)]
        n: usize,
        #[arg(long, default_value_t = 128)]
        k: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run a full MLP block (norm + gate/up/down + GELU GLU) on real Gemma
    /// weights, comparing GPU output to a host-reference computation that
    /// dequantizes the same FP4 bytes.
    TestMlp {
        model: PathBuf,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run GELU-tanh on random input and compare to a CPU reference.
    TestGelu {
        #[arg(long, default_value_t = 4096)]
        n: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run a full attention block (norm + QKV + qk-norm + RoPE + naive attn +
    /// o_proj) on real Gemma weights for one layer, comparing GPU to host ref.
    TestAttnLayer {
        model: PathBuf,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 4)]
        tokens: usize,
        #[arg(long, default_value_t = 0)]
        q_pos_base: i32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Self-consistency decode test: run prefill for T tokens, extract the
    /// last-row attention output, then re-run attention with (T_q=1,
    /// T_kv=T, q_pos_base=T-1) using the same Q/K/V. The two outputs must
    /// match bit-closely — proves the decode-shape code path computes the
    /// same math as the "last row of prefill".
    TestAttnDecode {
        model: PathBuf,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 4)]
        tokens: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Exercise KvCache: prefill (T-1) tokens via append-to-cache, then one
    /// decode step, and verify the final attention output matches a
    /// reference run that processes all T tokens at once.
    TestKvCache {
        model: PathBuf,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 4)]
        tokens: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Print the KV-sharing map derived from the safetensors header (which
    /// layers own their KV vs reuse from elsewhere).
    KvMap { model: PathBuf },
    /// Gather PLE (per-layer embedding) rows from host mmap for a few token
    /// IDs, upload to device, and verify against a direct host slice. Then
    /// splits out one layer's [T, 256] and runs per_layer_input_gate to
    /// prove the dim pipeline works with real weights.
    TestPle {
        model: PathBuf,
        #[arg(long, default_value = "2,108,107,1")]
        ids: String,
        #[arg(long, default_value_t = 0)]
        layer: usize,
    },
    /// Apply RoPE on random input and compare to a CPU reference.
    TestRope {
        #[arg(long, default_value_t = 4)]
        tokens: usize,
        #[arg(long, default_value_t = 8)]
        heads: usize,
        #[arg(long, default_value_t = 256)]
        head_dim: usize,
        /// Rotary dim; must be <= head_dim and even. Defaults to head_dim.
        #[arg(long)]
        rotary_dim: Option<usize>,
        #[arg(long, default_value_t = 10_000.0)]
        theta: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run masked softmax on random scores and compare to a CPU reference.
    TestSoftmax {
        #[arg(long, default_value_t = 4)]
        rows: usize,
        #[arg(long, default_value_t = 1)]
        t_q: usize,
        #[arg(long, default_value_t = 64)]
        t_kv: usize,
        #[arg(long, default_value_t = 16.0)]
        scale: f32,
        #[arg(long, default_value_t = 0)]
        q_pos_base: i32,
        /// 0 = no sliding window (full causal).
        #[arg(long, default_value_t = 0)]
        window: i32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Gather real token embeddings and verify against host mmap reference.
    TestEmbed {
        model: PathBuf,
        #[arg(long, default_value = "2,108,107")]
        ids: String,
    },
    /// Memory-map the model and upload tensors (matching an optional prefix)
    /// to device memory. Reports bytes, elapsed, and effective GB/s.
    Upload {
        model: PathBuf,
        /// Only upload tensors whose names start with this prefix.
        #[arg(long)]
        prefix: Option<String>,
        /// Stop uploading once cumulative bytes exceed this cap (bytes).
        #[arg(long, default_value_t = 1u64 << 30)]
        limit_bytes: u64,
        /// Verify each upload by copying back and comparing bytes.
        #[arg(long)]
        verify: bool,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match Args::parse().cmd {
        Command::Info { model } => cmd_info(model),
        Command::Load { model } => cmd_load(model),
        Command::List { model, pattern, limit } => cmd_list(model, pattern, limit),
        Command::Sanity => cmd_sanity(),
        Command::Vram => cmd_vram(),
        Command::TestRmsnorm { rows, hidden, eps, seed } => cmd_test_rmsnorm(rows, hidden, eps, seed),
        Command::TestDequant { rows, cols, seed } => cmd_test_dequant(rows, cols, seed),
        Command::TestGemm { m, n, k, seed } => cmd_test_gemm(m, n, k, seed),
        Command::TestLinear { m, n, k, seed } => cmd_test_linear(m, n, k, seed),
        Command::TestGelu { n, seed } => cmd_test_gelu(n, seed),
        Command::TestMlp { model, layer, batch, seed } => cmd_test_mlp(model, layer, batch, seed),
        Command::TestRope { tokens, heads, head_dim, rotary_dim, theta, seed } =>
            cmd_test_rope(tokens, heads, head_dim, rotary_dim.unwrap_or(head_dim), theta, seed),
        Command::TestSoftmax { rows, t_q, t_kv, scale, q_pos_base, window, seed } =>
            cmd_test_softmax(rows, t_q, t_kv, scale, q_pos_base, window, seed),
        Command::TestEmbed { model, ids } => cmd_test_embed(model, ids),
        Command::TestAttnLayer { model, layer, tokens, q_pos_base, seed } =>
            cmd_test_attn_layer(model, layer, tokens, q_pos_base, seed),
        Command::TestAttnDecode { model, layer, tokens, seed } =>
            cmd_test_attn_decode(model, layer, tokens, seed),
        Command::TestKvCache { model, layer, tokens, seed } =>
            cmd_test_kv_cache(model, layer, tokens, seed),
        Command::KvMap { model } => cmd_kv_map(model),
        Command::TestPle { model, ids, layer } => cmd_test_ple(model, ids, layer),
        Command::Upload { model, prefix, limit_bytes, verify } => cmd_upload(model, prefix, limit_bytes, verify),
    }
}

fn cmd_info(dir: PathBuf) -> anyhow::Result<()> {
    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let tc = &cfg.text_config;

    println!("=== xenon-cli info ===");
    println!("path                     {}", dir.display());
    println!("model_type               {}", cfg.model_type);
    if !cfg.architectures.is_empty() {
        println!("architectures            {:?}", cfg.architectures);
    }
    println!();
    println!("-- text config --");
    println!("hidden_size              {}", tc.hidden_size);
    println!("intermediate_size        {}", tc.intermediate_size);
    println!("num_hidden_layers        {}", tc.num_hidden_layers);
    println!("num_attention_heads      {}", tc.num_attention_heads);
    println!("num_key_value_heads      {}", tc.num_key_value_heads);
    println!("head_dim                 {}", tc.head_dim);
    println!("global_head_dim          {:?}", tc.global_head_dim);
    println!("vocab_size               {}", tc.vocab_size);
    println!("max_position_embeddings  {}", tc.max_position_embeddings);
    println!("sliding_window           {:?}", tc.sliding_window);
    println!("num_kv_shared_layers     {:?}", tc.num_kv_shared_layers);
    println!("hidden_size_per_layer    {:?}", tc.hidden_size_per_layer_input);
    println!("vocab_size_per_layer     {:?}", tc.vocab_size_per_layer_input);
    println!("rms_norm_eps             {}", tc.rms_norm_eps);
    println!("final_logit_softcapping  {:?}", tc.final_logit_softcapping);
    println!("hidden_activation        {:?}", tc.hidden_activation);
    println!("tie_word_embeddings      {}", tc.tie_word_embeddings);

    let (mut sliding, mut full) = (0usize, 0usize);
    for k in &tc.layer_types {
        match k {
            LayerKind::SlidingAttention => sliding += 1,
            LayerKind::FullAttention => full += 1,
        }
    }
    println!();
    println!("-- layer layout --");
    println!("  sliding_attention      {}", sliding);
    println!("  full_attention         {}", full);

    let quant_path = dir.join("hf_quant_config.json");
    if quant_path.exists() {
        let q = QuantConfig::from_path(&quant_path)?;
        println!();
        println!("-- quantization --");
        println!("producer                 {} {}", q.producer.name, q.producer.version);
        println!("quant_algo               {}", q.quantization.quant_algo);
        println!("group_size               {}", q.quantization.group_size);
        println!("kv_cache_quant_algo      {:?}", q.quantization.kv_cache_quant_algo);
        println!("exclude_modules          {:?}", q.quantization.exclude_modules);
    }

    let hidden = tc.hidden_size as u64;
    let inter = tc.intermediate_size as u64;
    let n_layers = tc.num_hidden_layers as u64;
    let head_dim = tc.head_dim as u64;
    let n_q_heads = tc.num_attention_heads as u64;
    let n_kv_heads = tc.num_key_value_heads as u64;
    let vocab = tc.vocab_size as u64;

    let qkv_out = n_q_heads * head_dim + 2 * n_kv_heads * head_dim;
    let attn_bytes = n_layers * (hidden * qkv_out + n_q_heads * head_dim * hidden) / 2;
    let mlp_bytes = n_layers * (3 * hidden * inter) / 2;
    let lm_head_bytes = vocab * hidden * 2;
    let total_bytes = attn_bytes + mlp_bytes + lm_head_bytes;

    println!();
    println!("-- decode roofline (rough) --");
    println!("attn weights / token     {:>6.1} MiB", attn_bytes as f64 / 1048576.0);
    println!("mlp  weights / token     {:>6.1} MiB", mlp_bytes as f64 / 1048576.0);
    println!("lm_head (bf16)  / token  {:>6.1} MiB", lm_head_bytes as f64 / 1048576.0);
    println!("total streamed  / token  {:>6.2} GiB", total_bytes as f64 / 1073741824.0);
    println!(
        "  @ 167 GB/s D2D:        {:>6.1} ms/token ({:>5.1} tok/s)",
        total_bytes as f64 / 167e9 * 1e3,
        1.0 / (total_bytes as f64 / 167e9)
    );

    Ok(())
}

fn cmd_load(dir: PathBuf) -> anyhow::Result<()> {
    let st_path = dir.join("model.safetensors");
    if !st_path.exists() {
        anyhow::bail!("expected {} to exist", st_path.display());
    }

    let header = SafetensorsHeader::from_path(&st_path)?;
    let b = WeightBreakdown::from_header(&header);

    println!("=== xenon-cli load ===");
    println!("file                     {}", st_path.display());
    println!("file_bytes               {:.2} GiB ({} bytes)", b.file_bytes as f64 / 1073741824.0, b.file_bytes);
    println!("header_bytes             {:.1} KiB", b.header_bytes as f64 / 1024.0);
    println!("tensor_count             {}", b.tensor_count);

    println!();
    println!("-- bytes by dtype --");
    for (dt, bytes) in &b.bytes_by_dtype {
        let count = b.count_by_dtype.get(dt).copied().unwrap_or(0);
        println!("  {:>10}  {:>5} tensors  {:>8.2} GiB", dt, count, *bytes as f64 / 1073741824.0);
    }

    println!();
    println!("-- NVFP4 weight+scale pairs --");
    println!("  pair count             {}", b.quant_pairs.len());
    println!("  packed FP4 weights     {:>6.2} GiB", b.quant_weight_bytes() as f64 / 1073741824.0);
    println!("  block scales           {:>6.2} GiB", b.quant_scale_bytes() as f64 / 1073741824.0);
    let extra_count = b.quant_pairs.iter().filter(|p| p.extra_scale.is_some()).count();
    println!("  with weight_scale_2    {}", extra_count);

    println!();
    println!("-- plain (unquantized) .weight tensors --");
    println!("  count                  {}", b.plain_weights.len());
    println!("  bytes                  {:>6.2} GiB", b.plain_weight_bytes(&header) as f64 / 1073741824.0);

    if !b.orphan_scales.is_empty() {
        println!();
        println!("!! orphan scales (scale tensor without matching weight) !!");
        for s in &b.orphan_scales {
            println!("   {s}");
        }
    }

    let quant_path = dir.join("hf_quant_config.json");
    if quant_path.exists() {
        let q = QuantConfig::from_path(&quant_path)?;
        let patterns = &q.quantization.exclude_modules;

        let mut quant_violations = Vec::new();
        for pair in &b.quant_pairs {
            let full = format!("{}.weight", pair.module);
            if is_excluded(&full, patterns) {
                quant_violations.push(pair.module.clone());
            }
        }
        let mut plain_violations = Vec::new();
        for name in &b.plain_weights {
            if is_excluded(name, patterns) {
                continue;
            }
            let is_norm = name.contains("norm") || name.contains("layernorm");
            let is_embed = name.contains("embed") || name.contains("embedding");
            if is_norm || is_embed {
                continue;
            }
            plain_violations.push(name.clone());
        }

        println!();
        println!("-- exclude_modules invariant --");
        if quant_violations.is_empty() && plain_violations.is_empty() {
            println!("  OK: excluded modules are plain, everything else is FP4 paired.");
        } else {
            if !quant_violations.is_empty() {
                println!("  WARN: modules marked excluded but quantized:");
                for m in &quant_violations {
                    println!("    {m}");
                }
            }
            if !plain_violations.is_empty() {
                println!("  WARN: non-excluded, non-norm, non-embed plain tensors:");
                for m in &plain_violations {
                    println!("    {m}");
                }
            }
        }
    }

    Ok(())
}

fn cmd_list(dir: PathBuf, pattern: Option<String>, limit: usize) -> anyhow::Result<()> {
    let st_path = dir.join("model.safetensors");
    let header = SafetensorsHeader::from_path(&st_path)?;
    let matches: Vec<(&String, &xenon_core::TensorInfo)> = header
        .tensors
        .iter()
        .filter(|(n, _)| {
            pattern
                .as_ref()
                .map(|p| xenon_core::weights::matches_glob(n, p))
                .unwrap_or(true)
        })
        .collect();
    let total = matches.len();
    let shown = if limit == 0 { total } else { total.min(limit) };

    println!("=== xenon-cli list ===");
    if let Some(p) = &pattern {
        println!("pattern                  {p}");
    }
    println!("matches                  {total} (showing {shown})");
    println!();
    println!("{:<64}  {:>10}  {:>24}  {:>12}", "name", "dtype", "shape", "bytes");
    for (name, info) in matches.iter().take(shown) {
        let shape = format!("{:?}", info.shape);
        println!(
            "{:<64}  {:>10}  {:>24}  {:>12}",
            name, info.dtype, shape, info.bytes()
        );
    }
    Ok(())
}

fn cmd_sanity() -> anyhow::Result<()> {
    let r = xenon_kernels::hello(21).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("xk_hello(21) = {r} (expected 43)");
    anyhow::ensure!(r == 43, "sanity kernel returned unexpected value");
    println!("OK: Rust <-> nvcc <-> CUDA runtime pipeline works.");
    Ok(())
}

fn cmd_vram() -> anyhow::Result<()> {
    let n = Device::count().map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("cuda devices: {n}");
    if n == 0 {
        return Ok(());
    }
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (free, total) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "device 0  free {:>6.2} GiB  total {:>6.2} GiB  used {:>6.2} GiB",
        free as f64 / 1073741824.0,
        total as f64 / 1073741824.0,
        (total - free) as f64 / 1073741824.0
    );
    Ok(())
}

fn cmd_test_rmsnorm(rows: usize, hidden: usize, eps: f32, seed: u64) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..rows * hidden)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect();
    let weight_host: Vec<bf16> = (0..hidden)
        .map(|_| bf16::from_f32(rng.gen_range(0.5..1.5)))
        .collect();

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(rows * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_w: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_y: DeviceBuffer<bf16> = DeviceBuffer::new(rows * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;

    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_w.copy_from_host(&weight_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Warmup + timed run.
    rmsnorm_bf16(&mut d_y, &d_x, &d_w, rows, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let t0 = Instant::now();
    let iters = 50;
    for _ in 0..iters {
        rmsnorm_bf16(&mut d_y, &d_x, &d_w, rows, hidden, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let y_gpu = d_y.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let y_ref = rmsnorm_bf16_reference(&x_host, &weight_host, rows, hidden, eps);
    let (max_abs_diff, max_rel_diff, _global) = compare_bf16(&y_gpu, &y_ref);

    println!("=== xenon-cli test-rmsnorm ===");
    println!("rows x hidden           {} x {}", rows, hidden);
    println!("eps                     {}", eps);
    println!("per-launch time         {:>7.2} us", per_call_us);
    println!("max abs diff vs ref     {:.3e}", max_abs_diff);
    println!("max rel diff vs ref     {:.3e}", max_rel_diff);

    // bf16 has a 7-bit mantissa -> ~1 ULP = 2^-7 ~= 7.8e-3 relative. The GPU
    // and host reductions accumulate in different orders, so the strict test
    // is relative error within a couple of ULPs.
    let tol_rel: f32 = 1e-2;
    if max_rel_diff <= tol_rel {
        println!("OK: within bf16 tolerance (rel {tol_rel:.1e} = ~1 ULP)");
    } else {
        anyhow::bail!("max rel diff {max_rel_diff} exceeds bf16 tolerance {tol_rel}");
    }
    Ok(())
}

/// Compare two bf16 arrays. Returns `(max_abs, max_per_elem_rel, global_rel)`.
/// `global_rel = max_abs / max(|want|)` is the right measure for tensor ops
/// where values can pass through zero; per-elem rel blows up there.
fn compare_bf16(got: &[bf16], want: &[bf16]) -> (f32, f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_per_elem_rel = 0.0f32;
    let mut max_mag = 0.0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        let fa = a.to_f32();
        let fb = b.to_f32();
        let d = (fa - fb).abs();
        if d > max_abs {
            max_abs = d;
        }
        let mb = fb.abs();
        if mb > max_mag {
            max_mag = mb;
        }
        let denom = mb.max(1e-6);
        let r = d / denom;
        if r > max_per_elem_rel {
            max_per_elem_rel = r;
        }
    }
    let global_rel = if max_mag > 0.0 { max_abs / max_mag } else { 0.0 };
    (max_abs, max_per_elem_rel, global_rel)
}

fn cmd_test_dequant(rows: usize, cols: usize, seed: u64) -> anyhow::Result<()> {
    anyhow::ensure!(cols % 16 == 0, "cols must be a multiple of 16");
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Packed FP4: any byte pattern is valid.
    let packed_host: Vec<u8> = (0..rows * cols / 2).map(|_| rng.gen()).collect();
    // Scales: sample UE4M3 byte space but keep sign bit off so every byte is a
    // valid non-negative scale. Restrict to exp in [1, 12] for values that
    // don't overflow bf16 after multiplication with |fp4| up to 6.
    let scales_host: Vec<u8> = (0..rows * cols / 16)
        .map(|_| {
            let exp = rng.gen_range(1u8..=12);
            let man = rng.gen_range(0u8..=7);
            (exp << 3) | man
        })
        .collect();
    let global_scale = 0.125f32 + rng.gen::<f32>() * 0.25f32;

    let mut d_packed: DeviceBuffer<u8> = DeviceBuffer::new(packed_host.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_scales: DeviceBuffer<u8> = DeviceBuffer::new(scales_host.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_out: DeviceBuffer<bf16> = DeviceBuffer::new(rows * cols).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_packed.copy_from_host(&packed_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_scales.copy_from_host(&scales_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    fp4_dequant_bf16(&mut d_out, &d_packed, &d_scales, global_scale, rows, cols, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let t0 = Instant::now();
    let iters = 20;
    for _ in 0..iters {
        fp4_dequant_bf16(&mut d_out, &d_packed, &d_scales, global_scale, rows, cols, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let got = d_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = fp4_dequant_bf16_reference(&packed_host, &scales_host, global_scale, rows, cols);

    let (max_abs, _per_elem, _global) = compare_bf16(&got, &want);
    println!("=== xenon-cli test-dequant ===");
    println!("rows x cols             {} x {}", rows, cols);
    println!("global_scale            {:.6}", global_scale);
    println!("per-launch time         {:>7.2} us", per_us);
    println!("max abs diff            {:.3e}", max_abs);
    // Kernel and reference share identical math; any difference is a bug.
    anyhow::ensure!(max_abs == 0.0, "dequant kernel diverges from reference (abs {max_abs})");
    println!("OK: kernel matches reference bit-for-bit.");
    Ok(())
}

fn cmd_test_gemm(m: usize, n: usize, k: usize, seed: u64) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let a_host: Vec<bf16> = (0..m * k).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    let b_host: Vec<bf16> = (0..k * n).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();

    let mut d_a: DeviceBuffer<bf16> = DeviceBuffer::new(m * k).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_b: DeviceBuffer<bf16> = DeviceBuffer::new(k * n).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_d: DeviceBuffer<bf16> = DeviceBuffer::new(m * n).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_a.copy_from_host(&a_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_b.copy_from_host(&b_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Warmup + timed runs.
    lt.matmul_bf16_rm(&mut d_d, &d_a, &d_b, None, m, n, k, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let t0 = Instant::now();
    let iters = 20;
    for _ in 0..iters {
        lt.matmul_bf16_rm(&mut d_d, &d_a, &d_b, None, m, n, k, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let tflops = (2.0 * m as f64 * n as f64 * k as f64) / (per_us * 1e-6) / 1e12;

    let got = d_d.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = matmul_bf16_reference(&a_host, &b_host, m, n, k, 1.0, 0.0, None);

    let (max_abs, _per_elem, global_rel) = compare_bf16(&got, &want);
    // GEMM outputs can pass through zero; the right measure is
    // max_abs / max(|ref|), which is a bounded fraction of 1 bf16 ULP.
    println!("=== xenon-cli test-gemm ===");
    println!("shape                   M={m} N={n} K={k}");
    println!("per-launch time         {:>7.2} us  ({:>5.2} TFLOP/s)", per_us, tflops);
    println!("max abs diff            {:.3e}", max_abs);
    println!("global rel diff         {:.3e}", global_rel);
    let tol_rel: f32 = 2e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel diff {global_rel} exceeds tolerance {tol_rel}");
    println!("OK: within bf16 tolerance (global rel {tol_rel:.1e}).");
    Ok(())
}

fn cmd_test_gelu(n: usize, seed: u64) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..n).map(|_| bf16::from_f32(rng.gen_range(-4.0..4.0))).collect();

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(n).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_y: DeviceBuffer<bf16> = DeviceBuffer::new(n).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    gelu_tanh_bf16(&mut d_y, &d_x, Some(&stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let got = d_y.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = gelu_tanh_bf16_reference(&x_host);

    let (max_abs, _per_elem, global_rel) = compare_bf16(&got, &want);
    println!("=== xenon-cli test-gelu ===");
    println!("n                       {n}");
    println!("max abs diff            {:.3e}", max_abs);
    println!("global rel diff         {:.3e}", global_rel);
    // GELU crosses zero, so per-elem rel is unbounded near the origin;
    // global rel is the right metric.
    let tol_rel: f32 = 1e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel diff {global_rel} exceeds tolerance {tol_rel}");
    println!("OK: within bf16 tolerance (global rel {tol_rel:.1e}).");
    Ok(())
}

fn cmd_upload(dir: PathBuf, prefix: Option<String>, limit_bytes: u64, verify: bool) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;

    let st_path = dir.join("model.safetensors");
    let mm = MmapWeights::open(&st_path)?;
    let (free_before, total) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("=== xenon-cli upload ===");
    println!("file                   {}", st_path.display());
    println!("vram free before       {:>6.2} / {:>6.2} GiB",
             free_before as f64 / 1073741824.0, total as f64 / 1073741824.0);
    if let Some(p) = &prefix {
        println!("prefix filter          {}", p);
    }
    println!("byte cap               {:.2} GiB", limit_bytes as f64 / 1073741824.0);
    println!();

    // Walk tensors in name order, upload until cap exceeded.
    let mut uploaded_bytes: u64 = 0;
    let mut uploaded_count: usize = 0;
    // Keep uploaded buffers alive until the end so we can report steady-state
    // VRAM occupancy; they drop on function exit.
    let mut resident: Vec<DeviceBuffer<u8>> = Vec::new();
    let t0 = Instant::now();

    for (name, info) in &mm.header.tensors {
        if let Some(p) = &prefix {
            if !name.starts_with(p) {
                continue;
            }
        }
        let bytes = info.bytes();
        if uploaded_bytes + bytes > limit_bytes {
            break;
        }
        let src = mm.tensor_bytes(name)?;
        let mut d: DeviceBuffer<u8> = DeviceBuffer::new(bytes as usize)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        d.copy_from_host_bytes(src).map_err(|e| anyhow::anyhow!("{e}"))?;

        if verify {
            let mut dst = vec![0u8; bytes as usize];
            d.copy_to_host(&mut dst).map_err(|e| anyhow::anyhow!("{e}"))?;
            anyhow::ensure!(dst == src, "round-trip byte mismatch for '{}'", name);
        }

        resident.push(d);
        uploaded_bytes += bytes;
        uploaded_count += 1;
    }

    device_synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let elapsed = t0.elapsed();
    let (free_after, _) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;

    let gbps = uploaded_bytes as f64 / elapsed.as_secs_f64() / 1e9;
    println!("tensors uploaded       {}", uploaded_count);
    println!("bytes                  {:.2} GiB", uploaded_bytes as f64 / 1073741824.0);
    println!("elapsed                {:.3} s", elapsed.as_secs_f64());
    println!("throughput             {:.2} GB/s", gbps);
    if verify {
        println!("round-trip verified    OK");
    }
    println!("vram free after        {:>6.2} GiB  (delta {:>+6.2} GiB)",
             free_after as f64 / 1073741824.0,
             (free_after as f64 - free_before as f64) / 1073741824.0);
    Ok(())
}

fn cmd_test_linear(m: usize, n: usize, k: usize, seed: u64) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..m * k).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    // W is [N, K] row-major in HF convention.
    let w_host: Vec<bf16> = (0..n * k).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(m * k).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_w: DeviceBuffer<bf16> = DeviceBuffer::new(n * k).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_y: DeviceBuffer<bf16> = DeviceBuffer::new(m * n).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_w.copy_from_host(&w_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    lt.linear_bf16(&mut d_y, &d_x, &d_w, None, m, n, k, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let t0 = Instant::now();
    let iters = 20;
    for _ in 0..iters {
        lt.linear_bf16(&mut d_y, &d_x, &d_w, None, m, n, k, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let per_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let tflops = (2.0 * m as f64 * n as f64 * k as f64) / (per_us * 1e-6) / 1e12;

    let got = d_y.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = linear_bf16_reference(&x_host, &w_host, m, n, k, 1.0, 0.0, None);
    let (max_abs, _per_elem, global_rel) = compare_bf16(&got, &want);

    println!("=== xenon-cli test-linear ===");
    println!("shape                   y[M,N] = x[M,K] * W[N,K]^T; M={m} N={n} K={k}");
    println!("per-launch time         {:>7.2} us  ({:>5.2} TFLOP/s)", per_us, tflops);
    println!("max abs diff            {:.3e}", max_abs);
    println!("global rel diff         {:.3e}", global_rel);
    let tol_rel: f32 = 2e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel diff {global_rel} exceeds tolerance {tol_rel}");
    println!("OK: within bf16 tolerance (global rel {tol_rel:.1e}).");
    Ok(())
}

fn cmd_test_rope(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    seed: u64,
) -> anyhow::Result<()> {
    anyhow::ensure!(rotary_dim <= head_dim && rotary_dim % 2 == 0, "rotary_dim must be even and <= head_dim");
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = tokens * heads * head_dim;
    let x_host: Vec<bf16> = (0..n).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    let pos_host: Vec<i32> = (0..tokens).map(|i| i as i32 * 3 + 7).collect(); // non-trivial positions

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(n).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_pos: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_pos.copy_from_host(&pos_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    rope_bf16(&mut d_x, &d_pos, tokens, heads, head_dim, rotary_dim, theta, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let got = d_x.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = rope_bf16_reference(&x_host, &pos_host, tokens, heads, head_dim, rotary_dim, theta);
    let (max_abs, _per, global_rel) = compare_bf16(&got, &want);

    println!("=== xenon-cli test-rope ===");
    println!("tokens x heads x head_dim  {} x {} x {}", tokens, heads, head_dim);
    println!("rotary_dim                 {}", rotary_dim);
    println!("theta                      {}", theta);
    println!("max abs diff               {:.3e}", max_abs);
    println!("global rel diff            {:.3e}", global_rel);
    let tol_rel: f32 = 1e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel {global_rel} exceeds {tol_rel}");
    println!("OK: RoPE matches reference within {tol_rel:.1e}.");
    Ok(())
}

fn cmd_test_softmax(
    rows: usize,
    t_q: usize,
    t_kv: usize,
    scale: f32,
    q_pos_base: i32,
    window: i32,
    seed: u64,
) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = rows * t_kv;
    let scores_host: Vec<bf16> = (0..n)
        .map(|_| bf16::from_f32(rng.gen_range(-3.0..3.0)))
        .collect();

    let mut d_scores: DeviceBuffer<bf16> = DeviceBuffer::new(n).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_scores.copy_from_host(&scores_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    softmax_attn_bf16(&mut d_scores, rows, t_q, t_kv, scale, q_pos_base, window, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let got = d_scores.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = softmax_attn_bf16_reference(&scores_host, rows, t_q, t_kv, scale, q_pos_base, window);
    let (max_abs, _per, global_rel) = compare_bf16(&got, &want);

    // Row-sum check: probabilities in each row should sum to ~1 (or ~0 if row
    // was fully masked; impossible when q_pos >= 0).
    let mut worst_sum_dev: f32 = 0.0;
    for r in 0..rows {
        let s: f32 = got[r * t_kv..(r + 1) * t_kv].iter().map(|v| v.to_f32()).sum();
        let dev = (s - 1.0).abs();
        if dev > worst_sum_dev { worst_sum_dev = dev; }
    }

    println!("=== xenon-cli test-softmax ===");
    println!("rows x t_q x t_kv          {} x {} x {}", rows, t_q, t_kv);
    println!("scale / q_pos_base / window {} / {} / {}", scale, q_pos_base, window);
    println!("worst row-sum dev from 1    {:.3e}", worst_sum_dev);
    println!("max abs diff vs ref         {:.3e}", max_abs);
    println!("global rel diff vs ref      {:.3e}", global_rel);
    anyhow::ensure!(worst_sum_dev <= 3e-2, "row probs don't sum to 1 (worst dev {worst_sum_dev})");
    let tol_rel: f32 = 2e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel {global_rel} exceeds {tol_rel}");
    println!("OK: masked softmax matches reference within {tol_rel:.1e}.");
    Ok(())
}

fn cmd_test_embed(dir: PathBuf, ids_csv: String) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let vocab = cfg.text_config.vocab_size;
    let hidden = cfg.text_config.hidden_size;

    let ids_host: Vec<i32> = ids_csv
        .split(',')
        .map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()?;
    for id in &ids_host {
        anyhow::ensure!((*id as usize) < vocab, "id {id} >= vocab {vocab}");
    }
    let tokens = ids_host.len();

    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;
    let table_bytes = mm.load_bf16("model.language_model.embed_tokens.weight")?;
    anyhow::ensure!(table_bytes.len() == vocab * hidden * 2, "embed table size");

    let mut d_table: DeviceBuffer<bf16> = DeviceBuffer::new(vocab * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ids: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;

    let upload_start = Instant::now();
    d_table.copy_from_host_bytes(table_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ids.copy_from_host(&ids_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    device_synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let upload_ms = upload_start.elapsed().as_secs_f64() * 1e3;

    embed_gather_bf16(&mut d_out, &d_table, &d_ids, tokens, vocab, hidden, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let got = d_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Host reference: direct slice out of the mmap.
    let table_host = bytes_to_bf16_vec(table_bytes);
    let mut want = vec![bf16::ZERO; tokens * hidden];
    for (t, &id) in ids_host.iter().enumerate() {
        let src = &table_host[id as usize * hidden..(id as usize + 1) * hidden];
        want[t * hidden..(t + 1) * hidden].copy_from_slice(src);
    }

    let mut mismatches = 0usize;
    for (a, b) in got.iter().zip(want.iter()) {
        if a.to_bits() != b.to_bits() { mismatches += 1; }
    }

    println!("=== xenon-cli test-embed ===");
    println!("tokens                   {}", tokens);
    println!("ids                      {:?}", ids_host);
    println!("table upload             {:>6.1} ms ({:.2} GiB)", upload_ms, (vocab * hidden * 2) as f64 / 1073741824.0);
    println!("mismatched bf16 elements {}", mismatches);
    anyhow::ensure!(mismatches == 0, "embedding gather produced {mismatches} mismatches");
    println!("OK: gather is bit-identical to host slice.");
    Ok(())
}

fn cmd_test_attn_layer(
    dir: PathBuf,
    layer: usize,
    tokens: usize,
    q_pos_base: i32,
    seed: u64,
) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let tc = &cfg.text_config;
    anyhow::ensure!(layer < tc.num_hidden_layers, "layer out of range");

    let hidden = tc.hidden_size;
    let h = tc.num_attention_heads;
    let h_kv = tc.num_key_value_heads;
    let d = cfg.head_dim_for_layer(layer);
    let kind = tc.layer_types[layer];
    let window: i32 = if matches!(kind, LayerKind::SlidingAttention) {
        tc.sliding_window.unwrap_or(0) as i32
    } else { 0 };
    let (rope_theta, rotary_dim) = cfg.rope_for_layer(layer);
    let eps = tc.rms_norm_eps as f32;
    let scale = 1.0f32 / (d as f32).sqrt();

    println!("=== xenon-cli test-attn-layer ===");
    println!("layer {layer}  kind {:?}  T={tokens}  q_pos_base={q_pos_base}", kind);
    println!("  hidden/H/Hkv/D         {hidden} / {h} / {h_kv} / {d}");
    println!("  rope theta / rotary    {rope_theta} / {rotary_dim}");
    println!("  window / scale         {window} / {:.6}", scale);

    let prefix = format!("model.language_model.layers.{layer}");
    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;

    let in_norm_bytes = mm.load_bf16(&format!("{prefix}.input_layernorm.weight"))?;
    let q_norm_bytes  = mm.load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?;
    let k_norm_bytes  = mm.load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?;
    let q_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.q_proj"))?;
    let k_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.k_proj"))?;
    let v_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.v_proj"))?;
    let o_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.o_proj"))?;

    anyhow::ensure!(q_proj.out_features == h * d && q_proj.in_features == hidden,
        "q_proj: got [{}, {}], want [{}, {}]", q_proj.out_features, q_proj.in_features, h * d, hidden);
    anyhow::ensure!(k_proj.out_features == h_kv * d && k_proj.in_features == hidden,
        "k_proj: got [{}, {}]", k_proj.out_features, k_proj.in_features);
    anyhow::ensure!(v_proj.out_features == h_kv * d && v_proj.in_features == hidden,
        "v_proj: got [{}, {}]", v_proj.out_features, v_proj.in_features);
    anyhow::ensure!(o_proj.out_features == hidden && o_proj.in_features == h * d,
        "o_proj: got [{}, {}]", o_proj.out_features, o_proj.in_features);
    anyhow::ensure!(q_norm_bytes.len() == d * 2, "q_norm size");
    anyhow::ensure!(k_norm_bytes.len() == d * 2, "k_norm size");

    // Random input.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..tokens * hidden)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect();
    let pos_host: Vec<i32> = (0..tokens).map(|i| q_pos_base + i as i32).collect();

    // ----- Device tensors -----
    let upload_start = Instant::now();
    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_in_norm_w: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_q_norm_w:  DeviceBuffer<bf16> = DeviceBuffer::new(d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_k_norm_w:  DeviceBuffer<bf16> = DeviceBuffer::new(d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Dequantized projection weights.
    let mut d_qw: DeviceBuffer<bf16> = DeviceBuffer::new(q_proj.out_features * q_proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_kw: DeviceBuffer<bf16> = DeviceBuffer::new(k_proj.out_features * k_proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_vw: DeviceBuffer<bf16> = DeviceBuffer::new(v_proj.out_features * v_proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ow: DeviceBuffer<bf16> = DeviceBuffer::new(o_proj.out_features * o_proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Scratch buffers for dequant source (packed + scales).
    let (mut d_qp, mut d_qs) = (
        DeviceBuffer::<u8>::new(q_proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(q_proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    let (mut d_kp, mut d_ks) = (
        DeviceBuffer::<u8>::new(k_proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(k_proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    let (mut d_vp, mut d_vs) = (
        DeviceBuffer::<u8>::new(v_proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(v_proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    let (mut d_op, mut d_os) = (
        DeviceBuffer::<u8>::new(o_proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(o_proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );

    // Intermediate buffers.
    let mut d_x_normed: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_q: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_k: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_kv * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_v: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_kv * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_attn_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;

    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&pos_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_in_norm_w.copy_from_host_bytes(in_norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_q_norm_w.copy_from_host_bytes(q_norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_k_norm_w.copy_from_host_bytes(k_norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_qp.copy_from_host_bytes(q_proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_qs.copy_from_host_bytes(q_proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_kp.copy_from_host_bytes(k_proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ks.copy_from_host_bytes(k_proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_vp.copy_from_host_bytes(v_proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_vs.copy_from_host_bytes(v_proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_op.copy_from_host_bytes(o_proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_os.copy_from_host_bytes(o_proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;

    fp4_dequant_bf16(&mut d_qw, &d_qp, &d_qs, q_proj.global_scale, q_proj.out_features, q_proj.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_kw, &d_kp, &d_ks, k_proj.global_scale, k_proj.out_features, k_proj.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_vw, &d_vp, &d_vs, v_proj.global_scale, v_proj.out_features, v_proj.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_ow, &d_op, &d_os, o_proj.global_scale, o_proj.out_features, o_proj.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let upload_ms = upload_start.elapsed().as_secs_f64() * 1e3;

    let run_forward = |lt: &mut CublasLt,
                       d_q: &mut DeviceBuffer<bf16>,
                       d_k: &mut DeviceBuffer<bf16>,
                       d_v: &mut DeviceBuffer<bf16>,
                       d_attn_out: &mut DeviceBuffer<bf16>,
                       d_out: &mut DeviceBuffer<bf16>,
                       d_x_normed: &mut DeviceBuffer<bf16>|
     -> anyhow::Result<()> {
        rmsnorm_bf16(d_x_normed, &d_x, &d_in_norm_w, tokens, hidden, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        lt.linear_bf16(d_q, d_x_normed, &d_qw, None, tokens, h * d, hidden, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        lt.linear_bf16(d_k, d_x_normed, &d_kw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        lt.linear_bf16(d_v, d_x_normed, &d_vw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // q_norm / k_norm: per-head RMSNorm over D.
        // Clone into scratch since rmsnorm takes &x (read) + &mut out.
        let q_copy = clone_buffer(d_q)?;
        rmsnorm_bf16(d_q, &q_copy, &d_q_norm_w, tokens * h, d, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let k_copy = clone_buffer(d_k)?;
        rmsnorm_bf16(d_k, &k_copy, &d_k_norm_w, tokens * h_kv, d, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // RoPE in-place on q and k (the rotary dims only).
        rope_bf16(d_q, &d_positions, tokens, h,    d, rotary_dim, rope_theta, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        rope_bf16(d_k, &d_positions, tokens, h_kv, d, rotary_dim, rope_theta, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Attention: Q[T, H, D] x K/V[T, Hkv, D] -> [T, H, D]
        attn_naive_bf16(d_attn_out, d_q, d_k, d_v,
                        tokens, tokens, h, h_kv, d, scale, q_pos_base, window, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // O projection back to hidden.
        lt.linear_bf16(d_out, d_attn_out, &d_ow, None, tokens, hidden, h * d, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    };

    // Warmup.
    run_forward(&mut lt, &mut d_q, &mut d_k, &mut d_v, &mut d_attn_out, &mut d_out, &mut d_x_normed)?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let iters = 10;
    let fwd_start = Instant::now();
    for _ in 0..iters {
        run_forward(&mut lt, &mut d_q, &mut d_k, &mut d_v, &mut d_attn_out, &mut d_out, &mut d_x_normed)?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let fwd_ms = fwd_start.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let gpu_out = d_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;

    // ----- Host reference -----
    let ref_start = Instant::now();
    let in_norm_w_host = bytes_to_bf16_vec(in_norm_bytes);
    let q_norm_w_host  = bytes_to_bf16_vec(q_norm_bytes);
    let k_norm_w_host  = bytes_to_bf16_vec(k_norm_bytes);
    let q_w_host = fp4_dequant_bf16_reference(q_proj.packed, q_proj.scales, q_proj.global_scale, q_proj.out_features, q_proj.in_features);
    let k_w_host = fp4_dequant_bf16_reference(k_proj.packed, k_proj.scales, k_proj.global_scale, k_proj.out_features, k_proj.in_features);
    let v_w_host = fp4_dequant_bf16_reference(v_proj.packed, v_proj.scales, v_proj.global_scale, v_proj.out_features, v_proj.in_features);
    let o_w_host = fp4_dequant_bf16_reference(o_proj.packed, o_proj.scales, o_proj.global_scale, o_proj.out_features, o_proj.in_features);

    let x_normed_host = rmsnorm_bf16_reference(&x_host, &in_norm_w_host, tokens, hidden, eps);
    let q_flat = linear_bf16_reference(&x_normed_host, &q_w_host, tokens, h * d, hidden, 1.0, 0.0, None);
    let k_flat = linear_bf16_reference(&x_normed_host, &k_w_host, tokens, h_kv * d, hidden, 1.0, 0.0, None);
    let v_flat = linear_bf16_reference(&x_normed_host, &v_w_host, tokens, h_kv * d, hidden, 1.0, 0.0, None);
    let q_normed = rmsnorm_bf16_reference(&q_flat, &q_norm_w_host, tokens * h, d, eps);
    let k_normed = rmsnorm_bf16_reference(&k_flat, &k_norm_w_host, tokens * h_kv, d, eps);
    let q_roped = rope_bf16_reference(&q_normed, &pos_host, tokens, h,    d, rotary_dim, rope_theta);
    let k_roped = rope_bf16_reference(&k_normed, &pos_host, tokens, h_kv, d, rotary_dim, rope_theta);
    let attn_out_host = attn_naive_bf16_reference(&q_roped, &k_roped, &v_flat,
                                                  tokens, tokens, h, h_kv, d,
                                                  scale, q_pos_base, window);
    let cpu_out = linear_bf16_reference(&attn_out_host, &o_w_host, tokens, hidden, h * d, 1.0, 0.0, None);
    let ref_ms = ref_start.elapsed().as_secs_f64() * 1e3;

    let (max_abs, _, global_rel) = compare_bf16(&gpu_out, &cpu_out);
    let gpu_max: f32 = gpu_out.iter().map(|v| v.to_f32().abs()).fold(0.0, f32::max);
    let gpu_finite = gpu_out.iter().all(|v| v.to_f32().is_finite());

    println!();
    println!("upload + dequant        {:>7.1} ms", upload_ms);
    println!("gpu forward (avg)       {:>7.2} ms", fwd_ms);
    println!("cpu reference forward   {:>7.1} ms", ref_ms);
    println!("gpu output max |.|      {:>8.3}", gpu_max);
    println!("gpu finite              {}", gpu_finite);
    println!("max abs diff            {:.3e}", max_abs);
    println!("global rel diff         {:.3e}", global_rel);
    anyhow::ensure!(gpu_finite, "GPU output contains NaN/inf");
    anyhow::ensure!(gpu_max > 0.0, "GPU output is identically zero");
    let tol_rel: f32 = 5e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel {global_rel} exceeds {tol_rel}");
    println!("OK: full attention layer matches host reference within {tol_rel:.1e}.");
    Ok(())
}

/// D2D-copy a device buffer. Used to avoid alias when a kernel wants separate
/// read and write views of the same logical tensor.
fn clone_buffer(src: &DeviceBuffer<bf16>) -> anyhow::Result<DeviceBuffer<bf16>> {
    let mut dst = DeviceBuffer::<bf16>::new(src.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    dst.copy_from_device(src).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(dst)
}

/// Run the attention block up to (and including) the naive-attention kernel,
/// returning the attn output [T, H, D] and the roped Q, K, V used to produce
/// it. Shared helper for test-attn-layer and test-attn-decode.
#[allow(clippy::too_many_arguments)]
fn run_prefill_attn(
    dir: &PathBuf,
    layer: usize,
    tokens: usize,
    q_pos_base: i32,
    seed: u64,
    stream: &Stream,
    lt: &mut CublasLt,
) -> anyhow::Result<PrefillAttn> {
    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let tc = &cfg.text_config;
    anyhow::ensure!(layer < tc.num_hidden_layers, "layer out of range");
    let hidden = tc.hidden_size;
    let h = tc.num_attention_heads;
    let h_kv = tc.num_key_value_heads;
    let d = cfg.head_dim_for_layer(layer);
    let kind = tc.layer_types[layer];
    let window: i32 = if matches!(kind, LayerKind::SlidingAttention) {
        tc.sliding_window.unwrap_or(0) as i32
    } else { 0 };
    let (rope_theta, rotary_dim) = cfg.rope_for_layer(layer);
    let eps = tc.rms_norm_eps as f32;
    let scale = 1.0f32 / (d as f32).sqrt();

    let prefix = format!("model.language_model.layers.{layer}");
    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;
    let in_norm_bytes = mm.load_bf16(&format!("{prefix}.input_layernorm.weight"))?;
    let q_norm_bytes  = mm.load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?;
    let k_norm_bytes  = mm.load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?;
    let q_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.q_proj"))?;
    let k_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.k_proj"))?;
    let v_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.v_proj"))?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..tokens * hidden)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect();
    let pos_host: Vec<i32> = (0..tokens).map(|i| q_pos_base + i as i32).collect();

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_in_norm_w: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_q_norm_w:  DeviceBuffer<bf16> = DeviceBuffer::new(d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_k_norm_w:  DeviceBuffer<bf16> = DeviceBuffer::new(d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_qw: DeviceBuffer<bf16> = DeviceBuffer::new(q_proj.out_features * q_proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_kw: DeviceBuffer<bf16> = DeviceBuffer::new(k_proj.out_features * k_proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_vw: DeviceBuffer<bf16> = DeviceBuffer::new(v_proj.out_features * v_proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;

    let (mut d_qp, mut d_qs) = (
        DeviceBuffer::<u8>::new(q_proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(q_proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    let (mut d_kp, mut d_ks) = (
        DeviceBuffer::<u8>::new(k_proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(k_proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    let (mut d_vp, mut d_vs) = (
        DeviceBuffer::<u8>::new(v_proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(v_proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );

    let mut d_x_normed: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_q: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_k: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_kv * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_v: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_kv * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_attn_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h * d).map_err(|e| anyhow::anyhow!("{e}"))?;

    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&pos_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_in_norm_w.copy_from_host_bytes(in_norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_q_norm_w.copy_from_host_bytes(q_norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_k_norm_w.copy_from_host_bytes(k_norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_qp.copy_from_host_bytes(q_proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_qs.copy_from_host_bytes(q_proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_kp.copy_from_host_bytes(k_proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ks.copy_from_host_bytes(k_proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_vp.copy_from_host_bytes(v_proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_vs.copy_from_host_bytes(v_proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;

    fp4_dequant_bf16(&mut d_qw, &d_qp, &d_qs, q_proj.global_scale, q_proj.out_features, q_proj.in_features, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_kw, &d_kp, &d_ks, k_proj.global_scale, k_proj.out_features, k_proj.in_features, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_vw, &d_vp, &d_vs, v_proj.global_scale, v_proj.out_features, v_proj.in_features, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;

    rmsnorm_bf16(&mut d_x_normed, &d_x, &d_in_norm_w, tokens, hidden, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_q, &d_x_normed, &d_qw, None, tokens, h * d, hidden, 1.0, 0.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_k, &d_x_normed, &d_kw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_v, &d_x_normed, &d_vw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    {
        let q_copy = clone_buffer(&d_q)?;
        rmsnorm_bf16(&mut d_q, &q_copy, &d_q_norm_w, tokens * h, d, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
        let k_copy = clone_buffer(&d_k)?;
        rmsnorm_bf16(&mut d_k, &k_copy, &d_k_norm_w, tokens * h_kv, d, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    rope_bf16(&mut d_q, &d_positions, tokens, h,    d, rotary_dim, rope_theta, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    rope_bf16(&mut d_k, &d_positions, tokens, h_kv, d, rotary_dim, rope_theta, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    attn_naive_bf16(&mut d_attn_out, &d_q, &d_k, &d_v, tokens, tokens, h, h_kv, d, scale, q_pos_base, window, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(PrefillAttn {
        q: d_q, k: d_k, v: d_v, attn_out: d_attn_out,
        tokens, hidden, h, h_kv, d,
        scale, window, kind,
    })
}

struct PrefillAttn {
    q: DeviceBuffer<bf16>,
    k: DeviceBuffer<bf16>,
    v: DeviceBuffer<bf16>,
    attn_out: DeviceBuffer<bf16>,
    tokens: usize,
    #[allow(dead_code)]
    hidden: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    scale: f32,
    window: i32,
    kind: LayerKind,
}

fn cmd_test_ple(dir: PathBuf, ids_csv: String, layer: usize) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let tc = &cfg.text_config;
    let hidden = tc.hidden_size;
    let per_layer = tc.hidden_size_per_layer_input
        .ok_or_else(|| anyhow::anyhow!("config missing hidden_size_per_layer_input"))?;
    let n_layers = tc.num_hidden_layers;
    let ple_width = per_layer * n_layers;
    anyhow::ensure!(layer < n_layers, "layer out of range");

    let ids_host: Vec<i32> = ids_csv
        .split(',')
        .map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()?;
    let tokens = ids_host.len();

    println!("=== xenon-cli test-ple ===");
    println!("tokens                     {}", tokens);
    println!("ids                        {:?}", ids_host);
    println!("layers / per_layer         {} / {}", n_layers, per_layer);
    println!("ple_width (L * per_layer)  {}", ple_width);

    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;
    let table_name = "model.language_model.embed_tokens_per_layer.weight";
    let info = mm.info(table_name).ok_or_else(|| anyhow::anyhow!("no PLE table"))?;
    anyhow::ensure!(
        info.shape == vec![tc.vocab_size, ple_width],
        "PLE table shape {:?} != [{}, {}]", info.shape, tc.vocab_size, ple_width
    );
    println!("ple table bytes            {:.2} GiB ({} rows)",
             info.bytes() as f64 / 1073741824.0, info.shape[0]);

    // Host gather + upload.
    let gather_start = Instant::now();
    let host_gather = mm.gather_rows_bf16(table_name, &ids_host, ple_width)?;
    let gather_ms = gather_start.elapsed().as_secs_f64() * 1e3;
    anyhow::ensure!(host_gather.len() == tokens * ple_width, "gather length");

    let mut d_gather: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let upload_start = Instant::now();
    d_gather.copy_from_host(&host_gather).map_err(|e| anyhow::anyhow!("{e}"))?;
    device_synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let upload_ms = upload_start.elapsed().as_secs_f64() * 1e3;
    let upload_gb = (tokens * ple_width * 2) as f64 / 1e9;
    println!("host gather                {:>6.1} ms", gather_ms);
    println!("h2d upload                 {:>6.1} ms ({:.2} GB/s)",
             upload_ms, upload_gb / (upload_ms / 1e3));

    // Spot-check: for each token, bytes from the mmap row should match the
    // device round-trip at every layer slice.
    let got = d_gather.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut mismatches = 0usize;
    for (a, b) in got.iter().zip(host_gather.iter()) {
        if a.to_bits() != b.to_bits() { mismatches += 1; }
    }
    anyhow::ensure!(mismatches == 0, "round-trip mismatch: {mismatches}");

    // Slice: PLE for this layer is columns [layer*per_layer, (layer+1)*per_layer).
    // Since the gather is [T, ple_width] row-major, a per-layer slice would be
    // non-contiguous. For the test, rebuild a contiguous [T, per_layer] on the
    // host and upload it.
    let begin = layer * per_layer;
    let end = begin + per_layer;
    let mut slice_host = Vec::with_capacity(tokens * per_layer);
    for t in 0..tokens {
        let row = &host_gather[t * ple_width..(t + 1) * ple_width];
        slice_host.extend_from_slice(&row[begin..end]);
    }
    let mut d_ple: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ple.copy_from_host(&slice_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!();
    println!("--- layer {layer} PLE slice ---");
    println!("  slice shape            [{tokens}, {per_layer}]");
    let ple_max = slice_host.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
    println!("  |max|                  {:.4}", ple_max);

    // Now run per_layer_input_gate on a random hidden-state input, just to
    // prove dim wiring. This is a sanity on the linear path, not the PLE math.
    let prefix = format!("model.language_model.layers.{layer}");
    let gate = mm.load_quant_linear(&format!("{prefix}.per_layer_input_gate"))?;
    let proj = mm.load_quant_linear(&format!("{prefix}.per_layer_projection"))?;
    anyhow::ensure!(gate.out_features == per_layer && gate.in_features == hidden,
        "per_layer_input_gate: got [{}, {}] want [{per_layer}, {hidden}]",
        gate.out_features, gate.in_features);
    anyhow::ensure!(proj.out_features == hidden && proj.in_features == per_layer,
        "per_layer_projection: got [{}, {}] want [{hidden}, {per_layer}]",
        proj.out_features, proj.in_features);
    println!("  per_layer_input_gate   FP4 [{per_layer} x {hidden}]");
    println!("  per_layer_projection   FP4 [{hidden} x {per_layer}]");

    // Simulate a hidden-state for the gate+proj dim plumbing:
    // gate maps hidden -> per_layer, projection maps per_layer -> hidden.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let h_host: Vec<bf16> = (0..tokens * hidden)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect();

    // Dequantize gate/proj weights on device.
    let mut d_gate_w: DeviceBuffer<bf16> = DeviceBuffer::new(gate.out_features * gate.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_proj_w: DeviceBuffer<bf16> = DeviceBuffer::new(proj.out_features * proj.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (mut d_gp, mut d_gs) = (
        DeviceBuffer::<u8>::new(gate.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(gate.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    let (mut d_pp, mut d_ps) = (
        DeviceBuffer::<u8>::new(proj.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
        DeviceBuffer::<u8>::new(proj.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    d_gp.copy_from_host_bytes(gate.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_gs.copy_from_host_bytes(gate.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_pp.copy_from_host_bytes(proj.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ps.copy_from_host_bytes(proj.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_gate_w, &d_gp, &d_gs, gate.global_scale,
                     gate.out_features, gate.in_features, Some(&stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_proj_w, &d_pp, &d_ps, proj.global_scale,
                     proj.out_features, proj.in_features, Some(&stream)).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_gated: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_back: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host(&h_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    lt.linear_bf16(&mut d_gated, &d_h, &d_gate_w, None, tokens, per_layer, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_back, &d_gated, &d_proj_w, None, tokens, hidden, per_layer, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Host-ref check for gate path correctness.
    let gate_w_host = fp4_dequant_bf16_reference(gate.packed, gate.scales, gate.global_scale, gate.out_features, gate.in_features);
    let proj_w_host = fp4_dequant_bf16_reference(proj.packed, proj.scales, proj.global_scale, proj.out_features, proj.in_features);
    let gated_host = linear_bf16_reference(&h_host, &gate_w_host, tokens, per_layer, hidden, 1.0, 0.0, None);
    let back_host  = linear_bf16_reference(&gated_host, &proj_w_host, tokens, hidden, per_layer, 1.0, 0.0, None);
    let got_back = d_back.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (max_abs, _p, global_rel) = compare_bf16(&got_back, &back_host);
    println!("  gate+proj vs ref       global_rel {:.3e}", global_rel);
    let tol_rel: f32 = 5e-2;
    anyhow::ensure!(global_rel <= tol_rel, "gate+proj path diverges: {global_rel}");
    let _ = (max_abs, d_ple);
    println!("OK: PLE gather + per-layer gate/projection paths wired and correct.");
    Ok(())
}

fn cmd_kv_map(dir: PathBuf) -> anyhow::Result<()> {
    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let tc = &cfg.text_config;
    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;
    let mut owned = Vec::new();
    let mut shared = Vec::new();
    for l in 0..tc.num_hidden_layers {
        if mm.layer_owns_kv(l) {
            owned.push(l);
        } else {
            shared.push(l);
        }
    }
    println!("=== xenon-cli kv-map ===");
    println!("layers          {}", tc.num_hidden_layers);
    println!("num_kv_shared   {:?}", tc.num_kv_shared_layers);
    println!("owns KV         {} layers: {:?}", owned.len(), owned);
    println!("shares KV       {} layers: {:?}", shared.len(), shared);
    println!();
    println!("per-layer kind (S=sliding, F=full) and ownership (O=own, S=shared):");
    for l in 0..tc.num_hidden_layers {
        let kind = match tc.layer_types[l] {
            LayerKind::SlidingAttention => 'S',
            LayerKind::FullAttention    => 'F',
        };
        let own = if mm.layer_owns_kv(l) { 'O' } else { 's' };
        print!("{l:02}:{kind}{own}  ");
        if (l + 1) % 6 == 0 { println!(); }
    }
    if tc.num_hidden_layers % 6 != 0 { println!(); }
    Ok(())
}

fn cmd_test_kv_cache(dir: PathBuf, layer: usize, tokens: usize, seed: u64) -> anyhow::Result<()> {
    anyhow::ensure!(tokens >= 2, "need at least 2 tokens");
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("=== xenon-cli test-kv-cache ===");
    // Reference run: full prefill on T tokens.
    let p = run_prefill_attn(&dir, layer, tokens, 0, seed, &stream, &mut lt)?;
    let t = p.tokens;
    let row_k = p.h_kv * p.d;
    let row_q = p.h * p.d;

    println!("layer {layer}  kind {:?}  T={t}", p.kind);
    println!("  hidden/H/Hkv/D         {} / {} / {} / {}", p.hidden, p.h, p.h_kv, p.d);

    // Build a KvCache for just this layer (one slot).
    let spec = SlotSpec { h_kv: p.h_kv, head_dim: p.d };
    let mut cache = KvCache::new(vec![spec], vec![0], t).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Prefill phase: copy the first T-1 K/V rows from p.k / p.v into a scratch
    // buffer sized [T-1, Hkv, D] then append to cache.
    let mut prefill_k: DeviceBuffer<bf16> = DeviceBuffer::new((t - 1) * row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut prefill_v: DeviceBuffer<bf16> = DeviceBuffer::new((t - 1) * row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    prefill_k.copy_region_from_device(0, &p.k, 0, (t - 1) * row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    prefill_v.copy_region_from_device(0, &p.v, 0, (t - 1) * row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    cache.append(0, &prefill_k, &prefill_v, t - 1).map_err(|e| anyhow::anyhow!("{e}"))?;
    cache.advance(t - 1);
    println!("prefill done; cache.len = {}", cache.len());

    // Decode phase: append the last token's K/V, then attend.
    let mut new_k: DeviceBuffer<bf16> = DeviceBuffer::new(row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut new_v: DeviceBuffer<bf16> = DeviceBuffer::new(row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    new_k.copy_region_from_device(0, &p.k, (t - 1) * row_k, row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    new_v.copy_region_from_device(0, &p.v, (t - 1) * row_k, row_k).map_err(|e| anyhow::anyhow!("{e}"))?;
    cache.append(0, &new_k, &new_v, 1).map_err(|e| anyhow::anyhow!("{e}"))?;
    cache.advance(1);
    println!("decode step appended; cache.len = {}", cache.len());
    assert_eq!(cache.len(), t);

    // Extract the last Q row for the decode query.
    let mut q_last: DeviceBuffer<bf16> = DeviceBuffer::new(row_q).map_err(|e| anyhow::anyhow!("{e}"))?;
    q_last.copy_region_from_device(0, &p.q, (t - 1) * row_q, row_q).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Run attention using the cache's K/V buffers.
    let mut out_cache: DeviceBuffer<bf16> = DeviceBuffer::new(row_q).map_err(|e| anyhow::anyhow!("{e}"))?;
    attn_naive_bf16(
        &mut out_cache, &q_last, cache.k_buf(0), cache.v_buf(0),
        1, cache.len(), p.h, p.h_kv, p.d,
        p.scale, (t as i32) - 1, p.window, Some(&stream),
    ).map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let got = out_cache.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let prefill_host = p.attn_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = &prefill_host[(t - 1) * row_q..t * row_q];
    let (max_abs, _per, global_rel) = compare_bf16(&got, want);
    let mut mismatches = 0usize;
    for (a, b) in got.iter().zip(want.iter()) {
        if a.to_bits() != b.to_bits() { mismatches += 1; }
    }

    println!();
    println!("mismatched bf16 elements  {} / {}", mismatches, row_q);
    println!("max abs diff              {:.3e}", max_abs);
    println!("global rel diff           {:.3e}", global_rel);
    anyhow::ensure!(mismatches == 0,
        "cached decode diverges from prefill last row by {mismatches} elements");
    println!("OK: cached decode matches full-prefill last row bit-for-bit.");
    Ok(())
}

fn cmd_test_attn_decode(dir: PathBuf, layer: usize, tokens: usize, seed: u64) -> anyhow::Result<()> {
    anyhow::ensure!(tokens >= 2, "need at least 2 tokens for a meaningful decode test");
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("=== xenon-cli test-attn-decode ===");
    let p = run_prefill_attn(&dir, layer, tokens, 0, seed, &stream, &mut lt)?;

    let t = p.tokens;
    let row = p.h * p.d;
    println!("layer {layer}  kind {:?}  T={t}", p.kind);
    println!("  hidden/H/Hkv/D         {} / {} / {} / {}", p.hidden, p.h, p.h_kv, p.d);
    println!("  window / scale         {} / {:.6}", p.window, p.scale);

    // Extract Q[T-1:T] as a [1, H, D] buffer.
    let mut q_last: DeviceBuffer<bf16> = DeviceBuffer::new(row).map_err(|e| anyhow::anyhow!("{e}"))?;
    q_last.copy_region_from_device(0, &p.q, (t - 1) * row, row)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Run decode-shape attention: T_q=1, T_kv=T, q_pos_base=T-1.
    // Full K and V from the prefill are the "cache".
    let mut out_decode: DeviceBuffer<bf16> = DeviceBuffer::new(row).map_err(|e| anyhow::anyhow!("{e}"))?;
    attn_naive_bf16(
        &mut out_decode, &q_last, &p.k, &p.v,
        1, t, p.h, p.h_kv, p.d,
        p.scale, (t as i32) - 1, p.window, Some(&stream),
    ).map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Compare against prefill's last row.
    let got = out_decode.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let prefill_host = p.attn_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want: &[bf16] = &prefill_host[(t - 1) * row..t * row];

    let (max_abs, _per, global_rel) = compare_bf16(&got, want);
    let mut mismatches = 0usize;
    for (a, b) in got.iter().zip(want.iter()) {
        if a.to_bits() != b.to_bits() { mismatches += 1; }
    }

    println!();
    println!("mismatched bf16 elements  {} / {}", mismatches, row);
    println!("max abs diff              {:.3e}", max_abs);
    println!("global rel diff           {:.3e}", global_rel);
    // Same math on same inputs — we expect bit-for-bit match (reduction order
    // is identical per (q_tok, q_head) and here q_tok is fixed).
    anyhow::ensure!(mismatches == 0,
        "decode output diverges from prefill last row by {mismatches} elements");
    println!("OK: decode-shape attention matches last-row of prefill bit-for-bit.");
    Ok(())
}

fn cmd_test_mlp(dir: PathBuf, layer: usize, batch: usize, seed: u64) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let tc = &cfg.text_config;
    let hidden = tc.hidden_size;
    let inter = tc.intermediate_size;
    let eps = tc.rms_norm_eps as f32;
    anyhow::ensure!(layer < tc.num_hidden_layers, "layer {layer} >= num_hidden_layers {}", tc.num_hidden_layers);

    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;
    let prefix = format!("model.language_model.layers.{layer}");

    let norm_bytes = mm.load_bf16(&format!("{prefix}.pre_feedforward_layernorm.weight"))?;
    let gate = mm.load_quant_linear(&format!("{prefix}.mlp.gate_proj"))?;
    let up = mm.load_quant_linear(&format!("{prefix}.mlp.up_proj"))?;
    let down = mm.load_quant_linear(&format!("{prefix}.mlp.down_proj"))?;

    anyhow::ensure!(
        gate.out_features == inter && gate.in_features == hidden,
        "gate_proj shape mismatch: got [{}, {}], expected [{inter}, {hidden}]",
        gate.out_features, gate.in_features
    );
    anyhow::ensure!(
        up.out_features == inter && up.in_features == hidden,
        "up_proj shape mismatch: got [{}, {}], expected [{inter}, {hidden}]",
        up.out_features, up.in_features
    );
    anyhow::ensure!(
        down.out_features == hidden && down.in_features == inter,
        "down_proj shape mismatch: got [{}, {}], expected [{hidden}, {inter}]",
        down.out_features, down.in_features
    );

    println!("=== xenon-cli test-mlp ===");
    println!("layer {layer}  batch {batch}  hidden {hidden}  intermediate {inter}");
    println!("  gate.global_scale      {}", gate.global_scale);
    println!("  up.global_scale        {}", up.global_scale);
    println!("  down.global_scale      {}", down.global_scale);

    // Random input.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..batch * hidden)
        .map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0)))
        .collect();

    // ----- GPU path -----
    let upload_start = Instant::now();

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(batch * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_norm_w: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_gate_packed: DeviceBuffer<u8> = DeviceBuffer::new(gate.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_gate_scales: DeviceBuffer<u8> = DeviceBuffer::new(gate.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_gate_w: DeviceBuffer<bf16> = DeviceBuffer::new(gate.out_features * gate.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_up_packed: DeviceBuffer<u8> = DeviceBuffer::new(up.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_up_scales: DeviceBuffer<u8> = DeviceBuffer::new(up.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_up_w: DeviceBuffer<bf16> = DeviceBuffer::new(up.out_features * up.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_down_packed: DeviceBuffer<u8> = DeviceBuffer::new(down.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_down_scales: DeviceBuffer<u8> = DeviceBuffer::new(down.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_down_w: DeviceBuffer<bf16> = DeviceBuffer::new(down.out_features * down.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_normed: DeviceBuffer<bf16> = DeviceBuffer::new(batch * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_gate_out: DeviceBuffer<bf16> = DeviceBuffer::new(batch * inter).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_up_out: DeviceBuffer<bf16> = DeviceBuffer::new(batch * inter).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_act: DeviceBuffer<bf16> = DeviceBuffer::new(batch * inter).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_out: DeviceBuffer<bf16> = DeviceBuffer::new(batch * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;

    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_norm_w.copy_from_host_bytes(norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_gate_packed.copy_from_host_bytes(gate.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_gate_scales.copy_from_host_bytes(gate.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_up_packed.copy_from_host_bytes(up.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_up_scales.copy_from_host_bytes(up.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_down_packed.copy_from_host_bytes(down.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_down_scales.copy_from_host_bytes(down.scales).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Dequantize weights on device (one-time).
    fp4_dequant_bf16(&mut d_gate_w, &d_gate_packed, &d_gate_scales,
                     gate.global_scale, gate.out_features, gate.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_up_w, &d_up_packed, &d_up_scales,
                     up.global_scale, up.out_features, up.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_down_w, &d_down_packed, &d_down_scales,
                     down.global_scale, down.out_features, down.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let upload_ms = upload_start.elapsed().as_secs_f64() * 1e3;

    // Warmup: first cuBLASLt call on a new (M, N, K) shape pays an algorithm-
    // selection / JIT cost that's unrelated to steady-state perf.
    let mut run_forward = |lt: &mut CublasLt| -> anyhow::Result<()> {
        rmsnorm_bf16(&mut d_normed, &d_x, &d_norm_w, batch, hidden, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        lt.linear_bf16(&mut d_gate_out, &d_normed, &d_gate_w, None, batch, inter, hidden, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        lt.linear_bf16(&mut d_up_out, &d_normed, &d_up_w, None, batch, inter, hidden, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        gelu_tanh_glu_bf16(&mut d_act, &d_gate_out, &d_up_out, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        lt.linear_bf16(&mut d_out, &d_act, &d_down_w, None, batch, hidden, inter, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    };

    run_forward(&mut lt)?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Timed runs.
    let iters = 20;
    let fwd_start = Instant::now();
    for _ in 0..iters {
        run_forward(&mut lt)?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let fwd_ms = fwd_start.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let gpu_out = d_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;

    // ----- Host reference -----
    let ref_start = Instant::now();
    let norm_w_host = bytes_to_bf16_vec(norm_bytes);
    let gate_w_host = fp4_dequant_bf16_reference(gate.packed, gate.scales, gate.global_scale, gate.out_features, gate.in_features);
    let up_w_host   = fp4_dequant_bf16_reference(up.packed,   up.scales,   up.global_scale,   up.out_features,   up.in_features);
    let down_w_host = fp4_dequant_bf16_reference(down.packed, down.scales, down.global_scale, down.out_features, down.in_features);

    let normed_host = rmsnorm_bf16_reference(&x_host, &norm_w_host, batch, hidden, eps);
    let gate_out_host = linear_bf16_reference(&normed_host, &gate_w_host, batch, inter, hidden, 1.0, 0.0, None);
    let up_out_host   = linear_bf16_reference(&normed_host, &up_w_host,   batch, inter, hidden, 1.0, 0.0, None);

    const K0: f32 = 0.7978845608028654;
    const K1: f32 = 0.044715;
    let act_host: Vec<bf16> = gate_out_host.iter().zip(up_out_host.iter()).map(|(g, u)| {
        let gv = g.to_f32();
        let uv = u.to_f32();
        let inner = K0 * (gv + K1 * gv * gv * gv);
        let gelu = 0.5 * gv * (1.0 + inner.tanh());
        bf16::from_f32(gelu * uv)
    }).collect();
    let cpu_out = linear_bf16_reference(&act_host, &down_w_host, batch, hidden, inter, 1.0, 0.0, None);
    let ref_ms = ref_start.elapsed().as_secs_f64() * 1e3;

    let (max_abs, _, global_rel) = compare_bf16(&gpu_out, &cpu_out);

    // Magnitude check: if GPU output is mostly zero or NaN, something's wrong
    // structurally even if GPU and CPU happen to agree on the zeros.
    let gpu_max: f32 = gpu_out.iter().map(|v| v.to_f32().abs()).fold(0.0, f32::max);
    let gpu_finite = gpu_out.iter().all(|v| v.to_f32().is_finite());

    println!();
    println!("upload + dequant        {:>7.1} ms", upload_ms);
    println!("gpu forward (avg)       {:>7.2} ms", fwd_ms);
    println!("cpu reference forward   {:>7.1} ms", ref_ms);
    println!("gpu output max |.|      {:>8.3}", gpu_max);
    println!("gpu finite              {}", gpu_finite);
    println!("max abs diff            {:.3e}", max_abs);
    println!("global rel diff         {:.3e}", global_rel);
    anyhow::ensure!(gpu_finite, "GPU output contains NaN/inf");
    anyhow::ensure!(gpu_max > 0.0, "GPU output is identically zero");
    let tol_rel: f32 = 5e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel diff {global_rel} exceeds tolerance {tol_rel}");
    println!("OK: full MLP matches host reference within {tol_rel:.1e}.");
    Ok(())
}
