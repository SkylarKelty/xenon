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
    cuda::{device_synchronize, mem_info, Device, DeviceBuffer, Stream},
    fp4_dequant_bf16, fp4_dequant_bf16_reference, gelu_tanh_bf16, gelu_tanh_bf16_reference,
    matmul_bf16_reference, rmsnorm_bf16, rmsnorm_bf16_reference, CublasLt,
};

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
    /// Run GELU-tanh on random input and compare to a CPU reference.
    TestGelu {
        #[arg(long, default_value_t = 4096)]
        n: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
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
        Command::Sanity => cmd_sanity(),
        Command::Vram => cmd_vram(),
        Command::TestRmsnorm { rows, hidden, eps, seed } => cmd_test_rmsnorm(rows, hidden, eps, seed),
        Command::TestDequant { rows, cols, seed } => cmd_test_dequant(rows, cols, seed),
        Command::TestGemm { m, n, k, seed } => cmd_test_gemm(m, n, k, seed),
        Command::TestGelu { n, seed } => cmd_test_gelu(n, seed),
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
