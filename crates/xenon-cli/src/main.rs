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
    rmsnorm_bf16, rmsnorm_bf16_reference,
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

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    for (a, b) in y_gpu.iter().zip(y_ref.iter()) {
        let fa = a.to_f32();
        let fb = b.to_f32();
        let d = (fa - fb).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        let denom = fb.abs().max(1e-6);
        let rel = d / denom;
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
    }

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
