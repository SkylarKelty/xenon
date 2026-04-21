use std::path::PathBuf;

use clap::{Parser, Subcommand};
use xenon_core::{
    config::QuantConfig,
    weights::{is_excluded, SafetensorsHeader, WeightBreakdown},
    GemmaConfig, LayerKind,
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
    Info {
        /// Model snapshot directory.
        model: PathBuf,
    },
    /// Walk the safetensors header and categorize tensors (NVFP4 pairs vs
    /// plain) without touching tensor data. Validates the modelopt exclude
    /// list: excluded modules should be plain (bf16), everything else should
    /// be FP4 weight+scale pairs.
    Load {
        /// Model snapshot directory.
        model: PathBuf,
    },
    /// Run the sanity kernel to verify the Rust <-> CUDA build pipeline.
    Sanity,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    match args.cmd {
        Command::Info { model } => cmd_info(model),
        Command::Load { model } => cmd_load(model),
        Command::Sanity => cmd_sanity(),
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

    // Rough decode-bandwidth roofline.
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
    println!("attn weights / token     {:>6.1} MiB", attn_bytes as f64 / 1024.0 / 1024.0);
    println!("mlp  weights / token     {:>6.1} MiB", mlp_bytes as f64 / 1024.0 / 1024.0);
    println!("lm_head (bf16)  / token  {:>6.1} MiB", lm_head_bytes as f64 / 1024.0 / 1024.0);
    println!("total streamed  / token  {:>6.2} GiB", total_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
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
        println!(
            "  {:>10}  {:>5} tensors  {:>8.2} GiB",
            dt,
            count,
            *bytes as f64 / 1073741824.0
        );
    }

    println!();
    println!("-- NVFP4 weight+scale pairs --");
    println!("  pair count             {}", b.quant_pairs.len());
    println!(
        "  packed FP4 weights     {:>6.2} GiB",
        b.quant_weight_bytes() as f64 / 1073741824.0
    );
    println!(
        "  block scales           {:>6.2} GiB",
        b.quant_scale_bytes() as f64 / 1073741824.0
    );
    let extra_count = b.quant_pairs.iter().filter(|p| p.extra_scale.is_some()).count();
    println!("  with weight_scale_2    {}", extra_count);

    println!();
    println!("-- plain (unquantized) .weight tensors --");
    println!("  count                  {}", b.plain_weights.len());
    println!(
        "  bytes                  {:>6.2} GiB",
        b.plain_weight_bytes(&header) as f64 / 1073741824.0
    );

    if !b.orphan_scales.is_empty() {
        println!();
        println!("!! orphan scales (scale tensor without matching weight) !!");
        for s in &b.orphan_scales {
            println!("   {s}");
        }
    }

    // Validate the exclude_modules invariant.
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
            // Allow norms and embeddings to be plain even if not in the exclude
            // list (modelopt doesn't quantize those by construction).
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
