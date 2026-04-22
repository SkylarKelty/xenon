use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use half::bf16;
use rand::{Rng, SeedableRng};
use xenon_core::{
    config::QuantConfig,
    weights::{is_excluded, MmapWeights, SafetensorsHeader, WeightBreakdown},
    GemmaConfig, LayerKind, Tokenizer,
};
use xenon_kernels::{
    add_scale_bf16, attn_flash_bf16, attn_naive_bf16, attn_naive_bf16_reference,
    cuda::{device_synchronize, mem_info, Device, DeviceBuffer, Stream},
    embed_gather_bf16, fp4_dequant_bf16, fp4_dequant_bf16_reference, fp4_gemv_bf16, gelu_tanh_bf16,
    gelu_tanh_bf16_reference, gelu_tanh_glu_bf16, linear_bf16_reference, matmul_bf16_reference,
    nvfp4_quantize_bf16, per_layer_slice_bf16, rmsnorm_bf16, rmsnorm_bf16_reference, rope_bf16,
    rope_bf16_reference, round_up, scale_bf16, softcap_bf16, softmax_attn_bf16,
    softmax_attn_bf16_reference, swizzle_blockscale_ue4m3, CublasLt, KvCache, SlotSpec,
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
    /// Round-trip correctness for the bf16→NVFP4 activation quantizer.
    TestNvfp4Roundtrip {
        #[arg(long, default_value_t = 4)]
        rows: usize,
        #[arg(long, default_value_t = 256)]
        cols: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run the native NVFP4-weight × bf16-activation GEMM on a real quantized
    /// Gemma weight and compare against the existing dequant+bf16 path.
    TestNvfp4Linear {
        model: PathBuf,
        /// Full prefix of a quantized linear (e.g. model.language_model.layers.0.mlp.gate_proj).
        #[arg(long)]
        module: String,
        #[arg(long, default_value_t = 1)]
        m: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Run the fused FP4×bf16 gemv (M=1 decode fast path) on a real quantized
    /// Gemma weight; compare against dequant+linear_bf16 and time both.
    TestFp4Gemv {
        model: PathBuf,
        /// Full prefix of a quantized linear (e.g. model.language_model.layers.0.mlp.gate_proj).
        #[arg(long)]
        module: String,
        #[arg(long, default_value_t = 100)]
        iters: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Validate xk_swizzle_blockscale against a reference byte file produced
    /// by vLLM's Python swizzle. Both src and expected are raw u8[M, Kb].
    TestSwizzle {
        #[arg(long)]
        src: PathBuf,
        #[arg(long)]
        expected: PathBuf,
        #[arg(long)]
        m: usize,
        #[arg(long)]
        k_blocks: usize,
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
    /// Load tokenizer.json, encode+decode text, verify round-trip and vocab
    /// matches config.vocab_size.
    TestTokenizer {
        model: PathBuf,
        #[arg(long, default_value = "Hello, xenon!")]
        text: String,
        #[arg(long, default_value_t = true)]
        add_specials: bool,
    },
    /// Diff our embed_gather output against an HF reference activation dump.
    /// Runs embed_tokens(ids) * sqrt(hidden) on device and compares to
    /// HF's `embed_tokens_scaled` tensor captured by tools/hf-ref/capture.py.
    TestVsHfEmbed {
        model: PathBuf,
        #[arg(long)]
        hf: PathBuf,
        #[arg(long, default_value = "2,108,107,1")]
        ids: String,
    },
    /// Diff PLE assembly (raw + projected) against HF. Runs the full top-
    /// level PLE pipeline: embed_tokens_per_layer gather + sqrt(Hl) scale,
    /// per_layer_model_projection linear + 1/sqrt(H) scale, per-layer
    /// RMSNorm, combine (ctx + raw) * 1/sqrt(2).
    TestVsHfPle {
        model: PathBuf,
        #[arg(long)]
        hf: PathBuf,
        #[arg(long, default_value = "2,108,107,1")]
        ids: String,
    },
    /// Run one full decoder layer end-to-end on real weights and diff the
    /// output against HF's `layer_NN.final`. Exercises the composition of
    /// every primitive: pre/post layernorm, attention (QKV + norms + RoPE
    /// + naive attn + o_proj), residual, MLP, PLE gate/GLU/projection, and
    /// the layer_scalar tail multiply.
    TestVsHfLayer {
        model: PathBuf,
        #[arg(long)]
        hf: PathBuf,
        #[arg(long, default_value = "2,108,107,1")]
        ids: String,
        #[arg(long, default_value_t = 0)]
        layer: usize,
    },
    /// Diff the post-stack tail: take HF's `layer_41.final`, apply final
    /// RMSNorm + tied lm_head + softcap, compare to HF at each of the three
    /// stages.
    TestVsHfTail {
        model: PathBuf,
        #[arg(long)]
        hf: PathBuf,
    },
    /// Full 42-layer forward pass: embed + PLE + 42 chained decoder layers
    /// (with KV sharing for layers 24..41) + final norm + lm_head + softcap.
    /// Diffs every major intermediate against HF's activation dump.
    TestVsHfFull {
        model: PathBuf,
        #[arg(long)]
        hf: PathBuf,
        #[arg(long, default_value = "2,108,107,1")]
        ids: String,
    },
    /// Run N copies of a prompt in a single batched generate call.
    /// Reports per-request decode ms + total tok/s (summed) vs wall time.
    BenchBatch {
        model: PathBuf,
        #[arg(long, default_value = "Write a haiku about GPUs.")]
        prompt: String,
        #[arg(long)]
        chat: bool,
        #[arg(long, default_value_t = 4)]
        batch: usize,
        #[arg(long, default_value_t = 20)]
        max_new: usize,
        #[arg(long, default_value_t = 1024)]
        max_len: usize,
    },
    /// Measure prefill + decode throughput. Runs N warmup iterations then M
    /// measured iterations with a fixed prompt, records per-step timings,
    /// dumps summary + JSON to results/.
    Bench {
        model: PathBuf,
        #[arg(long, default_value = "Write a haiku about GPUs.")]
        prompt: String,
        #[arg(long)]
        chat: bool,
        #[arg(long, default_value_t = 32)]
        max_new: usize,
        #[arg(long, default_value_t = 4096)]
        max_len: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 3)]
        iters: usize,
        /// Optional label for the results JSON (helps diff runs).
        #[arg(long, default_value = "baseline")]
        label: String,
    },
    /// Generate text from a prompt: tokenize, prefill, then greedy-sample
    /// decode until EOS or `max_new_tokens`. Loads all 42 layers' weights
    /// once (FP4 packed, dequant per step into a transient scratch buffer).
    Generate {
        model: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 32)]
        max_new: usize,
        /// Max context length — sizes the KV cache. Default leaves headroom
        /// for the biggest full-attn slot at 4K.
        #[arg(long, default_value_t = 4096)]
        max_len: usize,
        /// Wrap the prompt in Gemma's chat template so the instruction-tuned
        /// model treats it as a user turn and actually responds.
        #[arg(long)]
        chat: bool,
    },
    /// Run the FlashAttention-2 tiled kernel vs the naive kernel on random
    /// Q/K/V and verify they agree. Exercises the online-softmax path at
    /// T_kv values where the naive kernel would run out of shared memory.
    TestAttnFlash {
        #[arg(long, default_value_t = 1)]
        t_q: usize,
        #[arg(long, default_value_t = 1024)]
        t_kv: usize,
        #[arg(long, default_value_t = 8)]
        h: usize,
        #[arg(long, default_value_t = 2)]
        h_kv: usize,
        #[arg(long, default_value_t = 256)]
        head_dim: usize,
        /// 0 = no sliding window (full causal).
        #[arg(long, default_value_t = 0)]
        window: i32,
        /// q_pos_base (decode shape: T_kv-T_q for cached).
        #[arg(long)]
        q_pos_base: Option<i32>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
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
        Command::TestNvfp4Linear { model, module, m, seed } => cmd_test_nvfp4_linear(model, module, m, seed),
        Command::TestFp4Gemv { model, module, iters, seed } => cmd_test_fp4_gemv(model, module, iters, seed),
        Command::TestSwizzle { src, expected, m, k_blocks } => cmd_test_swizzle(src, expected, m, k_blocks),
        Command::TestNvfp4Roundtrip { rows, cols, seed } => cmd_test_nvfp4_roundtrip(rows, cols, seed),
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
        Command::TestTokenizer { model, text, add_specials } =>
            cmd_test_tokenizer(model, text, add_specials),
        Command::TestAttnFlash { t_q, t_kv, h, h_kv, head_dim, window, q_pos_base, seed } =>
            cmd_test_attn_flash(t_q, t_kv, h, h_kv, head_dim, window, q_pos_base, seed),
        Command::TestVsHfEmbed { model, hf, ids } => cmd_test_vs_hf_embed(model, hf, ids),
        Command::TestVsHfPle { model, hf, ids } => cmd_test_vs_hf_ple(model, hf, ids),
        Command::TestVsHfLayer { model, hf, ids, layer } =>
            cmd_test_vs_hf_layer(model, hf, ids, layer),
        Command::TestVsHfTail { model, hf } => cmd_test_vs_hf_tail(model, hf),
        Command::TestVsHfFull { model, hf, ids } => cmd_test_vs_hf_full(model, hf, ids),
        Command::Generate { model, prompt, max_new, max_len, chat } =>
            cmd_generate(model, prompt, max_new, max_len, chat),
        Command::Bench { model, prompt, chat, max_new, max_len, warmup, iters, label } =>
            cmd_bench(model, prompt, chat, max_new, max_len, warmup, iters, label),
        Command::BenchBatch { model, prompt, chat, batch, max_new, max_len } =>
            cmd_bench_batch(model, prompt, chat, batch, max_new, max_len),
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
    rmsnorm_bf16(&mut d_y, &d_x, Some(&d_w), rows, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let t0 = Instant::now();
    let iters = 50;
    for _ in 0..iters {
        rmsnorm_bf16(&mut d_y, &d_x, Some(&d_w), rows, hidden, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let y_gpu = d_y.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let y_ref = rmsnorm_bf16_reference(&x_host, Some(&weight_host), rows, hidden, eps);
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

fn cmd_test_nvfp4_roundtrip(rows: usize, cols: usize, seed: u64) -> anyhow::Result<()> {
    // Round-trip: random bf16 -> quantize (FP4 + UE4M3 scales) -> dequant via
    // our proven fp4_dequant_bf16 -> compare to original.
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    anyhow::ensure!(cols % 16 == 0, "cols must be multiple of 16");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..rows * cols)
        .map(|_| bf16::from_f32(rng.gen_range(-3.0..3.0))).collect();

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(rows * cols).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_packed: DeviceBuffer<u8> = DeviceBuffer::new(rows * cols / 2).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_scales: DeviceBuffer<u8> = DeviceBuffer::new(rows * cols / 16).map_err(|e| anyhow::anyhow!("{e}"))?;
    nvfp4_quantize_bf16(&mut d_packed, &mut d_scales, &d_x, rows, cols, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_deq: DeviceBuffer<bf16> = DeviceBuffer::new(rows * cols).map_err(|e| anyhow::anyhow!("{e}"))?;
    // global_scale = 1.0 (activation has no per-tensor scale).
    fp4_dequant_bf16(&mut d_deq, &d_packed, &d_scales, 1.0, rows, cols, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let got = d_deq.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let scales_host = d_scales.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let packed_host = d_packed.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (max_abs, _per, global_rel) = compare_bf16(&got, &x_host);
    println!("=== xenon-cli test-nvfp4-roundtrip ===");
    println!("rows x cols              {} x {}", rows, cols);
    println!("max abs diff             {:.3e}", max_abs);
    println!("global rel diff          {:.3e}", global_rel);
    println!();
    println!("first 16 elements (one block):");
    println!("  {:>4}  {:>10}  {:>10}  {:>4}", "i", "input", "deq", "fp4");
    for i in 0..16 {
        let byte = packed_host[i / 2];
        let code = if i & 1 == 0 { byte & 0xF } else { byte >> 4 };
        println!("  {:>4}  {:>10.4}  {:>10.4}  {:>4}",
                 i, x_host[i].to_f32(), got[i].to_f32(), code);
    }
    println!("scale_byte[0]            0x{:02x} -> {:.4}",
             scales_host[0], xenon_kernels::ue4m3_to_f32(scales_host[0]));
    // FP4 quant error per element is up to ~10% of block max; global rel
    // against the original input is expected ~1-10%.
    anyhow::ensure!(global_rel <= 0.2, "round-trip rel {global_rel} too large");
    println!("OK: FP4 round-trip max_rel {:.1}% (expected ~5-10%).", global_rel * 100.0);
    Ok(())
}

fn cmd_test_nvfp4_linear(dir: PathBuf, module: String, m: usize, seed: u64) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;
    let q = mm.load_quant_linear(&module)?;
    let (n, k) = (q.out_features, q.in_features);
    println!("=== xenon-cli test-nvfp4-linear ===");
    println!("module                 {module}");
    println!("shape (M, N, K)        ({m}, {n}, {k})");
    println!("global_scale           {}", q.global_scale);

    // Random bf16 activation.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..m * k).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();

    // Upload FP4 weight + scales.
    let mut d_w_packed: DeviceBuffer<u8> = DeviceBuffer::new(q.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_w_scales: DeviceBuffer<u8> = DeviceBuffer::new(q.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_w_packed.copy_from_host_bytes(q.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_w_scales.copy_from_host_bytes(q.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(m * k).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Quantize activation to NVFP4 (packed + UE4M3 scales).
    let mut d_x_packed: DeviceBuffer<u8> = DeviceBuffer::new(m * (k / 2)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_x_scales: DeviceBuffer<u8> = DeviceBuffer::new(m * (k / 16)).map_err(|e| anyhow::anyhow!("{e}"))?;
    nvfp4_quantize_bf16(&mut d_x_packed, &mut d_x_scales, &d_x, m, k, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_y_nvfp4: DeviceBuffer<bf16> = DeviceBuffer::new(m * n).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t0 = Instant::now();
    lt.linear_nvfp4(&mut d_y_nvfp4, &d_x_packed, &d_x_scales,
                     &d_w_packed, &d_w_scales, q.global_scale, None,
                     m, n, k, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let nvfp4_us = t0.elapsed().as_secs_f64() * 1e6;

    // Reference path: dequant to bf16, then bf16 linear.
    let mut d_w: DeviceBuffer<bf16> = DeviceBuffer::new(n * k).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_w, &d_w_packed, &d_w_scales, q.global_scale, n, k, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_y_ref: DeviceBuffer<bf16> = DeviceBuffer::new(m * n).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t1 = Instant::now();
    lt.linear_bf16(&mut d_y_ref, &d_x, &d_w, None, m, n, k, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let ref_us = t1.elapsed().as_secs_f64() * 1e6;

    let got = d_y_nvfp4.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = d_y_ref.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (max_abs, _per, global_rel) = compare_bf16(&got, &want);

    println!();
    println!("nvfp4 GEMM              {:>7.1} us", nvfp4_us);
    println!("dequant + bf16 GEMM     {:>7.1} us", ref_us);
    println!("max abs diff            {:.3e}", max_abs);
    println!("global rel diff         {:.3e}", global_rel);
    // NVFP4×NVFP4 vs bf16-weight-from-dequant×bf16-activation: the activation
    // is now FP4 too, so per-element error compounds. For random inputs
    // (unstructured magnitudes) ~50% rel is normal; real activations give
    // tighter match.
    let tol: f32 = 0.6;
    anyhow::ensure!(global_rel <= tol, "nvfp4 vs ref: {global_rel} > {tol}");
    println!("OK: native NVFP4 GEMM runs correctly (rel {:.1}% vs dequanted-bf16 on random input; real activations match tighter).",
             global_rel * 100.0);
    Ok(())
}

fn cmd_test_fp4_gemv(dir: PathBuf, module: String, iters: usize, seed: u64) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mm = MmapWeights::open(&dir.join("model.safetensors"))?;
    let q = mm.load_quant_linear(&module)?;
    let (n, k) = (q.out_features, q.in_features);
    println!("=== xenon-cli test-fp4-gemv ===");
    println!("module                 {module}");
    println!("shape (N, K)           ({n}, {k})");
    println!("global_scale           {}", q.global_scale);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let x_host: Vec<bf16> = (0..k).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();

    let mut d_x: DeviceBuffer<bf16> = DeviceBuffer::new(k).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_x.copy_from_host(&x_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_w_packed: DeviceBuffer<u8> = DeviceBuffer::new(q.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_w_scales: DeviceBuffer<u8> = DeviceBuffer::new(q.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_w_packed.copy_from_host_bytes(q.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_w_scales.copy_from_host_bytes(q.scales).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Reference: dequant to bf16 weight, then bf16 linear.
    let mut d_w_bf16: DeviceBuffer<bf16> = DeviceBuffer::new(n * k).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_w_bf16, &d_w_packed, &d_w_scales, q.global_scale, n, k, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_y_ref: DeviceBuffer<bf16> = DeviceBuffer::new(n).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_y_ref, &d_x, &d_w_bf16, None, 1, n, k, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Fused: FP4×bf16 gemv directly.
    let mut d_y_fused: DeviceBuffer<bf16> = DeviceBuffer::new(n).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_gemv_bf16(&mut d_y_fused, &d_x, &d_w_packed, &d_w_scales, q.global_scale, n, k, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let got = d_y_fused.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let want = d_y_ref.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (max_abs, _per, global_rel) = compare_bf16(&got, &want);
    println!();
    println!("max abs diff            {:.3e}", max_abs);
    println!("global rel diff         {:.3e}", global_rel);

    // Timing: N iterations each, synchronize once at the end.
    if iters > 0 {
        // Warmup.
        for _ in 0..5 {
            fp4_gemv_bf16(&mut d_y_fused, &d_x, &d_w_packed, &d_w_scales, q.global_scale, n, k, Some(&stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            fp4_dequant_bf16(&mut d_w_bf16, &d_w_packed, &d_w_scales, q.global_scale, n, k, Some(&stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            lt.linear_bf16(&mut d_y_ref, &d_x, &d_w_bf16, None, 1, n, k, 1.0, 0.0, Some(&stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

        let t0 = Instant::now();
        for _ in 0..iters {
            fp4_gemv_bf16(&mut d_y_fused, &d_x, &d_w_packed, &d_w_scales, q.global_scale, n, k, Some(&stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        let fused_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            fp4_dequant_bf16(&mut d_w_bf16, &d_w_packed, &d_w_scales, q.global_scale, n, k, Some(&stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            lt.linear_bf16(&mut d_y_ref, &d_x, &d_w_bf16, None, 1, n, k, 1.0, 0.0, Some(&stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        let ref_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!();
        println!("fused fp4 gemv          {:>7.1} us  ({} iters)", fused_us, iters);
        println!("dequant + bf16 gemv     {:>7.1} us  ({} iters)", ref_us, iters);
        println!("speedup                 {:>7.2}x", ref_us / fused_us);
    }

    // Accumulation-order differences between the two reductions mean bit-for-
    // bit equality isn't guaranteed, but global rel should be tiny.
    let tol: f32 = 0.01;
    anyhow::ensure!(global_rel <= tol, "fused vs dequant: rel {global_rel} > {tol}");
    println!("OK: fused fp4 gemv matches dequant+bf16 linear within {:.3}% rel.",
             global_rel * 100.0);
    Ok(())
}

fn cmd_test_swizzle(src_path: PathBuf, exp_path: PathBuf, m: usize, kb: usize) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let src = std::fs::read(&src_path)?;
    let expected = std::fs::read(&exp_path)?;
    let m_padded = round_up(m, 128);
    let kb_padded = round_up(kb, 4);
    anyhow::ensure!(src.len() == m * kb, "src len {} != m*kb {}", src.len(), m * kb);
    anyhow::ensure!(expected.len() == m_padded * kb_padded,
        "expected len {} != m_padded*kb_padded {}", expected.len(), m_padded * kb_padded);

    let mut d_src: DeviceBuffer<u8> = DeviceBuffer::new(m * kb).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_src.copy_from_host_bytes(&src).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_dst: DeviceBuffer<u8> = DeviceBuffer::new(m_padded * kb_padded).map_err(|e| anyhow::anyhow!("{e}"))?;
    swizzle_blockscale_ue4m3(&mut d_dst, &d_src, m, kb, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let got = d_dst.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut diffs = 0usize;
    let mut first = None;
    for i in 0..got.len() {
        if got[i] != expected[i] {
            diffs += 1;
            if first.is_none() { first = Some(i); }
        }
    }
    if diffs == 0 {
        println!("OK: swizzle kernel matches reference byte-for-byte ({} bytes)", got.len());
    } else {
        let i = first.unwrap();
        let s = i.saturating_sub(8);
        let e = (i + 8).min(got.len());
        println!("FAIL: {diffs}/{} bytes differ. First diff at byte {i}:", got.len());
        println!("  got[{s}..{e}]:      {:?}", &got[s..e]);
        println!("  expected[{s}..{e}]: {:?}", &expected[s..e]);
        anyhow::bail!("swizzle kernel diverged from reference");
    }
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

    // Gemma's ScaledWordEmbedding multiplies by sqrt(hidden_size) inside forward.
    let embed_scale = (hidden as f32).sqrt();
    embed_gather_bf16(&mut d_out, &d_table, &d_ids, tokens, vocab, hidden, embed_scale, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let got = d_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Host reference: direct slice * scale.
    let table_host = bytes_to_bf16_vec(table_bytes);
    let mut want = vec![bf16::ZERO; tokens * hidden];
    for (t, &id) in ids_host.iter().enumerate() {
        let src = &table_host[id as usize * hidden..(id as usize + 1) * hidden];
        for (i, v) in src.iter().enumerate() {
            want[t * hidden + i] = bf16::from_f32(v.to_f32() * embed_scale);
        }
    }

    let (max_abs, _per, global_rel) = compare_bf16(&got, &want);

    println!("=== xenon-cli test-embed ===");
    println!("tokens                   {}", tokens);
    println!("ids                      {:?}", ids_host);
    println!("embed_scale (sqrt H)     {:.4}", embed_scale);
    println!("table upload             {:>6.1} ms ({:.2} GiB)", upload_ms, (vocab * hidden * 2) as f64 / 1073741824.0);
    println!("max abs diff vs ref      {:.3e}", max_abs);
    println!("global rel diff vs ref   {:.3e}", global_rel);
    let tol_rel: f32 = 1e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel {global_rel} exceeds {tol_rel}");
    println!("OK: scaled gather matches host ref within {tol_rel:.1e}.");
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
    // Gemma 4 passes scaling=1.0 to attention (no 1/sqrt(D) factor).
    let scale = 1.0f32;

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
        rmsnorm_bf16(d_x_normed, &d_x, Some(&d_in_norm_w), tokens, hidden, eps, Some(&stream))
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
        rmsnorm_bf16(d_q, &q_copy, Some(&d_q_norm_w), tokens * h, d, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let k_copy = clone_buffer(d_k)?;
        rmsnorm_bf16(d_k, &k_copy, Some(&d_k_norm_w), tokens * h_kv, d, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // v_norm: RMSNorm with_scale=False, per-head over D. No weight tensor.
        let v_copy = clone_buffer(d_v)?;
        rmsnorm_bf16(d_v, &v_copy, None, tokens * h_kv, d, eps, Some(&stream))
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

    let x_normed_host = rmsnorm_bf16_reference(&x_host, Some(&in_norm_w_host), tokens, hidden, eps);
    let q_flat = linear_bf16_reference(&x_normed_host, &q_w_host, tokens, h * d, hidden, 1.0, 0.0, None);
    let k_flat = linear_bf16_reference(&x_normed_host, &k_w_host, tokens, h_kv * d, hidden, 1.0, 0.0, None);
    let v_flat = linear_bf16_reference(&x_normed_host, &v_w_host, tokens, h_kv * d, hidden, 1.0, 0.0, None);
    let q_normed = rmsnorm_bf16_reference(&q_flat, Some(&q_norm_w_host), tokens * h, d, eps);
    let k_normed = rmsnorm_bf16_reference(&k_flat, Some(&k_norm_w_host), tokens * h_kv, d, eps);
    let v_normed = rmsnorm_bf16_reference(&v_flat, None, tokens * h_kv, d, eps);
    let q_roped = rope_bf16_reference(&q_normed, &pos_host, tokens, h,    d, rotary_dim, rope_theta);
    let k_roped = rope_bf16_reference(&k_normed, &pos_host, tokens, h_kv, d, rotary_dim, rope_theta);
    let attn_out_host = attn_naive_bf16_reference(&q_roped, &k_roped, &v_normed,
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
    // Gemma 4 passes scaling=1.0 to attention (no 1/sqrt(D) factor). The
    // learned q_norm/k_norm weights absorb whatever scaling the network
    // wants on Q·K.
    let scale = 1.0f32;

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

    rmsnorm_bf16(&mut d_x_normed, &d_x, Some(&d_in_norm_w), tokens, hidden, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_q, &d_x_normed, &d_qw, None, tokens, h * d, hidden, 1.0, 0.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_k, &d_x_normed, &d_kw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_v, &d_x_normed, &d_vw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
    {
        let q_copy = clone_buffer(&d_q)?;
        rmsnorm_bf16(&mut d_q, &q_copy, Some(&d_q_norm_w), tokens * h, d, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
        let k_copy = clone_buffer(&d_k)?;
        rmsnorm_bf16(&mut d_k, &k_copy, Some(&d_k_norm_w), tokens * h_kv, d, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
        // v_norm: RMSNorm with_scale=False — pure RMS normalization on V,
        // per-head over head_dim. No learned weight tensor in the safetensors.
        let v_copy = clone_buffer(&d_v)?;
        rmsnorm_bf16(&mut d_v, &v_copy, None, tokens * h_kv, d, eps, Some(stream)).map_err(|e| anyhow::anyhow!("{e}"))?;
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

fn cmd_test_vs_hf_embed(model_dir: PathBuf, hf_path: PathBuf, ids_csv: String) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&model_dir.join("config.json"))?;
    let vocab = cfg.text_config.vocab_size;
    let hidden = cfg.text_config.hidden_size;
    let embed_scale = (hidden as f32).sqrt();

    let ids_host: Vec<i32> = ids_csv
        .split(',')
        .map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()?;
    let tokens = ids_host.len();

    println!("=== xenon-cli test-vs-hf-embed ===");
    println!("ids                      {:?}", ids_host);
    println!("vocab / hidden / scale   {} / {} / {:.4}", vocab, hidden, embed_scale);

    // Our path.
    let mm = MmapWeights::open(&model_dir.join("model.safetensors"))?;
    let table_bytes = mm.load_bf16("model.language_model.embed_tokens.weight")?;

    let mut d_table: DeviceBuffer<bf16> = DeviceBuffer::new(vocab * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ids: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_table.copy_from_host_bytes(table_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ids.copy_from_host(&ids_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    embed_gather_bf16(&mut d_out, &d_table, &d_ids, tokens, vocab, hidden, embed_scale, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let ours = d_out.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;

    // HF reference. Saved as bf16 [1, T, H]; we flatten to [T, H].
    let hf_mm = MmapWeights::open(&hf_path)?;
    let hf_bytes = hf_mm.load_bf16("embed_tokens_scaled")?;
    anyhow::ensure!(hf_bytes.len() == tokens * hidden * 2,
        "HF embed_tokens_scaled has {} bytes; expected {}",
        hf_bytes.len(), tokens * hidden * 2);
    let theirs = bytes_to_bf16_vec(hf_bytes);

    let (max_abs, _per, global_rel) = compare_bf16(&ours, &theirs);
    let mut mismatches = 0usize;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        if a.to_bits() != b.to_bits() { mismatches += 1; }
    }

    println!();
    println!("our max |.|              {:>8.3}", ours.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max));
    println!("hf  max |.|              {:>8.3}", theirs.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max));
    println!("mismatched bf16 elements {} / {}", mismatches, tokens * hidden);
    println!("max abs diff             {:.3e}", max_abs);
    println!("global rel diff          {:.3e}", global_rel);
    // bf16 ULP is ~7.8e-3 relative; allow a few for any summation-order noise.
    let tol_rel: f32 = 1e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel {global_rel} exceeds {tol_rel}");
    println!("OK: embed_tokens * sqrt(H) matches HF within {tol_rel:.1e}.");
    Ok(())
}

fn cmd_test_vs_hf_ple(model_dir: PathBuf, hf_path: PathBuf, ids_csv: String) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&model_dir.join("config.json"))?;
    let tc = &cfg.text_config;
    let vocab = tc.vocab_size;
    let hidden = tc.hidden_size;
    let per_layer = tc.hidden_size_per_layer_input
        .ok_or_else(|| anyhow::anyhow!("hidden_size_per_layer_input missing"))?;
    let n_layers = tc.num_hidden_layers;
    let ple_width = per_layer * n_layers;
    let eps = tc.rms_norm_eps as f32;

    let ids_host: Vec<i32> = ids_csv
        .split(',')
        .map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()?;
    let tokens = ids_host.len();

    println!("=== xenon-cli test-vs-hf-ple ===");
    println!("ids                      {:?}", ids_host);
    println!("L / Hl / ple_width       {} / {} / {}", n_layers, per_layer, ple_width);

    let mm = MmapWeights::open(&model_dir.join("model.safetensors"))?;

    // ---- per_layer_inputs_raw: host-gather the 5.25 GiB PLE table, scale sqrt(Hl).
    let raw_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens_per_layer.weight", &ids_host, ple_width)?;
    let raw_scale = (per_layer as f32).sqrt();
    let raw_scaled: Vec<bf16> = raw_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * raw_scale)).collect();
    // Compare raw_scaled against HF per_layer_inputs_raw.
    let hf_mm = MmapWeights::open(&hf_path)?;
    let hf_raw_bytes = hf_mm.load_bf16("per_layer_inputs_raw")?;
    anyhow::ensure!(hf_raw_bytes.len() == tokens * ple_width * 2,
        "hf raw length mismatch: {} vs {}", hf_raw_bytes.len(), tokens * ple_width * 2);
    let hf_raw = bytes_to_bf16_vec(hf_raw_bytes);
    let (raw_abs, _, raw_rel) = compare_bf16(&raw_scaled, &hf_raw);
    println!();
    println!("-- per_layer_inputs_raw (scaled gather) --");
    println!("  max abs diff           {:.3e}", raw_abs);
    println!("  global rel diff        {:.3e}", raw_rel);
    anyhow::ensure!(raw_rel <= 1e-2, "raw gather vs HF: global rel {raw_rel}");

    // ---- per_layer_model_projection linear: need inputs_embeds (scaled) on device.
    let embed_bytes = mm.load_bf16("model.language_model.embed_tokens.weight")?;
    let mut d_embed_table: DeviceBuffer<bf16> = DeviceBuffer::new(vocab * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ids: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_inputs_embeds: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_embed_table.copy_from_host_bytes(embed_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ids.copy_from_host(&ids_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    let embed_scale = (hidden as f32).sqrt();
    embed_gather_bf16(&mut d_inputs_embeds, &d_embed_table, &d_ids, tokens, vocab, hidden, embed_scale, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Dequant per_layer_model_projection (FP4 [ple_width, hidden]) onto device.
    let plmp = mm.load_quant_linear("model.language_model.per_layer_model_projection")?;
    anyhow::ensure!(plmp.out_features == ple_width && plmp.in_features == hidden,
        "plmp shape");
    let mut d_plmp_w: DeviceBuffer<bf16> = DeviceBuffer::new(ple_width * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_plmp_p: DeviceBuffer<u8> = DeviceBuffer::new(plmp.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_plmp_s: DeviceBuffer<u8> = DeviceBuffer::new(plmp.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_plmp_p.copy_from_host_bytes(plmp.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_plmp_s.copy_from_host_bytes(plmp.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_plmp_w, &d_plmp_p, &d_plmp_s, plmp.global_scale,
                     plmp.out_features, plmp.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Linear with alpha = 1/sqrt(H) (HF scales the projection before reshape).
    let proj_scale = (hidden as f32).powf(-0.5);
    let mut d_ctx: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_ctx, &d_inputs_embeds, &d_plmp_w, None,
                   tokens, ple_width, hidden, proj_scale, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // RMSNorm with per_layer_projection_norm over the Hl axis for each (t, l).
    // View ctx as [tokens*L, Hl] rows.
    let pln_bytes = mm.load_bf16("model.language_model.per_layer_projection_norm.weight")?;
    anyhow::ensure!(pln_bytes.len() == per_layer * 2, "pln weight size");
    let mut d_pln: DeviceBuffer<bf16> = DeviceBuffer::new(per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_pln.copy_from_host_bytes(pln_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_ctx_normed: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_ctx_normed, &d_ctx, Some(&d_pln),
                 tokens * n_layers, per_layer, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Combine: (ctx_normed + raw_scaled) * 1/sqrt(2). Upload raw_scaled first.
    let mut d_raw: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_raw.copy_from_host(&raw_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_combined: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let combine_scale = (0.5f32).sqrt();  // 1/sqrt(2)
    add_scale_bf16(&mut d_combined, &d_ctx_normed, &d_raw, combine_scale, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    let combined = d_combined.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let hf_proj_bytes = hf_mm.load_bf16("per_layer_inputs_projected")?;
    anyhow::ensure!(hf_proj_bytes.len() == tokens * ple_width * 2,
        "hf projected length mismatch");
    let hf_proj = bytes_to_bf16_vec(hf_proj_bytes);
    let (proj_abs, _, proj_rel) = compare_bf16(&combined, &hf_proj);
    println!();
    println!("-- per_layer_inputs_projected (full assembly) --");
    println!("  max abs diff           {:.3e}", proj_abs);
    println!("  global rel diff        {:.3e}", proj_rel);
    let tol: f32 = 5e-2;
    anyhow::ensure!(proj_rel <= tol, "projected vs HF: global rel {proj_rel} > {tol}");
    println!("OK: full PLE assembly matches HF within {tol:.1e}.");
    Ok(())
}

fn cmd_test_vs_hf_tail(model_dir: PathBuf, hf_path: PathBuf) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&model_dir.join("config.json"))?;
    let tc = &cfg.text_config;
    let hidden = tc.hidden_size;
    let vocab = tc.vocab_size;
    let n_layers = tc.num_hidden_layers;
    let eps = tc.rms_norm_eps as f32;
    let softcap = tc.final_logit_softcapping.unwrap_or(30.0) as f32;

    println!("=== xenon-cli test-vs-hf-tail ===");
    println!("hidden / vocab / softcap {} / {} / {}", hidden, vocab, softcap);

    let mm = MmapWeights::open(&model_dir.join("model.safetensors"))?;
    let hf_mm = MmapWeights::open(&hf_path)?;

    // Input: HF's layer_{last}.final.
    let last_final_name = format!("layer_{:02}.final", n_layers - 1);
    let last_bytes = hf_mm.load_bf16(&last_final_name)?;
    let tokens = last_bytes.len() / (hidden * 2);
    anyhow::ensure!(last_bytes.len() == tokens * hidden * 2, "last_final shape");

    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host_bytes(last_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Stage 1: final RMSNorm.
    let norm_bytes = mm.load_bf16("model.language_model.norm.weight")?;
    let mut d_norm_w: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_norm_w.copy_from_host_bytes(norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_final_norm: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_final_norm, &d_h, Some(&d_norm_w), tokens, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let ours_final_norm = d_final_norm.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let hf_final_norm_bytes = hf_mm.load_bf16("final_norm_out")?;
    let hf_final_norm = bytes_to_bf16_vec(hf_final_norm_bytes);
    let (fn_abs, _, fn_rel) = compare_bf16(&ours_final_norm, &hf_final_norm);
    println!();
    println!("-- final RMSNorm --");
    println!("  max abs diff           {:.3e}", fn_abs);
    println!("  global rel diff        {:.3e}", fn_rel);
    anyhow::ensure!(fn_rel <= 5e-2, "final norm rel {fn_rel}");

    // Stage 2: lm_head (tied: y = h @ embed_tokens^T).
    let embed_bytes = mm.load_bf16("model.language_model.embed_tokens.weight")?;
    let mut d_lm_head: DeviceBuffer<bf16> = DeviceBuffer::new(vocab * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t_upload = Instant::now();
    d_lm_head.copy_from_host_bytes(embed_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1e3;
    println!();
    println!("-- lm_head upload --");
    println!("  bytes                  {:.2} GiB", (vocab * hidden * 2) as f64 / 1073741824.0);
    println!("  elapsed                {:.1} ms", upload_ms);

    let mut d_logits: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    let t_gemm = Instant::now();
    lt.linear_bf16(&mut d_logits, &d_final_norm, &d_lm_head, None,
                    tokens, vocab, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let gemm_ms = t_gemm.elapsed().as_secs_f64() * 1e3;
    let ours_pre = d_logits.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let hf_pre_bytes = hf_mm.load_bf16("logits_pre_softcap")?;
    let hf_pre = bytes_to_bf16_vec(hf_pre_bytes);
    let (pre_abs, _, pre_rel) = compare_bf16(&ours_pre, &hf_pre);
    println!();
    println!("-- lm_head (tied) --");
    println!("  gemm                   {:.1} ms", gemm_ms);
    println!("  max abs diff           {:.3e}", pre_abs);
    println!("  global rel diff        {:.3e}", pre_rel);
    anyhow::ensure!(pre_rel <= 5e-2, "logits_pre rel {pre_rel}");

    // Stage 3: softcap.
    let mut d_logits_capped: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    softcap_bf16(&mut d_logits_capped, &d_logits, softcap, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let ours_post = d_logits_capped.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let hf_post_bytes = hf_mm.load_bf16("logits_post_softcap")?;
    let hf_post = bytes_to_bf16_vec(hf_post_bytes);
    let (post_abs, _, post_rel) = compare_bf16(&ours_post, &hf_post);
    println!();
    println!("-- softcap tanh(x/{})*{} --", softcap, softcap);
    println!("  max abs diff           {:.3e}", post_abs);
    println!("  global rel diff        {:.3e}", post_rel);
    anyhow::ensure!(post_rel <= 5e-2, "logits_post rel {post_rel}");

    // Top-k sanity: our next-token predictions should match HF's.
    let last_row = &ours_post[(tokens - 1) * vocab..];
    let hf_last_row = &hf_post[(tokens - 1) * vocab..];
    let mut ours_topk: Vec<(usize, f32)> = last_row.iter().enumerate()
        .map(|(i, v)| (i, v.to_f32())).collect();
    ours_topk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut hf_topk: Vec<(usize, f32)> = hf_last_row.iter().enumerate()
        .map(|(i, v)| (i, v.to_f32())).collect();
    hf_topk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!();
    println!("-- top-5 next token for last row --");
    println!("  {:<10}  {:<20}  {:<10}  {:<20}", "ours id", "ours logit", "hf id", "hf logit");
    for i in 0..5 {
        println!("  {:<10}  {:<20.3}  {:<10}  {:<20.3}",
                 ours_topk[i].0, ours_topk[i].1,
                 hf_topk[i].0, hf_topk[i].1);
    }

    println!();
    println!("OK: post-stack tail (final norm + lm_head + softcap) matches HF.");
    Ok(())
}

/// NVFP4-quantized linear kept on-device in its packed form. Callers dequant
/// to a bf16 scratch buffer just before they need the weight for a GEMM; that
/// way we can keep all 42 layers' weights resident (~2.1 GiB total packed vs
/// ~7.5 GiB dequanted) and only pay for peak scratch once per layer.
struct QuantLinearDev {
    packed: DeviceBuffer<u8>,
    scales: DeviceBuffer<u8>,
    scales_swizzled: DeviceBuffer<u8>,
    global_scale: f32,
    input_scale: f32,
    out_features: usize,
    in_features: usize,
}

impl QuantLinearDev {
    fn load(q: &xenon_core::QuantLinearRef<'_>) -> anyhow::Result<Self> {
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
            input_scale: q.input_scale,
            out_features: n,
            in_features: k,
        })
    }

    fn dequant_to(&self, out: &mut DeviceBuffer<bf16>, stream: &Stream) -> anyhow::Result<()> {
        fp4_dequant_bf16(out, &self.packed, &self.scales, self.global_scale,
                          self.out_features, self.in_features, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Three-way dispatch by M — see xenon-engine's QuantLinearDev::forward
    /// doc for the full rationale.
    fn forward(
        &self,
        lt: &mut CublasLt,
        y: &mut DeviceBuffer<bf16>,
        x: &DeviceBuffer<bf16>,
        m: usize,
        stream: &Stream,
    ) -> anyhow::Result<()> {
        let n = self.out_features;
        let k = self.in_features;
        let k_blocks = k / 16;
        if m == 1 {
            fp4_gemv_bf16(y, x, &self.packed, &self.scales, self.global_scale,
                           n, k, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))
        } else if m >= 128 {
            let mut x_packed: DeviceBuffer<u8> =
                DeviceBuffer::new_async(m * (k / 2), stream).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut x_scales_linear: DeviceBuffer<u8> =
                DeviceBuffer::new_async(m * k_blocks, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
            nvfp4_quantize_bf16(&mut x_packed, &mut x_scales_linear, x, m, k, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let m_padded = round_up(m, 128);
            let kb_padded = round_up(k_blocks, 4);
            let mut x_scales_swizzled: DeviceBuffer<u8> =
                DeviceBuffer::new_async(m_padded * kb_padded, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
            swizzle_blockscale_ue4m3(&mut x_scales_swizzled, &x_scales_linear, m, k_blocks, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let alpha_scale = self.global_scale;
            lt.linear_nvfp4(y, &x_packed, &x_scales_swizzled,
                             &self.packed, &self.scales_swizzled,
                             alpha_scale, None, m, n, k, 1.0, 0.0, Some(stream))
                .map_err(|e| anyhow::anyhow!("{e}"))
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

/// Decoder-layer weights resident on device. Quantized linears stay FP4;
/// norms and `layer_scalar` are bf16/f32. Owner-KV layers populate
/// `k_proj`/`v_proj`/`k_norm`; shared-KV layers leave them `None`.
struct LayerWeights {
    input_layernorm: DeviceBuffer<bf16>,
    post_attention_layernorm: DeviceBuffer<bf16>,
    pre_feedforward_layernorm: DeviceBuffer<bf16>,
    post_feedforward_layernorm: DeviceBuffer<bf16>,
    post_per_layer_input_norm: DeviceBuffer<bf16>,
    q_norm: DeviceBuffer<bf16>,
    k_norm: Option<DeviceBuffer<bf16>>,

    q_proj: QuantLinearDev,
    k_proj: Option<QuantLinearDev>,
    v_proj: Option<QuantLinearDev>,
    o_proj: QuantLinearDev,

    gate_proj: QuantLinearDev,
    up_proj: QuantLinearDev,
    down_proj: QuantLinearDev,

    per_layer_input_gate: QuantLinearDev,
    per_layer_projection: QuantLinearDev,

    layer_scalar: f32,
}

fn load_layer_weights(
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

/// Per-layer metadata passed to the forward step.
///
/// `t_q` is the number of query tokens (1 for decode, prompt-length for
/// prefill). `t_kv` is the total cached K/V length *after* this call's
/// append (== `kv.cur_len` + `t_q` for own-KV layers, or the owner's
/// cached length for shared layers). `q_pos_base` is the absolute position
/// of the first query token: 0 for prefill, `t_kv - 1` for a single
/// decode step.
#[derive(Clone, Copy)]
struct LayerMeta {
    layer_idx: usize,
    t_q: usize,
    t_kv: usize,
    q_pos_base: i32,
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

/// Run one decoder layer's forward, updating `h` in place and writing K/V to
/// the cache for owner layers. Shared layers skip the K/V path and read from
/// `kv` via `slot_for_layer`.
///
/// `h` is `[t_q, hidden]`. For prefill t_q = T; for decode t_q = 1.
/// `positions` holds the RoPE position for each of the `t_q` query tokens
/// (prefill: `[0..T)`, decode: `[cur_len]`).
#[allow(clippy::too_many_arguments)]
fn layer_forward(
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

    // Scratch buffers via cudaMallocAsync (default stream-ordered pool).
    // This caches freed blocks per-size, so the 20 per-layer allocs become
    // near-free after the first step instead of hitting cudaMalloc each time.
    let mut d_residual: DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_normed:   DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_tmp:      DeviceBuffer<bf16> = DeviceBuffer::new_async(t_q * hidden, stream).map_err(|e| anyhow::anyhow!("{e}"))?;

    // ---- Attention block ----
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

    // ---- MLP block ----
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

    // ---- PLE block ----
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

    // ---- layer_scalar tail multiply ----
    let h_in = clone_buffer_async(h, stream)?;
    scale_bf16(h, &h_in, lw.layer_scalar, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

fn clone_buffer_async(src: &DeviceBuffer<bf16>, stream: &Stream) -> anyhow::Result<DeviceBuffer<bf16>> {
    let mut dst = DeviceBuffer::<bf16>::new_async(src.len(), stream).map_err(|e| anyhow::anyhow!("{e}"))?;
    dst.copy_from_device(src).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(dst)
}

/// Weights shared across all decoder layers. `lm_head` is the 1.34 GiB
/// `embed_tokens` table kept device-resident — with the decoder stack at
/// ~11 ms post fused-FP4-gemv + shmem LUT, the PCIe transfer it used to
/// do per-step no longer hides and became the dominant host-side stall.
struct TopLevelWeights {
    per_layer_model_projection: QuantLinearDev,
    per_layer_projection_norm: DeviceBuffer<bf16>,
    norm: DeviceBuffer<bf16>,
    lm_head: DeviceBuffer<bf16>,
}

/// Shape/meta precomputed once for a model.
#[derive(Clone, Copy)]
struct ModelShape {
    hidden: usize,
    inter: usize,
    vocab: usize,
    h_heads: usize,
    h_kv: usize,
    per_layer: usize,
    n_layers: usize,
    ple_width: usize,
    eps: f32,
    softcap: f32,
}

/// Build the KV sharing map: layers 0..first_shared own; shared layers
/// point at the last own-KV layer of the same kind.
fn build_kv_cache(mm: &MmapWeights, cfg: &GemmaConfig, max_len: usize)
    -> anyhow::Result<KvCache>
{
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

/// Run one forward pass for `ids` (prefill for T>1 or a decode step for T=1).
/// Appends new K/V to `kv` and advances `cur_len` by `ids.len()`. Returns the
/// post-softcap logits for the LAST row as a `[vocab]` host vector (argmax
/// on a small host copy is trivial and removes a device dep).
///
/// `stream_lm` is a separate CUDA stream used to upload the 1.25 GiB lm_head
/// weights asynchronously while the decoder stack runs on `stream`. Pass
/// the same stream if you don't want overlap.
#[allow(clippy::too_many_arguments)]
fn forward_step(
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
    let _ = stream_lm; // retained for signature compatibility; lm_head is resident.

    // positions for RoPE: absolute indices q_pos_base..q_pos_base+t_q.
    let positions_host: Vec<i32> = (0..t_q as i32).map(|i| q_pos_base + i).collect();
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(t_q).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&positions_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    // ---- Input embedding (host-offloaded) ----
    let ie_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens.weight", ids, hidden)?;
    let embed_scale = (hidden as f32).sqrt();
    let ie_scaled: Vec<bf16> = ie_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * embed_scale)).collect();
    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host(&ie_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    // ---- PLE assembly ----
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
    drop(d_plmp_w);
    drop(d_raw);
    drop(d_ctx);
    drop(d_ctx_normed);

    // ---- Decoder stack ----
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

    // ---- Tail: final norm + lm_head + softcap on the LAST row only ----
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

fn cmd_bench_batch(
    model_dir: PathBuf,
    prompt: String,
    chat: bool,
    batch: usize,
    max_new: usize,
    max_len: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(batch >= 1, "batch must be >= 1");
    let t_load = Instant::now();
    let mut engine = xenon_engine::Engine::load(&model_dir, max_len)?;
    let load_s = t_load.elapsed().as_secs_f64();

    let text = if chat { xenon_engine::wrap_chat_prompt(&prompt) } else { prompt.clone() };
    let prompt_ids: Vec<u32> = engine.tokenize(&text, true)?;
    println!("=== xenon-cli bench-batch ===");
    println!("loaded in              {:.1} s", load_s);
    println!("batch / prompt_len     {} / {}", batch, prompt_ids.len());
    println!("max_new per request    {}", max_new);

    let requests: Vec<xenon_engine::BatchRequest> = (0..batch)
        .map(|_| xenon_engine::BatchRequest {
            prompt_ids: prompt_ids.clone(),
            max_new,
        }).collect();

    let t0 = Instant::now();
    let results = engine.generate_batch(requests, xenon_engine::GEMMA4_EOS, |_i, _tok| true)?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

    let total_tokens: usize = results.iter().map(|r| r.generated.len()).sum();
    let prefill_total_ms: f64 = results.iter().map(|r| r.prefill_ms).sum();
    let decode_ms_total: f64 = if !results[0].decode_ms.is_empty() {
        results[0].decode_ms.iter().sum()
    } else { 0.0 };
    let decode_steps = results[0].decode_ms.len();
    let per_step_mean = if decode_steps > 0 { decode_ms_total / decode_steps as f64 } else { 0.0 };

    println!();
    println!("prefill  (total serial)  {:>7.1} ms", prefill_total_ms);
    println!("decode   ({} steps batched) {:>7.1} ms  ({:.2} ms/step)",
             decode_steps, decode_ms_total, per_step_mean);
    println!("wall                     {:>7.1} ms", wall_ms);
    println!();
    println!("tokens   total           {}", total_tokens);
    println!("tok/s    aggregate       {:>6.2}  (total tokens / wall)",
             total_tokens as f64 / (wall_ms / 1000.0));
    println!("tok/s    per-request     {:>6.2}  (aggregate / batch)",
             total_tokens as f64 / (wall_ms / 1000.0) / batch as f64);
    println!("tok/s    decode-only agg {:>6.2}  (tokens in decode phase / decode wall)",
             (decode_steps * batch) as f64 / (decode_ms_total / 1000.0));

    // Print one request's output as a sanity check.
    let first_text = engine.decode(&results[0].generated, true)?;
    println!();
    println!("request 0 output: {}", first_text);
    Ok(())
}

fn cmd_generate(model_dir: PathBuf, prompt: String, max_new: usize, max_len: usize, chat: bool) -> anyhow::Result<()> {
    use std::io::Write;
    let t_load = Instant::now();
    let mut engine = xenon_engine::Engine::load(&model_dir, max_len)?;
    let (mem_free, mem_total) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let text = if chat { xenon_engine::wrap_chat_prompt(&prompt) } else { prompt.clone() };
    let prompt_ids: Vec<u32> = engine.tokenize(&text, true)?;
    eprintln!(
        "[xenon] loaded {} layers in {:.1} s; vram {:.2}/{:.2} GiB free; prompt {} tokens",
        engine.shape.n_layers, load_ms / 1000.0,
        mem_free as f64 / 1073741824.0, mem_total as f64 / 1073741824.0,
        prompt_ids.len()
    );

    // Stream tokens as they're generated. Decode the accumulated ids after
    // each step and emit the diff so subword pieces assemble into proper UTF-8.
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut generated: Vec<u32> = Vec::new();
    let mut prev_decoded = String::new();
    let tok = engine.tokenizer.clone();

    let stats = engine.generate(&prompt_ids, max_new, xenon_engine::GEMMA4_EOS, |id| {
        if xenon_engine::GEMMA4_EOS.contains(&id) && !generated.is_empty() { return true; }
        generated.push(id);
        if let Ok(s) = tok.decode(&generated, /*skip_special*/ true) {
            if s.len() >= prev_decoded.len() && s.starts_with(prev_decoded.as_str()) {
                let _ = out.write_all(s[prev_decoded.len()..].as_bytes());
                let _ = out.flush();
                prev_decoded = s;
            }
        }
        true
    })?;
    writeln!(out)?;
    drop(out);

    let decode_total_ms: f64 = stats.decode_ms.iter().sum();
    eprintln!("[xenon] prefill {:.1} ms ({:.1} tok/s)",
             stats.prefill_ms,
             stats.prompt_len as f64 / (stats.prefill_ms / 1000.0));
    eprintln!("[xenon] decoded {} tokens in {:.1} ms ({:.1} tok/s, prefill excluded)",
             stats.decode_ms.len(), decode_total_ms,
             stats.decode_ms.len() as f64 / (decode_total_ms / 1000.0));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_bench(
    model_dir: PathBuf,
    prompt: String,
    chat: bool,
    max_new: usize,
    max_len: usize,
    warmup: usize,
    iters: usize,
    label: String,
) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&model_dir.join("config.json"))?;
    let tc = &cfg.text_config;
    let shape = ModelShape {
        hidden: tc.hidden_size,
        inter: tc.intermediate_size,
        vocab: tc.vocab_size,
        h_heads: tc.num_attention_heads,
        h_kv: tc.num_key_value_heads,
        per_layer: tc.hidden_size_per_layer_input.ok_or_else(|| anyhow::anyhow!("missing"))?,
        n_layers: tc.num_hidden_layers,
        ple_width: tc.hidden_size_per_layer_input.unwrap_or(0) * tc.num_hidden_layers,
        eps: tc.rms_norm_eps as f32,
        softcap: tc.final_logit_softcapping.unwrap_or(30.0) as f32,
    };

    let tok = Tokenizer::from_dir(&model_dir)?;
    let mm = MmapWeights::open(&model_dir.join("model.safetensors"))?;

    let text = if chat {
        format!("<|turn>user\n{}<turn|>\n<|turn>model\n", prompt)
    } else { prompt.clone() };
    let prompt_ids_u: Vec<u32> = tok.encode(&text, true)?;
    let prompt_ids: Vec<i32> = prompt_ids_u.iter().map(|&x| x as i32).collect();
    anyhow::ensure!(prompt_ids.len() + max_new <= max_len, "exceeds max_len");

    println!("=== xenon-cli bench ===");
    println!("label               {}", label);
    println!("prompt_len          {}", prompt_ids.len());
    println!("max_new             {}", max_new);
    println!("warmup / iters      {} / {}", warmup, iters);

    let t_load = Instant::now();
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
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let (mem_free, mem_total) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("weights loaded      {:.1} s  (vram {:.2}/{:.2} GiB free)",
             load_ms / 1000.0,
             mem_free as f64 / 1073741824.0, mem_total as f64 / 1073741824.0);

    let eos: Vec<i32> = vec![1, 106];

    // One run = fresh KvCache + prefill + up-to-max_new decode steps, with
    // per-step timings captured.
    let stream_lm = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let run_once = |layers: &[LayerWeights], top: &TopLevelWeights,
                    lt: &mut CublasLt, stream: &Stream, stream_lm: &Stream|
        -> anyhow::Result<(f64, Vec<f64>, usize)>
    {
        let mut kv = build_kv_cache(&mm, &cfg, max_len)?;
        let t_prefill = Instant::now();
        let logits0 = forward_step(&mm, &cfg, shape, layers, top, &mut kv,
                                    &prompt_ids, 0, lt, stream, stream_lm)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        let mut next = argmax_bf16(&logits0);
        let mut decode_mss: Vec<f64> = Vec::with_capacity(max_new);
        let mut steps = 0usize;
        for step in 0..max_new {
            if eos.contains(&next) && step > 0 { break; }
            let q_pos = kv.len() as i32;
            let t_step = Instant::now();
            let logits = forward_step(&mm, &cfg, shape, layers, top, &mut kv,
                                       &[next], q_pos, lt, stream, stream_lm)?;
            decode_mss.push(t_step.elapsed().as_secs_f64() * 1e3);
            next = argmax_bf16(&logits);
            steps += 1;
        }
        Ok((prefill_ms, decode_mss, steps))
    };

    // Warmup.
    for i in 0..warmup {
        let (pfm, dec, steps) = run_once(&layers, &top, &mut lt, &stream, &stream_lm)?;
        println!("warmup {:>2}          prefill {:>6.1} ms   decode {} steps   avg {:>6.1} ms",
                 i, pfm, steps,
                 if !dec.is_empty() { dec.iter().sum::<f64>() / dec.len() as f64 } else { 0.0 });
    }

    // Measured runs.
    let mut all_prefill_ms: Vec<f64> = Vec::new();
    let mut all_decode_ms: Vec<f64> = Vec::new();
    let mut all_steps: Vec<usize> = Vec::new();
    for i in 0..iters {
        let (pfm, dec, steps) = run_once(&layers, &top, &mut lt, &stream, &stream_lm)?;
        println!("iter   {:>2}          prefill {:>6.1} ms   decode {} steps   avg {:>6.1} ms",
                 i, pfm, steps,
                 if !dec.is_empty() { dec.iter().sum::<f64>() / dec.len() as f64 } else { 0.0 });
        all_prefill_ms.push(pfm);
        all_decode_ms.extend(dec);
        all_steps.push(steps);
    }

    // Single profiled decode step: time the major phases with host-side
    // stream.synchronize() to attribute wall time to each block. Adds
    // sync overhead, so treat as diagnostic, not throughput.
    {
        let mut kv = build_kv_cache(&mm, &cfg, max_len)?;
        let _ = forward_step(&mm, &cfg, shape, &layers, &top, &mut kv,
                              &prompt_ids, 0, &mut lt, &stream, &stream_lm)?;
        let next_tok = 107i32;
        let q_pos = kv.len() as i32;
        let phases = forward_step_profiled(&mm, &cfg, shape, &layers, &top, &mut kv,
                                           &[next_tok], q_pos, &mut lt, &stream, &stream_lm)?;
        println!();
        println!("=== profiled decode step (phase times, ms) ===");
        for (name, ms) in &phases {
            println!("  {:<28} {:>7.2}", name, ms);
        }
    }

    // Summary.
    let prefill_mean = mean(&all_prefill_ms);
    let (p50, p95) = percentiles(&all_decode_ms);
    let decode_mean = mean(&all_decode_ms);
    let prefill_tps = prompt_ids.len() as f64 / (prefill_mean / 1000.0);
    let decode_tps = 1.0 / (decode_mean / 1000.0);
    println!();
    println!("=== summary ===");
    println!("prefill             {:>7.1} ms ({:>6.1} tok/s at T={})", prefill_mean, prefill_tps, prompt_ids.len());
    println!("decode per-step     mean {:.2} ms  p50 {:.2}  p95 {:.2}", decode_mean, p50, p95);
    println!("decode throughput   {:>7.2} tok/s (mean)", decode_tps);

    // Dump JSON.
    let ts = chrono_now_iso();
    let git_hash = current_git_hash().unwrap_or_else(|| "unknown".into());
    let results_dir = std::path::Path::new("results");
    std::fs::create_dir_all(results_dir)?;
    let fname = format!("bench-{}-{}.json", label, ts);
    let path = results_dir.join(&fname);
    let j = serde_json::json!({
        "timestamp": ts,
        "label": label,
        "git_hash": git_hash,
        "model": model_dir.to_string_lossy(),
        "prompt": prompt,
        "chat": chat,
        "prompt_len": prompt_ids.len(),
        "max_new": max_new,
        "max_len": max_len,
        "warmup_iters": warmup,
        "measured_iters": iters,
        "weights_load_s": load_ms / 1000.0,
        "vram_free_gib": mem_free as f64 / 1073741824.0,
        "vram_total_gib": mem_total as f64 / 1073741824.0,
        "prefill_ms_per_run": all_prefill_ms,
        "decode_ms_per_step": all_decode_ms,
        "decode_steps_per_run": all_steps,
        "summary": {
            "prefill_ms_mean": prefill_mean,
            "prefill_tok_per_s": prefill_tps,
            "decode_ms_mean": decode_mean,
            "decode_ms_p50": p50,
            "decode_ms_p95": p95,
            "decode_tok_per_s_mean": decode_tps,
        },
    });
    std::fs::write(&path, serde_json::to_string_pretty(&j)?)?;
    println!();
    println!("wrote {}", path.display());
    Ok(())
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}
fn percentiles(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() { return (0.0, 0.0); }
    let mut s: Vec<f64> = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| {
        let idx = ((s.len() - 1) as f64 * q).round() as usize;
        s[idx]
    };
    (p(0.50), p(0.95))
}
fn current_git_hash() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"]).output().ok()?;
    if !out.status.success() { return None; }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
fn chrono_now_iso() -> String {
    // Very simple local-time ISO-ish stamp; avoids pulling chrono in.
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let secs = now.as_secs();
    format!("{}", secs)
}

/// Variant of forward_step that synchronizes and times each major phase.
/// Sync overhead inflates wall time; use only for diagnostic breakdown.
#[allow(clippy::too_many_arguments)]
fn forward_step_profiled(
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
) -> anyhow::Result<Vec<(&'static str, f64)>> {
    let tc = &cfg.text_config;
    let t_q = ids.len();
    let t_kv = kv.len() + t_q;
    let ModelShape { hidden, inter: _, vocab, h_heads, h_kv, per_layer, n_layers, ple_width, eps, softcap } = shape;

    let mut phases: Vec<(&'static str, f64)> = Vec::new();
    let mark = |phases: &mut Vec<(&'static str, f64)>, name: &'static str, t: Instant, stream: &Stream| -> anyhow::Result<()> {
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        phases.push((name, t.elapsed().as_secs_f64() * 1e3));
        Ok(())
    };

    let _ = stream_lm; // retained for signature compatibility; lm_head is resident.
    // lm_head is device-resident as of the lm_head-resident change; nothing
    // to upload per-step. Kept phase name so the output columns stay stable.
    phases.push(("lm_head H2D issue", 0.0));

    let positions_host: Vec<i32> = (0..t_q as i32).map(|i| q_pos_base + i).collect();
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(t_q).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&positions_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    let t_embed = Instant::now();
    let ie_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens.weight", ids, hidden)?;
    let embed_scale = (hidden as f32).sqrt();
    let ie_scaled: Vec<bf16> = ie_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * embed_scale)).collect();
    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(t_q * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host(&ie_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;
    mark(&mut phases, "embed gather+upload", t_embed, stream)?;

    let t_ple = Instant::now();
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
    mark(&mut phases, "ple assembly", t_ple, stream)?;

    let t_stack = Instant::now();
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
            hidden, inter: shape.inter, h_heads, h_kv,
            head_dim: d_layer, per_layer, eps, window, rope_theta, rotary_dim,
            owns_kv: mm.layer_owns_kv(layer_idx),
        };
        layer_forward(&layers[layer_idx], meta, &mut d_h, &d_ple_layer,
                       &d_positions, kv, lt, stream)?;
    }
    kv.advance(t_q);
    mark(&mut phases, "decoder stack (42 layers)", t_stack, stream)?;

    let t_tail = Instant::now();
    let mut d_last_row: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_last_row.copy_region_from_device(0, &d_h, (t_q - 1) * hidden, hidden)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_normed: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, &d_last_row, Some(&top.norm), 1, hidden, eps, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    mark(&mut phases, "final RMSNorm", t_tail, stream)?;

    // lm_head is resident — no H2D to wait for. Kept row for phase stability.
    phases.push(("wait for lm_head H2D", 0.0));

    let t_lm = Instant::now();
    let mut d_logits: DeviceBuffer<bf16> = DeviceBuffer::new(vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_logits, &d_normed, &top.lm_head, None,
                    1, vocab, hidden, 1.0, 0.0, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_capped: DeviceBuffer<bf16> = DeviceBuffer::new(vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    softcap_bf16(&mut d_capped, &d_logits, softcap, Some(stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    mark(&mut phases, "lm_head GEMM + softcap", t_lm, stream)?;

    let t_d2h = Instant::now();
    let _host = d_capped.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    phases.push(("logits D2H", t_d2h.elapsed().as_secs_f64() * 1e3));

    Ok(phases)
}

fn argmax_bf16(row: &[bf16]) -> i32 {
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

fn cmd_test_vs_hf_full(model_dir: PathBuf, hf_path: PathBuf, ids_csv: String) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&model_dir.join("config.json"))?;
    let tc = &cfg.text_config;
    let hidden = tc.hidden_size;
    let inter = tc.intermediate_size;
    let vocab = tc.vocab_size;
    let h_heads = tc.num_attention_heads;
    let h_kv = tc.num_key_value_heads;
    let per_layer = tc.hidden_size_per_layer_input
        .ok_or_else(|| anyhow::anyhow!("hidden_size_per_layer_input missing"))?;
    let n_layers = tc.num_hidden_layers;
    let ple_width = per_layer * n_layers;
    let eps = tc.rms_norm_eps as f32;
    let softcap = tc.final_logit_softcapping.unwrap_or(30.0) as f32;
    let num_kv_shared = tc.num_kv_shared_layers.unwrap_or(0);

    let ids_host: Vec<i32> = ids_csv
        .split(',').map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()?;
    let tokens = ids_host.len();
    let pos_host: Vec<i32> = (0..tokens as i32).collect();

    let (mem_free_start, mem_total) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("=== xenon-cli test-vs-hf-full ===");
    println!("tokens                   {}", tokens);
    println!("hidden / vocab / layers  {} / {} / {}", hidden, vocab, n_layers);
    println!("num_kv_shared            {}", num_kv_shared);
    println!("vram at start            {:>6.2} / {:>6.2} GiB free",
             mem_free_start as f64 / 1073741824.0, mem_total as f64 / 1073741824.0);

    let mm = MmapWeights::open(&model_dir.join("model.safetensors"))?;
    let hf_mm = MmapWeights::open(&hf_path)?;

    // ---- Build KV cache slot map (only own-KV layers get their own slot;
    //      shared layers point at the last own-KV layer of the same kind). ----
    let first_shared = n_layers - num_kv_shared;
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
    let mut kv = KvCache::new(slot_specs, slot_for_layer, tokens).map_err(|e| anyhow::anyhow!("{e}"))?;

    // ---- TOP-LEVEL: embed + PLE assembly ----
    // Input embedding is host-offloaded: gather just the T rows we need on
    // the host and upload [T, H] instead of the full [V, H] table (saves
    // 1.25 GiB resident VRAM vs. the naive "upload whole table" approach).
    let embed_scale = (hidden as f32).sqrt();
    let ie_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens.weight", &ids_host, hidden)?;
    let ie_scaled: Vec<bf16> = ie_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * embed_scale)).collect();
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&pos_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_h.copy_from_host(&ie_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    // PLE assembly.
    let raw_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens_per_layer.weight", &ids_host, ple_width)?;
    let raw_scale = (per_layer as f32).sqrt();
    let raw_scaled: Vec<bf16> = raw_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * raw_scale)).collect();
    let mut d_raw: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_raw.copy_from_host(&raw_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    let plmp = mm.load_quant_linear("model.language_model.per_layer_model_projection")?;
    let mut d_plmp_w: DeviceBuffer<bf16> = DeviceBuffer::new(ple_width * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    {
        let mut dp: DeviceBuffer<u8> = DeviceBuffer::new(plmp.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ds: DeviceBuffer<u8> = DeviceBuffer::new(plmp.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        dp.copy_from_host_bytes(plmp.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
        ds.copy_from_host_bytes(plmp.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
        fp4_dequant_bf16(&mut d_plmp_w, &dp, &ds, plmp.global_scale,
                          plmp.out_features, plmp.in_features, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let pln_bytes = mm.load_bf16("model.language_model.per_layer_projection_norm.weight")?;
    let mut d_pln: DeviceBuffer<bf16> = DeviceBuffer::new(per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_pln.copy_from_host_bytes(pln_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_ctx: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ctx_normed: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_combined: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_ctx, &d_h, &d_plmp_w, None,
                    tokens, ple_width, hidden, (hidden as f32).powf(-0.5), 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_ctx_normed, &d_ctx, Some(&d_pln),
                  tokens * n_layers, per_layer, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut d_combined, &d_ctx_normed, &d_raw, (0.5f32).sqrt(), Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    // ---- Decoder stack ----
    let t_stack = Instant::now();
    let mut d_ple_layer: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    for layer_idx in 0..n_layers {
        let lw = load_layer_weights(&mm, &cfg, layer_idx, &stream)?;
        per_layer_slice_bf16(&mut d_ple_layer, &d_combined,
                              tokens, n_layers, per_layer, layer_idx, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let (rope_theta, rotary_dim) = cfg.rope_for_layer(layer_idx);
        let d_layer = cfg.head_dim_for_layer(layer_idx);
        let window = match tc.layer_types[layer_idx] {
            LayerKind::SlidingAttention => tc.sliding_window.unwrap_or(0) as i32,
            LayerKind::FullAttention => 0,
        };
        let meta = LayerMeta {
            layer_idx,
            t_q: tokens, t_kv: tokens, q_pos_base: 0,
            hidden, inter,
            h_heads, h_kv,
            head_dim: d_layer,
            per_layer, eps, window, rope_theta, rotary_dim,
            owns_kv: mm.layer_owns_kv(layer_idx),
        };
        layer_forward(&lw, meta, &mut d_h, &d_ple_layer, &d_positions, &mut kv, &mut lt, &stream)?;
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let stack_ms = t_stack.elapsed().as_secs_f64() * 1e3;
    println!("decoder stack (42 layers) {:.1} ms", stack_ms);

    // Diff final layer's output (should match `layer_{last}.final`).
    let final_layer_name = format!("layer_{:02}.final", n_layers - 1);
    let hf_last_final = bytes_to_bf16_vec(hf_mm.load_bf16(&final_layer_name)?);
    let ours_last_final = d_h.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (ll_abs, _, ll_rel) = compare_bf16(&ours_last_final, &hf_last_final);
    println!();
    println!("-- after {} layers --", n_layers);
    println!("  vs layer_{:02}.final     max_abs {:.3e}  global_rel {:.3e}",
             n_layers - 1, ll_abs, ll_rel);

    // ---- Tail: final norm + lm_head + softcap ----
    let norm_bytes = mm.load_bf16("model.language_model.norm.weight")?;
    let mut d_norm_w: DeviceBuffer<bf16> = DeviceBuffer::new(hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_norm_w.copy_from_host_bytes(norm_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_final_norm: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_final_norm, &d_h, Some(&d_norm_w), tokens, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // lm_head is host-offloaded: transient 1.25 GiB upload just for this
    // GEMM, then freed. Saves permanent VRAM at the cost of ~60 ms H2D
    // per call (can be overlapped with decoder compute in phase 4).
    let (mem_free_before_lm, _) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;
    let lm_head_bytes = mm.load_bf16("model.language_model.embed_tokens.weight")?;
    let mut d_logits: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_logits_capped: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * vocab).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (mem_free_peak, t_lm);
    {
        let mut d_lm_head: DeviceBuffer<bf16> = DeviceBuffer::new(vocab * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
        t_lm = Instant::now();
        d_lm_head.copy_from_host_bytes(lm_head_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        (mem_free_peak, _) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;
        lt.linear_bf16(&mut d_logits, &d_final_norm, &d_lm_head, None,
                        tokens, vocab, hidden, 1.0, 0.0, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        // d_lm_head drops here, VRAM freed.
    }
    let lm_ms = t_lm.elapsed().as_secs_f64() * 1e3;
    softcap_bf16(&mut d_logits_capped, &d_logits, softcap, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (mem_free_after_lm, _) = mem_info().map_err(|e| anyhow::anyhow!("{e}"))?;
    let gib = 1073741824.0f64;
    println!();
    println!("-- lm_head (host-offloaded) --");
    println!("  vram free pre  {:>6.2} GiB", mem_free_before_lm as f64 / gib);
    println!("  vram free peak {:>6.2} GiB   (transient {:.2} GiB)",
             mem_free_peak as f64 / gib,
             (mem_free_before_lm as f64 - mem_free_peak as f64) / gib);
    println!("  vram free post {:>6.2} GiB", mem_free_after_lm as f64 / gib);
    println!("  H2D+GEMM       {:.1} ms", lm_ms);

    // Diff final logits.
    let hf_post = bytes_to_bf16_vec(hf_mm.load_bf16("logits_post_softcap")?);
    let ours_post = d_logits_capped.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (p_abs, _, p_rel) = compare_bf16(&ours_post, &hf_post);
    println!("-- final logits --");
    println!("  vs logits_post_softcap max_abs {:.3e}  global_rel {:.3e}", p_abs, p_rel);

    // Top-5 on last row.
    let our_last = &ours_post[(tokens - 1) * vocab..];
    let hf_last  = &hf_post[(tokens - 1) * vocab..];
    let topk = |row: &[bf16]| {
        let mut v: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, x)| (i, x.to_f32())).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v.truncate(5);
        v
    };
    let ours_top = topk(our_last);
    let hf_top = topk(hf_last);
    println!();
    println!("top-5 next-token (last input position):");
    println!("  {:<10}  {:<12}  {:<10}  {:<12}", "ours id", "ours logit", "hf id", "hf logit");
    for (a, b) in ours_top.iter().zip(hf_top.iter()) {
        println!("  {:<10}  {:<12.3}  {:<10}  {:<12.3}", a.0, a.1, b.0, b.1);
    }

    let tol: f32 = 1e-1;
    anyhow::ensure!(ll_rel <= tol, "layer_last vs HF: {ll_rel}");
    anyhow::ensure!(p_rel  <= tol, "logits vs HF: {p_rel}");
    // Top-1 agreement is the real integration gate.
    anyhow::ensure!(ours_top[0].0 == hf_top[0].0,
        "top-1 mismatch: ours {} vs hf {}", ours_top[0].0, hf_top[0].0);
    println!();
    println!("OK: full 42-layer forward matches HF (top-1 agrees; logits rel ≤ {tol:.1e}).");
    Ok(())
}

fn cmd_test_vs_hf_layer(
    model_dir: PathBuf,
    hf_path: PathBuf,
    ids_csv: String,
    layer: usize,
) -> anyhow::Result<()> {
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut lt = CublasLt::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let cfg = GemmaConfig::from_path(&model_dir.join("config.json"))?;
    let tc = &cfg.text_config;
    anyhow::ensure!(layer < tc.num_hidden_layers, "layer out of range");
    let hidden = tc.hidden_size;
    let inter = tc.intermediate_size;
    let h_heads = tc.num_attention_heads;
    let h_kv = tc.num_key_value_heads;
    let d = cfg.head_dim_for_layer(layer);
    let per_layer = tc.hidden_size_per_layer_input
        .ok_or_else(|| anyhow::anyhow!("hidden_size_per_layer_input missing"))?;
    let n_layers = tc.num_hidden_layers;
    let ple_width = per_layer * n_layers;
    let eps = tc.rms_norm_eps as f32;
    let window: i32 = match tc.layer_types[layer] {
        LayerKind::SlidingAttention => tc.sliding_window.unwrap_or(0) as i32,
        LayerKind::FullAttention => 0,
    };
    let (rope_theta, rotary_dim) = cfg.rope_for_layer(layer);

    let ids_host: Vec<i32> = ids_csv
        .split(',')
        .map(|s| s.trim().parse::<i32>())
        .collect::<std::result::Result<_, _>>()?;
    let tokens = ids_host.len();
    let pos_host: Vec<i32> = (0..tokens as i32).collect();

    println!("=== xenon-cli test-vs-hf-layer ===");
    println!("layer {layer}  kind {:?}  T={tokens}", tc.layer_types[layer]);
    println!("  rope theta / rotary    {rope_theta} / {rotary_dim}  window {window}");

    let mm = MmapWeights::open(&model_dir.join("model.safetensors"))?;

    // --- TOP-LEVEL: inputs_embeds and PLE (same code path as test-vs-hf-ple). ---
    let embed_bytes = mm.load_bf16("model.language_model.embed_tokens.weight")?;
    let vocab = tc.vocab_size;
    let mut d_embed_table: DeviceBuffer<bf16> = DeviceBuffer::new(vocab * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ids: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_inputs_embeds: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_positions: DeviceBuffer<i32> = DeviceBuffer::new(tokens).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_embed_table.copy_from_host_bytes(embed_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_ids.copy_from_host(&ids_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_positions.copy_from_host(&pos_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    embed_gather_bf16(&mut d_inputs_embeds, &d_embed_table, &d_ids,
                       tokens, vocab, hidden, (hidden as f32).sqrt(), Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // PLE assembly: raw + projected -> combined [T, L, Hl].
    let raw_host = mm.gather_rows_bf16(
        "model.language_model.embed_tokens_per_layer.weight", &ids_host, ple_width)?;
    let raw_scale = (per_layer as f32).sqrt();
    let raw_scaled: Vec<bf16> = raw_host.iter()
        .map(|v| bf16::from_f32(v.to_f32() * raw_scale)).collect();
    let mut d_raw: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_raw.copy_from_host(&raw_scaled).map_err(|e| anyhow::anyhow!("{e}"))?;

    let plmp = mm.load_quant_linear("model.language_model.per_layer_model_projection")?;
    let mut d_plmp_w: DeviceBuffer<bf16> = DeviceBuffer::new(ple_width * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_plmp_p: DeviceBuffer<u8> = DeviceBuffer::new(plmp.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_plmp_s: DeviceBuffer<u8> = DeviceBuffer::new(plmp.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_plmp_p.copy_from_host_bytes(plmp.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_plmp_s.copy_from_host_bytes(plmp.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
    fp4_dequant_bf16(&mut d_plmp_w, &d_plmp_p, &d_plmp_s, plmp.global_scale,
                      plmp.out_features, plmp.in_features, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let pln_bytes = mm.load_bf16("model.language_model.per_layer_projection_norm.weight")?;
    let mut d_pln: DeviceBuffer<bf16> = DeviceBuffer::new(per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_pln.copy_from_host_bytes(pln_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_ctx: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ctx_normed: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_combined: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * ple_width).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_ctx, &d_inputs_embeds, &d_plmp_w, None,
                    tokens, ple_width, hidden, (hidden as f32).powf(-0.5), 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_ctx_normed, &d_ctx, Some(&d_pln),
                  tokens * n_layers, per_layer, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut d_combined, &d_ctx_normed, &d_raw,
                    (0.5f32).sqrt(), Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Extract this layer's per_layer_input [T, Hl].
    let mut d_ple_layer: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    per_layer_slice_bf16(&mut d_ple_layer, &d_combined,
                          tokens, n_layers, per_layer, layer, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // --- Load all layer weights. ---
    let prefix = format!("model.language_model.layers.{layer}");
    let in_norm_bytes = mm.load_bf16(&format!("{prefix}.input_layernorm.weight"))?;
    let post_attn_norm_bytes = mm.load_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?;
    let pre_ffn_norm_bytes = mm.load_bf16(&format!("{prefix}.pre_feedforward_layernorm.weight"))?;
    let post_ffn_norm_bytes = mm.load_bf16(&format!("{prefix}.post_feedforward_layernorm.weight"))?;
    let q_norm_bytes = mm.load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?;
    let k_norm_bytes = mm.load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?;
    let post_ple_norm_bytes = mm.load_bf16(&format!("{prefix}.post_per_layer_input_norm.weight"))?;
    let layer_scalar_bytes = mm.load_bf16(&format!("{prefix}.layer_scalar"))?;
    let layer_scalar = {
        let b = layer_scalar_bytes;
        bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
    };

    let q_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.q_proj"))?;
    let k_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.k_proj"))?;
    let v_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.v_proj"))?;
    let o_proj = mm.load_quant_linear(&format!("{prefix}.self_attn.o_proj"))?;
    let gate = mm.load_quant_linear(&format!("{prefix}.mlp.gate_proj"))?;
    let up_proj = mm.load_quant_linear(&format!("{prefix}.mlp.up_proj"))?;
    let down_proj = mm.load_quant_linear(&format!("{prefix}.mlp.down_proj"))?;
    let ple_gate = mm.load_quant_linear(&format!("{prefix}.per_layer_input_gate"))?;
    let ple_proj = mm.load_quant_linear(&format!("{prefix}.per_layer_projection"))?;

    println!("  layer_scalar           {:.4}", layer_scalar);
    println!("  loading + dequanting weights...");

    // Upload + dequant all quantized weights. Helper closure to keep it terse.
    let load_quant = |q: &xenon_core::QuantLinearRef<'_>, stream: &Stream|
      -> anyhow::Result<DeviceBuffer<bf16>> {
        let mut dp: DeviceBuffer<u8> = DeviceBuffer::new(q.packed.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut ds: DeviceBuffer<u8> = DeviceBuffer::new(q.scales.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
        dp.copy_from_host_bytes(q.packed).map_err(|e| anyhow::anyhow!("{e}"))?;
        ds.copy_from_host_bytes(q.scales).map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut dw: DeviceBuffer<bf16> = DeviceBuffer::new(q.out_features * q.in_features).map_err(|e| anyhow::anyhow!("{e}"))?;
        fp4_dequant_bf16(&mut dw, &dp, &ds, q.global_scale,
                          q.out_features, q.in_features, Some(stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(dw)
    };

    let load_bf16 = |bytes: &[u8], len: usize| -> anyhow::Result<DeviceBuffer<bf16>> {
        assert_eq!(bytes.len(), len * 2);
        let mut d: DeviceBuffer<bf16> = DeviceBuffer::new(len).map_err(|e| anyhow::anyhow!("{e}"))?;
        d.copy_from_host_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(d)
    };

    let d_in_norm = load_bf16(in_norm_bytes, hidden)?;
    let d_post_attn_norm = load_bf16(post_attn_norm_bytes, hidden)?;
    let d_pre_ffn_norm = load_bf16(pre_ffn_norm_bytes, hidden)?;
    let d_post_ffn_norm = load_bf16(post_ffn_norm_bytes, hidden)?;
    let d_q_norm_w = load_bf16(q_norm_bytes, d)?;
    let d_k_norm_w = load_bf16(k_norm_bytes, d)?;
    let d_post_ple_norm = load_bf16(post_ple_norm_bytes, hidden)?;

    let d_qw = load_quant(&q_proj, &stream)?;
    let d_kw = load_quant(&k_proj, &stream)?;
    let d_vw = load_quant(&v_proj, &stream)?;
    let d_ow = load_quant(&o_proj, &stream)?;
    let d_gate_w = load_quant(&gate, &stream)?;
    let d_up_w = load_quant(&up_proj, &stream)?;
    let d_down_w = load_quant(&down_proj, &stream)?;
    let d_plg_w = load_quant(&ple_gate, &stream)?;
    let d_plp_w = load_quant(&ple_proj, &stream)?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    // --- Decoder layer forward. d_h is the rolling hidden state. ---
    // Layer N's input = layer (N-1)'s output. For layer 0, input = inputs_embeds.
    // For layers > 0, use HF's captured output from the previous layer so the
    // diff isolates THIS layer's math (otherwise errors from earlier layers
    // would dominate the comparison).
    let mut d_h: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    let hf_mm_input = MmapWeights::open(&hf_path)?;
    if layer == 0 {
        d_h.copy_from_device(&d_inputs_embeds).map_err(|e| anyhow::anyhow!("{e}"))?;
    } else {
        let prev_name = format!("layer_{:02}.final", layer - 1);
        let prev_bytes = hf_mm_input.load_bf16(&prev_name)?;
        anyhow::ensure!(prev_bytes.len() == tokens * hidden * 2, "prev shape");
        d_h.copy_from_host_bytes(prev_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    // residual buffer (owns a copy).
    let mut d_residual: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_residual.copy_from_device(&d_h).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Step 1-5: attention block.
    // h = RMSNorm(input, input_layernorm)
    let mut d_normed: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, &d_h, Some(&d_in_norm), tokens, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // QKV projections.
    let mut d_q: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_heads * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_k: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_kv * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_v: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_kv * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_q, &d_normed, &d_qw, None, tokens, h_heads * d, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_k, &d_normed, &d_kw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_v, &d_normed, &d_vw, None, tokens, h_kv * d, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // q_norm / k_norm / v_norm (v has with_scale=False -> weight=None).
    {
        let q_tmp = clone_buffer(&d_q)?;
        rmsnorm_bf16(&mut d_q, &q_tmp, Some(&d_q_norm_w), tokens * h_heads, d, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let k_tmp = clone_buffer(&d_k)?;
        rmsnorm_bf16(&mut d_k, &k_tmp, Some(&d_k_norm_w), tokens * h_kv, d, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let v_tmp = clone_buffer(&d_v)?;
        rmsnorm_bf16(&mut d_v, &v_tmp, None, tokens * h_kv, d, eps, Some(&stream))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    // RoPE on Q and K only.
    rope_bf16(&mut d_q, &d_positions, tokens, h_heads, d, rotary_dim, rope_theta, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    rope_bf16(&mut d_k, &d_positions, tokens, h_kv, d, rotary_dim, rope_theta, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Naive attention with scale=1.0 (Gemma 4 relies on learned q/k norms).
    let mut d_attn_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * h_heads * d).map_err(|e| anyhow::anyhow!("{e}"))?;
    attn_naive_bf16(&mut d_attn_out, &d_q, &d_k, &d_v,
                     tokens, tokens, h_heads, h_kv, d, 1.0, 0, window, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // o_proj back to hidden.
    let mut d_attn_hidden: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_attn_hidden, &d_attn_out, &d_ow, None,
                    tokens, hidden, h_heads * d, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // post_attention_layernorm, then residual add.
    let mut d_tmp: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_tmp, &d_attn_hidden, Some(&d_post_attn_norm),
                  tokens, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut d_h, &d_residual, &d_tmp, 1.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Step 6-10: MLP block.
    d_residual.copy_from_device(&d_h).map_err(|e| anyhow::anyhow!("{e}"))?;
    rmsnorm_bf16(&mut d_normed, &d_h, Some(&d_pre_ffn_norm), tokens, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut d_gate_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * inter).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_up_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * inter).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_act: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * inter).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_mlp_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_gate_out, &d_normed, &d_gate_w, None,
                    tokens, inter, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_up_out, &d_normed, &d_up_w, None,
                    tokens, inter, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    gelu_tanh_glu_bf16(&mut d_act, &d_gate_out, &d_up_out, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    lt.linear_bf16(&mut d_mlp_out, &d_act, &d_down_w, None,
                    tokens, hidden, inter, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    rmsnorm_bf16(&mut d_tmp, &d_mlp_out, Some(&d_post_ffn_norm),
                  tokens, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    add_scale_bf16(&mut d_h, &d_residual, &d_tmp, 1.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Step 11-16: PLE block.
    d_residual.copy_from_device(&d_h).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_gate_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_glu: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * per_layer).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_ple_proj_out: DeviceBuffer<bf16> = DeviceBuffer::new(tokens * hidden).map_err(|e| anyhow::anyhow!("{e}"))?;

    // h_256 = per_layer_input_gate(h)  [T, Hl]
    lt.linear_bf16(&mut d_ple_gate_out, &d_h, &d_plg_w, None,
                    tokens, per_layer, hidden, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // h_256 = gelu(h_256) * per_layer_input[layer, :]
    gelu_tanh_glu_bf16(&mut d_ple_glu, &d_ple_gate_out, &d_ple_layer, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // h = per_layer_projection(h_256)  [T, H]
    lt.linear_bf16(&mut d_ple_proj_out, &d_ple_glu, &d_plp_w, None,
                    tokens, hidden, per_layer, 1.0, 0.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // h = post_per_layer_input_norm(h)
    rmsnorm_bf16(&mut d_tmp, &d_ple_proj_out, Some(&d_post_ple_norm),
                  tokens, hidden, eps, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // h = residual + h
    add_scale_bf16(&mut d_h, &d_residual, &d_tmp, 1.0, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Step 17: layer_scalar broadcast multiply.
    let d_h_in = clone_buffer(&d_h)?;
    scale_bf16(&mut d_h, &d_h_in, layer_scalar, Some(&stream))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;

    // --- Compare to HF reference ---
    let ours = d_h.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let hf_mm = MmapWeights::open(&hf_path)?;
    let hf_name = format!("layer_{:02}.final", layer);
    let hf_bytes = hf_mm.load_bf16(&hf_name)?;
    anyhow::ensure!(hf_bytes.len() == tokens * hidden * 2,
        "HF tensor shape mismatch");
    let theirs = bytes_to_bf16_vec(hf_bytes);

    let (max_abs, _per, global_rel) = compare_bf16(&ours, &theirs);
    let our_max: f32 = ours.iter().map(|v| v.to_f32().abs()).fold(0.0, f32::max);
    let hf_max:  f32 = theirs.iter().map(|v| v.to_f32().abs()).fold(0.0, f32::max);

    println!();
    println!("our max |.|              {:>8.3}", our_max);
    println!("hf  max |.|              {:>8.3}", hf_max);
    println!("max abs diff             {:.3e}", max_abs);
    println!("global rel diff          {:.3e}", global_rel);
    let tol: f32 = 5e-2;
    anyhow::ensure!(global_rel <= tol, "global rel {global_rel} exceeds {tol}");
    println!("OK: layer {layer} forward matches HF within {tol:.1e}.");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_test_attn_flash(
    t_q: usize,
    t_kv: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    window: i32,
    q_pos_base: Option<i32>,
    seed: u64,
) -> anyhow::Result<()> {
    anyhow::ensure!(h % h_kv == 0, "h must be divisible by h_kv");
    anyhow::ensure!(d % 128 == 0, "head_dim must be a multiple of 128");
    anyhow::ensure!(t_kv >= t_q, "T_kv must be >= T_q for a cached-decode-like shape");
    Device(0).set().map_err(|e| anyhow::anyhow!("{e}"))?;
    let stream = Stream::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    // Default q_pos_base so the last Q token lines up with the last KV position
    // (decode shape: q_pos_base = T_kv - T_q).
    let q_pos_base = q_pos_base.unwrap_or((t_kv as i32) - (t_q as i32));
    let scale = 1.0f32 / (d as f32).sqrt();

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let q_host: Vec<bf16> = (0..t_q * h * d).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    let k_host: Vec<bf16> = (0..t_kv * h_kv * d).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();
    let v_host: Vec<bf16> = (0..t_kv * h_kv * d).map(|_| bf16::from_f32(rng.gen_range(-1.0..1.0))).collect();

    let mut d_q: DeviceBuffer<bf16> = DeviceBuffer::new(q_host.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_k: DeviceBuffer<bf16> = DeviceBuffer::new(k_host.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_v: DeviceBuffer<bf16> = DeviceBuffer::new(v_host.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_naive: DeviceBuffer<bf16> = DeviceBuffer::new(q_host.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut d_flash: DeviceBuffer<bf16> = DeviceBuffer::new(q_host.len()).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_q.copy_from_host(&q_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_k.copy_from_host(&k_host).map_err(|e| anyhow::anyhow!("{e}"))?;
    d_v.copy_from_host(&v_host).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Naive kernel: only run if T_kv small enough to fit scores in shmem.
    // BLOCK_THREADS=128, ~48 KiB / (T_kv*4 + 128*4) bytes.
    let naive_shmem_bytes = (t_kv * 4) + (128 * 4);
    let naive_fits = naive_shmem_bytes < 47 * 1024;
    let want: Vec<bf16> = if naive_fits {
        attn_naive_bf16(
            &mut d_naive, &d_q, &d_k, &d_v,
            t_q, t_kv, h, h_kv, d, scale, q_pos_base, window, Some(&stream),
        ).map_err(|e| anyhow::anyhow!("{e}"))?;
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
        d_naive.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        attn_naive_bf16_reference(&q_host, &k_host, &v_host,
                                   t_q, t_kv, h, h_kv, d, scale, q_pos_base, window)
    };

    // Flash kernel.
    let flash_start = Instant::now();
    let iters = 10;
    attn_flash_bf16(
        &mut d_flash, &d_q, &d_k, &d_v,
        t_q, t_kv, h, h_kv, d, scale, q_pos_base, window, Some(&stream),
    ).map_err(|e| anyhow::anyhow!("{e}"))?;
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let t0 = Instant::now();
    for _ in 0..iters {
        attn_flash_bf16(
            &mut d_flash, &d_q, &d_k, &d_v,
            t_q, t_kv, h, h_kv, d, scale, q_pos_base, window, Some(&stream),
        ).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow::anyhow!("{e}"))?;
    let per_flash_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let _ = flash_start;

    let got = d_flash.copy_to_host_vec().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (max_abs, _per, global_rel) = compare_bf16(&got, &want);

    println!("=== xenon-cli test-attn-flash ===");
    println!("shape                    T_q={t_q}, T_kv={t_kv}, H={h}, Hkv={h_kv}, D={d}");
    println!("scale / q_pos_base       {:.6} / {}", scale, q_pos_base);
    println!("window                   {window}");
    println!("reference                {}", if naive_fits { "gpu naive kernel" } else { "host reference" });
    println!("flash per-launch         {:>7.2} us", per_flash_us);
    println!("max abs diff             {:.3e}", max_abs);
    println!("global rel diff          {:.3e}", global_rel);
    let tol_rel: f32 = 5e-2;
    anyhow::ensure!(global_rel <= tol_rel, "global rel {global_rel} exceeds {tol_rel}");
    println!("OK: flash kernel matches reference within {tol_rel:.1e}.");
    Ok(())
}

fn cmd_test_tokenizer(dir: PathBuf, text: String, add_specials: bool) -> anyhow::Result<()> {
    let cfg = GemmaConfig::from_path(&dir.join("config.json"))?;
    let tok = Tokenizer::from_dir(&dir)?;
    let vocab = tok.vocab_size();

    println!("=== xenon-cli test-tokenizer ===");
    println!("text                     {:?}", text);
    println!("add_special_tokens       {}", add_specials);
    println!("tokenizer vocab_size     {}", vocab);
    println!("config.vocab_size        {}", cfg.text_config.vocab_size);
    // Config vocab includes special tokens beyond the tokenizer's learned vocab
    // for Gemma 4 (boi/eoi/audio tokens etc.). Tokenizer's with_added=true
    // should reach or approach config.vocab_size.
    anyhow::ensure!(
        vocab <= cfg.text_config.vocab_size,
        "tokenizer vocab {vocab} larger than config vocab {}",
        cfg.text_config.vocab_size
    );

    let ids = tok.encode(&text, add_specials)?;
    println!();
    println!("encoded ids ({} tokens)", ids.len());
    for (i, id) in ids.iter().enumerate() {
        if i < 32 {
            print!("  {id}");
            if (i + 1) % 8 == 0 { println!(); }
        }
    }
    if ids.len() > 32 { println!("  ..."); }
    else if ids.len() % 8 != 0 { println!(); }

    for id in &ids {
        anyhow::ensure!((*id as usize) < cfg.text_config.vocab_size,
            "id {id} >= config.vocab_size {}", cfg.text_config.vocab_size);
    }

    let decoded = tok.decode(&ids, /*skip_special*/ true)?;
    println!();
    println!("decoded (skip_specials)  {:?}", decoded);
    println!("round-trip equal         {}", decoded == text);
    // Not all tokenizers round-trip perfectly (normalization, BOS handling),
    // so decoding with specials kept often matches better; make the assertion
    // on that.
    let decoded_raw = tok.decode(&ids, /*skip_special*/ false)?;
    println!("decoded (keep specials)  {:?}", decoded_raw);
    anyhow::ensure!(decoded_raw.contains(&text) || decoded == text,
        "round-trip lost input text");
    println!("OK: tokenizer loads, encode/decode round-trips the input.");
    Ok(())
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
        rmsnorm_bf16(&mut d_normed, &d_x, Some(&d_norm_w), batch, hidden, eps, Some(&stream))
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

    let normed_host = rmsnorm_bf16_reference(&x_host, Some(&norm_w_host), batch, hidden, eps);
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
