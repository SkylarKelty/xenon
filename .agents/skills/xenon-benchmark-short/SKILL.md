---
name: xenon-benchmark-short
description: Benchmark the Xenon LLM inference engine with short prompts (<256 tokens). Use when asked to benchmark, profile, or measure prefill/decode performance on prompts of 256 tokens or fewer. Covers quick latency tests, regression checks, and prefill-vs-decode characterization for the decode fast path.
---

# Xenon Benchmark — Short Prompts (<256 tokens)

Benchmark the engine with prompts ≤ 256 tokens. This exercises the decode fast path (FP4 GEMV, split-KV attention) and measures prefill latency at small-to-medium batch sizes.

## Prerequisites

- CUDA device available and `xenon-cli` built (`cargo build --release --bin xenon-cli`)
- Model path configured (default: `~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b`)
- For chat prompts, use `--chat` flag to wrap in Gemma template

## Workflow

1. **Pick a prompt length** from the reference catalog (see `references/prompts.md`)
2. **Run benchmark**: `cargo run --release --bin xenon-cli -- bench "$MODEL" --prompt "..." --max-new 32 --warmup 1 --iters 3 --label short-<N>tok`
3. **Check output**: prefill ms, decode ms/step, tok/s
4. **Compare results**: Look at `results/bench-short-<N>tok-*.json`

## Key Parameters

| Flag | Default | Short Prompt Guidance |
|------|---------|----------------------|
| `--max-new` | 32 | Keep at 32 for consistent decode characterization |
| `--warmup` | 1 | Increase to 2–3 for stable GPU clocks |
| `--iters` | 3 | 3 is enough for short prompts; 5 for publication |
| `--max-len` | 4096 | Can lower to 512 for very short tests to reduce KV alloc |
| `--chat` | off | Enable for instruction-tuned behavior |

## Quick Commands

```bash
MODEL=~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b

# 32 tokens — minimal prefill
P="Explain quantum computing in one sentence."
cargo run --release --bin xenon-cli -- bench "$MODEL" --prompt "$P" --max-new 32 --label short-32tok

# 128 tokens — moderate prefill
P="Write a Python function that reads a CSV file and returns a list of dictionaries. Include docstring, type hints, and error handling for missing files."
cargo run --release --bin xenon-cli -- bench "$MODEL" --prompt "$P" --max-new 32 --label short-128tok

# 256 tokens — boundary case (just under long-prompt threshold)
P=$(cat <<'EOF'
Design a REST API for a task management app. Support creating tasks with title, description, priority, due date, and assignee. Organize tasks into projects. Each project has a name, description, and owner. Users can comment on tasks and upload attachments. Implement filtering by status, priority, and assignee. Include pagination with cursor-based navigation. Document all endpoints with request/response examples and error codes.
EOF
)
cargo run --release --bin xenon-cli -- bench "$MODEL" --prompt "$P" --max-new 32 --label short-256tok
```

## Expected Performance (Gemma-4-E4B, RTX PRO 2000)

| Prompt Len | Prefill | Decode/step |
|-----------|---------|-------------|
| 32 tokens | ~5–10 ms | ~8–12 ms |
| 128 tokens | ~15–25 ms | ~8–12 ms |
| 256 tokens | ~30–50 ms | ~8–12 ms |

Decode latency should be roughly flat across prompt lengths (amortized KV cache warmup). If decode latency grows with prompt_len, investigate attention kernel selection.

## Diagnosing Issues

- **High prefill**: Check if `attn_flash_tc` is dispatching (T_q ≥ 16). If prefill uses `attn_naive`, shared memory may be limiting.
- **High decode**: Verify `fp4_gemv_bf16` is used for M=1 decode GEMV. Check `attn_split_kv` chunk size.
- **Out of VRAM**: Reduce `--max-len` or run `cargo run --release --bin xenon-cli -- vram` first.

## Prompt Catalog

See `references/prompts.md` for ready-made prompts at 32, 64, 128, 200, and 256 tokens.
