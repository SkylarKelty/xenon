# xenon

Single-stream LLM inference engine built from scratch in **Rust + CUDA**. Targets one
model (`cosmicproc/gemma-4-E4B-it-NVFP4`) on one GPU (RTX PRO 2000 Blackwell, `sm_120a`,
7.53 GiB VRAM). Optimised for **cold start, idle power, and operational simplicity** —
not throughput.

One static binary, no Python runtime, OpenAI-compatible HTTP API.

## Why

Existing inference stacks (vLLM, TensorRT-LLM, llama.cpp, ollama) are generalists. For
a single known model on a single known box, you can:

- Keep the whole weight footprint on device by host-offloading PLE + lm_head.
- Pick one quant format (NVFP4) and specialise kernels for it end-to-end.
- Skip the runtime flexibility budget — everything is AOT, there's no NVRTC, no Python,
  no model-loader fallback chain.
- Cold-start in under 2 seconds.

Xenon is the result.

## Current performance

Measured on `cosmicproc/gemma-4-E4B-it-NVFP4`, RTX PRO 2000 Blackwell Laptop, T_prompt=155,
max_new=50, warm GPU:

| Metric | xenon | ollama 0.21 (same model) | ratio |
| --- | --- | --- | --- |
| Decode | ~60 tok/s | 23.79 tok/s | **2.5×** |
| Prefill | 3242 tok/s | 2693 tok/s | **1.20×** |

ollama runs `gemma4:e4b` Q4_K_M with ~68% of the model on CPU because it doesn't fit the
7.5 GiB VRAM. Xenon keeps everything resident via per-layer-embedding and lm_head
host-offloading.

## What's in the box

| Feature | Status |
| --- | --- |
| 42-layer Gemma 4 E4B forward pass (sliding + full attention, GQA, KV sharing) | ✅ |
| NVFP4 quantised weights, native `cuBLASLt` FP4 GEMM for prefill (M ≥ 128) | ✅ |
| Hand-rolled kernels: fused FP4 gemv, flash attention (tensor-core, `cp.async`), split-KV | ✅ |
| KV cache with GQA layout + shared-slot indirection | ✅ |
| OpenAI-compatible HTTP server: `/v1/chat/completions` (SSE), `/v1/completions`, `/v1/models` | ✅ |
| Concurrent request batching at the engine and server level | ✅ |
| HuggingFace-reference verification harness | ✅ |
| Multi-model, multi-GPU, other architectures | ❌ v0 |
| Vision / audio towers | ❌ v0 |
| LoRA, speculative decoding, continuous batching | ❌ v0 |

See `PLAN.md` for the full phase plan and status.

## Quickstart

### Build

Requires CUDA 13.x, `nvcc`, and Rust 1.75+. `sm_120a` is the default; override with
`XENON_ARCH=sm_90a` etc.

```bash
cargo build --release --workspace
```

### Download the model

```bash
huggingface-cli download cosmicproc/gemma-4-E4B-it-NVFP4
MODEL=~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/<snapshot-hash>
```

### Generate

```bash
./target/release/xenon-cli generate "$MODEL" --chat --max-new 64 \
  --prompt "Explain fp4 tensor cores in 3 sentences."
```

### Serve an OpenAI-compatible API

```bash
./target/release/xenon-server --model "$MODEL" --bind 127.0.0.1:8080
```

In another shell:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-E4B-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

Also supported: `POST /v1/completions`, `GET /v1/models`, `GET /health`.

### Benchmark

```bash
./target/release/xenon-cli bench "$MODEL" --chat \
  --prompt "Write a haiku about GPUs." \
  --max-new 50 --warmup 3 --iters 5 --label my-bench
# Results: results/bench-my-bench-<ts>.json
```

## Architecture

Cargo workspace. Strict dependency layering — anything that touches CUDA lives below
the `xenon-engine` boundary.

```
crates/
  xenon-core/       pure Rust — safetensors mmap, config, tokenizer
  xenon-kernels/    CUDA FFI + hand-rolled .cu kernels + cuBLASLt wrappers
  xenon-engine/     forward_step / layer_forward / KV cache / batching
  xenon-server/     axum HTTP server (OpenAI endpoints)
  xenon-cli/        dev/debug/bench harness
tools/
  hf-ref/           Python capture of HF reference activations
  gpubench/         standalone CUDA roofline bench (predates xenon)
```

Kernels compile via `nvcc` in `xenon-kernels/build.rs` into a single static library.
Every `.cu` under `src/cu/` is picked up automatically.

Correctness is validated against a captured HuggingFace reference — see
`.agents/skills/xenon-verify-engine/` for the six-level diff workflow.

## Hardware target

v0 is explicitly scoped to one GPU: RTX PRO 2000 Blackwell Laptop (`sm_120a`,
26 SMs, 7.53 GiB VRAM). Dispatch constants (SM count, shmem budgets, attention
thresholds) are tuned for this part. Running on other Blackwell/Hopper parts
may work — `XENON_ARCH` lets you flip the gencode — but performance tuning is
not portable.

## Licence

MIT OR Apache-2.0.

## Status

Pre-v1. Phases 0-5 complete (foundations, primitives, full forward, decode,
server). Phase 6 perf polish ongoing. No release channel yet.
