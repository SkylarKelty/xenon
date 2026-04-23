# CLAUDE.md

Repo-level notes for Claude sessions. Don't duplicate what's in `PLAN.md` —
this file is the stuff that isn't obvious from reading the code.

## What this is

Xenon: from-scratch Rust + CUDA single-stream inference engine for one model
(`cosmicproc/gemma-4-E4B-it-NVFP4`) on one GPU (RTX PRO 2000 Blackwell,
`sm_120a`, 7.53 GiB VRAM). Optimised for **cold start, idle power, and
operational simplicity** — not throughput. See `PLAN.md` for phases, goals,
and non-goals. Current perf: ~60 tok/s decode (2.5× ollama) and 3242 tok/s
prefill at T=155 (1.20× ollama).

## Workspace layout

```
crates/xenon-core/     pure Rust: safetensors mmap, config, tokenizer
crates/xenon-kernels/  CUDA FFI + .cu kernels + cuBLASLt glue
crates/xenon-engine/   forward_step, layer_forward, KV cache, batching
crates/xenon-server/   axum HTTP server (OpenAI endpoints)
crates/xenon-cli/      dev/debug + bench harness; mirrors engine internals
tools/hf-ref/          Python capture of HF reference activations
tools/gpubench/        standalone CUDA roofline bench (pre-xenon reference)
```

Layering is intentional:
- `xenon-core` never touches CUDA. It stays compilable on hosts without a GPU.
- All CUDA linking lives in `xenon-kernels/build.rs`; rpath is baked in so
  binaries run without `LD_LIBRARY_PATH`.
- `xenon-cli` has its own `QuantLinearDev` and `layer_forward` that mirror
  the engine's — they are intentionally duplicated so kernel tests can reach
  low-level primitives without going through the engine.
- `xenon-server` consumes `xenon-engine` and never talks to kernels directly.

## Build & run

```bash
# Default nvcc arch is sm_120a. Override with XENON_ARCH=sm_90a etc.
cargo build --release -p xenon-cli
cargo build --release -p xenon-server

MODEL=~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b

# One-shot generation
./target/release/xenon-cli generate "$MODEL" --chat --prompt "hi" --max-new 32

# Bench (see skill `xenon-benchmark-short` / `xenon-benchmark-long`)
./target/release/xenon-cli bench "$MODEL" --chat --prompt "..." --max-new 50 \
  --warmup 3 --iters 5 --label <label>

# Server (OpenAI-compatible)
./target/release/xenon-server --model "$MODEL" --bind 127.0.0.1:8080
```

## Workflows — use the skills first

Three slash-skill workflows live in `.agents/skills/`:
- `xenon-benchmark-short` — prompts ≤ 256 tokens, decode-focused
- `xenon-benchmark-long` — prompts > 256 tokens, long-context focused
- `xenon-verify-engine` — correctness vs HF reference (6 levels)

**Prefer the skills over improvising bench/verify commands** — they document
the expected shapes, tolerances, and diagnostic steps for this model.

## Perf measurement gotchas

GPU clock ramp matters a lot on this laptop part. Between-run variance from
a cold (P5) vs warm (P0) start is ~5-8%. When comparing commits:
- Use `--warmup 3 --iters 5` minimum.
- Run 3 back-to-back benches for each variant.
- Do the back-to-back comparison in the **same shell session** — switching
  branches with a `cargo build` in between keeps clocks warm.
- `nvidia-smi --query-gpu=pstate,clocks.gr --format=csv` will tell you if
  the GPU is in P0 (max) or lower; report this if variance feels off.

Also always re-bench ollama on the same GPU state for any headline claim:
`ollama run gemma4:e4b --verbose "<prompt>"` after a warmup run.

## Editing kernels

- New `.cu` files drop into `crates/xenon-kernels/src/cu/`. `build.rs` picks
  them up automatically — no CMake list to maintain.
- Each kernel needs: a C-ABI `extern "C"` entry point in the `.cu`, an
  `extern "C" fn` FFI decl in `src/kernels.rs`, a safe Rust wrapper in
  `src/kernels.rs`, and a re-export in `src/lib.rs`.
- Add a CLI test (`test-<kernel>`) in `crates/xenon-cli/src/main.rs` whenever
  you add a kernel. The bar is "matches a host reference within a documented
  tolerance" before anything gets wired into `layer_forward`.
- Shared memory >48 KiB needs `cudaFuncSetAttribute` opt-in (see
  `attn_flash_tc.cu` for the pattern).

## Attention dispatch (recent, easy to confuse)

`layer_forward` in both `xenon-engine` and `xenon-cli` picks an attention
kernel by `t_q * h_heads`:

```
if t_q * h_heads < 2 * SM_count (=52): attn_split_kv_bf16   (decode)
elif t_q >= 16:                         attn_flash_tc_bf16   (prefill, any D)
else:                                    attn_naive_bf16      (edge cases)
```

- Split-KV is specific to decode shapes — adding a 3rd grid dim on T_kv
  chunks gets 40-64 blocks on 26 SMs instead of 8. Wiring it for prefill
  regresses ~14% because of the fp32 numerator DRAM round-trip.
- Flash-TC is `mma.m16n8k16` + `cp.async` K/V double-buffer. Works at both
  D=256 and D=512 since the cp.async overlap landed (e77db3f). Without
  cp.async D=512 regresses due to >48 KiB smem dropping occupancy.
- Naive is only hit by oddly-sized T_q (< MIN_TQ but > SATURATION/H);
  not reached in the Gemma 4 E4B prompt shapes we care about today.

See `memory/project_xenon_attention_kernel.md` for the full history and
measured numbers.

## Dead ends — don't retry

- **cudarc**: we want small, predictable FFI. Our surface is ~10 runtime +
  ~10 cuBLASLt calls, hand-rolled.
- **ONNX export**: throws away the weight layout work and the NVFP4 path.
- **CUDA graphs**: benched as a no-op at our layer count and memory
  pattern. See `PLAN.md` phase 5 note.
- **M<128 on cuBLASLt NVFP4**: silently returns garbage on sm_120a. The
  `>= 128` threshold in `QuantLinearDev::forward` is load-bearing.
- **bf16 sO in flash-tc for D=512**: considered as a way to fit occupancy;
  rejected — online softmax accumulates enough drift across tiles that
  precision fails. Use cp.async overlap instead (already landed).

## Memory conventions

- Row-major throughout. cuBLASLt wrapper has an operand-swap trick to bridge
  with its col-major native layout. Don't flip row/col without reading
  `matmul_bf16_reference`.
- bf16 for activations; fp32 for accumulators in CUDA kernels.
- FP4 packed weights are `[N, K/2]` U8 with UE4M3 per-16 block scales stored
  two ways: `scales` (row-major `[N, K/16]`) for `fp4_dequant_bf16`, and
  `scales_swizzled` (128×4-interleaved) for cuBLASLt `VEC16_UE4M3` mode.
  See `project_xenon_nvfp4_swizzle.md` memory for the *why*.

## Correctness first

- A new kernel lands only after its CLI test passes (global rel ≤ 5e-3 or
  ≤ 5e-2 for GELU-bearing paths) AND HF-ref verification (`test-vs-hf-*`)
  still passes if the change can affect model outputs.
- `xenon-cli generate --chat --prompt "hi" --max-new 32` is the cheap smoke
  test after any change to `layer_forward` or dispatch. It should produce
  coherent English text.

## Auto-memory

User-level memory lives at `~/.claude/projects/-home-k1811651-git-ai-gpubench/memory/`.
The key files are:
- `MEMORY.md` — index (always loaded into context)
- `project_xenon_decode_bottleneck.md` — session-to-session decode perf log
- `project_xenon_attention_kernel.md` — full attention kernel history
- `project_xenon_nvfp4_swizzle.md` — NVFP4 GEMM + shared-activation follow-up
- `project_xenon_ollama_baseline.md` — ollama tok/s on same hw/model

Update these when a non-obvious finding or regression root-cause comes up.
Don't duplicate content that already lives in `PLAN.md` or commit messages.
