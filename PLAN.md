# xenon — plan

Inference engine for a single model on a single box, built from scratch in Rust
+ CUDA. Optimised for **cold start, idle power, and operational simplicity**,
not for matching vLLM's server throughput.

## Goals

- Sub-2 s cold start to first token for a ~5 B-active model at NVFP4.
- Under 20 W idle when the API is quiet (Hot → Warm → Cold → Frozen state machine).
- One static binary, no Python runtime, OpenAI-compatible HTTP API.
- Correctness vs HuggingFace reference to within a few bf16 ULPs per layer.

## Non-goals

- Server-tier multi-tenant throughput. Continuous batching, speculative decoding,
  LoRA hot-swap are explicitly not v1.
- Multi-model, multi-hardware. This engine is specialised to one model +
  `sm_120a`. Portability is a later-phase concern.
- Beating vLLM/TensorRT-LLM on raw tok/s. Realistic envelope is 0.9-1.2× vLLM
  for single-stream; the win is in ops/idle/footprint.

## Target

| Thing | Value |
| --- | --- |
| Model | `cosmicproc/gemma-4-E4B-it-NVFP4` |
| Architecture | `Gemma4ForConditionalGeneration` (text-only path in v0) |
| Layers | 42 total: 35 sliding-attention (window 512) + 7 full-attention |
| Hidden / MLP | 2560 / 10240 |
| Heads | 8 Q, 2 KV (GQA 4:1); head_dim 256 (sliding) / 512 (full) |
| Vocab | 262,144 |
| Max context | 131,072 |
| Quant | NVFP4 g=16 on attn + MLP; `lm_head` / vision / audio / embeddings in bf16 |
| Weights on disk | 9.50 GiB (text only after offload: ~2.8 GiB on device) |
| GPU | NVIDIA RTX PRO 2000 Blackwell Laptop, `sm_120`, 7.53 GiB total |
| CUDA | 13.2 toolkit, driver 595.58 |

## Roofline (measured via `tools/gpubench`)

| Metric | This box | Used by |
| --- | --- | --- |
| FP4 GEMM peak | 260 TFLOP/s | eventual native-FP4 path |
| BF16 GEMM via cuBLASLt | 60 TFLOP/s @ 2048³ | MLP in phases 1-3 |
| D2D bandwidth | 167 GB/s | decode ceiling (bandwidth-bound) |
| H2D pinned | 22.9 GB/s | cold-start weight load |
| Kernel launch | 1.57 µs | decode per-step overhead before CUDA Graphs |

Decode ceiling (weights path only, `lm_head` not offloaded): ~3.04 GiB/tok →
~51 tok/s single-stream. With PLE + lm_head host-offloaded, roofline climbs to
~90 tok/s. Target v1: **30-40 tok/s short-ctx decode**, i.e. 40-70% of the
bandwidth ceiling.

## Architecture

Cargo workspace at repo root.

```
crates/
  xenon-core/          pure Rust: config, safetensors header, mmap
  xenon-kernels/       CUDA FFI + hand-rolled kernels + cuBLASLt glue
    build.rs           compiles every src/cu/*.cu via nvcc -> static lib
    src/cuda.rs        Device / Stream / DeviceBuffer<T>
    src/cublas.rs      cuBLASLt FFI + bf16 matmul wrapper
    src/kernels.rs     Rust wrappers for each .cu entry point
    src/cu/*.cu        the actual kernels
  xenon-cli/           binary: info / load / test-* / upload / sanity / vram
  xenon-server/        binary: axum HTTP server (OpenAI endpoints)
tools/
  gpubench/            standalone CUDA benchmark (unchanged, dev reference)
```

Dependency layering is intentional:
- `xenon-core` never touches CUDA — stays compilable on hosts without a GPU.
- All CUDA linking lives in `xenon-kernels/build.rs`; rpath is baked in so
  binaries run without `LD_LIBRARY_PATH`.
- `xenon-cli` is the test harness for every kernel; regressions surface
  there before they reach the server.

## Phase plan

Legend: `[x]` done, `[>]` in progress, `[ ]` todo.

### Phase 0 — foundations [x]
- [x] Cargo workspace + gpubench moved to `tools/`
- [x] HF `config.json` and `hf_quant_config.json` parsing
- [x] Safetensors header walker + NVFP4 pair classification
- [x] modelopt exclude-module invariant check
- [x] `nvcc` build pipeline → static lib → Rust link
- [x] CLI: `info`, `load`, `sanity`

### Phase 1 — MLP primitives

#### Week 1 [x]
- [x] Minimal CUDA runtime FFI (`Device`, `Stream`, `DeviceBuffer<T>`)
- [x] mmap safetensors (`MmapWeights::tensor_bytes`)
- [x] RMSNorm kernel (bf16 I/O, fp32 accumulate, shared-mem reduce)
- [x] CLI: `vram`, `test-rmsnorm`, `upload --verify`

#### Week 2 [x]
- [x] NVFP4 dequant kernel (UE4M3 block scale + fp32 global scale → bf16)
- [x] Hand-rolled cuBLASLt FFI + `matmul_bf16_rm` (row-major operand order)
- [x] GELU-tanh kernel (+ fused GLU variant)
- [x] CLI: `test-dequant`, `test-gemm`, `test-gelu`

#### Week 3 [ ] — real-weight MLP end-to-end
- [ ] `list <model> --pattern <glob>` CLI utility to surface tensor names
- [ ] Tensor-family loader: `(weight, weight_scale, weight_scale_2)` tuple for
      a given module prefix, with shape validation against config
- [ ] `test-mlp <model> --layer 0`: dequant gate/up/down, run full MLP chain,
      compare GPU vs host reference (same FP4 bytes, fp32 math, bf16 round
      at the end). Target: global rel diff ≤ 5e-2.
- [ ] Decide layer prefix convention (`model.language_model.layers.N.*` vs
      `model.layers.N.*`) from tensor listing

### Phase 2 — attention [ ]
- [ ] Tokenizer (load `tokenizer.json`, SentencePiece-compatible via
      `tokenizers` crate)
- [ ] Token embedding gather (bf16)
- [ ] Per-layer embedding gather from pinned host memory
- [ ] RoPE kernel (dual head_dim: 256 for sliding, 512 for global layers)
- [ ] Naive attention (Q@Kᵀ → softmax → @V) for correctness baseline
- [ ] KV cache allocator with GQA layout (2 KV heads × head_dim per layer)
- [ ] Sliding-window mask (window 512) + full attention variants
- [ ] KV sharing: 18 of 42 layers reuse KV from earlier layers — map indices
- [ ] FlashAttention-style tile kernel (fp32 accumulate, bf16 I/O), one
      version per head_dim variant

### Phase 3 — full forward pass [ ]
- [ ] 42-layer chain: embed → (norm → attn → norm → mlp + PLE inject) × 42 →
      norm → lm_head → logits
- [ ] Host-offloaded `lm_head` with cudaMemcpyAsync overlap (reclaims 1.28 GiB VRAM)
- [ ] Host-offloaded per-layer embedding tables (reclaims ~5.4 GiB VRAM —
      confirmed bf16 in the modelopt checkpoint, not FP4)
- [ ] Final `logit_softcapping = 30.0` kernel
- [ ] HF reference comparison: Python harness captures per-layer activations
      to .safetensors; Rust diffs against them. Target: global rel diff ≤
      5e-2 per layer, ≤ 1e-1 cumulative at final logits.

### Phase 4 — decode + CUDA Graphs [ ]
- [ ] Single-token decode path (Q=1 over cached K/V)
- [ ] Sampling: top-K, top-P, temperature (one kernel)
- [ ] CUDA Graph capture per decode shape; lazy recapture on prompt-length change
- [ ] First end-to-end `xenon-cli generate <prompt>` — report tok/s vs roofline

### Phase 5 — server + OpenAI API [ ]
- [ ] `POST /v1/chat/completions` with Gemma chat template
- [ ] SSE streaming
- [ ] `POST /v1/completions` (legacy)
- [ ] `GET /v1/models`
- [ ] Concurrent-request queuing (one request at a time on device in v1)

### Phase 6 — idle state machine [ ]
State machine timings are targets, not deadlines; measured and tuned later.

| State | After idle | Resident on device | TTFT on wake |
| --- | --- | --- | --- |
| Hot | (active) | weights + KV pool + graphs | <50 ms |
| Warm | 30 s | weights + CUDA context (no KV, no graphs) | ~200 ms |
| Cold | 5 min | CUDA context only | ~1.5 s |
| Frozen | 1 h | process exited | ~2 s (systemd socket activation) |

- [ ] Idle timer + transitions
- [ ] Wake-path warmup kernel (PCIe ASPM resume stall mitigation)
- [ ] systemd socket-activated unit file

### Phase 7 — polish + bench [ ]
- [ ] Nsight Compute profile passes, fix obvious tuning wins
- [ ] PLE prefetch overlap (one layer ahead)
- [ ] H2D overlap during decode if bandwidth-bound
- [ ] Benchmark suite vs Ollama (same prompts, same model, same GPU)

### Phase 8 - emotion vectors [ ]
- [ ] Investigate https://transformer-circuits.pub/2026/emotions/index.html#toc-15 - add in support for measuring/mapping emotion vectors
- [ ] Implement extra api return "emotions": ["happy": 0.2, "anxious": 0.1], etc

## Key technical decisions

| Decision | Rationale |
| --- | --- |
| Hand-rolled CUDA FFI (no `cudarc`) | Small, predictable, no dep churn across CUDA versions |
| Hand-rolled cuBLASLt bindings | Surface is ~10 functions; bindgen is more than we need |
| FP4 dequant-to-bf16 then bf16 GEMM | Correctness-first. Native FP4 operand GEMM moves to phase 7 when we have HF reference to validate against |
| Kernels compiled by `build.rs` (not NVRTC) | AOT, reproducible, faster startup; no runtime nvcc needed |
| `sm_120a` only (not broad gencode) | Single target; add compat later if we deploy elsewhere |
| Text-only in v0 | Vision + audio towers save 1.3 GiB VRAM and a lot of integration surface |
| PLE + lm_head host-offloaded | 6.7 GiB VRAM saved; per-token PCIe cost is <1 µs; this is *the* enabler for fitting on 7.53 GiB |
| row-major tensors throughout | Matches HF convention; operand-swap pattern in cuBLASLt wrapper handles the col-major conversion |
| Correctness metric: `max_abs / max(|ref|)` | GEMM and GELU pass through zero; per-element relative diff is unbounded there. Global rel diff is the defensible number |

## Deferred / open questions

- **HF reference comparison.** Phase 3 needs bit-exact-ish baseline. Options:
  (a) one-off Python script captures per-layer activations, (b) ONNX export,
  (c) run HF transformers via PyO3. Lean: (a). Defer until phase 3.
- **Weight layout on disk.** Currently read HF safetensors as-is. Preprocessing
  to a custom contiguous layout (tensors pre-aligned, scales interleaved) would
  cut cold-start time ~2×. Defer until the ops story demands it.
- **KV-sharing semantics.** `num_kv_shared_layers: 18` means 18 layers reuse
  KV, but the paper/config doesn't directly specify the mapping. Must be
  derived from the tensor presence pattern (layers without their own K/V
  proj weights reuse earlier ones). Resolve in phase 2.
- **Attention kernel.** Hand-rolled vs CUTLASS FMHA vs FlashInfer C++. Start
  hand-rolled for understanding and small-head-dim support; swap if perf
  demands it.
- **Native FP4 GEMM migration.** Post-phase-3 we swap dequant+bf16 GEMM for
  cuBLASLt NVFP4 operand mode. Expected win: ~2-3× MLP throughput when
  compute-bound, nothing when bandwidth-bound (which is most of decode).
- **Multi-modal.** Vision/audio/image gen are out of v0 scope. Revisit once
  text-only is shipping well.
- **Multiple concurrent requests.** v1 serves one at a time. Batching is a
  phase 8+ discussion once we know the actual traffic pattern.

## Out-of-scope (explicitly not doing)

- PagedAttention / continuous batching
- LoRA adapters
- Speculative decoding
- Training / fine-tuning
- Multi-GPU
- Quantization schemes other than NVFP4 in v1
- Vision / audio / image generation towers
