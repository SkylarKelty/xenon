# xenon — plan

Inference engine for a single model on a single box, built from scratch in Rust
+ CUDA. Optimised for **cold start, idle power, and operational simplicity**,
not for matching vLLM's server throughput.

## Current status (2026-04-23, commit `c7e6aa1`)

Single-stream, `cosmicproc/gemma-4-E4B-it-NVFP4` on RTX PRO 2000 Blackwell,
T_prompt=155, max_new=50, warm GPU:

| Metric | xenon | ollama 0.21 (same model) | ratio |
| --- | --- | --- | --- |
| Decode | ~60 tok/s | 23.79 tok/s | **2.5×** |
| Prefill | 3242 tok/s | 2693 tok/s | **1.20×** |

(ollama runs the same `gemma4:e4b` Q4_K_M with 68% of the model on CPU because
it doesn't fit the 7.5 GiB VRAM; xenon keeps everything on device via PLE /
lm_head host-offload.)

Phases 0–5 complete (foundations, MLP, attention, full forward, decode, server
with OpenAI API + streaming + request batching). Phase 4.1 native NVFP4 GEMM
wired for prefill; phase 6 perf polish ongoing.

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

#### Week 3 [x] — real-weight MLP end-to-end
- [x] `list <model> --pattern <glob>` CLI utility to surface tensor names
- [x] Tensor-family loader: `(weight, weight_scale, weight_scale_2)` tuple for
      a given module prefix, with shape validation against config
      (`MmapWeights::load_quant_linear` / `load_bf16`)
- [x] `CublasLt::linear_bf16` — `y = x @ W^T` for W stored `[N, K]` HF-style
- [x] `test-mlp <model> --layer 0`: dequant gate/up/down, run full MLP chain,
      compare GPU vs host reference. Result: global rel diff 3.1e-3 at
      batch=1 (well under 5e-2 target); 0.54 ms/forward post-warmup.
- [x] Layer prefix convention confirmed: `model.language_model.layers.N.*`
- [x] Discovered (for phase 2): `per_layer_input_gate` + `per_layer_projection`
      are the PLE injection path (quantized), not just gathered embeddings.
      Also `self_attn.{q,k}_norm` weights present — QK LayerNorm is enabled.

**Key finding:** cuBLASLt algorithm selection is ~90 ms on the first call per
unique shape. Must be pre-warmed at init for every shape that might appear
during serving; phase 4 CUDA Graph capture subsumes this.

### Phase 2 — attention [x]
- [x] Tokenizer via HF `tokenizers` crate. `test-tokenizer` encode/decode
      round-trip is exact; vocab matches config (262144).
- [x] Token embedding gather (bf16) — bit-identical to host slice
- [x] Per-layer embedding gather from host mmap. `MmapWeights::gather_rows_bf16`
      reads the 5.25 GiB PLE table (262144 × 10752 bf16) without uploading it;
      per-forward gather cost is T × 21 KiB. Host cold-cache ~7 ms, warm ~0 ms.
      Pinning H2D buffer is a later optimization.
- [x] RoPE kernel (dual head_dim: 256 sliding full rotary, 128/512 partial
      for full-attention layers; rope_theta 10k / 1M via config)
- [x] Naive attention (Q@Kᵀ → softmax → @V, GQA-aware) for correctness
      baseline. Shared-memory scores; bounded at T_kv ≤ ~11K fp32 entries.
- [x] Softmax kernel (causal + optional sliding-window mask) — exact match
      vs ref, row-sum within ~1e-3 of 1.0
- [x] KV cache allocator with GQA layout (2 KV heads × head_dim per layer).
      `KvCache` holds per-slot `[max_len, h_kv, head_dim]` K/V buffers with
      a `slot_for_layer` indirection for sharing. `append` writes at
      `cur_len`; `advance` marks N tokens consumed.
- [x] Sliding-window mask (window 512) + full attention variants — same
      kernel, window=0 for full attention
- [x] KV sharing: 18 of 42 layers reuse KV from earlier layers. Detected
      via `MmapWeights::layer_owns_kv`: layers missing `k_proj.input_scale`
      are the shared ones. Map: layers 0..23 own their KV, layers 24..41
      share (strongly suggestive mapping: layer `n` shares with `n-24`,
      same-kind — to confirm vs HF in phase 3).
- [x] FlashAttention-2 tile kernel (bf16 I/O, fp32 accumulate) — tiles
      K/V in BR=16 chunks with online softmax; shared mem is O(BR*D)
      independent of T_kv. Matches naive kernel within ~1e-4 global rel,
      handles T_kv up to 65K+. Not perf-tuned (Phase 6).
- [x] CLI: `test-rope`, `test-softmax`, `test-embed`, `test-attn-layer`,
      `test-attn-decode`, `test-kv-cache`, `kv-map`. End-to-end: real
      weights, ~8 layers across both kinds pass with global rel ≤ 5.6e-3
      at T=4. Decode-shape and cache-append paths match prefill's last
      row bit-for-bit.

### Phase 2 — HF-alignment fixes (applied after source review) [x]
- [x] RoPE: inv_freq denominator uses `head_dim` (not `rotary_dim`); pair
      stride is `head_dim/2` (not `rotary_dim/2`). Matches HF
      `_compute_proportional_rope_parameters` + `rotate_half` for partial
      rotary. For sliding layers (prf=1.0), both are equivalent.
- [x] Attention scaling: pass `scaling=1.0` to attn (not `1/sqrt(D)`).
      Gemma 4 lets the learned q_norm/k_norm weights control Q·K magnitude.
      No attention softcap (unlike Gemma 2).
- [x] V normalization: RMSNorm extended to accept `weight: Option<_>` —
      null pointer = `with_scale=False` (pure RMS, no learned weight).
      Applied to V after v_proj. No corresponding tensor in safetensors.
- [x] Embedding scale: `embed_gather_bf16` takes a scalar multiplier.
      Apply `sqrt(hidden_size) ≈ 50.6` for `embed_tokens` and
      `sqrt(hidden_size_per_layer_input) = 16` for `embed_tokens_per_layer`
      (matches `Gemma3TextScaledWordEmbedding`).
- [x] `layer_scalar`: per-layer `[1]`-bf16 multiplier at the very end of
      each decoder layer. `scale_bf16` tail multiply wired into both
      `xenon-engine::layer_forward` (prefill + batched) and the
      `xenon-cli` mirror.

### Phase 3 — full forward pass [x]
- [x] 42-layer chain: embed → (norm → attn → norm → mlp + PLE inject) × 42 →
      norm → lm_head → logits. Wired in `cmd_test_vs_hf_full` with a
      `LayerWeights` struct + `layer_forward` helper, dequanting each layer
      on the fly to stay within VRAM. 610 ms for T=4 on the first call
      (cuBLASLt algo selection dominates).
- [x] KV sharing: layers 0..23 own KV; 24..41 share via the last non-shared
      layer of the same type (22 for sliding, 23 for full). Driven by
      `MmapWeights::layer_owns_kv` + `KvCache`'s `slot_for_layer` indirection.
- [x] Final `logit_softcapping = 30.0` — `softcap_bf16` kernel. Bit-exact
      formula match: `tanh(x/30) * 30`.
- [x] HF reference harness (`tools/hf-ref/`): Python dequants NVFP4
      ourselves (sidestepping modelopt/transformers<5 conflict) and feeds
      a vanilla bf16 Gemma4. Captures embed, per-layer attn/mlp/final,
      final norm, pre/post softcap logits.
- [x] HF diff tests: embed (4e-3), PLE assembly (8e-3), per-layer forward
      (3e-4 to 1e-2 across all own-KV layers), tail (5e-3), full chain
      (logits global_rel 2.85e-2, top-1/top-5 predictions identical to HF).
- [x] Host-offloaded `lm_head` (tied to embed_tokens): transient 1.25 GiB
      upload for the GEMM then freed. Saves 1.25 GiB *permanent* VRAM at
      the cost of ~110 ms H2D+GEMM per call. Async overlap with decoder
      compute is a phase-4 optimization.
- [x] Host-offloaded per-layer embedding table — already host-resident
      via `MmapWeights::gather_rows_bf16`; only ~21 KiB/token touches
      device per forward.
- [x] Host-offloaded input embedding: `embed_tokens` rows gathered on
      host and uploaded as [T, H] instead of the full [V, H] table.

### Phase 4 — decode + generate [x]
- [x] Single-token decode path (t_q=1 over cached K/V). `LayerMeta` carries
      `t_q` / `t_kv` / `q_pos_base` so `layer_forward` handles prefill and
      decode identically.
- [x] Persistent KV cache across steps via `KvCache::append`+`advance` on
      each forward pass.
- [x] First end-to-end `xenon-cli generate <prompt>` — tokenize, prefill,
      greedy-decode, stop at EOS (`<turn|>`, 106). `--chat` flag wraps with
      Gemma 4 turn markers.
- [x] `xenon-cli bench` — warmup + N iters, per-step timing, JSON output
      to `results/`. Phase profiler dumps per-block timings.
- [x] Async `lm_head` H2D on a second stream. `PinnedBuffer<T>` via
      `cudaMallocHost` so the 1.25 GiB PCIe transfer overlaps with the
      decoder stack. Hidden inside runtime by the end.
- [x] cuBLASLt algorithm cache per (m, n, k, opA, opB) on `CublasLt`.
      Minor win on its own; prereq for graph captures later.
- [x] `DeviceBuffer::new_async` via `cudaMallocAsync` on the default
      stream-ordered pool. Transient scratch hits a cached block after
      warmup. Biggest single decode win (~60%).
- [x] Sampling: top-K, top-P, temperature. Device-side top-K kernel
      (`xk_sample_topk_bf16` in `crates/xenon-kernels/src/cu/sample.cu`)
      does temperature-scaled softmax + iterative top-K extraction in a
      single block; host applies top-P cutoff + inverse-CDF sampling via
      a seeded ChaCha PRNG on the compact K-sized slice. Greedy is a
      kernel fast path and stays on host argmax by default (no
      per-step D2H+alloc overhead in the common case). Wired through
      `Engine::generate` / `generate_batch` and the server. CLI: `bench`
      and `generate` gained `--temperature / --top-p / --top-k / --seed`;
      defaults preserve the greedy baseline so historical bench numbers
      remain comparable. `test-sample` validates vs a host reference.
      Measured: sampler overhead indistinguishable from noise at T=0.7
      top_p=0.9 top_k=40 (58.4 vs 58.5 tok/s decode, same prompt).

**Perf progression (T_prompt=15, max_new=15, haiku prompt):**

| Config                | decode ms/step | decode tok/s | vs baseline |
| --------------------- | -------------- | ------------ | ----------- |
| baseline (phase 3)    | 311            | 3.21         | —           |
| async `lm_head`       | 291            | 3.43         | +7%         |
| + algo cache          | 296            | 3.38         | (flat)      |
| + cudaMallocAsync     | **184**        | **5.43**     | **+69%**    |

Profile after all three:
```
decoder stack (42 layers)   114 ms
lm_head GEMM + softcap        4 ms
wait for lm_head H2D          0 ms   (fully overlapped)
other                         ~1 ms
```

The dominant remaining cost is MLP FP4→bf16 dequant bandwidth per layer
per step (~40 ms across the 42 layers). Only native FP4 GEMM fixes
that — deferred to the open-questions / future-phase bucket.

### Phase 4.1 — native NVFP4 GEMM (experimental) [~]
- [x] `CublasLt::linear_nvfp4`: FP4×FP4 GEMM, bf16 output, fp32 accumulate.
      Both operands use `CUDA_R_4F_E2M1` + `VEC16_UE4M3` per-block scales.
      Per-tensor weight global_scale folded into alpha.
- [x] `nvfp4_quantize_bf16` kernel: bf16 → packed FP4 + UE4M3 block scales.
      Round-trip rel ~13% on random bf16 in [-3, 3] (expected for FP4).
- [x] `test-nvfp4-roundtrip` and `test-nvfp4-linear` CLI tests.
- [ ] **Integration blocked by tensor-core M-threshold.** On this box
      (sm_120a / Blackwell), NVFP4 GEMM silently returns garbage at
      M < ~128 — minimum tile size for FP4 WGMMA. cuBLASLt does not
      error, which is dangerous.
      - At M=1 (decode) and M=4..32 (typical prefill for a short prompt),
        native NVFP4 is unusable.
      - Only starts giving reasonable answers at M ≥ 128.
      - Kept as a future-ready primitive for batched serving (phase 5+)
        and long-prefill paths, NOT wired into `layer_forward`.

### Phase 5 — server + OpenAI API [x]
- [x] `POST /v1/chat/completions` with Gemma chat template + SSE streaming
      (`8a8c340`).
- [x] `POST /v1/completions` legacy endpoint; `cli generate` migrated to
      share the `xenon-engine` code path (`6321902`).
- [x] `GET /v1/models`.
- [x] Concurrent-request handling: engine-level request batching
      (`dcc44fa`) + server batcher that shares a forward pass across
      concurrent requests (`1618d35`).
- [~] **CUDA Graph capture — investigated, deprioritized.** Prereq
      (persistent `DecodeScratch` covering all per-step buffers) was
      built and benched: no meaningful change vs the cudaMallocAsync
      pool (189 ms/step either way; run variance ≈ ±2%). Profiling
      shows the decoder stack alone is ~121 ms of kernel time, so
      decode is compute- and bandwidth-bound, not launch-bound (42
      layers × ~20 launches × 1.57 µs ≈ 1.3 ms total launch overhead
      — <1% of wall time). Graphs won't recover ms here; reverted the
      scratch wiring to keep the code lean. Kernel wrappers now accept
      `buf.len() >= required` (was `==`) so the infrastructure is
      still compatible if we revisit.
- [x] **Dispatch NVFP4 GEMM for large-M paths.** `QuantLinearDev::forward`
      picks `m==1` → fused FP4 gemv, `m>=128` → native NVFP4 (cuBLASLt
      VEC16_UE4M3), otherwise → dequant + bf16 linear. The shared-activation
      path (`prepare_fp4_activation` + `forward_fp4_prepacked`) reuses the
      quantized activation across co-sourced projections for +8.9% prefill.

### Phase 6 — polish + bench [~]
- [ ] Nsight Compute profile passes, fix obvious tuning wins
- [ ] PLE prefetch overlap (one layer ahead)
- [ ] H2D overlap during decode if bandwidth-bound
- [ ] Benchmark suite vs Ollama (same prompts, same model, same GPU)
- [x] **Pre-dequant `per_layer_model_projection` at load** (2026-04-23).
      Was calling `dequant_to` (647 μs) every forward_step; now dequanted
      once into a 43 MB bf16 buffer at load, mirroring the lm_head pattern.
      `ple assembly` phase 1.20 → 0.24 ms.
- [x] **`DeviceBuffer::new` → `new_async` in forward_step** (2026-04-23).
      All 11 call sites in the outer forward_step driver switched to
      stream-ordered alloc; H2D uploads use `copy_from_host_bytes_async`.
      Same change mirrored to `forward_step_batched` for the server path.
      `forward_step_profiled` left sync (its job is per-phase timing via
      host syncs).
- [x] **Shared NVFP4 activation across co-sourced projections** (2026-04-23,
      commit `c7e6aa1`). `nvfp4_quantize_bf16` was 27% of prefill GPU time because
      it was re-running on the same post-norm activation for each of q/k/v
      and each of gate/up. Added `QuantLinearDev::prepare_fp4_activation` +
      `forward_fp4_prepacked`: quantize once per shared activation, reuse
      for all co-sourced projections. Saved ~90 quantize calls per forward
      (3 per own-KV layer × 24 + 1 per shared-KV × 18). Bench: prefill
      2976 → 3242 tok/s (+8.9%), decode unchanged. Now **1.20× ollama
      prefill** (was 1.10×).
- [x] **Tensor-core flash-attention for prefill, all D** (2026-04-23,
      commits `1d87afb` + cp.async follow-up `e77db3f`).
      New `xk_attn_flash_tc_bf16_kernel` — flash-attention-2 style tiled
      kernel on `mma.m16n8k16` bf16. BR=BC=16, one warp per block, O_acc in
      smem. K and V in separate smem buffers; `cp.async.cg` prefetches
      V[i] (overlaps QK+softmax) and K[i+1] (overlaps PV). This doubles
      memory/compute throughput and lets D=512 win too (it was a
      regression without the overlap).
      Kernel-level: D=256 window=0 3.66×, D=256 window=512 3.39×,
      D=512 window=0 1.78×, D=512 window=512 1.81× vs naive.
      Bench (3 warm runs): prefill 2541 → 3097 tok/s (+21.9%), decode
      ~60 tok/s (unchanged within noise). Now **1.15× ollama prefill**
      (2693 tok/s) — first time we cross the ollama line.
      `test-mma-bf16` validates the mma PTX + fragment layout on sm_120a;
      `test-attn-flash-tc` times the kernel vs naive.
- [x] **Attention kernel rewrite — split-KV** (2026-04-23, commit `da09d5d`).
      New `xk_attn_split_kv_*_bf16_kernel` pair: partial kernel grids over
      `(q_tok × q_head × kv_chunk)` with naive's work pattern per chunk
      (threads partition T_kv slice, each does a full D-dot, no intra-block
      reduction); merge kernel combines the n_chunks partials via online
      softmax. Dispatched in `layer_forward` only when `T_q × H <
      2 × SM_count` (decode shapes). Prefill still uses naive — adding
      split-KV there is pure merge-kernel overhead (see loss row below).

**Attention kernel findings (2026-04-23, updated with split-KV results).**

Per-launch measured via `test-attn-flash` and `test-attn-split-kv`:

| Shape | naive | flash (rewritten) | split-KV (auto cs) | split-KV speedup |
| --- | --- | --- | --- | --- |
| T_q=1, T_kv=255, D=256 | 48 μs | 119 μs | 20 μs (cs=32) | **2.43×** |
| T_q=1, T_kv=255, D=512 | 94 μs | 253 μs | 32 μs (cs=16) | **2.97×** |
| T_q=1, T_kv=512, D=256 | 94 μs | — | 32 μs (cs=64) | **3.00×** |
| T_q=155, T_kv=155, D=256 | 442 μs | 1010 μs | 482 μs (cs=64) | 0.86× |
| T_q=155, T_kv=155, D=512 | 858 μs | 3383 μs | 975 μs (cs=32) | 0.89× |

Naive wins vs. flash because threads partition T_kv and each does a full
D-dot independently — no intra-block reduction. Flash partitions D and
cooperates per K row, needing BR=16 block-wide reductions per tile; that
only amortizes for T_kv ≥ ~10K where naive's O(T_kv) shmem blows the 48
KiB budget. Improved flash kept as long-context fallback (naive caps
around T_kv≈12K).

Split-KV wins on decode because grid size is the bottleneck, not kernel
internals. Decode's 1×8=8 blocks only uses 31% of ~26 SMs; split-KV at
cs=32–74 grows that to 40–64 blocks per launch (~90%+ occupancy).
For prefill, T_q×H=1240 already oversubscribes the SMs, so split-KV's
merge-kernel overhead is pure loss — hence the conditional dispatch.

**Full-bench impact (commit TBD, T_prompt=155, max_new=50):**
- decode 54.05 → 61.48 tok/s (+13.7%); decoder stack 13.96 → 12.22 ms
- prefill 2441 → 2522 tok/s (+3.3%, within noise)
- 2.58× ollama's 23.8 tok/s decode (was 2.17×)

### Phase 7 - emotion vectors [ ]
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
- **Attention kernel — further tensor-core work.** `mma.m16n8k16` +
  `cp.async` landed for all prefill D (+22% prefill bench, 1.15× ollama).
  Remaining levers:
  - Upgrade to `wgmma.mma_async` (warpgroup, 128 threads) — lets 4 warps
    pipeline K/V loads against MMA compute at instruction level (multiple
    in-flight MMA groups). Bigger rewrite; another 2-3× expected on top
    of the current mma path.
  - Larger BR (32 or 64) — amortize per-tile overhead across more rows
    (needs register/smem budget work; may require wgmma first).
  - For decode (currently split-KV): tensor cores don't help (M=1 < MMA
    tile m=16). Decode bottleneck now shifts to lm_head GEMM (4 ms/step).
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
