# IsoQuant KV Cache Compression — Research Plan

> Status: Phase 6 follow-up / Phase 8 candidate. Not yet scheduled.  
> Paper: [arXiv:2603.28430](https://arxiv.org/abs/2603.28430) "IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression" (Ji, 2026).  
> Code: https://github.com/ParaMind2025/isoquant

## What it is

IsoQuant compresses KV cache vectors via blockwise quaternion rotation followed by low-bit scalar quantization. It is a successor to RotorQuant that replaces 3D Clifford rotors with 4D quaternion blocks, yielding better hardware alignment and fewer FMAs.

## Mathematical Core

For each 4D block `v ∈ R⁴` (identified with a quaternion):

### IsoQuant-Full (6 DOF, strongest mixing)
```
encode:  ṽ = q_L · v · q̄_R          (quaternion sandwich)
quantize: v̂ = Q(ṽ)                 // per-coordinate scalar quantizer
store:   (v̂, ρ)                     // quantized direction + norm

decode:  v_rec = q̄_L · v̂ · q_R     // inverse sandwich
scale:   x̂ = ρ · v_rec
```

### IsoQuant-Fast (3 DOF, lower overhead)
```
encode:  ṽ = q_L · v
quantize: v̂ = Q(ṽ)
decode:  v_rec = q̄_L · v̂
```

### 2D Special Case (1 DOF, minimal)
```
encode:  ũ = R(θ) · u              // 2×2 planar rotation
quantize: û = Q(ũ)
decode:  u_rec = R(-θ) · û
```

**Key property:** Rotation is orthogonal → inverse is conjugate transpose → no matrix inversion needed. The `q_L, q_R` pair parameterizes the full `SO(4)` via `so(4) ≅ su(2)_L ⊕ su(2)_R`.

## Arithmetic Cost (at d = 128)

| Method | FMAs (forward) | FMAs (inverse) | Params/block |
|--------|---------------|----------------|--------------|
| Dense orthogonal | 16,384 | 16,384 | d² = 16,384 |
| RotorQuant (3D) | ~2,408 | ~2,408 | 3 scalars |
| **IsoQuant-Full** | **1,024** | **1,024** | **2×4 = 8 floats** |
| **IsoQuant-Fast** | **512** | **512** | **4 floats** |
| 2D planar | ~256 | ~256 | 1 float |

(1 quaternion = 4 floats; 2 quaternions for Full, 1 for Fast)

## Xenon Integration — Scope

### Where it fits

| KV Cache Op | Current | With IsoQuant |
|-------------|---------|---------------|
| **Append** (prefill) | `bf16` memcpy to `KV[slot][cur_len]` | rotate → quantize → pack `u8` → store |
| **Read** (decode attention) | `__bfloat162float(K[i]) * __bfloat162float(Q[j])` | unpack `u8` → dequantize → inv_rotate → dot |
| **Storage per slot** | `2 × max_len × h_kv × head_dim × 2` bytes | `2 × max_len × h_kv × head_dim × bw/8` bytes + rotation params |

For Gemma 4 (head_dim = 256 sliding, 512 full; h_kv = 2; 24 owner slots):

| Config | Current bf16 | IsoQuant 4-bit | IsoQuant 2-bit |
|--------|-------------|----------------|----------------|
| max_len = 4096 | **~261 MiB** | **~65 MiB** | **~33 MiB** |
| max_len = 131072 | **~8.4 GiB** (OOM) | **~2.1 GiB** | **~1.0 GiB** |

### Kernel Changes Needed

#### 1. `kv_append_isoquant.cu` — new file

Replace the current `kv_append_bf16` kernel with a fused rotate+quantize path:

```cuda
__global__ void kv_append_isoquant_kernel(
    uint8_t* __restrict__ out_k,      // quantized K
    uint8_t* __restrict__ out_v,      // quantized V
    const __nv_bfloat16* __restrict__ k_src,
    const __nv_bfloat16* __restrict__ v_src,
    int head_dim, int num_blocks,      // head_dim/4 blocks
    const float4* __restrict__ qL_k,   // per-block left quaternions (K)
    const float4* __restrict__ qR_k,   // per-block right quaternions (K) — null for Fast
    const float4* __restrict__ qL_v,   // per-block left quaternions (V)
    const float4* __restrict__ qR_v,   // per-block right quaternions (V) — null for Fast
    float global_scale,                // per-tensor scale for VQ
    int cur_len, int t_q)
{
    // One thread per (head, token, block)
    // 1. Load 4 bf16 → quaternion
    // 2. Rotate: ṽ = q_L * v * q̄_R (or q_L * v for Fast)
    // 3. Scalar quantize each float component to bw bits
    // 4. Pack bw×4 bits into ceil(bw/2) bytes
    // 5. Write to out_k[out_offset]
}
```

**Quantization details:**
- Bit widths: 2, 3, or 4 bits per coordinate
- Per-block or per-head scale (like the existing UE4M3 block scales in FP4)
- Paper uses symmetric uniform quantization around zero after rotation
- Optional: per-token absmax scaling for better dynamic range

**Rotation parameters:**
- Stored as `float4` per `head_dim/4` blocks per layer
- 24 owner slots × (head_dim/4) × 4 floats ≈ 6-12 KiB total (negligible)
- Calibrated offline on reference activations, not learned online

#### 2. `attn_split_kv` — modify Q·Kᵀ and V·softmax paths

In the partial kernel, replace:
```cuda
// CURRENT (bf16):
acc += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
```

With:
```cuda
// IsoQuant path (dequantize + inv_rotate on the fly):
// Load packed u8 chunk → unpack to 4 floats → dequantize
// inv_rotate: v = q̄_L * v̂ * q_R (or q̄_L * v̂ for Fast)
// accumulate dot product
```

**Critical:** The existing `attn_split_kv` kernel already loads K/V from global memory into registers per thread. We can fuse dequantization+inv_rotation into the load without extra memory traffic — the packed bytes are smaller than bf16, so bandwidth actually *decreases*.

#### 3. `KvCache` — extend layout

Current:
```rust
pub struct KvCache {
    k: Vec<DeviceBuffer<bf16>>,  // per physical slot
    v: Vec<DeviceBuffer<bf16>>,
    cur_len: usize,
    cur_len_dev: DeviceBuffer<i32>,
}
```

Extended:
```rust
pub struct KvCache {
    k: Vec<DeviceBuffer<u8>>,     // quantized K
    v: Vec<DeviceBuffer<u8>>,     // quantized V
    cur_len: usize,
    cur_len_dev: DeviceBuffer<i32>,
    // IsoQuant params per physical slot
    isoquant_mode: IsoQuantMode,  // Full / Fast / 2D
    bit_width: u8,                // 2, 3, or 4
    qL_k: DeviceBuffer<float4>,   // left quats for K (per slot, per block)
    qR_k: Option<DeviceBuffer<float4>>,  // right quats for K (Full only)
    qL_v: DeviceBuffer<float4>,   // left quats for V
    qR_v: Option<DeviceBuffer<float4>>,  // right quats for V (Full only)
}
```

### Calibration (Offline)

IsoQuant needs rotation quaternions fitted to the K/V distribution of the target model. This is **not** learned online — it's a one-time calibration:

1. **Capture reference K/V** from `test-vs-hf-layer` (already exists for correctness validation)
2. **Fit quaternions per layer:**
   - For each layer, sample N K/V vectors at various context lengths
   - Compute per-block covariance
   - Extract dominant eigenvectors → quaternions via SVD on SO(4) Lie algebra
   - Or simpler: random Haar-distributed quaternions (paper shows random works well)
3. **Validate round-trip MSE:**
   - `rotate(v) → quantize → dequantize → inv_rotate(v)` vs original
   - Target: MSE < 1% of vector norm (paper achieves this at 4-bit)

### Validation Matrix

| Test | Current | With IsoQuant | Tolerance |
|------|---------|---------------|-----------|
| `test-vs-hf-embed` | Pass | Unchanged | n/a |
| `test-vs-hf-ple` | Pass | Unchanged | n/a |
| `test-vs-hf-layer` | Pass | Likely pass | ≤ 2e-2 (MSE additive) |
| `test-vs-hf-tail` | Pass | Unchanged | n/a |
| `test-vs-hf-full` | Pass | Likely pass | logits top-1/top-5 match |
| **New: `test-isoquant-roundtrip`** | n/a | **Required** | MSE < 1% norm |
| `generate "hi" --chat` | "Hello!..." | Must match | Exact text |

## Implementation Phases

### Phase 8a — Kernel Primitives (1-2 weeks)
- [ ] `isoquant_rotate_bf16_kernel` — single-vector rotate+quantize (unit test)
- [ ] `isoquant_inv_rotate_bf16_kernel` — dequantize+inv_rotate (unit test)
- [ ] `isoquant_roundtrip_bf16` — fused encode→decode, measure MSE vs bf16
- [ ] CLI: `test-isoquant-roundtrip --layer N --bit-width 4 --mode fast`

### Phase 8b — KV Integration (1-2 weeks)
- [ ] `kv_append_isoquant` — replace bf16 append with rotate+quantize path
- [ ] `attn_split_kv` — modified to read quantized K/V + inv_rotate on-the-fly
- [ ] `KvCache` extended with `IsoQuantMode` + rotation params
- [ ] Calibration harness: `calibrate-isoquant --out isoquant_params.bin`

### Phase 8c — End-to-End Validation (1 week)
- [ ] `test-vs-hf-layer` with IsoQuant KV — verify within tolerance
- [ ] `generate` sanity — text quality check
- [ ] Benchmark: short (4K) and long (32K+) context decode throughput
- [ ] Memory footprint measurement

### Phase 8d — Optional — `attn_flash_tc` Integration (stretch)
- [ ] Extend tensor-core flash attention to read quantized K/V tiles
- [ ] Pre-fetch + inv_rotate in shared memory during QK matmul
- [ ] Impact: minimal at 4K, significant at 32K+ where K/V bandwidth dominates

## Open Questions

1. **Bit-width tradeoff for K vs V.** K and V may have different sensitivity. Can we use 4-bit for K and 2-bit for V? Or per-head adaptive width?
2. **Shared KV layers.** Layers 24–41 share KV with an owner layer. Do shared layers use the same rotation quats as their owner? (Probably yes — simpler.)
3. **Dynamic vs static quantization.** Current plan uses static per-layer quaternions. Dynamic (per-token) would adapt to context but adds compute.
4. **Calibration data coverage.** How many tokens / layers / positions needed for robust quaternion fitting? Paper uses synthetic; real activations may need more.

## Hardware Impact

| Metric | bf16 KVCache | IsoQuant-Fast 4-bit | IsoQuant-Fast 2-bit |
|--------|-------------|--------------------|--------------------|
| Storage @ 4K | 261 MiB | 65 MiB | 33 MiB |
| Storage @ 32K | 2.1 GiB | 524 MiB | 262 MiB |
| Decode bandwidth / step | 2×K + 2×V = 4 bytes/elt | 0.5 bytes/elt | 0.25 bytes/elt |
| Decode compute overhead | 0 | ~512 FMAs per head | ~512 FMAs per head |
| Roofline bound | Memory (167 GB/s) | Compute-bound? | Compute-bound? |

At d=256, IsoQuant-Fast adds ~512 FMAs per K/V read. On Blackwell, 512 FMAs ≈ 0.3 µs at 60 TFLOP/s. The bandwidth saved is 1.5 bytes/elt × ~4096 elts ≈ 6 KiB → 0.04 µs at 167 GB/s. So ** IsoQuant-Fast is compute-limited on decode** — the FMA overhead may outweigh the bandwidth savings at short context. The real win is at **long context** where KV cache size is the bottleneck (enables 32K+ context on this 7.5 GiB card).

## Decision Gate

**Do not start Phase 8 until:**
1. Phase 6 perf targets are met (~60 tok/s decode sustained, 3K+ tok/s prefill)
2. Phase 7 (emotion vectors) is scoped or deprioritized
3. At least one long-context benchmark (32K tokens) is regularly run — IsoQuant only pays off there

