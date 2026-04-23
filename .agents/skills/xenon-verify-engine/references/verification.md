# Detailed Verification Procedures

## Table of Contents

1. [HF Reference Capture](#hf-reference-capture)
2. [Test ID Selection](#test-id-selection)
3. [Per-Test Details](#per-test-details)
4. [Numerical Tolerance Deep-Dive](#numerical-tolerance-deep-dive)
5. [Common Failure Modes](#common-failure-modes)

## HF Reference Capture

### capture.py Options

```bash
uv run python capture.py --model "$MODEL_DIR" --ids "2,108,107,1" --out activations
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Path to model directory with config.json, tokenizer.json, model.safetensors |
| `--ids` | "2,108,107,1" | Comma-separated token IDs to use as input |
| `--out` | "activations" | Output directory for safetensors + metadata |
| `--device` | "cpu" | Device for forward pass (cpu/cuda) |

The capture script dequantizes NVFP4 weights to bf16, runs a forward pass, and saves intermediate activations. CPU is recommended for reproducibility.

### Capturing with Custom Token IDs

To verify with different sequence lengths or token patterns:

```bash
# Short sequence (4 tokens)
uv run python capture.py --model "$MODEL_DIR" --ids "2,108,107,1" --out activations_short

# Medium sequence (16 tokens) — use repeating pattern
uv run python capture.py --model "$MODEL_DIR" --ids "2,108,107,1,2,108,107,1,2,108,107,1,2,108,107,1" --out activations_med

# Long sequence — generate from text
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-4-4b-it')
ids = tok.encode('The quick brown fox jumps over the lazy dog. ' * 10, add_special_tokens=False)[:32]
print(','.join(map(str, ids)))
"
```

### Metadata Format

`activations/metadata.json` contains:

```json
{
  "model_dir": "/path/to/model",
  "ids": [2, 108, 107, 1],
  "seq_len": 4,
  "hidden_size": 2560,
  "num_layers": 42,
  "dtype": "bfloat16",
  "captured_at": "2025-04-22T10:30:00"
}
```

## Test ID Selection

### Default IDs "2,108,107,1"

These correspond to Gemma-4 token IDs:
- `2` — `<bos>` (beginning of sequence)
- `108` — "The"
- `107` — " "
- `1` — `<eos>` (end of sequence)

This short sequence tests the basic pipeline with mixed special and content tokens.

### Edge Case ID Sets

| IDs | Description | When to Use |
|-----|-------------|-------------|
| `2,1` | Just BOS+EOS | Minimal forward pass sanity |
| `2,108` | BOS + one content token | Embedding gather only |
| `2,235366` | BOS + rare token | Tests embedding table coverage |
| `2,108,107,1,2,108` | Sequence with repeated pattern | Tests attention causality |

## Per-Test Details

### test-vs-hf-embed

**What it tests**: `embed_tokens(ids) * sqrt(hidden_size)`

**Implementation in Rust** (`cmd_test_vs_hf_embed`):
1. Load tokenizer and encode IDs
2. Gather embedding rows from mmap
3. Upload to device
4. Scale by `sqrt(hidden_size)`
5. Compare to HF's `embed_tokens_scaled`

**Expected output**:
```
=== xenon-cli test-vs-hf-embed ===
ids                     [2, 108, 107, 1]
max diff                0.0000e+00
```

**Failure indicators**:
- Non-zero diff → tokenizer vocab mismatch or embedding table endianness issue
- Large diff → check `bytemuck` alignment or bf16 byte order

### test-vs-hf-ple

**What it tests**: Full PLE (Per-Layer Embedding) pipeline

**Pipeline stages**:
1. `embed_tokens_per_layer(ids)` gather → `[B, T, L, Hl]`
2. Scale by `sqrt(Hl)`
3. `per_layer_model_projection` linear: `[B, T, L, Hl] @ [Hl, Hl]` → `[B, T, L, Hl]`
4. Per-layer RMSNorm
5. Combine: `(ctx + raw) * 1/sqrt(2)`

**Expected output**:
```
=== xenon-cli test-vs-hf-ple ===
ids                     [2, 108, 107, 1]
per_layer_inputs_raw    max_abs=...  max_rel=...  global_rel=...
per_layer_inputs_projected  max_abs=...  max_rel=...  global_rel=...
```

Both stages should show `global_rel < 1e-2`.

### test-vs-hf-layer

**What it tests**: One complete decoder layer

**Layer computation**:
```
input → pre_attn_norm → attention(QKV→qk_norm→RoPE→attn→o_proj) → residual
      → pre_mlp_norm → MLP(gate_proj→GELU→up_proj * down_proj) → residual
      → PLE_gate → PLE_GLU → PLE_projection → layer_scalar multiply
```

**KV sharing**: Layers 24–41 reuse KV from earlier layers. When testing these layers, the engine must read K/V from the owner layer, not compute new ones.

**Expected output**:
```
=== xenon-cli test-vs-hf-layer ===
layer                   0
ids                     [2, 108, 107, 1]
-- attention --
attn_after_oproj        max_abs=...  max_rel=...  global_rel=...
-- mlp --
mlp_out                 max_abs=...  max_rel=...  global_rel=...
-- layer final --
layer_final             max_abs=...  max_rel=...  global_rel=...
```

### test-vs-hf-tail

**What it tests**: Post-stack tail operations

**Stages**:
1. Final RMSNorm on last layer output
2. Tied lm_head (same weights as embed_tokens)
3. Logit softcapping: `tanh(logits / 30) * 30`

**Expected output**:
```
=== xenon-cli test-vs-hf-tail ===
-- final norm --
max_abs=...  max_rel=...  global_rel=...
-- lm_head --
max_abs=...  max_rel=...  global_rel=...
-- softcap --
max_abs=...  max_rel=...  global_rel=...
```

### test-vs-hf-full

**What it tests**: Complete 42-layer forward pass

**Process**:
1. Embed + PLE (same as test-vs-hf-ple)
2. Layer 0 (full attention, owns KV)
3. Layers 1–23 (full attention, each owns KV)
4. Layers 24–41 (sliding attention, KV shared from earlier layers)
5. Final norm + lm_head + softcap (same as test-vs-hf-tail)

**Expected output**: Per-layer diffs printed sequentially, all within tolerance.

## Numerical Tolerance Deep-Dive

### bf16 Precision

- 7-bit mantissa → ~1 ULP = 2^-7 ≈ 7.8e-3 relative
- Operations accumulate in different orders on GPU vs CPU
- CUDA reductions may use different associativity than numpy

### Why Global Rel is the Right Metric

Per-element relative error blows up when values pass through zero (e.g., GELU, attention logits). Global rel = `max_abs_diff / max(|reference|)` is stable because the denominator is the maximum magnitude across the tensor.

### Tolerance Budget by Operation

| Operation | Expected Global Rel | Reason |
|-----------|-------------------|--------|
| Embedding gather | 0.0 | Deterministic, no arithmetic |
| RMSNorm | < 1e-2 | Single reduction + multiply |
| Matmul | < 2e-2 | Accumulation order differences |
| GELU | < 2e-2 | Approximation + values near zero |
| Attention | < 1e-2 | Softmax normalization reduces error |
| RoPE | < 1e-2 | Deterministic trig, but bf16 angles |
| Softcap | < 1e-2 | tanh is well-behaved |

## Common Failure Modes

### KV Sharing Mismatch

**Symptom**: Layers 24+ show high diffs, layers 0–23 are fine.
**Cause**: Engine reads K/V from wrong owner layer.
**Fix**: Check `cmd_kv_map` output and verify `KvCache` slot assignment.

### Sliding Window Misconfiguration

**Symptom**: Attention diffs grow with sequence length.
**Cause**: Wrong `window` parameter passed to attention kernel.
**Fix**: Verify `sliding_window` config value matches HF (typically 4096 for Gemma-4).

### Chat Template Mismatch

**Symptom**: Generation produces different text than expected.
**Cause**: Chat template wrapping differs between engine and HF.
**Fix**: Compare `xenon_engine::wrap_chat_prompt` output with HF tokenizer's `apply_chat_template`.

### Quantization Dequant Mismatch

**Symptom**: Diffs present from layer 0, growing through the network.
**Cause**: NVFP4 dequantization differs between Rust and Python.
**Fix**: Run `test-dequant` to verify kernel matches reference.
