---
name: xenon-verify-engine
description: Verify the Xenon LLM inference engine against HuggingFace reference outputs. Use when asked to validate correctness, diff against a reference implementation, check numerical accuracy, or verify that engine outputs match HF. Covers single-layer tests, full forward pass, embedding/PLE accuracy, and tail/layer stack verification.
---

# Xenon Engine Verification

Verify the Xenon engine against HuggingFace `transformers` reference outputs. The verification suite compares every major intermediate activation between the Rust engine and a Python HF reference capture.

## Prerequisites

- CUDA device available and `xenon-cli` built (`cargo build --release --bin xenon-cli`)
- Model path configured (default: `~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b`)
- HF reference activations captured (see `tools/hf-ref/`)

## HF Reference Capture

Before running verification, capture HF reference activations:

```bash
cd tools/hf-ref
uv sync
MODEL_DIR=~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b
uv run python capture.py --model "$MODEL_DIR" --ids "2,108,107,1" --out activations
```

Output: `activations/activations.safetensors` + `activations/metadata.json`.

The reference captures these tensors:

| Name | Shape | Description |
|------|-------|-------------|
| `embed_tokens_scaled` | `[B, T, H]` | `embed_tokens(ids) * sqrt(H)` |
| `per_layer_inputs_raw` | `[B, T, L, Hl]` | `embed_tokens_per_layer(ids) * sqrt(Hl)` reshaped |
| `per_layer_inputs_projected` | `[B, T, L, Hl]` | After per-layer model projection + norm |
| `layer_NN.attn_after_oproj` | `[B, T, H]` | Attention output after `o_proj` |
| `layer_NN.mlp_out` | `[B, T, H]` | MLP block output |
| `layer_NN.final` | `[B, T, H]` | Layer output (after residual, PLE, layer_scalar) |
| `final_norm_out` | `[B, T, H]` | After final RMSNorm |
| `logits_pre_softcap` | `[B, T, V]` | `lm_head(final_norm_out)` |
| `logits_post_softcap` | `[B, T, V]` | After `tanh(pre/30)*30` |

## Verification Levels

### Level 1: Embeddings (`test-vs-hf-embed`)
Verify the token embedding gather + sqrt(H) scale:

```bash
HF=tools/hf-ref/activations
cargo run --release --bin xenon-cli -- test-vs-hf-embed "$MODEL" "$HF" --ids "2,108,107,1"
```

Expected: max abs diff near zero (embed gather is deterministic).

### Level 2: PLE Pipeline (`test-vs-hf-ple`)
Verify per-layer embedding (PLE) assembly: gather + sqrt(Hl) scale, per-layer projection linear, per-layer RMSNorm, combine (ctx + raw) * 1/sqrt(2):

```bash
cargo run --release --bin xenon-cli -- test-vs-hf-ple "$MODEL" "$HF" --ids "2,108,107,1"
```

Expected: global rel diff ≤ 1e-2 (bf16 tolerance).

### Level 3: Single Layer (`test-vs-hf-layer`)
Run one full decoder layer end-to-end and diff against HF `layer_N.final`. Exercises every primitive: pre/post layernorm, attention (QKV + norms + RoPE + attn + o_proj), residual, MLP, PLE gate/GLU/projection, and layer_scalar:

```bash
# Test layer 0 (first layer, no KV sharing)
cargo run --release --bin xenon-cli -- test-vs-hf-layer "$MODEL" "$HF" --ids "2,108,107,1" --layer 0

# Test layer 24 (first KV-shared layer)
cargo run --release --bin xenon-cli -- test-vs-hf-layer "$MODEL" "$HF" --ids "2,108,107,1" --layer 24

# Test layer 41 (last layer)
cargo run --release --bin xenon-cli -- test-vs-hf-layer "$MODEL" "$HF" --ids "2,108,107,1" --layer 41
```

Expected: global rel diff ≤ 2e-2. Pay attention to:
- **Attention output**: Should match closely (≤ 1e-2)
- **MLP output**: Slightly more tolerance due to GELU approximation
- **Layer final**: Should be within 2e-2

### Level 4: Tail Stack (`test-vs-hf-tail`)
Verify the post-stack tail: final RMSNorm + tied lm_head + softcap, comparing to HF at each stage:

```bash
cargo run --release --bin xenon-cli -- test-vs-hf-tail "$MODEL" "$HF"
```

Expected: Each stage (final_norm, lm_head, softcap) within bf16 tolerance.

### Level 5: Full Forward Pass (`test-vs-hf-full`)
Run the complete 42-layer forward pass and diff every intermediate:

```bash
cargo run --release --bin xenon-cli -- test-vs-hf-full "$MODEL" "$HF" --ids "2,108,107,1"
```

Expected: All layers within tolerance. This is the comprehensive correctness check.

### Level 6: Text Generation Verification
After numerical verification passes, verify text generation produces coherent output:

```bash
# Short prompt generation test
cargo run --release --bin xenon-cli -- generate "$MODEL" --prompt "The capital of France is" --max-new 10

# Chat prompt generation test
cargo run --release --bin xenon-cli -- generate "$MODEL" --prompt "What is 2+2?" --max-new 10 --chat
```

Expected: Sensible continuations. For "The capital of France is", expect " Paris" or similar. Note: `--chat` is required because the default model (`gemma-4-4b-it-NVFP4`) is instruction-tuned. Without it, the model may fall into repetition loops.

## Full Verification Checklist

Run this sequence for a complete engine verification:

```bash
MODEL=/home/k1811651/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/02ecc99b351ea32f7d332fc7566e44bfc79baf0b
HF=tools/hf-ref/activations
IDS="2,108,107,1"

echo "=== Level 1: Embeddings ==="
cargo run --release --bin xenon-cli -- test-vs-hf-embed "$MODEL" "$HF" --ids "$IDS"

echo "=== Level 2: PLE Pipeline ==="
cargo run --release --bin xenon-cli -- test-vs-hf-ple "$MODEL" "$HF" --ids "$IDS"

echo "=== Level 3: Single Layers ==="
for LAYER in 0 24 41; do
  echo "--- Layer $LAYER ---"
  cargo run --release --bin xenon-cli -- test-vs-hf-layer "$MODEL" "$HF" --ids "$IDS" --layer $LAYER
done

echo "=== Level 4: Tail Stack ==="
cargo run --release --bin xenon-cli -- test-vs-hf-tail "$MODEL" "$HF"

echo "=== Level 5: Full Forward Pass ==="
cargo run --release --bin xenon-cli -- test-vs-hf-full "$MODEL" "$HF" --ids "$IDS"

echo "=== Level 6: Text Generation ==="
cargo run --release --bin xenon-cli -- generate "$MODEL" --prompt "The capital of France is" --max-new 10
cargo run --release --bin xenon-cli -- generate "$MODEL" --prompt "What is 2+2?" --max-new 10 --chat
```

## Tolerance Interpretation

| Metric | Typical Value | Interpretation |
|--------|--------------|----------------|
| max abs diff | 0.0–0.1 | Exact or near-exact match |
| max abs diff | 0.1–1.0 | Within ~1 ULP of bf16 |
| global rel diff | < 1e-2 | Excellent (bit-exact-ish) |
| global rel diff | 1e-2–2e-2 | Good (bf16 tolerance) |
| global rel diff | > 2e-2 | Investigate — possible bug |

## Troubleshooting

- **High diff in embeddings**: Check tokenizer encoding (add_special_tokens=True/False mismatch)
- **High diff in attention**: Verify RoPE theta/base_freq matches HF config
- **High diff in MLP**: GELU approximation difference; check if using `gelu_tanh` vs `gelu_pytorch_tanh`
- **High diff in layer 24+**: Check KV sharing map — shared layers must read K/V from the owner layer
- **Full pass fails but layers pass individually**: Check residual connections and layer_scalar application order
