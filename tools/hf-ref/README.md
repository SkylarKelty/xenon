# xenon-hf-ref

Reference harness: runs `cosmicproc/gemma-4-E4B-it-NVFP4` through HuggingFace
`transformers` on a fixed input and dumps intermediate activations. Used by
the Rust `xenon` implementation to diff its forward pass layer by layer.

## Why this exists

`transformers` main (5.x) supports Gemma 4, but NVIDIA's `modelopt` quant
plugin (which handles NVFP4 loading) pins to `transformers<5`. We resolve
the conflict by **dequantizing NVFP4 ourselves in Python** (mirroring the
CUDA kernel in `xenon-kernels/src/cu/fp4_dequant.cu`) and feeding a plain
bf16 model.

## Setup

Requires `uv` and ~2 GB free for the venv.

```bash
cd tools/hf-ref
uv sync
```

## Capture

```bash
MODEL_DIR=~/.cache/huggingface/hub/models--cosmicproc--gemma-4-E4B-it-NVFP4/snapshots/<HASH>
uv run python capture.py --model "$MODEL_DIR" --ids "2,108,107,1" --out activations
```

Output: `activations/activations.safetensors` + `metadata.json`.

CPU bf16 forward is ~1–2 min for T=4 on a recent laptop. GPU is possible if
you edit `.to(dtype)` to also move to CUDA, but CPU is plenty for this.

## What's captured

| Name | Shape | Notes |
| --- | --- | --- |
| `embed_tokens_scaled` | `[B, T, H]` | `embed_tokens(ids) * sqrt(H)` |
| `per_layer_inputs_raw` | `[B, T, L, Hl]` | `embed_tokens_per_layer(ids) * sqrt(Hl)` reshaped |
| `per_layer_inputs_projected` | `[B, T, L, Hl]` | `project_per_layer_inputs(inputs_embeds, raw)` |
| `layer_NN.attn_after_oproj` | `[B, T, H]` | output of `self_attn` (after `o_proj`) |
| `layer_NN.mlp_out` | `[B, T, H]` | output of MLP block |
| `layer_NN.final` | `[B, T, H]` | layer output (incl. PLE + `layer_scalar`) |
| `final_norm_out` | `[B, T, H]` | after the stack-final RMSNorm |
| `logits_pre_softcap` | `[B, T, V]` | `lm_head(final_norm_out)` |
| `logits_post_softcap` | `[B, T, V]` | `tanh(pre/30)*30` |

## Files

- `dequant.py` — NVFP4 → bf16 host-side dequant (≈30 lines).
- `capture.py` — load, run, hook, save.
- `pyproject.toml` — `uv`-managed env.
