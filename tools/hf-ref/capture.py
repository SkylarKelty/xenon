"""Run HF Gemma 4 on a fixed input and dump activations for xenon diffing.

Strategy: load the NVFP4 checkpoint *without* HF's quant-plugin path (modelopt
pins transformers<5 which collides with gemma4 support). We walk the
safetensors ourselves, dequantize NVFP4 triples on the fly via `dequant.py`,
and hand a fully-bf16 state dict to a vanilla text-only `Gemma4ForCausalLM`.
Hooks capture the intermediate tensors; results saved as one safetensors
file per (layer, point).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from dequant import dequant_nvfp4


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Read model.safetensors and dequant NVFP4 triples on the fly.

    Returns a flat `name -> tensor` dict in the shape the bf16 model expects.
    Drops modelopt-specific helper tensors (input_scale, weight_scale,
    weight_scale_2) once their owning triple is decoded.
    """
    path = model_dir / "model.safetensors"
    with safe_open(str(path), framework="pt") as f:
        names = f.keys()
        metadata = {n: f.get_slice(n).get_shape() for n in names}

        # Group by module prefix: things ending in .weight, .weight_scale,
        # .weight_scale_2 form a quantized triple if all three are present.
        quant_prefixes = set()
        bf16_names = set()
        for n in names:
            if n.endswith(".weight_scale"):
                quant_prefixes.add(n[: -len(".weight_scale")])
            elif n.endswith(".weight_scale_2") or n.endswith(".input_scale"):
                pass  # handled / skipped
            elif n.endswith(".weight"):
                prefix = n[: -len(".weight")]
                if f"{prefix}.weight_scale" in names:
                    continue  # part of a triple; decoded below
                bf16_names.add(n)
            else:
                # scalars / buffers like layer_scalar; keep as-is
                bf16_names.add(n)

        out: dict[str, torch.Tensor] = {}
        for n in bf16_names:
            out[n] = f.get_tensor(n)

        for prefix in sorted(quant_prefixes):
            w_name  = f"{prefix}.weight"
            s_name  = f"{prefix}.weight_scale"
            s2_name = f"{prefix}.weight_scale_2"
            if w_name not in names or s2_name not in names:
                raise RuntimeError(f"missing triple pieces for '{prefix}'")
            packed = f.get_tensor(w_name).numpy().astype(np.uint8)
            scales = f.get_tensor(s_name).view(torch.uint8).numpy().astype(np.uint8)
            s2 = f.get_tensor(s2_name).item()
            out_features, in_packed = packed.shape
            in_features = in_packed * 2
            dequant = dequant_nvfp4(packed, scales, s2, out_features, in_features)
            out[w_name] = dequant
    return out


def build_text_model(cfg_path: Path, dtype=torch.bfloat16):
    """Build a fresh bf16 Gemma 4 text model (no vision / audio towers)."""
    full = AutoConfig.from_pretrained(cfg_path)
    text_cfg = full.text_config
    # Strip the quantization config so HF doesn't try to wrap Linear layers.
    if hasattr(text_cfg, "quantization_config"):
        text_cfg.quantization_config = None
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    model = Gemma4ForCausalLM(text_cfg).to(dtype)
    model.eval()
    return model, text_cfg


def load_into_model(model, state_dict: dict[str, torch.Tensor]) -> None:
    """Strip the multimodal `model.language_model.*` prefix and load the rest."""
    remapped: dict[str, torch.Tensor] = {}
    prefix = "model.language_model."
    for k, v in state_dict.items():
        if k.startswith(prefix):
            remapped["model." + k[len(prefix):]] = v
        # Drop any tower / non-text keys silently.
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        for m in missing[:10]:
            print(f"  MISSING: {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    if unexpected:
        for u in unexpected[:10]:
            print(f"  UNEXPECTED: {u}")
        if len(unexpected) > 10:
            print(f"  ... and {len(unexpected) - 10} more")


def capture(model, input_ids: torch.Tensor, out_dir: Path, num_layers: int):
    """Run forward with hooks, save activations to safetensors.

    Captures:
      - inputs_embeds: embed_tokens(ids) * sqrt(hidden_size)
      - per_layer_inputs_projected: after project_per_layer_inputs()
      - for each layer i: `attn_after_oproj` (self_attn output), `mlp_out`
        (mlp output), `final` (decoder layer output including PLE and
        layer_scalar)
      - final_norm_out, logits_pre_softcap, logits_post_softcap
    """
    activations: dict[str, torch.Tensor] = {}
    hooks: list = []

    def tag(name: str, idx: int = 0):
        def _hook(_mod, _inputs, output):
            t = output[idx] if isinstance(output, tuple) else output
            activations[name] = t.detach().clone().to(torch.bfloat16)
        return _hook

    hooks.append(model.model.embed_tokens.register_forward_hook(tag("embed_tokens_scaled")))
    for i in range(num_layers):
        layer = model.model.layers[i]
        hooks.append(layer.self_attn.register_forward_hook(tag(f"layer_{i:02d}.attn_after_oproj")))
        hooks.append(layer.mlp.register_forward_hook(tag(f"layer_{i:02d}.mlp_out")))
        hooks.append(layer.register_forward_hook(tag(f"layer_{i:02d}.final")))
    hooks.append(model.model.norm.register_forward_hook(tag("final_norm_out")))
    hooks.append(model.lm_head.register_forward_hook(tag("logits_pre_softcap")))

    # Pre-compute PLE components so we can save them too. These duplicate what
    # model.forward() does internally, but there's no sub-module to hook for
    # `project_per_layer_inputs()` since it's a plain method.
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        per_layer = model.model.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_projected = model.model.project_per_layer_inputs(inputs_embeds, per_layer)
        activations["per_layer_inputs_raw"] = per_layer.detach().clone().to(torch.bfloat16)
        activations["per_layer_inputs_projected"] = per_layer_projected.detach().clone().to(torch.bfloat16)

        out = model(input_ids=input_ids, use_cache=False)
    activations["logits_post_softcap"] = out.logits.detach().clone().to(torch.bfloat16)

    for h in hooks:
        h.remove()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "activations.safetensors"
    save_file(activations, str(out_file))
    # Save a small JSON sidecar describing the input + config info so the
    # Rust consumer has a known ground truth it can assert on.
    sidecar = {
        "input_ids": input_ids.flatten().tolist(),
        "num_layers": num_layers,
        "tensor_names": sorted(activations.keys()),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\nSaved {len(activations)} tensors to {out_file}")
    for k in sorted(activations.keys()):
        print(f"  {k:48s}  {tuple(activations[k].shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model directory")
    ap.add_argument("--ids", default="2,108,107,1",
                    help="comma-separated token IDs to feed as input")
    ap.add_argument("--out", default="activations", help="output directory")
    args = ap.parse_args()

    model_dir = Path(args.model)
    out_dir = Path(args.out)

    print(f"building model from {model_dir}")
    model, text_cfg = build_text_model(model_dir)

    print("loading weights (dequant NVFP4 on the fly)...")
    sd = load_state_dict(model_dir)
    load_into_model(model, sd)

    ids = torch.tensor([[int(x) for x in args.ids.split(",")]], dtype=torch.long)
    print(f"input_ids shape: {tuple(ids.shape)}")

    capture(model, ids, out_dir, num_layers=text_cfg.num_hidden_layers)


if __name__ == "__main__":
    main()
