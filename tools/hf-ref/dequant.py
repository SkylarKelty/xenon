"""Host-side NVFP4 dequant to bf16, matching xenon-kernels' fp4_dequant.cu.

NVFP4 = packed E2M1 values (2 per byte, low nibble = even index) with two-
level scaling:
  - per-16-element-block UE4M3 scale (F8_E4M3 stored non-negative)
  - per-tensor fp32 global scale (`weight_scale_2`)

We keep this separate from any framework so it's auditable against the
Rust kernel.
"""

from __future__ import annotations

import numpy as np
import torch

# E2M1 decode table (16 entries). Index bit 3 = sign.
_E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def ue4m3_to_f32(byte: np.ndarray) -> np.ndarray:
    """Decode UE4M3 (FP8 E4M3 with sign bit ignored) bytes to f32.

    Normal: (8 + mantissa) * 2^(exp - 10)
    Subnormal (exp==0): mantissa / 512
    """
    x = byte.astype(np.uint32) & 0x7F
    exp = (x >> 3) & 0xF
    man = x & 0x7
    normal = (8 + man).astype(np.float32) * np.power(2.0, exp.astype(np.float32) - 10.0)
    subnormal = man.astype(np.float32) * (1.0 / 512.0)
    return np.where(exp == 0, subnormal, normal)


def dequant_nvfp4(packed: np.ndarray, scales: np.ndarray, global_scale: float,
                  out_features: int, in_features: int) -> torch.Tensor:
    """Dequantize an NVFP4 linear weight to bf16 `[out, in]`.

    `packed` is `[out, in/2]` u8; `scales` is `[out, in/16]` u8 (UE4M3);
    `global_scale` is fp32.
    """
    assert packed.shape == (out_features, in_features // 2), packed.shape
    assert scales.shape == (out_features, in_features // 16), scales.shape
    assert in_features % 16 == 0

    # Unpack: even index = low nibble, odd index = high nibble.
    lo = (packed & 0xF).astype(np.uint8)
    hi = (packed >> 4).astype(np.uint8)
    interleaved = np.empty((out_features, in_features), dtype=np.uint8)
    interleaved[:, 0::2] = lo
    interleaved[:, 1::2] = hi
    fp4 = _E2M1[interleaved]  # [out, in] fp32

    # Broadcast block scales across 16 columns.
    bs = ue4m3_to_f32(scales)  # [out, in/16]
    bs_expanded = np.repeat(bs, 16, axis=1)  # [out, in]

    dequant = fp4 * bs_expanded * float(global_scale)
    return torch.from_numpy(dequant).to(torch.bfloat16)
