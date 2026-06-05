"""Weight quantization and the AWQ scaling used to measure activation-aware benefit.

* ``quantize_weight`` — symmetric per-output-channel RTN (the baseline).
* ``awq_channel_scales`` / ``awq_dequant_weight`` — AWQ's activation-aware protection:
  scale up the salient input channels before quantizing, then unscale.
* ``output_error_from_accumulators`` — real layer-output relative error
  ``||W x - Wq x|| / ||W x||`` from accumulated calibration statistics.
"""
from __future__ import annotations

from typing import Dict

import torch


def quantize_weight(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel uniform quantization, returned de-quantized.

    W shape is ``[out_features, in_features]`` (PyTorch Linear convention), so the
    per-output-channel scale is taken over ``dim=1``.
    """
    n_levels = 2 ** bits
    w_max = W.abs().amax(dim=1, keepdim=True)          # per output channel
    scale = (w_max / (n_levels // 2)).clamp(min=1e-10)
    q = (W / scale).round().clamp(-(n_levels // 2), n_levels // 2 - 1)
    return q * scale


def awq_channel_scales(act_scale: torch.Tensor, alpha: float) -> torch.Tensor:
    """AWQ per-input-channel scaling factors s_j = (mean|x_j|)^alpha, mean-normalized.

    這就是 AWQ 的 activation-aware 核心：把 activation 大（salient）的 channel 對應的
    weight 放大，量化後再縮回去，等價於用更多 bit 保護重要 channel。
    alpha=0 → s≡1（退化成純 RTN）；alpha=1 → 完全跟著 activation magnitude 縮放。
    """
    s = act_scale.clamp(min=1e-6).pow(alpha)
    return s / s.mean().clamp(min=1e-10)


def awq_dequant_weight(W: torch.Tensor, bits: int, scales: torch.Tensor) -> torch.Tensor:
    """Quantize W *after* scaling input channels by `scales`, then unscale.

    effective Ŵ[:,j] = dequant( quant( W[:,j] * s_j ) ) / s_j
    在 full precision 下 (x/s)(sW) = xW 不變；量化後，salient channel 因為被放大，
    相對量化誤差變小 → 這就是 AWQ 保護重要 channel 的機制。
    """
    s = scales.unsqueeze(0)                            # [1, in]
    return quantize_weight(W * s, bits) / s


def output_error_from_accumulators(
    num: Dict[int, float],
    den: float,
) -> Dict[int, float]:
    """Turn accumulated Frobenius numerators/denominator into relative errors."""
    den = max(den, 1e-12)
    return {bits: num[bits] / den for bits in num}
