"""Weight quantization and per-layer error metrics.

我們不做真正的部署量化，而是「模擬」不同 bit-width 下的量化誤差，用來診斷哪些
layer 對低 bit 最敏感。Two error notions are provided:

* **proxy error** — activation-weighted weight MSE. Cheap, needs only weights +
  per-channel activation magnitude (this is the AWQ-flavoured saliency view).
* **output error** — the *actual* relative error of the layer output
  ``||W x - Wq x|| / ||W x||`` measured on real calibration activations.

Comparing the two is one of the experiments: does the cheap proxy track the real
layer-output degradation?
"""
from __future__ import annotations

from typing import Dict, Sequence

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


def proxy_error_sweep(
    W: torch.Tensor,
    act_mag: torch.Tensor,
    bit_widths: Sequence[int],
) -> Dict[int, float]:
    """Activation-weighted relative weight-quantization error per bit-width.

    error = sum_j act_mag[j] * mean_i (W[i,j] - Wq[i,j])^2
            / sum_j act_mag[j] * mean_i W[i,j]^2
    把 activation magnitude 當權重，讓重要 channel 的誤差被放大，概念上更接近 AWQ。
    """
    act_mag = act_mag.to(W.dtype)
    baseline = (W.pow(2).mean(dim=0) * act_mag).sum().clamp(min=1e-10)
    out: Dict[int, float] = {}
    for bits in bit_widths:
        Wq = quantize_weight(W, bits)
        per_channel_mse = (W - Wq).pow(2).mean(dim=0)          # [in_features]
        weighted = (per_channel_mse * act_mag).sum()
        out[bits] = (weighted / baseline).item()
    return out


def output_error_from_accumulators(
    num: Dict[int, float],
    den: float,
) -> Dict[int, float]:
    """Turn accumulated Frobenius numerators/denominator into relative errors."""
    den = max(den, 1e-12)
    return {bits: num[bits] / den for bits in num}


def jump_ratio(errors: Dict[int, float], hi_bit: int, lo_bit: int) -> float:
    """error(lo_bit) / error(hi_bit) — e.g. 3-bit error / 4-bit error."""
    return errors[lo_bit] / max(errors[hi_bit], 1e-15)
