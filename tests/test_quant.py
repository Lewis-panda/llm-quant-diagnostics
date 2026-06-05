"""Unit tests for the quantization core (run with `pytest`, CPU-only, no model)."""
from __future__ import annotations

import torch

from awq_diag.quant import (
    awq_channel_scales,
    awq_dequant_weight,
    output_error_from_accumulators,
    quantize_weight,
)


def test_quantize_shape_and_range():
    W = torch.randn(16, 32)
    Wq = quantize_weight(W, bits=4)
    assert Wq.shape == W.shape
    # de-quantized values must stay within the original per-channel max
    assert (Wq.abs() <= W.abs().amax(dim=1, keepdim=True) + 1e-6).all()


def test_more_bits_means_less_error():
    torch.manual_seed(0)
    W = torch.randn(64, 128)
    err = {b: (W - quantize_weight(W, b)).pow(2).mean().item() for b in (8, 6, 4, 3, 2)}
    # monotonic: fewer bits -> larger reconstruction error
    assert err[8] < err[6] < err[4] < err[3] < err[2]


def test_awq_alpha0_is_plain_rtn():
    # alpha=0 -> unit scales -> AWQ reduces exactly to RTN
    W = torch.randn(32, 64)
    act = torch.rand(64) + 0.1
    s = awq_channel_scales(act, alpha=0.0)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    assert torch.allclose(awq_dequant_weight(W, 3, s), quantize_weight(W, 3), atol=1e-6)


def test_awq_helps_when_activations_are_skewed():
    # with one dominant (salient) channel, AWQ scaling should lower the
    # activation-weighted output error vs plain RTN.
    torch.manual_seed(0)
    W = torch.randn(48, 64)
    act = torch.rand(64) * 0.1
    act[7] = 20.0  # a single large/salient channel
    x = torch.randn(100, 64) * act  # activations dominated by channel 7

    y = x @ W.T
    rtn = (y - x @ quantize_weight(W, 3).T).pow(2).sum()
    awq_s = awq_channel_scales(act, alpha=1.0)
    awq = (y - x @ awq_dequant_weight(W, 3, awq_s).T).pow(2).sum()
    assert awq < rtn


def test_output_error_accumulator():
    out = output_error_from_accumulators({4: 2.0, 3: 8.0}, den=4.0)
    assert out[4] == 0.5 and out[3] == 2.0
