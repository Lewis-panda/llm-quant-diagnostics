"""Unit tests for the quantization core (run with `pytest`, CPU-only, no model)."""
from __future__ import annotations

import torch

from awq_diag.quant import (
    jump_ratio,
    output_error_from_accumulators,
    proxy_error_sweep,
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


def test_proxy_error_sweep_monotonic():
    torch.manual_seed(0)
    W = torch.randn(64, 128)
    act = torch.rand(128)
    errs = proxy_error_sweep(W, act, (8, 6, 4, 3, 2))
    assert errs[8] < errs[4] < errs[2]


def test_jump_ratio():
    errs = {4: 0.01, 3: 0.04}
    assert abs(jump_ratio(errs, 4, 3) - 4.0) < 1e-9


def test_output_error_accumulator():
    out = output_error_from_accumulators({4: 2.0, 3: 8.0}, den=4.0)
    assert out[4] == 0.5 and out[3] == 2.0
