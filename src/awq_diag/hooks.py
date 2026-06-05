"""Forward-hook activation collector.

Hook 是 PyTorch 的機制：在任何 layer 掛一個 callback，該 layer 每次 forward 時
PyTorch 會把 input / output 傳進來，這就是我們「攔截」activation 的方式。

For every Linear inside a Transformer block we collect, per input channel:
    channel_magnitude  mean |x|        (AWQ saliency signal)
    channel_variance   var(x)
    channel_max        max |x|
    kurtosis           excess kurtosis (outlier heaviness, normal == 0)
    outlier_ratio      fraction of |x| > k*sigma

and, using the *real* activations, the output-error accumulators needed to
measure ``||W x - Wq x|| / ||W x||`` at each bit-width (output-error tracing).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiagConfig
from .model_utils import iter_block_linears
from .quant import awq_channel_scales, awq_dequant_weight, quantize_weight


class ActivationCollector:
    def __init__(self, model: nn.Module, cfg: DiagConfig):
        self.model = model
        self.cfg = cfg
        # running per-channel stats (kept on CPU as float32 tensors)
        self.stats: Dict[str, dict] = {}
        # output-error accumulators: name -> {"num": {bits: float}, "den": float}
        self.output_acc: Dict[str, dict] = {}
        # optional raw activations for a few named layers (qualitative only)
        self.raw: Dict[str, List[torch.Tensor]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, name: str):
        cfg = self.cfg

        def hook(module: nn.Linear, inputs, output):
            x = inputs[0].detach().float()          # [B, S, in_features]

            # ---- per-channel activation statistics (reduce over batch+seq) ----
            ch_mag = x.abs().mean(dim=(0, 1))
            ch_var = x.var(dim=(0, 1))
            ch_max = x.abs().amax(dim=(0, 1))

            mu = x.mean(dim=(0, 1), keepdim=True)
            sigma = x.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            z = (x - mu) / sigma
            kurt = z.pow(4).mean(dim=(0, 1)) - 3.0  # excess kurtosis

            thresh = cfg.outlier_sigma * sigma.squeeze(0)        # [1, in]
            outlier_ratio = (x.abs() > thresh.unsqueeze(0)).float().mean(dim=(0, 1))

            self._accumulate_stats(
                name, ch_mag, ch_var, ch_max, kurt, outlier_ratio, x.shape[-1]
            )

            # ---- output-error tracing on the real activations ----
            # bias cancels in (W x + b) - (Wq x + b), so we ignore it.
            W = module.weight.detach().float()
            y = F.linear(x, W)
            den = y.pow(2).sum().item()
            num = {}
            for bits in cfg.bit_widths:
                Wq = quantize_weight(W, bits)
                num[bits] = (y - F.linear(x, Wq)).pow(2).sum().item()
            self._accumulate_output(name, num, den)

            # ---- optional raw activation capture ----
            if name in cfg.save_raw_layers:
                self.raw.setdefault(name, []).append(x.cpu())

        return hook

    def _accumulate_stats(self, name, ch_mag, ch_var, ch_max, kurt, olr, hidden_dim):
        ch_mag, ch_var = ch_mag.cpu(), ch_var.cpu()
        ch_max, kurt, olr = ch_max.cpu(), kurt.cpu(), olr.cpu()
        if name in self.stats:
            old = self.stats[name]
            n = old["count"]
            self.stats[name] = {
                "channel_magnitude": (old["channel_magnitude"] * n + ch_mag) / (n + 1),
                "channel_variance": (old["channel_variance"] * n + ch_var) / (n + 1),
                "channel_max": torch.max(old["channel_max"], ch_max),
                "kurtosis": (old["kurtosis"] * n + kurt) / (n + 1),
                "outlier_ratio": (old["outlier_ratio"] * n + olr) / (n + 1),
                "hidden_dim": hidden_dim,
                "count": n + 1,
            }
        else:
            self.stats[name] = {
                "channel_magnitude": ch_mag,
                "channel_variance": ch_var,
                "channel_max": ch_max,
                "kurtosis": kurt,
                "outlier_ratio": olr,
                "hidden_dim": hidden_dim,
                "count": 1,
            }

    def _accumulate_output(self, name, num: Dict[int, float], den: float):
        if name not in self.output_acc:
            self.output_acc[name] = {"num": {b: 0.0 for b in num}, "den": 0.0}
        acc = self.output_acc[name]
        for b, v in num.items():
            acc["num"][b] += v
        acc["den"] += den

    # --------------------------------------------------------------- lifecycle
    def register(self) -> int:
        for name, module in iter_block_linears(self.model):
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return len(self._handles)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def run_calibration(self, cal_tokens, device) -> None:
        """Run forward passes over tokenized calibration samples."""
        for tokens in cal_tokens:
            self.model(tokens["input_ids"].to(device))


class AWQErrorCollector:
    """Second pass: measure the real layer-output error of AWQ-scaled quantization.

    For every Linear and every bit-width, grid-search the AWQ scaling exponent alpha
    that minimizes output error on the calibration set (alpha=0 is exactly plain RTN).
    The win of `min_alpha(err) vs err(alpha=0)` is how much activation-aware protection
    helps that layer — the thing AWQ-Diag previously *computed importance for but never
    actually used*.
    """

    def __init__(self, model, cfg: DiagConfig, act_scales: Dict[str, torch.Tensor],
                 alphas=(0.0, 1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6, 1.0),
                 quantizer=quantize_weight):
        self.model = model
        self.cfg = cfg
        self.act_scales = act_scales              # name -> mean|x| per input channel
        self.alphas = tuple(alphas)
        self.quantizer = quantizer                # base quantizer (per-channel RTN or group-wise)
        # acc[name][bits][alpha] = [sum_num, sum_den]
        self.acc: Dict[str, dict] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        sx = self.act_scales[name]
        bit_widths = self.cfg.bit_widths
        alphas = self.alphas

        @torch.no_grad()
        def hook(module: nn.Linear, inputs, output):
            x = inputs[0].detach().float()
            W = module.weight.detach().float()
            y = F.linear(x, W)
            den = y.pow(2).sum().item()
            sx_dev = sx.to(W.device)
            scale_cache = {a: awq_channel_scales(sx_dev, a) for a in alphas}
            entry = self.acc.setdefault(
                name, {b: {a: [0.0, 0.0] for a in alphas} for b in bit_widths}
            )
            for bits in bit_widths:
                for a in alphas:
                    Wq = awq_dequant_weight(W, bits, scale_cache[a], quantizer=self.quantizer)
                    num = (y - F.linear(x, Wq)).pow(2).sum().item()
                    entry[bits][a][0] += num
                    entry[bits][a][1] += den

        return hook

    def register(self) -> int:
        for name, module in iter_block_linears(self.model):
            if name in self.act_scales:
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return len(self._handles)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def run_calibration(self, cal_tokens, device) -> None:
        for tokens in cal_tokens:
            self.model(tokens["input_ids"].to(device))

    def finalize(self) -> Dict[str, dict]:
        """name -> {awq_output_error:{bits:err}, rtn_output_error:{bits:err}, best_alpha:{bits:a}}."""
        out: Dict[str, dict] = {}
        for name, per_bit in self.acc.items():
            awq_err, rtn_err, best_alpha = {}, {}, {}
            for bits, per_alpha in per_bit.items():
                errs = {a: num / max(den, 1e-12) for a, (num, den) in per_alpha.items()}
                a_star = min(errs, key=errs.get)
                awq_err[bits] = errs[a_star]
                rtn_err[bits] = errs[0.0]          # alpha=0 == plain RTN
                best_alpha[bits] = a_star
            out[name] = {
                "awq_output_error": awq_err,
                "rtn_output_error": rtn_err,
                "best_alpha": best_alpha,
            }
        return out

    def error_curves(self) -> Dict[str, Dict[int, Dict[float, float]]]:
        """Full relative-output-error response surface: name -> {bits -> {alpha -> err}}.

        Lets us study *how* error depends on the scaling exponent (not just the argmin),
        which is what the search-free-AWQ question needs.
        """
        out: Dict[str, Dict[int, Dict[float, float]]] = {}
        for name, per_bit in self.acc.items():
            out[name] = {
                bits: {a: num / max(den, 1e-12) for a, (num, den) in per_alpha.items()}
                for bits, per_alpha in per_bit.items()
            }
        return out
