"""Aggregate raw collector output into the diagnostic record + summary stats."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats

from .config import DiagConfig
from .hooks import ActivationCollector
from .model_utils import iter_block_linears, layer_idx_from_name, module_type_from_name
from .quant import jump_ratio, output_error_from_accumulators, proxy_error_sweep


def top1pct_importance_share(W: torch.Tensor, act_mag: torch.Tensor) -> float:
    """AWQ importance = mean_i |W[i,j]| * act_mag[j]; what % is in the top 1%?

    這就是 paper Figure 1 的 hockey-stick：top 1% channel 佔多少總 importance。
    """
    importance = (W.abs().mean(dim=0) * act_mag).cpu().numpy()
    order = np.sort(importance)[::-1]
    k = max(1, len(order) // 100)
    return float(order[:k].sum() / max(order.sum(), 1e-12) * 100.0)


def build_layer_records(
    model: nn.Module,
    collector: ActivationCollector,
    cfg: DiagConfig,
) -> Dict[str, dict]:
    """One record per analyzed Linear layer."""
    records: Dict[str, dict] = {}
    for name, module in iter_block_linears(model):
        if name not in collector.stats:
            continue
        s = collector.stats[name]
        W = module.weight.detach().cpu().float()
        act_mag = s["channel_magnitude"]

        proxy = proxy_error_sweep(W, act_mag, cfg.bit_widths)
        acc = collector.output_acc[name]
        output = output_error_from_accumulators(acc["num"], acc["den"])

        records[name] = {
            "module_type": module_type_from_name(name),
            "layer_idx": layer_idx_from_name(name),
            "hidden_dim": int(s["hidden_dim"]),
            "mean_kurtosis": float(s["kurtosis"].mean()),
            "max_kurtosis": float(s["kurtosis"].max()),
            "mean_outlier_ratio": float(s["outlier_ratio"].mean()),
            "max_outlier_ratio": float(s["outlier_ratio"].max()),
            "top1pct_importance_share": top1pct_importance_share(W, act_mag),
            "proxy_error": {str(b): proxy[b] for b in cfg.bit_widths},
            "output_error": {str(b): output[b] for b in cfg.bit_widths},
            "proxy_jump_4to3": jump_ratio(proxy, cfg.jump_hi_bit, cfg.jump_lo_bit),
            "output_jump_4to3": jump_ratio(output, cfg.jump_hi_bit, cfg.jump_lo_bit),
        }
    return records


def _dist(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "max": float(a.max()),
        "num_above_5x": int((a > 5.0).sum()),
    }


def _spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)


def build_summary(records: Dict[str, dict], cfg: DiagConfig) -> dict:
    names = list(records)
    kurt = [records[n]["mean_kurtosis"] for n in names]
    proxy_jump = [records[n]["proxy_jump_4to3"] for n in names]
    output_jump = [records[n]["output_jump_4to3"] for n in names]
    proxy_3bit = [records[n]["proxy_error"]["3"] for n in names]

    # module-family aggregation
    family: Dict[str, dict] = {}
    grouped = defaultdict(list)
    for n in names:
        grouped[records[n]["module_type"]].append(n)
    for mtype, members in grouped.items():
        family[mtype] = {
            "count": len(members),
            "mean_kurtosis": float(np.mean([records[m]["mean_kurtosis"] for m in members])),
            "mean_proxy_jump": float(np.mean([records[m]["proxy_jump_4to3"] for m in members])),
            "mean_output_jump": float(np.mean([records[m]["output_jump_4to3"] for m in members])),
            "mean_3bit_proxy_error": float(np.mean([records[m]["proxy_error"]["3"] for m in members])),
            "mean_3bit_output_error": float(np.mean([records[m]["output_error"]["3"] for m in members])),
        }

    top_kurt_name = max(names, key=lambda n: records[n]["mean_kurtosis"])

    return {
        "proxy_jump_4to3": _dist(proxy_jump),
        "output_jump_4to3": _dist(output_jump),
        "correlations": {
            "kurtosis_vs_proxy_jump_spearman": _spearman(kurt, proxy_jump),
            "kurtosis_vs_output_jump_spearman": _spearman(kurt, output_jump),
            "proxy_jump_vs_output_jump_spearman": _spearman(proxy_jump, output_jump),
            "kurtosis_vs_3bit_proxy_error_spearman": _spearman(kurt, proxy_3bit),
        },
        "top_kurtosis_layer": {
            "name": top_kurt_name,
            "mean_kurtosis": records[top_kurt_name]["mean_kurtosis"],
            "proxy_jump_4to3": records[top_kurt_name]["proxy_jump_4to3"],
        },
        "module_family": family,
    }
