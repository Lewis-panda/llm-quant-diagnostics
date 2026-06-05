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
from .quant import output_error_from_accumulators


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
    awq_results: Dict[str, dict] | None = None,
) -> Dict[str, dict]:
    """One record per analyzed Linear layer."""
    records: Dict[str, dict] = {}
    for name, module in iter_block_linears(model):
        if name not in collector.stats:
            continue
        s = collector.stats[name]
        W = module.weight.detach().cpu().float()
        act_mag = s["channel_magnitude"]

        acc = collector.output_acc[name]
        output = output_error_from_accumulators(acc["num"], acc["den"])

        rec = {
            "module_type": module_type_from_name(name),
            "layer_idx": layer_idx_from_name(name),
            "hidden_dim": int(s["hidden_dim"]),
            "mean_kurtosis": float(s["kurtosis"].mean()),
            "max_kurtosis": float(s["kurtosis"].max()),
            "mean_outlier_ratio": float(s["outlier_ratio"].mean()),
            "top1pct_importance_share": top1pct_importance_share(W, act_mag),
            "output_error": {str(b): output[b] for b in cfg.bit_widths},
        }

        if awq_results and name in awq_results:
            a = awq_results[name]
            lo = cfg.jump_lo_bit                       # report AWQ benefit at 3-bit
            rtn3 = a["rtn_output_error"][lo]
            awq3 = a["awq_output_error"][lo]
            rec["awq_output_error"] = {str(b): a["awq_output_error"][b] for b in cfg.bit_widths}
            rec["awq_best_alpha"] = {str(b): a["best_alpha"][b] for b in cfg.bit_widths}
            # how many times AWQ shrinks the 3-bit output error vs plain RTN (>=1)
            rec["awq_reduction_3bit"] = rtn3 / max(awq3, 1e-12)

        records[name] = rec
    return records


def _dist(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())}


def _spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)


def build_summary(records: Dict[str, dict], cfg: DiagConfig) -> dict:
    names = list(records)
    kurt = [records[n]["mean_kurtosis"] for n in names]
    has_awq = all("awq_reduction_3bit" in records[n] for n in names)

    # module-family aggregation (where importance / outliers live, and where AWQ helps)
    family: Dict[str, dict] = {}
    grouped = defaultdict(list)
    for n in names:
        grouped[records[n]["module_type"]].append(n)
    for mtype, members in grouped.items():
        family[mtype] = {
            "count": len(members),
            "mean_kurtosis": float(np.mean([records[m]["mean_kurtosis"] for m in members])),
            "mean_top1pct_importance_share": float(
                np.mean([records[m]["top1pct_importance_share"] for m in members])),
            "mean_3bit_output_error": float(np.mean([records[m]["output_error"]["3"] for m in members])),
        }
        if has_awq:
            family[mtype]["mean_awq_reduction_3bit"] = float(
                np.mean([records[m]["awq_reduction_3bit"] for m in members]))

    top_kurt_name = max(names, key=lambda n: records[n]["mean_kurtosis"])
    bits = list(cfg.bit_widths)

    summary = {
        # how big is low-bit error (context for why activation-aware protection matters)
        "per_bit_median_output_error": {
            str(b): float(np.median([records[n]["output_error"][str(b)] for n in names])) for b in bits},
        "top_kurtosis_layer": {
            "name": top_kurt_name,
            "mean_kurtosis": records[top_kurt_name]["mean_kurtosis"],
        },
        "module_family": family,
        "correlations": {},
    }

    if has_awq:
        reduction = [records[n]["awq_reduction_3bit"] for n in names]
        summary["awq_reduction_3bit"] = _dist(reduction)
        summary["correlations"]["kurtosis_vs_awq_reduction_spearman"] = _spearman(kurt, reduction)
        summary["correlations"]["outlier_ratio_vs_awq_reduction_spearman"] = _spearman(
            [records[n]["mean_outlier_ratio"] for n in names], reduction)

    return summary
