"""End-to-end diagnostic run: model -> JSON record + figures."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import torch

from . import plotting
from .analysis import build_layer_records, build_summary
from .config import DiagConfig
from .data import get_calibration_texts
from .hooks import ActivationCollector, AWQErrorCollector
from .model_utils import (
    get_model_info,
    iter_block_linears,
    load_model_and_tokenizer,
    set_seed,
)


def _tokenize(tokenizer, texts, max_length):
    return [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        for t in texts
    ]


def _demo_importance_curves(model, collector, n_layer0: int = 4) -> Dict[str, np.ndarray]:
    """Per-channel importance for the first block's Linear layers (saliency fig)."""
    curves: Dict[str, np.ndarray] = {}
    for name, module in iter_block_linears(model):
        if "layers.0." in name and name in collector.stats:
            W = module.weight.detach().cpu().float()
            act_mag = collector.stats[name]["channel_magnitude"]
            curves[name] = (W.abs().mean(dim=0) * act_mag).numpy()
        if len(curves) >= n_layer0:
            break
    return curves


def _maxpool_cols(M: np.ndarray, max_cols: int):
    """Max-pool columns down to <= max_cols so outlier spikes survive on a 3D mesh."""
    L = M.shape[1]
    if L <= max_cols:
        return M, np.arange(L)
    bin_size = int(np.ceil(L / max_cols))
    nbins = int(np.ceil(L / bin_size))
    padded = np.zeros((M.shape[0], nbins * bin_size), dtype=M.dtype)
    padded[:, :L] = M
    pooled = padded.reshape(M.shape[0], nbins, bin_size).max(axis=2)
    return pooled, np.arange(nbins) * bin_size


def _importance_surface(model, collector, module_type: str, max_cols: int = 512):
    """Build (matrix[n_layers, n_cols], channel_index, layer_index) of AWQ importance
    for every layer of one module family, or None if the family is absent."""
    rows = []
    for name, module in iter_block_linears(model):
        if name.split(".")[-1] != module_type or name not in collector.stats:
            continue
        idx = int(name.split("layers.")[1].split(".")[0])
        W = module.weight.detach().cpu().float()
        imp = (W.abs().mean(dim=0) * collector.stats[name]["channel_magnitude"]).numpy()
        rows.append((idx, imp))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    matrix = np.stack([r[1] for r in rows])
    layer_index = np.array([r[0] for r in rows])
    matrix, channel_index = _maxpool_cols(matrix, max_cols)
    return matrix, channel_index, layer_index


def run_diagnostic(cfg: DiagConfig, make_figures: bool = True, verbose: bool = True) -> dict:
    set_seed(cfg.seed)

    if verbose:
        print(f"[1/6] Loading {cfg.model_name} ...")
    model, tokenizer, device = load_model_and_tokenizer(cfg.model_name, cfg.dtype, cfg.device)
    model_info = get_model_info(model)

    if verbose:
        print(f"      params={model_info['num_params']/1e9:.2f}B  layers={model_info['num_layers']}  device={device}")
        print(f"[2/6] Registering hooks + running {len(get_calibration_texts())} calibration passes ...")
    collector = ActivationCollector(model, cfg)
    n_hooks = collector.register()
    cal_tokens = _tokenize(tokenizer, get_calibration_texts(), cfg.max_calibration_length)
    collector.run_calibration(cal_tokens, device)
    collector.remove()
    if verbose:
        print(f"      {n_hooks} hooks fired; collected stats for {len(collector.stats)} layers")

    if verbose:
        print("[3/5] Second pass: AWQ scaling search (RTN vs activation-aware) ...")
    act_scales = {n: collector.stats[n]["channel_magnitude"] for n in collector.stats}
    awq_collector = AWQErrorCollector(model, cfg, act_scales)
    awq_collector.register()
    awq_collector.run_calibration(cal_tokens, device)
    awq_collector.remove()
    awq_results = awq_collector.finalize()

    if verbose:
        print("[4/6] Computing per-layer records + summary ...")
    records = build_layer_records(model, collector, cfg, awq_results)
    summary = build_summary(records, cfg)

    result = {
        "model": cfg.model_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "bit_widths": list(cfg.bit_widths),
            "num_calibration_samples": len(cal_tokens),
            "max_calibration_length": cfg.max_calibration_length,
            "outlier_sigma": cfg.outlier_sigma,
            "jump_window": [cfg.jump_hi_bit, cfg.jump_lo_bit],
            "seed": cfg.seed,
            "torch_version": torch.__version__,
            "python": platform.python_version(),
        },
        "model_info": {**model_info, "num_linear_analyzed": len(records)},
        "summary": summary,
        "layers": records,
        # --- backward-compatible flat views (kept from the original notebook) ---
        "jump_ratios": {n: records[n]["proxy_jump_4to3"] for n in records},
        "kurtosis": {n: records[n]["mean_kurtosis"] for n in records},
    }

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.json_path.write_text(json.dumps(result, indent=2))
    if verbose:
        print(f"      wrote {cfg.json_path}")

    if make_figures:
        if verbose:
            print("[5/6] Rendering figures ...")
        d = cfg.model_figures_dir
        d.mkdir(parents=True, exist_ok=True)
        plotting.plot_saliency_curve(_demo_importance_curves(model, collector), d / "saliency_curve.png")
        plotting.plot_kurtosis_by_layer(records, d / "kurtosis_by_layer.png")
        plotting.plot_bitwidth_sweep(records, cfg, d / "bitwidth_error_sweep.png")
        plotting.plot_kurtosis_vs_jump(records, summary, d / "kurtosis_vs_jump_ratio.png")
        plotting.plot_module_family(summary, d / "module_family.png")
        plotting.plot_proxy_vs_output(records, summary, d / "proxy_vs_output_error.png")
        plotting.plot_awq_reduction(records, summary, d / "awq_reduction.png")
        n_fig = 7
        for mtype in ("down_proj", "o_proj"):  # the two highest-kurtosis families
            surf = _importance_surface(model, collector, mtype)
            if surf is not None:
                plotting.plot_importance_surface_3d(
                    *surf, mtype, d / f"importance_surface_{mtype}.png"
                )
                n_fig += 1
        if verbose:
            print(f"      wrote {n_fig} figures to {d}")

    if verbose:
        _print_headline(result)
        print("[6/6] Done.")
    # free GPU memory so multi-model runs don't accumulate
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _print_headline(result: dict) -> None:
    s = result["summary"]
    pj = s["proxy_jump_4to3"]
    rho, p = s["correlations"]["kurtosis_vs_proxy_jump_spearman"]
    tk = s["top_kurtosis_layer"]
    print("\n  ── headline ─────────────────────────────────────────")
    print(f"  4→3 proxy jump: min {pj['min']:.2f}x  median {pj['median']:.2f}x  "
          f"max {pj['max']:.2f}x  (>5x: {pj['num_above_5x']})")
    print(f"  kurtosis vs jump Spearman ρ = {rho:.3f} (p={p:.1e})")
    print(f"  highest-kurtosis layer: {tk['name']} (κ={tk['mean_kurtosis']:.2f})")
    if "awq_reduction_3bit" in s:
        ar = s["awq_reduction_3bit"]
        fam = s["module_family"]
        outlier = [fam[m]["mean_awq_reduction_3bit"] for m in ("down_proj", "o_proj") if m in fam]
        other = [fam[m]["mean_awq_reduction_3bit"] for m in fam if m not in ("down_proj", "o_proj")]
        print(f"  AWQ 3-bit error reduction: median {ar['median']:.2f}x  max {ar['max']:.2f}x")
        print(f"  AWQ helps outlier families most: down_proj/o_proj ~{np.mean(outlier):.2f}x "
              f"vs others ~{np.mean(other):.2f}x")
    print("  ─────────────────────────────────────────────────────")
