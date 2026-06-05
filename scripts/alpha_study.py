#!/usr/bin/env python
"""Search-free AWQ — can cheap activation statistics replace AWQ's per-layer grid search?

AWQ picks each layer's scaling exponent alpha (`s = act_scale^alpha`) by a grid search that
re-runs the block forward for every candidate — the bulk of calibration cost. This script asks
whether a cheap statistic predicts the optimal alpha, by comparing three strategies at one
bit-width:

    full search   per-layer argmin over a fine alpha grid      (the expensive baseline)
    global const  one alpha for every layer (the global argmin) (the dumb cheap baseline)
    predicted     alpha from a single stat, leave-one-out fit   (the proposed heuristic)

Outputs results/alpha_study_<model>.json and figures/<model>/alpha_study.png.

    python scripts/alpha_study.py --model Qwen/Qwen2.5-1.5B --bit 3
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from awq_diag.analysis import top1pct_importance_share  # noqa: E402
from awq_diag.config import DiagConfig  # noqa: E402
from awq_diag.data import get_calibration_texts  # noqa: E402
from awq_diag.hooks import ActivationCollector, AWQErrorCollector  # noqa: E402
from awq_diag.model_utils import iter_block_linears, load_model_and_tokenizer, set_seed  # noqa: E402


def _predictors(W, cm, s) -> dict:
    """Cheap per-layer stats that might predict the optimal alpha."""
    cm_np = cm.numpy()
    return {
        "kurtosis": float(s["kurtosis"].mean()),
        "outlier_ratio": float(s["outlier_ratio"].mean()),
        "top1pct_importance": top1pct_importance_share(W, cm),
        # how skewed is the activation magnitude (a few dominant channels)?
        "act_mag_skew": float(np.percentile(cm_np, 99) / max(np.percentile(cm_np, 50), 1e-9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--bit", type=int, default=3)
    ap.add_argument("--n-grid", type=int, default=21, help="alpha grid points in [0,1]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = DiagConfig(model_name=args.model, bit_widths=(args.bit,))
    alphas = [round(a, 4) for a in np.linspace(0.0, 1.0, args.n_grid)]

    print(f"[1/3] loading {args.model} ...")
    model, tok, device = load_model_and_tokenizer(cfg.model_name, cfg.dtype, cfg.device)
    cal = [tok(t, return_tensors="pt", truncation=True, max_length=cfg.max_calibration_length)
           for t in get_calibration_texts()]

    print("[2/3] collecting stats + fine alpha search ...")
    coll = ActivationCollector(model, cfg)
    coll.register()
    coll.run_calibration(cal, device)
    coll.remove()

    awq = AWQErrorCollector(model, cfg,
                            {n: coll.stats[n]["channel_magnitude"] for n in coll.stats},
                            alphas=alphas)
    awq.register()
    awq.run_calibration(cal, device)
    awq.remove()
    curves = awq.error_curves()                       # name -> {bit -> {alpha -> err}}

    # ---- per-layer table: best alpha, full curve, predictors ----
    names, rows = [], {}
    for name, module in iter_block_linears(model):
        if name not in curves:
            continue
        curve = curves[name][args.bit]
        W = module.weight.detach().cpu().float()
        preds = _predictors(W, coll.stats[name]["channel_magnitude"], coll.stats[name])
        rows[name] = {"curve": curve, "best_alpha": min(curve, key=curve.get),
                      "rtn": curve[0.0], **preds}
        names.append(name)

    a_grid = np.array(alphas)
    err = np.array([[rows[n]["curve"][a] for a in alphas] for n in names])   # [L, A]
    rtn = err[:, 0]                                                          # alpha=0
    best_alpha = np.array([rows[n]["best_alpha"] for n in names])
    err_search = err.min(axis=1)

    # ---- strategy: one global constant alpha (argmin of summed error) ----
    a_const_idx = int(err.sum(axis=0).argmin())
    a_const = alphas[a_const_idx]
    err_const = err[:, a_const_idx]

    # ---- predictor correlations + pick the strongest ----
    pred_names = ["kurtosis", "outlier_ratio", "top1pct_importance", "act_mag_skew"]
    P = {p: np.array([rows[n][p] for n in names]) for p in pred_names}
    corr = {p: float(sstats.spearmanr(P[p], best_alpha)[0]) for p in pred_names}
    best_pred = max(pred_names, key=lambda p: abs(corr[p]))

    # ---- strategy: predicted alpha, leave-one-out linear fit on the best predictor ----
    xfull = np.log10(np.clip(P[best_pred], 1e-6, None)) if best_pred == "act_mag_skew" else P[best_pred]
    err_pred = np.empty(len(names))
    pred_alpha = np.empty(len(names))
    for i in range(len(names)):
        mask = np.arange(len(names)) != i
        b, a0 = np.polyfit(xfull[mask], best_alpha[mask], 1)
        ahat = np.clip(a0 + b * xfull[i], 0.0, 1.0)
        j = int(np.abs(a_grid - ahat).argmin())            # snap to grid
        pred_alpha[i] = a_grid[j]
        err_pred[i] = err[i, j]

    def captured(e):  # fraction of the search's total error-reduction this strategy keeps
        return float((rtn - e).sum() / max((rtn - err_search).sum(), 1e-12))

    summary = {
        "model": args.model, "bit": args.bit, "n_layers": len(names), "n_grid": args.n_grid,
        "best_alpha_dist": {"min": float(best_alpha.min()), "median": float(np.median(best_alpha)),
                            "max": float(best_alpha.max()), "std": float(best_alpha.std())},
        "spearman_best_alpha_vs": corr,
        "best_predictor": best_pred,
        "global_const_alpha": a_const,
        "strategies": {
            "full_search":  {"mean_reduction": float(np.mean(rtn / err_search)), "captured": 1.0},
            "global_const": {"mean_reduction": float(np.mean(rtn / err_const)), "captured": captured(err_const)},
            "predicted":    {"mean_reduction": float(np.mean(rtn / err_pred)),  "captured": captured(err_pred)},
            "rtn":          {"mean_reduction": 1.0, "captured": 0.0},
        },
    }

    slug = args.model.split("/")[-1]
    out = Path(__file__).resolve().parents[1]
    (out / "results").mkdir(exist_ok=True)
    (out / "results" / f"alpha_study_{slug}.json").write_text(json.dumps(summary, indent=2))

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(xfull, best_alpha, s=28, alpha=0.7, color="steelblue", edgecolors="gray", linewidth=0.4)
    xl = f"log10({best_pred})" if best_pred == "act_mag_skew" else best_pred
    ax1.axhline(a_const, color="darkorange", linestyle="--", label=f"global const α={a_const:.2f}")
    ax1.set_xlabel(xl)
    ax1.set_ylabel(f"best α (at {args.bit}-bit)")
    ax1.set_title(f"Does a cheap stat predict the optimal α?\nSpearman ρ({best_pred}) = {corr[best_pred]:.2f}")
    ax1.legend()

    st = summary["strategies"]
    order = ["rtn", "global_const", "predicted", "full_search"]
    ax2.bar(range(4), [st[s]["captured"] * 100 for s in order],
            color=["gray", "darkorange", "seagreen", "steelblue"])
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["RTN", f"const α={a_const:.2f}", f"predicted({best_pred})", "full search"],
                        rotation=20, ha="right")
    ax2.set_ylabel("% of full-search error-reduction captured")
    ax2.set_title("Search-free strategies vs the expensive search")
    fig.tight_layout()
    (out / "figures" / slug).mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / slug / "alpha_study.png")
    plt.close(fig)

    # ---- print ----
    print("[3/3] done.\n")
    print(f"  best α: median {summary['best_alpha_dist']['median']:.2f} "
          f"[{best_alpha.min():.2f}, {best_alpha.max():.2f}]  std {best_alpha.std():.2f}")
    print(f"  global best constant α = {a_const}")
    print("  Spearman(best α, predictor):  " + "  ".join(f"{p}={corr[p]:+.2f}" for p in pred_names))
    print(f"  strongest predictor: {best_pred}\n")
    print(f"  {'strategy':<16} {'mean reduction':>14} {'% search captured':>18}")
    for s in order:
        print(f"  {s:<16} {st[s]['mean_reduction']:>13.2f}x {st[s]['captured']*100:>17.1f}%")


if __name__ == "__main__":
    main()
