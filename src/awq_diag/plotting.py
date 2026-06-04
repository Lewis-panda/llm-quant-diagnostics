"""Figure generation. Each function saves one PNG and returns its path."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import numpy as np

from .config import DiagConfig

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

_RED = "indianred"
_BLUE = "steelblue"


def _short(name: str) -> str:
    return name.replace("model.layers.", "L").replace("model.", "")


def plot_saliency_curve(importance_curves: Dict[str, np.ndarray], path: Path) -> Path:
    """AWQ hockey-stick: sorted per-channel importance (log y) for demo layers."""
    layers = list(importance_curves)
    fig, axes = plt.subplots(len(layers), 1, figsize=(12, 2.6 * len(layers)))
    if len(layers) == 1:
        axes = [axes]
    for ax, name in zip(axes, layers):
        imp = np.sort(importance_curves[name])[::-1]
        ax.semilogy(imp, linewidth=1.0, color=_BLUE)
        k = max(1, len(imp) // 100)
        share = imp[:k].sum() / imp.sum() * 100
        ax.axvline(k, color="red", linestyle="--", alpha=0.7)
        ax.fill_between(range(k), imp[:k], alpha=0.15, color="red")
        ax.set_title(f"{_short(name)}  |  top 1% channels hold {share:.1f}% of importance",
                     fontsize=10)
        ax.set_xlabel("channel rank (sorted)")
        ax.set_ylabel("importance (log)")
    fig.suptitle("AWQ weight importance — hockey-stick curve", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_kurtosis_by_layer(records: Dict[str, dict], path: Path) -> Path:
    names = list(records)
    mean_k = np.array([records[n]["mean_kurtosis"] for n in names])
    max_k = np.array([records[n]["max_kurtosis"] for n in names])
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    thr = np.percentile(mean_k, 90)
    ax1.bar(x, mean_k, color=np.where(mean_k > thr, _RED, _BLUE), alpha=0.85)
    ax1.axhline(thr, color="red", linestyle=":", alpha=0.5)
    ax1.set_title("Mean excess kurtosis per Linear layer (red = top 10% outlier-heavy)")
    ax1.set_ylabel("excess kurtosis")

    thr2 = np.percentile(max_k, 90)
    ax2.bar(x, max_k, color=np.where(max_k > thr2, _RED, _BLUE), alpha=0.85)
    ax2.set_title("Max channel kurtosis per Linear layer (worst-case outlier channel)")
    ax2.set_ylabel("excess kurtosis")
    step = max(1, len(names) // 30)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([_short(names[i]) for i in x[::step]], fontsize=6, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bitwidth_sweep(records: Dict[str, dict], cfg: DiagConfig, path: Path) -> Path:
    bits = list(cfg.bit_widths)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for n in records:
        errs = [records[n]["proxy_error"][str(b)] for b in bits]
        jump = records[n]["proxy_jump_4to3"]
        if jump > 5:
            ax1.plot(bits, errs, color="red", alpha=0.7, linewidth=1.5)
        elif jump > 2:
            ax1.plot(bits, errs, color="orange", alpha=0.4, linewidth=1.0)
        else:
            ax1.plot(bits, errs, color="gray", alpha=0.25, linewidth=0.6)
    ax1.set_yscale("log")
    ax1.set_xticks(bits)
    ax1.set_xlabel("bit width")
    ax1.set_ylabel("weighted relative quant error (log)")
    ax1.set_title("Error vs bit-width per layer\nred = jump > 5x at 4→3 bit")
    ax1.axvspan(2.8, 3.2, alpha=0.1, color="red")

    jumps = np.array([records[n]["proxy_jump_4to3"] for n in records])
    ax2.hist(jumps, bins=40, color=_BLUE, alpha=0.75, edgecolor="white")
    ax2.axvline(np.median(jumps), color="orange", linestyle="--",
                label=f"median {np.median(jumps):.2f}x")
    ax2.axvline(5, color="red", linestyle="--", label="phase-transition threshold (5x)")
    ax2.set_xlabel("error jump ratio (3-bit / 4-bit)")
    ax2.set_ylabel("number of layers")
    ax2.set_title("Distribution of 4→3 bit error jump")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_kurtosis_vs_jump(records: Dict[str, dict], summary: dict, path: Path) -> Path:
    kurt = np.array([records[n]["mean_kurtosis"] for n in records])
    jump = np.array([records[n]["proxy_jump_4to3"] for n in records])
    rho, pval = summary["correlations"]["kurtosis_vs_proxy_jump_spearman"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    sc = ax1.scatter(kurt, jump, c=jump, cmap="RdYlBu_r", alpha=0.75,
                     edgecolors="gray", linewidth=0.5, s=40)
    fig.colorbar(sc, ax=ax1, label="jump ratio")
    ax1.set_xlabel("mean excess kurtosis")
    ax1.set_ylabel("4→3 bit error jump ratio")
    ax1.set_title("Can kurtosis predict the 4→3 bit jump?")
    ax1.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {pval:.2e}",
             transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

    med = np.median(kurt)
    low, high = jump[kurt <= med], jump[kurt > med]
    bp = ax2.boxplot([low, high], patch_artist=True,
                     tick_labels=["low kurtosis\n(≤ median)", "high kurtosis\n(> median)"])
    bp["boxes"][0].set_facecolor(_BLUE)
    bp["boxes"][1].set_facecolor(_RED)
    ax2.set_ylabel("4→3 bit error jump ratio")
    ax2.set_title("Jump ratio: high vs low kurtosis layers")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_module_family(summary: dict, path: Path) -> Path:
    fam = summary["module_family"]
    order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    order = [m for m in order if m in fam] + [m for m in fam if m not in order]
    x = np.arange(len(order))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(x, [fam[m]["mean_kurtosis"] for m in order], color=_RED, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=30, ha="right")
    ax1.set_ylabel("mean excess kurtosis")
    ax1.set_title("Activation outliers by module family")

    w = 0.4
    ax2.bar(x - w / 2, [fam[m]["mean_proxy_jump"] for m in order], w, label="proxy jump", color=_BLUE)
    ax2.bar(x + w / 2, [fam[m]["mean_output_jump"] for m in order], w, label="output jump", color="darkorange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=30, ha="right")
    ax2.set_ylabel("mean 4→3 bit jump ratio")
    ax2.set_title("Quantization sensitivity by module family")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_awq_reduction(records: Dict[str, dict], summary: dict, path: Path) -> Path:
    """Does activation-aware (AWQ) scaling help the high-kurtosis layers most?

    Left: per-layer 3-bit output-error reduction (RTN / AWQ, >1 = AWQ wins) vs kurtosis.
    Right: mean reduction by module family — the test of AWQ's core thesis.
    """
    names = list(records)
    kurt = np.array([records[n]["mean_kurtosis"] for n in names])
    red = np.array([records[n]["awq_reduction_3bit"] for n in names])
    mtypes = [records[n]["module_type"] for n in names]
    rho, pval = summary["correlations"]["kurtosis_vs_awq_reduction_spearman"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # highlight the two outlier families so the per-layer panel tells the story too
    palette = {"down_proj": _RED, "o_proj": "darkorange"}
    for fam_name, color, z in [("other", "lightsteelblue", 1),
                               ("o_proj", "darkorange", 2), ("down_proj", _RED, 3)]:
        mask = np.array([(m == fam_name) if fam_name != "other"
                         else (m not in palette) for m in mtypes])
        if mask.any():
            ax1.scatter(kurt[mask], red[mask], s=42, alpha=0.85, zorder=z,
                        color=color, edgecolors="gray", linewidth=0.4,
                        label=fam_name if fam_name != "other" else "other families")
    ax1.set_yscale("log")
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax1.set_xlabel("mean excess kurtosis")
    ax1.set_ylabel("AWQ 3-bit output-error reduction  (RTN / AWQ, ×, log)")
    ax1.set_title("Where does activation-aware scaling help?")
    ax1.text(0.05, 0.95,
             f"per-layer Spearman ρ = {rho:.3f} (p={pval:.1e})\n"
             f"→ weak per-layer, but concentrated\n   in the outlier families →",
             transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax1.legend(loc="lower right")

    fam = summary["module_family"]
    order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    order = [m for m in order if m in fam] + [m for m in fam if m not in order]
    vals = [fam[m]["mean_awq_reduction_3bit"] for m in order]
    colors = [_RED if v > np.median(vals) else _BLUE for v in vals]
    x = np.arange(len(order))
    ax2.bar(x, vals, color=colors, alpha=0.85)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=30, ha="right")
    ax2.set_ylabel("mean 3-bit error reduction (×)")
    ax2.set_title("AWQ benefit by module family")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_importance_surface_3d(
    matrix: np.ndarray,
    channel_index: np.ndarray,
    layer_index: np.ndarray,
    module_type: str,
    path: Path,
) -> Path:
    """The classic activation-aware-quant surface.

    x = input channel, y = layer depth, z = AWQ importance (|W| · |activation|).
    Spiky towers = the salient/outlier channels that AWQ protects. The same
    "outlier channel" picture appears in LLM.int8(), SmoothQuant and AWQ.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    X, Y = np.meshgrid(channel_index, layer_index)
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, matrix, cmap="viridis", linewidth=0, antialiased=True,
        rcount=matrix.shape[0], ccount=matrix.shape[1],
    )
    ax.set_xlabel("input channel", labelpad=10)
    ax.set_ylabel("layer", labelpad=10)
    ax.set_zlabel("AWQ importance  |W|·|x|", labelpad=8)
    ax.set_title(
        f"Activation-aware weight importance across depth — {module_type}\n"
        f"spiky towers = salient / outlier channels AWQ protects",
        fontsize=12,
    )
    ax.view_init(elev=32, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.02, label="importance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_proxy_vs_output(records: Dict[str, dict], summary: dict, path: Path) -> Path:
    proxy = np.array([records[n]["proxy_jump_4to3"] for n in records])
    output = np.array([records[n]["output_jump_4to3"] for n in records])
    rho, pval = summary["correlations"]["proxy_jump_vs_output_jump_spearman"]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(proxy, output, alpha=0.7, edgecolors="gray", linewidth=0.5, s=40, color=_BLUE)
    lim = [min(proxy.min(), output.min()), max(proxy.max(), output.max())]
    ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
    ax.set_xlabel("proxy 4→3 jump (act-weighted weight MSE)")
    ax.set_ylabel("output 4→3 jump (real activation output error)")
    ax.set_title("Does the cheap proxy track the real layer-output jump?")
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\np = {pval:.2e}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
