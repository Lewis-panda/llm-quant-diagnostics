#!/usr/bin/env python
"""Cross-model comparison from saved diagnostic JSONs.

Builds:
  * figures/cross_model_jump_distribution.png  (overlaid 4→3 jump histograms)
  * results/cross_model_summary.md             (markdown comparison table)

Example
-------
    python scripts/compare_models.py results/diagnostic_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load(paths: List[str]) -> Dict[str, dict]:
    out = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        out[data["model"].split("/")[-1]] = data
    return out


def plot_jump_distributions(results: Dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(results)))
    for (slug, data), c in zip(results.items(), colors):
        jumps = np.array(list(data["jump_ratios"].values()))
        med = np.median(jumps)
        ax.hist(jumps, bins=30, alpha=0.5, color=c, edgecolor="white",
                label=f"{slug}  (median {med:.2f}x, n={len(jumps)})")
        ax.axvline(med, color=c, linestyle="--", alpha=0.9)
    ax.axvline(5, color="red", linestyle=":", label="phase-transition threshold (5x)")
    ax.set_xlabel("4→3 bit error jump ratio (proxy)")
    ax.set_ylabel("number of layers")
    ax.set_title("4→3 bit error jump across models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary_table(results: Dict[str, dict], path: Path) -> str:
    head = (
        "| Model | Params | Linear layers | Median 4→3 jump | Max jump | "
        "Layers >5x | κ-vs-jump ρ | Top-κ layer |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for slug, d in results.items():
        s = d["summary"]
        pj = s["proxy_jump_4to3"]
        rho = s["correlations"]["kurtosis_vs_proxy_jump_spearman"][0]
        params = d["model_info"]["num_params"] / 1e9
        tk = s["top_kurtosis_layer"]["name"].replace("model.layers.", "L")
        rows.append(
            f"| {slug} | {params:.2f}B | {d['model_info']['num_linear_analyzed']} | "
            f"{pj['median']:.2f}x | {pj['max']:.2f}x | {pj['num_above_5x']} | "
            f"{rho:.3f} | {tk} |"
        )
    table = head + "\n".join(rows) + "\n"
    path.write_text("# Cross-model diagnostic summary\n\n" + table)
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsons", nargs="+", help="diagnostic_*.json files")
    args = p.parse_args()

    results = load(args.jsons)
    if not results:
        sys.exit("no JSON files loaded")

    fig_path = REPO / "figures" / "cross_model_jump_distribution.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_jump_distributions(results, fig_path)

    table = write_summary_table(results, REPO / "results" / "cross_model_summary.md")
    print(table)
    print(f"wrote {fig_path}")
    print(f"wrote {REPO / 'results' / 'cross_model_summary.md'}")


if __name__ == "__main__":
    main()
