#!/usr/bin/env python
"""Cross-model comparison of the AWQ benefit, from saved diagnostic JSONs.

Builds:
  * figures/cross_model_awq_reduction.png   (AWQ 3-bit error reduction by module family)
  * results/cross_model_summary.md          (markdown comparison table)

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
_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(paths: List[str]) -> Dict[str, dict]:
    out = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        out[data["model"].split("/")[-1]] = data
    return out


def plot_awq_reduction(results: Dict[str, dict], path: Path) -> None:
    """Grouped bars: AWQ 3-bit error reduction per module family, one group per model."""
    fams = [m for m in _ORDER if all(m in d["summary"]["module_family"] for d in results.values())]
    x = np.arange(len(fams))
    n = len(results)
    width = 0.8 / max(n, 1)
    colors = plt.cm.viridis(np.linspace(0.2, 0.75, n))

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, ((slug, d), c) in enumerate(zip(results.items(), colors)):
        fam = d["summary"]["module_family"]
        vals = [fam[m].get("mean_awq_reduction_3bit", float("nan")) for m in fams]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=slug, color=c)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="no benefit (RTN)")
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_ylabel("mean 3-bit output-error reduction (RTN / AWQ, ×)")
    ax.set_title("AWQ benefit by module family — across models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary_table(results: Dict[str, dict], path: Path) -> str:
    head = (
        "| Model | Params | Linear layers | Top-κ layer | "
        "AWQ reduction `down_proj` | `o_proj` | others | max |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for slug, d in results.items():
        s = d["summary"]
        fam = s["module_family"]
        others = np.mean([fam[m]["mean_awq_reduction_3bit"]
                          for m in fam if m not in ("down_proj", "o_proj")])
        params = d["model_info"]["num_params"] / 1e9
        tk = s["top_kurtosis_layer"]["name"].replace("model.layers.", "L")
        rows.append(
            f"| {slug} | {params:.2f}B | {d['model_info']['num_linear_analyzed']} | {tk} | "
            f"{fam['down_proj']['mean_awq_reduction_3bit']:.2f}x | "
            f"{fam['o_proj']['mean_awq_reduction_3bit']:.2f}x | "
            f"{others:.2f}x | {s['awq_reduction_3bit']['max']:.1f}x |"
        )
    table = head + "\n".join(rows) + "\n"
    path.write_text("# Cross-model AWQ-benefit summary\n\n"
                    "How much AWQ's activation-aware scaling reduces 3-bit output error, by family.\n\n"
                    + table)
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsons", nargs="+", help="diagnostic_*.json files")
    args = p.parse_args()

    results = load(args.jsons)
    if not results:
        sys.exit("no JSON files loaded")

    fig_path = REPO / "figures" / "cross_model_awq_reduction.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_awq_reduction(results, fig_path)

    table = write_summary_table(results, REPO / "results" / "cross_model_summary.md")
    print(table)
    print(f"wrote {fig_path}")
    print(f"wrote {REPO / 'results' / 'cross_model_summary.md'}")


if __name__ == "__main__":
    main()
