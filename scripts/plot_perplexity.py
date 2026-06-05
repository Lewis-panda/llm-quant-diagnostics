#!/usr/bin/env python
"""Plot the search-free AWQ perplexity comparison from results/perplexity_*.json.

    python scripts/plot_perplexity.py
writes figures/perplexity_search_free_awq.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
_STRATS = [("fp16", "fp16", "gray"), ("rtn", "RTN", "indianred"),
           ("const_awq", "const-α AWQ", "seagreen"), ("search_awq", "search-α AWQ", "steelblue")]


def main() -> None:
    files = sorted((REPO / "results").glob("perplexity_*.json"))
    data = [json.loads(f.read_text()) for f in files]
    # order configs by bit then model
    data.sort(key=lambda d: (d["bit"], d["model"]))
    labels = [f"{d['model'].split('/')[-1]}\n{d['bit']}-bit" for d in data]
    x = np.arange(len(data))
    w = 0.2

    fig, ax = plt.subplots(figsize=(max(9, 2.2 * len(data)), 6))
    for i, (key, name, color) in enumerate(_STRATS):
        vals = [d["ppl"][key] for d in data]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WikiText-2 perplexity (lower = better)")
    ax.set_title("Search-free AWQ — one global α matches/beats the per-layer search\n"
                 "(group-wise asymmetric quant, Qwen2.5)")
    ax.legend()
    fig.tight_layout()
    out = REPO / "figures" / "perplexity_search_free_awq.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
