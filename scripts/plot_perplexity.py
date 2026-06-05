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
           ("const_awq", "const-α AWQ", "seagreen"), ("block_awq", "AWQ (block-level)", "darkorange"),
           ("search_awq", "per-Linear search", "steelblue")]


def main() -> None:
    files = sorted((REPO / "results").glob("perplexity_*.json"))
    data = [json.loads(f.read_text()) for f in files]
    # order configs by bit then model
    data.sort(key=lambda d: (d["bit"], d["model"]))
    labels = [f"{d['model'].split('/')[-1]}\n{d['bit']}-bit" for d in data]
    x = np.arange(len(data))
    n = len(_STRATS)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(10, 2.6 * len(data)), 6))
    for i, (key, name, color) in enumerate(_STRATS):
        vals = [d["ppl"].get(key, float("nan")) for d in data]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WikiText-2 perplexity (lower = better)")
    ax.set_title("Search-free AWQ — one global α matches/beats the official-style per-block AWQ\n"
                 "(group-wise asymmetric quant, Qwen2.5)")
    ax.legend()
    fig.tight_layout()
    out = REPO / "figures" / "perplexity_search_free_awq.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
