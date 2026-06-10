#!/usr/bin/env python
"""Plot the "3-bit cliff": layer output error grows smoothly with fewer bits,
but model-level perplexity has a threshold — 4-bit RTN is fine, 3-bit breaks.

Reads results/diagnostic_*.json (layer error, symmetric per-channel RTN) and
results/perplexity_*.json (WikiText-2 ppl, group-wise asymmetric RTN).

    python scripts/plot_bit_error.py
writes figures/error_vs_bits.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
_COLORS = ["steelblue", "darkorange", "seagreen", "indianred"]


def main() -> None:
    diags = [json.loads(f.read_text()) for f in sorted((REPO / "results").glob("diagnostic_*.json"))]
    diags.sort(key=lambda d: d["model"])
    ppls = [json.loads(f.read_text()) for f in sorted((REPO / "results").glob("perplexity_*.json"))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # left: median layer output error vs bits — smooth exponential growth
    for d, color in zip(diags, _COLORS):
        err = d["summary"]["per_bit_median_output_error"]
        bits = sorted((int(b) for b in err), reverse=True)
        vals = [err[str(b)] for b in bits]
        name = d["model"].split("/")[-1]
        ax1.plot(bits, vals, "o-", color=color, label=name)
        for b, v in zip(bits, vals):
            ax1.annotate(f"{v:.3g}", (b, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color=color)
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_xticks([8, 6, 4, 3, 2])
    ax1.set_xlabel("weight bit-width")
    ax1.set_ylabel("median relative output error  ‖Wx − Ŵx‖/‖Wx‖  (log)")
    ax1.set_title("Layer error grows smoothly (≈3–4× per bit)\nRTN, symmetric per-channel")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # right: model-level perplexity — the threshold the layer view hides
    models = sorted({p["model"] for p in ppls})
    for model, color in zip(models, _COLORS):
        mine = sorted((p for p in ppls if p["model"] == model), key=lambda p: -p["bit"])
        bits = [p["bit"] for p in mine]
        rtn = [p["ppl"]["rtn"] for p in mine]
        fp16 = mine[0]["ppl"]["fp16"]
        name = model.split("/")[-1]
        ax2.plot(bits, rtn, "o-", color=color, label=f"{name} RTN")
        ax2.axhline(fp16, color=color, ls="--", lw=1, alpha=0.6)
        ax2.annotate(f"fp16 {fp16:.1f}", (bits[-1], fp16), textcoords="offset points",
                     xytext=(4, 4), fontsize=8, color=color)
        for b, v in zip(bits, rtn):
            ax2.annotate(f"{v:.1f}", (b, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color=color)
    ax2.invert_xaxis()
    ax2.set_xticks([4, 3])
    ax2.set_xlabel("weight bit-width")
    ax2.set_ylabel("WikiText-2 perplexity (lower = better)")
    ax2.set_title("…but model quality falls off a cliff at 3-bit\nRTN, group-wise asymmetric (g=128)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("The 3-bit cliff: layer error is smooth, model-level quality has a threshold", y=1.0)
    fig.tight_layout()
    out = REPO / "figures" / "error_vs_bits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
