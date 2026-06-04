# AWQ-Diag

**A diagnostic toolkit for understanding *why* low-bit LLM quantization fails — not another quantizer.**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-cu128-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AWQ-Diag instruments a Hugging Face causal LM with PyTorch forward hooks, reproduces the
[AWQ](https://arxiv.org/abs/2306.00978) activation-aware saliency picture, and runs a per-layer
bit-width sweep (8 → 2 bit) to ask a focused question:

> **Can a cheap, single-layer activation statistic (kurtosis) predict where a model breaks when you push it from 4-bit to 3-bit?**

The honest answer here is **no — and that negative result is the point.** Activation outliers
clearly exist and are heavily concentrated in specific module families, but they raise the
*absolute* low-bit error floor without producing the localized 4→3 bit "phase transition" the
naive hypothesis predicted. The finding replicates across two model sizes.

> ⚠️ This is a **learning-oriented diagnostic project**, not a claim of a new quantization method.
> It is designed to demonstrate understanding of AWQ, activation outliers, and low-bit failure —
> and to honestly report a hypothesis that did not hold. See [`docs/research_gap_plan.md`](docs/research_gap_plan.md)
> for the full honest positioning.

---

## Key findings

Measured on `Qwen/Qwen2.5-1.5B` and replicated on `Qwen/Qwen2.5-0.5B`:

| Question | Metric | Result | Verdict |
|---|---|---|---|
| Does AWQ saliency concentrate? | top-1% channel importance share | up to **17.6%** (≈18× the uniform 1%) | ✅ hockey-stick reproduced |
| Do activation outliers exist? | max excess kurtosis | **κ ≈ 12** (`layers.1.mlp.down_proj`) | ✅ yes, very heavy-tailed |
| Are outliers module-specific? | mean kurtosis by family | **`down_proj` & `o_proj` ≫ everything else** | ✅ strong structure |
| Is there a 4→3 bit phase transition? | jump ratio = err(3b)/err(4b) | median **3.9×**, **0 layers > 5×** | ❌ smooth, no transition |
| Does kurtosis predict the **jump**? | Spearman ρ(κ, jump) | **−0.36** (1.5B), **−0.26** (0.5B) | ❌ negative — it does not |
| Does kurtosis predict the **error level**? | Spearman ρ(κ, 3-bit error) | **+0.55** (1.5B), **+0.51** (0.5B) | ✅ yes — different thing! |
| Does the cheap proxy track real output error? | Spearman ρ(proxy jump, output jump) | **+0.62** (1.5B), **+0.66** (0.5B) | ⚠️ partially |

**The one-line takeaway:** *kurtosis explains the **level** of low-bit error (outliers raise the
floor), but not the **sensitivity** to bit reduction (the 4→3 jump is uniform across layers).*
Low-bit failure is therefore unlikely to be a pure single-layer-statistics phenomenon — which
points the next investigation toward **inter-layer error propagation**.

---

## Figures

| AWQ saliency (hockey-stick) | Kurtosis by layer |
|---|---|
| ![saliency](figures/Qwen2.5-1.5B/saliency_curve.png) | ![kurtosis](figures/Qwen2.5-1.5B/kurtosis_by_layer.png) |

| Bit-width sweep & 4→3 jump | Kurtosis vs jump (the negative result) |
|---|---|
| ![sweep](figures/Qwen2.5-1.5B/bitwidth_error_sweep.png) | ![kvj](figures/Qwen2.5-1.5B/kurtosis_vs_jump_ratio.png) |

| Module-family breakdown | Proxy vs real output error |
|---|---|
| ![family](figures/Qwen2.5-1.5B/module_family.png) | ![pvo](figures/Qwen2.5-1.5B/proxy_vs_output_error.png) |

**Cross-model replication** (the negative result is not a 1.5B fluke):

![cross](figures/cross_model_jump_distribution.png)

The **module-family** figure is the clearest summary: `o_proj` and `down_proj` have *dramatically*
higher activation kurtosis than every other projection, yet the 4→3 bit jump ratio is essentially
**flat across all families**. Outliers ≠ quantization sensitivity.

---

## What it measures

For every `nn.Linear` inside the Transformer blocks, a forward hook collects **per-input-channel**:

| Statistic | Meaning |
|---|---|
| `channel_magnitude` | mean \|x\| — the AWQ saliency signal |
| `channel_variance` | spread of the activation distribution |
| `channel_max` | worst-case activation |
| `kurtosis` | excess kurtosis (0 = Gaussian; ≫0 = heavy-tailed / outliers) |
| `outlier_ratio` | fraction of \|x\| > 6σ |

Then, per layer, two complementary error notions across `{8,6,4,3,2}`-bit symmetric
per-output-channel weight quantization:

- **proxy error** — activation-weighted weight MSE (cheap; weights + activation magnitude only).
- **output error** — the *real* relative layer-output error `‖Wx − Ŵx‖ / ‖Wx‖` measured on the
  actual calibration activations (the "ground truth" the proxy is checked against).

See [`docs/report.md`](docs/report.md) for the full method, math, and interpretation.

---

## Quickstart

The environment is managed with **micromamba** (or conda/mamba).

```bash
# 1. Create the environment (PyTorch cu128 — adjust for your CUDA / CPU)
micromamba env create -f environment.yml
micromamba activate awq-diag

# 2. Run the diagnostic on one model (writes results/ + figures/)
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-1.5B
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-0.5B

# 3. Build the cross-model comparison
python scripts/compare_models.py results/diagnostic_*.json

# 4. (optional) run the unit tests
pytest
```

CPU-only / non-CUDA machines:

```bash
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-0.5B --device cpu --dtype float32
```

Outputs land in:

```
results/diagnostic_<model>.json      # full per-layer record + summary (see schema below)
figures/<model>/*.png                # 6 per-model figures
figures/cross_model_jump_distribution.png
results/cross_model_summary.md
```

---

## Repository layout

```
AWQ-Diag/
├── src/awq_diag/          # the package
│   ├── config.py          # DiagConfig — one object controls a run
│   ├── data.py            # calibration texts
│   ├── model_utils.py     # model loading + layer bookkeeping
│   ├── hooks.py           # ActivationCollector (the core: stats + output-error tracing)
│   ├── quant.py           # symmetric per-channel quant + error metrics
│   ├── analysis.py        # per-layer records, summary, module-family, correlations
│   ├── plotting.py        # the 6 figures
│   ├── pipeline.py        # end-to-end orchestration
│   └── cli.py             # `awq-diag` console entry
├── scripts/
│   ├── run_diagnostic.py  # run one model
│   └── compare_models.py  # cross-model summary
├── results/               # JSON outputs + cross-model table
├── figures/               # generated PNGs
├── notebooks/
│   └── awq_diagnostic.ipynb   # the original exploratory notebook (bilingual, educational)
├── docs/
│   ├── report.md          # full write-up (method → findings → limitations → next steps)
│   ├── note.md            # author's original project note (中文)
│   └── research_gap_plan.md   # honest positioning & research-gate analysis
├── tests/                 # pytest (quant core, CPU-only, no model download)
├── environment.yml        # micromamba/conda environment
├── requirements.txt       # pip fallback
└── pyproject.toml
```

The `.py` pipeline is the canonical, reproducible entry point and **exactly reproduces** the
original notebook's headline numbers (median 3.92× jump, ρ = −0.360, top-κ layer
`layers.1.mlp.down_proj` at κ ≈ 12).

---

## Output JSON schema (v2)

```jsonc
{
  "model": "Qwen/Qwen2.5-1.5B",
  "config":      { "bit_widths": [8,6,4,3,2], "outlier_sigma": 6.0, "seed": 0, ... },
  "model_info":  { "num_params": ..., "num_layers": 28, "num_linear_analyzed": 196, ... },
  "summary": {
    "proxy_jump_4to3":  { "min": .., "median": .., "mean": .., "max": .., "num_above_5x": 0 },
    "output_jump_4to3": { ... },
    "correlations": {
      "kurtosis_vs_proxy_jump_spearman":        [-0.360, 2.2e-07],
      "kurtosis_vs_3bit_proxy_error_spearman":  [ 0.553, 4.1e-17],
      "proxy_jump_vs_output_jump_spearman":     [ 0.623, 1.8e-22]
    },
    "module_family": { "down_proj": { "mean_kurtosis": .., "mean_proxy_jump": .., ... }, ... }
  },
  "layers": {
    "model.layers.0.self_attn.q_proj": {
      "module_type": "q_proj", "layer_idx": 0,
      "mean_kurtosis": .., "top1pct_importance_share": ..,
      "proxy_error":  { "8": .., "4": .., "3": .., "2": .. },
      "output_error": { ... },
      "proxy_jump_4to3": .., "output_jump_4to3": ..
    }
  },
  "jump_ratios": { ... },   // backward-compatible flat views from the original notebook
  "kurtosis":    { ... }
}
```

---

## Limitations & honest scope

- **One architecture family** (Qwen2.5, two sizes). Conclusions are not claimed to generalize to
  Llama/Gemma/Phi.
- **Proxy / output error are layer-local.** They are *not* end-task quality (perplexity / accuracy).
  The proxy↔output comparison is a first step toward closing that gap, but it stops at layer output.
- **Small calibration set** (4 paragraphs) — enough to characterize per-channel distributions, not
  to make distributional claims about rare events.
- **No comparison to a real AWQ scale search / GPTQ baseline.** This is a *diagnostic*, not a method.

## Next steps

The negative result sharpens the next hypothesis: low-bit failure is likely an **inter-layer
error-propagation** phenomenon rather than a single-layer one. Concretely:

1. **Single-layer quantization injection** — quantize one layer, measure final-logit KL / perplexity,
   and trace how local error grows (or is absorbed by the residual stream) downstream.
2. **Connect proxy error to model-level metrics** (logit KL, perplexity).
3. **Broader model families** to test whether `down_proj`/`o_proj` outlier concentration is universal.

A staged validation-gate plan for turning this into research is in
[`docs/research_gap_plan.md`](docs/research_gap_plan.md).

## References

- AWQ — [Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) (MLSys 2024 Best Paper)
- GPTQ — [Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- SmoothQuant — [arxiv 2211.10438](https://arxiv.org/abs/2211.10438)
- OmniQuant — [arxiv 2308.13137](https://arxiv.org/abs/2308.13137)

## License

MIT — see [LICENSE](LICENSE).
