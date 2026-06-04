# AWQ-Diag Report

*Understanding and visualizing AWQ on small Qwen2.5 models — is "activation importance" real, and does it matter?*

---

## 1. Motivation

**AWQ** ([Lin et al., MLSys 2024](https://arxiv.org/abs/2306.00978)) is the canonical
activation-aware weight-quantization method, built on one idea: not all weights matter equally — a
weight's importance scales with the activation it multiplies, so the few "salient" channels should be
protected rather than quantized bluntly.

That idea is usually presented as a couple of plots and an intuition. This project makes it
**concrete, visual, and testable** on a real model: reproduce AWQ's importance picture, visualize it
across the network, and — the key step — *implement* the protection and measure whether it actually
helps, and **where**. The guiding question:

> **What does AWQ's "activation importance" look like inside a real LLM, and does protecting the important channels really reduce quantization error?**

## 2. Background

- **Weight-only quantization.** Map FP weights to low-bit integers with a per-output-channel scale.
- **Activation-aware importance (AWQ).** `importance ∝ mean|weight| × mean|activation|`. A few
  channels dominate (the "hockey-stick" curve); protecting them recovers most of the quality.
- **Activation outliers.** Transformer activations are heavy-tailed — a handful of channels carry
  very large values (the channels AWQ cares about). Excess kurtosis quantifies this (0 = Gaussian).
- **AWQ's mechanism.** Scale up the salient input channels before quantizing (`W·diag(s)`) and undo
  it on the activation side (`x·diag(1/s)`); the product is unchanged in full precision, but the
  salient channels now survive quantization with less relative error.

## 3. Method

- **Models.** `Qwen/Qwen2.5-1.5B` (28 blocks, 196 block-internal Linear layers) and, for
  replication, `Qwen/Qwen2.5-0.5B` (24 blocks, 168 Linear layers).
- **Calibration.** 4 short, topic-diverse English paragraphs, truncated to 512 tokens.
- **Forward hooks.** A hook on every block-internal `nn.Linear` intercepts the input activation `x`
  and computes, per input channel: `channel_magnitude = mean|x|` (the AWQ saliency signal),
  `kurtosis = E[z⁴] − 3` (confirms the salient channels are genuine outliers), `outlier_ratio`,
  `variance`, `max`.
- **AWQ importance.** `importance[j] = mean_i |W[i,j]| · channel_magnitude[j]`; we report the share
  of total importance held by the top-1% of channels.
- **Bit-width sweep (context).** Symmetric per-output-channel RTN at `{8,6,4,3,2}` bits; we measure
  both an activation-weighted *proxy* error and the real *output* error `‖Wx − Ŵx‖²/‖Wx‖²`. This
  quantifies how large low-bit error is — i.e. why protection is worth doing.
- **AWQ scaling search (the key experiment).** Implement AWQ's mechanism: per-input-channel scaling
  `s = (mean|x|)^α` (mean-normalized), `Ŵ = quant(W·diag(s))·diag(1/s)`, with `α` grid-searched per
  layer/bit to minimize output error (`α = 0` is exactly RTN, so AWQ can never be worse). Report the
  **output-error reduction `RTN / AWQ`** per layer — i.e. how much protecting the salient channels
  helps.

Everything is driven by a single `DiagConfig` and a fixed seed, so a run is reproducible.

## 4. Experiments & results

Numbers are for `Qwen2.5-1.5B`; `Qwen2.5-0.5B` is noted where it replicates.

### 4.1 AWQ importance is concentrated (hockey-stick)

Per layer, the top 1% of channels hold up to **17.6%** of total importance (median 4.8%) — ≈18× the
uniform expectation. The premise "a few channels dominate" holds in a real model.

### 4.2 The salient channels are genuine outliers, concentrated in specific modules

Excess kurtosis ranges from ~0 to **~12** (`model.layers.1.mlp.down_proj`). By module family, the
outliers — and therefore the importance — are overwhelmingly concentrated in two projections:

| Module | Mean κ (1.5B) | Mean κ (0.5B) |
|---|---|---|
| q/k/v_proj | ~0.08 | ~0.10 |
| **o_proj** | **2.86** | **3.61** |
| gate/up_proj | ~0.09 | ~0.13 |
| **down_proj** | **4.31** | **3.54** |

`o_proj` (reads the attention output) and `down_proj` (reads the MLP intermediate) are the outlier
hot-spots — ≈35–55× the other projections — consistent with where activation outliers are known to
live.

### 4.3 Visualizing importance across the network

The 3D surface `importance_surface_*.png` (`x` = input channel, `y` = layer, `z` = `|W|·|x|`) shows
the classic spiky outlier-channel structure: a flat base with a few towering salient channels,
strongest in `down_proj`/`o_proj`. This is the picture that motivates activation-aware quantization,
drawn from a real model rather than a schematic.

### 4.4 Low-bit error is large (why protection is worth doing)

Median real output error grows quickly as bits drop — roughly ~4× per bit, the analytic scaling of
uniform-quantization MSE:

| bit | 8 | 6 | 4 | 3 | 2 |
|---|---|---|---|---|---|
| median output error | 0.0001 | 0.0014 | 0.022 | 0.079 | 0.268 |

So at 3–4 bit there is real error to be recovered — which sets up the main experiment.

### 4.5 Importance is *meaningful*: protecting the salient channels works (and lands where it should)

Implementing AWQ scaling and measuring the 3-bit output-error reduction `RTN / AWQ`:

| Metric | Qwen2.5-1.5B | Qwen2.5-0.5B |
|---|---|---|
| median reduction (all layers) | 1.17× | 1.15× |
| max reduction (single layer) | **25.9×** (`L2.mlp.down_proj`) | **28.9×** |
| mean reduction, `down_proj` | **2.31×** | ~2.3× |
| mean reduction, `o_proj` | **1.63×** | ~2.3× |
| mean reduction, other 5 families | 1.13–1.27× | ~1.2× |

The benefit lands **exactly** on the high-importance / high-outlier families (`down_proj`, `o_proj`)
and barely touches the low-importance ones; the **top-8 layers** by AWQ benefit are *all*
`down_proj`/`o_proj`. Protecting the channels that AWQ calls "important" demonstrably reduces error,
and does so precisely where the importance says it should — the empirical payoff that makes the
importance notion *meaningful*, not just intuitive.

> **An honest nuance.** The *per-layer* rank correlation ρ(kurtosis, reduction) ≈ 0. That is not a
> contradiction: ~5/7 of all layers are low-outlier families whose kurtosis is all ~0.08, forming a
> flat blob that dilutes the rank correlation. The signal is **categorical** (by module family) and
> very clean — which is exactly what AWQ predicts.

## 5. Findings (summary)

1. AWQ importance concentration is real and reproducible (top-1% share up to 17.6%).
2. The salient channels are genuine activation outliers (κ up to ~12), concentrated in
   `o_proj`/`down_proj` (≈35–55× the other projections).
3. Importance is visualized across channels × layers as the classic outlier-channel surface.
4. **Protecting the salient channels (AWQ scaling) measurably reduces quantization error — most
   exactly where importance concentrates** (`down_proj` 2.31×, `o_proj` 1.63×, others ~1.2×; up to
   25.9× for one layer).
5. All of the above **replicate** on Qwen2.5-0.5B.

## 6. Interpretation

AWQ's central claim — that activation-aware importance identifies the channels worth protecting — is
not just a heuristic on these models: importance is sharply concentrated in a small set of genuine
outlier channels, and protecting *those* channels is what produces the error reduction. The fact that
the benefit is concentrated in `o_proj`/`down_proj` (and absent in the near-Gaussian projections)
links the *diagnostic* (where outliers live) directly to the *mechanism* (where protection pays off).
The project therefore both **reproduces** AWQ's picture and **validates** its premise end-to-end:
look → it's concentrated; measure → protecting it helps; check where → exactly the outlier modules.

## 7. Limitations

- **Simplified quantizer.** Base is symmetric per-output-channel RTN; the AWQ pass adds the
  activation-aware scaling search. This captures AWQ's *mechanism* but is not the full deployed AWQ
  (group-wise + asymmetric zero-point + folded scales), and there is no GPTQ baseline — absolute error
  magnitudes are illustrative, not production numbers.
- **Layer-local error.** The AWQ benefit is measured at the layer output, not propagated to
  model-level quality (perplexity / accuracy).
- **Small calibration set** (4 paragraphs) and **kurtosis is a high-variance 4th-moment estimate**, so
  the per-layer statistics are indicative.
- **One architecture family** (Qwen2.5, two sizes) — no claim of generality to Llama/Gemma/Phi.

## 8. Next steps

1. **Group-wise + asymmetric AWQ** to move from a mechanism demo toward the real quantizer.
2. **Connect the layer-level AWQ benefit to model-level quality** (perplexity / logit KL): does
   protecting the important channels recover end-task accuracy, not just layer-output error?
3. **More model families** (Llama / Gemma / Phi) to test whether the `o_proj`/`down_proj` importance
   concentration is universal.

## References

- AWQ — [Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) (MLSys 2024 Best Paper)
- GPTQ — [arxiv 2210.17323](https://arxiv.org/abs/2210.17323) ·
  SmoothQuant — [arxiv 2211.10438](https://arxiv.org/abs/2211.10438) ·
  LLM.int8() — [arxiv 2208.07339](https://arxiv.org/abs/2208.07339)
- See [`research_gap_plan.md`](research_gap_plan.md) for the broader landscape (OmniQuant, QuaRot,
  SpinQuant, ParetoQ, KIVI, QServe, FLUTE).
