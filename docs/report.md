# AWQ-Diag Report

*A diagnostic study of layer-wise low-bit quantization sensitivity in small Qwen2.5 models.*

---

## 1. Motivation

Weight-only quantization is the cheapest way to shrink an LLM's memory footprint and speed up
inference, and **AWQ** ([Lin et al., MLSys 2024](https://arxiv.org/abs/2306.00978)) is the canonical
activation-aware method: not all weights matter equally, so the channels that multiply large
activations should be protected. AWQ is mature and widely deployed.

What is *less* well understood — and more interesting for a learning project — is **why** models
collapse when pushed below 4-bit. A common intuition is:

> "Layers with heavy activation outliers (high kurtosis) are the ones that break first; their
> quantization error should jump sharply when going from 4-bit to 3-bit."

This project builds a reproducible pipeline to test that intuition directly, and reports what it
actually found.

## 2. Background

- **Weight-only quantization.** Map FP weights to low-bit integers with a per-output-channel scale.
- **Activation-aware saliency (AWQ).** Channel importance ∝ mean |weight| × mean |activation|. A few
  channels dominate (the "hockey-stick" curve), so protecting them recovers most of the quality.
- **Activation outliers.** Transformer activations are heavy-tailed; a handful of channels carry
  very large values. Excess kurtosis quantifies this (0 = Gaussian, ≫0 = heavy tails).
- **Low-bit difficulty.** 3-bit and below is where uniform PTQ tends to break, motivating GPTQ,
  SmoothQuant, OmniQuant, QuaRot/SpinQuant, ParetoQ, etc.

## 3. Method

- **Models.** `Qwen/Qwen2.5-1.5B` (28 blocks, 196 block-internal Linear layers) and, for
  replication, `Qwen/Qwen2.5-0.5B` (24 blocks, 168 Linear layers).
- **Calibration.** 4 short, topic-diverse English paragraphs (ML / math / general prose),
  truncated to 512 tokens — enough to characterize per-channel distributions.
- **Forward hooks.** A hook on every block-internal `nn.Linear` intercepts the input activation
  `x` (shape `[1, seq, in]`) and computes, per input channel, over the batch+sequence axes:
  - `channel_magnitude = mean|x|` (AWQ saliency signal)
  - `channel_variance`, `channel_max`
  - `kurtosis = E[z⁴] − 3` with `z = (x − μ)/σ` (excess kurtosis)
  - `outlier_ratio = P(|x| > 6σ)`
- **AWQ importance.** `importance[j] = mean_i |W[i,j]| · channel_magnitude[j]`; we report the
  share of total importance held by the top-1% of channels.
- **Bit-width sweep.** Symmetric per-output-channel uniform quantization at `{8,6,4,3,2}` bits:
  `scale = max|W_row| / (2^{b−1})`, `Ŵ = clip(round(W/scale)) · scale`.
- **Two error metrics:**
  - **proxy error** (activation-weighted weight MSE):
    `Σ_j act_mag[j]·mean_i(W−Ŵ)² / Σ_j act_mag[j]·mean_i W²`.
  - **output error** (real layer-output relative error on calibration activations):
    `‖Wx − Ŵx‖² / ‖Wx‖²`, accumulated over calibration tokens (bias cancels in the difference).
- **Phase-transition metric.** `jump = error(3-bit) / error(4-bit)` per layer; a "phase transition"
  is flagged at `jump > 5×`.

Everything is driven by a single `DiagConfig` and a fixed seed, so a run is reproducible.

## 4. Experiments & results

All numbers below are for `Qwen2.5-1.5B`; `Qwen2.5-0.5B` values are in parentheses where they
replicate the trend.

### 4.1 AWQ saliency reproduces

The hockey-stick curve is clearly present: per-layer, the top 1% of channels hold up to **17.6%**
of total importance (median 4.8%) — i.e. up to ~18× the uniform expectation. AWQ's premise holds.

### 4.2 Outliers exist and are module-specific

Excess kurtosis ranges from ~0 to **~12** (`model.layers.1.mlp.down_proj`). Aggregated by module
family, outliers are *overwhelmingly* concentrated in two projections:

| Module | Mean κ (1.5B) | Mean κ (0.5B) | Mean 3-bit proxy error |
|---|---|---|---|
| q/k/v_proj | ~0.08 | ~0.10 | 0.087–0.100 |
| **o_proj** | **2.86** | **3.61** | 0.103 |
| gate/up_proj | ~0.09 | ~0.13 | 0.078–0.082 |
| **down_proj** | **4.31** | **3.54** | **0.135** |

`o_proj` (reads the attention output) and `down_proj` (reads the MLP intermediate) are the outlier
hot-spots — consistent with the literature on where activation outliers live.

### 4.3 No 4→3 bit phase transition

The 4→3 jump ratio is **smooth and tightly clustered**: median **3.92×**, range **[2.99×, 4.02×]**,
with **0 / 196 layers above the 5× threshold** (0.5B: median 3.85×, 0 / 168 above 5×). There is no
abrupt single-layer collapse in this setup.

### 4.4 Kurtosis does *not* predict the jump — but *does* predict the level

This is the central result, and it has two halves that are easy to conflate:

| Relationship | Spearman ρ (1.5B) | ρ (0.5B) | Reading |
|---|---|---|---|
| κ vs **4→3 jump ratio** | **−0.36** (p=2e-7) | −0.26 | ❌ higher-κ layers do *not* jump more — if anything, less |
| κ vs **absolute 3-bit error** | **+0.55** (p=4e-17) | +0.51 | ✅ higher-κ layers *do* have a higher error floor |

So outliers **raise the error floor uniformly** rather than creating a **localized phase
transition**. The naive hypothesis ("high kurtosis → sharp 4→3 collapse") is **false** here; the
subtler true statement is "high kurtosis → consistently higher low-bit error at every bit-width."

### 4.5 The cheap proxy partially tracks the real output error

The activation-weighted weight-MSE proxy correlates with the *real* layer-output jump at
**ρ = 0.62** (0.5B: 0.66) — useful as a screening signal, but it systematically over-estimates the
real output jump for many layers (the scatter sits below `y = x`). The proxy is a reasonable, cheap
stand-in but not a substitute for measuring actual output error.

### 4.6 Implementing AWQ scaling: where does activation-aware protection actually help?

The experiments above compute AWQ-style importance but quantize with plain RTN — i.e. they never
*use* the importance. So we close the loop and implement AWQ's actual mechanism: a per-input-channel
scaling `s = (mean|x|)^α` (mean-normalized) applied as `Ŵ = quant(W·diag(s))·diag(1/s)`, with `α`
grid-searched per layer/bit to minimize output error (`α = 0` is exactly RTN, so AWQ can never be
worse). We report the **3-bit output-error reduction** `RTN / AWQ` per layer.

| Metric | Qwen2.5-1.5B | Qwen2.5-0.5B |
|---|---|---|
| median reduction (all layers) | 1.17× | 1.15× |
| max reduction (single layer) | **25.9×** (`L2.mlp.down_proj`) | **28.9×** |
| mean reduction, `down_proj` | **2.31×** | ~2.3× |
| mean reduction, `o_proj` | **1.63×** | ~2.3× |
| mean reduction, other 5 families | 1.13–1.27× | ~1.2× |
| per-layer Spearman ρ(κ, reduction) | 0.04 (n.s.) | 0.06 (n.s.) |

The story is **categorical, not per-layer**: AWQ rescues *exactly* the two high-kurtosis families
(`down_proj`, `o_proj`) and barely touches the low-outlier families — a clean, **positive**
confirmation of AWQ's core thesis. The per-layer rank correlation with kurtosis is ~0 only because
~5/7 of all layers are low-outlier families that form a flat blob with no internal signal; the
top-8 layers by AWQ benefit are *all* `down_proj`/`o_proj`. This is also the answer to "why RTN?":
RTN is the fixed, assumption-free baseline, and AWQ is measured *relative to it*.

## 5. Findings (summary)

1. AWQ saliency concentration is real and reproducible.
2. Activation outliers are heavy (κ up to ~12) and concentrated in `o_proj` / `down_proj`.
3. The 4→3 bit transition is **smooth** — no layer exceeds the 5× phase-transition threshold.
   (And note: median jump ≈ 3.9× is essentially the analytic ~4×/bit scaling of uniform-quant MSE
   — see §7 — so this metric *cannot by construction* show a phase transition.)
4. **Kurtosis predicts the low-bit error *level* (ρ≈+0.55) but not the 4→3 *jump* (ρ≈−0.36).**
5. Both (3) and (4) **replicate** on a second model size (Qwen2.5-0.5B).
6. The cheap proxy moderately tracks the real layer-output error (ρ≈0.62).
7. Implemented AWQ scaling cuts 3-bit error ~2× on the outlier families (up to 25.9× for one
   layer), ~1.2× elsewhere — activation-aware protection helps *exactly* where the outliers are.

## 6. Interpretation

A single per-layer activation statistic captures the *magnitude* of a layer's quantization
difficulty (its error floor) but not its *bit-reduction dynamics*. Because the 4→3 jump is uniform
across layers — including across module families with 50× differences in kurtosis — low-bit failure
in these models is unlikely to be explained by single-layer activation distributions alone. The more
promising hypothesis is **inter-layer error propagation**: how local quantization error grows or is
absorbed by the residual stream and normalization as it flows downstream.

## 7. Limitations

- **The jump-ratio metric is dominated by an analytic constant.** Uniform-quantization MSE scales
  as `4^(−bits)`, so *every* layer's error multiplies by ≈4× per bit dropped, independent of its
  distribution. The measured per-bit ratios confirm this almost exactly (8→6→4→3→2 give
  4.00 / 3.99 / 3.92 / 3.65× per bit). This means the layer-local RTN proxy **cannot, by
  construction, reveal a phase transition** — a true transition would only appear in *model-level*
  metrics (perplexity / logit KL). The smooth "no >5× jump" result should be read in that light.
- One architecture family (Qwen2.5), two sizes — no claim of generality to Llama/Gemma/Phi.
- Proxy and output error are **layer-local**; neither is end-task quality (perplexity / accuracy).
- **Kurtosis is a 4th-moment estimate** from a small calibration set (4 paragraphs, a few hundred
  tokens/layer) — inherently high-variance, so the exact correlation values are indicative, not tight.
- **Simplified quantizer.** The base is symmetric per-output-channel RTN (a deliberate fixed probe).
  The AWQ pass adds activation-aware scaling but is not the full deployed AWQ (group-wise +
  asymmetric + folded scales), and there is no GPTQ baseline — absolute error magnitudes are not
  comparable to production AWQ/GPTQ.

## 8. Next steps

1. **Single-layer quantization injection** — quantize one layer, run the full forward, and measure
   final-logit KL / perplexity to test propagation directly.
2. **Connect proxy error to model-level metrics** (logit KL, perplexity) to validate or replace it.
3. **Broaden model families** to test whether `o_proj`/`down_proj` outlier concentration is universal.
4. **Toy adaptive precision** (extension only): keep the highest-error layers at 4-bit, drop the rest
   to 3-bit, and compare against random / depth heuristics under a fixed average-bit budget.

A staged validation-gate plan is in [`research_gap_plan.md`](research_gap_plan.md).

## References

See [`research_gap_plan.md`](research_gap_plan.md) for the full reference list (AWQ, GPTQ,
SmoothQuant, OmniQuant, QuaRot, SpinQuant, ParetoQ, KIVI, QServe, FLUTE).
