# AWQ-Diag Research Gap Plan

這份 plan 是針對 `AWQ-Diag/` 目前內容寫的研究定位建議。

目前資料夾內已有：

- `note.md`：AWQ-Diag project note
- `awq_diagnostic.ipynb`：使用 Qwen2.5-1.5B 做 activation / kurtosis / quantization error 診斷
- `diagnostic_Qwen2.5-1.5B.json`：196 個 Linear layer 的 jump ratio 與 kurtosis 結果

核心判斷：

> 不建議把這個 project 包裝成「提出新的 AWQ quantization method」。AWQ / GPTQ / SmoothQuant / OmniQuant / rotation-based PTQ / low-bit QAT / serving-system co-design 都已經非常擁擠。以目前 AWQ-Diag 的規模，直接宣稱 research gap 會很勉強。

更合理的定位是：

> 把 AWQ-Diag 轉成一個 learning-oriented diagnostic project：用可重複的實驗流程理解 AWQ、activation-aware saliency、low-bit failure、layer sensitivity，以及為什麼 single-layer statistics 不足以解釋 quantization collapse。

如果之後真的要往 research 走，應該先通過明確的 validation gates，而不是現在直接宣稱 novelty。

---

## 1. Honest Saturation Assessment

### 1.1 AWQ 本身已經不是新 gap

AWQ 的核心主張已經很完整：它發現不是所有 weights 一樣重要，應該根據 activation distribution 找 salient channels，並用等價縮放保護重要 channels，而不是直接做硬體不友善的 mixed precision。AWQ 也包含 TinyChat deployment system，並已經是 MLSys 2024 Best Paper。

這代表：

- 「我用 activation magnitude 找 important weights」不是新貢獻。
- 「我重現 AWQ saliency curve」是很好的學習成果，但不是 research gap。
- 「我做 4-bit weight-only quantization」本身也不是新方向。

### 1.2 低 bit quantization 已經有很多強 baseline

相關方向已經包含：

- GPTQ：one-shot second-order weight quantization，已處理 3/4-bit，甚至討論 2-bit / ternary regime。
- SmoothQuant：把 activation quantization difficulty 遷移到 weights，讓 W8A8 PTQ 可行。
- OmniQuant：用 learnable clipping 和 learnable equivalent transformation 處理 low-bit PTQ。
- QuaRot / SpinQuant：用 rotation 去除 outliers，處理 weights、activations、KV cache 的 4-bit quantization。
- ParetoQ：直接研究 1-bit、1.58-bit、2-bit、3-bit、4-bit 的 scaling law 和 low-bit transition。
- KIVI / QServe：把戰場推到 KV cache quantization 和 serving-system co-design。
- FLUTE：處理 3-bit / LUT quantization 的高效 kernel 問題。

所以如果題目只是：

```text
Can activation statistics predict quantization error?
Can we allocate precision layer-wise?
Can we diagnose low-bit quantization failure?
```

這些問題不是不能做，但必須非常小心，因為相鄰工作很多。要成為研究貢獻，需要更強的 empirical evidence、更多模型、真實 downstream evaluation，以及和強 baseline 比較。

### 1.3 本地結果不支持原始 strong hypothesis

目前 `diagnostic_Qwen2.5-1.5B.json` 顯示：

- Linear layers 數量：196
- 4-bit 到 3-bit jump ratio：
  - min：約 2.99x
  - max：約 4.02x
  - 沒有超過 5x 的 layer
- 最高 kurtosis：
  - `model.layers.1.mlp.down_proj`
  - mean kurtosis 約 11.99
- `note.md` 已記錄：
  - 沒有觀察到劇烈 phase transition
  - kurtosis 和 jump ratio 的 Spearman correlation 約為 -0.36

這意味著原本假設：

```text
High activation kurtosis predicts 4-bit -> 3-bit phase transition.
```

目前是不成立的。這不是壞事，但它讓 project 更適合定位成 diagnostic / learning project，而不是 claim 一個已經驗證的 predictor。

---

## 2. Research Gap Candidates

### Candidate A: Kurtosis as a lightweight phase-transition predictor

Verdict: 不建議作為主題。

原因：

- 本地結果已經反證：最高 kurtosis layer 並不是最高 jump ratio。
- Spearman correlation 為負，不支持正向 predictor 假設。
- 單一模型、少量 calibration text、layer-wise proxy error 還不足以支撐嚴格結論。

可以保留為 negative finding：

```text
Per-layer kurtosis captures activation outliers, but it does not reliably predict 4-to-3 bit activation-weighted quantization error jumps in Qwen2.5-1.5B under this diagnostic setup.
```

這是很好的學習型 project 結果。

### Candidate B: Inter-layer error propagation diagnostic

Verdict: 有一點研究潛力，但目前還太早。

目前 note 裡最有價值的 observation 是：

```text
低 bit failure 可能不是 single-layer activation distribution 的問題，而是 inter-layer error propagation、residual stream、attention / MLP interaction 的問題。
```

這比 kurtosis predictor 更有意思，但要變成 research，需要做新的實驗：

- 單獨量化某一層，觀察後續 layers hidden states 的 error growth。
- 分別量化 attention projections、MLP projections、down_proj，追蹤 residual stream error。
- 比較 layer-local weight error、layer output error、final logits error、perplexity drop。
- 檢查 error 是否在特定 block 後被放大或被 residual path 吸收。

如果能做到，可能的研究問題是：

```text
Can inter-layer error propagation explain low-bit quantization sensitivity better than single-layer activation outlier statistics?
```

但這需要比目前 notebook 多很多實驗。

### Candidate C: Cross-model quantization sensitivity map

Verdict: 適合學習型 empirical project；作為 research gap 偏弱。

可以把同一套 diagnostic pipeline 跑在：

- Qwen2.5-0.5B / 1.5B
- Llama-3.2-1B / 3B if accessible
- Phi / Gemma small models if hardware permits

目標不是宣稱新方法，而是建立比較：

- 不同 architecture 的 sensitive modules 是否一致？
- MLP down_proj 是否普遍 high-kurtosis？
- 3-bit error 是否在某些 model family 更不穩？
- Qwen 的 GQA / attention projection shape 是否影響 layer sensitivity？

這個方向適合寫成：

```text
A diagnostic study of layer-wise low-bit quantization sensitivity across small open LLMs.
```

但它比較像 empirical report，不一定有足夠 novelty。

### Candidate D: Diagnostic-guided adaptive precision allocation

Verdict: 有 project 價值，但 research 風險高。

做法：

- 根據 diagnostic score 選出 top-k sensitive layers 保留 4-bit。
- 其他 layers 使用 3-bit 或 2-bit。
- 比較：
  - uniform 4-bit
  - uniform 3-bit
  - random mixed precision
  - kurtosis-guided mixed precision
  - activation-weighted-error-guided mixed precision

問題：

- AWQ 已經是 activation-aware。
- mixed precision allocation 本身也不是新概念。
- 如果沒有真實 perplexity / benchmark evaluation，只看 proxy error 不夠。
- 如果沒有 kernel / storage accounting，只說 bit average 也不夠完整。

適合當 Level 3 extension，不適合當目前主線。

### Candidate E: AWQ-Diag as a learning toolkit

Verdict: 最推薦。

這個定位最誠實，也最可完成：

```text
AWQ-Diag is a learning-oriented diagnostic toolkit for understanding why low-bit LLM quantization is difficult. It reproduces AWQ-style activation-aware saliency, measures per-layer activation outliers, simulates bit-width sweeps, and shows that simple single-layer statistics such as kurtosis are insufficient to explain low-bit quantization error jumps.
```

貢獻不是新 SOTA，而是：

- 讀懂 AWQ 的 core mechanism。
- 用 PyTorch hooks 抓 activation。
- 建立 per-layer quantization diagnostic pipeline。
- 做出 negative result。
- 從 negative result 推導下一步研究問題。

這很適合課程、portfolio、research training 或 advisor discussion。

---

## 3. Recommended Project Positioning

### Project Title

```text
AWQ-Diag: A Diagnostic Learning Toolkit for Understanding Low-Bit LLM Quantization Failure
```

### One-sentence Summary

```text
AWQ-Diag uses activation hooks and layer-wise bit-width sweeps to study how activation outliers, AWQ-style saliency, and activation-weighted quantization error vary across Transformer layers, showing that single-layer kurtosis alone does not explain 4-bit to 3-bit quantization sensitivity in Qwen2.5-1.5B.
```

### Project Type

Learning-oriented empirical diagnostic project.

### Not a Claim

不要宣稱：

- 我提出新的 AWQ。
- 我找到 phase transition predictor。
- 我能改善 SOTA quantization。
- kurtosis 可以預測 quantization collapse。

### Valid Claim

可以宣稱：

- 我重現並理解 AWQ-style activation-aware saliency。
- 我建立了 layer-wise diagnostic pipeline。
- 我發現 Qwen2.5-1.5B 中 activation outliers 存在但不直接對應 4-to-3 bit error jump。
- 這支持下一步研究 inter-layer error propagation，而不是只看 single-layer statistics。

---

## 4. Research Validation Gates

如果未來想把 AWQ-Diag 往 research paper 方向推，必須先通過以下 gates。

### Gate 1: Multi-model replication

最低要求：

- 至少 3 個模型。
- 至少 2 個 model families。
- 每個模型使用相同 calibration protocol。
- 每個模型輸出相同 diagnostic JSON schema。

要回答：

```text
Kurtosis 和 jump ratio 的弱相關是否只發生在 Qwen2.5-1.5B，還是跨模型都成立？
```

通過標準：

- 結果跨模型穩定。
- 或者不同 architecture 顯示明確差異，而且能解釋。

### Gate 2: From proxy error to model behavior

目前的 error 是 activation-weighted weight quantization proxy，不是真正 downstream quality。

需要加入：

- layer output MSE
- final logits KL divergence
- perplexity
- small downstream task score
- generation quality spot check

要回答：

```text
Layer-wise proxy error 是否真的對應 model-level degradation？
```

通過標準：

- proxy error 能預測至少一種 model-level metric。
- 或者證明 proxy error 不足，並提出更好的 diagnostic metric。

### Gate 3: Inter-layer error propagation

需要做 injection / ablation：

- 只量化單一 layer。
- 只量化 attention。
- 只量化 MLP。
- 只量化 down_proj。
- 追蹤 hidden state error 如何往後傳。

要回答：

```text
哪些 layers 的 local quantization error 會被後續 layers 放大？
哪些 layers 的 local error 會被 residual stream 或 normalization 吸收？
```

通過標準：

- 找到比 kurtosis 更有解釋力的 propagation metric。

### Gate 4: Adaptive precision baseline

如果要做 method improvement，必須比較：

- uniform 4-bit
- uniform 3-bit
- random mixed precision
- layer-depth heuristic
- kurtosis-guided mixed precision
- activation-weighted-error-guided mixed precision
- ideally AWQ / GPTQ / OmniQuant baseline

要回答：

```text
Diagnostic-guided mixed precision 是否真的比簡單 heuristic 好？
```

通過標準：

- 在相同 average bit budget 下，模型品質明顯較好。
- 或在相同品質下，average bit 明顯較低。

---

## 5. Learning Project Plan

### Goal

把 AWQ-Diag 做成一個完整、可展示、可解釋的學習型 project。

不是追求 SOTA，而是展示：

- 理解 LLM quantization literature。
- 能讀懂 Transformer module structure。
- 能用 hooks 收集中間 activation。
- 能設計 layer-wise diagnostic metrics。
- 能誠實處理 hypothesis 不成立的結果。

### Final Deliverables

```text
AWQ-Diag/
├── README.md
├── note.md
├── awq_diagnostic.ipynb
├── diagnostic_Qwen2.5-1.5B.json
├── research_gap_plan.md
├── figures/
│   ├── saliency_curve.png
│   ├── kurtosis_by_layer.png
│   ├── bitwidth_error_sweep.png
│   └── kurtosis_vs_jump_ratio.png
└── report.md
```

### Suggested Report Structure

```md
# AWQ-Diag Report

## 1. Motivation
- Why LLM quantization matters
- Why AWQ is important
- Why low-bit failure is worth diagnosing

## 2. Background
- Weight-only quantization
- Activation-aware saliency
- AWQ
- Activation outliers
- 3-bit / 2-bit quantization difficulty

## 3. Method
- Model: Qwen2.5-1.5B
- Calibration text
- Forward hooks
- Activation statistics
- AWQ-style importance
- Symmetric per-output-channel quantization
- Activation-weighted relative error

## 4. Experiments
- Saliency curve reproduction
- Kurtosis by layer
- Bit-width sweep
- 4-bit to 3-bit jump ratio
- Kurtosis vs jump ratio

## 5. Findings
- Activation outliers exist
- MLP down_proj can have high kurtosis
- 4-to-3 bit jump is smooth in this setup
- Kurtosis does not predict jump ratio

## 6. Interpretation
- Single-layer statistics are insufficient
- Inter-layer propagation is a stronger next hypothesis
- Proxy error must be connected to model-level behavior

## 7. Limitations
- One model
- Small calibration set
- Proxy error only
- No true AWQ scale search comparison
- No perplexity / downstream evaluation yet

## 8. Next Steps
- Multi-model replication
- Output/logit error tracing
- Inter-layer propagation experiment
- Adaptive mixed precision as extension
```

---

## 6. Concrete Next Experiments

### Experiment 1: Clean up current diagnostic pipeline

目的：

- 把 notebook 從 exploratory 變成 reproducible。

輸出：

- deterministic seed
- model name parameter
- calibration text parameter
- saved JSON schema
- saved figures

成功標準：

- 重新執行 notebook 可以產生同樣的 JSON 和 figures。

### Experiment 2: Add output error tracing

目的：

- 從 weight proxy error 走向 actual layer output error。

做法：

- 對每個 Linear layer：
  - 保留原始 output。
  - 用 quantized weight 重新計算 output。
  - 計算 output MSE / relative error。

成功標準：

- 可以比較：
  - weight quantization proxy error
  - actual layer output error
  - jump ratio

### Experiment 3: Single-layer quantization injection

目的：

- 測 inter-layer propagation。

做法：

- 一次只量化一個 layer。
- 跑完整模型 forward。
- 計算：
  - final logits KL
  - final logits MSE
  - next-token rank change
  - optional perplexity

成功標準：

- 找出 local error 和 final output degradation 是否一致。

### Experiment 4: Module-family comparison

目的：

- 看 q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj 是否有穩定差異。

輸出：

```md
| Module Type | Mean Kurtosis | Mean 4-to-3 Jump | Mean 3-bit Error | Interpretation |
|---|---|---|---|---|
```

成功標準：

- 能說明哪類 module 比較敏感。

### Experiment 5: Multi-model replication

目的：

- 判斷目前 negative result 是否只是 Qwen2.5-1.5B 的特例。

建議模型：

- Qwen2.5-0.5B
- Qwen2.5-1.5B
- Llama-3.2-1B or 3B if accessible
- Gemma / Phi small model if hardware permits

成功標準：

- 每個模型產生同格式 diagnostic JSON。
- 可以做 cross-model summary table。

### Experiment 6: Toy adaptive precision

目的：

- 做 project extension，不當主 claim。

做法：

- 根據 diagnostic score 選 top 10% sensitive layers 用 4-bit。
- 其他 layers 用 3-bit。
- 和 random top 10%、depth-based heuristic 比較。

成功標準：

- 至少在 proxy metric 或 output error 上優於 random heuristic。

---

## 7. What Not To Do

不要做：

- 不要宣稱自己提出新 AWQ。
- 不要只用 kurtosis 做 predictor，因為目前結果已經不支持。
- 不要只跑一個模型就宣稱 general finding。
- 不要只看 layer-wise proxy error 就宣稱 model performance。
- 不要做 custom CUDA kernel，除非 project 目標轉成 systems。
- 不要追 2-bit SOTA，除非有 QAT、benchmark、compute 和強 baseline。

---

## 8. Minimal 2-Week Plan

### Week 1

1. 整理 notebook。
2. 補 README。
3. 固定 JSON schema。
4. 產出 4 張核心圖：
   - saliency curve
   - kurtosis by layer
   - bit-width error sweep
   - kurtosis vs jump ratio
5. 寫 `report.md` 前半：motivation、background、method。

### Week 2

1. 加入 output error tracing。
2. 做 module-family comparison。
3. 整理 negative result。
4. 寫 `report.md` 後半：experiments、findings、limitations、next steps。
5. 準備一份 8 到 10 分鐘 presentation。

---

## 9. Research-Oriented 6-Week Plan

只有在你真的想推 research 時才走這條。

### Week 1: Reproducible diagnostic pipeline

- notebook parameterization
- figures
- JSON schema

### Week 2: Model-level metrics

- output error
- logits KL
- optional perplexity

### Week 3: Inter-layer propagation

- single-layer quantization injection
- attention vs MLP comparison
- residual stream error tracing

### Week 4: Cross-model replication

- at least 3 small models
- same calibration protocol
- cross-model summary

### Week 5: Adaptive precision toy method

- heuristic baselines
- diagnostic-guided allocation
- average bit budget accounting

### Week 6: Research decision

如果結果顯示：

- diagnostic score 能預測 model-level degradation，或
- propagation metric 明顯比 single-layer stats 好，或
- adaptive precision 明顯優於 heuristic，

才考慮寫成 research-oriented project。

否則，維持 learning project 定位。

---

## 10. Final Recommendation

目前最誠實的結論是：

```text
AWQ-Diag 不應該現在被包裝成一個新的 AWQ research gap project。
```

原因：

- AWQ 本身已成熟。
- 低 bit quantization 方向非常擁擠。
- 本地結果沒有支持 phase transition predictor。
- kurtosis hypothesis 目前是 negative result。
- 現階段缺少 multi-model、downstream metric、baseline comparison。

最好的 project positioning 是：

```text
AWQ-Diag 是一個學習型、診斷型、可展示的 LLM quantization project。它的價值在於復現 AWQ 的核心直覺、建立 layer-wise diagnostic pipeline、產出負面結果，並把下一步研究問題明確收斂到 inter-layer error propagation。
```

如果未來要轉 research，真正值得追的不是「kurtosis predictor」，而是：

```text
Can inter-layer error propagation explain low-bit quantization sensitivity better than single-layer activation statistics?
```

這個方向仍然有可能，但必須用更嚴格的實驗證明。

---

## References

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- [SpinQuant: LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)
- [ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](https://arxiv.org/abs/2502.02631)
- [Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in Large Language Models From Edge to Giant](https://arxiv.org/abs/2409.11055)
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
- [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/abs/2405.04532)
- [Fast Matrix Multiplications for Lookup Table-Quantized LLMs](https://arxiv.org/abs/2407.10960)
