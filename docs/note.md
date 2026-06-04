# AWQ-Diag Project Note

## Project Overview

AWQ-Diag 是一個針對大型語言模型量化行為的診斷型研究專案。這個專案的重點不是直接把模型量化到更低 bit，而是分析「為什麼量化會失敗」，尤其是模型從 4-bit 降到 3-bit 或 2-bit 時，某些 layer 的誤差是否會突然爆炸，形成所謂的 phase transition。

專案使用 `Qwen/Qwen2.5-1.5B` 作為主要實驗模型，透過 PyTorch forward hook 擷取 Transformer 中每個 Linear layer 的 input activation，計算 activation magnitude、variance、kurtosis、maximum value 和 outlier ratio 等統計量。這些統計量再被用來分析 AWQ-style weight importance，以及不同 bit-width 下的量化誤差變化。

## Motivation

LLM 量化可以大幅降低模型記憶體需求與推論成本，但 aggressive quantization 往往會導致模型表現突然崩潰。一般量化實作多半著重在「如何量化」，而這個 project 更關心的是「哪些 layer 為什麼比較難量化」。

這個專案的核心問題是：

- LLM activation 中的 outlier 是否會影響某些 layer 的量化穩定性？
- Per-layer kurtosis 能不能作為預測 phase transition 的 lightweight signal？
- 4-bit 到 3-bit 的誤差跳躍是否集中在特定 layer 或特定 module，例如 attention projection 或 MLP projection？
- 如果能提前預測容易崩潰的 layer，是否能進一步設計 layer-wise adaptive precision allocation？

## What I Built

我建立了一個完整的 Jupyter Notebook 實驗流程，用來觀察 AWQ 與低 bit 量化下的模型行為。主要流程包含：

1. 載入 Hugging Face 上的 causal language model 與 tokenizer。
2. 觀察 Qwen2.5-1.5B 的 Transformer block 結構，確認 attention、MLP 與 Linear layer 的命名與 shape。
3. 使用少量 calibration text 跑 forward pass，模擬 AWQ calibration 的資料收集流程。
4. 在每個 Linear layer 上註冊 forward hook，攔截 input activation 並計算 per-channel statistics。
5. 用 activation magnitude 和 weight magnitude 計算 AWQ-style importance，復現 AWQ paper 中提到的 hockey-stick saliency curve。
6. 對所有 Transformer block 的 Linear layers 做 kurtosis 分析，找出 activation outlier 最嚴重的 layer。
7. 對每個 layer 模擬 8-bit、6-bit、4-bit、3-bit、2-bit quantization，計算 activation-weighted relative quantization error。
8. 分析 4-bit 到 3-bit 的 error jump ratio，觀察是否存在 phase transition。
9. 將 jump ratio 和 kurtosis 存成 JSON，方便之後跨模型比較。

## Technical Details

這個專案使用的核心方法是 PyTorch forward hook。每當模型中的 Linear layer 被執行時，hook 會取得該 layer 的 input activation，並針對 hidden dimension 做 per-channel 統計。

主要統計量包括：

- `channel_magnitude`: 每個 input channel 的平均絕對 activation，作為 AWQ saliency signal。
- `channel_variance`: 每個 channel 的變異數，用來描述 activation 分布的離散程度。
- `kurtosis`: excess kurtosis，用來衡量 activation distribution 是否具有 heavy-tail 與 outlier。
- `channel_max`: 每個 channel 的最大絕對值。
- `outlier_ratio`: 超過 6 sigma 的 activation 比例。

在 bit-width sweep 中，我對每個 Linear layer 的 weight 做 symmetric per-output-channel quantization，並計算 activation-weighted quantization error。這樣的 error metric 不是單純看 weight MSE，而是把 activation magnitude 納入權重，讓重要 channel 的誤差被放大，概念上更接近 AWQ 關心的量化敏感度。

## Current Result

目前已經完成 `Qwen/Qwen2.5-1.5B` 的診斷結果，並輸出到 `diagnostic_Qwen2.5-1.5B.json`。

這次實驗共分析了 196 個 Transformer block 內的 Linear layers。4-bit 到 3-bit 的 jump ratio 統計如下：

- minimum: 約 2.99x
- median: 約 3.92x
- mean: 約 3.86x
- maximum: 約 4.02x
- jump ratio 大於 5x 的 layer: 0 個

這代表在目前這次 Qwen2.5-1.5B 的實驗設定下，4-bit 到 3-bit 的量化誤差普遍會上升，但沒有觀察到超過 5x threshold 的劇烈 phase transition。

另一方面，kurtosis 分布顯示某些 layer 的 activation outlier 明顯較嚴重。例如最高 kurtosis 出現在 `model.layers.1.mlp.down_proj`，mean kurtosis 約為 11.99，但它的 4-to-3 bit jump ratio 約為 3.46x，並不是最高的量化誤差跳躍。整體來看，這次資料中 kurtosis 與 jump ratio 的 Spearman correlation 約為 -0.36，表示單靠 per-layer kurtosis 並不能直接解釋這次實驗中的 phase transition。

## Key Finding

這個 project 的重要發現是：activation outlier 確實存在，而且不同 layer 的 kurtosis 差異很大，但在目前 Qwen2.5-1.5B 的實驗結果中，kurtosis 並沒有直接正向預測 4-bit 到 3-bit 的 error jump。

這個結果反而指出一個更值得深入研究的方向：低 bit 量化失敗可能不只是 single-layer activation distribution 的問題，也可能與 inter-layer error propagation、residual stream、attention/MLP 之間的誤差累積有關。

## Skills Demonstrated

這個專案展現了我在以下幾個面向的能力：

- 深入理解 Transformer block、attention projection、MLP projection 與 tokenizer 的實際運作。
- 使用 PyTorch hook 擷取模型中間 activation，並設計可重複的模型診斷流程。
- 理解 AWQ 的 activation-aware quantization 概念，並實作 AWQ-style saliency analysis。
- 設計 bit-width sweep 實驗，模擬不同精度下的 layer-wise quantization error。
- 使用 kurtosis、outlier ratio、weighted MSE 等統計方法分析 LLM quantization failure。
- 將實驗結果整理成 JSON，為後續跨模型比較與論文方向探索做準備。

## How I Would Describe This Project

這是一個研究 LLM 低 bit 量化失敗原因的診斷工具。我使用 Qwen2.5-1.5B 作為實驗模型，透過 PyTorch forward hook 收集每個 Linear layer 的 activation statistics，並分析 activation outlier、AWQ weight importance 與不同 bit-width 下的 quantization error。專案的核心目標是找出哪些 layer 在 4-bit 降到 3-bit 時容易出現 phase transition，並驗證 kurtosis 是否能作為預測量化崩潰的指標。實驗結果顯示，activation outlier 的確在不同 layer 間有明顯差異，但單靠 kurtosis 無法完整解釋量化誤差跳躍，暗示低 bit 量化失敗可能還涉及跨 layer 的 error propagation。

## Future Work

後續可以把同樣流程擴展到更大的模型，例如 LLaMA-3.2-3B、LLaMA-2-7B 或其他 Qwen 模型，觀察模型大小與架構差異是否會改變 phase transition 行為。

如果未來要把這個 project 發展成更完整的研究，可以加入更多 predictor，例如 spectral norm、layer depth、attention entropy、residual stream magnitude，並進一步設計 adaptive precision allocation：對量化敏感的 layer 保留較高 bit，對穩定 layer 使用更低 bit，讓模型在壓縮率與 accuracy 之間取得更好的平衡。
