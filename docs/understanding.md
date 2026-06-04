# AWQ-Diag 從 0 到 100：完整理解這個 Project

> 這份文件假設你完全不熟量化，從最基本的概念一路堆到這個 project 的每一個設計決策、
> 每一張圖、每一個數字。讀完你應該能對任何人（包含面試官）把這個 project 講清楚。

目錄：
- [Part 0 — 一句話 + 心智模型](#part-0--一句話--心智模型)
- [Part 1 — 為什麼要量化](#part-1--為什麼要量化)
- [Part 2 — 量化到底怎麼做](#part-2--量化到底怎麼做symmetric-per-channel)
- [Part 3 — 為什麼低 bit 會崩：outliers 與 AWQ](#part-3--為什麼低-bit-會崩outliers-與-awq)
- [Part 4 — 這個 project 量了什麼](#part-4--這個-project-到底量了什麼method)
- [Part 5 — 假設與實驗（八張圖）](#part-5--假設與實驗八張圖)
- [Part 6 — 結果與「正確的」解讀](#part-6--結果與正確的解讀重點)
- [Part 7 — 程式架構](#part-7--程式架構code-怎麼組織)
- [Part 8 — 怎麼跑](#part-8--怎麼跑)
- [Part 9 — 限制與下一步](#part-9--限制與下一步)
- [Part 10 — 面試 Q&A 預備](#part-10--面試-qa-預備)

---

## Part 0 — 一句話 + 心智模型

**一句話：**
> AWQ-Diag 不是做一個「更強的量化方法」，而是一個**診斷工具**：用 PyTorch hook 把模型每一層的
> activation 抓出來分析，問一個問題——「能不能用一個便宜的單層統計量（kurtosis），預測模型在
> 4-bit 降到 3-bit 時哪裡會壞掉？」

**最重要的心智模型（先記住，後面才有方向感）：**

```
量化 = 把連續的數字塞進「有限的格子」裡 → 一定有誤差
bit 越少 = 格子越少 = 誤差越大
這個 project 在問：誤差變大的時候，是「平均地」變大，還是「某幾層突然爆炸」(phase transition)？
                  如果是突然爆炸，能不能事先用 activation 的統計量預測？
```

**最後的答案（先破梗）：**
- 在 Qwen2.5-1.5B / 0.5B 上，**沒有觀察到「突然爆炸」**——4→3 bit 的誤差是平滑上升的。
- kurtosis（outlier 嚴重程度）**不能**預測「跳躍幅度」，但**能**預測「誤差的絕對水平」。
- 這是一個**誠實的負面結果**，而且它把下一個更好的研究問題指出來了。

---

## Part 1 — 為什麼要量化

### LLM 很大
一個參數用 16-bit（bfloat16）存，Qwen2.5-1.5B 有 ~15 億參數 → 約 3 GB。7B 模型 → 約 14 GB。
跑起來還要再加 activation、KV cache。記憶體與頻寬就是錢、就是速度。

### 量化 = 用更少的 bit 存權重
把每個 16-bit 浮點數，近似成一個低 bit 整數（例如 4-bit 整數只有 16 個可能值）。
- 4-bit → 模型大小直接掉到 1/4。
- 整數運算在硬體上也更快。

這個 project 只做 **weight-only quantization**（只量化權重，不量化 activation），因為這是
AWQ 的設定，也是最常見的部署設定。

### bit-width 的直覺
| bits | 可表示的數值個數 | 直覺 |
|---|---|---|
| 8-bit | 256 | 很細，幾乎無損 |
| 4-bit | 16 | 還行，業界主流 |
| 3-bit | 8 | 開始痛 |
| 2-bit | 4 | 通常崩 |

「為什麼 3-bit 以下特別難」就是整個低 bit 量化研究的核心戰場（GPTQ / AWQ / OmniQuant /
QuaRot / ParetoQ 全在打這塊）。這個 project 就是來「觀察」這個變難的過程。

---

## Part 2 — 量化到底怎麼做（symmetric per-channel）

這個 project 用的是最基本、最標準的 **symmetric per-output-channel uniform quantization**。
拆開來解釋：

一個 Linear layer 的權重是一個矩陣 `W`，形狀 `[out_features, in_features]`
（PyTorch 慣例：`y = x @ W.T`）。

**步驟（對每一個 output channel，也就是 W 的每一列，獨立做）：**

```python
n_levels = 2 ** bits                       # 例如 3-bit → 8 個格子
w_max   = W.abs().amax(dim=1, keepdim=True) # 這一列裡最大的絕對值
scale   = w_max / (n_levels // 2)           # 一格有多寬
q       = round(W / scale).clamp(-(n_levels//2), n_levels//2 - 1)  # 量化成整數
W_deq   = q * scale                         # 還原回浮點（dequantize）
```

- **symmetric**：格子對稱地分布在 0 兩側（不像 asymmetric 會有 zero-point 偏移）。
- **per-output-channel**：每一列用自己的 scale，因為不同 output channel 的權重大小差很多，
  共用一個 scale 會犧牲小的那些。
- **uniform**：格子等寬（相對於 non-uniform / lookup-table 量化）。

**一個具體例子（3-bit）：**
某列 `W = [0.10, -0.40, 0.05, 0.80]`
- `n_levels=8`，整數範圍 `-4..3`
- `w_max=0.80`，`scale=0.80/4=0.20`
- `q = round([0.5, -2, 0.25, 4]) = [0, -2, 0, 4] → clamp → [0, -2, 0, 3]`
- `W_deq = [0, -0.40, 0, 0.60]`

注意兩件事：
1. `0.10` 和 `0.05` 都被壓成 `0`（小值在粗格子裡消失）。
2. `0.80` 被 clamp 成 `0.60`（symmetric 上界只到 `3*scale`）。
   → **bit 越少，格子越粗，這種「四捨五入掉」和「clamp 掉」的誤差就越大。**

這就是量化誤差的來源。整個 project 就是在量「這個誤差，在不同層、不同 bit 數下，長什麼樣」。

---

## Part 3 — 為什麼低 bit 會崩：outliers 與 AWQ

### activation 是什麼
模型 forward 的時候，每一個 Linear layer 都會收到一個輸入向量 `x`（叫 activation），
形狀 `[batch, seq_len, in_features]`。例如 Qwen2.5-1.5B 的 hidden dim = 1536，
一段 52 個 token 的文字 → `x` 是 `[1, 52, 1536]`。

把 `in_features` 這個維度叫做 **channel**（特徵維度）。

### outlier channel 現象
Transformer 的 activation 有一個著名特性：**少數幾個 channel 的數值會異常地大**
（其他 channel 都在 ±1 附近，某幾個 channel 卻到 ±50）。這叫 activation outlier，
是 LLM.int8()、SmoothQuant、AWQ 都在處理的核心痛點。

**為什麼 outlier 讓量化變難？**
量化的 scale 是被「最大值」決定的。一個 outlier 把 `w_max`（或 activation 的範圍）撐很大
→ scale 變很粗 → 其他正常的值全部擠在少數幾個格子裡 → 資訊全毀。

### AWQ 的洞見
AWQ（Activation-aware Weight Quantization，MLSys 2024 最佳論文）說：
> 不是所有 weight 一樣重要。一個 weight 重不重要，要看它乘上的 activation 有多大。

定義 **importance（saliency）**：
```
importance[channel j] = mean_i |W[i, j]|  ×  mean |activation[j]|
                         (這個 channel 的權重大小)  (這個 channel 的 activation 大小)
```
AWQ 發現如果把 importance 排序，會看到一條 **hockey-stick curve**（曲棍球桿）：
極少數 channel 的 importance 遠遠高於其他。AWQ 的做法是用等價縮放去「保護」這些重要 channel。

**這個 project 不實作 AWQ 的縮放**——我們只「借用」它的 importance 概念來做診斷：
1. 復現 hockey-stick（證明我理解 AWQ）。
2. 用 importance 當權重去算「activation-aware 的量化誤差」。
3. 用 kurtosis 量 outlier，看它能不能預測哪層難量化。

---

## Part 4 — 這個 project 到底量了什麼（method）

### 4.1 Forward hook（怎麼把 activation 抓出來）
PyTorch 的 hook 是一個「掛在某層上的 callback」：那層每次 forward，PyTorch 就自動呼叫你的
函式，把 input / output 交給你。我們在**每一個 Transformer block 裡的 Linear layer**都掛一個
hook，攔截它的輸入 activation `x`。

> 程式在 `src/awq_diag/hooks.py` 的 `ActivationCollector`。

### 4.2 每層算 5 個 per-channel 統計量
對攔到的 `x`（形狀 `[1, seq, in]`），沿著 batch+seq 壓掉，對每個 channel 算：

| 統計量 | 公式（直覺） | 用途 |
|---|---|---|
| `channel_magnitude` | `mean|x|` | AWQ 的 saliency 信號 |
| `channel_variance` | `var(x)` | 分布有多散 |
| `channel_max` | `max|x|` | 最壞情況 |
| `kurtosis` | `E[z⁴] - 3`，`z=(x-μ)/σ` | **outlier 嚴重程度** |
| `outlier_ratio` | `P(|x| > 6σ)` | 超過 6 倍標準差的比例 |

**kurtosis（峰度）要特別理解，它是這個 project 的主角：**
- 它衡量分布「尾巴有多厚 / 有多少 outlier」。
- 正態分布的 kurtosis 剛好 = 3，所以我們減 3 變成 **excess kurtosis**，讓「正態 = 0」當基準。
- `= 0`：跟正態一樣正常。`>> 0`：又尖又厚尾，有嚴重 outlier。
- 直覺例子：1000 個 ~N(0,1) 的點 → kurtosis ≈ 0；如果摻幾個 ±10 的點，`z⁴` 被那幾個點
  主宰（10⁴ = 10000），kurtosis 立刻飆高。**所以 kurtosis 高 = 有 outlier channel。**

**為什麼 outlier threshold 選 6σ？**
3σ 在正態下約 0.27% 的點會超過（太鬆，正常波動就會觸發）；6σ 在正態下幾乎不可能發生
（約十億分之二），所以「超過 6σ」幾乎一定是真 outlier，不是正常分布的尾巴。

### 4.3 Bit-width sweep（核心實驗）
對每一層，模擬 `{8, 6, 4, 3, 2}`-bit 量化，算量化誤差。重點不是只看一個 bit，而是看
**誤差怎麼隨 bit 數變化的曲線**。

### 4.4 兩種誤差（這是這個 project 比原版更扎實的地方）

**(a) proxy error — activation-weighted weight MSE（便宜版）**
```
proxy = Σ_j act_mag[j] · mean_i (W - Ŵ)²[i,j]   /   Σ_j act_mag[j] · mean_i W²[i,j]
```
把每個 channel 的權重誤差，用該 channel 的 activation 大小加權（重要 channel 的誤差被放大）。
只需要權重 + activation magnitude，很便宜。這是「AWQ 風格」的誤差。

**(b) output error — 真實 layer 輸出誤差（誠實版）**
```
output = ‖Wx - Ŵx‖²  /  ‖Wx‖²
```
直接拿真實的 activation `x`，算「量化前的輸出」和「量化後的輸出」差多少。這才是這一層
真正造成的輸出退化（bias 會在相減時抵消，所以不用管）。

> 為什麼要兩種？因為 (a) 便宜但是是 proxy（代理指標），(b) 貴但接近真相。其中一個實驗就是
> 問：**便宜的 proxy 到底能不能代表真實的 output error？**（答案：部分能，ρ≈0.62。）

### 4.5 Jump ratio 與 phase transition 的定義
```
jump = error(3-bit) / error(4-bit)      ← 4 bit 降到 3 bit，誤差變幾倍
```
如果某層 `jump > 5`，就標記為「phase transition」（誤差突然爆炸）。
原始假設是：**high-kurtosis 的層，jump 應該特別大。**

---

## Part 5 — 假設與實驗（八張圖）

整個 pipeline 跑完，對每個模型輸出 8 張圖 + 一個 JSON。每張圖對應一個問題：

| # | 圖檔 | 在問什麼 | 看到什麼 |
|---|---|---|---|
| 1 | `saliency_curve.png` | AWQ 的 importance 真的集中嗎？ | hockey-stick：top 1% channel 佔最多 17.6% importance（≈18× 均勻） |
| 2 | `kurtosis_by_layer.png` | 哪些層 outlier 最重？ | 少數層 kurtosis 特別高（最高 κ≈12，`layers.1.mlp.down_proj`） |
| 3 | `bitwidth_error_sweep.png` | 有沒有 phase transition？ | 4→3 jump 中位數 3.92×，**0 層超過 5×**（平滑，沒爆炸） |
| 4 | `kurtosis_vs_jump_ratio.png` | **kurtosis 能預測 jump 嗎？** | **不能**：Spearman ρ = −0.36（甚至負相關！） |
| 5 | `module_family.png` | 哪類 module 比較敏感？ | outlier 集中在 `o_proj`/`down_proj`，但 jump 在各 family 幾乎一樣平 |
| 6 | `proxy_vs_output_error.png` | 便宜 proxy 能代表真實輸出誤差嗎？ | 部分能：ρ≈0.62，但系統性高估 |
| 7 | `importance_surface_*.png` | （3D）importance 在 channel×layer 上長怎樣 | 經典 outlier-channel 尖塔 surface |
| 8 | `cross_model_jump_distribution.png` | 換個模型結論還成立嗎？ | 0.5B 與 1.5B 都一樣（中位數 ~3.9×，沒有 >5×） |

第 4 張是整個 project 的高潮：它直接驗證原始假設，而且**推翻了它**。

---

## Part 6 — 結果與「正確的」解讀（重點）

這段是面試最該講清楚的地方，因為它有一個很容易講錯的 nuance。

### 6.1 原始假設被推翻
「high kurtosis → 4→3 大跳躍」這個假設，在資料上是 **ρ = −0.36**（1.5B）、−0.26（0.5B）。
不只是「沒有正相關」，而是**輕微負相關**——kurtosis 高的層，jump 反而稍微小一點。

### 6.2 但是！kurtosis 預測的是「水平」不是「跳躍」
這是最關鍵的細節：

| 關係 | Spearman ρ (1.5B) | 意思 |
|---|---|---|
| kurtosis vs **4→3 jump ratio** | **−0.36** | ❌ outlier 多 ≠ 對降 bit 更敏感 |
| kurtosis vs **絕對 3-bit error** | **+0.55** | ✅ outlier 多 = 誤差的「地板」更高 |

用一個比喻：
> 想像每一層是一個爬樓梯的人。
> - **絕對 error level** = 他現在站在幾樓（kurtosis 高的人站得比較高樓 → +0.55）。
> - **jump ratio** = 從 4 樓爬到 3 樓他「多累」（所有人爬一階都差不多累 → 跟他站幾樓無關 → −0.36）。
>
> 所以 outlier **整體抬高了誤差的水平**，但**不會讓某一層在降 bit 時特別敏感**。

這個區分（level vs sensitivity）就是這個 project 真正的 insight，比原本 note 裡只說
「弱相關」更精準、更有深度。

### 6.3 module-family 證據
- `down_proj`（讀 MLP 中間層）和 `o_proj`（讀 attention 輸出）的 kurtosis 是其他 projection
  的 **約 35~55 倍**（κ≈4.3 / 2.9 vs ~0.08）。
- 它們的**絕對 3-bit error 也最高**（0.135 / 0.103 vs ~0.08）→ 呼應 6.2 的「水平」。
- **但它們的 jump ratio 並沒有比較高**（甚至 down_proj 最低 3.68×）→ 呼應 6.2 的「敏感度」。
- 第 5 張圖把這個「左邊 kurtosis 差很多、右邊 jump 一樣平」並排，是最有說服力的一張。

### 6.4 proxy vs output
便宜的 proxy jump 和真實的 output jump 相關 ρ≈0.62（兩個模型都 0.6+）。
→ proxy 可以當「篩選信號」，但會系統性高估真實輸出退化，不能完全取代。

### 6.5 cross-model
同樣的結論（沒有 phase transition、kurtosis 不預測 jump）在 0.5B 和 1.5B 都成立。
→ 不是 1.5B 的偶然特例，至少在 Qwen2.5 family 內穩定。

### 6.6 為什麼這是個「好的」負面結果
1. 它**誠實**——資料不支持假設就說不支持，沒有硬凹。
2. 它**有 actionable 的下一步**：既然單層統計量不夠，問題可能在**跨層誤差傳遞
   (inter-layer error propagation)**——local error 怎麼沿著 residual stream 被放大或吸收。
3. 它展示了完整的科學流程：提假設 → 設計可重複實驗 → 量化驗證 → 推翻 → 收斂出更好的問題。

---

## Part 7 — 程式架構（code 怎麼組織）

### 資料流
```
model (HF Qwen2.5)
   │  load_model_and_tokenizer            (model_utils.py)
   ▼
ActivationCollector  ──register hooks──►  每個 Linear forward 時：
   │  (hooks.py)                            • 算 5 個 per-channel 統計量
   │                                        • 用真實 x 算 output-error 累加器
   ▼
collector.stats / collector.output_acc
   │  build_layer_records / build_summary  (analysis.py)
   │     • proxy_error_sweep / output_error  (quant.py)
   │     • top-1% importance, jump ratio
   │     • module-family 聚合, Spearman 相關
   ▼
result dict ──► JSON (results/) + 8 figures (figures/)   (pipeline.py + plotting.py)
```

### 每個檔案的職責
| 檔案 | 做什麼 |
|---|---|
| `config.py` | `DiagConfig`：一個物件控制整次實驗（model、bit_widths、seed、輸出路徑） |
| `data.py` | calibration 用的 4 段文字 |
| `model_utils.py` | 載入模型、列出 block 裡的 Linear、判斷 module 類型 / layer index、設 seed |
| `hooks.py` | **核心**：`ActivationCollector`，收集統計量 + output-error tracing |
| `quant.py` | symmetric per-channel 量化、proxy/output 誤差、jump ratio（有 pytest） |
| `analysis.py` | 把原始收集結果 → 每層 record + summary + module-family + 相關係數 |
| `plotting.py` | 8 張圖（含 3D importance surface） |
| `pipeline.py` | 串起來：model → JSON + figures，印 headline |
| `cli.py` / `scripts/` | 命令列入口、跨模型比較 |

> 設計原則：**notebook 是探索用的草稿，.py pipeline 才是 canonical、可重跑的版本**。
> 而且重構後跑出來的數字和原 notebook 一模一樣（median 3.92×、ρ=−0.360、top-κ 同一層），
> 證明重構沒改壞任何邏輯。

---

## Part 8 — 怎麼跑

```bash
# 環境（micromamba，PyTorch cu128）
micromamba env create -f environment.yml
micromamba activate awq-diag

# 跑單一模型 → 產生 results/ + figures/
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-1.5B
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-0.5B

# 跨模型比較
python scripts/compare_models.py results/diagnostic_*.json

# 單元測試（純 CPU，不用下載模型）
pytest
```

純 CPU 機器：`--device cpu --dtype float32`。

---

## Part 9 — 限制與下一步

### 誠實的限制
- **只有一個 architecture family**（Qwen2.5，兩個 size）——不宣稱能推廣到 Llama/Gemma/Phi。
- **proxy / output error 都是 layer-local**——不是 end-task 品質（perplexity / 準確率）。
  proxy↔output 的比較只是往「真實品質」邁的第一步，但還停在 layer 輸出。
- **calibration 很小**（4 段文字）——足以刻畫 per-channel 分布，不足以對罕見事件下結論。
- **沒有跟真正的 AWQ scale search / GPTQ baseline 比**——這是診斷工具，不是新方法。

### 下一步（負面結果指出的方向）
1. **單層量化注入**：一次只量化一層，跑完整 forward，量 final-logit KL / perplexity，
   直接測 error 怎麼往後傳。
2. **把 proxy error 接到 model-level 指標**（logit KL、perplexity）來驗證或替換它。
3. **更多 model family**，看 `o_proj`/`down_proj` 的 outlier 集中是不是普遍現象。
4. **toy adaptive precision**（純 extension）：誤差最大的層留 4-bit、其他降 3-bit，
   跟 random / depth heuristic 在固定平均 bit 預算下比。

> `docs/research_gap_plan.md` 有更完整的「要變成 research 必須通過哪些 validation gate」的分析。

---

## Part 10 — 面試 Q&A 預備

**Q：為什麼用 kurtosis 當 predictor？**
A：kurtosis 直接量「尾巴厚度 / outlier 嚴重度」，而 outlier 正是量化變難的已知主因
（LLM.int8 / SmoothQuant）。它是個便宜的單層信號，所以是很自然的第一個 hypothesis——
如果它能預測 phase transition，就有一個 lightweight predictor。結果它不行，這本身就是發現。

**Q：你的 proxy error 跟「真的把模型量化」差在哪？**
A：proxy 是 activation-weighted 的 weight MSE，只看單層權重的重建誤差；真正的量化退化還包含
activation 量化、跨層誤差累積、以及對最終 logits / perplexity 的影響。我加了 output-error
（用真實 activation 算 `‖Wx−Ŵx‖/‖Wx‖`）就是為了往真相靠一步，並驗證 proxy 的可信度（ρ≈0.62）。

**Q：為什麼沒看到 phase transition？是不是你的設定太弱？**
A：完全可能，這也是我列在 limitation 的。我用的是 layer-local 的 proxy/output error，不是
perplexity；calibration 也小。所以我**不宣稱**「LLM 沒有 phase transition」，只說「在這個
診斷設定下，4→3 是平滑的」。真正要下強結論需要 model-level 指標——這正是我的下一步。

**Q：為什麼 outlier threshold 是 6σ 不是 3σ？**
A：3σ 在正態下約 0.27% 會超過，太鬆，正常波動就觸發；6σ 幾乎不可能由正態產生，所以超過的
幾乎一定是真 outlier。這讓 outlier_ratio 這個指標更乾淨。

**Q：為什麼 `down_proj` / `o_proj` 的 outlier 特別重？**
A：它們的輸入分別是 MLP 的中間 activation（SwiGLU 之後）和 attention 的輸出，這些位置已知
容易產生大幅值的 channel。我的資料顯示它們 kurtosis 是其他 projection 的約 35~55 倍，
而且絕對 3-bit error 也最高——但 jump ratio 沒有比較高，這就是「水平 vs 敏感度」的區別。

**Q：負面結果為什麼還算貢獻？**
A：因為它（1）誠實地驗證並推翻了一個看似合理的假設，（2）區分出「kurtosis 預測誤差水平
（ρ=+0.55）而非降 bit 敏感度（ρ=−0.36）」這個精確的 nuance，（3）把下一個更有希望的假設
（inter-layer propagation）明確指出來。比起硬湊一個 SOTA，這更像真實研究會發生的事。

**Q：如果給你兩週/兩個月，你會怎麼往 research 推？**
A：先做單層量化注入 + logit KL，把 layer-local error 接到 model-level degradation；
再做跨層 propagation 的追蹤（local error 沿 residual stream 怎麼放大）。如果發現
propagation metric 比單層統計量更能解釋 model-level 退化，那才是真正的 research gap。

---

### 一段話總結（可以直接背）
> 「AWQ-Diag 是一個診斷工具，用 forward hook 抓每層 activation，復現 AWQ 的 importance、
> 量 outlier（kurtosis）、做 8→2 bit 的 sweep。核心發現是：activation outlier 確實存在且集中在
> `o_proj`/`down_proj`，它會抬高低 bit 誤差的**絕對水平**（kurtosis vs 3-bit error ρ≈+0.55），
> 但**不能**預測 4→3 bit 的誤差**跳躍幅度**（ρ≈−0.36），而且沒有觀察到劇烈的 phase transition——
> 這個結論在 0.5B 和 1.5B 都成立。所以低 bit 崩潰比較可能是跨層誤差傳遞的問題，而不是單層
> activation 分布就能解釋的。」
