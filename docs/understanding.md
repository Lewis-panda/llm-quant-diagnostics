# AWQ-Diag 從 0 到 100

目錄：
- [Part 0 — 一句話 + 必要概念](#part-0--一句話--必要概念)
- [Part 1 — 為什麼要量化](#part-1--為什麼要量化)
- [Part 2 — 量化到底怎麼做](#part-2--量化到底怎麼做symmetric-per-channel)
- [Part 3 — 為什麼低 bit 會崩：outliers 與 AWQ](#part-3--為什麼低-bit-會崩outliers-與-awq)
- [Part 4 — 這個 project 量了什麼](#part-4--這個-project-到底量了什麼method)
- [Part 5 — 假設與實驗（九張圖）](#part-5--假設與實驗九張圖)
- [Part 6 — 結果與解讀](#part-6--結果與解讀)
- [Part 7 — 程式架構](#part-7--程式架構code-怎麼組織)
- [Part 8 — 怎麼跑](#part-8--怎麼跑)
- [Part 9 — 限制與下一步](#part-9--限制與下一步)
- [Part 10 — Q&A](#part-10--qa)

---

## Part 0 — 一句話 + 必要概念

**一句話：**
> AWQ-Diag 是一個**診斷工具**(不是做一個更強的量化方法)：用 PyTorch hook 把模型每一層的
> activation 抓出來分析，問一個問題——「能不能用一個便宜的單層統計量（kurtosis），預測模型在
> 4-bit 降到 3-bit 時哪裡會壞掉？」

**必要概念：**

```
量化(Quantization) = 把連續的數字塞進「有限的格子」裡 → 一定有誤差
bit 越少 = 格子越少 = 誤差越大
這個 project 在問：誤差變大的時候，是「平均地」變大，還是「某幾層突然爆炸」(phase transition)？
                  如果是突然爆炸，能不能事先用 activation 的統計量預測？
```

**Results：**
- 在 Qwen2.5-1.5B / 0.5B 上，**沒有觀察到「突然爆炸」**——4→3 bit 的誤差是平滑上升的。
- kurtosis（outlier 嚴重程度）**不能**預測「跳躍幅度」，但**能**預測「誤差的絕對水平」。
- 額外實作了 AWQ 的 activation-aware scaling，它**剛好**最會救 outlier 最重的兩類 layer。
- 這是一個**誠實的負面結果**，並指出下一個更好的研究問題。

---

## Part 1 — 為什麼要量化

### LLM 很大
一個參數用 16-bit（bfloat16）存，Qwen2.5-1.5B 有 ~15 億參數 → 約 3 GB。7B 模型 → 約 14 GB。
跑起來還要再加 activation、KV cache。記憶體與頻寬就是錢、就是速度。

### 量化 = 用更少的 bit 存權重
把每個 16-bit 浮點數，近似成一個低 bit 整數。
- 例如 4-bit → 模型大小直接掉到 1/4。
- 整數運算在硬體上也更快。

這個 project 只做 **weight-only quantization**（只量化權重，不量化 activation），因為這是
AWQ 的設定，也是常見的部署設定。

### bit-width 的直覺
| bits | 可表示的數值個數 | 直覺 |
|---|---|---|
| 8-bit | 256 | 很細，幾乎無損 |
| 4-bit | 16 | 還行，業界主流 |
| 3-bit | 8 | 開始有問題 |
| 2-bit | 4 | 通常崩 |

「為什麼 3-bit 以下特別難」就是整個低 bit 量化研究的核心戰場（GPTQ / AWQ / OmniQuant /
QuaRot / ParetoQ 全在打這塊）。這個 project 就是「觀察」這個變難的過程。

---

## Part 2 — 量化到底怎麼做（symmetric per-channel）

base 用的是標準的 **symmetric per-output-channel uniform quantization**(就是 RTN，
round-to-nearest)。一個 Linear 的權重是矩陣 `W`，形狀 `[out_features, in_features]`
(PyTorch 慣例 `y = x @ W.T`)。

**步驟(對每一個 output channel，也就是 W 的每一列獨立做)：**

```python
n_levels = 2 ** bits                        # 例如 3-bit → 8 個格子
w_max   = W.abs().amax(dim=1, keepdim=True) # 這一列裡最大的絕對值
scale   = w_max / (n_levels // 2)           # 一格有多寬
q       = round(W / scale).clamp(-(n_levels//2), n_levels//2 - 1)  # 量化成整數
W_deq   = q * scale                         # 還原回浮點(dequantize)
```

- **symmetric**：格子對稱分布在 0 兩側(沒有 zero-point 偏移)。
- **per-output-channel**：每列用自己的 scale，因為不同 output channel 的權重大小差很多。
- **uniform**：格子等寬(相對於 non-uniform / lookup-table 量化)。

**具體例子(3-bit)：** 某列 `W = [0.10, -0.40, 0.05, 0.80]`
- `n_levels=8`，整數範圍 `-4..3`；`w_max=0.80`，`scale=0.20`
- `q = round([0.5, -2, 0.25, 4]) = [0,-2,0,4] → clamp → [0,-2,0,3]`
- `W_deq = [0, -0.40, 0, 0.60]`

兩個誤差來源：
1. `0.10`、`0.05` 被壓成 `0`(小值在粗格子裡消失)。
2. `0.80` 被 clamp 成 `0.60`(symmetric 上界只到 `3*scale`)。

→ bit 越少，格子越粗，誤差越大。整個 project 就是量這個誤差在不同層、不同 bit 下長什麼樣。

> **為什麼用 RTN，不用真 AWQ？** RTN 是無最佳化、無資料相依的固定探針——要比較「層本身有多難」，
> 就不能讓量化器去把難度優化掉。真 AWQ 會在這個 base 上加保護(見 Part 4.6)，我們是拿它跟
> RTN 比，量出「保護有多少用」。

---

## Part 3 — 為什麼低 bit 會崩：outliers 與 AWQ

### activation 是什麼
模型 forward 時，每個 Linear 都會收到一個輸入 `x`(activation)，形狀
`[batch, seq_len, in_features]`。例如 Qwen2.5-1.5B 的 hidden dim = 1536，52 個 token 的文字
→ `x` 是 `[1, 52, 1536]`。`in_features` 這個維度叫 **channel**(特徵維度)。

### outlier channel 現象
Transformer 的 activation 有一個著名特性：**少數幾個 channel 的數值異常地大**(其他 channel
在 ±1 附近，某幾個卻到 ±50)。這叫 activation outlier，是 LLM.int8()、SmoothQuant、AWQ 都在
處理的核心痛點。

**為什麼 outlier 讓量化變難？** 量化的 scale 被「最大值」決定。一個 outlier 把範圍撐很大 →
scale 變粗 → 其他正常值全部擠在少數格子裡 → 資訊損失。

### AWQ 的洞見
AWQ(Activation-aware Weight Quantization，MLSys 2024 最佳論文)：weight 重不重要，要看它乘上的
activation 有多大。定義 **importance(saliency)**：

```
importance[channel j] = mean_i |W[i, j]|  ×  mean |activation[j]|
                         (這個 channel 的權重大小)    (這個 channel 的 activation 大小)
```

把 importance 排序會看到 **hockey-stick curve**：極少數 channel 的 importance 遠高於其他。AWQ
用等價縮放去保護這些重要 channel(Part 4.6 會實作)。

這個 project 借用 importance 概念做三件事：
1. 復現 hockey-stick(證明理解 AWQ)。
2. 用 importance 當權重算 activation-aware 的量化誤差。
3. 用 kurtosis 量 outlier，看能不能預測哪層難量化。

---

## Part 4 — 這個 project 到底量了什麼（method）

### 4.1 Forward hook
PyTorch 的 hook 是「掛在某層上的 callback」：那層每次 forward，PyTorch 自動呼叫你的函式，
把 input / output 交給你。我們在每個 Transformer block 裡的 Linear 都掛一個 hook，攔截輸入
activation `x`。程式在 `src/awq_diag/hooks.py` 的 `ActivationCollector`。

### 4.2 每層算 5 個 per-channel 統計量
對 `x`(形狀 `[1, seq, in]`)沿 batch+seq 壓掉，對每個 channel 算：

| 統計量 | 公式 | 用途 |
|---|---|---|
| `channel_magnitude` | `mean|x|` | AWQ 的 saliency 信號 |
| `channel_variance` | `var(x)` | 分布有多散 |
| `channel_max` | `max|x|` | 最壞情況 |
| `kurtosis` | `E[z⁴] - 3`，`z=(x-μ)/σ` | outlier 嚴重程度 |
| `outlier_ratio` | `P(|x| > 6σ)` | 超過 6σ 的比例 |

**kurtosis(峰度)** 衡量分布尾巴有多厚。正態分布 kurtosis = 3，減 3 變成 **excess kurtosis**，
讓「正態 = 0」當基準：`=0` 跟正態一樣，`>>0` 又尖又厚尾。直覺：1000 個 ~N(0,1) 的點
kurtosis ≈ 0；摻幾個 ±10 的點，`z⁴` 被那幾個點主宰(10⁴=10000)，kurtosis 立刻變大。

**6σ threshold：** 3σ 在正態下約 0.27% 會超過(太鬆)；6σ 幾乎不可能由正態產生，所以超過 6σ
幾乎一定是真 outlier。

### 4.3 Bit-width sweep
對每層模擬 `{8, 6, 4, 3, 2}`-bit 量化，看誤差隨 bit 數變化的曲線。

### 4.4 兩種誤差

**(a) proxy error — activation-weighted weight MSE(便宜)**
```
proxy = Σ_j act_mag[j] · mean_i (W - Ŵ)²[i,j]   /   Σ_j act_mag[j] · mean_i W²[i,j]
```
把每個 channel 的權重誤差用 activation 大小加權。只需要權重 + activation magnitude。

**(b) output error — 真實 layer 輸出誤差(誠實)**
```
output = ‖Wx - Ŵx‖²  /  ‖Wx‖²
```
直接拿真實 `x`，算量化前後輸出差多少(bias 在相減時抵消)。

→ 一個實驗就是問：便宜的 proxy 能不能代表真實的 output error(答案：部分能，ρ≈0.62)。

### 4.5 Jump ratio 與 phase transition
```
jump = error(3-bit) / error(4-bit)      ← 4 bit 降到 3 bit，誤差變幾倍
```
`jump > 5` 標記為 phase transition。原始假設：high-kurtosis 的層，jump 應該特別大。

### 4.6 實作 AWQ scaling(把「算了 importance 但沒用」補起來)
4.1–4.5 算了 importance 卻用 RTN 量化，從沒「用」importance。這裡補上 AWQ 真正的機制：
per-input-channel scaling `s = (mean|x|)^α`(mean-normalized)，套成
`Ŵ = quant(W·diag(s))·diag(1/s)`，把 activation 大的 salient channel 放大、量化後再縮回去。

對每層每個 bit grid-search α 讓 output error 最小(`α=0` 就是 RTN，所以 AWQ 不可能比 RTN 差)，
回報 **3-bit error reduction = RTN / AWQ**。程式在 `hooks.py` 的 `AWQErrorCollector`。

---

## Part 5 — 假設與實驗（九張圖）

pipeline 跑完每個模型輸出 9 張圖 + 一個 JSON。每張圖對應一個問題：

| # | 圖檔 | 在問什麼 | 看到什麼 |
|---|---|---|---|
| 1 | `saliency_curve.png` | AWQ importance 真的集中嗎？ | hockey-stick：top 1% channel 佔最多 17.6% importance(≈18× 均勻) |
| 2 | `kurtosis_by_layer.png` | 哪些層 outlier 最重？ | 少數層 kurtosis 特別高(最高 κ≈12，`layers.1.mlp.down_proj`) |
| 3 | `bitwidth_error_sweep.png` | 有沒有 phase transition？ | 4→3 jump 中位數 3.92×，0 層超過 5×(平滑) |
| 4 | `kurtosis_vs_jump_ratio.png` | kurtosis 能預測 jump 嗎？ | 不能：Spearman ρ = −0.36(負相關) |
| 5 | `module_family.png` | 哪類 module 比較敏感？ | outlier 集中在 `o_proj`/`down_proj`，但 jump 在各 family 幾乎一樣平 |
| 6 | `proxy_vs_output_error.png` | 便宜 proxy 能代表真實輸出誤差嗎？ | 部分能：ρ≈0.62，但系統性高估 |
| 7 | `awq_reduction.png` | AWQ 的保護在哪裡幫最大？ | 剛好幫 `down_proj`(2.3×)/`o_proj`(1.6×)，其他 ~1.2× |
| 8 | `importance_surface_*.png` | (3D)importance 在 channel×layer 上長怎樣 | 經典 outlier-channel 尖塔 surface |
| 9 | `cross_model_jump_distribution.png` | 換個模型結論還成立嗎？ | 0.5B 與 1.5B 都一樣 |

第 4 張直接驗證原始假設並推翻它；第 7 張是「實作 AWQ 並量出它的效益」。

---

## Part 6 — 結果與解讀

### 6.1 原始假設被推翻
「high kurtosis → 4→3 大跳躍」在資料上是 ρ = −0.36(1.5B)、−0.26(0.5B)。不只是沒有正相關，
而是輕微負相關——kurtosis 高的層，jump 反而稍微小一點。

### 6.2 kurtosis 預測的是「水平」不是「跳躍」
最關鍵的細節：

| 關係 | Spearman ρ (1.5B) | 意思 |
|---|---|---|
| kurtosis vs **4→3 jump ratio** | **−0.36** | outlier 多 ≠ 對降 bit 更敏感 |
| kurtosis vs **絕對 3-bit error** | **+0.55** | outlier 多 = 誤差的「地板」更高 |

比喻：每一層是一個爬樓梯的人。
- **error level** = 他站在幾樓(kurtosis 高的人站比較高樓 → +0.55)。
- **jump ratio** = 從 4 樓爬到 3 樓多累(所有人爬一階都差不多累，跟站幾樓無關 → −0.36)。

→ outlier 整體抬高誤差水平，但不會讓某層在降 bit 時特別敏感。這個 level vs sensitivity 的
區分，就是這個 project 真正的 insight。

### 6.3 module-family 證據
- `down_proj`(讀 MLP 中間)和 `o_proj`(讀 attention 輸出)的 kurtosis 是其他 projection 的
  約 35~55 倍(κ≈4.3 / 2.9 vs ~0.08)。
- 它們的絕對 3-bit error 也最高(0.135 / 0.103 vs ~0.08)→ 呼應 6.2 的「水平」。
- 但 jump ratio 沒有比較高(down_proj 甚至最低 3.68×)→ 呼應 6.2 的「敏感度」。
- 第 5 張圖把「左邊 kurtosis 差很多、右邊 jump 一樣平」並排，是最有說服力的一張。

### 6.4 為什麼根本看不到 phase transition：~4×/bit 是數學必然
這是一個必須主動講的點。uniform 量化的 MSE ∝ `4^(-bits)`(step 每減半，MSE 變 4 倍)，所以
**任何層、任何分布**降一個 bit，誤差都會 ≈4×。實測完全吻合：

```
8→6→4→3→2 每個 bit step 的誤差倍率：4.00 / 3.99 / 3.92 / 3.65 ×
```

含義：layer-local 的 RTN proxy error 在設計上**幾乎不可能**顯示 phase transition——它是 bit 的
平滑解析函數。所以「沒有層超過 5×」一半是指標本身決定的。真正的 transition(若存在)只會出現在
**model-level 指標**(perplexity / logit KL)，這正是下一步要接的東西。

### 6.5 proxy vs output
proxy jump 和真實 output jump 相關 ρ≈0.62(兩個模型都 0.6+)。→ proxy 可當篩選信號，但系統性
高估真實退化，不能完全取代。

### 6.6 實作 AWQ：保護有效，而且剛好救到 outlier family
AWQ scaling 把 3-bit output error 降下來，效果**集中在高 kurtosis 的兩類 layer**：

| module family | AWQ 3-bit error reduction (RTN / AWQ) | mean kurtosis |
|---|---|---|
| `down_proj` | **2.31×** | 4.31 |
| `o_proj` | **1.63×** | 2.86 |
| 其他 5 類(q/k/v/gate/up) | 1.13–1.27× | ~0.08 |

單一最大受益層是 `L2.mlp.down_proj`，error 被砍 **25.9×**；top-8 受益層全是 down_proj/o_proj。
這在 0.5B 也成立(down_proj/o_proj ~2.3× vs 其他 ~1.2×，max 28.9×)。

注意一個 nuance：**per-layer 的 ρ(kurtosis, reduction) ≈ 0**，因為 7 類 module 裡有 5 類是低
outlier、kurtosis 全擠在 ~0.08，形成一個沒有內部訊號的 blob，把 rank correlation 稀釋掉。但
**categorical(module-family)層級**訊號非常乾淨：AWQ 的保護剛好救到有 outlier 的那兩類——這是
AWQ 核心 thesis 的正面驗證。

### 6.7 cross-model
沒有 phase transition、kurtosis 不預測 jump、AWQ 救 outlier family——三個結論在 0.5B 和 1.5B
都成立，不是單一模型的偶然特例(至少在 Qwen2.5 family 內穩定)。

### 6.8 為什麼這是個「好的」負面結果
1. 誠實：資料不支持假設就說不支持。
2. 有 actionable 的下一步：既然單層統計量不夠，問題可能在跨層誤差傳遞
   (inter-layer error propagation)——local error 怎麼沿 residual stream 被放大或吸收。
3. 完整的科學流程：提假設 → 可重複實驗 → 量化驗證 → 推翻 → 收斂出更好的問題。

---

## Part 7 — 程式架構（code 怎麼組織）

### 資料流
```
model (HF Qwen2.5)
   │  load_model_and_tokenizer            (model_utils.py)
   ▼
ActivationCollector  ──hooks──►  每個 Linear forward：5 個 per-channel 統計量
   │  (hooks.py)                                + 用真實 x 算 RTN output-error
   ▼
AWQErrorCollector    ──第二趟──►  grid-search α，量 AWQ vs RTN 的 output-error
   │  (hooks.py)
   ▼
build_layer_records / build_summary   (analysis.py)
   │     • proxy_error_sweep / output_error / awq scaling   (quant.py)
   │     • top-1% importance, jump ratio, awq reduction
   │     • module-family 聚合, Spearman 相關
   ▼
result dict ──► JSON (results/) + 9 figures (figures/)   (pipeline.py + plotting.py)
```

### 每個檔案的職責
| 檔案 | 做什麼 |
|---|---|
| `config.py` | `DiagConfig`：一個物件控制整次實驗(model、bit_widths、seed、輸出路徑) |
| `data.py` | calibration 用的 4 段文字 |
| `model_utils.py` | 載入模型、列出 block 裡的 Linear、判斷 module 類型 / layer index、設 seed |
| `hooks.py` | `ActivationCollector`(統計量 + RTN output error)、`AWQErrorCollector`(AWQ scaling search) |
| `quant.py` | symmetric per-channel 量化、AWQ scaling、proxy/output 誤差、jump ratio(有 pytest) |
| `analysis.py` | 原始收集結果 → 每層 record + summary + module-family + 相關係數 |
| `plotting.py` | 9 張圖(含 AWQ reduction、3D importance surface) |
| `pipeline.py` | 串起來：model → JSON + figures，印 headline |
| `cli.py` / `scripts/` | 命令列入口、跨模型比較 |

> 設計原則：notebook 是探索草稿，.py pipeline 才是 canonical、可重跑的版本。重構後跑出來的
> 數字和原 notebook 一模一樣(median 3.92×、ρ=−0.360、top-κ 同一層)，證明重構沒改壞邏輯。

---

## Part 8 — 怎麼跑

```bash
# 環境(micromamba，PyTorch cu128)
micromamba env create -f environment.yml
micromamba activate awq-diag

# 跑單一模型 → 產生 results/ + figures/
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-1.5B
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-0.5B

# 跨模型比較
python scripts/compare_models.py results/diagnostic_*.json

# 單元測試(純 CPU，不用下載模型)
pytest
```

純 CPU 機器：`--device cpu --dtype float32`。

---

## Part 9 — 限制與下一步

### 限制
- **jump ratio 被一個解析常數主宰：** uniform 量化 MSE ∝ `4^(-bits)`，每降一個 bit 誤差就 ≈4×，
  跟分布無關。所以 layer-local RTN proxy **天生看不到 phase transition**——真正的 transition 只會
  在 model-level 指標出現。「沒有層 >5×」要這樣讀。
- **kurtosis 是四階矩**，從小 calibration set(4 段、每層幾百 token)估，本來就高變異，所以相關
  係數是定性訊號，不要當精確值。
- **proxy / output error 都是 layer-local**，不是 end-task 品質(perplexity / 準確率)。
- **量化器是簡化版：** base 是 symmetric per-output-channel RTN(刻意的固定探針)；AWQ pass 加了
  scaling，但不是完整的部署 AWQ(group-wise + asymmetric + 折進前一層)，也沒有 GPTQ baseline，
  所以絕對誤差數字不能拿去跟生產級 AWQ/GPTQ 比。
- 只有一個 architecture family(Qwen2.5，兩個 size)。

### 下一步
1. **單層量化注入**：一次只量化一層，跑完整 forward，量 final-logit KL / perplexity，直接測
   error 怎麼往後傳。
2. **把 proxy error 接到 model-level 指標**(logit KL、perplexity)來驗證或替換它。
3. **更多 model family**，看 `o_proj`/`down_proj` 的 outlier 集中是不是普遍現象。
4. **toy adaptive precision**(extension)：誤差最大的層留 4-bit、其他降 3-bit，跟 random /
   depth heuristic 在固定平均 bit 預算下比。

> `docs/research_gap_plan.md` 有完整的「要變成 research 必須通過哪些 validation gate」。

---

## Part 10 — Q&A

**Q：為什麼用 kurtosis 當 predictor？**
kurtosis 直接量尾巴厚度 / outlier 嚴重度，而 outlier 是量化變難的已知主因(LLM.int8 /
SmoothQuant)。它是便宜的單層信號，所以是很自然的第一個 hypothesis；結果它不行，這本身就是發現。

**Q：為什麼用 RTN 不用真 AWQ？**
RTN 是無最佳化、無資料相依的固定探針。要量「層本身有多難」，就不能讓量化器把難度優化掉——用
AWQ 會量到「AWQ optimizer 有多強」而不是「層有多難」。而且我有實作 AWQ scaling(Part 4.6)，拿它
跟 RTN 比，量出保護的效益。所以 RTN 是 baseline，AWQ 是相對於它的改善。

**Q：你的 proxy error 跟真的量化差在哪？**
proxy 是 activation-weighted 的 weight MSE，只看單層權重重建誤差；真正的退化還包含 activation
量化、跨層誤差累積、對 logits / perplexity 的影響。我加了 output-error(用真實 activation)往真相
靠一步，並驗證 proxy 的可信度(ρ≈0.62)。

**Q：為什麼沒看到 phase transition？是不是設定太弱？**
有一半是指標決定的：uniform 量化 MSE ∝ `4^(-bits)`，每降一 bit 都 ≈4×(實測 4.00/3.99/3.92/
3.65×)，所以 layer-local RTN proxy 天生看不到 transition。我**不宣稱**「LLM 沒有 phase
transition」，只說在這個診斷設定下 4→3 是平滑的；要下強結論得用 model-level 指標。

**Q：AWQ 那個結果怎麼解讀？per-layer ρ≈0 不是代表沒關係嗎？**
不是。per-layer ρ≈0 是因為 7 類 module 有 5 類是低 outlier、kurtosis 全擠在 ~0.08，形成沒有訊號
的 blob，把 rank correlation 稀釋掉。但 categorical 層級訊號很乾淨：AWQ 的保護剛好集中在
down_proj(2.3×)/o_proj(1.6×)，其他 ~1.2×，top-8 受益層全是這兩類。這是 AWQ thesis 的正面驗證。

**Q：為什麼 `down_proj` / `o_proj` 的 outlier 特別重？**
它們的輸入分別是 MLP 中間 activation(SwiGLU 之後)和 attention 輸出，這些位置已知容易產生大幅值
channel。我的資料顯示它們 kurtosis 是其他 projection 的約 35~55 倍，絕對 3-bit error 也最高，AWQ
對它們幫助也最大——但 4→3 jump 沒有比較高，這就是「水平 vs 敏感度」的區別。

**Q：為什麼 outlier threshold 是 6σ 不是 3σ？**
3σ 在正態下約 0.27% 會超過，太鬆；6σ 幾乎不可能由正態產生，所以超過的幾乎一定是真 outlier，
讓 outlier_ratio 這個指標更乾淨。

**Q：負面結果為什麼還算貢獻？**
因為它(1)誠實驗證並推翻一個看似合理的假設，(2)區分出「kurtosis 預測誤差水平(ρ=+0.55)而非降
bit 敏感度(ρ=−0.36)」這個精確 nuance，(3)把下一個更有希望的假設(inter-layer propagation)指出來。
比起硬湊一個 SOTA，這更接近真實研究會發生的事。

**Q：如果給你兩週/兩個月，你會怎麼往 research 推？**
先做單層量化注入 + logit KL，把 layer-local error 接到 model-level degradation；再做跨層
propagation 的追蹤(local error 沿 residual stream 怎麼放大)。如果 propagation metric 比單層統計量
更能解釋 model-level 退化，那才是真正的 research gap。

---

### 一段話總結
> AWQ-Diag 是一個診斷工具，用 forward hook 抓每層 activation，復現 AWQ importance、量 outlier
> (kurtosis)、做 8→2 bit sweep，並實作 AWQ scaling 跟 RTN 比。核心發現：activation outlier 集中
> 在 `o_proj`/`down_proj`，會抬高低 bit 誤差的**絕對水平**(ρ≈+0.55)，但**不能**預測 4→3 bit 的
> 誤差**跳躍幅度**(ρ≈−0.36)；而且因為 uniform 量化 MSE 是 ~4×/bit 的解析必然，layer-local proxy
> 本來就看不到 phase transition。AWQ 的 activation-aware 保護剛好最會救那兩類 outlier layer
> (down_proj 2.3×、單層最高 25.9×)。結論在 0.5B 和 1.5B 都成立，指向下一步：低 bit 崩潰比較可能
> 是跨層誤差傳遞，而不是單層 activation 分布就能解釋。
