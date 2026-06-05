# AWQ-Diag 從 0 到 100

目錄：
- [Part 0 — 一句話 + 必要概念](#part-0--一句話--必要概念)
- [Part 1 — 為什麼要量化](#part-1--為什麼要量化)
- [Part 2 — 量化到底怎麼做](#part-2--量化到底怎麼做symmetric-per-channel)
- [Part 3 — AWQ 的核心：activation importance](#part-3--awq-的核心activation-importance)
- [Part 4 — 這個 project 量了什麼](#part-4--這個-project-量了什麼method)
- [Part 5 — 四張核心圖](#part-5--四張核心圖)
- [Part 6 — 結果：importance 真的有意義](#part-6--結果importance-真的有意義)
- [Part 7 — 程式架構](#part-7--程式架構code-怎麼組織)
- [Part 8 — 怎麼跑](#part-8--怎麼跑)
- [Part 9 — 限制與下一步](#part-9--限制與下一步)
- [Part 10 — Q&A](#part-10--qa)

---

## Part 0 — 一句話 + 必要概念

**一句話：**
> AWQ-Diag 是一個**理解 + 視覺化 AWQ** 的 project：用 PyTorch hook 把每層 activation 抓出來，
> 看 AWQ 講的「activation importance」到底長什麼樣，並驗證它**真的有意義**——保護重要 channel
> 真的能降低量化誤差。

**必要概念(AWQ 的核心主張)：**

```
不是所有 weight 一樣重要。
一個 weight 多重要 = |W| × |它乘上的 activation|       ← activation-aware importance
少數「salient channel」撐起大部分 importance。
保護這些 channel(放大後再量化)= activation-aware quantization = AWQ。
```

**Results：**
- **importance 高度集中**(hockey-stick)：top 1% channel 在某層佔到 17.6% 的總 importance(≈18× 均勻)。
- 這些 salient channel 是**真的 activation outlier**，集中在 `o_proj` / `down_proj`。
- 把 importance **視覺化**成 channel × layer 的 3D surface(經典 outlier 尖塔)。
- **importance 真的有意義**：實作 AWQ scaling 後，誤差下降**剛好集中在 importance 最重的地方**
  (`down_proj` 2.3×、單層最高 25.9×)，低 importance 的 module 幾乎不動。
- 在 Qwen2.5-1.5B / 0.5B 都成立。

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

bit 越低誤差越大——所以「**怎麼用同樣的 bit 數把誤差壓低**」就是 AWQ 在解的問題。

---

## Part 2 — 量化到底怎麼做（symmetric per-channel）

base 用標準的 **symmetric per-output-channel uniform quantization**(就是 RTN，
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

誤差來源：小值被四捨五入成 0、大值被 clamp。bit 越少格子越粗、誤差越大。

> RTN 是這個 project 的 **baseline**(無最佳化、無資料相依)。AWQ 是「在這個 base 上加保護」，
> 我們會拿 AWQ 跟 RTN 比，量出保護到底有多少用(Part 4.5)。

---

## Part 3 — AWQ 的核心：activation importance

### activation 是什麼
模型 forward 時，每個 Linear 都會收到輸入 `x`(activation)，形狀
`[batch, seq_len, in_features]`。例如 Qwen2.5-1.5B 的 hidden dim = 1536，52 個 token 的文字
→ `x` 是 `[1, 52, 1536]`。`in_features` 這個維度叫 **channel**。

### outlier channel 現象
Transformer 的 activation 有一個著名特性：**少數幾個 channel 的數值異常地大**(其他在 ±1，
某幾個到 ±50)。這叫 activation outlier，是 LLM.int8() / SmoothQuant / AWQ 共同的關注點。

### AWQ 的洞見
> 一個 weight 重不重要，不是只看 weight 本身，而是看它**乘上的 activation 有多大**。

定義 **importance(saliency)**：
```
importance[channel j] = mean_i |W[i, j]|  ×  mean |activation[j]|
                         (weight 大小)         (activation 大小)
```
把 importance 排序會看到 **hockey-stick curve**：極少數 channel 的 importance 遠高於其他。AWQ 的
做法是用等價縮放**放大 salient channel** → 它在量化時相對誤差變小 → 等於用更多 bit 保護重要
channel，但維持硬體友善的 uniform 量化。

這個 project 就是要把這條洞見**看清楚、量出來**：importance 真的集中嗎?salient channel 真的是
outlier 嗎?保護它們真的有用嗎?

---

## Part 4 — 這個 project 量了什麼（method）

### 4.1 Forward hook
hook 是「掛在某層上的 callback」，那層每次 forward，PyTorch 自動把 input / output 交給你。我們在
每個 block 裡的 Linear 都掛 hook，攔截輸入 `x`。程式在 `hooks.py` 的 `ActivationCollector`。

### 4.2 per-channel 統計量
對 `x` 沿 batch+seq 壓掉，對每個 channel 算：

| 統計量 | 公式 | 用途 |
|---|---|---|
| `channel_magnitude` | `mean|x|` | **AWQ 的 saliency 信號** |
| `kurtosis` | `E[z⁴] - 3` | 確認 salient channel 是真的 heavy-tail outlier |
| `outlier_ratio` | `P(|x| > 6σ)` | 超過 6σ 的比例 |
| `channel_variance` / `channel_max` | — | 分布散度 / 最壞情況 |

kurtosis(峰度)衡量尾巴有多厚：正態 = 3，減 3 變成 excess kurtosis，`>>0` 代表又尖又厚尾、有
outlier。6σ 門檻是因為正態幾乎不可能超過 6σ，所以超過的幾乎一定是真 outlier。

### 4.3 importance 與 hockey-stick
用 `mean|W|·mean|x|` 算每個 channel 的 importance，排序看集中度(top 1% 佔多少總 importance)。

### 4.4 量化誤差(看低 bit 有多痛)
對每層做 `{8,6,4,3,2}`-bit 量化，量兩種誤差：
- **proxy error**：activation-weighted 的 weight MSE(便宜，只要權重 + activation magnitude)。
- **output error**：真實的 `‖Wx - Ŵx‖² / ‖Wx‖²`(用真實 activation，最接近真相)。

這是背景：低 bit 誤差成長很快(每降一 bit ~4×)——所以才需要 AWQ 的保護。

### 4.5 AWQ scaling search（核心：把保護「做出來」並量效益）
這是整個 project 的重點。實作 AWQ 真正的機制：per-input-channel scaling `s = (mean|x|)^α`
(mean-normalized)，套成 `Ŵ = quant(W·diag(s))·diag(1/s)`——把 activation 大的 salient channel
放大、量化後再縮回去。

對每層每個 bit grid-search α 讓 output error 最小(`α=0` 就是 RTN，所以 AWQ 不可能比 RTN 差)，
回報 **error reduction = RTN / AWQ**。這個數字就是「**保護 salient channel 到底有多少用**」。
程式在 `hooks.py` 的 `AWQErrorCollector`。

---

## Part 5 — 四張核心圖

| # | 圖 | 在講什麼 | 看到什麼 |
|---|---|---|---|
| 1 | `saliency_curve.png` | importance 集中嗎? | hockey-stick：top 1% channel 佔最多 17.6%(≈18× 均勻) |
| 2 | `importance_surface_*.png` | (3D)importance 長什麼樣? | channel × layer 的尖塔 surface，salient channel 一目了然 |
| 3 | `module_family.png` | importance / outlier 在哪? | 集中在 `o_proj` / `down_proj`，遠高於其他 projection |
| 4 | `awq_reduction.png` | 保護它們有用嗎? | AWQ 把誤差降下來，**剛好降在 importance 最重的兩類**，其他幾乎不動 |

(另外還產生 `kurtosis_by_layer`(輔助診斷)和 `cross_model_awq_reduction`(跨模型比較 AWQ 效益)。)

第 4 張是收尾：它把「importance 不只是直覺，而是真的有操作意義」一槌定音。

---

## Part 6 — 結果：importance 真的有意義

### 6.1 importance 高度集中
hockey-stick 很明顯：top 1% channel 在某層佔到 17.6% 的總 importance(≈18× 均勻分配)。AWQ「少數
channel 撐起大部分重要性」的前提，在真實模型上成立。

### 6.2 salient channel 是真的 outlier，而且集中在特定 module
- kurtosis 最高到 ~12(`layers.1.mlp.down_proj`)。
- 按 module family 聚合，outlier **壓倒性集中在 `down_proj` 和 `o_proj`**，是其他 projection 的
  約 35~55 倍(κ≈4.3 / 2.9 vs ~0.08)。
- 它們的輸入分別是 MLP 中間 activation(SwiGLU 之後)和 attention 輸出——已知的 outlier 熱點。

### 6.3 保護它們真的有用（importance 的「意義」就在這）
實作 AWQ scaling 後，3-bit output error 的下降幅度：

| module family | AWQ error reduction (RTN / AWQ) | mean kurtosis |
|---|---|---|
| `down_proj` | **2.31×** | 4.31 |
| `o_proj` | **1.63×** | 2.86 |
| 其他 5 類(q/k/v/gate/up) | 1.13–1.27× | ~0.08 |

- 單層最高：`L2.mlp.down_proj` 誤差被砍 **25.9×**；top-8 受益層**全部**是 down_proj/o_proj。
- 也就是說：**AWQ 的保護剛好落在 importance / outlier 最重的地方**——這正是 AWQ thesis 的正面驗證。
- 0.5B 也一樣(down_proj/o_proj ~2.3× vs 其他 ~1.2×，單層最高 28.9×)。

一個誠實的 nuance：**per-layer 的 ρ(kurtosis, reduction) ≈ 0**，因為 7 類 module 有 5 類是低
outlier、kurtosis 全擠在 ~0.08，形成沒有訊號的 blob，把 rank correlation 稀釋掉。但 **module-family
層級**的訊號非常乾淨。第 4 張圖左右兩 panel 剛好把「per-layer 看起來雜、但 outlier family 明顯
被救」並排呈現。

### 6.4 為什麼這個 project 站得住
- 它**復現**並**視覺化**了 AWQ 的核心(importance 集中、outlier 分布、3D surface)。
- 它把 AWQ 的機制**實作**出來(不只是讀懂)，並用真實 output error 量出效益。
- 它**驗證** AWQ 的中心主張：保護「重要」channel 真的有用，而且用在對的地方。

> 附帶觀察(不是主線)：低 bit 的 weight-quant 誤差大致隨 ~4×/bit 成長，這也是「為什麼低 bit 需要
> AWQ 這種保護」的背景。

---

## Part 7 — 程式架構（code 怎麼組織）

### 資料流
```
model (HF Qwen2.5)
   │  load_model_and_tokenizer            (model_utils.py)
   ▼
ActivationCollector  ──hooks──►  每個 Linear forward：per-channel 統計量 + RTN output-error
   │  (hooks.py)
   ▼
AWQErrorCollector    ──第二趟──►  grid-search α，量 AWQ vs RTN 的 output-error reduction
   │  (hooks.py)
   ▼
build_layer_records / build_summary   (analysis.py)
   │     • importance / top-1% share
   │     • AWQ reduction、module-family 聚合
   ▼
result dict ──► JSON (results/) + 9 figures (figures/)   (pipeline.py + plotting.py)
```

### 每個檔案的職責
| 檔案 | 做什麼 |
|---|---|
| `config.py` | `DiagConfig`：一個物件控制整次實驗 |
| `data.py` | calibration 用的 4 段文字 |
| `model_utils.py` | 載入模型、列出 Linear、判斷 module 類型 / layer index |
| `hooks.py` | `ActivationCollector`(統計量 + error)、`AWQErrorCollector`(AWQ scaling search) |
| `quant.py` | symmetric per-channel 量化、AWQ scaling、誤差(有 pytest) |
| `analysis.py` | importance、AWQ reduction、module-family、summary |
| `plotting.py` | 9 張圖(含 3D importance surface、AWQ benefit) |
| `pipeline.py` | 串起來：model → JSON + figures |
| `cli.py` / `scripts/` | 命令列入口、跨模型比較 |

---

## Part 8 — 怎麼跑

```bash
micromamba env create -f environment.yml
micromamba activate awq-diag

python scripts/run_diagnostic.py --model Qwen/Qwen2.5-1.5B
python scripts/run_diagnostic.py --model Qwen/Qwen2.5-0.5B
python scripts/compare_models.py results/diagnostic_*.json
pytest                              # 純 CPU 單元測試
```

純 CPU 機器：`--device cpu --dtype float32`。

---

## Part 9 — 限制與下一步

### 限制
- **量化器是簡化版**：base 是 symmetric per-output-channel RTN，AWQ pass 加了 scaling，但不是完整
  部署 AWQ(group-wise + asymmetric + 折進前一層)，也沒有 GPTQ baseline，所以絕對誤差數字是示意，
  不是生產級數字。
- **誤差是 layer-local**，不是 end-task 品質(perplexity / 準確率)——AWQ 效益量在 layer 輸出，還沒
  接到 model-level。
- **kurtosis 是四階矩**、calibration 也小(4 段)，所以統計量是定性訊號。
- 只有一個 architecture family(Qwen2.5，兩個 size)。

### 下一步
1. **group-wise + asymmetric AWQ**，從「機制 demo」往真實量化器靠。
2. **把 layer-level 的 AWQ 效益接到 model-level 品質**(perplexity / logit KL)：保護重要 channel
   是不是真的救回 end-task 準確率，而不只是 layer 輸出誤差?
3. **更多 model family**(Llama / Gemma / Phi)，看 `o_proj`/`down_proj` 的集中是不是普遍。

---

## Part 10 — Q&A

**Q：AWQ 的 importance 為什麼是 `|W|·|activation|`？**
因為輸出貢獻 ≈ weight × activation。一個 weight 再大，如果乘上的 activation 很小，對輸出影響也小；
反之 activation 大的 channel 即使 weight 普通，也很關鍵。所以「重要性」要同時看兩邊。

**Q：你怎麼證明 importance「有意義」，不是只畫個漂亮的圖？**
我把 AWQ 的保護**實作**出來(per-channel scaling search)，量它把真實 output error 降多少。結果保護的
效益**剛好集中在 importance / outlier 最重的 `down_proj`/`o_proj`**(2.3× / 1.6×，單層最高 25.9×)，
低 importance 的 module 幾乎不動。保護「重要」channel 真的有用、而且用在對的地方——這就是意義。

**Q：為什麼用 RTN 當 base，不直接用真 AWQ？**
RTN 是無最佳化的固定 baseline。AWQ 是「在 RTN 上加 activation-aware 保護」，我要量的是**這個保護
帶來多少改善**，所以需要一個沒有保護的對照組。RTN 就是那個對照組(也等於 scaling α=0)。

**Q：AWQ reduction 的 per-layer ρ(kurtosis) ≈ 0，不是代表沒關係嗎？**
不是。7 類 module 有 5 類是低 outlier、kurtosis 全擠在 ~0.08，形成沒有訊號的 blob，把 rank
correlation 稀釋掉。但 module-family 層級訊號很乾淨：受益最大的 top-8 層全是 down_proj/o_proj。
所以是 categorical 訊號強、per-layer rank 被低 outlier 群稀釋。

**Q：為什麼 `down_proj` / `o_proj` 的 outlier / importance 特別重？**
它們的輸入分別是 MLP 中間 activation(SwiGLU 之後)和 attention 輸出，這些位置已知容易產生大幅值
channel。資料顯示它們的 kurtosis 是其他 projection 的約 35~55 倍。

**Q：這跟真正部署的 AWQ 差在哪？**
真 AWQ 是 group-wise + asymmetric(有 zero-point)+ 把 scale 折進前一層，還有完整的 α 搜尋。我的是
per-output-channel + symmetric 的簡化版，抓的是**機制與直覺**，不是生產級數字。

**Q：下一步最想做什麼？**
把 layer-level 的 AWQ 效益接到 **model-level 品質**(perplexity / logit KL)——證明保護重要 channel
不只降 layer 輸出誤差，而是真的救回 end-task 準確率。

---

### 一段話總結
> AWQ-Diag 是一個理解 + 視覺化 AWQ 的 project。用 forward hook 抓每層 activation，復現 AWQ 的
> importance(`|W|·|x|`)、確認 salient channel 是真的 outlier(集中在 `o_proj`/`down_proj`)、把它
> 畫成 channel×layer 的 3D surface，並**實作 AWQ 的 scaling** 來驗證:保護這些重要 channel 真的把
> 量化誤差降下來，而且**剛好降在 importance 最重的地方**(down_proj 2.3×、單層最高 25.9×)。結論在
> 0.5B 和 1.5B 都成立——activation importance 不只是直覺,而是有操作意義的。
