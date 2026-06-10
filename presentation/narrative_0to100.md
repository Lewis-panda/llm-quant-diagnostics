# 投影片敘事弧線：0 → 100

> 給觀眾的設定：軟韌體工程師，有基本 ML 知識，不是量化專家。
> 原則：每一段都回答「所以呢？」，每個主張都掛一張圖或一個數字。
> 圖的狀態標記：✅ = repo 已有、🖌️ = 投影片裡畫的示意圖（不是實驗）。

---

## 前段（0→30）：為什麼有這個問題

**1. 動機 — LLM inference 是 memory-bound** 🖌️
Decode 階段每生成一個 token 都要把全部權重讀一遍，算術強度低，瓶頸是
memory bandwidth 不是算力。權重壓到 4-bit = 搬運量 ÷4。
（示意圖：權重從 DRAM/HBM 搬進晶片的頻寬瓶頸；FP16 vs INT4 的模型大小對比。）

**2. 量化是什麼** 🖌️
FP16 → 低 bit 整數的數線映射：scale、rounding、格點數 = 2^bits。
4-bit 16 格、3-bit 8 格。每少一個 bit，格點砍半。
（示意圖：數線上的格點與 rounding error。）

**3. 最樸素的做法（RTN）能撐到哪裡？** ✅ `figures/error_vs_bits.png`
左欄：layer output error 隨 bit 數平滑指數成長（每少一 bit ≈ ×3–4）。
右欄：但 model-level perplexity 有一個門檻——**4-bit RTN 還可用
（ppl 9.7 → 10.9），3-bit 直接崩（→ 28.4）**。layer error 是平滑的，
模型品質的容忍度是非線性的。

**橋樑句（照念）：**
> 「Group-wise 4-bit RTN 已接近無損，所以往下推到 3-bit 才是 frontier；
> 但 3-bit RTN 崩潰。要活在 3-bit，需要比 RTN 聰明的方法——AWQ 是其中
> 一種，而我的 project 就是去驗證它的核心機制是不是真的。」

**4. AWQ 的 idea** 🖌️
不是所有 weight 等價——weight 的重要性正比於它乘到的 activation。
做法：把 salient channel 先放大再量化（`W·s`, `x/s`，`s=(mean|x|)^α`），
浮點下等價，但量化時 salient weight 佔住更多格點。
（示意圖：AWQ paper Fig. 2 風格，自己重畫。）
關鍵轉折：**paper 的證據多是用結果反推的——我想直接量測它。**

---

## 中段（30→80）：我量到了什麼（全部有現成圖）

**5. 工具怎麼做** 🖌️
PyTorch forward hooks → 每層 Linear（196 層）收 per-input-channel 統計
→ bit sweep {8,6,4,3,2} → AWQ α search。
（示意圖：pipeline 流程圖。）

**6. Importance 高度集中（hockey-stick）** ✅ `saliency_curve.png`
top-1% channels 最高占 17.6% importance（≈18× 均勻值）。

**7. 你可以親眼看到它** ✅ `importance_surface_down_proj.png`
3D 曲面：x=channel、y=layer、z=importance。spiky towers = salient channels。
**這張是全場主視覺，停留久一點。**

**8. Outlier 是真的、而且住在特定地方** ✅ `module_family.png`（+ 備用 `kurtosis_by_layer.png`）
max kurtosis κ≈12（`layers.1.mlp.down_proj`）；mean kurtosis：
`down_proj` 4.31、`o_proj` 2.86，其餘 ~0.08（35–55× 差距）。
為什麼是這兩個：它們的輸入是 attention 輸出 / SwiGLU 乘積——非線性互動後的訊號。

**9. 保護它們真的有用** ✅ `awq_reduction.png`
3-bit error reduction vs RTN：`down_proj` 2.31×、`o_proj` 1.63×、
其他 ~1.2×，單層最高 25.9×。
**punchline：效果剛好落在 importance 說它該落的地方——importance 不只是直覺，是可操作的訊號。**

**10. 跨 size 重現** ✅ `cross_model_awq_reduction.png`
0.5B 同樣 pattern（單層最高 28.9×）。

---

## 後段（80→100）：延伸 finding 與下一步

**11. 那 α search 是必要的嗎？** ✅ `perplexity_search_free_awq.png`
AWQ 校正成本主要在 per-layer α grid search。量 model-level perplexity：
**單一全域 α 在每個 config 打平甚至贏過 block-level search**
（1.5B 3-bit：15.6 vs 16.7）。

**12. 深度點（最值得講的對比）** ✅ `alpha_study.png`
layer-level 上 const-α 只捕捉 full search 77% 的 reduction，
但 model-level 上 gap 消失甚至反轉——**每層各自最優 ≠ 端到端最優**
（search 的 optimization target 是 block-local error，不是 perplexity）。
誠實框架：與 SmoothQuant 全域 α=0.5 先例一致，不 claim novelty；
只在 Qwen2.5 ≤1.5B 驗證。

**13. Limitations + 下一步**
- 沒做 weight clipping、fake quant 沒量 latency、一個 model family。
- 2-bit 連 AWQ 都救不了（layer error 0.268）→ 「保護」不夠，要「打散」：
  rotation-based 方法（QuaRot/SpinQuant）——我的 hook 工具鏈可以直接量
  rotation 前後 outlier 結構的變化，這是下一步。
- 對韌體聽眾加碼：Triton W4A16 dequant kernel + 實測 throughput。

---

## 圖總表

| 順序 | 圖 | 狀態 |
|---|---|---|
| 1–2 | 動機 / 量化基礎示意 | 🖌️ 投影片畫 |
| 3 | `figures/error_vs_bits.png`（3-bit cliff，兩欄） | ✅ `scripts/plot_bit_error.py` 生成 |
| 4 | AWQ scaling 機制示意 | 🖌️ 投影片畫 |
| 5 | pipeline 流程圖 | 🖌️ 投影片畫 |
| 6 | `figures/Qwen2.5-1.5B/saliency_curve.png` | ✅ |
| 7 | `figures/Qwen2.5-1.5B/importance_surface_down_proj.png` | ✅ 主視覺 |
| 8 | `figures/Qwen2.5-1.5B/module_family.png`（備用 `kurtosis_by_layer.png`） | ✅ |
| 9 | `figures/Qwen2.5-1.5B/awq_reduction.png` | ✅ |
| 10 | `figures/cross_model_awq_reduction.png` | ✅ |
| 11 | `figures/perplexity_search_free_awq.png` | ✅ |
| 12 | `figures/Qwen2.5-1.5B/alpha_study.png` | ✅ |

注意：`error_vs_bits.png` 左欄是 symmetric per-channel（診斷設定）、右欄是
group-wise asymmetric g=128（部署設定）——兩個 quantizer 不同，被問到要說清楚
（圖上小標題已標明）。
