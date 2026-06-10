# AWQ-Diag 面試 cheat sheet（local-only，不進 git）

> 面試官：軟韌體工程師，有基本 ML 知識。重點是講清楚，不是炫技。
> 所有數字都有出處（標在括號裡），被追問時可以打開對應檔案。

---

## 一句話定位

AWQ-Diag **不是新的 quantizer**，是一套診斷工具：直接量測並視覺化 AWQ 的
「activation-aware importance」在真實 LLM 裡長什麼樣子，驗證「保護 salient channels」
是否真的降低量化誤差，再延伸檢驗 AWQ 的 per-layer α search 是否必要。

---

## 90 秒 pitch（對軟韌體工程師的版本）

1. **動機（用他們的語言）**：LLM inference 是 memory-bandwidth bound——瓶頸是把權重從
   DRAM/HBM 搬進晶片，不是算力。把 FP16 權重壓到 4-bit/3-bit，搬運量直接除以 4。
   問題：4-bit 用最簡單的 round-to-nearest (RTN) 幾乎無損，**3-bit 就崩**。
2. **AWQ 的 idea**（MLSys 2024 Best Paper）：不是所有 weight 等價——weight 的重要性
   正比於它乘到的 activation 大小；少數 salient channels 要保護。但 paper 裡這件事
   多半是用結果反推的，我想**直接量測並視覺化它**。
3. **我做了什麼**：用 PyTorch forward hooks 對 Qwen2.5 每一層 Linear（196 層）收
   per-channel 統計，證實三件事——importance 高度集中（top-1% channel 最高占 17.6%）、
   那些 channel 是真的 heavy-tailed outlier（kurtosis ≈ 12）、而且集中在特定模組
   （`o_proj`/`down_proj`）。然後實作 AWQ scaling 驗證：保護那些 channel 把 3-bit
   layer error 平均降 2.3×（單層最高 25.9×），**效果剛好落在 importance 說它該落的地方**。
4. **延伸 finding**：AWQ 校正成本主要在 per-layer α grid search。我量到**單一全域 α
   在 model-level perplexity 上打平甚至更好**——和 SmoothQuant 用固定 α=0.5 的先例一致，
   所以我誠實框成 reproduction-quality 的觀察，不 claim novelty。

---

## 關鍵數字表（背這張）

| 主張 | 數字 | 出處 |
|---|---|---|
| importance 集中（hockey-stick） | top-1% channels 最高占 **17.6%** importance（≈18× 均勻值） | README Key findings / `saliency_curve.png` |
| outlier 是真的 | max excess kurtosis **κ ≈ 12.0**，在 `layers.1.mlp.down_proj` | `diagnostic_*.json` → `top_kurtosis_layer` |
| 集中在哪 | mean kurtosis：`down_proj` **4.31**、`o_proj` **2.86**，其餘 ~0.08–0.09（35–55× 差距） | `summary.module_family` |
| 保護有用（3-bit, vs RTN） | error reduction：`down_proj` **2.31×**、`o_proj` **1.63×**、其他 ~1.1–1.3×；單層最高 **25.9×** | `awq_reduction.png` |
| 跨 size 重現（0.5B） | 同樣 pattern，單層最高 **28.9×** | `cross_model_awq_reduction.png` |
| 3-bit 是懸崖 | median output error：8-bit 8.6e-5 → 6-bit 1.4e-3 → 4-bit **0.022** → 3-bit **0.079** → 2-bit **0.268** | `summary.per_bit_median_output_error` |

**Perplexity（WikiText-2，group-wise asymmetric，group=128，fake quant）：**

| Model / bits | FP16 | RTN | const-α | block-AWQ | 結論 |
|---|---|---|---|---|---|
| 1.5B @ 3-bit | 9.70 | 28.40 | **15.58** (α=0.4) | 16.65 | const-α **贏** |
| 1.5B @ 4-bit | 9.70 | 10.88 | 10.79 | 11.06 | 全部接近；4-bit RTN 已近無損 |
| 0.5B @ 3-bit | 13.12 | 51.85 | 27.12 (α=0.35) | 27.31 | 打平 |
| 0.5B @ 4-bit | 13.12 | 15.61 | 14.89 | 14.88 | 打平 |

**α study（layer-level，3-bit，1.5B，196 層 × 21-point grid）**：best-α median 0.2、
全域 const-α 只捕捉 full search **77%** 的 layer-level reduction——
**但傳到 model-level perplexity 時這個 gap 消失甚至反轉**。

> 這個對比是最好的深度展示點：per-layer search 最小化的是 **layer-local error**，
> 每層各自最優 ≠ end-to-end 最優（greedy local optimization 不保證全域最好）。
> 韌體工程師對「local optimum ≠ global optimum」很有感。

---

## 名詞速答（他們最可能問的基礎題）

- **為什麼量化會變快？** Decode 階段每生成一個 token 都要把全部權重讀一遍，
  算術強度低 → memory-bound。權重壓到 1/4 大小 ≈ 搬運時間省 1/4。
  Dequant 在 on-chip 做（W4A16：權重 4-bit 存、載入後還原成 FP16 做矩陣乘），
  開銷遠小於省下的頻寬。
- **RTN**：round-to-nearest，最樸素的量化——scale 之後直接四捨五入。
- **symmetric vs asymmetric**：symmetric 只有 scale（零點固定在 0）；asymmetric 加
  zero-point，能貼合不對稱分佈。**per-channel vs group-wise**：scale 的粒度——每個
  output channel 一組 vs 每 128 個 weight 一組（group-wise 是實際部署的標準做法）。
- **我用哪個？** 診斷部分用 symmetric per-output-channel（隔離機制、乾淨）；
  perplexity 延伸用 group-wise asymmetric g=128（貼近真實部署）。
- **AWQ scaling 的數學**：選 per-input-channel scale `s = (mean|x|)^α`，
  把 `W → W·diag(s)`、`x → x/s`——浮點下數學等價，但量化 `W·s` 時 salient channel
  的數值被放大、佔住更多量化格點，有效解析度變高。`α=0` 退化成 RTN。
- **為什麼不直接把 outlier channel 留 FP16（mixed precision）？** 那是 LLM.int8() 的
  做法，hardware-unfriendly（kernel 要處理兩種格式）。AWQ 用 scaling 維持 uniform INT
  格式——**這正是它對部署友善的原因**，對韌體面試官特別值得講。
- **perplexity**：exp(平均 negative log-likelihood)，越低越好；WikiText-2 是標準語料。
- **kurtosis 為什麼用？** 量 heavy-tail 程度，Gaussian 的 excess kurtosis = 0；
  κ≈12 表示分佈尾巴極重，少數 channel 的值遠超其他。
- **calibration set 多大？** 4 段文字——誠實說很小；但 AWQ 本身就標榜 data-efficient，
  且 importance 統計（mean|x|）對 calibration 集不敏感（paper 的 claim，與我觀察一致）。

---

## 預期深挖 Q&A

**Q: 你的 AWQ 實作和官方差在哪？**
A: 三點：(1) 沒做 weight clipping（官方 AWQ 的第二個 trick）；(2) scale 沒有 fold 進
前一層/LayerNorm，是模擬等價變換；(3) 全程 fake quant，沒有 INT kernel，所以只量
quality 不量 speed。另外官方 `llm-awq` 在 Qwen2.5 + 新版 transformers 上跑不起來，
所以 block-level baseline 是我照官方邏輯（per-block 共用 α grid search）faithful 重實作的。

**Q: const-α 贏過 search 不是很奇怪嗎？**
A: 不奇怪，有三個理由：(1) search 最小化的是 block-local output error，不是 perplexity——
optimization target 和 evaluation metric 不同；(2) SmoothQuant 用全域固定 α=0.5 早就
work，這是一致的先例；(3) 我只在 Qwen2.5 ≤1.5B 驗證，不 claim 普適——更大的 model、
更極端的 bit 數可能不同。

**Q: 為什麼 outlier 集中在 o_proj / down_proj？**
A: 觀察上，這兩個模組的輸入分別是 attention 的加權輸出和 SwiGLU 的逐元素乘積——
都是經過非線性互動後的訊號，outlier 在這裡聚集。「為什麼 LLM 會長出 outlier channel」
本身是 open question（LLM.int8()/SmoothQuant 都觀察到，與訓練動態有關），
我的貢獻是把「它在哪、多嚴重」量出來。

**Q: 為什麼選 Qwen2.5？**
A: 夠新、有多個小 size 可以做 cross-size replication、跑得進我的單卡（RTX 50 系列）。
0.5B 和 1.5B 的 pattern 一致，增加結論可信度。

**Q: 如果要部署到 edge / 自家晶片，你會怎麼做？**
A: 順序是：(1) 加 weight clipping 補齊 AWQ；(2) 真的 pack 成 INT4 + 寫 dequant kernel
（GPU 上是 Triton W4A16；MCU/NPU 上看 ISA 支援什麼粒度的 unpack）；(3) 量實際
latency/throughput 而不只 perplexity；(4) group size 和 zero-point 格式要配合硬體的
對齊方式選。我目前的 repo 故意停在 quality 層，因為要先確定「機制是對的」再花力氣寫 kernel。

**Q: 你量到的 layer error 降 2.3×，這對使用者有什麼意義？**
A: 單看沒意義，所以我才做 perplexity 延伸把它接到 end-to-end：3-bit 時 AWQ 把
perplexity 從 RTN 的 28.4 拉回 15.6（FP16 是 9.7）——模型從「不能用」變「可用」。
這正是 README 從 layer-level 走到 model-level 的原因。

---

## 被問「接下來想做什麼」的標準答案

1. **Rotation-based 方法的 diagnostic（首選，講這個）**：我量到 outlier 集中在少數
   channel，AWQ 的對策是「保護」它們；2024 後的主流（QuaRot、SpinQuant）改用 rotation
   （Hadamard transform）把 outlier **打散**，讓分佈均勻好量化。我的 hook 工具鏈可以
   直接量 rotation 前後 kurtosis / importance 集中度的變化——驗證「打散」是不是真的發生、
   發生在哪些模組。這把新方向掛在我自己的量測上。
2. **對韌體面試官加碼**：寫 Triton W4A16 dequant kernel、量實際 throughput——
   現在我只證明了 error 面，還沒證明 speed 面。
3. 更多 model families（Llama/Gemma/Phi）驗證 `o_proj`/`down_proj` 集中是否普適。

---

## 誠實雷區（不要說過頭）

- ❌「我發明了 search-free AWQ」→ ✅「我量測到 per-layer search 在這個 setting 下
  不必要，與 SmoothQuant 的全域 α 先例一致」。
- ❌ 任何加速 claim——全程 fake quant，沒量過 latency。
- ❌ 普適性 claim——一個 model family（兩個 size）、4 段 calibration 文字、
  WikiText-2 一個 benchmark。
- 被問倒時的標準回法：「這個我沒量過，但我的工具可以直接量——做法會是…」
  （把不知道轉成 experiment design，這是 diagnostic toolkit 的優勢。）
