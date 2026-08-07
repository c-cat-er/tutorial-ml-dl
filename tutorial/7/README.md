# 專案七：無監督領域自適應 (UDA，Unsupervised Domain Adversarial Training) (DaNN)

## 任務目標

- **Source Data**：真實照片，5000 張（32×32 RGB），有標籤（10 類，編號 0–9）
- **Target Data**：手畫塗鴉，100000 張（28×28 灰階），無標籤
- **目標**：用 Source 訓練模型，讓模型能正確預測 Target（手繪塗鴉）的類別
- **評估指標**：準確率 (Accuracy)
- **重要提示**：Source 和 Target 的類別分佈都是平衡的（每類數量相同）→ 可用來做強基線技巧（例如用預測結果的類別分佈去校正/約束模型輸出）

## 核心方法：DaNN 原理

1.  **問題本質**：一般 CNN 若測試資料跟訓練資料分佈不同，輸出容易爆走（因為沒看過該分佈的資料，也沒有標籤可學）。
2.  **解法**：把模型拆成兩部分：
    - **Feature Extractor（前半）**：把 Source/Target 圖片轉成 feature
    - **Label Predictor（後半／Classifier）**：根據 feature 做分類
3.  **關鍵技巧**：加入一個 Domain Classifier（判別器）
    - Domain Classifier 負責判斷某個 feature 來自 Source 還是 Target
    - Feature Extractor 則要學會「騙過」Domain Classifier，讓兩個 domain 的 feature 分佈盡量相似（類似 GAN 的對抗訓練）
    - **最終目標**：讓 Source 和 Target 的 feature 落在同一個分佈上，這樣 Label Predictor 對 Target 也能正常工作
4.  **訓練上通常用** Gradient Reversal Layer (GRL) 加上一個 \(\lambda\) 係數來實作這種對抗訓練（文件中提到「在 DaNN 演算法中設定合適的 \(\lambda\) 值」= 中等基線的關鍵）

## 資料格式

```text
train_data/
0/ 0.bmp ... 499.bmp
1/ 500.bmp ... 999.bmp
...
9/

test_data/
0/ 00000.bmp ... 99999.bmp (100000 張，無真實標籤，只是資料夾結構)
```

### 載入方式：

```python
source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
```

## Baseline 要求

| 等級         | 做法                                 |
| :----------- | :----------------------------------- |
| **簡單基線** | 調整學習率                           |
| **中等基線** | DaNN 演算法中設定合適的 $\lambda$ 值 |
| **強基線**   | 利用「測試集標籤平衡」這個額外資訊   |

## 進階提升方向（非必要但可加分）

- 調整優化器 / lr_scheduler
- Model ensemble（整合多個模型或輸出）
- 更進階的對抗式領域適應演算法：MCD、MSDA、DIRT-T
- 半監督式學習 / 無監督式學習（例如通用域適應 Universal Domain Adaptation）

## 規則限制（務必遵守）

- ❌ 不可搜尋或使用額外資料
- ❌ 不可在網路上搜尋標籤或資料集
- ❌ 不可在任何影像資料集上使用預訓練模型

## 核心流程

---

## 優化重點摘要

- 新增校正函數（建議放在 Inference 區塊之前）
- 完整 Inference + 強基線校正區塊（替換原本的 Inference）
- 可選進階版（結合 Temperature Scaling + Iterative Calibration）(為了更穩定)

1. 先完成中等基線：訓練 DaNN 時設定合適的 λ（通常 0.1~1.0 之間，需實驗），讓 Feature Extractor 學到 domain-invariant 的特徵。

2. 全量推論 Target：

- 用訓練好的模型對 100000 張 test_data 做推論，取得 softmax 機率或 logits。
- 計算目前預測的類別分佈（10 類平均預測次數或平均機率）。

3. 分佈校正（Post-processing）：

- 常見做法（Prior Shift Correction）：
    - 對 logits 做調整：adjusted_logits = logits - log(預測類別頻率 + ε)
    - 或對機率做重新縮放：p_corrected = p / avg_pred_dist，再 softmax/renormalize。- 目標：讓校正後的整體預測分佈盡量接近 [0.1, 0.1, ..., 0.1]。

4. 驗證與微調：

- 觀察校正前後的類別分佈是否更平衡。
- 若仍有偏差，可結合溫度縮放（Temperature Scaling）或多次實驗不同 λ 與校正強度。

5. 其他可搭配的強基線技巧：

- 多模型 ensemble 後再做校正。
- 在 pseudo-labeling 時加入平衡約束（例如限制每類 pseudo-label 數量）。

---

## 優化步驟

---

### 評估指標 (Evaluation Metric)

---

## 執行建議

- 先把中等基線（λ 調優）訓練到穩定。
- 替換 Inference 區塊為上方強基線版本。
- 觀察 pred_dist 是否接近 [0.1, 0.1, ...]，若仍偏差可微調 eps 或使用迭代版。

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
