# HW8

## Task Description

- **網路壓縮**：使用小型模型來模擬大型模型的預測結果/準確率。
- 在這個任務中，您需要訓練一個非常小的模型來完成作業3，也就是對 food-11 資料集進行分類。

## Task-Food Classification

- 這些圖像來自 food-11 資料集，該資料集分為 11 個類別。
- 此處的資料集略有修改：
    - **訓練集**：280 張 \* 11 類已標註圖像 + 6786 張未標註圖像
    - **驗證集**：60 張 \* 11 類已標註影像
    - **測試集**：3347 張圖像
- 請勿使用原始資料集或標籤。

## Intro

- 網路/模型壓縮有很多種類型，這裡我們介紹兩種：
    - **知識蒸餾**：讓小模型透過觀察大模型的學習行為（預測）來學習更好。（簡單來說：讓小模型從大模型中提取知識）
    - **設計架構**：使用較少的參數來表示原始層。（例如：普通卷積 → 深度卷積和逐點卷積）

## Intro-Knowledge Distillation 知識蒸餾

- 訓練小型模型時，可以添加一些來自大型模型的資訊（例如預測的機率分佈），以幫助小型模型更好地學習。
- 我們提供了一個訓練良好的網絡來幫助您進行知識蒸餾（準確率約為 0.788）。
- 請注意，您只能在完成作業時使用我們提供的預訓練模型。

## Intro-Design Architecture 設計架構

- **深度卷積層與逐點卷積層**（MobileNet 中提出）
    - 您可以將原始卷積視為 Dense/Linear 層，但每條線/每個權重都是濾波器，而原始乘法運算則變為卷積運算。（輸入 _ 重量 → 輸入 _ 濾波器）
    - **深度卷積**：每個通道首先通過各自的濾波器，然後每個像素通過共享權重的 Dense/Linear 層。
    - **逐點卷積**：是一個 1x1 卷積。
- 強烈建議您使用類似的技術來設計您的模型。（NMkk / Nkk+NM）

## Regulations

- 嚴禁搜尋或使用其他資料。
- 嚴禁在網路上搜尋標籤或資料集。
- 嚴禁在任何影像資料集上使用預訓練模型。

## Special Regulations-1

- 請確保模型的參數總數**小於或等於 100,000**。
- 請勿將測驗資料用於推理以外的用途。
    - 因為如果您使用教師網絡預測測驗資料的偽標籤，則只能使用學生網路來過擬合這些偽標籤，而無法使用訓練/未標記資料。這樣，您的 Kaggle 準確率將與教師網絡一樣高，但實際上，您只是過度擬合了測試數據，而您的真實測試準確率非常低。
    - 這與本次作業的目的（網路壓縮）相矛盾；因此，您不應濫用測試資料。

## Special Regulations-2

- 我們強烈建議您使用 `torchsummary` 套件來測量模型的參數數量。請注意，**不可訓練的參數也應考慮**。
- 允許使用整合方法（或其他任何多模型方法）。但您需要將所有參數的數量相加，並確保總和不超過 100,000。

## Baseline Guides

- **簡單基線**（2 分，準確率 ≥ 0.54751）
    - 只需運行程式碼並調整學習率。
- **中等基線**（2 分，準確率 ≥ 0.62163）
    - 完善知識蒸餾中的損失函數，並控制 alpha 和 T 值。
- **強基線**（2 分，準確率 ≥ 0.64375）
    - 使用深度卷積層和逐點卷積層修改模型架構。
    - 或者，您可以從 MobileNet、ShuffleNet、DenseNet、SqueezeNet、GhostNet 等模式的優秀想法中學習。
    - 您在 HW3-CNN 中學到的任何技巧和方法。例如，加強資料增強、改進半監督學習等。

## 核心流程

---

## 優化重點摘要

---

## 優化步驟

- 優化後的 StudentNet（使用 Depthwise + Pointwise）
    - 注意：使用 torchsummary 確認參數量必須 ≤ 100,000。
- 優化後的 Knowledge Distillation Loss (loss_fn_kd)
- 建議加強的 Data Augmentation（替換原 train_tfm）(transforms.Compose)
- 訓練設定建議（優化版）

```bash
    # Optimizer 建議改用較小 lr + weight decay
    optimizer = torch.optim.AdamW(student_net.parameters(), lr=3e-4, weight_decay=1e-4)
    # 可加入 scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
```

1. 架構設計：改用 Depthwise + Pointwise (MobileNet 風格) 取代一般 Conv，目標參數 ≤ 100k（用 torchsummary 驗證，含不可訓練參數）。
2. 知識蒸餾：完善 KD Loss，調整 alpha（0.5~0.9）與 T（4~8），結合 teacher soft label 與 hard label。
3. 資料增強：加強 HW3 技巧（RandAugment、Mixup、Cutout、強 ColorJitter）。
4. 半監督學習：善用 6786 張 unlabeled 資料（pseudo-label 或 consistency regularization），嚴禁對 test set 產生 pseudo-label。
5. 訓練策略：使用 Label Smoothing、較長 epoch、cosine decay LR、gradient accumulation。

---

### 評估指標 (Evaluation Metric)

---

## 執行建議

- 替換 StudentNet + loss_fn_kd。
- 先以簡單 CNN + KD 達到中等基線，再替換為 depthwise/pointwise block。
- 執行 summary 確認參數 ≤ 100k。
- 每改一版立即檢查參數量與 val acc。
- 調整 T=6, alpha=0.7 開始訓練，再微調架構。
- 若 val acc 仍不足，再加入加強的 data augmentation。
- 最終 ensemble 時所有模型參數總和仍需 ≤ 100k。

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
