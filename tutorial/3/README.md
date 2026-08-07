# 專案三：Image Classification

## Dataset

- **來源與類別**：影像收集自 food-11 資料集，共分為 11 個類別。
- **訓練集 (Training set)**：280 \* 11 張有標籤影像 (labeled images) + 6786 張無標籤影像 (unlabeled images)。
- **驗證集 (Validation set)**：60 \* 11 張有標籤影像。
- **測試集 (Testing set)**：3347 張影像。

## Task

- 使用**卷積神經網路 (Convolutional Neural Network, CNN)** 進行影像分類。
- 若使用知名模型架構（例如 ResNet），**切勿（NOT）**載入預訓練權重（pre-trained weights）作為初始化。
- 實作**半監督式學習的偽標籤流程 (pseudo-label 流程，即 `get_pseudo_labels`)**、資料增強 (data augmentation) 以及學習率調整與模型改善。
- **基本目標**：利用不同模型架構或資料增強方式來提升有標籤影像的分類效能。
- **進階挑戰 (Hard)**：
    - 利用額外的無標籤影像來改善效能。
    - 獨立完成範例程式碼中的 `TODO` 區塊。
    - 允許在此處使用無標籤的測試資料 (unlabeled testing data)。
- **提示 (Hint)**：可採用半監督式學習 (Semi-supervised learning) 或自監督式學習 (Self-supervised learning)。

---

## 自監督式學習 (Self-supervised learning)

- 完全不需要人工標籤。
- 讓模型自己設計「**預測任務 (pretext task)**」，從資料本身主動產生標籤。
- 模型會先學習資料的特徵表示 (representation)，隨後再利用這些特徵來進行下游任務（例如影像分類）。

---

## 半監督式學習 (Semi-supervised learning, SSL)

- **核心概念**：
  模型在一開始先使用少量的有標籤資料 (labeled data) 進行訓練，接著利用該模型對大量未標籤資料 (unlabeled data) 進行預測，並將預測結果（如偽標籤 pseudo-labels）作為擴充的訓練資料，反覆疊代更新模型。
- **基本流程**：
    1. 利用模型的**信心值 (confidence)** 挑選出預測結果可靠的未標籤資料。
    2. 將這些高信心值的偽標籤 (pseudo-label) 樣本加入原本的訓練集中。
    3. 重複「訓練 → 標記 → 過濾 → 再訓練」的循環。

### 偽標籤生成流程 (Pseudo-labeling 流程)

1. **輸入**：CNN 模型輸入未標籤的影像。
2. **計算**：模型輸出 logits，經過 Softmax 函數計算後得到機率分佈。
3. **門檻過濾**：若最大機率（例如 0.831）超過設定的信心門檻（例如 0.8），則視為「模型夠自信」。
4. **擴充訓練**：將該預測類別作為 pseudo-label（例如 label = 7），合併加入訓練資料中重新訓練。

### SSL 完整循環流程

- **有標籤資料** → 訓練 CNN。
- 模型對**未標籤資料**進行預測與標註 (Labeling)。
- 經過**信心濾除 (Filtering)** → 篩選形成偽標籤子集 (Pseudo-labeled subset)。
- 將此子集與原來的有標籤資料進行**合併 (Combining)**，再度投入重新訓練。

---

## 核心流程

---

## 優化重點摘要

- 有優化的檔案：

---

## 優化步驟 1

(忘了註記優化的區塊)

- 基礎模型：使用 ResNet（不載預訓練權重）+ 適當資料增強（Resize + RandomFlip/Rotation/ColorJitter）。
- 實作 Pseudo-label：完成 get_pseudo_labels，設定信心門檻（0.8~0.95）篩選高信心未標籤資料。
- 半監督迭代：有標籤資料訓練 → 預測未標籤 → 合併高信心偽標籤 → 重新訓練（2~3 輪）。
- 調優：加入 LR Scheduler（CosineAnnealing）、Dropout、Label Smoothing；調整 batch size 與 threshold。
- 評估：以 validation accuracy 為主，追蹤 pseudo-label 準確度與數量變化。

## 優化步驟 2

- 資料增強：使用 RandAugment / AutoAugment + Mixup/CutMix，提升泛化。
- 模型架構：從 ResNet-18/34 開始（不載入預訓權重），逐步加深或加 SE/Attention 模組。
- 偽標籤流程：調整信心門檻（0.8→0.9），逐步加入高信心 unlabeled 資料，並迭代 2-3 輪。
- 學習率排程：使用 Cosine Annealing + Warmup、Label Smoothing、Mixup/CutMix、早停機制，避免 plateau。
- 正則化：增加 Dropout、Label Smoothing、Weight Decay。
- 評估與融合：用驗證集早停 + 模型集成（Ensemble）提升最終分數。

---

### 評估指標 (Evaluation Metric)

---

## 執行建議 1

- 先跑 baseline（僅有標籤資料）再加入 SSL。
- 優先實驗 augmentation 與 threshold 對 pseudo-label 品質的影響。
- 記錄每輪 validation accuracy 與 pseudo 樣本數，避免過度擬合。
- 硬體允許可加大模型深度或使用混合精度加速。

## 執行建議 2

- 先跑 baseline（有標籤資料 + 簡單 CNN），記錄 val acc。
- 逐步加入 pseudo-labeling 與 data aug，觀察 unlabeled 資料貢獻。
- 每輪 pseudo-label 後重新訓練，並用 val set 驗證門檻效果。
- 最後測試集用 TTA（Test Time Augmentation）提升穩定性。

## 執行建議 3

- 先替換 get_pseudo_labels（最重要）
- 改用 ResNet18 模型
- 加入 CosineAnnealingLR
- 調整 threshold（建議 0.80~0.90）

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
