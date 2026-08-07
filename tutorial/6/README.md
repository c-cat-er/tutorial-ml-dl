# 專案六：異常檢測 (Anomaly Detection)

## 1. 任務目標 (Goal)

- 利用自編碼器（Autoencoder）進行影像異常檢測。
- **無監督異常檢測**：利用機器學習模型判斷測試影像與訓練影像是否屬於同一類別（分佈）。
- **訓練與推理邏輯**：模型**僅使用「正常資料」進行訓練**，之後需要判斷出哪些是正常 (Seen Normal)、哪些是異常 (Unseen Anomaly)。

## 2. 資料集說明 (Data)

- **訓練集**：約 **14 萬張人臉圖像**（尺寸：`64*64*3`，代表影像高度 64 像素、寬度 64 像素、RGB 3 顏色通道）。
- **測試集**：
    - **正常圖像**：1 萬張與訓練集分佈相同的正常資料（**類別標籤：0**）。
    - **異常圖像**：1 萬張來自其他分佈的人臉圖像（**類別標籤：1**）。
- **限制條件**：**禁止**使用額外的訓練資料和預訓練模型。
- **資料格式與路徑**：
    - 解壓縮命令：`tar zxvf data-bin.tar.gz`
    - 檔案路徑：`data-bin/trainingset.npy`、`data-bin/testingset.npy`

## 3. 核心方法：自編碼器 (Method - Autoencoder)

- **訓練階段 (Training)**：輸入正常資料，訓練編碼器（Encoder）與解碼器（Decoder），使重建影像與輸入影像越接近越好 (As close as possible)。
- **測試階段 (Testing)**：當輸入為異常狀況 (Anomaly) 時，模型無法有效重建 (Cannot be reconstructed)。
- **停止訓練時機**：當**均方誤差損失（MSE Loss）收斂**時，訓練應該停止。
- **異常值定義（推理過程）**：計算「輸入影像」與「重建影像」之間的**重建誤差**，此誤差稱為**異常值（異常分數）**。影像的異常值可以衡量其分佈在訓練過程中未被觀測到的可能性，因此將其用作預測值。

### 卷積自編碼器 (Conv Autoencoder)

- **結構**：編碼器結構由「卷積層 + pooling」構成。
- **特點**：
    - 保留圖像的空間特徵（邊緣、形狀、區塊）。
    - 壓縮更有效。
    - 非常適合 image anomaly detection、denoising、feature extraction。
- **參考資源**：[GitHub - PyTorch CIFAR-10 Autoencoder](https://github.com)

## 4. 評估指標 (Metric - ROC_AUC score)

- **不使用準確率（Accuracy）的原因**：本任務的模型充當**感測器（或偵測器）**而非分類器。若使用準確率，則需要嘗試單一模型的所有可能閾值才能獲得令人滿意的分數。然而，作業想要的是一個在所有可能閾值下，平均準確率最高的感測器。
- **優秀感測器的標準**：
    - 對異常資料賦予**高異常值**，對正常資料賦予**低異常值**。
    - 兩組數據的得分應有較大差距。
- **ROC 曲線與 AUC**：
    - **ROC 曲線**：適用於此任務。曲線上的每個點代表特定閾值下的真陽性率（TPR）和假陽性率（FPR）。
    - **AUC**：ROC 曲線下的面積，用於衡量模型的一般效能。
    - **計算步驟**：用 Anomaly Score **由高到低排序** → 逐筆計算未正規化的 fp (false positive) 和 tp (true positive) → 正規化 → 畫出 ROC curve → 計算 AUC（講義範例之 Area Under Curve 計算結果為 0.8333）。

## 5. Baseline 通過指南 (Baseline Guides)

- **簡單 (Simple Baseline)**：調整學習率。
- **中等 (Medium Baseline)**：
    - 使用 **CNN 自編碼器**。
    - 嘗試更小的模型（更少的層）。
    - 使用 Smaller batch size。
- **強 (Strong Baseline)**：
    - 加入 **BatchNorm**（批次正規化）。
    - 延長訓練時間。

## 核心流程

---

## 優化重點摘要

- 有優化的檔案：

---

## 優化步驟

- 6 Training 超參數設定區塊（6.1） — 最需優先修改
- 4 Autoencoder - Models & loss CNN 自編碼器模型區塊（加入 BatchNorm 版 — Strong Baseline）
- 6.2 Training loop 訓練迴圈區塊（建議微調）

1. Simple Baseline：調整學習率（最優先）。
2. Medium Baseline：

- 改用 CNN 自編碼器（保留空間特徵）。
- 試更小模型（較少層數）+ Smaller batch size。

3. Strong Baseline：

- 加入 BatchNorm。
- 延長訓練時間，直到 MSE Loss 收斂再停止。
- 推理時以「輸入 vs 重建影像」的 MSE 重建誤差 作為異常分數（Anomaly Score）。

---

### 評估指標 (Evaluation Metric)

---

## 執行建議

- 資料：只用 trainingset.npy（正常）訓練，測試集用 testingset.npy 計算 ROC-AUC。
- 模型：Conv Autoencoder（Encoder: Conv+Pool；Decoder: Upsample+Conv）。
- 訓練：MSE Loss，正常資料重建越好越好；收斂即停。
- 評估：異常分數由高到低排序 → 計算 ROC-AUC（目標分數越高越好，正常/異常分數需有明顯差距）。
- 限制：禁止額外資料或預訓練模型。

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
