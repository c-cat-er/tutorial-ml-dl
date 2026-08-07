# 專案一：COVID-19 預測 (未與code整合)

(README1 與 README2 用 ChatGPT 5.5 重點摘要版)

## 專案目標

- 任務類型：利用 <my-orange>PyTorch 深度神經網路 (DNN) 預測 COVID-19 第三天新增確診比例（迴歸問題 Regression）</my-orange>。
- 評估指標：<my-orange>RMSE</my-orange> (Root Mean Squared Error)，<my-orange>數值越小代表模型預測越精準</my-orange>。

---

## 1. 資料前處理 (Dataset)

### 開發重點

- 數據讀取：載入 `covid.train.csv` 與 `covid.test.shuffle.csv`。
- <my-orange>特徵選擇</my-orange> (Feature Selection)：
    - 40 個州別的 One-Hot 編碼特徵。
    - `tested_positive` 相關特徵（位於第 57、75 欄）。
    - 當設定 `target_only=True` 時，精簡為 42 維特徵（Strong Baseline 配置）。??
- 數據切分：將訓練集與驗證集（Dev Set）依照 <my-orange>9:1</my-orange> 的比例進行<my-orange>切分</my-orange>。
- 特徵標準化 (Normalization)：對連續數值特徵進行標準化處理，以加速模型收斂。
- 批次打包：使用 `DataLoader` 包裝資料，且 Batch Size 設為可調整參數。

---

## 2. DNN 模型架構

### 基本架構

```text
Input ──> Linear ──> ReLU ──> Linear ──> Output
```

### 強大基準線 (Strong Baseline) 的優化配置

- **增加深度**：將隱藏層增加至 **3~5 層**（通常建議約 4 層）。
- **彈性維度**：隱藏層維度（Hidden Dimension）設定為可調參數。
- **批次標準化 (BatchNorm)**：穩定層與層之間的數據分佈。
- **隨機丟棄 (Dropout)**：防止模型產生過擬合 (Overfitting)。
- **損失函數 (Loss Function)**：採用 `nn.MSELoss`。
- **權重衰減 (L2 Regularization)**：透過優化器的 `weight_decay` 參數加入阻尼，約束權重大小。

---

## 3. 訓練與執行流程

### 核心函式設計

- `train()`：執行模型的訓練與權重更新。
- `dev()`：在驗證集上評估模型表現。
- `test()`：對未知的測試集進行最終預測。

### 訓練與最佳化技巧

- **優化器選擇**：選用 **Adam**，收斂速度與效果皆優於傳統的 SGD。
- **早停機制 (Early Stopping)**：當驗證集損失不再下降時提早結束，避免無效訓練。
- **模型儲存**：自動監控並儲存表現最佳的權重至 `model.pth`。
- **數據視覺化**：
    - 繪製 **Learning Curve**（訓練與驗證損失隨時間的變化趨勢）。
    - 繪製 **Prediction Scatter Plot**（預測值與真實值的對比散佈圖）。

---

## 4. 關鍵優化與 Bug 修正（期末/面試重點）

1. **特徵篩選 (Feature Selection)**：捨棄冗餘特徵，僅保留 40 州編碼與 2 個關鍵檢測陽性特徵。
2. **建構更深的 DNN**：擴充至約 4 層的結構，並完整搭配 `BatchNorm` 與 `Dropout`。
3. **優化器升級**：全面改用 `Adam` 優化器。
4. **正規化 (Regularization)**：在優化器中開啟 `weight_decay` 實作 L2 正則化。
5. **代碼 Bug 修正**：修正原始程式碼中，當 `target_only=True` 時產生的 `feats` 未定義問題。

---

## 5. 評估指標數學公式

### RMSE (Root Mean Squared Error)

$$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(f(x_i)-y_i)^2}$$

> **重要觀念**：RMSE 的數值越小，代表預測結果越精準。

---

## 6. 完整程式執行順序

1. **Download Data**：下載訓練與測試 CSV 檔。
2. **Import Packages**：匯入 PyTorch、NumPy、Matplotlib 等套件。
3. **Utilities**：定義裝置切換 (`get_device`) 與繪圖工具。
4. **Dataset**：實作 `COVID19Dataset` 類別。
5. **DataLoader**：封裝批次讀取機制。
6. **NeuralNet**：宣告神經網路架構模型。
7. **Training**：撰寫主要訓練迴圈。
8. **Validation**：撰寫驗證計算邏輯。
9. **Testing**：實作測試集預測邏輯。
10. **Setup Hyper-parameters**：配置超參數字典。
11. **Load Data**：宣告實例化的 DataLoaders。
12. **Start Training**：啟動訓練並產出學習曲線。
13. **輸出結果**：將最終預測儲存為 `pred.csv`。

---

## 7. 核心超參數建議配置

| 超參數名稱 (Hyper-parameter) | 建議設定值 | 備註說明                        |
| :--------------------------- | :--------- | :------------------------------ |
| **Epoch**                    | 800        | 提供充足的訓練疊代次數          |
| **Batch Size**               | 256        | 平衡記憶體吞吐量與梯度穩定度    |
| **Learning Rate**            | 0.0005     | Adam 優化器建議搭配較小的學習率 |
| **Optimizer**                | Adam       | 適應性學習率，收斂速度快        |
| **Weight Decay**             | 1e-5 (L2)  | 用於限制模型複雜度，抑制過擬合  |
| **Early Stopping**           | 80 ~ 200   | 當驗證損失持續未改善時切斷訓練  |

---

## ⭐ 考前與面試必背濃縮版

- **基本任務**：基於 PyTorch 的 COVID-19 確診率迴歸預測問題。
- **評估指標**：RMSE (越小越好)。
- **特徵維度**：精簡後的 42 維（40 州 + 2 個 tested_positive）。
- **模型核心**：加深的全連接層 (DNN)（Linear → ReLU → Linear）。
- **Loss**：MSELoss。
- **Optimizer**：Adam。
- **泛化技巧**：搭配 BatchNorm、Dropout、L2 Regularization。
- **控制策略**：實作 Early Stopping 並動態儲存最佳權重模型。
- **最終產出**：產生預測檔案 `pred.csv`。

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
