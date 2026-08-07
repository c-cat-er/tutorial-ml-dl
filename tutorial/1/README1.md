# 專案一機器學習：COVID-19 趨勢預測，使用 PyTorch DNN 解決迴歸問題

(參考 Google 雲端檔案，專案一\_重點簡述)

- 根據美國各州過去 3 天的資料，<my-cyan>預測</my-cyan>第 3 天新增檢測陽性病例的百分比（<my-cyan>RMSE 評估</my-cyan>）。
- 先<my-orange>特徵選擇</my-orange>，再逐步優化架構、<my-orange>正規化與訓練參數</my-orange>，即可有效<my-orange>降低 RMSE</my-orange>。

---

## 核心架構流程

### 1. 資料前處理 (COVID19Dataset)

- 讀取 `covid.train.csv` / `covid.test.csv`。
- **特徵選擇 (Medium Baseline)**：使用 40 個州別的 one-hot（索引 0~39）加上 2 個 `tested_positive` 特徵（索引 57、75），共 42 維（`target_only=True`）。
- 訓練/驗證分割 (9:1)、<my-orange>特徵正規化</my-orange>（mean/std，只對後續特徵）。
- DataLoader 打包（<my-orange>batch size 可調</my-orange>）。

### 2. 模型架構 (NeuralNet)

- **簡單全連接 DNN**：Linear $\rightarrow$ ReLU $\rightarrow$ Linear。
- **Strong Baseline 改進**：
    - <my-orange>增加層數</my-orange>（例如 3~5 層）、<my-orange>調整 hidden dim</my-orange>（64 $\rightarrow$ 128 $\rightarrow$ 64 或更大）。
    - 使用 <my-orange>BatchNorm1d + Dropout</my-orange> 提升穩定性與泛化。
    - 輸出層 squeeze 成 1 維。
- **Loss**：`MSELoss`，可加入 <my-yellow>L2 正規化 (weight_decay)</my-yellow>。

### 3. 訓練與驗證

- `train()`：<my-orange>SGD/Adam + momentum, early stopping（patience 200）</my-orange>。
- `dev()`：計算驗證集平均 MSE。
- 儲存最佳模型 (`model.pth`)。
- 繪製 learning curve 與 prediction scatter。

### 4. 測試與輸出

- `test()` 產生預測值，存成 `pred.csv`（含 id 與 `tested_positive`）。

### 5. 超參數調整重點 (config)

- `batch_size`、`lr`、`optimizer`（建議試 Adam）、`n_epochs`、`early_stop`。
- 修正樣例程式可能的 bug（例如正規化範圍、索引錯誤）。

---

## 優化重點摘要

- **特徵選擇**：40 個州 + 2 個 `tested_positive` (索引 57、75) $\rightarrow$ `target_only=True`。
- **更強的 DNN 架構**：<my-orange>4 層 + BatchNorm + Dropout（提升泛化）</my-orange>。
- **訓練優化**：改用 <my-orange>Adam + `weight_decay` (L2 正規化) + 適當 `learning rate`</my-orange>。
- **Bug 修正**：修正樣例程式 `target_only=True` 時 `feats` 未定義的問題。
- **其餘流程**：保留並微調參數（early stopping、儲存最佳模型、繪圖）。

---

## 優化步驟

- 有優化的檔案：`Dataset`、`Deep Neural Network`、`Setup Hyper-parameters`

---

### 評估指標 (Evaluation Metric)

1. **Root Mean Squared Error (<my-cyan>RMSE</my-cyan>)**:
   $$RMSE = \sqrt{\frac{1}{N}\sum_{n=1}^{N}(f(x^{n})-\hat{y}^{n})^{2}}$$
    - $f$: 您的模型 (your model)
    - $x^{n}$: 輸入特徵 (input features / testing data)
    - $\hat{y}^{n}$: 真實標籤 (ground truth label / correct answer)

---

## 執行建議

1. 全部替換後，先執行到 **Cell 11** 載入資料，確認印出 `dim = 42`。
2. 開始訓練（**Cell 12**）。
3. 訓練完後執行 **Cell 13** 產生 `pred.csv` 提交。
4. 如果想再衝更好成績，可以再試：
    - 把 `target_only=False` + 更深網路。
    - 調整 `Dropout` / `weight_decay` / `lr`。
    - 把 `batch_size` 改 128 或 512 再跑幾次。

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
