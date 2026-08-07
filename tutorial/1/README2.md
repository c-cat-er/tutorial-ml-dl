## 1. 資料下載與環境準備

1. Download Data：使用 gdown 從雲端下載<my-orange>訓練集</my-orange>（covid.train.csv）與<my-orange>測試集</my-orange>（covid.test.csv）資料。
2. Import Some Packages：載入核心套件，包括<my-orange>深度學習框架 torch、數據處理的 numpy 與 csv，以及繪圖工具 matplotlib</my-orange>。同時設定隨機種子（myseed = 42069）以確保實驗的可重複性。

## 2. 工具函式與資料處理

3. Some Utilities：
    - get_device()：自動偵測並回傳可用的硬體設備（GPU cuda 或 cpu）。
    - plot_learning_curve()：繪製訓練與驗證階段的損失曲線（Learning Curve）。
    - plot_pred()：繪製真實值與預測值的散佈圖，用以評估模型擬合狀況。
4. Dataset：
    - 定義繼承自 PyTorch 的 COVID19Dataset 類別，負責讀取 CSV、特徵選擇（包含 40 個州和 2 個檢測陽性特徵的強基準線配置）、資料分割（9:1 拆分訓練與驗證集）以及特徵標準化處理。
5. DataLoader：
    - 定義 prep_dataloader() 函式，將自訂的 Dataset 包裝成 PyTorch 的 DataLoader，以利批次（Batch）讀取和隨機打亂資料。

## 3. 模型設計與核心算法

6. Deep Neural Network：
    - 建立 NeuralNet 類別。這是一個多層的全連接神經網路（Fully-Connected DNN），其架構包含：
        - 線性層（Linear）：維度由 input_dim → 64 → 32 → 16 → 1。
        - 批次正規化（BatchNorm1d）與非線性激活函數（ReLU）。
        - 丟棄法（Dropout）：防止模型過擬合。
        - 損失函數：採用均方誤差損失（nn.MSELoss）。

## 4. 訓練、驗證與測試流程

7. Training：
    - train() 函式負責完整的訓練循環（Training Loop），包含梯度歸零、前向傳播、計算損失、反向傳播、更新權重，並具備早停機制（Early Stopping）與最佳模型權重儲存功能。
8. Validation：
    - dev() 函式在每個 Epoch 結束後評估模型在驗證集上的表現，此時會關閉梯度計算（torch.no_grad()）。
9. Testing：
    - test() 函式對不帶標籤的測試集進行預測並回傳結果。

## 5. 主程式執行流

10. Setup Hyper-parameters：

- 設定超參數字典 config（如 800 個 Epochs、Batch size 256、Adam 優化器、學習率 0.0005、L2 正則化權重衰減、早停門檻 80 次等）。

11. Load data and model：建立訓練、驗證、測試的三個 DataLoader，並實例化神經網路模型移至指定硬體。
12. Start Training!：執行訓練、繪製損失曲線，隨後重新載入表現最佳的模型權重，並繪製預測散佈圖。
13. Testing：調用測試函式，並透過 save_pred() 將最終的預測結果輸出並儲存為 pred.csv 檔案。

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
