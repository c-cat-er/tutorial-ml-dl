# 專案二：Framewise Phoneme Classification (逐幀音素分類)

(參考 Google 雲端檔案，專案二\_重點簡述)

## 任務目標 (Task)

- <my-cyan>**多類別分類 (Multiclass Classification)**</my-cyan>：從語音訊號中進行<my-orange>逐幀音素預測</my-orange>。本作業將訓練一個深度神經網路分類器，以預測 TIMIT 語音語料庫中每個畫面的音素。
- **什麼是<my-cyan>音素 (Phoneme)</my-cyan>？** <my-orange>語言中用來區分單字的最小語音單位</my-orange>。
    - `bat` → `b` `at`；`bad` → `ba` `d`。
    - `MachineLearning` → `M` `AH` `SH` `IH` `N` `L` `ER` `N` `IH` `NG`。

---

## 資料預處理 (Data Preprocessing)

### 1. 語音切片 + 特徵序列

- 將原始語音訊號<my-orange>切成多個 **frame**（使用 **25ms** 的視窗，每 **10ms** 移動一次）</my-orange>。
- <my-orange>每個 frame 提取出 d 維的特徵向量（例如：**39 維 MFCC** 或 **80 維 filter bank**）</my-orange>。
- 最終的結果是一個長度為 T × 維度 d 的矩陣，此即為<my-orange>**輸入模型的資料形式**</my-orange>。

### 2. MFCC 特徵抽取細節 (Mel Frequency Cepstral Coefficients)

「為什麼每個 frame 可以變成一個 d 維向量」的聲學特徵轉換流程：

1. <my-orange>**原始語音 (Waveform)** → 透過 **DFT (離散傅立葉轉換)** 進行頻譜分析</my-orange>。
2. 經過 <my-orange>**Mel filter bank** 模擬人耳對頻率的聽覺感知</my-orange>。
3. <my-orange>取 **log** → 再透過 **DCT (離散餘弦變換)** 來壓縮資訊並去除特徵間的相關性</my-orange>。
4. 最終得到<my-orange>每個 frame 專屬的 **MFCC 向量**</my-orange>。

### 3. 上下文視窗 (Context Window)

- **設計目的**：單一 frame（僅包含 25ms 語音）的資訊不足以代表完整音素，因為一個音素通常跨越多個幀。為了<my-orange>**增加時序上下文資訊**</my-orange>，會將「前後幾個 frame」<my-orange>串接（concatenate）</my-orange>看。
- **本作業設定**：串接過去與未來的 5 幀（**前 5 + 當前 1 + 後 5 = 共 11 個 frame**）。
    - 每個 frame 為 39 維 → 串接後特徵維度為 $11 \times 39 = \mathbf{429}$ **維**。
    - 輸入形狀為 `(1, 429)`，可將其重新整形（reshape）為 `(11, 39)` 以獲得分離的 11 幀。
- **注意**：**標籤（Label）僅對應於中央框架（中心幀）**，上下文的設計僅是用來輔助分類器更準確地判斷該中心幀的音素。
- **提示**：由於音素具備時序連續性，<my-orange>**後處理（Post-processing）**</my-orange>可能會對提升表現有所幫助。

---

## 資料集與格式 (Dataset & DataFormat)

- **資料集**：TIMIT Acoustic-Phonetic Continuous Speech Corpus（助教已進行預處理，目錄為 `timit_11/`）。
- **資料格式**：
    - **訓練資料 (`train_11.npy`)**：包含 **922,449** 個訓練樣本，形狀為 `(# of training frames, 429)`。
    - **訓練標籤 (`train_label_11.npy`)**：逐幀的音素標籤，類別為 **0-38（共 39 類）**。
    - **測試資料 (`test_11.npy`)**：包含 **307,483** 個測試樣本，只有 MFCC 特徵，**不提供標籤（Label）**。
- **評估指標**：<my-orange>**分類準確率 (Accuracy)**</my-orange>。
  $$\text{Acc} = \frac{\text{正確預測數}}{\text{總樣本數}}$$

### 規範限制

- **嚴禁人工標註**：禁止尋找測試標籤、偷看 label 或透過人工聆聽音檔進行標註。

---

## 39 類目標音素對照表 (Label Set)

每個 frame 輸出的 Label 均為以下 39 個類別之一：

| Class  | Phoneme | Example  | Class  | Phoneme |  Example  | Class  | Phoneme |     Example     |
| :----: | :-----: | :------: | :----: | :-----: | :-------: | :----: | :-----: | :-------------: |
| **0**  |   iy    | b**ee**t | **13** |    l    |  **l**ay  | **26** |   dx    |    mu**dd**y    |
| **1**  |   ih    | b**i**t  | **14** |    r    |  **r**ay  | **27** |    g    |     **g**ay     |
| **2**  |   eh    | b**e**t  | **15** |    y    | **y**acht | **28** |    p    |     **p**ea     |
| **3**  |   ae    | b**a**t  | **16** |    w    |  **w**ay  | **29** |    t    |     **t**ea     |
| **4**  |   ah    | b**u**t  | **17** |   er    | b**i**rd  | **30** |    k    |     **k**ey     |
| **5**  |   uw    | b**oo**t | **18** |    m    |  **m**om  | **31** |    z    |    **z**one     |
| **6**  |   uh    | b**oo**k | **19** |    n    | **n**oon  | **32** |    v    |     **v**an     |
| **7**  |   aa    | b**o**b  | **20** |   ng    | si**ng**  | **33** |    f    |     **f**in     |
| **8**  |   ey    | b**ai**t | **21** |   ch    | **ch**oke | **34** |   th    |    **th**in     |
| **9**  |   ay    | b**i**te | **22** |   jh    | **j**oke  | **35** |    s    |     **s**ea     |
| **10** |   oy    | b**oy**  | **23** |   dh    | **th**en  | **36** |   sh    |     **sh**e     |
| **11** |   aw    | b**ou**t | **24** |    b    |  **b**ee  | **37** |   hh    |     **h**ay     |
| **12** |   ow    | b**oa**t | **25** |    d    |  **d**ay  | **38** |   sil   | silence/closure |

---

## 基準線與最佳化方向 (Baseline Guidelines)

- **Simple baseline**
    - 調整<my-orange>學習率 (Learning Rate, lr)</my-orange>。
- **Strong baseline**
    - **模型架構 (Model Architecture)**：設計<my-orange>多層隱藏層（Layers）、調整每層維度（Dimension），並嘗試不同的激活函數（Activation function，如 ReLU, LeakyReLU, Swish 等）</my-orange>。
    - **模型訓練 (Training)**：調整<my-orange>批次大小（Batch size）、更換優化器（Optimizer，如 Adam, AdamW）、調整學習率排程與增加訓練輪數（Epoch）</my-orange>。
    - **實用技巧 (Tips)**：加入<my-orange>批次正規化（Batch Normalization）、暫退法（Dropout）以及權重衰減（Regularization / Weight Decay）以防止模型過擬合（Overfitting）</my-orange>。

## 核心流程

---

## 優化重點摘要

檔案?: 優化甚麼?

1. 基礎調整（Simple Baseline）
   調整 Learning Rate（建議從 1e-3 或 5e-4 開始，搭配 Adam）
   先用較小 epoch 快速驗證（e.g. 10~20）
2. 模型架構優化（Strong Baseline 重點）
   設計 多層隱藏層（建議 3~5 層）
   每層維度建議：429 → 1024 → 512 → 256 → 128 → 39
   激活函數嘗試：ReLU / LeakyReLU / Swish
   輸入可 reshape 為 (11, 39) 再用 1D-CNN 或直接 flatten
3. 訓練技巧調整
   Batch size：32~128（依顯存調整）
   Optimizer：Adam → AdamW（加 weight decay）
   Learning rate schedule：ReduceLROnPlateau 或 CosineAnnealing
   Epoch：增加至 50~100，並加入 early stopping
4. 防止 Overfitting（實用技巧）
   加入 Batch Normalization（每層後）
   加入 Dropout（0.2~0.5，建議 hidden layer 後）
   Weight Decay（1e-4 ~ 1e-5）
   可考慮 Label Smoothing（尤其是 39 類問題）
5. 後處理建議（可提升準確率）
   音素具時序連續性，可對預測結果做簡單 smoothing 或 majority vote
   建議優先順序：
   Learning Rate → 模型深度/寬度 → BatchNorm + Dropout → Optimizer + LR schedule

## 優化步驟

### 1. Create Model

- 增加第 4 層隱藏層（提升模型容量）
- 調整維度為 429 → 1024 → 512 → 256 → 128 → 39
- Dropout 比例微調

### 2. 參數

- num_epoch = 80
- learning_rate = 5e-4
- BATCH_SIZE = 128 # 可依顯存調整為 64~256
- weight_decay = 5e-4

### 3. kfold

```bash
   optimizer = torch.optim.AdamW(
      model.parameters(), lr=learning_rate,
      weight_decay=weight_decay
   )
   criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
   scheduler = ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5,
      patience=6, min_lr=1e-6, verbose=True
   )
```

## early_stopping = EarlyStopping(patience=20, verbose=True, delta=1e-4)

### 評估指標 (Evaluation Metric)

---

## 執行建議

1. 先跑 1 個 Fold 驗證

- 將 kf = KFold(n_splits=5) 暫改為 n_splits=1，快速確認 pipeline 無誤。

2. 資源管理

- 每次 Fold 結束後執行：

```bash
   del train_set, val_set, train_loader, val_loader, model
   gc.collect()
   torch.cuda.empty_cache()
```

3. 監控重點指標

- 觀察 Val Acc 是否持續上升
- 若 Train Acc >> Val Acc → 增加 Dropout / Weight Decay
- 若 Val Loss plateau → 降低 Learning Rate 或增加 Epoch

4. 最終提交前

- 使用全部 5 Fold 的模型做 Logits Average
- 確認測試資料已正確使用各 Fold 的 mu / sd 標準化

5. 可額外嘗試的方向（若時間允許）

- 把輸入 reshape 成 (11, 39) 後使用 1D-CNN / Transformer Encoder
- 嘗試 Label Smoothing=0.05 或 0.15
- 使用 CosineAnnealingLR 替代 ReduceLROnPlateau

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 亮紅，錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 橘，警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 綠，正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 藍，提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 灰，次要註解 */
</style>
