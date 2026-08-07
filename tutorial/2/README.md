# 逐幀音素分類專案指南

(README1 ChatGPT 5.5 重點摘要版)

## 1. 專案目標

- **任務**：逐幀音素分類 (Framewise Phoneme Classification)
- **類型**：39 類多分類 (Multiclass Classification)
- **輸入**：每個語音 frame 的 MFCC 特徵
- **輸出**：預測該 frame 對應的音素 (39 類)

---

## 2. 資料前處理

### 語音切片

- 使用 25 ms 視窗
- 每 10 ms 移動一次

### 特徵抽取 (MFCC)

**流程**：

1. Waveform
2. DFT (傅立葉轉換)
3. Mel Filter Bank
4. Log
5. DCT
6. 得到 MFCC 特徵向量

---

## 3. Context Window (最重要)

由於單一 frame 資訊不足，因此加入前後資訊。

**本作業設定**：

- 前 5 frame + 當前 + 後 5 frame = 共 11 個 frame

**維度計算**：

- 若每個 frame 為 39 維：$11 \times 39 = 429$ 維
- **模型輸入**：429 維
- **Label**：只對應中間那一幀

---

## 4. Dataset

### 資料

- **`train_11.npy`**
    - 922,449 筆
    - `shape=(N, 429)`
- **`train_label_11.npy`**
    - 39 類 Label
- **`test_11.npy`**
    - 307,483 筆
    - 無 Label

### 評估指標

- Accuracy

---

## 5. Baseline 提升方向

依照難度：

### (1) Simple Baseline

- **先調**：Learning Rate

### (2) Strong Baseline

#### Model

- 增加 Hidden Layer
- 調整 Hidden Dimension
- 激活函數：ReLU、LeakyReLU、Swish

#### Training

- Batch Size
- Optimizer
- Learning Rate Scheduler
- Epoch

#### 防止 Overfitting

- BatchNorm
- Dropout
- Weight Decay

---

## 6. 最推薦的模型設定 (作者整理)

### Network 架構

429 → 1024 → 512 → 256 → 128 → 39

- **Hidden Layer**：4~5 層
- **Activation**：ReLU、LeakyReLU、Swish

---

## 7. 推薦超參數

| 參數            | 建議值 |
| :-------------- | :----- |
| Learning Rate   | 5e-4   |
| Epoch           | 80     |
| Batch Size      | 128    |
| Weight Decay    | 5e-4   |
| Optimizer       | AdamW  |
| Label Smoothing | 0.1    |

---

## 8. Learning Rate Scheduler

- **建議**：ReduceLROnPlateau、CosineAnnealingLR
- **搭配**：Early Stopping (patience=20)

---

## 9. 防止 Overfitting

建議加入：

- BatchNorm
- Dropout (0.2~0.5)
- Weight Decay
- Label Smoothing

---

## 10. 執行流程

建議步驟：

1. 先跑一個 Fold
2. 確認 Pipeline
3. 再跑完整 5 Fold
4. 每 Fold 清除 GPU 記憶體
5. 最後做 Logits Average Ensemble

---

## 11. 監控重點

訓練時注意：

- Val Accuracy 持續上升
- **Train Acc ≫ Val Acc**：代表過擬合，需增加 Dropout / Weight Decay
- **Val Loss 不下降**：需降低 Learning Rate 或增加 Epoch

---

## 12. 可再提升的方法

若有更多時間，可嘗試：

- 將輸入 reshape 為 (11, 39)，改用 1D-CNN
- 使用 Transformer Encoder
- 調整 Label Smoothing (0.05 或 0.15)
- 改用 CosineAnnealingLR

---

## ★ 考試 / 實作必記重點

- 輸入維度：429 ($11 \times 39$)
- 39 類音素分類
- Context Window：前 5 + 中 1 + 後 5
- MFCC 流程：DFT → Mel Filter Bank → Log → DCT
- 推薦模型：429 → 1024 → 512 → 256 → 128 → 39
- 推薦 Optimizer：AdamW
- 推薦 Learning Rate：5e-4
- 推薦 Batch Size：128
- 推薦 Epoch：80
- 使用 BatchNorm、Dropout、Weight Decay、Early Stopping 防止過擬合
- 可搭配 ReduceLROnPlateau 或 CosineAnnealingLR 提升訓練效果
