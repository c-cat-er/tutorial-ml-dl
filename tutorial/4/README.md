# 專案四：說話者分類 (Speaker Classification)

## 📌 任務概述

- **核心目標**：根據給定的語音訊號，預測其所屬的說話者類別。
- **架構優勢**：結合 RNN（考慮完整序列）與 CNN（並行處理）的優點。
- **學習重點**：掌握 Transformer 模型的應用方法。

## 📊 資料集與格式

- **訓練集**：69,438 個帶有標籤的已處理音訊特徵。
- **測試集**：6,000 個不帶標籤的已處理音訊特徵。
- **類別總數**：共 600 個類別，每個類別代表一位特定的說話者。
- **目錄架構**：包含 `metadata.json`、`testdata.json`、`mapping.json` 及多個 `uttr-{隨機字串}.pt` 特徵檔。
- **元數據內容**：`metadata.json` 記載梅爾頻譜圖維度（`n_mels`）與說話者字典（含 `feature_path` 與 `mel_len`）。

## 🛠️ 資料預處理

- **聲學特徵生成流程**：
    1. **Waveform（波形）**：將語音時間訊號轉為頻率訊號。
    2. **DFT（離散傅立葉轉換）**：生成 Spectrogram（頻譜圖）。
    3. **Mel-scale filter bank**：模擬人耳聽覺頻帶，轉換為 Mel-spectrogram（梅爾頻譜）。
    4. **Log（對數運算）**：取對數後得到最終輸入模型的特徵矩陣。
- **訓練時資料分割 (Data Segmentation)**：
    - 因語音長度不一，訓練時會隨機將長語音切成固定數量的片段（如 Segment=2）。
    - **目的**：增強模型泛化能力，並使 batch 內語音長度一致以利於並行運算。

## 🧠 模型核心架構與技術

- **自注意力說話者嵌入 (Self-Attentive Speaker Embeddings)**：
    - 取代傳統計算平均與標準差的統計池化（Statistics Pooling）。
    - 利用 Attention 機制動態學習各時間步的權重，聚合 hidden states 以捕捉最能代表說話者特徵的語音片段。
- **Conformer (構象體)**：
    - Transformer 的改進版本，完美融合 CNN 局部與 Transformer 全域的特徵擷取能力。
    - **模組組成**：Feed Forward Module（特徵抽象）、Convolution Module（局部時序特徵）、Multi-Head Self Attention（全域時序關聯）、LayerNorm + Dropout（正規化與防過擬合）。
- **Additive Margin Softmax**：
    - 輸出層的優化分類技巧，在 Softmax 中加入固定邊界（margin）。
    - **效果**：使同類樣本距離更近（類內緊密）、不同類樣本距離更遠（類間分離），顯著提升說話者辨識與驗證的準確率。

## 📈 Baseline 達成指南

- **Simple Baseline**：
    - 執行範例程式碼，理解 Transformer 的基礎使用並調整學習率（Learning Rate）。
    - 利用範例建構出自注意力網路（Self-attention network）進行說話者分類。
- **Medium Baseline**：
    - 掌握並修改範例程式碼中 Transformer 模組的各項超參數。
- **Hard Baseline**：
    - 自行建構出 Transformer 的變體——Conformer 架構。
    - 藉由加入 Conformer 層（構象層）來大幅提升模型預測效能。

---

## 核心流程

---

## 優化重點摘要

- 有優化的檔案：

---

## 優化步驟

- 4.1 Self-Attention Pooling + ConformerBlock
- 4.2 Model Classifier 模型（已整合 Conformer + SelfAttnPool）
-   10. Main function of Inference 訓練設定（parse_args 優化後的 config）

- 調整學習率（Learning Rate）
- 自行建構 Conformer 架構（取代標準 Transformer）。
- 實作 Conformer 模組：Feed Forward + Convolution Module + Multi-Head Self-Attention + LayerNorm。
- 將 Conformer 層替換或加入 Encoder。
- 搭配 Additive Margin Softmax 輸出層。

---

### 評估指標 (Evaluation Metric)

---

## 執行建議

- 從 Medium Baseline 的 Transformer 程式碼開始修改。
- 參考 Conformer 論文或 huggingface/transformers 實作。
- 先用小 batch 驗證架構正確性，再調整層數、kernel size、attention head。
- 目標：大幅提升測試集準確率（Hard Baseline 門檻）。

## 優化建議

- Additive Margin Softmax（可再提升 Hard Baseline 分數）
- 增加 num_conformer 到 4~6 層
- 調整 kernel_size（15~31）、nhead（4~8）、d_model（256）

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
