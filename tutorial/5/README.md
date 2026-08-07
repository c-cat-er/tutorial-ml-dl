# 專案五：BERT 中文問答任務

### 任務目標

給定段落 + 問題，模型要從段落中「抽取」出答案（輸出起始/結束 token 位置 s, e），屬於抽取式問答（Extractive QA），基於 BERT 微調。

### 核心概念

BERT 輸入格式：`[CLS] Question tokens [SEP] Document tokens [SEP]`

- 預訓練用 MLM + NSP 學語意
- 微調時接一個輸出層，預測答案的 start/end 位置

**為什麼段落要切窗口：**

- BERT 最大長度限制 512（因為 Self-Attention 是 $O(n^2)$ 複雜度）
- 長段落塞不下 $\rightarrow$ 切成多個 window，各自預測 start/end，再取最大分數的答案

---

### 這次作業要改的 4 個重點

| 項目               | 問題                                                                                                                    | 建議做法                                                                                                                                                     |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **doc_stride**     | 原本 150（等於視窗長度，幾乎無重疊），答案容易被切斷在交界                                                              | 改小，例如 `max_paragraph_len=150` 時用 `doc_stride=64~96`                                                                                                   |
| **Preprocessing**  | 訓練時答案永遠置中在窗口中間 $\rightarrow$ 模型學到「答案在中間」的捷徑，不會真正理解                                   | 加隨機偏移，讓答案在窗口中的位置分佈更均勻                                                                                                                   |
| **Postprocessing** | 直接對整段做 argmax，可能選到 question/`[CLS]`/`[SEP]`/padding；沒保證 end $\ge$ start；用相加分數而非真正搜尋最佳 span | 1) 遮蔽掉非答案區域（用 `token_type_ids` / `attention_mask`）<br>2) 在合法範圍內搜尋 start+end 分數最大且 end $\ge$ start、長度受限的組合<br>3) 再跨窗口比較 |
| **學習率衰減**     | 原始 `learning_rate=0.9` 太誇張，容易爆炸                                                                               | 改成 `3e-5~5e-5`，並用 `get_linear_schedule_with_warmup`（或手動線性遞減）                                                                                   |

---

### 進階優化選項（optional）

- **AMP 混合精度（FP16）**：加速 1.5~3 倍
- **Gradient Accumulation**：小 batch 累積梯度模擬大 batch
- **Ensemble**：多模型集成
- **換預訓練模型**：如 `deepset/roberta-base-squad2`（只能用 HuggingFace 上的模型，用外部模型視為作弊）
- **超參數調整**：調整 epoch 數（1 通常不夠，建議 2~3）、`doc_stride=128`、`max_query_length=64` 等超參數

---

### 建議實作優先順序

1. 先修正學習率（避免訓練直接爆炸）
2. 修正 postprocessing bug（這是影響分數最直接的地方）
3. 調整 doc_stride + 加隨機偏移（防止 shortcut learning）
4. 有餘力再嘗試 AMP / Gradient Accumulation / 換模型 / Ensemble

## 核心流程

---

## 優化重點摘要

- 有優化的檔案：

---

## 優化步驟

-   7. Dataset and Dataloader Dataset 類別優化（doc_stride + 隨機偏移 Preprocessing）
-   8. Function for Evaluation Postprocessing 完整優化版（Boss 關鍵）
-   9. Training 學習率 + Scheduler 優化（Boss 建議）
- 建議超參數設定（Boss 級別）

1. Improve Postprocessing（最關鍵）

- 用 token_type_ids / attention_mask 遮蔽非答案區域（question、[CLS]、[SEP]、padding）
- 在合法範圍內搜尋 start + end 分數最大、且 end ≥ start、長度受限的最佳 span
- 跨多個窗口比較，取最高分答案

2. Further improve the above hints

- doc_stride：從 150 改為 64~96（增加重疊，避免答案被切斷）
- Preprocessing：加入隨機偏移，讓答案在窗口中的位置更均勻（避免 shortcut learning）
- 學習率：改為 3e-5 ~ 5e-5 + 使用 get_linear_schedule_with_warmup 線性衰減

3. 進階加分（可選）

- AMP 混合精度（加速 1.5~3 倍）
- Gradient Accumulation（小 batch 模擬大 batch）
- Ensemble 多模型集成
- 調整 epoch=2~3、max_query_length=64 等超參數

---

### 評估指標 (Evaluation Metric)

---

## 執行建議

-順序：Postprocessing → 學習率 → doc_stride + 隨機偏移 → 其餘進階優化。

<style>
  my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
  my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
  my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
  my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
  my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
  my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
  my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
