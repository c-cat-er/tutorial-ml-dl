未完

## 1. 資料下載

## 關鍵設計亮點

| 項目                     | 說明                 | 對應代碼位置                                    |
| ------------------------ | -------------------- | ----------------------------------------------- |
| 每 Fold 獨立標準化       | 避免資料洩漏         | mu = train[train_idx].mean...                   |
| Label Smoothing          | 提升泛化能力         | nn.CrossEntropyLoss(label_smoothing=0.1)        |
| ReduceLROnPlateau        | 驗證損失停滯時自動降 | LR scheduler = ReduceLROnPlateau(...)           |
| EarlyStopping            | 防止過擬合           | EarlyStopping(patience=20)                      |
| AdamW + Weight Decay     | 更好的正則化         | torch.optim.AdamW(..., weight_decay=...)        |
| pin_memory + num_workers | 加速資料載入         | DataLoader(..., num_workers=4, pin_memory=True) |

## 可優化

加入 混淆矩陣 與 學習曲線 視覺化
測試集推論時使用各 fold 的 \*\_norm.npz 進行相同標準化
嘗試 Residual Block 或 Transformer Encoder 替換全連接層
使用 混合精度訓練 (torch.cuda.amp) 加速

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
