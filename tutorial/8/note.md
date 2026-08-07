未完成

模型壓縮 + 知識蒸餾」兩大核心，參數量控制與 KD Loss 是評分重點。
整體流程：資料前處理 → 2. 設計輕量 StudentNet（dwpw_conv）→ 3. 定義 KD Loss → 4. 載入 Teacher → 5. （可選）產生 Pseudo Label → 6. 使用 KD Loss 訓練 80 epochs + CosineAnnealingLR → 7. 輸出 predict.csv。

<style>
   my-red    { color: #d32f2f; font-weight: bold; } /* 錯誤/危險 */
   my-orange { color: #ed6c02; font-weight: bold; } /* 警告/注意 */
   my-yellow { background-color: #fff176; color: #000000; padding: 0 4px; } /* 重點標記 */
   my-green  { color: #2e7d32; font-weight: bold; } /* 正常/完成 */
   my-blue   { color: #0288d1; font-weight: bold; } /* 提示/說明 */
   my-cyan   { color: #00a8cc; font-weight: bold; } /* 青色/新增設定 */
   my-gray   { color: #8c8c8c; font-size: 0.9em; } /* 次要註解 */
</style>
