# 06 — 階梯實驗預先登記（跑之前寫死，2026-09-23）

依 `docs/handoff-v3.md` §1。以每個 task 的 test 集 accuracy 為主，配對 bootstrap 95% CI（1000 次），McNemar 用不一致格。

- 「相當」：CI 重疊且 McNemar p ≥ 0.05。
- 「明顯掉」：差距 ≥ 5 個百分點且 p < 0.05。

| D0 現有合成集 | 新難題集（D1-cue / D2） | 判讀 |
|---|---|---|
| E4B ≈ 26B | E4B ≈ 26B | H1 強版：題目對這一級全部飽和 |
| E4B ≈ 26B | E4B 明顯掉 | H1（合成集）+ 新集有鑑別力 |
| E4B 明顯掉 | E4B 明顯掉 | H2：現有難度已能分出等級 |
| E4B 明顯掉 | E4B ≈ 26B | 不一致，優先懷疑新集 |

E2B 在某 task ≥ 0.95 → 該 task 直接 H1。
另量：錯誤偵測 AUROC（raw conf 預測對錯）、26B 在「E4B 錯 / 26B 對」題上的信心分布。
模型：E2B Q8_0（5.05 GB）、E4B Q8_0（8.19 GB）、26B-A4B UD-Q4_K_M（沿用 D0 結果；prompt.py 與 llama-server 版本自 D0 後未變）。全部 `--swa-full`。
