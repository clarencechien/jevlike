# data/v14 — 100 份 SOP 路由（v14，`docs/handoff-v14-sop-routing.md`）

- `sop_catalog.json`：Gemini（`gemini-3.5-flash`）依 `gen_v14.py catalog` 的規格寫的 100 份 SOP（12 類，每類 8–9 份），每份 `id／category／title／scope`；20 組近似孿生（40 份帶 `twin_of`），一次產生即通過驗證（100 份、標題不重複、孿生對稱）。
- `queries.jsonl`：同一個模型依 `gen_v14.py queries` 每份 SOP 寫 3 則 LINE 群組口吻的現場訊息（共 300）；style 0 直接描述、1 口語縮寫、2 夾雜無關資訊。近似孿生的兩份，Gemini 會同時看到兩份並被要求只寫得出唯一答案的訊息。沒有任何一則包含完整標題（`title_leak` 全為 false）。
- `eg2_top10.json`：EmbeddingGemma 2（CPU 參考跑）的前 10 名，依目錄順序排列，給混合式 LLM 臂讀。
- 切分：依 SOP 分組（seed 0），cal 50 份 × 3、test 50 份 × 3；選項永遠是全部 100 份。

## 抽查（Claude，隨機 30 則，seed 14）

28 則清楚指向標準答案。2 則邊界：
- `S094-q0`（換線後首件確認）：「首片過完爐，錫膏印刷有點偏，印刷機壓力跟對位要再調」——也可歸到印刷調整類。
- `S064-q0`（首件自主檢驗與登錄）：「登錄量測數據時系統一直轉圈圈」——也可歸到 MES 系統異常。

邊界率約 7%，與 v3 盲寫題的抽查同量級；這兩類不從資料中剔除，報告照實計分。
