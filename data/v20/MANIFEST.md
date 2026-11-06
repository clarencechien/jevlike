# data/v20 — eg2poc 專案狀態題（v20 速度對照用）

來源：clarencechien/eg2poc @ `ee37646e901fac14ca6e726c07ddbef10b7ded1e`，`data/synthetic-v2.1/analysis-plan.json` 的 `benchmark_pair_ids`（100 個 pair，zh 50／en 50）取 `new_text`。
題目：`single_instructions[lang]`；選項 `question` = `labels_alt[lang]`（A 按計畫、B 延後、C 阻礙、D 取消），`question2` = `labels[lang]`（S3 換題用）。
gold：pair 的 `label`（1 = 狀態已改變）；照 eg2poc 的 `single_score`，分數 = P(B+C+D)，答對 = (答案 ≠ A) == label。
EG2 設定照 eg2poc config：document 模板 `title: none | text: {text}`、classification 模板 `task: classification | query: {text}`、temperature 0.045。
合成資料（規則生成），eg2poc README 註明未經獨立判讀。
