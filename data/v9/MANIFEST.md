# data/v9 MANIFEST

`python3 data/v9/gen_v9.py --seed 9`。gold 全部來自構造（模板、程序等級、段落身分），不是模型標的。

| task | n | gold 分布 | easy / hard | lang | 來源 |
|---|---|---|---|---|---|
| g_input_guard | 198 | {'A': 89, 'B': 53, 'C': 56} | 138 / 60 | {'zh': 169, 'en': 19, 'mixed': 10} | Counter({'template': 138, 'handwritten': 60}) |
| t_tool_gate | 198 | {'A': 47, 'B': 75, 'C': 76} | 138 / 60 | {'zh': 140, 'mixed': 58} | Counter({'template': 138, 'handwritten': 60}) |
| e_answer_score | 600 | {'A': 120, 'B': 120, 'C': 120, 'D': 120, 'E': 120} | 360 / 240 | {'zh': 600} | Counter({'procedure': 600}) |
| r_rerank | 200 查詢 × 20 候選；語料 300 段 | 每查詢 1 正解（rel 2）+ 2 同症狀他意圖（rel 1）+ 2 鄰近症狀（rel 0）+ 隨機 | — | zh | raw/r_symptoms.json |
