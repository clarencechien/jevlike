# D2 — Gemini 盲寫、雙標註

出題：gemini-3.5-flash，只看 task 名稱、選項名稱與 20 筆 D0 語氣參考，**不看 criteria**。
標註：Gemini（gemini-3.5-flash，temperature 0.2）與 Claude（Fable，三個獨立 subagent）各自拿完整 criteria 獨立標，不互看。
一致的進 D2；不一致的存 `disagreements.jsonl` 待人裁決（同時是 borderline 子集）。

| task | 寫出 | 兩邊都有標 | 一致 | 一致率 | κ | Gemini 低信心(≤0.6) | Fable 低信心 | D2 label 分布 |
|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | 60 | 60 | 53 | 0.88 | 0.82 | 0 | 9 | A:17, B:16, C:20 |
| m_alarm_category | 60 | 60 | 60 | 1.00 | 1.00 | 0 | 3 | A:12, B:12, C:12, D:12, E:12 |
| m_needs_dispatch | 60 | 60 | 60 | 1.00 | 1.00 | 0 | 1 | A:29, B:31 |
| q_spc_action | 60 | 60 | 60 | 1.00 | 1.00 | 0 | 5 | A:15, B:15, C:15, D:15 |
| q_defect_root | 60 | 60 | 55 | 0.92 | 0.90 | 0 | 2 | A:6, B:7, C:9, D:11, E:10, F:12 |
| p_uph_anomaly | 60 | 60 | 60 | 1.00 | 1.00 | 0 | 2 | A:31, B:29 |
| p_line_change | 60 | 60 | 60 | 1.00 | 1.00 | 0 | 1 | A:21, B:20, C:19 |
| x_ticket_route | 60 | 60 | 60 | 1.00 | 1.00 | 0 | 1 | A:15, B:15, C:15, D:15 |
| x_escalate | 60 | 60 | 60 | 1.00 | 1.00 | 0 | 1 | A:30, B:30 |
| x_10way_intent | 60 | 60 | 56 | 0.93 | 0.93 | 0 | 5 | A:6, B:5, C:8, D:6, E:6, F:2, G:5, H:6, I:6, J:6 |
| **全部** | 600 | 600 | 584 | 0.97 | | | | |
