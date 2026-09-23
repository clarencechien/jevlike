# MANIFEST — data/synthetic

seed=7, easy per task=140, en share=0.2

| task | kind | n | easy | hard | zh | en | pairs (easy/hard) | gold distribution | hard types |
|---|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | score | 200 | 140 | 60 | 159 | 41 | 70/15 | A:68, B:62, C:70 | borderline:12, distractor:12, incomplete:12, inverted:12, noisy:12 |
| m_alarm_category | choice | 200 | 140 | 60 | 159 | 41 | 70/15 | A:42, B:41, C:40, D:40, E:37 | borderline:12, distractor:12, incomplete:12, inverted:12, noisy:12 |
| m_needs_dispatch | noul | 200 | 140 | 60 | 160 | 40 | 70/15 | A:100, B:100 | borderline:12, distractor:12, incomplete:12, inverted:12, noisy:12 |
| q_spc_action | choice | 200 | 140 | 60 | 159 | 41 | 70/15 | A:51, B:48, C:53, D:48 | borderline:13, distractor:14, incomplete:11, inverted:11, noisy:11 |
| q_defect_root | choice | 200 | 140 | 60 | 159 | 41 | 70/15 | A:35, B:34, C:34, D:34, E:32, F:31 | borderline:13, distractor:12, incomplete:11, inverted:12, noisy:12 |
| p_uph_anomaly | noul | 200 | 140 | 60 | 160 | 40 | 70/15 | A:100, B:100 | borderline:12, distractor:12, incomplete:14, inverted:10, noisy:12 |
| p_line_change | choice | 200 | 140 | 60 | 160 | 40 | 70/16 | A:67, B:66, C:67 | borderline:12, distractor:12, incomplete:12, inverted:12, noisy:12 |
| x_ticket_route | choice | 200 | 140 | 60 | 160 | 40 | 70/16 | A:51, B:52, C:50, D:47 | borderline:12, distractor:12, incomplete:12, inverted:12, noisy:12 |
| x_escalate | noul | 200 | 140 | 60 | 160 | 40 | 70/16 | A:100, B:100 | borderline:12, distractor:12, incomplete:12, inverted:12, noisy:12 |
| x_10way_intent | choice | 200 | 140 | 60 | 160 | 40 | 70/17 | A:20, B:20, C:21, D:19, E:21, F:21, G:19, H:19, I:20, J:20 | borderline:14, distractor:12, incomplete:12, inverted:10, noisy:12 |

## 產法

- easy：`data/gen/{task}.py` 的孿生模板（每個 generator 回傳一對只差一個事實、gold 不同的例子）× 隨機參數，`gen_expand.py` 展開並去重；gold 由模板規格決定。
- hard：`data/hard/{task}.jsonl`，Claude（Fable）依 `data/HARD_GUIDE.md` 手寫，先定 label 再寫情境；每筆有 `rationale`。
- 出題者（Claude/Fable）與受測者（Gemma 4 26B）不同家；Fable 不作答。
- 抽查：見 `data/synthetic/SPOTCHECK.md`。
