# MANIFEST — data/synthetic

seed=7, easy per task=140, en share=0.2

| task | kind | n | easy | hard | zh | en | pairs (easy/hard) | gold distribution | hard types |
|---|---|---|---|---|---|---|---|---|---|
| m_alarm_category | choice | 140 | 140 | 0 | 112 | 28 | 70/0 | A:30, B:29, C:28, D:28, E:25 |  |
| m_needs_dispatch | noul | 140 | 140 | 0 | 112 | 28 | 70/0 | A:70, B:70 |  |

## 產法

- easy：`data/gen/{task}.py` 的孿生模板（每個 generator 回傳一對只差一個事實、gold 不同的例子）× 隨機參數，`gen_expand.py` 展開並去重；gold 由模板規格決定。
- hard：`data/hard/{task}.jsonl`，Claude（Fable）依 `data/HARD_GUIDE.md` 手寫，先定 label 再寫情境；每筆有 `rationale`。
- 出題者（Claude/Fable）與受測者（Gemma 4 26B）不同家；Fable 不作答。
- 抽查：見 `data/synthetic/SPOTCHECK.md`。
