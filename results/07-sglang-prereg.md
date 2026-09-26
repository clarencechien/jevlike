# 07 — SGLang 後端評估：預先登記（2026-09-26，跑前寫死）

依 `docs/handoff-v4-sglang.md` §3。卡：Modal L40S 48 GB。四個 arm：A0 llama-server UD-Q4_K_M、A1 llama-server Q8_0、B1 SGLang FP8（`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` @ ed35d7ab）、B2 選配。

| 門檻 | 條件 |
|---|---|
| G1 單題不退步 | B1 L1 p50 ≤ A1 × 1.10 |
| G2 多題有感 | K=10 與 K=16（state ≈1,100 token）B1 端到端 ≤ A1 ÷ 1.5 |
| G3 併發不退步 | c=8 與 c=32 時 B1 decisions/s ≥ A1 |
| G4 共存不更差 | L6 下 B1 的 p95 退化倍數 ≤ A1 的 |

前提：B1 共用前綴的 cached_tokens 命中率 ≥ 90%；三 arm smoke 的 missing=0、first token 是字母。
護欄：B1 vs A1 每 task acc 差 ±2 點內且 McNemar p ≥ 0.05；錯誤偵測 AUROC 不低於 A1 超過 0.05。
判定：G1–G4 全過 → Phase 2；G2 過、G1/G3 沒過 → 分工使用；G2 沒過 → 停，維持 llama-server。
A0/A1 用各自最佳做法（`-np` 多 slot + cache_prompt 並行送出）。不開推測解碼。
