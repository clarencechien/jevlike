# 實際費用與 GPU 時間（對照 handoff v2 §2 估算）

日期：2026-09-23（v2/v3）、2026-09-26（v4）。v2/v3 GPU 全部為 Modal L4（約 $0.80/h），v4 用 L40S 與 H100；下列時間取自各 run 的 `_run.json` 與 CLI 觀察，Modal dashboard 為準。

| 里程碑 | 內容 | Modal 時間 | 估算費用 | v2 估算 |
|---|---|---|---|---|
| M0 環境 | image build（CPU）、probe（L4 約 1 分鐘）、GGUF 下載 17 GB（CPU 容器約 3 分鐘） | GPU 0.02 h | < $0.1 | 0.5 h / < $1 |
| M1 Smoke | 2 次（第一次 thinking 未關，第二次採用設定），每次含 17 s 模型載入 | GPU 0.05 h | < $0.1 | 0.2 h / < $0.5 |
| M2 資料 | Claude Code（Fable）撰寫 seeds / 產生器 / 難例 / 抽查；無 GPU | 0 | 訂閱額度 | 0 |
| M3 延遲 | L1–L7 一次跑完 1355 s + 額外 `--swa-full` 對照（L1/L4，n=100） | GPU ≈ 0.55 h | ≈ $0.45 | 0.5–1 h / ~$1 |
| M4 準確率 | 10 task × 200 筆 logprob + 10 × 100 筆 JSON 對照，724 s | GPU 0.2 h | ≈ $0.17 | 1–1.5 h / ~$1.5 |
| M5 分析 | 本機 CPU | 0 | 0 | 0 |
| 除錯重跑 | image entrypoint / click 參數名 crash loop（CPU，數分鐘） | ≈ 0 | < $0.1 | 1–2 h / ~$2 |
| v3 階梯 | E2B/E4B 下載與 smoke、3 模型 × D0/D1×3/D2（約 14,000 題）、E2B D1-cue 重跑 | GPU ≈ 1.2 h | ≈ $1 | 1.5 h / $1.5 |
| **合計（v2 + v3）** | | **GPU ≈ 2.1 h** | **≈ $2** | 5.5–7.5 h / $5.5–7.5 |
| v4 SGLang（L40S $1.95/h） | llama-server A0/A1 各 3 段 v4bench + Q8 準確率 D0/D1-cue；SGLang smoke ×4（每次冷啟 3–9 分鐘）、v4bench ×3、準確率 D0/D1-cue ×2/D0-w1/D0-noradix | GPU ≈ 2.8 h | ≈ $5.5 | v4 估 4–6 h / $8–12 |
| v4 bf16 對照（H100 $3.95/h） | llama-server BF16（50 GB GGUF）與 SGLang BF16 各跑一次 D0 test | GPU ≈ 0.4 h | ≈ $1.6 | 選配 |
| **合計（v2–v4）** | | **GPU ≈ 5.3 h** | **≈ $9** | |

比估算省很多的原因：MoE A4B 在 L4 上每題 200 ms 級，2000 筆 + 1000 筆對照只要 12 分鐘；資料由 Claude Code 端產生不吃 GPU。
AI Studio：v2 僅能力探測；v3 用 gemini-3.5-flash 盲寫 600 筆 D2 與標註 600 筆，約 80 次呼叫，免費額度內。
