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
| v5 TypeLLM 補測（L4） | T1 兩個變體 × 4 task × 200、T2 置換平均 4 task × 200 × 6–8 次前向 | GPU ≈ 0.35 h | ≈ $0.3 | 0.4 h / < $0.5 |
| v5 T4 SGLang smoke（L40S） | 指定 token 讀機率驗收，跑 3 次（前兩次分別撞 `/v1/tokenize` 序列化錯誤、欄位沒存） | GPU ≈ 0.4 h | ≈ $0.8 | 10 分鐘 / $0.3 |
| v6 SGLang 穩定性（L40S） | E1 餵 llama ids ×1、E2 deterministic D0 ×2 + 速度 ×1、修正輸入後穩定性 ×3（ids 重跑、ids+deterministic ×2）；L4 上切 token 一次 | GPU ≈ 1.8 h | ≈ $3.5 | 上限 $3、一天 |
| v7 packed readout（L40S） | dry run + 主跑 + 占位符變體 + K=7，四次冷啟 | GPU ≈ 0.7 h | ≈ $1.4 | < $1 |
| v8 JevBench 自跑（L4） | dry run ×2（一次 OOM）+ 26B 231 題正反序 + E4B | GPU ≈ 0.25 h | ≈ $0.2 | < $1 |
| v9 九格補測（L4 + L40S） | 護欄、工具守門 × 3 模型、外部題 × 3、評分 × 3（含 JSON 對照）、重排序 × 3（L4）與 SGLang 一次（L40S）；Gemini 寫題與標註約 16 次呼叫（免費額度） | GPU ≈ 1.3 h | ≈ $1.5 | < $2 |
| v10 Clef 對照（L40S + H100） | 下載 74 GB（CPU 容器）；L40S：flash BF16 smoke + 延遲 ×2、量化等價檢查 FP8 ×2（一次缺套件）、FP8-nola、int8、NF4、27B FP8，flash 主跑（D0 2,000 + 字母 ID 2,400 + D2 + v9 + R 4,000 + JevBench 509 + packed 600 + 延遲 + profile，13 分鐘）；H100：27B BF16 參考 50 題 + 延遲，27B 主跑（31 分鐘）。image 建了 4 次（torch／torchvision／kernels 版本） | L40S ≈ 0.7 h、H100 ≈ 0.75 h | ≈ $4.5 | $3–4，上限 $8 |
| v11 Clef on llama-server + jevify（L4 + L40S + H100） | llama.cpp b11371 編譯（image builder 慢放棄一次、32 核 CPU 編兩次，一次連結失敗）；GGUF 下載 136 GB（CPU）；L40S：Clef smoke ×4、主跑 ×4、26B 延遲；H100：27B BF16 smoke；L4：jevify J1／J2（5 個 suite）／J3、混用 ×3、26B D0 新版 | L4 ≈ 2.7 h、L40S ≈ 1.6 h、H100 ≈ 0.1 h、CPU 編譯 ≈ 1 h × 32 核 | ≈ $7 | ≈ $5，上限 $8 |
| v12 借 Gemma 4 新做法（L4 + L40S） | 12B／31B GGUF 下載（CPU）；12B smoke + D0/D2/JevBench/延遲（L4）；31B smoke（OOM 一次）+ 主跑 + JevBench 1 slot 重跑（L40S）；26B 思考尾段步驟 0、弱題主跑、JevBench 2 slot 重跑（L4）；E1/E2 離線 | GPU ≈ 1.3 h | ≈ $1.5 | 約 $4 |
| v13 EmbeddingGemma 2 D0（本機 CPU） | 10 類 × 200 筆，雲端容器 4 核 CPU 約 15 分鐘 | 0 | $0 | — |
| v14 SOP 路由（L4） | E4B／26B 淘汰賽與混合式各一次（300 則）、EmbeddingGemma 2 GPU 延遲一次；Gemini 出題約 101 次呼叫（免費額度） | GPU ≈ 0.5 h | ≈ $0.4 | — |
| v15 前 K 名小型淘汰賽（L4） | 26B 前 20／30／50 名各一次（300 則，4 則並行）；單則延遲容器一個（test 前 30 則 × 5 種做法） | GPU ≈ 0.5 h | ≈ $0.4 | — |
| v16 打散分組（L4） | 26B 四種發牌方式各一次（300 則，4 則並行；D100 每則 11 次讀取） | GPU ≈ 0.6 h | ≈ $0.5 | — |
| v18 MTP 開關（L4） | 判斷套件（D0 2,000 題 + v4bench L1/L4/L6）base／mtp 各一個容器約 25 分鐘；思考 base／mtp 各約 8 分鐘；旗標檢查與草稿頭下載 | GPU ≈ 1.1 h | ≈ $1 | — |
| **合計（v2–v18）** | | **GPU ≈ 19.6 h** | **≈ $32.5** | |

比估算省很多的原因：MoE A4B 在 L4 上每題 200 ms 級，2000 筆 + 1000 筆對照只要 12 分鐘；資料由 Claude Code 端產生不吃 GPU。
AI Studio：v2 僅能力探測；v3 用 gemini-3.5-flash 盲寫 600 筆 D2 與標註 600 筆，約 80 次呼叫，免費額度內。
