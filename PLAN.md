# 執行計畫與環境評估（Claude Code web + Modal + AI Studio）

日期：2026-09-23。依 `docs/handoff-v2.md` 執行；背景與指標定義見 `docs/handoff-v1.md`。

## 1. 環境評估（M0 前置，已實測）

| 項目 | 結果 | 影響 |
|---|---|---|
| Modal token | env 變數名是 `modal` / `modal_secret`（非官方名），執行時映射成 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | 不寫進 repo，所有 `modal` 指令經 `scripts/modal.sh` 包一層 |
| Modal API 連線 | 一開始「Could not connect」：本環境走 HTTPS proxy，Modal 的 gRPC 需要 `python-socks`。`pip install 'modal[api-proxy-support]'` 後正常，`modal app list` / `modal volume list` 可用 | **不需要 Azure VM 備援** |
| AI Studio key | `gemini_key` 可用；模型表含 `gemma-4-26b-a4b-it`、`gemma-4-31b-it`、gemini 3.x 系列 | 可用 |
| AI Studio logprobs | Gemma 4 26B / 31B 皆回 `Logprobs is not enabled for this model`；且 `maxOutputTokens=1` 回空字串（thinking token 先被吐出） | **AI Studio 不能當 typed-decision 後端**，只能做 (a) JSON 生成對照組（Gemini）、(b) 補產難例情境；31B dense 對照只能用生成式作答，不能讀 logprob |
| HF GGUF | `unsloth/gemma-4-26B-A4B-it-GGUF` 不需 token（gated=false）。實際檔名 `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`，16.95 GB（handoff 寫的 `Q4_K_M` 檔名不存在，最接近的是 UD-Q4_K_M）。另有 `ggml-org/gemma-4-26B-A4B-it-GGUF` 的 `Q4_0` 14.6 GB 與 Google 官方 QAT q4_0 14.4 GB | 預設 UD-Q4_K_M；L4 24 GB OOM 時退到 Q4_0 |
| 本機 | 4 vCPU / 15 GB RAM，無 GPU；numpy / sklearn / matplotlib 已裝 | M5 CPU 分析在本機做 |
| GB10 | 不在本環境 | 依 v2 §0：延遲只報 L4 上界 |

## 2. Milestone 對應與判斷準則

| M | 做什麼 | 進下一階段的條件 |
|---|---|---|
| M0 | `modal_app.py`：image（`ghcr.io/ggml-org/llama.cpp:server-cuda`）、Volume、下載 GGUF、GPU probe | image 起得來、`llama-server` 版本可讀、GGUF 在 Volume |
| M1 | `bench/smoke.py`：5 例；印 `/apply-template` 結果、`/tokenize` 的 `A` vs ` A`、第一個 token、`missing` | first_token 是字母、missing 為空、明顯案例 p>0.9；否則修 `prompt.py`（關 thinking / 換 prefill）再跑 |
| M2 | seeds（Fable 寫）→ `gen_expand.py` 展開 → hard + 孿生（Fable 寫）→ MANIFEST | 10 task × ≥200 筆、80/20 語言比、30% hard、孿生 pair_id |
| M3 | `bench/latency.py` L1–L7 | 每組 200 次 + warm-up 20；L4 開/關 cache_prompt；L6 只報倍數 |
| M4 | `bench/accuracy.py`：全部讀 logprob 存 raw；JSON 生成對照 easy/hard 各 50 | raw jsonl 拉回 repo |
| M5 | `bench/analyze.py`（本機 CPU）：校準、AUROC、ECE、選擇性準確率、學習曲線、規則與 TF-IDF 基線 | 圖存 `results/fig/` |
| M6 | `results/REPORT.md`、`results/cost.md` | 填 v1 §7 表與 Q1–Q8，加「本版限制」 |

## 3. 設計決定（與 handoff 不同或 handoff 未定之處）

1. **模型檔**：`unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`。GB10 用哪份未知，接回時若不同要重跑 M4（v2 §10）。
2. **Prompt 渲染**：不手刻 chat template。用 llama-server `/apply-template` 取得模型自帶模板渲染結果，再接 assistant 前綴；smoke 時決定 thinking 要怎麼關（`--reasoning-budget 0` / `chat_template_kwargs`）。
3. **AI Studio 用途縮小**：只做 JSON 生成對照（Gemini 3.x flash，免費額度）與補產難例；不做 31B logprob 對照。
4. **長任務一律 `modal run --detach`**，結果逐筆 append 到 Volume，`accuracy.py` 支援 `--resume`。
5. **費用上限**：L4 約 $0.80/h，目標 GPU 總時 < 6 h；每次 `modal run` 後把實際秒數記進 `results/cost.md`。

## 4. 沒有 GB10 不能回答的（原樣留給 v1）

aarch64 相容性、與現有生成負載搶資源的絕對數字、真實資料校準。
