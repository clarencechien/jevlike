# 01 — Smoke（M1）

模型：`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`，llama-server b11118，L4，`-c 16384 -np 4 --jinja --reasoning-budget 0`。原始輸出：`results/modal/smoke.json`。

## Tokenize 檢查

| 字串 | tokens |
|---|---|
| `A` | `[236776 "A"]` |
| ` A` | `[562 " A"]`（**與 `A` 不同 token**） |
| `答案：A` | `[56996 "答案"] [237184 "："] [236776 "A"]` |
| `答案： A` | `… [562 " A"]` |

→ client 比對 token 時 strip 空白後對字母（同一字母取最大 logprob）。

## Thinking 的影響（關鍵發現）

Gemma 4 的 chat template 用 `enable_thinking` 控制：開啟時在 system 回合最前面塞 `<|think|>\n`；關閉時不塞，並在 `<|turn>model\n` 後補一個**空的思考區塊** `<|channel>thought\n<channel|>`。`--reasoning-budget 0` 只作用在 `/v1/chat/completions`，**對 `/apply-template` 與 `/completion` 無效**；要自己在 `/apply-template` 帶 `chat_template_kwargs: {"enable_thinking": false}`。

| 變體 | 第一個 token | 5 例對幾題 | 備註 |
|---|---|---|---|
| think 開 + 無 prefill | `<|channel>` | 0（全部 missing） | 模型要先思考 |
| think 開 + prefill「答案：」 | `答案`（想再寫一次） | 3 | 分布平坦（p≈0.7），s1 明顯題答錯 |
| **think 關 + system + 無 prefill** | **字母** | **4** | top-5 全是字母；cached 55 ms / 未 cached 90 ms；**採用** |
| think 關 + system + prefill「答案：」 | 字母 | 4 | top 出現 `**`（markdown），略差 |
| think 關 + 無 system + 無 prefill | 字母 | 4 | s4 改答 A；system 有幫助 |
| think 關 + prefill「答案： 」（尾空白） | `\n` | — | 尾空白會讓模型先換行，**不要留空白** |

## 5 例結果（採用設定）

| id | 題 | gold | chosen | p(chosen) | missing | 延遲 (cached) |
|---|---|---|---|---|---|---|
| s1 | 回焊超溫板在爐內 → 急迫度 | C | C | 1.00 | — | 58 ms |
| s2 | E4021 NOZZLE VACUUM LOW → 類別 | A | A | 1.00 | D（第 2 次） | 52 ms |
| s3 | feeder 缺料操作員已補 → 派工？ | B | B | 1.00 | A | 59 ms |
| s4 | 連續 7 點下方且下降、未超限 → SPC 動作 | B | **C** | **0.996** | — | 55 ms |
| s5 | UPH 高於目標 → 異常？ | B | B | 1.00 | — | 54 ms |

觀察：
1. 分布極度尖銳（p≈1.0），s4 是**高信心錯誤**，M4 的校準與選擇性準確率要特別看這種案例。
2. `missing` 來自 n_probs=20 內被 `_`、全形字母、`\n` 等雜 token 擠掉，機率本來就 <1e-4；改 n_probs=40 並把 missing 視為 0 機率（分析時另記 missing 率）。
3. 對照組 `/v1/chat/completions` 生成 JSON：s1、s2 皆正確，未 cached 約 460–500 ms（含 15 個輸出 token）；正式數字在 M3 L7。
4. 每次 server 冷啟動 17 s（模型 17 GB 從 Volume 載入）。
