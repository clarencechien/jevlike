# 06 — 對比：Jev、gemma-jev、我們的 Gemma 4 26B

日期：2026-09-23。外部數字取自 gemma-jev README／`docs/benchmarks.md`（catonooka/gemma-jev，master）與 JevBench v1.3.0（benchmarkheaven.com/jev-models，2026-09-21 評分）、v1 交接文件 §1 的獨立實測。**資料集、硬體、題型都不同，只能看量級，不能排名。**

## 速度（單題決策延遲）

| 系統 | 硬體 | 模型 | p50 | p95 | 條件 | 來源 |
|---|---|---|---|---|---|---|
| Jev 1.13（hosted） | TypeSafe 雲端 | 閉源 | 352 ms | 818 ms | 含網路，gemma-jev 量 | gemma-jev README |
| Jev（獨立實測） | 雲端 | 閉源 | 236–276 ms | — | 含網路 | v1 handoff §1.1 |
| gemma-jev | RTX 3090（936 GB/s） | Gemma 4 E4B Q8_0（4.5B） | 33.5 ms | 80.9 ms | 暖機、~100-token state | gemma-jev benchmarks.md |
| gemma-jev | RTX 3090 | **Gemma 4 26B-A4B QAT q4_0** | **48.4 ms** | 141.1 ms | 同上，需關 thinking | 同上 |
| gemma-jev | Threadripper 8 thread CPU | E4B Q4_K_M | 221 ms | 232 ms | 同上 | 同上 |
| **我們** | **L4（300 GB/s）** | **Gemma 4 26B-A4B UD-Q4_K_M** | **206 ms** | 225 ms | ~190 prompt token、每次不同 state、預設無前綴重用 | 03-latency L1 |
| 我們 | L4 | 同上，`--swa-full` | 137 ms | 153 ms | 模板前綴 28 token 命中 | 03-latency L8 |
| 我們 | L4 | 同上，共用 state 第 2 題起 | 73 ms | — | state 已在 cache，只算問題段 + decode | L8 |
| 我們 | L4 | 同上，同 prompt 重複（decode 地板） | 26 ms | 27 ms | 全 cache 命中 | L8 |
| 我們 | L4 | 同模型改生成 JSON | 496 ms | 520 ms | 15 個輸出 token | L7 |

判讀：
- 同一顆 26B-A4B，gemma-jev 在 3090 量到 48 ms、我們在 L4 量到 206 ms，差距主要是 **prefill 算力與頻寬**（3090 頻寬是 L4 的 3 倍）加上我們的 prompt 較長（含 criteria，190 token vs ~100）。decode 地板兩邊接近（26 vs 33–48 ms）。
- GB10 頻寬（273 GB/s）與 L4 同級，所以 GB10 上比較可能落在 100–200 ms 而不是 48 ms；除非 state 共用讓 prefill 攤掉（73 ms/題）。
- 本地任何一種都比 hosted Jev 的 236–352 ms 快，且不出網路。

## 準確率

| 系統 | 資料集 | 結果 | n | 來源 |
|---|---|---|---|---|
| Jev 1.13 | JevBench v1.3.0（534 題，220 難題） | **74.4**（第 1 名） | 534 | JevBench |
| SemIf（Qwen3.5-4B，讀 logits） | JevBench | 73.1（第 2 名） | 534 | JevBench |
| Gemma 4 26B-A4B NVFP4（razorback16 提交） | JevBench | 66.4 | 534 | JevBench |
| Gemma 4 E2B LoRA on L4 | JevBench | 66.6 | 534 | JevBench |
| Laya（零樣本） | JevBench | 54.4（第 33 名） | 534 | JevBench |
| **Cygnet**（凍結 Gemma-4-12B-it，vLLM，單 token 讀字母 + 一個溫度，無訓練） | JevBench v1.4.2 官方 534+308 題 | **61.8（第 4 名）**；公開集 87.9%、密封集 33.8%；L40S p50 50–66 ms | 842 | JevBench；與我們同一派、同一家族模型 |
| decider-4b v2（Qwen3.5-4B + 8k 筆 LoRA） | 同上 | 64.1（第 1 名，贏 Jev 1.13 的 63.3） | 842 | JevBench |
| TypeLLM（Qwen3.8-27B NVFP4，無訓練，不開思考） | JevBench **公開 231 題子集**（與上列 534 題不能直接排名） | 195/231 = 84.4%；Brier 0.241、ECE 0.052 | 231 | TypeLLM evals/jevbench（2026-09-23） |
| TypeLLM（同上，開思考，均 919 token/題，p95 61 s） | 同上 | 228/231 = 98.7%；ECE 0.017 | 231 | 同上 |
| Jev（官方） | TypeSafe 自家 4-workflow | ~68% | — | v1 handoff |
| Jev（官方公布） | Banking77 全 77 類 | 80.3% | — | gemma-jev README |
| Jev（官方公布） | SNIPS 7 類 | 97.9% | — | 同上 |
| gemma-jev E4B | Banking77 8 類子集 | 100% | 40 | gemma-jev |
| gemma-jev E4B | SNIPS 10 類 | 66.7% | 30 | 同上 |
| gemma-jev E4B | prompt injection | 91.7% | 12 | 同上 |
| gemma-jev E4B | 自家 controller set | 95.5% | 22 | 同上 |
| **gemma-jev 26B-A4B QAT** | SNIPS 10 類 / injection / controller | **86.7% / 100% / 100%** | 30 / 12 / 22 | gemma-jev benchmarks.md |
| **我們 26B-A4B** | 自製產線 10 task（合成，中文為主，30% 難題） | **平均 0.955**；7/10 ≥ 0.98；最低 0.82（急迫度）、0.86（SPC） | 10 × 200 | 04-accuracy |
| 我們 26B-A4B | 同上，只看難題 | 平均 0.91；最低 0.75 | 10 × 60 | 04-accuracy |
| 我們 26B-A4B | 10 類相近意圖（對應 SNIPS 10 類的壓力測試） | **1.00**（難題 0.98） | 200 | 04-accuracy |
| 我們 26B-A4B | 改寫 criteria 後（SPC / 急迫度） | 0.95 / 0.885（held-out 0.945 / 0.875） | 200 | 05-criteria-v2 |
| 我們 26B-A4B | 同模型生成 JSON 對照 | 平均約 0.93；SPC 0.72 | 10 × 100 | 04-accuracy |

判讀：
- **同一顆 26B-A4B 在不同人手上的數字一致地「高」**：gemma-jev 的 SNIPS 86.7% / injection 100%，我們的 10 類意圖 100%、7 個 task ≥ 98%。這顆模型做 typed decision 的能力不是我們資料太簡單才看到的。
- 但 **JevBench 上 26B-A4B 只有 66.4，落後 Jev 8 分、落後 SemIf 的 4B 模型 6.7 分**。JevBench 的 220 道難題是刻意設計的邊界案例，和我們 hard 子集 0.91 的落差說明：我們的難題還是比 JevBench 溫和，真實資料上要預期往 JevBench 那個方向掉。
- gemma-jev 的樣本數 n=12–40，只能當方向；我們每 task 200、外加 10% 獨立抽查，統計上比較站得住，但**資料是合成的**，這點和 JevBench（人工題）不同。
- **TypeLLM 不支援 Gemma 4**（`protocol.py` 遇到 `<|turn>`+`<|channel>` 直接 raise）。他們 2026-09-22 用 E2B 實測 42 題只對 24：log 顯示第一個 token 是 `The`（logprob −0.001），A/B/C 全在 −11 以下，因為沒有放空的 thought channel。我們的 `GEMMA4_TEMPLATE_NOTHINK`（`<|channel>thought\n<channel|>` 空思考塊）解掉了這題。TypeLLM 的 84.4% 是 Qwen3.8-27B、231 題公開子集，與 JevBench 534 題的 74.4 分不是同一把尺；能對照的只有「無訓練也能到 84%，開思考到 98.7%」這個量級。後續要補的做法見 `docs/handoff-v5-typellm.md`。
- **後端提醒（v6）**：用 HF tokenizer 的引擎（SGLang、vLLM、TypeLLM）對 Gemma 4 預設**不加 `<bos>`**，llama-server 會加。我們量到這一個 token 值 2–3 分（`09-sglang-stability.md`）。TypeLLM 的 Gemma 4 log 裡 prompt 也是以 `<bos>` 開頭，所以他們沒踩到這個，踩到的是空 thought channel。
- Jev 的獨特賣點是校準過的信心；我們量到 raw 信心無鑑別力（ECE 0.18），gemma-jev 沒有報這一項。

生態全景（三十幾個替代品分四類、哪些可抄、哪些我們領先）：`10-landscape.md`。

## 一句話

速度：本地 26B-A4B 的 decode 地板兩邊都在 30–50 ms，差在 prefill 硬體；準確率：這顆模型在「選項少、標準寫清楚」的題上和 Jev 同級或更好，在刻意刁難的 JevBench 上落後 8 分。對產線的封閉判斷題，這個差距不影響「值得做」的結論；對開放、模糊的判斷，Jev 或 SemIf 等級的專門化仍有優勢。
