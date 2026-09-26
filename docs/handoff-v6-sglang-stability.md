# Handoff v6 — SGLang 掉分與不穩的兩個對照實驗（2026-09-26，跑前寫死）

接 v4（`results/07-sglang-accuracy.md`）：SGLang 在 Gemma 4 26B-A4B 上比 llama-server 低 2–3 點，且同題重跑答案會變。
v4 已排除 FP8、RadixAttention、併發三個原因，沒有找到真正的原因。本檔先排兩個最便宜、最可能命中的實驗；
每個都是一次 D0 test（10 task × 100 題）、L40S 約 15 分鐘、約 $0.5。

## 0. 前提：先問要不要救

v4 要 SGLang 的理由是「多題共用上下文快 3.7 倍、併發快 6 倍」，這是在 L4/L40S 上量的。GB10 有 128 GB 統一記憶體，
llama-server 可以開很多 slot，L4 上「併發不加速」的結論在 GB10 上不一定成立。**`bench/v4bench.py` 的 L5（c=1/8/32）在 GB10 上先跑一次**
（`REPORT.md` §5 本來就有這項）。若 llama-server 在 GB10 上 c=32 的 decisions/s 已達即時需求，本檔的實驗可以不做。

## 1. 兩個問題，兩個實驗

| 問題 | 症狀（v4 數字） | 實驗 |
|---|---|---|
| 系統性掉分 | D0 0.959 → 0.931；bf16/H100 仍低 2.6 點 | **E1 tokenizer 對齊** |
| 同題重跑會變 | 三次同設定答案兩兩只 97% 一致；翻掉的 50 題有 32 題原本把握 >0.9 | **E2 批次不變推理** |

### E1 tokenizer 對齊：把 llama-server 切的 token 直接餵給 SGLang

**假設**：同一段 prompt，HF tokenizer 切 116 個 token、GGUF tokenizer 切 117 個（`07-env-sglang.md`），模型看到的輸入本來就不同，
2–3 點可能全部或部分來自這裡，而不是 kernel。

**依據**：兩邊 vocab 的 token id 相同（v5 T4 量到字母 id A=236776、C=236780 在 HF 與 llama-server 一致），所以 id 可以直接互餵。

**做法**：
1. `decide/client.py` 加 `read_option_probs_sglang_ids(base_url, input_ids, letters, token_ids)`：`/generate` 送 `input_ids` 而不是 `text`，
   `token_ids_logprob` 讀字母（v5 T4 已有）。
2. `bench/accuracy.py` 加 `--tokenizer llama`：先用 llama-server `/tokenize`（`add_special: true`，與 `/completion` 一致）把 prompt 切好，
   存 ids；SGLang 端用 ids 讀機率。需要兩個 server 同時在：llama-server 只當 tokenizer（CPU 容器即可，不用 GPU；或先在 L4 批次切好存成 `data/tokenized/D0.jsonl`）。
   **先切好存檔**比較簡單：新增 `bench/tokenize_dump.py`，在 llama-server 上把 D0 test 1,001 題的 prompt ids 存成檔，SGLang 那邊讀檔。
3. 同時記錄兩邊 tokenization 的差異在哪裡（第一個不同的 token 位置與字串），寫進報告。

**Arm**：B1 原設定（v4 已有，不重跑）、**B1-ids**（同 server、同 FP8、餵 llama ids）、A1 llama-server Q8（v4 已有）。

**判定（pre-registered）**：
- B1-ids 比 A1 每 task 差 ±2 點內且 McNemar p ≥ 0.05 的 task 數 ≥ 16/20（D0 + D1-cue；D1-cue 若不跑則 ≥ 8/10）→ **掉分是 tokenizer 造成**，
  解法是 SGLang 端換用與 GGUF 一致的 tokenizer 行為（或永遠用 ids 餵）；v4 的「後端固有」結論撤回。
- B1-ids 與 B1 差 < 1 點 → tokenizer 不是原因，進 E3。
- 介於中間 → tokenizer 是部分原因，報告拆開寫。

### E2 批次不變推理：`--enable-deterministic-inference`

**假設**：不穩來自 batch 組成不同 → 矩陣乘法分塊與歸約順序不同 → bf16 小數位不同 → 兩個字母機率相近的題翻面。
SGLang 0.5.x 的 `--enable-deterministic-inference` 用批次不變 kernel，理論上同一題不論跟誰湊 batch 結果都一樣。

**做法**：`modal_sglang.py --server-extra "--enable-deterministic-inference"`，跑 D0 兩次（`--out-sub D0-det-a`、`D0-det-b`，workers=8）。
另外重跑 v4bench 的 L1、L4（warm）、L5，量速度代價。

**判定（pre-registered）**：
- 兩次答案一致率 ≥ 99.5%（v4 是 97%）**且**翻掉的題中把握 >0.9 的為 0 → **不穩解掉**。
- 一致率 ≥ 99.5% 但 G2（K=10,16 ≥ 1.5×）或 G3（c=8,32 ≥ A1）掉到沒過 → 解掉但速度優勢不夠，SGLang 只給批次。
- 一致率 < 99.5% → 不穩另有原因（radix 命中路徑、CUDA graph），記錄後停。
- 注意 E2 只治「會變」，不治「低 2–3 分」；若 E1 沒過而 E2 過，SGLang 仍是「穩定地低 2–3 分」，分工使用。

### E3（E1 沒命中才做）：kernel 與精度 — **不做，E1 已命中（差距 0.001）**；剩下 0.15% 的翻面留到 GB10 實測後再看

各一次 D0，各約 $0.5，任何一個把差距縮到 1 點內就停：
1. `--attention-backend flashinfer`（L40S 上 v4 自動選了 triton）。
2. `--moe-runner-backend` 換一種（v4 的 MoE Triton 設定檔是手寫保守版）；或在 H100 上用 SGLang 自帶的設定檔重跑 bf16（B2）。
3. lm_head / logprob 以 fp32 計算（查 SGLang 有無對應選項；TypeLLM 用「NVFP4 權重 + BF16 lm_head」的權重檔，說明這一層敏感）。

## 2. 順序與預算

1. GB10 上 llama-server L5（不花錢，半小時）→ 夠用就結案，本檔其餘不做。
2. E1（$0.5）與 E2（$0.5 + 速度重量 $0.5）可以平行跑。
3. E3 只在 E1 沒命中時做，最多 $1.5。
合計上限約 $3、一天。

## 3. 輸出

`results/09-sglang-stability.md`（E1 tokenizer 差異位置表、B1-ids vs A1 護欄表、E2 兩次一致率與翻面把握分布、速度代價）、`bench/analyze_v6.py`、
REPORT.md §3d 改寫結論。若 E1 命中，`06-comparison.md` 與 HTML 的「後端」一段要撤回「後端固有」的說法。

## 4. 一句話結論（三選一，跑完填）

**跑完（2026-09-26）**：E1 命中，差異在第 0 個 token：llama-server 加 `<bos>`、SGLang text 路徑不加；餵同樣 id 後平均 0.958 vs 0.959，護欄 8/10。
E2 單獨開旗標（仍無 `<bos>`）一致率 99.2%；補上 `<bos>` 再開旗標 99.85%，2,000 題翻 3 題、1 題高把握，差門檻一題。
速度代價：開旗標單題 63 → 81 ms、K=16 217 → 301 ms、c=32 111.7 → 85.5 dec/s，G2/G3 仍過、G1 沒過。
→ 選第一句（掉分有解、不穩接近解），**SGLang 回到候選**；修正已進 `decide/prompt.py`（`GEMMA4_TEMPLATE_NOTHINK_BOS`），smoke 驗證 token 數一致。


- 「掉分是 tokenizer 差異、不穩是 batch 數值路徑，兩個都有解，**SGLang 可回到候選**；重跑 v4 門檻。」
- 「不穩解掉、掉分解不掉，SGLang **穩定地低 2–3 分**，只給可容忍的批次工作。」
- 「兩個都沒解，**維持 llama-server**，GB10 上用多 slot 換併發。」
