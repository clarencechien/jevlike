# Handoff v5 — 參考 TypeLLM 之後要補測、補試的計劃（2026-09-26，跑前寫死）

來源：<https://github.com/TypeLLM/TypeLLM>（commit 9d622e0）。他們用 SGLang + Qwen3.8-27B 做 JSON Schema 型別安全輸出，
JevBench 公開 231 題子集無思考 195/231、開思考 228/231。與 jevlike 的差別、能抄的東西見 §1；本檔只列**要動手驗證**的項目，
每項先寫門檻再跑（同 v3/v4 慣例）。

## 0. 先講兩個已經確認、不用跑 GPU 的事

1. **TypeLLM 不支援 Gemma 4**。`typellm/protocol.py` 遇到 `<|turn>`+`<|channel>` 直接 raise；他們 9/22 用 E2B 實測 42 題只對 24，
   log 顯示第一個 token 是 `The`（logprob −0.001），A/B/C 全在 −11 以下，原因是沒放空的 thought channel。jevlike 的 `GEMMA4_TEMPLATE_NOTHINK`
   解掉了這題，寫進 `06-comparison.md`。
2. **top-k 讀不到的字母不影響結果**。TypeLLM 用 `token_ids_logprob` 指定 token 讀機率；我們用 `n_probs=40` top-k，讀不到的字母當 0。
   檢查 v2 D0 原始資料：2,000 筆有 661 筆至少一個字母不在 top-40，但這些筆 top-40 的地板 logprob 最高 −14.1，
   漏掉的機率上界 7.4e-7，對 softmax 與信心門檻無影響。llama-server 路徑不改；SGLang 路徑改用 `token_ids_logprob`（T4，順手）。

## 1. 對照：TypeLLM 有、jevlike 沒有的

| TypeLLM 做法 | jevlike 現況 | 處置 |
|---|---|---|
| 標籤先過 tokenizer 驗證（單 token、decode 回同字串） | 假設 A–J 單 token，沒驗 | **T0-a** 加啟動自檢 |
| 評測附獨立驗證腳本（不用模型重算 acc / Brier / ECE / 機率和） | 原始 logprobs 有，驗證腳本無 | **T0-b** `bench/verify.py` |
| 答案 prefill 成 JSON 形狀 `{"name": "`，下一個 token 就是標籤 | 試過「答案：」變差；JSON 形狀沒試 | **T1** |
| 選項順序置換平均（permutations） | 沒做；三個弱 task 都是相鄰等級 | **T2** |
| SGLang `token_ids_logprob` 指定 token | top-40 | **T4**（順手，不影響結論） |
| nullable / 「以上皆非」統一語法 | `x_escalate` 有 noul 選項，無統一語法 | 不做：抽查只有 5 筆 ARGUABLE，沒有足夠正例；等真實資料 |
| `depends_on` 依賴鏈 + 前綴重用 | S2 場景 7 題獨立 | 不做：合成資料每 task 各自的 state，沒有鏈式題；真實資料有「先分類再判急迫度」時再開 |
| thinking mode（JevBench 84%→98.7%，均 919 token/題，p95 61 s） | 關 | 不做：產線 gating 用不起；3c 級聯是更便宜的換法 |
| 數值欄位逐位受限解碼 | v4 Phase 2 未開 | 不做；若日後開，抄他們 `thinking_diagnosis_report.md` 的教訓（指令與文法完全對齊、允許集全部低機率時失敗不硬選） |

## 2. 要跑的項目與門檻

環境：Modal L4（與 v2/v3 同卡，數字可直接對照）、`unsloth` UD-Q4_K_M、`--swa-full`、`-np 4`。資料：`data/synthetic` D0，
split 同 `bench/analyze.py`（pair_id 分組 seed 0，50/50 cal/test）。弱 task 指 `m_alarm_severity`、`q_spc_action`、`p_uph_anomaly`（v1 criteria，
與 04-accuracy 對照）；控制 task `m_alarm_category`（強、3 選項）。

### T0 不用 GPU（先做）

- **T0-a 標籤自檢** `decide/labels.py`：對 llama-server `/tokenize`（`add_special: false`, `with_pieces`）與 SGLang `/v1/tokenize`
  驗證每個字母是單 token、且 detokenize 回同字串；`bench/smoke.py` 啟動時呼叫，失敗即停。驗收：既有 smoke 結果不變、記錄 token id 表到 `01-smoke.md`。
- **T0-b 驗證腳本** `bench/verify.py`：讀 `results/modal/accuracy/**/*.jsonl`，重算每檔 n、acc、Brier、ECE(10 bins)、機率和是否在 1±1e-3、
  missing 率，輸出 `results/verify.md`；與 `analysis.json` 的 raw_test acc 對得上（±0.001）才算過。
- **T0-c** `06-comparison.md` 加 TypeLLM 列（231 題子集，與 Jev 534 題不能直接排名）與 §0.1 的 Gemma 4 註記。

### T1 JSON 形狀 prefill（L4，約 5 分鐘）

變體（同一組 prompt 只改結尾）：
- V0 現行：`…<|turn>model\n<|channel>thought\n<channel|>`，下一 token 即字母。
- V1 prefill：V0 + `{"answer": "`（無尾空白；下一 token 即字母）。
- V2 prefill + 指令：user 段最後一句改為「以 {"answer": "<字母>"} 回答」，prefill 同 V1。

步驟：smoke 5 題 × 3 變體，first token 必須是字母、missing=0；然後 3 弱 task + 控制 task × D0 全 200 筆 × V1/V2（V0 用既有 `04-accuracy` 資料）。
判定（test 100 筆/ task，McNemar 對 V0）：
- **採用**：弱 task 平均 acc ≥ V0 + 2 點，且沒有任何 task（含控制）低於 V0 − 1 點。
- **不採用**：其餘情況。V1 與 V2 都過時取 acc 高者；同分取 V1（不改指令）。

### T2 選項順序置換平均（L4，約 10 分鐘）

對每題把選項順序重排，機率對齊回原標籤後平均。次數：3 選項 task 用全部 6 種，4–5 選項用 8 種抽樣（seed 0，含原順序）。
task：3 弱 + 控制，D0 全 200 筆（cal 也要，因為要重做校準）。每題記錄每個順序的機率向量。
指標（test 100 筆）：acc、ECE、AUROC、sel@0.9 coverage、**順序敏感度**（同一題不同順序 argmax 不一致的比例）。
判定：
- **建議用在升級路徑／離線**：弱 task 平均 acc ≥ 單次 + 2 點，**或** ECE 降一半以上且 acc 不掉；控制 task 不掉超過 1 點。
- **不建議**：其餘。另外無論過不過，順序敏感度 ≥ 10% 的 task 要寫進 REPORT 當風險。
成本：每題 6–8 次前向，state 共用（`--swa-full`），估 (6+8+8+6)×200 ≈ 5,600 次 × ~80 ms ≈ 8 分鐘。

### T4 SGLang 指定 token 讀機率（順手，選配，約 $0.3）

`read_option_probs_sglang` / batch 版改用 `token_ids_logprob`（字母 token id 由 T0-a 的自檢取得），去掉 `top_logprobs_num=40`。
驗收：`bench/smoke_sglang.py` 5 題機率與 v4 smoke 的差 < 1e-3。不改 v4 結論（SGLang 仍不進即時層）。

## 3. 順序、預算、輸出

順序：T0 → T1 → T2 →（T4）。T1 若採用，T2 用採用後的 prefill 跑。
預算：L4 約 0.4 h、< $0.5；T4 另 L40S 約 10 分鐘。
輸出：`results/08-typellm-followups.md`（T1/T2 表 + 判定）、`results/verify.md`、`06-comparison.md` 更新、REPORT.md 加 §3e、
HTML 只在 T1 或 T2 有「採用／建議」時更新一段，否則只補一句。

## 4. 一句話結論（三選一，跑完填）

- 「JSON 形狀 prefill／順序平均讓弱題再升 X 點，**納入預設做法**。」
- 「順序平均只在升級路徑值得（每題多 6–8 次前向），**預設不開**；prefill 無差。」
- 「兩者都無差，**維持現行 prompt**；TypeLLM 可抄的只剩標籤自檢與驗證腳本。」
