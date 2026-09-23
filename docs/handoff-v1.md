# GB10 × Gemma 4 26B：Jev 式 typed decision 實驗 — Claude Code 交接文件

日期：2026-09-23
負責人：Clarence
執行者：Claude Code（本文件是起點，請照 Phase 順序推進，每個 Phase 結束時把結果寫回 `results/`）

---

## 0. 一句話目標

在已經部署好的 Gemma 4 26B（GB10 / DGX Spark）上，**不重新部署模型**，加一層「讀選項 logprob」的決策 API，用自己產的合成資料量出：

1. 反應速度（p50 / p95，單題與多題共用 state）
2. 準確率與校準（accuracy、AUROC、ECE、reliability curve）
3. 產線上哪類題目走「LLM 生成」、哪類走「typed decision」的分流規則

最終產出一份 `results/REPORT.md`，含數據表和分流建議。

---

## 1. 背景：這個 session 已經查過的東西

### 1.1 Jev 是什麼（TypeSafe AI，2026-09-15 發表）

- 不生成文字。輸入 state + typed questions（`choice` / `score` / `noul`），一次 forward 回傳選項機率與 confidence。
- 官方宣稱 40–200x 快、$0.042 / M input tokens；獨立實測 p50 236–276 ms（含網路）。
- 「不會幻覺」精確地說是**不會輸出 schema 以外的值**；答案還是可能錯。
- 訓練法 RLCD（Reinforcement Learning for Calibrated Decisions），細節、架構、校準曲線皆未公開；訓練資料 100% 合成。
- 閉源、僅雲端 API → **不能放進 GB10**，只能當對照組。
- 在 TypeSafe 自家 4-workflow benchmark 約 68% 準確率，接近中階 LLM。
- 已知弱點：DAIR Emotion 上 16% 樣本給正確答案 0 機率（靠 confidence 分流時是硬傷）。

### 1.2 Laya（ConvAI Innovations，Apache-2.0，`pip install laya`）

- ModernBERT-large 421M（英文）/ mmBERT-base 322M（多語）encoder，T4 上單題 33 ms。
- **零樣本接近亂猜**（typed-decisions 0.362，隨機 0.318，多數類 0.461）；0.766 是在 benchmark 自己的訓練集 fine-tune 後的數字。
- 英文 checkpoint 遇到非拉丁文字會「很有自信地錯」（孟加拉文 acc 0.080 / conf 0.945）。中文一定要走多語版且自己驗證。
- 選項 >20 時準確率崩（Banking77 0.425 vs Jev 0.870）。
- 結論：**是拿來特化的快速底座，不是零樣本引擎**。本實驗先不用；Phase 6 才視情況拿來蒸餾。

### 1.3 已有人做過的「LLM 讀 logits」實作（可直接引用）

| 專案 | 做法 | 對本實驗的價值 |
|---|---|---|
| **SemIf**（原 OpenJev，TheoLeeCJ/SemIf，3k+ stars） | 凍結 Qwen3.5-4B / 35B-A3B，一次 forward 讀選項 logits；支援 CUDA / MLX / llama.cpp / WebGPU；附 per-workload temperature scaling | JevBench #2（73.1 vs Jev 74.4），最成熟的參考實作 |
| **gemma-jev**（catonooka/gemma-jev） | Gemma 4 E4B on llama-server，`max_tokens:1` + `logprobs`，選項映成單字母 A/B/C，softmax 成分布；後端可換成任何有 logprobs 的 OpenAI 相容 server；26B-A4B QAT q4_0 在 3090 上 p50 48 ms | 直接對應我們的 Gemma 4 26B；但 0 星、10 commits、benchmark n=12–40，**當參考實作，不當依賴** |
| **JEV-CPU-Gemma4**（HeapHeapHooray） | SemIf 的 CPU 移植，gemma-4-E2B-it，bf16 約 3.5 GB RAM，~1 s/題 | 沒 GPU 時的 fallback |
| **Bespoke Nimble**（bespokelabsai/nimble） | Qwen3.5-9B LoRA，對比式造資料（改一個事實讓 label 翻轉），只在答案 token 上訓練；原模型 66.4% → 90.1% | Phase 6 若要 fine-tune，照這個配方 |
| **Kev-0.5B** | Qwen2.5-0.5B LoRA + readout head，MacBook <2 h 訓完，held-out ECE 0.065 | 證明 fine-tune 成本很低 |
| awesome-jev-family / awesome-jev-typesafe | 兩份清單 | 查其他實作 |

### 1.4 gemma-jev 作者整理的四條設計規則（照做）

1. **State = 證據。** 貼原始文字（alarm 原文、CFX event、SPC 數值），不要貼 label。
2. **選項少而正交。** 二選一 / 三選一可到 95–100%，10 個相近選項會掉。
3. **組合交給程式。** 模糊時單一 choice 會在 p≈0.5 搖擺；拆成數個正交 yes/no 再用程式組合。門檻與終止條件是 code 的事。
4. **有「以上皆非」時，Choice 要配 Noul。** Choice 一定選出贏家。

### 1.5 校準的重要區分

- **Temperature scaling** 只修 confidence，**不改變 argmax**（SemIf 文件明講）。
- **Platt scaling（含 bias 項）** 才能移門檻。Laya 釣魚信實驗：raw recall 1.2%，AUROC 0.678（排序能力其實跟 Jev 0.689 差不多），fit 一個 bias 項後 acc 0.611。
- 判讀順序：AUROC 好但 acc 差 → 門檻問題，校準即可；AUROC 本身差 → 改 criteria / 換大模型 / fine-tune。

### 1.6 Line Manager 的背景（給合成資料用）

- IPC-CFX 收 EMS 產線資料，分 **P（生產）/ Q（品質）/ M（機台）** 三類；已有 CT / ATT / UPH 與 SPC；線頭看板 PQM 指標。
- Digital Line Manager 目標：自動查線、自動除錯、自動派工；方法論 5W2H1E + PDCA。
- 語言：工單、alarm、SOP 多為**台灣正體中文**夾英文機台代碼；合成資料要長這樣。

### 1.7 主管版報告（已產出）

`jev-on-gb10-feasibility.html` 已對主管承諾：實驗回答 Q1–Q8（見 §7），並主張「標註需求從每題幾千筆降到幾百筆」。Phase 4 的學習曲線就是用來驗證這句話的；若結果不支持，REPORT.md 要明講，主管版報告要改。

---

## 2. Phase 0：盤點現有部署（先做，決定後面走哪條路）

> 目的：確認現有 Gemma 4 26B 服務能不能直接共用。**不要重新部署模型。**

### 2.1 要查清楚的事（寫進 `results/00-inventory.md`）

```
[ ] 推論引擎：llama-server / vLLM / SGLang / Ollama / 其他？版本？
[ ] 模型檔：Gemma 4 26B 的哪個變體？（26B-A4B？dense？）量化格式（Q4_K_M / QAT q4_0 / FP8 / BF16）？
[ ] 端點：base URL、port、是否有 auth
[ ] API 相容性：
    [ ] /v1/completions（raw prompt）是否可用？
    [ ] 是否支援 logprobs / top_logprobs？最多回幾個？
    [ ] max_tokens=1 是否正常？
[ ] Thinking：模型是否會先吐 thinking-channel token？（gemma-jev 提到 QAT build 會）
    → 用 curl 送一個 prompt，看第一個 token 是什麼
[ ] 併發：llama-server 的 -np（parallel slots）設幾個？vLLM 是否開 continuous batching？
[ ] prefix cache：llama-server 是否開 cache_prompt？vLLM 是否開 prefix caching？
[ ] 目前這台的生成負載：誰在用、尖峰時段
[ ] 空間：GB10 128GB 統一記憶體目前用了多少（能否再放一個 E4B 5GB 做專用決策）
```

### 2.2 驗證指令（依引擎擇一）

```bash
# llama-server
curl -s http://$HOST:$PORT/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "<渲染好的 prompt，見 3.2>",
  "max_tokens": 1, "temperature": 0, "logprobs": 20
}' | jq '.choices[0].logprobs'

# vLLM（OpenAI 相容）
curl -s http://$HOST:$PORT/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "<model name>", "prompt": "...", "max_tokens": 1, "temperature": 0, "logprobs": 20
}' | jq '.choices[0].logprobs'
```

### 2.3 Phase 0 的分支決策

| 盤點結果 | 路線 |
|---|---|
| `/v1/completions` + logprobs 可用，無 thinking token 干擾 | **路線 A（預設）**：共用現有 26B，自寫決策層（Phase 1） |
| logprobs 可用，但會先吐 thinking token | 路線 A + 在 prompt 中關掉 thinking（模型的 system 指令或 chat template 參數），確認第一個 token 是答案字母 |
| 引擎不支援 logprobs（例如某些 Ollama 版本） | **路線 B**：另起一個 llama-server 跑同一個 26B GGUF（模型檔共用、process 分開）；或直接跑 E4B Q4_K_M 專用決策（5 GB） |
| 生成負載重、決策延遲不穩 | **路線 C**：E4B 專用決策 server（高頻 yes/no gating）+ 26B 負責多選項難題與生成 |
| 上面都不通 | **路線 D**：SemIf（Qwen3.5-4B，llama.cpp backend，GB10 有預編 sm_121a binary：`merve/llama.cpp-dgx-spark-gb10-sm121a`） |

> GB10 是 aarch64 + Blackwell（sm_121a）。任何要新裝的東西都要確認 arm64 wheel / 容器；llama.cpp 要自己 build 或用上面那份預編 binary。

---

## 3. Phase 1：自寫最小決策層（不依賴 gemma-jev）

> 為什麼自寫：gemma-jev 核心就是一次 `max_tokens:1` + softmax，幾十行；自寫可以直接嵌進 EAP / Line Manager，也避免依賴一個 0 星專案。gemma-jev 的 `server.py`、`docs/api.md` 可以讀來參考。

### 3.1 檔案結構

```
gb10-decide/
├── decide/
│   ├── __init__.py
│   ├── client.py        # 對 OpenAI 相容 server 發 completions + logprobs
│   ├── prompt.py        # 渲染 state + typed question → prompt 字串
│   ├── types.py         # choice / score / noul 的 schema
│   └── calibrate.py     # temperature / Platt（Phase 4）
├── data/
│   ├── gen_synthetic.py # Phase 2
│   └── synthetic/       # 產出的 jsonl
├── bench/
│   ├── latency.py       # Phase 3
│   └── accuracy.py      # Phase 4
├── results/
│   ├── 00-inventory.md
│   ├── 01-smoke.md
│   ├── 03-latency.md
│   ├── 04-accuracy.md
│   └── REPORT.md
└── README.md
```

### 3.2 Prompt 渲染（`prompt.py`）

原則：用模型自己的 chat template 渲染，**然後在 assistant 回合開頭停住**，讓下一個 token 就是答案字母。

```
<system>
你是產線決策引擎。只回答一個字母。
<user>
【狀態】
{state}

【問題】{instructions}
{options rendered as}
A. {label_a}：{criteria_a}
B. {label_b}：{criteria_b}
...

只回答代表正確選項的字母。
<assistant>
答案：
```

實作注意：

- **選項字母 token 化**：Gemma tokenizer 可能把 `A` 和 ` A`（前面帶空白）視為不同 token。渲染時決定答案位置前面是否有空白，然後對 logprobs 回來的 token 同時比對兩種形式。先用 tokenizer 印出 `A`、` A`、`B`、` B` 的 token id 確認。
- **選項 ≤ 8** 時用 A–H；超過就照設計規則拆題。
- **`top_logprobs` 要 ≥ 選項數 + 5**。回傳裡若缺某個字母，記錄為 `missing`，不要當 0。
- **`noul`**（yes/no）= 兩選項 choice（A. 是 / B. 否），回傳 P(是)。
- **`score`**（序位 rubric）= 選項為等級（A. 不急 / B. 盡快 / C. 阻斷），回傳分布與期望值。
- **多題共用 state**：同一份 state、N 個問題 → N 次呼叫，但 prompt 前綴一致，靠 server 的 prefix cache 減少重算。Phase 3 要量「有 prefix cache / 沒有」的差異。

### 3.3 `client.py` 介面

```python
@dataclass
class Decision:
    question_id: str
    kind: Literal["choice", "score", "noul"]
    chosen: str                  # label
    probs: dict[str, float]      # label → p（softmax over option letters）
    confidence: float            # max prob（raw；校準後另存 calibrated_confidence）
    missing: list[str]           # 沒出現在 top_logprobs 的選項
    latency_ms: float
    prompt_tokens: int

def decide(state: str | dict, questions: dict[str, Question], *, backend: Backend) -> dict[str, Decision]: ...
```

### 3.4 Smoke test（寫進 `results/01-smoke.md`）

用 5 個手寫例子（中文 alarm、工單、SPC 數值）跑一輪，確認：

- 第一個 token 是字母（不是 thinking token、不是換行、不是「答」）
- 分布合理（明顯案例 p > 0.9）
- 延遲落在幾十到幾百 ms

---

## 4. Phase 2：合成資料（`gen_synthetic.py`）

### 4.1 題目清單（對應 PQM 與 Digital Line Manager）

每個 task 一個 jsonl，每筆含 `state`、`question`、`gold`、`difficulty`（easy / hard）、`lang`（zh / mixed）。

| task id | 類別 | 題型 | 選項數 | 說明 |
|---|---|---|---|---|
| `m_alarm_severity` | M | score | 3 | 機台 alarm 原文 → 不急 / 盡快 / 停線 |
| `m_alarm_category` | M | choice | 5 | alarm → 機構 / 電控 / 物料 / 程式 / 環境 |
| `m_needs_dispatch` | M | noul | 2 | 是否要派工給設備工程師 |
| `q_spc_action` | Q | choice | 4 | SPC 描述（連續點、超限、趨勢）→ 持續監控 / 抽檢 / 停線複檢 / 呼叫 QE |
| `q_defect_root` | Q | choice | 6 | 不良描述 → 錫膏 / 貼片 / 回焊 / 來料 / 治具 / 人為 |
| `p_uph_anomaly` | P | noul | 2 | CT / UPH 數列描述 → 是否異常 |
| `p_line_change` | P | choice | 3 | 工單狀態 → 正常換線 / 缺料等待 / 設備待修 |
| `x_ticket_route` | 通用 | choice | 4 | 工單文字 → 產線 / 設備 / IT / 品保 |
| `x_escalate` | 通用 | noul | 2 | 是否需要升級給線長 |
| `x_10way_intent` | 壓力測試 | choice | 10 | 刻意做 10 類相近意圖，驗證「選項多會掉」 |

### 4.2 產生方式

- 用**現有的 Gemma 4 26B 生成**（走 `/v1/chat/completions`，這是 LLM 該做的事），或由 Claude Code 直接寫。每個 task **至少 200 筆**，其中 30% 標 `hard`（訊息不完整、兩類邊界、夾錯字）。
- 語言：80% 台灣正體中文夾機台代碼（例：`SMT-L3 印刷機 DEK-02 錫膏厚度 CPK 1.02 連續 7 點下降`），20% 純英文 alarm code（例：`E4021 NOZZLE VACUUM LOW HEAD2`）。
- **對比式造資料**（Nimble 配方）：每筆 easy 例子再生一筆「改一個事實讓 gold 翻轉」的孿生例子，並記 `pair_id`。這批之後可直接當 fine-tune 資料。
- Gold label 由生成 prompt 指定（先給 label 再生 state），生成完隨機抽 10% 人工看過，記錄不合理率。
- 存檔：`data/synthetic/{task_id}.jsonl`，附 `data/synthetic/MANIFEST.md`（筆數、難度比例、生成 prompt、抽查結果）。

---

## 5. Phase 3：延遲實驗（`bench/latency.py` → `results/03-latency.md`）

固定條件：warm-up 20 次後量測；每組 200 次；記 p50 / p95 / p99；同時記 server 端 `prompt_tokens`。

| 實驗 | 變因 | 要回答的問題 |
|---|---|---|
| L1 單題基準 | state 約 100 token，2 選項 | GB10 上的底線是多少（對照 3090 上 26B-A4B 48 ms） |
| L2 state 長度 | 100 / 500 / 2000 token | prefill 隨長度怎麼長 |
| L3 選項數 | 2 / 5 / 10 | 選項數對延遲影響（理論上很小） |
| L4 多題共用 state | 1 / 5 / 10 題，開/關 prefix cache | 共用 state 的 N 題是接近 1 次還是 N 次的成本 |
| L5 併發 | 1 / 4 / 8 個 client 同時發 | server 的 parallel slot 夠不夠 |
| L6 背景生成負載 | 同時跑一個 512 token 的生成請求 | 決策延遲被生成拖慢多少（決定要不要走路線 C） |
| L7 對照 | 同一題改用 `/v1/chat/completions` 生成 JSON 答案 | typed decision 相對於「LLM 輸出 JSON」快多少 |

（若走路線 C，L1 / L6 再對 E4B 跑一遍。）

---

## 6. Phase 4：準確率與校準（`bench/accuracy.py` → `results/04-accuracy.md`）

每個 task 切 **calibration 50% / test 50%**（依 `pair_id` 切，孿生例子不能跨集）。

指標（每 task、每 difficulty、每 lang 各報一份）：

- accuracy、macro-F1
- **AUROC**（二元題）/ top-2 accuracy（多選項）
- **ECE**（raw、temperature scaling 後、Platt 後）
- reliability diagram（存 png 到 `results/fig/`）
- `missing` 率（選項字母沒進 top_logprobs 的比例）
- **選擇性準確率**：只接受 confidence ≥ 0.8 / 0.9 時的 accuracy 與 coverage（這是分流門檻的依據）

校準做法：

1. temperature scaling（多類）— 在 calibration 集上 fit 一個 T
2. Platt / vector scaling（含 bias）— 二元題用 Platt；多類用 per-class bias
3. 校準**不能改善 argmax** 時記下來，這代表要改 criteria 或換模型

對照組（每 task 至少跑 easy/hard 各 50 筆）：

- `/v1/chat/completions` 讓 26B 生成 JSON 答案（temperature 0）→ 比 accuracy 與延遲
- **規則基線**：每個 task 手寫 10–20 條關鍵字規則（例：`NOZZLE|VACUUM` → 機構），量 accuracy。這是「傳統做法」的地板
- **傳統 ML 基線**：TF-IDF（char n-gram，中文不用斷詞）+ logistic regression，用 calibration 集訓練、test 集評估
- 若時間允許：同一批資料丟 Laya 多語版（零樣本）當「小 encoder 底線」

**標註量學習曲線**（給主管報告用，回答「還要不要 learning」）：

- 對 `m_alarm_category`、`x_ticket_route`、`q_spc_action` 三個 task，從 calibration 集抽 N = 25 / 50 / 100 / 200 / 500 筆，各跑 3 個 seed
- 三條線：傳統 ML（用 N 筆訓練）、typed decision 零樣本（N 筆只做校準，argmax 不變）、typed decision + 校準後的選擇性準確率 @0.9
- 存 `results/fig/learning-curve-{task}.png`；REPORT.md 要寫「達到 90% 各需要幾筆」

---

## 7. Phase 5：分流規則（寫進 `results/REPORT.md`）

用 Phase 3 / 4 的數據填這張表，**不要先入為主**：

| 題目特徵 | 預期走 | 判斷依據（實測後填） |
|---|---|---|
| 答案是封閉 label、選項 ≤ 5、每天百次以上 | typed decision | 選擇性準確率 @0.9 ≥ 95% 且 coverage ≥ 70% |
| 10 類以上細粒度分類 | 先拆題再 typed；不行則 LLM | `x_10way_intent` 的結果 |
| 需要說明「為什麼」（RCA、報告、派工單內容） | LLM 生成 | 本質上要文字 |
| 需要多步推理或查資料才能決定 | LLM（或 LangGraph playbook 內部再呼叫 typed decision） | — |
| 模糊、需要「以上皆非」 | typed：choice + noul 配對，低信心升級 LLM | `x_escalate` 的校準曲線 |
| 中文夾機台代碼 | 看 `lang=mixed` 分項 | 若中文明顯差 → 把 state 先正規化 |
| 關鍵字規則就能解 | code（規則） | 規則基線 ≥ 95% 的 task 不上模型 |
| 數值進、數值出（SPC、CT 分布、時序異常） | 統計 / 傳統 ML | 不在本實驗範圍；typed decision 只讀「已判定的文字描述」 |

分流架構草案（照 gemma-jev 的三層說法）：

```
code（always）   ：門檻、組合、終止、數學（CT/UPH/SPC 本來就是公式）
typed decision（often）：判斷、分類、gating、排序 — 每天數百到數千次
LLM（rare）      ：規劃、寫 RCA、處理意外 — 由 typed decision 低信心時觸發
```

REPORT.md 要回答的具體問題：

1. GB10 上 26B 做 typed decision 的 p50 / p95 是多少；跟生成 JSON 相比快幾倍
2. 哪些 task 零樣本就夠（acc、選擇性 acc）；哪些需要改 criteria；哪些校準也救不回來
3. 多題共用 state 的實際成本（決定 playbook 裡一次問幾題）
4. 生成負載對決策延遲的影響（決定要不要獨立 E4B）
5. 中文 vs 英文 alarm 的落差
6. 建議的 confidence 門檻（每個 task 一個，不是全系統一個）
7. 標註量：傳統 ML 與 typed decision 各需要幾筆標註才到 90%；typed decision 在 0 筆（純零樣本）時落在哪
8. 規則基線在哪些 task 已經夠用（若關鍵字規則就 95%，那個 task 不該上模型）

---

## 8. Phase 6（選做）：什麼時候才 fine-tune

只有 Phase 4 出現「AUROC 本身差、改 criteria 也沒用」的 task 才進來。

- 先試換更大的 backbone（若現在是 A4B MoE，試 dense 或更高精度）。
- 再試 **Nimble 配方**：LoRA、對比式資料（Phase 2 已產 `pair_id` 孿生例子）、只在答案 token 上訓練。GB10 128GB 記憶體夠，但頻寬低，預期數小時等級；先用 500–2000 筆試。
- 若要壓到 30 ms 等級且量很大：拿 26B 的輸出當 teacher，蒸餾 **Laya 多語版**（mmBERT-base 322M），照官方 Kaggle notebook 流程。

---

## 9. 交付檢核

```
[ ] results/00-inventory.md   現有部署盤點 + 路線決定
[ ] decide/ 可 import，README 有 3 行 quickstart
[ ] results/01-smoke.md       5 例 smoke test 通過
[ ] data/synthetic/*.jsonl    10 個 task × ≥200 筆，MANIFEST.md
[ ] results/03-latency.md     L1–L7 表格
[ ] results/04-accuracy.md    每 task 指標 + reliability diagram + 規則/傳統 ML 基線
[ ] results/fig/learning-curve-*.png  三個 task 的標註量學習曲線
[ ] results/REPORT.md         分流規則 + 8 個問題的答案 + 下一步
```

## 10. 參考連結

- TypeSafe 發表文：https://typesafe.ai/blog/introducing-system-one-models-and-jev
- SemIf：https://github.com/TheoLeeCJ/SemIf
- gemma-jev：https://github.com/catonooka/gemma-jev
- JEV-CPU-Gemma4：https://github.com/HeapHeapHooray/JEV-CPU-Gemma4
- Laya：https://huggingface.co/convaiinnovations/laya
- Laya vs Jev 獨立實測（5090）：https://huggingface.co/datasets/Luni/laya-jev-benchmark
- awesome-jev-family：https://github.com/notsointresting/awesome-jev-family
- JevBench：https://benchmarkheaven.com/jev-models
- Jev 架構外部推測：https://archerhume.com/posts/jevs-architecture-unmasked/
- llama.cpp GB10 預編 binary：https://huggingface.co/merve/llama.cpp-dgx-spark-gb10-sm121a
