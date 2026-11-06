# Handoff v20 — 同一批題、同一張卡：LLM call、jevlike E4B、jevlike 26B、EG2 的速度（2026-10-15，跑前寫死）

來源：eg2poc 的三車道圖（LLM call = prefill + decode；jevlike = 只 prefill；EG2 = encode 一次 + 矩陣乘）與它在 4 vCPU 上量到的數字（EG2 投影每筆 241 ms、E2B 7.2 s、E4B 16.1 s）。
那是 CPU 對 CPU、不同題。這輪把四個引擎放到**同一張 L4、同一批題**上量，速度與準確率放同一張表。

## 0. 規則

- 題目只挑 EG2 做得來的題型（v13 的主題型 + 一題中間型，加 eg2poc 自己的四類狀態題）。每個引擎答**完全相同**的題。
- 合成資料。不訓練、不改模板。26B UD-Q4_K_M、E4B Q8_0、llama.cpp b11371、`-c 16384 -np 4 --swa-full --reasoning-budget 0`、不開 MTP。EG2 `google/embeddinggemma-2` @ `914f7f89…`，sentence-transformers。
- 計時一律 `perf_counter` 包整個請求（client 端看到的）；每個情境先暖機再量。
- 準確率在同一批題上重算，不引用舊輪的數字（v13 的數字用來寫預測）。

## 1. 題

| 題組 | 來源 | 題數 | 選項 | gold |
|---|---|---|---|---|
| 工單分派 | D0 test | 100 | 4 | 字母 |
| 十種意圖 | D0 test | 100 | 10 | 字母 |
| alarm 原因分類 | D0 test | 100 | 5 | 字母 |
| 停線原因 | D0 test | 100 | 3 | 字母 |
| **專案狀態（eg2poc）** | `data/v20/eg2poc_status.jsonl`：eg2poc `synthetic-v2.1/analysis-plan.json` 的 100 個 benchmark pair 的 `new_text`（zh 50／en 50） | 100 | 4（`labels_alt`：按計畫／延後／阻礙／取消） | eg2poc 的 pair `label`（1 = 狀態已改變 → 答案應在 B/C/D；0 → A），照它的 `single_score` 規則，分數 = P(B+C+D) |

專案狀態題的問法完全照 eg2poc 的 J 單邊模式：`single_instructions[lang]`、選項 = `labels_alt[lang]`、英文題用 `SYSTEM_EN`。

## 2. 引擎

| 引擎 | 做法 | 回傳 |
|---|---|---|
| **A LLM call** | 26B `/v1/chat/completions`，同一份 state + 題目 + 選項，答案句改成「以 `{"answer": "<字母>"}` 回答」，貪婪、`max_tokens 16`，parse 字母 | 字母，無機率 |
| **B jevlike E4B** | 讀字母第一 token | 機率 |
| **C jevlike 26B** | 同上 | 機率 |
| **D EG2 GPU（L4）** | D0 題用 v13 的 Z1：分類前綴編 state 與選項（label：criteria），選項向量題內中心化，cosine，softmax(S／0.045)；專案狀態題照 eg2poc：text 用 document 模板、標籤用 classification 模板、不中心化、同一個 softmax | 機率 |
| **D′ EG2 CPU（4 vCPU、4 threads）** | 同 D | 同 D |

E2B／E4B 的 CPU 數字引用 eg2poc（7.2 s／16.1 s 每題，Xeon 8573C 4 threads），不重跑。

## 3. 情境（對應三車道圖的「100 筆」算式）

| 情境 | 怎麼量 | LLM | jevlike | EG2 |
|---|---|---|---|---|
| **S1 單筆** | 500 題依序、一次一筆，p50／p95 | prefill + decode | prefill | encode + 矩陣乘 |
| **S2 批次 100** | 每題組 100 筆，各引擎自己最好的批法：llama 4 個 slot 並行；EG2 batch 64 | 100 × (prefill + decode) | 100 × prefill | 一次 encode |
| **S3 同一批再問一題** | 同 100 筆 text，換第二道題（工單分派 ↔ alarm 分類、十種意圖 ↔ 停線原因互換題目；專案狀態題換 `labels` 標籤組）；只量時間不計分 | 再 100 次完整呼叫 | 再 100 次，state 前綴在快取（記 `cache_n`） | 只編新標籤 + 矩陣乘，text 不重 encode |

每個情境報：每百筆秒數、每筆 ms、換算 L4 費用（$0.80／h）。另記載入時間、prewarm 時間、GPU 記憶體（不設門檻）。

## 4. 預測

- **S1 GPU 單筆**：A 約 450–600 ms；C 約 140 ms；B 約 65 ms；D 約 70 ms。**EG2 單筆不比 E4B 快。** D′ CPU 約 250–350 ms。
- **S2 批次 100**：D 約 0.3 s；B 約 3.5 s；C 約 7 s；A 約 15–25 s。D 比 B 快 ≥ 10×、比 A 快 ≥ 50×。
- **S3 再問一題**：D 約 0.02 s；C 約 2–4 s（快取命中，`cache_n` ≥ state 長度）；B 約 1–2 s；A 與 S2 同量級。
- **準確率**：D0 四題 EG2 0.65–0.85，E4B ≥ 0.92，26B ≥ 0.98，A 與 C 同量級（v1 量過生成 JSON 與讀字母同準）；專案狀態題 EG2 與 E4B 差 ≤ 0.05（eg2poc 的 T6 結論）。

## 5. 判定規則

這輪是量測，不是賭；但「某題型該交給 EG2」的規則先寫死：

- **R1**：該題型 EG2 準確率 ≥ E4B − 0.05，**且** S2 批次速度 ≥ E4B 的 10 倍。兩條都要。
- 預測：D0 四題沒有一題過 R1 第一條（速度再快也不用）；**專案狀態題是唯一兩條都過的**。
- **R2 LLM call 的位置**：若 A 的準確率與 C 差 ≤ 0.01 而 S1 慢 ≥ 3×、S2 慢 ≥ 2×，結論維持「讀字母取代生成」。
- 預測錯了照實寫。

## 6. 結論（2026-10-15，照 §5 填；`results/24-speed.md`、`v20.json`）

### 規則

| 規則 | 結果 | 判定 |
|---|---|---|
| R1 工單分派 | EG2 0.780 對 E4B 1.000；批次快 48× | 準確率 ✗ → 不用 EG2 |
| R1 十種意圖 | 0.713 對 0.921；64× | ✗ |
| R1 alarm 原因分類 | 0.650 對 0.920；39× | ✗ |
| R1 停線原因 | 0.840 對 0.980；30× | ✗ |
| **R1 專案狀態（eg2poc）** | 0.700 對 0.720；32× | **兩條都過，唯一交給 EG2 的題型** |
| R2 LLM call | D0 四類 0.970 對 0.990；S1 慢 2.7×、S2 慢 1.5× | S2 沒到 2× → ✗（理由見下） |

### 數字（L4；每筆 ms）

| 引擎 | S1 單筆 p50 | S2 批次 100（冷） | 第二題（同 state，S3b） | 準確率 D0 四類 |
|---|---|---|---|---|
| A LLM call 26B | 391–425 | 209–272 | 355–403 | 0.950–0.990 |
| C jevlike 26B | 131–172 | 124–190 | 97–154 | 0.980–1.000 |
| B jevlike E4B | 71–101 | 70–111 | 49–83 | 0.920–1.000 |
| D EG2 L4 | 70–73 | **2–3** | ≈ 0 | 0.650–0.840 |
| D′ EG2 CPU（Modal 4 vCPU） | 79–117 | 37–65 | ≈ 0 | 同上 |

每百萬筆 L4 費用（S2 冷）：LLM call $50.6、jevlike 26B $33.6、E4B $20.2、EG2 $0.5。

### 三件跑了才知道的事

1. **第一次套件跑的 S2 是假的。** S2 緊接在 S1 之後送一模一樣的 prompt，llama-server 的 prompt 快取整段重播（26B 每筆 31 ms），不是批次速度。診斷跑（同一個 server，冷 → 換題 → 暖）證實：冷 147 ms、換題 130 ms、暖 129 ms，server 日誌有「making room for prompt cache entry」。**量批次要冷跑**，寫進坑。
2. **llama-server 在 L4 上 4 個 slot 並行不加吞吐。** 冷批次每筆 147 ms（26B）≈ 單筆 144 ms，4 路在飛時每請求 577 ms。v1 的 L5 早就量到（5.5 req/s 卡住），這次又量到一次。要批次就要 SGLang（v4：K=16 快 3.7×、c=32 快 6×），這輪沒跑。
3. **同一份 state 再問一題，state 短的時候省不多。** 這批 state 約 100 token，第二題只省 10–30%（26B 137–154 ms 對單筆 144）；v2 的「第二題起 73 ms」是 1,100 token 的 state。EG2 的第二題是真的接近 0。

### 預測對答案（15 條對 7 條）

對：jevlike 26B 單筆約 140、EG2 單筆約 70、EG2 批次 ≤ 0.5 s、LLM 批次 15–25 s、EG2 批次 ≥ 10× E4B、≥ 50× LLM、EG2 第二題 ≈ 0、R1 只有 eg2poc 過。
錯：LLM 單筆 391–425（預測 450–600）；E4B 單筆 71–101（預測 65）；**EG2 單筆比 E4B 快一點**（70 對 83，預測打平）；EG2 CPU 79–117（預測 250–350，Modal 的 CPU 比 eg2poc 的 Xeon 快兩三倍）；E4B 批次 9.1 s、26B 15.1 s（預測 3.5／7，因為 llama-server 不會並行）；jevlike 26B 第二題 140 ms（預測每筆 20–40 ms，state 太短）；eg2poc 題 EG2 與 E4B 差 ≤ 0.05 ✓ 但那題所有引擎都只有 0.70–0.73（AUROC 26B 0.984、E4B 0.963、EG2 0.884）。

### 決定

- EG2 只在「粗類、互斥、像哪一類、準確率跟 E4B 打平」的題上用；這輪五題只有 eg2poc 的專案狀態題符合。D0 四類差 10 到 27 點，速度再快也不用。預設不變。
- LLM call 單筆慢 2.7 倍、批次慢 1.5 倍、準確率低 2 點；R2 的「批次慢 ≥ 2×」沒過是因為兩邊都不批次。「讀字母取代生成」維持，理由改成單筆延遲與準確率。
- GB10 若要批次，判斷層用 SGLang 或接受 llama-server 的單筆速度。
費用：L4 約 1.4 h（套件 2 個 llama 容器 + EG2 + 三個診斷）、CPU 容器 10 分鐘，約 $1.2。
