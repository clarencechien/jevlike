# Handoff v10 — Cloudflare Clef / Clef-flash 同尺對照（2026-10-02，跑前寫死）

來源：2026-10-01 Cloudflare 發布的 Clef（Qwen3.8-27B）與 Clef-flash（Qwen3.5-9B）。兩者都是凍結主模型，加 rank-256 LoRA 和 joint schema head，一次 prefill 對所有選項打分。Apache 2.0，權重在 HF（`cloudflare/clef`、`cloudflare/clef-flash`）。
官方數字只有英文意圖分類（BANKING77、CLINC150）和自家 edge 的延遲（flash p50 38.8 ms、27B 209 ms、Jev 524 ms），**沒有中文，也沒有 JevBench 公開子集的分數**（我們抓到的頁面都沒有）。
本輪要回答一個問題：**在我們的中文產線題上，Clef 能不能取代「凍結 Gemma 26B 讀字母」？**

## 0. 值不值得跑

**值得，這是九輪以來第一個可能直接改變上線選型的實驗。**

- 如果 Clef-flash 在 D0 上與 26B 打平，延遲又是 1/5，GB10 上的判斷層就該換成它。9B 的 BF16 約 18 GB，能把 26B 的記憶體讓給生成模型。
- 如果它在我們的弱題（急迫度、SPC、UPH）或 v9 沒過的兩格（工具守門 90.0%、重排序 nDCG@5 0.840）明顯更好，就是訓練判斷頭比零訓練讀字母多出來的價值，這正是 `10-landscape.md` 一直沒能量的那一欄。
- 如果中文不行，也要有數字才能說不換。

得不到的：它在真實產線資料上的表現（要到 GB10 上做，見 §7）、RL 微調後的效果（官方還沒開放自助）。

## 1. 安全與授權

- 只用合成資料：D0、`data/hard`、v9、JevBench 公開集。**真實產線資料不上 Modal，也不送 Workers AI。**
- 權重 Apache 2.0，可以下載到 Modal volume，但**不 commit 進 repo**；`MANIFEST` 記 HF revision 與 sha256。
- Workers AI arm 是選配（§4 W），只送合成題，而且要使用者自己提供 `cloudflare_token` 環境變數；沒給就跳過，不阻擋本輪。

## 2. 步驟 0：讀卡、寫 adapter（無 GPU）

1. 讀兩個 HF 模型卡和 `joint_schema_model.py`，在 `results/15-clef.md` §0 記錄：
   - 輸入 schema 長什麼樣；它說「與 Jev API 相容」，具體是哪個版本的欄位。
   - 選項上限、有沒有 criteria 欄位、system prompt 能不能改。
   - 回傳的是 softmax 機率還是 logit。
   - 需要的 torch / transformers 版本（卡上寫 torch 2.11、transformers 5.10.2）。
2. 寫 `decide/clef_adapter.py`，把我們的題目（state、question、options 及每個選項的 criteria）轉成它的 schema。
   - **不改題目內容**：選項文字、criteria、題序都照 D0 原樣。
   - 唯一允許的改動是格式轉換，有任何不得不改的地方（例如選項數上限）逐條記錄。
3. Score 題（v9 的 E 評分、JevBench Score）照 v8 的做法：期望等級與 argmax 都報。
4. Noul 題用二選一 `Yes / No`，同 v8。

## 2.5 硬體怎麼選（為什麼 27B 只在 H100 上跑 50 題）

- **記憶體**：L40S 只有 48 GB。27B 的 BF16 權重約 54 GB 放不下，flash 的 BF16 約 18 GB 放得下。
  MoE 不省記憶體：我們的 26B-A4B 也要存完整 26B 參數，之前放得進 L4／L40S 是因為用了 Q4（約 16 GB）與 FP8（約 26 GB）。
- **參考值一定要 BF16**：判斷層是自訂的，BF16 跑 transformers 加官方 `load_release_model` 是唯一確定照官方方式算的版本。
  所以 27B 只在 H100 上用 BF16 跑 50 題當參考，主跑改在 L40S 用 FP8 或 Q8（約 27–29 GB），前提是先和參考值比對過關（§3）。
- **L40S 不等於 GB10**：
  - 準確率跟硬體無關（同權重、同精度），L40S 量的就是 GB10 的。
  - 延遲不行。GB10 是 128 GB 統一記憶體，頻寬約 L40S 的三分之一，算力也低一截，所以 L40S 的延遲會比 GB10 樂觀。延遲表一律標「L40S，非 GB10」。
- **MoE 影響的是速度**：
  - 讀第一個 token 的機率幾乎都是 prefill，主要吃算力。
  - 26B-A4B 每個 token 只啟用約 4B 參數；Clef 27B 是 dense，計算量約 7 倍；flash 9B 約 2 倍。
  - 所以 Cloudflare edge 上的 38.8 ms 不能直接搬到 GB10。本輪每個 arm 另報一欄**每題計算量**（啟用參數 × 輸入 token 數），用來推估 GB10 上的快慢排序，GB10 上再實測確認（§7）。

## 3. 步驟 1：參考值與等價檢查（smoke）

參考實作是 transformers 加官方 `load_release_model`（BF16）。卡上另外列了 vLLM、SGLang、llama.cpp 和 15 個量化版本，但**判斷頭是自訂的，這些 runtime 不一定真的跑了它**。

- **參考值**：
  - flash：L40S 上 BF16。
  - 27B：H100 上 BF16，只跑 D0 抽出的 50 題，結果存 volume 後關機。
- **等價檢查**：同 50 題和參考值比，每題機率向量最大絕對差 ≤ 0.01，且 argmax 一致 ≥ 49/50，才算等價。要比的組合：
  - flash：vLLM／SGLang（BF16），以及 Q8 與 Q4（GB10 會想用）。
  - 27B：L40S 上的 FP8 或 Q8（transformers 載入或 vLLM／SGLang，哪個先過用哪個），這是主跑能不能放 L40S 的關鍵。
- 不等價的 runtime 只記錄，不拿來報準確率。
- 27B 在 L40S 上沒有任何量化版過關 → 27B 主跑改回 H100 BF16（預算多約 $3，§6）。

## 4. 步驟 2：主跑

| arm | 模型 | 後端 | 硬體 |
|---|---|---|---|
| C1 | Clef-flash（9B） | transformers BF16（參考） | L40S 48 GB |
| C2 | Clef（27B） | FP8 或 Q8（§3 過關的那個） | L40S 48 GB |
| C2ref | Clef（27B） | transformers BF16，只跑 50 題 | H100 80 GB |
| C1s | Clef-flash | 等價檢查過的 vLLM 或 SGLang（量延遲用） | L40S |
| G26 | 我們的 26B-A4B | 準確率用既有結果（llama-server UD-Q4_K_M、SGLang 帶 `<bos>`）；延遲在同一張 L40S 上重量 | L40S |
| G4 | 我們的 E4B | 既有結果 | — |
| W（選配） | `@cf/cloudflare/clef-flash` | Workers AI REST | Cloudflare edge |

資料（每個 arm 都跑同一份）：

- **D0**：10 類，每類 cal 100、test 100，pair_id 分組 seed 0。cal 用來擬溫度和反推門檻，test 報準確率。
- **`data/hard`** 手寫難例、**`data/blind`** 盲寫題。
- **v9**：G 護欄、T 工具守門、E 評分、R 重排序（每段二選一，200 查詢 × 20 段）、v9 blind。
- **JevBench 公開 231 題**：正序加反序，沿用 `bench/jevbench.py` 的抓題與釘版；標示一律寫 self-run。

量測：

- 每 task 的準確率；與 G26 做 McNemar（同題配對）。
- 語言切片：D0 裡 10–20% 的英文與中英夾雜題，跟中文題分開報。這是這輪最想知道的事。
- 校準：
  - raw 的 ECE / Brier 一起報。
  - 用 cal 擬溫度後再報一次。
- 門檻：`bench/thresholds.py` 的 `risk_threshold`，ε = 5% 和 2%，報 test 實際錯誤率與 coverage；三段式（自動／確認／轉人）比例同 v9 C9。
- 順序敏感度：D0 test 與 JevBench 各跑一次選項反序，報 argmax 翻面率。對照組是 v5 量到的急迫度 21%、SPC 19%，以及 v8 JevBench 的 6.5%。
- 延遲：
  - 單題與「同 state 多題」（K=3、K=7，同 v7 P1）兩種，報 p50 / p95。
  - G26（SGLang 帶 `<bos>`）、C1s、C2 都在**同一張 L40S** 上量。硬體、batch、runtime 寫清楚，表頭標「L40S，非 GB10」。
  - 另報每題計算量（啟用參數 × 輸入 token 數）與換算的 GB10 預估排序，明寫是推估。
- 記憶體：每個 arm 的峰值 GPU 記憶體，以及換算到 GB10（128 GB 統一記憶體）能不能和生成模型並存。

**注意 option mass 不適用**：它的頭本來就只在選項上做 softmax，總和恆為 1，v7 的模板健康檢查換不過去。改用「同題反序的機率向量差」當健康檢查，記錄但不設門檻。

## 5. 判定（pre-registered）

**A. runtime**：至少一個最佳化 runtime 或量化版通過 §3 等價檢查，GB10 部署才算可行；都不過的話，結論只能寫「transformers BF16 可用、速度未最佳化」。27B 的主跑數字只有在 L40S 量化版過關時才算數，否則以 H100 BF16 重跑的為準。

**B. Clef-flash 取代 26B（逐 task 判）**：

- 準確率 ≥ G26 − 1 點，且 McNemar p ≥ 0.05；
- 同一張 L40S 上單題 p50 ≤ G26 × 0.5，而且每題計算量不高於 G26（否則 L40S 上快、GB10 上不一定快，記為「待 GB10 確認」）；
- 中文題切片的準確率 ≥ G26 − 2 點。

三條都過的 task 列入「可換 flash」。10 類中 ≥ 8 類可換 → 判斷層預設改 Clef-flash，26B 留給生成。

**C. Clef（27B）比 26B 好**：任一條成立就記為「訓練判斷頭有價值」。

- 三類弱題（急迫度、SPC、UPH）平均 +3 點以上；
- T 工具守門 ≥ 95%，且危險放行（真答案 C 判成 A）≤ 1%；
- R 重排序 nDCG@5 ≥ 0.85；
- JevBench 公開 231 題 hard ≥ 85%（26B 是 77.5%）。

**D. 順序敏感度**：D0 test 的反序翻面率比 G26 低一半以上，就把上線條件裡的「題序要固定」放寬給 Clef。這條只是加分，不影響 B、C。

**E. 不換的條件**：flash 在中文切片比 G26 低 > 5 點，或 ε=5% 時守得住的 task < 8/10，就寫明「英文意圖分類的成績在中文產線題上不成立」。

## 6. 順序、預算、輸出

1. 步驟 0：讀卡、adapter（無 GPU，約半天）。
2. 參考值：H100 下載 27B（約 54 GB）進 volume，BF16 跑 50 題後關機，含下載約 15–20 分鐘，≈ $1。
3. Smoke：L40S 下載 flash（約 18 GB）與 27B 量化版，跑 §3 的等價檢查。
4. 主跑（全在 L40S）：每個模型約 7,000 次讀取（D0 2,000 + hard/blind + v9 約 600 + R 4,000 + JevBench 462 + D0 反序 1,000）。
   - C1 flash：約 45 分鐘，≈ $1.5。
   - C2 27B 量化：約 30–40 分鐘，≈ $1.2。
5. 延遲：G26、C1s、C2 同一張 L40S 各量一次（約 20 分鐘），≈ $0.7。
6. W（選配）：Workers AI 只跑 D0 test 300 題加延遲 100 次；照 Cloudflare 定價頁另計，預期在免費額度內。

**合計 GPU 約 $3–4，上限 $8**。若 27B 量化版等價檢查沒過、主跑改回 H100 BF16，再加約 $3。

輸出：

- `results/15-clef.md`：§0 讀卡紀錄、等價檢查、主跑表、語言切片、門檻、延遲、記憶體。
- `results/v10.json`。
- `decide/clef_adapter.py`、`bench/clef_run.py`、`bench/analyze_v10.py`。
- 更新：
  - `10-landscape.md` 補 Clef 一列（歸在「訓練 LoRA + 決策頭」類，與 Open-Jev 同類）；
  - `13-jevbench.md` 對照表加 self-run 的 Clef 兩列；
  - REPORT §3i、HTML 參考章與 TL;DR、README 九輪表（改十輪）與 `cost.md`。

## 7. GB10 後續（本輪不做，寫給 `handoff-gb10.md`）

本輪結論若選第一或第二句，GB10 上要補三件事：

1. 用本機能跑的 runtime（等價檢查過的量化版）重跑 D0 smoke，確認 GB10 的 ARM 與 CUDA 環境下數字不漂。
2. 用真實產線的校準集（≥ 300 筆，不出機器）重擬溫度與 `risk_threshold`，寫進 lock 檔，用 `verify.py --check-lock` 守。
3. 生成模型與 Clef-flash 同時常駐時，量記憶體與延遲。

## 一句話結論（三選一，跑完填）

**跑完（2026-10-02）**：選第二句，補 27B 的準確率優勢。Clef 27B D0 96.6%（26B 95.5%，SPC 97% 對 86% p=0.007，弱題 +4.3 點）、重排序 nDCG@5 0.879 過門檻、
三題一起判不掉準確率、護欄外部題 98.2%（26B 74.5%）；Clef-flash D0 93.0%、根因站別 −12 點。中文沒有掉。但判斷層讀的是每個位置的 hidden state，
**沒有推論引擎能跑**，量化版（FP8 出 NaN、int8、NF4）全部沒過等價檢查，只能 transformers BF16；本機單題 flash 61 ms（L40S）、27B 87 ms（H100），沒有比 26B 的 63 ms 快，
H100 上新長度首次 ≈ 1 秒。→ 判斷層預設不換；Clef 27B 列為 SPC、重排序、同 state 多題與護欄的候選，GB10 實測後再定。
工具守門 Clef 更差（85%，危險放行 2.7%），照舊用規則。偏離計劃之處見 `results/15-clef.md` §9；成本約 $4.5。


- 「Clef-flash 在 X/10 類與 26B 打平、快 Y 倍、中文不掉：**判斷層換 flash**，26B 只留給生成。」
- 「Clef 準確率接近但速度優勢只在 Cloudflare edge 上成立（本機 runtime 等價檢查沒過／太慢）：**先不換**，等官方 runtime 支援。」
- 「英文意圖分類的成績在中文產線題上不成立（中文切片 −X 點）：**不換**，繼續凍結 Gemma 讀字母。」
