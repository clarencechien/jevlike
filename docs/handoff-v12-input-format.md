# Handoff v12 — 輸入要怎麼準備：JSON 還是文字？要不要先算好？（2026-11-05，跑前寫死；已跑，結論見文末與 `results/17-input-format.md`）

來源：使用者的問題「jevlike 的 input 應該是結構還是非結構？直接拿 JSON log 會比自然語言的文字報告好嗎？」
到 v11 為止所有 state 都是一兩句中文描述，而且關鍵比較已經寫好（「超上限 9°C」）。上線時手上的往往是機台的 JSON log。
v9 已經看到相關的弱點：工具守門的 state 帶 JSON 參數，錯的 14 題全是「參數超規格」與「影響全產線」；UPH、JevBench 時間與數字題也是同一類。

本輪要回答兩件事，並把答案寫成一份**輸入準備指南**（`docs/input-guide.md`），讓上線的人照著做：
1. **格式**：同樣的資訊，寫成 JSON 或寫成文字，準確率有沒有差？
2. **前處理**：數字比較、代碼翻譯先用程式做完（衍生欄位），跟丟原始值讓模型自己比，差多少？加上雜訊欄位（ID、時間戳、韌體版本）會不會變差？

不在範圍：事後人寫的文字報告（裡面常已含判斷，會高估準確率，另案處理）；圖片；真實資料（GB10 上再做，見 §7）。

## 0. 設計：一個事件、五種寫法（2 × 2 + 雜訊）

每一筆資料先產生一個**事件**（dict），gold 由事件與規則決定；再用五個 renderer 寫成 state。同一個事件的五種寫法 gold 相同，可以逐題配對比較。

| 代號 | 格式 | 數字比較／代碼意義 | 雜訊欄位 | 回答的問題 |
|---|---|---|---|---|
| **T0** | 中文文字 | 原始值＋規格（「第 7 區 254°C，上限 245°C」），不寫比較結果 | 無 | 文字、未前處理 |
| **T1** | 中文文字 | 加衍生（「超上限 9°C，已持續 3 分鐘」、代碼翻成意思） | 無 | 文字、已前處理（≈ 現行 D0 寫法） |
| **J0** | JSON（compact） | 原始值＋規格欄位，不給比較結果；代碼只給代碼與機台原文訊息 | 無 | JSON、未前處理 |
| **J1** | JSON | J0 再加衍生欄位（`over_limit_c: 9`, `over_limit: true`, `code_meaning: "真空不足"`） | 無 | JSON、已前處理 |
| **JN** | JSON | 同 J0，再加 6–10 個無關欄位（event_id、uuid、時間戳、韌體版本、操作員工號、PLC 位址…） | 有 | 原始 log 直接丟 |

**資訊對等原則**（產生器要 assert）：
- 五種寫法都包含判斷所需的全部**事實**（量測值、規格上下限、狀態、持續時間、代碼）。少了事實就是題目不能答，不是格式的問題。
- 「衍生」只能是事實的**確定性計算**（差值、是否超限、比例、持續分鐘數、代碼查表），不能是判斷（不能寫「應停線」「嚴重」這種跟答案同義的字）。
- T0 與 J0 資訊完全相同，只差格式；T1 與 J1 也是。這樣 T0 vs J0 量的是格式，T0 vs T1 量的是前處理。

### 範例（急迫度，一個事件）

```
事件：line=SMT-L4, device=R-03, zone=7, temp=254, limit_hi=245, duration_min=3, board_in_oven=true, conveyor=running
gold：C 停線（超限且板子仍在爐內）

T0：SMT-L4 回焊爐 R-03 第 7 區溫度 254°C，上限 245°C，已 3 分鐘；爐內有板，輸送帶運轉中
T1：SMT-L4 回焊爐 R-03 第 7 區溫度超上限 9°C，持續 3 分鐘；爐內有板，輸送帶運轉中
J0：{"line":"SMT-L4","device":"R-03","zone":7,"temp_c":254,"limit_hi_c":245,"duration_min":3,"board_in_oven":true,"conveyor":"running"}
J1：J0 ＋ "over_limit":true,"over_by_c":9
JN：J0 ＋ "event_id":"EV-20261105-004812","uuid":"…","ts":"2026-11-05T03:12:44.218Z","fw":"RX-4.2.17","plc_addr":"D1204","op_id":"A10233"
```
twin 的另一半：只改一個事實讓 gold 換（例如 `board_in_oven=false, conveyor=stopped` → B 盡快；或 `temp=247` 在警戒內 → A）。
**數字型 twin**（只改數值、讓它從規格內跨到規格外）至少佔一半，因為這正是要測的弱點。

## 1. 題目（6 類，各 120 個事件 = 60 組 twin）

挑「上線時輸入本來就是 log／結構資料」的題型；`x_*`（人寫的訊息、工單）不在範圍。題目定義與判斷標準沿用現行，急迫度與 SPC 用 `data/seeds/tasks_v2.json`。

| task | 事件欄位（事實） | 衍生欄位 | 數字比較在哪 |
|---|---|---|---|
| `m_alarm_severity` | 溫度／壓力／NG 率等量測值、上下限、持續時間、板子在不在、輸送帶狀態、影響站數 | 超限與否、超多少、持續分鐘 | 量測值 vs 上限、NG 率 vs 30% |
| `m_alarm_category` | alarm 代碼、機台原文訊息（英文，如 `VACUUM LOW HEAD 3`）、子系統 | 代碼的中文意思 | （以代碼語意為主，數字少） |
| `m_needs_dispatch` | 事件、操作員已做的動作序列、是否已復歸、重試次數 | 是否已依 SOP 復歸、重試超過上限與否 | 重試次數 vs 上限 |
| `q_spc_action` | 最近 25 點量測值、CL／UCL／LCL | 超出管制界限的點數、最長同側連續點數、最長單向連續點數 | **規則全靠數列計算**（最難的一類） |
| `p_uph_anomaly` | 目標 UPH、最近 4 小時每小時產出、CT | 相對目標的百分比、是否有突然下滑 | 實際 vs 目標的比例 |
| `t_tool_gate` | 呼叫者角色與負責線別、工具名、參數（JSON）、參數允許範圍、影響範圍 | 參數是否超出允許範圍、影響線數、是否寫入 | 參數 vs 範圍（v9 錯 7/7 的那類） |

每類另外標 `needs_compare`（gold 是否取決於數字比較）與 `hard_type`，分析時切片。

## 2. 資料怎麼做

1. Claude Code 寫 `data/v12/gen_v12.py`：每類一個事件產生器（twin pair）＋五個 renderer＋資訊對等 assert。
   輸出 `data/v12/<T0|T1|J0|J1|JN>/<task>.jsonl`，欄位同 `data/synthetic`（`id, task, state, gold, question, difficulty, lang, pair_id`），
   同一事件五種寫法的 `id` 相同（只有目錄不同），另加 `needs_compare`、`event`（原始事件 dict，分析用）。
2. **抽查**：另一個 agent（不是寫產生器的那個）盲抽 10%（每類 12 個事件 × 五種寫法），檢查三件事：
   (a) 五種寫法事實一致；(b) 衍生欄位沒有洩漏答案（沒有「停線」「嚴重」這類字）；(c) gold 依判斷標準合理。不合理率 > 5% 就改產生器重生。結果寫 `data/v12/SPOTCHECK.md`。
3. 記錄每種寫法的平均 token 數（用 26B 的 tokenizer，`/tokenize`）。
4. 只用合成資料；`data/v12/MANIFEST.md` 記產生器 seed 與筆數。

## 3. 跑什麼

| arm | 模型 | 後端 | 硬體 |
|---|---|---|---|
| G26 | 26B UD-Q4_K_M（現行讀字母） | llama-server **b11371**（`modal_clef_gguf.py::run_l4`，v11 已驗證與 b11118 數字一致） | L4 |
| G4 | E4B Q8_0（小模型可能更需要前處理） | 同上 | L4 |
| C27 | Clef 27B Q8_0 | llama-server `/v1/systemone`（`systemone_bench.py --suites rows`） | L40S |

```bash
# 26B / E4B：每種寫法一次（--data-dir 指到該寫法目錄）
scripts/modal.sh run modal_clef_gguf.py::run_l4 --name 26b --which accuracy \
  --args "--data-dir /root/data/v12/J0 --tasks m_alarm_severity,m_alarm_category,m_needs_dispatch,q_spc_action,p_uph_anomaly,t_tool_gate --criteria /root/data/seeds/tasks_v2.json --out-sub v12/J0 --control-n 0"
# Clef 27B
scripts/modal.sh run modal_clef_gguf.py::run_l40s --name clef-q8 \
  --args "--backend systemone --suites rows --rows-dir /root/data/v12/J0 --tasks ... --criteria /root/data/seeds/tasks_v2.json --out-sub v12/J0"
```
（E4B 需要在 `modal_clef_gguf.py` 的 `_run` 加一個 `e4b` 名稱，模型檔已在 volume。）
`t_tool_gate` 的題目定義在 `data/v9/tasks_v9.json`，不在 `tasks_v2.json`；產生器直接把 question 寫進列裡即可。

## 4. 量測

- 每 task × 每寫法 × 每模型：準確率（**全部 120 題**，不分 cal/test，因為這輪比的是寫法不是門檻）、McNemar（同事件配對）、ECE（raw）、平均 token 數、單題 p50。
- 切片：`needs_compare = true` 的題；`t_tool_gate` 的「參數超規格」「全產線範圍」兩個 hard_type（v9 的錯誤來源）。
- 錯誤分析：每類各抽 10 個「T1 對、J0 錯」的事件，寫出模型錯在哪（比錯數字？代碼不認得？被雜訊帶偏？）。

## 5. 判定（pre-registered）

以 26B 為主判（現行上線模型），E4B 與 Clef 報數字並比較方向是否一致。

- **H1 格式本身不重要**：T1 與 J1 每類差 ≤ 2 點，6 類平均差 ≤ 1 點；T0 與 J0 同樣。成立 → 指南寫「JSON 或文字都可以，選維護方便的」。
- **H2 前處理很重要**：`needs_compare` 切片上，J1 比 J0 高 ≥ 5 點（或 T1 比 T0 高 ≥ 5 點），McNemar p < 0.05。
  成立 → 指南第一條寫「數字比較與代碼翻譯一律先用程式算好再給模型」。
- **H3 雜訊有代價**：JN 比 J0 低 ≥ 2 點，或 token 多 ≥ 40%、單題延遲多 ≥ 20%。成立 → 指南寫「只送判斷需要的欄位」。
- **H4 工具守門可以救**：`t_tool_gate` 的「參數超規格」「全產線」兩類，J1 比 J0 高 ≥ 30 點（v9 這兩類 0/7、0/7）。
  成立 → v9「工具守門交給規則」的結論改寫成「衍生欄位算好後交給模型」，並重算工具守門整體準確率是否 ≥ 95%。
- **H5 SPC 數列**：`q_spc_action` 的 J0（只給 25 個點與管制線）預期接近亂猜；J1（給計算好的規則統計）預期回到 ≥ 90%。
  兩者差 ≥ 20 點 → 指南寫「SPC 規則判定用程式算，模型只判處置」。

三個模型方向不一致時（例如 Clef 不需要前處理），照實寫，指南以 26B 為準、另註 Clef 的差異。

## 6. 順序、預算、輸出

1. 產生器＋抽查（無 GPU，約半天）。
2. L4：26B × 5 寫法（720 × 5 = 3,600 次讀取，約 15 分鐘）、E4B 同（約 10 分鐘）。L40S：Clef 27B Q8 × 5 寫法（約 15 分鐘）。
3. 分析：`bench/analyze_v12.py`。

**合計 GPU 約 $1.5，上限 $3。**

輸出：
- `results/17-input-format.md`：表、判定、錯誤分析、偏離計劃之處。
- **`docs/input-guide.md`**：給上線的人的輸入準備指南（依判定寫成可照做的規則：哪些要先算、哪些欄位該留、JSON 或文字、範例各一），README 的「給 GB10 上的 Claude Code」段落加一行指向它。
- `data/v12/`（產生器、五種寫法、MANIFEST、SPOTCHECK）、`results/v12.json`、REPORT §3k（若 GB10-Clef 先佔用 §3k 就順延）、HTML 一列、README 輪次表一列、`cost.md`。

## 7. GB10 後續（本輪不做）

真實 log 上要重驗一次：從 `data/real/` 挑 50–100 筆有原始 log 的事件，用指南的前處理寫成 J1，跟原始 JN 各跑一次 26B（`bench/run_local.py --which accuracy`），差距寫進 `handoff-gb10.md` 的結案報告。真實資料不出 GB10。

## 一句話結論（三選一，跑完填）

- 「格式不重要、前處理很重要：JSON 或文字都可以，但數字比較與代碼翻譯先算好，準確率差 X 點；工具守門因此從 Y% 回到 Z%。」
- 「格式與前處理都不重要（差 < 2 點）：模型自己會比數字，原始 log 可以直接丟，只要拿掉雜訊欄位。」
- 「JSON 本身比文字差（或好）X 點：上線用 Y 格式；前處理另有 Z 點。」

**結論（2026-11-05 跑完）**：選第一句，附但書——
「格式不重要、前處理很重要：JSON 或文字六類平均差 < 2 點（但單類可差 5–13 點、方向不一，三個模型最好的都是中文句子＋衍生值）；
要比數字的題先算好 26B +8～+15 點（E4B +22、Clef +9），雜訊欄位 −3.6 點；工具守門參數超規格 80% → 100%，但跨線範圍仍 50%，整體 86.7%，未到 95%。」
H1 ✗、H2 ✓、H3 ✓、H4 ✗（超規格救回、跨線沒救）、H5 26B JSON ✗（欄位名被誤讀）／文字與 E4B、Clef ✓。偏離計劃：改在 L40S 跑（L4 排不到）、T1 保留原始值。
上線規則寫在 `docs/input-guide.md`。

