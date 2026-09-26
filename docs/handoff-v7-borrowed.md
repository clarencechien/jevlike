# Handoff v7 — 從生態調查抄三件事：packed readout、option mass、conformal 門檻（2026-09-26，跑前寫死）

來源：`results/10-landscape.md` §5 的前三項。三件事只有 P1 要 GPU（SGLang L40S 一次，約 $0.5）；P2、P3 全部用既有的原始 logprobs 離線算。

## P1 packed readout：K 題一個前向（抄 open-alternative-jev）

**現況**：共用 state 問 K 題，llama-server 走同 slot 順序送 K 次（v4 K=16 813 ms），SGLang 暖一題再整批（217 ms）。每題都是一個請求、一次 decode，SGLang 每請求固定開銷 62 ms。

**做法**：state 一次，後面接 K 個「問題回合 + 占位答案回合」，一次前向，讀每個占位符位置的字母 logprob。Gemma 4 版型：

```
<bos><|turn>system\n{sys}<turn|>\n<|turn>user\n【狀態】\n{state}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>_<turn|>\n
<|turn>user\n{q1}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>_<turn|>\n
<|turn>user\n{q2}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>_<turn|>\n ...
```

占位符 `_`（單 token，不是任何字母）；讀 `_` 這個位置的預測分布（即 `<channel|>` 之後的下一個 token）。後端：**SGLang** `/generate` 帶 `input_ids`、`return_logprob`、`logprob_start_len=0`、`token_ids_logprob=[字母 id]`，回傳 `input_token_ids_logprobs` 每個位置都有字母的 logprob，取占位符索引。llama-server `/completion` 拿不到 prompt 位置的 logprob，**P1 只在 SGLang 上做**；llama-server 的順序送已經是它能做到的最好（省的只有 K−1 次 decode 地板 13 ms）。

**資料**：需要「同一 state 多題」而且有 gold。用 alarm 家族：`m_alarm_category`、`m_alarm_severity`、`m_needs_dispatch` 三個 task 的 state 都是 alarm 文字，每個 state 打包這三題（K=3），gold 只有原 task 那一題。600 個 state × 3 題；另外從 v4bench S2 的 120 個會議視窗 × 7 題只量速度（無 gold）。

**Arm**：separate（現行：暖 state 再整批，K 個請求）、packed（1 個請求）。同權重（FP8）、同 `<bos>`、同題序。

**判定（pre-registered，D0 test 300 題有 gold）**：
- 準確率：packed 與 separate 每 task 差 ±1 點內且 McNemar p ≥ 0.05（3/3 task）。
- 干擾率：同一（state, q）packed 與 separate 的 argmax 不同的比例 ≤ 5%（open-alternative-jev 量到 6–9%，我們的門檻比他們嚴，因為我們要拿它做 gating）。
- 速度：K=3 端到端 p50 packed ≤ separate × 0.6；K=7（S2）≤ × 0.4。
- 三項全過 → **packed 進 SGLang 批次路徑的預設**；準確率過、干擾率 5–10% → 只給離線批次，即時不用；準確率沒過 → 不採用，記錄干擾例。
- 額外記錄：占位符換成 `A` 或空字串時準確率是否變（占位符本身的偏誤），各 100 題。

## P2 option mass：字母正規化前拿到的總機率（抄 verdict）

**做法**：`read_option_probs` 與 SGLang 讀法回傳新欄位 `option_mass = Σ_letters exp(logprob)`（top-k 內；SGLang 指定 token 時是精確值）。`accuracy.py` 每列存；`analyze*.py` 每 task 報 p50 / p5 / <0.5 的比例；smoke 加一條：5 題 option mass 最低值 ≥ 0.9，否則 fail。

**回溯驗證（跑前已用既有資料做，寫在這裡當已知）**：
- `<bos>` 有無：`sglang/D0`（無）與 `sglang/D0-ids`（有）的 option mass 都是 1.000（p5 也是 1.000）。**option mass 抓不到 `<bos>` 這類錯**：模型仍把全部機率放在字母上，只是選錯字母。原本以為它能抓到 v6 的問題，不成立，先講清楚。
- v2 smoke 的五個模板變體：採用的三個變體 mass 0.998–1.000；「prefill 答案：加尾空白」mass 中位數 0.020（first token 是換行）、「think 開」mass 0.003（first token 是「答案」）。**模板等級的錯它抓得到**，而且是 0 與 1 的差，門檻好設。
- 所以 P2 的定位是：smoke 與每次上線前的「模板健康檢查」（門檻 0.9），不是準確率訊號。仍要報 llama-server D0 上 mass 與答對率的 AUROC，預期接近 0.5。

## P3 conformal 門檻 + lock 檔 + CI 重測（抄 poorjev、jevcal）

**現況**：Q6 報的是「sel@0.9 / 0.95 的準確率與 coverage」，是描述。長官問的形狀是「我允許 5% 錯，多少比例能自動處理」。

**做法**（`bench/thresholds.py`，離線）：
1. 每 task 用既有 cal/test split（pair_id 分組 seed 0）。信心量測三選一：top_prob、margin、entropy，先用溫度校準後的 top_prob，`--measure auto` 時挑 cal 上 AUROC 最高的。
2. Split conformal：在 cal 上算 nonconformity s = 1 − conf，門檻 q = 第 ⌈(n+1)(1−ε)⌉/n 分位；test 上 conf ≥ 1−q 才自動處理。ε 取 2%、5%、10%。
3. 輸出 `results/thresholds.lock.json`：每 task × ε 的門檻、n_cal、預期 coverage、模型檔名與 commit；`results/11-thresholds.md` 報 test 上的實際錯誤率、coverage、要升級的比例，並與 Q6 的 sel@0.9 對照。
4. `bench/verify.py --check-lock <results dir>`：用 lock 重算，實際錯誤率 > ε × 1.5 或 coverage 掉 > 10 個百分點就 exit 1。這是之後 GB10 換模型檔、換後端時的 CI。

**判定（pre-registered）**：
- 保證是否成立：ε=5% 時 test 實際錯誤率 ≤ 5% 的 task 數 ≥ 8/10（100 筆的抽樣誤差允許兩題超）；ε=2% 時 ≥ 6/10。
- 有用與否：ε=5% 的 coverage，七類強題 ≥ 0.95、三類弱題報數字不設門檻。
- 過 → Q6 改寫成 conformal 形式，HTML「條件二」改成「給錯誤預算，得自動處理比例」；沒過 → 保留 sel@0.9，寫明 100 筆校準不足以給保證，真實資料要 300 筆以上。

## 順序、預算、輸出

P2 → P3 → P1（前兩個不用 GPU，先做；P1 的 smoke 用 P2 的 option mass 當健康檢查）。預算：L40S 約 20 分鐘、< $1。
輸出：`results/11-thresholds.md`、`thresholds.lock.json`、`12-packed.md`、`bench/thresholds.py`、`bench/packed.py`、`bench/analyze_v7.py`；REPORT.md §3f；HTML 只在 P1 或 P3 過時改一段。

## 一句話結論（三選一，跑完填）

- 「三件都成立：K 題一個前向快 X 倍、option mass 當模板健康檢查、給 5% 錯誤預算能自動處理 Y%，**全部納入**。」
- 「門檻與 option mass 納入，packed 只給離線批次（干擾 X%）。」
- 「只納入 option mass 與 lock 檔；conformal 在 100 筆上給不了保證，packed 干擾太高。」
