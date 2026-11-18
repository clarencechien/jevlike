# Handoff v24 — 補三個沒試過的：空白輸入扣偏好、isotonic／Dirichlet 校準、「資訊不足」選項（2026-11-18，跑前寫死，未跑）

來源：一段「不重新訓練、只在推論時把決策模型做到接近商用水準」的建議（第 0 步先量；第 1 層：換位平均、contextual calibration、每類一組校準參數、conformal 門檻、加「需要先查或問人」選項）。
逐項對照之前的實驗，大部分已經做過：

| 建議 | 已做 | 結果 |
|---|---|---|
| 順序一致率 | v5、v19 | 弱題沒過 97%：急迫度 21%、SPC 19% 會因順序改答案（v19 旋轉翻轉率弱題 0.110） |
| 各類可靠度圖、ECE | v1/v2 | 十類各一張（`results/fig/reliability-*.png`），本來就是每類一個溫度 |
| 選擇性準確率曲線 | v1、v7→v9 | sel@0.9 與 2%／5%／10% 錯誤預算下的 coverage（`11-thresholds.md`） |
| 換位平均 | v5（隨機置換）、v19（循環旋轉） | 不採用：v5 弱題 −1.3 點；v19 弱題 0.867 → 0.867 |
| 扣掉先天偏好 | v19（**從 cal 旋轉平均估先驗**） | 零改變：弱題 0.867 → 0.867、翻轉率 0.110 → 0.110 |
| 每類一組參數 | v1/v2：溫度、affine（`analyze.py::fit_affine` = 對 logprob 的多項式 LR，L2 C=1，等於沒有 ODIR 的 Dirichlet） | affine 在 6 選項、100 筆 cal 已過擬合（0.98 → 0.95） |
| conformal 門檻＋轉人 | v7→v9 | 採用：5% 預算 10/10 類守住；三段式 11/12 |

**沒做過、這輪要補的三件：**
1. **E1 空白輸入扣偏好**（Zhao et al. 2021 *Calibrate Before Use* 的原版）：餵「N/A」這類沒有內容的狀態，量模型對各位置字母的偏好再除掉。v19 的先驗是用有標註的 cal 估的，不是空白輸入，而且 v19 沒有把 E1 跟每類溫度、5% 預算放在一起看。
2. **E2 isotonic、正規化的 Dirichlet、vector scaling**，以及**要幾筆標註才值得換**的學習曲線。建議說「每類幾百筆就夠」，我們只有每類 100 筆 cal，要量出拐點，給 GB10 真實資料校準時參考。
3. **E3「資訊不足，要先查或問人」選項**（When2Call 那種能力）：同一事件拿掉判斷需要的事實，看模型會不會選這個選項，而不是硬選一個。同時跟「不加選項、靠 conformal 門檻轉人」比，回答「要不要在題目裡加這個選項」。

不在範圍：重新訓練、換模型、改判斷標準。

## 0. 共同設定

- 26B UD-Q4_K_M、llama.cpp **b11371**（`modal_clef_gguf.py`，與 v19／v23 同一個 build）。
- D0 的 cal／test 照舊：`analyze.group_split` 依 pair_id、seed 0，各 100 題。溫度、門檻、所有擬合的參數只在 cal 擬，判定一律在 test（E2 另外報 grouped 5-fold CV）。
- 弱題 = 急迫度、SPC、UPH；強題 = 其餘七類（同 v19）。
- 錯誤預算門檻一律用 `bench/thresholds.py` 的選擇性風險控制（v9 修正版），信心換成各方法校準後的 top 機率。

## 1. E1 空白輸入扣偏好（GPU 很少，L4）

**讀取**：`bench/v24_cf.py`，沿用 `bench/v19.py` 的 `load()` 與旋轉寫法（同一份題目與選項），只把狀態換成三個空白輸入：`N/A`、`（無）`、空字串。
每類 × K 個旋轉 × 3 個輸入：K 合計 41，共 **123 次讀取**。每個旋轉得到 p_cf(位置字母)，三個輸入取平均。

**離線**（`bench/analyze_v24.py`，用 `results/modal/v19/l0/` 已存的 200 題 × K 個旋轉）：

| 讀法 | 做法 | 需要標註 |
|---|---|---|
| raw | 旋轉 0（= v19 raw） | 否 |
| prior19 | v19 的 cal 先驗除法（重算，對照用） | 是 |
| **cf** | 旋轉 0 的機率除以該旋轉的 p_cf，再正規化（Zhao 的 W = diag(1/p_cf)、b = 0） | **否** |
| **cf-L0** | 每個旋轉先除 p_cf，再對回原選項取對數平均（K 倍讀取） | 否 |

每種讀法都再擬每類溫度、定 5% 門檻。量：test 準確率（弱／強平均）、旋轉翻轉率（cf：每個旋轉各除各的 p_cf 後，argmax 跟旋轉 0 不同的比例）、5% 預算 coverage 與實際錯誤率、top-label ECE。

**判定 G1（cf 當預設）**：弱題準確率 ≥ raw + 0.02，**或**弱題 5% coverage ≥ raw + 0.05；強題準確率不掉超過 0.005，且 5% 預算守住的類數不少於 raw。
**G1b（對稱性）**：cf 單次讀取的弱題翻轉率 ≤ 0.055（raw 的一半），且弱題準確率不掉超過 0.005。
**預測**：G1 ✗、G1b ✗。v19 的除先驗零改變，表示翻答案跟題目內容有關，不是固定的字母偏好；空白輸入估出來的偏好只會更粗。弱題準確率變動在 ±0.01 內。

## 2. E2 校準方法與要幾筆標註（零 GPU）

**資料**：v19 旋轉 0 的 logprob（200 題／類）。缺值照 `analyze.logit_matrix` 補底。

| 方法 | 定義 | 會不會改答案 |
|---|---|---|
| temp（基準） | 每類一個溫度（現行） | 不會 |
| vector | 每類 logit 乘對角 W 加 b（K + K 個參數） | 會 |
| affine（現行對照） | `fit_affine`：多項式 LR，L2 C = 1 | 會 |
| dirichlet-odir | 同 affine 的完整矩陣，改用 ODIR 正規化（非對角與截距分開罰，λ 在 cal 內 3-fold 選） | 會 |
| isotonic | 先套 temp，再對 top 機率做 isotonic（top-label） | 不會 |

**評估**：
- 固定 split（cal 100／test 100）。
- **grouped 5-fold CV**（依 pair_id 分組，200 題全用，每折約 160 擬、40 量）。**判定看 CV**，固定 split 對照用。
- 指標：準確率、top-label ECE（10 bins）、classwise-ECE、NLL、Brier、2%／5% 預算的 coverage 與實際錯誤率。
- **學習曲線**：cal 抽 25／50／100／150 筆（grouped，重抽 20 次），每種方法在固定 test 上的 ECE 與 5% coverage。報「isotonic／dirichlet 的 ECE 開始低於 temp 的筆數」。

**判定 G2（換掉每類溫度）**：CV 下某方法的弱題平均 5% coverage ≥ temp + 0.05，且守住預算的類數 ≥ temp，強題 coverage 不掉超過 0.01，十類平均 ECE ≤ temp。
**預測**：100 筆下沒有方法通過。vector ≈ temp；isotonic 的 ECE 略低但 coverage 不穩；dirichlet-odir 比 affine 少過擬合，但 6／10 選項的類別仍不贏 temp。學習曲線的拐點在 150 筆以上，或量不到。

## 3. E3「資訊不足」選項（新資料，GPU）

### 3.1 資料：`data/v24/gen_v24.py`

沿用 `data/v23/gen_v23.py` 的事件家族與 renderer（同 seed 12）。每類取 60 組 twin，每組取一邊，a／b 輪流以平衡 gold。每個事件產生三列：

| 變體 | 做法 | gold |
|---|---|---|
| **C 完整** | v23 原事件 | 原答案 |
| **M 缺關鍵事實** | 拿掉一個判斷需要的事實 | 「資訊不足」 |
| **R 缺無關事實** | 拿掉一個判斷用不到的事實（對照：缺東西就喊不足？） | 原答案 |

各家族拿掉的欄位（產生器裡寫成表，盲審檢查）：

| 家族 | M 拿掉 | R 拿掉 |
|---|---|---|
| SevReflow 數字型 | `limit_hi_c` | `duration_min` |
| SevReflow 事實型 | `board_in_oven`＋`conveyor`（爐內狀態） | `duration_min` |
| SevNg | `inspected` | `upstream` |
| SevStations | `waiting_stations` | （無） |
| SevPaste | `paste_life_h` | `boards_printed_since_open` |
| SevThrow | `control_limit_pct` | （無） |
| CatCode | `code`＋`message` | `machine_state` |
| CatAir | `facility_supply_bar` | （無） |
| CatVolt | `plant_supply_v` | （無） |
| DispRetry | `sop_reset_limit_per_shift` | `running_min_since_last_reset` |
| DispMetric | `sop_pass_min_pct` | （無） |
| DispAdjust | `operator_range_abs` | （無） |
| DispAction | `finding` | `alarm_code` |
| Spc（全部） | `ucl`＋`lcl` | （無） |
| UphHourly／UphAvg／UphCt | `target_uph`／`target_ct_s` | UphHourly：`hours`；其餘（無） |
| ToolParam | `param_spec` | `context` |
| ToolScope | `params.targets` | `context` |
| ToolRead | `caller_lines` | `context` |
| ToolRole | `caller_role` | `context` |

產生器 assert：
- **M 有效**：把拿掉的欄位在該家族抽樣範圍內換 20 個值，gold 至少出現兩種。也就是說，不知道這個值就真的判不了。
- **R 有效**：同樣換 20 個值，gold 不變。
- 拿掉後仍保留 v23 的資訊對等與 `FORBID` 檢查。

**寫法**（每列三種）：
- **T0**：文字原始值，缺的那句不寫。
- **J0**：JSON，缺的 key 不出現。
- **J1n**：JSON＋衍生欄位。算不出的衍生值寫 `null`，模擬「程式算好、缺值標 null」。

**題目兩臂**：
- **Q+**：在原選項後面加一個選項「資訊不足」，判準寫：「判斷這題需要的事實（數值、規格上下限、狀態、角色…）在狀態裡找不到，要先查資料或問人；事實齊全時不要選」。急迫度是 score 型，這個選項加在等級之後，字母讀法不受影響。
- **Q−**：原選項（強迫選一個）。

題目直接寫進每列，**跑的時候不帶 `--criteria`**，避免急迫度與 SPC 被覆寫掉新選項。

**筆數**：
- 每類 C 60、M 60、R 0–60 列，六類約 1,000 列。
- 三種寫法 × 兩臂約 6,000 列，每個模型約 6,000 次讀取。
- **盲審**：同 v23，另一個 agent 每類抽 12 個事件，檢查 M 真的判不了、R 真的不影響、沒有洩漏。不合理率 > 5% 就改產生器重生。結果寫 `data/v24/SPOTCHECK.md`。

### 3.2 跑什麼

- 模型：**26B**（主判）、**E4B**。
- 硬體：L40S，跟 v23 同卡，因為 Q− 的門檻要用 v23 的完整題擬（見下）。
- Clef 不跑：score 型題加選項會破壞它的等級語意。
- `bench/v23_formats.py` 加 `--data-root`（指到 `data/v24/<Q+|Q->`）與 `--criteria none`，其餘不變。

### 3.3 量什麼

- **Q+**：
  - M 選「資訊不足」的比例（recall）。
  - C、R 誤選「資訊不足」的比例。
  - C 的準確率跟 Q− 比（McNemar）。
  - 依家族列出哪一類缺值最常被忽略（數字型規格、狀態布林、角色）。
- **Q−（靠門檻轉人）**：
  - 60 組 pair 依 pair_id、seed 0 分兩半。
  - 用 v23 同模型、同寫法的完整題（T0／J0／J1）中 **cal 半的 pair** 擬溫度與 5% 門檻。
  - 在 v24 **test 半**量：M 被門檻擋下（轉人）的比例，以及 C 仍被自動處理的比例。
- 信心：M 列在 Q− 的 top 機率分布，看「缺事實時模型是不是仍然很有把握」。

### 3.4 判定 G3（26B，主看 J0）

- **G3a 加選項可用**：M recall ≥ 0.80；C 誤選 ≤ 0.05；R 誤選 ≤ 0.10；C 準確率 Q+ 比 Q− 掉不超過 1 點，且 McNemar p ≥ 0.05。
  成立 → 指南加「log 型題目一律附『資訊不足』選項」。
- **G3b 選項比門檻好**：Q+ 的 M recall ≥ Q− 門檻轉人比例 + 0.20。
- **G3c 程式標缺值就夠**：J1n 的 M recall ≥ 0.95。
  成立 → 指南寫「缺值由程式檢查、直接轉人；模型的『資訊不足』選項只是第二道」。

**預測**：
- Q− 門檻轉人比例 < 0.30：v23 看到錯的題 86% 機率 ≥ 0.95，缺事實時大概也一樣有把握。
- Q+ 的 M recall：J0 0.5–0.7、T0 相近、J1n ≥ 0.9。缺規格數字比缺狀態布林容易被發現。
- C 誤選：26B ≤ 0.05、E4B 較高。R 誤選 0.05–0.15。
- 綜合：G3a ✗（J0 recall 不到 0.8）、G3b ✓、G3c ✓。

## 4. 順序、預算、輸出

1. E2（零 GPU，可以先做）。
2. E1：L4 上 123 次讀取，約 5 分鐘含載入；L4 排不到就改 L40S，寫進偏離。
3. E3：產生器＋盲審（無 GPU）→ smoke（每類 5 列 × 兩臂 × J0）→ 26B、E4B 各約 6,000 次讀取（L40S，26B 約 15 分鐘、E4B 約 8 分鐘）。

**合計 GPU 約 $1.5，上限 $3。**

輸出：
- `results/28-calibration-abstain.md`：表、判定、偏離計劃之處。
- `results/28-calibration-abstain-tables.md`、`results/v24.json`。
- `data/v24/`：產生器、資料、MANIFEST、SPOTCHECK。
- `bench/v24_cf.py`、`bench/analyze_v24.py`。
- REPORT §3s、README 輪次表一列、`cost.md`、HTML 一列。
- 依判定更新 `docs/input-guide.md`（缺值怎麼處理）與 `docs/handoff-gb10.md`（真實資料校準要用幾筆、用哪個方法）。

## 5. GB10 後續（本輪不做）

- E2 的學習曲線決定 GB10 真實資料每類要標幾筆，以及用 temp 還是換方法。
- E3 成立的話，在真實 log 上抽 50 筆、手動拿掉關鍵欄位重驗一次。真實資料不出 GB10。

## 一句話結論（三選一，跑完填）

- 「三件都不用換：空白輸入扣偏好與 isotonic／Dirichlet 在 100 筆下都不贏每類溫度（拐點約 X 筆）；『資訊不足』交給程式檢查缺值，模型選項只當第二道（recall X）。」
- 「『資訊不足』選項值得加：J0 recall X、誤選 Y，比靠門檻轉人多抓 Z；校準維持每類溫度。」
- 「有一件改變預設：＿＿（數字），其餘維持。」
