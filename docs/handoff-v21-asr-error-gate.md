# Handoff v21 — Breeze 聽、E4B 抓錯：ASR 之後的「這段有沒有錯」閘門（2026-10-15，跑前寫死）

起因：使用者問「ASR 前能不能先用分類器把純中文跟晶晶體分開」。讀了 kikemu（clarencechien/kikemu @ `1a9cea4`）的 exp5 之後題目改了形狀：
Breeze ASR 25 會保留拉丁字母，所以「純中文對晶晶體」在它的輸出上是一條正規式；它在晶晶體上真正的失誤是**英文認錯字**（core→cool、Bing→bin、compare to→comparative），
把英文寫成漢字（F1）整份語料只有 T1 的 2–4 處。中文錯字（鄰里→林地、搜尋結果→攝影節目）在吵雜條件下更多。
所以這輪要量的是：**Breeze 單跑之後，E4B 用讀字母能不能標出「這個視窗有錯」**，以及免費訊號（Breeze 自己的 logprob）夠不夠、音訊有沒有加分、26B 純文字的上限在哪。
抓到的視窗轉人或轉大模型，**本輪不做自動改錯**。

oracle：**exp5 的正解照用**（兩路 Gemini 合併 + 16 處拉丁字母分歧的人工裁決；kikemu `pairing_report.md` §7a 標了它是 Gemini 產的，本輪使用者決定直接當 oracle，寫進限制）。
Scribe × Gemini 分歧裁決那條（36 處、基準永遠選 Scribe 0.58）**不在本輪**，見 §9 選配。

## 0. 規則

- 不訓練、不改 `decide/prompt.py` 模板本體。E4B Q8_0、26B UD-Q4_K_M、llama.cpp ≥ b11371、`-c 16384 -np 4 --swa-full --reasoning-budget 0`、判斷 server 不開 MTP（v18）。
- Breeze：`MediaTek-Research/Breeze-ASR-25` revision `cffe7ccb404d025296a00758d0a33468bec3a9d0`，fp16，`chunk_length_s=0`、`return_timestamps=True`、不指定 language——與 kikemu `Xbrz_gpu` 完全同設定。
- 音訊照 kikemu manifest 重建，sha256 12/12 相符才能跑；音檔不進 repo。
- 正解、術語表、tokenizer（含 OpenCC s2t）、召回判定全部沿用 kikemu `exp5/scripts/score.py`，不另發明。
- **GB10 優先**（語料是公開 podcast，不是機密，Modal L4 可當開發備援；正式數字以 GB10 為準，兩邊都跑就並列）。
- 標籤由機械對齊產生（§2），在任何偵測器跑之前先產生並凍結（`data/v21/windows.jsonl` 的 sha256 寫進 §6）。
- 校準資料只用來擬溫度、定門檻；所有判定在 test 折。預測錯了照實寫。

## 1. 語料（`data/v21/kikemu/`，MANIFEST 有 sha256）

exp5：3 段台灣科技 podcast（T1 賽事轉播技術、T2 區塊鏈錢包、T3 開發者社群）× 4 個聲學條件（M0 原音、M1 講堂殘響、M2 + 辦公室噪音 20 dB、M3 + 會議人聲 12 dB），每檔 300 秒，共 1 小時。
正解每段約 1,500 token，英文術語 83 型、每條件 158 實例。

Breeze（`raw/Xbrz_gpu`）既有輸出的錯誤普查（本輪寫前算的，`bench/v21_labels.py --census` 重現）：1,423 個 Breeze 小段（中位數 2.2 秒），乾淨 499、英文術語出錯 139、只有中文替換 387、只有中文增刪 398；
召回 tolerant 0.78–0.93、TER 0.11–0.24。**這就是為什麼判斷單位要合併成視窗**：2 秒的小段沒有上下文。

## 2. 判斷單位與標籤（Phase 1，純計算）

**視窗**：把 Breeze 小段依時間順序合併，直到 ≥ 8 秒或 ≥ 25 token，上限 15 秒（單一小段超過 15 秒就自成視窗）。預估 12 檔約 350 個視窗。
視窗以**被評的那次 Breeze 跑**為準（GB10 跑過等價檢查就用 GB10 的；Modal 備援用 kikemu 的 `Xbrz_gpu`）。

**標籤**（整段逐字稿對正解做 token 對齊 `difflib.SequenceMatcher(autojunk=False)`，opcode 落到視窗；tokenizer 同 score.py）：

| 標籤 | 定義 |
|---|---|
| **E 英文錯** | 視窗內任一非 equal 的 opcode，其正解側含拉丁 token（= score.py 的 F1 音譯／F2 認錯／F4 漏掉） |
| **C 中文錯字** | 視窗內任一 `replace`，正解側 ≥ 2 個連續 CJK token 或假設側 ≥ 2 個 CJK token；**單 token 增刪不算**（語助詞，正解自己不穩） |
| C-strong | 同上但 ≥ 3 token（次要，雜訊更低） |
| **N 乾淨** | 非 E 非 C |
| has_latin | 視窗假設側有拉丁 token（D0，晶晶體旗標） |

多標籤；Q3 四選一由 (E, C) 推出：A 乾淨、B 只有英文錯、C 只有中文錯、D 兩者都有。

**雜訊地板**：把 `verify/*.gm36.txt`（第二路 Gemini）當假設，對正解做同一套標籤，報「被標成 C 的視窗比例」。這是 C 標籤的雜訊上限；若超過 20%，G2 只報不判。

**切分**：留一段（T1／T2／T3）當 test、另兩段四條件當校準，三折輪流；test 預測合併後算 AUROC，另依條件拆。cluster 只有 3 個，CI 必然寬，照實報。

## 3. 偵測器

| 代號 | 做法 | 回傳 | 要什麼 |
|---|---|---|---|
| **D0** | 正規式：視窗有沒有拉丁字母 | 旗標 | 無 |
| **D1** | Breeze 自己的信心：改用 `model.generate(return_dict_in_generate=True, output_scores=True)` 取每 token logprob；視窗特徵 = 平均 logprob（主）、最低 logprob、logprob < −1 的比例（次） | 分數 | 重跑 Breeze（Phase 0） |
| **D2** | E4B 純文字，jevlike 讀字母：state = 「台灣科技 podcast 的 ASR 逐字稿，中英夾雜」+ 前一視窗（標「上一段」）+ 目前視窗（標「這一段」）；Q1「這一段有沒有英文字被聽錯或寫錯（例如 core 寫成 cool、Bing 寫成 bin）」A 沒有／B 有；Q2「這一段有沒有中文字聽錯、不成詞」A 沒有／B 有；Q3 四選一 | 機率 | E4B server |
| **D2′** | D2 的 Q1／Q2 選項對調（B／A）、Q3 四個旋轉；報翻轉率與平均 \|Δp\|（v19 的旋轉程式） | 機率 | 同上 |
| **D3** | E4B 聽音訊 + 文字：同 D2 的題，`/completion` 的 prompt 改成 `{"prompt_string": "...<__media__>", "multimodal_data": ["<視窗 wav 16 kHz base64>"]}`、`n_probs 40`；音訊放在文字**之後**（模型卡 §4）；server 加 `--mmproj` BF16 | 機率 | E4B server + mmproj；**Phase 4 smoke 過了才全跑** |
| **D4** | 26B 純文字，同 D2 的題 | 機率 | 26B server |
| **D5** | 雙 ASR 分歧：同視窗 Breeze 對 E4B-ASR（`raw/Xgma_e4b_greedy`，只有 M0／M3）的正規化編輯距離 | 分數 | 無（6 檔） |

校準：D2／D3／D4 各題在校準折擬一個溫度（`analyze.fit_temperature`），錯誤預算 ε = 0.10 的選擇性門檻（`thresholds.risk_threshold`）；test 折報可靠度表（5 桶：說 0.9 時真的約 90% 對嗎）。

## 4. 速度（GB10；Modal 備援另列）

| 量什麼 | 怎麼量 |
|---|---|
| Breeze 每 300 秒檔的秒數、載入秒數 | Phase 0 |
| E4B 每視窗 D2 三題（含前一視窗的 state）、D2′ | 第一輪冷跑的 p50／p95（視窗互不相同，沒有快取重播問題；D2′ 與 D2 共享 state 前綴會命中快取，分開記） |
| E4B 每視窗 D3（音訊約 25 token／秒，10 秒視窗 ≈ 250 個音訊 token） | 同上 |
| 26B 每視窗 D4 | 同上 |
| 閘門總時間 ÷ Breeze 時間 | 12 檔合計 |

## 5. 預測（寫死）

- D0：含拉丁字母的視窗 60–75%（T1 密度 11.6 詞／分鐘）；純中文視窗是少數。
- D1 平均 logprob：E 的 AUROC 0.65–0.72，C 0.70–0.78；M3 比 M0 好抓。
- D2：E 0.85、C 0.72、Q3 四選一 test 準確率 0.60；D2′ 翻轉率 < 5%、平均 |Δp| < 0.05。
- D3：對 D2 的 E 只高 ≤ 0.03（不到 G4）；而且 **一半機率 smoke 就過不了**（多模態 `/completion` 讀不到字母機率）。
- D4 26B：E 0.88、C 0.76。語言先驗的上限比 E4B 高 3–4 點。
- D5：E 的 AUROC 0.75（M0／M3）。
- 速度：GB10 上 Breeze 300 秒檔 ≤ 60 秒；E4B D2 每視窗 80–120 ms（三題合計 ≤ 300 ms）；D3 每視窗 300–600 ms；26B 150 ms。閘門（D2 三題）加在 Breeze 後面 ≤ 10%。

## 6. 門檻（跑前寫死）

| 門檻 | 條件 | 過了代表 |
|---|---|---|
| **G0 同一把尺** | GB10 的 Breeze 對 kikemu `Xbrz_gpu`：12 檔術語召回差 ≤ 0.02、TER 差 ≤ 0.01 | GB10 數字可與 kikemu 並列；不過就另記 arm `Xbrz_gb10`，差異寫出來，仍往下跑 |
| **G1 英文錯** | D2 Q1 test AUROC ≥ 0.85，且抓到 80% 的 E 視窗時精確率 ≥ 0.60 | E4B 文字閘門可上線 |
| **G2 中文錯字** | D2 Q2 test AUROC ≥ 0.75（雜訊地板 ≤ 20% 才判） | 錯字也能抓 |
| **G3 免費訊號** | D1 主特徵 AUROC ≥ 0.75 → 先用它當第一道；D2 要比 D1 高 ≥ 0.05 才值得加 E4B | 否則 E4B 不加 |
| **G4 音訊** | D3 比 D2 高 ≥ 0.05（E 與 C 各算） | 否則文字版就好 |
| **G5 速度** | GB10 上 D2 三題每視窗 ≤ 300 ms，閘門總時間 ≤ Breeze 的 10% | 閘門不拖慢產線 |
| **Kill** | D2 與 D4 的 E AUROC 都 < 0.70 | 讀字母抓不到 ASR 錯，這條線收掉 |

## 7. 階段

**Phase 0 環境與 Breeze（GB10，半天）**
1. `git clone https://github.com/clarencechien/kikemu` 到 repo 外、checkout `1a9cea4`，唯讀。
2. Python：`torch`（aarch64 CUDA 版）、`transformers>=5.15`、`librosa soundfile scipy numpy imageio-ffmpeg requests opencc-python-reimplemented`。
3. 重建音訊：`python3 exp5/scripts/prep_audio.py && python3 exp5/scripts/degrade.py`（噪音素材自動下載，MIT IR 12 MB + DEMAND 172 MB），`bench/v21_breeze.py --check-audio` 比對 12 個 sha256。
4. `python3 bench/v21_breeze.py --audio <kikemu>/exp5/corpus/conditions --out results/gb10/v21/breeze`：同 kikemu runner 的設定，多存每 token logprob（D1）；輸出沿用 kikemu 的 JSON 形狀（`transcript`、`segments`、`meta`）+ `token_logprobs`。
5. `python3 bench/v21_labels.py --breeze results/gb10/v21/breeze --score` → G0。

**Phase 1 標籤（純計算，10 分鐘）**：`bench/v21_labels.py` → `data/v21/windows.jsonl`、雜訊地板、D0、D5。sha256 寫進 §8 後才開 server。

**Phase 2 文字判斷（GB10，1 小時）**：E4B 與 26B 各起一個 llama-server（v20 的旗標）；`bench/v21_judge.py --engine e4b|26b --mode text|swap` → `results/gb10/v21/{d2,d2s,d4}/*.jsonl`。

**Phase 3 音訊 smoke（半天內）**，三件都過才全跑 D3：
1. E4B server 加 `--mmproj mmproj-gemma-4-E4B-it-BF16.gguf`（BF16：PR #21421 作者說 F16／Q8_0 會重複輸出）；`/completion` 同時帶 `multimodal_data` 與 `n_probs`，字母機率要讀得到、`option_mass` ≥ 0.9。
2. 先跑模型卡官方音訊樣本做轉錄當基準線（kikemu 鐵律 7），再餵一個 10 秒晶晶體視窗要它轉錄：輸出繁體、英文保留拉丁字母。
3. 拿一個明顯的 core→cool 視窗問 Q1，P(B) > 0.5；拿一個乾淨視窗，P(A) > 0.5。
過了：`bench/v21_judge.py --engine e4b --mode audio`。

**Phase 4 分析與報告**：`bench/analyze_v21.py` → `results/25-asr-error-gate.md`、`results/v21.json`；README 新一列、REPORT §3p、`results/cost.md`、`handoff-gb10.md` §6c 加一條。

**Modal 備援**：`modal_v21.py`（L4：Breeze 用 torch image；E4B／26B 用既有 volume 的 gguf；mmproj 另下載到 volume）。音訊由本機重建後上傳，不在容器裡抓 podcast。預估 $2–3。

## 8. 結論（2026-10-15，Modal L4 備援跑完；照 §6 填。`results/25-asr-error-gate.md`、`results/v21.json`）

### 門檻

| 門檻 | 數字 | 判定 |
|---|---|---|
| G0 同一把尺 | 12 檔術語召回最大差 0.019 ✓、TER 最大差 0.013 ✗（T3 M3：L4 0.177 對 T4 0.191，L4 反而好） | ✗ → 另記 arm `Xbrz_l4`，往下跑。差在 generate() 與 pipeline 兩條解碼路徑偶爾挑不同的字（同權重），pipeline 版與 T4 的差在門檻內（0.010／0.007） |
| **G1 英文錯** | D2 E4B 純文字 AUROC **0.713**（預測 0.85）；抓到 80% 的 E 視窗時精確率 0.29（要標 59% 的視窗） | **✗** |
| G2 中文錯字 | D2 0.629（C-strong 0.686）；雜訊地板 31.6% > 20% | 只報不判（E4B 與 26B 都沒過 0.75，26B 0.666） |
| **G3 免費訊號** | D1 Breeze 平均 logprob：E 0.726、C 0.783（C-strong 0.835）；D2 − D1 = **−0.013**（E）、−0.154（C） | D1 當第一道 ✗（0.726 < 0.75，但 `d1_min` 0.747 很接近）；**加 E4B 不值得 ✗** |
| G4 音訊 | D3 E4B 聽音訊 + 文字：E 0.602、C 0.618；對 D2 −0.111／−0.011；smoke 3 項過 2 項（Q1 方向 ✗） | **✗** |
| G5 速度（L4） | E4B 三題每視窗 452 ms（每題 p50 151 ms）；閘門 246 s 對 Breeze 650 s = 0.38 | ✗（三題 > 300 ms；一題就夠的話 151 ms，比例 0.13 也還是 > 0.10；GB10 再量） |
| Kill | D2 E 0.713、D4 26B 0.777 | 未觸發（26B ≥ 0.70） |

### 數字

| 偵測器 | E 英文錯 | C 中文錯字 | C-strong | 每題 p50 |
|---|---|---|---|---|
| D1 Breeze 自己的 logprob（平均／最低） | 0.726／0.747 | 0.783／0.737 | 0.835／0.793 | 免費 |
| D5 Breeze 對 E4B-ASR 的分歧 | 0.665 | 0.775 | 0.833 | 免費（有第二套 ASR 的話） |
| D2 E4B 純文字 | 0.713 | 0.629 | 0.686 | 151 ms |
| D4 26B 純文字 | 0.777 | 0.666 | 0.718 | 307 ms |

- 選項對調（D2′）：Q1 翻轉 0.7%（E4B）／3.3%（26B）、平均 |Δp| 0.02／0.03，位置偏誤小；Q2 翻轉 10.8%／8.5%。Q3 四選一 E4B 0.45、26B 0.56；E4B 的 Q3 四個旋轉任一翻轉 50%（讀 Q3 不穩，Q1／Q2 分開問比較穩）。
- 校準（留一段三折、ε = 0.10）：E4B Q1 的 P(有錯) 幾乎不過 0.5（平均 0.04），可靠度表 0.4–0.6 桶只有 12 題；26B 0.8–1.0 桶 39 題裡 74% 真的有錯——26B 說「有錯」時多半真的有錯，但它只抓到一小部分。
- 抓到的是什麼：兩個模型最有把握的都是 bin（Bing）、cool（core）、comparative（compare to）、quality（wallet）、poor（porn）這種「英文字在上下文裡不通」的錯；漏掉的是 e-政府→一政府、queue→刻度、參考稿對齊產生的 wallet 刪除這種讀文字看不出來的。

### 三件跑了才知道的事

1. **讀文字抓不到 ASR 的錯，因為錯的字多半「看起來通」。** Breeze 認錯的英文字（cool、bin）在句子裡仍然像一個合理的詞，純文字的判斷只能靠「這個詞在這個領域不合」的先驗；26B 先驗強一點（0.78），E4B 不夠（0.71）。Breeze 自己的 logprob 是聲學證據，反而比 E4B 讀文字有用。
2. **中文錯字的正解本身不穩。** 第二路 Gemini 當假設時 32% 的視窗會被標成「有中文錯字」，所以 C 標籤有三成是正解的雜訊，不是 Breeze 的錯；G2 的門檻在這份語料上量不出來。
3. **E4B 幾乎不說「有錯」。** Q1 的 P(B) 平均 0.04，排名有一點訊號（0.71）但沒有可用的門檻；選擇性回答在 ε = 0.10 下自動處理 42%，處理的部分錯誤率 14%（沒達到 ε）。

### 預測對答案（12 條對 5 條）

對：D1 C 0.70–0.78（0.783）、D3 對 D2 ≤ +0.03（實際 −0.11，方向對、幅度錯）、D3 smoke 可能過不了（`/completion` 路徑真的不行，chat 路徑救回來；Q1 方向沒過）、26B 比 E4B 高 3–4 點（高 6 點，方向對）、D2′ 翻轉率 < 5%（Q1 對；Q2 錯）、E4B 每題 p50 80–120 ms 附近（151，略慢）。
錯：**D2 E 0.85（實際 0.713）**、D2 C 0.72（0.629）、D1 E 0.65–0.72（0.726，略高於預測）、D5 E 0.75（0.665）、Q3 準確率 0.60（0.45）、含拉丁字母的視窗 60–75%（49%）、D4 26B E 0.88（0.777）。

### 決定

- **Breeze 之後的「這段有沒有錯」閘門，第一道用 Breeze 自己的 logprob，不加 E4B。** E4B 讀文字的 AUROC 沒贏過免費訊號；26B 贏 5 點但每視窗要 0.3 秒 × 題數，而且抓 80% 的錯要標一半的視窗。
- 要上線的形狀：logprob 低於門檻的視窗轉人（或轉第二套 ASR 比對，D5 在中文錯字上 0.78）；不要用 E4B 讀文字當閘門。
- 這條線要再往前，需要（a）自發晶晶體的真實錄音與人工校對的正解（C 標籤的雜訊地板要壓到 10% 以下），（b）**不是**同一顆模型聽音訊：E4B 聽了反而更確定沒錯（D3 0.602），要抓 Breeze 的錯得用聽法不同的第二套 ASR（D5 分歧在中文錯字上 0.78）或 ASR 自己的信心。

### 音訊 smoke（Phase 3）

三個檢查：(1) 讀得到字母機率 ✓——但不是走 `/completion`：b11371 的原始 prompt 多模態形式（`{"prompt_string", "multimodal_data"}`）對 `<__media__>`／`<__image__>` 都回「找到 0 個媒體標記」400；改走 `/v1/chat/completions` 加 `input_audio` + `logprobs`、並送 `chat_template_kwargs.enable_thinking=false`（否則第一個 token 跟 `<|channel>` 搶，option mass 0.5–0.8）之後 option mass 1.0。(2) 轉錄 ✓——jfk.wav 全對；晶晶體視窗保留拉丁字母、繁體。但 E4B 自己聽那個視窗轉出來是「就是冰，就是冰水。我在非常酷的 Thing」：**Bing→冰、core→酷、team→Thing**，它聽的跟 Breeze 認錯的是同一批字。(3) Q1 方向 ✗——有錯的視窗加了音訊 P(有錯) 從 0.17 掉到 0.06。

全跑 D3（544 視窗，chat 路徑，每題 p50 265 ms、三題 795 ms）：**E 0.602、C 0.618**，比純文字 D2 低 0.111／0.011，**G4 ✗**。音訊沒有加分，反而更差：E4B 聽到的和 Breeze 寫的一致，所以它更確定「沒錯」。這就是 §2 討論時說的「同一顆模型的盲點是同一個」，量出來了。

### 補算（2026-10-15，`bench/v21_extra.py`、`results/v21_extra.json`，零 GPU）

1. **把握度加讀字母合起來**：E 的 AUROC，Breeze logprob 單用 0.726、26B 單用 0.777，兩者排名平均 **0.802**；logprob 加 E4B 0.762。兩個訊號互補（一個聽、一個讀）。留一段三折的邏輯斯迴歸反而低（0.74，折太少）。C 上沒有任何組合贏過 logprob 單用的 0.783。→ E4B 讀文字仍不值得加；**26B 加在 logprob 後面值得，0.802 是下一輪要打的數字**。
2. **中文錯字的代理正解**（Breeze 與正解不同、而第二路 Gemini 與正解一致才算錯；兩路都不同的 96 個視窗排除）：正例 210 → 114。AUROC 幾乎不動：logprob 0.781、E4B 0.604、26B 0.648、兩套辨識器分歧 0.802。**正解雜訊不是 E4B 在中文錯字上輸的原因**，結論不變。

費用：Modal L4 約 1.4 h（Breeze 12 檔 27 分鐘含 pipeline 對照、四個文字判斷容器各 2–4 分鐘、音訊 smoke 與探針五次各 3–10 分鐘、D3 全跑 5 分鐘）、mmproj 下載 CPU 2 分鐘，約 $1.3。

## 9. 限制（跑前就知道的）與選配

- 正解是 Gemini 產的（kikemu §7a）；本輪照用。Gemini 自己錯而兩路一致的地方沒人查過，E 標籤有未知的漏報；C 標籤有 §2 的雜訊地板。
- 只有 3 段講者、4 個條件高度相關；cluster = 3，所有 CI 都寬。這輪能回答「有沒有訊號、值不值得上線試」，訂產線門檻要自發晶晶體的真實錄音（會議記錄線的待辦）。
- Breeze 只在中文站得住（kikemu handoff-v8 A：日文 0.415），結論只限中文晶晶體。
- 不做自動改錯；不做日文。
- **選配 A（下一輪）**：Scribe × Gemini 36 處分歧讓 E4B 聽音訊裁決，四組對照（永遠選 Scribe 0.58、26B 純文字、E4B 聽音訊、E4B 對調），smoke 級不訂門檻；做之前要人工聽那 36 處（3 分鐘）寫 overrides，否則標籤偏向 Gemini。
