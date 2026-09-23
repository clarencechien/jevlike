# Handoff v3：題目問題，還是模型強度問題？（E2B / E4B / 26B 階梯實驗）

日期：2026-09-23
負責人：Clarence
執行者：Claude Code web + Modal（L4）+ AI Studio（Gemini，只用來改寫題目）
前置：v1（題目與指標定義）、v2（Modal 執行環境）、`results/REPORT.md`（v2 結果）。本文件只寫新增的實驗。

---

## 0. 要回答的一個問題

v2 的結果：Gemma 4 26B-A4B 在 10 個 task 上零樣本 7 個 ≥ 0.98，連 10 類相近意圖都 1.00，raw confidence 平均 0.99。

這有兩種解釋，v2 的資料分不出來：

- **H1 題目太簡單**：state 是拿著 criteria 寫出來的，線索太明顯；任何會讀中文、能對 criteria 的模型都接近滿分。
- **H2 26B 真的強**：這個難度已經能區分模型等級，26B 的 0.98 反映真實能力。

分辨的方法是**兩條階梯交叉**：模型從小到大、題目從易到難。如果小模型在現有題目上也滿分，是 H1；如果小模型明顯掉，是 H2；如果小模型只在新的難題上掉，那現有題目是 H1、新題目才有鑑別力。

**注意這不是「題目太爛」。** v2 的規則基線最高 0.885、TF-IDF 加 100 筆只有 0.85，題目對淺層方法並不簡單；它只是對 LLM 這一級的讀者太簡單。實驗要找的是「難度落在哪兩個模型之間」。

---

## 1. 預先登記的判讀規則（跑之前寫死，跑完不准改）

以每個 task 的 test 集 accuracy 為主，配對 bootstrap 95% CI（1000 次重抽）。「相當」定義為 CI 重疊且 McNemar p ≥ 0.05；「明顯掉」定義為差距 ≥ 5 個百分點且 p < 0.05。

| 現有合成集 | 新難題集（§3） | 判讀 | 後果 |
|---|---|---|---|
| E4B ≈ 26B | E4B ≈ 26B | **H1 強版**：兩級模型都分不出，題目對這一級全部飽和 | GB10 決策層可用 E4B；真實資料驗證前不得引用任何「26B 強」的說法；新難題集也要再加難 |
| E4B ≈ 26B | E4B 明顯掉 | **H1（合成集）+ 新集有鑑別力** | 合成集降級為 smoke test；後續所有比較改用新難題集；26B 在新集上的分數才是可引用數字 |
| E4B 明顯掉 | E4B 明顯掉 | **H2**：現有難度已能分出等級 | 26B 的 0.98 可信；GB10 決策層要用 26B；E4B 不夠 |
| E4B 明顯掉 | E4B ≈ 26B | 不一致，優先懷疑新集有問題（label 錯、改寫改變語意） | 回頭抽查新集，先不下結論 |

E2B 當地板：E2B 在某個 task 也 ≥ 0.95 的話，那個 task 直接判 H1，不用看 E4B。

另外兩個獨立於準確率的指標，用來看「模型有沒有在猶豫」：

- **錯誤偵測 AUROC**：以 raw confidence 當「這題會不會錯」的分數，算 AUROC。接近 0.5 表示信心和對錯無關（純表面匹配）；≥ 0.8 表示模型知道自己哪裡不確定。v2 的 26B 在 `m_alarm_severity` 上這個值預期很低，要量出來。
- **26B 對 E4B 的信心差**：在 E4B 答錯而 26B 答對的題目上，26B 的 raw confidence 分布。若仍是 0.99，表示 26B 也只是碰巧線索夠。

---

## 2. 模型階梯

| 模型 | 檔案（HF，以頁面為準） | 預估大小 | 在 L4 上 | 備註 |
|---|---|---|---|---|
| Gemma 4 E2B | `unsloth/gemma-4-E2B-it-GGUF` Q8_0 | ~3 GB | 可 | 地板 |
| Gemma 4 E4B | `unsloth/gemma-4-E4B-it-GGUF` Q8_0 | ~5 GB | 可 | 主角；gemma-jev 用的就是這個 |
| Gemma 4 26B-A4B | v2 同一份 UD-Q4_K_M | 16.95 GB | 可 | **不重跑**，直接用 `results/accuracy/*.jsonl` 的 raw logprobs；但新資料集要跑 |
| （選配）Gemma 4 31B dense | `unsloth/gemma-4-31B-it-GGUF` Q4_K_M | ~18–19 GB | 勉強（`-c 2048`） | 只在 H2 成立、想看「dense 比 MoE 多多少」時做 |

三個規則：

1. **prompt 完全相同**。`decide/prompt.py` 不動；三個模型吃一模一樣的字串。若 v2 之後改過 prompt.py 或 llama-server 版本，26B 也要重跑，不能混用舊 logprobs。
2. **每個模型都要過 smoke**：`first_token` 是字母、`missing` 為空、`/tokenize` 確認 `A` / ` A` 的 token id（Gemma 4 系列應共用 tokenizer，但要驗）。E2B/E4B 可能沒有 thinking channel，也要確認。
3. **全部開 `--swa-full`**（v2 Q3 的結論），延遲順便記，但本實驗不比延遲。

---

## 3. 題目階梯

### 3.1 D0：現有合成集（v2）

10 task × 200 筆，沿用 v2 的 calibration / test 切分（依 `pair_id`）。不動。

### 3.2 D1：擾動集（程式產生，不需要 LLM）

對 D0 的 **test 集**做三種擾動，label 不變，各產一份：

| 擾動 | 做法 | 測什麼 |
|---|---|---|
| D1-typo 錯字 | 中文：同音／形近錯字表（Fable 寫 `data/seeds/typos_zh.json`，≥ 200 對），每句隨機替換 1–2 處；英文：字元交換／漏字 1–2 處 | 對雜訊的穩健度 |
| D1-cue 線索移除 | 找出 state 與該 task criteria 文字的重疊 n-gram（char 2–4 gram），把重疊詞換成泛稱（「溫度超上限」→「數值異常」）；記錄每筆移除了幾個線索 | **對 criteria 措辭的依賴** — 這是 H1 的直接證據 |
| D1-mix 混訊 | 在 state 前後接上另一個 task 的無關句子（同一條線的其他訊息） | 對干擾的抵抗 |

D1-cue 要人工抽 20 筆確認移除後 label 仍成立；不成立的筆數記進 MANIFEST，超過 10% 就縮小移除範圍。

### 3.3 D2：盲寫難題集（Gemini 寫 state，看不到 criteria）

這是 v2 沒有的東西：**出題時不給 criteria，label 事後由兩個獨立標註者決定。**

1. **寫 state**：用 AI Studio 的 Gemini（不是 Gemma，避免受測者自己出題；也不是 Fable，避免和 D0 同一種筆法）。輸入只有：task 名稱、選項名稱（不含 criteria）、20 筆 D0 的 state 當風格參考、以及指示「寫成產線真實訊息：操作員口語、半句話、縮寫、alarm code 對照表可能過期、可能兩件事混在一則」。每 task 60 筆，共 600 筆。免費額度 1,500 次／天，分兩天。
2. **獨立標註**：Fable 和 Gemini 各自拿完整 criteria 獨立標 label，不互看。
3. **一致性**：算每 task 的 Cohen's kappa。兩者一致的筆數進 D2；不一致的筆數由人（Clarence）裁決，裁決結果另存 `D2-adjudicated`，同時保留為「真正的 borderline 子集」。
4. **切分**：D2 全部當 test，不做校準（校準沿用 D0 的 calibration 集，這本身也是一個測試：D0 上校準的門檻在 D2 上還有效嗎）。

D2 的 kappa 本身就是結論的一部分：如果 Fable 和 Gemini 只有 0.8 一致，模型 0.8 就是到了「兩個強模型的一致水準」，這個數字要寫進主管版報告，取代 1.00 當參考天花板。

---

## 4. 執行

### 4.1 步驟與時間

| 步驟 | 內容 | 時間 | GPU |
|---|---|---|---|
| S1 | 下載 E2B、E4B GGUF 進 Volume；各跑 smoke | 0.5 h | 0.2 h |
| S2 | Fable 寫 `typos_zh.json`；`data/perturb.py` 產 D1 三份；人工抽查 D1-cue | 2 h | 0 |
| S3 | Gemini 寫 D2 state（600 筆）；Fable 與 Gemini 獨立標；算 kappa；人裁決不一致筆 | 1 天（受免費額度限制）+ 人 1 h | 0 |
| S4 | E2B、E4B 跑 D0 + D1 + D2；26B 跑 D1 + D2（D0 沿用） | 0.5 h | 約 1 h（約 3 模型 × 4,600 題，每題 < 0.3 s） |
| S5 | `bench/analyze_ladder.py`：配對 bootstrap、McNemar、錯誤偵測 AUROC、信心差、逐題交叉表 | 1 h | 0 |
| S6 | `results/06-ladder.md` + 更新 `REPORT.md` 加「題目 vs 模型」一節 + 更新 imitator HTML | 1 h | 0 |
| **合計** | | **約 2 個工作天（含等 Gemini 額度）** | **約 1.5 h，L4 約 $1.5** |

### 4.2 `modal_app.py` 改動

- `MODEL_FILE` 改成參數，`run_bench` 多接 `model` 引數，Volume 裡放三份 GGUF。
- `bench/accuracy.py` 多接 `--dataset {D0,D1-typo,D1-cue,D1-mix,D2}`，輸出路徑 `results/accuracy/{model}/{dataset}/{task}.jsonl`。
- 用 `--detach` 一次排三個模型；結果逐筆 append，支援 `--resume`。

### 4.3 `bench/analyze_ladder.py` 要輸出

1. **主表**：task × 模型 × 資料集 的 accuracy（含 95% CI），加 hard 子集分項。
2. **判讀表**：每個 task 套 §1 的規則，直接標 H1 強版 / H1 / H2 / 不一致。
3. **逐題交叉表**：每個資料集上，(E4B 對, 26B 對) / (E4B 錯, 26B 對) / (E4B 對, 26B 錯) / (都錯) 四格；McNemar 用第二和第三格。
4. **錯誤偵測 AUROC**：每模型 × 資料集。
5. **D1-cue 的線索依賴曲線**：以「移除線索數」分組，畫 accuracy 隨移除數的下降（三條線 = 三個模型）。這張圖是 H1 最直觀的證據。
6. **D2 kappa 表**：Fable vs Gemini 每 task 的 kappa，以及模型對 D2 一致筆的 accuracy。
7. **校準遷移**：用 D0 calibration 集擬合的 affine 參數，在 D2 上的 ECE 與選擇性準確率 @0.9；對照在 D2 上重新擬合的數字。

---

## 5. 報告要怎麼寫（`results/06-ladder.md`，並併入 REPORT.md）

第一段直接回答：**每個 task 落在 H1 還是 H2**，附一句依據。不要平均成一個總分，10 個 task 的答案很可能不同（例：`x_ticket_route` 大概率 H1，`m_alarm_severity` 可能 H2）。

第二段回答「GB10 該放哪一個」：H1 的 task 用 E4B、H2 的 task 用 26B，或全部用 26B 但知道哪些其實 E4B 就夠。

第三段回答「拿什麼跟 Qwen 122B / GPT-5 比」：只有被判定有鑑別力的資料集（D1-cue、D2）才能拿去比；D0 明說已飽和，不再引用。

第四段更新主管版報告的措辭：把「零樣本近滿分」改成「在乾淨題目上近滿分；在盲寫難題上 X，兩個強模型的互相一致率是 Y，模型已到／未到人類一致水準」。

---

## 6. 交付檢核

```
[ ] results/06-env-ladder.md      E2B/E4B 檔名、大小、smoke、tokenize 檢查
[ ] data/perturbed/D1-*/           三份擾動集 + MANIFEST（每筆移除線索數、抽查結果）
[ ] data/blind/D2/                 Gemini state + 兩份獨立 label + kappa + 裁決記錄
[ ] results/accuracy/{model}/{dataset}/*.jsonl   raw logprobs
[ ] results/fig/cue-removal-curve.png、ladder-*.png
[ ] results/06-ladder.md           §1 判讀表逐 task 填好、§4.3 七項齊全
[ ] results/REPORT.md              新增「題目 vs 模型」一節；D0 標記為已飽和
[ ] imitator HTML                  第 03 判定表的 △ 項依結果改 ○ 或 ×；新增「兩個強模型的一致率」數字
[ ] results/cost.md                累計費用
```

## 7. 之後（不在 v3 範圍）

- 真實資料 200–500 筆 + 兩位工程師標註 + kappa（v1 §6）：只能在 GB10 做。
- Qwen 122B（地端 vLLM，logprobs 可用）與雲端前沿模型的比較：用 D2 與真實集；推理型 API 若不開 logprobs，只比生成準確率。
- 若 H2 成立且想壓延遲：拿 26B 的 D2 輸出當 teacher，蒸餾 E4B 或 Laya 多語版（v1 §8）。
