# Handoff v8 — JevBench 公開子集自跑（self-run，不排名）計劃（2026-09-26，跑前寫死）

## 0. 值不值得跑

**值得，但只值一個下午和一美元，而且要先講清楚它不會改變任何產線決策。**

能得到的：
1. 一個和別人**同一把尺**的數字。現在 `06-comparison.md` 裡引用的「26B-A4B 66.4 分」是別人提交的 NVFP4 版官方 534 題分，
   和公開 231 題子集的 Cygnet 87.9%、TypeLLM 84.4%、Open-Jev 85.3% 不是同一尺。自跑 231 題後，我們的 26B-A4B 和這三個可以並排。
2. 一次**英文通用題**的體檢。我們所有數字都是中文合成產線題；若英文通用題只有 70%，代表模板或讀法有問題，不是模型；若 85% 以上，
   代表 v2 的高分不是題目量身訂做的結果。
3. 順便量 E4B，補 v3 階梯的外部座標。

得不到的：排名（要 842 題含密封集）、對產線題準確率的任何新資訊、與 Jev 的直接勝負（Jev 沒有公開子集數字，只有官方 63.3 分）。

**判斷**：GPU 約 5 分鐘、< $0.5；寫 adapter 與整理約 2–3 小時。跑。

## 1. 資料與規則

- 來源：`fstandhartinger/jevbench` 釘 revision `f8ce71361165846101d02ebc83ad44e47ae44fc3`（TypeLLM 用的同一版），`datasets/public/{original,easy,hard}.jsonl` = 72 + 48 + 111 = 231 題；
  資料集 sha256 `dc3995d8…dde51` 要對上。題型 Choice 139、Noul 74、Score 18。
- 授權：只有 72 題 original 是 MIT 可重散布；easy/hard 公開題**不進 repo**，容器啟動時從 GitHub raw 釘版拉，只 commit 我們的答案檔（每題附 upstream 連結與 hash，照 TypeLLM 的做法）。
- 標示：所有輸出一律寫「**self-run on the public split (231), not an official JevBench result**」，這是 JevBench 作者對自跑分的要求（issue #7）。
- 不做：思考模式、順序平均、任何用 JevBench 題調參。

## 2. Adapter（`bench/jevbench.py`）

題目 → 我們的 prompt：
- system：英文版 `You are a decision engine. Answer with exactly one letter.`（新增 `SYSTEM_EN`；中文 system 不能直接用在英文題）。
- user：`[State]\n{state}\n\n[Question] {instructions}\nA. {label}: {criteria}\n…\n\nAnswer with the letter of the correct option only.`
- **Choice**：選項照原順序給字母，讀字母機率。
- **Noul**：二選一 `A. Yes / B. No`，P(true) = P(A)。
- **Score**：有序等級照 rubric 順序給字母；回傳完整機率向量；typed answer = 機率加權期望等級（與 TypeLLM 同），準確率用 argmax。
- 機率向量必須恰好覆蓋題目的 labels 且和為 1（嚴格 1e-3），不重新正規化 → 用 v5 T4 的精確讀法（llama-server 用 top-40，缺的字母補 1e-9 後正規化並記錄 `renormalized` 旗標；預期缺字母率 < 1%）。
- 溫度：兩個 arm 都報。raw（T=1）與 **T 取自我們合成資料 D0 cal 半**（跨 task 中位數，`analysis.json` 的 `temperature`），明寫「溫度不在 JevBench 上擬合」。
  Cygnet 是在 241 題自產題上擬合 T=3.4；我們的 T 來自不同語言與領域，ECE 可能不好看，如實報。
- 順序敏感度：每題另跑一次選項反序，報 argmax 翻面比例（JevBench 提交規範要求申報這一項）。

後端：llama-server L4、UD-Q4_K_M、`--swa-full`；231 × 2（正反序）× 2 模型（26B、E4B）= 924 次讀取，每次 ~200 ms → 約 3 分鐘加載入。
選配：SGLang FP8（帶 `<bos>`）同題再跑一次，作為 v6 結論的英文題複驗（+$0.5）。

## 3. 輸出

- `results/modal/jevbench/{26b,e4b}/answers.jsonl`：照 TypeLLM 的欄位（task_id、tier、family、question_type、labels、expected、predicted、correct、answer、probabilities、strict_valid、latency_s、task_source）。
- `results/13-jevbench.md`：
  - 表一：tier（original / easy / hard）× arm（26B raw、26B T、E4B raw）的 acc、Brier、top-label ECE、Score 題的 ordinal MAE、p50 延遲、缺字母率、反序翻面率。
  - 表二：與公開子集自跑者並排：Cygnet 87.9（Gemma-4-12B 凍結）、Open-Jev-27B v1.1 85.3（訓練）、TypeLLM 84.4 / 98.7 思考（Qwen3.8-27B）、JevK5 hard 0.739（訓練）、imajev-4B hard 0.703（訓練）。
  - 用 upstream 的 `python -m jevbench.cli summarize` 重算一次，與我們自己算的對上才寫。
- `06-comparison.md` 準確率表加一列（標 self-run），HTML「08 · 參考」對照表加一列。

## 4. 判定（pre-registered）

| 26B raw 公開 231 題 acc | 解讀 |
|---|---|
| ≥ 85% | 與 Cygnet / Open-Jev 同級：26B-A4B 凍結讀字母在英文通用題上站得住，v2 高分不是題目量身訂做 |
| 78–85% | 與 TypeLLM 無思考同級：可用，但 12B dense 的 Cygnet 比 26B-A4B MoE 好，值得記一筆 |
| < 78% | 模板或讀法在英文題上有問題（先查 system prompt、Score 題的 rubric 排法、缺字母率），修正後重跑一次；仍低就如實報 |

E4B 預期落在 26B 之下 5–15 點（v3 階梯的比例）；若 E4B ≥ 26B，表示 26B 的 prompt 有問題。
反序翻面率預期 5–10%（SemIf 102 題翻 10、open-alternative-jev 6–9%）；> 15% 要寫進 REPORT 當風險。

## 5. 順序與預算

1. 寫 adapter、本機用 5 題 dry-run 檢查 prompt 與機率向量格式（無 GPU）。
2. L4 一次跑完 26B + E4B 正反序（`--which jevbench`），約 5 分鐘、< $0.3。
3. 上游 summarize 重算、寫 13-jevbench.md、更新 06 與 HTML。
4. 選配 SGLang 複驗（+$0.5）。

合計：GPU < $1；人力半天。

## 6. 一句話結論（三選一，跑完填）

**跑完（2026-09-26）**：選第一句。26B raw 88.7%（hard 77.5%）、E4B 78.8%、反序翻面 6.5% / 10.0%；合成資料的 T 套上去 ECE 0.093 → 0.044。L4 上 64k ctx + `--swa-full` 會 OOM，改 32k ctx、2 slot、不開 swa-full。


- 「26B-A4B 凍結讀字母在 JevBench 公開子集 X%，與 Cygnet / Open-Jev 同級，**v2 的高分不是題目量身訂做**。」
- 「X%，與 TypeLLM 無思考同級；12B dense 在英文通用題上略勝 26B-A4B MoE，產線題結論不變。」
- 「X%，明顯落後；原因是＿＿（模板／Score 排法／缺字母），修正後 Y%。」
