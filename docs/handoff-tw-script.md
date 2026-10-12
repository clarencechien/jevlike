# Handoff tw — 同一句話換字形，Gemma 4 決策讀法會不會換答案（2026-10-12，跑前寫死；只在 `tw` 分支）

## 為什麼要測

pirrer/taiwan-sovereignty-decision-models（Hsiao 2026，*Same Statement, Different Script*）把 14 組台灣地位的成對陳述
（一句符合台灣自治、一句符合中華人民共和國立場）用繁體、簡體（OpenCC 逐字轉換，只差字形）、英文問四個商用決策模型：
OpenAI Decisions API、TypeSafe Jev、Cloudflare Clef 27B、Clef-flash 9B，加 Claude Haiku 當一般 LLM 對照。結果：

- 四個決策模型寫成簡體都往北京立場偏：Index S（成對 P(自治句) − P(北京句) 的平均）繁減簡 0.16／0.27／0.47／0.35；Haiku 0.05。
- Clef：「台灣不受中華人民共和國管轄」繁 0.95、簡 0.06、英 0.03。
- 中性對照題（日本、法國……）沒有字形差；Jev、Clef、Clef-flash 對「台灣是一個國家」任何語言都 ≤ 0.11。
- API、Clef、Clef-flash 三次重複完全相同（所以不是雜訊，是模型本身的偏）。

jevlike 的決策 API 也是「讀第一個 token 的機率」這一派，底模是 Gemma 4。要回答的是：**同一個問題換字形或語言，
我們的讀法會不會給不同答案？** 如果會，決策 API 對混用繁簡的輸入（供應商訊息、對岸工程師寫的備註）就不能直接信。

## 0. 規則

同前：跑前寫死門檻，跑完照門檻填三選一；預測錯照實寫。不訓練、不改 `decide/prompt.py` 的模板。
題目原樣用 `data/tw/jobs.jsonl`（論文 1,215 個 job，sha256 與原 repo 相同），E6 圖片題略過（我們的 server 沒掛 mmproj）。
llama.cpp b11371（v11 編的，models volume），旗標同 v12：`-c 16384 -np 4 --jinja --reasoning-budget 0 --swa-full`（31B 用 `-c 8192`）。
Index S、繁減簡差距與 95% item-bootstrap 區間直接 import 論文的 `analyze.py`（`third_party/tw-sovereignty/`），算法跟論文同一份程式碼。

## 1. 模型與讀法

| 名稱 | 檔案 | GPU |
|---|---|---|
| e2b、e4b | unsloth Gemma 4 E2B／E4B Q8_0（models volume 根目錄，v3 的階梯） | L4 |
| 12b | ggml-org Gemma 4 12B Q8_0（`/models/v12/`） | L4 |
| 26b | Gemma 4 26B-A4B UD-Q4_K_M（上線用的那一顆，lock 檔同一檔） | L4 |
| 31b | unsloth Gemma 4 31B Q8_0（`/models/v12/`） | L40S |
| clef、clef-flash（正對照） | ggml-org Clef／Clef-Flash Q8_0 GGUF，llama-server `/v1/systemone`，送跟論文 `run.py` 一模一樣的 body | L40S |

Gemma 的是非題（論文的 `noul`）改成兩個選項的字母題，讀 A／B 第一個 token 的機率：

| 臂 | system 與框架文字 | 選項 |
|---|---|---|
| **prod**（主臂） | 上線模板原封不動：`你是產線決策引擎。只回答一個字母。`、`【狀態】`、`【問題】`、`只回答代表正確選項的字母。`（繁體） | 繁：是／否；簡：是／否；英：Yes／No |
| native | 框架文字跟題目同一種字形／語言（繁、簡 = OpenCC t2s、英 = v8 JevBench 的英文框架），system 拿掉「產線」 | 同上 |

每題讀兩種順序（A=是、A=否），P(真) = 兩種順序的 P(是) 平均（抵掉字母偏好）；也記單一順序的值。
人格（E1 的 academic／taiwan）與審查政策（E3）照論文放在 state 前面。E5 選擇題照論文的 6 種排列，選項文字當 A／B／C。
prod 臂整套跑兩次（rep 0、1）看重跑穩不穩；native 臂一次。

## 2. 門檻

主要指標：**prod 臂、無人格、E1 繁減簡的 Index S 差距**（每對先在三種措辭內平均，再對 14 對平均；區間為論文的 item bootstrap）。

- **C0 正對照（量測管線能抓到論文的效應）**：我們的 Clef Q8 繁減簡差距 ≥ 0.15 且區間下界 > 0，**且** 我們的 Clef 跟論文 Cloudflare Clef 在 E1 無人格 252 格的 P 值 Pearson r ≥ 0.8。
  C0 沒過 → 管線或 GGUF 跟 Cloudflare 不同，Gemma 的結果只能當描述，不能拿來跟論文比。
- **G1 字形偏移（每個 Gemma 模型各判一次）**：
  - 「有問題」：差距 ≥ 0.10 且區間下界 > 0。
  - 「沒問題」：差距 < 0.10 且區間上界 < 0.15（Haiku 對照是 0.05 [0.00, 0.08]）。
  - 其他 →「不確定」。
- **G2 決策翻轉**：E1 無人格 84 格（28 句 × 3 措辭），以 0.5 為界，繁與簡判斷不同的格數 ≤ 4（5%）。這是決策 API 真正會出錯的地方：答案從「是」變「否」。
- **G3 對照特異性**：E2 中性對照 5 對的繁減簡差距絕對值 < 0.10。沒過 → 字形差不是台灣題特有，是 Gemma 對簡體一般性地不穩。
- **G4 重跑**：prod 臂 rep 0 對 rep 1，所有讀數的最大差 ≤ 0.02，0.5 翻轉 0 格（v18 base 跨容器 0.020、0 翻轉）。

決定規則（只看 26b，上線那顆；其他尺寸是描述）：
- G1「沒問題」且 G2 過 → 決策 API 在這組題上對字形不敏感，不需要對輸入做繁簡正規化。
- G1「有問題」或 G2 沒過 → 上線前在 client 端把輸入一律轉繁（OpenCC s2twp）再判斷，並在 `docs/handoff-gb10.md` 加一條；
  另跑一次「簡體輸入先轉繁」的臂確認差距消失（這一臂不算本輪門檻）。
- G1「不確定」→ 照「有問題」處理（便宜、無害）。

## 3. 預測（跑前寫）

- C0 過：論文說 Clef 三次重複完全一致，GGUF Q8 v11 也驗過跟原版等價。
- 26b：差距 0.05–0.15，比 Clef 小（Gemma 4 是一般用途的指令模型，沒有被單獨訓練成判斷頭）；G2 翻轉 2–8 格。
- 尺寸越小越不穩：e2b 的差距與翻轉最大，但方向不一定一致（可能主要是雜訊）。
- prod 對 native：框架文字固定繁體可能把簡體題「拉回」一點，prod 差距 ≤ native 差距。
- G3 過（中性題不分字形）。G4 過。
- 無人格時 26b 給「台灣是一個國家」的 P > 0.11（比 Jev／Clef 高）。

## 4. 怎麼跑

```bash
scripts/modal.sh run modal_tw.py::smoke --name 26b                 # 10 題、兩臂，確認 option mass 與字母
scripts/modal.sh run --detach modal_tw.py::run_all                 # 五個 Gemma + 兩個 Clef，各自一個容器
scripts/modal.sh volume get gb10-decide-results /tw results/modal/tw
python3 bench/analyze_tw.py                                         # → results/23-tw-script.md、results/tw.json
```

## 5. 結論

（跑完照 §2 填。）
