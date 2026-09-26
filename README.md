# jevlike — Jev 式 typed decision on Gemma 4 26B（GB10 前置實驗）

讀選項字母的第一個 token logprob 當決策 API；不生成文字。合成的台灣 SMT 產線資料（10 類判斷題 × 200 筆），全部在 Modal 雲端 GPU 上跑，真實資料不出廠。

- 長官版 HTML（公開）：<https://imitator.ai-apps.work/r/gb10-typed-decisions>
- 工程報告：`results/REPORT.md`；分項報告 `results/00`–`12`；費用 `results/cost.md`
- 生態調查與借鏡：`results/10-landscape.md`（本檔 §5 是摘要）

## 1. 一句話

值得做、用現有的 Gemma 4 26B-A4B 自己做、不採購 Jev。十類題七類零標註 ≥ 98%；弱的三類靠改寫判斷標準與幾十筆校準；一次判斷 L4 上 0.2 秒（共用現場狀況 0.07 秒）。推理引擎 llama-server 與 SGLang 都可用，SGLang 批次快 5 倍，前提是 prompt 帶 `<bos>`。八輪實驗約 8.9 GPU 小時、約 $15。在 JevBench 公開子集自跑 88.7%，高於同派的 Cygnet 與訓練過的 Open-Jev。

## 2. 走到哪裡了（八輪，每輪先寫門檻再跑）

| 輪 | 交接文件 | 問的問題 | 答案 | 結果檔 |
|---|---|---|---|---|
| v1/v2 | `docs/handoff-v1.md`、`v2.md` | 讀選項機率當決策 API，準不準、快不快、要標多少？ | 七類零標註 ≥ 0.98，L1 206 ms（`--swa-full` 137），多數題零筆標註；raw 信心無鑑別力，門檻要校準 | `results/00`–`04`、`REPORT.md` |
| v2 補 | — | 弱題是不是標準沒寫清楚？ | 把「連續幾點、超限幾次」寫成可數規則：SPC 0.815 → 0.95、急迫度 0.81 → 0.885；held-out 種子等幅 | `05-criteria-v2.md` |
| v3 | `handoff-v3.md` | 題目太簡單，還是模型真的強？ | E2B/E4B/26B 階梯：四類 E2B 就夠（H1）、六類 26B 真的強（H2）；E4B 先答、34% 再問 26B，acc 不變、26B 負載剩三分之一 | `06-ladder*.md`、`cascade.json` |
| v4 | `handoff-v4-sglang.md` | 換 SGLang 值不值？ | 速度門檻 G1–G4 全過（K=16 3.7×、c=32 6×），但 acc 低 2.8 點、重跑不穩 → 視同沒過 | `07-*.md` |
| v5 | `handoff-v5-typellm.md` | TypeLLM 的技巧有用嗎？ | JSON prefill +1.3 點、順序平均 −1.3 點，都沒過；量到弱題順序敏感度 21% / 19%；加了標籤自檢與驗證腳本 | `08-typellm-followups.md`、`verify.md` |
| v6 | `handoff-v6-sglang-stability.md` | SGLang 掉分的真正原因？ | 差在第 0 個 token：llama-server 加 `<bos>`、SGLang 不加。餵同樣 id 後 0.958 vs 0.959；補 `<bos>` + 批次不變旗標重跑一致率 99.85% → SGLang 回到候選 | `09-sglang-stability.md` |
| v8 | `handoff-v8-jevbench.md` | 同一把尺：JevBench 公開 231 題自跑（不排名） | 26B **88.7%**（hard 77.5%），高於 Cygnet 87.9、Open-Jev 85.3、TypeLLM 84.4；E4B 78.8%；合成資料的溫度搬過去 ECE 0.093 → 0.044 | `13-jevbench.md`、`data/jevbench/` |
| v7 | `handoff-v7-borrowed.md` | 三十幾個開源替代品有什麼可抄？ | conformal 門檻 + lock 檔（採用，`--check-lock` 抓到換錯引擎 10 項）、option mass 模板檢查（採用）、packed readout（不採用：後面的題翻 11–13%） | `11-*.md`、`12-packed.md`、`thresholds.lock.json` |

## 3. 效果多好（D0 test 每 task 100 筆，L4 / L40S）

| 指標 | 數字 | 註 |
|---|---|---|
| JevBench 公開 231 題（self-run） | 88.7%，hard 77.5% | Cygnet 87.9、Open-Jev-27B 85.3、TypeLLM 84.4（各自自跑）；不是官方排名 |
| 零標註 ≥ 0.98 的 task | 7 / 10 | `m_alarm_category`、`m_needs_dispatch`、`q_defect_root`、`p_line_change`、`x_ticket_route`、`x_escalate`、`x_10way_intent` |
| 三類弱題（改寫標準後） | 急迫度 0.885、SPC 0.95、UPH 0.93 | 錯集中在相鄰等級；UPH 該用公式 |
| 5% 錯誤預算下強題 coverage | 0.91–1.00，實際錯 0–3.3% | conformal，保證在 8/10 task 成立（`11-thresholds.md`） |
| 單題 p50 | L4 137 ms（`--swa-full`）、L40S 61 ms | 生成 JSON 對照 496 ms |
| 共用 state 問 10 / 16 題 | L4 每題 73 ms；L40S llama 813 ms vs SGLang 217 ms（K=16） | SGLang 要帶 `<bos>` |
| E4B 級聯（門檻 0.999） | acc 0.952 vs 全 26B 0.955，34% 送 26B | 平均延遲 −20% |
| 順序敏感度 | 急迫度 21%、SPC 19%、其餘 ≤ 3% | 上線選項順序固定 |
| 累計費用 | ≈ $15、8.6 GPU h | 原估 $5.5–7.5 只算前兩輪 |

## 4. Quickstart

```bash
pip install 'modal[api-proxy-support]' numpy scikit-learn matplotlib
# llama-server（L4）
scripts/modal.sh run modal_app.py --which download && scripts/modal.sh run modal_app.py --which smoke
scripts/modal.sh run --detach modal_app.py --which accuracy --server-extra "--swa-full"
scripts/modal.sh volume get gb10-decide-results / results/modal/ && python3 bench/analyze.py
# SGLang（L40S，FP8；靜態模板已含 <bos>）
scripts/modal.sh run modal_sglang.py --which download && scripts/modal.sh run modal_sglang.py --which smoke_sglang
scripts/modal.sh run --detach modal_sglang.py --which accuracy --args "--out-sub D0 --control-n 0"
# 離線：重算所有結果檔 / conformal 門檻與 lock 檔 / 換模型或後端後的漂移檢查
python3 bench/verify.py && python3 bench/thresholds.py && python3 bench/verify.py --check-lock results/modal/accuracy/q8/D0
```

`scripts/modal.sh` 把本環境的 `modal` / `modal_secret` 環境變數映射成 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`；`GB10_GPU` 選卡（L4 / L40S / H100）。token 只走環境變數，不進 repo。

### 結構

- `PLAN.md`；`docs/` 七份交接文件（每份先寫門檻，跑完在 §結論填三選一）
- `modal_app.py` llama-server 入口（`MODELS`：26b UD-Q4_K_M / q8 / bf16 / e4b / e2b）；`modal_sglang.py` SGLang 入口（fp8 / bf16，L40S 需自帶 fused-MoE Triton 設定檔）
- `decide/` `prompt.py`（模板：`/apply-template` 學來，或 SGLang 用的 `GEMMA4_TEMPLATE_NOTHINK_BOS`；T1 變體）、`client.py`（llama `/completion`；SGLang `/generate` 指定 token id / `input_ids`；每次回傳 `option_mass`）、`labels.py`（字母單 token 自檢）
- `data/` `seeds/`、`gen/`、`hard/`、`rules/`、`synthetic/`（D0 + MANIFEST + SPOTCHECK）、`heldout/`、`perturbed/`（D1）、`blind/`（Gemini 盲寫 D2）、`tokenized/`（llama-server 切好的 token id）
- `bench/` `smoke*.py`、`latency.py`、`accuracy.py`、`permute.py`、`packed.py`、`v4bench.py`、`jevbench.py`、`tokenize_dump.py`、`option_mass.py`、`thresholds.py`、`analyze*.py`（v2 / ladder / v4–v8）、`verify.py`（含 `--check-lock`）、`render_html_report.py`
- `data/jevbench/` JevBench 公開 231 題（MIT，釘版）；`third_party/jevbench/` 評分 harness（MIT）
- `results/` 分項報告 `00`–`13`、`REPORT.md`、`cost.md`、`thresholds.lock.json`、`fig/`、`modal/`（原始 logprobs）

### 踩過的坑（接 GB10 時先看）

- **`<bos>`**：llama-server 對字串 prompt 會加，HF tokenizer（SGLang、vLLM、TypeLLM）對 Gemma 4 預設不加。少這一個 token 十類平均掉 2–3 分（v6）。smoke 會比對兩邊 prompt token 數。
- **thinking**：`/apply-template` 帶 `chat_template_kwargs: {"enable_thinking": false}`，模板結尾是空的 `<|channel>thought\n<channel|>`。TypeLLM 因此不支援 Gemma 4。
- **前綴 cache**：llama-server 開 `--swa-full`，同一 slot 順序送；SGLang 先暖一題再整批。多題排成一列一個前向（packed）在 Gemma 4 上會讓後面的題答案變，不用。
- **選項順序**：弱題兩成會因順序改答案，上線順序固定、校準用同一順序。
- **信心**：raw 信心平均 0.99，門檻用校準後信心；`thresholds.lock.json` + `--check-lock` 當 CI。
- **SGLang 重跑**：帶 `<bos>` 99.7%，加 `--enable-deterministic-inference` 99.85%（單題慢三成）；`gemma4-mtp` 映像的 `/v1/tokenize` 會崩，自檢改走 HF tokenizer。

## 5. 參考：別人怎麼做、我們借鏡了什麼

Jev（TypeSafe，2026-09-15）發表後兩週，開源替代品三十幾個，分四類（詳 `results/10-landscape.md`，含來源連結）：

| 做法 | 代表 | 自報成績 | 對我們 |
|---|---|---|---|
| 零樣本讀選項機率（我們這一派） | **Cygnet**（凍結 Gemma-4-12B，vLLM，一個溫度）、open-alternative-jev（packed readout）、SemIf、openjev-sglang、verdict、gemma-jev | Cygnet JevBench 官方第 4（61.8）、公開題 87.9% | 同一條路在公開評測站得住；Gemma 4 只有 Cygnet 和我們 |
| LoRA + 決策頭 | **decider-4b**（8k 筆，榜首 64.1）、JevK5（大模型開思考蒸餾）、Open-Jev（148k 筆、反事實資料）、Kev、imajev | 4B 訓練後贏 Jev（63.3） | 真實工單到手後的下一步：26B 開思考出題標答 → E4B LoRA |
| 小型非自迴歸 | Laya、von、poorjev 的 NLI | CPU 可跑 | 我們量到 Laya 零樣本接近亂猜 |
| 校準與門檻工具 | poorjev（conformal）、jevcal（lock 檔 + CI）、jevkit | ECE 0.170 → 0.071 | 已抄成 `bench/thresholds.py` + `verify.py --check-lock` |

**借鏡了什麼，結果如何**：留下的四件是 conformal 門檻 + lock 檔（v7 P3）、option mass 模板檢查（P2）、標籤自檢與驗證腳本（v5 T0）、指定 token 讀機率（T4）；不留的三件是 JSON prefill（+1 點）、順序平均（−1 點）、packed readout（後面的題翻 11–13%）。

**我們比別人多做的**：Gemma 4 的正確模板（空 thought channel、`<bos>`），沒有任何專案寫到；題目簡單還是模型強的階梯；兩個後端同題對照；每輪預先登記門檻、held-out 驗證、獨立抽查。
