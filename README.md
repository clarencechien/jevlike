# jevlike — Jev 式 typed decision on Gemma 4 26B（GB10 前置實驗）

讀選項字母的第一個 token logprob 當決策 API；不生成文字。合成的台灣 SMT 產線資料（10 類判斷題 × 200 筆），全部在 Modal 雲端 GPU 上跑，真實資料不出廠。

- 長官版 HTML（公開）：<https://imitator.ai-apps.work/r/gb10-typed-decisions>
- 懶人包（公開）：<https://imitator.ai-apps.work/r/jevlike-primer>（原理、溫度怎麼校、為什麼準、什麼時候會失靈）
- 工程報告：`results/REPORT.md`；分項報告 `results/00`–`21`；費用 `results/cost.md`
- 生態調查與借鏡：`results/10-landscape.md`（本檔 §5 是摘要）

### 給在 GB10 上接手的 Claude Code：從這裡開始

依序做，每一份都先寫門檻再跑、跑完填結論、commit + push（分支 `gb10/<日期>`）。**真實產線資料不離開 GB10、不進 git**（`data/real/`、`results/gb10-real/` 已 ignore）。

1. **主線**：`docs/handoff-gb10.md`。26B 重現 → GB10 重量速度 → 真實資料校準門檻 → 結案。不用 Modal，所有 bench 用 `bench/run_local.py --which <模組>`。
   llama.cpp 直接用 **≥ b11371**（v11 驗證 26B 換到這版 check-lock 0 失敗），之後 Clef 也用同一個 build。
2. **選配：Clef 27B 實測**：`docs/handoff-gb10-clef.md`。主線的真實資料標好之後做最有用（§6 要用到）；沒有真實資料也可以先做 §1–§5（速度與記憶體）。
   回答「SPC（合成資料 97% 對 86%）與重排序（0.878 對 0.840）要不要改走 Clef 27B」，判定門檻在該檔 §7。
3. **選配：v12 留下的兩個 GB10 待測**（`results/17-v12.md`）：31B Q8 的延遲（JevBench 公開 91.3% 最高，但 L40S 上單題是 26B 的 3 倍）；只對 SPC、UPH 開「低信心才思考」要先重新預登記（v12 是事後看到的），L4 上每題 11–19 秒。
4. **選配：IPC 全層事故的案例推理系統**：`docs/handoff-v17-ipc-cbr.md`。提案，從 Phase 0（定分層、定卡、手填 30 張歷史卡，不用模型）開始。
5. **不要做**：訓練 LoRA、用 jevify、讓判斷跟生成共用帶 `--lora` 的 server（v11 證據見 `results/16-clef-llamacpp.md` §4）；不改 `decide/prompt.py` 的模板；不要對急迫度開思考（v12 E4 變差）；不要用 EmbeddingGemma 2 當選擇器，它只能當前 20 名的粗篩（v13–v16）；判斷 server 不開 MTP 草稿頭、也不跟開了 MTP 的生成 server 共用（v18）。

要先讀的背景：本檔 §1–§3、`results/REPORT.md` §0／§4／§5、`results/16-clef-llamacpp.md` 的「一句話」。

## 1. 一句話

值得做、用現有的 Gemma 4 26B-A4B 自己做、不採購 Jev。十類題七類零標註 ≥ 98%；弱的三類靠改寫判斷標準與幾十筆校準；一次判斷 L4 上 0.2 秒（共用現場狀況 0.07 秒）。推理引擎 llama-server 與 SGLang 都可用，SGLang 批次快 5 倍，前提是 prompt 帶 `<bos>`。十八輪（v17 是提案未跑）實驗約 19.6 GPU 小時、約 $32.5。v18：Gemma 4 MTP 草稿頭讓生成快 1.7 倍，但判斷 server 開了之後第一個 token 的機率重跑會變（2,000 題翻 7 題、最大差 0.93），判斷 server 不開、生成 server 另開。v13–v16 試了 Google 自家 Decision Maker 用的 EmbeddingGemma 2：零樣本是召回約 0.95 的粗篩，不是選擇器（十類 0.637、100 份 SOP 選一份 0.527）；100 選 1 維持 26B 完整淘汰賽 0.947。在 JevBench 公開子集自跑 88.7%，高於同派的 Cygnet 與訓練過的 Open-Jev。Cloudflare 開源的 Clef 27B 在我們的題上更準（96.6% 對 95.5%）；llama.cpp 已原生支援、Q8 可用，但速度沒贏，判斷層先不換（v10、v11）。v12 借了 Gemma 4 這條線上別人的四個零訓練做法，都沒有改變預設；Gemma 4 31B 在英文 JevBench 公開題最高（91.3%），但慢 3 倍。

## 2. 走到哪裡了（十八輪，每輪先寫門檻再跑）

| 輪 | 交接文件 | 問的問題 | 答案 | 結果檔 |
|---|---|---|---|---|
| v1/v2 | `docs/handoff-v1.md`、`v2.md` | 讀選項機率當決策 API，準不準、快不快、要標多少？ | 七類零標註 ≥ 0.98，L1 206 ms（`--swa-full` 137），多數題零筆標註；raw 信心無鑑別力，門檻要校準 | `results/00`–`04`、`REPORT.md` |
| v2 補 | — | 弱題是不是標準沒寫清楚？ | 把「連續幾點、超限幾次」寫成可數規則：SPC 0.815 → 0.95、急迫度 0.81 → 0.885；held-out 種子等幅 | `05-criteria-v2.md` |
| v3 | `handoff-v3.md` | 題目太簡單，還是模型真的強？ | E2B/E4B/26B 階梯：四類 E2B 就夠（H1）、六類 26B 真的強（H2）；E4B 先答、34% 再問 26B，acc 不變、26B 負載剩三分之一 | `06-ladder*.md`、`cascade.json` |
| v4 | `handoff-v4-sglang.md` | 換 SGLang 值不值？ | 速度門檻 G1–G4 全過（K=16 3.7×、c=32 6×），但 acc 低 2.8 點、重跑不穩 → 視同沒過 | `07-*.md` |
| v5 | `handoff-v5-typellm.md` | TypeLLM 的技巧有用嗎？ | JSON prefill +1.3 點、順序平均 −1.3 點，都沒過；量到弱題順序敏感度 21% / 19%；加了標籤自檢與驗證腳本 | `08-typellm-followups.md`、`verify.md` |
| v6 | `handoff-v6-sglang-stability.md` | SGLang 掉分的真正原因？ | 差在第 0 個 token：llama-server 加 `<bos>`、SGLang 不加。餵同樣 id 後 0.958 vs 0.959；補 `<bos>` + 批次不變旗標重跑一致率 99.85% → SGLang 回到候選 | `09-sglang-stability.md` |
| v8 | `handoff-v8-jevbench.md` | 同一把尺：JevBench 公開 231 題自跑（不排名） | 26B **88.7%**（hard 77.5%），高於 Cygnet 87.9、Open-Jev 85.3、TypeLLM 84.4；E4B 78.8%；合成資料的溫度搬過去 ECE 0.093 → 0.044 | `13-jevbench.md`、`data/jevbench/` |
| v7 | `handoff-v7-borrowed.md` | 三十幾個開源替代品有什麼可抄？ | 錯誤預算門檻 + lock 檔（採用；方法 v9 修正，`--check-lock` 抓到換錯引擎 7 項）、option mass 模板檢查（採用）、packed readout（不採用：後面的題翻 11–13%） | `11-*.md`、`12-packed.md`、`thresholds.lock.json` |
| v9 | `handoff-v9-nine-places.md` | ByteByteGo 九格裡沒測過的四格 + 三段式門檻 | 護欄（26B 攔 96%，誤擋 5%）、LLM 評分（99.7%）、三段式門檻（11/12）成立；工具守門 90%（數字與範圍交給規則）、重排序 nDCG@5 0.84（差 0.01）沒過；修正 v7 門檻方法 | `14-nine-places.md`、`data/v9/` |
| v10 | `handoff-v10-clef.md` | Cloudflare 開源的 Clef（訓練判斷頭）能不能取代我們？ | Clef 27B D0 96.6%（SPC +11 點）、重排序 0.879 過、三題一起不掉、護欄外部題 98%；flash 93.0%。但判斷頭接不上推論引擎、量化版不等價（FP8 NaN），只能 transformers BF16，單題 61–87 ms 不比我們快 → 不換，27B 留作弱題與重排序候選 | `15-clef.md`、`modal_clef.py` |
| v11 | `handoff-v11-clef-llamacpp.md` | llama.cpp 原生支援 Clef 之後能不能換？同底模的 jevify LoRA 有沒有用？ | llama-server 版與原版等價、Q8／Q4 可用；同一張 L40S 上 flash Q8 48 ms 對 26B 56 ms、27B Q8 142 ms，速度沒贏 → 不換，27B Q8 可上 GB10。jevify D0 92.5% 比零訓練差，不採用；`--lora` 混用會拖慢判斷 3 倍。26B 換 b11371 不漂 | `16-clef-llamacpp.md`、`modal_clef_gguf.py` |
| v12 | `handoff-v12-borrowed-gemma.md` | Gemma 4 這條線的新做法（Cygnet、decisio、Rune）零訓練能借的四件，有用嗎？ | 四件都沒改變預設：分數題改讀機率加權等級 +1.1 點（持平）；每題型溫度在十類內輸給每類溫度，但轉到 JevBench ECE 0.010；12B 當級聯第一階過準確率門檻，但 L4 上比 26B 慢（168 對 137 ms）；31B 弱題 +2 點未達 +3、JevBench 公開 91.3%、慢 3 倍；低信心才思考 +2 點但自動處理只多 2 點（SPC、UPH 有效，急迫度有害） | `17-v12*.md`、`v12.json` |
| v13 | `handoff-v13-embeddinggemma2.md` | Google MediaPipe Decision Maker 的 EmbeddingGemma 2 後端（bi-encoder、零樣本）在 D0 到什麼程度？ | 十類平均 0.637，低於 E2B 0.807、E4B 0.885、26B 0.955；主題型三類 0.714，三條門檻全沒過。答案相反的孿生題 45% 給同一個答案（26B 6%）。本機 CPU 免費 | `18-embeddinggemma2.md`、`v13.json` |
| v14 | `handoff-v14-sop-routing.md` | 給 EmbeddingGemma 2 它的主場（100 份 SOP 路由，Gemini 盲寫），對 E4B、26B 誰贏？ | 主場也輸：EG top-1 0.527、E4B 淘汰賽 0.907、26B 0.947；EG 前 10 名給 26B 讀一次 0.860（輸在 EG 召回 0.887）。EG 前 20 名召回 0.953，可當寬鬆的前置 | `19-sop-routing.md`、`v14.json`、`data/v14/` |
| v15 | `handoff-v15-shortlist-tournament.md` | EG 先縮到前 20／30／50 名、26B 打小型淘汰賽，能不能追平完整淘汰賽？ | 追不平：前 20／30／50 名 0.893／0.900／0.893，完整淘汰賽 0.947。召回補回來了，但候選越寬、進了名單後答對越低（前 50 名 p ≈ 0.008）。單則延遲：前 20 名 0.72 秒、完整淘汰賽 2.7 秒。100 選 1 仍用完整淘汰賽 | `20-shortlist-tournament.md`、`v15.json` |
| v16 | `handoff-v16-dealt-groups.md` | 把長得像的對手打散到不同組，小型淘汰賽會不會變準？ | 不會：前 20 名四種排法 0.873–0.893，全部 100 份打散 0.927（照目錄 0.947）。打散只把近親搬進決賽。錯誤跟著訊息走、不跟著分組走；剩下的槓桿是召回或每次讀取的判斷力 | `21-dealt-groups.md`、`v16.json` |
| v17（提案，未跑） | `handoff-v17-ipc-cbr.md` | 一群 Ubuntu + microk8s 的 IPC 全層進 LGTM，出事時判層、找舊案、套 SOP、掉人工再存回，EG2／E4B／26B 怎麼分工？ | 案例推理：卡生成器 → EG2 卡對卡檢索與 novelty → E4B 先答 → 26B 四道選擇題；Keep 開源版 + Tempo service graph，不訓練。P0–P4 門檻已寫死 | — |
| v18 | `handoff-v18-mtp.md` | Gemma 4 的 MTP 草稿頭（llama-server `--spec-type draft-mtp`）開了，判斷會不會變、生成快多少？ | 生成 1.72×、思考模式 0.73×（接受率 58%）；單題不變慢；但判斷等價沒過：2,000 題翻 7 題、最大機率差 0.93，而且 MTP 臂自己重跑也差到 0.84；背景生成下判斷反而更慢（1.33× → 1.60×）。判斷 server 不開，生成 server 另開 | `22-mtp.md`、`v18.json` |

## 3. 效果多好（D0 test 每 task 100 筆，L4 / L40S）

| 指標 | 數字 | 註 |
|---|---|---|
| JevBench 公開 231 題（self-run） | 88.7%，hard 77.5% | Cygnet 87.9、Open-Jev-27B 85.3、TypeLLM 84.4（各自自跑）；不是官方排名。v12：同讀法 Gemma 4 31B 91.3%、12B 86.1% |
| 零標註 ≥ 0.98 的 task | 7 / 10 | `m_alarm_category`、`m_needs_dispatch`、`q_defect_root`、`p_line_change`、`x_ticket_route`、`x_escalate`、`x_10way_intent` |
| 三類弱題（改寫標準後） | 急迫度 0.885、SPC 0.95、UPH 0.93 | 錯集中在相鄰等級；UPH 該用公式 |
| 5% 錯誤預算下強題 coverage | 0.98–1.00，實際錯 0–2% | 選擇性風險控制，10/10 類守住；急迫度做不到就不自動（`11-thresholds.md`，v9 修正 v7 的方法） |
| 單題 p50 | L4 137 ms（`--swa-full`）、L40S 61 ms | 生成 JSON 對照 496 ms |
| 共用 state 問 10 / 16 題 | L4 每題 73 ms；L40S llama 813 ms vs SGLang 217 ms（K=16） | SGLang 要帶 `<bos>` |
| E4B 級聯（門檻 0.999） | acc 0.952 vs 全 26B 0.955，34% 送 26B | 平均延遲 −20% |
| 順序敏感度 | 急迫度 21%、SPC 19%、其餘 ≤ 3% | 上線選項順序固定 |
| 累計費用 | ≈ $31.5、18.5 GPU h | 原估 $5.5–7.5 只算前兩輪；明細 `results/cost.md` |

## 4. Quickstart（Modal；GB10 本機見 `docs/handoff-gb10.md` 與 `docs/handoff-gb10-clef.md`）

```bash
pip install 'modal[api-proxy-support]' numpy scikit-learn matplotlib
# llama-server（L4）
scripts/modal.sh run modal_app.py --which download && scripts/modal.sh run modal_app.py --which smoke
scripts/modal.sh run --detach modal_app.py --which accuracy --server-extra "--swa-full"
scripts/modal.sh volume get gb10-decide-results / results/modal/ && python3 bench/analyze.py
# SGLang（L40S，FP8；靜態模板已含 <bos>）
scripts/modal.sh run modal_sglang.py --which download && scripts/modal.sh run modal_sglang.py --which smoke_sglang
scripts/modal.sh run --detach modal_sglang.py --which accuracy --args "--out-sub D0 --control-n 0"
# 本機 server（GB10）：任何 bench 模組都可不經 Modal 跑
python3 bench/run_local.py --which accuracy --base-url http://127.0.0.1:8080 --out-dir results/gb10/accuracy --args "--control-n 0"
# 離線：重算所有結果檔 / conformal 門檻與 lock 檔 / 換模型或後端後的漂移檢查
python3 bench/verify.py && python3 bench/thresholds.py && python3 bench/verify.py --check-lock results/modal/accuracy/q8/D0
```

`scripts/modal.sh` 把本環境的 `modal` / `modal_secret` 環境變數映射成 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`；`GB10_GPU` 選卡（L4 / L40S / H100）。token 只走環境變數，不進 repo。

### 結構

- `PLAN.md`；`docs/` 交接文件（每份先寫門檻，跑完在 §結論填三選一）
- `modal_clef_gguf.py` llama.cpp b11371 編譯與 Clef GGUF（L4／L40S／H100）
- `modal_v12.py` v12：Gemma 4 12B（L4）／31B（L40S，ctx 8192）階梯與 26B 思考尾段，用 b11371 binary；`--slots` 換 slot 數（長題要更長的 slot context）
- `modal_clef.py` Clef／Clef-flash 入口（釘版下載、L40S／H100）；`decide/clef_adapter.py` 題目轉 Clef schema
- `modal_app.py` llama-server 入口（`MODELS`：26b UD-Q4_K_M / q8 / bf16 / e4b / e2b）；`modal_sglang.py` SGLang 入口（fp8 / bf16，L40S 需自帶 fused-MoE Triton 設定檔）
- `decide/` `prompt.py`（模板：`/apply-template` 學來，或 SGLang 用的 `GEMMA4_TEMPLATE_NOTHINK_BOS`；T1 變體）、`client.py`（llama `/completion`；SGLang `/generate` 指定 token id / `input_ids`；每次回傳 `option_mass`）、`labels.py`（字母單 token 自檢）
- `data/` `seeds/`、`gen/`、`hard/`、`rules/`、`synthetic/`（D0 + MANIFEST + SPOTCHECK）、`heldout/`、`perturbed/`（D1）、`blind/`（Gemini 盲寫 D2）、`tokenized/`（llama-server 切好的 token id）
- `bench/` `run_local.py`（GB10 本機跑）、`clef_under_load.py`（GB10：生成負載下的 Clef 延遲）、`analyze_gb10_clef.py`（26B 對 Clef，合成或真實資料）、`rerank.py`、`smoke*.py`、`latency.py`、`accuracy.py`、`permute.py`、`packed.py`、`v4bench.py`、`jevbench.py`、`tokenize_dump.py`、`option_mass.py`、`thresholds.py`、`analyze*.py`（v2 / ladder / v4–v10）、`clef_run.py`（Clef，transformers）、`systemone_bench.py`（llama-server `/v1/systemone` 與 jevify）、`lora_mix.py`、`verify.py`（含 `--check-lock`）、`render_html_report.py`
- `data/v9/` 護欄、工具守門、評分、重排序四份資料（產生器、抽查、Gemini 盲寫外部題）
- `data/jevbench/` JevBench 公開 231 題（MIT，釘版）；`third_party/jevbench/` 評分 harness（MIT）
- `bench/embed_d0.py`、`bench/analyze_v13.py`（v13：EmbeddingGemma 2 零樣本，CPU）
- `modal_v14.py`、`bench/embed_route.py`、`bench/route_bench.py`（淘汰賽／混合式／v15 前 K 名小型淘汰賽）、`bench/analyze_v14.py`、`bench/analyze_v15.py`、`bench/analyze_v16.py`（v16 打散分組，`--mode deal`）、`modal_v18.py`、`bench/analyze_v18.py`（v18 MTP 開關）；`data/v14/`（Gemini 寫的 100 份 SOP 與 300 則訊息）
- `bench/analyze_v12.py`（E1/E2 離線）、`bench/analyze_v12_gpu.py`（E3/E4）、`bench/think_tail.py`（低信心才思考的兩段讀法）
- `results/` 分項報告 `00`–`22`、`REPORT.md`、`cost.md`、`thresholds.lock.json`、`fig/`、`modal/`（原始 logprobs）

### 踩過的坑（接 GB10 時先看）

- **`<bos>`**：llama-server 對字串 prompt 會加，HF tokenizer（SGLang、vLLM、TypeLLM）對 Gemma 4 預設不加。少這一個 token 十類平均掉 2–3 分（v6）。smoke 會比對兩邊 prompt token 數。
- **thinking**：`/apply-template` 帶 `chat_template_kwargs: {"enable_thinking": false}`，模板結尾是空的 `<|channel>thought\n<channel|>`。TypeLLM 因此不支援 Gemma 4。
- **前綴 cache**：llama-server 開 `--swa-full`，同一 slot 順序送；SGLang 先暖一題再整批。多題排成一列一個前向（packed）在 Gemma 4 上會讓後面的題答案變，不用。
- **選項順序**：弱題兩成會因順序改答案，上線順序固定、校準用同一順序。
- **信心**：raw 信心平均 0.99，門檻用校準後信心；`thresholds.lock.json` + `--check-lock` 當 CI。
- **MTP 草稿頭**（`--spec-type draft-mtp`，b11371 已支援）：生成快 1.7 倍，但判斷 server 開了之後字母機率重跑會變（v18）；判斷與生成分兩個 server。
- **`--reasoning-budget 0`**：會強制關掉思考；要讓模型先想再答（v12 E4）就拿掉，字母讀法走 `/completion` 自帶模板，不受影響。
- **slot 的 context = `-c` ÷ `-np`**：JevBench 有題目約 3.9k token，2048 的 slot 會直接回 400；31B Q8 加 `--swa-full` 在 48 GB 卡只能開 8192（v12）。
- **SGLang 重跑**：帶 `<bos>` 99.7%，加 `--enable-deterministic-inference` 99.85%（單題慢三成）；`gemma4-mtp` 映像的 `/v1/tokenize` 會崩，自檢改走 HF tokenizer。

## 5. 參考：別人怎麼做、我們借鏡了什麼

Jev（TypeSafe，2026-09-15）發表後兩週，開源替代品三十幾個，分四類（詳 `results/10-landscape.md`，含來源連結）：

| 做法 | 代表 | 自報成績 | 對我們 |
|---|---|---|---|
| 零樣本讀選項機率（我們這一派） | **Cygnet**（凍結 Gemma-4-12B，vLLM，一個溫度）、open-alternative-jev（packed readout）、SemIf、openjev-sglang、verdict、gemma-jev、decisio（凍結 Gemma 4 31B） | Cygnet JevBench v1.5.4 官方第 1（73.70，舊版第 4 的 61.8 不同尺）、公開題 87.9% | 同一條路在公開評測站得住；Gemma 4 只有 Cygnet 和我們 |
| LoRA + 決策頭 | **decider-4b**（8k 筆，榜首 64.1）、JevK5（大模型開思考蒸餾）、Open-Jev（148k 筆、反事實資料）、Kev、imajev、**Clef**（Cloudflare，joint schema head） | 4B 訓練後贏 Jev（63.3）；Clef 27B 我們自跑公開題 87.4%、D0 96.6% | 真實工單到手後的下一步：26B 開思考出題標答 → E4B LoRA；Clef 27B 是現成的候選，但只能 transformers BF16 |
| 小型非自迴歸 | Laya、von、poorjev 的 NLI | CPU 可跑 | 我們量到 Laya 零樣本接近亂猜 |
| 校準與門檻工具 | poorjev（conformal）、jevcal（lock 檔 + CI）、jevkit | ECE 0.170 → 0.071 | 已抄成 `bench/thresholds.py` + `verify.py --check-lock` |

**借鏡了什麼，結果如何**：留下的四件是 conformal 門檻 + lock 檔（v7 P3）、option mass 模板檢查（P2）、標籤自檢與驗證腳本（v5 T0）、指定 token 讀機率（T4）；不留的三件是 JSON prefill（+1 點）、順序平均（−1 點）、packed readout（後面的題翻 11–13%）。

**ByteByteGo「九個使用位置」逐格對照**（`results/10-landscape.md` §8，含原圖；v9 補測後九格全測）：成立七格，工具守門與重排序沒過但原因明確。要改兩處：信心門檻 0.9 / 0.5 不能照抄，要用資料反推並鎖檔；重排序要逐段讀、不要打包。

補測結果：`results/14-nine-places.md`（計劃 `docs/handoff-v9-nine-places.md`）。

**Cloudflare Clef 同尺對照**（v10，`results/15-clef.md`）：訓練判斷頭的價值是真的，集中在我們最弱的地方（SPC、外部寫法、同 state 多題、重排序）；但 Cloudflare 宣稱的速度只在他們自己的 serving 上，本機只能 transformers BF16、沒有推論引擎與可用的量化版。工具守門 Clef 更差（危險放行 2.7%）。

**2026-10-06 補調查**（`results/10-landscape.md` §9）：官方 JevBench 前排出現三個 Gemma 4 系統（Cygnet、Winnow-12B、Jev-Omni，都是 12B），Rune 在同一顆 26B-A4B 上做了「低信心才思考」，decisio 的凍結 31B 比凍結 12B 高 8 分。零訓練能借的四件在 v12 量完，都沒改變預設（`results/17-v12.md`）。

**我們比別人多做的**：Gemma 4 的正確模板（空 thought channel、`<bos>`），沒有任何專案寫到；題目簡單還是模型強的階梯；兩個後端同題對照；每輪預先登記門檻、held-out 驗證、獨立抽查。
