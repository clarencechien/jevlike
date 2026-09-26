# Handoff GB10 — 給在 GB10 上跑的 Claude Code：重現、重量、結案（2026-09-26）

你接手的是一個已經在 Modal 雲端跑完八輪的實驗 repo。所有數字都是在租用的 L4 / L40S / H100 上、用合成資料量的。
你的工作只有三件：**在 GB10 上重現一次證明環境對、把速度數字換成 GB10 的、用真實資料把門檻校準好，然後結案。**
先讀 `README.md` §1–3 和 `results/REPORT.md` §0、§4、§5，再回來這裡。

## 0. 規則（不能違反）

1. **真實產線資料不離開 GB10**。不上 Modal、不上任何雲端、不進 git（`data/real/` 已在 `.gitignore`，你要先確認）。報告裡只放彙總數字，不放原文。
2. 每一輪先寫門檻再跑，跑完照門檻填結論；沒過的也照實寫。本檔 §3–§6 的門檻已經寫死。
3. 不做的事：不訓練、不改 prompt 模板（`decide/prompt.py` 的 `SYSTEM`、`GEMMA4_TEMPLATE_NOTHINK_BOS`、`ANSWER_LINE` 原樣）、不用真實資料調任何會回頭影響準確率的東西，只用它校準門檻。
4. 每個階段結束 commit + push（分支自己開，`gb10/<日期>`），結果放 `results/gb10/`，原始 logprobs 一起進 repo（合成資料的可以；真實資料的**不可以**，只留 `_summary.json` 與門檻 lock）。

## 1. 環境（Phase 0，半天）

| 項目 | 要求 | 怎麼確認 |
|---|---|---|
| llama.cpp | llama-server，CUDA sm121 build，版本 ≥ b11118（Modal 用的） | `llama-server --version`；`/props` 的 `build_info` 寫進 `results/gb10/00-env.md` |
| 模型檔（llama） | `unsloth/gemma-4-26B-A4B-it-GGUF`：`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`（主力，17 GB）與 `Q8_0`（27 GB）；`unsloth/gemma-4-E4B-it-GGUF` Q8_0；`unsloth/gemma-4-E2B-it-GGUF` Q8_0 | 檔名、sha256 寫進 00-env；若 GB10 現有部署用的是別的量化（Google QAT q4_0 等），**也要量**，見 §4 |
| llama-server 參數 | `--jinja --reasoning-budget 0 --swa-full -np 4 -c 65536`（26B）；GB10 記憶體夠，`-np` 可加到 8–16 給 L5 用 | smoke 會檢查模板與 `<bos>` |
| SGLang（選配） | 映像 `xomoxcc/dgx-spark-sglang` 的 `gemma4-sm121` tag；權重 `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` @ `ed35d7abe5d9`；`--context-length 4096 --max-running-requests 32`；**MTP 不開**、reasoning parser 不開 | `bench/run_local.py --which smoke_sglang` 的 `bos_check.ok` 與 `option_mass_check.ok` 都要 True |
| Python | `numpy scikit-learn matplotlib requests`；SGLang 路徑另需 `transformers`（標籤自檢與 packed 用 HF tokenizer） | |
| 字型 | `bench/analyze.py` 畫圖用 WenQuanYi Zen Hei；沒有就 `--no-fig` 或裝字型 | |

不用 Modal：所有 bench 模組都改用 `bench/run_local.py --which <模組> --base-url <server> --out-dir results/gb10/<模組> --args "<原本的 args>"`。
資料路徑自動落到 repo 的 `data/`（`bench/accuracy.py` 的 `DATA_ROOT`）。

## 2. Smoke（Phase 1，10 分鐘）

```bash
python3 bench/run_local.py --which smoke --base-url http://127.0.0.1:8080 --out-dir results/gb10/smoke
```
必須全部成立，否則停下來修環境，不要往下跑：
- `label_token_ids` 十個字母都是單 token（A=236776 … 與 `results/01-smoke.md` 一致）。
- 採用變體 `nothink+sys+noprefill` 的 5 題 first token 全是字母、missing 全空、`option_mass_check.ok` True。
- s1 的 prompt token 數 = 117（v2 同題）。差一個就是 `<bos>` 或模板問題，看 `README.md` 踩過的坑第 1、2 條。
- 5 題答案與 `results/modal/smoke.json` 的採用變體一致（s4 那題兩邊都錯是正常的）。

SGLang 若有裝，`--which smoke_sglang --backend sglang --base-url http://127.0.0.1:30000`，多看 `bos_check.ok`、`token_ids_max_abs_diff == 0`、batch 三題答 C,C,C。

## 3. 準確率重現（Phase 2，20 分鐘）— 證明「同權重、同結果」

```bash
python3 bench/run_local.py --which accuracy --out-dir results/gb10/accuracy --args "--control-n 0 --workers 4"
ACC_DIR=results/gb10/accuracy python3 bench/analyze.py          # 若 analyze.py 不吃 ACC_DIR，看它頂端註解的環境變數名
python3 bench/verify.py --check-lock results/gb10/accuracy
```
門檻（pre-registered）：
- 10 task 的 test acc 與 `results/analysis.json` 的 `raw_test.acc` 每 task 差 ≤ 1 點（同 UD-Q4_K_M 權重；llama-server 不同 build 允許個位數題翻面）。
- `--check-lock` 0 failures（Q8 權重在 Modal 上也是 0；SGLang 無 `<bos>` 是 10）。
- 過 → 環境對，往下。沒過 → 先比 `_run.json` 的 build、模型檔 sha256、`/props` 的 chat_template；最常見是 llama.cpp 版本換了模板行為。

若 GB10 部署用的不是 UD-Q4_K_M（例如 Google QAT q4_0），**兩種都跑**，分別存 `results/gb10/accuracy` 與 `results/gb10/accuracy-<量化名>`；門檻同上，另報兩者差。

## 4. 速度重量（Phase 3，1 小時）— 把 REPORT 裡所有「L4 上界」換掉

```bash
python3 bench/run_local.py --which latency --out-dir results/gb10/latency                       # L1–L7，注意 L7 是生成 JSON 對照
python3 bench/run_local.py --which v4bench --out-dir results/gb10/v4bench --args "--n 200 --warmup 20 --only L1,L2,L4,L5,L6,S2,S3 --l4-mode seq"
# SGLang（選配）
python3 bench/run_local.py --which v4bench --backend sglang --base-url http://127.0.0.1:30000 --out-dir results/gb10/v4bench/sglang --args "--n 200 --warmup 20 --only L1,L2,L4,L5,L6,S2,S3 --l4-mode warm"
```
要回答的問題與預期範圍（超出範圍不是錯，是要解釋）：
- **L1 單題 p50**：預期 60–210 ms（L40S 61、L4 137）。GB10 記憶體頻寬 273 GB/s 與 L4 同級，算力較強，落在中間最合理。
- **decode 地板**（同 prompt 重複）：預期 15–30 ms。
- **L4 共用 state K=10 / 16**：`--swa-full` 下每題預期 40–80 ms。
- **L5 併發 c=1/8/32**：這是本檔最重要的一項。L4 上 llama-server 併發不加速（throughput 卡 5.5 req/s）；GB10 上 `-np 16` 能不能加速決定要不要 SGLang。判定：**c=32 的 decisions/s ≥ 3 × c=1** → 即時層 llama-server 即可，SGLang 只給批次；否則 SGLang 進即時層候選，照 `docs/handoff-v4-sglang.md` 的 G1–G4 重跑一次判定（用 `bench/analyze_v4.py`，把路徑改成 gb10 的）。
- **L6 生成負載下的決策延遲**：GB10 上 26B 同時要寫報告，p95 退化倍數預期 1.3–1.5×；> 2× 就要考慮把判斷放到獨立的 E4B 進程（v3 級聯）。
- **E4B / E2B 也量 L1 與 L4**（`--model e4b` 只是記錄用，server 要自己換模型檔），填 `results/speed_ladder.json` 的 GB10 欄。

輸出：`results/gb10/03-latency.md`（`bench/render_latency.py` 指到 gb10 目錄）與 `07-sglang-speed.md` 的 GB10 版。

## 5. 真實資料校準（Phase 4，一到兩天，最重要）

1. 收 **200–500 筆**真實 alarm / 工單 / 訊息，依 `data/seeds/tasks.json` 的十類（或其中實際會上線的幾類）標 gold。標註規則用 `data/seeds/tasks_v2.json` 的可數標準（急迫度、SPC 那兩題）。兩人標、算 kappa，κ < 0.7 的題先回頭修標準，不要拿去校準。
2. 放 `data/real/<task>.jsonl`（欄位同 `data/synthetic`：`id, state, gold, question, difficulty, lang, pair_id`；`pair_id` 可用 id），確認 `.gitignore` 擋住。
3. ```bash
   python3 bench/run_local.py --which accuracy --out-dir /secure/gb10-real/accuracy --args "--data-dir data/real --control-n 0 --criteria data/seeds/tasks_v2.json"
   python3 bench/thresholds.py --acc-dir /secure/gb10-real/accuracy --measure auto --eps 0.02,0.05,0.10
   ```
   `thresholds.lock.json` 可以進 repo（只有門檻與筆數）；原始 logprobs 不行。
4. 門檻（pre-registered）：
   - 零標註 test acc：預期比合成資料低 5–15 點；**< 0.80 的 task 不上線**，回報是題目（標準）還是資料（標籤品質）問題。
   - conformal ε=5%：保證成立的 task ≥ 上線 task 數的 80%；強題 coverage ≥ 0.90。
   - 順序敏感度：用 `bench/permute.py`（`--perms 4` 即可）量一次，≥ 20% 的 task 上線時選項順序寫死並在 REPORT 註明。
   - 級聯：若 E4B 也跑了真實資料，用 `bench/analyze_ladder.py` 的 cascade 邏輯重算門檻；沒跑就維持「六類 26B、四類 E4B」的查表。
5. 每三個月或模型檔更換時：`python3 bench/verify.py --check-lock /secure/gb10-real/accuracy-<新>`，失敗就重校準。

## 6. 結案（Phase 5，半天）

REPORT.md 要改的地方，一條一條對：
- §0 一句話：把「L4 上界」的延遲換成 GB10 實測；加一句真實資料上的零標註 acc 與 5% 預算下的 coverage。
- §2 Q1 / Q3 / Q4：延遲表加 GB10 欄，L4 欄保留當對照。
- §2 Q6：換成真實資料的 conformal 表（`11-thresholds.md` 的格式），合成資料版移到附註。
- §3d 後端選擇：填 L5 判定與 SGLang 是否進即時層。
- §4 限制：刪掉第 1（L4 上界）、3（aarch64 未驗證）、4（校準只有 100 筆）條，補上真實資料的限制（筆數、標註者、時間範圍）。
- §5 接回 GB10：全部打勾或改成「未做，原因」。
- `results/cost.md` 加 GB10 電費／時間一列；`README.md` §2 表加「GB10」一列、§3 記分板換 GB10 數字，標題改「（已在 GB10 結案）」。
- HTML：`bench/render_html_report.py` 讀的是 `results/analysis.json` 等檔，把 GB10 的分析輸出成同名檔放 `results/`（原本的改名 `-modal`），重新 `scripts/publish_report.sh`（需要 `IMITATOR_TOKEN` 環境變數）；TL;DR 第 3 條與第 6 條改成 GB10 數字，「條件四 GB10 實測」改成已完成。
- 最後一段寫「結案判定」三選一：**上線**（七類強題 + 校準門檻）／**部分上線**（列哪幾類）／**不上線**（原因）。

## 7. 你會踩到的坑（都踩過了）

- `<bos>`：HF tokenizer 引擎（SGLang、vLLM）對 Gemma 4 預設不加，llama-server 會加；差 2–3 分。靜態模板已含，smoke 會查 token 數。
- thinking：`/apply-template` 要帶 `chat_template_kwargs: {"enable_thinking": false}`，結尾是空的 `<|channel>thought\n<channel|>`。
- `--swa-full` 一定要開，否則共用 state 的多題是 N 倍時間；64k ctx 加 swa-full 在 24 GB 卡會 OOM，GB10 128 GB 沒這問題。
- SGLang 重跑 0.15–0.3% 的題會翻面，`--enable-deterministic-inference` 可壓到 0.15%，單題慢三成。
- 多題排成一列一個前向（packed）在 Gemma 4 上會讓後面的題答案變，不要用。
- llama-server `/completion` 的 top-40 讀不到的字母機率上界 7e-7，不用管。
- `bench/analyze.py` 匯入 matplotlib，沒有字型會 warning 不會掛；`bench/v4bench.py` 的 S3 曾因為匯入 analyze 掛過，現在已拆開。

## 8. 交付清單

- [ ] `results/gb10/00-env.md`（卡、build、模型檔 sha256、參數）
- [ ] smoke 四項全過
- [ ] 合成 D0 acc 每 task ±1 點內、`--check-lock` 0 failures
- [ ] `results/gb10/03-latency.md`、v4bench L1–S3（llama；SGLang 選配）與 L5 判定
- [ ] `thresholds.lock.json`（真實資料）、`11-thresholds.md` GB10 版
- [ ] REPORT.md 依 §6 改完、README、cost、HTML 重新發佈
- [ ] 結案判定一句話
