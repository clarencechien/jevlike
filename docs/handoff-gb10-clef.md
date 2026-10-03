# Handoff GB10-Clef — 在 GB10 上實測 Clef 27B，決定 SPC／重排序要不要改走它（2026-10-03，跑前寫死）

給在 GB10 上跑的 Claude Code。這份是 `docs/handoff-gb10.md` 的**選配後續**：主線（26B 重現、重量、真實資料校準、結案）照那份做；
這份只回答一個問題：**Clef 27B 在 GB10 上夠不夠快、真實資料上有沒有比 26B 好，值不值得把 SPC 與重排序交給它。**

先讀（約 15 分鐘）：
1. `README.md` §1–2（十一輪走到哪裡）。
2. `results/16-clef-llamacpp.md`（v11：llama-server 跑 Clef 的結果、量化、速度、§0 讀程式碼的發現）。
3. `results/15-clef.md` 的「一句話」與 §2、§7（v10：Clef 在合成資料上強在哪、弱在哪）。

## 已知的事（Modal 上量的，合成資料）

| | 26B（現行） | Clef 27B Q8_0 |
|---|---|---|
| D0 平均／SPC／急迫度 | 95.5% / 86% / 82% | 96.8% / **97%**（p = 0.007）/ 81% |
| 重排序 nDCG@5（門檻 0.85） | 0.840 | **0.878** |
| 同 state 三題一起 | v7 packed −14 點，不能用 | 不掉準確率 |
| 工具守門 | 90%，危險放行 0 | 85%，危險放行 2.7% → **不交給 Clef** |
| L40S 單題 p50（同一個 llama.cpp b11371） | 56 ms | 142 ms（**2.5 倍**） |
| 記憶體 | UD-Q4_K_M 17 GB | Q8_0 28.7 GB |

v11 的判定是「速度沒過、預設不換」。GB10 的算力與頻寬都比 L40S 低，27B 是 dense，預期只會更慢；但 GB10 上 26B 也會變慢，**比例才是要量的東西**。

## 0. 規則（不能違反，同 `handoff-gb10.md` §0）

1. **真實產線資料不離開 GB10**：不上雲、不進 git。`data/real/`、`results/gb10-real/` 已在 `.gitignore`，動手前 `git check-ignore -v data/real/x results/gb10-real/x` 確認。真實資料的逐列輸出一律寫到 `/secure/...`（或 repo 裡已被 ignore 的 `results/gb10-real/`）；下文的 `/secure/gb10-real/...` 換成哪一個都可以，跟 `handoff-gb10.md` §5 用同一個。
2. **門檻先寫死再跑**（§8 已寫好），沒過照實寫。
3. **不訓練、不改 prompt**：Clef 用 llama-server 內建的 systemone 模板，題目內容照 `data/seeds/tasks_v2.json`（與 26B 同一份標準）。
4. **不要用 LoRA 混用**（v11：判斷延遲被生成拖到 3 倍以上），也不要用 jevify（v11：比零訓練差）。
5. 每個階段結束 commit + push，分支 `gb10/clef-<日期>`；結果放 `results/gb10/clef/`（只放合成資料的逐列輸出與所有彙總數字）。

## 1. 環境（Phase 0，半天）

| 項目 | 要求 | 怎麼確認 |
|---|---|---|
| llama.cpp | **≥ `b11371`**（= commit `99b95488cac0f00ce3f05af113a8c1e287753f87`，Clef 支援從這裡開始）。26B 與 Clef 用**同一個 build**（v11 驗證過 26B 換到 b11371 的 check-lock 0 失敗） | `llama-server --version`；啟動 Clef 時 log 要出現 `decision model type: clef` |
| 編法 | 有 Docker：`ghcr.io/ggml-org/llama.cpp:server-cuda` 的 **arm64** 映像，tag 的 `org.opencontainers.image.version` 要 ≥ b11371（2026-10-03 時還是 b11347，不夠）。否則自己編：`git clone --branch b11371 https://github.com/ggml-org/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build --target llama-server -j` | `nvidia-smi --query-gpu=compute_cap --format=csv` 應為 12.1；`nvcc --version` 要支援 sm_121。在有 GPU 驅動的機器上編，不需要 v11 的 libcuda stub 處理 |
| Clef 權重 | `ggml-org/Clef-GGUF` revision **`5f70656b6670c65eb85ad07a11efe211b5f211bd`**：`Clef-Q8_0.gguf`（28.73 GB，主力）。選配：`Clef-BF16.gguf`（54.06 GB，等價檢查用） | `huggingface-cli download ggml-org/Clef-GGUF Clef-Q8_0.gguf --revision 5f70656b... --local-dir /models/clef-gguf`；sha256 記進 `results/gb10/clef/00-env.md` |
| flash（選配） | `ggml-org/Clef-Flash-GGUF` revision `4a7a08c09bc63baf043b62b5ba89dd67a0357d95`：`Clef-Flash-Q8_0.gguf`（9.66 GB） | 同上 |
| 26B | 跟 `handoff-gb10.md` §1 同一個檔（`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`），用新 build 起 | |
| Python | `numpy scikit-learn requests` | `python3 -c "import numpy, sklearn, requests"` |

## 2. 起兩個 server

```bash
# 26B（生成 + 現行判斷），參數同 handoff-gb10.md §1
llama-server -m /models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -ngl 99 --port 8080 --jinja --reasoning-budget 0 --swa-full -np 4 -c 65536 --metrics
# Clef 27B（只提供 /v1/systemone；整個 prompt 要放進一個 ubatch；判斷層一次只算一個請求）
llama-server -m /models/clef-gguf/Clef-Q8_0.gguf -ngl 99 --port 8090 -c 16384 -ub 4096 -b 4096 -np 2 --metrics
```
兩個都起來後記下：`nvidia-smi` 的記憶體、`free -g`（GB10 是統一記憶體，兩邊加起來看）、兩個 `/props` 的 `build_info`，寫進 `00-env.md`。

## 3. Smoke（Phase 1，5 分鐘）— 證明「同權重、同結果」

```bash
python3 bench/run_local.py --which systemone_bench --base-url http://127.0.0.1:8090 --out-dir results/gb10/clef/q8 --args "--backend systemone --suites smoke"
python3 - <<'EOF'
import sys; sys.path.insert(0, "bench"); import analyze_v10 as A
print(A.equiv(A.jl("results/modal/v11/clef/clef-q8/smoke.jsonl"), A.jl("results/gb10/clef/q8/smoke.jsonl")))
EOF
```
必須：答案 **≥ 49/50** 與 Modal 的 Q8 一致，最大機率差 ≤ 0.02（同量化、不同硬體）。沒過就停，先查 build 版本、`-ub`、模型檔 sha256。
選配：BF16 也跑一次，對 `results/modal/clef/clef-bf16/smoke.jsonl`（v10 transformers 參考值）最大差 ≤ 0.01。

## 4. 合成資料重現（Phase 2，約 30 分鐘）

```bash
python3 bench/run_local.py --which systemone_bench --base-url http://127.0.0.1:8090 --out-dir results/gb10/clef/q8 \
  --args "--backend systemone --suites d0,v9,rerank,packed"
python3 bench/analyze_gb10_clef.py --ours results/modal/accuracy --clef results/gb10/clef/q8 --out results/gb10/clef/d0-compare.md
```
預期與 v11（`results/16-clef-llamacpp-tables.md` 的「27B Q8_0」欄）一致：D0 test 平均 96.8% ± 0.5 點、SPC 97%、重排序 nDCG@5 0.878 ± 0.01。差更多就是環境問題，回 §3。

## 5. 延遲（Phase 3，約 1 小時）— 這一輪的關鍵

同一個 build、同一台 GB10，三組都量：

```bash
# a) Clef 單機（26B 閒置）：單題、同 state 三題、併發 1/4/8 吞吐
python3 bench/run_local.py --which systemone_bench --base-url http://127.0.0.1:8090 --out-dir results/gb10/clef/q8 --args "--backend systemone --suites latency"
# b) 26B 同一套（L1 單題、L4 共用 state、L5 併發），對照組
python3 bench/run_local.py --which latency --base-url http://127.0.0.1:8080 --out-dir results/gb10/clef/latency-26b --args "--only L1,L4,L5 --n 200 --out latency_26b.json"
# c) Clef 在 26B 生成負載下（0／1／2／4 條生成串流同時跑）
python3 bench/clef_under_load.py --clef-url http://127.0.0.1:8090 --gen-url http://127.0.0.1:8080 --streams 0,1,2,4 --n 100 --out results/gb10/clef/under_load.json
```
(c) 同時記 26B 生成的 tokens/s：單獨跑（`gen_tok_per_s_per_stream_alone`）與 Clef 同時跑（`gen_tok_per_s_per_stream`）。

## 6. 真實資料（Phase 4，只有 `handoff-gb10.md` §5 的真實資料集已經標好才做）

```bash
# 26B：handoff-gb10.md §5 已經跑過，輸出在 /secure/gb10-real/accuracy
# Clef：同一份 data/real、同一份標準
python3 bench/run_local.py --which systemone_bench --base-url http://127.0.0.1:8090 --out-dir /secure/gb10-real/clef-q8 \
  --args "--backend systemone --suites rows --rows-dir data/real --tasks q_spc_action,m_alarm_severity --criteria data/seeds/tasks_v2.json"
python3 bench/analyze_gb10_clef.py --ours /secure/gb10-real/accuracy --clef /secure/gb10-real/clef-q8 \
  --tasks q_spc_action,m_alarm_severity --out results/gb10/clef/real-compare.md
```
`real-compare.md` 只有彙總數字，可以進 repo；`/secure/...` 的逐列檔不可以。
重排序的真實資料版（真實 SOP 段落 + 真實查詢）**不在本輪範圍**：`bench/clef_run.py` 的 rerank suite 只讀 `data/v9/`，要做得另外開一輪、先寫資料格式與門檻。

## 7. 判定（pre-registered）

- **G1 環境**：§3 smoke 過（答案 ≥ 49/50、最大差 ≤ 0.02）。不過 → 停，回報。
- **G2 重現**：§4 D0 平均與 v11 差 ≤ 0.5 點、SPC ≥ 95%、重排序 ≥ 0.86。不過 → 停，回報。
- **G3 單機速度**：GB10 上 Clef 27B 單題 p50 ≤ 26B 單題 p50 的 **2 倍**（L40S 是 2.5 倍）。
- **G4 負載下**：2 條生成串流時，Clef 單題 **p95 ≤ 1,000 ms**，且 26B 生成每串流 tokens/s 掉 ≤ 30%（對它單獨跑時）。
- **G5 記憶體**：兩個 server 加上 26B 的 `-c 65536 -np 4` 都常駐時，系統可用記憶體 ≥ 10%。
- **G6 真實資料（逐 task）**：test 半 ≥ 100 筆，且 Clef 比 26B 高 ≥ 3 點、McNemar p < 0.05；或 ε=5% 下自動處理比例高 ≥ 10 點、且自動段錯誤率 ≤ 5%。

**結論規則**：
- G1–G5 全過、G6 對某 task 成立 → 該 task 改走 Clef 27B（只考慮 SPC、重排序；急迫度在合成資料上 ε=5% 守不住，工具守門 Clef 更差，都不換）。
- G1–G5 全過、還沒有真實資料 → 寫「可行，待真實資料」，不上線切換。
- G3、G4 或 G5 任一沒過 → 維持 26B，Clef 不部署；寫明差多少（例如「慢 3.1 倍」），留給之後更快的 build 或更小的量化再試。
- 選配：flash Q8 也照 §3–§5 跑一次。它在 L40S 上比 26B 快 15% 但準確率略輸，只在「同 state 問多題」的場景有意義（三題 84 ms 對 26B 約 150 ms）。

## 8. 交付

1. `results/gb10/clef/00-env.md`：build、模型檔 sha256、兩個 server 參數、記憶體。
2. `results/gb10/17-clef-gb10.md`：§3–§6 的數字與 §7 判定，格式學 `results/16-clef-llamacpp.md`（一句話 → 表 → 判定 → 對上線的意思 → 偏離計劃之處）。
3. 本檔最後的「一句話結論」填好（三選一）。
4. `README.md` §2 的輪次表加一列「GB10-Clef」；`results/REPORT.md` 加 §3k。
5. 若 G6 讓某 task 改走 Clef：在 `handoff-gb10.md` §6 的結案清單寫明路由（哪個 task 打哪個 port），並把 Clef 的門檻（用 Clef 自己在真實資料上的分數，`bench/thresholds.py` 的 `risk_threshold`）一起鎖檔。**Q8 對 BF16 只是「可用」不是「等價」，門檻一定要用 Q8 自己的分數反推。**

## 9. 會踩到的坑（v10／v11 踩過的）

- **build 太舊**：< b11371 的 llama-server 不認得 `clef` 架構，會載入失敗或當一般模型。看 log 的 `decision model type: clef`。
- **ubatch**：整個 prompt 要放進一個 ubatch（`-ub 4096 -b 4096`）；合成資料最長的是 v9 評分題約 670 token（D0 最長 455）。真實資料若有長工單，先量最長的 token 數，超過就調大 `-ub`／`-b`，否則直接報錯。
- **一次一個請求**：Clef 判斷層讀整個 batch，多 slot 不會同時算；吞吐不會隨併發增加（L40S：27B 約 7 題/秒）。要更多吞吐只能開第二個 server。
- **跑 Clef 的 server 只有 `/v1/systemone`**，不能生成、也不能用來跑 26B 的 `/completion`。兩個 server 分開。
- **只支援文字**，`images` 會被拒。
- **溫度**：Clef GGUF 沒存溫度，回傳的是溫度 1 的機率；校準照我們的 `bench/thresholds.py` 自己做。
- **不要在 transformers 裡量化 Clef**（v10：FP8 出 NaN、int8／NF4 機率偏 0.23）。GGUF 的 Q8／Q4 沒問題。
- **選項 ID**：llama-server 跟 Python 版一樣會把 choice 選項照 ID 排序；`decide/clef_adapter.py` 用標籤當 ID，不要改。

## 一句話結論（三選一，跑完填）

- 「GB10 上 Clef 27B 單題 X 倍、負載下 p95 Y ms、記憶體夠；真實資料上 SPC 高 Z 點：**SPC（與重排序）改走 Clef 27B**，其餘維持 26B。」
- 「GB10 上可行（G1–G5 過），但真實資料上沒有贏（或還沒有真實資料）：維持 26B，Clef 留作候選。」
- 「GB10 上太慢或記憶體不夠（G3／G4／G5 沒過，差 X）：維持 26B，Clef 不部署。」
