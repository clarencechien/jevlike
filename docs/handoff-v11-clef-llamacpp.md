# Handoff v11 — Clef 走 llama-server 原生 `/v1/systemone`（釘 llama.cpp `99b9548`，2026-10-03，跑前寫死）

來源：v10（`results/15-clef.md`）的結論是「Clef 27B 比 26B 準，但只能 transformers BF16、本機不比我們快 → 先不換」。
2026-10-03 llama.cpp 合併了 PR #29831（commit `99b9548`「model: add support for clef decision model (text-only)」）：
判斷層用 C++ 重寫（`src/models/clef.cpp`），llama-server 直接提供 `/v1/systemone`，ggml-org 放了官方轉檔的 GGUF。
v10 卡住的三件事（沒有推論引擎、量化不能用、速度）這一輪都可以重量。

本輪要回答：**用 llama-server 跑 Clef，結果跟 Cloudflare 的 PyTorch 版一樣嗎？量化後還能用嗎？速度有沒有贏 26B？**
另外加測 jevify（§4b）：**同一顆 Gemma 4 26B，LoRA 訓練過的讀法比我們零訓練好多少，能不能用 `--lora` 跟生成混用？**

## 0. 版本與檔案（全部釘死）

- **llama.cpp**：`99b95488cac0f00ce3f05af113a8c1e287753f87`（master，2026-10-03 02:50 +0200）。
  **= tag `b11371`**（2026-10-03 查到，tag 正好指向這個 commit，原始碼完全相同）。
  但 `b11371` 的 release 檔案還沒上傳（各平台壓縮檔都是 404，`b11368` 的同名檔是 200），Docker `server-cuda` 仍是 `b11347`。
  而且 llama.cpp 本來就不出 Linux CUDA 的預編檔（歷來只有 Windows CUDA、Linux CPU／Vulkan／arm64），Modal 上一律從 `b11371` 原始碼編。
  GB10（arm64 + CUDA）之後可用 Docker `server-cuda` 的 arm64 映像，等它更新到 ≥ `b11371`。
- **GGUF**（ggml-org，用 `ggml-org/convert` 自動轉的）：
  - `ggml-org/Clef-Flash-GGUF` revision `4a7a08c09bc63baf043b62b5ba89dd67a0357d95`：BF16 18.16 GB、Q8_0 9.66 GB、Q4_K_M 6.49 GB。
  - `ggml-org/Clef-GGUF` revision `5f70656b6670c65eb85ad07a11efe211b5f211bd`：BF16 54.06 GB、Q8_0 28.73 GB、Q4_K_M 19.23 GB。
- **參考值**：v10 的 transformers BF16 輸出（`results/modal/clef/clef-flash-bf16/`、`results/modal/clef/clef-bf16/`），不重跑。
- **26B**：沿用 `unsloth/gemma-4-26B-A4B-it-GGUF` UD-Q4_K_M，同一個 `99b9548` build 重跑一次（§5 D）。
- 只用合成資料（D0、D2、v9、JevBench 公開題）。權重不進 repo。

## 1. 步驟 0：讀程式碼、確認行為（無 GPU，約 2 小時）

在 `99b9548` 上讀這幾個檔，結果寫進 `results/16-clef-llamacpp.md` §0：
1. `conversion/clef.py`：
   - GGUF 存了哪些 metadata（`clef.decision.type`、systemone 模板）。
   - **有沒有存溫度**。server README 說「機率會套用模型檔裡的溫度」；若溫度 ≠ 1，跟 v10 參考值比之前要先換算回來，否則等價檢查會失敗。
2. `src/models/clef.cpp`、`tools/server` 的 systemone 處理：
   - prompt 怎麼拼：要對得上 `joint_schema_model.encode_record` 的 system prompt、`STATE:`、`SCHEMA FIELDS`、每個選項 `{"option_id","description"}` 的 JSON。
   - **choice 選項有沒有照 ID 排序**（Python 版會 `sorted()`）。沒排序的話，順序敏感度要重新量（§4）。
   - `noul` 沒給 criteria 時用的預設描述，跟 Python 版是否相同。
   - 一個 request 多題時，是不是真的一次判（README 說 clef 會「jointly」）。
3. server 的限制：整個 prompt 要放得進 `--ubatch-size`；跑 clef 的 server 只提供 `/v1/systemone`；只支援文字。
4. 用 `gguf-py` 讀下載的 GGUF header，記錄 tensor 型別。Q8_0／Q4_K_M 的 `ssm_alpha`／`ssm_beta`（gated delta net 的 gate）有沒有保留高精度：v10 的 FP8 NaN 就是這裡溢位，Livesport 的配方把它們留在 F32。

## 2. 步驟 1：建環境（Modal，CPU 編譯）

- 新增 `modal_clef_gguf.py`：
  - image：`nvidia/cuda:12.8.1-devel-ubuntu22.04`，`git checkout 99b9548`，`cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89;90"`（L40S、H100），只編 `llama-server`。
    image 建置在 CPU 上，不吃 GPU 費用。
  - 下載函式（CPU 容器）把 GGUF 釘版放進 `gb10-decide-models` volume 的 `/models/clef-gguf/`。
  - server 參數：`-c 16384 -ub 4096 -b 4096 -np 4`，`--embeddings` 等由步驟 0 確認後決定；記錄 `llama-server --version`。
- 新增 `bench/clef_systemone.py`：
  - 用 `decide/clef_adapter.py` 把題目轉成 Jev 格式，POST `/v1/systemone`，再把 `probabilities` 對回我們的字母。
  - 輸出欄位跟 `bench/clef_run.py` 一樣（`id`、`task`、`gold`、`chosen`、`correct`、`probs`、`prompt_tokens`、`latency_ms`），
    讓 `bench/analyze_v10.py` 的函式可以直接用。
  - suites 同 v10：`smoke`、`d0`（含字母 ID 正反序）、`blind`、`v9`、`rerank`、`jevbench`、`packed`、`latency`。
- 已知差異：API 只回機率，不回 logit。v10 的溫度擬合需要 logit，改用 `log(p)` 當 logit（差一個常數，不影響 softmax 與溫度擬合）。

## 3. 步驟 2：等價檢查（smoke，50 題，與 v10 同一批）

拿 v10 的 transformers BF16 參考值當標準。分兩級，事先寫死：
- **等價**：每題機率最大差 ≤ 0.01，且答案一致 ≥ 49/50（同 v10）。
- **可用**（只給量化版）：答案一致 ≥ 49/50，而且完整 D0 的準確率跟 BF16 GGUF 差 ≤ 1 點、McNemar p ≥ 0.05。
  理由：Livesport 在 GB10 上量到 Q8_0 對 BF16 的平均機率差 0.0018，但我們的門檻是最大差，量化版可能過不了「等價」，卻仍然可以上線。

| arm | 要達到的等級 | 沒達到時 |
|---|---|---|
| flash BF16 GGUF | **等價** | 停下來。C++ 實作跟 Cloudflare 版不同，記錄差異題、對照 §1 讀到的 prompt 差異；不做後面的步驟 |
| flash Q8_0 | 可用 | 不用於 GB10 |
| flash Q4_K_M | 可用 | 不用於 GB10 |
| 27B BF16 GGUF（H100，只跑 smoke） | **等價** | 同 flash BF16 |
| 27B Q8_0（L40S） | 可用 | 27B 在 GB10 上只能用 BF16（55 GB） |

## 4. 步驟 3：主跑

- **flash**：BF16 GGUF 與每個達到「可用」的量化版，跑全部 suites（L40S）。
- **27B Q8_0**：達到「可用」才跑全部 suites（L40S，28.7 GB 放得下，v10 的 BF16 放不下）。
- **順序**：若步驟 0 發現 llama-server **沒有**把 choice 選項排序，在 `d0` 加一組「標籤 ID、原順序反過來送」，報翻面率；有排序就照 v10 只做字母 ID 正反。
- **三題一起**：同 v10 `packed`（alarm 家族 600 列），跟單題比。

## 4b. 加測：jevify（同一顆 Gemma 4 26B-A4B，LoRA 訓練過的讀法）

來源：`kushalpatil/jevify-gemma4-26b-a4b`。底模跟我們一樣，LoRA r=64 只改 attention，用約 4.7 萬筆（state、題目、目標分布）訓練，loss = KL；
讀法也跟我們一樣：一次 prefill、讀答案標籤的下一個 token 機率。自報 6 個 OOD 資料集 acc 0.834、ECE 0.061，訓練資料是英文公開資料集。
這一格回答的是 v10 沒能回答的一半：**同一顆底模，訓練過的 LoRA 比我們零訓練讀字母好多少？混用（同一份底模又生成又判斷）要付多少代價？**

釘版：
- 合併版 GGUF：`mradermacher/jevify-gemma4-26b-a4b-GGUF` revision `8917f8f469b91a789fe1dc5ae6835f6da5f3dd0f`，Q4_K_M 16.8 GB（與我們的 UD-Q4_K_M 同量級）、Q8_0 26.9 GB。
- 只有 LoRA：`kushalpatil/jevify-gemma4-26b-a4b-lora` revision `ec4a3d221b0989853e8a7ab0f55ef38945ff3a92`（0.18 GB）。
  用 llama.cpp 的 `convert_lora_to_gguf.py` 轉成 GGUF LoRA（CPU，需要 `google/gemma-4-26B-A4B-it` 的 config 與 tokenizer）。
- 合併版 HF 權重（只拿 `chat_template.jinja` 與讀 prompt 格式）：`kushalpatil/jevify-gemma4-26b-a4b` revision `d4c0d1d455892957d274f68ec64e9fc0881011c8`。
- jevify 程式碼：`github.com/kushalpatil07/jevify` commit `66f49bff53b3276fecc6b4bd16a31de277bebfaf`。
- 授權：Gemma license，跟我們用的底模一樣。

步驟 0 追加（無 GPU）：讀 jevify 的 prompt 組法與讀法（標籤 token 怎麼選、`noul`／`score` 怎麼對應、有沒有 `<bos>`、thinking 怎麼關），
在 `bench/jevify_client.py` 重現，用 `/completion` + `n_probs` 讀機率（跟我們的 client 同一條路）。先用 jevify 自己的 transformers 實作跑 20 題當參考，確認重現無誤（答案 20/20 一致、機率差 ≤ 0.01）。

arm（全部 L4，llama-server 用 **b11118**，跟我們 v1–v9 同一版，排除版本差異；§5 D 另外處理新版）：

| arm | 權重 | prompt | 回答的問題 |
|---|---|---|---|
| J0 | 我們的 UD-Q4_K_M（既有結果，不重跑） | 我們的（v2 模板） | 對照組 |
| J1 | jevify 合併版 Q4_K_M | jevify 的 | 照作者的用法，在我們的題上多準 |
| J2 | jevify 合併版 Q4_K_M | 我們的 | 訓練效果能不能搬到我們的 prompt |
| J3 | 我們的 UD-Q4_K_M + `--lora` jevify（每個 request 設 scale 1） | jevify 的 | 混用：底模只放一份，判斷時才套 LoRA；跟 J1 應該只差在量化（LoRA 套在量化過的底模上） |

suites：D0（2,000）、D2 盲寫、v9 G／T／E、JevBench 公開 231 題（正反序）。重排序與 packed 不做（jevify 不是為這兩種設計的）。

混用的代價另外量（J3 的 server）：
- 判斷題 scale 1、生成題 scale 0 交錯送時，判斷題的 p50 延遲、生成的 tokens/s，對照「只有判斷」與「只有生成」。
- prefix cache 命中率（scale 不同的 request 不能共用 cache，量實際損失）。
- 生成題 scale 0 時的輸出，跟沒載 LoRA 的 server 逐 token 比，確認底模行為完全沒變（50 題 greedy，期望 100% 一致）。

判定（pre-registered）：
- **J-a 訓練有價值**：J1 的 D0 平均 ≥ J0 + 1 點，或 ECE（raw，不擬溫度）≤ J0 擬溫度後的 ECE。
  兩條都沒過 → 寫明「通用英文資料訓練的 LoRA 在中文產線題上沒有比零訓練好」。
- **J-b 混用可行**：J3 與 J1 的答案一致 ≥ 98%；生成題 scale 0 與無 LoRA 完全一致；判斷題延遲 ≤ J1 的 1.2 倍。
- **J-c 速度**：預期 J1 與 J0 單題延遲相差 ≤ 10%（同架構、同一次 prefill），只是確認，不是要贏的門檻。

## 5. 判定（pre-registered）

**A 實作正確**：flash BF16 GGUF 與 27B BF16 GGUF 都達到「等價」。沒過就是結論，後面不判。

**B 速度（同一張 L40S、同一個 `99b9548` build）**：
- 單題 p50：flash Q8_0（或最好的「可用」量化版）≤ 26B 的 0.5 倍。這是 v10 判定 B 的速度條件，這次終於能在同引擎下量。
- 三題一起（alarm K=3）p50：Clef ≤ 26B 三題分開送（同 slot 依序、共用前綴）的 0.5 倍。
- 吞吐：並發 1／4／8 時每秒判斷數，Clef 對 26B 都報。
- 27B Q8_0 單題 p50 ≤ 26B 的 2 倍（`handoff-gb10.md` §6b 的 GB10 門檻，先在 L40S 上預看）。

**C 準確率不掉**：達到「可用」的量化版，D0 平均與 v10 BF16 差 ≤ 1 點；v10 過的格（G、E，27B 的 R）仍然過。

**D 26B 換到新版不漂**：26B 用 `99b9548` build 重跑 D0，`bench/verify.py --check-lock` 通過 → GB10 可以只用一個 llama.cpp 版本（26B 與 Clef 兩個 server 同版本）。沒過 → GB10 上兩個版本並存（26B 留 b11118）。

## 6. 順序、預算、輸出

1. 步驟 0：讀程式碼（無 GPU）。
2. image 編譯（CPU，約 15 分鐘）＋ GGUF 下載（CPU）：flash 三個 34 GB、27B Q8_0 29 GB；27B BF16 54 GB 只在 smoke 用。
3. jevify（L4，b11118）：下載 Q4_K_M 17 GB 與 LoRA、轉 LoRA GGUF（CPU）；J1／J2／J3 各跑 D0、D2、v9、JevBench，加混用代價量測。約 1.5 小時，≈ $1.2。
4. L40S：flash 三個 smoke → 主跑（BF16 + 可用的量化版）→ 27B Q8_0 smoke → 主跑 → 延遲（含 26B 同 build 對照）→ 26B D0 + check-lock。約 1.5 小時，≈ $3。
5. H100：27B BF16 GGUF smoke 50 題，約 15 分鐘，≈ $1。

**合計 ≈ $5，上限 $8。**

輸出：`results/16-clef-llamacpp.md`（含 §jevify）、`results/v11.json`、`modal_clef_gguf.py`、`bench/clef_systemone.py`、`bench/jevify_client.py`、`bench/analyze_v11.py`；
更新 `15-clef.md`（補一節「v11 之後」）、REPORT §3j、HTML、README、`handoff-gb10.md` §6b（若過，GB10 改用 llama-server 跑 Clef）。

## 一句話結論（三選一，跑完填）

**跑完（2026-10-03）**：Clef 選第二句的變形：實作等價（BF16 GGUF 最大機率差 0.008／0.009）、量化可用（Q8／Q4 與 BF16 差 ≤ 0.35 點），但速度沒贏（flash 單題 0.85 倍、
三題一起 0.56 倍；27B Q8 2.5 倍）→ 維持 v10「預設不換」，差別是 27B 可以 Q8 + llama-server 上 GB10，列為 SPC（97% 對 86%）、重排序（0.878）、同 state 多題的候選。
jevify 選第二句：D0 92.5–93.6% 低於零訓練 95.5%，通用 LoRA 不採用；混用時生成逐字相同，但判斷延遲被生成拖到 3.3 倍。26B 換 b11371：check-lock 0 失敗，GB10 用一個版本。
詳見 `results/16-clef-llamacpp.md`；偏離計劃之處見該檔 §8。


- 「llama-server 版與 Cloudflare 版等價，Q8_0 可用、單題快 X 倍、三題一起快 Y 倍：**v10 的『先不換』改成『27B 接 SPC 與重排序、flash 接多題同 state』**，GB10 用 llama-server。」
- 「實作等價但量化版不可用或速度沒贏：Clef 只能 BF16 上 GB10，維持 v10 的結論，差別只是不用 transformers。」
- 「C++ 實作與 Cloudflare 版不等價（差異在 X）：等上游修正，維持 v10 的結論。」

jevify（§4b）另填一句，二選一：
- 「jevify 在我們的題上比零訓練好 X 點／ECE 從 A 降到 B，`--lora` 混用代價 Y：**值得自己用產線資料訓練一個 LoRA**。」
- 「jevify 沒有比零訓練好（英文通用資料訓練搬不過來）：要訓練也得用自己的產線資料，通用 LoRA 不採用。」
