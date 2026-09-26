# Handoff v4：SGLang 後端評估 — 先過速度門檻，再談特異功能

日期：2026-09-24
負責人：Clarence
執行者：Claude Code web + Modal
前置：v1（題目與指標）、v2（Modal 環境、延遲 L1–L7）、v3（E2B/E4B/26B 階梯，`results/06-ladder.md`）

---

## 0. 這一版只回答一個問題

**jevlike 換成 SGLang 後端，速度有沒有比現在的 llama-server 好？**

jevlike 賣的一大半是速度。SGLang 的賣點（RadixAttention 自動共用前綴、同一份上下文 prefill 一次再分叉）理論上正好打中 jevlike 的工作型態，TypeLLM 在 Qwen3.8-27B 上實測 16 題平行比循序快 5.8 倍。但那是另一個模型、另一張卡、另一種比較基準。

所以本版分三個 Phase，**Phase 1 不過，後面全部不做**：

| Phase | 內容 | 過關條件 |
|---|---|---|
| 1 | 速度對決（同卡、同題、同 prompt 字串） | §3 的速度門檻 + 準確率護欄全過 |
| 2 | 特異功能：思考後作答、數值欄位、循序模式 | 每項各有自己的判準（§5） |
| 3 | 搬上 GB10（sm_121 社群映像、長時間負載） | 不凍結、記憶體與 26B 生成 server 共存 |

**不做的事**：推測解碼（MTP drafter）。jevlike 每題只出一個 token，推測解碼幫不上決策層，只會增加變因。決策 server 一律關閉。

---

## 1. 背景（本 session 查到的事實）

- SGLang 在 Gemma 4 發表當天即支援四個尺寸，含圖片／影片，音訊限 E2B/E4B；有 gemma4 reasoning parser。Google 在 HF 上確認 vLLM 與 SGLang 皆官方支援。
- 官方 Docker：x86 有 `lmsysorg/sglang:gemma4`（CUDA 12.9）與 `cu13-gemma4`；ARM64 映像標示給 GB200/GB300（**不是 GB10 的 sm_121**）。
- GB10 的社群映像：`xomoxcc/dgx-spark-sglang` 有 `gemma4-sm121` 系列 tag；31B 已在 4 節點 DGX Spark 叢集驗證，26B-A4B 的測試當時仍在進行。
- 已知風險：NVIDIA 論壇有「Gemma 4 在 DGX Spark 使用率 > 80% 系統凍結、sm_121 kernel 問題」的討論串。
- 已知取捨：有人在 GB10 上測 vLLM 與 SGLang，**NVFP4 比 FP8 慢 32%**。本版權重一律用 FP8。
- **換後端 = 換權重**：SGLang 吃 HF 格式，不吃 v2 驗證過的 GGUF UD-Q4_K_M。準確率與校準要重新確認（§3.3 護欄）。

---

## 2. 實驗設計

### 2.1 卡

**L40S（48 GB）**。理由：FP8 的 26B-A4B 權重約 26–27 GB，L4 的 24 GB 放不下。

**v2 的 llama-server 基準要在 L40S 上重跑**，不能拿 L4 的數字比。

### 2.2 四個 arm

| arm | 後端 | 權重 | 目的 |
|---|---|---|---|
| **A0** | llama-server | v2 同一份 UD-Q4_K_M GGUF | 現行基準 |
| **A1** | llama-server | Q8_0 GGUF | 精度對照組：把「量化精度」和「後端」兩個變因拆開 |
| **B1** | SGLang（自寫 client，§2.4） | FP8 HF 權重（記錄確切 repo 與 revision） | 主角 |
| **B2** | SGLang + TypeLLM client（改成 Gemma 4 格式） | 同 B1 | 選配。只在 B1 過關後做，確認 TypeLLM 本身沒有額外開銷 |

A1 vs B1 的差距 ≈ 後端的差距；A0 vs A1 的差距 ≈ 量化的差距。**結論要寫清楚是哪一個造成的。**

### 2.3 控制變因

- **prompt 字串完全相同**：沿用 `decide/prompt.py` 渲染，三個 arm 吃一模一樣的字串。
- **tokenizer 驗證**：SGLang 用 HF tokenizer、llama-server 用 GGUF 內嵌 tokenizer。smoke 時對同一 prompt 比對 token 數與 `A`/` A` 的 token id，不一致就停下來查。
- **thinking 關閉**、temperature 0、`max_new_tokens=1`。
- **context 上限一致**（4096），並記錄各 arm 的實際 KV 配置。
- **warm-up 20 次**後量測，每組 ≥ 200 次，報 p50 / p95 / p99。

### 2.4 B1 的 client（`decide/backends/sglang.py`）

不依賴 TypeLLM，直接打 SGLang 原生 `/generate`：

```python
payload = {
    "text": prompts,                      # str 或 list[str]；list 代表同批送出
    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    "return_logprob": True,
    "top_logprobs_num": 20,
    "logprob_start_len": -1,              # 只要輸出位置的 logprob
}
# 回傳 meta_info 內含 prompt_tokens、cached_tokens、output_top_logprobs
```

多題共用 state：把 K 個 prompt（共同前綴 + 各自題目）**以 list 一次送出**，靠 RadixAttention 自動共用前綴。**每次都要記錄 `cached_tokens`**：如果共用前綴的請求 `cached_tokens` 是 0，代表快取沒命中，速度數字無效。

> Gemma 4 有滑動視窗層。v2 在 llama-server 上需要 `--swa-full` 才能重用前綴。SGLang 對混合 SWA 的前綴快取行為要實測：若 `cached_tokens` 為 0 或偏低，查當版文件中關閉 hybrid SWA 記憶體最佳化的選項（名稱以當版為準），並記錄開關前後的記憶體與速度。

SGLang server 啟動參數（以當版文件為準）：

```bash
python -m sglang.launch_server \
  --model-path <FP8 repo> \
  --context-length 4096 \
  --max-running-requests 32 \
  --mem-fraction-static 0.8 \
  --host 127.0.0.1 --port 30000
# 不加任何 speculative 參數；reasoning parser 不開（thinking 關）
```

### 2.5 量測項目（沿用 v2 L1–L7，加三項）

| 代號 | 內容 | 說明 |
|---|---|---|
| L1 | 單題，state ≈ 100 token，2 選項 | 底線 |
| L2 | state 長度 100 / 500 / 2000 | prefill 隨長度 |
| L4 | **多題共用 state：K = 1 / 5 / 10 / 16，state ≈ 1,100 token** | **本版最關鍵**；對齊 TypeLLM 的量測條件 |
| L5 | 併發 1 / 8 / 32 個 client | 吞吐（decisions/s） |
| L6 | 背景持續生成 512 token 時的決策延遲 | 對應 GB10 上與 RCA 生成共存 |
| **S1** | 冷啟動時間、權重載入後的記憶體 | 不設門檻，GB10 預算用 |
| **S2** | 一場會議的模擬負載：120 個視窗 × 7 題，視窗間有共同系統提示 | 對應會議記錄的實際工作型態 |
| **S3** | 整批 D0 test（1,000 題）端到端完成時間 | 對應批次 ETL |

A0/A1 在 L4 上要用**它們各自最好的做法**：llama-server 開 `-np` 多 slot + `cache_prompt` 並行送出，不是循序一題一題送。比的是兩個後端各自的最佳狀態。

---

## 3. 過關門檻（跑之前寫死，跑完不准改）

### 3.1 速度門檻（以 A1 為主要對照；A0 並列報告）

| 門檻 | 條件 | 理由 |
|---|---|---|
| **G1 單題不退步** | B1 的 L1 p50 ≤ A1 × 1.10 | 單題是 gating 的主流量，不能變慢 |
| **G2 多題有感** | K = 10 與 K = 16 時，B1 端到端延遲 ≤ A1 ÷ 1.5 | SGLang 的主要賣點；沒快 1.5 倍就不值得換 |
| **G3 併發不退步** | c = 8 與 c = 32 時，B1 decisions/s ≥ A1 | 多線同時進來的情境 |
| **G4 共存不更差** | L6 下 B1 的 p95 退化倍數 ≤ A1 的退化倍數 | GB10 上要和 RCA 生成共存 |

判定：
- **G1–G4 全過** → 進 Phase 2
- **G2 過、G1 或 G3 沒過** → 「分工使用」：SGLang 只給批次型工作（會議記錄、ETL，對照 S2/S3），即時 gating 留在 llama-server。寫進 REPORT，進 Phase 2 但範圍限批次
- **G2 沒過** → 停。結論是「在 Gemma 4 26B 上，SGLang 的前綴共用優勢不足以抵銷換後端的成本」，維持 llama-server

### 3.2 前提檢查（不過就先除錯，不判定）

- L4 的 B1 請求中，共用前綴的 `cached_tokens` 命中率 ≥ 90%
- 三個 arm 在 D0 smoke 上 `missing` 率 = 0、first token 是字母

### 3.3 準確率護欄（速度贏了但這條沒過，視同沒過）

資料集：D0 test、D0 hard 子集、D1-cue（v3）。

- B1 vs A1：每個 task 的 accuracy 差距在 ±2 個百分點內，且配對 McNemar p ≥ 0.05
- B1 的錯誤偵測 AUROC 不低於 A1 超過 0.05
- 若 B1 顯著較好：記錄下來，但**不能當成 SGLang 的功勞**，要先看 A0 vs A1 的量化效應

---

## 4. Phase 1 執行步驟

| 步驟 | 內容 | GPU 時間 |
|---|---|---|
| P1-1 | Modal 上建 SGLang image（官方 `lmsysorg/sglang:gemma4`）；下載 FP8 權重與 Q8_0 GGUF 進 Volume | 0.5 h |
| P1-2 | 三個 arm 的 smoke：token 比對、first token、missing、`cached_tokens` | 0.3 h |
| P1-3 | L1 / L2 / L4 / L5 / L6，三個 arm | 1.5 h |
| P1-4 | S2 / S3 | 0.5 h |
| P1-5 | 準確率護欄（D0 test + hard + D1-cue，三個 arm） | 0.5 h |
| P1-6 | `results/07-sglang-speed.md`：門檻判定表、每個 G 附數字與 CI | 0 |
| **合計** | | **約 3.5 h，L40S 約 $7** |

Modal L40S 約 $1.95/h。本月免費額度若已用掉一部分，先看 `results/cost.md`。

---

## 5. Phase 2：特異功能（只在 Phase 1 過關後做）

每一項都要同時報**準確率收益**和**速度代價**。收益是用速度換來的，要算得出划不划算。

### 5.1 先思考再作答 — 只用在低信心的題目

目的：v3 裡 `m_alarm_severity` 是唯一沒解決的題（AUROC 0.70）。

做法：**同模型兩段式**。

1. 不思考快判（跟 Phase 1 一樣）
2. 若最高機率 < 門檻（取 D0 calibration 集上 sel@0.9 對應的門檻），才用 thinking 重判；thinking 預算 256 / 1024 兩檔

報告：
- 被升級到 thinking 的比例
- 升級子集上，thinking 前後的 accuracy 與錯誤偵測 AUROC
- 整體 p50 / p95 延遲（含升級的題）
- 判準：**H2 六題中至少兩題**在升級子集上 accuracy 提升 ≥ 5 點且 p < 0.05，**同時**整體 p50 不超過 Phase 1 的 2 倍。否則 thinking 不進主線

額外看一組：E4B 開 thinking 能不能在 H2 題上追到 26B 不思考的水準。能的話，GB10 上的決策層可以全用 E4B。

### 5.2 數值欄位

目的：AI ETL 需要直接抽出整數或小數（不良數、溫度、CT）。

做法：新增一個合成抽取集 `x_numeric_extract`（Fable 出題，200 筆，含單位換算、多個數字干擾、中文數字），用 SGLang 的受限解碼（regex 或 JSON schema 限制為數值）。

判準：精確匹配率、格式錯誤率（應為 0）、每題延遲。對照組：讓 26B 自由生成再用程式解析。

### 5.3 循序 vs 平行 + 程式組合

目的：TypeLLM 的循序模式讓後題看到前題答案。要知道這是升級還是風險。

做法：選一條決策鏈（alarm 類別 → 急迫度 → 是否派工），比較
- 循序：後題 prompt 帶前題 argmax
- 平行：三題各自獨立，由程式規則組合

報告：最終決策的準確率、**錯誤傳染率**（第一題錯時後題也錯的比例）、延遲。

### 5.4 圖片輸入（選配）

只在前三項有結論後做。50 張 HMI 螢幕照片，問 alarm 類別，與同內容的文字版比較準確率與延遲，視覺 token 預算 140 / 280 / 560 三檔。

Phase 2 合計 GPU 約 3–4 h，約 $6–8。

---

## 6. Phase 3：搬上 GB10（只在 Phase 1 過關後做）

1. 映像：`xomoxcc/dgx-spark-sglang` 的 gemma4-sm121 系列 tag（記錄確切 tag）；或 sparkrun 的 Gemma 4 配方（注意它預設是 vLLM）
2. 權重：與 Modal 相同的 FP8 repo 與 revision
3. **長時間負載測試**：使用率 > 80% 持續 60 分鐘（S2 的會議負載反覆跑），監控是否凍結、記憶體是否漂移
4. **共存測試**：SGLang 決策 server 與 llama-server 的 26B 生成 server 同時運行，量 G4
5. 重跑 L1 / L4 / S2，確認 Modal 上的相對結論在 GB10 上仍成立

若出現凍結：記錄條件、回報給論壇討論串，GB10 決策層維持 llama-server。

---

## 7. 交付檢核

```
[ ] results/07-env-sglang.md        映像、權重 repo 與 revision、server 參數、tokenizer 比對
[ ] results/07-sglang-speed.md      L1/L2/L4/L5/L6/S1–S3 × 3 arm；G1–G4 判定表；cached_tokens 命中率
[ ] results/07-sglang-accuracy.md   準確率護欄；A0 vs A1（量化效應）與 A1 vs B1（後端效應）分開寫
[ ] （Phase 2）results/08-features.md   thinking 升級、數值欄位、循序 vs 平行、（選配）圖片
[ ] （Phase 3）results/09-gb10-sglang.md 長時間負載、共存、GB10 重跑
[ ] results/REPORT.md               新增「後端選擇」一節：維持 llama-server / 分工使用 / 換 SGLang，附依據
[ ] results/cost.md                 累計費用
```

## 8. REPORT 要寫的一句話結論（三選一）

- 「SGLang 在多題共用上下文時快 X 倍、單題不退步、準確率無差異，**決策層換 SGLang**。」
- 「SGLang 只在批次工作快 X 倍，**即時 gating 留 llama-server，會議記錄與 ETL 走 SGLang**。」
- 「SGLang 的前綴共用優勢在 Gemma 4 26B 上不足 1.5 倍，**維持 llama-server**；需要數值欄位時用 llama-server 的 grammar 實作。」
