# Jev 式 typed decision on Gemma 4 26B — 實驗報告（v2：無 GB10，Modal L4 上界版）

日期：2026-09-23。執行：Claude Code web + Modal（L4）+ AI Studio（僅能力探測）。
模型：`unsloth/gemma-4-26B-A4B-it-GGUF` / `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`（16.95 GB），llama-server b11118。
資料：合成 10 task × 200 筆（`data/synthetic/`，MANIFEST + 10% 獨立抽查：不合理率 2.5%、0 筆錯）。
分項報告：`00-env.md`、`01-smoke.md`、`03-latency.md`、`04-accuracy.md`、`06-ladder.md`、`07-sglang-*.md`、`cost.md`。HTML 整理版（公開）：<https://imitator.ai-apps.work/r/gb10-typed-decisions>。

## 0. 一句話結論

在 L4 上，Gemma 4 26B-A4B 讀第一個 token 的選項 logprob 這條路**可行且零樣本就很強**：10 個 task 中 7 個 test 準確率 ≥ 0.98、選擇性準確率 @0.9 ≥ 0.99 且 coverage ≥ 0.99；3 個 task（alarm 急迫度、SPC 動作、UPH 異常）零樣本 0.82–0.93，且模型**極度過度自信**（raw confidence 平均 0.99），要靠 100 筆以內的校準把「信心門檻」變得可用。延遲在 L4 上單題 p50 206 ms（開 `--swa-full` 後 137 ms，共用 state 的後續題 73 ms），比 LLM 生成 JSON 快 2.4–3.6 倍，但 llama-server 在 L4 上併發不會加速（throughput 卡在 5.5 req/s）。 v4 追加：換 SGLang 後端，共用上下文快 3.7 倍、併發快 6 倍，但準確率低 2–3 點且重跑不穩，**即時決策層維持 llama-server**（§3d）。

## 1. 分流表（v1 §7，實測後填）

| 題目特徵 | 預期走 | 實測判斷依據 | 結論 |
|---|---|---|---|
| 封閉 label、選項 ≤ 5、每天百次以上 | typed decision | 7/10 task 零樣本 sel@0.9 ≥ 0.99、coverage ≥ 0.99（`m_alarm_category`、`m_needs_dispatch`、`q_defect_root`、`p_line_change`、`x_ticket_route`、`x_escalate`、`x_10way_intent`） | **走 typed decision，零樣本即可** |
| 10 類以上細粒度分類 | 先拆題再 typed | `x_10way_intent`（10 類相近意圖）test acc 1.00、hard 0.98、top-2 1.00；延遲 10 選項只比 2 選項多 40 ms（prompt 長） | **26B 不需要拆題**；「選項多會掉」在此模型／此資料上不成立（合成資料上；真實聊天訊息待驗） |
| 需要說明「為什麼」 | LLM 生成 | 本質上要文字 | LLM |
| 多步推理／查資料 | LLM（內部再呼叫 typed） | — | LLM |
| 模糊、需要「以上皆非」 | typed：choice + noul，低信心升級 | `x_escalate` acc 1.00、ECE 0.000；但 `m_alarm_severity` raw 信心無鑑別力（≥0.9 的 coverage 0.97 但 acc 只 0.835） | **低信心升級只能用校準後的信心**，raw 信心不能當門檻 |
| 中文夾機台代碼 | 看 lang 分項 | zh 普遍 ≥ en（見 Q5） | 不需先正規化 |
| 關鍵字規則就能解 | code | 規則基線最高 0.885（`p_line_change`），無 task ≥ 0.95 | **沒有 task 該只用規則**；規則可當 easy 案例的前置過濾（`p_line_change` easy 0.96） |
| 數值進、數值出 | 統計／傳統 ML | `p_uph_anomaly` 讀「數列文字」只有 0.93，hard 0.80 | UPH/CT 該用公式判定，再把判定結果當 state 給模型 |

三層架構草案維持：code（門檻、組合、公式）→ typed decision（判斷、分類、gating）→ LLM（規劃、RCA、意外）；由**校準後**低信心觸發升級。

## 2. Q1–Q8

### Q1 延遲（L4 = GB10 保守上界）

| 條件 | p50 | p95 | 備註 |
|---|---|---|---|
| 單題、state 100 token、2 選項、每次不同 state | 206 ms | 225 ms | 預設設定，prompt 191 token，cache 命中 0 |
| 同上 + `--swa-full` | 137 ms | 153 ms | 模板前綴 28 token 命中 |
| decode 地板（同 prompt 重複，`--swa-full`） | 26 ms | 27 ms | 對照 gemma-jev 3090 上 48 ms |
| state 500 / 2000 token | 493 / 1364 ms | 611 / 1966 ms | prefill 約 0.5 ms/token（L4） |
| LLM 生成 JSON（同題，15 個輸出 token） | 496 ms | 520 ms | **typed 快 2.4x（預設）／3.6x（swa-full）** |

GB10 記憶體頻寬與 L4 相近但算力較弱，prefill 主導的數字（206 ms）在 GB10 上可能更慢，decode 地板（26 ms）應接近。真實數字待 GB10 重跑 L1–L7（約 30 分鐘）。

### Q2 哪些 task 零樣本夠、哪些要改題、哪些救不回

| 判定 | task | test acc（raw → affine 校準） | 備註 |
|---|---|---|---|
| 零樣本夠用 | `x_ticket_route`、`x_escalate`、`x_10way_intent` | 1.00 | hard 0.97–1.00 |
| 零樣本夠用 | `m_needs_dispatch`、`p_line_change`、`m_alarm_category`、`q_defect_root` | 0.98–0.99 | hard 0.92–1.00；錯的都是 borderline / distractor |
| 要校準門檻（排序能力好） | `p_uph_anomaly` | 0.93 → 0.95（AUROC 0.991） | hard 0.80；incomplete / distractor 型 0.67–0.71 |
| 要校準 + 改 criteria | `q_spc_action` | 0.86 → 0.92（top-2 0.99）；**改寫 criteria 為可數規則後 0.95（held-out 0.945），見 `05-criteria-v2.md`** | hard 0.75；混淆集中在 B 抽檢 ↔ C 停線複檢、C ↔ D 呼叫 QE，criteria 的「連串 vs 超限」「單點 vs 多點」邊界要寫成可數的規則 |
| 要校準 + 改 criteria | `m_alarm_severity` | 0.82 → 0.90（top-2 1.00）；**改寫 criteria 後 0.90（held-out 0.875）** | hard 0.75、en 0.71；B 盡快 ↔ C 停線邊界（機台仍運作但有品質風險）模型偏向 C |
| 校準也救不回 | （無） | — | 三個弱 task 的 top-2 / AUROC 都 ≥ 0.99，是門檻與 criteria 問題，不需 fine-tune |

Smoke 就看到的現象在全量上重現：raw confidence 平均 0.99–1.00，`m_alarm_severity` raw ECE 0.178、`q_spc_action` 0.137——**模型幾乎永遠說自己有 99% 把握**，temperature 要拉到 T≈6 才校準得回來。

### Q3 多題共用 state 的成本

| N 題共用 300-token state | 預設 | `--swa-full` |
|---|---|---|
| 1 題 | 334 ms | 223 ms |
| 5 題 | 1253 ms（3.75x） | 518 ms（2.32x） |
| 10 題 | 2382 ms（7.13x） | 880 ms（3.94x） |
| 第 2 題起每題 | 231 ms | **73 ms** |

原因：Gemma 4 用 sliding-window attention，llama-server 預設 SWA cache 無法部分重用前綴（`cache_prompt` 開關完全無差）。**GB10 部署必開 `--swa-full`**（記憶體夠）。開了之後 playbook 一次問 5–10 題是划算的：每加一題約 +70 ms（L4）。

### Q4 生成負載對決策延遲的影響

- 同容器背景持續生成 512 token：決策 p50 206 → 285 ms（**1.39x**）。
- 併發 1 / 4 / 8 client：p50 211 / 736 / 1437 ms，throughput 4.7 / 5.4 / 5.5 req/s——llama-server 在 L4 上 `-np 4` **沒有把 prefill 批次化**，併發只是排隊。
- 決定要不要獨立 E4B（路線 C）：以 L4 數字看，每秒 5 題以內 26B 共用就夠；GB10 併發行為與實際生成尖峰要重測後才能定案。

### Q5 中文 vs 英文 alarm

| task | zh acc | en acc |
|---|---|---|
| m_alarm_severity | 0.84 | **0.71** |
| m_alarm_category | 0.99 | 0.93 |
| q_defect_root | 0.99 | 0.93 |
| p_line_change | 1.00 | 0.93 |
| q_spc_action | 0.82 | 0.78 |
| 其餘 5 task | 0.92–1.00 | 0.93–1.00（≥ zh） |

英文 alarm log 在 4 個 task 上略差，主因是英文樣本刻意寫成極簡 log（`REFLOW R-02 ZONE4 TEMP HIGH ...`），資訊比中文句少，是資料風格與語言的混淆；中文夾機台代碼**不需要正規化**。

### Q6 每個 task 的信心門檻（合成資料版，真實資料校準待 GB10）

| task | 建議門檻 | 用哪個信心 | 門檻內 acc / coverage |
|---|---|---|---|
| x_ticket_route、x_escalate、x_10way_intent、m_needs_dispatch、p_line_change | raw ≥ 0.9 | raw 即可 | ≥ 0.99 / ≥ 0.99 |
| m_alarm_category、q_defect_root | raw ≥ 0.9 | raw 即可（affine 反而略過擬合） | 0.99 / 0.99 |
| p_uph_anomaly | affine ≥ 0.9 | 校準後 | 0.967 / 0.92 |
| q_spc_action | affine ≥ 0.9 | 校準後 | 0.965 / 0.86 |
| m_alarm_severity | affine ≥ 0.9 | 校準後 | 1.00 / **0.40**（六成要升級 LLM 或人）|

校準用 100 筆（calibration 集），affine 對 6 選項的 `q_defect_root` 已出現過擬合（0.98 → 0.95），真實資料校準時 200–500 筆較穩。

### Q7 標註量（回答「還要不要 learning」）

| task | TF-IDF+LR @N=25 / 50 / 100 | typed 零樣本（N=0） | typed 校準後 sel@0.9 @N=25（coverage） |
|---|---|---|---|
| m_alarm_category | 0.51 / 0.78 / 0.85 | 0.98 | 0.985（0.93） |
| q_spc_action | 0.47 / 0.68 / 0.77 | 0.86 | 0.953（0.79） |
| x_ticket_route | 0.51 / 0.76 / 0.84 | 1.00 | 1.000（0.97） |

- 傳統 ML 在 100 筆內**沒有一個 task 到 0.9**（hard 子集只有 0.46–0.57）；本資料集只有 200 筆／task，N=200/500 無法量。
- typed decision：7 個 task **0 筆**就 ≥ 0.98；弱的 3 個用 25–100 筆做校準，即可在 0.79–0.93 coverage 下達到 0.95+。
- 主管版「從每題幾千筆降到幾百筆」的說法**在合成資料上成立**，且比說法更強（多數 task 是 0 筆）；但要注意 easy 例子是模板展開、規律性高，真實資料的零樣本數字預期會低於這裡。

### Q8 規則基線

| task | 規則 all / easy / hard |
|---|---|
| p_line_change | 0.885 / 0.964 / 0.70 |
| m_needs_dispatch | 0.870 / 0.857 / 0.90 |
| p_uph_anomaly | 0.755 / 0.857 / 0.52 |
| m_alarm_category | 0.740 / 0.764 / 0.68 |
| 其餘 | 0.47–0.72 |

沒有 task 的規則 ≥ 0.95，**不建議任何 task 只用規則**；`p_line_change` 的 easy 案例可先用規則過濾（0.96），剩下的交 typed decision。

## 3. 對照組：LLM 生成 JSON

同一模型、同一題、`temperature 0` 生成 `{"answer": "X"}`：10 task 準確率與 typed 相當或略低（例外：`q_spc_action` JSON 0.72 vs typed 0.78、`p_uph_anomaly` 0.88 vs 0.90），但 typed 多給一個可校準的分布，且延遲低 2.4–3.6 倍。**同一個 26B 不需要為了準確率走生成路徑。**

## 3b. 題目問題還是模型能力問題（v3 階梯實驗，`06-ladder.md`）

用 Gemma 4 E2B / E4B / 26B 三級模型 × D0（合成集 test）/ D1（錯字、線索移除、混訊）/ D2（Gemini 盲寫、雙標註一致 584 筆）交叉：

- **6 個 task 是 H2**（26B 真的強）：alarm 分類、急迫度、派工、SPC、根因、UPH，26B 比 E4B 高 6–13 點、比 E2B 高 9–30 點（McNemar p<0.05），錯字擾動下差距不變。
- **4 個 task 是 H1**（題目對這一級太簡單）：工單狀態、派給誰、升級、10 類意圖，E2B 就 ≥0.95。GB10 上這四類可用小模型。
- D2 盲寫集比 D0 容易（雙標註一致率 0.97；E4B 在 D2 上 8/10 ≥0.93），不能當「更難的題」，D0 的 hard 子集才是拉開差距的地方。26B 在 D2 平均 0.96，已到兩個強模型互相一致的水準。
- 線索移除沒有看到「靠 criteria 措辭作答」的證據（26B 移除 2+ 線索仍 1.00），但子集小、統計力不足。
- D0 校準的門檻套到 D2 仍有效（sel@0.9 準確率 0.945–1.00）。


### 3c. 先分類再派：題型查表 + E4B 級聯（`results/cascade.json`、`speed_ladder.json`）

**怎麼分的**：分的是題型不是單題。每類 100 題 test 集、三個模型同一批同一 prompt，規則預先登記：E2B 也 ≥0.95 → 題目對小模型也簡單（4 類）；26B 比 E4B 高 ≥5 點且 McNemar p<0.05 → 26B 真的強（6 類）。十類剛好 4+6，沒有落在中間。

**能不能先分類再選模型**：兩層。
1. 題型層級不需要分類器：問題是系統自己發的，發問時就知道是哪一類，查表路由。
2. 題目層級用 E4B 自己的把握度當路由：先問 E4B，raw 把握度低於門檻再問 26B。D0 test 1,001 題模擬：

| 做法 | 平均 acc | 送 26B 比例 | 期望延遲（L4，`--swa-full`） |
|---|---|---|---|
| 全部 E4B | 0.885 | 0% | 63 ms |
| E4B 先答，把握<0.99 才問 26B | 0.939 | 20% | 91 ms |
| E4B 先答，把握<0.999 才問 26B | 0.952 | 34% | 110 ms |
| 全部 26B | 0.955 | 100% | 137 ms |

四類簡單題只有 3–20% 被送上 26B，六類難題 18–86%（急迫度 86%）：系統自己把難題挑出來了。提醒：用的是 raw 把握度，上線前要校準；合成資料的門檻要在真實資料重定。

**E4B 到底多快**（L4、`--swa-full`、p50）：

| | 26B-A4B | E4B Q8 | E2B Q8 |
|---|---|---|---|
| 單題（~190 token，每次不同 state） | 137 ms | 63 ms | 39 ms |
| decode 地板（全 cache 命中） | 26 ms | 31 ms | 18 ms |
| 共用 state 問 10 題 | 880 ms | 497 ms | 302 ms |
| 模型檔 | 17 GB | 8.2 GB | 5.1 GB |

E4B 單題快 2.2 倍、十題快 1.8 倍，但 **decode 地板兩者一樣**（26B-A4B 是 MoE，每 token 只算 4B），E4B 的優勢在 prefill 與記憶體，不在 decode。

**這樣做多值**：級聯（門檻 0.999）準確率 95.2% ≈ 全用 26B 的 95.5%，平均延遲 −20%（137 → 110 ms），**26B 的負載只剩三分之一**。在 GB10 上 26B 同時要做生成，這三分之二的判斷流量移走才是主要價值；純速度上的收益有限。四類簡單題若只放 E4B，記憶體 8 GB、單題 63 ms，可以放在更小的機器上。

## 3d. 後端選擇：SGLang 值不值（v4，`07-sglang-speed.md`、`07-sglang-accuracy.md`、`07-env-sglang.md`）

**一句話**：SGLang 在多題共用上下文時快 3.7 倍、併發時快 6 倍、單題不退步，**但準確率平均低 2–3 點、而且同一題重跑答案會變**；即時決策層維持 llama-server，SGLang 只給可以容忍掉分的批次工作。

**怎麼比**：同一張 L40S、同一組 prompt 字串（`GEMMA4_TEMPLATE_NOTHINK`，比對過等於 llama-server 的模板輸出）、門檻跑前登記（`07-sglang-prereg.md`）。三個 arm：A0 llama-server UD-Q4_K_M、A1 llama-server Q8_0、B1 SGLang FP8（RedHatAI FP8-dynamic）。llama-server 用各自最佳做法（`--swa-full`、多 slot、同 slot 順序送），SGLang 先暖一題再整批送（RadixAttention 只重用已經在 cache 裡的前綴）。

**速度（p50，ms）**：

| | A1 llama-server Q8 | B1 SGLang FP8 | 差 |
|---|---|---|---|
| 單題（~100 token state） | 61 | 63 | 持平（G1 過） |
| decode 地板（全 cache） | 14 | 62 | SGLang 每請求固定開銷 ~60 ms |
| 共用 state（1,100 token）問 16 題 | 813 | 217 | **3.7×**（G2 過） |
| 併發 32 路 decisions/s | 16.6 | 111.7 | **6.7×**（G3 過） |
| 背景生成時 p95 退化 | 1.38× | 1.04× | SGLang 較穩（G4 過） |
| D0 test 1,001 題整批 | 67 s | 11 s | 6× |

**準確率護欄（D0 test，每 task 100 題，±2 點內且 McNemar p≥0.05）**：20 項只過 4 項。B1 比 A1 平均低 2.8 點（D0 0.959 → 0.931；D1-cue 0.954 → 0.904，低 5 點），`q_spc_action`、`q_defect_root` 在 D1-cue 各低 8 點。三個排除實驗：

1. **不是 FP8**：換成 bf16 權重、同一張 H100，SGLang 仍比 llama-server 低 2.6 點，答案只有 95.9% 一致。
2. **不是併發、不是 RadixAttention**：workers=1 與 `--disable-radix-cache` 的準確率（0.930 / 0.930）和原本（0.931）一樣。
3. **SGLang 同題重跑會變**：三次同設定的答案兩兩只 97% 一致；翻掉的 50 題裡 32 題原本給了 >0.9 的把握。這直接打到「拿信心當門檻」：門檻對 SGLang 不穩。

**為什麼**：差在後端本身（Triton attention／MoE kernel 的數值路徑、HF tokenizer 與 GGUF tokenizer 差一個 token），不是我們的設定。L40S 上還要自己補 fused-MoE 的 Triton 設定檔才跑得起來（`07-env-sglang.md`），這是部署成本。

**分工建議**：
- 即時 gating、派工、升級判斷：**llama-server**（準確率高、重跑穩定、單題一樣快）。
- 會議記錄事後整理、歷史工單 ETL、一次幾千題的批次標註：**SGLang**，快 5–6 倍，掉 2–3 點在可接受範圍時用；用它的結果做校準或門檻要另外校。
- GB10 上：SGLang 需要 `xomoxcc/dgx-spark-sglang` 的 sm121 映像 + FP8 權重，MTP（推測解碼）對只讀一個 token 的決策沒用，不開。

## 3e. 參考 TypeLLM 之後補的四件事（v5，`08-typellm-followups.md`、`verify.md`、`docs/handoff-v5-typellm.md`）

**一句話**：JSON 形狀 prefill 與選項順序置換平均都沒有讓弱題過門檻，**維持現行 prompt**；TypeLLM 可抄的是標籤自檢與獨立驗證腳本（已加），另外量到一個風險：**急迫度與 SPC 兩題有兩成的題目換個選項順序答案就變**。

| 項目 | 結果 | 處置 |
|---|---|---|
| T0-a 標籤自檢（每個字母單 token、可回轉） | `decide/labels.py`，smoke 啟動時檢查 | 已加 |
| T0-b 獨立驗證腳本（不呼叫模型重算 acc / Brier / ECE） | `bench/verify.py`：237 個結果檔、與 analysis.json 0 筆不一致 | 已加 |
| T0-c top-40 讀不到的字母 | 2,000 筆有 661 筆缺字母，但漏掉的機率上界 7.4e-7 | 不用改 |
| T1 JSON prefill `{"answer": "` | 弱 task 平均 +1.3 點（V1）/ +1.0 點（V2），門檻 +2；McNemar p 全 ≥ 0.5 | 不採用 |
| T2 順序置換平均（3 選項 6 種、其餘 8 種） | 弱 task 平均 −1.3 點；SPC 的 ECE 0.136 → 0.081、AUROC 0.88 → 0.95、sel@0.9 0.869 → 0.952（coverage 0.99 → 0.83） | 不建議當預設；平均後的信心比較會分辨對錯，升級路徑可考慮 |
| T4 SGLang 指定 token 讀機率（`token_ids_logprob`） | 與 top-40 讀法機率差 0.0；字母 token id 與 TypeLLM 的 Gemma 4 log 相同 | 已加（SGLang 路徑），不改 v4 結論 |
| 順序敏感度 | 急迫度 21%、SPC 19%、UPH 3%、控制題 1% | **風險**：弱題的答案有兩成受選項順序影響，上線前選項順序固定、校準用同一順序 |

TypeLLM 本身不支援 Gemma 4（`06-comparison.md`），它的 JevBench 84.4% 是 Qwen3.8-27B 在 231 題公開子集，與 534 題的 74.4 不同尺。它的 thinking mode（84% → 98.7%，每題均 919 token）產線 gating 用不起；`depends_on` 依賴鏈與 nullable 語法等真實資料有鏈式題再開。

## 4. 本版的限制

1. **延遲全部是 L4 上界**。GB10 實測待補（v1 §5 L1–L7，約 30 分鐘；記得 `--swa-full`）。
2. **合成資料**：出題者是 Claude（Fable），受測者是 Gemma；easy 70% 為模板展開，hard 30% 手寫。抽查不合理率 2.5%（全是 hard 的 borderline）。零樣本準確率在真實 alarm／工單上預期會下降；校準門檻（Q6）必須用真實資料重做（v1 §6）。
3. **aarch64、與現有生成負載搶資源**未驗證；L5 顯示 llama-server 在 L4 上不會因併發而提升 throughput，GB10 要重量。
4. **校準集只有 100 筆**，affine 校準在多選項 task 有過擬合跡象。
5. **模型檔**：用 unsloth UD-Q4_K_M；若 GB10 用的是 Google QAT q4_0 或其他量化，M4 要在 GB10 重跑（同格式 raw logprobs 可直接進 `analyze.py`）。
6. AI Studio 上的 Gemma 4 不開放 logprobs，31B dense 對照因此沒做。
8. **順序置換平均只跑一次、每 task 100 筆 test**：−1.3 點在 McNemar 上不顯著（p 0.22–1.0），「不建議」是依門檻，不是證明它有害。
7. **SGLang 對照只做到 Phase 1**：速度門檻全過、準確率護欄沒過，依預先登記沒進 Phase 2/3（grammar/數值欄位、GB10 sm121 實測）。SGLang 掉分的根因（哪個 kernel）沒有再往下挖；換 attention backend（flashinfer）或關掉 CUDA graph 是下一個可試的旋鈕。

## 5. 接回 GB10 時要做的

1. Phase 0 盤點：引擎、模型檔、`/completion` + `n_probs` 是否可用；**確認 `--swa-full` 或等效設定**。
2. 用 `bench/latency.py` 重跑 L1–L7（`--only` 可分段）。
3. 用 `bench/accuracy.py` 在 200–500 筆真實 alarm／工單上讀 logprob，`bench/analyze.py` 重算 Q6 門檻；若模型檔不同，合成集也重跑一次比對。
4. ~~依 Q2 改寫 criteria~~ 已做（`05-criteria-v2.md`）：SPC 0.815 → 0.95、急迫度 0.81 → 0.885；用真實資料再驗一次，急迫度可能還要一輪。
