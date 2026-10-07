# Handoff v13 — EmbeddingGemma 2（Google MediaPipe Decision Maker 的 bi-encoder 後端）在 D0 上的程度（2026-10-07，跑前寫死）

來源：Google 2026-10-06 發佈 MediaPipe Decision Maker（<https://developers.google.com/edge/mediapipe/solutions/decision/decision_maker>），
三個後端裡 EmbeddingGemma 標「ultra-low latency」，但官方沒有公布任何決策準確率。本輪只回答一件事：
**EmbeddingGemma 2 零樣本做我們的十類產線判斷，到什麼程度？哪些類能用、哪些不行？**

## 0. 規則

1. 只用合成資料 D0（`data/synthetic/<task>.jsonl`，每類 200 筆）。切分同 jevlike：`group_split(rows, seed=0)`，test 每類 100 筆。
2. 不訓練模型。監督式的臂（原型、LR）只用 cal 半的標註，當「embedding 裡有多少資訊」的參考，不當判定。
3. 模型：`google/embeddinggemma-2` @ `914f7f89142e33e77833254d9c9b90c3cef7303b`（未 gated，1.49 GB，含圖片與聲音編碼器，只用文字）。輸出 768 維、已 L2 正規化。
4. 跑在本機 CPU（4 vCPU，無 GPU），延遲只報這台機器的數字，不代表手機或 GB10。
5. 結果：`results/18-embeddinggemma2.md`、`results/v13.json`、原始向量相似度 `results/modal/v13/`；程式 `bench/embed_d0.py`、`bench/analyze_v13.py`。

## 1. 做法（模仿 Decision Maker 的 prewarm → evaluate）

每一題的選項文字 = `「{label}：{criteria}」`（與我們送 26B 的判準同一段字）。

| 臂 | state 前綴 | 選項前綴 | 選項向量處理 | 角色 |
|---|---|---|---|---|
| **Z1（主臂）** | `task: classification \| query: ` | 同左（對稱，模型卡對分類的建議） | 同一題 K 個選項減去它們的平均（題內中心化，即頁面說的 centroid whitening），再 L2 正規化 | 判定用 |
| Z2 | 同 Z1 | 同 Z1 | 不中心化 | 消融 |
| Z3 | `task: search result \| query: ` | `title: {label} \| text: {criteria}` | 中心化 | 消融（非對稱檢索式） |
| P1 | — | — | 每類用 cal 半該選項的 state 向量平均當原型 | 參考：100 筆標註的原型分類 |
| P2 | — | — | cal 半訓練 logistic regression（C=1） | 參考：100 筆標註的 LR |

分數 = 餘弦相似度；答案 = argmax。機率 = softmax(分數 / τ)，τ 在 cal 半用 NLL 擬合（格點同 jevlike），只影響 ECE 與選擇性風險控制，不影響答案。

## 2. 預測（跑前寫，跑完對答案）

- 主題型三類（`x_10way_intent`、`x_ticket_route`、`m_alarm_category`）：Z1 ≥ 0.80。
- 要看數字、狀態或條件的五類（`x_escalate`、`m_needs_dispatch`、`m_alarm_severity`、`q_spc_action`、`p_uph_anomaly`）：Z1 ≤ 0.65，接近亂猜（二選一 0.5、三選一 0.33、四選一 0.25）。
- `q_defect_root`、`p_line_change`：介於中間。

## 3. 判定門檻

- **A 可當意圖路由前置**：主題型三類 Z1 test 平均 ≥ 0.90。
- **B 可取代 26B 的類別**：任一類 Z1 test acc ≥ 26B（`results/modal/accuracy`）− 1 點，列出來。
- **C 零樣本整體**：Z1 十類平均 ≥ E4B（0.885）→ 「手機上 EmbeddingGemma 可替代 E4B」；否則「不能」。
- 三條各自判；預測錯了照實寫。

## 4. 結論（2026-10-07，照 §3 填）

| 門檻 | 結果 | 判定 |
|---|---|---|
| A 意圖路由前置（主題型三類 ≥ 0.90） | 0.714（十種意圖 0.69、工單分派 0.80、alarm 原因 0.65） | **不行** |
| B 可取代 26B 的類別 | 沒有任何一類到 26B − 1 點 | **沒有** |
| C 十類平均 ≥ E4B 0.885 | 0.637（E2B 0.807） | **不能替代 E4B，連 E2B 都不如** |

預測兩條都錯，方向有資訊：主題型比預期差（十種意圖 0.69，E2B 零樣本 0.95）；「要不要派工」0.89 比預期好，因為它的判準靠用詞（維修、換零件 vs 補料、清潔、重啟），不靠數字，當初分類分錯。
孿生題（只差一個數字或狀態、答案相反）855 對裡，EmbeddingGemma 2 有 45% 給同一個答案，26B 6%；UPH 82%、SPC 68%。
就算給 100 筆標註做原型分類，十類平均 0.777，仍低於 E2B 零樣本。結論：bi-encoder 只能當「這段話在講哪個主題」的粗分，不能當我們的決策層。
