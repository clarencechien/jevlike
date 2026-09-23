# 難例（hard）與規則基線的撰寫規範

適用：`data/hard/{task}.jsonl` 與 `data/rules/{task}.json`。task 定義（instructions、選項、criteria）在 `data/seeds/tasks.json`，**gold 必須能單靠 state + criteria 判定**。

## 原則

1. **先定 label，再寫情境。** 每筆先決定 gold，再寫出能支持它的證據；`rationale` 一句話說明為何是這個 gold。
2. **state 只放證據，不放結論。** 不可出現選項的 label 字面（例：「停線」「派工」「品保」「異常」當結論用）；可以描述事實（溫度、數量、時間、誰做了什麼）。
3. **每筆手寫，不用模板。** 同一 task 內句型、機台、料號、時間要有變化。
4. 語言：**約 80% 台灣正體中文夾機台代碼**（`lang: "zh"`），**約 20% 純英文**（`lang: "en"`，像 alarm log / MES 訊息，例：`E4021 NOZZLE VACUUM LOW HEAD2`）。
5. 難例類型（每 task 都要涵蓋，大致平均）：
   - `incomplete`：訊息不完整，但仍有足夠線索判定
   - `borderline`：兩個選項之間的邊界，靠一個關鍵事實決定
   - `noisy`：夾錯字、注音錯字、縮寫、口語、中英夾雜、多餘符號
   - `distractor`：含干擾資訊（無關的數字、另一台機台的正常狀態）
   - `inverted`：先講結果再講原因、或用否定句（「不是缺料，是……」）
6. **孿生例（twin）**：每 task 至少 15 對（30 筆）。同一對兩筆只改**一個事實**讓 gold 翻轉，共用 `pair_id`，`difficulty` 都是 `hard`。其餘 30 筆 `pair_id: null`。
7. label 分布：各選項都要有；二選一的 task 大約 50/50；多選項的 task 每個選項至少 6 筆。
8. 每 task **60 筆**。

## 場景詞彙（可用，不限於此）

- 線別：SMT-L1～L6、DIP-L1、FATP-L2；工單 `WO-20260923-017`
- 機台：印刷機 DEK-01/02、SPI-01、貼片機 NXT-03 / SIPLACE-02 / M-05 / YSM-04、回焊爐 R-01/02（1–10 區）、AOI-01/02、爐後 AOI、X-ray、ICT-02、FCT-01、選擇焊 SW-01、分板機
- 物料：0402/0603/0805 電阻電容、QFN、BGA、連接器、料帶、料盤、錫膏 SAC305、鋼網、feeder slot 23、料號 R123-0402-10K
- 指標：UPH、CT、ATT、稼動率、CPK、SPC 管制圖、X-bar R、連續 7 點
- 人員：操作員、線長、設備工程師（ME）、製程工程師（PE）、品保（QE/QA）、IT、倉庫
- 英文 alarm 樣式：`E4021 NOZZLE VACUUM LOW HEAD2`、`F-1207 FEEDER SLOT 23 EMPTY`、`SERVO ALARM X-AXIS OVERLOAD`、`REFLOW ZONE4 TEMP HIGH 255C LIMIT 245C`、`MES CONNECTION TIMEOUT`

## jsonl 欄位

```json
{"id": "m_alarm_severity-h001", "task": "m_alarm_severity", "state": "...", "gold": "C",
 "difficulty": "hard", "hard_type": "borderline", "lang": "zh", "pair_id": "m_alarm_severity-hp01",
 "source": "hand", "rationale": "板子仍在爐內且溫度超上限，批量品質已受影響"}
```

- `id`：`{task}-h001` … `-h060`
- `pair_id`：`{task}-hp01` … 或 `null`
- `question` 欄位**不用寫**，展開時由 `tasks.json` 補上
- 一行一筆合法 JSON，UTF-8，不要有 BOM、不要有多餘逗號

## 規則基線 `data/rules/{task}.json`

**先寫規則、再寫難例**（規則代表「工程師憑 criteria 會寫的關鍵字規則」，不是針對難例調的）。10–20 條，由上而下第一條命中的為準：

```json
{"task": "m_alarm_category", "default": "A",
 "rules": [
   {"pattern": "NOZZLE|VACUUM|吸嘴|真空|皮帶|馬達", "label": "A", "note": "機構"},
   {"pattern": "SERVO|伺服|驅動|通訊|PLC|I/O", "label": "B", "note": "電控"}
 ]}
```

`pattern` 是 Python regex，比對時忽略大小寫。

## 驗證

寫完執行 `python3 data/validate.py`，必須零錯誤。
