# SPOTCHECK — data/synthetic 獨立抽查報告

- 抽查者：獨立審核（非資料作者），僅依 `state` 文字與各選項 `criteria` 判斷，不參考 `rationale` 作為依據。
- 抽樣方式：每個任務以 `random.Random(2026).sample(rows, 20)` 抽 20 筆（10%），共 200 筆。
- 判定標準：
  - OK：依 criteria，gold 明顯是最佳答案。
  - ARGUABLE：另一選項在 criteria 文字下亦可成立。
  - WRONG：gold 與 criteria 矛盾，或 state 沒有支持 gold 的證據。
- 另外檢查：標籤外洩（選項 label 字詞在 state 內當作結論使用）、中英文不自然／文法錯誤、看起來重複的列、`lang=en` 但實際非英文。

## 1. 總表

| task | sampled | OK | ARGUABLE | WRONG | 不合理率 | leakage / style 問題數 |
|---|---|---|---|---|---|---|
| m_alarm_severity | 20 | 18 | 2 | 0 | 10.0% | 0 / 1 |
| m_alarm_category | 20 | 19 | 1 | 0 | 5.0% | 0 / 1 |
| m_needs_dispatch | 20 | 20 | 0 | 0 | 0.0% | 0 / 1（同一模板問題，影響 6 筆）|
| q_spc_action | 20 | 20 | 0 | 0 | 0.0% | 0 / 0 |
| q_defect_root | 20 | 19 | 1 | 0 | 5.0% | 0 / 1 |
| p_uph_anomaly | 20 | 20 | 0 | 0 | 0.0% | 0 / 1 |
| p_line_change | 20 | 20 | 0 | 0 | 0.0% | 0 / 1 |
| x_ticket_route | 20 | 19 | 1 | 0 | 5.0% | 0 / 1 |
| x_escalate | 20 | 20 | 0 | 0 | 0.0% | 0 / 0 |
| x_10way_intent | 20 | 20 | 0 | 0 | 0.0% | 0 / 0 |
| **合計** | **200** | **195** | **5** | **0** | **2.5%** | **0 / 7** |

- 標籤外洩：200 筆抽樣中 0 筆。全檔掃描 label 字詞出現在 state 的情況（如「程式」「錫膏」「是」）皆為領域用語（貼片程式、錫膏體積、是否報廢），非把結論寫進題面；也未發現「屬機構問題」「需派工」「通知線長」這類結論句。
- `lang=en` 列：抽樣 41 筆 en 列與全檔 en 列都不含 CJK 字元，均為英文。
- hard 列全部附 `rationale`，easy 列均無；`rationale` 本身多次直接複述 criteria 用語（如「屬單站超限」「屬人為」），若受測時把 `rationale` 一併餵入會形成外洩，抽查時已刻意忽略。

## 2. 各任務 ARGUABLE / WRONG 明細

### m_alarm_severity（2 筆 ARGUABLE）

| id | gold | 建議 | 理由 |
|---|---|---|---|
| m_alarm_severity-h014 | C | ARGUABLE → B | AOI 漏檢 3 片短路但已被下游 ICT 攔下、機台仍在運作，符合 B「有品質風險但機台仍可運作」；C 要求「已造成或即將造成批量品質問題」，證據偏弱。 |
| m_alarm_severity-h001 | C | ARGUABLE → B | 第 4 區 251°C 僅超上限 1°C，與 easy 模板「超上限 9°C」量級差很多，B「有品質風險、應本班內處理」同樣可成立，C 的「批量品質問題」偏強。 |

### m_alarm_category（1 筆 ARGUABLE）

| id | gold | 建議 | 理由 |
|---|---|---|---|
| m_alarm_category-h026 | A | ARGUABLE → C | state 只有「feeder 報 EMPTY 但料盤還剩 2400」，沒有任何機構檢查結果；料帶斷裂／蓋帶撕裂（同檔 e0026 的 C 情境）同樣會產生此現象，A 與 C 皆可成立。 |

### m_needs_dispatch

無 ARGUABLE／WRONG。

### q_spc_action

無 ARGUABLE／WRONG。

### q_defect_root（1 筆 ARGUABLE）

| id | gold | 建議 | 理由 |
|---|---|---|---|
| q_defect_root-h020 | A | ARGUABLE → F | 「刮刀壓力被設成 9kg，標準 6kg」是有人把參數設錯，criteria F「未依 SOP 操作」可成立；A 的 criteria 列的是少錫／多錫／偏移等印刷現象，未明列參數設定錯誤。 |

### p_uph_anomaly

無 ARGUABLE／WRONG。

### p_line_change

無 ARGUABLE／WRONG。

### x_ticket_route（1 筆 ARGUABLE）

| id | gold | 建議 | 理由 |
|---|---|---|---|
| x_ticket_route-h014 | B | ARGUABLE → C | criteria C 明列「資料未上傳」，state 主訴正是「output data not uploading to MES」；雖然結論指向機台通訊板需更換（B），但兩個 criteria 都被字面命中，題目沒有給「該由誰處理」的規則來裁決。 |

### x_escalate

無 ARGUABLE／WRONG。（h014「FCT-01 停 48 分鐘、產出 0，但已由 OP 依 SOP 換 pogo pin」同時命中 A 的「停線超過 30 分鐘」與 B 的「已依 SOP 處理完畢」，但 30 分鐘是硬門檻且 48 分鐘不算「短暫」，判 OK。）

### x_10way_intent

無 ARGUABLE／WRONG。

## 3. Leakage / 文字風格 / 重複問題明細

| task | id | 問題 |
|---|---|---|
| m_alarm_severity | h057 | 「誤判率 2%（跟ㄧ樣）」缺受詞（應為「跟平常一樣」），超出 noisy 類型刻意加入的注音噪音，讀起來像漏字。 |
| m_alarm_category | e0125 | 英文 easy 模板結尾直接寫「PROGRAM DATA WRONG」，等於把答案寫在題面。 |
| m_needs_dispatch | e0081, e0128, e0131, e0057, e0107, e0125 | easy 模板句尾固定寫「需更換／需重新校正／… REQUIRED」，等同宣告結論；A/B 對照幾乎可用「有沒有『需』『REQUIRED』」判斷，難度趨近於零。 |
| q_defect_root | e0057 vs e0081 | 同模板（program_ef）、同操作員「小陳」、只差線別／機台／µm 數值，抽樣 20 筆就撞到 2 筆，觀感重複。 |
| p_uph_anomaly | h020 | 統計數字自相矛盾：AVG 41.2s、MIN 39.5s、但「每 10 片一次 90–95s」時平均應約 45s，且 STD 40.0s 與資料形狀不符（實際約 15s）。標籤（A，間歇性掉產能）本身正確。 |
| p_line_change | e0131 | 「ETA 1 DAYS」單複數錯誤（模板未處理 1 day）。 |
| x_ticket_route | e0131 | 「pull shipment in by 1 days」單複數錯誤（同上模板問題）。 |

重複度（全檔，以去除數字／線別／機台／人名後的字串分群）：

| task | 近似重複群數 | 落在重複群的列數 / 200 | 最大群 |
|---|---|---|---|
| m_alarm_severity | 33 | 140 | 6 |
| m_alarm_category | 33 | 122 | 5 |
| m_needs_dispatch | 34 | 132 | 5 |
| q_spc_action | 29 | 85 | 5 |
| q_defect_root | 27 | 73 | 5 |
| p_uph_anomaly | 47 | 162 | 10 |
| p_line_change | 31 | 106 | 5 |
| x_ticket_route | 27 | 91 | 5 |
| x_escalate | 32 | 98 | 5 |
| x_10way_intent | 35 | 80 | 3 |

easy 列本來就是模板 × 參數展開，這是設計使然；但 p_uph_anomaly 的「貼片 UPH 目標 X，本班平均 Y，達成率 Z%」一個模板就佔 10 列，同質性偏高。

## 4. 整體觀察

- **標籤品質整體良好**：200 筆中無 WRONG，5 筆 ARGUABLE 全部落在 hard 列（borderline 2、distractor 2、incomplete 1）。easy 列 140 筆全部 OK，孿生模板的 gold 與 criteria 對應沒有發現系統性偏差。
- **hard borderline 的「量級」需要注意**：m_alarm_severity 的 borderline 把「超上限 1°C」與 easy 的「超上限 9°C」同樣標成 C；hard 作者顯然採「超限 + 板子在爐內 ⇒ 停線」規則，但 criteria 寫的是「批量品質問題」，量級太小時 B 就可辯護。建議 borderline 情境的數值至少要讓 C 的門檻明顯成立，或在 criteria 補一句「超出上限即視為……」。
- **incomplete / distractor 類要確保剩下的證據仍足以唯一決定**：m_alarm_category-h026 把檢查結果全部拿掉後，剩下的現象同時對應機構與物料兩種根因；q_defect_root-h020 的「被設成 9kg」則同時命中 A 與 F。這兩類 hard 列刪證據時要保留「排他」的那一句。
- **x_ticket_route 的 criteria 有重疊區**：C 明列「資料未上傳」、B 明列「零件更換」；當機台通訊板壞掉導致資料未上傳時兩者同時成立（h014）。建議在 C 的 criteria 加「（機台本身硬體正常）」之類的排他條件，或在 B 加「含機台通訊板卡」。
- **m_needs_dispatch 與 m_alarm_category 的 easy 英文／中文模板把結論寫進題面**（「需更換」「REQUIRED」「PROGRAM DATA WRONG」），使 easy 題可靠關鍵字作答。若目的是測理解而非關鍵字比對，建議把結尾改成純現象描述（例如「重啟 3 次仍 AL.24」而非「驅動器需更換」）。
- **小的文字問題**：英文模板未處理單複數（「1 DAYS」「1 days」出現於 p_line_change、x_ticket_route 兩個 generator）；noisy 類型偶有超出「注音／口語」設計的漏字（m_alarm_severity-h057）；p_uph_anomaly-h020 的統計數字不自洽。這些都不影響 gold，但影響資料可信度。

## 5. 結論

- 整體不合理率（ARGUABLE + WRONG）／抽樣：**5 / 200 = 2.5%**（WRONG 0%）。
- 最差 5 筆：
  1. m_alarm_category-h026（A，可辯 C：無檢查證據，機構與物料都可解釋）
  2. x_ticket_route-h014（B，可辯 C：criteria C 明列「資料未上傳」）
  3. m_alarm_severity-h014（C，可辯 B：漏檢已被 ICT 攔下、機台仍運作）
  4. q_defect_root-h020（A，可辯 F：參數被人設錯）
  5. m_alarm_severity-h001（C，可辯 B：僅超上限 1°C）
