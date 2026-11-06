# data/v23 MANIFEST

`python3 data/v23/gen_v23.py --seed 12`。gold 全部由事件與規則決定（不是模型標的）。五種寫法同一個 `id`，只差目錄。每列另有 `needs_compare`、`family`、`twin_kind`、`event`（事實）、`derived`（衍生欄位）。

| task | n | gold 分布 | needs_compare | numeric twin 對數 | families |
|---|---|---|---|---|---|
| m_alarm_severity | 120 | {'A': 32, 'B': 38, 'C': 50} | 100 | 50 / 60 | SevReflow:numeric 12, SevReflow:fact 10, SevNg 10, SevStations 8, SevPaste 10, SevThrow 10 |
| m_alarm_category | 120 | {'A': 27, 'B': 27, 'C': 12, 'D': 12, 'E': 42} | 60 | 30 / 60 | CatAir 15, CatVolt 15, CatCode 30 |
| m_needs_dispatch | 120 | {'A': 60, 'B': 60} | 90 | 45 / 60 | DispRetry 15, DispMetric 15, DispAdjust 15, DispAction 15 |
| q_spc_action | 120 | {'A': 42, 'B': 26, 'C': 34, 'D': 18} | 120 | 52 / 60 | Spc:run7 10, Spc:mono6 8, Spc:two3 8, Spc:one_out 16, Spc:second_out 10, Spc:recheck 8 |
| p_uph_anomaly | 120 | {'A': 60, 'B': 60} | 120 | 60 / 60 | UphHourly 20, UphAvg 20, UphCt 20 |
| t_tool_gate | 120 | {'A': 13, 'B': 60, 'C': 47} | 70 | 35 / 60 | ToolParam 20, ToolScope 15, ToolRead 13, ToolRole 12 |

平均字元數（state）：

| task | T0 | T1 | J0 | J1 | JN |
|---|---|---|---|---|---|
| m_alarm_severity | 59 | 77 | 128 | 164 | 317 |
| m_alarm_category | 64 | 100 | 128 | 202 | 320 |
| m_needs_dispatch | 72 | 90 | 117 | 151 | 311 |
| q_spc_action | 150 | 229 | 225 | 398 | 420 |
| p_uph_anomaly | 48 | 74 | 86 | 134 | 286 |
| t_tool_gate | 123 | 172 | 206 | 326 | 413 |

檢查（產生時 assert）：五種寫法事實一致（數字逐字、布林／列舉用中文詞）；T1 含所有衍生值；任何寫法都不含選項標籤或判斷詞（`FORBID`）；每組 twin 的 gold 不同。
T1 是 T0 加上括號裡的衍生值（保留原始值），所以 T1 與 J1 資訊相同；handoff 範例的 T1 省掉原始值，這裡沒有照做，以維持資訊對等。
