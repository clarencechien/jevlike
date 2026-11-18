# data/v24 MANIFEST

`python3 data/v24/gen_v24.py --seed 24`（seed 只用在有效性檢查的抽樣；事件本身是 data/v23 的 seed 12）。
每類從 v23 的 60 組 twin 各取一邊（偶數組 a、奇數組 b），每個事件 C 完整／M 缺關鍵事實／R 缺無關事實（有對應欄位的家族才有 R）。
Qplus 在原選項後加「資訊不足」選項（M 的 gold 是它）；Qminus 是原選項（M 的 gold 記成 `-`，只看會不會被門檻轉人）。

| task | C | M | R | M 略過（剩下的事實已能判定） | R 略過 | 加的選項字母 |
|---|---|---|---|---|---|---|
| m_alarm_severity | 60 | 60 | 42 | 0 | 0 | D |
| m_alarm_category | 60 | 60 | 30 | 0 | 0 | F |
| m_needs_dispatch | 60 | 58 | 30 | 2 | 0 | C |
| q_spc_action | 60 | 56 | 0 | 4 | 0 | E |
| p_uph_anomaly | 60 | 50 | 20 | 10 | 0 | C |
| t_tool_gate | 60 | 60 | 60 | 0 | 0 | D |

| task｜family | M 拿掉 | R 拿掉 | 事件數 |
|---|---|---|---|
| m_alarm_severity｜SevReflow:numeric | limit_hi_c | duration_min | 12 |
| m_alarm_severity｜SevReflow:fact | board_in_oven, conveyor | duration_min | 10 |
| m_alarm_severity｜SevNg | inspected | upstream | 10 |
| m_alarm_severity｜SevStations | waiting_stations | （無） | 8 |
| m_alarm_severity｜SevPaste | paste_life_h | boards_printed_since_open | 10 |
| m_alarm_severity｜SevThrow | control_limit_pct | （無） | 10 |
| m_alarm_category｜CatAir | facility_supply_bar, facility_min_bar | （無） | 15 |
| m_alarm_category｜CatVolt | plant_supply_v, rated_v, tolerance_pct | （無） | 15 |
| m_alarm_category｜CatCode | code, message | machine_state | 30 |
| m_needs_dispatch｜DispRetry | sop_reset_limit_per_shift | running_min_since_last_reset | 13 |
| m_needs_dispatch｜DispMetric | sop_pass_min_pct | （無） | 15 |
| m_needs_dispatch｜DispAdjust | operator_range_abs | （無） | 15 |
| m_needs_dispatch｜DispAction | finding | alarm_code | 15 |
| q_spc_action｜Spc:run7 | ucl, lcl | （無） | 7 |
| q_spc_action｜Spc:mono6 | ucl, lcl | （無） | 7 |
| q_spc_action｜Spc:two3 | ucl, lcl | （無） | 8 |
| q_spc_action｜Spc:one_out | ucl, lcl | （無） | 16 |
| q_spc_action｜Spc:second_out | ucl, lcl | （無） | 10 |
| q_spc_action｜Spc:recheck | ucl, lcl | （無） | 8 |
| p_uph_anomaly｜UphHourly |  | hours | 10 |
| p_uph_anomaly｜UphAvg | target_uph | （無） | 20 |
| p_uph_anomaly｜UphCt | target_ct_s | （無） | 20 |
| t_tool_gate｜ToolParam | param_spec | context | 20 |
| t_tool_gate｜ToolScope | params.targets | context | 15 |
| t_tool_gate｜ToolRead | caller_lines | context | 13 |
| t_tool_gate｜ToolRole | caller_role | context | 12 |

檢查（產生時 assert）：要刪的句段在 v23 T0 恰好出現一次；M 有效才出 M 列（被拿掉的事實換 24–72 個值：數字另外乘 0.2–5 倍，gold 至少兩種，或超出規則定義範圍；剩下的事實已能判定的事件不出 M 列，含逐時產出有 ≥ 10% 驟降的 UPH——標準把突然下滑也算異常）；R 有效才出 R 列（換值後 gold 全不變）；J1n 的衍生值凡是會隨被拿掉的事實改變的（換值測試，加上 SPC 2σ／超限計數、工具線別檢查的明列清單），一律 null。C 列三種寫法與 v23 的 T0／J0／J1 逐字相同。
