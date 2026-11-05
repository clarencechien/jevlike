# v12 錯誤樣本：26B，T1 對、J0 錯（每類最多 10 個）

## m_alarm_severity（共 17 個）

- `m_alarm_severity-v12-000b` gold A，J0 選 B {'B': 0.84, 'C': 0.13, 'A': 0.02}（SevReflow:numeric）  
  `{"line":"SMT-L3","device":"R-03","zone":7,"temp_c":251,"limit_hi_c":255,"duration_min":7,"board_in_oven":true,"conveyor":"running"}`
- `m_alarm_severity-v12-003b` gold A，J0 選 C {'C': 0.96, 'A': 0.03, 'B': 0.01}（SevReflow:numeric）  
  `{"line":"SMT-L5","device":"R-02","zone":8,"temp_c":242,"limit_hi_c":245,"duration_min":6,"board_in_oven":true,"conveyor":"running"}`
- `m_alarm_severity-v12-004b` gold A，J0 選 C {'C': 0.98, 'B': 0.02, 'A': 0.0}（SevReflow:numeric）  
  `{"line":"SMT-L6","device":"R-01","zone":4,"temp_c":257,"limit_hi_c":260,"duration_min":4,"board_in_oven":true,"conveyor":"running"}`
- `m_alarm_severity-v12-005b` gold A，J0 選 C {'C': 1.0, 'B': 0.0, 'A': 0.0}（SevReflow:numeric）  
  `{"line":"SMT-L5","device":"R-03","zone":4,"temp_c":253,"limit_hi_c":260,"duration_min":8,"board_in_oven":true,"conveyor":"running"}`
- `m_alarm_severity-v12-008b` gold A，J0 選 C {'C': 1.0, 'B': 0.0, 'A': 0.0}（SevReflow:numeric）  
  `{"line":"SMT-L5","device":"R-02","zone":5,"temp_c":248,"limit_hi_c":250,"duration_min":7,"board_in_oven":true,"conveyor":"running"}`
- `m_alarm_severity-v12-009b` gold A，J0 選 C {'C': 0.87, 'B': 0.08, 'A': 0.04}（SevReflow:numeric）  
  `{"line":"SMT-L3","device":"R-02","zone":6,"temp_c":234,"limit_hi_c":240,"duration_min":6,"board_in_oven":true,"conveyor":"running"}`
- `m_alarm_severity-v12-012b` gold B，J0 選 C {'C': 0.95, 'B': 0.05, 'A': 0.0}（SevReflow:fact）  
  `{"line":"SMT-L4","device":"R-03","zone":5,"temp_c":274,"limit_hi_c":260,"duration_min":3,"board_in_oven":false,"conveyor":"stopped"}`
- `m_alarm_severity-v12-020b` gold B，J0 選 C {'C': 0.98, 'B': 0.02, 'A': 0.0}（SevReflow:fact）  
  `{"line":"SMT-L1","device":"R-01","zone":3,"temp_c":269,"limit_hi_c":255,"duration_min":3,"board_in_oven":false,"conveyor":"stopped"}`
- `m_alarm_severity-v12-032b` gold B，J0 選 C {'C': 1.0, 'B': 0.0, 'A': 0.0}（SevStations）  
  `{"line":"SMT-L2","device":"YSM-04","alarm":"供料器卡料","machine_state":"stopped","waiting_stations":0}`
- `m_alarm_severity-v12-033b` gold B，J0 選 C {'C': 1.0, 'B': 0.0, 'A': 0.0}（SevStations）  
  `{"line":"SMT-L2","device":"M-05","alarm":"吸料失敗連續 5 次","machine_state":"stopped","waiting_stations":0}`

## m_alarm_category（共 30 個）

- `m_alarm_category-v12-000a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L3","device":"YSM-06","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.6,"facility_min_bar":5.0,"machine_inlet_bar":3.5,"machine_min_bar":4.5}`
- `m_alarm_category-v12-002a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L6","device":"YSM-06","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.6,"facility_min_bar":5.0,"machine_inlet_bar":3.9,"machine_min_bar":4.5}`
- `m_alarm_category-v12-003a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L6","device":"NXT-05","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.5,"facility_min_bar":5.0,"machine_inlet_bar":3.3,"machine_min_bar":4.5}`
- `m_alarm_category-v12-004a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L3","device":"NXT-05","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.8,"facility_min_bar":5.0,"machine_inlet_bar":3.8,"machine_min_bar":4.5}`
- `m_alarm_category-v12-005a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L6","device":"YSM-06","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.5,"facility_min_bar":5.0,"machine_inlet_bar":3.5,"machine_min_bar":4.5}`
- `m_alarm_category-v12-007a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L6","device":"M-07","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.9,"facility_min_bar":5.0,"machine_inlet_bar":3.6,"machine_min_bar":4.5}`
- `m_alarm_category-v12-008a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L3","device":"YSM-04","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.7,"facility_min_bar":5.0,"machine_inlet_bar":3.4,"machine_min_bar":4.5}`
- `m_alarm_category-v12-009a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L3","device":"YSM-06","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.7,"facility_min_bar":5.0,"machine_inlet_bar":3.9,"machine_min_bar":4.5}`
- `m_alarm_category-v12-010a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L5","device":"M-07","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.6,"facility_min_bar":5.0,"machine_inlet_bar":3.7,"machine_min_bar":4.5}`
- `m_alarm_category-v12-011a` gold A，J0 選 E {'E': 1.0, 'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}（CatAir）  
  `{"line":"SMT-L3","device":"M-05","code":"E-2120","message":"AIR PRESS LO","facility_supply_bar":5.6,"facility_min_bar":5.0,"machine_inlet_bar":3.8,"machine_min_bar":4.5}`

## m_needs_dispatch（共 17 個）

- `m_needs_dispatch-v12-000b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispRetry）  
  `{"line":"SMT-L4","device":"M-05","alarm":"置件偏移","sop_reset_limit_per_shift":3,"resets_this_shift":2,"running_min_since_last_reset":36}`
- `m_needs_dispatch-v12-007b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispRetry）  
  `{"line":"SMT-L3","device":"NXT-03","alarm":"卡板","sop_reset_limit_per_shift":3,"resets_this_shift":2,"running_min_since_last_reset":32}`
- `m_needs_dispatch-v12-010b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispRetry）  
  `{"line":"SMT-L4","device":"M-07","alarm":"置件偏移","sop_reset_limit_per_shift":3,"resets_this_shift":2,"running_min_since_last_reset":21}`
- `m_needs_dispatch-v12-011b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispRetry）  
  `{"line":"SMT-L2","device":"YSM-04","alarm":"卡板","sop_reset_limit_per_shift":3,"resets_this_shift":2,"running_min_since_last_reset":12}`
- `m_needs_dispatch-v12-014b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispRetry）  
  `{"line":"SMT-L5","device":"NXT-03","alarm":"卡板","sop_reset_limit_per_shift":3,"resets_this_shift":2,"running_min_since_last_reset":32}`
- `m_needs_dispatch-v12-031b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispAdjust）  
  `{"line":"SMT-L1","device":"DEK-03","param":"印刷壓力","unit":"kg","current":6.0,"required":6.4,"operator_range_abs":0.5}`
- `m_needs_dispatch-v12-032b` gold B，J0 選 A {'A': 0.99, 'B': 0.01}（DispAdjust）  
  `{"line":"SMT-L5","device":"DEK-02","param":"印刷壓力","unit":"kg","current":6.0,"required":5.8,"operator_range_abs":0.5}`
- `m_needs_dispatch-v12-033b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispAdjust）  
  `{"line":"SMT-L5","device":"YSM-04","param":"置件頭 Z 軸高度補償","unit":"mm","current":0.0,"required":-0.02,"operator_range_abs":0.05}`
- `m_needs_dispatch-v12-035b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispAdjust）  
  `{"line":"SMT-L6","device":"NXT-05","param":"置件頭 Z 軸高度補償","unit":"mm","current":0.0,"required":0.04,"operator_range_abs":0.05}`
- `m_needs_dispatch-v12-036b` gold B，J0 選 A {'A': 1.0, 'B': 0.0}（DispAdjust）  
  `{"line":"SMT-L2","device":"DEK-02","param":"印刷壓力","unit":"kg","current":6.0,"required":6.2,"operator_range_abs":0.5}`

## q_spc_action（共 22 個）

- `q_spc_action-v12-000a` gold A，J0 選 B {'B': 0.85, 'A': 0.15, 'D': 0.0, 'C': 0.0}（Spc:run7）  
  `{"line":"SMT-L6","station":"SPI-02","chart":"錫膏厚度","unit":"µm","cl":150,"ucl":171,"lcl":129,"points":[158,147,144,144,147,150,143,149,151,142,158,145,149,152,150,145,140,162,154,152,147,158,156,148,154],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-000b` gold B，J0 選 A {'A': 0.8, 'B': 0.2, 'D': 0.0, 'C': 0.0}（Spc:run7）  
  `{"line":"SMT-L6","station":"SPI-02","chart":"錫膏厚度","unit":"µm","cl":150,"ucl":171,"lcl":129,"points":[158,147,144,144,147,150,143,149,151,142,158,145,149,152,150,145,140,162,154,152,156,158,156,156,154],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-003b` gold B，J0 選 A {'A': 0.98, 'B': 0.02, 'D': 0.0, 'C': 0.0}（Spc:run7）  
  `{"line":"SMT-L2","station":"DEK-03","chart":"印刷 Y 偏移","unit":"µm","cl":0,"ucl":24,"lcl":-24,"points":[-9,-8,-4,4,-5,-3,-3,12,3,6,-6,12,-2,1,-2,-13,4,0,-11,-6,-3,-5,-10,-6,-8],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-006b` gold B，J0 選 A {'A': 0.99, 'B': 0.01, 'D': 0.0, 'C': 0.0}（Spc:run7）  
  `{"line":"SMT-L2","station":"DEK-02","chart":"印刷 Y 偏移","unit":"µm","cl":0,"ucl":24,"lcl":-24,"points":[8,3,-1,2,-6,2,-3,-3,8,-4,-4,6,3,9,9,2,-8,-10,6,8,9,3,6,9,9],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-007b` gold B，J0 選 A {'A': 1.0, 'B': 0.0, 'D': 0.0, 'C': 0.0}（Spc:run7）  
  `{"line":"SMT-L5","station":"SPI-01","chart":"錫膏體積","unit":"%","cl":100,"ucl":124,"lcl":76,"points":[105,103,92,102,96,104,99,109,104,100,100,105,105,100,86,110,95,100,90,96,89,97,96,92,96],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-008b` gold B，J0 選 A {'A': 0.99, 'B': 0.01, 'D': 0.0, 'C': 0.0}（Spc:run7）  
  `{"line":"SMT-L3","station":"DEK-02","chart":"印刷 Y 偏移","unit":"µm","cl":0,"ucl":24,"lcl":-24,"points":[-14,-14,1,12,3,-6,-1,-5,2,8,2,4,-8,5,1,-4,8,5,-4,-10,-3,-3,-6,-9,-9],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-009a` gold A，J0 選 B {'B': 1.0, 'A': 0.0, 'C': 0.0, 'D': 0.0}（Spc:run7）  
  `{"line":"SMT-L1","station":"R-02","chart":"回焊峰值溫度","unit":"°C","cl":245,"ucl":251,"lcl":239,"points":[246,246,245,244,243,244,244,245,247,245,242,245,242,245,245,245,247,246,246,246,244,247,247,244,247],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-010a` gold A，J0 選 B {'B': 0.85, 'A': 0.15, 'D': 0.0, 'C': 0.0}（Spc:mono6）  
  `{"line":"SMT-L2","station":"DEK-02","chart":"印刷 Y 偏移","unit":"µm","cl":0,"ucl":24,"lcl":-24,"points":[-2,0,6,-8,-8,0,-8,2,0,2,6,-5,-5,-1,7,1,3,-1,8,7,3,-2,3,-12,-17],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-010b` gold B，J0 選 A {'A': 0.91, 'B': 0.09, 'D': 0.0, 'C': 0.0}（Spc:mono6）  
  `{"line":"SMT-L2","station":"DEK-02","chart":"印刷 Y 偏移","unit":"µm","cl":0,"ucl":24,"lcl":-24,"points":[-2,0,6,-8,-8,0,-8,2,0,2,6,-5,-5,-1,7,1,3,-1,8,7,3,-2,-7,-12,-17],"recheck":"none","recheck_value":null}`
- `q_spc_action-v12-011a` gold A，J0 選 B {'B': 0.85, 'A': 0.15, 'D': 0.0, 'C': 0.0}（Spc:mono6）  
  `{"line":"SMT-L5","station":"SPI-01","chart":"錫膏厚度","unit":"µm","cl":150,"ucl":171,"lcl":129,"points":[140,157,157,151,139,153,138,147,154,155,141,149,147,160,151,151,147,154,147,155,151,148,151,140,139],"recheck":"none","recheck_value":null}`

## p_uph_anomaly（共 1 個）

- `p_uph_anomaly-v12-001b` gold B，J0 選 A {'A': 0.75, 'B': 0.25}（UphHourly）  
  `{"line":"SMT-L5","device":"NXT-03","target_uph":900,"hours":"11:00–15:00","hourly_output":[905,924,930,874]}`

## t_tool_gate（共 9 個）

- `t_tool_gate-v12-001a` gold C，J0 選 B {'B': 0.99, 'C': 0.01, 'A': 0.0}（ToolParam）  
  `{"caller_role":"製程工程師","caller_lines":["SMT-L5"],"tool":"mounter.set_speed_pct","params":{"line":"SMT-L5","machine":"NXT-03","speed_pct":39},"param_spec":{"speed_pct":[50,100]},"context":"拋料偏高，想調整置件速度"}`
- `t_tool_gate-v12-004a` gold C，J0 選 B {'B': 1.0, 'C': 0.0, 'A': 0.0}（ToolParam）  
  `{"caller_role":"製程工程師","caller_lines":["SMT-L5"],"tool":"aoi.set_threshold","params":{"line":"SMT-L5","station":"AOI-02","threshold":57},"param_spec":{"threshold":[60,90]},"context":"誤判太多，想調整判定門檻"}`
- `t_tool_gate-v12-009a` gold C，J0 選 B {'B': 1.0, 'C': 0.0, 'A': 0.0}（ToolParam）  
  `{"caller_role":"技術員","caller_lines":["SMT-L1"],"tool":"aoi.set_threshold","params":{"line":"SMT-L1","station":"AOI-02","threshold":96},"param_spec":{"threshold":[60,90]},"context":"誤判太多，想調整判定門檻"}`
- `t_tool_gate-v12-010a` gold C，J0 選 B {'B': 1.0, 'C': 0.0, 'A': 0.0}（ToolParam）  
  `{"caller_role":"製程工程師","caller_lines":["SMT-L2"],"tool":"mounter.set_speed_pct","params":{"line":"SMT-L2","machine":"SIPLACE-04","speed_pct":39},"param_spec":{"speed_pct":[50,100]},"context":"拋料偏高，想調整置件速度"}`
- `t_tool_gate-v12-011a` gold C，J0 選 B {'B': 1.0, 'C': 0.0, 'A': 0.0}（ToolParam）  
  `{"caller_role":"設備工程師","caller_lines":["SMT-L1"],"tool":"aoi.set_threshold","params":{"line":"SMT-L1","station":"AOI-02","threshold":52},"param_spec":{"threshold":[60,90]},"context":"誤判太多，想調整判定門檻"}`
- `t_tool_gate-v12-014a` gold C，J0 選 B {'B': 0.84, 'C': 0.16, 'A': 0.0}（ToolParam）  
  `{"caller_role":"設備工程師","caller_lines":["SMT-L2"],"tool":"mounter.set_speed_pct","params":{"line":"SMT-L2","machine":"YSM-04","speed_pct":34},"param_spec":{"speed_pct":[50,100]},"context":"拋料偏高，想調整置件速度"}`
- `t_tool_gate-v12-015a` gold C，J0 選 B {'B': 1.0, 'C': 0.0, 'A': 0.0}（ToolParam）  
  `{"caller_role":"技術員","caller_lines":["SMT-L6"],"tool":"aoi.set_threshold","params":{"line":"SMT-L6","station":"AOI-02","threshold":94},"param_spec":{"threshold":[60,90]},"context":"誤判太多，想調整判定門檻"}`
- `t_tool_gate-v12-016a` gold C，J0 選 B {'B': 1.0, 'C': 0.0, 'A': 0.0}（ToolParam）  
  `{"caller_role":"技術員","caller_lines":["SMT-L1"],"tool":"aoi.set_threshold","params":{"line":"SMT-L1","station":"AOI-02","threshold":99},"param_spec":{"threshold":[60,90]},"context":"誤判太多，想調整判定門檻"}`
- `t_tool_gate-v12-041b` gold B，J0 選 A {'A': 0.87, 'B': 0.13, 'C': 0.0}（ToolRead）  
  `{"caller_role":"技術員","caller_lines":["SMT-L3","SMT-L6"],"tool":"aoi.export_report","params":{"line":"SMT-L4","station":"AOI-02","date":"2026-11-04","format":"csv"},"context":"早會要用昨天的 AOI 不良報表"}`

