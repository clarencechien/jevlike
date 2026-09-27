# 11 — conformal 門檻：給錯誤預算，得自動處理比例

`python3 bench/thresholds.py`。結果目錄 `results/modal/accuracy`，pair_id 分組 seed 0，cal/test 各半。信心 = 溫度校準後的 top_prob（auto 時每 task 取 cal 上 AUROC 最高者）。選擇性風險控制：cal 上取最低的信心切點，使「自動處理的題」的加一修正錯誤率（錯題+1）/（處理題+1）≤ ε；test 上 conf ≥ 切點才自動處理。（v7 版用的是涵蓋率分位數，不控制錯誤率，v9 修正。）門檻見 `docs/handoff-v7-borrowed.md` §P3。

## ε = 2%

| task | 信心量測 | T | test acc | 門檻 | coverage（自動處理比例） | 實際錯誤率 | 處理筆數 | 保證成立 |
|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | top_prob | 6.23 | 0.820 | inf | 0.00 | 0.000 | 0 | ✓ |
| m_alarm_category | top_prob | 4.11 | 0.980 | 0.9782 | 0.77 | 0.000 | 77 | ✓ |
| m_needs_dispatch | top_prob | 0.05 | 0.990 | 1.0000 | 1.00 | 0.010 | 100 | ✓ |
| q_spc_action | top_prob | 5.81 | 0.860 | inf | 0.00 | 0.000 | 0 | ✓ |
| q_defect_root | top_prob | 2.53 | 0.980 | 0.9692 | 0.79 | 0.000 | 79 | ✓ |
| p_uph_anomaly | top_prob | 6.02 | 0.930 | 0.9296 | 0.77 | 0.000 | 77 | ✓ |
| p_line_change | top_prob | 3.11 | 0.990 | 0.7655 | 1.00 | 0.010 | 100 | ✓ |
| x_ticket_route | top_prob | 2.71 | 1.000 | 0.8400 | 0.98 | 0.000 | 98 | ✓ |
| x_escalate | top_prob | 0.05 | 1.000 | 1.0000 | 1.00 | 0.000 | 100 | ✓ |
| x_10way_intent | top_prob | 3.45 | 1.000 | 0.9341 | 0.83 | 0.000 | 84 | ✓ |

保證成立 10/10；七類強題 coverage 最低 0.77、平均 0.91。

## ε = 5%

| task | 信心量測 | T | test acc | 門檻 | coverage（自動處理比例） | 實際錯誤率 | 處理筆數 | 保證成立 |
|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | top_prob | 6.23 | 0.820 | inf | 0.00 | 0.000 | 0 | ✓ |
| m_alarm_category | top_prob | 4.11 | 0.980 | 0.6800 | 0.99 | 0.010 | 99 | ✓ |
| m_needs_dispatch | top_prob | 0.05 | 0.990 | 1.0000 | 1.00 | 0.010 | 100 | ✓ |
| q_spc_action | top_prob | 5.81 | 0.860 | 0.8565 | 0.69 | 0.014 | 69 | ✓ |
| q_defect_root | top_prob | 2.53 | 0.980 | 0.5269 | 1.00 | 0.020 | 100 | ✓ |
| p_uph_anomaly | top_prob | 6.02 | 0.930 | 0.8968 | 0.83 | 0.012 | 83 | ✓ |
| p_line_change | top_prob | 3.11 | 0.990 | 0.6955 | 1.00 | 0.010 | 100 | ✓ |
| x_ticket_route | top_prob | 2.71 | 1.000 | 0.6192 | 0.99 | 0.000 | 99 | ✓ |
| x_escalate | top_prob | 0.05 | 1.000 | 1.0000 | 1.00 | 0.000 | 100 | ✓ |
| x_10way_intent | top_prob | 3.45 | 1.000 | 0.7678 | 0.98 | 0.000 | 99 | ✓ |

保證成立 10/10；七類強題 coverage 最低 0.98、平均 0.99。

## ε = 10%

| task | 信心量測 | T | test acc | 門檻 | coverage（自動處理比例） | 實際錯誤率 | 處理筆數 | 保證成立 |
|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | top_prob | 6.23 | 0.820 | 0.8637 | 0.58 | 0.121 | 58 | ✗ |
| m_alarm_category | top_prob | 4.11 | 0.980 | 0.6800 | 0.99 | 0.010 | 99 | ✓ |
| m_needs_dispatch | top_prob | 0.05 | 0.990 | 1.0000 | 1.00 | 0.010 | 100 | ✓ |
| q_spc_action | top_prob | 5.81 | 0.860 | 0.7889 | 0.76 | 0.053 | 76 | ✓ |
| q_defect_root | top_prob | 2.53 | 0.980 | 0.5269 | 1.00 | 0.020 | 100 | ✓ |
| p_uph_anomaly | top_prob | 6.02 | 0.930 | 0.5317 | 0.99 | 0.071 | 99 | ✓ |
| p_line_change | top_prob | 3.11 | 0.990 | 0.6955 | 1.00 | 0.010 | 100 | ✓ |
| x_ticket_route | top_prob | 2.71 | 1.000 | 0.6192 | 0.99 | 0.000 | 99 | ✓ |
| x_escalate | top_prob | 0.05 | 1.000 | 1.0000 | 1.00 | 0.000 | 100 | ✓ |
| x_10way_intent | top_prob | 3.45 | 1.000 | 0.7678 | 0.98 | 0.000 | 99 | ✓ |

保證成立 9/10；七類強題 coverage 最低 0.98、平均 0.99。

**判定**：**過**（ε=5% 保證成立 10/10，門檻 8；ε=2% 10/10，門檻 6）；強題 ε=5% coverage 最低 0.98（門檻 0.95：✓）。 Q6 改寫成 conformal 形式；lock 檔 `results/thresholds.lock.json` 給 `bench/verify.py --check-lock` 用。

與 Q6 的差別：sel@0.9 是「信心 ≥ 0.9 時的準確率與 coverage」（描述）；這裡是「先定錯誤預算，反推門檻」（保證，在可交換資料上成立，100 筆的抽樣誤差約 ±2 個百分點）。
門檻是在合成資料上算的，真實資料要用同一支程式重算；lock 檔記錄 commit 與結果目錄，換模型檔或後端時 `--check-lock` 會重測。
