# 11 — conformal 門檻：給錯誤預算，得自動處理比例

`python3 bench/thresholds.py`。結果目錄 `results/modal/accuracy`，pair_id 分組 seed 0，cal/test 各半。信心 = 溫度校準後的 top_prob（auto 時每 task 取 cal 上 AUROC 最高者）。split conformal：cal 上 s = 1 − conf，q = ⌈(n+1)(1−ε)⌉/n 分位，test 上 conf ≥ 1−q 才自動處理。門檻見 `docs/handoff-v7-borrowed.md` §P3。

## ε = 2%

| task | 信心量測 | T | test acc | 門檻 | coverage（自動處理比例） | 實際錯誤率 | 處理筆數 | 保證成立 |
|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | top_prob | 6.23 | 0.820 | 0.5391 | 0.99 | 0.182 | 99 | ✗ |
| m_alarm_category | top_prob | 4.11 | 0.980 | 0.7946 | 0.99 | 0.010 | 99 | ✓ |
| m_needs_dispatch | top_prob | 0.05 | 0.990 | 1.0000 | 1.00 | 0.010 | 100 | ✓ |
| q_spc_action | top_prob | 5.81 | 0.860 | 0.4921 | 0.98 | 0.122 | 98 | ✗ |
| q_defect_root | top_prob | 2.53 | 0.980 | 0.6162 | 1.00 | 0.020 | 100 | ✓ |
| p_uph_anomaly | top_prob | 6.02 | 0.930 | 0.6569 | 0.96 | 0.052 | 96 | ✗ |
| p_line_change | top_prob | 3.11 | 0.990 | 0.7655 | 1.00 | 0.010 | 100 | ✓ |
| x_ticket_route | top_prob | 2.71 | 1.000 | 0.8400 | 0.98 | 0.000 | 98 | ✓ |
| x_escalate | top_prob | 0.05 | 1.000 | 1.0000 | 1.00 | 0.000 | 100 | ✓ |
| x_10way_intent | top_prob | 3.45 | 1.000 | 0.8137 | 0.98 | 0.000 | 99 | ✓ |

保證成立 7/10；七類強題 coverage 最低 0.98、平均 0.99。

## ε = 5%

| task | 信心量測 | T | test acc | 門檻 | coverage（自動處理比例） | 實際錯誤率 | 處理筆數 | 保證成立 |
|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | top_prob | 6.23 | 0.820 | 0.5641 | 0.95 | 0.147 | 95 | ✗ |
| m_alarm_category | top_prob | 4.11 | 0.980 | 0.8835 | 0.99 | 0.010 | 99 | ✓ |
| m_needs_dispatch | top_prob | 0.05 | 0.990 | 1.0000 | 1.00 | 0.010 | 100 | ✓ |
| q_spc_action | top_prob | 5.81 | 0.860 | 0.5681 | 0.96 | 0.115 | 96 | ✗ |
| q_defect_root | top_prob | 2.53 | 0.980 | 0.7768 | 0.98 | 0.000 | 98 | ✓ |
| p_uph_anomaly | top_prob | 6.02 | 0.930 | 0.7630 | 0.91 | 0.033 | 91 | ✓ |
| p_line_change | top_prob | 3.11 | 0.990 | 0.9929 | 0.91 | 0.000 | 91 | ✓ |
| x_ticket_route | top_prob | 2.71 | 1.000 | 0.8822 | 0.97 | 0.000 | 97 | ✓ |
| x_escalate | top_prob | 0.05 | 1.000 | 1.0000 | 1.00 | 0.000 | 100 | ✓ |
| x_10way_intent | top_prob | 3.45 | 1.000 | 0.8904 | 0.93 | 0.000 | 94 | ✓ |

保證成立 8/10；七類強題 coverage 最低 0.91、平均 0.97。

## ε = 10%

| task | 信心量測 | T | test acc | 門檻 | coverage（自動處理比例） | 實際錯誤率 | 處理筆數 | 保證成立 |
|---|---|---|---|---|---|---|---|---|
| m_alarm_severity | top_prob | 6.23 | 0.820 | 0.6151 | 0.90 | 0.122 | 90 | ✗ |
| m_alarm_category | top_prob | 4.11 | 0.980 | 0.9674 | 0.89 | 0.000 | 89 | ✓ |
| m_needs_dispatch | top_prob | 0.05 | 0.990 | 1.0000 | 1.00 | 0.010 | 100 | ✓ |
| q_spc_action | top_prob | 5.81 | 0.860 | 0.6606 | 0.87 | 0.080 | 87 | ✓ |
| q_defect_root | top_prob | 2.53 | 0.980 | 0.9135 | 0.93 | 0.000 | 93 | ✓ |
| p_uph_anomaly | top_prob | 6.02 | 0.930 | 0.8378 | 0.86 | 0.023 | 86 | ✓ |
| p_line_change | top_prob | 3.11 | 0.990 | 0.9962 | 0.90 | 0.000 | 90 | ✓ |
| x_ticket_route | top_prob | 2.71 | 1.000 | 0.9869 | 0.94 | 0.000 | 94 | ✓ |
| x_escalate | top_prob | 0.05 | 1.000 | 1.0000 | 1.00 | 0.000 | 100 | ✓ |
| x_10way_intent | top_prob | 3.45 | 1.000 | 0.9033 | 0.93 | 0.000 | 94 | ✓ |

保證成立 9/10；七類強題 coverage 最低 0.89、平均 0.94。

**判定**：**過**（ε=5% 保證成立 8/10，門檻 8；ε=2% 7/10，門檻 6）；強題 ε=5% coverage 最低 0.91（門檻 0.95：✗）。 Q6 改寫成 conformal 形式；lock 檔 `results/thresholds.lock.json` 給 `bench/verify.py --check-lock` 用。

與 Q6 的差別：sel@0.9 是「信心 ≥ 0.9 時的準確率與 coverage」（描述）；這裡是「先定錯誤預算，反推門檻」（保證，在可交換資料上成立，100 筆的抽樣誤差約 ±2 個百分點）。
門檻是在合成資料上算的，真實資料要用同一支程式重算；lock 檔記錄 commit 與結果目錄，換模型檔或後端時 `--check-lock` 會重測。
