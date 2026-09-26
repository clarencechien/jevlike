# 12 — packed readout：K 題一個前向（SGLang）

門檻預先登記於 `docs/handoff-v7-borrowed.md` §P1。alarm 家族三題打包（K=3），每個 state 只有自己 task 的那題有 gold；separate = 第一題單獨（暖前綴）+ 其餘整批；packed = 一個請求，讀每個占位符位置的字母 logprob。L40S、FP8、含 `<bos>`。

## 準確率與干擾（占位符 `_`，D0 test 100 筆／task；干擾率用全部 200 筆）

| task | K | separate acc | packed acc | 差 | McNemar p | 干擾率（所有題） | 自己那題翻 | 後面的題翻 | 缺字母 | tokens | separate ms | packed ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m_alarm_category | 3 | 0.990 | 0.990 | +0.000 | 1.000 | 7.7% | 0.0% | 11.5% | 0.000 | 513 | 359 | 306 |
| m_alarm_severity | 3 | 0.820 | 0.680 | -0.140 | 0.004 | 13.2% | 18.0% | 16.0% | 0.000 | 493 | 365 | 279 |
| m_needs_dispatch | 3 | 1.000 | 0.940 | -0.060 | 0.031 | 13.2% | 4.5% | 19.2% | 0.000 | 506 | 400 | 258 |

## K=7（3 題 + 4 個一般是非題，只量速度，100 state／task）

| task | tokens | separate ms | packed ms | packed/separate | 干擾率 |
|---|---|---|---|---|---|
| m_alarm_category | 701 | 631 | 366 | 0.58 | 8.1% |
| m_alarm_severity | 676 | 558 | 343 | 0.61 | 15.4% |
| m_needs_dispatch | 698 | 529 | 391 | 0.74 | 15.5% |

## 占位符換成 `A`（100 state／task，全部筆）

| task | packed acc（`A`） | packed acc（`_`，同 100 筆） | 干擾率（`A`） |
|---|---|---|---|
| m_alarm_category | 1.000 | 1.000 | 10.0% |
| m_alarm_severity | 0.740 | 0.710 | 15.0% |
| m_needs_dispatch | 0.960 | 0.960 | 10.7% |

門檻：每 task ±1 點且 p ≥ 0.05（1/3）；干擾率 ≤ 5%（11.3%）；K=3 packed ≤ separate × 0.6（0.75）、K=7 ≤ × 0.4（0.64）。判定：**準確率沒過 → 不採用**

