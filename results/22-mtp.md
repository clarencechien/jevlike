# 22 — Gemma 4 MTP 草稿頭對 jevlike 有沒有影響（v18）

計劃與門檻：`docs/handoff-v18-mtp.md`（跑前寫死）。L4、llama.cpp b11371、26B UD-Q4_K_M（lock 檔同一檔）、草稿頭 unsloth `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf`、`--spec-type draft-mtp --spec-draft-n-max 4`。兩臂同容器型號、同旗標，只差草稿頭。

## 判定

- **G1 判斷等價**（翻轉 ≤ 2／1,000、check-lock 0 失敗、最大機率差 ≤ 0.05）：翻轉 7／2000、check-lock 0 失敗（base 臂 0）、最大差 0.9282 → ✗
- **G2 單題不變慢**（L1 p50 比 ≤ 1.10）：1.00 → ✓
- **G3 生成有感**（背景生成 token/s 比 ≥ 1.3 且思考生成時間比 ≤ 0.75）：1.72、0.73 → ✓
- **G4 負載下判斷不變差**（L6 slowdown 比 ≤ 1.05）：1.33 → 1.60 → ✗
- **決定**：判斷 server 不開 MTP（v4 規定不變）；生成 server 開
- 預測對答案：A_flips_0_2 ✗、B_L1_within_5pct ✓、B_L6_gen_1.5x ✓、B_L6_slowdown_not_worse ✗、C_genms_0.6x ✗、C_flips_le_2pct ✗

## A 判斷等價（D0 全部 2,000 題，逐題比）

| 題型 | n | 翻轉 | base 準確率 | mtp 準確率 | base 單題 p50 | mtp 單題 p50 |
|---|---|---|---|---|---|---|
| m_alarm_category | 200 | 0 | 0.97 | 0.97 | 621 ms | 686 ms |
| m_alarm_severity | 200 | 1 | 0.81 | 0.81 | 465 ms | 529 ms |
| m_needs_dispatch | 200 | 0 | 0.99 | 0.99 | 477 ms | 488 ms |
| q_spc_action | 200 | 2 | 0.81 | 0.82 | 731 ms | 703 ms |
| q_defect_root | 200 | 3 | 0.98 | 0.98 | 766 ms | 796 ms |
| p_uph_anomaly | 200 | 1 | 0.93 | 0.93 | 483 ms | 479 ms |
| p_line_change | 200 | 0 | 0.98 | 0.98 | 557 ms | 576 ms |
| x_ticket_route | 200 | 0 | 0.99 | 0.99 | 606 ms | 590 ms |
| x_escalate | 200 | 0 | 1.00 | 1.00 | 448 ms | 439 ms |
| x_10way_intent | 200 | 0 | 0.98 | 0.98 | 760 ms | 753 ms |

機率差：平均 0.00419、p99 0.07931、最大 0.92822。

## B 延遲（v4bench，n 100）

| 項 | base | mtp | 比 |
|---|---|---|---|
| L1 單題 p50 | 136 ms | 135 ms | 1.00 |
| L4 共用 state 16 題 p50 | 4293 ms | 4416 ms | 1.03 |
| L6 背景生成時的判斷 p50 | 180 ms | 216 ms | 1.20 |
| L6 判斷被拖慢的倍數（對 L1） | 1.33 | 1.60 | — |
| L6 背景生成 token/s | 28 | 49 | 1.72 |

## C 思考模式（三類弱題各 50 題，信心 < 0.9 的進階段 B，貪婪生成 ≤ 512 token）

| 題型 | 進階段 B | base 生成 p50 | mtp 生成 p50 | base token/s | mtp token/s | 思考後字母翻轉 | 思考文字完全相同 |
|---|---|---|---|---|---|---|---|
| m_alarm_severity | 24 | 15.8 s | 12.9 s | 29 | 39 | 1 | 0 |
| q_spc_action | 33 | 17.4 s | 12.1 s | 28 | 38 | 3 | 0 |
| p_uph_anomaly | 3 | 23.7 s | 11.7 s | 21 | 39 | 0 | 0 |

全部：60 題進階段 B，生成時間中位數比 0.73，字母翻轉 4。

## D 草稿接受率

- `mtp-q8-n4` /metrics：llamacpp:spec_decode_num_draft_tokens_total 33940；llamacpp:spec_decode_num_accepted_tokens_total 19700；llamacpp:spec_decode_num_drafts_total 8498；llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 6787；llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="1"} 5299；llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="2"} 4220；llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="3"} 3394

## 跑後對照：機率差有多少是 MTP、有多少是跑兩次就會有的（不是門檻）

思考臂的階段 A 在另一個容器上重讀了同樣的題（三類弱題各前 50 題，不生成）。base 對 base 跨容器的差就是「跑兩次」的雜訊；拿它跟 base 對 mtp 比。

| 比較 | n | 最大機率差 | 平均 | > 0.01 的題 | > 0.05 的題 | 翻轉 |
|---|---|---|---|---|---|---|
| base_vs_base | 150 | 0.0200 | 0.00031 | 3 | 0 | 0 |
| mtp_vs_mtp | 150 | 0.8397 | 0.01311 | 6 | 4 | 2 |
| base_vs_mtp_same_rows | 150 | 0.8397 | 0.01553 | 10 | 5 | 2 |

server 旗標與版本記在 `results/modal/v18/*/*/_run.json`；server 日誌 `results/modal/v18/_server_*.log`。
