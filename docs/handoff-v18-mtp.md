# Handoff v18 — Gemma 4 MTP 草稿頭對 jevlike 有沒有影響（2026-10-08，跑前寫死）

v4 的交接寫過「決策 server 一律關推測解碼」，理由是每題只出一個 token，MTP 幫不上。那是推論，沒量過。
2026-06 之後 llama.cpp 原生支援 Gemma 4 的 MTP 草稿頭（`--spec-type draft-mtp`，PR #23398；Google 把草稿頭當獨立的 assistant 模型發布，unsloth 的 GGUF 包裡附 `MTP/` 資料夾），
GB10 上同一顆 26B 很可能會同時供判斷與生成，所以值得量一次：**開了 MTP，判斷的機率會不會變、單題會不會變慢、生成與思考模式快多少、生成負載下判斷有沒有好轉。**

## 0. 規則

同前。L4、llama.cpp b11371（v11 編的；若它沒有 `--spec-type draft-mtp`，先用 `modal_clef_gguf.py` 的做法編新版，記下 tag，兩臂都用新版）。
目標模型 `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`（models volume 根目錄，與 lock 檔同一檔）。
草稿頭 `unsloth/gemma-4-26B-A4B-it-GGUF` @ `c099eb48e663fd284577b04978a94ffccb261841` 的 `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf`（462 MB）。
兩臂同一個容器型號、同一組旗標，只差 `--model-draft … --spec-type draft-mtp --spec-draft-n-max 4`。合成資料。不訓練、不改模板。

## 1. 臂

| 臂 | server 旗標 |
|---|---|
| base | `-ngl 99 -c 16384 -np 4 --jinja --swa-full --metrics`（判斷 server 加 `--reasoning-budget 0`；思考 server 不加） |
| mtp | 同上 + `--model-draft <Q8 草稿頭> --spec-type draft-mtp --spec-draft-n-max 4` |

## 2. 量什麼

| 項 | 工具 | 內容 |
|---|---|---|
| A 等價 | `bench/accuracy.py --control-n 0 --workers 4`（D0 test 1,000 題） | 兩臂逐題比：argmax 翻轉數、校準前最大機率的差；mtp 臂跑 `verify.py --check-lock` |
| B 延遲 | `bench/v4bench.py --n 100 --warmup 10 --only L1,L4,L6` | L1 單題 p50；L4 共用 state K=16；L6 背景生成 512 token 時的判斷延遲與背景生成的 token 數／秒 |
| C 思考 | `bench/think_tail.py --tasks m_alarm_severity,q_spc_action,p_uph_anomaly --trigger 0.9 --budget 512 --workers 4 --jevbench 0 --limit 50` | 階段 B 每題生成時間（`B_gen_ms`）、思考後字母翻轉數、`n_tokens` |
| D 接受率 | server `/metrics` 與日誌 | 草稿接受率（有就記，沒有就寫沒有） |

## 3. 預測

- A：翻轉 0–2 題，check-lock 0 項失敗（第一個 token 的分佈來自主模型 prefill，草稿頭沒上場）。
- B：L1 p50 差在 ±5% 內（可能慢一點點）；L6 背景生成 token／秒 mtp ≥ 1.5× base；L6 的判斷 slowdown mtp 不比 base 差。
- C：階段 B 生成時間 mtp ≤ 0.6× base；字母翻轉 ≤ 2%（貪婪解碼，理論上同句）。

## 4. 判定門檻

- **G1 判斷等價**：翻轉 ≤ 2／1,000，且 check-lock 0 項失敗，且逐題校準前最大機率差的最大值 ≤ 0.05。
- **G2 單題不變慢**：L1 p50 mtp／base ≤ 1.10。
- **G3 生成有感**：L6 背景生成 token／秒 mtp／base ≥ 1.3，**且** 階段 B 生成時間中位數 mtp／base ≤ 0.75。
- **G4 負載下判斷不變差**：L6 slowdown_p50 mtp ≤ base × 1.05。
- 決定規則：
  - G1 或 G2 沒過 → 判斷 server 維持不開（v4 的規定不變），生成 server 看 G3。
  - G1、G2 過且 G4 過 → 允許判斷與生成**共用一個開 MTP 的 server**（GB10 省一份權重）。
  - G3 沒過 → 生成 server 也不開，v4 規定原樣。
- 預測錯了照實寫。

## 5. 結論（跑完填）

（待填）
