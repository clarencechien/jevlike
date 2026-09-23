# 03 — 延遲（M3）— **L4，GB10 保守上界**

每組 n=200（warm-up 20），client 端量測（同容器 localhost，不含 Modal 網路層）。原始：`results/modal/latency/latency.json`。
模型 Gemma 4 26B-A4B UD-Q4_K_M，llama-server `-c 16384 -np 4`，`cache_prompt=true` 除非另註。

## L1 單題基準（state ≈100 token，2 選項）

| 條件 | p50 ms | p95 ms | p99 ms | prompt tokens | server prompt_ms p50 | cache 命中 tokens | 備註 |
|---|---|---|---|---|---|---|---|
| L1 每次不同 state | 206 | 225 | 233 | 191 | 152 | 0 | 只有模板前綴命中 cache |
| 參考：同一 prompt 重複 | 97 | 107 | 124 | 187 | 90 | 182 | 全 cache 命中 = 1 token decode 地板 |
| 參考：cache_prompt=false | 206 | 226 | 233 | 191 | 153 | 0 | 全部重算 |

## L2 state 長度（2 選項）

| 條件 | p50 ms | p95 ms | p99 ms | prompt tokens | server prompt_ms p50 | cache 命中 tokens | 備註 |
|---|---|---|---|---|---|---|---|
| 100 token | 206 | 225 | 233 | 191 | 152 | 0 |  |
| 500 token | 493 | 611 | 646 | 647 | 343 | 0 |  |
| 2000 token | 1364 | 1966 | 2015 | 2435 | 1019 | 0 |  |

## L3 選項數（state ≈100 token）

| 條件 | p50 ms | p95 ms | p99 ms | prompt tokens | server prompt_ms p50 | cache 命中 tokens | 備註 |
|---|---|---|---|---|---|---|---|
| 2 選項 | 206 | 225 | 233 | 191 | 152 | 0 |  |
| 5 選項 | 224 | 239 | 244 | 210 | 167 | 0 |  |
| 10 選項 | 246 | 274 | 311 | 241 | 181 | 0 |  |

## L4 多題共用 state（state ≈300 token；總時間 = N 題全部答完）

| N 題 | cache on 總 p50 ms | cache on 每題 p50 | cache on 每題 prompt_ms | cache off 總 p50 ms | cache off 每題 p50 | 倍數（cache off / on） | N 題 vs 1 題（cache on） |
|---|---|---|---|---|---|---|---|
| 1 | 334 | 333 | 222 | 322 | 322 | 0.96x | 1.00x |
| 5 | 1253 | 234 | 224 | 1257 | 234 | 1.00x | 3.75x |
| 10 | 2382 | 231 | 223 | 2395 | 231 | 1.01x | 7.13x |

## L5 併發（每次不同 state，2 選項）

| 併發 client | p50 ms | p95 ms | p99 ms | throughput req/s |
|---|---|---|---|---|
| 1 | 211 | 233 | 244 | 4.7 |
| 4 | 736 | 799 | 843 | 5.4 |
| 8 | 1437 | 1534 | 1560 | 5.5 |

## L6 背景生成負載（同容器一個執行緒持續生成 512 token）

| 條件 | p50 ms | p95 ms | p99 ms | prompt tokens | server prompt_ms p50 | cache 命中 tokens | 備註 |
|---|---|---|---|---|---|---|---|
| 無背景負載（= L1） | 206 | 225 | 233 | 191 | 152 | 0 |  |
| 有背景生成 | 285 | 307 | 316 | 191 | 214 | 0 | 背景完成 2 次生成、1024 tokens |

**拖慢倍數（p50）：1.39x**（L4 上的相對值；GB10 併發行為要重測）

## L7 對照：同一題改用 `/v1/chat/completions` 生成 JSON（temperature 0，max_tokens 32）

| 方法 | p50 ms | p95 ms | p99 ms | 輸出 tokens |
|---|---|---|---|---|
| typed decision（L1） | 206 | 225 | 233 | 1 |
| LLM 生成 JSON | 496 | 520 | 541 | 15.0 |

**JSON 生成 / typed decision（p50）= 2.4x**

註：所有數字皆為 L4（300 GB/s、算力弱於 GB10）；GB10 要用 v1 §5 的 L1–L7 重跑一次（約 30 分鐘）。

## L8 追加：`--swa-full`（Gemma 4 sliding-window attention 與前綴 cache）

上面所有 `cache 命中 tokens` 都是 0：Gemma 4 用 sliding-window attention（SWA），llama-server 預設的 SWA KV cache **只能整段命中、不能部分重用前綴**，所以「多題共用 state」和「共用 system/模板前綴」都沒有省到。加 `--swa-full`（SWA 層也保留完整 KV，記憶體較大）後前綴重用生效：

n=100，`-c 8192 -np 4 --swa-full`。

| 條件 | 預設 p50 ms | `--swa-full` p50 ms | swa-full cache 命中 tokens |
|---|---|---|---|
| L1 每次不同 state | 206 | 137 | 28（模板前綴） |
| 同一 prompt 重複（decode 地板） | 97 | 26 | 186 |
| cache_prompt=false | 206 | 169 | 0 |

| N 題共用 state（cache on） | 預設 總 p50 | 預設 每題 | swa-full 總 p50 | swa-full 每題 | swa-full 每題 cache 命中 | swa-full N 題 vs 1 題 |
|---|---|---|---|---|---|---|
| 1 | 334 | 333 | 223 | 223 | 28 | 1.00x |
| 5 | 1253 | 234 | 518 | 74 | 323 | 2.32x |
| 10 | 2382 | 231 | 880 | 73 | 344 | 3.94x |

→ 共用 state 的第 2 題起，每題只剩 decode + 問題段 prefill（約 70 ms，L4）。**GB10 部署時要開 `--swa-full`**（128 GB 統一記憶體夠），否則 Q3 的答案是「N 題 = N 倍」。
