# 07 — SGLang 後端評估：速度（L40S，同題、同 prompt 字串）

門檻預先登記於 `07-sglang-prereg.md`。原始：`results/modal/v4bench/*/v4.json`。llama-server 用 `-np` 多 slot + cache_prompt + `--swa-full` 並行送出；SGLang 用 list 一次送出（RadixAttention）。

## L1 / L2 單題延遲 p50 / p95（ms）

| 條件 | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 |
|---|---|---|
| L1 單題 ~100-tok state | 53 / 63 | 61 / 65 |
| decode 地板（全 cache） | 13 / 14 | 14 / 15 |
| L2 state 500 tok | 123 / 144 | 133 / 153 |
| L2 state 2000 tok | 325 / 504 | 336 / 524 |

## L4 多題共用 state（~1,100 tok），K 題端到端 p50 ms（每題 cached tokens 命中率）

| K | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 |
|---|---|---|
| 1 | 208（2%） | 220（2%） |
| 5 | 333（74%） | 396（74%） |
| 10 | 481（81%） | 586（81%） |
| 16 | 664（87%） | 813（87%） |

## L5 併發：decisions/s（p50 ms）

| c | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 |
|---|---|---|
| 1 | 15.2（58） | 13.8（66） |
| 8 | 19.9（402） | 18.3（436） |
| 32 | 19.3（1329） | 16.6（1551） |

## L6 背景生成 512 token 時的決策延遲

| | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 |
|---|---|---|
| p50 無/有負載 | 53 → 75 | 61 → 84 |
| p95 退化倍數 | 1.31x | 1.38x |
| 背景生成 tokens | 1024 | 1024 |

## S2 會議負載（120 視窗 × 7 題）與 S3 D0 test 1,001 題端到端

| | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 |
|---|---|---|
| S2 wall s（decisions/s） | 62.2（13.5） | 65.7（12.8） |
| S3 wall s（decisions/s） | 60.1（16.7） | 67.3（14.9） |
