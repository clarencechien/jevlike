# 07 — SGLang 後端評估：速度（L40S，同題、同 prompt 字串）

門檻預先登記於 `07-sglang-prereg.md`。原始：`results/modal/v4bench/*/v4.json`。llama-server 用 `-np` 多 slot + cache_prompt + `--swa-full` 並行送出；SGLang 用 list 一次送出（RadixAttention）。

## L1 / L2 單題延遲 p50 / p95（ms）

| 條件 |  |
|---|
| L1 單題 ~100-tok state |  |
| decode 地板（全 cache） |  |
| L2 state 500 tok |  |
| L2 state 2000 tok |  |

## L4 多題共用 state（~1,100 tok），K 題端到端 p50 ms（每題 cached tokens 命中率）

| K |  |
|---|
| 1 |  |
| 5 |  |
| 10 |  |
| 16 |  |

## L5 併發：decisions/s（p50 ms）

| c |  |
|---|
| 1 |  |
| 8 |  |
| 32 |  |

## L6 背景生成 512 token 時的決策延遲

| |  |
|---|
| p50 無/有負載 |  |
| p95 退化倍數 |  |
| 背景生成 tokens |  |

## S2 會議負載（120 視窗 × 7 題）與 S3 D0 test 1,001 題端到端

| |  |
|---|
| S2 wall s（decisions/s） |  |
| S3 wall s（decisions/s） |  |
