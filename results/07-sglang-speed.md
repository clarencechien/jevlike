# 07 — SGLang 後端評估：速度（L40S，同題、同 prompt 字串）

門檻預先登記於 `07-sglang-prereg.md`。原始：`results/modal/v4bench/*/v4.json`。llama-server 用 `-np` 多 slot + cache_prompt + `--swa-full` 並行送出；SGLang 用 list 一次送出（RadixAttention）。

## L1 / L2 單題延遲 p50 / p95（ms）

| 條件 | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 | B1 SGLang FP8 |
|---|---|---|---|
| L1 單題 ~100-tok state | 53 / 63 | 61 / 65 | 63 / 73 |
| decode 地板（全 cache） | 13 / 14 | 14 / 15 | 62 / 65 |
| L2 state 500 tok | 123 / 144 | 133 / 153 | 63 / 66 |
| L2 state 2000 tok | 325 / 504 | 336 / 524 | 133 / 147 |

## L4 多題共用 state（~1,100 tok），K 題端到端 p50 ms（每題 cached tokens 命中率）

| K | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 | B1 SGLang FP8 |
|---|---|---|---|
| 1 | 208（2%） | 220（2%） | 85（2%） |
| 5 | 333（74%） | 396（74%） | 157（77%） |
| 10 | 481（81%） | 586（81%） | 217（86%） |
| 16 | 664（87%） | 813（87%） | 217（90%） |

## L5 併發：decisions/s（p50 ms）

| c | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 | B1 SGLang FP8 |
|---|---|---|---|
| 1 | 15.2（58） | 13.8（66） | 16.0（62） |
| 8 | 19.9（402） | 18.3（436） | 58.8（134） |
| 32 | 19.3（1329） | 16.6（1551） | 111.7（267） |

## L6 背景生成 512 token 時的決策延遲

| | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 | B1 SGLang FP8 |
|---|---|---|---|
| p50 無/有負載 | 53 → 75 | 61 → 84 | 63 → 72 |
| p95 退化倍數 | 1.31x | 1.38x | 1.04x |
| 背景生成 tokens | 1024 | 1024 | 2560 |

## S2 會議負載（120 視窗 × 7 題）與 S3 D0 test 1,001 題端到端

| | A0 llama-server UD-Q4_K_M | A1 llama-server Q8_0 | B1 SGLang FP8 |
|---|---|---|---|
| S2 wall s（decisions/s） | 62.2（13.5） | 65.7（12.8） | 13.0（64.7） |
| S3 wall s（decisions/s） | 60.1（16.7） | 67.3（14.9） | 10.8（92.8） |

## 門檻判定（B1 vs A1）

| 門檻 | 數字 | 過？ |
|---|---|---|
| G1 單題不退步（B1 ≤ A1×1.10） | B1/A1 = 1.03 | ✓ |
| G2 多題有感（K=10,16 ≥1.5×） | A1/B1 = 2.70×（K=10）, 3.74×（K=16） | ✓ |
| G3 併發不退步（c=8,32） | B1 58.8/111.7 vs A1 18.3/16.6 dec/s | ✓ |
| G4 共存不更差（p95 退化） | B1 1.04× vs A1 1.38× | ✓ |
| 前提：B1 前綴命中率 ≥90% | 最低 77% | ✗ |

**判定：G1–G4 全過 → 進 Phase 2**

但準確率護欄 4/20 過（`07-sglang-accuracy.md`）：依預先登記，護欄沒過**視同沒過**，Phase 2 不進。 速度結論保留：多題共用 state 3.7×、併發 6×、批次 1,001 題 5.6×，這些是 SGLang 給批次型工作的價值。
