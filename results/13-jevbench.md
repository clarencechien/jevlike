# 13 — JevBench 公開子集自跑（self-run on the public split, not an official JevBench result）

231 題（original 72 / easy 48 / hard 111），來源與 hash 見 `data/jevbench/SOURCE.md`；評分用同 revision 的 harness（`third_party/jevbench`）。讀法同 v2：字母第一 token logprob；英文 system prompt；每題另跑一次選項反序。溫度 T=3.28 取自我們合成資料 D0 cal 半的每 task 溫度中位數，**不在 JevBench 上擬合**。

## 結果

| arm | 全部 acc | original | easy | hard | Brier | top-label ECE | Score 題 ordinal MAE | 嚴格有效 | 反序翻面率 | p50 s | 平均 input tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 26B raw | **88.7%**（205/231） | 98.6% | 100.0% | 77.5% | 0.198 | 0.093 | 0.127 | 100% | 6.5%（反序 acc 88.7%） | 0.29 | 671 |
| 26B T=3.28 | **88.7%**（205/231） | 98.6% | 100.0% | 77.5% | 0.161 | 0.044 | 0.147 | 100% | 6.5%（反序 acc 88.7%） | 0.29 | 671 |
| E4B raw | **78.8%**（182/231） | 95.8% | 100.0% | 58.6% | 0.349 | 0.162 | 0.232 | 100% | 10.0%（反序 acc 77.1%） | 0.36 | 667 |

## 同一子集的其他自跑數字（各自報，模型與設定不同）

| 系統 | 公開 231 題 acc | 備註 |
|---|---|---|
| Cygnet（凍結 Gemma-4-12B-it，vLLM，T=3.4） | 87.9% | easy 48/48、standard 70/72、hard 85/111；官方榜第 4 |
| Open-Jev-27B v1.1（Qwen3.8-27B + LoRA + 決策頭，訓練） | 85.3% | hard 72.1% |
| TypeLLM（Qwen3.8-27B NVFP4，無訓練，不開思考） | 84.4% | 開思考 98.7%（每題均 919 token） |
| Open-Jev-9B（訓練） | 77.5% | hard 59.5% |
| JevK5 v0.2（Qwen3.5-4B + 蒸餾 LoRA） | — | hard 0.739 vs 未訓練 0.613 |
| Open-Jev-2B（訓練） | 64.9% | hard 41.4% |

## 26B raw 依題族

| family | acc |
|---|---|
| temporal_numeric | 0.267 |
| tradeoff | 0.667 |
| probability | 0.700 |
| long_policy | 0.737 |
| judge_hard | 0.824 |
| ambiguous | 0.857 |
| policy | 0.917 |
| adequacy | 1.000 |
| adversarial | 1.000 |
| extraction | 1.000 |
| fact | 1.000 |
| intent | 1.000 |
| multi_hop | 1.000 |
| ordinal | 1.000 |
| routing | 1.000 |
| routing_hard | 1.000 |
| tool_selection | 1.000 |
| trap | 1.000 |

## 判定（`docs/handoff-v8-jevbench.md` §4）

26B-A4B 凍結讀字母在 JevBench 公開子集 88.7%，與 Cygnet / Open-Jev 同級：**v2 的高分不是題目量身訂做**。 E4B 78.8%（低 10.0 點，預期 5–15）。 反序翻面率 6.5%（預期 5–10%）。
