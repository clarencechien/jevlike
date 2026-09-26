# jevlike — Jev 式 typed decision on Gemma 4 26B（GB10 前置實驗，Modal 版）

讀選項字母的第一個 token logprob 當決策 API；不生成文字。合成的台灣 SMT 產線資料（10 類判斷題 × 200 筆），全部在 Modal 雲端 GPU 上跑，真實資料不出廠。

- 工程報告：`results/REPORT.md`（§0 一句話結論、§1 分流表、§2 Q1–Q8、§3b–3e 追加實驗、§4 限制、§5 接回 GB10）
- 長官版 HTML（公開）：<https://imitator.ai-apps.work/r/gb10-typed-decisions>（`scripts/publish_report.sh` 重新產生並發佈）
- 六輪實驗共約 8 GPU 小時、約 $14（`results/cost.md`）

## 一句話結論

值得做、用現有的 Gemma 4 26B-A4B 自己做、不採購 Jev。十類題七類零標註 ≥ 98%；弱的三類靠改寫判斷標準與幾十筆校準；一次判斷 L4 上 0.2 秒（共用現場狀況時 0.07 秒）。推理引擎 llama-server 與 SGLang 都可用：SGLang 批次快 5 倍，前提是 prompt 要帶 `<bos>`。

## 實驗輪次（每輪先寫門檻再跑）

| 輪 | 交接文件 | 問題 | 結果檔 |
|---|---|---|---|
| v1/v2 | `docs/handoff-v1.md`、`handoff-v2.md` | 延遲、準確率、校準、標註量、規則基線 | `results/00`–`05`、`REPORT.md` |
| v3 | `docs/handoff-v3.md` | 題目太簡單還是模型真的強（E2B/E4B/26B × D0/D1/D2 階梯） | `results/06-ladder*.md`、`cascade.json` |
| v4 | `docs/handoff-v4-sglang.md` | SGLang 後端值不值（速度門檻 G1–G4 + 準確率護欄） | `results/07-*.md` |
| v5 | `docs/handoff-v5-typellm.md` | 參考 TypeLLM：JSON prefill、順序置換平均、標籤自檢、驗證腳本 | `results/08-typellm-followups.md`、`verify.md` |
| v6 | `docs/handoff-v6-sglang-stability.md` | SGLang 掉分與不穩的原因（tokenizer 對齊、批次不變推理） | `results/09-sglang-stability.md` |

對照別人的數字（Jev、gemma-jev、JevBench、TypeLLM）：`results/06-comparison.md`。

## Quickstart

```bash
pip install 'modal[api-proxy-support]' numpy scikit-learn matplotlib
# llama-server（L4）
scripts/modal.sh run modal_app.py --which download && scripts/modal.sh run modal_app.py --which smoke
scripts/modal.sh run --detach modal_app.py --which accuracy --server-extra "--swa-full"
scripts/modal.sh volume get gb10-decide-results / results/modal/ && python3 bench/analyze.py
# SGLang（L40S，FP8）
scripts/modal.sh run modal_sglang.py --which download && scripts/modal.sh run modal_sglang.py --which smoke_sglang
scripts/modal.sh run --detach modal_sglang.py --which accuracy --args "--out-sub D0 --control-n 0"
# 不呼叫模型重算所有結果檔
python3 bench/verify.py
```

`scripts/modal.sh` 把本環境的 `modal` / `modal_secret` 環境變數映射成 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`；`GB10_GPU` 選卡（L4 / L40S / H100）。token 只走環境變數，不進 repo。

## 結構

- `PLAN.md` 計畫與環境評估；`docs/` 六份交接文件
- `modal_app.py` llama-server 入口（image、volume、下載、bench；`MODELS`：26b UD-Q4_K_M / q8 / bf16 / e4b / e2b）；`modal_sglang.py` SGLang 入口（fp8 / bf16，L40S 需自帶 fused-MoE Triton 設定檔）
- `decide/` `prompt.py`（模板：`/apply-template` 學來，或 SGLang 用的靜態 `GEMMA4_TEMPLATE_NOTHINK_BOS`；T1 prompt 變體）、`client.py`（llama `/completion` n_predict=1 + n_probs；SGLang `/generate` 指定 token id 或 `input_ids`）、`labels.py`（字母單 token 自檢）
- `data/` `seeds/`（10 task 定義、v2 可數標準、錯字表）、`gen/`、`hard/`、`rules/`、`synthetic/`（D0 + MANIFEST + SPOTCHECK）、`heldout/`、`perturbed/`（D1）、`blind/`（Gemini 盲寫 D2）、`tokenized/`（llama-server 切好的 token id）
- `bench/` `smoke*.py`、`latency.py`、`accuracy.py`（`--variant`、`--ids-dir`、`--out-sub`）、`permute.py`、`v4bench.py`、`tokenize_dump.py`、`analyze*.py`（v2/ladder/v4/v5/v6）、`verify.py`、`render_html_report.py`
- `results/` 分項報告 `00`–`09`、`REPORT.md`、`cost.md`、`fig/`、`modal/`（從 Volume 拉回的原始 logprobs）

## 踩過的坑（接 GB10 時先看）

- **`<bos>`**：llama-server 對字串 prompt 會加，HF tokenizer（SGLang、vLLM、TypeLLM）對 Gemma 4 預設不加。少這一個 token 十類平均掉 2–3 分（v6）。用 `bench/verify.py` 或 smoke 確認兩邊 prompt token 數一致。
- **thinking**：Gemma 4 要在 `/apply-template` 帶 `chat_template_kwargs: {"enable_thinking": false}`，模板結尾是空的 `<|channel>thought\n<channel|>`，否則第一個 token 是 `<|channel>`。TypeLLM 因此不支援 Gemma 4。
- **前綴 cache**：llama-server 要開 `--swa-full`，否則共用 state 的多題成本是 N 倍；同一 slot 順序送才會命中。SGLang 的 RadixAttention 只重用已在 cache 裡的前綴，整批送之前先暖一題。
- **選項順序**：急迫度與 SPC 兩類弱題有兩成的題目換順序答案就變（v5），上線時順序固定、校準用同一順序。
- **信心**：raw 信心平均 0.99、對錯都一樣，門檻只能用校準後的信心（v2 Q6）。
- **SGLang 重跑**：帶 `<bos>` 後兩次一致率 99.7%，開 `--enable-deterministic-inference` 99.85%（單題慢三成）；`gemma4-mtp` 映像的 `/v1/tokenize` 會崩，標籤自檢改走本機 HF tokenizer。
