# jevlike — Jev 式 typed decision on Gemma 4 26B（GB10 前置實驗，Modal L4 版）

讀選項字母的第一個 token logprob 當決策 API；不生成文字。報告：`results/REPORT.md`；HTML 版（公開）：<https://imitator.ai-apps.work/r/gb10-typed-decisions>（`scripts/publish_report.sh` 重新產生並發佈）。

## Quickstart（3 行）

```bash
pip install 'modal[api-proxy-support]' numpy scikit-learn matplotlib
scripts/modal.sh run modal_app.py --which download && scripts/modal.sh run modal_app.py --which smoke
scripts/modal.sh run --detach modal_app.py --which accuracy && scripts/modal.sh volume get gb10-decide-results / results/modal/ && python3 bench/analyze.py
```

`scripts/modal.sh` 把本環境的 `modal` / `modal_secret` 環境變數映射成 `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`。

## 結構

- `PLAN.md` 計畫與環境評估；`docs/handoff-v1.md`、`docs/handoff-v2.md` 原始交接文件
- `modal_app.py` Modal 入口（image、volume、下載、llama-server、bench）
- `decide/` `prompt.py`（用 `/apply-template` 學模板、關 thinking）、`client.py`（`/completion` n_predict=1 + n_probs）
- `data/` `seeds/tasks.json`（10 task 定義）、`gen/`（easy 模板產生器）、`hard/`（手寫難例）、`rules/`（規則基線）、`gen_expand.py`、`validate.py`、`synthetic/`（合併輸出 + MANIFEST + SPOTCHECK）
- `bench/` `smoke.py`、`latency.py`（L1–L7，`--only`）、`accuracy.py`（raw logprobs + JSON 對照，`--resume`）、`analyze.py`（CPU：校準、AUROC、ECE、學習曲線、基線）、`render_latency.py`
- `results/` `00-env.md`、`01-smoke.md`、`03-latency.md`、`04-accuracy.md`、`REPORT.md`、`cost.md`、`fig/`、`modal/`（從 Volume 拉回的原始資料）

## 重要設定

- Gemma 4 要在 `/apply-template` 帶 `chat_template_kwargs: {"enable_thinking": false}`，否則第一個 token 是 `<|channel>`。
- 前綴 cache 要開 `--swa-full`，否則共用 state 的多題成本是 N 倍。
