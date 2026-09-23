# 00 — 環境（M0）

日期：2026-09-23

| 項目 | 值 |
|---|---|
| 執行環境 | Claude Code web（4 vCPU / 15 GB，無 GPU）+ Modal + AI Studio |
| Modal client | 1.5.5；需 `modal[api-proxy-support]`（本環境走 HTTPS proxy，gRPC 要 python-socks） |
| Modal GPU | L4，23034 MiB，driver 580.95.05 |
| Image | `ghcr.io/ggml-org/llama.cpp:server-cuda` + python 3.11；**要 `.entrypoint([])`**（image ENTRYPOINT 是 llama-server，否則 Modal runner 指令被吃掉、crash loop） |
| llama-server | `/app/llama-server`，version 0.4.1-dev (build 11118, commit e6ab7c1a4)，x86_64 |
| 模型 | `unsloth/gemma-4-26B-A4B-it-GGUF` / `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`，HF 標示 16.95 GB（實際大小下載後補） |
| Volumes | `gb10-decide-models`（模型）、`gb10-decide-results`（結果） |
| AI Studio | key 可用；`gemma-4-26b-a4b-it` / `gemma-4-31b-it` **不支援 logprobs**（`Logprobs is not enabled for this model`），且 `maxOutputTokens=1` 回空字串。只能當生成對照與資料補產 |
| Azure VM 備援 | 不需要 |

已踩的坑：
1. Modal「Could not connect」→ 缺 `python-socks`。
2. `modal run` 的 local_entrypoint 參數不能叫 `ctx`（與 click 衝突）。
3. image ENTRYPOINT 問題見上表。
4. Gemma 4（sliding-window attention）在 llama-server 預設下**前綴 cache 不能部分重用**；要 `--swa-full`（見 03-latency.md L8）。
5. `/apply-template` 要自己帶 `chat_template_kwargs.enable_thinking=false`；`--reasoning-budget 0` 只作用在 chat 路徑（見 01-smoke.md）。
