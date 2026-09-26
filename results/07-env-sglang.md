# 07 — SGLang 環境（Phase 1）

| 項目 | 值 |
|---|---|
| 卡 | Modal L40S（NVIDIA L40S, 46068 MiB） |
| 映像 | `lmsysorg/sglang:gemma4-mtp`（2026-06-05；SGLang 0.5.12.post2.dev932+gbf172c492）。4 月的 `gemma4` tag（0.5.10rc0）**載不進** compressed-tensors FP8 的 MoE expert 權重（載入警告、輸出全是 `<pad>`），不能用 |
| 權重 | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` @ `ed35d7abe5d9`（28.7 GB，llm-compressor FP8 dynamic） |
| 必要修補 | L40S（Ada，shared memory 101 KB）沒有 Gemma 4 MoE（E=128, N=704, FP8）的 fused-MoE Triton 設定檔，預設 block 需 147 KB → 第一個 forward 就 OOM。映像內自己放一份保守設定（BLOCK 16/64×64×128、3 stages），未調速 |
| server 參數 | `--context-length 4096 --max-running-requests 32 --mem-fraction-static 0.8`，attention backend 自動選 triton，MoE runner triton；不開 speculative、不開 reasoning parser |
| 冷啟動 | 213.3 s（含 CUDA graph capture）；載入後 GPU 記憶體 37870 MiB |
| Prompt | 與 llama-server 完全相同的字串（`GEMMA4_TEMPLATE_NOTHINK`，經比對等於 llama-server `/apply-template` 的輸出） |
| tokenizer 比對 | 同一 prompt SGLang（HF tokenizer）116 tokens，llama-server（GGUF）117 tokens：差 1 個 token，`A`/` A` 的字母 token 兩邊都能對到，missing=0 |
| smoke | 5 例 first token 都是字母、missing=0；重複 prompt cached_tokens 115/116；不同 prompt 共用 system 前綴 24 tokens 命中；s4（SPC）SGLang 選 A、llama 選 C（gold B，兩邊都錯但錯不同） |
| 每請求固定開銷 | 全 cache 命中時仍 62 ms（llama-server 13 ms）：HTTP + scheduler 的固定成本，決定了 SGLang 單題不會比 llama 快 |
