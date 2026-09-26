# JevBench public split（231 題）

- 來源：<https://github.com/fstandhartinger/jevbench>，revision `f8ce71361165846101d02ebc83ad44e47ae44fc3`（與 TypeLLM evals 同版）。
- 檔案：`original.jsonl` 72、`easy.jsonl` 48、`hard.jsonl` 111；每題 `provenance.license` 皆為 MIT，可重散布。
- `cat original.jsonl easy.jsonl hard.jsonl | sha256sum` = `519b0b7b57ae8c61a7a1e19733c03cba179521c56796ab8520dc2d2ab7237da0`。
- 評分程式碼 `third_party/jevbench/`（MIT，同 revision）只用於本機重算 `score_task` / `summarize`。
- 我們的數字一律標示 **self-run on the public split, not an official JevBench result**。
