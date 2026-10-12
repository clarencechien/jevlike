# third_party/tw-sovereignty

複製自 <https://github.com/pirrer/taiwan-sovereignty-decision-models> @ `ebafab7`（MIT 程式碼、CC BY 4.0 資料）：

- `analyze.py`：原作者的分析程式（MIT），`bench/analyze_tw.py` 直接 import 它的 `aggregate`、`pair_values`、`S_cell`、`gap_list`、`boot`，讓 Index S 與 bootstrap 區間的算法跟論文同一份程式碼。
- `results/clef.jsonl`、`results/clef-flash.jsonl`：論文在 Cloudflare Workers AI 上跑 Clef／Clef-flash 的原始結果（CC BY 4.0），用來跟我們 llama.cpp 上的 Clef GGUF 逐題對照。
- `LICENSE`、`THIRD_PARTY_NOTICES.md`：原 repo 的授權與第三方聲明。
