# Jev 式 typed decision 實驗 — Handoff v2（Claude Code web + Modal + AI Studio）

日期：2026-09-23
負責人：Clarence
執行者：Claude Code（網頁版）
前一版：`gb10-typed-decisions-handoff.md`（v1，以 GB10 為執行環境；背景資料、題目設計、指標定義都在 v1，本文件不重複，只寫「沒有 GB10 時怎麼跑」）

---

## 0. 這一版要做到什麼

**沒有 GB10 也能把實驗跑到 80%。** 用 Fable 5 出題、Modal 上的 L4 跑 Gemma 4 26B 讀 logprob、AI Studio 當選配，最後只留「延遲真實數字」和「真實資料校準」兩件事給 GB10。

能回答與不能回答（對應 v1 §7 的 Q1–Q8）：

| 問題 | 沒有 GB10 能回答到什麼程度 |
|---|---|
| Q1 GB10 上的延遲 | **只有上界**。L4 頻寬（300 GB/s）接近 GB10（273 GB/s），但算力較弱；L4 的決策延遲可當 GB10 的保守上界 |
| Q2 哪些 task 零樣本夠、哪些要改題、哪些救不回 | **完整回答**（合成資料上）。同一份模型檔，準確率與 GPU 無關 |
| Q3 多題共用 state 的成本 | **完整回答**（相對倍數），絕對毫秒數以 GB10 為準 |
| Q4 生成負載對決策延遲的影響 | **部分**。L4 上可以模擬，但 GB10 的併發行為要重測 |
| Q5 中文 vs 英文 alarm | **完整回答** |
| Q6 每個 task 的信心門檻 | **合成資料版**。真實資料校準只能在 GB10 做 |
| Q7 標註量學習曲線 | **完整回答** |
| Q8 規則基線在哪些 task 夠用 | **完整回答**（純 CPU） |
| aarch64 相容性、與現有生成負載搶資源 | **不能**，留給 GB10 |

---

## 1. 環境分工

```
Claude Code web  ── 寫程式、產資料（Fable 5）、發 modal run、讀結果、寫報告
       │
       ├── Modal (L4)  ── llama-server + Gemma 4 26B-A4B GGUF；跑 smoke / latency / accuracy
       │                  結果寫進 Modal Volume，再 modal volume get 拉回 repo
       │
       ├── AI Studio   ── 選配：31B dense 對照、或 Claude Code 用量吃緊時補產資料
       │
       └── (備援) Azure 開發 VM ── 若 Claude Code web 連不到 Modal，就 push repo 到 VM 執行 modal run
```

**真實產線資料不進任何一個雲端環境。** 本版所有資料都是合成的。

---

## 2. Milestone 與時間／成本估算

| 里程碑 | 內容 | 人工／Claude Code 時間 | Modal GPU 時間 | 費用 |
|---|---|---|---|---|
| M0 環境 | Modal token 可用、llama.cpp CUDA image 起得來、GGUF 下載進 Volume | 0.5–1 h | 0.5 h（含下載） | < $1 |
| M1 Smoke | 5 個手寫例子跑通，確認第一個 token 是答案字母 | 0.5 h | 0.2 h | < $0.5 |
| M2 資料 | 10 個 task × ≥200 筆 + 孿生例子 + MANIFEST | 2–4 h（Fable 寫素材與難例；程式展開） | 0 | 訂閱額度 |
| M3 延遲 | L1–L7（L6 在 L4 上模擬） | 0.5 h | 0.5–1 h | ~$1 |
| M4 準確率 | 10 task 讀 logprob、存 raw logits；JSON 生成對照組 | 0.5 h | 1–1.5 h | ~$1.5 |
| M5 CPU 分析 | 校準、AUROC、ECE、學習曲線、規則與 TF-IDF 基線 | 1–2 h | 0 | 0 |
| M6 報告 | `results/REPORT.md`，填 v1 §7 的表 | 1 h | 0 | 0 |
| 除錯與重跑 | — | 2 h | 1–2 h | ~$2 |
| **合計** | | **約 1.5–2 個工作天** | **4–6 h** | **L4 約 $4–6，含重跑 < $20** |

L4 在 Modal 約 $0.80/h。指定 region 或關掉搶佔會加價；本實驗不需要指定 region，接受搶佔（結果寫 Volume，中斷可續跑）。

AI Studio 免費額度約 15 RPM / 1,500 次/天，只在選配時用。

---

## 3. M0：環境

### 3.1 Claude Code web 要先確認的事

```
[ ] pip install modal 成功
[ ] 環境變數 MODAL_TOKEN_ID / MODAL_TOKEN_SECRET 已設（在 Claude Code 環境設定中加入，不要寫進 repo）
[ ] modal token 驗證：python -c "import modal; print(modal.__version__)" 後 `modal profile current`
[ ] 網路能連到 modal.com API（Claude Code web 的網路白名單可能要加）
    → 不行：把 repo push 到 GitHub，改在 Azure 開發 VM 上執行 modal run，其餘不變
[ ] （選配）GEMINI_API_KEY 已設
```

### 3.2 repo 結構（沿用 v1，多一個 modal_app.py）

```
gb10-decide/
├── modal_app.py          # Modal 入口：image、volume、下載模型、起 llama-server、跑 bench
├── decide/
│   ├── client.py         # 對 llama-server /completion 發 n_predict=1 + n_probs
│   ├── prompt.py         # 渲染 state + question（v1 §3.2）
│   └── calibrate.py      # temperature / Platt（CPU）
├── data/
│   ├── seeds/            # Fable 寫的素材：機台代碼表、句型、錯字表（yaml/json）
│   ├── gen_expand.py     # 模板組合展開
│   ├── hard/             # Fable 手寫的難例與孿生例子（jsonl）
│   └── synthetic/        # 合併後的 {task}.jsonl + MANIFEST.md
├── bench/
│   ├── smoke.py
│   ├── latency.py
│   ├── accuracy.py       # 只負責讀 logprob 存 raw；分析在 analyze.py
│   └── analyze.py        # CPU：指標、校準、學習曲線、基線
└── results/
```

### 3.3 `modal_app.py` 骨架

設計原則：**client 和 llama-server 跑在同一個容器內**，延遲量的是 localhost，不含 Modal 的網路層；沒有常駐 web endpoint，跑完就關，不燒閒置費。

```python
import modal, subprocess, time, json, os, urllib.request

MODEL_REPO = "unsloth/gemma-4-26b-a4b-it-GGUF"      # 預設；若之後知道 GB10 用哪份檔，換成同一份
MODEL_FILE = "gemma-4-26b-a4b-it-Q4_K_M.gguf"        # 檔名以 HF 頁面為準，下載後記實際大小（預估 15–17 GB）
GPU = "L4"                                           # v1 §0 的分支：FP8 → L40S；BF16 → A100/H100

app = modal.App("gb10-decide")
models = modal.Volume.from_name("gb10-decide-models", create_if_missing=True)
results = modal.Volume.from_name("gb10-decide-results", create_if_missing=True)

# llama.cpp 官方 CUDA server image；加 python 以便在同容器跑 client
image = (
    modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.11")
    .pip_install("huggingface_hub", "numpy", "requests")
    .add_local_dir("decide", remote_path="/root/decide")
    .add_local_dir("bench", remote_path="/root/bench")
    .add_local_dir("data/synthetic", remote_path="/root/data/synthetic")
)

@app.function(image=image, volumes={"/models": models}, timeout=60*60)
def download():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir="/models")
    models.commit()
    return os.path.getsize(p)

def start_server(n_parallel=4, ctx=4096, port=8080):
    # image 內 server 執行檔路徑以 image 為準（通常在 /app/llama-server），起不來就 `find / -name llama-server`
    proc = subprocess.Popen([
        "/app/llama-server", "-m", f"/models/{MODEL_FILE}",
        "-ngl", "99", "-c", str(ctx), "-np", str(n_parallel),
        "--port", str(port), "--host", "127.0.0.1",
    ])
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("llama-server did not become healthy")

@app.function(image=image, gpu=GPU, volumes={"/models": models, "/results": results}, timeout=60*60*2)
def run_bench(which: str, args: str = ""):
    """which ∈ {smoke, latency, accuracy}; 結果寫 /results/<which>/…，再 modal volume get 拉回"""
    import sys; sys.path.insert(0, "/root")
    proc = start_server()
    try:
        mod = __import__(f"bench.{which}", fromlist=["main"])
        mod.main(base_url="http://127.0.0.1:8080", out_dir=f"/results/{which}", args=args)
        results.commit()
    finally:
        proc.terminate()

@app.local_entrypoint()
def main(which: str = "smoke", args: str = ""):
    if which == "download":
        print("bytes:", download.remote()); return
    run_bench.remote(which, args)
```

執行順序：

```bash
modal run modal_app.py --which download
modal run modal_app.py --which smoke
modal run modal_app.py --which latency
modal run --detach modal_app.py --which accuracy      # 較長，用 detach 避免 Claude Code session 逾時
modal volume get gb10-decide-results / ./results/modal/
```

### 3.4 `decide/client.py` 核心（llama-server 原生 `/completion`）

```python
import requests, math, time

def read_option_probs(base_url, prompt, letters, n_probs=20, cache_prompt=True):
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/completion", json={
        "prompt": prompt, "n_predict": 1, "temperature": 0,
        "n_probs": n_probs, "cache_prompt": cache_prompt,
    }, timeout=120).json()
    dt = (time.perf_counter() - t0) * 1000
    # 回傳結構：r["completion_probabilities"][0]["top_logprobs"] 或 ["probs"]，依版本不同；smoke 時印出來確認
    top = r["completion_probabilities"][0]
    cand = top.get("top_logprobs") or top.get("probs")
    logp = {}
    for c in cand:
        tok = (c.get("token") or c.get("tok_str") or "").strip()
        if tok in letters:
            v = c.get("logprob", None)
            if v is None:  # 舊版給 prob
                v = math.log(max(c["prob"], 1e-12))
            logp[tok] = max(logp.get(tok, -1e9), v)
    missing = [l for l in letters if l not in logp]
    m = max(logp.values()) if logp else 0.0
    z = sum(math.exp(v - m) for v in logp.values())
    probs = {k: math.exp(v - m) / z for k, v in logp.items()}
    return {"probs": probs, "missing": missing, "latency_ms": dt,
            "first_token": r.get("content"), "prompt_tokens": r.get("tokens_evaluated"),
            "raw_logprobs": logp}
```

Smoke 時一定要看的三件事：`first_token` 是字母而不是 thinking token 或換行；`missing` 為空；用 `/tokenize` 確認 `A` 和 ` A` 的 token 是否不同，並據此決定 prompt 結尾要不要留空白。

---

## 4. M2：資料（Fable 出題，Gemma 作答）

出題者和受測者刻意不同家，避免「自己考自己」。**標準答案先定再寫**：Fable 先拿到 label，再寫情境；gold 來自出題規格，不是事後判斷。**不要拿 Fable 當作答對照組**，它出的題它自己答會偏高。

### 4.1 三層產法

| 層 | 誰做 | 產出 | 佔比 |
|---|---|---|---|
| 素材 | Fable（Claude Code 直接寫檔） | `data/seeds/`：機台代碼表（SMT/AOI/回焊/貼片…）、alarm 句型、SPC 描述模板、常見錯字與縮寫、英文 alarm code 表 | — |
| 展開 | `gen_expand.py` | 模板 × 隨機參數 → 每 task 的 easy 例子 | ~70% |
| 難例 | Fable 手寫 | `data/hard/`：訊息不完整、兩類邊界、夾錯字；每筆 easy 再配一筆「改一個事實讓 label 翻轉」的孿生例，記 `pair_id` | ~30% + 孿生 |

10 個 task 的定義沿用 v1 §4.1。每 task ≥ 200 筆，80% 台灣正體中文夾機台代碼，20% 純英文 alarm code。

### 4.2 每筆 jsonl 欄位

```json
{"id": "m_alarm_severity-0137", "task": "m_alarm_severity", "state": "SMT-L3 回焊爐 R-02 第 4 區溫度 245°C 超上限 10°C 持續 3 分鐘，板子仍在爐內",
 "question": {"type": "score", "instructions": "這則 alarm 的處置急迫度？", "options": {"A": "不急", "B": "盡快", "C": "停線"}},
 "gold": "C", "difficulty": "easy", "lang": "zh", "pair_id": null, "source": "expand"}
```

### 4.3 MANIFEST.md 要記

每 task 筆數、難度比例、語言比例、孿生對數、Fable 用的出題 prompt、隨機抽 10% 人工看過的不合理率。

### 4.4 選配：AI Studio 補產

Claude Code 用量吃緊時，`gen_expand.py` 可加一個 `--llm gemini` 模式，用 `gemma-4-26b-a4b-it` 或 `gemini` 系列補寫難例。**只用來寫情境，不用它標答案**。注意用 Gemma 補產會回到「自己考自己」，優先用 Gemini。

---

## 5. M3：延遲（L4 上量上界）

沿用 v1 §5 的 L1–L7，每組 200 次，warm-up 20 次，記 p50/p95/p99 與 `tokens_evaluated`。

差異：

- L4 = **多題共用 state** 要開／關 `cache_prompt` 各跑一次，回報倍數
- L6 = 在同容器內另開一個執行緒持續打 `/completion` 生成 512 token，同時量決策延遲；因為是 L4 不是 GB10，**只報「拖慢倍數」**，不報絕對值
- L7 = 同一題改走 `/v1/chat/completions` 生成 JSON 答案（`temperature 0`），比延遲
- 每張表都註明「L4，GB10 保守上界」

---

## 6. M4：準確率（只讀 logprob，分析全在 CPU）

`bench/accuracy.py` 只做一件事：每筆資料讀一次 option logprobs，把 **raw logprobs 全部存下來**（`results/accuracy/{task}.jsonl`）。之後所有分析、校準、學習曲線都在 `analyze.py` 用 CPU 重算，不用再開 GPU。

對照組同時跑：`/v1/chat/completions` 生成 JSON（每 task easy/hard 各 50 筆），存 accuracy 與延遲。

### 6.1 `analyze.py`（CPU，沿用 v1 §6 定義）

依 `pair_id` 切 calibration 50% / test 50%，孿生例子不跨集。輸出：

- 每 task × difficulty × lang：accuracy、macro-F1、AUROC（二元）、top-2、ECE（raw / temperature / Platt）、`missing` 率、選擇性準確率 @0.8 / @0.9 與 coverage
- reliability diagram → `results/fig/`
- 學習曲線：`m_alarm_category`、`x_ticket_route`、`q_spc_action`，N = 25/50/100/200/500，3 seeds；三條線 = TF-IDF+LR、typed decision 零樣本（N 筆只做校準）、typed decision 校準後 @0.9
- 規則基線：每 task 手寫 10–20 條關鍵字規則（Fable 寫、放 `data/rules/`），報 accuracy
- 哪些 task 校準後 argmax 仍不對 → 標記為「要改 criteria 或換模型」

---

## 7. M6：`results/REPORT.md`

填 v1 §7 的分流表與 Q1–Q8，並加一節「**本版的限制**」：

1. 延遲全部是 L4 上界，GB10 實測待補（約半小時，v1 §5 的 L1–L7 重跑一次）
2. 校準用的是合成資料，真實資料校準待補（v1 §6，只能在 GB10 做）
3. aarch64 與併發行為未驗證

---

## 8. 常見卡點與備援

| 卡點 | 處理 |
|---|---|
| Claude Code web 連不到 Modal | push repo 到 GitHub → Azure 開發 VM 執行 `modal run`；結果 commit 回 repo |
| `ghcr.io/ggml-org/llama.cpp:server-cuda` 起不來或 server 路徑不對 | `find / -name llama-server`；或改用 `nvidia/cuda:12.x-devel` image 自己 `cmake -DGGML_CUDA=ON` build |
| GGUF 檔名／大小與預期不符 | 以 HF 頁面為準，先 `huggingface_hub.list_repo_files` 列出來 |
| 第一個 token 是 thinking token | 在 `prompt.py` 渲染時關掉 thinking（chat template 參數或 system 指令），smoke 不過不進 M2 |
| `missing` 率高 | 提高 `n_probs`；檢查字母前後空白的 token 化 |
| L4 OOM | 降 `-c` 到 2048、`-np` 到 2；再不行換 L40S |
| Modal 搶佔中斷 | 結果逐筆 append 到 Volume，`accuracy.py` 支援 `--resume` |
| Claude Code session 逾時 | 長任務一律 `modal run --detach`，之後 `modal app logs` 查進度 |

---

## 9. 交付檢核

```
[ ] results/00-env.md            Modal 可用、image、GGUF 實際檔名與大小、llama-server 版本
[ ] results/01-smoke.md          first_token / missing / tokenize 檢查
[ ] data/synthetic/*.jsonl       10 task × ≥200 筆 + MANIFEST.md
[ ] results/03-latency.md        L1–L7（標註 L4 上界）
[ ] results/accuracy/*.jsonl     raw logprobs（之後 GB10 校準也用同格式）
[ ] results/04-accuracy.md       指標 + 基線 + reliability diagram
[ ] results/fig/learning-curve-*.png
[ ] results/REPORT.md            Q1–Q8 + 本版限制 + GB10 待補清單
[ ] results/cost.md              實際 Modal 費用與 GPU 小時（對照本文件 §2 估算）
```

## 10. 接回 GB10 時要做的（留給 v1）

1. Phase 0 盤點確切模型檔；若與 Modal 用的不同，M4 要在 GB10 重跑一次
2. L1–L7 重跑（約 30 分鐘）
3. 用 200–500 筆真實 alarm／工單做校準，覆寫 Q6 的門檻
