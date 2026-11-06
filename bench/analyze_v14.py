"""v14 analysis (docs/handoff-v14-sop-routing.md; gates and predictions written before the data existed).

  python3 bench/analyze_v14.py  ->  results/19-sop-routing.md, results/v14.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from embed_route import split  # noqa: E402

V14 = os.path.join(ROOT, "results/modal/v14")
DATA = os.path.join(ROOT, "data/v14")
GRID = np.exp(np.linspace(math.log(1e-3), math.log(5), 300))


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)


def fit_tau(Z, y):
    v = [-np.log(np.clip(softmax(Z / t)[np.arange(len(y)), y], 1e-12, 1)).mean() for t in GRID]
    return float(GRID[int(np.argmin(v))])


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def main():
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, HERE)
    from thresholds import risk_threshold
    qs = load_jsonl(os.path.join(DATA, "queries.jsonl")); qid = [q["id"] for q in qs]
    twin = np.array([q["has_twin"] for q in qs]); cal = split(qs); te = ~cal
    d = np.load(os.path.join(V14, "eg_cpu/eg_route.npz")); y = d["gold"]; S = d["S_eg"]
    rank = np.argsort(-S, 1)
    rec = {k: float(np.mean([y[i] in rank[i, :k] for i in np.where(te)[0]])) for k in (1, 5, 10, 20, 30, 50)}
    tau = fit_tau(S[cal], y[cal]); P = softmax(S / tau); conf = P.max(1); ok = P.argmax(1) == y
    cut, _ = risk_threshold(conf[cal], ok[cal], 0.05); h = conf[te] >= cut
    eg = {"top1": rec[1], "recall5": rec[5], "recall10": rec[10], "recall_curve": rec, "top1_twin": float(ok[te & twin].mean()), "top1_nontwin": float(ok[te & ~twin].mean()),
          "top1_cls": float((d["S_cls"].argmax(1)[te] == y[te]).mean()), "auroc": float(roc_auc_score(ok[te], conf[te])),
          "coverage_eps5": float(h.mean()), "error_eps5": float((~ok[te][h]).mean()) if h.any() else 0.0,
          "by_style": {s: float(ok[te & (np.array([q["style"] for q in qs]) == s)].mean()) for s in (0, 1, 2)}}
    run_cpu = json.load(open(os.path.join(V14, "eg_cpu/_run_cpu.json")))
    gpu_p = os.path.join(V14, "eg_gpu/_run_cuda.json")
    run_gpu = json.load(open(gpu_p)) if os.path.exists(gpu_p) else None
    if run_gpu and os.path.exists(os.path.join(V14, "eg_gpu/eg_route.npz")):
        g = np.load(os.path.join(V14, "eg_gpu/eg_route.npz"))
        eg["gpu_same_top1_rate"] = float((g["S_eg"].argmax(1) == S.argmax(1)).mean())
    arms = {}
    idx = {k: i for i, k in enumerate(qid)}
    for name in ("e4b", "26b"):
        for mode in ("tournament", "hybrid"):
            R = load_jsonl(os.path.join(V14, name, f"route_{mode}.jsonl"))
            if not R:
                continue
            m_te = [r for r in R if te[idx[r["id"]]]]
            lat = [r["latency_ms"] for r in m_te]
            arms[f"{'T' if mode == 'tournament' else 'H'}-{name.upper()}"] = {
                "n_test": len(m_te), "top1": float(np.mean([r["correct"] for r in m_te])),
                "top1_twin": float(np.mean([r["correct"] for r in m_te if twin[idx[r["id"]]]])),
                "top1_nontwin": float(np.mean([r["correct"] for r in m_te if not twin[idx[r["id"]]]])),
                "gold_in_final": float(np.mean([r["gold_in_final"] for r in m_te])), "reads": float(np.mean([r["reads"] for r in m_te])),
                "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95)),
                "mass_min": float(min(r["final_mass"] or 0 for r in m_te))}
    t26 = arms.get("T-26B"); h26 = arms.get("H-26B")
    def in_short(name):
        R = [r for r in load_jsonl(os.path.join(V14, name, "route_hybrid.jsonl")) if te[idx[r["id"]]] and r["gold_in_final"]]
        return float(np.mean([r["correct"] for r in R])) if R else float("nan")
    h26_in, he4_in = in_short("26b"), in_short("e4b")
    t26_read = float(np.median([r["latency_ms"] / r["reads"] for r in load_jsonl(os.path.join(V14, "26b", "route_tournament.jsonl"))]))
    eg_gpu_ms = run_gpu["latency_ms_batch1"]["p50"] if run_gpu else None
    g1 = bool(t26 and eg_gpu_ms is not None and eg["top1"] >= t26["top1"] - 0.05 and eg_gpu_ms <= t26["latency_p50_ms"] / 10)
    g2 = bool(t26 and h26 and h26["top1"] >= t26["top1"] - 0.01)
    g3 = eg["recall10"] >= 0.95
    pred = {"eg_top1_0.75_0.90": 0.75 <= eg["top1"] <= 0.90, "eg_recall10_ge_0.95": g3, "eg_twin_le_0.70": eg["top1_twin"] <= 0.70,
            "t26_ge_0.90": bool(t26 and t26["top1"] >= 0.90), "t_e4b_about_0.80": bool(arms.get("T-E4B") and abs(arms["T-E4B"]["top1"] - 0.80) <= 0.05),
            "h26_within_1pt_of_t26": g2, "eg_gpu_le_20ms": bool(eg_gpu_ms is not None and eg_gpu_ms <= 20)}
    res = {"eg": eg, "arms": arms, "eg_cpu_run": run_cpu, "eg_gpu_run": run_gpu,
           "gates": {"G1_home_ground": g1, "G2_hybrid_standard": g2, "G3_recall10": g3}, "predictions": pred}
    json.dump(res, open(os.path.join(ROOT, "results/v14.json"), "w"), ensure_ascii=False, indent=1)
    L = ["# 19 — 100 份 SOP 路由：EmbeddingGemma 2 的主場 vs E4B、26B（v14）", "",
         "計劃與門檻：`docs/handoff-v14-sop-routing.md`（資料產生前寫死並 commit）。資料 `data/v14/`：Gemini 寫的 100 份 SOP（20 組近似孿生）與 300 則現場訊息；"
         "依 SOP 分組切半，test 150 則、選項永遠 100 份。LLM：llama.cpp b11371、L4、`--swa-full`；一則訊息內的讀取依序進行，四則訊息並行。", "",
         "## 判定", "",
         f"- **G1 這題是 EG 的主場**（EG top-1 ≥ T-26B − 0.05 且 GPU 延遲 ≤ T-26B 的 1/10）：EG {eg['top1']:.3f} vs T-26B {f(t26 and t26['top1'])}；"
         f"EG GPU {f(eg_gpu_ms, 1)} ms vs T-26B {f(t26 and t26['latency_p50_ms'], 0)} ms → {'✓' if g1 else '✗'}",
         f"- **G2 混合式當標準做法**（H-26B ≥ T-26B − 0.01）：{f(h26 and h26['top1'])} vs {f(t26 and t26['top1'])} → {'✓' if g2 else '✗'}",
         f"- **G3 EG 召回夠當前置**（recall@10 ≥ 0.95）：{eg['recall10']:.3f} → {'✓' if g3 else '✗'}",
         "- 預測對答案：" + "、".join(f"{k} {'✓' if v else '✗'}" for k, v in pred.items()), "",
         "## 準確率與成本（test 150 則）", "",
         "| 做法 | top-1 | 近似孿生 | 其他 | 正解有進最後一輪 | 每則 LLM 讀取 | 每則延遲 p50 |", "|---|---|---|---|---|---|---|",
         f"| EG（EmbeddingGemma 2，CPU 參考） | {eg['top1']:.3f} | {eg['top1_twin']:.3f} | {eg['top1_nontwin']:.3f} | recall@10 {eg['recall10']:.3f} | 0 | "
         f"CPU {run_cpu['latency_ms_batch1']['p50']:.0f} ms／L4 {f(eg_gpu_ms, 1)} ms |"]
    for k in ("T-E4B", "T-26B", "H-E4B", "H-26B"):
        if k in arms:
            a = arms[k]
            L.append(f"| {k} | {a['top1']:.3f} | {a['top1_twin']:.3f} | {a['top1_nontwin']:.3f} | {a['gold_in_final']:.3f} | {a['reads']:.0f} | {a['latency_p50_ms']:.0f} ms |")
    L += ["", "T = 淘汰賽（每 10 份一組讀字母，10 個勝出者再讀一次，共 11 次）；H = 混合（EG 前 10 名依目錄順序給 LLM 讀一次）。H 的延遲只算 LLM，另加 EG 一次編碼。", "",
          "## EmbeddingGemma 2 細項", "",
          f"- recall@1／@5／@10：{eg['top1']:.3f}／{eg['recall5']:.3f}／{eg['recall10']:.3f}；對稱分類前綴（v13 的 Z1）top-1 {eg['top1_cls']:.3f}。",
          f"- 依訊息風格 top-1：直接描述 {eg['by_style'][0]:.3f}、口語縮寫 {eg['by_style'][1]:.3f}、夾雜無關資訊 {eg['by_style'][2]:.3f}。",
          f"- 信心 AUROC {eg['auroc']:.2f}；5% 錯誤預算下自動處理 {eg['coverage_eps5']:.0%}（其中錯 {eg['error_eps5']:.0%}）。",
          f"- 速度：CPU（4 核、fp32）單則 p50 {run_cpu['latency_ms_batch1']['p50']:.0f} ms、批次 {run_cpu['throughput_per_s_batch64']} 則／秒；"
          + (f"L4 GPU 單則 p50 {run_gpu['latency_ms_batch1']['p50']:.1f} ms、批次 {run_gpu['throughput_per_s_batch64']} 則／秒；GPU 與 CPU 的 top-1 一致 {eg.get('gpu_same_top1_rate', float('nan')):.1%}。" if run_gpu else "GPU 未跑。"),
          "- 資料抽查 30 則有 2 則邊界（`data/v14/MANIFEST.md`），照實計分。",
          f"- GPU 與 CPU 的 top-1 一致 {eg.get('gpu_same_top1_rate', float('nan')):.0%}，不是 100%：fp32 在兩種硬體上的數值誤差讓少數接近同分的題換了名次。判定一律用 CPU 參考跑的排名。",
          "", "## 拆解（跑完後補算，不判定）", "",
          f"- **混合式輸在召回，不在 LLM**：正解有進 EG 前 10 名時，H-26B 答對 {h26_in:.3f}、H-E4B {he4_in:.3f}，與淘汰賽的 T-26B {f(t26 and t26['top1'])} 同級；"
          f"輸掉的部分是 EG 前 10 名漏掉正解的 {1 - eg['recall10']:.0%}。",
          "- **EG 的召回曲線**：" + "、".join(f"前 {k} 名 {v:.3f}" for k, v in rec.items()) + "。"
          f"若混合式改成取前 20 名（2 組 + 決賽 = 3 次讀取）或前 30 名（4 次），依上面兩個數字相乘推算約 {rec[20] * h26_in:.2f}／{rec[30] * h26_in:.2f}，"
          "讀取次數是淘汰賽的 1/4 到 1/3。這是推算，沒有實跑。",
          "- **延遲是吞吐量受限的數字**：LLM 臂是四則訊息同時在跑，每則 11 次讀取依序進行；L4 上 llama-server 並行不會變快（v2 已量過），"
          f"所以 T-26B 每次讀取約 {t26_read:.0f} ms、遠高於單題 137 ms。單則依序跑會快很多，但 G1 輸在準確率，不受影響。"]
    open(os.path.join(ROOT, "results/19-sop-routing.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
