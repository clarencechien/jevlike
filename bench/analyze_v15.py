"""v15 analysis (docs/handoff-v15-shortlist-tournament.md; gates written before the run).

  python3 bench/analyze_v15.py  ->  results/20-shortlist-tournament.md, results/v15.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from embed_route import split  # noqa: E402
from math import comb  # noqa: E402

V14, V15, DATA = (os.path.join(ROOT, p) for p in ("results/modal/v14", "results/modal/v15", "data/v14"))
EG_GPU_MS = None


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def diagnose(te_ids, qs, src):
    """Where the gold is lost (not shortlisted / lost in its group / lost in the final), how many look-alikes (EG top-10 of
    that query) share the gold's first group, and an exact McNemar test of each arm against T100 on the same queries."""
    cat = [s["id"] for s in json.load(open(os.path.join(DATA, "sop_catalog.json"), encoding="utf-8"))["sops"]]
    gold = {q["id"]: q["sop_id"] for q in qs}
    top10 = json.load(open(os.path.join(DATA, "eg2_top10.json"), encoding="utf-8"))
    R = {k: {r["id"]: r for r in load(p) if r["id"] in te_ids} for k, p in src.items()}
    out = {}
    for k in ("H20", "H30", "H50", "T100"):
        if not R.get(k):
            continue
        rs = R[k].values()
        rec = (lambda r: r.get("gold_in_shortlist", True)) if k != "T100" else (lambda r: True)
        sl = json.load(open(os.path.join(DATA, f"eg2_top{k[1:]}.json"), encoding="utf-8")) if k != "T100" else None
        dens = []
        for i in te_ids:
            L = sorted(sl[i], key=cat.index) if sl else cat
            if gold[i] in L:
                g = L.index(gold[i]) // 10
                dens.append(sum(x in top10[i] for x in L[g * 10:g * 10 + 10] if x != gold[i]))
        b = sum(R["T100"][i]["correct"] and not R[k][i]["correct"] for i in te_ids)
        c = sum(not R["T100"][i]["correct"] and R[k][i]["correct"] for i in te_ids)
        n = b + c
        p = min(1.0, 2 * sum(comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
        out[k] = {"not_shortlisted": sum(not rec(r) for r in rs),
                  "lost_in_group": sum(rec(r) and not r["correct"] and not r["gold_in_final"] for r in rs),
                  "lost_in_final": sum(rec(r) and not r["correct"] and r["gold_in_final"] for r in rs),
                  "lookalikes_in_gold_group": float(np.mean(dens)), "vs_T100_only_T100_right": b, "vs_T100_only_this_right": c, "mcnemar_p": p}
    return out


def main():
    qs = load(os.path.join(DATA, "queries.jsonl")); te_ids = {q["id"] for q, t in zip(qs, ~split(qs)) if t}
    twin = {q["id"]: q["has_twin"] for q in qs}
    eg_gpu = json.load(open(os.path.join(V14, "eg_gpu/_run_cuda.json")))["latency_ms_batch1"]["p50"]
    src = {"H10": os.path.join(V14, "26b/route_hybrid.jsonl"), "H20": os.path.join(V15, "26b/route_shortlist20.jsonl"),
           "H30": os.path.join(V15, "26b/route_shortlist30.jsonl"), "H50": os.path.join(V15, "26b/route_shortlist50.jsonl"),
           "T100": os.path.join(V14, "26b/route_tournament.jsonl")}
    lat_src = {"H10": "route_shortlist10_test30.jsonl", "H20": "route_shortlist20_test30.jsonl", "H30": "route_shortlist30_test30.jsonl",
               "H50": "route_shortlist50_test30.jsonl", "T100": "route_tournament_test30.jsonl"}
    arms = {}
    for k, p in src.items():
        R = [r for r in load(p) if r["id"] in te_ids]
        if not R:
            continue
        L = load(os.path.join(V15, "26b", lat_src[k]))
        a = {"n": len(R), "top1": float(np.mean([r["correct"] for r in R])),
             "top1_twin": float(np.mean([r["correct"] for r in R if twin[r["id"]]])),
             "recall": float(np.mean([r.get("gold_in_shortlist", r["gold_in_final"]) if k != "T100" else 1.0 for r in R])),
             "acc_given_recalled": float(np.mean([r["correct"] for r in R if (r.get("gold_in_shortlist", r["gold_in_final"]) if k != "T100" else True)])),
             "reads": float(np.mean([r["reads"] for r in R]))}
        if L:
            lat = [r["latency_ms"] for r in L]
            a.update({"single_p50_ms": float(np.percentile(lat, 50)), "single_p95_ms": float(np.percentile(lat, 95)),
                      "single_per_read_ms": float(np.median([r["latency_ms"] / r["reads"] for r in L])),
                      "single_total_p50_ms": float(np.percentile(lat, 50)) + (eg_gpu if k != "T100" else 0.0)})
        arms[k] = a
    t = arms["T100"]["top1"]
    g1 = "H20" in arms and arms["H20"]["top1"] >= t - 0.03
    g2 = "H30" in arms and arms["H30"]["top1"] >= t - 0.01
    ok_k = [k for k in ("H10", "H20", "H30", "H50") if k in arms and arms[k]["top1"] >= t - 0.01]
    rec = ok_k[0] if ok_k else "T100"
    pred = {"H20_about_0.92": "H20" in arms and abs(arms["H20"]["top1"] - 0.92) <= 0.02, "H30_about_0.94": "H30" in arms and abs(arms["H30"]["top1"] - 0.94) <= 0.02,
            "H50_about_0.95": "H50" in arms and abs(arms["H50"]["top1"] - 0.95) <= 0.02,
            "per_read_150_250ms": "single_per_read_ms" in arms["T100"] and 150 <= arms["T100"]["single_per_read_ms"] <= 250}
    diag = diagnose(te_ids, qs, src)
    res = {"arms": arms, "diagnosis": diag, "eg_gpu_ms": eg_gpu, "gates": {"G1_top20_enough": g1, "G2_top30_equals_full": g2, "recommended": rec}, "predictions": pred}
    json.dump(res, open(os.path.join(ROOT, "results/v15.json"), "w"), ensure_ascii=False, indent=1)
    name = {"H10": "前 10 名", "H20": "前 20 名", "H30": "前 30 名", "H50": "前 50 名", "T100": "完整淘汰賽（100 份）"}
    L = ["# 20 — EmbeddingGemma 2 先縮到前 K 名，26B 打小型淘汰賽（v15）", "",
         "計劃與門檻：`docs/handoff-v15-shortlist-tournament.md`（跑前寫死）。資料與切分同 v14；26B UD-Q4_K_M、llama.cpp b11371、L4。"
         "前 10 名與完整淘汰賽沿用 v14 的跑法，前 20／30／50 名是本輪新跑；單則延遲另開一個容器，一次只跑一則，取 test 前 30 則。", "",
         "## 判定", "",
         f"- **G1 前 20 名夠用**（≥ 完整淘汰賽 − 0.03）：{arms.get('H20', {}).get('top1', float('nan')):.3f} vs {t:.3f} → {'✓' if g1 else '✗'}",
         f"- **G2 前 30 名等同完整淘汰賽**（≥ − 0.01）：{arms.get('H30', {}).get('top1', float('nan')):.3f} vs {t:.3f} → {'✓' if g2 else '✗'}",
         f"- **推薦設定**：{name[rec]}" + ("（最小的 K 達到完整淘汰賽 − 0.01）" if rec != "T100" else "（沒有任何 K 達到，EG 不當前置）"),
         "- 預測對答案：" + "、".join(f"{k} {'✓' if v else '✗'}" for k, v in pred.items()), "",
         "## 準確率對成本（test 150 則）", "",
         "| 做法 | top-1 | 近似孿生 | 正解在候選裡 | 在候選裡時答對 | 每則讀取 | 單則延遲 p50（含 EG 70 ms） | 每次讀取 |", "|---|---|---|---|---|---|---|---|"]
    for k in ("H10", "H20", "H30", "H50", "T100"):
        if k in arms:
            a = arms[k]
            L.append(f"| {name[k]} | **{a['top1']:.3f}** | {a['top1_twin']:.3f} | {a['recall']:.3f} | {a['acc_given_recalled']:.3f} | {a['reads']:.0f} | "
                     f"{a.get('single_total_p50_ms', float('nan')):.0f} ms | {a.get('single_per_read_ms', float('nan')):.0f} ms |")
    L += ["", "準確率是全部 150 則 test、4 則並行跑出來的；延遲是另外單則依序跑 test 前 30 則量的（兩者用同一支程式、同一個模型檔）。"
          "EG 的 70 ms 是 v14 在 L4 上量的單則編碼（批次 1），只加在用到 EG 的做法上。"
          "前 10 名的每次讀取（343 ms）偏高，因為它是延遲容器裡第一個跑的，前綴快取還是空的；其他四組約 220–250 ms。", "",
          "## 為什麼縮得越寬沒有越準（跑後診斷，不是預先寫好的門檻）", "",
          "| 做法 | 正解沒進候選 | 在第一輪小組被淘汰 | 在決賽輸掉 | 正解那組裡有幾個「長得像的」 | 只有完整淘汰賽答對／只有這組答對 | McNemar p |", "|---|---|---|---|---|---|---|"]
    for k in ("H20", "H30", "H50", "T100"):
        if k in diag:
            d = diag[k]
            L.append(f"| {name[k]} | {d['not_shortlisted']} | {d['lost_in_group']} | {d['lost_in_final']} | {d['lookalikes_in_gold_group']:.2f} | "
                     + ("—" if k == "T100" else f"{d['vs_T100_only_T100_right']}／{d['vs_T100_only_this_right']}") + " | "
                     + ("—" if k == "T100" else f"{d['mcnemar_p']:.3f}") + " |")
    L += ["", "「長得像的」＝這則訊息在 EG 前 10 名裡、又剛好和正解分在同一組的其他 SOP。",
          "完整淘汰賽照目錄順序分組，正解那組平均只有 1.55 個長得像的；EG 前 20 名再照目錄排，正解那組平均有 4.41 個。"
          "EG 把召回補回來了，但也把最難分的對手全塞進同一組、同一場決賽，26B 每一輪都在打硬仗。",
          "所以「召回 × 0.97」的推算錯在把 0.97 當常數：那是只讀一次前 10 名時的條件準確率，候選越像，這個數字越低（前 20 名 0.937、前 50 名 0.905）。",
          "150 則裡差 7–8 則，前 20／30 名對完整淘汰賽 p ≈ 0.06，前 50 名 p ≈ 0.008。", "",
          "**更正（v16 之後）**：上面「近親塞進同一組」的解釋被 v16（`21-dealt-groups.md`）推翻：把近親打散到不同組，第一輪被淘汰的沒有變少，總準確率也沒變好。"
          "前 20 名比完整淘汰賽多錯的 8 則裡，7 則是正解沒進前 20 名（召回）。"]
    open(os.path.join(ROOT, "results/20-shortlist-tournament.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
