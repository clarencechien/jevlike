"""v20 analysis (docs/handoff-v20-speed.md; predictions and rules written before the run).

  python3 bench/analyze_v20.py [--dir results/modal/v20]  ->  results/24-speed.md, results/v20.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASKS = ["x_ticket_route", "x_10way_intent", "m_alarm_category", "p_line_change", "eg2poc_status"]
NAME = {"x_ticket_route": "工單分派（4）", "x_10way_intent": "十種意圖（10）", "m_alarm_category": "alarm 原因分類（5）", "p_line_change": "停線原因（3）",
        "eg2poc_status": "專案狀態（eg2poc，4）"}
ARMS = ["llm_26b", "jev_26b", "jev_e4b", "eg_cuda", "eg_cpu"]
ANAME = {"llm_26b": "A LLM call 26B", "jev_26b": "C jevlike 26B", "jev_e4b": "B jevlike E4B", "eg_cuda": "D EG2 L4", "eg_cpu": "D′ EG2 CPU"}
L4_USD_PER_H = 0.80


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    a = dict(zip(*[iter(sys.argv[1:])] * 2)); d = os.path.join(ROOT, a.get("--dir", "results/modal/v20"))
    S = {}; rows = {}
    for arm in ARMS:
        p = os.path.join(d, arm, f"{arm}_summary.json")
        if os.path.exists(p):
            S[arm] = json.load(open(p)); rows[arm] = load(os.path.join(d, arm, f"{arm}.jsonl"))
    # eg2poc status: AUROC of P(B+C+D) against the pair label, for arms with probabilities
    au = {}
    for arm in S:
        R = [r for r in rows[arm] if r["task"] == "eg2poc_status" and r.get("probs")]
        if R:
            au[arm] = auroc([sum(v for k, v in r["probs"].items() if k != "A") for r in R], [bool(r["label"]) for r in R])
    g = lambda arm, t, *ks: _get(S.get(arm, {}).get("tasks", {}).get(t, {}), ks)  # noqa: E731

    def _get(x, ks):
        for k in ks:
            x = x.get(k) if isinstance(x, dict) else None
            if x is None:
                return None
        return x

    def ratio(x, y):
        return None if x is None or y in (None, 0) else x / y

    # rules
    r1 = {}
    for t in TASKS:
        acc_eg, acc_e4 = g("eg_cuda", t, "S1_acc"), g("jev_e4b", t, "S1_acc")
        sp = ratio(g("jev_e4b", t, "S2", "per_item_ms"), g("eg_cuda", t, "S2", "per_item_ms"))
        r1[t] = {"acc_eg": acc_eg, "acc_e4b": acc_e4, "acc_ok": acc_eg is not None and acc_e4 is not None and acc_eg >= acc_e4 - 0.05 - 1e-9,
                 "speed_x": sp, "speed_ok": sp is not None and sp >= 10, }
        r1[t]["pass"] = r1[t]["acc_ok"] and r1[t]["speed_ok"]
    d0 = [t for t in TASKS if t != "eg2poc_status"]
    acc_a = np.mean([g("llm_26b", t, "S1_acc") or 0 for t in d0]); acc_c = np.mean([g("jev_26b", t, "S1_acc") or 0 for t in d0])
    s1_ratio = np.mean([ratio(g("llm_26b", t, "S1", "p50"), g("jev_26b", t, "S1", "p50")) or 0 for t in d0])
    s2_ratio = np.mean([ratio(g("llm_26b", t, "S2", "per_item_ms"), g("jev_26b", t, "S2", "per_item_ms")) or 0 for t in d0])
    r2 = {"acc_llm": float(acc_a), "acc_jev26b": float(acc_c), "s1_slower_x": float(s1_ratio), "s2_slower_x": float(s2_ratio),
          "pass": abs(acc_a - acc_c) <= 0.01 + 1e-9 and s1_ratio >= 3 and s2_ratio >= 2}
    m = lambda arm, key: float(np.mean([v for v in (g(arm, t, *key) for t in TASKS) if v is not None])) if any(g(arm, t, *key) is not None for t in TASKS) else None  # noqa: E731
    pred = {"S1_llm_450_600": m("llm_26b", ("S1", "p50")) is not None and 450 <= m("llm_26b", ("S1", "p50")) <= 600,
            "S1_jev26b_about_140": m("jev_26b", ("S1", "p50")) is not None and 120 <= m("jev_26b", ("S1", "p50")) <= 160,
            "S1_e4b_about_65": m("jev_e4b", ("S1", "p50")) is not None and 55 <= m("jev_e4b", ("S1", "p50")) <= 80,
            "S1_eg_about_70": m("eg_cuda", ("S1", "p50")) is not None and 60 <= m("eg_cuda", ("S1", "p50")) <= 85,
            "S1_eg_not_faster_than_e4b": m("eg_cuda", ("S1", "p50")) is not None and m("jev_e4b", ("S1", "p50")) is not None and m("eg_cuda", ("S1", "p50")) >= m("jev_e4b", ("S1", "p50")) * 0.9,
            "S1_egcpu_250_350": m("eg_cpu", ("S1", "p50")) is not None and 250 <= m("eg_cpu", ("S1", "p50")) <= 350,
            "S2_eg_about_0.3s_per100": m("eg_cuda", ("S2", "wall_s")) is not None and m("eg_cuda", ("S2", "wall_s")) <= 0.5,
            "S2_e4b_about_3.5s": m("jev_e4b", ("S2", "wall_s")) is not None and 2.5 <= m("jev_e4b", ("S2", "wall_s")) <= 4.5,
            "S2_jev26b_about_7s": m("jev_26b", ("S2", "wall_s")) is not None and 5 <= m("jev_26b", ("S2", "wall_s")) <= 9,
            "S2_llm_15_25s": m("llm_26b", ("S2", "wall_s")) is not None and 15 <= m("llm_26b", ("S2", "wall_s")) <= 25,
            "S2_eg_10x_e4b": m("eg_cuda", ("S2", "per_item_ms")) is not None and m("jev_e4b", ("S2", "per_item_ms")) / m("eg_cuda", ("S2", "per_item_ms")) >= 10,
            "S2_eg_50x_llm": m("eg_cuda", ("S2", "per_item_ms")) is not None and m("llm_26b", ("S2", "per_item_ms")) / m("eg_cuda", ("S2", "per_item_ms")) >= 50,
            "S3_eg_about_0.02s": m("eg_cuda", ("S3", "wall_s")) is not None and m("eg_cuda", ("S3", "wall_s")) <= 0.05,
            "S3_jev26b_2_4s": m("jev_26b", ("S3", "wall_s")) is not None and 2 <= m("jev_26b", ("S3", "wall_s")) <= 4,
            "R1_only_eg2poc_passes": all((r1[t]["pass"] == (t == "eg2poc_status")) for t in TASKS),
            "eg2poc_eg_within_0.05_of_e4b": r1["eg2poc_status"]["acc_ok"]}
    res = {"summaries": S, "auroc_eg2poc_pBCD": au, "R1": r1, "R2": r2, "predictions": pred}
    json.dump(res, open(os.path.join(ROOT, "results/v20.json"), "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    f1 = lambda x: "—" if x is None else f"{x:.0f}"  # noqa: E731
    f2 = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    f3 = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    L = ["# 24 — 同一批題、同一張卡：LLM call、jevlike E4B、jevlike 26B、EG2 的速度（v20）", "",
         "計劃、預測與規則：`docs/handoff-v20-speed.md`（跑前寫死）。500 題（D0 四類 test 各 100 + eg2poc 專案狀態 100），L4；EG2 另在 4 vCPU 容器量一次。"
         "計時包整個請求；S1 一次一筆、S2 每題組 100 筆各引擎自己最好的批法（llama 4 slot、EG2 batch 64）、S3 同 100 筆換第二道題。", "",
         "## 規則", ""]
    for t in TASKS:
        x = r1[t]
        L.append(f"- **R1 {NAME[t]}**（EG2 準確率 ≥ E4B − 0.05 且批次 ≥ 10×）：準確率 {f3(x['acc_eg'])} 對 {f3(x['acc_e4b'])}（{'✓' if x['acc_ok'] else '✗'}）、批次快 {f2(x['speed_x'])}×（{'✓' if x['speed_ok'] else '✗'}） → {'**交給 EG2**' if x['pass'] else '不用 EG2'}")
    L += [f"- **R2 LLM call 的位置**（A 與 C 準確率差 ≤ 0.01，S1 慢 ≥ 3×、S2 慢 ≥ 2×）：D0 四類準確率 {f3(r2['acc_llm'])} 對 {f3(r2['acc_jev26b'])}；S1 慢 {f2(r2['s1_slower_x'])}×、S2 慢 {f2(r2['s2_slower_x'])}× → {'✓ 維持讀字母取代生成' if r2['pass'] else '✗'}",
          "- 預測對答案：" + "、".join(f"{k} {'✓' if v else '✗'}" for k, v in pred.items()), "",
          "## S1 單筆延遲 p50（ms；括號 p95）", "", "| 題組 | " + " | ".join(ANAME[a] for a in ARMS if a in S) + " |", "|---|" + "---|" * len([a for a in ARMS if a in S])]
    for t in TASKS:
        L.append(f"| {NAME[t]} | " + " | ".join(f"{f1(g(a, t, 'S1', 'p50'))}（{f1(g(a, t, 'S1', 'p95'))}）" for a in ARMS if a in S) + " |")
    L += ["", "## S2 批次 100 筆：每百筆秒數（每筆 ms）", "", "| 題組 | " + " | ".join(ANAME[a] for a in ARMS if a in S) + " |", "|---|" + "---|" * len([a for a in ARMS if a in S])]
    for t in TASKS:
        L.append(f"| {NAME[t]} | " + " | ".join(f"{f2(g(a, t, 'S2', 'wall_s'))} s（{f1(g(a, t, 'S2', 'per_item_ms'))}）" for a in ARMS if a in S) + " |")
    L += ["", "## S3 同 100 筆再問一題：每百筆秒數（jevlike／LLM 括號內是平均快取命中 token 數）", "", "| 題組 | " + " | ".join(ANAME[a] for a in ARMS if a in S) + " |", "|---|" + "---|" * len([a for a in ARMS if a in S])]
    for t in TASKS:
        cells = []
        for a in ARMS:
            if a not in S:
                continue
            w = g(a, t, "S3", "wall_s"); c = g(a, t, "S3", "cache_n_mean")
            cells.append(f"{f2(w)} s" + (f"（快取 {f1(c)}）" if c is not None else ""))
        L.append(f"| {NAME[t]} | " + " | ".join(cells) + " |")
    L += ["", "## 準確率（同一批題，S1 的答案）", "", "| 題組 | " + " | ".join(ANAME[a] for a in ARMS if a in S) + " |", "|---|" + "---|" * len([a for a in ARMS if a in S])]
    for t in TASKS:
        L.append(f"| {NAME[t]} | " + " | ".join(f3(g(a, t, "S1_acc")) for a in ARMS if a in S) + " |")
    L += ["", "專案狀態題照 eg2poc 的規則計分：答案 ≠ A 視為「狀態已改變」，對 pair label；P(B+C+D) 的 AUROC：" + "、".join(f"{ANAME[a]} {f3(v)}" for a, v in au.items()), "",
          "## 每百筆換算 L4 費用（$0.80／h；S2）", "", "| 引擎 | 每百筆秒數（五題組平均） | 每百筆美元 | 每百萬筆美元 |", "|---|---|---|---|"]
    for a in ARMS:
        if a in S and a != "eg_cpu":
            w = m(a, ("S2", "wall_s"))
            if w is not None:
                L.append(f"| {ANAME[a]} | {w:.2f} | ${w / 3600 * L4_USD_PER_H:.5f} | ${w / 3600 * L4_USD_PER_H * 1e4:.1f} |")
    L += ["", "載入與其他：" + "；".join(f"{ANAME[a]} 載入 {S[a].get('load_s', 0):.1f} s" + (f"、GPU 記憶體 {S[a]['gpu_mem_mb']:.0f} MB" if S[a].get("gpu_mem_mb") else "") for a in ("eg_cuda", "eg_cpu") if a in S),
          "E2B／E4B 的 CPU 數字引用 eg2poc（Xeon 8573C、4 threads、Q8_0）：每題 7.2 s／16.1 s，未重跑。"]
    open(os.path.join(ROOT, "results/24-speed.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
