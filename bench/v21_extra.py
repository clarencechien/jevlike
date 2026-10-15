"""v21 follow-up (pure computation on existing outputs, no GPU):
  1. does E4B / 26B add anything on top of Breeze's own logprob?  rank-average and leave-one-segment-out logistic combos
  2. proxy Chinese-error labels without human proofreading: Breeze differs from the reference AND the second Gemini pass
     agrees with the reference at that span (windows where gm36 also differs are excluded as "uncertain")
  python3 bench/v21_extra.py   -> results/v21_extra.json + markdown on stdout
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from analyze_v21 import auroc, load_jsonl, pB  # noqa: E402

SEGS = ["T1", "T2", "T3"]
W = load_jsonl(os.path.join(ROOT, "data/v21/windows.jsonl")); ids = [w["id"] for w in W]
J = {t: {r["id"]: r for r in load_jsonl(os.path.join(ROOT, f"results/modal/v21/{t}/judge.jsonl"))} for t in ("e4b_text", "26b_text", "e4b_audio")}
seg = np.array([w["seg"] for w in W]); E = np.array([w["E"] for w in W], bool); C = np.array([w["C"] for w in W], bool)
feat = {"d1_mean": np.array([-w["d1_mean"] for w in W]), "d1_min": np.array([-w["d1_min"] for w in W]), "d1_low": np.array([w["d1_low"] for w in W]),
        "d5": np.array([w["d5_disagree"] for w in W]),
        "e4b_q1": np.array([pB(J["e4b_text"][i], "q1") for i in ids]), "e4b_q2": np.array([pB(J["e4b_text"][i], "q2") for i in ids]),
        "26b_q1": np.array([pB(J["26b_text"][i], "q1") for i in ids]), "26b_q2": np.array([pB(J["26b_text"][i], "q2") for i in ids]),
        "e4b_audio_q1": np.array([pB(J["e4b_audio"][i], "q1") for i in ids])}


def rank(x):
    o = np.argsort(np.argsort(x)); return o / (len(x) - 1)


def loso_logistic(names, y):
    """Leave-one-segment-out logistic regression on standardized features; pooled test AUROC."""
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([feat[n] for n in names]); p = np.zeros(len(y))
    for s in SEGS:
        tr, te = seg != s, seg == s
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        lr = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr]); p[te] = lr.predict_proba((X[te] - mu) / sd)[:, 1]
    return auroc(p, y)


out = {"part1": {}, "part2": {}}
# ---- part 1
rows = []
for lab, y, q in (("E", E, "q1"), ("C", C, "q2")):
    base = feat["d1_mean"]; r = {"label": lab, "d1_mean": auroc(base, y), "d1_min": auroc(feat["d1_min"], y), "d5": auroc(feat["d5"], y)}
    for m in ("e4b", "26b"):
        s = feat[f"{m}_{q}"]; r[f"{m}"] = auroc(s, y)
        r[f"rankavg_d1+{m}"] = auroc(rank(base) + rank(s), y)
        r[f"rankavg_d1+d1min+{m}"] = auroc(rank(base) + rank(feat["d1_min"]) + rank(s), y)
        r[f"logit_d1x3+{m}"] = loso_logistic(["d1_mean", "d1_min", "d1_low", f"{m}_{q}"], y)
    r["rankavg_d1+d5"] = auroc(rank(base) + rank(feat["d5"]), y)
    r["logit_d1x3"] = loso_logistic(["d1_mean", "d1_min", "d1_low"], y)
    r["logit_d1x3+d5"] = loso_logistic(["d1_mean", "d1_min", "d1_low", "d5"], y)
    r["logit_all"] = loso_logistic(["d1_mean", "d1_min", "d1_low", "d5", f"e4b_{q}", f"26b_{q}"], y)
    rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
out["part1"] = rows
# ---- part 2: proxy C
noise = np.array([w["noise_C"] for w in W], bool)
keep = ~(C & noise)            # drop windows where Breeze differs but gm36 also differs there (uncertain)
pos = C & ~noise               # Breeze differs, gm36 agrees with the reference
neg_uncertain = (~C) & noise   # Breeze agrees with reference but gm36 differs: keep as negative (Breeze matched the reference there)
Cp = pos
p2 = {"windows": len(W), "C_original": int(C.sum()), "C_proxy_positive": int(pos.sum()), "excluded_uncertain": int((C & noise).sum()),
      "negatives": int((keep & ~pos).sum()), "note": "positive = Breeze differs from reference (>=2 CJK replace) and the second Gemini pass agrees with the reference in that span; windows where both differ are excluded"}
for n in ("d1_mean", "d1_min", "d5", "e4b_q2", "26b_q2"):
    p2[f"auroc_{n}_proxy"] = round(auroc(feat[n][keep], Cp[keep]), 3); p2[f"auroc_{n}_original"] = round(auroc(feat[n], C), 3)
p2["logit_d1x3_proxy"] = None
# per condition counts
p2["per_cond"] = {c: {"pos": int(sum(1 for w, p_ in zip(W, pos) if w["cond"] == c and p_)), "excluded": int(sum(1 for w, x in zip(W, C & noise) if w["cond"] == c and x))} for c in ("M0", "M1", "M2", "M3")}
out["part2"] = p2
json.dump(out, open(os.path.join(ROOT, "results/v21_extra.json"), "w"), ensure_ascii=False, indent=1)
# ---- markdown
print("### 1. 把握度加讀字母合起來（AUROC；留一段三折的邏輯斯迴歸）\n")
keys = ["d1_mean", "d1_min", "d5", "e4b", "26b", "rankavg_d1+e4b", "rankavg_d1+26b", "logit_d1x3", "logit_d1x3+e4b", "logit_d1x3+26b", "logit_d1x3+d5", "logit_all"]
print("| 標籤 | " + " | ".join(keys) + " |"); print("|---|" + "---|" * len(keys))
for r in rows:
    print(f"| {r['label']} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
print("\n### 2. 中文錯字的代理正解（Breeze 不同、第二路 Gemini 與正解一致）\n")
print(f"原本 C {p2['C_original']} 個視窗 → 代理正例 {p2['C_proxy_positive']}、排除不確定 {p2['excluded_uncertain']}、負例 {p2['negatives']}。")
print("| 偵測器 | 原本 C | 代理 C |"); print("|---|---|---|")
for n in ("d1_mean", "d1_min", "d5", "e4b_q2", "26b_q2"):
    print(f"| {n} | {p2[f'auroc_{n}_original']:.3f} | {p2[f'auroc_{n}_proxy']:.3f} |")
