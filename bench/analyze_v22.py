"""v22 analysis: arms x engines on the English-error label (E) and the proxy Chinese label; word-level accuracy; rank-average
with Breeze's logprob; cascade; gates G1-G6 + kill.  python3 bench/analyze_v22.py -> results/26-glossary-words.md, results/v22.json"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from analyze_v21 import auroc, load_jsonl, pB  # noqa: E402

RES = os.path.join(ROOT, "results/modal/v22"); ARMS = ["glossary", "glossary_mixed", "words", "words_glossary"]; ENG = ["e4b", "26b"]
W = load_jsonl(os.path.join(ROOT, "data/v21/windows.jsonl")); ids = [w["id"] for w in W]; N = len(W)
E = np.array([w["E"] for w in W], bool); C = np.array([w["C"] for w in W], bool); noise = np.array([w["noise_C"] for w in W], bool)
keepC = ~(C & noise); Cp = C & ~noise
lp = np.array([-w["d1_mean"] for w in W]); has_latin = np.array([w["has_latin"] for w in W], bool)
V21 = {t: {r["id"]: r for r in load_jsonl(os.path.join(ROOT, f"results/modal/v21/{t}/judge.jsonl"))} for t in ("e4b_text", "26b_text")}


def rank(x):
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def cascade(arm_score, frac=0.4):
    """Ask the arm only on the `frac` least-confident windows (by logprob); others keep their logprob rank below them."""
    r = rank(lp); cut = np.quantile(r, 1 - frac); low = r >= cut
    s = r.copy(); sub = (rank(lp[low]) + rank(arm_score[low])) / 2
    s[low] = cut + (1 - cut) * sub
    return s


def score_of(recs):
    s = np.full(N, 0.0)
    for i, w in enumerate(W):
        r = recs.get(w["id"])
        if not r or "res" not in r or not r["res"].get("probs"):
            continue
        s[i] = (1 - r["p_ok"]) if r.get("words") else (r.get("p_err") or 0.0)
    return s


R = {"arms": {}, "gates": {}}
base = {e: np.array([pB(V21[f"{e}_text"][i], "q1") for i in ids]) for e in ENG}
for e in ENG:
    R["arms"][f"{e}_v21"] = {"auroc_E": round(auroc(base[e], E), 3), "auroc_E_plus_lp": round(auroc(rank(base[e]) + rank(lp), E), 3),
                             "auroc_Cproxy": round(auroc(base[e][keepC], Cp[keepC]), 3)}
R["arms"]["logprob"] = {"auroc_E": round(auroc(lp, E), 3), "auroc_Cproxy": round(auroc(lp[keepC], Cp[keepC]), 3)}
for e in ENG:
    for arm in ARMS:
        recs = {r["id"]: r for r in load_jsonl(os.path.join(RES, f"{e}_{arm}", "judge.jsonl"))}
        if not recs:
            continue
        s = score_of(recs); d = {"n": len(recs), "asked": sum(1 for r in recs.values() if "res" in r), "errors": sum(1 for r in recs.values() if "res" in r and "error" in r["res"]),
                                 "option_mass_mean": round(float(np.mean([r["res"]["option_mass"] for r in recs.values() if "res" in r and "option_mass" in r["res"]])), 4)}
        d["auroc_E"] = round(auroc(s, E), 3); d["auroc_E_latin_only"] = round(auroc(s[has_latin], E[has_latin]), 3)
        d["auroc_E_plus_lp"] = round(auroc(rank(s) + rank(lp), E), 3); d["auroc_E_cascade40"] = round(auroc(cascade(s), E), 3)
        d["auroc_Cproxy"] = round(auroc(s[keepC], Cp[keepC]), 3)
        d["per_cond_E"] = {c: round(auroc(s[[w["cond"] == c for w in W]], E[[w["cond"] == c for w in W]]), 3) for c in ("M0", "M1", "M2", "M3")}
        lat = sorted(r["res"]["latency_ms"] for r in recs.values() if "res" in r and "latency_ms" in r["res"]); d["latency_p50_ms"] = round(lat[len(lat) // 2], 1) if lat else None
        if arm.startswith("words"):
            g = [(r, w) for w in W for r in [recs.get(w["id"])] if r and r.get("gold_words") and r.get("pred_word")]
            d["word_n"] = len(g); d["word_top1"] = round(float(np.mean([r["pred_word"] in r["gold_words"] for r, _ in g])), 3) if g else None
            # among windows with gold words: how often is "all ok" NOT the argmax (i.e. it flags something)
            flag = [max(r["res"]["probs"], key=r["res"]["probs"].get) != r["ok_letter"] for r, _ in g]
            d["word_flagged_when_error"] = round(float(np.mean(flag)), 3) if g else None
            ok_pos = [r["res"]["probs"].get(r["ok_letter"], 0) for r in recs.values() if "res" in r and r.get("words")]
            d["p_ok_mean"] = round(float(np.mean(ok_pos)), 3)
            d["p_ok_mean_by_n_options"] = {k: round(float(np.mean([r["res"]["probs"].get(r["ok_letter"], 0) for r in recs.values() if "res" in r and r.get("words") and len(r["letters"]) == k])), 3)
                                           for k in range(2, 11) if any("res" in r and r.get("words") and len(r["letters"]) == k for r in recs.values())}
        R["arms"][f"{e}_{arm}"] = d

A = R["arms"]; G = R["gates"]
def best(e, arms):  # noqa: E302
    c = [(A[f"{e}_{a}"]["auroc_E"], a) for a in arms if f"{e}_{a}" in A]; return max(c) if c else (None, None)
bE, bA = best("e4b", ["glossary", "words_glossary"]); bEall, bAall = best("e4b", ARMS); b26, b26A = best("26b", ARMS)
G["G1"] = {"e4b_best_glossary_arm": bA, "auroc": bE, "v21": A["e4b_v21"]["auroc_E"], "pass": bE is not None and bE >= 0.80 and bE - A["e4b_v21"]["auroc_E"] >= 0.05}
plus = {k: v["auroc_E_plus_lp"] for k, v in A.items() if "auroc_E_plus_lp" in v}; bp = max(plus.items(), key=lambda kv: kv[1]) if plus else (None, None)
G["G2"] = {"best_plus_lp": bp[0], "auroc": bp[1], "v21_lp_plus_26b": A["26b_v21"]["auroc_E_plus_lp"], "pass": bp[1] is not None and bp[1] >= 0.85}
gE = A.get("e4b_glossary", {}).get("auroc_E"); G["G3"] = {"e4b_glossary": gE, "26b_v21": A["26b_v21"]["auroc_E"], "pass": gE is not None and gE >= A["26b_v21"]["auroc_E"]}
G["G4"] = {e: {"glossary": A.get(f"{e}_glossary", {}).get("auroc_E"), "mixed": A.get(f"{e}_glossary_mixed", {}).get("auroc_E")} for e in ENG}
G["G4"]["pass"] = all(G["G4"][e]["glossary"] is not None and G["G4"][e]["mixed"] is not None and G["G4"][e]["glossary"] - G["G4"][e]["mixed"] <= 0.03 for e in ENG)
wt = {e: A.get(f"{e}_words", {}).get("word_top1") for e in ENG}; G["G5"] = {"word_top1": wt, "pass": any(v is not None and v >= 0.60 for v in wt.values())}
cas = {k: (v["auroc_E_plus_lp"], v["auroc_E_cascade40"]) for k, v in A.items() if "auroc_E_cascade40" in v}
G["G6"] = {"full_vs_cascade": cas, "pass": bool(cas) and all(full - c40 <= 0.02 for full, c40 in cas.values())}
G["kill"] = {"e4b_best": bEall, "26b_best": b26, "triggered": (bEall is None or bEall - A["e4b_v21"]["auroc_E"] < 0.05) and (b26 is None or b26 < 0.82)}
json.dump(R, open(os.path.join(ROOT, "results/v22.json"), "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

f3 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))  # noqa: E731
L = ["# 26 — v22：給它詞表、把英文字列成選項（Modal L4，2026-10-15）", "",
     "規則、預測與門檻：`docs/handoff-v22-glossary-words.md`。語料與標籤沿用 v21 的凍結檔。數字：`results/v22.json`、`results/modal/v22/`。", "",
     "## 1. 臂 × 引擎（英文錯 AUROC；代理中文錯字只報）", "",
     "| 臂 | 引擎 | E | E（只算有英文字的 267 個） | E + logprob 排名平均 | 級聯只問 40% | 代理 C | 詞層 top-1（n） | 每題 p50 |", "|---|---|---|---|---|---|---|---|---|",
     f"| （對照）Breeze logprob | — | {f3(A['logprob']['auroc_E'])} | | | | {f3(A['logprob']['auroc_Cproxy'])} | | 免費 |"]
for e in ENG:
    v = A[f"{e}_v21"]; L.append(f"| （對照）v21 原題 | {e} | {f3(v['auroc_E'])} | | {f3(v['auroc_E_plus_lp'])} | | {f3(v['auroc_Cproxy'])} | | |")
    for arm in ARMS:
        d = A.get(f"{e}_{arm}")
        if d:
            L.append(f"| {arm} | {e} | **{f3(d['auroc_E'])}** | {f3(d['auroc_E_latin_only'])} | {f3(d['auroc_E_plus_lp'])} | {f3(d['auroc_E_cascade40'])} | {f3(d['auroc_Cproxy'])} | {f3(d.get('word_top1'))}（{d.get('word_n', '')}） | {d['latency_p50_ms']} ms |")
L += ["", "每條件 E：" + "；".join(f"{k} " + "、".join(f"{c} {f3(x)}" for c, x in v["per_cond_E"].items()) for k, v in A.items() if "per_cond_E" in v), ""]
for k, v in A.items():
    if "p_ok_mean_by_n_options" in v:
        L.append(f"- {k}：「都對」（最後一個字母）的平均機率 {f3(v['p_ok_mean'])}；依選項數：" + "、".join(f"{n} 選 {f3(p)}" for n, p in v["p_ok_mean_by_n_options"].items()) + f"；有錯字的視窗裡標出東西的比例 {f3(v['word_flagged_when_error'])}")
L += ["", "## 2. 門檻", "", "| 門檻 | 數字 | 判定 |", "|---|---|---|",
      f"| G1 詞表有用 | E4B 最佳詞表臂 {G['G1']['e4b_best_glossary_arm']} {f3(G['G1']['auroc'])}，v21 原題 {f3(G['G1']['v21'])} | {'✓' if G['G1']['pass'] else '✗'} |",
      f"| G2 贏過上一輪的組合 | 最佳臂 + logprob {G['G2']['best_plus_lp']} {f3(G['G2']['auroc'])}（v21 補算 logprob + 26B {f3(G['G2']['v21_lp_plus_26b'])}） | {'✓' if G['G2']['pass'] else '✗'} |",
      f"| G3 小模型加詞表 ≥ 大模型不加 | E4B 詞表 {f3(G['G3']['e4b_glossary'])} 對 26B 原題 {f3(G['G3']['26b_v21'])} | {'✓' if G['G3']['pass'] else '✗'} |",
      f"| G4 混合詞表站得住 | " + "；".join(f"{e} 精確 {f3(G['G4'][e]['glossary'])} 混合 {f3(G['G4'][e]['mixed'])}" for e in ENG) + f" | {'✓' if G['G4']['pass'] else '✗'} |",
      f"| G5 列字題型 | 詞層 top-1：" + "、".join(f"{e} {f3(v)}" for e, v in G['G5']['word_top1'].items()) + f" | {'✓' if G['G5']['pass'] else '✗'} |",
      f"| G6 級聯 | 全問對只問 40%：" + "；".join(f"{k} {f3(a)}→{f3(b)}" for k, (a, b) in G['G6']['full_vs_cascade'].items()) + f" | {'✓' if G['G6']['pass'] else '✗'} |",
      f"| Kill | E4B 最佳 {f3(G['kill']['e4b_best'])}、26B 最佳 {f3(G['kill']['26b_best'])} | {'觸發' if G['kill']['triggered'] else '未觸發'} |", ""]
open(os.path.join(ROOT, "results/26-glossary-words.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
