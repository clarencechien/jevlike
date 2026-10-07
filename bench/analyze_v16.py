"""v16 analysis (docs/handoff-v16-dealt-groups.md; gates written before the run).

  python3 bench/analyze_v16.py  ->  results/21-dealt-groups.md, results/v16.json
"""
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from embed_route import split  # noqa: E402

V14, V15, V16, DATA = (os.path.join(ROOT, p) for p in ("results/modal/v14", "results/modal/v15", "results/modal/v16", "data/v14"))
ARMS = ("H20", "D20", "S20", "P20", "D100", "T100")
NAME = {"H20": "前 20 名照目錄切 2 組（v15）", "D20": "前 20 名發牌 2 組", "S20": "前 20 名發牌 4 組 × 5 份", "P20": "前 20 名發牌 4 組 × 5 份 + 5 份不相干",
        "D100": "全部 100 份發牌 10 組", "T100": "全部 100 份照目錄 10 組（v14）"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def mcnemar(A, B, ids):
    b = sum(A[i]["correct"] and not B[i]["correct"] for i in ids); c = sum(not A[i]["correct"] and B[i]["correct"] for i in ids)
    n = b + c
    return b, c, (min(1.0, 2 * sum(comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n) if n else 1.0)


def gold_reads(arm, r, cat, sl20):
    """The reads the gold took part in, as member lists: its first-round group, then the final if it got there."""
    g = r["gold"]
    if arm in ("D20", "S20", "P20", "D100"):
        first = next((s["members"] for s in r["stages"] if g in s["members"]), None)
        final = sorted(s["winner"] for s in r["stages"])
    else:
        pool = cat if arm == "T100" else sorted(sl20[r["id"]])
        if g not in pool:
            return []
        k = pool.index(g) // 10; first = pool[k * 10:k * 10 + 10]
        final = sorted(s["winner"] for s in r["stages"] if s.get("level", 0) == 0)
    out = [first] if first else []
    if first and g in final:
        out.append(final)
    return out


def main():
    qs = load(os.path.join(DATA, "queries.jsonl")); te = [q["id"] for q, t in zip(qs, ~split(qs)) if t]
    twin = {q["id"]: q["has_twin"] for q in qs}
    cat = [s["id"] for s in json.load(open(os.path.join(DATA, "sop_catalog.json"), encoding="utf-8"))["sops"]]
    rank = json.load(open(os.path.join(DATA, "eg2_rank.json"), encoding="utf-8"))
    sl20 = json.load(open(os.path.join(DATA, "eg2_top20.json"), encoding="utf-8"))
    src = {"H20": os.path.join(V15, "26b/route_shortlist20.jsonl"), "T100": os.path.join(V14, "26b/route_tournament.jsonl"),
           **{a: os.path.join(V16, f"26b/route_deal{a}.jsonl") for a in ("D20", "S20", "P20", "D100")}}
    R = {a: {r["id"]: r for r in load(p) if r["id"] in set(te)} for a, p in src.items()}
    R = {a: v for a, v in R.items() if len(v) == len(te)}
    arms = {}
    for a in ARMS:
        if a not in R:
            continue
        rs = [R[a][i] for i in te]
        rec = [r.get("gold_in_shortlist", True) if a != "T100" else True for r in rs]
        hard = easy = hard_err = easy_err = 0; lk_first = []
        lk_final = [len({s["winner"] for s in r["stages"] if s.get("level", 0) == 0} & (set(rank[r["id"]][:10]) - {r["gold"]})) for r in rs if r["stages"]]
        for r in rs:
            look = set(rank[r["id"]][:10]) - {r["gold"]}
            reads = gold_reads(a, r, cat, sl20)
            if reads:
                lk_first.append(sum(x in look for x in reads[0]))
            for n, m in enumerate(reads):
                lost_here = (r["pred"] != r["gold"]) and (n == len(reads) - 1)
                if any(x in look for x in m):
                    hard += 1; hard_err += lost_here
                else:
                    easy += 1; easy_err += lost_here
        b_h, c_h, p_h = mcnemar(R["H20"], R[a], te) if "H20" in R else (0, 0, 1.0)
        b_t, c_t, p_t = mcnemar(R["T100"], R[a], te) if "T100" in R else (0, 0, 1.0)
        arms[a] = {"n": len(rs), "top1": float(np.mean([r["correct"] for r in rs])), "top1_twin": float(np.mean([r["correct"] for r in rs if twin[r["id"]]])),
                   "recall": float(np.mean(rec)), "reads": float(np.mean([r["reads"] for r in rs])),
                   "not_shortlisted": int(sum(not x for x in rec)),
                   "lost_in_group": int(sum(x and not r["correct"] and not r["gold_in_final"] for r, x in zip(rs, rec))),
                   "lost_in_final": int(sum(x and not r["correct"] and r["gold_in_final"] for r, x in zip(rs, rec))),
                   "lookalikes_in_gold_group": float(np.mean(lk_first)), "lookalikes_in_final": float(np.mean(lk_final)), "hard_reads_per_query": hard / len(rs),
                   "err_per_hard_read": hard_err / hard if hard else float("nan"), "err_per_easy_read": easy_err / easy if easy else float("nan"),
                   "vs_H20": {"only_H20": b_h, "only_this": c_h, "p": p_h}, "vs_T100": {"only_T100": b_t, "only_this": c_t, "p": p_t}}
    h, t = arms["H20"]["top1"], arms["T100"]["top1"]
    best = max(arms.get("S20", {}).get("top1", 0), arms.get("P20", {}).get("top1", 0))
    g1, g2 = best >= h + 0.02 - 1e-9, best >= t - 0.01 - 1e-9  # 1e-9: k/150 differences are exact in counts
    g3 = "D100" in arms and arms["D100"]["top1"] >= t + 0.02 - 1e-9
    hypo = "A" if g1 else ("B" if all(arms[a]["lost_in_group"] >= arms["H20"]["lost_in_group"] for a in ("S20", "P20") if a in arms) else "不明")
    pred = {f"{a}_within_0.02_of_H20": a in arms and abs(arms[a]["top1"] - h) <= 0.02 + 1e-9 for a in ("D20", "S20", "P20")}
    pred["D100_within_0.02_of_T100"] = "D100" in arms and abs(arms["D100"]["top1"] - t) <= 0.02 + 1e-9
    res = {"arms": arms, "gates": {"G1_dealing_helps": g1, "G2_dealt_top20_replaces_full": g2, "G3_dealt_full_beats_catalog": g3, "hypothesis": hypo}, "predictions": pred}
    json.dump(res, open(os.path.join(ROOT, "results/v16.json"), "w"), ensure_ascii=False, indent=1)

    f = lambda x: f"{x:.3f}"  # noqa: E731
    L = ["# 21 — 打散分組：把長得像的對手分到不同組（v16）", "",
         "計劃與門檻：`docs/handoff-v16-dealt-groups.md`（跑前寫死）。資料、切分、26B、L4 同 v14／v15；4 則並行跑 300 則、test 150 則計分。"
         "「發牌」＝依 EG 名次輪流分組；組內與決賽一律照目錄順序。", "",
         "## 判定", "",
         f"- **G1 打散有用**（max(S20, P20) ≥ H20 + 0.02）：{f(best)} vs {f(h)} → {'✓' if g1 else '✗'}",
         f"- **G2 打散的前 20 名能取代完整淘汰賽**（≥ T100 − 0.01）：{f(best)} vs {f(t)} → {'✓' if g2 else '✗'}",
         f"- **G3 打散的完整淘汰賽勝過照目錄**（D100 ≥ T100 + 0.02）：{f(arms.get('D100', {}).get('top1', float('nan')))} vs {f(t)} → {'✓' if g3 else '✗'}",
         f"- 假說判讀：{ {'A': '支持 A（密度）', 'B': '支持 B（難關數）', '不明': '兩個都不乾淨'}[hypo] }",
         "- 預測對答案：" + "、".join(f"{k} {'✓' if v else '✗'}" for k, v in pred.items()), "",
         "## 結果（test 150 則）", "",
         "| 做法 | top-1 | 近似孿生 | 每則讀取 | 正解沒進候選 | 第一輪被淘汰 | 決賽輸掉 | 正解那組的近親數 | 對 H20（只 H20 對／只這組對，p） | 對 T100（只 T100 對／只這組對，p） |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        if a in arms:
            x = arms[a]
            vh = "—" if a == "H20" else f"{x['vs_H20']['only_H20']}／{x['vs_H20']['only_this']}，{x['vs_H20']['p']:.3f}"
            vt = "—" if a == "T100" else f"{x['vs_T100']['only_T100']}／{x['vs_T100']['only_this']}，{x['vs_T100']['p']:.3f}"
            L.append(f"| {NAME[a]} | **{f(x['top1'])}** | {f(x['top1_twin'])} | {x['reads']:.0f} | {x['not_shortlisted']} | {x['lost_in_group']} | {x['lost_in_final']} | "
                     f"{x['lookalikes_in_gold_group']:.2f} | {vh} | {vt} |")
    L += ["", "「近親」＝這則訊息的 EG 前 10 名（不含正解）。", "",
          "## 難關數（跑後診斷）", "",
          "把每則訊息裡「正解有參加的讀取」分成兩種：同場有至少一個近親的叫難關，沒有的叫簡單關。正解輸在哪一關就算那一關錯。", "",
          "| 做法 | 每則難關數 | 每次難關出錯率 | 每次簡單關出錯率 | 決賽裡的近親數 |", "|---|---|---|---|---|"]
    for a in ARMS:
        if a in arms:
            x = arms[a]
            L.append(f"| {NAME[a]} | {x['hard_reads_per_query']:.2f} | {x['err_per_hard_read']:.3f} | {x['err_per_easy_read']:.3f} | {x['lookalikes_in_final']:.2f} |")
    L += ["", "決賽裡的近親數＝第一輪各組勝出者之中，有幾個是這則訊息的 EG 前 10 名（不含正解）。", "",
          "## 讀法", "",
          "- **假說 A（密度）被推翻**：發牌 4 組把正解那組的近親從 4.41 個降到 1.86 個，第一輪被淘汰的反而從 4 則變 7 則；top-1 沒有變好。",
          "- **假說 B 照預先寫的判讀規則算是「支持」，但字面上的版本也不成立**：全部 100 份發牌（D100）每則只有 1.09 次難關，是最少的，卻不比照目錄好。"
          "原因是打散把難度搬到決賽：每組都有一個近親，近親大多會贏下自己那組，決賽裡的近親（不含正解）從照目錄的 2.17 個變成 3.67 個，決賽輸掉從 5 則變 8 則。",
          "- 比較像的說法：難的訊息不管怎麼排都會在某一場碰到近親，錯誤率大致跟著訊息走，不跟著分組走。分組方式只是決定錯在哪一輪。",
          "- 只有 5 個字母（S20 0.873）沒有比補滿 10 個字母（P20 0.893）好；補進去的不相干 SOP 沒有造成傷害。",
          "- 150 則的雜訊下，前 20 名的四種排法（0.873–0.893）彼此都分不開；全部 100 份的兩種排法（0.927、0.947）也分不開（p = 0.375）。"]
    open(os.path.join(ROOT, "results/21-dealt-groups.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
