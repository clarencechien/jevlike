"""tw2 (docs/handoff-tw2-decision-apis.md): tables and gates. Offline.

A  hosted decision APIs on the paper's jobs: results/modal/tw2/a/<model>/{api,api_s2t}.jsonl, next to tw's 26b / Clef readings
   (results/modal/tw/main) and the paper's numbers. Index S, gaps and intervals via the paper's analyze.py (same as bench/analyze_tw.py).
B  other disputes in the claimants' languages: data/tw2/jobs.jsonl; APIs results/modal/tw2/b/<model>/api.jsonl;
   26b / clef results/modal/tw2/b-local/<model>/{prod,prod_norm,prod_pivot,systemone}.jsonl.

  python3 bench/analyze_tw2.py   -> results/tw2-decision-apis-tables.md, results/tw2.json
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_tw as T  # noqa: E402  (also loads the paper's analyze.py as T.A)

A = T.A
REPO = T.REPO
RA = os.path.join(REPO, "results", "modal", "tw2", "a")
RB = os.path.join(REPO, "results", "modal", "tw2", "b")
RBL = os.path.join(REPO, "results", "modal", "tw2", "b-local")
APIS = [("msft-1.1", "Microsoft-Decision-1.1（Gemma 4 26B-A4B）"), ("msft-1.0", "Microsoft-Decision-1（Qwen3.5-9B）"), ("gemini-d1", "Gemini Decision 1")]


def read(p):
    return T.read(p)


# ====================================================================== A
def part_a():
    jobs = T.load_jobs()
    raw = {}
    for m, _ in APIS:
        for arm in ("api", "api_s2t"):
            recs = read(os.path.join(RA, m, f"{arm}.jsonl"))
            if recs:
                raw[f"{m}/{arm}"] = recs
    tw = T.load_raw()
    for k in ("26b/prod", "26b/prod_s2t", "clef/ours", "clef-flash/ours"):
        if k in tw:
            raw[k] = tw[k]
    agg, stab = A.aggregate(jobs, raw)
    out, md = {}, ["# tw2 — 新的商用 decision 模型與其他主權爭議：完整表格（`bench/analyze_tw2.py` 產生）", "",
                   "## A. 論文考卷（台灣），E1 無人格：Index S 與繁減簡差距", "",
                   "| 模型 | 繁 | 簡 | 英 | 繁減簡 [95% CI] | 繁減英 [CI] | G1 | 翻面 繁 vs 簡（/84） | 翻面 繁 vs 英 | G2 | E2 中性繁減簡 | G3 | 重跑最大差（翻 0.5） | G4 |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = dict(APIS)
    rows = [(f"{m}/api", names[m]) for m, _ in APIS] + [(f"{m}/api_s2t", names[m] + "，簡體先轉繁") for m, _ in APIS] + \
           [("26b/prod", "Gemma 4 26B（我們，tw）"), ("26b/prod_s2t", "Gemma 4 26B，簡體先轉繁（tw）"), ("clef/ours", "Clef 27B（我們跑的 GGUF，tw）")]
    for key, label in rows:
        if key not in agg:
            continue
        r = T.e1(jobs, agg, key)
        f, n, items = T.flips(jobs, agg, key, "E1")
        fe, ne, _ = T.flips(jobs, agg, key, "E1", "zh_tw", "en")
        g2, g2n, _ = T.e2_gap(jobs, agg, key, "neutral")
        s = stab.get(key, {})
        g4 = None
        if s.get("cells"):
            g4 = (s["max_range"] is not None and s["max_range"] <= 0.02 and s["threshold_flips"] == 0)
        out[key] = {"S": r["S"], "ts": r["ts"], "ts_ci": r["ts_ci"], "te": r["te"], "te_ci": r["te_ci"], "G1": T.g1(r),
                    "flips_ts": f, "flips_n": n, "flips_te": fe, "flip_items": items, "G2": f <= 4, "e2_neutral": g2, "e2_neutral_ci": g2n,
                    "G3": g2 is not None and abs(g2) < 0.10, "stab": s, "G4": g4}
        md.append(f"| {label} | {A.fmt(r['S']['zh_tw'])} | {A.fmt(r['S']['zh_cn'])} | {A.fmt(r['S']['en'])} | {A.fmt(r['ts'])} [{A.fmt(r['ts_ci'][0])}, {A.fmt(r['ts_ci'][1])}] | "
                  f"{A.fmt(r['te'])} [{A.fmt(r['te_ci'][0])}, {A.fmt(r['te_ci'][1])}] | {T.g1(r)} | {f}/{n} | {fe}/{ne} | {'✓' if f <= 4 else '✗'} | "
                  f"{A.fmt(g2)} | {'✓' if out[key]['G3'] else '✗'} | "
                  + (f"{A.fmt(s['max_range'], 3)}（{s['threshold_flips']}） | {'✓' if g4 else '✗'} |" if g4 is not None else "— | — |"))
    for lab, t, s_, e, gap in T.PAPER_ROWS:
        md.append(f"| {lab} | {t:.2f} | {s_:.2f} | {e:.2f} | {gap} | | | | | | | | | |")
    md.append("")
    # inheritance: msft-1.1 vs our 26b, E1 no persona, all 252 cells
    inh = {}
    for m in ("msft-1.1", "msft-1.0", "gemini-d1"):
        for ref in ("26b/prod", "clef/ours"):
            xs, ys = [], []
            for k, j in jobs.items():
                if j["exp"] == "E1" and j["persona"] == "none":
                    a, b = A.P(agg, f"{m}/api", k), A.P(agg, ref, k)
                    if a is not None and b is not None:
                        xs.append(a); ys.append(b)
            if xs:
                inh[f"{m}~{ref}"] = {"r": T.pearson(xs, ys), "n": len(xs)}
    out["_inherit"] = inh
    md += ["### 跟誰像：E1 無人格 252 格的 P 值相關（Pearson r）", "", "| 模型 | 對 Gemma 4 26B（我們） | 對 Clef 27B（Qwen 系） |", "|---|---|---|"]
    for m, lab in APIS:
        a_, b_ = inh.get(f"{m}~26b/prod"), inh.get(f"{m}~clef/ours")
        md.append(f"| {lab} | {A.fmt(a_['r']) if a_ else '—'} | {A.fmt(b_['r']) if b_ else '—'} |")
    md.append("")
    # flipped statements for the APIs
    md += ["### 繁 vs 簡翻面的陳述（E1 無人格，P(真) 繁 → 簡）", ""]
    for m, lab in APIS:
        x = out.get(f"{m}/api")
        if not x:
            continue
        its = defaultdict(list)
        for item, ph, pa, pb in x["flip_items"]:
            its[item].append(f"{ph} {pa:.2f}→{pb:.2f}")
        md.append(f"- **{lab}**（{x['flips_ts']}/84）：" + ("；".join(f"{it} {', '.join(v)}" for it, v in sorted(its.items())) or "無"))
    md.append("")
    # E3 moderation and E5 choices (descriptive)
    md += ["### E3 審查（每則貼文每個問題的平均 P，依語言）與 E5 選擇題（多數選項）", ""]
    for m, lab in APIS:
        key = f"{m}/api"
        if key not in agg:
            continue
        e3 = defaultdict(lambda: defaultdict(list))
        e5 = defaultdict(lambda: defaultdict(list))
        for k, j in jobs.items():
            v = agg[key].get(k)
            if not v:
                continue
            if j["exp"] == "E3":
                for q, a in v.items():
                    if a.get("p") is not None:
                        e3[(j["item"], q)][j["lang"]].append(a["p"])
            if j["exp"] == "E5":
                for q, a in v.items():
                    if a.get("choice"):
                        e5[j["item"]][j["lang"]].append(a["choice"])
        md.append(f"- **{lab}** E3：" + "；".join(f"{it}/{q} " + "／".join(f"{l} {st.mean(ps):.2f}" for l, ps in sorted(d.items()))
                                                for (it, q), d in sorted(e3.items())))
        md.append(f"  E5：" + "；".join(f"{it} " + "／".join(f"{l} {max(set(cs), key=cs.count)}" for l, cs in sorted(d.items())) for it, d in sorted(e5.items())))
    md.append("")
    # cost / served build
    for m, _ in APIS:
        recs = read(os.path.join(RA, m, "api.jsonl")) + read(os.path.join(RA, m, "api_s2t.jsonl"))
        builds = sorted({r.get("served_model") for r in recs if r.get("served_model")})
        cost = sum((r.get("usage") or {}).get("cost") or 0 for r in recs)
        errs = sum(bool(r.get("error")) for r in recs)
        refusals = sum(1 for r in recs if r.get("answers") for a in r["answers"].values() if a.get("refusal"))
        out.setdefault("_meta", {})[m] = {"builds": builds, "cost_usd": cost, "errors": errs, "refusals": refusals, "calls": len(recs)}
    md += ["呼叫紀錄：" + "；".join(f"{m} {v['calls']} 次、錯誤 {v['errors']}、拒答 {v['refusals']}、版本 {', '.join(v['builds']) or '—'}、OpenRouter 計費 ${v['cost_usd']:.4f}"
                                  for m, v in out.get("_meta", {}).items()), ""]
    return out, md


# ====================================================================== B
def load_b():
    jobs = {j["job"]: j for j in map(json.loads, open(os.path.join(REPO, "data", "tw2", "jobs.jsonl"), encoding="utf-8"))}
    master = json.load(open(os.path.join(REPO, "data", "tw2", "master_en.json"), encoding="utf-8"))
    runs = {}
    for m, _ in APIS:
        recs = read(os.path.join(RB, m, "api.jsonl"))
        if recs:
            runs[m] = recs
    for m, arms in (("26b", ("prod", "prod_norm", "prod_pivot")), ("clef", ("systemone",))):
        for arm in arms:
            recs = read(os.path.join(RBL, m, f"{arm}.jsonl"))
            if recs:
                runs[m if arm in ("prod", "systemone") else f"{m}/{arm}"] = recs
    P = {}
    for m, recs in runs.items():
        acc = defaultdict(list)
        for r in recs:
            if r.get("error") or not r.get("answers"):
                continue
            a = r["answers"].get("true") or {}
            if a.get("p") is not None:
                acc[r["job"]].append(a["p"])
        P[m] = {k: st.mean(v) for k, v in acc.items()}
    return jobs, master, P


def b_index(jobs, Pm, dispute, lang):
    """per pair: mean over phrasings of P(X) - P(Y); returns {pair: value}"""
    cell = defaultdict(lambda: defaultdict(dict))
    for k, j in jobs.items():
        if j["dispute"] != dispute or j["lang"] != lang or j["neutral"] or k not in Pm:
            continue
        cell[j["pair"]][j["phrasing"]][j["side"]] = Pm[k]
    out = {}
    for pair, phs in cell.items():
        vals = [d["X"] - d["Y"] for d in phs.values() if "X" in d and "Y" in d]
        if vals:
            out[pair] = st.mean(vals)
    return out


def b_neutral(jobs, Pm, dispute, lang):
    vals = defaultdict(dict)
    for k, j in jobs.items():
        if j["dispute"] == dispute and j["lang"] == lang and j["neutral"] and k in Pm:
            vals[(j["pair"], j["phrasing"])][j["side"]] = Pm[k]
    d = [v["true"] - v["false"] for v in vals.values() if "true" in v and "false" in v]
    return st.mean(d) if d else None


def b_flips(jobs, Pm, dispute, la, lb):
    n = f = 0
    for k, j in jobs.items():
        if j["dispute"] != dispute or j["lang"] != la or j["neutral"]:
            continue
        kb = k.replace(f"|{la}|", f"|{lb}|")
        if k in Pm and kb in Pm:
            n += 1; f += (Pm[k] > 0.5) != (Pm[kb] > 0.5)
    return f, n


def verdict(gap, lo, hi):
    if gap is None or lo != lo:
        return "—"
    if gap >= 0.10 and lo > 0:
        return "有問題"
    if gap < 0.10 and hi < 0.15:
        return "沒問題"
    return "不確定"


def part_b():
    if not os.path.exists(os.path.join(REPO, "data", "tw2", "jobs.jsonl")):
        return None, []
    jobs, master, P = load_b()
    models = [m for m in ["msft-1.1", "msft-1.0", "gemini-d1", "26b", "clef", "26b/prod_norm", "26b/prod_pivot"] if m in P]
    out = {"index": {}, "gaps": {}, "neutral": {}, "summary": {}}
    md = ["## B. 其他主權爭議，用當事方的語言／字形問", "",
          "Index = 每對 P(X 方立場句為真) − P(Y 方立場句為真)，三種問法平均後對 4 對平均；+1 = 一致站 X 方。主場偏差 = Index(X 方的語言) − Index(Y 方的語言)，正值 = 用哪一方的語言問就往哪一方偏。",
          ""]
    for d in master["disputes"]:
        md += [f"### {d['id']}（X = {d['X']}，Y = {d['Y']}）", "", "| 模型 | " + " | ".join(f"{l}" for l in d["langs"]) + " | 中性對照（各語言 P真−P假） |", "|---|" + "---|" * (len(d["langs"]) + 1)]
        for m in models:
            idx = {l: b_index(jobs, P[m], d["id"], l) for l in d["langs"]}
            neu = {l: b_neutral(jobs, P[m], d["id"], l) for l in d["langs"]}
            out["index"][f"{m}|{d['id']}"] = {l: (st.mean(v.values()) if v else None) for l, v in idx.items()}
            out["neutral"][f"{m}|{d['id']}"] = neu
            md.append(f"| {m} | " + " | ".join(A.fmt(st.mean(idx[l].values()) if idx[l] else None) for l in d["langs"]) + " | "
                      + "、".join(f"{l} {A.fmt(neu[l])}" for l in d["langs"]) + " |")
        # home-bias gaps: every (X-language, Y-language) pair; Serbian Cyrillic vs Latin as the script-only pair (no expected sign)
        xs = [l for l, s in d["home"].items() if s == "X"]
        ys = [l for l, s in d["home"].items() if s == "Y"]
        comps = [(a, b, "home") for a in xs for b in ys]
        if d["id"] == "kosovo":
            comps.append(("sr_latn", "sr_cyrl", "script"))
        md += ["", "| 模型 | 比較 | 差距 [95% CI] | 翻面（/24） | 讀得懂（兩語言中性 ≥ 0.8） | 判定 |", "|---|---|---|---|---|---|"]
        for m in models:
            for la, lb, kind in comps:
                ia, ib = b_index(jobs, P[m], d["id"], la), b_index(jobs, P[m], d["id"], lb)
                g = [ia[p] - ib[p] for p in ia if p in ib]
                gap = st.mean(g) if g else None
                lo, hi = A.boot(g) if len(g) >= 2 else (float("nan"), float("nan"))
                f, n = b_flips(jobs, P[m], d["id"], la, lb)
                na, nb = b_neutral(jobs, P[m], d["id"], la), b_neutral(jobs, P[m], d["id"], lb)
                ok = na is not None and nb is not None and na >= 0.8 and nb >= 0.8
                if kind == "script":  # no home side: judge the size of the gap in either direction
                    if gap is None or lo != lo:
                        v = "—"
                    elif abs(gap) >= 0.10 and (lo > 0 or hi < 0):
                        v = "有問題"
                    elif abs(gap) < 0.10 and max(abs(lo), abs(hi)) < 0.15:
                        v = "沒問題"
                    else:
                        v = "不確定"
                else:
                    v = verdict(gap, lo, hi)
                v = v if ok else "讀不懂，不判"
                out["gaps"][f"{m}|{d['id']}|{la}-{lb}"] = {"gap": gap, "ci": [lo, hi], "flips": f, "n": n, "neutral": [na, nb], "readable": ok, "verdict": v,
                                                          "GB2": f <= 2, "kind": kind, "group": d["group"]}
                md.append(f"| {m} | {la} − {lb}{'（只換字母）' if kind == 'script' else ''} | {A.fmt(gap)} [{A.fmt(lo)}, {A.fmt(hi)}] | {f}/{n} | {'✓' if ok else '✗'} | {v} |")
        md.append("")
    # summary per model: disputes with a home-bias problem (any of its X-Y comparisons) / judgeable disputes
    md += ["### 彙總：每個模型在幾個爭議上有主場偏差（GB1 有問題 或 GB2 翻面 > 2/24）", "", "| 模型 | B1（只換字形：中文四題＋科索沃） | B2（換語言） | 合計 |", "|---|---|---|---|"]
    for m in models:
        row = {}
        for grp in ("B1", "B2"):
            prob = judg = 0
            for d in master["disputes"]:
                if d["group"] != grp:
                    continue
                gs = [v for k, v in out["gaps"].items() if k.startswith(f"{m}|{d['id']}|") and v["kind"] == "home" and v["readable"]]
                if not gs:
                    continue
                judg += 1
                prob += any(v["verdict"] == "有問題" or not v["GB2"] for v in gs)
            row[grp] = (prob, judg)
        out["summary"][m] = row
        md.append(f"| {m} | {row['B1'][0]}/{row['B1'][1]} | {row['B2'][0]}/{row['B2'][1]} | {row['B1'][0] + row['B2'][0]}/{row['B1'][1] + row['B2'][1]} |")
    md.append("")
    return out, md


def main():
    a, mda = part_a()
    b, mdb = part_b()
    json.dump({"A": a, "B": b}, open(os.path.join(REPO, "results", "tw2.json"), "w"), ensure_ascii=False, indent=1, default=str)
    open(os.path.join(REPO, "results", "tw2-decision-apis-tables.md"), "w", encoding="utf-8").write("\n".join(mda + mdb) + "\n")
    print("\n".join(mda + mdb))


if __name__ == "__main__":
    main()
