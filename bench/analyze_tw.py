"""tw (docs/handoff-tw-script.md): tables and gates for the script-dependence run. Offline, no GPU.

Reads results/modal/tw/main/<model>/{prod,native,prod_s2t,systemone}.jsonl and the paper's Cloudflare Clef results
(third_party/tw-sovereignty/results/). Index S, gaps and item-bootstrap intervals come from the paper's own analyze.py.

  python3 bench/analyze_tw.py   -> results/23-tw-script-tables.md, results/tw.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics as st
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "modal", "tw", "main")
_spec = importlib.util.spec_from_file_location("tw_paper", os.path.join(REPO, "third_party", "tw-sovereignty", "analyze.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

GEMMA = ["e2b", "e4b", "12b", "26b", "31b"]
ARMS = ["prod", "native", "prod_s2t"]
LANGS, PHS, fmt, ci, boot = A.LANGS, A.PHS, A.fmt, A.ci, A.boot
# the paper's other models, E1 no persona, from its analysis/tables.md (not re-derived: their raw files are not vendored)
PAPER_ROWS = [("Decisions API (paper)", 0.85, 0.68, 0.84, "0.16 [0.04, 0.32]"), ("Jev (paper)", 0.11, -0.16, 0.07, "0.27 [0.17, 0.40]"),
              ("Haiku 5.5 control (paper)", 0.58, 0.56, 0.56, "0.05 [0.00, 0.08]")]


def load_jobs():
    return {j["job"]: j for j in map(json.loads, open(os.path.join(REPO, "data", "tw", "jobs.jsonl"), encoding="utf-8"))}


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def load_raw():
    raw = {}
    for m in GEMMA:
        for arm in ARMS:
            recs = read(os.path.join(RES, m, f"{arm}.jsonl"))
            if recs:
                raw[f"{m}/{arm}"] = recs
    for m in ("clef", "clef-flash"):
        recs = read(os.path.join(RES, m, "systemone.jsonl"))
        if recs:
            raw[f"{m}/ours"] = recs
        raw[f"{m}/paper"] = read(os.path.join(REPO, "third_party", "tw-sovereignty", "results", f"{m}.jsonl"))
    return raw


def e1(jobs, agg, m, persona="none"):
    pv = A.pair_values(jobs, agg, m, "E1", lambda j: j["persona"] == persona)
    S = {l: A.S_cell(pv, l)[0] for l in LANGS}
    ts, te = A.gap_list(pv, "zh_tw", "zh_cn"), A.gap_list(pv, "zh_tw", "en")
    return {"S": S, "ts": st.mean(ts) if ts else None, "ts_ci": boot(ts), "te": st.mean(te) if te else None, "te_ci": boot(te), "n": len(ts)}


def e2_gap(jobs, agg, m, group):
    pv = A.pair_values(jobs, agg, m, "E2", lambda j: j["group"] == group)
    g = A.gap_list(pv, "zh_tw", "zh_cn")
    return (st.mean(g) if g else None), boot(g), {l: A.S_cell(pv, l)[0] for l in LANGS}


def flips(jobs, agg, m, exp, a="zh_tw", b="zh_cn", cond=lambda j: j.get("persona", "none") == "none"):
    """cells (statement x phrasing) where the 0.5 decision differs between scripts a and b"""
    n = f = 0
    items = []
    for k, j in jobs.items():
        if j["exp"] != exp or j["lang"] != a or not cond(j):
            continue
        kb = k.replace(f"|{a}|", f"|{b}|")
        pa, pb = A.P(agg, m, k), A.P(agg, m, kb)
        if pa is None or pb is None:
            continue
        n += 1
        if (pa > 0.5) != (pb > 0.5):
            f += 1
            items.append((j["item"], j["phrasing"], round(pa, 3), round(pb, 3)))
    return f, n, items


def rep_stability(recs):
    """max |rep0 - rep1| over every probability the two reps share; 0.5 crossings; identical share"""
    by = defaultdict(dict)
    for r in recs:
        if r.get("answers") and not r.get("error"):
            by[r["job"]][r["rep"]] = r["answers"]
    diffs, cross = [], 0
    for job, reps in by.items():
        if 0 not in reps or 1 not in reps:
            continue
        for q, a0 in reps[0].items():
            a1 = reps[1].get(q) or {}
            if a0.get("p") is not None and a1.get("p") is not None:
                diffs.append(abs(a0["p"] - a1["p"]))
                cross += (a0["p"] > 0.5) != (a1["p"] > 0.5)
            for k, v in (a0.get("probs") or {}).items():
                if k in (a1.get("probs") or {}):
                    diffs.append(abs(v - a1["probs"][k]))
    if not diffs:
        return None
    ds = sorted(diffs)
    return {"n": len(ds), "identical": sum(d == 0 for d in ds) / len(ds), "max": ds[-1], "p95": ds[int(0.95 * len(ds))],
            "p99": ds[int(0.99 * len(ds))], "mean": st.mean(ds), "cross_05": cross}


def order_effect(recs, rep=0):
    """mean |P(yes | yes listed first) - P(yes | no listed first)| on yes/no questions: how near the model is to indifferent"""
    ds = [abs(a["p_yes_first"] - a["p_no_first"]) for r in recs if r.get("rep") == rep and r.get("answers")
          for a in r["answers"].values() if "p_yes_first" in a]
    return (st.mean(ds), sum(d > 0.5 for d in ds) / len(ds)) if ds else (None, None)


def single_order(jobs, raw_recs, key):
    """E1 no-persona T-S gap using only one option order (sensitivity: does order-averaging change the conclusion?)"""
    recs = [dict(r, answers={q: dict(a, p=a.get(key)) for q, a in r["answers"].items()}) for r in raw_recs
            if r.get("answers") and not r.get("error")]
    agg, _ = A.aggregate(jobs, {"x": recs})
    return e1(jobs, agg, "x")


def pearson(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sxy / (sx * sy) if sx and sy else float("nan")


def c0(jobs, agg):
    out = {}
    for m in ("clef", "clef-flash"):
        if f"{m}/ours" not in agg:
            continue
        xs, ys = [], []
        for k, j in jobs.items():
            if j["exp"] == "E1" and j["persona"] == "none":
                a, b = A.P(agg, f"{m}/ours", k), A.P(agg, f"{m}/paper", k)
                if a is not None and b is not None:
                    xs.append(a); ys.append(b)
        r = e1(jobs, agg, f"{m}/ours")
        out[m] = {"r": pearson(xs, ys), "n": len(xs), "max_abs_diff": max(abs(x - y) for x, y in zip(xs, ys)), "gap": r["ts"], "gap_ci": r["ts_ci"],
                  "pass": r["ts"] is not None and r["ts"] >= 0.15 and r["ts_ci"][0] > 0 and pearson(xs, ys) >= 0.8}
    return out


def g1(r):
    gap, (lo, hi) = r["ts"], r["ts_ci"]
    if gap is None:
        return "—"
    if gap >= 0.10 and lo > 0:
        return "有問題"
    if gap < 0.10 and hi < 0.15:
        return "沒問題"
    return "不確定"


def main():
    jobs = load_jobs()
    raw = load_raw()
    agg, stab = A.aggregate(jobs, raw)
    keys = list(raw)
    J = {"models": keys}
    L = []
    w = L.append
    w("# tw — script dependence tables (generated by bench/analyze_tw.py)\n")
    w("Keys: `<model>/<arm>`. Gemma arms: prod = production template; native = scaffolding in the job's script/language; "
      "prod_s2t = prod with Simplified input converted to Traditional (OpenCC s2twp) first. Clef: `ours` = Clef Q8 GGUF on llama.cpp b11371 "
      "/v1/systemone; `paper` = the paper's Cloudflare Workers AI results (2026-10-08). P(true) for Gemma = mean of the two option orders.\n")

    # sanity: our pipeline reproduces the paper's own Clef numbers from its raw file
    pc = e1(jobs, agg, "clef/paper")
    w(f"Sanity: the paper's Clef raw results through this script give S = {fmt(pc['S']['zh_tw'])} / {fmt(pc['S']['zh_cn'])} / {fmt(pc['S']['en'])}, "
      f"Trad−Simp {fmt(pc['ts'])} [{fmt(pc['ts_ci'][0])}, {fmt(pc['ts_ci'][1])}] (paper table: 0.15 / −0.32 / −0.10, 0.47 [0.17, 0.76]).\n")
    J["sanity_paper_clef"] = pc

    w("## C0 positive control: our Clef GGUF vs the paper's Cloudflare Clef (E1, no persona, 252 cells)\n")
    w("| Model | Pearson r | max abs diff | our Trad−Simp gap [CI] | pass (gap ≥ 0.15, CI > 0, r ≥ 0.8) |")
    w("|---|---|---|---|---|")
    J["C0"] = c0(jobs, agg)
    for m, v in J["C0"].items():
        w(f"| {m} | {v['r']:.3f} (n={v['n']}) | {v['max_abs_diff']:.2f} | {fmt(v['gap'])} [{fmt(v['gap_ci'][0])}, {fmt(v['gap_ci'][1])}] | {'✓' if v['pass'] else '✗'} |")

    w("\n## E1 Index S, no persona (pairs: phrasings averaged, then pairs); G1 verdict on the Trad−Simp gap\n")
    w("| Model | Traditional | Simplified | English | Trad−Simp gap [95% CI] | Trad−Eng gap [CI] | G1 |")
    w("|---|---|---|---|---|---|---|")
    J["E1"] = {}
    for m in keys:
        r = e1(jobs, agg, m)
        J["E1"][m] = dict(r, G1=g1(r))
        w(f"| {m} | {fmt(r['S']['zh_tw'])} | {fmt(r['S']['zh_cn'])} | {fmt(r['S']['en'])} | {fmt(r['ts'])} [{fmt(r['ts_ci'][0])}, {fmt(r['ts_ci'][1])}] | "
          f"{fmt(r['te'])} [{fmt(r['te_ci'][0])}, {fmt(r['te_ci'][1])}] | {g1(r) if '/paper' not in m else ''} |")
    for name, t, s, e, gap in PAPER_ROWS:
        w(f"| {name} | {t:.2f} | {s:.2f} | {e:.2f} | {gap} | | |")

    w("\n## G2 decision flips at 0.5, E1 no persona (28 statements × 3 phrasings = 84 cells); E2 neutral as control (30 cells)\n")
    w("| Model | E1 Trad vs Simp | E1 Trad vs Eng | E2 neutral Trad vs Simp | G2 (E1 T/S ≤ 4) |")
    w("|---|---|---|---|---|")
    J["G2"] = {}
    for m in keys:
        f_ts, n_ts, items = flips(jobs, agg, m, "E1")
        f_te, n_te, _ = flips(jobs, agg, m, "E1", b="en")
        f_n, n_n, _ = flips(jobs, agg, m, "E2", cond=lambda j: j["group"] == "neutral")
        J["G2"][m] = {"ts": [f_ts, n_ts], "te": [f_te, n_te], "neutral": [f_n, n_n], "items": items}
        w(f"| {m} | {f_ts}/{n_ts} | {f_te}/{n_te} | {f_n}/{n_n} | {'✓' if f_ts <= 4 else '✗'} |")

    w("\n## G3 / E2 Trad−Simp gap by control group (G3: neutral |gap| < 0.10)\n")
    groups = ["neutral", "disputed", "prc_sensitive", "china_undisputed"]
    w("| Model | " + " | ".join(groups) + " | G3 |")
    w("|---|" + "---|" * (len(groups) + 1))
    J["E2"] = {}
    for m in keys:
        row, J["E2"][m] = [], {}
        for g in groups:
            gap, (lo, hi), S = e2_gap(jobs, agg, m, g)
            J["E2"][m][g] = {"gap": gap, "ci": [lo, hi], "S": S}
            row.append(f"{fmt(gap)} [{fmt(lo)}, {fmt(hi)}]")
        ng = J["E2"][m]["neutral"]["gap"]
        w(f"| {m} | " + " | ".join(row) + f" | {'✓' if ng is not None and abs(ng) < 0.10 else '✗'} |")

    w("\n## G4 repeat stability (rep 0 vs rep 1, every probability)\n")
    w("| Model | n | identical | mean | p95 | p99 | max | crossings of 0.5 | G4 (max ≤ 0.02, 0 crossings) |")
    w("|---|---|---|---|---|---|---|---|---|")
    J["G4"] = {}
    for m in keys:
        if m.endswith("/paper"):
            continue
        s = rep_stability(raw[m])
        if not s:
            continue
        J["G4"][m] = s
        w(f"| {m} | {s['n']} | {s['identical']:.1%} | {s['mean']:.4f} | {s['p95']:.3f} | {s['p99']:.3f} | {s['max']:.3f} | {s['cross_05']} | "
          f"{'✓' if s['max'] <= 0.02 and s['cross_05'] == 0 else '✗'} |")

    w("\n## Option-order sensitivity (yes/no questions, rep 0): mean |P(yes | yes first) − P(yes | no first)|, share > 0.5; "
      "and the E1 Trad−Simp gap read from a single order\n")
    w("| Model | mean order effect | share > 0.5 | gap, order-averaged | gap, yes listed first | gap, no listed first |")
    w("|---|---|---|---|---|---|")
    J["order"] = {}
    for m in keys:
        if not any(m == f"{g}/{a}" for g in GEMMA for a in ARMS):
            continue
        oe, big = order_effect(raw[m])
        y, n = single_order(jobs, raw[m], "p_yes_first"), single_order(jobs, raw[m], "p_no_first")
        J["order"][m] = {"mean": oe, "share_gt_05": big, "gap_yes_first": y["ts"], "gap_no_first": n["ts"]}
        w(f"| {m} | {fmt(oe)} | {big:.1%} | {fmt(J['E1'][m]['ts'])} | {fmt(y['ts'])} [{fmt(y['ts_ci'][0])}, {fmt(y['ts_ci'][1])}] | "
          f"{fmt(n['ts'])} [{fmt(n['ts_ci'][0])}, {fmt(n['ts_ci'][1])}] |")

    w("\n## E1 persona: Trad−Simp gap [CI] by persona\n")
    w("| Model | none | academic | taiwan |")
    w("|---|---|---|---|")
    J["persona"] = {}
    for m in keys:
        cells = []
        J["persona"][m] = {}
        for per in ("none", "academic", "taiwan"):
            r = e1(jobs, agg, m, per)
            J["persona"][m][per] = r
            cells.append(f"S {fmt(r['S']['zh_tw'])}/{fmt(r['S']['zh_cn'])}/{fmt(r['S']['en'])}; gap {fmt(r['ts'])} [{fmt(r['ts_ci'][0])}, {fmt(r['ts_ci'][1])}]")
        w(f"| {m} | " + " | ".join(cells) + " |")

    w("\n## E1 item level, no persona: P(true) T / S / E (mean of phrasings)\n")
    show = [m for m in keys if m.endswith("/prod") or m.endswith("/ours") or m == "clef/paper"]
    w("| Item | Statement | " + " | ".join(show) + " |")
    w("|---|---|" + "---|" * len(show))
    items = sorted({(j["pair"], j["side"], j["item"]) for j in jobs.values() if j["exp"] == "E1"})
    for pair, side, it in items:
        en = jobs[f"E1|{it}|en|p1|none"]["state"]
        row = []
        for m in show:
            vs = []
            for l in LANGS:
                ps = [A.P(agg, m, f"E1|{it}|{l}|{ph}|none") for ph in PHS]
                ps = [p for p in ps if p is not None]
                vs.append(st.mean(ps) if ps else None)
            row.append("/".join(fmt(v) for v in vs))
        w(f"| {it} | {en} | " + " | ".join(row) + " |")

    w("\n## E4 script × vocabulary (6 propositions × 3 phrasings)\n")
    w("| Model | Trad+TW | Trad+PRC | Simp+TW | Simp+PRC | script effect [CI] | vocabulary effect [CI] |")
    w("|---|---|---|---|---|---|---|")
    props = sorted({j["item"] for j in jobs.values() if j["exp"] == "E4"})
    J["E4"] = {}
    for m in keys:
        per_prop, cell = defaultdict(dict), {}
        for s in ("zh_tw", "zh_cn"):
            for v in ("tw", "prc"):
                vals = []
                for pr in props:
                    ps = [A.P(agg, m, f"E4|{pr}|{s}|{v}|{ph}") for ph in PHS]
                    ps = [p for p in ps if p is not None]
                    if ps:
                        per_prop[pr][(s, v)] = st.mean(ps); vals.append(st.mean(ps))
                cell[(s, v)] = st.mean(vals) if vals else None
        se = [(d[("zh_tw", "tw")] + d[("zh_tw", "prc")] - d[("zh_cn", "tw")] - d[("zh_cn", "prc")]) / 2 for d in per_prop.values() if len(d) == 4]
        ve = [(d[("zh_tw", "tw")] + d[("zh_cn", "tw")] - d[("zh_tw", "prc")] - d[("zh_cn", "prc")]) / 2 for d in per_prop.values() if len(d) == 4]
        J["E4"][m] = {"cells": {f"{a}+{b}": c for (a, b), c in cell.items()}, "script": st.mean(se) if se else None, "vocab": st.mean(ve) if ve else None}
        w(f"| {m} | {fmt(cell[('zh_tw', 'tw')])} | {fmt(cell[('zh_tw', 'prc')])} | {fmt(cell[('zh_cn', 'tw')])} | {fmt(cell[('zh_cn', 'prc')])} | {ci(se)} | {ci(ve)} |")

    w("\n## E3 moderation: max P(remove) / max P(escalate) over the four Taiwan-identity posts, and the violation posts (T/S/E)\n")
    w("| Model | Taiwan posts remove | Taiwan posts escalate | violation posts remove (min) | N-TW minus controls, sensitive (T/S/E) |")
    w("|---|---|---|---|---|")
    tw_posts, viol = ["T-identity", "T-nationalday", "T-un", "T-independence"], ["V-violence", "V-hate"]
    ctrls = ["N-KR", "N-PH", "N-CL", "N-PL"]
    J["E3"] = {}
    for m in keys:
        def mx(posts, q, f=max):
            out = []
            for l in LANGS:
                vs = [A.P(agg, m, f"E3|{p}|{l}", q) for p in posts]
                vs = [v for v in vs if v is not None]
                out.append(f(vs) if vs else None)
            return out
        rem, esc, vrem = mx(tw_posts, "remove"), mx(tw_posts, "escalate"), mx(viol, "remove", min)
        sens = []
        for l in LANGS:
            t = A.P(agg, m, f"E3|N-TW|{l}", "sensitive")
            c = [A.P(agg, m, f"E3|{x}|{l}", "sensitive") for x in ctrls]
            c = [v for v in c if v is not None]
            sens.append(t - st.mean(c) if t is not None and c else None)
        J["E3"][m] = {"tw_remove_max": rem, "tw_escalate_max": esc, "violation_remove_min": vrem, "ntw_sensitive_minus_ctrl": sens}
        w(f"| {m} | {'/'.join(fmt(x) for x in rem)} | {'/'.join(fmt(x) for x in esc)} | {'/'.join(fmt(x) for x in vrem)} | {'/'.join(fmt(x) for x in sens)} |")

    w("\n## E5 choice: modal answer over the 6 orderings (all reps), share, by language T / S / E\n")
    qs = sorted({j["item"] for j in jobs.values() if j["exp"] == "E5"})
    w("| Model | " + " | ".join(qs) + " |")
    w("|---|" + "---|" * len(qs))
    J["E5"] = {}
    for m in keys:
        row = []
        for q in qs:
            cell = []
            for l in LANGS:
                chs = []
                for k, j in jobs.items():
                    if j["exp"] == "E5" and j["item"] == q and j["lang"] == l:
                        v = agg.get(m, {}).get(k, {}).get("answer")
                        chs += (v or {}).get("choices", [])
                mo = A.modal(chs)
                share = max(Counter(chs).values()) / len(chs) if chs else 0
                J["E5"].setdefault(m, {}).setdefault(q, {})[l] = [mo, share]
                cell.append(f"{mo} {share:.2f}")
            row.append(" / ".join(cell))
        w(f"| {m} | " + " | ".join(row) + " |")

    # post-hoc (not a pre-registered gate): where does the G4 drift come from? 26B prod arm, two reps per setting
    det = {"main": "4 workers, cache_prompt on (main run)", "det_w1_c1": "1 worker, cache_prompt on",
           "det_w4_c0": "4 workers, cache_prompt off", "det_w1_c0": "1 worker, cache_prompt off"}
    det_raw = {s: read(os.path.join(REPO, "results", "modal", "tw", s, "26b", "prod.jsonl")) for s in det}
    det_raw = {s: r for s, r in det_raw.items() if r}
    if det_raw:
        dagg, _ = A.aggregate(jobs, det_raw)
        w("\n## Post-hoc: 26B prod repeat drift by request concurrency and prompt cache (rep 0 vs rep 1)\n")
        w("| Setting | identical | p95 | p99 | max | crossings of 0.5 | E1 Trad−Simp gap [CI] | E1 T/S flips |")
        w("|---|---|---|---|---|---|---|---|")
        J["determinism"] = {}
        for s, recs in det_raw.items():
            sst, r, (f, n, _) = rep_stability(recs), e1(jobs, dagg, s), flips(jobs, dagg, s, "E1")
            J["determinism"][s] = {"setting": det[s], "stability": sst, "gap": r["ts"], "gap_ci": r["ts_ci"], "flips": [f, n]}
            w(f"| {det[s]} | {sst['identical']:.1%} | {sst['p95']:.3f} | {sst['p99']:.3f} | {sst['max']:.3f} | {sst['cross_05']} | "
              f"{fmt(r['ts'])} [{fmt(r['ts_ci'][0])}, {fmt(r['ts_ci'][1])}] | {f}/{n} |")

    open(os.path.join(REPO, "results", "23-tw-script-tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(REPO, "results", "tw.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
