"""tw3 (docs/handoff-tw3-place-names.md): language x place name, and the pivot + name-table pipeline. Offline.

Readings
  tw2 (reused cells): APIs results/modal/tw2/b/<m>/api.jsonl; 26b results/modal/tw2/b-local/26b/{prod,prod_pivot}.jsonl;
                      clef results/modal/tw2/b-local/clef/systemone.jsonl
  tw3 (new cells):    APIs results/modal/tw3/<m>/api.jsonl; 26b results/modal/tw3/26b/prod.jsonl; clef results/modal/tw3/clef/systemone.jsonl
Cells: (dispute, language, pair, name arm X/Y/N, side, phrasing). tw2 rows enter as the name arm their translation already is
(data/tw3/names.json `orig`); English dokdo's tw2 dual name "Dokdo (Takeshima)" is arm `dual` (shown, not in the effects).

  python3 bench/analyze_tw3.py   -> results/tw3-place-names-tables.md, results/tw3.json
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_tw as T  # noqa: E402
import tw3_build as B  # noqa: E402

A = T.A
REPO = T.REPO
R2B, R2L, R3 = (os.path.join(REPO, "results", "modal", *p) for p in (("tw2", "b"), ("tw2", "b-local"), ("tw3",)))
MODELS = ["msft-1.1", "msft-1.0", "gemini-d1", "26b", "clef"]
LABEL = {"msft-1.1": "微軟 1.1", "msft-1.0": "微軟 1.0", "gemini-d1": "Gemini D1", "26b": "26B（我們）", "clef": "Clef"}
COMPS = {"dokdo": [("ko", "ja")], "scs": [("vi", "zh_cn"), ("fil", "zh_cn")], "falklands": [("en", "es")]}
ARMS = ("X", "Y", "N")


def P_of(path):
    acc = defaultdict(list)
    for r in T.read(path):
        if r.get("error") or not r.get("answers"):
            continue
        p = (r["answers"].get("true") or {}).get("p")
        if p is not None:
            acc[r["job"]].append(p)
    return {k: st.mean(v) for k, v in acc.items()}


def load():
    jobs2 = {j["job"]: j for j in map(json.loads, open(os.path.join(REPO, "data", "tw2", "jobs.jsonl"), encoding="utf-8")) if j["dispute"] in B.DISPUTES}
    jobs3 = {j["job"]: j for j in map(json.loads, open(os.path.join(REPO, "data", "tw3", "jobs.jsonl"), encoding="utf-8"))}
    src2 = {m: os.path.join(R2B, m, "api.jsonl") for m in MODELS[:3]} | {"26b": os.path.join(R2L, "26b", "prod.jsonl"), "clef": os.path.join(R2L, "clef", "systemone.jsonl")}
    src3 = {m: os.path.join(R3, m, "api.jsonl") for m in MODELS[:3]} | {"26b": os.path.join(R3, "26b", "prod.jsonl"), "clef": os.path.join(R3, "clef", "systemone.jsonl")}
    E1 = {m: {} for m in MODELS}     # (d, lang, pid, arm, side, ph) -> p
    NEU = {m: {} for m in MODELS}    # (d, lang, pid, side, ph) -> p   (tw2 neutral pairs, native language)
    E0 = {m: {} for m in MODELS}     # (d, lang, pid, side, ph) -> p   (all tw2 pairs, as translated)
    E2 = {m: {} for m in MODELS}     # (d, src, arm, pid, side, ph) -> p
    for m in MODELS:
        P2, P3 = P_of(src2[m]), P_of(src3[m])
        for k, j in jobs2.items():
            if k not in P2:
                continue
            d, lang, pid = j["dispute"], j["lang"], j["pair"].split(":")[1]
            if j["neutral"]:
                NEU[m][(d, lang, pid, j["side"], j["phrasing"])] = P2[k]
                continue
            E0[m][(d, lang, pid, j["side"], j["phrasing"])] = P2[k]
            key = B.NAME_PAIRS[d].get(pid)
            if key:
                arm = B.orig_arm(d, lang, key) or ("dual" if d == "dokdo" else "orig_lc")
                E1[m][(d, lang, pid, arm, j["side"], j["phrasing"])] = P2[k]
        for k, j in jobs3.items():
            if k not in P3:
                continue
            pid = j["pair"].split(":")[1]
            if j["exp"] == "T1":
                E1[m][(j["dispute"], j["lang"], pid, j["name"], j["side"], j["phrasing"])] = P3[k]
            else:
                E2[m][(j["dispute"], j["src"], j["arm"], pid, j["side"], j["phrasing"])] = P3[k]
    # 26b's tw2 pivot readings: a rerun check against tw3's fresh pivot arm
    P26 = P_of(os.path.join(R2L, "26b", "prod_pivot.jsonl"))
    E2["26b/tw2"] = {}
    for k, j in jobs2.items():
        if k in P26 and not j["neutral"]:
            E2["26b/tw2"][(j["dispute"], j["lang"], "pivot", j["pair"].split(":")[1], j["side"], j["phrasing"])] = P26[k]
    return E1, NEU, E0, E2


def index(cells, prefix, pids):
    """{pid: mean over phrasings of P(X side) - P(Y side)} for cells keyed prefix + (pid, side, ph)"""
    out = {}
    for pid in pids:
        v = [cells[prefix + (pid, "X", ph)] - cells[prefix + (pid, "Y", ph)] for ph in ("p1", "p2", "p3")
             if prefix + (pid, "X", ph) in cells and prefix + (pid, "Y", ph) in cells]
        if v:
            out[pid] = st.mean(v)
    return out


def neutral(NEUm, d, lang):
    v = [NEUm[(d, lang, pid, "true", ph)] - NEUm[(d, lang, pid, "false", ph)] for pid in ("n1", "n2") for ph in ("p1", "p2", "p3")
         if (d, lang, pid, "true", ph) in NEUm and (d, lang, pid, "false", ph) in NEUm]
    return st.mean(v) if v else None


def flips(cells, pa, pb, pids):
    n = f = 0
    for pid in pids:
        for side in ("X", "Y"):
            for ph in ("p1", "p2", "p3"):
                ka, kb = pa + (pid, side, ph), pb + (pid, side, ph)
                if ka in cells and kb in cells:
                    n += 1; f += (cells[ka] > 0.5) != (cells[kb] > 0.5)
    return f, n


def gap_stats(ia, ib):
    g = [ia[p] - ib[p] for p in ia if p in ib]
    if not g:
        return None, float("nan"), float("nan"), []
    lo, hi = A.boot(g) if len(g) >= 2 else (float("nan"), float("nan"))
    return st.mean(g), lo, hi, g


def verdict(gap, lo, hi):
    if gap is None or lo != lo:
        return "—"
    if gap >= 0.10 and lo > 0:
        return "有問題"
    if gap < 0.10 and hi < 0.15:
        return "沒問題"
    return "不確定"


def mean_dicts(ds):
    keys = set.intersection(*[set(d) for d in ds]) if ds else set()
    return {k: st.mean(d[k] for d in ds) for k in keys}


def main():
    E1, NEU, E0, E2 = load()
    f = A.fmt
    out = {"E0": {}, "E1": {}, "H1": {}, "H2": {}, "E2": {}, "H3": {}, "rerun_26b_pivot": {}}
    md = ["# tw3 tables（`bench/analyze_tw3.py` 產生）", "",
          "Index = 每對 P(X 方立場句為真) − P(Y 方立場句為真)，三種問法平均；表內數字再對該爭議有地名的各對平均。+1 = 一致站 X 方。",
          "讀得懂 = 該語言 tw2 中性題 P真−P假 ≥ 0.8；讀不懂的語言不進效應、不判。", ""]
    readable = {m: {(d, l): (neutral(NEU[m], d, l) or 0) >= 0.8 for d in B.DISPUTES for l in B.MASTER[d]["langs"]} for m in MODELS}

    # ---------------- E0
    md += ["## E0：南海，tw2 讀數拆成「沒地名的兩對」與「有地名的兩對」（描述，不判）", "",
           "| 模型 | 比較 | 沒地名（九段線、仲裁） | 有地名（黃岩島、南沙） | 差（≈ 地名的額外效應） |", "|---|---|---|---|---|"]
    for m in MODELS:
        for la, lb in COMPS["scs"]:
            if not (readable[m][("scs", la)] and readable[m][("scs", lb)]):
                md.append(f"| {LABEL[m]} | {la} − {lb} | 讀不懂，不算 | | |"); continue
            ia, ib = index(E0[m], ("scs", la), ["ninedash", "ruling", "scarborough", "spratlys"]), index(E0[m], ("scs", lb), ["ninedash", "ruling", "scarborough", "spratlys"])
            g0 = st.mean([ia[p] - ib[p] for p in ("ninedash", "ruling")]); g1 = st.mean([ia[p] - ib[p] for p in ("scarborough", "spratlys")])
            out["E0"][f"{m}|{la}-{lb}"] = {"no_name": g0, "name": g1, "diff": g1 - g0}
            md.append(f"| {LABEL[m]} | {la} − {lb} | {f(g0)} | {f(g1)} | {f(g1 - g0)} |")
    md.append("")

    # ---------------- E1 grid + H1 + H2
    for d in B.DISPUTES:
        pids = list(B.NAME_PAIRS[d])
        langs = B.MASTER[d]["langs"]
        arms_shown = list(ARMS) + (["dual"] if d == "dokdo" else ["orig_lc"] if d == "falklands" else [])
        md += [f"## E1 · {d}（X = {B.MASTER[d]['X']}，Y = {B.MASTER[d]['Y']}；地名對 {len(pids)} 對）", "",
               "Index：列 = 語言，欄 = 地名臂（X 方叫法／Y 方叫法／中立" + ("／tw2 雙名 Dokdo (Takeshima)" if d == "dokdo" else "／tw2 原句（句首小寫）" if d == "falklands" else "") + "）。* = tw2 已有的格子。", "",
               "| 模型 | 語言 | 讀得懂 | " + " | ".join(arms_shown) + " |", "|---|---|---|" + "---|" * len(arms_shown)]
        for m in MODELS:
            for l in langs:
                row = []
                for arm in arms_shown:
                    ix = index(E1[m], (d, l), pids) if False else {p: v for p, v in index({k[:3] + k[4:]: v for k, v in E1[m].items() if k[3] == arm and k[0] == d}, (d, l), pids).items()}
                    tag = "*" if any(arm == (B.orig_arm(d, l, B.NAME_PAIRS[d][p]) or ("dual" if d == "dokdo" else "orig_lc")) for p in pids) else ""
                    row.append(f(st.mean(ix.values())) + tag if ix else "—")
                    out["E1"][f"{m}|{d}|{l}|{arm}"] = st.mean(ix.values()) if ix else None
                md.append(f"| {LABEL[m]} | {l} | {'✓' if readable[m][(d, l)] else '✗'} | " + " | ".join(row) + " |")
        md.append("")

        def ix_arm(m, l, arm):
            return index({k[:3] + k[4:]: v for k, v in E1[m].items() if k[3] == arm and k[0] == d}, (d, l), pids)

        def cells_arm(m, arm):
            return {k[:3] + k[4:]: v for k, v in E1[m].items() if k[3] == arm and k[0] == d}

        md += ["**H1（地名主導）與 H2（原語言＋中立地名）**", "",
               "地名效應 = 各讀得懂語言的 Index(X 方叫法) − Index(Y 方叫法) 平均；語言效應 = 三種地名臂下 Index(X 方語言) − Index(Y 方語言) 平均。"
               "H1 成立 = 地名效應 ≥ 語言效應 + 0.10 且地名效應區間下界 > 0。H2 = 中立地名下的主場偏差，照 tw2 判法；「減半」= ≤ 原寫法偏差的一半（同一組對）。", "",
               "| 模型 | 地名效應 [95% CI] | 語言效應 [95% CI] | H1 | 比較 | 原寫法偏差（tw2） | 中立地名偏差 [95% CI] | 翻面 | H2 判定 | 減半 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            rl = [l for l in langs if readable[m][(d, l)]]
            nm = [{p: ix_arm(m, l, "X")[p] - ix_arm(m, l, "Y")[p] for p in pids if p in ix_arm(m, l, "X") and p in ix_arm(m, l, "Y")} for l in rl]
            nm = mean_dicts([x for x in nm if x])
            e_name = st.mean(nm.values()) if nm else None
            en_lo, en_hi = A.boot(list(nm.values())) if len(nm) >= 2 else (float("nan"), float("nan"))
            lg = []
            for la, lb in COMPS[d]:
                if la in rl and lb in rl:
                    for arm in ARMS:
                        ia, ib = ix_arm(m, la, arm), ix_arm(m, lb, arm)
                        lg.append({p: ia[p] - ib[p] for p in pids if p in ia and p in ib})
            lgm = mean_dicts([x for x in lg if x])
            e_lang = st.mean(lgm.values()) if lgm else None
            el_lo, el_hi = A.boot(list(lgm.values())) if len(lgm) >= 2 else (float("nan"), float("nan"))
            h1 = (e_name is not None and e_lang is not None and en_lo == en_lo and e_name >= e_lang + 0.10 and en_lo > 0)
            h1s = "—" if e_name is None or e_lang is None else ("成立" if h1 else "不成立")
            out["H1"][f"{m}|{d}"] = {"E_name": e_name, "E_name_ci": [en_lo, en_hi], "E_lang": e_lang, "E_lang_ci": [el_lo, el_hi], "H1": h1s, "langs": rl}
            first = True
            for la, lb in COMPS[d]:
                ok = la in rl and lb in rl
                origA = {}
                for p in pids:  # each language's own tw2 form
                    a_arm = B.orig_arm(d, la, B.NAME_PAIRS[d][p]) or ("orig_lc" if d == "falklands" else "dual")
                    b_arm = B.orig_arm(d, lb, B.NAME_PAIRS[d][p]) or ("orig_lc" if d == "falklands" else "dual")
                    ia, ib = ix_arm(m, la, a_arm), ix_arm(m, lb, b_arm)
                    if p in ia and p in ib:
                        origA[p] = ia[p] - ib[p]
                g_orig = st.mean(origA.values()) if origA else None
                gN, lo, hi, _ = gap_stats(ix_arm(m, la, "N"), ix_arm(m, lb, "N"))
                fl, n = flips(cells_arm(m, "N"), (d, la), (d, lb), pids)
                v = verdict(gN, lo, hi) if ok else "讀不懂，不判"
                half = (gN is not None and g_orig is not None and g_orig > 0 and gN <= 0.5 * g_orig) if ok else None
                out["H2"][f"{m}|{d}|{la}-{lb}"] = {"orig": g_orig, "neutral": gN, "ci": [lo, hi], "flips": fl, "n": n, "verdict": v, "halved": half}
                lead = (f"| {LABEL[m]} | {f(e_name)} [{f(en_lo)}, {f(en_hi)}] | {f(e_lang)} [{f(el_lo)}, {f(el_hi)}] | {h1s} " if first
                        else "| | | | ")
                md.append(lead + f"| {la} − {lb} | {f(g_orig)} | {f(gN)} [{f(lo)}, {f(hi)}] | {fl}/{n} | {v} | {'—' if half is None else ('✓' if half else '✗')} |")
                first = False
        md.append("")

    # ---------------- E2 / H3
    md += ["## E2 · 上線做法：先翻英文（pivot）vs 先翻英文再套地名對照表（pivot＋names）", "",
           "全部 4 對。H3 通過 = pivot＋names 的主場偏差「沒問題」且翻面 ≤ 2/24，該爭議每個比較都要過。讀得懂用 pivot 文字自己的中性題。", "",
           "| 模型 | 爭議 | 比較 | pivot 偏差 [CI]／翻面 | pivot＋names 偏差 [CI]／翻面 | 讀得懂 | pivot＋names 判定 | H3 |", "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for d in B.DISPUTES:
            pids = [p["pid"] for p in B.MASTER[d]["pairs"]]
            rows, passes = [], []
            for la, lb in COMPS[d]:
                res = {}
                for arm in ("pivot", "pivot_names"):
                    c = {k[:2] + k[3:]: v for k, v in E2[m].items() if k[0] == d and k[2] == arm}
                    g, lo, hi, _ = gap_stats(index(c, (d, la), pids), index(c, (d, lb), pids))
                    fl, n = flips(c, (d, la), (d, lb), pids)
                    res[arm] = (g, lo, hi, fl, n)
                # readability on the pivot texts' own neutral pairs (pivot_names neutrals are the same text)
                neu = []
                for l in (la, lb):
                    c = {k[:2] + k[3:]: v for k, v in E2[m].items() if k[0] == d and k[2] == "pivot_names"}
                    vv = [c[(d, l, pid, "true", ph)] - c[(d, l, pid, "false", ph)] for pid in ("n1", "n2") for ph in ("p1", "p2", "p3")
                          if (d, l, pid, "true", ph) in c and (d, l, pid, "false", ph) in c]
                    neu.append(st.mean(vv) if vv else None)
                ok = all(x is not None and x >= 0.8 for x in neu)
                g, lo, hi, fl, n = res["pivot_names"]
                v = verdict(g, lo, hi) if ok else "讀不懂，不判"
                p3 = ok and v == "沒問題" and fl <= 2
                passes.append(p3)
                out["E2"][f"{m}|{d}|{la}-{lb}"] = {a: {"gap": r[0], "ci": [r[1], r[2]], "flips": r[3], "n": r[4]} for a, r in res.items()} | {"neutral": neu, "verdict": v, "pass": p3}
                pv = res["pivot"]
                rows.append(f"| {LABEL[m]} | {d} | {la} − {lb} | {f(pv[0])} [{f(pv[1])}, {f(pv[2])}]／{pv[3]} | {f(g)} [{f(lo)}, {f(hi)}]／{fl} | {'✓' if ok else '✗'} | {v} | {'✓' if p3 else '✗'} |")
            out["H3"][f"{m}|{d}"] = all(passes)
            md += rows
    md.append("")
    md += ["| 模型 | H3 三個爭議都過 |", "|---|---|"] + [f"| {LABEL[m]} | {'✓' if all(out['H3'][f'{m}|{d}'] for d in B.DISPUTES) else '✗'}（"
                                                    + "、".join(f"{d} {'✓' if out['H3'][f'{m}|{d}'] else '✗'}" for d in B.DISPUTES) + "） |" for m in MODELS]
    # 26b rerun check: tw2 prod_pivot vs tw3 pivot arm (same text, same server build)
    a, b = E2["26b/tw2"], {k: v for k, v in E2["26b"].items() if k[2] == "pivot"}
    diffs = [abs(a[k] - b[k]) for k in a if k in b]
    fl = sum((a[k] > 0.5) != (b[k] > 0.5) for k in a if k in b)
    out["rerun_26b_pivot"] = {"n": len(diffs), "max_abs": max(diffs) if diffs else None, "mean_abs": st.mean(diffs) if diffs else None, "flips": fl}
    md += ["", f"26B pivot 重跑檢查（tw2 prod_pivot vs tw3 pivot，同文字）：{len(diffs)} 筆，最大差 {f(max(diffs) if diffs else None, 3)}，平均差 "
           f"{f(st.mean(diffs) if diffs else None, 3)}，翻面 {fl}。", ""]
    json.dump(out, open(os.path.join(REPO, "results", "tw3.json"), "w"), ensure_ascii=False, indent=1, default=str)
    open(os.path.join(REPO, "results", "tw3-place-names-tables.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
