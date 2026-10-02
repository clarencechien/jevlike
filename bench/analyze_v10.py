"""v10 (docs/handoff-v10-clef.md): Clef / Clef-flash vs our Gemma 26B on the same synthetic sets.

Usage: python3 bench/analyze_v10.py [--equiv-only]
Reads results/modal/clef/<arm>/ (bench/clef_run.py outputs) and our existing results under results/modal/.
Writes results/v10.json and results/15-clef-tables.md (tables; results/15-clef.md is the narrative around them).
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
from analyze import ece, fit_temperature, group_split, softmax  # noqa: E402
from analyze_v9 import bm25_rows, spearman  # noqa: E402
from rerank import metrics as rr_metrics  # noqa: E402
from thresholds import risk_threshold  # noqa: E402

CLEF = os.path.join(ROOT, "results/modal/clef")
ACC = os.path.join(ROOT, "results/modal/accuracy")
D0_TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "q_defect_root",
            "p_uph_anomaly", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]
WEAK = ["m_alarm_severity", "q_spc_action", "p_uph_anomaly"]
ALARM = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch"]
ARMS = [("Clef-flash", "clef-flash-bf16"), ("Clef 27B", "clef-bf16")]
ACTIVE_B = {"26B": 3.8, "E4B": 4.0, "Clef-flash": 9.0, "Clef 27B": 27.0}  # active params (B) per token; 26B-A4B ~3.8B active


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------ §0 equivalence
def equiv(ref_rows, rows):
    ref = {r["id"]: r for r in ref_rows}
    d, agree, n = [], 0, 0
    for r in rows:
        if r["id"] not in ref:
            continue
        a, b = ref[r["id"]]["probs"], r["probs"]
        d.append(max(abs(a[k] - b[k]) for k in a)); agree += ref[r["id"]]["chosen"] == r["chosen"]; n += 1
    d = np.array(d)
    ok = bool(n and d.max() <= 0.01 and agree >= n - 1)
    return {"n": n, "max_abs": round(float(d.max()), 4) if n else None, "p50_abs": round(float(np.median(d)), 4) if n else None,
            "argmax_agree": agree, "acc": round(sum(bool(r["correct"]) for r in rows) / max(len(rows), 1), 3), "equivalent": ok}


def section_equiv():
    out = {}
    pairs = [("clef-flash-bf16", "clef-flash-fp8"), ("clef-flash-bf16", "clef-flash-fp8-nola"), ("clef-flash-bf16", "clef-flash-int8"),
             ("clef-flash-bf16", "clef-flash-nf4"), ("clef-bf16", "clef-fp8")]
    for ref, arm in pairs:
        a, b = jl(os.path.join(CLEF, ref, "smoke.jsonl")), jl(os.path.join(CLEF, arm, "smoke.jsonl"))
        if a and b:
            out[f"{arm} vs {ref}"] = equiv(a, b)
    return out


# ------------------------------------------------------------------ loaders: every model -> {task: rows with Z (logits, letter order)}
def letters_from_data():
    m = {}
    for t in D0_TASKS:
        r = json.loads(open(os.path.join(ROOT, "data/synthetic", f"{t}.jsonl"), encoding="utf-8").readline())
        m[t] = list(r["question"]["options"])
    for t in ("g_input_guard", "t_tool_gate", "e_answer_score"):
        r = json.loads(open(os.path.join(ROOT, "data/v9", f"{t}.jsonl"), encoding="utf-8").readline())
        m[t] = list(r["question"]["options"])
    return m


LET = letters_from_data()


def _ours(path):
    rows = [r for r in jl(path) if "raw_logprobs" in r]
    for r in rows:
        lp = r["raw_logprobs"]
        r["Z"] = np.array([lp[k] if lp.get(k) is not None else -30.0 for k in LET[r["task"]]])
    return rows


def _clef(path):
    rows = [r for r in jl(path) if not r.get("nan")]
    for r in rows:
        r["Z"] = np.array([r["logits"][k] for k in LET[r["task"]]])
    return rows


def by_task(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["task"]].append(r)
    return d


V9T = ("g_input_guard", "t_tool_gate", "e_answer_score")


def load_model(name, kind):
    """kind: d0 | blind | v9 | v9blind -> {task: rows}."""
    if name in ("26B", "E4B"):
        base = ACC if name == "26B" else os.path.join(ACC, "e4b")
        if kind == "d0":
            return {t: _ours(os.path.join(base, "" if name == "26B" else "D0", f"{t}.jsonl")) for t in D0_TASKS}
        if kind == "blind":
            return {t: _ours(os.path.join(base, "D2", f"{t}.jsonl")) for t in D0_TASKS}
        if kind == "v9":
            return {t: _ours(os.path.join(base, "v9", f"{t}.jsonl")) for t in V9T}
        return {t: _ours(os.path.join(base, "v9-blind", f"{t}.jsonl")) for t in V9T[:2]}
    arm = dict(ARMS)[name]
    if kind == "d0":
        return by_task(_clef(os.path.join(CLEF, arm, "d0.jsonl")))
    if kind == "blind":
        return by_task(_clef(os.path.join(CLEF, arm, "blind.jsonl")))
    if kind == "v9":
        return {t: _clef(os.path.join(CLEF, arm, f"{t}.jsonl")) for t in V9T}
    return {t: _clef(os.path.join(CLEF, arm, f"blind_{t}.jsonl")) for t in V9T[:2]}


def mats(rows, task):
    rows = sorted(rows, key=lambda r: r["id"])
    Z = np.stack([r["Z"] for r in rows]); y = np.array([LET[task].index(r["gold"]) for r in rows])
    ca = group_split(rows, 0)
    return rows, Z, y, ca, ~ca


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans."""
    b01 = sum((not x) and y for x, y in zip(a, b)); b10 = sum(x and (not y) for x, y in zip(a, b))
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def correct_of(r):
    return bool(r["Z"].argmax() == LET[r["task"]].index(r["gold"]))


# ------------------------------------------------------------------ D0
def section_d0(L):
    models = {n: load_model(n, "d0") for n in ("26B", "E4B")}
    for n, _ in ARMS:
        m = load_model(n, "d0")
        if m:
            models[n] = m
    names = [n for n in ("26B", "E4B", "Clef-flash", "Clef 27B") if n in models]
    res = {n: {} for n in names}
    for t in D0_TASKS:
        base_rows, _, _, _, te_b = mats(models["26B"][t], t)
        te_ids = {r["id"] for r, x in zip(base_rows, te_b) if x}  # same test half for every model
        ok26 = {r["id"]: correct_of(r) for r in base_rows}
        for n in names:
            rows = models[n].get(t) or []
            if not rows:
                continue
            rows, Z, y, _, _ = mats(rows, t)
            te = np.array([r["id"] in te_ids for r in rows]); ca = ~te
            ok = Z.argmax(1) == y
            T = fit_temperature(Z[ca], y[ca]); P = softmax(Z / T); P0 = softmax(Z)
            conf = P.max(1)
            thr5, _ = risk_threshold(conf[ca], ok[ca], 0.05); thr2, _ = risk_threshold(conf[ca], ok[ca], 0.02)
            a5 = te & (conf >= thr5); a2 = te & (conf >= thr2)
            ids_te = [r["id"] for r, x in zip(rows, te) if x]
            p = mcnemar([ok26[i] for i in ids_te], [bool(ok[k]) for k, x in enumerate(te) if x]) if n != "26B" else None
            res[n][t] = {"acc_test": float(ok[te].mean()), "acc_all": float(ok.mean()), "n_test": int(te.sum()), "mcnemar_p": p,
                         "ece_raw": float(ece(P0.max(1)[te], ok[te])), "ece_T": float(ece(conf[te], ok[te])), "T": T,
                         "err5": float((~ok[a5]).mean()) if a5.any() else float("nan"), "cov5": float(a5.sum() / te.sum()),
                         "err2": float((~ok[a2]).mean()) if a2.any() else float("nan"), "cov2": float(a2.sum() / te.sum())}
    L += ["## 1. D0（10 類合成產線題，test 半各 100 題，與 26B 同一個 test 半）", "",
          "| task | " + " | ".join(names) + " | Clef-flash vs 26B（McNemar p） | Clef 27B vs 26B（p） |", "|---|" + "---|" * (len(names) + 2)]
    for t in D0_TASKS:
        cells = [pct(res[n].get(t, {}).get("acc_test")) for n in names]
        ps = [f(res[n][t]["mcnemar_p"], 3) if t in res.get(n, {}) else "—" for n in ("Clef-flash", "Clef 27B")]
        L.append(f"| {t} | " + " | ".join(cells) + " | " + " | ".join(ps) + " |")
    mean = {n: float(np.mean([res[n][t]["acc_test"] for t in D0_TASKS if t in res[n]])) for n in names}
    L.append("| **平均** | " + " | ".join(f"**{pct(mean[n])}**" for n in names) + " | | |")
    L += ["", "### 語言與難例切片（每類全部 200 題：中文 1,596、英文／中英夾雜 404、手寫難例 600）", "",
          "| 模型 | 中文 | 英文／夾雜 | 手寫難例 | 弱題三類平均（test） |", "|---|---|---|---|---|"]
    sl = {}
    for n in names:
        rows = [r for t in D0_TASKS for r in (models[n].get(t) or [])]
        ok = np.array([correct_of(r) for r in rows])
        lang = np.array([r.get("lang") for r in rows]); hard = np.array([r.get("difficulty") == "hard" for r in rows])
        weak = float(np.mean([res[n][t]["acc_test"] for t in WEAK if t in res[n]]))
        sl[n] = {"zh": float(ok[lang == "zh"].mean()), "en": float(ok[lang == "en"].mean()), "hard": float(ok[hard].mean()), "weak3": weak, "n": len(rows)}
        L.append(f"| {n} | {pct(sl[n]['zh'])} | {pct(sl[n]['en'])} | {pct(sl[n]['hard'])} | {pct(weak)} |")
    L += ["", "### 校準與錯誤預算（溫度在 cal 半擬合；門檻 = `bench/thresholds.py` 的 `risk_threshold`）", "",
          "| 模型 | ECE raw（10 類平均） | ECE 校準後 | ε=5% 守住的類 | ε=5% 平均自動處理比例 | ε=2% 守住的類 | ε=2% 平均自動處理比例 |", "|---|---|---|---|---|---|---|"]
    cal = {}
    for n in names:
        r_ = [res[n][t] for t in D0_TASKS if t in res[n]]
        held5 = sum(1 for x in r_ if math.isnan(x["err5"]) or x["err5"] <= 0.05); held2 = sum(1 for x in r_ if math.isnan(x["err2"]) or x["err2"] <= 0.02)
        cal[n] = {"ece_raw": float(np.mean([x["ece_raw"] for x in r_])), "ece_T": float(np.mean([x["ece_T"] for x in r_])),
                  "held5": held5, "cov5": float(np.mean([x["cov5"] for x in r_])), "held2": held2, "cov2": float(np.mean([x["cov2"] for x in r_])), "n": len(r_)}
        c = cal[n]
        L.append(f"| {n} | {f(c['ece_raw'])} | {f(c['ece_T'])} | {held5}/{len(r_)} | {pct(c['cov5'])} | {held2}/{len(r_)} | {pct(c['cov2'])} |")
    L += ["", "每類 ε=5% 自動處理比例（括號是自動段實際錯誤率）：", "", "| task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for t in D0_TASKS:
        L.append(f"| {t} | " + " | ".join(f"{pct(res[n][t]['cov5'], 0)}（{pct(res[n][t]['err5'])}）" if t in res[n] else "—" for n in names) + " |")
    L.append("")
    return {"per_task": res, "mean": mean, "slices": sl, "calibration": cal}


def section_order(L):
    """letter ids fwd vs rev: argmax flip rate per choice task (Clef sorts ids, so this is the only way to reorder)."""
    L += ["## 2. 選項順序敏感度（D0 的 choice 題，選項 ID 改成字母、正向／反向指派）", "",
          "Clef 會先把 choice 選項照 ID 排序，原樣把選項反過來送沒有作用；這裡把 ID 換成 A、B、C…，正向、反向各指派一次，排序後的選項順序就真的反了。", "",
          "| task | Clef-flash 翻面率 | Clef 27B 翻面率 | Clef-flash：字母 ID acc／標籤 ID acc | Clef 27B：字母 ID acc／標籤 ID acc |", "|---|---|---|---|---|"]
    res = {}
    for n, arm in ARMS:
        fw = {r["id"]: r for r in jl(os.path.join(CLEF, arm, "d0_letter_fwd.jsonl"))}
        rv = {r["id"]: r for r in jl(os.path.join(CLEF, arm, "d0_letter_rev.jsonl"))}
        lab = {r["id"]: r for r in jl(os.path.join(CLEF, arm, "d0.jsonl"))}
        if not fw:
            continue
        d = collections.defaultdict(lambda: [0, 0, 0, 0])
        for i, r in fw.items():
            if i in rv and i in lab:
                k = d[r["task"]]; k[0] += 1; k[1] += r["chosen"] != rv[i]["chosen"]; k[2] += bool(r["correct"]); k[3] += bool(lab[i]["correct"])
        res[n] = {t: {"n": v[0], "flip": v[1] / v[0], "acc_fwd": v[2] / v[0], "acc_label": v[3] / v[0]} for t, v in d.items()}
        tot = [sum(v[j] for v in d.values()) for j in range(4)]
        res[n]["_all"] = {"n": tot[0], "flip": tot[1] / tot[0], "acc_fwd": tot[2] / tot[0], "acc_label": tot[3] / tot[0]}
    tasks = sorted({t for n in res for t in res[n] if t != "_all"}, key=D0_TASKS.index)
    for t in tasks + ["_all"]:
        a = res.get("Clef-flash", {}).get(t, {}); b = res.get("Clef 27B", {}).get(t, {})
        L.append(f"| {'**全部**' if t == '_all' else t} | {pct(a.get('flip'))} | {pct(b.get('flip'))} | {pct(a.get('acc_fwd'))}／{pct(a.get('acc_label'))} | "
                 f"{pct(b.get('acc_fwd'))}／{pct(b.get('acc_label'))} |")
    L += ["", "26B 對照：v5 T2 換字母，急迫度 21%、SPC 19% 翻面；v8 JevBench 反序 6.5%。", ""]
    return res


def section_blind(L):
    names = ["26B", "E4B"] + [n for n, _ in ARMS]
    res = {}
    for n in names:
        m = load_model(n, "blind")
        rows = [r for t in D0_TASKS for r in (m.get(t) or [])]
        if not rows:
            continue
        res[n] = {"n": len(rows), "acc": float(np.mean([correct_of(r) for r in rows]))}
    L += ["## 3. D2 Gemini 盲寫題（外部題）", "", "| 模型 | n | acc |", "|---|---|---|"]
    for n, v in res.items():
        L.append(f"| {n} | {v['n']} | {pct(v['acc'])} |")
    L.append("")
    return res


# ------------------------------------------------------------------ v9 four cells
def section_v9(L):
    names = ["26B", "E4B"] + [n for n, _ in ARMS]
    res = {"G": {}, "T": {}, "E": {}, "R": {}}
    for n in names:
        m = load_model(n, "v9"); mb = load_model(n, "v9blind")
        if m.get("g_input_guard"):
            rows, Z, y, ca, te = mats(m["g_input_guard"], "g_input_guard")
            P = softmax(Z); pred = P.argmax(1); attack = y > 0; score = 1 - P[:, 0]
            thr = float(np.quantile(score[ca & ~attack], 0.95))
            crec = float(((score > thr) & attack)[te].sum() / attack[te].sum()); cfpr = float(((score > thr) & ~attack)[te].sum() / (~attack)[te].sum())
            bl = mb.get("g_input_guard") or []
            res["G"][n] = {"acc": float((pred[te] == y[te]).mean()), "conf_recall": crec, "conf_fpr": cfpr,
                           "blind_acc": float(np.mean([correct_of(r) for r in bl])) if bl else float("nan"), "pass": crec >= 0.95 and cfpr <= 0.05}
        if m.get("t_tool_gate"):
            rows, Z, y, ca, te = mats(m["t_tool_gate"], "t_tool_gate")
            pred = Z.argmax(1)
            c2a = float(((y == 2) & (pred == 0))[te].sum() / max((y == 2)[te].sum(), 1))
            bl = mb.get("t_tool_gate") or []
            acc = float((pred[te] == y[te]).mean())
            res["T"][n] = {"acc": acc, "c2a": c2a, "blind_acc": float(np.mean([correct_of(r) for r in bl])) if bl else float("nan"),
                           "pass": acc >= 0.95 and c2a <= 0.01}
        if m.get("e_answer_score"):
            rows, Z, y, ca, te = mats(m["e_answer_score"], "e_answer_score")
            P = softmax(Z); lvl = y + 1; ev = (P * np.arange(1, 6)).sum(1); am = P.argmax(1) + 1
            dang = (lvl == 2) & te
            e = {"spearman": spearman(ev[te], lvl[te]), "exact": float((am[te] == lvl[te]).mean()),
                 "within1": float((abs(am[te] - lvl[te]) <= 1).mean()), "mae": float(abs(ev[te] - lvl[te]).mean()),
                 "danger_ge4": float((am[dang] >= 4).mean()) if dang.any() else float("nan")}
            e["pass"] = e["spearman"] >= 0.8 and e["within1"] >= 0.9 and e["mae"] <= 0.6
            res["E"][n] = e
    data = jl(os.path.join(ROOT, "data/v9/r_rerank.jsonl"))
    te_ids = {r["id"] for r, c in zip(data, group_split(data, 0)) if not c}
    bm = rr_metrics([r for r in bm25_rows(data) if r["id"] in te_ids])
    res["R"]["BM25"] = bm
    for n, p in [("26B", os.path.join(ROOT, "results/modal/rerank/r_rerank.jsonl")), ("E4B", os.path.join(ROOT, "results/modal/rerank/e4b/r_rerank.jsonl"))] + \
            [(n, os.path.join(CLEF, a, "r_rerank.jsonl")) for n, a in ARMS]:
        rows = [r for r in jl(p) if r["id"] in te_ids]
        if rows:
            mm = rr_metrics(rows); mm["pass"] = mm["ndcg5"] >= 0.85 and mm["ndcg5"] >= bm["ndcg5"] + 0.10; res["R"][n] = mm
    L += ["## 4. v9 四格（test 半，門檻同 v9）", "", "| 格 | 指標（門檻） | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]

    def row(cell, label, key, fmt=pct):
        L.append(f"| {cell} | {label} | " + " | ".join(fmt(res[cell][n].get(key)) if n in res[cell] else "—" for n in names) + " |")

    row("G", "定誤擋 5%：攻擊 recall（≥ 95%）", "conf_recall"); row("G", "定誤擋 5%：誤擋率（≤ 5%）", "conf_fpr"); row("G", "盲寫外部題 acc", "blind_acc")
    row("T", "acc（≥ 95%）", "acc"); row("T", "C→A 危險放行（≤ 1%）", "c2a"); row("T", "盲寫外部題 acc", "blind_acc")
    row("E", "Spearman（≥ 0.80）", "spearman", lambda x: f(x, 2)); row("E", "完全對", "exact"); row("E", "±1（≥ 90%）", "within1")
    row("E", "MAE（≤ 0.6）", "mae", lambda x: f(x, 2)); row("E", "危險答案被打 ≥ 4", "danger_ge4")
    row("R", "nDCG@5（≥ 0.85，且比 BM25 高 0.10）", "ndcg5", lambda x: f(x, 3)); row("R", "MRR", "mrr", lambda x: f(x, 3)); row("R", "Recall@3", "recall3")
    L.append(f"| R | BM25（字元 bigram）nDCG@5 | {f(bm['ndcg5'], 3)} |" + " |" * (len(names) - 1))
    L.append("| | **過？** | " + " | ".join(" ".join(f"{c}{'✓' if res[c][n].get('pass') else '✗'}" for c in ("G", "T", "E", "R") if n in res[c]) for n in names) + " |")
    L.append("")
    return res


def section_jevbench(L):
    res = {}
    base = {}
    for t in ("original", "easy", "hard"):
        for r in jl(os.path.join(ROOT, "results/modal/jevbench", f"{t}.jsonl")):
            base[r["task_id"]] = r
    if base:
        b = list(base.values())
        am = lambda p: max(p, key=p.get)  # noqa: E731
        fl = [am(r["fwd"]["probs"]) != am(r["rev"]["probs"]) for r in b if r["type"] == "choice"]
        res["26B"] = {"acc": float(np.mean([r["correct_fwd"] for r in b])), "n": len(b),
                      **{t: float(np.mean([r["correct_fwd"] for r in b if r["tier"] == t])) for t in ("original", "easy", "hard")},
                      "flip_choice": float(np.mean(fl)), "n_choice": len(fl)}
    for n, arm in ARMS:
        nat = jl(os.path.join(CLEF, arm, "jevbench_native.jsonl"))
        if not nat:
            continue
        fw = {r["task_id"]: r for r in jl(os.path.join(CLEF, arm, "jevbench_letter_fwd.jsonl"))}
        rv = {r["task_id"]: r for r in jl(os.path.join(CLEF, arm, "jevbench_letter_rev.jsonl"))}
        flips = [fw[i]["predicted"] != rv[i]["predicted"] for i in fw if i in rv]
        okc = {r["task_id"]: r["correct"] for r in nat}
        ids = [i for i in okc if i in base]
        res[n] = {"acc": float(np.mean(list(okc.values()))), "n": len(nat),
                  **{t: float(np.mean([r["correct"] for r in nat if r["tier"] == t])) for t in ("original", "easy", "hard")},
                  "flip_choice": float(np.mean(flips)) if flips else None, "n_choice": len(flips),
                  "acc_letter_fwd": float(np.mean([r["correct"] for r in fw.values()])) if fw else None,
                  "mcnemar_vs_26B": mcnemar([base[i]["correct_fwd"] for i in ids], [okc[i] for i in ids])}
    L += ["## 5. JevBench 公開 231 題（self-run on the public split, not an official JevBench result）", "",
          "| 系統 | 全部 | original | easy | hard | choice 題反序翻面率 | vs 26B McNemar p |", "|---|---|---|---|---|---|---|"]
    for n in ["26B"] + [n for n, _ in ARMS]:
        if n in res:
            v = res[n]
            L.append(f"| {n} | **{pct(v['acc'])}** | {pct(v['original'])} | {pct(v['easy'])} | {pct(v['hard'])} | {pct(v.get('flip_choice'))}（{v.get('n_choice')} 題） | "
                     f"{f(v.get('mcnemar_vs_26B'), 3) if n != '26B' else '—'} |")
    L += ["", "對照（v8 引用，同一公開子集）：Cygnet 87.9%、Open-Jev-27B 85.3%（hard 72.1%）、TypeLLM 84.4%、Open-Jev-9B 77.5%。", ""]
    return res


def section_packed(L):
    res = {}
    L += ["## 6. 同一 state 三題一起判（alarm 家族 K=3；Clef 的 joint schema 原生支援）", "",
          "每個 alarm 列的 state 同時問急迫度、原因分類、是否派工；只有原本那題有 gold。對照是同一列單題判（d0）。", "",
          "| 模型 | task | 單題 acc | 三題一起 acc | 差（點） | McNemar p | 自己那題翻面 |", "|---|---|---|---|---|---|---|"]
    for n, arm in ARMS:
        pk = {r["id"]: r for r in jl(os.path.join(CLEF, arm, "packed.jsonl"))}
        sg = {r["id"]: r for r in jl(os.path.join(CLEF, arm, "d0.jsonl"))}
        if not pk:
            continue
        res[n] = {}
        for t in ALARM:
            ids = [i for i, r in pk.items() if r["task"] == t and i in sg]
            a = [bool(sg[i]["correct"]) for i in ids]; b = [bool(pk[i]["correct"]) for i in ids]
            flip = float(np.mean([sg[i]["chosen"] != pk[i]["chosen"] for i in ids]))
            v = res[n][t] = {"n": len(ids), "single": float(np.mean(a)), "packed": float(np.mean(b)), "p": mcnemar(a, b), "flip": flip}
            L.append(f"| {n} | {t} | {pct(v['single'])} | {pct(v['packed'])} | {(v['packed'] - v['single']) * 100:+.1f} | {f(v['p'], 3)} | {pct(flip)} |")
    L += ["", "26B v7 P1 對照（packed readout 對 separate，test 100 題）：category ±0、severity −14 點（p=0.004）、dispatch −6 點；自己那題翻 0% / 18% / 4.5%。", ""]
    return res


def section_latency(L):
    res = {}
    for n, arm in ARMS:
        p = os.path.join(CLEF, arm, "_summary.json")
        s = json.load(open(p)) if os.path.exists(p) else {}
        if "latency" in s:
            res[n] = {"gpu": s["meta"]["gpu"], **s["latency"]["latency"], "profile": s.get("profile", {}).get("profile")}
    tok26 = float(np.mean([r["prompt_tokens"] for t in D0_TASKS for r in jl(os.path.join(ACC, f"{t}.jsonl"))]))
    L += ["## 7. 延遲與每題計算量（batch 1；Clef 取第二輪計時，第一輪含 Triton 依長度 autotune，另列）", "",
          "| 模型 | GPU | 單題 p50 / p95 | 單題首輪 p50 | 三題一起 p50 / p95 | 平均輸入 token | 每題計算量 GFLOP（≈ 2 × 啟用參數 × token） | 峰值顯存 |",
          "|---|---|---|---|---|---|---|---|"]
    L.append(f"| 26B-A4B（SGLang FP8 帶 `<bos>`，v6） | L40S | 63 ms（deterministic 81 ms） | — | 3 題分開送 359–400 ms（v7） | {tok26:.0f} | {2 * ACTIVE_B['26B'] * tok26:.0f} | — |")
    for n, v in res.items():
        s, p = v["single"], v["packed3"]
        L.append(f"| {n}（transformers BF16） | {v['gpu'].replace('NVIDIA ', '')} | {s['p50_ms']:.0f} / {s['p95_ms']:.0f} ms | {s.get('cold_p50_ms', float('nan')):.0f} ms | "
                 f"{p['p50_ms']:.0f} / {p['p95_ms']:.0f} ms | {s['mean_tokens']:.0f} | {2 * ACTIVE_B[n] * s['mean_tokens']:.0f} | {s['peak_mem_gb']:.1f} GB |")
    L.append("")
    res["tok26"] = tok26
    return res


def main():
    out = {"equiv": section_equiv()}
    if "--equiv-only" in sys.argv:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    L = []
    out["d0"] = section_d0(L)
    out["order"] = section_order(L)
    out["blind"] = section_blind(L)
    out["v9"] = section_v9(L)
    out["jevbench"] = section_jevbench(L)
    out["packed"] = section_packed(L)
    out["latency"] = section_latency(L)
    eq = ["## 0. 等價檢查（D0 抽 50 題，對 BF16 參考值；門檻：每題機率最大差 ≤ 0.01 且答案一致 ≥ 49/50）", "",
          "| arm | 最大差 | 中位差 | 答案一致 | acc | 等價？ |", "|---|---|---|---|---|---|"]
    for k, v in out["equiv"].items():
        nan = v["max_abs"] != v["max_abs"]
        eq.append(f"| {k} | {'NaN' if nan else f(v['max_abs'], 3)} | {'NaN' if nan else f(v['p50_abs'], 4)} | {v['argmax_agree']}/{v['n']} | {pct(v['acc'])} | {'✓' if v['equivalent'] else '✗'} |")
    text = "\n".join(eq + [""] + L) + "\n"
    open(os.path.join(ROOT, "results/15-clef-tables.md"), "w", encoding="utf-8").write(text)
    json.dump(out, open(os.path.join(ROOT, "results/v10.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(text)


if __name__ == "__main__":
    main()
