"""v23 analysis (docs/handoff-v23-input-format.md §4–5): five renderings × six tasks × three models.

Inputs:  results/modal/v23/{26b,e4b,clef-q8}/<R>/<task>.jsonl  (rows with id, chosen, correct, probs; llama rows also
         prompt_tokens, latency_ms), joined by id with data/v23/T0/<task>.jsonl (needs_compare, hard_type, family).
Outputs: results/v23.json, results/27-input-format-tables.md, results/27-input-format-errors.md
"""
import json
import math
import os
import statistics as st

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "modal", "v23")
TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "p_uph_anomaly", "t_tool_gate"]
SHORT = {"m_alarm_severity": "急迫度", "m_alarm_category": "原因分類", "m_needs_dispatch": "派工", "q_spc_action": "SPC", "p_uph_anomaly": "UPH", "t_tool_gate": "工具守門"}
RS = ["T0", "T1", "J0", "J1", "JN"]
MODELS = [("26b", "26B"), ("e4b", "E4B"), ("clef-q8", "Clef 27B Q8")]


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if os.path.exists(p) else []


META = {r["id"]: r for t in TASKS for r in jl(os.path.join(ROOT, "data", "v23", "T0", f"{t}.jsonl"))}
STATE = {R: {r["id"]: r["state"] for t in TASKS for r in jl(os.path.join(ROOT, "data", "v23", R, f"{t}.jsonl"))} for R in RS}


def mcnemar(a, b):
    b01 = sum((not x) and y for x, y in zip(a, b)); b10 = sum(x and (not y) for x, y in zip(a, b))
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def load(model):
    out = {}
    for R in RS:
        for t in TASKS:
            for r in jl(os.path.join(RES, model, R, f"{t}.jsonl")):
                if "correct" in r:
                    out[(R, r["id"])] = r
    return out


def acc(rows):
    return round(100 * sum(r["correct"] for r in rows) / len(rows), 1) if rows else None


def paired(D, ra, rb, ids):
    ids = [i for i in ids if (ra, i) in D and (rb, i) in D]
    a = [D[(ra, i)]["correct"] for i in ids]; b = [D[(rb, i)]["correct"] for i in ids]
    if not ids:
        return None
    return {"n": len(ids), ra: round(100 * sum(a) / len(ids), 1), rb: round(100 * sum(b) / len(ids), 1),
            "diff": round(100 * (sum(b) - sum(a)) / len(ids), 1), "p": round(mcnemar(a, b), 4)}


def analyze(model):
    D = load(model)
    if not D:
        return None
    ids_t = {t: [i for i in META if META[i]["task"] == t] for t in TASKS}
    res = {"acc": {}, "mean": {}, "nc": {}, "tokens": {}, "lat_p50": {}, "ece": {}, "n": {}}
    for R in RS:
        res["acc"][R] = {t: acc([D[(R, i)] for i in ids_t[t] if (R, i) in D]) for t in TASKS}
        vals = [v for v in res["acc"][R].values() if v is not None]
        res["mean"][R] = round(sum(vals) / len(vals), 1) if vals else None
        rows = [D[k] for k in D if k[0] == R]
        res["n"][R] = len(rows)
        res["nc"][R] = acc([D[(R, i)] for i in META if META[i]["needs_compare"] and (R, i) in D])
        tok = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
        res["tokens"][R] = round(st.mean(tok), 1) if tok else None
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
        res["lat_p50"][R] = round(st.median(lat), 1) if lat else None
        conf = [max(r["probs"].values()) for r in rows if r.get("probs")]
        res["ece"][R] = round(ece(conf, [r["correct"] for r in rows if r.get("probs")]), 3) if conf else None
    all_ids = list(META)
    nc_ids = [i for i in META if META[i]["needs_compare"]]
    res["H1"] = {"T1_vs_J1": {t: paired(D, "T1", "J1", ids_t[t]) for t in TASKS}, "T0_vs_J0": {t: paired(D, "T0", "J0", ids_t[t]) for t in TASKS},
                 "all_T1_vs_J1": paired(D, "T1", "J1", all_ids), "all_T0_vs_J0": paired(D, "T0", "J0", all_ids)}
    res["H2"] = {"nc_J0_vs_J1": paired(D, "J0", "J1", nc_ids), "nc_T0_vs_T1": paired(D, "T0", "T1", nc_ids),
                 "all_J0_vs_J1": paired(D, "J0", "J1", all_ids), "all_T0_vs_T1": paired(D, "T0", "T1", all_ids),
                 "per_task_J0_vs_J1": {t: paired(D, "J0", "J1", ids_t[t]) for t in TASKS},
                 "per_task_T0_vs_T1": {t: paired(D, "T0", "T1", ids_t[t]) for t in TASKS},
                 "not_nc_J0_vs_J1": paired(D, "J0", "J1", [i for i in META if not META[i]["needs_compare"]])}
    res["H3"] = {"J0_vs_JN": paired(D, "J0", "JN", all_ids),
                 "tokens_ratio": round(res["tokens"]["JN"] / res["tokens"]["J0"], 2) if res["tokens"].get("JN") and res["tokens"].get("J0") else None,
                 "lat_ratio": round(res["lat_p50"]["JN"] / res["lat_p50"]["J0"], 2) if res["lat_p50"].get("JN") and res["lat_p50"].get("J0") else None}
    tg = {}
    for ht in ("param_out_of_range", "multi_line_scope", "readonly_other_line", "role_permission"):
        tg[ht] = {R: acc([D[(R, i)] for i in ids_t["t_tool_gate"] if META[i]["hard_type"] == ht and (R, i) in D]) for R in RS}
        tg[ht]["J0_vs_J1"] = paired(D, "J0", "J1", [i for i in ids_t["t_tool_gate"] if META[i]["hard_type"] == ht])
    res["H4"] = tg
    res["H5"] = {"J0_vs_J1": paired(D, "J0", "J1", ids_t["q_spc_action"]), "T0_vs_T1": paired(D, "T0", "T1", ids_t["q_spc_action"]),
                 "by_kind": {k: {R: acc([D[(R, i)] for i in ids_t["q_spc_action"] if META[i]["hard_type"] == k and (R, i) in D]) for R in RS}
                             for k in ("run7", "mono6", "two3", "one_out", "second_out", "recheck")}}
    fam = {}
    for i, m in META.items():
        fam.setdefault((m["task"], m["family"]), []).append(i)
    res["family"] = {f"{t}|{f}": {R: acc([D[(R, i)] for i in ids if (R, i) in D]) for R in RS} for (t, f), ids in sorted(fam.items())}
    return res, D


def errors(D, k=10):
    """per task up to k events that T1 gets right and J0 gets wrong"""
    out = {}
    for t in TASKS:
        ids = [i for i in META if META[i]["task"] == t and ("T1", i) in D and ("J0", i) in D and D[("T1", i)]["correct"] and not D[("J0", i)]["correct"]]
        out[t] = [{"id": i, "gold": META[i]["gold"], "J0_chosen": D[("J0", i)]["chosen"], "family": META[i]["family"], "J0": STATE["J0"][i],
                   "J0_probs": D[("J0", i)].get("probs")} for i in ids[:k]]
        out[t + "_n"] = len(ids)
    return out


def fmt(v):
    return "—" if v is None else (f"{v:+.1f}" if isinstance(v, float) and False else str(v))


def main():
    full, tables, errs = {}, [], []
    for m, label in MODELS:
        r = analyze(m)
        if r is None:
            continue
        res, D = r
        full[m] = res
        tables += [f"## {label}", "", "### 準確率（%，每格 120 題）", "", "| task | " + " | ".join(RS) + " |", "|---|" + "---|" * len(RS)]
        for t in TASKS:
            tables.append(f"| {SHORT[t]} | " + " | ".join(fmt(res["acc"][R][t]) for R in RS) + " |")
        tables.append("| **平均** | " + " | ".join(f"**{fmt(res['mean'][R])}**" for R in RS) + " |")
        tables.append("| needs_compare 切片 | " + " | ".join(fmt(res["nc"][R]) for R in RS) + " |")
        tables.append("| ECE（raw） | " + " | ".join(fmt(res["ece"][R]) for R in RS) + " |")
        if res["tokens"].get("J0"):
            tables.append("| 平均 prompt token | " + " | ".join(fmt(res["tokens"][R]) for R in RS) + " |")
            tables.append("| 單題 p50 ms | " + " | ".join(fmt(res["lat_p50"][R]) for R in RS) + " |")
        tables += ["", "### 配對比較（同事件，McNemar exact）", "", "| 比較 | 切片 | n | 前 | 後 | 差 | p |", "|---|---|---|---|---|---|---|"]

        def row(name, sl, x, a, b):
            if x:
                tables.append(f"| {name} | {sl} | {x['n']} | {x[a]} | {x[b]} | {x['diff']:+.1f} | {x['p']} |")
        row("格式 T1→J1", "全部", res["H1"]["all_T1_vs_J1"], "T1", "J1")
        row("格式 T0→J0", "全部", res["H1"]["all_T0_vs_J0"], "T0", "J0")
        row("前處理 J0→J1", "needs_compare", res["H2"]["nc_J0_vs_J1"], "J0", "J1")
        row("前處理 T0→T1", "needs_compare", res["H2"]["nc_T0_vs_T1"], "T0", "T1")
        row("前處理 J0→J1", "不需比較", res["H2"]["not_nc_J0_vs_J1"], "J0", "J1")
        row("雜訊 J0→JN", "全部", res["H3"]["J0_vs_JN"], "J0", "JN")
        for t in TASKS:
            row("格式 T1→J1", SHORT[t], res["H1"]["T1_vs_J1"][t], "T1", "J1")
        for t in TASKS:
            row("格式 T0→J0", SHORT[t], res["H1"]["T0_vs_J0"][t], "T0", "J0")
        for t in TASKS:
            row("前處理 J0→J1", SHORT[t], res["H2"]["per_task_J0_vs_J1"][t], "J0", "J1")
        for t in TASKS:
            row("前處理 T0→T1", SHORT[t], res["H2"]["per_task_T0_vs_T1"][t], "T0", "T1")
        tables += ["", f"雜訊：JN／J0 token {res['H3']['tokens_ratio']} 倍、p50 {res['H3']['lat_ratio']} 倍。", "",
                   "### 工具守門 hard_type（%）", "", "| hard_type | " + " | ".join(RS) + " | J0→J1 p |", "|---|" + "---|" * (len(RS) + 1)]
        for ht, v in res["H4"].items():
            tables.append(f"| {ht} | " + " | ".join(fmt(v[R]) for R in RS) + f" | {v['J0_vs_J1']['p'] if v['J0_vs_J1'] else '—'} |")
        tables += ["", "### SPC 依 twin 種類（%）", "", "| 種類 | " + " | ".join(RS) + " |", "|---|" + "---|" * len(RS)]
        for k, v in res["H5"]["by_kind"].items():
            tables.append(f"| {k} | " + " | ".join(fmt(v[R]) for R in RS) + " |")
        tables += ["", "### 依 family（%）", "", "| task｜family | " + " | ".join(RS) + " |", "|---|" + "---|" * len(RS)]
        for f, v in res["family"].items():
            tables.append(f"| {f} | " + " | ".join(fmt(v[R]) for R in RS) + " |")
        tables.append("")
        if m == "26b":
            e = errors(D)
            full[m]["errors_T1_right_J0_wrong_n"] = {t: e[t + "_n"] for t in TASKS}
            errs += ["# v23 錯誤樣本：26B，T1 對、J0 錯（每類最多 10 個）", ""]
            for t in TASKS:
                errs += [f"## {t}（共 {e[t + '_n']} 個）", ""]
                for x in e[t]:
                    pr = {k: round(v, 2) for k, v in (x["J0_probs"] or {}).items()}
                    errs.append(f"- `{x['id']}` gold {x['gold']}，J0 選 {x['J0_chosen']} {pr}（{x['family']}）  \n  `{x['J0']}`")
                errs.append("")
    json.dump(full, open(os.path.join(ROOT, "results", "v23.json"), "w"), ensure_ascii=False, indent=1)
    open(os.path.join(ROOT, "results", "27-input-format-tables.md"), "w", encoding="utf-8").write(
        "# 27 — 輸入格式：完整表格（`bench/analyze_v23.py` 產生）\n\n" + "\n".join(tables) + "\n")
    if errs:
        open(os.path.join(ROOT, "results", "27-input-format-errors.md"), "w", encoding="utf-8").write("\n".join(errs) + "\n")
    print("\n".join(tables))


if __name__ == "__main__":
    main()
