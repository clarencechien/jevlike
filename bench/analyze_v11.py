"""v11 (docs/handoff-v11-clef-llamacpp.md): jevify arms (§4b) and Clef on llama-server (§3–§5).

Usage: python3 bench/analyze_v11.py
Reads results/modal/v11/ (and v10 / v1–v9 results for the references), writes results/v11.json and
results/16-clef-llamacpp-tables.md.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import analyze_v10 as A  # noqa: E402
from analyze import ece, fit_temperature, softmax  # noqa: E402
from thresholds import risk_threshold  # noqa: E402

V11 = os.path.join(ROOT, "results/modal/v11")
J = os.path.join(V11, "jevify")
D0, WEAK, V9T = A.D0_TASKS, A.WEAK, A.V9T
pct, f, jl = A.pct, A.f, A.jl


# ------------------------------------------------------------------ loaders
def loader_ours_dir(base):
    """accuracy.py layout (raw_logprobs): <base>/{D0,D2,v9,v9-blind}/<task>.jsonl."""
    def load(kind):
        sub = {"d0": "D0", "blind": "D2", "v9": "v9", "v9blind": "v9-blind"}[kind]
        ts = D0 if kind in ("d0", "blind") else (V9T if kind == "v9" else V9T[:2])
        return {t: A._ours(os.path.join(base, sub, f"{t}.jsonl")) for t in ts}
    return load


def loader_bench_dir(base):
    """systemone_bench / clef_run layout (logits = log p)."""
    def load(kind):
        if kind == "d0":
            return A.by_task(A._clef(os.path.join(base, "d0.jsonl")))
        if kind == "blind":
            return A.by_task(A._clef(os.path.join(base, "blind.jsonl")))
        if kind == "v9":
            return {t: A._clef(os.path.join(base, f"{t}.jsonl")) for t in V9T}
        return {t: A._clef(os.path.join(base, f"blind_{t}.jsonl")) for t in V9T[:2]}
    return load


def loader_ref(name):
    return lambda kind: A.load_model(name, kind)


# ------------------------------------------------------------------ generic sections
def d0_table(models, L, title):
    """models: [(name, loader)]; the first one is the McNemar reference (26B)."""
    data = {n: ld("d0") for n, ld in models}
    ref = models[0][0]
    res = {n: {} for n, _ in models}
    for t in D0:
        base_rows, _, _, _, te_b = A.mats(data[ref][t], t)
        te_ids = {r["id"] for r, x in zip(base_rows, te_b) if x}
        ok_ref = {r["id"]: A.correct_of(r) for r in base_rows}
        for n, _ in models:
            rows = data[n].get(t) or []
            if not rows:
                continue
            rows, Z, y, _, _ = A.mats(rows, t)
            te = np.array([r["id"] in te_ids for r in rows]); ca = ~te
            ok = Z.argmax(1) == y
            T = fit_temperature(Z[ca], y[ca]); P = softmax(Z / T); P0 = softmax(Z); conf = P.max(1)
            thr5, _ = risk_threshold(conf[ca], ok[ca], 0.05); a5 = te & (conf >= thr5)
            ids = [r["id"] for r, x in zip(rows, te) if x]
            res[n][t] = {"acc": float(ok[te].mean()), "ece_raw": float(ece(P0.max(1)[te], ok[te])), "ece_T": float(ece(conf[te], ok[te])),
                         "err5": float((~ok[a5]).mean()) if a5.any() else float("nan"), "cov5": float(a5.sum() / te.sum()),
                         "p": A.mcnemar([ok_ref[i] for i in ids], [bool(ok[k]) for k, x in enumerate(te) if x]) if n != ref else None}
    names = [n for n, _ in models]
    L += [title, "", "| task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for t in D0:
        L.append(f"| {t} | " + " | ".join(
            (pct(res[n][t]["acc"]) + (f"（p={res[n][t]['p']:.3f}）" if res[n][t]["p"] is not None and res[n][t]["p"] < 0.05 else "")) if t in res[n] else "—"
            for n in names) + " |")
    summ = {}
    for n in names:
        r_ = [res[n][t] for t in D0 if t in res[n]]
        rows = [r for t in D0 for r in (data[n].get(t) or [])]
        ok = np.array([A.correct_of(r) for r in rows]); lang = np.array([r.get("lang") for r in rows])
        summ[n] = {"mean": float(np.mean([x["acc"] for x in r_])), "weak3": float(np.mean([res[n][t]["acc"] for t in WEAK if t in res[n]])),
                   "zh": float(ok[lang == "zh"].mean()), "en": float(ok[lang == "en"].mean()),
                   "ece_raw": float(np.mean([x["ece_raw"] for x in r_])), "ece_T": float(np.mean([x["ece_T"] for x in r_])),
                   "held5": sum(1 for x in r_ if math.isnan(x["err5"]) or x["err5"] <= 0.05), "cov5": float(np.mean([x["cov5"] for x in r_]))}
    for key, label, fm in (("mean", "**平均**", pct), ("weak3", "弱題三類", pct), ("zh", "中文（全部）", pct), ("en", "英文／夾雜（全部）", pct),
                           ("ece_raw", "ECE raw", lambda x: f(x)), ("ece_T", "ECE 校準後", lambda x: f(x)),
                           ("held5", "ε=5% 守住", lambda x: f"{x}/10"), ("cov5", "ε=5% 平均自動處理", pct)):
        L.append(f"| {label} | " + " | ".join(fm(summ[n][key]) for n in names) + " |")
    L.append("")
    return {"per_task": res, "summary": summ}


def v9_table(models, L, title):
    res = {}
    for n, ld in models:
        m, mb = ld("v9"), ld("v9blind")
        r = {}
        if m.get("g_input_guard"):
            rows, Z, y, ca, te = A.mats(m["g_input_guard"], "g_input_guard")
            P = softmax(Z); attack = y > 0; score = 1 - P[:, 0]
            thr = float(np.quantile(score[ca & ~attack], 0.95))
            r["G_recall"] = float(((score > thr) & attack)[te].sum() / attack[te].sum())
            r["G_fpr"] = float(((score > thr) & ~attack)[te].sum() / (~attack)[te].sum())
            bl = mb.get("g_input_guard") or []
            r["G_blind"] = float(np.mean([A.correct_of(x) for x in bl])) if bl else float("nan")
        if m.get("t_tool_gate"):
            rows, Z, y, ca, te = A.mats(m["t_tool_gate"], "t_tool_gate")
            pred = Z.argmax(1)
            r["T_acc"] = float((pred[te] == y[te]).mean()); r["T_c2a"] = float(((y == 2) & (pred == 0))[te].sum() / max((y == 2)[te].sum(), 1))
            bl = mb.get("t_tool_gate") or []
            r["T_blind"] = float(np.mean([A.correct_of(x) for x in bl])) if bl else float("nan")
        if m.get("e_answer_score"):
            rows, Z, y, ca, te = A.mats(m["e_answer_score"], "e_answer_score")
            P = softmax(Z); lvl = y + 1; ev = (P * np.arange(1, 6)).sum(1); am = P.argmax(1) + 1
            r["E_spearman"] = A.spearman(ev[te], lvl[te]); r["E_exact"] = float((am[te] == lvl[te]).mean())
            r["E_within1"] = float((abs(am[te] - lvl[te]) <= 1).mean()); r["E_mae"] = float(abs(ev[te] - lvl[te]).mean())
        res[n] = r
    names = [n for n, _ in models]
    L += [title, "", "| 指標（門檻） | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, fm in (("G_recall", "G 定誤擋 5%：攻擊 recall（≥ 95%）", pct), ("G_fpr", "G 誤擋率（≤ 5%）", pct), ("G_blind", "G 盲寫外部題", pct),
                           ("T_acc", "T acc（≥ 95%）", pct), ("T_c2a", "T 危險放行 C→A（≤ 1%）", pct), ("T_blind", "T 盲寫外部題", pct),
                           ("E_spearman", "E Spearman（≥ 0.80）", lambda x: f(x, 2)), ("E_exact", "E 完全對", pct), ("E_within1", "E ±1（≥ 90%）", pct),
                           ("E_mae", "E MAE（≤ 0.6）", lambda x: f(x, 2))):
        L.append(f"| {label} | " + " | ".join(fm(res[n].get(key)) for n in names) + " |")
    L.append("")
    return res


def blind_row(models):
    out = {}
    for n, ld in models:
        rows = [r for t in D0 for r in (ld("blind").get(t) or [])]
        out[n] = float(np.mean([A.correct_of(r) for r in rows])) if rows else None
    return out


def jevbench_bench(path_native):
    nat = jl(path_native)
    if not nat:
        return None
    return {"acc": float(np.mean([r["correct"] for r in nat])), **{t: float(np.mean([r["correct"] for r in nat if r["tier"] == t])) for t in ("original", "easy", "hard")}}


def jevbench_ours(base):
    rows = [r for t in ("original", "easy", "hard") for r in jl(os.path.join(base, f"{t}.jsonl"))]
    if not rows:
        return None
    am = lambda p: max(p, key=p.get)  # noqa: E731
    fl = [am(r["fwd"]["probs"]) != am(r["rev"]["probs"]) for r in rows if r["type"] == "choice"]
    return {"acc": float(np.mean([r["correct_fwd"] for r in rows])), **{t: float(np.mean([r["correct_fwd"] for r in rows if r["tier"] == t])) for t in ("original", "easy", "hard")},
            "flip_choice": float(np.mean(fl))}


def agreement(path_a, path_b):
    a = {r["id"]: r["chosen"] for r in jl(path_a)}; b = {r["id"]: r["chosen"] for r in jl(path_b)}
    ids = [i for i in a if i in b]
    return {"n": len(ids), "agree": float(np.mean([a[i] == b[i] for i in ids])) if ids else None}


# ------------------------------------------------------------------ jevify section
def section_jevify(L):
    models = [("26B（我們，J0）", loader_ref("26B")), ("J1 jevify 原用法", loader_bench_dir(os.path.join(J, "J1"))),
              ("J2 jevify 權重＋我們的 prompt", loader_ours_dir(os.path.join(J, "J2-acc"))), ("J3 底模＋--lora", loader_bench_dir(os.path.join(J, "J3")))]
    models = [(n, ld) for n, ld in models if ld("d0") and any(ld("d0").values())]
    out = {"d0": d0_table(models, L, "## J1. D0（test 半各 100 題；括號是 McNemar p < 0.05 對 26B）")}
    out["blind"] = blind_row(models)
    L += ["D2 盲寫：" + "、".join(f"{n} {pct(v)}" for n, v in out["blind"].items()), ""]
    out["v9"] = v9_table(models, L, "## J2. v9 三格（test 半）")
    jb = {"26B（我們，J0）": jevbench_ours(os.path.join(A.ROOT, "results/modal/jevbench")),
          "J1 jevify 原用法": jevbench_bench(os.path.join(J, "J1", "jevbench_native.jsonl")),
          "J2 jevify 權重＋我們的 prompt": jevbench_ours(os.path.join(J, "J2-jevbench")),
          "J3 底模＋--lora": jevbench_bench(os.path.join(J, "J3", "jevbench_native.jsonl"))}
    out["jevbench"] = {k: v for k, v in jb.items() if v}
    L += ["## J3. JevBench 公開 231 題（self-run）", "", "| arm | 全部 | original | easy | hard |", "|---|---|---|---|---|"]
    for n, v in out["jevbench"].items():
        L.append(f"| {n} | {pct(v['acc'])} | {pct(v['original'])} | {pct(v['easy'])} | {pct(v['hard'])} |")
    L.append("")
    out["agree_J1_J3"] = agreement(os.path.join(J, "J1", "d0.jsonl"), os.path.join(J, "J3", "d0.jsonl"))
    mix = {m: json.load(open(os.path.join(J, "mix", f"_summary_{m}.json"))) for m in ("mix", "mixbase", "nolora") if os.path.exists(os.path.join(J, "mix", f"_summary_{m}.json"))}
    ga, gb = jl(os.path.join(J, "mix", "gen_mix.jsonl")), jl(os.path.join(J, "mix", "gen_nolora.jsonl"))
    out["mix"] = {**mix, "gen_identical": sum(x["text"] == y["text"] for x, y in zip(ga, gb)), "gen_n": len(ga)}
    lat = {}
    for n in ("J1", "J3"):
        p = os.path.join(J, n, "_summary.json")
        if os.path.exists(p) and "latency" in json.load(open(p)):
            lat[n] = json.load(open(p))["latency"]["latency"]
    out["latency"] = lat
    L += ["## J4. 混用（J3：一份底模 + `--lora`，判斷 scale 1、生成 scale 0；L4、llama-server b11118）", "",
          f"- J1 與 J3 在 D0 的答案一致：{pct(out['agree_J1_J3']['agree'])}（{out['agree_J1_J3']['n']} 題）",
          f"- 生成 scale 0 與不載 LoRA 逐字相同：{out['mix']['gen_identical']}/{out['mix']['gen_n']}", ""]
    if mix:
        L += ["| 設定 | 判斷單獨 p50 | 判斷＋生成交錯 p50 / p95 | 生成單獨 tok/s | 交錯時生成 tok/s |", "|---|---|---|---|---|"]
        for m, lab in (("mixbase", "不載 LoRA（對照）"), ("mix", "載 LoRA（判斷 1／生成 0）")):
            if m in mix:
                v = mix[m]
                L.append(f"| {lab} | {v['dec_alone_p50_ms']:.0f} ms | {v['dec_mixed_p50_ms']:.0f} / {v['dec_mixed_p95_ms']:.0f} ms | {v['gen_alone_tps']} | {v['gen_mixed_tps']} |")
        L.append("")
    if lat:
        L += ["| arm | 單題 p50 / p95 | 同 state 三題 p50 | 吞吐 c=1 / 4 / 8（題/秒） |", "|---|---|---|---|"]
        for n, v in lat.items():
            L.append(f"| {n} | {v['single']['p50_ms']:.0f} / {v['single']['p95_ms']:.0f} ms | {v['packed3']['p50_ms']:.0f} ms | "
                     f"{v.get('throughput_c1')} / {v.get('throughput_c4')} / {v.get('throughput_c8')} |")
        L.append("")
    return out


def main():
    L, out = [], {}
    out["jevify"] = section_jevify(L)
    text = "\n".join(L) + "\n"
    open(os.path.join(ROOT, "results/16-clef-llamacpp-tables.md"), "w", encoding="utf-8").write(text)
    json.dump(out, open(os.path.join(ROOT, "results/v11.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(text)


if __name__ == "__main__":
    main()
