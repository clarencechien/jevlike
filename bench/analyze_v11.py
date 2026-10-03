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


# ------------------------------------------------------------------ Clef on llama-server
C = os.path.join(V11, "clef")
CLEF_ARMS = [("flash BF16 GGUF", "flash-bf16"), ("flash Q8_0", "flash-q8"), ("flash Q4_K_M", "flash-q4"), ("27B Q8_0", "clef-q8")]


def section_clef(L):
    out = {}
    # §3 equivalence vs v10 transformers BF16
    refs = {"flash": jl(os.path.join(A.CLEF, "clef-flash-bf16", "smoke.jsonl")), "clef": jl(os.path.join(A.CLEF, "clef-bf16", "smoke.jsonl"))}
    eq = {}
    for name in ("flash-bf16", "flash-q8", "flash-q4", "clef-bf16", "clef-q8"):
        rows = jl(os.path.join(C, name, "smoke.jsonl"))
        if rows:
            eq[name] = A.equiv(refs["flash" if name.startswith("flash") else "clef"], rows)
    out["equiv"] = eq
    L += ["## C0. 等價檢查（50 題，對 v10 transformers BF16 參考值；等價 = 最大機率差 ≤ 0.01 且答案 ≥ 49/50）", "",
          "| arm | 硬體 | 最大差 | 中位差 | 答案一致 | 等價？ |", "|---|---|---|---|---|---|"]
    for k, v in eq.items():
        hw = "H100" if k == "clef-bf16" else "L40S"
        L.append(f"| {k} | {hw} | {f(v['max_abs'], 4)} | {f(v['p50_abs'], 4)} | {v['argmax_agree']}/{v['n']} | {'✓' if v['equivalent'] else '✗'} |")
    L.append("")
    # D0 / v9 tables
    models = [("26B（我們）", loader_ref("26B")), ("flash（v10 transformers）", loader_ref("Clef-flash"))] + \
             [(lab, loader_bench_dir(os.path.join(C, d))) for lab, d in CLEF_ARMS[:3]] + \
             [("27B（v10 transformers）", loader_ref("Clef 27B")), (CLEF_ARMS[3][0], loader_bench_dir(os.path.join(C, CLEF_ARMS[3][1])))]
    models = [(n, ld) for n, ld in models if any(ld("d0").values())]
    out["d0"] = d0_table(models, L, "## C1. D0（test 半；括號是對 26B 的 McNemar p < 0.05）")
    out["blind"] = blind_row(models)
    L += ["D2 盲寫：" + "、".join(f"{n} {pct(v)}" for n, v in out["blind"].items()), ""]
    # 「可用」: quant vs its BF16 (flash: BF16 GGUF; 27B: v10 transformers BF16, shown equivalent to the BF16 GGUF in C0)
    usable = {}
    for q, base, smoke_key in (("flash Q8_0", "flash BF16 GGUF", "flash-q8"), ("flash Q4_K_M", "flash BF16 GGUF", "flash-q4"), ("27B Q8_0", "27B（v10 transformers）", "clef-q8")):
        if q not in out["d0"]["summary"] or base not in out["d0"]["summary"]:
            continue
        lq = dict(models)[q]("d0"); lb = dict(models)[base]("d0")
        a = {r["id"]: A.correct_of(r) for t in D0 for r in lb.get(t, [])}
        b = {r["id"]: A.correct_of(r) for t in D0 for r in lq.get(t, [])}
        ids = [i for i in a if i in b]
        acc_a, acc_b = float(np.mean([a[i] for i in ids])), float(np.mean([b[i] for i in ids]))
        p = A.mcnemar([a[i] for i in ids], [b[i] for i in ids])
        ag = eq.get(smoke_key, {}).get("argmax_agree", 0)
        usable[q] = {"acc_base_all": acc_a, "acc_quant_all": acc_b, "diff_pt": (acc_b - acc_a) * 100, "p": p, "smoke_agree": ag,
                     "usable": ag >= 49 and abs(acc_b - acc_a) <= 0.01 and p >= 0.05}
    out["usable"] = usable
    L += ["## C2. 量化版「可用」（答案一致 ≥ 49/50，完整 D0 2,000 題與 BF16 差 ≤ 1 點且 McNemar p ≥ 0.05）", "",
          "| 量化版 | 對照 BF16 | BF16 acc | 量化 acc | 差（點） | p | smoke 一致 | 可用？ |", "|---|---|---|---|---|---|---|---|"]
    for q, v in usable.items():
        base = "flash BF16 GGUF" if q.startswith("flash") else "27B BF16（v10，與 GGUF 等價）"
        L.append(f"| {q} | {base} | {pct(v['acc_base_all'])} | {pct(v['acc_quant_all'])} | {v['diff_pt']:+.2f} | {f(v['p'], 3)} | {v['smoke_agree']}/50 | {'✓' if v['usable'] else '✗'} |")
    L.append("")
    out["v9"] = v9_table(models, L, "## C3. v9 三格（test 半）")
    # rerank, jevbench, packed
    data = A.jl(os.path.join(ROOT, "data/v9/r_rerank.jsonl"))
    te_ids = {r["id"] for r, c in zip(data, A.group_split(data, 0)) if not c}
    rr = {"26B（我們）": os.path.join(ROOT, "results/modal/rerank/r_rerank.jsonl"), "flash（v10 transformers）": os.path.join(A.CLEF, "clef-flash-bf16", "r_rerank.jsonl"),
          "flash BF16 GGUF": os.path.join(C, "flash-bf16", "r_rerank.jsonl"), "flash Q8_0": os.path.join(C, "flash-q8", "r_rerank.jsonl"),
          "27B（v10 transformers）": os.path.join(A.CLEF, "clef-bf16", "r_rerank.jsonl"), "27B Q8_0": os.path.join(C, "clef-q8", "r_rerank.jsonl")}
    out["rerank"] = {n: A.rr_metrics([r for r in jl(p) if r["id"] in te_ids]) for n, p in rr.items() if jl(p)}
    jbp = {"flash（v10 transformers）": os.path.join(A.CLEF, "clef-flash-bf16", "jevbench_native.jsonl"), "flash BF16 GGUF": os.path.join(C, "flash-bf16", "jevbench_native.jsonl"),
           "flash Q8_0": os.path.join(C, "flash-q8", "jevbench_native.jsonl"), "27B（v10 transformers）": os.path.join(A.CLEF, "clef-bf16", "jevbench_native.jsonl"),
           "27B Q8_0": os.path.join(C, "clef-q8", "jevbench_native.jsonl")}
    out["jevbench"] = {"26B（我們）": jevbench_ours(os.path.join(ROOT, "results/modal/jevbench")), **{n: jevbench_bench(p) for n, p in jbp.items() if jl(p)}}
    L += ["## C4. 重排序 nDCG@5（test 半，門檻 0.85）與 JevBench 公開 231 題（self-run）", "", "| arm | nDCG@5 | MRR | JevBench 全部 | hard |", "|---|---|---|---|---|"]
    for n in out["jevbench"]:
        r = out["rerank"].get(n, {}); j = out["jevbench"][n] or {}
        L.append(f"| {n} | {f(r.get('ndcg5'), 3)} | {f(r.get('mrr'), 3)} | {pct(j.get('acc'))} | {pct(j.get('hard'))} |")
    L.append("")
    pk = {}
    for n, d in (("flash BF16 GGUF", "flash-bf16"), ("flash Q8_0", "flash-q8"), ("27B Q8_0", "clef-q8")):
        P = {r["id"]: r for r in jl(os.path.join(C, d, "packed.jsonl"))}; S = {r["id"]: r for r in jl(os.path.join(C, d, "d0.jsonl"))}
        if not P:
            continue
        pk[n] = {}
        for t in A.ALARM:
            ids = [i for i, r in P.items() if r["task"] == t and i in S]
            a = [bool(S[i]["correct"]) for i in ids]; b = [bool(P[i]["correct"]) for i in ids]
            pk[n][t] = {"single": float(np.mean(a)), "packed": float(np.mean(b)), "p": A.mcnemar(a, b), "flip": float(np.mean([S[i]["chosen"] != P[i]["chosen"] for i in ids]))}
    out["packed"] = pk
    L += ["## C5. 同 state 三題一起判（alarm 600 列）", "", "| arm | 急迫度 單題→三題 | 分類 | 派工 |", "|---|---|---|---|"]
    for n, v in pk.items():
        L.append(f"| {n} | " + " | ".join(f"{pct(v[t]['single'])} → {pct(v[t]['packed'])}（p={v[t]['p']:.3f}）" for t in A.ALARM) + " |")
    L.append("")
    # latency (same L40S, same b11371 build)
    lat = {}
    for n, d in CLEF_ARMS:
        p = os.path.join(C, d, "_summary.json")
        if os.path.exists(p) and "latency" in json.load(open(p)):
            lat[n] = json.load(open(p))["latency"]["latency"]
    r26 = json.load(open(os.path.join(V11, "latency-26b", "latency_b11371_l40s.json")))["results"]
    def p50(key):
        v = r26.get(key)
        return v["summary"]["p50"] if v else None
    s1, q1, q5, q10 = p50("L1_single_100tok_2opt"), p50("L4_shared_state_1q_cache_on"), p50("L4_shared_state_5q_cache_on"), p50("L4_shared_state_10q_cache_on")
    q3 = q1 + (q5 - q1) * 2 / 4 if q1 and q5 else None  # linear between the measured 1q and 5q points
    thr26 = {c: c / (p50(f"L5_concurrency_{c}") / 1000) for c in (1, 4, 8) if p50(f"L5_concurrency_{c}")}
    out["latency"] = {"clef": lat, "26b": {"single": s1, "shared_1q": q1, "shared_3q_interp": q3, "shared_5q": q5, "shared_10q": q10, "throughput": thr26}}
    L += ["## C6. 延遲（同一張 L40S、同一個 llama.cpp b11371 build；batch 1）", "",
          "| | 單題 p50 | 同 state 三題 p50 | 吞吐 c=1 / 4 / 8（題/秒） |", "|---|---|---|---|",
          f"| 26B（我們，`--swa-full`，同 slot 依序送） | {s1:.0f} ms | ≈ {q3:.0f} ms（1 題 {q1:.0f}、5 題 {q5:.0f} ms 之間內插） | " + " / ".join(f"{thr26[c]:.1f}" for c in (1, 4, 8)) + " |"]
    for n, v in lat.items():
        L.append(f"| {n} | {v['single']['p50_ms']:.0f} ms | {v['packed3']['p50_ms']:.0f} ms | {v['throughput_c1']} / {v['throughput_c4']} / {v['throughput_c8']} |")
    L.append("")
    return out


def main():
    L, out = [], {}
    out["jevify"] = section_jevify(L)
    out["clef"] = section_clef(L)
    text = "\n".join(L) + "\n"
    open(os.path.join(ROOT, "results/16-clef-llamacpp-tables.md"), "w", encoding="utf-8").write(text)
    json.dump(out, open(os.path.join(ROOT, "results/v11.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(text)


if __name__ == "__main__":
    main()
