"""v21 analysis: detectors D0-D5 against the window labels, folds, calibration, gates G0-G5 and the kill criterion.

  python3 bench/analyze_v21.py [--res results/modal/v21] [--windows data/v21/windows.jsonl]
Writes results/25-asr-error-gate.md and results/v21.json.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "bench"))
from analyze import fit_temperature, softmax  # noqa: E402
from thresholds import risk_threshold  # noqa: E402
import v21_labels  # noqa: E402

SEGS, CONDS = ["T1", "T2", "T3"], ["M0", "M1", "M2", "M3"]
EPS = 0.10


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if y.all() or (~y).all() or len(y) == 0:
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s), float)
    # average ranks for ties
    i = 0; ss = s[order]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def prec_at_recall(scores, labels, target=0.8):
    s = np.asarray(scores, float); y = np.asarray(labels, bool); order = np.argsort(-s)
    tp = np.cumsum(y[order]); k = np.searchsorted(tp, math.ceil(target * y.sum()))
    if k >= len(s):
        return float("nan"), float("nan")
    return float(tp[k] / (k + 1)), float((k + 1) / len(s))


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def pB(rec, qk, letters=("B",)):
    r = rec.get(qk) or {}; p = r.get("probs") or {}
    return sum(p.get(L, 0.0) for L in letters) if p else float("nan")


def logits_AB(rec, qk):
    r = rec.get(qk) or {}; lp = r.get("raw_logprobs") or {}
    return [lp.get("A", -30.0), lp.get("B", -30.0)]


def folds_eval(recs, by_id, qk, label):
    """Leave-one-segment-out: temperature + risk threshold on two segments, judged on the third. Pooled test predictions."""
    rows = [(r, by_id[r["id"]]) for r in recs if r["id"] in by_id and (r.get(qk) or {}).get("probs")]
    out = {"n": len(rows)}; pooled_conf, pooled_ok, pooled_p, pooled_y, handled_flags = [], [], [], [], []
    per_fold = {}
    for test_seg in SEGS:
        cal = [(r, w) for r, w in rows if w["seg"] != test_seg]; te = [(r, w) for r, w in rows if w["seg"] == test_seg]
        if not cal or not te:
            continue
        Zc = np.array([logits_AB(r, qk) for r, _ in cal]); yc = np.array([int(w[label]) for _, w in cal])
        T = fit_temperature(Zc, yc); Pc = softmax(Zc / T)
        conf_c = Pc.max(1); ok_c = Pc.argmax(1) == yc
        t, frac = risk_threshold(conf_c, ok_c, EPS)
        Zt = np.array([logits_AB(r, qk) for r, _ in te]); yt = np.array([int(w[label]) for _, w in te]); Pt = softmax(Zt / T)
        conf_t = Pt.max(1); ok_t = Pt.argmax(1) == yt; handled = conf_t >= t
        per_fold[test_seg] = {"T": round(T, 3), "threshold": None if math.isinf(t) else round(t, 4), "cal_handled": round(frac, 3),
                              "test_handled": round(float(handled.mean()), 3), "test_err_handled": round(float((~ok_t[handled]).mean()), 3) if handled.any() else None,
                              "test_acc": round(float(ok_t.mean()), 3), "test_auroc": round(auroc(Pt[:, 1], yt.astype(bool)), 3)}
        pooled_conf += conf_t.tolist(); pooled_ok += ok_t.tolist(); pooled_p += Pt[:, 1].tolist(); pooled_y += yt.tolist(); handled_flags += handled.tolist()
    y = np.array(pooled_y, bool); p = np.array(pooled_p); h = np.array(handled_flags, bool); ok = np.array(pooled_ok, bool)
    out["folds"] = per_fold
    out["auroc_cal"] = round(auroc(p, y), 3) if len(y) else None
    out["acc"] = round(float(ok.mean()), 3) if len(y) else None
    out["handled"] = round(float(h.mean()), 3) if len(y) else None
    out["err_handled"] = round(float((~ok[h]).mean()), 3) if h.any() else None
    bins = []
    for lo in np.arange(0, 1, 0.2):
        m = (p >= lo) & (p < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))
        bins.append({"bin": f"{lo:.1f}-{lo + 0.2:.1f}", "n": int(m.sum()), "mean_p": round(float(p[m].mean()), 3) if m.any() else None, "frac_pos": round(float(y[m].mean()), 3) if m.any() else None})
    out["reliability"] = bins
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default=os.path.join(ROOT, "results/modal/v21")); ap.add_argument("--windows", default=os.path.join(ROOT, "data/v21/windows.jsonl"))
    a = ap.parse_args()
    wins = load_jsonl(a.windows); by_id = {w["id"]: w for w in wins}
    meta = json.load(open(a.windows.replace(".jsonl", "_meta.json")))
    R = {"windows_meta": meta, "detectors": {}, "gates": {}, "speed": {}}
    y = {lab: np.array([w[lab] for w in wins], bool) for lab in ("E", "C", "C_strong")}
    cond = np.array([w["cond"] for w in wins])

    # ---- G0: Breeze on L4 vs kikemu Xbrz_gpu (both the generate transcript and the pipeline transcript)
    brz = v21_labels.load_run(os.path.join(a.res, "breeze")); base = v21_labels.load_run(os.path.join(v21_labels.KIK, "raw/Xbrz_gpu"))
    terms = json.load(open(os.path.join(v21_labels.KIK, "terms.json")))
    g0 = {"files": {}, "max_d_recall": 0, "max_d_ter": 0, "max_d_recall_pipeline": 0, "max_d_ter_pipeline": 0}
    for k, d in sorted(brz.items()):
        s = v21_labels.score_file(k[0], d["transcript"], terms); sp = v21_labels.score_file(k[0], d.get("pipeline_transcript", ""), terms); b = v21_labels.score_file(k[0], base[k]["transcript"], terms)
        g0["files"]["__".join(k)] = {"l4": s, "l4_pipeline": sp, "t4": b, "elapsed_sec": d["meta"].get("elapsed_sec"), "pipeline_sec": d["meta"].get("pipeline_sec"), "argmax_agree": d["meta"].get("argmax_agree")}
        g0["max_d_recall"] = max(g0["max_d_recall"], abs(s["recall_tolerant"] - b["recall_tolerant"])); g0["max_d_ter"] = max(g0["max_d_ter"], abs(s["ter"] - b["ter"]))
        g0["max_d_recall_pipeline"] = max(g0["max_d_recall_pipeline"], abs(sp["recall_tolerant"] - b["recall_tolerant"])); g0["max_d_ter_pipeline"] = max(g0["max_d_ter_pipeline"], abs(sp["ter"] - b["ter"]))
    g0["pass"] = g0["max_d_recall"] <= 0.02 and g0["max_d_ter"] <= 0.01
    g0["breeze_total_sec"] = sum(v["elapsed_sec"] or 0 for v in g0["files"].values())
    R["gates"]["G0"] = g0

    # ---- D0, D1, D5
    D = R["detectors"]
    D["D0"] = {"has_latin_frac": round(float(np.mean([w["has_latin"] for w in wins])), 3), "per_cond": {c: round(float(np.mean([w["has_latin"] for w in wins if w["cond"] == c])), 3) for c in CONDS}}
    if all("d1_mean" in w for w in wins):
        D["D1"] = {}
        for feat, sign in (("d1_mean", -1), ("d1_min", -1), ("d1_low", 1)):
            s = np.array([sign * w[feat] for w in wins])
            D["D1"][feat] = {lab: round(auroc(s, y[lab]), 3) for lab in ("E", "C", "C_strong")}
            D["D1"][feat]["per_cond_E"] = {c: round(auroc(s[cond == c], y["E"][cond == c]), 3) for c in CONDS}
            D["D1"][feat]["per_cond_C"] = {c: round(auroc(s[cond == c], y["C"][cond == c]), 3) for c in CONDS}
    m5 = np.array(["d5_disagree" in w for w in wins])
    if m5.any():
        s = np.array([w.get("d5_disagree", 0) for w in wins])
        D["D5"] = {"n": int(m5.sum()), **{lab: round(auroc(s[m5], y[lab][m5]), 3) for lab in ("E", "C", "C_strong")}}

    # ---- D2 / D3 / D4 (+ swap)
    arms = {"D2": "e4b_text", "D3": "e4b_audio", "D4": "26b_text"}
    for key, tag in arms.items():
        recs = load_jsonl(os.path.join(a.res, tag, "judge.jsonl"))
        if not recs:
            continue
        rid = {r["id"] for r in recs}; idx = np.array([w["id"] in rid for w in wins]); rmap = {r["id"]: r for r in recs}
        d = {"n": len(recs), "option_mass_mean": round(float(np.mean([v["option_mass"] for r in recs for v in r.values() if isinstance(v, dict) and "option_mass" in v])), 4),
             "errors": sum(1 for r in recs for v in r.values() if isinstance(v, dict) and "error" in v)}
        ws = [w for w in wins if w["id"] in rid]
        def S(qk, letters):  # noqa: E306
            return np.array([pB(rmap[w["id"]], qk, letters) for w in ws])
        yy = {lab: np.array([w[lab] for w in ws], bool) for lab in ("E", "C", "C_strong")}; cc = np.array([w["cond"] for w in ws])
        for qk, lab, letters in (("q1", "E", ("B",)), ("q2", "C", ("B",)), ("q2", "C_strong", ("B",)), ("q3", "E", ("B", "D")), ("q3", "C", ("C", "D"))):
            s = S(qk, letters); ok = ~np.isnan(s)
            d[f"auroc_{qk}_{lab}"] = round(auroc(s[ok], yy[lab][ok]), 3)
            d[f"auroc_{qk}_{lab}_per_cond"] = {c: round(auroc(s[ok & (cc == c)], yy[lab][ok & (cc == c)]), 3) for c in CONDS}
        s = S("q1", ("B",)); ok = ~np.isnan(s); p80, frac80 = prec_at_recall(s[ok], yy["E"][ok]); d["q1_prec_at_recall80"] = round(p80, 3); d["q1_flagged_frac_at_recall80"] = round(frac80, 3)
        q3 = [(rmap[w["id"]].get("q3") or {}).get("probs") or {} for w in ws]
        pred = [max(p, key=p.get) if p else None for p in q3]
        d["q3_acc"] = round(float(np.mean([p == w["q3"] for p, w in zip(pred, ws) if p])), 3)
        d["q3_confusion"] = {g: {p: int(sum(1 for pp, w in zip(pred, ws) if w["q3"] == g and pp == p)) for p in "ABCD"} for g in "ABCD"}
        d["calib_q1_E"] = folds_eval(recs, by_id, "q1", "E"); d["calib_q2_C"] = folds_eval(recs, by_id, "q2", "C")
        lat = sorted(v["latency_ms"] for r in recs for v in r.values() if isinstance(v, dict) and "latency_ms" in v)
        d["latency_p50_ms"] = round(lat[len(lat) // 2], 1); d["latency_p95_ms"] = round(lat[int(len(lat) * 0.95)], 1)
        d["per_window_3q_ms"] = round(3 * lat[len(lat) // 2], 1)
        D[key] = d
        swap = load_jsonl(os.path.join(a.res, tag.replace("_text", "_swap"), "judge.jsonl")) if tag.endswith("_text") else []
        if swap:
            smap = {r["id"]: r for r in swap}; both = [w for w in ws if w["id"] in smap]
            sw = {"n": len(both)}
            for qk in ("q1", "q2"):
                po = np.array([pB(rmap[w["id"]], qk) for w in both]); ps = np.array([(smap[w["id"]].get(f"{qk}_r1") or {}).get("probs_orig", {}).get("B", np.nan) for w in both])
                ok = ~np.isnan(po) & ~np.isnan(ps)
                sw[qk] = {"flip_rate": round(float(np.mean((po[ok] > 0.5) != (ps[ok] > 0.5))), 3), "mean_abs_dp": round(float(np.mean(np.abs(po[ok] - ps[ok]))), 3),
                          "auroc_orig": round(auroc(po[ok], np.array([w["E" if qk == "q1" else "C"] for w in both], bool)[ok]), 3),
                          "auroc_swapped": round(auroc(ps[ok], np.array([w["E" if qk == "q1" else "C"] for w in both], bool)[ok]), 3),
                          "auroc_mean2": round(auroc((po[ok] + ps[ok]) / 2, np.array([w["E" if qk == "q1" else "C"] for w in both], bool)[ok]), 3),
                          "pB_mean_orig": round(float(po[ok].mean()), 3), "pB_mean_swapped": round(float(ps[ok].mean()), 3)}
            # q3: argmax over 4 rotations mapped back; log-mean of the four
            flips = 0; n = 0; acc_rot = 0
            for w in both:
                p0 = (rmap[w["id"]].get("q3") or {}).get("probs") or {}
                rots = [p0] + [(smap[w["id"]].get(f"q3_r{r}") or {}).get("probs_orig", {}) for r in (1, 2, 3)]
                rots = [p for p in rots if p]
                if len(rots) < 2:
                    continue
                am = [max(p, key=p.get) for p in rots]; n += 1; flips += len(set(am)) > 1
                lm = {L: sum(math.log(max(p.get(L, 1e-9), 1e-9)) for p in rots) / len(rots) for L in "ABCD"}
                acc_rot += max(lm, key=lm.get) == w["q3"]
            sw["q3"] = {"n": n, "any_flip_rate": round(flips / max(n, 1), 3), "acc_rot_logmean": round(acc_rot / max(n, 1), 3)}
            D[key]["swap"] = sw
    smoke_p = os.path.join(a.res, "audio_smoke.json")
    if os.path.exists(smoke_p):
        R["audio_smoke"] = json.load(open(smoke_p))

    # ---- gates
    G = R["gates"]; d2 = D.get("D2", {}); d1 = D.get("D1", {}); d3 = D.get("D3", {}); d4 = D.get("D4", {})
    G["G1"] = {"auroc": d2.get("auroc_q1_E"), "prec_at_recall80": d2.get("q1_prec_at_recall80"), "pass": bool(d2) and d2["auroc_q1_E"] >= 0.85 and d2["q1_prec_at_recall80"] >= 0.60}
    nf = meta["noise_floor"]["C_frac"]
    G["G2"] = {"auroc": d2.get("auroc_q2_C"), "noise_floor_C": nf, "judged": nf <= 0.20, "pass": (bool(d2) and d2["auroc_q2_C"] >= 0.75) if nf <= 0.20 else None,
               "auroc_C_strong": d2.get("auroc_q2_C_strong")}
    d1E = d1.get("d1_mean", {}).get("E"); d1C = d1.get("d1_mean", {}).get("C")
    G["G3"] = {"d1_auroc_E": d1E, "d1_auroc_C": d1C, "d1_first_gate": bool(d1) and (d1E or 0) >= 0.75,
               "d2_minus_d1_E": round(d2["auroc_q1_E"] - d1E, 3) if d2 and d1E is not None else None,
               "d2_minus_d1_C": round(d2["auroc_q2_C"] - d1C, 3) if d2 and d1C is not None else None,
               "e4b_worth_adding": bool(d2 and d1E is not None) and (d2["auroc_q1_E"] - d1E) >= 0.05}
    G["G4"] = {"d3_minus_d2_E": round(d3["auroc_q1_E"] - d2["auroc_q1_E"], 3) if d3 and d2 else None, "d3_minus_d2_C": round(d3["auroc_q2_C"] - d2["auroc_q2_C"], 3) if d3 and d2 else None,
               "pass": bool(d3 and d2) and (d3["auroc_q1_E"] - d2["auroc_q1_E"]) >= 0.05 and (d3["auroc_q2_C"] - d2["auroc_q2_C"]) >= 0.05,
               "smoke_all_pass": R.get("audio_smoke", {}).get("all_pass")}
    gate_s = (d2.get("per_window_3q_ms", 0) * len(wins) / 1000) if d2 else None
    G["G5"] = {"per_window_3q_ms": d2.get("per_window_3q_ms"), "gate_total_s": round(gate_s, 1) if gate_s else None, "breeze_total_s": round(g0["breeze_total_sec"], 1),
               "ratio": round(gate_s / g0["breeze_total_sec"], 3) if gate_s and g0["breeze_total_sec"] else None, "note": "L4 fallback numbers; the GB10 numbers decide",
               "pass": bool(d2) and d2["per_window_3q_ms"] <= 300 and gate_s / g0["breeze_total_sec"] <= 0.10}
    G["kill"] = {"d2_E": d2.get("auroc_q1_E"), "d4_E": d4.get("auroc_q1_E"), "triggered": bool(d2 and d4) and d2["auroc_q1_E"] < 0.70 and d4["auroc_q1_E"] < 0.70}
    json.dump(R, open(os.path.join(ROOT, "results/v21.json"), "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_report(R, wins)


def f3(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))


def write_report(R, wins):
    D, G, M = R["detectors"], R["gates"], R["windows_meta"]; c = M["counts"]
    L = ["# 25 — v21：Breeze 聽、E4B 抓錯（Modal L4 備援，2026-10-15）", "",
         "規則、預測與門檻：`docs/handoff-v21-asr-error-gate.md`（跑前寫死）。數字：`results/v21.json`、`results/modal/v21/`、`data/v21/windows.jsonl`。", "",
         "## 1. 語料與標籤", "",
         f"- 視窗 {c['windows']} 個（12 檔；中位數 {M['window_seconds_p50']} 秒、{M['window_tokens_p50']:.0f} token）：E 英文錯 {c['E']}、C 中文錯字 {c['C']}（C-strong {c['C_strong']}）、N 乾淨 {c['N']}；含拉丁字母 {c['has_latin']}。",
         f"- 雜訊地板（第二路 Gemini 當假設）：被標成 C 的視窗 **{M['noise_floor']['C_frac']:.1%}**、E {M['noise_floor']['E_frac']:.1%}。" + ("C 超過 20%，**G2 只報不判**。" if M["noise_floor"]["C_frac"] > 0.2 else ""),
         "- 每條件：" + "、".join(f"{k} n={v['n']} E={v['E']} C={v['C']}" for k, v in M["per_cond"].items()), ""]
    g0 = G["G0"]
    L += ["## 2. G0 同一把尺：Breeze L4（generate）對 kikemu Xbrz_gpu（T4 pipeline）", "",
          "| 檔 | 召回 L4 / T4 | TER L4 / T4 | 召回 L4-pipeline | 秒（generate / pipeline） | argmax 一致 |", "|---|---|---|---|---|---|"]
    for k, v in g0["files"].items():
        L.append(f"| {k} | {v['l4']['recall_tolerant']:.3f} / {v['t4']['recall_tolerant']:.3f} | {v['l4']['ter']:.3f} / {v['t4']['ter']:.3f} | {v['l4_pipeline']['recall_tolerant']:.3f} | {v['elapsed_sec']} / {v['pipeline_sec']} | {v['argmax_agree']} |")
    L += ["", f"最大差：召回 {g0['max_d_recall']:.3f}（≤ 0.02 → {'✓' if g0['max_d_recall'] <= 0.02 else '✗'}）、TER {g0['max_d_ter']:.3f}（≤ 0.01 → {'✓' if g0['max_d_ter'] <= 0.01 else '✗'}）；pipeline 版的最大差：召回 {g0['max_d_recall_pipeline']:.3f}、TER {g0['max_d_ter_pipeline']:.3f}。**G0 {'✓' if g0['pass'] else '✗'}**", ""]
    L += ["## 3. 偵測器（AUROC，越高越能把有錯的視窗排到前面；0.5 = 瞎猜）", "",
          "| 偵測器 | E 英文錯 | C 中文錯字 | C-strong | 備註 |", "|---|---|---|---|---|"]
    if "D1" in D:
        for feat, v in D["D1"].items():
            L.append(f"| D1 Breeze logprob `{feat}` | {f3(v['E'])} | {f3(v['C'])} | {f3(v['C_strong'])} | 免費；每條件 E：{', '.join(f'{k} {f3(x)}' for k, x in v['per_cond_E'].items())} |")
    if "D5" in D:
        L.append(f"| D5 Breeze 對 E4B-ASR 分歧 | {f3(D['D5']['E'])} | {f3(D['D5']['C'])} | {f3(D['D5']['C_strong'])} | 只有 M0／M3，n={D['D5']['n']} |")
    for key, name in (("D2", "D2 E4B 純文字"), ("D3", "D3 E4B 聽音訊 + 文字"), ("D4", "D4 26B 純文字")):
        if key in D:
            d = D[key]
            L.append(f"| {name} Q1／Q2 | {f3(d['auroc_q1_E'])} | {f3(d['auroc_q2_C'])} | {f3(d['auroc_q2_C_strong'])} | option mass {d['option_mass_mean']}；Q3 四選一 {f3(d['q3_acc'])}（E 由 Q3 {f3(d['auroc_q3_E'])}、C {f3(d['auroc_q3_C'])}）；抓 80% E 時精確率 {f3(d['q1_prec_at_recall80'])}（標 {f3(d['q1_flagged_frac_at_recall80'])} 的視窗）；每題 p50 {d['latency_p50_ms']} ms |")
    L.append("")
    for key in ("D2", "D3", "D4"):
        if key in D:
            d = D[key]
            L += [f"### {key} 每條件 AUROC", "", "| | " + " | ".join(CONDS) + " |", "|---|" + "---|" * 4,
                  "| Q1 → E | " + " | ".join(f3(d['auroc_q1_E_per_cond'][c]) for c in CONDS) + " |",
                  "| Q2 → C | " + " | ".join(f3(d['auroc_q2_C_per_cond'][c]) for c in CONDS) + " |", ""]
            for qk, lab in (("calib_q1_E", "Q1／E"), ("calib_q2_C", "Q2／C")):
                cal = d[qk]
                L += [f"### {key} {lab} 校準（留一段三折，溫度 + 錯誤預算 ε = {EPS}）", "",
                      f"合併 test：AUROC {f3(cal['auroc_cal'])}、準確率 {f3(cal['acc'])}、自動處理 {f3(cal['handled'])}、處理部分錯誤率 {f3(cal['err_handled'])}。",
                      "折：" + "；".join(f"test {s} T={v['T']} 門檻 {v['threshold']} 處理 {v['test_handled']} 錯 {v['test_err_handled']} acc {v['test_acc']}" for s, v in cal["folds"].items()), "",
                      "| P(有錯) 桶 | n | 平均 P | 真的有錯的比例 |", "|---|---|---|---|"] + [f"| {b['bin']} | {b['n']} | {f3(b['mean_p'])} | {f3(b['frac_pos'])} |" for b in cal["reliability"]] + [""]
            if "swap" in d:
                sw = d["swap"]
                L += [f"### {key} 選項對調（D2′，n={sw['n']}）", "",
                      "| 題 | 翻轉率 | 平均 \\|Δp\\| | AUROC 原序 / 對調 / 兩者平均 | P(B) 平均 原序 / 對調 |", "|---|---|---|---|---|"]
                for qk in ("q1", "q2"):
                    v = sw[qk]; L.append(f"| {qk} | {v['flip_rate']} | {v['mean_abs_dp']} | {v['auroc_orig']} / {v['auroc_swapped']} / {v['auroc_mean2']} | {v['pB_mean_orig']} / {v['pB_mean_swapped']} |")
                L += [f"| q3 四個旋轉 | 任一翻轉 {sw['q3']['any_flip_rate']} | | 旋轉對數平均四選一準確率 {sw['q3']['acc_rot_logmean']} | |", ""]
    if "audio_smoke" in R:
        s = R["audio_smoke"]
        L += ["## 4. 音訊 smoke（Phase 3）", "",
              f"- check 1 多模態 `/completion` + `n_probs` 讀得到字母機率：{'✓' if s.get('check1_probs_readable') else '✗'}",
              f"- check 2 轉錄（jfk.wav 基準線 {'✓' if s.get('jfk', {}).get('pass') else '✗'}；晶晶體視窗保留拉丁字母 {'✓' if s.get('err_window', {}).get('has_latin') else '✗'}、無簡體 {'✓' if not s.get('err_window', {}).get('has_simplified_marker', True) else '✗'}）：{'✓' if s.get('check2_transcription') else '✗'}",
              f"- check 3 Q1 方向（有錯的視窗 P(B) {f3(s.get('err_window', {}).get('q1_audio', {}).get('probs', {}).get('B'))}、乾淨視窗 P(A) {f3(s.get('clean_window', {}).get('q1_audio', {}).get('probs', {}).get('A'))}）：{'✓' if s.get('check3_q1_direction') else '✗'}",
              f"- 晶晶體視窗 E4B 自己的轉錄：「{(s.get('err_window', {}).get('transcribe', {}).get('text') or '')[:120]}」；Breeze：「{s.get('err_window', {}).get('breeze_text', '')[:120]}」", ""]
    L += ["## 5. 門檻", "", "| 門檻 | 數字 | 判定 |", "|---|---|---|",
          f"| G0 同一把尺 | 召回最大差 {g0['max_d_recall']:.3f}、TER {g0['max_d_ter']:.3f} | {'✓' if g0['pass'] else '✗'} |",
          f"| G1 英文錯 | D2 AUROC {f3(G['G1']['auroc'])}、抓 80% 時精確率 {f3(G['G1']['prec_at_recall80'])} | {'✓' if G['G1']['pass'] else '✗'} |",
          f"| G2 中文錯字 | D2 AUROC {f3(G['G2']['auroc'])}（C-strong {f3(G['G2']['auroc_C_strong'])}）；雜訊地板 {G['G2']['noise_floor_C']:.1%} | {'只報不判' if not G['G2']['judged'] else ('✓' if G['G2']['pass'] else '✗')} |",
          f"| G3 免費訊號 | D1 E {f3(G['G3']['d1_auroc_E'])}、C {f3(G['G3']['d1_auroc_C'])}；D2 − D1：E {f3(G['G3']['d2_minus_d1_E'])}、C {f3(G['G3']['d2_minus_d1_C'])} | D1 當第一道 {'✓' if G['G3']['d1_first_gate'] else '✗'}；加 E4B 值得 {'✓' if G['G3']['e4b_worth_adding'] else '✗'} |",
          f"| G4 音訊 | D3 − D2：E {f3(G['G4']['d3_minus_d2_E'])}、C {f3(G['G4']['d3_minus_d2_C'])}；smoke {G['G4']['smoke_all_pass']} | {'✓' if G['G4']['pass'] else ('✗' if G['G4']['d3_minus_d2_E'] is not None else '未跑')} |",
          f"| G5 速度（L4） | 三題每視窗 {f3(G['G5']['per_window_3q_ms'])} ms；閘門 {f3(G['G5']['gate_total_s'])} s 對 Breeze {f3(G['G5']['breeze_total_s'])} s = {f3(G['G5']['ratio'])} | {'✓' if G['G5']['pass'] else '✗'}（GB10 再量） |",
          f"| Kill | D2 E {f3(G['kill']['d2_E'])}、D4 E {f3(G['kill']['d4_E'])} 都 < 0.70？ | {'觸發' if G['kill']['triggered'] else '未觸發'} |", ""]
    open(os.path.join(ROOT, "results/25-asr-error-gate.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-10:]))


if __name__ == "__main__":
    main()
