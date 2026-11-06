"""v19 analysis (docs/handoff-v19-l0-numeric-coherence.md; gates written before the run).

  python3 bench/analyze_v19.py [--dir results/modal/v19]  ->  results/23-l0-numeric-coherence.md, results/v19.json
"""
import json
import math
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from analyze import MISSING_LOGP, ece, fit_temperature, softmax  # noqa: E402
from thresholds import risk_threshold  # noqa: E402

TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "q_defect_root", "p_uph_anomaly", "p_line_change",
         "x_ticket_route", "x_escalate", "x_10way_intent"]
WEAK = ["m_alarm_severity", "q_spc_action", "p_uph_anomaly"]
STRONG = [t for t in TASKS if t not in WEAK]
E3 = ["m_alarm_severity", "m_alarm_category", "q_spc_action", "q_defect_root", "x_ticket_route"]
READOUTS = ["raw", "rot", "rot_pmean", "prior", "L0"]
EPS = 0.05


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def mcnemar(a, b):
    """a, b: boolean arrays of correctness on the same rows. Returns (only_a, only_b, exact p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum()); n = x + y
    p = min(1.0, 2 * sum(comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def lp(d, letters):
    return np.array([MISSING_LOGP if d.get(L) is None else d[L] for L in letters])


# ------------------------------------------------------------------ E1
def e1_task(rows, letters):
    """Returns per-readout logit matrices (n, K) for all rows, plus the prior and flip stats."""
    K = len(letters); rows = [r for r in rows if "rotations" in r and len(r["rotations"]) == K]
    cal = [i for i, r in enumerate(rows) if r["split"] == "cal"]
    # position prior from the cal split, all rotations (each option sits at each position once -> content averages out)
    prior = np.zeros(K)
    for i in cal:
        for rot in rows[i]["rotations"]:
            prior += np.array([rot["pos_probs"].get(L, 0.0) for L in letters])
    prior /= max(1, len(cal) * K); prior = np.clip(prior, 1e-6, None); logprior = np.log(prior)
    Z = {k: np.zeros((len(rows), K)) for k in READOUTS}
    flips = {"raw_r0_vs_r1": [], "raw_any": [], "prior_r0_vs_r1": [], "prior_any": []}
    for i, r in enumerate(rows):
        A = np.array([lp(rot["aligned_logp"], letters) for rot in r["rotations"]])   # (K rotations, K options), aligned to options
        # prior removal: option j in rotation rr sits at position (j + rr) % K -> subtract that position's log prior
        P = np.array([[A[rr, j] - logprior[(j + rr) % K] for j in range(K)] for rr in range(K)])
        Z["raw"][i] = A[0]; Z["rot"][i] = A.mean(0); Z["prior"][i] = P[0]; Z["L0"][i] = P.mean(0)
        pm = np.array([list(rot["aligned_probs"].get(L, 0.0) for L in letters) for rot in r["rotations"]]).mean(0)
        Z["rot_pmean"][i] = np.log(np.clip(pm, 1e-12, None))
        am = A.argmax(1); pmx = P.argmax(1)
        flips["raw_r0_vs_r1"].append(am[0] != am[1 % K]); flips["raw_any"].append(len(set(am.tolist())) > 1)
        flips["prior_r0_vs_r1"].append(pmx[0] != pmx[1 % K]); flips["prior_any"].append(len(set(pmx.tolist())) > 1)
    return rows, Z, prior, flips


def e1(dirpath):
    out = {}
    for t in TASKS:
        rows = load(os.path.join(dirpath, "l0", f"{t}.jsonl"))
        if not rows:
            continue
        letters = list("ABCDEFGHIJ"[:len(rows[0]["rotations"])])
        rows, Z, prior, flips = e1_task(rows, letters)
        y = np.array([letters.index(r["gold"]) for r in rows]); te = np.array([r["split"] == "test" for r in rows]); ca = ~te
        res = {"n_test": int(te.sum()), "K": len(letters), "prior": prior.round(4).tolist(), "readouts": {}}
        for k in READOUTS:
            T = fit_temperature(Z[k][ca], y[ca]); P = softmax(Z[k] / T); conf = P.max(1); correct = P.argmax(1) == y
            thr, _ = risk_threshold(conf[ca], correct[ca], EPS); h = (conf >= thr) & te
            res["readouts"][k] = {"acc": float(correct[te].mean()), "T": T, "ece": ece(conf[te], correct[te]),
                                  "cov5": float(h.sum() / te.sum()), "err5": float((~correct[h]).mean()) if h.any() else 0.0,
                                  "correct_test": correct[te].tolist()}
        for k in ("rot", "rot_pmean", "prior", "L0"):
            res["readouts"][k]["vs_raw"] = mcnemar(res["readouts"]["raw"]["correct_test"], res["readouts"][k]["correct_test"])
        res["flip_test"] = {k: float(np.mean(np.array(v)[te])) for k, v in flips.items()}
        out[t] = res
    return out


# ------------------------------------------------------------------ E2
def e2(dirpath, e1res):
    out = {}
    for v in ("N1", "N2"):
        per = {}
        for t in TASKS:
            if t == "x_10way_intent":
                continue
            rows = [r for r in load(os.path.join(dirpath, "numeric", v, f"{t}.jsonl")) if "correct" in r]
            base = {r["id"]: r for r in load(os.path.join(dirpath, "l0", f"{t}.jsonl")) if r.get("split") == "test" and "rotations" in r}
            ids = [r["id"] for r in rows if r["id"] in base]
            if not ids:
                continue
            num = {r["id"]: r for r in rows}
            a = [base[i]["correct_raw"] for i in ids]; b = [num[i]["correct"] for i in ids]
            per[t] = {"n": len(ids), "acc_letters": float(np.mean(a)), "acc_numeric": float(np.mean(b)),
                      "mass_median": float(np.median([num[i]["option_mass"] for i in ids])), "vs_letters": mcnemar(a, b),
                      "lat_p50": float(np.median([num[i]["latency_ms"] for i in ids]))}
        out[v] = per
    return out


# ------------------------------------------------------------------ E3
def e3(dirpath, lock):
    out = {}
    gaps_all, agree_all, order_all = [], [], []
    for t in E3:
        rows = [r for r in load(os.path.join(dirpath, "coherence", f"{t}.jsonl")) if "labels" in r]
        if not rows:
            continue
        T = lock["tasks"][t]["T"]
        g, g_T, agree, order, sums, yes_mean, not_mean = [], [], [], [], [], [], []
        for r in rows:
            for L in (r["X"], r["Y"]):
                d = r["labels"][L]
                if d["p_yes"] is None or d["p_not"] is None:
                    continue
                g.append(abs(d["p_yes"] + d["p_not"] - 1)); yes_mean.append(d["p_yes"]); not_mean.append(d["p_not"])
                agree.append(abs(d["p_yes"] - r["p_k"].get(L, 0.0))); order.append(abs(d["p_yes"] - d["p_yes_swapped"]))
                py = softmax(lp(d["lp_yes"], ["A", "B"])[None, :] / T)[0, 0]; pn = softmax(lp(d["lp_not"], ["A", "B"])[None, :] / T)[0, 0]
                g_T.append(abs(py + pn - 1))
            if t == "m_alarm_severity":
                s = sum(r["labels"][L]["p_yes"] for L in r["labels"] if r["labels"][L]["p_yes"] is not None)
                if len(r["labels"]) == r["K"]:
                    sums.append(s)
        ci = 1.96 * np.std(g) / math.sqrt(len(g)) if g else 0.0
        out[t] = {"n_pairs": len(g), "gap_mean": float(np.mean(g)), "gap_ci95": float(ci), "gap_median": float(np.median(g)),
                  "gap_after_T": float(np.mean(g_T)), "agree_k_vs_yes": float(np.mean(agree)), "order_effect": float(np.mean(order)),
                  "p_yes_mean": float(np.mean(yes_mean)), "p_not_mean": float(np.mean(not_mean)),
                  "single_label_sum": float(np.mean(sums)) if sums else None, "n_sum": len(sums)}
        gaps_all += g; agree_all += agree; order_all += order
    out["all"] = {"n_pairs": len(gaps_all), "gap_mean": float(np.mean(gaps_all)) if gaps_all else None,
                  "gap_ci95": float(1.96 * np.std(gaps_all) / math.sqrt(len(gaps_all))) if gaps_all else None,
                  "agree_k_vs_yes": float(np.mean(agree_all)) if agree_all else None, "order_effect": float(np.mean(order_all)) if order_all else None}
    return out


def mean_over(res, tasks, k, field):
    v = [res[t]["readouts"][k][field] for t in tasks if t in res]
    return float(np.mean(v)) if v else float("nan")


def main():
    a = dict(zip(*[iter(sys.argv[1:])] * 2)); d = os.path.join(ROOT, a.get("--dir", "results/modal/v19"))
    lock = json.load(open(os.path.join(ROOT, "results/thresholds.lock.json")))
    r1 = e1(d); r2 = e2(d, r1); r3 = e3(d, lock)
    f3 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"  # noqa: E731
    # gates
    w_raw, w_L0, w_prior = (mean_over(r1, WEAK, k, "acc") for k in ("raw", "L0", "prior"))
    s_raw, s_L0, s_prior = (mean_over(r1, STRONG, k, "acc") for k in ("raw", "L0", "prior"))
    c_raw, c_L0 = mean_over(r1, WEAK, "raw", "cov5"), mean_over(r1, WEAK, "L0", "cov5")
    g1 = bool(r1) and w_L0 >= w_raw + 0.02 - 1e-9 and c_L0 >= c_raw + 0.05 - 1e-9 and s_L0 >= s_raw - 0.005 - 1e-9
    g1b = bool(r1) and w_prior >= w_raw + 0.01 - 1e-9 and s_prior >= s_raw - 0.005 - 1e-9
    g2 = {}
    nine = [t for t in TASKS if t != "x_10way_intent"]
    for v, per in r2.items():
        if not per:
            continue
        m_l = float(np.mean([per[t]["acc_letters"] for t in nine if t in per])); m_n = float(np.mean([per[t]["acc_numeric"] for t in nine if t in per]))
        w_l = float(np.mean([per[t]["acc_letters"] for t in WEAK if t in per])); w_n = float(np.mean([per[t]["acc_numeric"] for t in WEAK if t in per]))
        mass = float(np.median([per[t]["mass_median"] for t in per]))
        g2[v] = {"mean_letters": m_l, "mean_numeric": m_n, "weak_letters": w_l, "weak_numeric": w_n, "mass": mass,
                 "pass": m_n >= m_l + 0.01 - 1e-9 and w_n >= w_l + 0.02 - 1e-9 and mass >= 0.9}
    gap = r3.get("all", {}).get("gap_mean")
    g3 = None if gap is None else ("probabilities" if gap <= 0.10 else ("ranking_only" if gap < 0.30 else "worse_than_qwen"))
    flips_w = {k: float(np.mean([r1[t]["flip_test"][k] for t in WEAK if t in r1])) for k in ("raw_any", "prior_any", "raw_r0_vs_r1", "prior_r0_vs_r1")} if r1 else {}
    pred = {"E1_rot_weak_-1_to_0": bool(r1) and -0.01 - 1e-9 <= mean_over(r1, WEAK, "rot", "acc") - w_raw <= 0 + 1e-9,
            "E1_prior_weak_0_to_+1": bool(r1) and 0 - 1e-9 <= w_prior - w_raw <= 0.01 + 1e-9,
            "E1_L0_weak_about_+1": bool(r1) and 0.0 <= w_L0 - w_raw <= 0.02 + 1e-9,
            "E1_raw_flip_weak_about_20pct": bool(r1) and 0.15 <= flips_w.get("raw_any", 0) <= 0.25,
            "E1_prior_flip_weak_under_15pct": bool(r1) and flips_w.get("prior_any", 1) < 0.15,
            "E1_G1_fails": not g1,
            "E2_N1_within_half_point": "N1" in g2 and abs(g2["N1"]["mean_numeric"] - g2["N1"]["mean_letters"]) <= 0.005 + 1e-9,
            "E2_N2_mass_lower_than_N1": "N2" in g2 and "N1" in g2 and g2["N2"]["mass"] < g2["N1"]["mass"],
            "E2_G2_fails": not any(v["pass"] for v in g2.values()) if g2 else False,
            "E3_gap_0.10_to_0.25": gap is not None and 0.10 <= gap <= 0.25,
            "E3_single_sum_1.2_to_1.5": r3.get("m_alarm_severity", {}).get("single_label_sum") is not None and 1.2 <= r3["m_alarm_severity"]["single_label_sum"] <= 1.5,
            "E3_order_effect_0.05_to_0.15": r3.get("all", {}).get("order_effect") is not None and 0.05 <= r3["all"]["order_effect"] <= 0.15}
    res = {"e1": {t: {k: v for k, v in r1[t].items() if k != "readouts"} | {"readouts": {k: {kk: vv for kk, vv in rd.items() if kk != "correct_test"} for k, rd in r1[t]["readouts"].items()}} for t in r1},
           "e2": r2, "e3": r3, "gates": {"G1_L0_default": g1, "G1b_prior_free": g1b, "G2_numeric": g2, "G3_coherence": g3}, "predictions": pred,
           "means": {"weak": {"raw": w_raw, "rot": mean_over(r1, WEAK, "rot", "acc"), "rot_pmean": mean_over(r1, WEAK, "rot_pmean", "acc"), "prior": w_prior, "L0": w_L0},
                     "strong": {"raw": s_raw, "rot": mean_over(r1, STRONG, "rot", "acc"), "rot_pmean": mean_over(r1, STRONG, "rot_pmean", "acc"), "prior": s_prior, "L0": s_L0},
                     "cov5_weak": {k: mean_over(r1, WEAK, k, "cov5") for k in READOUTS}, "cov5_strong": {k: mean_over(r1, STRONG, k, "cov5") for k in READOUTS},
                     "flips_weak": flips_w}}
    json.dump(res, open(os.path.join(ROOT, "results/v19.json"), "w"), ensure_ascii=False, indent=1)

    L = ["# 23 — 借三件：字母先驗除法 + 旋轉平均、數字編號讀出、機率連貫性（v19）", "",
         "計劃與門檻：`docs/handoff-v19-l0-numeric-coherence.md`（跑前寫死）。D0、cal/test 各 100、26B UD-Q4_K_M、b11371、L4、不開 MTP。溫度與 5% 錯誤預算門檻一律在 cal 擬、在 test 量。", "",
         "## 判定", "",
         f"- **G1 L0 當預設**（弱題 ≥ raw + 0.02 且 5% 預算 coverage ≥ +0.05 且強題不掉 > 0.005）：弱題 {f3(w_raw)} → {f3(w_L0)}、coverage {f3(c_raw)} → {f3(c_L0)}、強題 {f3(s_raw)} → {f3(s_L0)} → {'✓' if g1 else '✗'}",
         f"- **G1b 先驗除法免費**（prior 一次讀取，弱題 ≥ +0.01 且強題不掉）：弱題 {f3(w_raw)} → {f3(w_prior)}、強題 {f3(s_raw)} → {f3(s_prior)} → {'✓' if g1b else '✗'}"]
    for v, gg in g2.items():
        L.append(f"- **G2 數字編號 {v}**（九類 ≥ letters + 0.01 且弱題 ≥ +0.02 且 mass ≥ 0.9）：九類 {f3(gg['mean_letters'])} → {f3(gg['mean_numeric'])}、弱題 {f3(gg['weak_letters'])} → {f3(gg['weak_numeric'])}、mass {f3(gg['mass'])} → {'✓' if gg['pass'] else '✗'}")
    L += [f"- **G3 機率連貫性**（否定差 ≤ 0.10 可當機率；< 0.30 只當排序；≥ 0.30 寫進限制）：{f3(gap)} → {g3}；對照 Jev 0.064、Qwen3.8-27B 第一 token 0.293",
          "- 預測對答案：" + "、".join(f"{k} {'✓' if v else '✗'}" for k, v in pred.items()), "",
          "## E1 四種讀法（test 100 題／類；T 與門檻在 cal 擬）", "",
          "| 題型 | K | raw | rot（對數平均） | rot（機率平均，v5） | prior | L0 | L0 對 raw（只 raw 對／只 L0 對，p） | 5% 預算 coverage raw → L0 | 旋轉翻轉率 raw → prior |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        if t in r1:
            R = r1[t]["readouts"]; m = R["L0"]["vs_raw"]; fl = r1[t]["flip_test"]
            L.append(f"| {t} | {r1[t]['K']} | {f3(R['raw']['acc'])} | {f3(R['rot']['acc'])} | {f3(R['rot_pmean']['acc'])} | {f3(R['prior']['acc'])} | **{f3(R['L0']['acc'])}** | {m[0]}／{m[1]}，{m[2]:.2f} | {f3(R['raw']['cov5'])} → {f3(R['L0']['cov5'])} | {f3(fl['raw_any'])} → {f3(fl['prior_any'])} |")
    if r1:
        L += [f"| **弱題平均** | | {f3(w_raw)} | {f3(res['means']['weak']['rot'])} | {f3(res['means']['weak']['rot_pmean'])} | {f3(w_prior)} | **{f3(w_L0)}** | | {f3(c_raw)} → {f3(c_L0)} | {f3(flips_w['raw_any'])} → {f3(flips_w['prior_any'])} |",
              f"| **強題平均** | | {f3(s_raw)} | {f3(res['means']['strong']['rot'])} | {f3(res['means']['strong']['rot_pmean'])} | {f3(s_prior)} | **{f3(s_L0)}** | | {f3(res['means']['cov5_strong']['raw'])} → {f3(res['means']['cov5_strong']['L0'])} | |", "",
              "位置先驗（cal 100 題 × K 旋轉的位置字母機率平均）：" + "；".join(f"{t} {r1[t]['prior']}" for t in TASKS if t in r1), ""]
    L += ["## E2 數字編號（九類 test 100 題；letters = 同容器 E1 的 raw）", "", "| 題型 | letters | N1 數字 | N1 對 letters（p） | N1 mass | N2 [數字] | N2 對 letters（p） | N2 mass |", "|---|---|---|---|---|---|---|---|"]
    for t in nine:
        if t in r2.get("N1", {}):
            a1 = r2["N1"][t]; a2 = r2.get("N2", {}).get(t)
            L.append(f"| {t} | {f3(a1['acc_letters'])} | {f3(a1['acc_numeric'])} | {a1['vs_letters'][0]}／{a1['vs_letters'][1]}，{a1['vs_letters'][2]:.2f} | {f3(a1['mass_median'])} | "
                     + (f"{f3(a2['acc_numeric'])} | {a2['vs_letters'][0]}／{a2['vs_letters'][1]}，{a2['vs_letters'][2]:.2f} | {f3(a2['mass_median'])} |" if a2 else "— | — | — |"))
    L += ["", "## E3 連貫性（五個多類題 test 100 題，每題兩個標籤；raw 機率）", "",
          "| 題型 | 配對數 | 否定差 \\|P(是)+P(非)−1\\| 平均 ± 95% CI | 中位 | 套各類溫度後 | P(是) 平均 | P(非) 平均 | yes/no 對 K 選 1 的差 | 順序效應 | 單標籤和 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in E3:
        if t in r3:
            x = r3[t]
            L.append(f"| {t} | {x['n_pairs']} | **{f3(x['gap_mean'])}** ± {f3(x['gap_ci95'])} | {f3(x['gap_median'])} | {f3(x['gap_after_T'])} | {f3(x['p_yes_mean'])} | {f3(x['p_not_mean'])} | {f3(x['agree_k_vs_yes'])} | {f3(x['order_effect'])} | {f3(x['single_label_sum']) if x['single_label_sum'] is not None else '—'} |")
    if "all" in r3 and r3["all"]["gap_mean"] is not None:
        L.append(f"| **全部** | {r3['all']['n_pairs']} | **{f3(r3['all']['gap_mean'])}** ± {f3(r3['all']['gap_ci95'])} | | | | | {f3(r3['all']['agree_k_vs_yes'])} | {f3(r3['all']['order_effect'])} | |")
    L += ["", "對照（論文 2609.33209）：Jev 否定差 0.064、Qwen3.8-27B 第一 token 0.293、口頭機率 0.122；Jev 三個單標籤機率和 1.14。"]
    open(os.path.join(ROOT, "results/23-l0-numeric-coherence.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
