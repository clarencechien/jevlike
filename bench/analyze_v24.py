"""v24 analysis (docs/handoff-v24-calibration-abstain.md).

E1  content-free prior (Calibrate Before Use) on v19 rotation rows: raw / prior19 / cf / cf-L0.
E2  calibration methods on v19 rotation-0 logits: temp / vector / affine / dirichlet-odir / isotonic;
    fixed cal-test split, grouped 5-fold CV (judged), learning curve over training size.
E3  'insufficient information' option on v24 rows (see e3()).

Inputs : results/modal/v19/l0/<task>.jsonl, results/modal/v24/cf/cf.jsonl (+ drift.jsonl), results/modal/v24/e3/...
Outputs: results/v24.json, results/28-calibration-abstain-tables.md
"""
import collections
import json
import math
import os
import random
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from analyze import MISSING_LOGP, ece, fit_affine, fit_temperature, softmax  # noqa: E402
from analyze_v19 import TASKS, WEAK, STRONG, e1_task, lp, mcnemar  # noqa: E402
from thresholds import risk_threshold  # noqa: E402

V19 = os.path.join(ROOT, "results", "modal", "v19", "l0")
V24 = os.path.join(ROOT, "results", "modal", "v24")
EPS_LIST = (0.02, 0.05)


def jl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if os.path.exists(p) else []


def letters_of(rows):
    return list("ABCDEFGHIJ"[:len(rows[0]["rotations"])])


def cov_err(conf, correct, tr, ev, eps):
    thr, _ = risk_threshold(conf[tr], correct[tr], eps)
    h = (conf >= thr) & ev
    return int(h.sum()), int((~correct[h]).sum()), int(ev.sum())


# ====================================================================== E1
def e1():
    cf = jl(os.path.join(V24, "cf", "cf.jsonl"))
    if not cf:
        return None
    out = {}
    for t in TASKS:
        rows = jl(os.path.join(V19, f"{t}.jsonl"))
        letters = letters_of(rows); K = len(letters)
        rows, Z, prior19, flips19 = e1_task(rows, letters)
        # content-free prior per rotation: mean over the three inputs of position probabilities
        C = np.zeros((K, K))
        for rr in range(K):
            ps = [np.array([x["pos_probs"].get(L, 0.0) for L in letters]) for x in cf if x["task"] == t and x["r"] == rr]
            p = np.mean(ps, 0)
            C[rr] = np.log(np.clip(p / max(p.sum(), 1e-12), 1e-6, None))
        Zcf = np.zeros((len(rows), K)); Zcf0 = np.zeros((len(rows), K)); flip_cf = []
        for i, r in enumerate(rows):
            A = np.array([lp(rot["aligned_logp"], letters) for rot in r["rotations"]])
            P = np.array([[A[rr, j] - C[rr, (j + rr) % K] for j in range(K)] for rr in range(K)])
            Zcf0[i] = P[0]; Zcf[i] = P.mean(0)
            flip_cf.append(len(set(P.argmax(1).tolist())) > 1)
        y = np.array([letters.index(r["gold"]) for r in rows]); te = np.array([r["split"] == "test" for r in rows]); ca = ~te
        res = {"K": K, "cf_prior_r0": np.exp(C[0]).round(4).tolist(), "prior19": prior19.round(4).tolist(), "readouts": {}}
        for k, M in (("raw", Z["raw"]), ("prior19", Z["prior"]), ("cf", Zcf0), ("cf-L0", Zcf)):
            T = fit_temperature(M[ca], y[ca]); Pm = softmax(M / T); conf = Pm.max(1); correct = Pm.argmax(1) == y
            h, e, n = cov_err(conf, correct, ca, te, 0.05)
            res["readouts"][k] = {"acc": float(correct[te].mean()), "T": T, "ece": ece(conf[te], correct[te]), "cov5": h / n,
                                  "err5": e / h if h else 0.0, "correct_test": correct[te].tolist()}
        for k in ("prior19", "cf", "cf-L0"):
            res["readouts"][k]["vs_raw"] = mcnemar(res["readouts"]["raw"]["correct_test"], res["readouts"][k]["correct_test"])
        res["flip_test"] = {"raw": float(np.mean(np.array(flips19["raw_any"])[te])), "prior19": float(np.mean(np.array(flips19["prior_any"])[te])),
                            "cf": float(np.mean(np.array(flip_cf)[te]))}
        out[t] = res
    # drift check: rotation-0 re-reads vs the stored v19 rows
    drift = jl(os.path.join(V24, "cf", "drift.jsonl")); dmax = 0.0; agree = 0
    v19 = {r["id"]: r for t in TASKS for r in jl(os.path.join(V19, f"{t}.jsonl"))}
    for d in drift:
        old = v19[d["id"]]["rotations"][0]["pos_probs"]
        dmax = max(dmax, max(abs(d["pos_probs"].get(L, 0.0) - old.get(L, 0.0)) for L in set(old) | set(d["pos_probs"])))
        agree += max(d["pos_probs"], key=d["pos_probs"].get) == max(old, key=old.get)
    out["_drift"] = {"n": len(drift), "max_abs_dp": round(dmax, 4), "argmax_agree": agree}
    return out


# ====================================================================== E2 methods
def norm_logp(Z):
    Z = np.clip(Z, MISSING_LOGP, 0)
    m = Z.max(1, keepdims=True)
    return Z - (m + np.log(np.exp(Z - m).sum(1, keepdims=True)))


def nll_of(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))))


def fit_linear(X, y, K, full, lam):
    """softmax(X W^T + b). full=False: diagonal W (vector scaling); full=True: Dirichlet calibration with ODIR.
    Analytic gradient (numerical gradients made the learning curve take hours)."""
    n = len(y); Y = np.eye(K)[y]
    off_mask = 1 - np.eye(K)

    def unpack(th):
        if full:
            return th[:K * K].reshape(K, K), th[K * K:]
        return np.diag(th[:K]), th[K:]

    def loss_grad(th):
        W, b = unpack(th)
        S = X @ W.T + b; S = S - S.max(1, keepdims=True)
        E = np.exp(S); P = E / E.sum(1, keepdims=True)
        L = -np.mean(np.log(np.clip(P[np.arange(n), y], 1e-300, None)))
        G = (P - Y) / n                       # dL/dS
        gW = G.T @ X; gb = G.sum(0)
        if full:
            L += lam * ((W * off_mask) ** 2).sum() / max(1, K * (K - 1)) + lam * (b ** 2).sum() / K
            gW = gW + 2 * lam * W * off_mask / max(1, K * (K - 1)); gb = gb + 2 * lam * b / K
            return L, np.concatenate([gW.ravel(), gb])
        d = np.diag(W)
        L += 1e-4 * ((d - 1) ** 2).sum()
        return L, np.concatenate([np.diag(gW) + 2e-4 * (d - 1), gb])

    th0 = np.concatenate([np.eye(K).ravel(), np.zeros(K)]) if full else np.concatenate([np.ones(K), np.zeros(K)])
    th = minimize(loss_grad, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    W, b = unpack(th)

    def pred(Xt):
        return softmax(Xt @ W.T + b)
    return pred


def odir_lambda(X, y, K, groups, seed=0):
    grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    folds = grouped_folds(groups, 3, seed)
    best, bl = None, float("inf")
    for lam in grid:
        v = []
        for f in folds:
            tr = ~f
            if len(set(y[tr].tolist())) < 2:
                continue
            P = fit_linear(X[tr], y[tr], K, True, lam)(X[f]); v.append(nll_of(P, y[f]))  # noqa: E702
        if v and np.mean(v) < bl:
            bl, best = float(np.mean(v)), lam
    return best or 1.0


def fit_method(method, Z, y, K, tr, groups):
    """Fit on the rows in `tr`; returns predict(mask) -> (P, conf)."""
    X = norm_logp(Z)
    if method == "temp":
        T = fit_temperature(Z[tr], y[tr])
        return lambda m: (lambda P: (P, P.max(1)))(softmax(Z[m] / T))
    if method == "vector":
        f = fit_linear(X[tr], y[tr], K, False, 0)
        return lambda m: (lambda P: (P, P.max(1)))(f(X[m]))
    if method == "affine":
        pr = fit_affine(Z[tr], y[tr], K)
        if pr is None:
            T = fit_temperature(Z[tr], y[tr])
            return lambda m: (lambda P: (P, P.max(1)))(softmax(Z[m] / T))
        return lambda m: (lambda P: (P, P.max(1)))(pr(Z[m]))
    if method == "dirichlet":
        lam = odir_lambda(X[tr], y[tr], K, groups[tr])
        f = fit_linear(X[tr], y[tr], K, True, lam)
        return lambda m: (lambda P: (P, P.max(1)))(f(X[m]))
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        T = fit_temperature(Z[tr], y[tr]); Ptr = softmax(Z[tr] / T)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(Ptr.max(1), (Ptr.argmax(1) == y[tr]).astype(float))

        def pred(m):
            P = softmax(Z[m] / T); c = iso.predict(P.max(1)); top = P.argmax(1); rest = 1 - P[np.arange(len(P)), top]
            Q = P * np.where(rest > 0, (1 - c) / np.clip(rest, 1e-12, None), 1.0)[:, None]
            Q[np.arange(len(P)), top] = c
            return Q, c
        return pred
    raise ValueError(method)


METHODS = ["temp", "vector", "affine", "dirichlet", "isotonic"]


def grouped_folds(groups, k, seed):
    g = sorted(set(groups.tolist())); rng = random.Random(seed); rng.shuffle(g)
    fold_of = {x: i % k for i, x in enumerate(g)}
    return [np.array([fold_of[x] == f for x in groups]) for f in range(k)]


def classwise_ece(P, y, K, bins=10):
    return float(np.mean([ece(P[:, k], (y == k).astype(float), bins) for k in range(K)]))


def brier(P, y, K):
    Y = np.eye(K)[y]
    return float(np.mean(((P - Y) ** 2).sum(1)))


def evaluate(method, Z, y, K, groups, splits):
    """splits: list of (train_mask, eval_mask). Confidence/threshold fitted on train, measured on eval; pooled."""
    allP, ally, allc, allcorr = [], [], [], []
    cov = {e: [0, 0, 0] for e in EPS_LIST}
    for tr, ev in splits:
        # the threshold is fitted on the train predictions of the same fitted calibrator
        pred = fit_method(method, Z, y, K, tr, groups)
        P_ev, c_ev = pred(ev); P_tr, c_tr = pred(tr)
        corr_ev = P_ev.argmax(1) == y[ev]; corr_tr = P_tr.argmax(1) == y[tr]
        for e in EPS_LIST:
            thr, _ = risk_threshold(c_tr, corr_tr, e)
            h = c_ev >= thr
            cov[e][0] += int(h.sum()); cov[e][1] += int((~corr_ev[h]).sum()); cov[e][2] += int(ev.sum())
        allP.append(P_ev); ally.append(y[ev]); allc.append(c_ev); allcorr.append(corr_ev)
    P = np.concatenate(allP); yy = np.concatenate(ally); c = np.concatenate(allc); corr = np.concatenate(allcorr)
    out = {"acc": float(corr.mean()), "ece": ece(c, corr), "cw_ece": classwise_ece(P, yy, K), "nll": nll_of(P, yy), "brier": brier(P, yy, K)}
    for e in EPS_LIST:
        h, err, n = cov[e]
        out[f"cov{int(e * 100)}"] = h / n; out[f"err{int(e * 100)}"] = err / h if h else 0.0
    return out


def task_data(t):
    rows = jl(os.path.join(V19, f"{t}.jsonl"))
    letters = letters_of(rows); K = len(letters)
    Z = np.array([lp(r["rotations"][0]["aligned_logp"], letters) for r in rows])
    y = np.array([letters.index(r["gold"]) for r in rows])
    groups = np.array([r.get("pair_id") or r["id"] for r in rows])
    te = np.array([r["split"] == "test" for r in rows])
    return Z, y, K, groups, te


def e2():
    out = {"fixed": {}, "cv": {}, "curve": {}}
    for t in TASKS:
        Z, y, K, groups, te = task_data(t)
        out["fixed"][t] = {m: evaluate(m, Z, y, K, groups, [(~te, te)]) for m in METHODS}
        folds = grouped_folds(groups, 5, 0)
        out["cv"][t] = {m: evaluate(m, Z, y, K, groups, [(~f, f) for f in folds]) for m in METHODS}
        print(f"[E2] {t} cv " + " ".join(f"{m}:{out['cv'][t][m]['ece']:.3f}/{out['cv'][t][m]['cov5']:.2f}" for m in METHODS), flush=True)
    # learning curve: 20 repetitions; each holds out 25 pair groups (~50 rows), trains on n rows (grouped) from the rest
    sizes = (25, 50, 100, 150)
    for t in TASKS:
        Z, y, K, groups, te = task_data(t)
        g_all = sorted(set(groups.tolist())); res = {m: {n: [] for n in sizes} for m in ("temp", "vector", "dirichlet", "isotonic")}
        for rep in range(20):
            rng = random.Random(1000 + rep); g = list(g_all); rng.shuffle(g)
            hold = set(g[:25]); pool = g[25:]
            ev = np.array([x in hold for x in groups])
            for n in sizes:
                take, cnt = set(), 0
                for x in pool:
                    if cnt >= n:
                        break
                    take.add(x); cnt += int((groups == x).sum())
                tr = np.array([x in take for x in groups])
                if len(set(y[tr].tolist())) < 2:
                    continue
                for m in res:
                    r = evaluate(m, Z, y, K, groups, [(tr, ev)])
                    res[m][n].append((r["ece"], r["cov5"], r["err5"], r["acc"]))
        out["curve"][t] = {m: {n: [float(np.mean([v[i] for v in res[m][n]])) for i in range(4)] for n in sizes if res[m][n]} for m in res}
        print(f"[E2 curve] {t} done", flush=True)
    return out


def e2_summary(e2r):
    s = {}
    for m in METHODS:
        cv = e2r["cv"]
        held = sum(cv[t][m]["err5"] <= 0.05 for t in TASKS)
        s[m] = {"weak_cov5": float(np.mean([cv[t][m]["cov5"] for t in WEAK])), "strong_cov5": float(np.mean([cv[t][m]["cov5"] for t in STRONG])),
                "held5": int(held), "ece_mean": float(np.mean([cv[t][m]["ece"] for t in TASKS])),
                "cw_ece_mean": float(np.mean([cv[t][m]["cw_ece"] for t in TASKS])), "nll_mean": float(np.mean([cv[t][m]["nll"] for t in TASKS])),
                "weak_acc": float(np.mean([cv[t][m]["acc"] for t in WEAK])), "strong_acc": float(np.mean([cv[t][m]["acc"] for t in STRONG]))}
    base = s["temp"]
    for m in METHODS[1:]:
        x = s[m]
        x["G2"] = bool(x["weak_cov5"] >= base["weak_cov5"] + 0.05 and x["held5"] >= base["held5"] and x["strong_cov5"] >= base["strong_cov5"] - 0.01
                       and x["ece_mean"] <= base["ece_mean"])
    # crossover: smallest training size where method ECE < temp ECE, averaged over tasks
    cross = {}
    for m in ("vector", "dirichlet", "isotonic"):
        cross[m] = {}
        for n in (25, 50, 100, 150):
            a = [e2r["curve"][t][m][n][0] for t in TASKS if n in e2r["curve"][t][m]]
            b = [e2r["curve"][t]["temp"][n][0] for t in TASKS if n in e2r["curve"][t]["temp"]]
            cross[m][n] = {"ece": float(np.mean(a)), "temp_ece": float(np.mean(b)),
                           "cov5": float(np.mean([e2r["curve"][t][m][n][1] for t in TASKS])), "temp_cov5": float(np.mean([e2r["curve"][t]["temp"][n][1] for t in TASKS]))}
    return s, cross


# ====================================================================== E3
E3_TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "p_uph_anomaly", "t_tool_gate"]
E3_R = ["T0", "J0", "J1n"]
V23_R = {"T0": "T0", "J0": "J0", "J1n": "J1"}
E3_MODELS = [("26b", "26B"), ("e4b", "E4B")]


def _logits(rows, letters):
    return np.array([[MISSING_LOGP if r["raw_logprobs"].get(L) is None else r["raw_logprobs"][L] for L in letters] for r in rows])


def cal_pairs(pair_ids, seed=0):
    g = sorted(set(pair_ids)); rng = random.Random(seed); rng.shuffle(g)
    return set(g[: len(g) // 2])


def e3():
    root = os.path.join(V24, "e3")
    if not os.path.isdir(root):
        return None
    out, md = {}, ["", "## E3「資訊不足」選項", ""]
    for m, label in E3_MODELS:
        if not os.path.isdir(os.path.join(root, m)):
            continue
        res = {}
        for R in E3_R:
            agg = collections.defaultdict(list); fam = collections.defaultdict(list); qminus = collections.defaultdict(list); per_task = {}
            for t in E3_TASKS:
                meta = {x["id"]: x for x in jl(os.path.join(ROOT, "data", "v24", "Qplus", R, f"{t}.jsonl"))}
                qp = {x["id"]: x for x in jl(os.path.join(root, m, "Qplus", R, f"{t}.jsonl")) if "chosen" in x}
                qm = {x["id"]: x for x in jl(os.path.join(root, m, "Qminus", R, f"{t}.jsonl")) if "chosen" in x}
                if not qp:
                    continue
                ins = next(iter(meta.values()))["insufficient_letter"]
                calset = cal_pairs([x["pair_id"] for x in meta.values()])
                tp = collections.defaultdict(list)
                for i, x in meta.items():
                    if i not in qp:
                        continue
                    v = x["variant"]; ch = qp[i]["chosen"]; test = x["pair_id"] not in calset
                    agg[f"{v}_ins"].append(ch == ins); tp[f"{v}_ins"].append(ch == ins)
                    if test:
                        agg[f"{v}_ins_test"].append(ch == ins)
                    if v == "M":
                        fam[x["family"]].append(ch == ins)
                    if v in ("C", "R"):
                        agg[f"{v}_acc_plus"].append(ch == x["orig_gold"])
                        if i in qm:
                            agg[f"{v}_acc_minus"].append(qm[i]["chosen"] == x["orig_gold"])
                per_task[t] = {k: float(np.mean(v)) for k, v in tp.items()}
                # Q-: temperature and 5% cut fitted on v23 complete rows (same model, rendering) of the cal-half pairs
                v23 = [x for x in jl(os.path.join(ROOT, "results", "modal", "v23", m, V23_R[R], f"{t}.jsonl")) if "correct" in x]
                letters = list("ABCDEFGHIJ"[:len(next(iter(meta.values()))["question"]["options"]) - 1])
                cal = [x for x in v23 if x["pair_id"] in calset]
                if not cal or not qm:
                    continue
                Zc = _logits(cal, letters); yc = np.array([letters.index(x["gold"]) for x in cal])
                T = fit_temperature(Zc, yc); Pc = softmax(Zc / T)
                thr, _ = risk_threshold(Pc.max(1), Pc.argmax(1) == yc, 0.05)
                test_rows = [(meta[i], qm[i]) for i in qm if i in meta and meta[i]["pair_id"] not in calset]
                Zt = _logits([r for _, r in test_rows], letters); conf = softmax(Zt / T).max(1); rawc = np.array([max(r["probs"].values()) if r["probs"] else 0 for _, r in test_rows])
                for (x, r), c, rc in zip(test_rows, conf, rawc):
                    qminus[f"{x['variant']}_routed"].append(bool(c < thr))
                    qminus[f"{x['variant']}_rawconf"].append(float(rc))
                    if x["variant"] == "C" and c >= thr:
                        qminus["C_auto_correct"].append(r["chosen"] == x["orig_gold"])
            if not agg:
                continue
            a_ = lambda k: float(np.mean(agg[k])) if agg[k] else None  # noqa: E731
            q_ = lambda k: float(np.mean(qminus[k])) if qminus[k] else None  # noqa: E731
            cp = agg["C_acc_plus"]; cm = agg["C_acc_minus"]
            n = min(len(cp), len(cm))
            mc = mcnemar(cm[:n], cp[:n]) if n else (0, 0, 1.0)
            res[R] = {"M_recall": a_("M_ins"), "M_recall_test": a_("M_ins_test"), "C_false": a_("C_ins"), "R_false": a_("R_ins"),
                      "C_acc_plus": a_("C_acc_plus"), "C_acc_minus": a_("C_acc_minus"), "C_mcnemar": mc, "R_acc_plus": a_("R_acc_plus"),
                      "Qminus_M_routed": q_("M_routed"), "Qminus_C_routed": q_("C_routed"), "Qminus_R_routed": q_("R_routed"),
                      "Qminus_C_auto_acc": q_("C_auto_correct"),
                      "Qminus_M_rawconf_median": float(np.median(qminus["M_rawconf"])) if qminus["M_rawconf"] else None,
                      "Qminus_M_rawconf_ge95": float(np.mean(np.array(qminus["M_rawconf"]) >= 0.95)) if qminus["M_rawconf"] else None,
                      "family_M_recall": {k: float(np.mean(v)) for k, v in sorted(fam.items())}, "per_task": per_task,
                      "n": {k: len(v) for k, v in agg.items()}}
        out[m] = res
        md += [f"### {label}", "", "| 寫法 | M 選資訊不足（recall） | C 誤選 | R 誤選 | C 準確率 Q+ ／ Q−（p） | Q− 門檻轉人：M ／ C | Q− 自動處理的 C 準確率 | Q− 時 M 的原始信心中位數（≥ 0.95 比例） |",
               "|---|---|---|---|---|---|---|---|"]
        for R in E3_R:
            if R not in res:
                continue
            x = res[R]
            md.append(f"| {R} | **{fmt(x['M_recall'])}** | {fmt(x['C_false'])} | {fmt(x['R_false'])} | {fmt(x['C_acc_plus'])} ／ {fmt(x['C_acc_minus'])}（{x['C_mcnemar'][2]:.2f}） | "
                      f"{fmt(x['Qminus_M_routed'])} ／ {fmt(x['Qminus_C_routed'])} | {fmt(x['Qminus_C_auto_acc'])} | {fmt(x['Qminus_M_rawconf_median'])}（{fmt(x['Qminus_M_rawconf_ge95'])}） |")
        md += ["", "M recall 依家族（" + "／".join(R for R in E3_R if R in res) + "）：", "", "| family | " + " | ".join(R for R in E3_R if R in res) + " |", "|---|" + "---|" * len([R for R in E3_R if R in res])]
        fams = sorted({f for R in res for f in res[R]["family_M_recall"]})
        for f_ in fams:
            md.append(f"| {f_} | " + " | ".join(fmt(res[R]["family_M_recall"].get(f_)) for R in E3_R if R in res) + " |")
        md += ["", "依題型（C 誤選／M recall／R 誤選）：", "", "| task | " + " | ".join(R for R in E3_R if R in res) + " |", "|---|" + "---|" * len([R for R in E3_R if R in res])]
        for t in E3_TASKS:
            md.append(f"| {t} | " + " | ".join(f"{fmt(res[R]['per_task'].get(t, {}).get('C_ins'), 2)}／{fmt(res[R]['per_task'].get(t, {}).get('M_ins'), 2)}／{fmt(res[R]['per_task'].get(t, {}).get('R_ins'), 2)}"
                                               for R in E3_R if R in res) + " |")
        md.append("")
    if "26b" in out and "J0" in out["26b"]:
        x = out["26b"]["J0"]
        g3a = (x["M_recall"] or 0) >= 0.80 and (x["C_false"] or 0) <= 0.05 and (x["R_false"] or 0) <= 0.10 and \
            (x["C_acc_plus"] or 0) >= (x["C_acc_minus"] or 0) - 0.01 and x["C_mcnemar"][2] >= 0.05
        g3b = (x["M_recall_test"] or 0) >= (x["Qminus_M_routed"] or 0) + 0.20
        g3c = (out["26b"].get("J1n", {}).get("M_recall") or 0) >= 0.95
        out["gates"] = {"G3a": bool(g3a), "G3b": bool(g3b), "G3c": bool(g3c)}
        md.append(f"判定（26B，J0）：G3a 加選項可用 {'✓' if g3a else '✗'}；G3b 選項比門檻好 {'✓' if g3b else '✗'}（Q+ test 半 M recall {fmt(x['M_recall_test'])} 對 Q− 轉人 {fmt(x['Qminus_M_routed'])}）；"
                  f"G3c 程式標 null 就夠 {'✓' if g3c else '✗'}。")
    return out, md


def fmt(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main():
    full, md = {}, ["# 28 — 校準與「資訊不足」選項：完整表格（`bench/analyze_v24.py` 產生）", ""]
    r1 = e1()
    if r1:
        full["E1"] = {t: {k: v for k, v in r1[t].items()} for t in TASKS}; full["E1"]["_drift"] = r1["_drift"]
        md += ["## E1 空白輸入扣偏好（test 100 題／類；溫度與 5% 門檻在 cal 擬）", "",
               "| 題型 | K | raw | prior19 | cf | cf-L0 | cf 對 raw（只 raw 對／只 cf 對，p） | 5% coverage raw → cf | 翻轉率 raw → prior19 → cf | cf 先驗（旋轉 0） |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for t in TASKS:
            x = r1[t]; R = x["readouts"]
            md.append(f"| {t} | {x['K']} | {R['raw']['acc']:.3f} | {R['prior19']['acc']:.3f} | **{R['cf']['acc']:.3f}** | {R['cf-L0']['acc']:.3f} | "
                      f"{R['cf']['vs_raw'][0]}／{R['cf']['vs_raw'][1]}，{R['cf']['vs_raw'][2]:.2f} | {R['raw']['cov5']:.2f} → {R['cf']['cov5']:.2f} | "
                      f"{x['flip_test']['raw']:.3f} → {x['flip_test']['prior19']:.3f} → {x['flip_test']['cf']:.3f} | {x['cf_prior_r0']} |")
        agg = {}
        for grp, ts in (("弱題", WEAK), ("強題", STRONG)):
            agg[grp] = {k: {"acc": float(np.mean([r1[t]["readouts"][k]["acc"] for t in ts])), "cov5": float(np.mean([r1[t]["readouts"][k]["cov5"] for t in ts])),
                            "ece": float(np.mean([r1[t]["readouts"][k]["ece"] for t in ts]))} for k in ("raw", "prior19", "cf", "cf-L0")}
            agg[grp]["flip"] = {k: float(np.mean([r1[t]["flip_test"][k] for t in ts])) for k in ("raw", "prior19", "cf")}
            md.append(f"| **{grp}平均** | | {agg[grp]['raw']['acc']:.3f} | {agg[grp]['prior19']['acc']:.3f} | **{agg[grp]['cf']['acc']:.3f}** | {agg[grp]['cf-L0']['acc']:.3f} | | "
                      f"{agg[grp]['raw']['cov5']:.3f} → {agg[grp]['cf']['cov5']:.3f} | {agg[grp]['flip']['raw']:.3f} → {agg[grp]['flip']['prior19']:.3f} → {agg[grp]['flip']['cf']:.3f} | |")
        w, s_ = agg["弱題"], agg["強題"]
        g1 = (w["cf"]["acc"] >= w["raw"]["acc"] + 0.02 or w["cf"]["cov5"] >= w["raw"]["cov5"] + 0.05) and s_["cf"]["acc"] >= s_["raw"]["acc"] - 0.005
        g1b = w["flip"]["cf"] <= 0.055 and w["cf"]["acc"] >= w["raw"]["acc"] - 0.005
        full["E1_summary"] = {"agg": agg, "G1": bool(g1), "G1b": bool(g1b)}
        md += ["", f"G1（cf 當預設）：{'✓' if g1 else '✗'}；G1b（翻轉率減半）：{'✓' if g1b else '✗'}。"
               f"重讀 drift：{r1['_drift']['n']} 題旋轉 0，最大機率差 {r1['_drift']['max_abs_dp']}，argmax 一致 {r1['_drift']['argmax_agree']}/{r1['_drift']['n']}。", ""]
    r2 = e2()
    s2, cross = e2_summary(r2)
    full["E2"] = r2; full["E2_summary"] = s2; full["E2_cross"] = cross
    md += ["## E2 校準方法（grouped 5-fold CV，200 題／類；判定看這張）", "",
           "| 方法 | 弱題 acc | 強題 acc | 弱題 5% coverage | 強題 5% coverage | 5% 守住類數 | 十類 ECE | classwise-ECE | NLL | G2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        x = s2[m]
        md.append(f"| {m} | {x['weak_acc']:.3f} | {x['strong_acc']:.3f} | {x['weak_cov5']:.3f} | {x['strong_cov5']:.3f} | {x['held5']}/10 | {x['ece_mean']:.3f} | "
                  f"{x['cw_ece_mean']:.3f} | {x['nll_mean']:.3f} | {'—' if m == 'temp' else ('✓' if x['G2'] else '✗')} |")
    md += ["", "### 每類（CV）：ECE ／ 5% coverage（實際錯誤率）", "", "| 題型 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for t in TASKS:
        md.append(f"| {t} | " + " | ".join(f"{r2['cv'][t][m]['ece']:.3f} ／ {r2['cv'][t][m]['cov5']:.2f}（{r2['cv'][t][m]['err5']:.3f}）" for m in METHODS) + " |")
    md += ["", "### 固定 split（cal 100 → test 100），對照用", "", "| 題型 | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for t in TASKS:
        md.append(f"| {t} | " + " | ".join(f"{r2['fixed'][t][m]['acc']:.2f}／{r2['fixed'][t][m]['ece']:.3f}／{r2['fixed'][t][m]['cov5']:.2f}" for m in METHODS) + " |")
    md += ["", "（格式：acc／ECE／5% coverage）", "", "### 學習曲線：訓練筆數 vs 十類平均 ECE 與 5% coverage（20 次重抽，每次留 25 組 pair 量）", "",
           "| 訓練筆數 | temp ECE | vector | dirichlet | isotonic | temp cov5 | vector | dirichlet | isotonic |", "|---|---|---|---|---|---|---|---|---|"]
    for n in (25, 50, 100, 150):
        md.append(f"| {n} | {cross['vector'][n]['temp_ece']:.3f} | {cross['vector'][n]['ece']:.3f} | {cross['dirichlet'][n]['ece']:.3f} | {cross['isotonic'][n]['ece']:.3f} | "
                  f"{cross['vector'][n]['temp_cov5']:.3f} | {cross['vector'][n]['cov5']:.3f} | {cross['dirichlet'][n]['cov5']:.3f} | {cross['isotonic'][n]['cov5']:.3f} |")
    r3 = e3()
    if r3:
        full["E3"] = r3[0]; md += r3[1]
    json.dump(full, open(os.path.join(ROOT, "results", "v24.json"), "w"), ensure_ascii=False, indent=1, default=float)
    open(os.path.join(ROOT, "results", "28-calibration-abstain-tables.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
