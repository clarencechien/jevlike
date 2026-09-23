"""M5 CPU analysis over raw logprobs pulled back from Modal.

Inputs : results/modal/accuracy/{task}.jsonl, results/modal/accuracy/control_{task}.jsonl,
         data/synthetic/{task}.jsonl, data/rules/{task}.json
Outputs: results/04-accuracy.md, results/analysis.json, results/fig/reliability-{task}.png,
         results/fig/learning-curve-{task}.png
Usage  : python3 bench/analyze.py [--seed 0]
"""
import argparse
import collections
import json
import math
import os
import random
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = json.load(open(os.path.join(ROOT, "data/seeds/tasks.json"), encoding="utf-8"))
ACC_DIR = os.path.join(ROOT, "results/modal/accuracy")
FIG_DIR = os.path.join(ROOT, "results/fig")
LC_TASKS = ["m_alarm_category", "x_ticket_route", "q_spc_action"]
LC_N = [25, 50, 100, 200, 500]
MISSING_LOGP = -30.0
# dataviz reference palette (light): series 1-4, ink, muted, grid, surface
C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "ink": "#0b0b0b", "muted": "#898781",
     "grid": "#e1e0d9", "surface": "#fcfcfb", "sec": "#52514e"}


# ---------------------------------------------------------------- loading
def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def logit_matrix(rows, letters):
    X = np.full((len(rows), len(letters)), MISSING_LOGP)
    for i, r in enumerate(rows):
        for j, L in enumerate(letters):
            v = r["raw_logprobs"].get(L)
            if v is not None:
                X[i, j] = v
    return X


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def group_split(rows, seed):
    groups = sorted({r.get("pair_id") or r["id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    is_cal = np.array([(r.get("pair_id") or r["id"]) in cal for r in rows])
    return is_cal


# ---------------------------------------------------------------- metrics
def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def macro_f1(y, yhat, K):
    f = []
    for k in range(K):
        tp = np.sum((y == k) & (yhat == k)); fp = np.sum((y != k) & (yhat == k)); fn = np.sum((y == k) & (yhat != k))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f.append(2 * p * r / (p + r) if p + r else 0)
    return float(np.mean(f)) if f else float("nan")


def auroc(y_bin, score):
    from sklearn.metrics import roc_auc_score
    if len(set(y_bin.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, score))


def selective(conf, correct, thr):
    m = conf >= thr
    return {"acc": float(correct[m].mean()) if m.any() else float("nan"), "coverage": float(m.mean())}


def metrics(P, y, K, prefix=""):
    yhat = P.argmax(1)
    conf = P.max(1)
    correct = (yhat == y)
    out = {"n": int(len(y)), "acc": float(correct.mean()), "macro_f1": macro_f1(y, yhat, K),
           "ece": ece(conf, correct), "sel80": selective(conf, correct, 0.8), "sel90": selective(conf, correct, 0.9),
           "sel95": selective(conf, correct, 0.95), "conf_mean": float(conf.mean())}
    if K == 2:
        out["auroc"] = auroc((y == 0).astype(int), P[:, 0])
    else:
        top2 = np.argsort(-P, 1)[:, :2]
        out["top2"] = float(np.mean([y[i] in top2[i] for i in range(len(y))]))
        try:
            from sklearn.metrics import roc_auc_score
            Y = np.eye(K)[y]
            keep = Y.sum(0) > 0
            out["auroc_ovr"] = float(roc_auc_score(Y[:, keep], P[:, keep], average="macro")) if keep.sum() > 1 else float("nan")
        except Exception:  # noqa: BLE001
            out["auroc_ovr"] = float("nan")
    return {prefix + k: v for k, v in out.items()}


# ---------------------------------------------------------------- calibration
def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def fit_temperature(Z, y):
    best_T, best = 1.0, float("inf")
    for T in np.exp(np.linspace(math.log(0.05), math.log(50), 200)):
        v = nll(softmax(Z / T), y)
        if v < best:
            best, best_T = v, float(T)
    return best_T


def fit_affine(Z, y, K):
    """Vector/Platt-style scaling with bias: multinomial LR on logprob features (can move argmax)."""
    from sklearn.linear_model import LogisticRegression
    Zc = np.clip(Z, -30, 0)
    if len(set(y.tolist())) < 2:
        return None
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(Zc, y)
    classes = list(lr.classes_)

    def predict(Zt):
        Pt = np.zeros((len(Zt), K))
        Pt[:, classes] = lr.predict_proba(np.clip(Zt, -30, 0))
        return Pt
    return predict


# ---------------------------------------------------------------- baselines
def rules_predict(task, states, letters):
    p = os.path.join(ROOT, "data/rules", f"{task}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    comp = [(re.compile(r["pattern"], re.I), r["label"]) for r in d["rules"]]
    out = []
    for s in states:
        lab = d["default"]
        for rx, L in comp:
            if rx.search(s):
                lab = L
                break
        out.append(letters.index(lab))
    return np.array(out)


def tfidf_lr(train_states, train_y, test_states, seed=0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    if len(set(train_y.tolist())) < 2:
        return np.full(len(test_states), int(train_y[0]))
    pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=1, sublinear_tf=True),
                         LogisticRegression(C=10, max_iter=3000, random_state=seed))
    pipe.fit(train_states, train_y)
    return pipe.predict(test_states)


# ---------------------------------------------------------------- plotting
def reliability_fig(task, arms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4), facecolor=C["surface"])
    if len(arms) == 1:
        axes = [axes]
    for ax, (name, conf, correct, e) in zip(axes, arms):
        ax.set_facecolor(C["surface"])
        edges = np.linspace(0, 1, 11)
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
            if m.any():
                xs.append((lo + hi) / 2); ys.append(correct[m].mean()); ns.append(m.sum())
        ax.plot([0, 1], [0, 1], color=C["grid"], lw=1.5, ls="--", zorder=1)
        ax.bar(xs, ys, width=0.09, color=C["s1"], edgecolor=C["surface"], linewidth=2, zorder=2)
        for x, yv, n in zip(xs, ys, ns):
            ax.text(x, min(yv + 0.02, 0.98), str(n), ha="center", va="bottom", fontsize=7, color=C["sec"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"{name}  ECE={e:.3f}", fontsize=10, color=C["ink"])
        ax.set_xlabel("confidence", color=C["sec"]); ax.set_ylabel("accuracy", color=C["sec"])
        ax.grid(axis="y", color=C["grid"], lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(C["grid"])
        ax.tick_params(colors=C["muted"], labelsize=8)
    fig.suptitle(f"{task} — reliability (test split, L4 / Gemma 4 26B-A4B Q4_K_M)", fontsize=11, color=C["ink"])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def learning_curve_fig(task, curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=C["surface"])
    ax.set_facecolor(C["surface"])
    series = [("TF-IDF + LR（N 筆訓練）", "tfidf", C["s1"]),
              ("typed decision 零樣本（argmax 不變）", "typed0", C["s2"]),
              ("typed decision 校準後 選擇性 acc @0.9", "typed_sel90", C["s3"]),
              ("↳ 其 coverage", "typed_cov90", C["s4"])]
    for label, key, col in series:
        ns = [n for n in curve["N"] if key in curve["mean"] and not math.isnan(curve["mean"][key][str(n)])]
        if not ns:
            continue
        m = [curve["mean"][key][str(n)] for n in ns]
        s = [curve["std"][key][str(n)] for n in ns]
        ls = "--" if key == "typed_cov90" else "-"
        ax.plot(ns, m, color=col, lw=2, ls=ls, marker="o", ms=6, label=label, zorder=3)
        ax.fill_between(ns, np.array(m) - np.array(s), np.array(m) + np.array(s), color=col, alpha=0.12, lw=0)
    ax.axhline(0.9, color=C["grid"], lw=1.2, ls=":", zorder=1)
    ax.text(curve["N"][0], 0.905, "0.90", fontsize=8, color=C["muted"])
    from matplotlib.ticker import NullFormatter, NullLocator
    ax.set_xscale("log"); ax.set_xticks(curve["N"]); ax.set_xticklabels([str(n) for n in curve["N"]])
    ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.02); ax.set_xlabel("標註筆數 N（calibration 集抽樣，3 seeds）", color=C["sec"]); ax.set_ylabel("accuracy / coverage（test 集）", color=C["sec"])
    ax.set_title(f"{task} — 標註量學習曲線", fontsize=11, color=C["ink"])
    ax.grid(axis="y", color=C["grid"], lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- per task
def analyze_task(task, seed):
    rows = [r for r in load_jsonl(os.path.join(ACC_DIR, f"{task}.jsonl")) if "raw_logprobs" in r]
    if not rows:
        return None
    data = {r["id"]: r for r in load_jsonl(os.path.join(ROOT, "data/synthetic", f"{task}.jsonl"))}
    rows = [r for r in rows if r["id"] in data]
    letters = list(TASKS[task]["options"])
    K = len(letters)
    Z = logit_matrix(rows, letters)
    y = np.array([letters.index(r["gold"]) for r in rows])
    states = [data[r["id"]]["state"] for r in rows]
    diff = np.array([r["difficulty"] for r in rows]); lang = np.array([r["lang"] for r in rows])
    missing = np.array([bool(r["missing"]) for r in rows])
    is_cal = group_split(rows, seed)
    te, ca = ~is_cal, is_cal
    P0 = softmax(Z)
    res = {"task": task, "kind": TASKS[task]["kind"], "K": K, "n_all": len(rows), "n_cal": int(ca.sum()), "n_test": int(te.sum()),
           "missing_rate": float(missing.mean()), "missing_rate_gold": float(np.mean([r["gold"] in r["missing"] for r in rows])),
           "latency_ms_p50": float(np.median([r["latency_ms"] for r in rows]))}
    # raw, all rows and test-only
    res["raw_all"] = metrics(P0, y, K)
    res["raw_test"] = metrics(P0[te], y[te], K)
    # breakdowns (all rows, raw — argmax is data-independent)
    res["by"] = {}
    for name, mask in [("easy", diff == "easy"), ("hard", diff == "hard"), ("zh", lang == "zh"), ("en", lang == "en")]:
        if mask.any():
            res["by"][name] = metrics(P0[mask], y[mask], K)
    htypes = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("hard_type"):
            htypes[r["hard_type"]].append(i)
    res["by_hard_type"] = {k: {"n": len(v), "acc": float((P0[v].argmax(1) == y[v]).mean())} for k, v in htypes.items()}
    # calibration on cal, evaluate on test
    T = fit_temperature(Z[ca], y[ca])
    Pt = softmax(Z / T)
    res["temperature"] = T
    res["temp_test"] = metrics(Pt[te], y[te], K)
    aff = fit_affine(Z[ca], y[ca], K)
    if aff is not None:
        Pa = aff(Z)
        res["affine_test"] = metrics(Pa[te], y[te], K)
    else:
        Pa = Pt
        res["affine_test"] = res["temp_test"]
    res["argmax_changed_by_affine"] = float(np.mean(Pa[te].argmax(1) != P0[te].argmax(1)))
    res["affine_fixes_raw_errors"] = float(np.mean((Pa[te].argmax(1) == y[te]) & (P0[te].argmax(1) != y[te])))
    # per-class confusion (raw, all)
    conf_mat = np.zeros((K, K), int)
    for yi, pi in zip(y, P0.argmax(1)):
        conf_mat[yi, pi] += 1
    res["confusion"] = conf_mat.tolist()
    res["letters"] = letters
    # baselines
    rp = rules_predict(task, states, letters)
    if rp is not None:
        res["rules_acc_all"] = float((rp == y).mean()); res["rules_acc_test"] = float((rp[te] == y[te]).mean())
        res["rules_acc_easy"] = float((rp[diff == "easy"] == y[diff == "easy"]).mean())
        res["rules_acc_hard"] = float((rp[diff == "hard"] == y[diff == "hard"]).mean())
    tp = tfidf_lr([s for s, c in zip(states, ca) if c], y[ca], [s for s, t in zip(states, te) if t], seed)
    res["tfidf_acc_test"] = float((tp == y[te]).mean())
    res["tfidf_acc_test_hard"] = float((tp[diff[te] == "hard"] == y[te][diff[te] == "hard"]).mean()) if (diff[te] == "hard").any() else float("nan")
    # control arm
    ctrl = [r for r in load_jsonl(os.path.join(ACC_DIR, f"control_{task}.jsonl")) if "correct" in r]
    if ctrl:
        ids = {r["id"] for r in ctrl}
        sub = [r for r in rows if r["id"] in ids]
        res["control"] = {"n": len(ctrl), "acc": float(np.mean([r["correct"] for r in ctrl])),
                          "acc_easy": float(np.mean([r["correct"] for r in ctrl if r["difficulty"] == "easy"] or [float("nan")])),
                          "acc_hard": float(np.mean([r["correct"] for r in ctrl if r["difficulty"] == "hard"] or [float("nan")])),
                          "unparsed": int(sum(r["chosen"] is None for r in ctrl)),
                          "latency_ms_p50": float(np.median([r["latency_ms"] for r in ctrl])),
                          "typed_acc_same_rows": float(np.mean([r["correct"] for r in sub])) if sub else float("nan"),
                          "typed_latency_ms_p50_same_rows": float(np.median([r["latency_ms"] for r in sub])) if sub else float("nan")}
    # reliability figure
    os.makedirs(FIG_DIR, exist_ok=True)
    arms = [("raw", P0[te].max(1), P0[te].argmax(1) == y[te], res["raw_test"]["ece"]),
            (f"temperature T={T:.2f}", Pt[te].max(1), Pt[te].argmax(1) == y[te], res["temp_test"]["ece"]),
            ("affine (bias)", Pa[te].max(1), Pa[te].argmax(1) == y[te], res["affine_test"]["ece"])]
    reliability_fig(task, arms, os.path.join(FIG_DIR, f"reliability-{task}.png"))
    # learning curve
    if task in LC_TASKS:
        res["learning_curve"] = learning_curve(task, Z, y, states, ca, te, K, seed)
        learning_curve_fig(task, res["learning_curve"], os.path.join(FIG_DIR, f"learning-curve-{task}.png"))
    return res


def learning_curve(task, Z, y, states, ca, te, K, seed):
    cal_idx = np.where(ca)[0]
    P0 = softmax(Z)
    typed0 = float((P0[te].argmax(1) == y[te]).mean())
    out = {"N": [n for n in LC_N if n <= len(cal_idx)], "n_cal_available": int(len(cal_idx)),
           "note": "N capped at calibration-set size; the synthetic set has 200 rows/task so N>100 is not reachable here",
           "mean": collections.defaultdict(dict), "std": collections.defaultdict(dict), "raw": {}}
    for n in out["N"]:
        vals = collections.defaultdict(list)
        for s in range(3):
            rng = np.random.RandomState(seed * 100 + s)
            sub = rng.choice(cal_idx, size=n, replace=False)
            tp = tfidf_lr([states[i] for i in sub], y[sub], [states[i] for i in np.where(te)[0]], s)
            vals["tfidf"].append(float((tp == y[te]).mean()))
            vals["typed0"].append(typed0)
            aff = fit_affine(Z[sub], y[sub], K)
            Pa = aff(Z) if aff is not None else softmax(Z / fit_temperature(Z[sub], y[sub]))
            m = metrics(Pa[te], y[te], K)
            vals["typed_cal_acc"].append(m["acc"])
            vals["typed_sel90"].append(m["sel90"]["acc"])
            vals["typed_cov90"].append(m["sel90"]["coverage"])
            vals["typed_ece"].append(m["ece"])
        out["raw"][str(n)] = dict(vals)
        for k, v in vals.items():
            out["mean"][k][str(n)] = float(np.nanmean(v)); out["std"][k][str(n)] = float(np.nanstd(v))
    out["mean"] = dict(out["mean"]); out["std"] = dict(out["std"])
    return out


# ---------------------------------------------------------------- report
def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_report(all_res, path):
    L = ["# 04 — 準確率與校準（M4/M5）", "",
         "模型：Gemma 4 26B-A4B `UD-Q4_K_M`，llama-server，thinking 關閉，讀第一個 token 的選項字母 logprob。",
         "資料：`data/synthetic/*.jsonl`（合成；見 MANIFEST）。切分：依 `pair_id` 分組 50/50 為 calibration / test，孿生例不跨集。",
         "raw logprobs：`results/modal/accuracy/{task}.jsonl`。圖：`results/fig/`。原始數字：`results/analysis.json`。", "",
         "## 1. 總表（test 集，raw = 未校準）", "",
         "| task | kind | K | n | acc | macro-F1 | AUROC / top-2 | ECE raw | ECE T | ECE affine | sel@0.9 acc / cov | sel@0.95 acc / cov | missing | rules | TF-IDF+LR | JSON 生成 acc |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in all_res:
        rt = r["raw_test"]
        au = f(rt.get("auroc")) if r["K"] == 2 else f"top2 {f(rt.get('top2'))}"
        L.append(f"| {r['task']} | {r['kind']} | {r['K']} | {rt['n']} | {f(rt['acc'])} | {f(rt['macro_f1'])} | {au} | {f(rt['ece'])} | {f(r['temp_test']['ece'])} | {f(r['affine_test']['ece'])} | "
                 f"{f(rt['sel90']['acc'])} / {f(rt['sel90']['coverage'],2)} | {f(rt['sel95']['acc'])} / {f(rt['sel95']['coverage'],2)} | {f(r['missing_rate'],2)} | "
                 f"{f(r.get('rules_acc_test'))} | {f(r['tfidf_acc_test'])} | {f(r.get('control',{}).get('acc'))} |")
    L += ["", "## 2. 分項：難度與語言（全部資料，raw argmax）", "",
          "| task | easy acc | hard acc | zh acc | en acc | hard: " + " / ".join(["incomplete", "borderline", "noisy", "distractor", "inverted"]) + " |", "|---|---|---|---|---|---|"]
    for r in all_res:
        b = r["by"]; ht = r["by_hard_type"]
        hts = " / ".join(f(ht.get(k, {}).get("acc"), 2) for k in ["incomplete", "borderline", "noisy", "distractor", "inverted"])
        L.append(f"| {r['task']} | {f(b.get('easy',{}).get('acc'))} | {f(b.get('hard',{}).get('acc'))} | {f(b.get('zh',{}).get('acc'))} | {f(b.get('en',{}).get('acc'))} | {hts} |")
    L += ["", "## 3. 校準能不能救（test 集）", "",
          "| task | T | acc raw | acc affine | affine 改了多少 argmax | 修正的 raw 錯誤比例 | sel@0.9 raw acc/cov | sel@0.9 affine acc/cov | 判讀 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in all_res:
        rt, at = r["raw_test"], r["affine_test"]
        au = rt.get("auroc", rt.get("auroc_ovr", float("nan")))
        if rt["acc"] >= 0.9:
            verdict = "零樣本夠用"
        elif at["acc"] >= 0.9 or (not math.isnan(au) and au >= 0.9):
            verdict = "排序能力好、門檻問題 → 校準可救"
        elif at["acc"] - rt["acc"] >= 0.05:
            verdict = "校準有幫助但未達 0.9 → 改 criteria"
        else:
            verdict = "AUROC 差／校準無效 → 改 criteria 或換模型／fine-tune"
        L.append(f"| {r['task']} | {f(r['temperature'],2)} | {f(rt['acc'])} | {f(at['acc'])} | {f(r['argmax_changed_by_affine'],2)} | {f(r['affine_fixes_raw_errors'],2)} | "
                 f"{f(rt['sel90']['acc'])}/{f(rt['sel90']['coverage'],2)} | {f(at['sel90']['acc'])}/{f(at['sel90']['coverage'],2)} | {verdict} |")
    L += ["", "## 4. 對照組：JSON 生成（`/v1/chat/completions`，temperature 0，每 task easy/hard 各 50）", "",
          "| task | n | JSON acc | JSON easy | JSON hard | typed acc（同樣本） | JSON p50 ms | typed p50 ms（同樣本） | 無法解析 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in all_res:
        c = r.get("control")
        if c:
            L.append(f"| {r['task']} | {c['n']} | {f(c['acc'])} | {f(c['acc_easy'])} | {f(c['acc_hard'])} | {f(c['typed_acc_same_rows'])} | {f(c['latency_ms_p50'],0)} | {f(c['typed_latency_ms_p50_same_rows'],0)} | {c['unparsed']} |")
    L += ["", "## 5. 基線", "",
          "| task | 規則 all | 規則 easy | 規則 hard | TF-IDF+LR test | TF-IDF+LR hard | typed raw test | typed hard |", "|---|---|---|---|---|---|---|---|"]
    for r in all_res:
        L.append(f"| {r['task']} | {f(r.get('rules_acc_all'))} | {f(r.get('rules_acc_easy'))} | {f(r.get('rules_acc_hard'))} | {f(r['tfidf_acc_test'])} | {f(r['tfidf_acc_test_hard'])} | {f(r['raw_test']['acc'])} | {f(r['by'].get('hard',{}).get('acc'))} |")
    L += ["", "## 6. 標註量學習曲線（3 seeds；N 受限於 calibration 集大小 ≈100）", ""]
    for r in all_res:
        lc = r.get("learning_curve")
        if not lc:
            continue
        L += [f"### {r['task']}", "", f"![](fig/learning-curve-{r['task']}.png)", "",
              "| N | TF-IDF+LR | typed 零樣本 | typed 校準後 acc | typed sel@0.9 acc | coverage | ECE |", "|---|---|---|---|---|---|---|"]
        for n in lc["N"]:
            n = str(n)
            L.append(f"| {n} | {f(lc['mean']['tfidf'][n])}±{f(lc['std']['tfidf'][n],2)} | {f(lc['mean']['typed0'][n])} | {f(lc['mean']['typed_cal_acc'][n])} | {f(lc['mean']['typed_sel90'][n])} | {f(lc['mean']['typed_cov90'][n],2)} | {f(lc['mean']['typed_ece'][n])} |")
        L.append("")
    L += ["## 7. 混淆矩陣（raw，全部資料；列 = gold，欄 = 預測）", ""]
    for r in all_res:
        ls = r["letters"]
        L += [f"**{r['task']}**", "", "| gold \\ pred | " + " | ".join(ls) + " |", "|---|" + "---|" * len(ls)]
        for i, row in enumerate(r["confusion"]):
            L.append(f"| {ls[i]} {TASKS[r['task']]['options'][ls[i]]['label']} | " + " | ".join(str(v) for v in row) + " |")
        L.append("")
    L += ["## 8. Reliability diagrams", ""] + [f"![](fig/reliability-{r['task']}.png)" for r in all_res] + [""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    all_res = []
    for task in TASKS:
        r = analyze_task(task, a.seed)
        if r:
            all_res.append(r)
            rt = r["raw_test"]
            print(f"{task:18s} n={r['n_all']} acc_test={rt['acc']:.3f} ece={rt['ece']:.3f} sel90={rt['sel90']['acc']:.3f}@{rt['sel90']['coverage']:.2f} "
                  f"affine_acc={r['affine_test']['acc']:.3f} rules={r.get('rules_acc_test', float('nan')):.3f} tfidf={r['tfidf_acc_test']:.3f} ctrl={r.get('control',{}).get('acc', float('nan')):.3f}")
    json.dump(all_res, open(os.path.join(ROOT, "results/analysis.json"), "w"), ensure_ascii=False, indent=1, default=float)
    write_report(all_res, os.path.join(ROOT, "results/04-accuracy.md"))
    print("wrote results/04-accuracy.md")


if __name__ == "__main__":
    main()
