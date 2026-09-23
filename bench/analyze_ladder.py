"""v3 ladder analysis: model ladder (E2B / E4B / 26B) x dataset ladder (D0 / D1-typo / D1-cue / D1-mix / D2).

Inputs (raw logprobs from bench/accuracy.py):
  26b : results/modal/accuracy/{task}.jsonl (D0), results/modal/accuracy/{dataset}/{task}.jsonl
  e4b, e2b : results/modal/accuracy/{model}/{dataset}/{task}.jsonl   (dataset incl. D0)
  D2 labels: data/blind/D2/labels_gemini.jsonl, labels_fable.jsonl, adjudicated.jsonl (optional)
Outputs: results/06-ladder.md, results/ladder.json, results/fig/cue-removal-curve.png, results/fig/ladder-*.png
"""
import collections
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import C, TASKS, ece, fit_affine, group_split, logit_matrix, metrics, softmax  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(ROOT, "results/modal/accuracy")
FIG = os.path.join(ROOT, "results/fig")
MODELS = ["e2b", "e4b", "26b"]
DATASETS = ["D0", "D1-typo", "D1-cue", "D1-cue≥1", "D1-mix", "D2"]
NEW_SETS = ["D1-cue≥1", "D2"]
ZH = {"e2b": "E2B", "e4b": "E4B", "26b": "26B-A4B"}
SEED = 0


def load(model, dataset, task):
    if model == "26b":
        p = os.path.join(ACC, f"{task}.jsonl") if dataset == "D0" else os.path.join(ACC, dataset, f"{task}.jsonl")
    else:
        p = os.path.join(ACC, model, dataset, f"{task}.jsonl")
    if not os.path.exists(p):
        return {}
    rows = {}
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if "correct" in r:
            rows[r["id"]] = r
    return rows


def d0_test_ids(task):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8")]
    is_cal = group_split(rows, SEED)
    return {r["id"] for r, c in zip(rows, is_cal) if not c}, {r["id"] for r, c in zip(rows, is_cal) if c}


def boot_ci(correct, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    c = np.asarray(correct, dtype=float)
    if len(c) == 0:
        return (float("nan"), float("nan"))
    accs = [c[rng.randint(0, len(c), len(c))].mean() for _ in range(n)]
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def mcnemar(a_correct, b_correct):
    """Exact binomial McNemar on discordant pairs. Returns (n01, n10, p)."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int(np.sum(~a & b))  # a wrong, b right
    n10 = int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return n01, n10, min(1.0, p)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def conf_of(rows, ids):
    return np.array([max(rows[i]["probs"].values()) if rows[i]["probs"] else 0.0 for i in ids])


def compare(task, dataset, ids, R):
    """Per-model acc/CI on the common ids, McNemar E4B vs 26B and E2B vs 26B."""
    out = {"n": len(ids)}
    for m in MODELS:
        if m in R:
            c = [R[m][i]["correct"] for i in ids]
            out[m] = {"acc": float(np.mean(c)), "ci": boot_ci(c), "n": len(c),
                      "err_auroc": auroc(c, conf_of(R[m], ids)), "conf_mean": float(conf_of(R[m], ids).mean())}
            if dataset != "D2":
                hard = [i for i in ids if R[m][i].get("difficulty") == "hard"]
                out[m]["acc_hard"] = float(np.mean([R[m][i]["correct"] for i in hard])) if hard else float("nan")
    if "e4b" in R and "26b" in R:
        a = [R["e4b"][i]["correct"] for i in ids]
        b = [R["26b"][i]["correct"] for i in ids]
        n01, n10, p = mcnemar(a, b)
        gap = out["26b"]["acc"] - out["e4b"]["acc"]
        ci_overlap = not (out["e4b"]["ci"][1] < out["26b"]["ci"][0] or out["26b"]["ci"][1] < out["e4b"]["ci"][0])
        status = "明顯掉" if (gap >= 0.05 and p < 0.05) else ("相當" if (ci_overlap and p >= 0.05) else "介於")
        both = sum(x and y for x, y in zip(a, b)); neither = sum((not x) and (not y) for x, y in zip(a, b))
        conf26 = [max(R["26b"][i]["probs"].values()) for i, x, y in zip(ids, a, b) if (not x) and y and R["26b"][i]["probs"]]
        out["e4b_vs_26b"] = {"gap": gap, "n_e4b_wrong_26b_right": n01, "n_e4b_right_26b_wrong": n10, "both_right": both, "both_wrong": neither,
                             "mcnemar_p": p, "ci_overlap": ci_overlap, "status": status,
                             "conf26_on_e4b_errors": {"n": len(conf26), "median": float(np.median(conf26)) if conf26 else float("nan"),
                                                      "frac_ge_0.99": float(np.mean([c >= 0.99 for c in conf26])) if conf26 else float("nan")}}
    if "e2b" in R and "26b" in R:
        a = [R["e2b"][i]["correct"] for i in ids]; b = [R["26b"][i]["correct"] for i in ids]
        n01, n10, p = mcnemar(a, b)
        out["e2b_vs_26b"] = {"gap": out["26b"]["acc"] - out["e2b"]["acc"], "mcnemar_p": p}
    return out


def verdict(task_res):
    """Apply the pre-registered table. task_res[dataset] = compare() output."""
    d0 = task_res.get("D0", {}).get("e4b_vs_26b", {}).get("status")
    e2b_d0 = task_res.get("D0", {}).get("e2b", {}).get("acc")
    new = {ds: task_res.get(ds, {}).get("e4b_vs_26b", {}).get("status") for ds in NEW_SETS if ds in task_res}
    if e2b_d0 is not None and e2b_d0 >= 0.95:
        return "H1（E2B 在 D0 已 ≥ 0.95）", d0, new
    if d0 is None:
        return "資料不足", d0, new
    new_drop = any(v == "明顯掉" for v in new.values())
    new_same = new and all(v == "相當" for v in new.values())
    if d0 == "相當" and new_same:
        return "H1 強版：兩級模型在新集也分不出", d0, new
    if d0 == "相當" and new_drop:
        return "H1（合成集飽和）＋ 新集有鑑別力", d0, new
    if d0 == "明顯掉" and (new_drop or not new):
        return "H2：現有難度已能分出等級", d0, new
    if d0 == "明顯掉" and new_same:
        return "不一致：先懷疑新集", d0, new
    return "介於（CI 重疊但 p<0.05 或差距 <5pt）", d0, new


def kappa(a, b):
    """Cohen's kappa for two label lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def d2_agreement():
    base = os.path.join(ROOT, "data/blind/D2")
    out = {}
    if not os.path.exists(os.path.join(base, "labels_gemini.jsonl")) or not os.path.exists(os.path.join(base, "labels_fable.jsonl")):
        return out
    g = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(os.path.join(base, "labels_gemini.jsonl"), encoding="utf-8")}
    f = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(os.path.join(base, "labels_fable.jsonl"), encoding="utf-8")}
    for task in TASKS:
        ids = [i for i in g if i in f and i.startswith(task + "-")]
        a, b = [g[i] for i in ids], [f[i] for i in ids]
        out[task] = {"n": len(ids), "agree": float(np.mean([x == y for x, y in zip(a, b)])) if ids else float("nan"), "kappa": kappa(a, b)}
    return out


def cue_curve(task_ids_rows):
    """task_ids_rows: list of (task, ids, R, cue_rows) -> acc by removed-cue bin per model."""
    bins = collections.OrderedDict([("0", (0, 0)), ("1", (1, 1)), ("2", (2, 2)), ("3+", (3, 99))])
    agg = {m: {b: [] for b in bins} for m in MODELS}
    for task, ids, R, cue in task_ids_rows:
        for i in ids:
            k = cue[i].get("n_cues_removed", 0)
            b = next(name for name, (lo, hi) in bins.items() if lo <= k <= hi)
            for m in MODELS:
                if m in R and i in R[m]:
                    agg[m][b].append(R[m][i]["correct"])
    return {m: {b: (float(np.mean(v)) if v else float("nan"), len(v)) for b, v in d.items()} for m, d in agg.items()}


def cue_fig(curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=C["surface"]); ax.set_facecolor(C["surface"])
    cols = {"e2b": C["s3"], "e4b": C["s2"], "26b": C["s1"]}
    for m in MODELS:
        xs = list(curve[m]); ys = [curve[m][b][0] for b in xs]
        if all(math.isnan(y) for y in ys):
            continue
        ax.plot(range(len(xs)), ys, marker="o", ms=6, lw=2, color=cols[m], label=ZH[m])
        for x, (y, n) in zip(range(len(xs)), [curve[m][b] for b in xs]):
            if not math.isnan(y):
                ax.text(x, y + 0.015, f"n={n}", fontsize=7, color=C["sec"], ha="center")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("移除的 criteria 線索數（D1-cue）", color=C["sec"]); ax.set_ylabel("accuracy", color=C["sec"])
    ax.set_title("線索依賴曲線：移除與 criteria 重疊的字後的準確率", fontsize=11, color=C["ink"])
    ax.grid(axis="y", color=C["grid"], lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def ladder_fig(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(results)
    dss = [d for d in DATASETS if any(d in results[t] for t in tasks)]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 4.6), facecolor=C["surface"], sharey=True)
    if len(dss) == 1:
        axes = [axes]
    cols = {"e2b": C["s3"], "e4b": C["s2"], "26b": C["s1"]}
    for ax, ds in zip(axes, dss):
        ax.set_facecolor(C["surface"])
        for j, m in enumerate(MODELS):
            ys = [results[t].get(ds, {}).get(m, {}).get("acc", float("nan")) for t in tasks]
            ax.barh(np.arange(len(tasks)) + (j - 1) * 0.27, ys, height=0.25, color=cols[m], label=ZH[m])
        ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks, fontsize=8); ax.invert_yaxis()
        ax.set_xlim(0.3, 1.0); ax.set_title(ds, fontsize=10, color=C["ink"]); ax.grid(axis="x", color=C["grid"], lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("模型階梯 × 資料階梯：accuracy", fontsize=11, color=C["ink"]); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def main():
    results, cue_inputs, calib = {}, [], {}
    for task in TASKS:
        results[task] = {}
        test_ids, cal_ids = d0_test_ids(task)
        for ds in DATASETS:
            if ds == "D1-cue≥1":
                continue
            R = {m: load(m, ds, task) for m in MODELS}
            R = {m: r for m, r in R.items() if r}
            if not R:
                continue
            common = set.intersection(*[set(r) for r in R.values()])
            if ds == "D0":
                common &= test_ids
            elif ds.startswith("D1"):
                common = {i for i in common if i.rsplit("-D1", 1)[0] in test_ids}
            ids = sorted(common)
            if not ids:
                continue
            results[task][ds] = compare(task, ds, ids, R)
            if ds == "D1-cue" and "26b" in R:
                cue_inputs.append((task, ids, R, R["26b"]))
                sub = [i for i in ids if R["26b"][i].get("n_cues_removed", 0) >= 1]
                if len(sub) >= 10:
                    results[task]["D1-cue≥1"] = compare(task, "D1-cue≥1", sub, R)
        # calibration transfer 26b: fit on D0 cal, apply to D2
        R0, R2 = load("26b", "D0", task), load("26b", "D2", task)
        if R0 and R2:
            letters = list(TASKS[task]["options"]); K = len(letters)
            cal_rows = [R0[i] for i in sorted(cal_ids) if i in R0]
            Zc = logit_matrix(cal_rows, letters); yc = np.array([letters.index(r["gold"]) for r in cal_rows])
            d2_rows = [R2[i] for i in sorted(R2)]
            Z2 = logit_matrix(d2_rows, letters); y2 = np.array([letters.index(r["gold"]) for r in d2_rows])
            aff = fit_affine(Zc, yc, K)
            m_raw = metrics(softmax(Z2), y2, K)
            m_tr = metrics(aff(Z2), y2, K) if aff else m_raw
            # refit on half of D2
            rng = random.Random(1); idx = list(range(len(d2_rows))); rng.shuffle(idx)
            h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
            aff2 = fit_affine(Z2[h1], y2[h1], K)
            m_re = metrics(aff2(Z2[h2]), y2[h2], K) if aff2 else None
            calib[task] = {"raw": m_raw, "transfer": m_tr, "refit_half": m_re}
    verdicts = {t: verdict(results[t]) for t in results if results[t]}
    kap = d2_agreement()
    curve = cue_curve(cue_inputs) if cue_inputs else None
    os.makedirs(FIG, exist_ok=True)
    if curve:
        cue_fig(curve, os.path.join(FIG, "cue-removal-curve.png"))
    if any(results.values()):
        ladder_fig({t: r for t, r in results.items() if r}, os.path.join(FIG, "ladder-accuracy.png"))
    json.dump({"results": results, "verdicts": verdicts, "kappa": kap, "cue_curve": curve, "calibration_transfer": calib},
              open(os.path.join(ROOT, "results/ladder.json"), "w"), ensure_ascii=False, indent=1, default=float)

    # ------------------------------------------------------------ report
    L = ["# 06 — 題目問題還是模型能力問題？（E2B / E4B / 26B × D0 / D1 / D2）", "",
         "判讀規則預先登記於 `06-ladder-prereg.md`。所有比較都在同一批題目、同一個 prompt、`--swa-full` 下進行；D0 只算 test 集（依 pair_id 分組切半，seed 0），D1 是 D0 test 集的擾動版，D2 是 Gemini 盲寫、雙標註一致的題。原始：`results/ladder.json`。", "",
         "## 1. 每個 task 的判定", "",
         "| task | D0：E4B vs 26B | D1-cue | D2 | E2B@D0 | 判定 |", "|---|---|---|---|---|---|"]
    for t, (v, d0, new) in verdicts.items():
        e2 = results[t].get("D0", {}).get("e2b", {}).get("acc")
        L.append(f"| {t} | {d0 or '—'} | {new.get('D1-cue', '—')} | {new.get('D2', '—')} | {f(e2)} | **{v}** |")
    L += ["", "## 2. 主表：accuracy（95% bootstrap CI）", ""]
    for ds in DATASETS:
        if not any(ds in results[t] for t in results):
            continue
        L += [f"### {ds}", "", "| task | n | E2B | E4B | 26B | 26B−E4B | McNemar p（E4B vs 26B） | E4B錯/26B對 | E4B對/26B錯 | hard: E4B / 26B |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in results:
            r = results[t].get(ds)
            if not r:
                continue
            cell = lambda m: f"{f(r[m]['acc'])} [{f(r[m]['ci'][0], 2)}–{f(r[m]['ci'][1], 2)}]" if m in r else "—"
            cmp_ = r.get("e4b_vs_26b", {})
            L.append(f"| {t} | {r['n']} | {cell('e2b')} | {cell('e4b')} | {cell('26b')} | {f(cmp_.get('gap'))} | {f(cmp_.get('mcnemar_p'))} | {cmp_.get('n_e4b_wrong_26b_right', '—')} | {cmp_.get('n_e4b_right_26b_wrong', '—')} | "
                     f"{f(r.get('e4b', {}).get('acc_hard'))} / {f(r.get('26b', {}).get('acc_hard'))} |")
        L.append("")
    L += ["## 3. 錯誤偵測 AUROC（raw confidence 預測「這題會不會對」；0.5 = 信心與對錯無關）", "",
          "| task | " + " | ".join(f"{ds} E2B/E4B/26B" for ds in DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
    for t in results:
        cells = []
        for ds in DATASETS:
            r = results[t].get(ds, {})
            cells.append(" / ".join(f(r.get(m, {}).get("err_auroc"), 2) for m in MODELS))
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "## 4. 26B 在「E4B 錯、26B 對」題目上的信心", "",
          "| task | 資料集 | n | 中位數 conf | ≥0.99 的比例 |", "|---|---|---|---|---|"]
    for t in results:
        for ds in DATASETS:
            c = results[t].get(ds, {}).get("e4b_vs_26b", {}).get("conf26_on_e4b_errors")
            if c and c["n"]:
                L.append(f"| {t} | {ds} | {c['n']} | {f(c['median'], 3)} | {f(c['frac_ge_0.99'], 2)} |")
    if curve:
        L += ["", "## 5. 線索依賴曲線（D1-cue）", "", "![](fig/cue-removal-curve.png)", "",
              "| 移除線索數 | " + " | ".join(f"{ZH[m]} acc (n)" for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
        for b in curve[MODELS[0]]:
            L.append(f"| {b} | " + " | ".join(f"{f(curve[m][b][0])} ({curve[m][b][1]})" for m in MODELS) + " |")
    if kap:
        L += ["", "## 6. D2 雙標註一致性（Gemini vs Claude，各自拿完整 criteria 獨立標）", "",
              "| task | n | 一致率 | Cohen's κ |", "|---|---|---|---|"]
        for t, k in kap.items():
            L.append(f"| {t} | {k['n']} | {f(k['agree'])} | {f(k['kappa'])} |")
        L.append(f"| **全部** | {sum(k['n'] for k in kap.values())} | {f(np.mean([k['agree'] for k in kap.values()]))} | {f(np.mean([k['kappa'] for k in kap.values() if not math.isnan(k['kappa'])]))} |")
    if calib:
        L += ["", "## 7. 校準遷移（26B：D0 calibration 集擬合的 affine，套到 D2）", "",
              "| task | D2 raw acc / ECE / sel@0.9 acc,cov | D0→D2 遷移 ECE / sel@0.9 acc,cov | D2 內重擬（半數）ECE / sel@0.9 acc,cov |", "|---|---|---|---|"]
        for t, c in calib.items():
            s = lambda m: f"{f(m['ece'])} / {f(m['sel90']['acc'])},{f(m['sel90']['coverage'], 2)}" if m else "—"
            L.append(f"| {t} | {f(c['raw']['acc'])} / {s(c['raw'])} | {s(c['transfer'])} | {s(c['refit_half'])} |")
    L += ["", "![](fig/ladder-accuracy.png)", ""]
    open(os.path.join(ROOT, "results/06-ladder.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote results/06-ladder.md")
    for t, v in verdicts.items():
        print(f"{t:18s} {v[0]}  D0={v[1]} new={v[2]}")


if __name__ == "__main__":
    main()
