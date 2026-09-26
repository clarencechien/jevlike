"""P3 (handoff v7): conformal confidence thresholds per task + lock file (after poorjev / jevcal).

For each task: pair_id group split (seed 0) into cal/test as in analyze.py. Confidence = temperature-scaled top
probability (T fitted on cal by NLL), or margin / entropy with --measure auto (highest cal AUROC wins). Split
conformal: nonconformity s = 1 - conf on cal; for error budget eps, q = the ceil((n+1)(1-eps))/n quantile of s;
auto-handle a row iff conf >= 1 - q. Guarantee: P(error | handled) <= eps on exchangeable data, up to sampling noise.

Outputs: results/11-thresholds.md, results/thresholds.lock.json (thresholds + evidence), results/v7.json["thresholds"].
Usage: python3 bench/thresholds.py [--acc-dir results/modal/accuracy] [--measure top_prob|margin|entropy|auto] [--eps 0.02,0.05,0.10]
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import TASKS, fit_temperature, group_split, load_jsonl, logit_matrix, softmax  # noqa: E402
from analyze_ladder import auroc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRONG = ["m_alarm_category", "m_needs_dispatch", "q_defect_root", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]


def confidence(P, measure):
    if measure == "top_prob":
        return P.max(1)
    if measure == "margin":
        s = np.sort(P, 1)
        return s[:, -1] - s[:, -2]
    if measure == "entropy":
        H = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
        return 1 - H / math.log(P.shape[1])
    raise ValueError(measure)


def conformal_q(scores_cal, eps):
    n = len(scores_cal)
    k = min(n, int(math.ceil((n + 1) * (1 - eps))))
    return float(np.sort(scores_cal)[k - 1])


def run(acc_dir, measure, eps_list, seed=0):
    lock = {"acc_dir": os.path.relpath(acc_dir, ROOT), "seed": seed, "measure_requested": measure, "eps": eps_list,
            "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), "tasks": {}}
    rows_out = []
    for task in TASKS:
        rows = [r for r in load_jsonl(os.path.join(acc_dir, f"{task}.jsonl")) if "raw_logprobs" in r]
        if not rows:
            continue
        letters = list(TASKS[task]["options"])
        Z = logit_matrix(rows, letters)
        y = np.array([letters.index(r["gold"]) for r in rows])
        ca = group_split(rows, seed); te = ~ca
        T = fit_temperature(Z[ca], y[ca])
        P = softmax(Z / T)
        correct = P.argmax(1) == y
        if measure == "auto":
            cands = {m: auroc(correct[ca].astype(int), confidence(P[ca], m)) for m in ("top_prob", "margin", "entropy")}
            cands = {k: (v if not math.isnan(v) else -1) for k, v in cands.items()}
            m_use = max(cands, key=cands.get)
        else:
            m_use = measure
        conf = confidence(P, m_use)
        entry = {"T": T, "measure": m_use, "n_cal": int(ca.sum()), "n_test": int(te.sum()), "acc_test": float(correct[te].mean()), "eps": {}}
        for eps in eps_list:
            q = conformal_q(1 - conf[ca], eps)
            thr = 1 - q
            h = conf[te] >= thr
            err = float((~correct[te][h]).mean()) if h.any() else 0.0
            cov = float(h.mean())
            entry["eps"][str(eps)] = {"threshold": thr, "coverage_cal": float((conf[ca] >= thr).mean()), "coverage_test": cov, "error_test": err,
                                      "n_handled_test": int(h.sum()), "guarantee_holds": err <= eps}
            rows_out.append((task, eps, thr, cov, err, int(h.sum()), err <= eps))
        lock["tasks"][task] = entry
    return lock, rows_out


def write_report(lock, rows_out, path):
    eps_list = lock["eps"]
    L = ["# 11 — conformal 門檻：給錯誤預算，得自動處理比例", "",
         f"`python3 bench/thresholds.py`。結果目錄 `{lock['acc_dir']}`，pair_id 分組 seed {lock['seed']}，cal/test 各半。信心 = 溫度校準後的 {lock['measure_requested']}"
         "（auto 時每 task 取 cal 上 AUROC 最高者）。split conformal：cal 上 s = 1 − conf，q = ⌈(n+1)(1−ε)⌉/n 分位，test 上 conf ≥ 1−q 才自動處理。門檻見 `docs/handoff-v7-borrowed.md` §P3。", ""]
    for eps in eps_list:
        L += [f"## ε = {eps:.0%}", "", "| task | 信心量測 | T | test acc | 門檻 | coverage（自動處理比例） | 實際錯誤率 | 處理筆數 | 保證成立 |", "|---|---|---|---|---|---|---|---|---|"]
        for task, e, thr, cov, err, nh, ok in rows_out:
            if e != eps:
                continue
            t = lock["tasks"][task]
            L.append(f"| {task} | {t['measure']} | {t['T']:.2f} | {t['acc_test']:.3f} | {thr:.4f} | {cov:.2f} | {err:.3f} | {nh} | {'✓' if ok else '✗'} |")
        n_ok = sum(1 for r in rows_out if r[1] == eps and r[6]); n_tot = sum(1 for r in rows_out if r[1] == eps)
        cov_strong = [r[3] for r in rows_out if r[1] == eps and r[0] in STRONG]
        L += ["", f"保證成立 {n_ok}/{n_tot}；七類強題 coverage 最低 {min(cov_strong):.2f}、平均 {np.mean(cov_strong):.2f}。", ""]
    # verdict per pre-registration
    ok5 = sum(1 for r in rows_out if r[1] == 0.05 and r[6]); ok2 = sum(1 for r in rows_out if r[1] == 0.02 and r[6])
    cov5 = [r[3] for r in rows_out if r[1] == 0.05 and r[0] in STRONG]
    passed = ok5 >= 8 and ok2 >= 6
    useful = min(cov5) >= 0.95 if cov5 else False
    v = ("**過**" if passed else "**沒過**") + f"（ε=5% 保證成立 {ok5}/10，門檻 8；ε=2% {ok2}/10，門檻 6）；強題 ε=5% coverage 最低 {min(cov5):.2f}（門檻 0.95：{'✓' if useful else '✗'}）。"
    if passed:
        v += " Q6 改寫成 conformal 形式；lock 檔 `results/thresholds.lock.json` 給 `bench/verify.py --check-lock` 用。"
    else:
        v += " 保留 sel@0.9；100 筆校準不足以給保證，真實資料要 300 筆以上。"
    L += ["**判定**：" + v, "",
          "與 Q6 的差別：sel@0.9 是「信心 ≥ 0.9 時的準確率與 coverage」（描述）；這裡是「先定錯誤預算，反推門檻」（保證，在可交換資料上成立，100 筆的抽樣誤差約 ±2 個百分點）。",
          "門檻是在合成資料上算的，真實資料要用同一支程式重算；lock 檔記錄 commit 與結果目錄，換模型檔或後端時 `--check-lock` 會重測。"]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return v


def main():
    a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    acc_dir = os.path.join(ROOT, a.get("--acc-dir", "results/modal/accuracy"))
    measure = a.get("--measure", "top_prob")
    eps_list = [float(x) for x in a.get("--eps", "0.02,0.05,0.10").split(",")]
    lock, rows_out = run(acc_dir, measure, eps_list)
    json.dump(lock, open(os.path.join(ROOT, "results/thresholds.lock.json"), "w"), ensure_ascii=False, indent=1, default=float)
    v = write_report(lock, rows_out, os.path.join(ROOT, "results/11-thresholds.md"))
    v7p = os.path.join(ROOT, "results/v7.json")
    v7 = json.load(open(v7p)) if os.path.exists(v7p) else {}
    v7["thresholds"] = {"verdict": v, "lock": lock}
    json.dump(v7, open(v7p, "w"), ensure_ascii=False, indent=1, default=float)
    print(v)


if __name__ == "__main__":
    main()
