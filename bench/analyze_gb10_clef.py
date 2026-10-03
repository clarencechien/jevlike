"""GB10 (docs/handoff-gb10-clef.md §7–§8): 26B vs Clef on the same rows, synthetic or real.

  python3 bench/analyze_gb10_clef.py --ours <26B accuracy dir> --clef <Clef rows dir> --tasks q_spc_action,m_alarm_severity \
      [--out results/gb10/clef/compare.md]

--ours : bench/accuracy.py output (<dir>/<task>.jsonl with raw_logprobs)
--clef : bench/systemone_bench.py output, either per-task files (suite "rows") or one d0.jsonl (suite "d0")
Same cal/test split as every other round (pair_id groups, seed 0; real rows without pair_id use their id). Prints and writes
per-task test acc, McNemar p, ECE (raw / temperature fitted on cal), and the epsilon=5% risk-controlled coverage and error.
Only aggregate numbers are written, so the output is safe to commit even for real data; the row files are not.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import analyze_v10 as A  # noqa: E402
from analyze import ece, fit_temperature, softmax  # noqa: E402
from thresholds import risk_threshold  # noqa: E402


def letters_of(rows):
    return list(rows[0]["question"]["options"]) if rows and "question" in rows[0] else None


def load_dir(path, tasks, kind):
    out = {}
    single = os.path.join(path, "d0.jsonl")
    pooled = collections.defaultdict(list)
    if kind == "clef" and os.path.exists(single):
        for r in A.jl(single):
            pooled[r["task"]].append(r)
    for t in tasks:
        rows = pooled.get(t) or A.jl(os.path.join(path, f"{t}.jsonl"))
        if not rows:
            continue
        out[t] = rows
    return out


def zmat(rows, letters, kind):
    if kind == "ours":
        return np.array([[r["raw_logprobs"].get(k) if r["raw_logprobs"].get(k) is not None else -30.0 for k in letters] for r in rows])
    return np.array([[r["logits"][k] for k in letters] for r in rows if not r.get("nan")])


def per_task(rows, letters, kind, te_ids):
    rows = sorted([r for r in rows if not r.get("nan")], key=lambda r: r["id"])
    Z = zmat(rows, letters, kind); y = np.array([letters.index(r["gold"]) for r in rows])
    te = np.array([r["id"] in te_ids for r in rows]); ca = ~te
    ok = Z.argmax(1) == y
    T = fit_temperature(Z[ca], y[ca]) if ca.any() else 1.0
    P = softmax(Z / T); P0 = softmax(Z); conf = P.max(1)
    thr, _ = risk_threshold(conf[ca], ok[ca], 0.05) if ca.any() else (float("inf"), 0)
    a5 = te & (conf >= thr)
    return {"ids": [r["id"] for r in rows], "ok": ok, "te": te, "acc": float(ok[te].mean()), "n_test": int(te.sum()),
            "ece_raw": float(ece(P0.max(1)[te], ok[te])), "ece_T": float(ece(conf[te], ok[te])),
            "cov5": float(a5.sum() / te.sum()), "err5": float((~ok[a5]).mean()) if a5.any() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--clef", required=True)
    ap.add_argument("--tasks", default=",".join(A.D0_TASKS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    ours, clef = load_dir(a.ours, tasks, "ours"), load_dir(a.clef, tasks, "clef")
    L = ["| task | n test | 26B acc | Clef acc | 差（點） | McNemar p | ECE 校準後 26B / Clef | ε=5% 自動處理 26B / Clef（錯誤率） |", "|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        if t not in ours or t not in clef:
            continue
        letters = letters_of(ours[t]) or letters_of(clef[t]) or A.LET.get(t)
        split_rows = sorted(ours[t], key=lambda r: r["id"])
        te_b = ~A.group_split(split_rows, 0)
        te_ids = {r["id"] for r, x in zip(split_rows, te_b) if x}
        o, c = per_task(ours[t], letters, "ours", te_ids), per_task(clef[t], letters, "clef", te_ids)
        okc = dict(zip(c["ids"], c["ok"])); oko = dict(zip(o["ids"], o["ok"]))
        ids = [i for i, x in zip(o["ids"], o["te"]) if x and i in okc]
        p = A.mcnemar([bool(oko[i]) for i in ids], [bool(okc[i]) for i in ids])
        L.append(f"| {t} | {o['n_test']} | {A.pct(o['acc'])} | {A.pct(c['acc'])} | {(c['acc'] - o['acc']) * 100:+.1f} | {p:.3f} | "
                 f"{A.f(o['ece_T'])} / {A.f(c['ece_T'])} | {A.pct(o['cov5'], 0)}（{A.pct(o['err5'])}）/ {A.pct(c['cov5'], 0)}（{A.pct(c['err5'])}） |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
