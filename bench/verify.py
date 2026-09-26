"""T0-b (handoff v5): recompute accuracy metrics from raw logprob rows without any model call.

Modelled on TypeLLM's evals/jevbench/verify.py: anyone can rerun this on results/modal/accuracy/**/*.jsonl and
check the numbers in 04-accuracy.md / analysis.json. Writes results/verify.md.

Checks per file: n, accuracy (argmax of `probs` == gold), Brier (multi-class), ECE (10 bins, top-label), probability
sum within 1 ± 1e-3, missing-letter rate. For the v2 D0 files it also recomputes the test-split accuracy (pair_id group
split, seed 0) and compares it with analysis.json raw_test.acc.
"""
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(ROOT, "results/modal/accuracy")


def ece(conf, correct, bins=10):
    n = len(conf)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if (conf[i] > lo or (b == 0 and conf[i] >= lo)) and conf[i] <= hi]
        if idx:
            e += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx))
    return e


def group_split_test_ids(rows, seed=0):
    groups = sorted({r.get("pair_id") or r["id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    return {r["id"] for r in rows if (r.get("pair_id") or r["id"]) not in cal}


def verify_file(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if "gold" in r and "probs" in r]
    if not rows:
        return None
    n = len(rows)
    correct, conf, brier, bad_sum, missing = [], [], 0.0, 0, 0
    for r in rows:
        p = r["probs"]
        letters = sorted(set(p) | set(r.get("missing") or []))
        chosen = max(p, key=p.get) if p else None
        c = chosen == r["gold"]
        correct.append(float(c))
        conf.append(p.get(chosen, 0.0) if chosen else 0.0)
        s = sum(p.values())
        if abs(s - 1.0) > 1e-3:
            bad_sum += 1
        brier += sum((p.get(L, 0.0) - (1.0 if L == r["gold"] else 0.0)) ** 2 for L in letters)
        missing += bool(r.get("missing"))
        if "correct" in r and bool(r["correct"]) != c:
            raise AssertionError(f"{path}: stored `correct` disagrees with argmax for {r['id']}")
    return {"n": n, "acc": sum(correct) / n, "brier": brier / n, "ece": ece(conf, correct), "bad_sum": bad_sum,
            "missing_rate": missing / n, "conf_mean": sum(conf) / n, "rows": rows}


def main():
    files = sorted(p for p in glob.glob(os.path.join(ACC, "**/*.jsonl"), recursive=True) if "control_" not in os.path.basename(p))
    analysis = {r["task"]: r for r in json.load(open(os.path.join(ROOT, "results/analysis.json")))}
    L = ["# verify — 從原始 logprob 重算（不呼叫模型）", "",
         "`python3 bench/verify.py`。每列一個結果檔：argmax 準確率、Brier、ECE(10 bins)、機率和超出 1±1e-3 的筆數、缺字母率。",
         "v2 D0 檔另重算 test split（pair_id 分組、seed 0）並與 `analysis.json` raw_test.acc 比對。", "",
         "| 檔 | n | acc | Brier | ECE | 機率和異常 | 缺字母率 | 平均信心 | test acc（重算） | analysis.json | 一致 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    n_files = n_mismatch = 0
    for p in files:
        v = verify_file(p)
        if v is None:
            continue
        n_files += 1
        rel = os.path.relpath(p, ACC)
        task = os.path.basename(p)[:-6]
        recomputed = ref = agree = "—"
        if os.path.dirname(rel) == "" and task in analysis:
            data_path = os.path.join(ROOT, f"data/synthetic/{task}.jsonl")
            if os.path.exists(data_path):
                test_ids = group_split_test_ids([json.loads(l) for l in open(data_path, encoding="utf-8")])
                tr = [r for r in v["rows"] if r["id"] in test_ids]
                acc_t = sum(max(r["probs"], key=r["probs"].get) == r["gold"] for r in tr) / len(tr)
                ref_v = analysis[task]["raw_test"]["acc"]
                ok = abs(acc_t - ref_v) <= 0.001
                n_mismatch += not ok
                recomputed, ref, agree = f"{acc_t:.3f}", f"{ref_v:.3f}", "✓" if ok else "✗"
        L.append(f"| {rel} | {v['n']} | {v['acc']:.3f} | {v['brier']:.3f} | {v['ece']:.3f} | {v['bad_sum']} | {v['missing_rate']:.2f} | {v['conf_mean']:.3f} | {recomputed} | {ref} | {agree} |")
    L += ["", f"共 {n_files} 檔；與 analysis.json 不一致：{n_mismatch}。缺字母（不在 top-40）的筆，其 top-40 地板 logprob 最高 −14.1，漏掉的機率上界 7.4e-7（handoff v5 §0.2）。"]
    out = os.path.join(ROOT, "results/verify.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{n_files} files, {n_mismatch} mismatches -> {out}")
    return 1 if n_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
