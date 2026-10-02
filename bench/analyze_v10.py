"""v10 (docs/handoff-v10-clef.md): Clef / Clef-flash vs our Gemma 26B on the same synthetic sets.

Usage: python3 bench/analyze_v10.py [--equiv-only]
Reads results/modal/clef/<arm>/ (bench/clef_run.py outputs) and our existing results under results/modal/.
Writes results/v10.json; the markdown tables in results/15-clef.md are built from it.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CLEF = os.path.join(ROOT, "results/modal/clef")
MOD = os.path.join(ROOT, "results/modal")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


# ------------------------------------------------------------------ §3 equivalence
def equiv(ref_rows, rows):
    ref = {r["id"]: r for r in ref_rows}
    d, agree, n = [], 0, 0
    for r in rows:
        if r["id"] not in ref:
            continue
        a, b = ref[r["id"]]["probs"], r["probs"]
        d.append(max(abs(a[k] - b[k]) for k in a)); agree += ref[r["id"]]["chosen"] == r["chosen"]; n += 1
    d = np.array(d)
    ok = bool(n and d.max() <= 0.01 and agree >= n - 1)
    return {"n": n, "max_abs": round(float(d.max()), 4) if n else None, "p50_abs": round(float(np.median(d)), 4) if n else None,
            "argmax_agree": agree, "acc": round(sum(r["correct"] for r in rows) / max(len(rows), 1), 3), "equivalent": ok}


def section_equiv():
    out = {}
    pairs = [("clef-flash-bf16", "clef-flash-fp8"), ("clef-flash-bf16", "clef-flash-nf4"), ("clef-bf16", "clef-fp8")]
    for ref, arm in pairs:
        a, b = jl(os.path.join(CLEF, ref, "smoke.jsonl")), jl(os.path.join(CLEF, arm, "smoke.jsonl"))
        if a and b:
            out[f"{arm} vs {ref}"] = equiv(a, b)
    return out


# ------------------------------------------------------------------ helpers shared with the main sections
def group_split(rows, seed=0):
    from bench.analyze import group_split as gs
    return gs(rows, seed)


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans."""
    b01 = sum((not x) and y for x, y in zip(a, b)); b10 = sum(x and (not y) for x, y in zip(a, b))
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)


if __name__ == "__main__":
    res = {"equiv": section_equiv()}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if "--equiv-only" not in sys.argv:
        json.dump(res, open(os.path.join(ROOT, "results/v10.json"), "w"), ensure_ascii=False, indent=1)
