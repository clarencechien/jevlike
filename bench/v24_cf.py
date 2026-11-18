"""v24 E1 (docs/handoff-v24-calibration-abstain.md §1): content-free priors per task x rotation (Calibrate Before Use).

Same questions and rotation scheme as bench/v19.py (mode l0); the state is replaced by three content-free inputs.
Also re-reads rotation 0 for the first 5 rows per task so the analysis can check drift against results/modal/v19/l0.

args: "--tasks a,b --workers 4"
Out: <out_dir>/cf.jsonl (task, input, r, pos_probs, pos_logp), <out_dir>/drift.jsonl (id, aligned_probs at r=0)
"""
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import ALL_TASKS, load  # noqa: E402
from bench.v19 import _read  # noqa: E402
from decide.prompt import TemplateRenderer, letters_for  # noqa: E402

CONTENT_FREE = ["N/A", "（無）", ""]


def rotated(question, K, rr, letters):
    opts = list(question["options"].values())
    order = [(i - rr) % K for i in range(K)]  # identical to v19: position i holds option order[i]
    return dict(question, options={letters[i]: opts[order[i]] for i in range(K)})


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    tasks = a.get("--tasks", ",".join(ALL_TASKS)).split(",")
    os.makedirs(out_dir, exist_ok=True)
    tr = TemplateRenderer(base_url)
    s = requests.Session()
    n = 0
    with open(os.path.join(out_dir, "cf.jsonl"), "w", encoding="utf-8") as f, open(os.path.join(out_dir, "drift.jsonl"), "w", encoding="utf-8") as g:
        for task in tasks:
            rows = load(task)
            q0 = rows[0]["question"]
            K = len(q0["options"]); letters = letters_for(K)
            for inp in CONTENT_FREE:
                for rr in range(K):
                    res = _read(base_url, tr.render(inp, rotated(q0, K, rr, letters)), letters, s, 0)
                    f.write(json.dumps({"task": task, "input": inp, "r": rr, "K": K, "pos_probs": res["probs"],
                                        "pos_logp": {L: res["raw_logprobs"].get(L) for L in letters}, "missing": res["missing"],
                                        "option_mass": res.get("option_mass")}, ensure_ascii=False) + "\n")
                    n += 1
            for r in rows[:5]:
                assert r["question"] == q0, "question differs within a task; content-free prior would not apply"
                res = _read(base_url, tr.render(r["state"], rotated(r["question"], K, 0, letters)), letters, s, 0)
                g.write(json.dumps({"id": r["id"], "task": task, "pos_probs": res["probs"]}, ensure_ascii=False) + "\n")
            print(f"[{task}] K={K} content-free reads done ({n} total)", flush=True)
    if kw.get("commit"):
        kw["commit"]()
    return {"reads": n}
