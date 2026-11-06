"""v23 (docs/handoff-v23-input-format.md): run the same six tasks in each of the five renderings with one server start.

args: "--backend llama|systemone --renderings T0,T1,J0,J1,JN --workers 1"
  llama      -> bench/accuracy.py per rendering (letter logprobs; latency_ms and prompt_tokens per row), control arm off
  systemone  -> bench/systemone_bench.py rows suite per rendering (Clef)
Outputs: <out_dir>/v23/<R>/<task>.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import accuracy, systemone_bench  # noqa: E402

TASKS = "m_alarm_severity,m_alarm_category,m_needs_dispatch,q_spc_action,p_uph_anomaly,t_tool_gate"
DATA = "/root/data/v23" if os.path.isdir("/root/data/v23") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "v23")
CRITERIA = os.path.join(os.path.dirname(DATA), "seeds", "tasks_v2.json")


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    backend = a.get("--backend", "llama")
    workers = a.get("--workers", "1")
    lim = f" --limit {a['--limit']}" if a.get("--limit") else ""  # smoke only (accuracy.py)
    sub_out = a.get("--out-sub", "v23")
    out = {}
    for r in a.get("--renderings", "T0,T1,J0,J1,JN").split(","):
        d = os.path.join(DATA, r)
        if backend == "llama":
            sub = f"--data-dir {d} --tasks {TASKS} --criteria {CRITERIA} --out-sub {sub_out}/{r} --control-n 0 --workers {workers}{lim}"
            out[r] = accuracy.main(base_url, out_dir, sub, **kw)
        else:
            sub = f"--backend systemone --suites rows --rows-dir {d} --tasks {TASKS} --criteria {CRITERIA} --out-sub {sub_out}/{r} --workers {workers}"
            out[r] = systemone_bench.main(base_url, out_dir, sub, **{k: v for k, v in kw.items() if k == "commit"})
        print(f"[v23] {r} done: {out[r]}", flush=True)
    return out
