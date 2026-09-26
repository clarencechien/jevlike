"""E1 (handoff v6): dump llama-server's tokenization of every D0 prompt so SGLang can be fed the same ids.

Runs against llama-server (GGUF tokenizer). For each row: prompt = TemplateRenderer(V0).render(...), ids from
POST /tokenize with add_special=true (what /completion does with a string prompt). Output one jsonl per task in
out_dir (default /results/tokenized), later copied into data/tokenized/ and mounted into the SGLang container.

args: "--tasks a,b --limit 0"
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import ALL_TASKS, load  # noqa: E402
from decide.prompt import TemplateRenderer  # noqa: E402


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    tasks = a.get("--tasks", ",".join(ALL_TASKS)).split(",")
    limit = int(a.get("--limit", 0))
    os.makedirs(out_dir, exist_ok=True)
    tr = TemplateRenderer(base_url)
    s = requests.Session()
    summary = {}
    for task in tasks:
        rows = load(task)
        if limit:
            rows = rows[:limit]
        t0 = time.time()
        n_tok = []
        with open(os.path.join(out_dir, f"{task}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                prompt = tr.render(r["state"], r["question"])
                res = s.post(f"{base_url}/tokenize", json={"content": prompt, "add_special": True, "with_pieces": False}, timeout=60).json()
                ids = res["tokens"]
                n_tok.append(len(ids))
                f.write(json.dumps({"id": r["id"], "task": task, "ids": ids, "n": len(ids), "prompt": prompt}, ensure_ascii=False) + "\n")
        summary[task] = {"n": len(rows), "tokens_mean": sum(n_tok) / len(n_tok), "s": round(time.time() - t0, 1)}
        print(f"[{task}] {summary[task]}", flush=True)
    json.dump({"template": tr.template, "summary": summary}, open(os.path.join(out_dir, "_summary.json"), "w"), ensure_ascii=False, indent=1)
    if kw.get("commit"):
        kw["commit"]()
    return summary
