"""v8 (handoff v8): self-run of the JevBench public split (231 items) with the first-token letter read.

Items: /root/data/jevbench/{original,easy,hard}.jsonl (MIT, pinned revision in data/jevbench/SOURCE.md).
Mapping: choice -> one letter per label in the given order, "X. label: criteria"; noul -> labels [no, yes] with the
false/true criteria; score -> level indices with the rubric text. Every item is read twice: original order and
reversed order (order-sensitivity disclosure). Raw letter logprobs are stored so temperature can be applied offline.
Scoring happens locally in bench/analyze_v8.py with the vendored harness (third_party/jevbench).

args: "--limit 0 --workers 4 --tiers original,easy,hard"
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.client import reader_for  # noqa: E402
from decide.prompt import SYSTEM_EN, TemplateRenderer, letters_for  # noqa: E402

DATA = "/root/data/jevbench" if os.path.isdir("/root/data/jevbench") else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/jevbench")


def option_lines(task, order):
    q = task["question"]; crit = q.get("criteria"); labels = task["labels"]
    out = []
    for i, j in enumerate(order):
        lab = labels[j]
        if q["type"] == "noul":
            desc = (crit or {}).get("true" if lab == "yes" else "false", "")
            text = f"{lab}" + (f": {desc}" if desc else "")
        elif q["type"] == "score":
            desc = crit[int(lab)] if isinstance(crit, list) and int(lab) < len(crit) else ""
            text = f"level {lab}" + (f": {desc}" if desc else "")
        else:
            desc = (crit or {}).get(lab, "") if isinstance(crit, dict) else ""
            text = f"{lab}" + (f": {desc}" if desc else "")
        out.append(f"{letters_for(len(labels))[i]}. {text}")
    return out


def user_text(task, order):
    state = task["state"] if isinstance(task["state"], str) else json.dumps(task["state"], ensure_ascii=False, indent=1)
    return (f"[State]\n{state}\n\n[Question] {task['question']['instructions']}\n" + "\n".join(option_lines(task, order)) +
            "\n\nAnswer with the letter of the correct option only.")


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    limit = int(a.get("--limit", 0)); workers = int(a.get("--workers", 4))
    tiers = a.get("--tiers", "original,easy,hard").split(",")
    os.makedirs(out_dir, exist_ok=True)
    backend = kw.get("backend", "llama")
    read = reader_for(backend)
    tr = TemplateRenderer(base_url, static=(backend == "sglang"))
    token_ids = None
    if backend == "sglang":
        from decide.labels import check_labels
        token_ids = check_labels(base_url, "sglang")
    tls = threading.local()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session()
        return tls.s

    lock = threading.Lock()
    summary = {}
    for tier in tiers:
        tasks = [json.loads(l) for l in open(os.path.join(DATA, f"{tier}.jsonl"), encoding="utf-8") if l.strip()]
        if limit:
            tasks = tasks[:limit]
        out_path = os.path.join(out_dir, f"{tier}.jsonl")
        print(f"[{tier}] {len(tasks)} items", flush=True)
        t0 = time.time(); n_ok = 0

        def one(task):
            K = len(task["labels"]); letters = letters_for(K)
            res = {}
            for name, order in (("fwd", list(range(K))), ("rev", list(range(K))[::-1])):
                msgs = [{"role": "system", "content": SYSTEM_EN}, {"role": "user", "content": user_text(task, order)}]
                prompt = tr.render_messages(msgs)
                for attempt in range(3):
                    try:
                        r = read(base_url, prompt, letters, session=sess(), **({"token_ids": token_ids} if token_ids else {}))
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2:
                            return {"task_id": task["id"], "tier": tier, "error": repr(e)}
                        time.sleep(1)
                # letter at position i -> label order[i]
                res[name] = {"probs": {task["labels"][j]: r["probs"].get(letters[i], 0.0) for i, j in enumerate(order)},
                             "raw_logprobs": {task["labels"][j]: r["raw_logprobs"].get(letters[i]) for i, j in enumerate(order)},
                             "missing": [task["labels"][order[letters.index(m)]] for m in r["missing"]], "first_token": r["first_token"],
                             "latency_ms": r["latency_ms"], "prompt_tokens": r["prompt_tokens"], "option_mass": r.get("option_mass")}
            pf = res["fwd"]["probs"]; pred = max(pf, key=pf.get) if pf else None
            return {"task_id": task["id"], "tier": tier, "family": task["family"], "type": task["question"]["type"], "labels": task["labels"],
                    "expected": task["expected"], "predicted_fwd": pred, "correct_fwd": pred == str(task["expected"]), **res}

        with open(out_path, "w", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            for i, row in enumerate(ex.map(one, tasks)):
                with lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_ok += bool(row.get("correct_fwd"))
                    if (i + 1) % 25 == 0:
                        print(f"  [{tier}] {i + 1}/{len(tasks)} acc_fwd {n_ok / (i + 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        summary[tier] = {"n": len(tasks), "acc_fwd": n_ok / max(len(tasks), 1), "s": round(time.time() - t0, 1)}
        print(f"[{tier}] {summary[tier]}", flush=True)
        if kw.get("commit"):
            kw["commit"]()
    json.dump({"summary": summary, "system": SYSTEM_EN, "backend": backend, "model": kw.get("model")}, open(os.path.join(out_dir, "_summary.json"), "w"), ensure_ascii=False, indent=1)
    if kw.get("commit"):
        kw["commit"]()
    return summary
