"""T2 (handoff v5): option-order permutation averaging, after TypeLLM's `permutations`.

For each row, render the question with the options in several orders, read the letter probabilities in each order,
map them back to the original options and average. Orders share the state prefix, so with `--swa-full` and one
server slot per worker the extra forward passes are mostly question-segment prefill + decode.

args: "--tasks a,b --perms 8 --variant V0 --workers 4 --resume 1 --limit 0 --out-sub T2"
Output rows: probs_single (original order), probs_mean (permutation average), per-order probabilities,
order_sensitive (argmax differs across orders).
"""
import itertools
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import ALL_TASKS, done_ids, load  # noqa: E402
from decide.client import reader_for  # noqa: E402
from decide.prompt import VARIANTS, TemplateRenderer, letters_for  # noqa: E402


def orders_for(K, perms, key):
    total = math.factorial(K)
    ident = tuple(range(K))
    if perms == "all" or int(perms) >= total:
        return list(itertools.permutations(range(K)))
    rng = random.Random(f"{key}")
    out = [ident]
    seen = {ident}
    while len(out) < int(perms):
        o = list(range(K))
        rng.shuffle(o)
        o = tuple(o)
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    tasks = a.get("--tasks", ",".join(ALL_TASKS)).split(",")
    perms = a.get("--perms", "8")
    workers = int(a.get("--workers", 4))
    resume = a.get("--resume", "1") == "1"
    limit = int(a.get("--limit", 0))
    commit = kw.get("commit")
    if a.get("--out-sub"):
        out_dir = os.path.join(out_dir, a["--out-sub"])
    os.makedirs(out_dir, exist_ok=True)
    backend = kw.get("backend", "llama")
    read = reader_for(backend)
    tr = TemplateRenderer(base_url, static=(backend == "sglang"), **VARIANTS[a.get("--variant", "V0")])
    lock = threading.Lock()
    tls = threading.local()
    slot_counter = itertools.count()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session()
            tls.slot = next(slot_counter) % workers
        return tls.s

    summary = {}
    for task in tasks:
        rows = load(task)
        if limit:
            rows = rows[:limit]
        K = len(rows[0]["question"]["options"])
        letters = letters_for(K)
        out_path = os.path.join(out_dir, f"{task}.jsonl")
        skip = done_ids(out_path) if resume else set()
        todo = [r for r in rows if r["id"] not in skip]
        print(f"[{task}] K={K} perms={perms} {len(rows)} rows, {len(todo)} to run", flush=True)
        t0 = time.time()
        n_single = n_mean = 0

        def one(r):
            s = sess()
            opts = list(r["question"]["options"].values())
            orders = orders_for(K, perms, r["id"])
            per = []
            t_row = time.perf_counter()
            for order in orders:
                q = dict(r["question"], options={letters[i]: opts[j] for i, j in enumerate(order)})
                prompt = tr.render(r["state"], q)
                for attempt in range(3):
                    try:
                        res = read(base_url, prompt, letters, session=s, slot_id=tls.slot)
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2:
                            return {"id": r["id"], "error": repr(e)}
                        time.sleep(1)
                # probability of original option j = probability of the letter at the position i where order[i] == j
                aligned = {letters[j]: res["probs"].get(letters[i], 0.0) for i, j in enumerate(order)}
                per.append({"order": list(order), "probs": aligned, "missing": res["missing"], "first_token": res["first_token"],
                            "cache_n": res.get("server_cache_n"), "latency_ms": res["latency_ms"]})
            n = len(per)
            mean = {L: sum(p["probs"][L] for p in per) / n for L in letters}
            single = per[0]["probs"]
            argmaxes = {max(p["probs"], key=p["probs"].get) for p in per if p["probs"]}
            ch_s = max(single, key=single.get) if single else None
            ch_m = max(mean, key=mean.get)
            return {"id": r["id"], "task": task, "gold": r["gold"], "K": K, "n_orders": n,
                    "difficulty": r["difficulty"], "lang": r["lang"], "pair_id": r.get("pair_id"), "hard_type": r.get("hard_type"),
                    "chosen_single": ch_s, "correct_single": ch_s == r["gold"], "probs_single": single,
                    "chosen_mean": ch_m, "correct_mean": ch_m == r["gold"], "probs_mean": mean,
                    "order_sensitive": len(argmaxes) > 1, "n_distinct_argmax": len(argmaxes),
                    "per_order": per, "row_ms": (time.perf_counter() - t_row) * 1000}

        with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            for i, res in enumerate(ex.map(one, todo)):
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    n_single += bool(res.get("correct_single"))
                    n_mean += bool(res.get("correct_mean"))
                    if (i + 1) % 50 == 0:
                        f.flush()
                        if commit:
                            commit()
                        print(f"  [{task}] {i + 1}/{len(todo)} single {n_single / (i + 1):.3f} mean {n_mean / (i + 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        if commit:
            commit()
        summary[task] = {"n": len(todo), "acc_single": n_single / max(len(todo), 1), "acc_mean": n_mean / max(len(todo), 1), "s": round(time.time() - t0, 1)}
        print(f"[{task}] {summary[task]}", flush=True)
    json.dump(summary, open(os.path.join(out_dir, "_summary.json"), "w"), ensure_ascii=False, indent=1)
    if commit:
        commit()
    return summary
