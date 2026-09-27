"""v9 R (docs/handoff-v9-nine-places.md): rerank SOP passages by reading P(A 能回答) for each (query, passage).

Each candidate is read separately (v7 P1 showed packing several passages into one forward pass contaminates later ones).
The query sits before the passage, so the template + query prefix is shared across a query's 20 candidates:
llama-server sends them in order on one slot (cache_prompt + --swa-full); SGLang warms the first and batches the rest.

args: "--limit 0 --workers 4"
Output: <out_dir>/r_rerank.jsonl (per query: candidate pids, rel, P(A), per-query wall ms), _summary.json (nDCG@5 etc.)
"""
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import DATA_ROOT  # noqa: E402
from decide.client import batch_read_option_probs_sglang, read_option_probs, read_option_probs_sglang  # noqa: E402
from decide.prompt import TemplateRenderer  # noqa: E402

Q = {"instructions": "這段文件能直接回答上面工程師的問題嗎？", "options": {"A": "能", "B": "不能"}}


def state_for(query, text):
    return f"工程師的問題：{query}\n\n文件段落：{text}"


def ndcg_at(rels_sorted, k=5):
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels_sorted[:k]))
    ideal = sorted(rels_sorted, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg else 0.0


def metrics(rows, key="score"):
    nd, mrr, r3 = [], [], []
    for r in rows:
        order = sorted(r["candidates"], key=lambda c: -c[key])
        rels = [c["rel"] for c in order]
        nd.append(ndcg_at(rels, 5))
        rank = next(i for i, c in enumerate(order) if c["rel"] == 2) + 1
        mrr.append(1 / rank); r3.append(rank <= 3)
    n = max(len(rows), 1)
    return {"n": len(rows), "ndcg5": sum(nd) / n, "mrr": sum(mrr) / n, "recall3": sum(r3) / n}


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    limit = int(a.get("--limit", 0)); workers = int(a.get("--workers", 4))
    backend = kw.get("backend", "llama")
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(DATA_ROOT, "v9/r_rerank.jsonl"), encoding="utf-8")]
    corpus = {json.loads(l)["pid"]: json.loads(l)["text"] for l in open(os.path.join(DATA_ROOT, "v9/r_corpus.jsonl"), encoding="utf-8")}
    if limit:
        rows = rows[:limit]
    tr = TemplateRenderer(base_url, static=(backend == "sglang"))
    token_ids = None
    if backend == "sglang":
        from decide.labels import check_labels
        token_ids = check_labels(base_url, "sglang")
    tls = threading.local(); slot_counter = iter(range(10 ** 6)); lock = threading.Lock()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session(); tls.slot = next(slot_counter) % workers
        return tls.s

    def one(r):
        s = sess()
        prompts = [tr.render(state_for(r["query"], corpus[c["pid"]]), Q) for c in r["candidates"]]
        t0 = time.perf_counter()
        if backend == "sglang":
            first = read_option_probs_sglang(base_url, prompts[0], ["A", "B"], session=s, token_ids=token_ids)
            rest, _ = batch_read_option_probs_sglang(base_url, prompts[1:], [["A", "B"]] * (len(prompts) - 1), session=s, token_ids=token_ids)
            res = [first] + rest
        else:
            res = [read_option_probs(base_url, p, ["A", "B"], session=s, slot_id=tls.slot) for p in prompts]
        wall = (time.perf_counter() - t0) * 1000
        cands = [dict(c, score=x["probs"].get("A", 0.0), missing=x["missing"], cache_n=x.get("server_cache_n")) for c, x in zip(r["candidates"], res)]
        return {"id": r["id"], "sid": r["sid"], "intent": r["intent"], "chapter": r["chapter"], "pair_id": r["pair_id"], "query": r["query"],
                "candidates": cands, "wall_ms": wall}

    out_path = os.path.join(out_dir, "r_rerank.jsonl")
    done = []
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(one, rows)):
            with lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); done.append(res)
                if (i + 1) % 25 == 0:
                    f.flush()
                    print(f"  [rerank] {i + 1}/{len(rows)} {metrics(done)} ({time.time() - t0:.0f}s)", flush=True)
                    if kw.get("commit"):
                        kw["commit"]()
    walls = sorted(r["wall_ms"] for r in done)
    summ = {**metrics(done), "wall_ms_p50": walls[len(walls) // 2], "backend": backend, "s": round(time.time() - t0, 1)}
    json.dump(summ, open(os.path.join(out_dir, "_summary.json"), "w"), ensure_ascii=False, indent=1)
    print("[rerank]", summ, flush=True)
    if kw.get("commit"):
        kw["commit"]()
    return summ
