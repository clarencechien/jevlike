"""v14 (docs/handoff-v14-sop-routing.md): an LLM routes a shop-floor message to one of 100 SOPs by reading letters.

Letters are A–J, so 100 options need a tournament: catalog order in groups of 10, one read per group, then one read
over the 10 winners (11 reads). Hybrid mode reads once over EmbeddingGemma 2's top-10 (data/v14/eg2_top10.json, in
catalog order). Reads within a query are sequential (the honest per-query cost); queries run in parallel slots.

args: "--mode tournament|hybrid|shortlist --k 20 --subset test30 --workers 4 --limit 0"
v15 shortlist mode: EG's top-K (data/v14/eg2_top<K>.json, catalog order), then the same group-of-10 tournament until at
most 10 remain, then a final read (K=10 -> 1 read, 20 -> 3, 30 -> 4, 50 -> 6; the full catalog would be 11).
--subset test30 runs only the first 30 test-split queries (single-stream latency runs use --workers 1).
Out: <out_dir>/route_<mode>[<k>][_<subset>].jsonl, <out_dir>/_summary_<same>.json
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.client import read_option_probs  # noqa: E402
from decide.prompt import TemplateRenderer, letters_for  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/root/data/v14" if os.path.isdir("/root/data/v14") else os.path.join(ROOT, "data/v14")
INSTR = "這則現場訊息應該參照哪一份 SOP？"


def question(sops):
    return {"type": "choice", "instructions": INSTR,
            "options": {L: {"label": s["title"], "criteria": s["scope"]} for L, s in zip(letters_for(len(sops)), sops)}}


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    mode = a.get("--mode", "tournament"); workers = int(a.get("--workers", 4)); limit = int(a.get("--limit", 0))
    cat = json.load(open(os.path.join(DATA, "sop_catalog.json"), encoding="utf-8"))["sops"]
    by_id = {s["id"]: s for s in cat}
    qs = [json.loads(l) for l in open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8") if l.strip()]
    subset = a.get("--subset", "")
    if subset == "test30":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from embed_route import split
        te = ~split(qs)
        qs = [q for q, t in zip(qs, te) if t][:30]
    if limit:
        qs = qs[:limit]
    k = int(a.get("--k", 10))
    shortlist = None
    if mode == "hybrid":
        shortlist = json.load(open(os.path.join(DATA, "eg2_top10.json"), encoding="utf-8"))
    elif mode == "shortlist":
        shortlist = json.load(open(os.path.join(DATA, f"eg2_top{k}.json"), encoding="utf-8"))
    tag = mode + (str(k) if mode == "shortlist" else "") + (f"_{subset}" if subset else "")
    tr = TemplateRenderer(base_url)
    tls = threading.local()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session()
        return tls.s

    def pick(state, sops):
        letters = letters_for(len(sops))
        r = read_option_probs(base_url, tr.render(state, question(sops)), letters, session=sess())
        p = {sops[letters.index(L)]["id"]: v for L, v in r["probs"].items()}
        best = max(p, key=p.get) if p else sops[0]["id"]
        return best, p, r["latency_ms"], r.get("option_mass")

    def one(q):
        t0 = time.perf_counter(); reads = 0; stages = []
        if mode in ("tournament", "shortlist"):
            pool = list(cat) if mode == "tournament" else [by_id[i] for i in shortlist[q["id"]]]
            level = 0
            while len(pool) > 10:  # groups of 10 in catalog order; winners go up a level
                winners = []
                for g in range(0, len(pool), 10):
                    w, p, lat, mass = pick(q["state"], pool[g:g + 10]); reads += 1; winners.append(w)
                    stages.append({"level": level, "group": g // 10, "winner": w, "p_winner": p.get(w), "mass": mass})
                pool = sorted((by_id[w] for w in winners), key=lambda s: s["id"]); level += 1
            final_sops = pool
        else:
            final_sops = [by_id[i] for i in shortlist[q["id"]]]
        pred, p, lat, mass = pick(q["state"], final_sops); reads += 1
        return {"id": q["id"], "gold": q["sop_id"], "pred": pred, "correct": pred == q["sop_id"], "final_probs": p, "final_mass": mass,
                "gold_in_final": q["sop_id"] in [s["id"] for s in final_sops],
                "gold_in_shortlist": (q["sop_id"] in shortlist[q["id"]]) if shortlist else True, "reads": reads, "latency_ms": (time.perf_counter() - t0) * 1000,
                "stages": stages}

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time(); ok = 0
    with open(os.path.join(out_dir, f"route_{tag}.jsonl"), "w", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, qs)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); ok += r["correct"]
            if (i + 1) % 50 == 0:
                print(f"  [{tag}] {i + 1}/{len(qs)} acc {ok / (i + 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
    summ = {"mode": mode, "k": k if mode == "shortlist" else None, "subset": subset, "workers": workers, "n": len(qs), "acc_all": ok / len(qs),
            "wall_s": round(time.time() - t0, 1)}
    json.dump(summ, open(os.path.join(out_dir, f"_summary_{tag}.json"), "w"), indent=1)
    return summ
