"""v14 (docs/handoff-v14-sop-routing.md): EmbeddingGemma 2 routes a shop-floor message to one of 100 SOPs.

Arms: EG (asymmetric retrieval prefixes, the model card's recommendation for search) and EG-cls (symmetric classification
prefix + within-question centring, as v13's Z1). Writes <out>/eg_route.npz (scores), <out>/_run_<device>.json (latency)
and, on the reference run, data/v14/eg2_top10.json (the shortlist the hybrid LLM arms read).

  python3 bench/embed_route.py --device cpu  --out results/modal/v14 --write-shortlist 1
  (Modal) modal_v14.py::eg_gpu runs the same with --device cuda for latency only
"""
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "/root/data/v14" if os.path.isdir("/root/data/v14") else os.path.join(ROOT, "data/v14")
MODEL, REV = "google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"
QRY, CLS = "task: search result | query: ", "task: classification | query: "


def load():
    cat = json.load(open(os.path.join(DATA, "sop_catalog.json"), encoding="utf-8"))["sops"]
    qs = [json.loads(l) for l in open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8") if l.strip()]
    return cat, qs


def split(qs, seed=0):
    """Group split by SOP (== bench/analyze.group_split with pair_id = sop_id): cal SOPs vs test SOPs."""
    groups = sorted({q["sop_id"] for q in qs}); rng = random.Random(seed); rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    return np.array([q["sop_id"] in cal for q in qs])


def main(argv=None):
    a = dict(zip((argv or sys.argv[1:])[0::2], (argv or sys.argv[1:])[1::2]))
    device = a.get("--device", "cpu"); out = a.get("--out", os.path.join(ROOT, "results/modal/v14"))
    from sentence_transformers import SentenceTransformer
    import torch
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    cat, qs = load()
    t0 = time.time(); m = SentenceTransformer(MODEL, revision=REV, device=device); load_s = time.time() - t0
    enc = lambda xs, bs=32: np.asarray(m.encode(xs, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)  # noqa: E731
    t1 = time.time()
    D = enc([f"title: {s['title']} | text: {s['scope']}" for s in cat])
    C = enc([CLS + f"{s['title']}：{s['scope']}" for s in cat]); C = C - C.mean(0, keepdims=True); C /= np.linalg.norm(C, axis=1, keepdims=True)
    prewarm_s = time.time() - t1
    Q = enc([QRY + q["state"] for q in qs]); Qc = enc([CLS + q["state"] for q in qs])
    S_eg, S_cls = Q @ D.T, Qc @ C.T
    ids = [s["id"] for s in cat]; gold = np.array([ids.index(q["sop_id"]) for q in qs])
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "eg_route.npz"), qids=np.array([q["id"] for q in qs]), gold=gold, cal=split(qs), S_eg=S_eg, S_cls=S_cls)
    if a.get("--write-shortlist") == "1":
        sl = {}
        for i, q in enumerate(qs):
            top = np.argsort(-S_eg[i])[:10]
            sl[q["id"]] = [ids[j] for j in sorted(top)]  # catalog order, not EG rank (handoff §1)
        json.dump(sl, open(os.path.join(DATA, "eg2_top10.json"), "w"), ensure_ascii=False, indent=0)
    # evaluate() latency: one message -> one embedding -> 100 dot products -> argmax, batch 1
    lat = []
    for i, q in enumerate(qs[:70]):
        t = time.perf_counter(); e = enc([QRY + q["state"]], bs=1); _ = int(np.argmax(D @ e[0]))
        if device != "cpu":
            torch.cuda.synchronize()
        if i >= 10:
            lat.append((time.perf_counter() - t) * 1000)
    t2 = time.time(); enc([QRY + q["state"] for q in qs], bs=64); thr = len(qs) / (time.time() - t2)
    run = {"device": device, "gpu": torch.cuda.get_device_name(0) if device != "cpu" else None, "load_s": round(load_s, 1), "prewarm_100_sops_s": round(prewarm_s, 2),
           "latency_ms_batch1": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "n": len(lat)}, "throughput_per_s_batch64": round(thr, 1),
           "top1_eg_all": float((S_eg.argmax(1) == gold).mean()), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(run, open(os.path.join(out, f"_run_{device}.json"), "w"), indent=1)
    print(json.dumps(run), flush=True)
    return run


if __name__ == "__main__":
    main()
