"""v13 (docs/handoff-v13-embeddinggemma2.md): EmbeddingGemma 2 as a zero-shot bi-encoder decision maker on D0.

Mimics MediaPipe Decision Maker's embedding backend: prewarm = embed each question's K options once (optionally
centre them within the question and re-normalise); evaluate = embed the state once, cosine against the K options.
Writes results/modal/v13/<task>.npz (ids, gold, split, scores per arm, state embeddings) and _run.json (latency).

  python3 bench/embed_d0.py [--device cpu] [--batch 32]
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
MODEL, REV = "google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"
TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "q_defect_root",
         "p_uph_anomaly", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]
CLS = "task: classification | query: "
QRY = "task: search result | query: "
OUT = os.path.join(ROOT, "results/modal/v13")


def group_split(rows, seed=0):  # == bench/analyze.group_split (inlined so this file has no matplotlib import)
    import random
    groups = sorted({r.get("pair_id") or r["id"] for r in rows})
    rng = random.Random(seed); rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    return np.array([(r.get("pair_id") or r["id"]) in cal for r in rows])


def option_texts(q):
    out = []
    for L, o in q["options"].items():
        lab = o["label"] if isinstance(o, dict) else o
        crit = o.get("criteria", "") if isinstance(o, dict) else ""
        out.append((L, lab, crit))
    return out


def centre(E):
    E = E - E.mean(0, keepdims=True)
    return E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)


def main():
    a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    device, batch = a.get("--device", "cpu"), int(a.get("--batch", 32))
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    t0 = time.time()
    model = SentenceTransformer(MODEL, revision=REV, device=device)
    load_s = time.time() - t0
    enc = lambda xs: np.asarray(model.encode(xs, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)  # noqa: E731
    os.makedirs(OUT, exist_ok=True)
    run = {"model": MODEL, "revision": REV, "device": device, "load_s": round(load_s, 1), "threads": torch.get_num_threads(), "tasks": {}}
    for task in TASKS:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8") if l.strip()]
        ca = group_split(rows)
        qkeys = [json.dumps(r["question"], ensure_ascii=False, sort_keys=True) for r in rows]
        uq = sorted(set(qkeys))
        t1 = time.time()
        opt = {}
        for k in uq:  # prewarm: K options per distinct question
            q = json.loads(k); ops = option_texts(q)
            E_cls = enc([CLS + f"{lab}：{crit}" if crit else CLS + lab for _, lab, crit in ops])
            E_doc = enc([f"title: {lab} | text: {crit or lab}" for _, lab, crit in ops])
            opt[k] = {"letters": [L for L, _, _ in ops], "cls": E_cls, "cls_c": centre(E_cls), "doc_c": centre(E_doc)}
        prewarm_s = time.time() - t1
        t2 = time.time()
        S_cls = enc([CLS + r["state"] for r in rows])
        enc_cls_s = time.time() - t2
        S_qry = enc([QRY + r["state"] for r in rows])
        letters = opt[qkeys[0]]["letters"]
        K = len(letters)
        Z1, Z2, Z3 = (np.zeros((len(rows), K), np.float32) for _ in range(3))
        for i, (r, k) in enumerate(zip(rows, qkeys)):
            o = opt[k]
            assert o["letters"] == letters, f"{task}: option letters differ between rows"
            Z1[i] = o["cls_c"] @ S_cls[i]; Z2[i] = o["cls"] @ S_cls[i]; Z3[i] = o["doc_c"] @ S_qry[i]
        gold = np.array([letters.index(r["gold"]) for r in rows])
        np.savez_compressed(os.path.join(OUT, f"{task}.npz"), ids=np.array([r["id"] for r in rows]), gold=gold, cal=ca, letters=np.array(letters),
                            Z1=Z1, Z2=Z2, Z3=Z3, S=S_cls.astype(np.float16))
        acc = {n: float((Z.argmax(1)[~ca] == gold[~ca]).mean()) for n, Z in (("Z1", Z1), ("Z2", Z2), ("Z3", Z3))}
        run["tasks"][task] = {"n": len(rows), "K": K, "distinct_questions": len(uq), "prewarm_s": round(prewarm_s, 2),
                              "encode_states_s": round(enc_cls_s, 2), "acc_test": acc}
        print(task, run["tasks"][task], flush=True)
    # single-state latency (batch 1), the evaluate() path: one state embedding + K dot products
    states = [json.loads(l)["state"] for l in open(os.path.join(ROOT, "data/synthetic/x_escalate.jsonl"), encoding="utf-8")][:60]
    o_last = opt[qkeys[0]]["cls_c"]
    for s in states[:10]:
        enc([CLS + s])
    lat = []
    for s in states[10:]:
        t = time.perf_counter(); e = enc([CLS + s]); _ = o_last @ e[0]; lat.append((time.perf_counter() - t) * 1000)
    run["latency_ms_batch1"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "n": len(lat)}
    run["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(run, open(os.path.join(OUT, "_run.json"), "w"), indent=1)
    print(json.dumps(run["latency_ms_batch1"]), flush=True)


if __name__ == "__main__":
    main()
