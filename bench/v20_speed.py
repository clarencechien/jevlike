"""v20 (docs/handoff-v20-speed.md): the same 500 items on the same card — LLM call / jevlike letters / EG2.

Items: D0 test split of x_ticket_route, x_10way_intent, m_alarm_category, p_line_change (100 each) + data/v20/eg2poc_status.jsonl (100).
Engines (--engine): llm (chat JSON, greedy, 16 tokens) | jev (first-token letters) | eg (EmbeddingGemma 2, sentence-transformers).
Scenarios: S1 one item at a time (p50/p95) | S2 batch of 100 per task, each engine's own best batching | S3 the same 100
states with a second question (jev/llm: cache hit on the state prefix; eg: new label matrix only).

args: "--engine jev --scenarios S1,S2,S3 --workers 4 --limit 0 --device cuda --tag jev_26b"
Out: <out_dir>/<tag>.jsonl (S1 per-item records) and <out_dir>/<tag>_summary.json
"""
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
from bench.accuracy import load, parse_letter  # noqa: E402

D0_TASKS = ["x_ticket_route", "x_10way_intent", "m_alarm_category", "p_line_change"]
SECOND = {"x_ticket_route": "m_alarm_category", "m_alarm_category": "x_ticket_route", "x_10way_intent": "p_line_change", "p_line_change": "x_10way_intent"}
EG_MODEL, EG_REV = "google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"
CLS, DOC, TEMP = "task: classification | query: ", "title: none | text: ", 0.045
LETTERS = "ABCDEFGHIJ"


def test_ids(rows, seed=0):
    groups = sorted({r.get("pair_id") or r["id"] for r in rows})
    rng = random.Random(seed); rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    return {r["id"] for r in rows if (r.get("pair_id") or r["id"]) not in cal}


def load_items(limit=0):
    """Returns {task: [items]} where item = {id, task, lang, state, question, question2, gold, label}."""
    out = {}
    q_of = {}
    for t in D0_TASKS:
        rows = load(t); te = test_ids(rows); rows = [r for r in rows if r["id"] in te]
        q_of[t] = rows[0]["question"]
        out[t] = [{"id": r["id"], "task": t, "lang": r["lang"], "state": r["state"], "question": r["question"], "gold": r["gold"], "label": None} for r in rows]
    for t in D0_TASKS:
        for it in out[t]:
            it["question2"] = q_of[SECOND[t]]
    p = os.path.join(ROOT, "data/v20/eg2poc_status.jsonl")
    out["eg2poc_status"] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if limit:
        out = {t: v[:limit] for t, v in out.items()}
    return out


def letters_of(q):
    return list(q["options"].keys())


def option_text(opt):
    if isinstance(opt, dict):
        return f"{opt['label']}：{opt['criteria']}" if opt.get("criteria") else opt["label"]
    return opt


def summarize(ms):
    a = np.asarray(ms, dtype=float)
    return {"n": int(len(a)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())} if len(a) else {}


# ------------------------------------------------------------------ llama-server engines
class LlamaEngine:
    def __init__(self, base_url, engine, workers):
        import itertools
        import requests
        from decide.prompt import SYSTEM, SYSTEM_EN, TemplateRenderer, build_messages
        from decide.client import chat_json, read_option_probs
        self.url, self.engine, self.workers = base_url, engine, workers
        self.tr = TemplateRenderer(base_url); self.SYS = {"zh": SYSTEM, "en": SYSTEM_EN}
        self.build, self.chat, self.read = build_messages, chat_json, read_option_probs
        self.tls = threading.local(); self.counter = itertools.count(); self.requests = requests

    def sess(self):
        if not hasattr(self.tls, "s"):
            self.tls.s = self.requests.Session(); self.tls.slot = next(self.counter) % self.workers
        return self.tls.s, self.tls.slot

    def one(self, item, q):
        s, slot = self.sess(); letters = letters_of(q); system = self.SYS.get(item.get("lang", "zh"), self.SYS["zh"])
        for attempt in range(3):
            try:
                if self.engine == "llm":
                    msgs = self.build(item["state"], q, system)
                    msgs[-1]["content"] += '\n\n以 JSON 回答：{"answer": "<字母>"}'
                    r = self.chat(self.url, msgs, max_tokens=16, session=s)
                    ch = parse_letter(r["content"] or "", letters)
                    u = r.get("usage") or {}
                    return {"chosen": ch, "probs": None, "latency_ms": r["latency_ms"], "prompt_tokens": u.get("prompt_tokens"),
                            "completion_tokens": u.get("completion_tokens"), "cache_n": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                            "content": r["content"]}
                prompt = self.tr.render(item["state"], q, system=system)
                r = self.read(self.url, prompt, letters, session=s, slot_id=slot)
                ch = max(r["probs"], key=r["probs"].get) if r["probs"] else None
                return {"chosen": ch, "probs": r["probs"], "latency_ms": r["latency_ms"], "prompt_tokens": r["prompt_tokens"], "cache_n": r["server_cache_n"],
                        "option_mass": r.get("option_mass")}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"error": repr(e), "latency_ms": None}
                time.sleep(1)

    def s1(self, items):
        return [self.one(it, it["question"]) for it in items]

    def batch(self, items, which="question"):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(lambda it: self.one(it, it[which]), items))
        return (time.perf_counter() - t0), res

    def s3b(self, items):
        """Shared-state second question the way it is used in production: Q1 then immediately Q2 on the same slot,
        one item at a time, so the state prefix is still in that slot's KV cache. Returns Q2 records (latency, cache_n)."""
        out = []
        for it in items:
            self.one(it, it["question"]); out.append(self.one(it, it["question2"]))
        return out


# ------------------------------------------------------------------ EmbeddingGemma 2
class EGEngine:
    def __init__(self, device):
        from sentence_transformers import SentenceTransformer
        import torch
        self.torch = torch
        if device == "cpu":
            torch.set_num_threads(4)
        t0 = time.perf_counter(); self.m = SentenceTransformer(EG_MODEL, revision=EG_REV, device=device); self.load_s = time.perf_counter() - t0
        self.device = device; self.L = {}

    def enc(self, xs, bs=64):
        e = np.asarray(self.m.encode(xs, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
        if self.device != "cpu":
            self.torch.cuda.synchronize()
        return e

    def labels(self, task, q):
        """D0: classification prefix for both sides, option vectors centred within the question (v13 Z1).
        eg2poc: document template for the text, classification template for the labels, no centring (their config)."""
        letters = letters_of(q); texts = [option_text(q["options"][L]) for L in letters]
        t0 = time.perf_counter()
        E = self.enc([CLS + x for x in texts], bs=16)
        if task != "eg2poc_status":
            E = E - E.mean(0, keepdims=True); E /= np.linalg.norm(E, axis=1, keepdims=True)
        return letters, E, time.perf_counter() - t0

    def state_prefix(self, task):
        return DOC if task == "eg2poc_status" else CLS

    @staticmethod
    def probs(X, L, letters):
        S = X @ L.T; Z = (S - S.max(1, keepdims=True)) / TEMP; P = np.exp(Z); P /= P.sum(1, keepdims=True)
        return [dict(zip(letters, map(float, row))) for row in P]

    def s1(self, items, task):
        letters, L, _ = self.labels(task, items[0]["question"]); pre = self.state_prefix(task); out = []
        self.enc([pre + items[0]["state"]], bs=1)  # warm
        for it in items:
            t = time.perf_counter(); x = self.enc([pre + it["state"]], bs=1); p = self.probs(x, L, letters)[0]
            out.append({"chosen": max(p, key=p.get), "probs": p, "latency_ms": (time.perf_counter() - t) * 1000})
        return out

    def batch(self, items, task, which="question", X=None):
        letters, L, lab_s = self.labels(task, items[0][which]); pre = self.state_prefix(task)
        t0 = time.perf_counter()
        if X is None:
            X = self.enc([pre + it["state"] for it in items], bs=64)
        P = self.probs(X, L, letters)
        return (time.perf_counter() - t0), [{"chosen": max(p, key=p.get), "probs": p} for p in P], X, lab_s


# ------------------------------------------------------------------ main
def score(items, res):
    ok = []
    for it, r in zip(items, res):
        if r.get("chosen") is None:
            ok.append(False); continue
        ok.append(((r["chosen"] != "A") == bool(it["label"])) if it["task"] == "eg2poc_status" else (r["chosen"] == it["gold"]))
    return float(np.mean(ok)) if ok else None


def main(base_url="", out_dir=".", args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    engine = a.get("--engine", "jev"); scen = set(a.get("--scenarios", "S1,S2,S3").split(",")); workers = int(a.get("--workers", 4))
    limit = int(a.get("--limit", 0)); device = a.get("--device", "cuda"); tag = a.get("--tag", engine)
    os.makedirs(out_dir, exist_ok=True)
    items = load_items(limit)
    eng = EGEngine(device) if engine == "eg" else LlamaEngine(base_url, engine, workers)
    summ = {"engine": engine, "tag": tag, "device": device if engine == "eg" else None, "workers": workers, "tasks": {}}
    if engine == "eg":
        summ["load_s"] = eng.load_s
    f = open(os.path.join(out_dir, f"{tag}.jsonl"), "w", encoding="utf-8")
    for task, its in items.items():
        T = {"n": len(its)}; t_all = time.time()
        if "S1" in scen:
            if engine == "eg":
                eng.enc([eng.state_prefix(task) + its[0]["state"]], bs=1)
            else:
                eng.one(its[0], its[0]["question"])  # warm
            res = eng.s1(its, task) if engine == "eg" else eng.s1(its)
            T["S1"] = summarize([r["latency_ms"] for r in res if r.get("latency_ms") is not None]); T["S1_acc"] = score(its, res)
            if engine != "eg":
                T["S1_cache_n_mean"] = float(np.mean([r.get("cache_n") or 0 for r in res]))
            for it, r in zip(its, res):
                f.write(json.dumps({"id": it["id"], "task": task, "gold": it["gold"], "label": it["label"], "tag": tag, **r}, ensure_ascii=False) + "\n")
            f.flush()
        X = None

        def req_stats(res):
            lat = [r["latency_ms"] for r in res if r.get("latency_ms") is not None]
            return {"req_p50": float(np.percentile(lat, 50)) if lat else None, "req_p95": float(np.percentile(lat, 95)) if lat else None,
                    "cache_n_mean": float(np.mean([r.get("cache_n") or 0 for r in res])), "prompt_tokens_mean": float(np.mean([r.get("prompt_tokens") or 0 for r in res])),
                    "errors": int(sum("error" in r for r in res))}

        if "S2" in scen:
            if engine == "eg":
                wall, res, X, lab_s = eng.batch(its, task, "question"); T["S2_prewarm_labels_s"] = lab_s
            else:
                wall, res = eng.batch(its, "question")
            T["S2"] = {"wall_s": wall, "per_item_ms": wall * 1000 / len(its), "items_per_s": len(its) / wall, "acc": score(its, res), **(req_stats(res) if engine != "eg" else {})}
        if "S3" in scen:
            if engine == "eg":
                wall, res, _, lab_s = eng.batch(its, task, "question2", X=X); T["S3"] = {"wall_s": wall, "per_item_ms": wall * 1000 / len(its), "labels_s": lab_s}
            else:
                wall, res = eng.batch(its, "question2")
                T["S3"] = {"wall_s": wall, "per_item_ms": wall * 1000 / len(its), **req_stats(res)}
        if "S2r" in scen and engine != "eg":  # S2 again after S3: does order (what the slots hold) explain a difference?
            wall, res = eng.batch(its, "question")
            T["S2r"] = {"wall_s": wall, "per_item_ms": wall * 1000 / len(its), **req_stats(res)}
        if "S3b" in scen and engine != "eg":  # the production shape: Q1 then Q2 on the same slot, one item at a time
            res = eng.s3b(its)
            T["S3b"] = {"q2_" + k: v for k, v in summarize([r["latency_ms"] for r in res if r.get("latency_ms") is not None]).items()} | req_stats(res)
        T["task_s"] = round(time.time() - t_all, 1); summ["tasks"][task] = T
        print(f"[{tag}] {task}: " + json.dumps({k: v for k, v in T.items() if k in ("S1", "S1_acc", "S2", "S3")}, ensure_ascii=False)[:400], flush=True)
    f.close()
    if engine == "eg" and device != "cpu":
        summ["gpu_mem_mb"] = float(eng.torch.cuda.max_memory_allocated() / 1e6)
    json.dump(summ, open(os.path.join(out_dir, f"{tag}_summary.json"), "w"), ensure_ascii=False, indent=1)
    return summ
