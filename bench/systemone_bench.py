"""v11 (docs/handoff-v11-clef-llamacpp.md): run Jev/SystemOne-format requests through a decision backend and reuse the
v10 suites (bench/clef_run.py) unchanged.

Backends (both take the same Jev record that decide/clef_adapter.py builds and return Jev answers):
  systemone  llama-server POST /v1/systemone (Clef GGUF on llama.cpp b11371, judgment head in C++)
  jevify     the jevify library (github.com/kushalpatil07/jevify, pinned) against llama-server /v1/chat/completions,
             i.e. the author's own llamacpp path; `--lora-scale` adds {"lora": [{"id": 0, "scale": s}]} to each request (J3)

The API returns probabilities, not logits: rows store log(p) as "logits" (softmax-invariant up to a constant, so the
v10 temperature fit and analysis code work unchanged).

args (run through modal_app.run_bench or modal_clef_gguf): "--backend systemone|jevify --suites smoke,d0,... --workers 4
  --lora-scale 1 --model-name clef"
"""
from __future__ import annotations

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
from bench import clef_run  # noqa: E402
from decide.clef_adapter import QID, row_record  # noqa: E402


def _answer_to_out(question, ans):
    """Jev answer -> (option_ids, probs, log-probs) in the option-id space the adapter's back-map expects."""
    t = question["type"]
    if t == "noul":
        p = float(ans["noul"])
        oids, probs = ["true", "false"], [p, 1.0 - p]
    else:
        probs_d = ans["probabilities"]
        oids = list(probs_d)
        probs = [float(probs_d[o]) for o in oids]
    logits = [math.log(max(p, 1e-12)) for p in probs]
    return oids, probs, logits


class SystemOneRunner:
    """POST /v1/systemone; one record per request (all its questions go in the same request, as Clef expects)."""

    def __init__(self, base_url, model_name="clef", workers=4):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model_name, self.workers = model_name, workers
        self.tls = threading.local()

    def _sess(self):
        if not hasattr(self.tls, "s"):
            self.tls.s = requests.Session()
        return self.tls.s

    def one(self, rec):
        body = {"model": self.model_name, "state": rec["state"], "questions": rec["questions"]}
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                r = self._sess().post(self.url, json=body, timeout=600)
                ms = (time.perf_counter() - t0) * 1000
                r.raise_for_status()
                d = r.json()
                out = {qid: _answer_to_out(q, d["answers"][qid]) for qid, q in rec["questions"].items()}
                return {"out": out, "n_tokens": d.get("usage", {}).get("input_tokens"), "ms": ms}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise RuntimeError(f"{rec.get('id')}: {e!r} {getattr(e, 'response', None) and e.response.text[:300]}")
                time.sleep(1)

    def run(self, recs, log=""):
        t0 = time.time(); res = [None] * len(recs)
        with ThreadPoolExecutor(self.workers) as ex:
            for i, x in enumerate(ex.map(self.one, recs)):
                res[i] = x
                if log and (i + 1) % 500 == 0:
                    print(f"  [{log}] {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
        return res


class JevifyRunner(SystemOneRunner):
    """jevify library, llamacpp runtime preset (chat completions, max_tokens 1, top_logprobs 20, thinking off)."""

    def __init__(self, base_url, model_name="jevify", workers=4, lora_scale=None):
        from jevify import Jevify
        extra = {"lora": [{"id": 0, "scale": float(lora_scale)}]} if lora_scale is not None else None
        self.jev_args = dict(runtime="llamacpp", model=model_name, base_url=base_url.rstrip("/") + "/v1", extra_body=extra)
        self.Jevify, self.workers, self.tls = Jevify, workers, threading.local()

    def _jev(self):
        if not hasattr(self.tls, "j"):
            self.tls.j = self.Jevify.from_runtime(**self.jev_args)
        return self.tls.j

    def one(self, rec):
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                d = self._jev().system_one(rec["state"], rec["questions"])
                ms = (time.perf_counter() - t0) * 1000
                answers = d.get("answers", d)
                out = {qid: _answer_to_out(q, answers[qid]) for qid, q in rec["questions"].items()}
                return {"out": out, "n_tokens": (d.get("usage") or {}).get("input_tokens"), "ms": ms}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise RuntimeError(f"{rec.get('id')}: {e!r}")
                time.sleep(1)


# ------------------------------------------------------------------ HTTP latency / throughput
def suite_latency(runner, out, n=200):
    import numpy as np
    rng = random.Random(1)
    rows = rng.sample(clef_run.d0_rows(), n)
    _, precs, _ = clef_run._packed_inputs()
    precs = rng.sample(precs, n)
    s = {}
    for name, recs in (("single", [row_record(r)[0] for r in rows]), ("packed3", precs)):
        for r in recs[:10]:
            runner.one(r)  # warm-up
        ms = [runner.one(r)["ms"] for r in recs]
        s[name] = {"n": n, "p50_ms": round(float(np.percentile(ms, 50)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1)}
        print(f"  [latency] {name} {s[name]}", flush=True)
    single = [row_record(r)[0] for r in rows]
    for c in (1, 4, 8):
        w0 = runner.workers; runner.workers = c
        t0 = time.time(); runner.run(single); dt = time.time() - t0
        runner.workers = w0
        s[f"throughput_c{c}"] = round(len(single) / dt, 2)
        print(f"  [latency] concurrency {c}: {s[f'throughput_c{c}']} decisions/s", flush=True)
    return {"latency": s}


SUITES = {k: v for k, v in clef_run.SUITES.items() if k not in ("latency", "profile")}
SUITES["latency"] = suite_latency


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    backend = a.get("--backend", "systemone"); workers = int(a.get("--workers", 4))
    suites = a.get("--suites", "smoke").split(",")
    if a.get("--out-sub"):
        out_dir = os.path.join(out_dir, a["--out-sub"])
    os.makedirs(out_dir, exist_ok=True)
    if backend == "systemone":
        runner = SystemOneRunner(base_url, a.get("--model-name", "clef"), workers)
    else:
        runner = JevifyRunner(base_url, a.get("--model-name", "jevify"), workers, a.get("--lora-scale"))
    sp = os.path.join(out_dir, "_summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    summary["meta"] = {"backend": backend, "args": args, "base_url": base_url}
    for name in suites:
        t0 = time.time()
        summary[name] = {**SUITES[name](runner, out_dir), "s": round(time.time() - t0, 1)}
        print(f"[systemone] {name} {summary[name]}", flush=True)
        json.dump(summary, open(sp, "w"), ensure_ascii=False, indent=1)
        if kw.get("commit"):
            kw["commit"]()
    return summary
