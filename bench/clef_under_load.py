"""GB10 (docs/handoff-gb10-clef.md §6): Clef decision latency while the 26B server is generating.

On GB10 the two llama-servers share one unified memory and one GPU, so the number that decides deployment is the Clef
latency *under generation load*, not on an idle box. Generation runs as N concurrent streams against the 26B server
(/v1/chat/completions, greedy, 128 tokens, thinking off); Clef answers single questions from D0 one at a time.

Usage (from the repo root, both servers already running):
  python3 bench/clef_under_load.py --clef-url http://127.0.0.1:8090 --gen-url http://127.0.0.1:8080 \
      --streams 0,1,2,4 --n 100 --out results/gb10/clef/under_load.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import clef_run  # noqa: E402
from bench.systemone_bench import SystemOneRunner  # noqa: E402
from decide.clef_adapter import row_record  # noqa: E402

PROMPT = "請用三句話說明這則產線訊息代表什麼、可能原因、建議處置：\n\n"


def gen_stream(url, states, stop, log):
    s = requests.Session(); i = 0
    while not stop.is_set():
        body = {"model": "x", "messages": [{"role": "user", "content": PROMPT + states[i % len(states)]}], "max_tokens": 128,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
        t0 = time.perf_counter()
        try:
            d = s.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=600).json()
            log.append(((d.get("usage") or {}).get("completion_tokens") or 0, time.perf_counter() - t0))
        except Exception:  # noqa: BLE001
            pass
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clef-url", default="http://127.0.0.1:8090")
    ap.add_argument("--gen-url", default="http://127.0.0.1:8080")
    ap.add_argument("--streams", default="0,1,2,4")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/gb10/clef/under_load.json")
    a = ap.parse_args()
    rng = random.Random(3)
    rows = rng.sample(clef_run.d0_rows(), a.n)
    recs = [row_record(r)[0] for r in rows]
    states = [r["state"] for r in rng.sample(clef_run.d0_rows(), 50)]
    clef = SystemOneRunner(a.clef_url, "clef", workers=1)
    for r in recs[:10]:
        clef.one(r)
    out = {}
    for k in [int(x) for x in a.streams.split(",")]:
        stop = threading.Event(); log = []
        ths = [threading.Thread(target=gen_stream, args=(a.gen_url, states, stop, log)) for _ in range(k)]
        for th in ths:
            th.start()
        base_tps = None
        if k:
            time.sleep(20)  # generation alone first: the baseline for the generation slowdown
            tok = sum(x for x, _ in log); sec = sum(s for _, s in log)
            base_tps = round(tok / sec, 1) if sec else None
            log.clear()
        ms = [clef.one(r)["ms"] for r in recs]
        stop.set()
        for th in ths:
            th.join()
        tok = sum(t for t, _ in log); sec = sum(s for _, s in log)
        out[f"streams_{k}"] = {"clef_p50_ms": round(float(np.percentile(ms, 50)), 1), "clef_p95_ms": round(float(np.percentile(ms, 95)), 1),
                               "gen_requests": len(log), "gen_tok_per_s_per_stream": round(tok / sec, 1) if sec else None,
                               "gen_tok_per_s_per_stream_alone": base_tps}
        print(f"[under_load] {k} generation streams: {out[f'streams_{k}']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
