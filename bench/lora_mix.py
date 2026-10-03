"""v11 J3 (docs/handoff-v11-clef-llamacpp.md §4b): cost of serving decisions (jevify LoRA, scale 1) and generation
(LoRA scale 0) from one llama-server with one copy of the base model.

  --mode mix     server started with --lora: decisions alone, generation alone, then both interleaved
  --mode nolora  server without --lora: generation alone (reference for the "scale 0 == no LoRA" check)

Generation: 50 D0 states, "summarise in one sentence", greedy, 64 tokens, thinking off, via /v1/chat/completions.
Outputs: <out_dir>/gen_<mode>.jsonl (text per prompt), _summary.json (p50s, tokens/s).
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import clef_run  # noqa: E402
from bench.systemone_bench import JevifyRunner  # noqa: E402
from decide.clef_adapter import row_record  # noqa: E402

GEN_PROMPT = "請用一句話摘要這則產線訊息，不要加任何說明：\n\n"


def gen_one(base_url, state, lora_scale=None):
    body = {"model": "x", "messages": [{"role": "user", "content": GEN_PROMPT + state}], "max_tokens": 64, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
    if lora_scale is not None:
        body["lora"] = [{"id": 0, "scale": float(lora_scale)}]
    t0 = time.perf_counter()
    r = requests.post(base_url.rstrip("/") + "/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    dt = time.perf_counter() - t0
    n = (d.get("usage") or {}).get("completion_tokens") or 0
    return {"text": d["choices"][0]["message"]["content"], "s": dt, "tokens": n}


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    mode = a.get("--mode", "mix")
    if a.get("--out-sub"):
        out_dir = os.path.join(out_dir, a["--out-sub"])
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(7)
    rows = clef_run.d0_rows()
    gen_states = [r["state"] for r in rng.sample(rows, 50)]
    dec_recs = [row_record(r)[0] for r in rng.sample(rows, 100)]
    s = {"mode": mode}

    def gen_all(scale):
        out = [gen_one(base_url, st, scale) for st in gen_states]
        tps = sum(o["tokens"] for o in out) / sum(o["s"] for o in out)
        return out, tps

    if mode == "nolora":
        out, tps = gen_all(None)
        s["gen_alone_tps"] = round(tps, 1)
    else:
        jev = JevifyRunner(base_url, "jevify", workers=1, lora_scale=1)
        for r in dec_recs[:5]:
            jev.one(r)
        ms = [jev.one(r)["ms"] for r in dec_recs]
        s["dec_alone_p50_ms"] = round(float(np.percentile(ms, 50)), 1)
        out, tps = gen_all(0)
        s["gen_alone_tps"] = round(tps, 1)
        # interleaved: a generation stream and a decision stream at the same time
        stop = threading.Event(); gen_log = []

        def gen_loop():
            i = 0
            while not stop.is_set():
                gen_log.append(gen_one(base_url, gen_states[i % len(gen_states)], 0)); i += 1

        th = threading.Thread(target=gen_loop); th.start()
        time.sleep(2)
        ms_mix = [jev.one(r)["ms"] for r in dec_recs]
        stop.set(); th.join()
        s["dec_mixed_p50_ms"] = round(float(np.percentile(ms_mix, 50)), 1)
        s["dec_mixed_p95_ms"] = round(float(np.percentile(ms_mix, 95)), 1)
        s["gen_mixed_tps"] = round(sum(g["tokens"] for g in gen_log) / max(sum(g["s"] for g in gen_log), 1e-9), 1)
        s["gen_mixed_n"] = len(gen_log)
    with open(os.path.join(out_dir, f"gen_{mode}.jsonl"), "w", encoding="utf-8") as f:
        for st, o in zip(gen_states, out):
            f.write(json.dumps({"state": st, **o}, ensure_ascii=False) + "\n")
    json.dump(s, open(os.path.join(out_dir, f"_summary_{mode}.json"), "w"), ensure_ascii=False, indent=1)
    print("[lora_mix]", s, flush=True)
    if kw.get("commit"):
        kw["commit"]()
    return s
