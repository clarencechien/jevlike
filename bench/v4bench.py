"""v4 speed suite, backend-agnostic. Each backend uses its own best method:
  llama-server : -np slots + cache_prompt, K prompts sent concurrently (threads)
  sglang       : K prompts sent as one list request (RadixAttention shares the prefix)

Items: L1 single (~100-tok state, distinct states), L2 state length 100/500/2000, L4 shared state K=1/5/10/16
(state ~1100 tok), L5 concurrency 1/8/32, L6 decision latency under a 512-token background generation,
S2 meeting load (120 windows x 7 questions, shared system prompt), S3 D0 test end-to-end (1001 rows).
args: "--n 200 --warmup 20 --only L1,L2,L4,L5,L6,S2,S3 --out v4.json"
"""
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.latency import LINES_EN, LINES_ZH, pct, question  # noqa: E402
from decide.client import batch_read_option_probs_sglang, generate_text, reader_for  # noqa: E402
from decide.prompt import TemplateRenderer, letters_for  # noqa: E402


def summarize(samples, key="latency_ms"):
    xs = [s[key] for s in samples if s.get(key) is not None]
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "p99": pct(xs, 0.99), "mean": statistics.fmean(xs) if xs else None,
            "prompt_tokens_mean": statistics.fmean([s["prompt_tokens"] for s in samples if s.get("prompt_tokens")] or [0]),
            "cache_n_mean": statistics.fmean([s["server_cache_n"] for s in samples if s.get("server_cache_n") is not None] or [0])}


def make_state_factory(rng, tokenize):
    def line():
        tpl = rng.choice(LINES_ZH if rng.random() < 0.8 else LINES_EN)
        return tpl.format(l=rng.randint(1, 6), r=rng.randint(1, 3), z=rng.randint(1, 10), t=rng.randint(150, 260), s=rng.randint(200, 250),
                          v=rng.randint(60, 90), p=rng.randint(80, 100), n=rng.randint(1, 500), c=round(rng.uniform(0.8, 2.0), 2),
                          d=rng.randint(10, 28), k=rng.randint(1000, 9999), m=rng.randint(500, 2000), u=rng.randint(900, 1300), g=rng.randint(1000, 1200))

    def make(target_tokens):
        parts = [line(), line(), line()]
        per_line = max(8, tokenize("\n".join(parts)) / 3)
        parts += [line() for _ in range(max(0, int(target_tokens / per_line) - 3))]
        n = tokenize("\n".join(parts))
        while n > target_tokens * 1.15 and len(parts) > 1:  # trim overshoot (keep prompts well under the 4096 context)
            parts.pop(); n = tokenize("\n".join(parts))
        while n < target_tokens:
            parts.append(line()); n = tokenize("\n".join(parts))
        return "\n".join(parts)
    return make


def main(base_url, out_dir, args="", backend="llama", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    N, WARM = int(a.get("--n", 200)), int(a.get("--warmup", 20))
    only = set(a.get("--only", "L1,L2,L4,L5,L6,S2,S3").split(","))
    out_name = a.get("--out", "v4.json")
    l4_mode = a.get("--l4-mode", "par")  # llama-server: par = K threads (different slots), seq = one slot with cache reuse
    rng = random.Random(42)
    sess = requests.Session()
    read = reader_for(backend)

    if backend == "sglang":
        def tokenize(text):
            r = sess.post(f"{base_url}/generate", json={"text": text, "sampling_params": {"max_new_tokens": 0}}, timeout=120).json()
            return r["meta_info"]["prompt_tokens"]
    else:
        def tokenize(text):
            return len(sess.post(f"{base_url}/tokenize", json={"content": text}, timeout=60).json()["tokens"])
    make_state = make_state_factory(rng, tokenize)
    tr = TemplateRenderer(base_url, static=(backend == "sglang"))
    out = {"backend": backend, "model": kw.get("model"), "n": N, "warmup": WARM, "results": {}}

    def one(p, L, s=None):
        return read(base_url, p, L, session=s or sess)

    def run_seq(name, prompts_letters, note=""):
        samples = []
        for i, (p, L) in enumerate(prompts_letters):
            r = one(p, L)
            if i >= WARM:
                samples.append({k: r.get(k) for k in ("latency_ms", "prompt_tokens", "server_cache_n", "missing", "first_token")})
        out["results"][name] = {"summary": summarize(samples), "note": note, "missing_rate": sum(bool(s["missing"]) for s in samples) / max(1, len(samples))}
        print(name, json.dumps(out["results"][name]["summary"]), flush=True)

    def run_group(prompts, letters_list):
        """K prompts that share a prefix: backend's best method. Returns (wall_ms, cached_tokens list, per-call results)."""
        if backend == "sglang":
            res, wall = batch_read_option_probs_sglang(base_url, prompts, letters_list, session=sess)
            return wall, [r["server_cache_n"] for r in res], res
        t0 = time.perf_counter()
        if l4_mode == "seq":
            res = [read(base_url, p, L, session=sess, slot_id=0) for p, L in zip(prompts, letters_list)]
        else:
            with ThreadPoolExecutor(len(prompts)) as ex:
                res = list(ex.map(lambda pl: read(base_url, pl[0], pl[1], session=requests.Session()), zip(prompts, letters_list)))
        return (time.perf_counter() - t0) * 1000, [r["server_cache_n"] for r in res], res

    q2 = question(2)
    # ---- L1 / L2
    if "L1" in only or "L2" in only:
        st100 = [make_state(100) for _ in range(N + WARM)]
        run_seq("L1_single_100tok_2opt", [(tr.render(s, q2), letters_for(2)) for s in st100])
        p0 = tr.render(st100[0], q2)
        run_seq("L1_ref_same_prompt_repeated", [(p0, letters_for(2))] * (N + WARM), "full cache hit = decode floor")
    if "L2" in only:
        for tok in (500, 2000):
            run_seq(f"L2_state_{tok}tok", [(tr.render(make_state(tok), q2), letters_for(2)) for _ in range(N + WARM)])
    # ---- L4: shared ~1100-token state, K questions, backend-best grouping; distinct state per group
    if "L4" in only:
        qs = [question(k) for k in (2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5)]
        for i, q in enumerate(qs):
            q["instructions"] = f"問題 {i + 1}：" + q["instructions"]
        for K in (1, 5, 10, 16):
            n_groups = max(20, min(100, N // K)) + 5
            walls, cached, per = [], [], []
            for gi in range(n_groups):
                s = make_state(1100)
                prompts = [tr.render(s, q) for q in qs[:K]]
                letters = [letters_for(len(q["options"])) for q in qs[:K]]
                wall, c, res = run_group(prompts, letters)
                if gi >= 5:
                    walls.append({"latency_ms": wall}); cached += [x or 0 for x in c]; per += [{"latency_ms": r["latency_ms"], "prompt_tokens": r["prompt_tokens"]} for r in res]
            ptoks = statistics.fmean([r["prompt_tokens"] for r in per if r["prompt_tokens"]] or [0])
            hit = statistics.fmean([min(1.0, c / ptoks) for c in cached]) if ptoks else 0
            out["results"][f"L4_shared_{K}q"] = {"summary": summarize(walls), "per_call": summarize(per), "cached_tokens_mean": statistics.fmean(cached) if cached else 0,
                                                 "prefix_hit_rate": hit, "note": f"total ms for K questions on one ~1100-token state; sglang=list batch, llama={l4_mode}"}
            print(f"L4_shared_{K}q", json.dumps(out["results"][f"L4_shared_{K}q"]["summary"]), "cached", round(statistics.fmean(cached) if cached else 0), "hit", round(hit, 2), flush=True)
    # ---- L5 concurrency
    if "L5" in only:
        for conc in (1, 8, 32):
            prompts = [tr.render(make_state(100), q2) for _ in range(N + WARM)]
            samples, lock = [], threading.Lock()

            def go(p):
                r = read(base_url, p, letters_for(2), session=requests.Session())
                with lock:
                    samples.append({"latency_ms": r["latency_ms"], "prompt_tokens": r["prompt_tokens"], "server_cache_n": r["server_cache_n"]})
            t0 = time.perf_counter()
            with ThreadPoolExecutor(conc) as ex:
                list(ex.map(go, prompts))
            wall = time.perf_counter() - t0
            summ = summarize(samples[WARM:]); summ["throughput_rps"] = len(prompts) / wall
            out["results"][f"L5_concurrency_{conc}"] = {"summary": summ}
            print(f"L5_concurrency_{conc}", json.dumps(summ), flush=True)
    # ---- L6 background generation
    if "L6" in only:
        stop = threading.Event(); gen = {"n": 0, "tokens": 0}

        def bg():
            s2 = requests.Session()
            while not stop.is_set():
                try:
                    gen["tokens"] += generate_text(base_url, backend, "請詳細說明 SMT 產線回焊爐溫度曲線的設定原則與常見異常的處理方式：", 512, s2); gen["n"] += 1
                except Exception:  # noqa: BLE001
                    time.sleep(0.5)
        th = threading.Thread(target=bg, daemon=True); th.start(); time.sleep(3)
        run_seq("L6_with_bg_generation", [(tr.render(make_state(100), q2), letters_for(2)) for _ in range(N + WARM)])
        stop.set(); th.join(timeout=120)
        base = out["results"].get("L1_single_100tok_2opt", {}).get("summary")
        out["results"]["L6_with_bg_generation"]["bg_generation"] = gen
        if base:
            s6 = out["results"]["L6_with_bg_generation"]["summary"]
            out["results"]["L6_with_bg_generation"]["slowdown_p50"] = s6["p50"] / base["p50"]
            out["results"]["L6_with_bg_generation"]["slowdown_p95"] = s6["p95"] / base["p95"]
    # ---- S2 meeting load: 120 windows x 7 questions, shared system prompt; windows ~300 tokens
    if "S2" in only:
        qs = [question(k) for k in (2, 3, 4, 5, 2, 3, 4)]
        for i, q in enumerate(qs):
            q["instructions"] = f"問題 {i + 1}：" + q["instructions"]
        windows = [make_state(300) for _ in range(120)]
        t0 = time.perf_counter()
        if backend == "sglang":
            # all 7 questions of a window as one list request, windows pipelined 8 at a time
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda w: batch_read_option_probs_sglang(base_url, [tr.render(w, q) for q in qs], [letters_for(len(q["options"])) for q in qs], session=requests.Session()), windows))
        else:
            jobs = [(tr.render(w, q), letters_for(len(q["options"]))) for w in windows for q in qs]
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda pl: read(base_url, pl[0], pl[1], session=requests.Session()), jobs))
        wall = time.perf_counter() - t0
        out["results"]["S2_meeting_120x7"] = {"wall_s": wall, "decisions": 120 * 7, "decisions_per_s": 840 / wall}
        print("S2", json.dumps(out["results"]["S2_meeting_120x7"]), flush=True)
    # ---- S3 D0 test end-to-end
    if "S3" in only:
        def group_split(rows_, seed):  # same split as analyze.py (no matplotlib import here)
            groups = sorted({r.get("pair_id") or r["id"] for r in rows_}); rr = random.Random(seed); rr.shuffle(groups)
            calset = set(groups[: len(groups) // 2])
            return [(r.get("pair_id") or r["id"]) in calset for r in rows_]
        rows = []
        for t in ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "q_defect_root", "p_uph_anomaly", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]:
            rs = [json.loads(l) for l in open(f"/root/data/synthetic/{t}.jsonl", encoding="utf-8")]
            cal = group_split(rs, 0)
            rows += [r for r, c in zip(rs, cal) if not c]
        jobs = [(tr.render(r["state"], r["question"]), letters_for(len(r["question"]["options"]))) for r in rows]
        t0 = time.perf_counter()
        if backend == "sglang":
            for i in range(0, len(jobs), 32):
                batch_read_option_probs_sglang(base_url, [j[0] for j in jobs[i:i + 32]], [j[1] for j in jobs[i:i + 32]], session=sess)
        else:
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda pl: read(base_url, pl[0], pl[1], session=requests.Session()), jobs))
        wall = time.perf_counter() - t0
        out["results"]["S3_d0_test_1001"] = {"wall_s": wall, "decisions": len(jobs), "decisions_per_s": len(jobs) / wall}
        print("S3", json.dumps(out["results"]["S3_d0_test_1001"]), flush=True)

    with open(os.path.join(out_dir, out_name), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if kw.get("commit"):
        kw["commit"]()
    return {k: (v.get("summary") or v) for k, v in out["results"].items()}
