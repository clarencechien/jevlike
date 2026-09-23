"""M3 latency: L1–L7. Writes <out_dir>/latency.json (raw samples + summary).

All numbers are on the Modal GPU (L4 by default) = conservative upper bound for GB10.
args: "--n 200 --warmup 20" (space separated key value pairs)
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
from decide.client import chat_json, read_option_probs  # noqa: E402
from decide.prompt import SYSTEM, TemplateRenderer, build_messages, letters_for  # noqa: E402

LINES_ZH = [
    "SMT-L{l} 回焊爐 R-0{r} 第 {z} 區溫度 {t}°C，設定 {s}°C，輸送速度 {v} cm/min",
    "SMT-L{l} 貼片機 NXT-0{r} head{z} 吸嘴吸著率 {p}%，拋料 {n} 顆/小時",
    "SMT-L{l} 印刷機 DEK-0{r} 錫膏厚度平均 {t} µm，CPK {c}，刮刀壓力 {n} kg",
    "SMT-L{l} AOI-0{r} 過去 {n} 分鐘檢出 {z} 片 NG，NG 率 {p}%",
    "工單 WO-202609{d}-{k} 目前完成 {n}/{m} 片，UPH {u}，目標 {g}",
    "SMT-L{l} feeder slot {z} 料號 R123-0402-10K 剩餘 {n} 顆，預估 {t} 分鐘用完",
    "SMT-L{l} 廠務空壓 {c} bar，氮氣純度 {p}%，環境溫度 {t}°C 濕度 {n}%",
    "SMT-L{l} SPI-0{r} 錫膏體積平均 {p}%，偏移 {z} µm，橋接 {n} 處",
]
LINES_EN = [
    "E{k} NOZZLE VACUUM LOW HEAD{z} NXT-0{r} PICKUP {p}%",
    "REFLOW R-0{r} ZONE{z} TEMP {t}C SET {s}C BELT {v}CM/MIN",
    "F-{k} FEEDER SLOT {z} LOW {n} PCS REMAINING",
    "SPC DEK-0{r} PASTE HEIGHT AVG {t}UM CPK {c}",
]


def make_state(rng, target_tokens, tokenize):
    def line():
        tpl = rng.choice(LINES_ZH if rng.random() < 0.8 else LINES_EN)
        return tpl.format(l=rng.randint(1, 6), r=rng.randint(1, 3), z=rng.randint(1, 10), t=rng.randint(150, 260),
                          s=rng.randint(200, 250), v=rng.randint(60, 90), p=rng.randint(80, 100), n=rng.randint(1, 500),
                          c=round(rng.uniform(0.8, 2.0), 2), d=rng.randint(10, 28), k=rng.randint(1000, 9999),
                          m=rng.randint(500, 2000), u=rng.randint(900, 1300), g=rng.randint(1000, 1200))
    parts = [line()]
    per_line = max(8, tokenize(parts[0]))
    parts += [line() for _ in range(max(0, target_tokens // per_line - 1))]
    while tokenize("\n".join(parts)) < target_tokens:
        parts.append(line())
    return "\n".join(parts)


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(samples, key="latency_ms"):
    xs = [s[key] for s in samples if s.get(key) is not None]
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "p99": pct(xs, 0.99),
            "mean": statistics.fmean(xs) if xs else None,
            "prompt_tokens_mean": statistics.fmean([s["prompt_tokens"] for s in samples if s.get("prompt_tokens")] or [0]),
            "server_prompt_ms_p50": pct([s["server_prompt_ms"] for s in samples if s.get("server_prompt_ms") is not None], 0.5),
            "cache_n_mean": statistics.fmean([s["server_cache_n"] for s in samples if s.get("server_cache_n") is not None] or [0])}


def question(n_opts):
    labels = ["不急", "盡快", "停線", "呼叫 QE", "抽檢", "持續監控", "派工", "換線", "缺料", "待修"][:n_opts]
    return {"instructions": "依上述狀態，最合適的處置是？", "options": {L: lab for L, lab in zip(letters_for(n_opts), labels)}}


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    N = int(a.get("--n", 200))
    WARM = int(a.get("--warmup", 20))
    rng = random.Random(42)
    sess = requests.Session()

    def tokenize(text):
        return len(sess.post(f"{base_url}/tokenize", json={"content": text}, timeout=60).json()["tokens"])

    tr = TemplateRenderer(base_url)
    out = {"gpu": os.environ.get("GB10_GPU", "L4"), "n": N, "warmup": WARM, "results": {}}

    def run(name, prompts_letters, n_warm=WARM, cache_prompt=True, note=""):
        samples = []
        for i, (p, L) in enumerate(prompts_letters):
            r = read_option_probs(base_url, p, L, cache_prompt=cache_prompt, session=sess)
            if i >= n_warm:
                samples.append({k: r[k] for k in ("latency_ms", "prompt_tokens", "tokens_cached", "server_prompt_ms", "server_predicted_ms", "server_cache_n", "missing")})
        out["results"][name] = {"summary": summarize(samples), "note": note, "samples": samples}
        print(f"{name}: {json.dumps(out['results'][name]['summary'])}", flush=True)

    def states(n, tokens):
        return [make_state(rng, tokens, tokenize) for _ in range(n)]

    # ---- L1: single question, ~100-token state, 2 options, distinct states (prefix cache only hits template)
    st100 = states(N + WARM, 100)
    q2 = question(2)
    run("L1_single_100tok_2opt", [(tr.render(s, q2), letters_for(2)) for s in st100], note="每次不同 state；只有模板前綴命中 cache")
    # same prompt repeated (full cache hit) as reference floor
    p0 = tr.render(st100[0], q2)
    run("L1_ref_same_prompt_repeated", [(p0, letters_for(2))] * (N + WARM), note="同一 prompt 重複；全 cache 命中，= 1 token decode 的地板")
    # no cache at all
    run("L1_ref_nocache", [(tr.render(s, q2), letters_for(2)) for s in st100], cache_prompt=False, note="cache_prompt=false，全部重算")

    # ---- L2: state length
    for tok in (500, 2000):
        run(f"L2_state_{tok}tok", [(tr.render(s, q2), letters_for(2)) for s in states(N + WARM, tok)])
    out["results"]["L2_state_100tok"] = out["results"]["L1_single_100tok_2opt"]

    # ---- L3: option count
    for k in (5, 10):
        qk = question(k)
        run(f"L3_{k}opt", [(tr.render(s, qk), letters_for(k)) for s in st100])
    out["results"]["L3_2opt"] = out["results"]["L1_single_100tok_2opt"]

    # ---- L4: N questions sharing one state, cache on/off. Measure total wall time per state.
    qs = [question(2), question(3), question(4), question(5), question(2), question(3), question(4), question(5), question(2), question(3)]
    for i, q in enumerate(qs):
        q["instructions"] = f"問題 {i + 1}：" + q["instructions"]
    for cache in (True, False):
        for nq in (1, 5, 10):
            n_states = max(20, N // nq)
            sts = states(n_states + 5, 300)
            totals, per_call = [], []
            for si, s in enumerate(sts):
                t0 = time.perf_counter()
                for q in qs[:nq]:
                    r = read_option_probs(base_url, tr.render(s, q), letters_for(len(q["options"])), cache_prompt=cache, session=sess)
                    per_call.append({"latency_ms": r["latency_ms"], "prompt_tokens": r["prompt_tokens"], "server_prompt_ms": r["server_prompt_ms"], "server_cache_n": r["server_cache_n"]})
                if si >= 5:
                    totals.append({"latency_ms": (time.perf_counter() - t0) * 1000, "prompt_tokens": None})
            name = f"L4_shared_state_{nq}q_cache_{'on' if cache else 'off'}"
            out["results"][name] = {"summary": summarize(totals), "per_call": summarize(per_call), "samples": totals,
                                    "note": "total ms for all N questions on one ~300-token state"}
            print(f"{name}: total {json.dumps(out['results'][name]['summary'])} per_call {json.dumps(out['results'][name]['per_call'])}", flush=True)

    # ---- L5: concurrency
    for conc in (1, 4, 8):
        sts = states(N + WARM, 100)
        prompts = [tr.render(s, q2) for s in sts]
        samples = []
        lock = threading.Lock()

        def one(p):
            r = read_option_probs(base_url, p, letters_for(2), session=requests.Session())
            with lock:
                samples.append({"latency_ms": r["latency_ms"], "prompt_tokens": r["prompt_tokens"], "server_prompt_ms": r["server_prompt_ms"], "server_cache_n": r["server_cache_n"]})
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            list(ex.map(one, prompts))
        wall = time.perf_counter() - t0
        summ = summarize(samples[WARM:])
        summ["throughput_rps"] = len(prompts) / wall
        out["results"][f"L5_concurrency_{conc}"] = {"summary": summ, "samples": samples[WARM:]}
        print(f"L5_concurrency_{conc}: {json.dumps(summ)}", flush=True)

    # ---- L6: background generation load (512-token generation running continuously)
    stop = threading.Event()
    gen_stats = {"n": 0, "tokens": 0}

    def bg_gen():
        s2 = requests.Session()
        while not stop.is_set():
            try:
                r = s2.post(f"{base_url}/completion", json={"prompt": "請詳細說明 SMT 產線回焊爐溫度曲線的設定原則與常見異常的處理方式：", "n_predict": 512, "temperature": 0.7, "cache_prompt": False}, timeout=600).json()
                gen_stats["n"] += 1
                gen_stats["tokens"] += r.get("tokens_predicted", 0)
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
    th = threading.Thread(target=bg_gen, daemon=True)
    th.start()
    time.sleep(3)
    run("L6_with_bg_generation", [(tr.render(s, q2), letters_for(2)) for s in states(N + WARM, 100)], note="同容器另一執行緒持續生成 512 token")
    stop.set()
    th.join(timeout=120)
    out["results"]["L6_with_bg_generation"]["bg_generation"] = gen_stats
    out["results"]["L6_with_bg_generation"]["slowdown_p50"] = out["results"]["L6_with_bg_generation"]["summary"]["p50"] / out["results"]["L1_single_100tok_2opt"]["summary"]["p50"]

    # ---- L7: same question via chat completions generating JSON
    samples = []
    for i, s in enumerate(states(N + WARM, 100)):
        msgs = build_messages(s, q2, SYSTEM)
        msgs[-1]["content"] += '\n\n以 JSON 回答：{"answer": "<字母>"}'
        r = chat_json(base_url, msgs, max_tokens=32, session=sess)
        if i >= WARM:
            samples.append({"latency_ms": r["latency_ms"], "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                            "completion_tokens": (r.get("usage") or {}).get("completion_tokens"), "content": r["content"]})
    summ = summarize(samples)
    summ["completion_tokens_mean"] = statistics.fmean([x["completion_tokens"] for x in samples if x.get("completion_tokens")] or [0])
    summ["speedup_vs_L1_p50"] = summ["p50"] / out["results"]["L1_single_100tok_2opt"]["summary"]["p50"]
    out["results"]["L7_chat_json"] = {"summary": summ, "samples": samples}
    print(f"L7_chat_json: {json.dumps(summ)}", flush=True)

    with open(os.path.join(out_dir, "latency.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if kw.get("commit"):
        kw["commit"]()
    return {k: v["summary"] for k, v in out["results"].items()}
