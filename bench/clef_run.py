"""v10 (docs/handoff-v10-clef.md): run Cloudflare Clef / Clef-flash in-process (transformers + official joint_schema_model.py).

The joint schema head reads every position's final hidden state, so no serving engine (vLLM / SGLang / llama.cpp)
can produce its output; transformers is the only runtime. Quantization is done in transformers:
  bf16 (reference), fp8 (FineGrainedFP8Config, block 128; lm_head / embeddings / vision kept bf16), nf4 (bitsandbytes).

Suites (each writes <out_dir>/<suite>/<name>.jsonl):
  smoke    50 D0 rows (5 per task, seed 0)                  -> reference / equivalence check
  d0       2000 D0 rows (label ids) + choice rows with letter ids fwd / rev (order arm)
  blind    D2 Gemini-written blind rows
  v9       g_input_guard, t_tool_gate, e_answer_score + v9 blind g / t
  rerank   200 queries x 20 passages, one noul record each ("能回答嗎")
  jevbench 231 public items (native ids) + choice items with letter ids fwd / rev
  packed   alarm family: every alarm row's state with all 3 alarm questions in one record (gold = origin question)
  latency  batch 1, 200 D0 rows single question + 200 packed (K=3), after warm-up; peak memory

Usage (inside the Modal container or on GB10):
  python3 bench/clef_run.py --model /models/clef/clef-flash --quant bf16 --out /results/clef/flash-bf16 --suites smoke,d0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from decide.clef_adapter import QID, jev_record, packed_record, row_record, to_ours  # noqa: E402

DATA = "/root/data" if os.path.isdir("/root/data") else os.path.join(REPO, "data")
D0_TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "q_defect_root",
            "p_uph_anomaly", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]
ALARM = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch"]
RERANK_Q = "這段文件能直接回答上面工程師的問題嗎？"  # same wording as bench/rerank.py
KEEP = ("id", "task", "gold", "difficulty", "hard_type", "lang", "pair_id", "source")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def d0_rows():
    return [r for t in D0_TASKS for r in jl(os.path.join(DATA, "synthetic", f"{t}.jsonl"))]


# ------------------------------------------------------------------ model
def load(model_path, quant):
    import torch
    sys.path.insert(0, model_path)
    import joint_schema_model as jsm
    kw = {}
    if quant in ("fp8", "fp8-nola"):
        from transformers import FineGrainedFP8Config
        skip = ["lm_head", r"model\.visual.*", "in_proj_a", "in_proj_b"]  # in_proj_a/b: out dim = heads, < block
        if quant == "fp8-nola":  # diagnostic: keep every linear-attention projection in bf16, quantize the rest
            skip.append(r".*linear_attn.*")
        kw["quantization_config"] = FineGrainedFP8Config(modules_to_not_convert=skip)
    elif quant == "int8":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head", "visual", "in_proj_a", "in_proj_b"])
    elif quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       llm_int8_skip_modules=["lm_head", "visual", "in_proj_a", "in_proj_b"])
    else:
        assert quant == "bf16", quant
    t0 = time.time()
    model, processor = jsm.load_release_model(model_path, device="cuda", dtype=torch.bfloat16, **kw)
    # the head indexes the output embedding matrix by token id; make sure it stayed a plain bf16 tensor
    w = model.language_model.get_output_embeddings().weight
    assert w.dtype == torch.bfloat16, w.dtype
    return model, processor, jsm, round(time.time() - t0, 1)


class Runner:
    def __init__(self, model, processor, jsm, token_budget):
        import torch
        self.torch, self.model, self.proc, self.jsm, self.budget = torch, model, processor, jsm, token_budget
        tok = processor.tokenizer
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def encode(self, rec):
        return self.jsm.encode_record(self.proc.tokenizer, rec, processor=None)

    def forward(self, encs):
        torch = self.torch
        batch = self.jsm.collate_records(encs, self.pad, torch.device("cuda"))
        with torch.inference_mode():
            logits = self.model(batch)
        out = []
        for enc, rl in zip(encs, logits):
            out.append({q.question_id: (list(q.option_ids), ql.float().softmax(-1).tolist(), ql.float().tolist())
                        for q, ql in zip(enc.questions, rl)})
        return out

    def run(self, recs, log=""):
        """recs: list of clef records -> list of per-record {qid: (option_ids, probs, logits)} plus n_tokens."""
        encs = [self.encode(r) for r in recs]
        order = sorted(range(len(encs)), key=lambda i: len(encs[i].input_ids))
        res = [None] * len(encs)
        i, t0, done = 0, time.time(), 0
        while i < len(order):
            L = len(encs[order[i]].input_ids)
            j = i + 1
            while j < len(order) and (j - i + 1) * len(encs[order[j]].input_ids) <= self.budget and j - i < 32:
                j += 1
            idx = order[i:j]
            for k, o in zip(idx, self.forward([encs[k] for k in idx])):
                res[k] = {"out": o, "n_tokens": len(encs[k].input_ids)}
            done += len(idx); i = j
            if log and (done // 200) != ((done - len(idx)) // 200):
                print(f"  [{log}] {done}/{len(encs)} ({time.time() - t0:.0f}s, last len {L})", flush=True)
        return res


# ------------------------------------------------------------------ suites
def rows_to_out(rows, arm, runner, out_path, id_mode="label"):
    recs, backs = zip(*[row_record(r, id_mode) for r in rows])
    res = runner.run(list(recs), log=os.path.basename(out_path))
    n_ok = n_nan = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r, b, x in zip(rows, backs, res):
            oids, probs, logits = x["out"][QID]
            p = to_ours(oids, probs, b[QID])
            bad = any(v != v for v in p.values())  # NaN guard: FP8 produced all-NaN logits once; never score those
            n_nan += bad
            chosen = None if bad else max(p, key=p.get)
            n_ok += chosen == r["gold"]
            f.write(json.dumps({**{k: r.get(k) for k in KEEP}, "arm": arm, "nan": bad, "qtype": r["question"]["type"], "chosen": chosen,
                                "correct": chosen == r["gold"], "probs": p,
                                "logits": dict(zip([b[QID][o] for o in oids], logits)), "prompt_tokens": x["n_tokens"]},
                               ensure_ascii=False) + "\n")
    return {"n": len(rows), "acc": round(n_ok / max(len(rows), 1), 4), "n_nan": n_nan}


def suite_smoke(runner, out):
    rng = random.Random(0)
    rows = []
    for t in D0_TASKS:
        rr = [r for r in d0_rows() if r["task"] == t]
        rows += rng.sample(rr, 5)
    return {"smoke": rows_to_out(rows, "label", runner, os.path.join(out, "smoke.jsonl"))}


def suite_d0(runner, out):
    rows = d0_rows()
    s = {"d0": rows_to_out(rows, "label", runner, os.path.join(out, "d0.jsonl"))}
    ch = [r for r in rows if r["question"]["type"] == "choice"]
    for m in ("letter_fwd", "letter_rev"):
        s[f"d0_{m}"] = rows_to_out(ch, m, runner, os.path.join(out, f"d0_{m}.jsonl"), id_mode=m)
    return s


def suite_blind(runner, out):
    rows = [r for t in D0_TASKS for r in jl(os.path.join(DATA, "blind", "D2", f"{t}.jsonl"))]
    return {"blind": rows_to_out(rows, "label", runner, os.path.join(out, "blind.jsonl"))}


def suite_v9(runner, out):
    s = {}
    for t in ("g_input_guard", "t_tool_gate", "e_answer_score"):
        s[t] = rows_to_out(jl(os.path.join(DATA, "v9", f"{t}.jsonl")), "label", runner, os.path.join(out, f"{t}.jsonl"))
    for t in ("g_input_guard", "t_tool_gate"):
        s[f"blind_{t}"] = rows_to_out(jl(os.path.join(DATA, "v9", "blind", f"{t}.jsonl")), "label", runner,
                                      os.path.join(out, f"blind_{t}.jsonl"))
    return s


def suite_rerank(runner, out):
    sys.path.insert(0, REPO)
    from bench.rerank import metrics, state_for
    corpus = {r["pid"]: r for r in jl(os.path.join(DATA, "v9", "r_corpus.jsonl"))}
    qs = jl(os.path.join(DATA, "v9", "r_rerank.jsonl"))
    recs, keys = [], []
    for qi, q in enumerate(qs):
        for ci, c in enumerate(q["candidates"]):
            recs.append({"id": f"{q['id']}|{c['pid']}", "state": state_for(q["query"], corpus[c["pid"]]["text"]),
                         "questions": {QID: {"type": "noul", "instructions": RERANK_Q}}})
            keys.append((qi, ci))
    res = runner.run(recs, log="rerank")
    rows = [{"id": q["id"], "query": q["query"], "candidates": [dict(c) for c in q["candidates"]]} for q in qs]
    for (qi, ci), x in zip(keys, res):
        oids, probs, _ = x["out"][QID]
        rows[qi]["candidates"][ci]["score"] = dict(zip(oids, probs))["true"]
    with open(os.path.join(out, "r_rerank.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"rerank": metrics(rows)}


def suite_jevbench(runner, out):
    tasks = [dict(json.loads(l), tier=t) for t in ("original", "easy", "hard")
             for l in open(os.path.join(DATA, "jevbench", f"{t}.jsonl"), encoding="utf-8") if l.strip()]
    s = {}
    for mode in ("native", "letter_fwd", "letter_rev"):
        tt = tasks if mode == "native" else [t for t in tasks if t["question"]["type"] == "choice"]
        recs, backs = zip(*[jev_record(t, mode) for t in tt])
        res = runner.run(list(recs), log=f"jevbench-{mode}")
        n_ok = 0
        with open(os.path.join(out, f"jevbench_{mode}.jsonl"), "w", encoding="utf-8") as f:
            for t, b, x in zip(tt, backs, res):
                oids, probs, logits = x["out"][QID]
                p = to_ours(oids, probs, b[QID])
                pred = max(p, key=p.get)
                n_ok += pred == str(t["expected"])
                f.write(json.dumps({"task_id": t["id"], "tier": t["tier"], "family": t["family"], "type": t["question"]["type"],
                                    "labels": t["labels"], "expected": t["expected"], "predicted": pred,
                                    "correct": pred == str(t["expected"]), "probs": p, "prompt_tokens": x["n_tokens"]},
                                   ensure_ascii=False) + "\n")
        s[f"jevbench_{mode}"] = {"n": len(tt), "acc": round(n_ok / len(tt), 4)}
    return s


def _packed_inputs():
    rows = d0_rows()
    qdef = {t: next(r["question"] for r in rows if r["task"] == t) for t in ALARM}
    al = [r for r in rows if r["task"] in ALARM]
    recs, backs = zip(*[packed_record(r["state"], qdef, r["id"]) for r in al])
    return al, list(recs), list(backs)


def suite_packed(runner, out):
    al, recs, backs = _packed_inputs()
    res = runner.run(recs, log="packed")
    n_ok = 0
    with open(os.path.join(out, "packed.jsonl"), "w", encoding="utf-8") as f:
        for r, b, x in zip(al, backs, res):
            oids, probs, _ = x["out"][r["task"]]
            p = to_ours(oids, probs, b[r["task"]])
            chosen = max(p, key=p.get)
            n_ok += chosen == r["gold"]
            other = {}
            for t in ALARM:
                if t != r["task"]:
                    po = to_ours(*x["out"][t][:2], b[t]); other[t] = max(po, key=po.get)
            f.write(json.dumps({**{k: r.get(k) for k in KEEP}, "arm": "packed3", "chosen": chosen, "correct": chosen == r["gold"],
                                "probs": p, "other_chosen": other, "prompt_tokens": x["n_tokens"]}, ensure_ascii=False) + "\n")
    return {"packed": {"n": len(al), "acc": round(n_ok / len(al), 4)}}


def suite_latency(runner, out, n=200):
    import numpy as np
    torch = runner.torch
    rng = random.Random(1)
    rows = rng.sample(d0_rows(), n)
    _, precs, _ = _packed_inputs()
    precs = rng.sample(precs, n)
    s = {}
    for name, recs in (("single", [row_record(r)[0] for r in rows]), ("packed3", precs)):
        encs = [runner.encode(r) for r in recs]
        # Triton kernels (fla) autotune per input length: pass 1 sees every length once (cold), pass 2 is timed (warm)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        cold, ms = [], []
        for e in encs:
            t0 = time.perf_counter(); runner.forward([e]); torch.cuda.synchronize()
            cold.append((time.perf_counter() - t0) * 1000)
        for e in encs:
            t0 = time.perf_counter(); runner.forward([e]); torch.cuda.synchronize()
            ms.append((time.perf_counter() - t0) * 1000)
        toks = [len(e.input_ids) for e in encs]
        s[name] = {"n": n, "p50_ms": round(float(np.percentile(ms, 50)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1),
                   "cold_p50_ms": round(float(np.percentile(cold, 50)), 1), "cold_p95_ms": round(float(np.percentile(cold, 95)), 1),
                   "mean_tokens": round(float(np.mean(toks)), 1), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        print(f"  [latency] {name} {s[name]}", flush=True)
    return {"latency": s}


def suite_profile(runner, out):
    """Same record 20x (per-call ms: a slow first call then fast = per-shape autotune), plus top CUDA ops of one call."""
    torch = runner.torch
    r = d0_rows()[0]
    e = runner.encode(row_record(r)[0])
    ms = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter(); runner.forward([e]); torch.cuda.synchronize()
        ms.append(round((time.perf_counter() - t0) * 1000, 1))
    # a second length, to see whether a new shape is slow again
    e2 = runner.encode(row_record(d0_rows()[5])[0])
    ms2 = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter(); runner.forward([e2]); torch.cuda.synchronize()
        ms2.append(round((time.perf_counter() - t0) * 1000, 1))
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        runner.forward([e]); torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    open(os.path.join(out, "profile.txt"), "w").write(table + "\n\n" + cpu_table)
    print(table, flush=True)
    return {"profile": {"tokens": len(e.input_ids), "ms": ms, "tokens2": len(e2.input_ids), "ms2": ms2}}


SUITES = {"smoke": suite_smoke, "d0": suite_d0, "blind": suite_blind, "v9": suite_v9, "rerank": suite_rerank,
          "jevbench": suite_jevbench, "packed": suite_packed, "latency": suite_latency,
          "profile": suite_profile}


def main(model_path, quant, out, suites, token_budget=16384, commit=None):
    import torch
    os.makedirs(out, exist_ok=True)
    model, processor, jsm, load_s = load(model_path, quant)
    runner = Runner(model, processor, jsm, token_budget)
    meta = {"model_path": model_path, "quant": quant, "load_s": load_s, "gpu": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__), "mem_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}
    try:
        import transformers
        meta["transformers"] = str(transformers.__version__)
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        meta["fast_path"] = bool(getattr(mq, "is_fast_path_available", False))
        meta["fla"] = mq.chunk_gated_delta_rule is not None
    except Exception as e:  # noqa: BLE001
        meta["meta_err"] = repr(e)
    print("[clef]", meta, flush=True)
    sp = os.path.join(out, "_summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    summary["meta"] = meta
    for name in suites:
        t0 = time.time()
        summary[name] = {**SUITES[name](runner, out), "s": round(time.time() - t0, 1)}
        print(f"[clef] {name} {summary[name]}", flush=True)
        json.dump(summary, open(sp, "w"), ensure_ascii=False, indent=1)
        if commit:
            commit()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="bf16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="smoke")
    ap.add_argument("--token-budget", type=int, default=16384)
    a = ap.parse_args()
    main(a.model, a.quant, a.out, a.suites.split(","), a.token_budget)
