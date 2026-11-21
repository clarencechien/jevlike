"""tw2 (docs/handoff-tw2-decision-apis.md): hosted decision APIs on the paper's jobs (A) or on data/tw2 disputes (B).

Models (keys from env only, never written anywhere):
  msft-1.1   OpenRouter POST /api/alpha/decisions  model microsoft/microsoft-decision-1.1   (env or_key)
  msft-1.0   OpenRouter                            model microsoft/microsoft-decision-1
  gemini-d1  Google AI Studio POST /v1beta/decisions  model gemini-decision-1             (env gemini_key)
Body: the System One shape the paper's run.py sent (state + questions; noul / choice with criteria id->label), except that
AI Studio calls the state field `input`. Rows mirror results/modal/tw/main/<m>/systemone.jsonl so bench/analyze_tw.py logic applies.

  python3 bench/tw2_api.py --model msft-1.1 --jobs data/tw/jobs.jsonl --out results/modal/tw2/a --arm api [--exps E1,E2,E3,E4,E5]
                           [--reps-e1-none 3] [--limit N] [--workers 6]
  --arm api_s2t converts Simplified-Chinese jobs (lang zh_cn) to Traditional with OpenCC s2twp before sending.
Out: <out>/<model>/<arm>.jsonl (+ _body_example.json, _errors.jsonl)
"""
import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

MODELS = {
    "msft-1.1": ("or", "microsoft/microsoft-decision-1.1"),
    "msft-1.0": ("or", "microsoft/microsoft-decision-1"),
    "gemini-d1": ("gemini", "gemini-decision-1"),
}
OR_URL = "https://openrouter.ai/api/alpha/decisions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
_S2T = None


def s2t(text):
    global _S2T
    if text is None:
        return None
    if _S2T is None:
        import opencc
        _S2T = opencc.OpenCC("s2twp")
    return _S2T.convert(text)


def state_text(job):
    """Same as the paper's run.py (and bench/tw_script.py): persona / policy context, a blank line, then the statement."""
    return f"{job['context']}\n\n{job['state']}" if job.get("context") else job["state"]


def convert_job(job):
    j = json.loads(json.dumps(job))
    j["context"] = s2t(j.get("context")); j["state"] = s2t(j["state"])
    for q in j["questions"]:
        q["instructions"] = s2t(q["instructions"])
        for c in q.get("choices", []):
            c["label"] = s2t(c["label"])
    return j


def body(provider, model_id, job):
    qs = {}
    for q in job["questions"]:
        if q["type"] == "noul":
            qs[q["name"]] = {"type": "noul", "instructions": q["instructions"]}
        else:
            qs[q["name"]] = {"type": "choice", "instructions": q["instructions"], "criteria": {c["id"]: c["label"] for c in q["choices"]}}
    key = "input" if provider == "gemini" else "state"
    return {"model": model_id, key: state_text(job), "questions": qs}


def parse(job, res):
    ans, out = res.get("answers", {}), {}
    for q in job["questions"]:
        a = ans.get(q["name"]) or {}
        if a.get("type") == "refusal":
            out[q["name"]] = {"refusal": True}
        elif q["type"] == "noul":
            out[q["name"]] = {"p": a.get("noul")} if a.get("noul") is not None else {"missing": True}
        else:
            by_id = {c["id"]: c["key"] for c in q["choices"]}
            probs = a.get("probabilities") or {}
            out[q["name"]] = {"choice": by_id.get(a.get("choice")), "confidence": a.get("confidence"),
                              "probs": {by_id.get(k, k): v for k, v in probs.items()}}
    return out


class Client:
    def __init__(self, name, max_rps):
        self.provider, self.model_id = MODELS[name]
        self.name = name
        self.tls = threading.local()
        self.lock = threading.Lock(); self.next_t = 0.0; self.gap = 1.0 / max_rps

    def _throttle(self):
        with self.lock:
            now = time.time(); wait = self.next_t - now
            self.next_t = max(now, self.next_t) + self.gap
        if wait > 0:
            time.sleep(wait)

    def call(self, b):
        if not hasattr(self.tls, "s"):
            self.tls.s = requests.Session()
        if self.provider == "or":
            url, headers = OR_URL, {"Authorization": f"Bearer {os.environ['or_key']}", "Content-Type": "application/json"}
        else:
            url, headers = GEMINI_URL, {"x-goog-api-key": os.environ["gemini_key"], "Content-Type": "application/json"}
        delay = 2.0
        for attempt in range(8):
            self._throttle()
            t0 = time.time()
            try:
                r = self.tls.s.post(url, headers=headers, json=b, timeout=120)
            except requests.RequestException as e:  # network: retry
                err = repr(e)[:300]
            else:
                if r.status_code == 200:
                    return r.json(), round((time.time() - t0) * 1000, 1), None
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (408, 429, 500, 502, 503, 524, 529):
                    return None, None, err  # 400 (e.g. safety block) etc.: record, do not retry
            time.sleep(delay + random.random()); delay = min(delay * 2, 60)
        return None, None, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--jobs", default="data/tw/jobs.jsonl")
    ap.add_argument("--out", default="results/modal/tw2/a")
    ap.add_argument("--arm", default="api", choices=["api", "api_s2t"])
    ap.add_argument("--exps", default="E1,E2,E3,E4,E5")
    ap.add_argument("--reps-e1-none", type=int, default=1, help="send E1 no-persona jobs this many times (rep 0..n-1)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rps", type=float, default=3.0)
    a = ap.parse_args()
    exps = set(a.exps.split(","))
    jobs = [json.loads(l) for l in open(a.jobs, encoding="utf-8") if l.strip()]
    jobs = [j for j in jobs if j.get("exp") in exps]
    if a.arm == "api_s2t":
        jobs = [convert_job(j) if j.get("lang") == "zh_cn" else j for j in jobs]
    if a.limit:
        jobs = jobs[:a.limit]
    tasks = []
    for j in jobs:
        n = a.reps_e1_none if (j.get("exp") == "E1" and j.get("persona", "none") == "none") else 1
        tasks += [(j, r) for r in range(n)]
    od = os.path.join(a.out, a.model); os.makedirs(od, exist_ok=True)
    out_path = os.path.join(od, f"{a.arm}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            x = json.loads(l)
            if not x.get("error"):
                done.add((x["job"], x["rep"]))
    tasks = [t for t in tasks if (t[0]["job"], t[1]) not in done]
    cl = Client(a.model, a.rps)
    json.dump(body(cl.provider, cl.model_id, jobs[0]), open(os.path.join(od, "_body_example.json"), "w"), ensure_ascii=False, indent=1)
    lock = threading.Lock(); n_err = 0; t0 = time.time()
    print(f"[{a.model}/{a.arm}] {len(tasks)} calls ({len(done)} already done)", flush=True)

    def one(t):
        j, rep = t
        res, ms, err = cl.call(body(cl.provider, cl.model_id, j))
        return {"model": a.model, "job": j["job"], "rep": rep, "exp": j.get("exp"), "arm": a.arm,
                "answers": parse(j, res) if res else None, "error": err, "ms": ms,
                "served_model": (res or {}).get("model"), "provider": (res or {}).get("provider"),
                "usage": (res or {}).get("usage")}

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(one, tasks)):
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_err += bool(row["error"])
                if (i + 1) % 200 == 0:
                    f.flush(); print(f"  {i + 1}/{len(tasks)} errors {n_err} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{a.model}/{a.arm}] done, errors {n_err}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
