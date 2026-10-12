"""tw (docs/handoff-tw-script.md): the Hsiao (2026) script-dependence jobs (data/tw/jobs.jsonl) through our decision readout.

Backends:
  llama      Gemma 4 on llama-server /completion: each yes/no (noul) question becomes a two-letter question, read in both orders
             (A=yes, A=no); P(true) = mean of the two orders' P(yes). E5 choice questions use the job's own option order as A/B/C.
  systemone  Clef GGUF on llama-server /v1/systemone with the same request body the paper's run.py sent to Cloudflare.

Arms (llama only):
  prod      the production template unchanged (decide/prompt.py SYSTEM + build_messages, Traditional scaffolding)
  native    scaffolding in the job's own script / language; "產線" dropped from the system prompt
  prod_s2t  prod, with zh_cn jobs converted to Traditional (OpenCC s2twp) before reading -- the client-side mitigation

Records follow the paper's results/<model>.jsonl schema ({"job", "rep", "exp", "answers": {q: {"p"} | {"choice", "probs"}}}),
so third_party/tw-sovereignty/analyze.py aggregates them unchanged. Extra per-question fields: p_yes_first, p_no_first, mass.

args: "--arms prod,native,prod_s2t --reps 2 --native-reps 1 --workers 4 --cache 1 --limit 0 --exps E1,E2,E3,E4,E5"
  --cache 0 sends cache_prompt=false (post-hoc determinism check, results/23-tw-script.md §G4)
Output: <out_dir>/<arm>.jsonl (llama) or <out_dir>/systemone.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.client import read_option_probs  # noqa: E402
from decide.prompt import SYSTEM, TemplateRenderer, build_messages  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/root/data" if os.path.isdir("/root/data") else os.path.join(REPO, "data")
JOBS = os.path.join(DATA_ROOT, "tw", "jobs.jsonl")

YES_NO = {"zh_tw": ("是", "否"), "zh_cn": ("是", "否"), "en": ("Yes", "No")}
# native arm: the production scaffolding with 產線 dropped, then OpenCC t2s for zh_cn; English = v8 JevBench scaffolding
NATIVE = {
    "zh_tw": {"system": "你是決策引擎。只回答一個字母。", "state": "【狀態】", "question": "【問題】", "answer": "只回答代表正確選項的字母。"},
    "zh_cn": {"system": "你是决策引擎。只回答一个字母。", "state": "【状态】", "question": "【问题】", "answer": "只回答代表正确选项的字母。"},
    "en": {"system": "You are a decision engine. Answer with exactly one letter.", "state": "[State]", "question": "[Question]",
           "answer": "Answer with the letter of the correct option only."},
}


def state_text(job):
    """Same as the paper's run.py: persona / policy context, a blank line, then the statement or post."""
    return f"{job['context']}\n\n{job['state']}" if job.get("context") else job["state"]


def native_messages(lang, state, instructions, labels):
    s = NATIVE[lang]
    sep = " " if lang == "en" else ""
    opts = "\n".join(f"{L}. {lab}" for L, lab in labels)
    user = f"{s['state']}\n{state}\n\n{s['question']}{sep}{instructions}\n{opts}\n\n{s['answer']}"
    return [{"role": "system", "content": s["system"]}, {"role": "user", "content": user}]


_S2T = None


def s2t(text):
    global _S2T
    if _S2T is None:
        import opencc
        _S2T = opencc.OpenCC("s2twp")
    return _S2T.convert(text) if text else text


class GemmaReader:
    def __init__(self, base_url, arm, cache_prompt=True):
        self.base_url, self.arm, self.cache_prompt = base_url, arm, cache_prompt
        self.tr = TemplateRenderer(base_url)
        self.tls = threading.local()

    def _sess(self):
        if not hasattr(self.tls, "s"):
            self.tls.s = requests.Session()
        return self.tls.s

    def _read(self, lang, state, instructions, labels):
        letters = [L for L, _ in labels]
        if self.arm == "native":
            prompt = self.tr.render_messages(native_messages(lang, state, instructions, labels))
        else:
            q = {"instructions": instructions, "options": dict(labels)}
            prompt = self.tr.render_messages(build_messages(state, q, SYSTEM))
        return read_option_probs(self.base_url, prompt, letters, session=self._sess(), cache_prompt=self.cache_prompt)

    def answer(self, job):
        lang = job["lang"]
        state = state_text(job)
        conv = self.arm == "prod_s2t" and lang == "zh_cn"
        if conv:
            state = s2t(state)
        out = {}
        for q in job["questions"]:
            instr = s2t(q["instructions"]) if conv else q["instructions"]
            if q["type"] == "noul":
                yes, no = YES_NO[lang]
                r1 = self._read(lang, state, instr, [("A", yes), ("B", no)])
                r2 = self._read(lang, state, instr, [("A", no), ("B", yes)])
                if not r1["probs"] or not r2["probs"]:
                    out[q["name"]] = {"missing": True, "top": [r1["top_tokens"][:3], r2["top_tokens"][:3]]}
                    continue
                py1, py2 = r1["probs"].get("A", 0.0), r2["probs"].get("B", 0.0)
                out[q["name"]] = {"p": (py1 + py2) / 2, "p_yes_first": py1, "p_no_first": py2,
                                  "mass": [r1["option_mass"], r2["option_mass"]], "missing_letters": r1["missing"] + r2["missing"]}
            else:
                labels = [(chr(65 + i), s2t(c["label"]) if conv else c["label"]) for i, c in enumerate(q["choices"])]
                r = self._read(lang, state, instr, labels)
                by_letter = {chr(65 + i): c["key"] for i, c in enumerate(q["choices"])}
                probs = {by_letter[L]: p for L, p in r["probs"].items()}
                choice = max(probs, key=probs.get) if probs else None
                out[q["name"]] = {"choice": choice, "confidence": probs.get(choice), "probs": probs, "mass": r["option_mass"]}
        return out


# ---- systemone (Clef): body and parse copied from the paper's run.py (MIT, Shang Nung Hsiao) ----
def s1_body(model, job):
    qs = {}
    for q in job["questions"]:
        if q["type"] == "noul":
            qs[q["name"]] = {"type": "noul", "instructions": q["instructions"]}
        else:
            qs[q["name"]] = {"type": "choice", "instructions": q["instructions"],
                             "criteria": {c["id"]: c["label"] for c in q["choices"]}}
    return {"model": model, "state": state_text(job), "questions": qs}


def s1_parse(job, res):
    ans, out = res.get("answers", {}), {}
    for q in job["questions"]:
        a = ans.get(q["name"]) or {}
        if q["type"] == "noul":
            out[q["name"]] = {"p": a.get("noul")} if a.get("noul") is not None else {"missing": True}
        else:
            by_id = {c["id"]: c["key"] for c in q["choices"]}
            probs = a.get("probabilities") or {}
            if isinstance(probs, list):
                probs = {d.get("value", d.get("label")): d.get("probability") for d in probs}
            out[q["name"]] = {"choice": by_id.get(a.get("choice")), "confidence": a.get("confidence"),
                              "probs": {by_id.get(k, k): v for k, v in probs.items()}}
    return out


class SystemOneReader:
    def __init__(self, base_url, model_name):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model_name = model_name
        self.tls = threading.local()

    def answer(self, job):
        if not hasattr(self.tls, "s"):
            self.tls.s = requests.Session()
        r = self.tls.s.post(self.url, json=s1_body(self.model_name, job), timeout=600)
        r.raise_for_status()
        return s1_parse(job, r.json())


def run_arm(reader, jobs, path, model, rep, workers, commit=None):
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("error"):
                done.add((r["job"], r["rep"]))
    todo = [j for j in jobs if (j["job"], rep) not in done]

    def one(job):
        rec = {"model": model, "job": job["job"], "rep": rep, "exp": job["exp"], "answers": None, "error": None}
        t0 = time.perf_counter()
        try:
            rec["answers"] = reader.answer(job)
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)[:400]
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return rec

    t0 = time.time(); n_err = 0
    with ThreadPoolExecutor(workers) as ex, open(path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(one, todo)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_err += bool(rec["error"])
            if (i + 1) % 200 == 0:
                f.flush()
                print(f"  {os.path.basename(path)} rep {rep}: {i + 1}/{len(todo)} ({time.time() - t0:.0f}s, errors {n_err})", flush=True)
                if commit:
                    commit()
    if commit:
        commit()
    return {"path": path, "rep": rep, "n": len(todo), "errors": n_err, "s": round(time.time() - t0, 1)}


def main(base_url, out_dir, args="", commit=None, model="26b", backend="llama", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    workers = int(a.get("--workers", 4))
    limit = int(a.get("--limit", 0))
    exps = set(a.get("--exps", "E1,E2,E3,E4,E5").split(","))
    jobs = [json.loads(l) for l in open(JOBS, encoding="utf-8")]
    jobs = [j for j in jobs if j["exp"] in exps and not j.get("image")][: limit or None]
    os.makedirs(out_dir, exist_ok=True)
    out = {"model": model, "n_jobs": len(jobs), "runs": []}
    if backend == "systemone":
        reader = SystemOneReader(base_url, a.get("--model-name", "clef"))
        for rep in range(int(a.get("--reps", 2))):
            out["runs"].append(run_arm(reader, jobs, os.path.join(out_dir, "systemone.jsonl"), model, rep, workers, commit))
        return out
    reps = int(a.get("--reps", 2)); native_reps = int(a.get("--native-reps", 1))
    for arm in a.get("--arms", "prod,native,prod_s2t").split(","):
        reader = GemmaReader(base_url, arm, cache_prompt=a.get("--cache", "1") == "1")
        out["template_honored_kwargs"] = reader.tr.server_honored_kwargs
        for rep in range(reps if arm == "prod" else native_reps):
            print(f"[{model}] arm {arm} rep {rep}: {len(jobs)} jobs", flush=True)
            out["runs"].append({"arm": arm, **run_arm(reader, jobs, os.path.join(out_dir, f"{arm}.jsonl"), model, rep, workers, commit)})
    return out
