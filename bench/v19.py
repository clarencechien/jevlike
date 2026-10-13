"""v19 (docs/handoff-v19-l0-numeric-coherence.md): three borrowed readouts on D0, one llama-server, no training.

  E1 l0        : K cyclic rotations per row (all 200 rows per task); per-rotation position probs + option-aligned probs.
                 raw / rot / prior / L0 are computed offline by analyze_v19.py (the prior needs the cal split).
  E2 numeric   : option labels 1..K (N1) or [1]..[K] with a '[' prefill (N2); test split; 10-way task excluded.
  E3 coherence : for two labels per row (raw argmax X, gold-or-runner-up Y) ask "is it L?", the same with A/B swapped,
                 and "is it something other than L?"; m_alarm_severity asks "is it L?" for all three labels.

args: "--mode l0|numeric|coherence --variant N1|N2 --tasks a,b --workers 4 --limit 0 --resume 1"
Out: <out_dir>/{task}.jsonl, <out_dir>/_summary.json (+ _labels.json for numeric)
"""
import itertools
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import ALL_TASKS, done_ids, load  # noqa: E402
from decide.client import read_option_probs  # noqa: E402
from decide.labels import check_labels_llama  # noqa: E402
from decide.prompt import TemplateRenderer, letters_for, render_question  # noqa: E402

E3_TASKS = ["m_alarm_severity", "m_alarm_category", "q_spc_action", "q_defect_root", "x_ticket_route"]
DIGITS = "123456789"
NUMERIC = {"N1": {"system": "你是產線決策引擎。只回答一個數字。", "answer": "只回答代表正確選項的數字。", "prefill": ""},
           "N2": {"system": "你是產線決策引擎。以 [數字] 回答。", "answer": "以 [數字] 回答，只輸出這個。", "prefill": "["}}


def test_ids(rows, seed=0):
    """Same split as bench/verify.py::group_split_test_ids (grouped by pair_id, seed 0)."""
    groups = sorted({r.get("pair_id") or r["id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    return {r["id"] for r in rows if (r.get("pair_id") or r["id"]) not in cal}


def _read(base_url, prompt, labels, session, slot, tries=3):
    for attempt in range(tries):
        try:
            return read_option_probs(base_url, prompt, labels, session=session, slot_id=slot)
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(1)


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    mode = a.get("--mode", "l0"); workers = int(a.get("--workers", 4)); limit = int(a.get("--limit", 0))
    resume = a.get("--resume", "1") == "1"; variant = a.get("--variant", "N1"); commit = kw.get("commit")
    default_tasks = {"l0": ALL_TASKS, "numeric": [t for t in ALL_TASKS if t != "x_10way_intent"], "coherence": E3_TASKS}[mode]
    tasks = a.get("--tasks", ",".join(default_tasks)).split(",")
    os.makedirs(out_dir, exist_ok=True)
    tr = TemplateRenderer(base_url)
    lock = threading.Lock(); tls = threading.local(); slot_counter = itertools.count()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session(); tls.slot = next(slot_counter) % workers
        return tls.s, tls.slot

    if mode == "numeric":
        chk = check_labels_llama(base_url, letters=DIGITS)
        json.dump(chk, open(os.path.join(out_dir, "_labels.json"), "w"), ensure_ascii=False, indent=1)
        print(f"[labels] digits single-token check: {chk}", flush=True)

    summary = {}
    for task in tasks:
        rows = load(task)
        te = test_ids(rows)
        if mode != "l0":
            rows = [r for r in rows if r["id"] in te]
        if limit:
            rows = rows[:limit]
        K = len(rows[0]["question"]["options"]); letters = letters_for(K)
        out_path = os.path.join(out_dir, f"{task}.jsonl")
        skip = done_ids(out_path) if resume else set()
        todo = [r for r in rows if r["id"] not in skip]
        print(f"[{task}] mode={mode} K={K} {len(rows)} rows, {len(todo)} to run", flush=True)
        t0 = time.time(); n_ok = 0

        def one_l0(r):
            s, slot = sess(); opts = list(r["question"]["options"].values()); rots = []
            for rr in range(K):
                order = [(i - rr) % K for i in range(K)]  # position i holds option order[i]; option j sits at (j + rr) % K
                q = dict(r["question"], options={letters[i]: opts[order[i]] for i in range(K)})
                res = _read(base_url, tr.render(r["state"], q), letters, s, slot)
                pos = res["probs"]; lp = res["raw_logprobs"]
                rots.append({"r": rr, "pos_probs": pos, "pos_logp": {L: lp.get(L) for L in letters},
                             "aligned_probs": {letters[j]: pos.get(letters[(j + rr) % K], 0.0) for j in range(K)},
                             "aligned_logp": {letters[j]: lp.get(letters[(j + rr) % K]) for j in range(K)},
                             "option_mass": res.get("option_mass"), "missing": res["missing"], "latency_ms": res["latency_ms"]})
            raw = rots[0]["aligned_probs"]; ch = max(raw, key=raw.get) if raw else None
            return {"id": r["id"], "task": task, "gold": r["gold"], "K": K, "split": "test" if r["id"] in te else "cal",
                    "difficulty": r["difficulty"], "lang": r["lang"], "pair_id": r.get("pair_id"), "chosen_raw": ch, "correct_raw": ch == r["gold"],
                    "rotations": rots}

        def one_numeric(r):
            s, slot = sess(); labels = list(DIGITS[:K]); v = NUMERIC[variant]
            q = dict(r["question"], options={d: o for d, o in zip(labels, r["question"]["options"].values())})
            msgs = [{"role": "system", "content": v["system"]},
                    {"role": "user", "content": f"【狀態】\n{r['state']}\n\n{render_question(q)}\n\n{v['answer']}"}]
            res = _read(base_url, tr.render_messages(msgs, prefill=v["prefill"]), labels, s, slot)
            probs = {letters[labels.index(d)]: p for d, p in res["probs"].items()}
            lp = {letters[labels.index(d)]: v2 for d, v2 in res["raw_logprobs"].items()}
            ch = max(probs, key=probs.get) if probs else None
            return {"id": r["id"], "task": task, "gold": r["gold"], "variant": variant, "chosen": ch, "correct": ch == r["gold"],
                    "difficulty": r["difficulty"], "lang": r["lang"], "pair_id": r.get("pair_id"), "probs": probs, "raw_logprobs": lp,
                    "missing": res["missing"], "option_mass": res.get("option_mass"), "first_token": res["first_token"],
                    "top_tokens": res["top_tokens"][:5], "latency_ms": res["latency_ms"], "prompt_tokens": res["prompt_tokens"]}

        def one_coh(r):
            s, slot = sess(); instr = r["question"]["instructions"]; opts = r["question"]["options"]
            raw = _read(base_url, tr.render(r["state"], r["question"]), letters, s, slot)
            pk = raw["probs"]; X = max(pk, key=pk.get)
            ranked = sorted(pk, key=pk.get, reverse=True)
            Y = r["gold"] if r["gold"] != X else (ranked[1] if len(ranked) > 1 else X)
            want = sorted({X, Y} | (set(letters) if task == "m_alarm_severity" else set()), key=letters.index)
            per = {}
            for L in want:
                o = opts[L]; label = o["label"] if isinstance(o, dict) else o
                crit = o.get("criteria") if isinstance(o, dict) else None
                full = f"{label}（{crit}）" if crit else label
                qa = {"instructions": f"{instr}這個狀況屬於「{full}」嗎？", "options": {"A": "是", "B": "否"}}
                qs = {"instructions": qa["instructions"], "options": {"A": "否", "B": "是"}}
                qn = {"instructions": f"{instr}這個狀況屬於「{label}」以外的其他類別嗎？", "options": {"A": "是", "B": "否"}}
                ra = _read(base_url, tr.render(r["state"], qa), ["A", "B"], s, slot)
                rs = _read(base_url, tr.render(r["state"], qs), ["A", "B"], s, slot)
                rn = _read(base_url, tr.render(r["state"], qn), ["A", "B"], s, slot) if L in (X, Y) else None
                per[L] = {"p_yes": ra["probs"].get("A"), "lp_yes": ra["raw_logprobs"], "mass_yes": ra.get("option_mass"),
                          "p_yes_swapped": rs["probs"].get("B"), "lp_yes_swapped": rs["raw_logprobs"],
                          "p_not": rn["probs"].get("A") if rn else None, "lp_not": rn["raw_logprobs"] if rn else None,
                          "mass_not": rn.get("option_mass") if rn else None}
            return {"id": r["id"], "task": task, "gold": r["gold"], "K": K, "difficulty": r["difficulty"], "pair_id": r.get("pair_id"),
                    "X": X, "Y": Y, "p_k": pk, "lp_k": raw["raw_logprobs"], "correct_raw": X == r["gold"], "labels": per}

        fn = {"l0": one_l0, "numeric": one_numeric, "coherence": one_coh}[mode]
        key = {"l0": "correct_raw", "numeric": "correct", "coherence": "correct_raw"}[mode]

        def safe(r):
            try:
                return fn(r)
            except Exception as e:  # noqa: BLE001
                return {"id": r["id"], "task": task, "error": repr(e)}

        with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            for i, res in enumerate(ex.map(safe, todo)):
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n"); n_ok += bool(res.get(key))
                    if (i + 1) % 50 == 0 or i + 1 == len(todo):
                        f.flush()
                        if commit:
                            commit()
                        print(f"  [{task}] {i + 1}/{len(todo)} raw acc {n_ok / (i + 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        summary[task] = {"n": len(todo), "acc_raw": n_ok / max(1, len(todo)), "s": round(time.time() - t0, 1)}
    json.dump({"mode": mode, "variant": variant if mode == "numeric" else None, "tasks": summary}, open(os.path.join(out_dir, "_summary.json"), "w"), indent=1)
    return summary
