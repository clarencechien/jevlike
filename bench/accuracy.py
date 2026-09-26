"""M4 accuracy: read option logprobs for every synthetic row (raw saved), plus JSON-generation control arm.

args: "--tasks a,b --control-n 50 --workers 4 --resume 1 --limit 0 --variant V0|V1|V2"
Outputs: <out_dir>/{task}.jsonl (typed decision, raw logprobs), <out_dir>/control_{task}.jsonl
"""
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.client import chat_json, read_option_probs, read_option_probs_sglang_ids, reader_for  # noqa: E402
from decide.prompt import SYSTEM, VARIANTS, TemplateRenderer, build_messages, letters_for  # noqa: E402

DATA_DIR = "/root/data/synthetic"
ALL_TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "q_defect_root",
             "p_uph_anomaly", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]


def load(task):
    p = os.path.join(DATA_DIR, f"{task}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def done_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    for l in open(path, encoding="utf-8"):
        try:
            ids.add(json.loads(l)["id"])
        except Exception:  # noqa: BLE001
            pass
    return ids


def parse_letter(content, letters):
    if not content:
        return None
    m = re.search(r'"answer"\s*:\s*"\s*([A-J])', content)
    if m and m.group(1) in letters:
        return m.group(1)
    m = re.search(r"\b([A-J])\b", content)
    return m.group(1) if m and m.group(1) in letters else None


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    tasks = a.get("--tasks", ",".join(ALL_TASKS)).split(",")
    control_n = int(a.get("--control-n", 50))
    workers = int(a.get("--workers", 4))
    resume = a.get("--resume", "1") == "1"
    limit = int(a.get("--limit", 0))
    commit = kw.get("commit")
    criteria = json.load(open(a["--criteria"], encoding="utf-8")) if a.get("--criteria") else {}
    global DATA_DIR
    DATA_DIR = a.get("--data-dir", DATA_DIR)
    if a.get("--out-sub"):
        out_dir = os.path.join(out_dir, a["--out-sub"])
        os.makedirs(out_dir, exist_ok=True)
    backend = kw.get("backend", "llama")
    read = reader_for(backend)
    variant = VARIANTS[a.get("--variant", "V0")]  # T1 (handoff v5)
    tr = TemplateRenderer(base_url, static=(backend == "sglang"), **variant)
    # E1 (handoff v6): --ids-dir <dir with {task}.jsonl of llama-server token ids>; SGLang is fed input_ids instead of text
    ids_dir = a.get("--ids-dir")
    ids_map, token_ids, hf_tok = {}, None, None
    if ids_dir:
        assert backend == "sglang", "--ids-dir is for the SGLang backend"
        from decide.labels import check_labels
        token_ids = check_labels(base_url, "sglang")
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(requests.get(f"{base_url}/get_model_info", timeout=30).json()["model_path"])
        except Exception as e:  # noqa: BLE001
            print("hf tokenizer unavailable for diff:", repr(e), flush=True)
        for t in tasks:
            for l in open(os.path.join(ids_dir, f"{t}.jsonl"), encoding="utf-8"):
                d = json.loads(l)
                ids_map[d["id"]] = d
        print(f"[ids] loaded {len(ids_map)} pre-tokenized prompts from {ids_dir}; letter ids {token_ids}", flush=True)
    lock = threading.Lock()
    tls = threading.local()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session()
        return tls.s

    summary = {}
    for task in tasks:
        rows = load(task)
        if task in criteria:
            t = criteria[task]
            q = {"type": t["kind"], "instructions": t["instructions"],
                 "options": {k: {"label": v["label"], "criteria": v["criteria"]} for k, v in t["options"].items()}}
            for r in rows:
                r["question"] = q
            print(f"[{task}] using criteria override from {a['--criteria']}", flush=True)
        if limit:
            rows = rows[:limit]
        letters = letters_for(len(rows[0]["question"]["options"]))
        # ---- typed decision arm
        out_path = os.path.join(out_dir, f"{task}.jsonl")
        skip = done_ids(out_path) if resume else set()
        todo = [r for r in rows if r["id"] not in skip]
        print(f"[{task}] {len(rows)} rows, {len(todo)} to run", flush=True)
        n_ok = 0
        t0 = time.time()

        tokdiff = []
        if ids_dir and hf_tok is not None:  # where do the two tokenizers first disagree? (first 20 rows per task)
            for r in rows[:20]:
                d = ids_map[r["id"]]
                hf_ids = hf_tok.encode(d["prompt"], add_special_tokens=True)
                gg = d["ids"]
                k = next((i for i in range(min(len(gg), len(hf_ids))) if gg[i] != hf_ids[i]), None)
                tokdiff.append({"id": r["id"], "n_llama": len(gg), "n_hf": len(hf_ids), "first_diff_pos": k,
                                "llama_piece": hf_tok.decode(gg[k:k + 3]) if k is not None else None,
                                "hf_piece": hf_tok.decode(hf_ids[k:k + 3]) if k is not None else None})
            json.dump(tokdiff, open(os.path.join(out_dir, f"_tokdiff_{task}.json"), "w"), ensure_ascii=False, indent=1)
            print(f"[{task}] tokenizer diff (first 3): {tokdiff[:3]}", flush=True)

        def one(r):
            prompt = tr.render(r["state"], r["question"])
            for attempt in range(3):
                try:
                    if ids_dir:
                        res = read_option_probs_sglang_ids(base_url, ids_map[r["id"]]["ids"], letters, token_ids, session=sess())
                    else:
                        res = read(base_url, prompt, letters, session=sess())
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        return {"id": r["id"], "error": repr(e)}
                    time.sleep(1)
            chosen = max(res["probs"], key=res["probs"].get) if res["probs"] else None
            return {"id": r["id"], "task": task, "gold": r["gold"], "chosen": chosen, "correct": chosen == r["gold"],
                    "difficulty": r["difficulty"], "lang": r["lang"], "pair_id": r.get("pair_id"), "hard_type": r.get("hard_type"),
                    "probs": res["probs"], "raw_logprobs": res["raw_logprobs"], "missing": res["missing"], "option_mass": res.get("option_mass"),
                    "first_token": res["first_token"], "top_tokens": res["top_tokens"][:5],
                    "latency_ms": res["latency_ms"], "prompt_tokens": res["prompt_tokens"], "server_cache_n": res["server_cache_n"]}

        with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            for i, res in enumerate(ex.map(one, todo)):
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    n_ok += bool(res.get("correct"))
                    if (i + 1) % 100 == 0:
                        f.flush()
                        if commit:
                            commit()
                        print(f"  [{task}] {i + 1}/{len(todo)} acc so far {n_ok / (i + 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        if commit:
            commit()
        # ---- control arm: chat completion generating JSON, easy/hard x control_n each
        ctrl_path = os.path.join(out_dir, f"control_{task}.jsonl")
        skip = done_ids(ctrl_path) if resume else set()
        rng = random.Random(f"ctrl-{task}")
        sel = []
        for diff in ("easy", "hard"):
            pool = [r for r in rows if r["difficulty"] == diff]
            rng.shuffle(pool)
            sel += pool[:control_n]
        sel = [r for r in sel if r["id"] not in skip]
        if backend == "sglang":
            sel = []  # control arm (chat JSON) only on llama-server
        print(f"[{task}] control arm: {len(sel)} rows", flush=True)

        def one_ctrl(r):
            msgs = build_messages(r["state"], r["question"], SYSTEM)
            msgs[-1]["content"] += '\n\n以 JSON 回答：{"answer": "<字母>"}'
            for attempt in range(3):
                try:
                    res = chat_json(base_url, msgs, max_tokens=48, session=sess())
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        return {"id": r["id"], "error": repr(e)}
                    time.sleep(1)
            chosen = parse_letter(res["content"], letters)
            return {"id": r["id"], "task": task, "gold": r["gold"], "chosen": chosen, "correct": chosen == r["gold"],
                    "difficulty": r["difficulty"], "lang": r["lang"], "pair_id": r.get("pair_id"),
                    "content": res["content"], "latency_ms": res["latency_ms"], "usage": res["usage"], "finish_reason": res["finish_reason"]}

        with open(ctrl_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            for res in ex.map(one_ctrl, sel):
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
        if commit:
            commit()
        # quick summary from file
        allrows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        allrows = [r for r in allrows if "correct" in r]
        ctrl = [json.loads(l) for l in open(ctrl_path, encoding="utf-8")]
        ctrl = [r for r in ctrl if "correct" in r]
        summary[task] = {"n": len(allrows), "acc": sum(r["correct"] for r in allrows) / max(1, len(allrows)),
                         "acc_easy": sum(r["correct"] for r in allrows if r["difficulty"] == "easy") / max(1, sum(r["difficulty"] == "easy" for r in allrows)),
                         "acc_hard": sum(r["correct"] for r in allrows if r["difficulty"] == "hard") / max(1, sum(r["difficulty"] == "hard" for r in allrows)),
                         "missing_rate": sum(bool(r["missing"]) for r in allrows) / max(1, len(allrows)),
                         "control_n": len(ctrl), "control_acc": sum(r["correct"] for r in ctrl) / max(1, len(ctrl)),
                         "control_unparsed": sum(r["chosen"] is None for r in ctrl)}
        print(f"[{task}] {json.dumps(summary[task])}", flush=True)
    with open(os.path.join(out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if commit:
        commit()
    return summary
