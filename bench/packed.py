"""P1 (handoff v7): packed readout on SGLang — K questions about one state in ONE forward pass.

Packed prompt = multi-turn: user(state + q1) / model(placeholder) / user(q2) / model(placeholder) / ... ; the letter
logprobs are read at every placeholder position from `input_token_ids_logprobs` (logprob_start_len=0, input_ids).
Separate arm = current practice: first question alone (warms the shared prefix), then the rest as one list request.

Data: alarm family (m_alarm_category K=5, m_alarm_severity K=3, m_needs_dispatch K=2) — every alarm state gets all
three questions; gold exists only for the state's own task. Optional K=7 speed-only pack adds 4 generic yes/no questions.

args: "--limit 0 --workers 8 --placeholder _ --extra 0 --out-sub P1"
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import load  # noqa: E402
from decide.client import _sglang_parse, batch_read_option_probs_sglang, read_option_probs_sglang  # noqa: E402
from decide.labels import check_labels  # noqa: E402
from decide.prompt import ANSWER_LINE, SYSTEM, TemplateRenderer, letters_for, render_question  # noqa: E402

ALARM = ["m_alarm_category", "m_alarm_severity", "m_needs_dispatch"]
EXTRA = [{"instructions": "這則 alarm 是否需要通知線長？", "options": {"A": "是", "B": "否"}},
         {"instructions": "這則 alarm 是否可能在一小時內重複發生？", "options": {"A": "是", "B": "否"}},
         {"instructions": "這則 alarm 是否影響當班產出？", "options": {"A": "是", "B": "否"}},
         {"instructions": "這則 alarm 是否需要記錄到日報？", "options": {"A": "是", "B": "否"}}]
TURN_U, TURN_END, TURN_M = "<|turn>user\n", "<turn|>\n", "<|turn>model\n<|channel>thought\n<channel|>"


def user_text(state, q, first):
    body = (f"【狀態】\n{state}\n\n" if first else "") + render_question(q) + "\n\n" + ANSWER_LINE["letter"]
    return body


def packed_ids(tok, state, qs, placeholder):
    """Return (input_ids, placeholder_positions). Position p means: logprobs at index p predict the placeholder token."""
    ids = tok.encode("<bos><|turn>system\n" + SYSTEM + TURN_END, add_special_tokens=False)
    pos = []
    for i, q in enumerate(qs):
        ids += tok.encode(TURN_U + user_text(state, q, i == 0) + TURN_END + TURN_M, add_special_tokens=False)
        pos.append(len(ids))
        ids += tok.encode(placeholder, add_special_tokens=False) + tok.encode(TURN_END, add_special_tokens=False)
    return ids, pos


def read_packed(base_url, sess, ids, pos, letter_ids, qs):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
            "logprob_start_len": 0, "token_ids_logprob": letter_ids, "top_logprobs_num": 1, "return_text_in_logprobs": True}
    t0 = time.perf_counter()
    r = sess.post(f"{base_url}/generate", json=body, timeout=300)
    dt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    meta = r.json()["meta_info"]
    inp = meta.get("input_token_ids_logprobs") or []
    # SGLang returns one entry per input position from logprob_start_len; entry j corresponds to position j (None at 0)
    out = []
    for p, q in zip(pos, qs):
        entry = inp[p] if p < len(inp) else None
        letters = letters_for(len(q["options"]))
        probs, missing, logp, _ = _sglang_parse({"output_token_ids_logprobs": [entry or []], "output_top_logprobs": [[]]}, letters,
                                                {L: tid for L, tid in zip(letters_for(len(letter_ids)), letter_ids)})
        out.append({"probs": probs, "missing": missing, "raw_logprobs": logp})
    return out, dt, {"n_input_entries": len(inp), "prompt_tokens": meta.get("prompt_tokens"), "cached": meta.get("cached_tokens")}


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    limit = int(a.get("--limit", 0)); workers = int(a.get("--workers", 8))
    placeholder = a.get("--placeholder", "_"); extra = int(a.get("--extra", 0))
    if a.get("--out-sub"):
        out_dir = os.path.join(out_dir, a["--out-sub"])
    os.makedirs(out_dir, exist_ok=True)
    commit = kw.get("commit")
    from transformers import AutoTokenizer
    model_path = requests.get(f"{base_url}/get_model_info", timeout=30).json()["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path)
    token_ids = check_labels(base_url, "sglang")
    letter_ids = [token_ids[L] for L in letters_for(5)]
    tr = TemplateRenderer(base_url, static=True)
    # sanity: our packed first turn must tokenize identically to the renderer's single-question prompt
    qdefs = {t: load(t)[0]["question"] for t in ALARM}
    ph_ids = tok.encode(placeholder, add_special_tokens=False)
    assert len(ph_ids) == 1, f"placeholder {placeholder!r} is not a single token: {ph_ids}"
    print(f"[packed] placeholder {placeholder!r} id {ph_ids[0]}; letter ids {letter_ids}", flush=True)
    tls = threading.local()

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session()
        return tls.s

    lock = threading.Lock()
    summary = {}
    for task in ALARM:
        rows = load(task)
        if limit:
            rows = rows[:limit]
        qs = [qdefs[t] for t in ALARM] + EXTRA[:extra]
        qtasks = ALARM + [f"extra{i}" for i in range(extra)]
        own = ALARM.index(task)
        out_path = os.path.join(out_dir, f"{task}.jsonl")
        done = set()
        if os.path.exists(out_path):
            done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
        todo = [r for r in rows if r["id"] not in done]
        print(f"[{task}] {len(todo)} states × K={len(qs)}", flush=True)
        t0 = time.time()
        n_sep = n_pack = n_int = n = 0

        def one(r):
            s = sess()
            # separate: first question alone (warms prefix), then the rest as a list
            t_s = time.perf_counter()
            p0 = tr.render(r["state"], qs[0])
            r0 = read_option_probs_sglang(base_url, p0, letters_for(len(qs[0]["options"])), session=s, token_ids=token_ids)
            rest, _ = batch_read_option_probs_sglang(base_url, [tr.render(r["state"], q) for q in qs[1:]], [letters_for(len(q["options"])) for q in qs[1:]], session=s, token_ids=token_ids)
            sep = [r0] + rest
            sep_ms = (time.perf_counter() - t_s) * 1000
            ids, pos = packed_ids(tok, r["state"], qs, placeholder)
            for attempt in range(3):
                try:
                    pk, pk_ms, meta = read_packed(base_url, s, ids, pos, letter_ids, qs)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        return {"id": r["id"], "error": repr(e)}
                    time.sleep(1)
            ch = lambda d: (max(d["probs"], key=d["probs"].get) if d["probs"] else None)
            per_q = [{"task": qt, "sep": ch(a_), "packed": ch(b_), "sep_probs": a_["probs"], "packed_probs": b_["probs"], "missing_packed": b_["missing"]}
                     for qt, a_, b_ in zip(qtasks, sep, pk)]
            return {"id": r["id"], "task": task, "gold": r["gold"], "difficulty": r["difficulty"], "pair_id": r.get("pair_id"), "K": len(qs),
                    "own_index": own, "sep_correct": per_q[own]["sep"] == r["gold"], "packed_correct": per_q[own]["packed"] == r["gold"],
                    "interference": sum(1 for x in per_q if x["sep"] != x["packed"]), "per_q": per_q,
                    "sep_ms": sep_ms, "packed_ms": pk_ms, "packed_meta": meta, "n_tokens": len(ids)}

        with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            for i, res in enumerate(ex.map(one, todo)):
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    if "error" not in res:
                        n += 1; n_sep += res["sep_correct"]; n_pack += res["packed_correct"]; n_int += res["interference"]
                    if (i + 1) % 50 == 0:
                        f.flush()
                        if commit:
                            commit()
                        print(f"  [{task}] {i + 1}/{len(todo)} sep {n_sep / max(n, 1):.3f} packed {n_pack / max(n, 1):.3f} interference/q {n_int / max(n * len(qs), 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        summary[task] = {"n": n, "acc_sep": n_sep / max(n, 1), "acc_packed": n_pack / max(n, 1), "interference_rate": n_int / max(n * len(qs), 1), "s": round(time.time() - t0, 1)}
        print(f"[{task}] {summary[task]}", flush=True)
        if commit:
            commit()
    json.dump({"placeholder": placeholder, "extra": extra, "summary": summary}, open(os.path.join(out_dir, "_summary.json"), "w"), ensure_ascii=False, indent=1)
    if commit:
        commit()
    return summary
