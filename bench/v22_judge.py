"""v22 (docs/handoff-v22-glossary-words.md): glossary in the state, English words as options. Text-only letter readout.

  main(base_url, out_dir, args="--engine e4b --arm glossary|glossary_mixed|words|words_glossary [--limit N] [--workers 4]")
"""
import itertools
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from decide.client import read_option_probs
from decide.prompt import LETTERS, TemplateRenderer, build_messages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_V22 = "你是語音辨識品管引擎。只回答一個字母。"
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
MAX_WORDS = 9
ALL_OK = "都對，沒有英文字聽錯"
TERMS = json.load(open(os.path.join(ROOT, "data/v21/kikemu/terms.json")))


def glossary(seg, mixed=False):
    segs = ["T1", "T2", "T3"] if mixed else [seg]
    seen, out = set(), []
    for s in segs:
        for t in TERMS[s]:
            if t["term"].lower() not in seen:
                seen.add(t["term"].lower()); out.append(t["term"])
    return "、".join(out)


def latin_words(text):
    seen, out = set(), []
    for w in LATIN.findall(text):
        if w.lower() not in seen:
            seen.add(w.lower()); out.append(w)
    return out[:MAX_WORDS]


def state_text(w, gl=None):
    prev = w.get("prev_text") or "（沒有）"
    head = "這是台灣科技 podcast 的語音辨識逐字稿，講者中英夾雜，沒有標點。"
    if gl:
        head += f"\n【這集的術語表】{gl}"
    return f"{head}\n【上一段】{prev}\n【這一段】{w['text']}"


def question(arm, w):
    """Returns (question dict, letters, meta) or None when the arm does not apply to this window."""
    if arm in ("glossary", "glossary_mixed"):
        return ({"instructions": "「這一段」裡有沒有術語表裡的詞被聽成別的字（例如 Bing 聽成 bin、core 聽成 cool）？",
                 "options": {"A": "沒有，術語都對", "B": "有，至少一個術語被聽成別的字"}}, ["A", "B"], {"words": None})
    words = latin_words(w["text"])
    if not words:
        return None
    letters = list(LETTERS[:len(words) + 1])
    opts = {L: wd for L, wd in zip(letters, words)}; opts[letters[-1]] = ALL_OK
    return ({"instructions": "「這一段」的英文字裡，哪一個是聽錯的（寫成了別的字）？", "options": opts}, letters, {"words": words, "ok_letter": letters[-1]})


def gold_words(w):
    """Latin tokens on the hypothesis side of F2 opcodes (the mis-heard words), lowercased."""
    out = set()
    for d in w.get("detail", []):
        if d["kind"] == "F2":
            out.update(t.lower() for t in LATIN.findall(d["hyp"]))
    return sorted(out)


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    engine = a.get("--engine", "e4b"); arm = a.get("--arm", "glossary"); workers = int(a.get("--workers", 4)); limit = int(a.get("--limit", 0))
    wins = [json.loads(l) for l in open(a.get("--windows", os.path.join(ROOT, "data/v21/windows.jsonl")))]
    if limit:
        wins = [w for w in wins if w["has_latin"]][::max(1, 267 // limit)][:limit]
    os.makedirs(out_dir, exist_ok=True); out_p = os.path.join(out_dir, "judge.jsonl")
    done = {json.loads(l)["id"] for l in open(out_p)} if os.path.exists(out_p) else set()
    todo = [w for w in wins if w["id"] not in done]
    tr = TemplateRenderer(base_url); lock = threading.Lock(); tls = threading.local(); slots = itertools.count()
    print(f"{engine} {arm}: {len(todo)} windows to do ({len(done)} done)", flush=True)

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session(); tls.slot = next(slots) % workers
        return tls.s, tls.slot

    def one(w):
        rec = {"id": w["id"], "seg": w["seg"], "cond": w["cond"], "E": w["E"], "C": w["C"], "engine": engine, "arm": arm, "gold_words": gold_words(w)}
        q = question(arm, w)
        if q is None:
            rec["skipped"] = "no latin words"; return rec
        qd, letters, meta = q
        gl = glossary(w["seg"], mixed=(arm == "glossary_mixed")) if arm in ("glossary", "glossary_mixed", "words_glossary") else None
        prompt = tr.render_messages(build_messages(state_text(w, gl), qd, SYSTEM_V22))
        s, slot = sess()
        for attempt in range(3):
            try:
                r = read_option_probs(base_url, prompt, letters, session=s, slot_id=slot); break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    r = {"error": str(e)[:300], "probs": {}}
                time.sleep(2)
        rec["res"] = r; rec["letters"] = letters; rec.update(meta)
        if meta.get("words"):
            p = r.get("probs") or {}
            rec["p_ok"] = p.get(meta["ok_letter"], float("nan")) if p else None
            word_p = {wd.lower(): p.get(L, 0.0) for L, wd in zip(letters, meta["words"])}
            rec["pred_word"] = max(word_p, key=word_p.get) if word_p and p else None
        else:
            rec["p_err"] = (r.get("probs") or {}).get("B")
        return rec

    t0 = time.time(); n = 0
    with ThreadPoolExecutor(workers) as ex, open(out_p, "a") as f:
        for rec in ex.map(one, todo):
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1
                if n % 100 == 0:
                    print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    recs = [json.loads(l) for l in open(out_p)]
    lat = sorted(r["res"]["latency_ms"] for r in recs if "res" in r and "latency_ms" in r["res"])
    masses = [r["res"]["option_mass"] for r in recs if "res" in r and "option_mass" in r["res"]]
    summ = {"engine": engine, "arm": arm, "n": len(recs), "asked": len(lat), "skipped": sum("skipped" in r for r in recs), "wall_s": round(time.time() - t0, 1),
            "latency_p50_ms": round(lat[len(lat) // 2], 1) if lat else None, "option_mass_mean": round(sum(masses) / max(len(masses), 1), 4),
            "errors": sum(1 for r in recs if "res" in r and "error" in r["res"])}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1); print(summ, flush=True)
    return summ
