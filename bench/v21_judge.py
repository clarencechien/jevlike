"""v21 Phase 2/3: E4B / 26B flag the Breeze windows with errors — typed decisions read from the first-token letter.

  main(base_url, out_dir, args="--engine e4b --mode text|swap|audio [--limit N] [--workers 4] [--windows data/v21/windows.jsonl] [--audio-dir DIR]")
  smoke(base_url, audio_dir)   # Phase 3: multimodal /completion + n_probs, transcription of a clean clip and a window, Q1 on two windows

Modes: text  = D2 (E4B) / D4 (26B): state = previous window + this window, three questions, letters A/B(/C/D)
       swap  = D2': Q1/Q2 with the two options swapped, Q3 with the three other rotations (position-bias check, v19 style)
       audio = D3: same prompt + the window's own audio (16 kHz wav, base64) after the text, via /completion multimodal_data
"""
import base64
import io
import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from decide.client import _parse_candidates, option_mass, read_option_probs
from decide.prompt import TemplateRenderer, build_messages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_V21 = "你是語音辨識品管引擎。只回答一個字母。"
MARKER = "<__media__>"
Q = {
    "q1": {"instructions": "「這一段」裡有沒有英文字被聽錯或寫錯（例如 core 寫成 cool、Bing 寫成 bin、compare to 寫成 comparative）？",
           "options": {"A": "沒有，英文字都對", "B": "有，至少一個英文字錯了"}},
    "q2": {"instructions": "「這一段」裡有沒有中文被聽錯（錯字、不成詞、不合上下文的詞）？",
           "options": {"A": "沒有，中文都對", "B": "有，至少一處中文錯了"}},
    "q3": {"instructions": "「這一段」的辨識品質是哪一種？",
           "options": {"A": "乾淨，沒有錯", "B": "只有英文字錯", "C": "只有中文錯", "D": "英文和中文都有錯"}},
}


def state_text(w):
    prev = w.get("prev_text") or "（沒有）"
    return ("這是台灣科技 podcast 的語音辨識逐字稿，講者中英夾雜，沒有標點。\n"
            f"【上一段】{prev}\n【這一段】{w['text']}")


def rotate(q, r):
    """Options rotated by r: original option j lands at position (j + r) mod K. Returns (question, pos->orig letter)."""
    letters = list(q["options"]); K = len(letters); labels = [q["options"][L] for L in letters]
    opts = {letters[(j + r) % K]: labels[j] for j in range(K)}
    opts = {L: opts[L] for L in letters}
    back = {letters[(j + r) % K]: letters[j] for j in range(K)}
    return {"instructions": q["instructions"], "options": opts}, back


def window_wav_b64(audio_dir, w, cache):
    import soundfile as sf
    key = (w["seg"], w["cond"])
    if key not in cache:
        cache[key] = sf.read(os.path.join(audio_dir, f"{w['seg']}__{w['cond']}.wav"), dtype="int16")
    data, sr = cache[key]
    t0 = float(w["t0"] or 0); t1 = float(w["t1"]) if w.get("t1") is not None else t0 + 15.0
    seg = data[int(t0 * sr): int(t1 * sr)]
    buf = io.BytesIO(); sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode(), round(len(seg) / sr, 2)


def read_mm(base_url, prompt, letters, wav_b64, session=None, n_probs=40, timeout=600, slot_id=None):
    """read_option_probs for a prompt that carries one audio clip at MARKER (llama-server /completion multimodal_data)."""
    import math
    sess = session or requests
    body = {"prompt": {"prompt_string": prompt, "multimodal_data": [wav_b64]}, "n_predict": 1, "temperature": 0, "n_probs": n_probs,
            "cache_prompt": True, "samplers": []}
    if slot_id is not None:
        body["id_slot"] = slot_id
    t0 = time.perf_counter(); resp = sess.post(f"{base_url}/completion", json=body, timeout=timeout); dt = (time.perf_counter() - t0) * 1000
    if resp.status_code != 200:
        raise RuntimeError(f"/completion {resp.status_code}: {resp.text[:400]}")
    r = resp.json(); top, cand = _parse_candidates(r); logp = {}
    for c in cand:
        tok = (c.get("token") or "").strip()
        if tok in letters:
            v = c.get("logprob"); v = math.log(max(c.get("prob", 0.0), 1e-12)) if v is None else v
            logp[tok] = max(logp.get(tok, -1e9), v)
    probs = {}
    if logp:
        m = max(logp.values()); z = sum(math.exp(v - m) for v in logp.values()); probs = {k: math.exp(v - m) / z for k, v in logp.items()}
    tm = r.get("timings") or {}
    return {"probs": probs, "raw_logprobs": logp, "option_mass": option_mass(logp), "latency_ms": dt, "first_token": r.get("content"),
            "prompt_tokens": r.get("tokens_evaluated"), "server_cache_n": tm.get("cache_n"), "server_prompt_ms": tm.get("prompt_ms"),
            "top_tokens": [{"token": c.get("token"), "logprob": c.get("logprob", c.get("prob"))} for c in cand[:8]]}


def read_mm_chat(base_url, w, q, letters, wav_b64, session=None, timeout=600, top=20):
    """First-token letter probabilities through /v1/chat/completions with input_audio + logprobs (the OpenAI-style route is the
    one that accepts audio on this build; the server applies the same no-think template, --reasoning-budget 0)."""
    import math
    sess = session or requests
    msgs = build_messages(state_text(w), q, SYSTEM_V21)
    body = {"messages": [msgs[0], {"role": "user", "content": [{"type": "text", "text": msgs[1]["content"] + "\n【這一段的聲音】"},
                                                             {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}]}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": top, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}  # else the first token competes with '<|channel>' (option mass 0.5-0.8 in the smoke)
    t0 = time.perf_counter(); resp = sess.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout); dt = (time.perf_counter() - t0) * 1000
    if resp.status_code != 200:
        raise RuntimeError(f"/v1/chat/completions {resp.status_code}: {resp.text[:400]}")
    r = resp.json(); ch = r["choices"][0]; lp_c = ((ch.get("logprobs") or {}).get("content") or [{}])[0]
    cand = lp_c.get("top_logprobs") or []; logp = {}
    for c in cand:
        tok = (c.get("token") or "").strip()
        if tok in letters:
            logp[tok] = max(logp.get(tok, -1e9), c.get("logprob", -1e9))
    probs = {}
    if logp:
        m = max(logp.values()); z = sum(math.exp(v - m) for v in logp.values()); probs = {k: math.exp(v - m) / z for k, v in logp.items()}
    tm = r.get("timings") or {}
    return {"probs": probs, "raw_logprobs": logp, "option_mass": option_mass(logp), "latency_ms": dt, "first_token": ch["message"].get("content"),
            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), "server_cache_n": tm.get("cache_n"), "server_prompt_ms": tm.get("prompt_ms"),
            "top_tokens": [{"token": c.get("token"), "logprob": c.get("logprob")} for c in cand[:8]], "route": "chat_logprobs"}


def render(tr, w, q, audio=False):
    msgs = build_messages(state_text(w), q, SYSTEM_V21)
    if audio:
        msgs[1]["content"] += f"\n【這一段的聲音】{MARKER}"
    return tr.render_messages(msgs)


def main(base_url, out_dir, args="", **kw):
    a = dict(zip(*[iter(args.split())] * 2)) if args else {}
    engine = a.get("--engine", "e4b"); mode = a.get("--mode", "text"); workers = int(a.get("--workers", 4)); limit = int(a.get("--limit", 0))
    windows_p = a.get("--windows", os.path.join(ROOT, "data/v21/windows.jsonl")); audio_dir = a.get("--audio-dir", "")
    os.makedirs(out_dir, exist_ok=True)
    wins = [json.loads(l) for l in open(windows_p)]
    if limit:
        wins = wins[::max(1, len(wins) // limit)][:limit]
    tr = TemplateRenderer(base_url)
    lock = threading.Lock(); tls = threading.local(); slots = itertools.count(); wav_cache = {}
    out_p = os.path.join(out_dir, "judge.jsonl"); done = set()
    if os.path.exists(out_p):
        done = {json.loads(l)["id"] for l in open(out_p)}
    todo = [w for w in wins if w["id"] not in done]
    print(f"{engine} {mode}: {len(todo)} windows to do ({len(done)} done)", flush=True)

    def sess():
        if not hasattr(tls, "s"):
            tls.s = requests.Session(); tls.slot = next(slots) % workers
        return tls.s, tls.slot

    def ask(w, q, letters, wav):
        s, slot = sess()
        prompt = render(tr, w, q, audio=wav is not None)
        for attempt in range(3):
            try:
                if wav is None:
                    return read_option_probs(base_url, prompt, letters, session=s, slot_id=slot)
                return read_mm_chat(base_url, w, q, letters, wav, session=s)
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"error": str(e)[:300], "probs": {}}
                time.sleep(2)

    def one(w):
        rec = {"id": w["id"], "seg": w["seg"], "cond": w["cond"], "E": w["E"], "C": w["C"], "q3": w["q3"], "engine": engine, "mode": mode}
        wav = None
        if mode == "audio":
            with lock:
                wav, rec["audio_s"] = window_wav_b64(audio_dir, w, wav_cache)
        if mode in ("text", "audio"):
            for qk in ("q1", "q2", "q3"):
                rec[qk] = ask(w, Q[qk], list(Q[qk]["options"]), wav)
        else:  # swap: q1/q2 rotation 1 (A<->B); q3 rotations 1..3, probabilities mapped back to the original letters
            for qk, rots in (("q1", [1]), ("q2", [1]), ("q3", [1, 2, 3])):
                for r in rots:
                    q, back = rotate(Q[qk], r)
                    res = ask(w, q, list(q["options"]), wav)
                    res["rotation"] = r; res["probs_orig"] = {back[L]: p for L, p in res.get("probs", {}).items()}
                    rec[f"{qk}_r{r}"] = res
        return rec

    t0 = time.time(); n = 0
    with ThreadPoolExecutor(workers) as ex, open(out_p, "a") as f:
        for rec in ex.map(one, todo):
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); n += 1
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    recs = [json.loads(l) for l in open(out_p)]
    lat = [v["latency_ms"] for r in recs for k, v in r.items() if isinstance(v, dict) and "latency_ms" in v]
    lat.sort()
    summary = {"engine": engine, "mode": mode, "n": len(recs), "wall_s": round(time.time() - t0, 1),
               "latency_p50_ms": lat[len(lat) // 2] if lat else None, "latency_p95_ms": lat[int(len(lat) * 0.95)] if lat else None,
               "errors": sum(1 for r in recs for v in r.values() if isinstance(v, dict) and "error" in v),
               "option_mass_mean": round(sum(v.get("option_mass", 0) for r in recs for v in r.values() if isinstance(v, dict) and "option_mass" in v) / max(len(lat), 1), 4)}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(summary, flush=True)
    return summary


def transcribe_chat(base_url, wav_b64, prompt, max_tokens=256):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}]}],
            "max_tokens": max_tokens, "temperature": 0}
    t0 = time.perf_counter(); r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600); dt = (time.perf_counter() - t0) * 1000
    r.raise_for_status(); j = r.json()
    return {"text": j["choices"][0]["message"].get("content"), "latency_ms": round(dt), "usage": j.get("usage")}


def smoke(base_url, audio_dir, windows_p=None):
    """Phase 3 smoke, three checks (handoff §7). Returns a dict with pass/fail per check."""
    import urllib.request
    windows_p = windows_p or os.path.join(ROOT, "data/v21/windows.jsonl")
    wins = [json.loads(l) for l in open(windows_p)]
    tr = TemplateRenderer(base_url); out = {}; cache = {}
    official = ("Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n"
                "* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.")
    # (2a) canonical clean English clip (whisper.cpp's jfk.wav, 11 s, 16 kHz) as the baseline the model card's sample would give
    try:
        jfk = urllib.request.urlopen("https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav", timeout=60).read()
        out["jfk"] = transcribe_chat(base_url, base64.b64encode(jfk).decode(), official)
        out["jfk"]["pass"] = "ask not what your country" in (out["jfk"]["text"] or "").lower()
    except Exception as e:  # noqa: BLE001
        out["jfk"] = {"error": str(e)[:300], "pass": False}
    # pick windows: one with a known English error (ref core -> hyp cool or similar F2) on M0, one clean on M0
    err = next((w for w in wins if w["cond"] == "M0" and any(d["kind"] == "F2" for d in w["detail"])), None)
    clean = next((w for w in wins if w["cond"] == "M0" and w["N"] and w["has_latin"]), None)
    for name, w in (("err_window", err), ("clean_window", clean)):
        if w is None:
            out[name] = {"error": "no such window"}; continue
        wav, secs = window_wav_b64(audio_dir, w, cache)
        rec = {"id": w["id"], "seconds": secs, "breeze_text": w["text"], "errors": [d for d in w["detail"] if d["kind"] in ("F1", "F2", "F4", "C")][:4]}
        rec["transcribe"] = transcribe_chat(base_url, wav, official + "\n* The speaker mixes Mandarin Chinese and English. Keep every English word verbatim in Latin script; do not translate or transliterate them.")
        txt = rec["transcribe"]["text"] or ""
        rec["has_latin"] = any(c.isascii() and c.isalpha() for c in txt)
        rec["has_traditional_marker"] = any(c in txt for c in "這個們說對來時為麼個還會後")  # characters that differ from Simplified forms
        rec["has_simplified_marker"] = any(c in txt for c in "这个们说对来时为么还会后")
        # (1)+(3): multimodal /completion with n_probs, Q1
        prompt = render(tr, w, Q["q1"], audio=True)
        try:
            rec["q1_audio_completion"] = read_mm(base_url, prompt, ["A", "B"], wav)
        except Exception as e:  # noqa: BLE001
            rec["q1_audio_completion"] = {"error": str(e)[:300], "probs": {}}
        try:
            rec["q1_audio"] = read_mm_chat(base_url, w, Q["q1"], ["A", "B"], wav)
        except Exception as e:  # noqa: BLE001
            rec["q1_audio"] = {"error": str(e)[:600], "probs": {}}
            # variant B: the OpenAI-style route cannot return first-token probabilities, but tell whether the marker is the problem
            try:
                body = {"prompt": prompt.replace(MARKER, "<__media__>"), "multimodal_data": [wav], "n_predict": 1, "n_probs": 40, "temperature": 0, "samplers": []}
                r2 = requests.post(f"{base_url}/completion", json=body, timeout=600)
                rec["q1_audio_variant_flat"] = {"status": r2.status_code, "body": r2.text[:400]}
            except Exception as e2:  # noqa: BLE001
                rec["q1_audio_variant_flat"] = {"error": str(e2)[:300]}
        rec["q1_text"] = read_option_probs(base_url, render(tr, w, Q["q1"]), ["A", "B"])
        out[name] = rec
    pa = out.get("err_window", {}).get("q1_audio", {}); pb = out.get("clean_window", {}).get("q1_audio", {})
    out["check1_probs_readable"] = bool(pa.get("probs")) and pa.get("option_mass", 0) >= 0.9 and bool(pb.get("probs")) and pb.get("option_mass", 0) >= 0.9
    out["check2_transcription"] = bool(out.get("jfk", {}).get("pass")) and out.get("err_window", {}).get("has_latin", False) and not out.get("err_window", {}).get("has_simplified_marker", True)
    out["check3_q1_direction"] = pa.get("probs", {}).get("B", 0) > 0.5 and pb.get("probs", {}).get("A", 0) > 0.5
    out["all_pass"] = out["check1_probs_readable"] and out["check2_transcription"] and out["check3_q1_direction"]
    return out
