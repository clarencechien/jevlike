"""Read option-letter log-probs from llama-server's native /completion endpoint."""
import math
import time

import requests


def option_mass(logp: dict) -> float:
    """Sum of the letters' raw probabilities before renormalisation (after verdict's 'option mass').
    ~1.0 = the model wanted to answer a letter; ~0 = the template is broken (thinking on, stray whitespace, ...)."""
    return float(sum(math.exp(v) for v in logp.values())) if logp else 0.0


def _parse_candidates(r: dict):
    cp = r.get("completion_probabilities") or []
    if not cp:
        return None, []
    top = cp[0]
    cand = top.get("top_logprobs") or top.get("top_probs") or top.get("probs") or []
    return top, cand


def read_option_probs(base_url, prompt, letters, n_probs=40, cache_prompt=True, session=None, timeout=300, slot_id=None):
    """One forward pass (n_predict=1). Returns softmax over `letters` from the top-n_probs list.

    Tokens are matched after stripping whitespace, so 'A' and ' A' both count for letter A (max taken).
    """
    sess = session or requests
    body = {
        "prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": n_probs,
        "cache_prompt": cache_prompt, "samplers": [],
    }
    if slot_id is not None:
        body["id_slot"] = slot_id
    t0 = time.perf_counter()
    resp = sess.post(f"{base_url}/completion", json=body, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    r = resp.json()
    top, cand = _parse_candidates(r)
    logp = {}
    for c in cand:
        tok = (c.get("token") or c.get("tok_str") or "").strip()
        if tok in letters:
            v = c.get("logprob")
            if v is None:
                v = math.log(max(c.get("prob", 0.0), 1e-12))
            logp[tok] = max(logp.get(tok, -1e9), v)
    missing = [l for l in letters if l not in logp]
    if logp:
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        probs = {k: math.exp(v - m) / z for k, v in logp.items()}
    else:
        probs = {}
    timings = r.get("timings") or {}
    return {
        "probs": probs, "missing": missing, "latency_ms": dt,
        "option_mass": option_mass(logp),  # P2 (handoff v7): raw mass on the letters before renormalising
        "first_token": r.get("content"),
        "first_token_id": (top or {}).get("id"),
        "prompt_tokens": r.get("tokens_evaluated"),
        "tokens_cached": r.get("tokens_cached"),
        "raw_logprobs": logp,
        "top_tokens": [{"token": c.get("token"), "logprob": c.get("logprob", c.get("prob"))} for c in cand[:10]],
        "server_prompt_ms": timings.get("prompt_ms"),
        "server_predicted_ms": timings.get("predicted_ms"),
        "server_prompt_n": timings.get("prompt_n"),
        "server_cache_n": timings.get("cache_n"),
    }


def chat_json(base_url, messages, max_tokens=64, session=None, timeout=300, extra=None):
    """Control arm: /v1/chat/completions generating a short JSON answer at temperature 0."""
    sess = session or requests
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True}
    if extra:
        body.update(extra)
    t0 = time.perf_counter()
    resp = sess.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    r = resp.json()
    msg = r["choices"][0]["message"]
    return {"content": msg.get("content"), "reasoning": msg.get("reasoning_content"), "latency_ms": dt,
            "usage": r.get("usage"), "finish_reason": r["choices"][0].get("finish_reason")}


# ---------------------------------------------------------------- SGLang native /generate
def _sglang_parse(meta, letters, token_ids=None):
    """token_ids: optional {letter: token_id}; when given, read SGLang's exact `output_token_ids_logprobs`
    (TypeLLM-style, handoff v5 T4) instead of scanning the top-k list."""
    top = (meta.get("output_top_logprobs") or [[]])[0]
    logp = {}
    first = None
    if token_ids:
        exact = (meta.get("output_token_ids_logprobs") or [[]])[0]
        by_id = {int(item[1]): item[0] for item in exact}
        logp = {L: by_id[tid] for L, tid in token_ids.items() if L in letters and tid in by_id}
        missing = [l for l in letters if l not in logp]
        m = max(logp.values()) if logp else 0.0
        z = sum(math.exp(v - m) for v in logp.values()) or 1.0
        probs = {k: math.exp(v - m) / z for k, v in logp.items()}
        return probs, missing, logp, [{"token": (i[2] if len(i) > 2 else None), "logprob": i[0]} for i in top[:10]]
    for item in top:
        lp, tid = item[0], item[1]
        tok = (item[2] if len(item) > 2 and item[2] is not None else "")
        if first is None:
            first = tok
        tok = tok.strip()
        if tok in letters:
            logp[tok] = max(logp.get(tok, -1e9), lp)
    missing = [l for l in letters if l not in logp]
    if logp:
        m = max(logp.values()); z = sum(math.exp(v - m) for v in logp.values())
        probs = {k: math.exp(v - m) / z for k, v in logp.items()}
    else:
        probs = {}
    return probs, missing, logp, [{"token": (i[2] if len(i) > 2 else None), "logprob": i[0]} for i in top[:10]]


def _sglang_body(text, letters, n_probs, token_ids):
    body = {"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
            "logprob_start_len": -1, "return_text_in_logprobs": True}
    if token_ids:
        ids = [token_ids[L] for L in letters]
        body["token_ids_logprob"] = [ids] * len(text) if isinstance(text, list) else ids
        body["top_logprobs_num"] = 5  # still keep a short top list for the first-token record
    else:
        body["top_logprobs_num"] = n_probs
    return body


def read_option_probs_sglang(base_url, prompt, letters, n_probs=40, session=None, timeout=300, token_ids=None, **_):
    sess = session or requests
    body = _sglang_body(prompt, letters, n_probs, token_ids)
    t0 = time.perf_counter()
    resp = sess.post(f"{base_url}/generate", json=body, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    r = resp.json()
    meta = r.get("meta_info", {})
    probs, missing, logp, top = _sglang_parse(meta, letters, token_ids)
    return {"probs": probs, "missing": missing, "latency_ms": dt, "first_token": r.get("text"), "first_token_id": None,
            "prompt_tokens": meta.get("prompt_tokens"), "tokens_cached": meta.get("cached_tokens"), "raw_logprobs": logp, "option_mass": option_mass(logp),
            "top_tokens": top, "server_prompt_ms": None, "server_predicted_ms": None, "server_prompt_n": meta.get("prompt_tokens"),
            "server_cache_n": meta.get("cached_tokens"), "e2e_latency_s": meta.get("e2e_latency")}


def read_option_probs_sglang_ids(base_url, input_ids, letters, token_ids, session=None, timeout=300, **_):
    """E1 (handoff v6): feed pre-tokenized ids (from llama-server /tokenize) instead of text, read exact letter logprobs."""
    sess = session or requests
    body = _sglang_body("", letters, 5, token_ids)
    del body["text"]
    body["input_ids"] = list(input_ids)
    t0 = time.perf_counter()
    resp = sess.post(f"{base_url}/generate", json=body, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    r = resp.json()
    meta = r.get("meta_info", {})
    probs, missing, logp, top = _sglang_parse(meta, letters, token_ids)
    return {"probs": probs, "missing": missing, "latency_ms": dt, "first_token": r.get("text"), "first_token_id": None,
            "prompt_tokens": meta.get("prompt_tokens"), "tokens_cached": meta.get("cached_tokens"), "raw_logprobs": logp, "option_mass": option_mass(logp),
            "top_tokens": top, "server_prompt_ms": None, "server_predicted_ms": None, "server_prompt_n": meta.get("prompt_tokens"),
            "server_cache_n": meta.get("cached_tokens"), "e2e_latency_s": meta.get("e2e_latency")}


def batch_read_option_probs_sglang(base_url, prompts, letters_list, n_probs=40, session=None, timeout=600, token_ids=None):
    """Send K prompts as one list request; RadixAttention shares the common prefix. Returns list of dicts + wall ms."""
    sess = session or requests
    body = _sglang_body(list(prompts), letters_list[0], n_probs, token_ids)
    if token_ids:
        body["token_ids_logprob"] = [[token_ids[L] for L in letters] for letters in letters_list]
    t0 = time.perf_counter()
    resp = sess.post(f"{base_url}/generate", json=body, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    out = []
    for r, letters in zip(resp.json(), letters_list):
        meta = r.get("meta_info", {})
        probs, missing, logp, top = _sglang_parse(meta, letters, token_ids)
        out.append({"probs": probs, "missing": missing, "latency_ms": dt, "first_token": r.get("text"), "prompt_tokens": meta.get("prompt_tokens"),
                    "tokens_cached": meta.get("cached_tokens"), "raw_logprobs": logp, "option_mass": option_mass(logp), "top_tokens": top, "server_cache_n": meta.get("cached_tokens"),
                    "server_prompt_ms": None, "server_predicted_ms": None, "server_prompt_n": meta.get("prompt_tokens")})
    return out, dt


def generate_text(base_url, backend, prompt, n_predict=512, session=None, timeout=600):
    """Background-load generator used by L6: returns tokens generated."""
    sess = session or requests
    if backend == "sglang":
        r = sess.post(f"{base_url}/generate", json={"text": prompt, "sampling_params": {"max_new_tokens": n_predict, "temperature": 0.7}}, timeout=timeout).json()
        return r.get("meta_info", {}).get("completion_tokens", 0)
    r = sess.post(f"{base_url}/completion", json={"prompt": prompt, "n_predict": n_predict, "temperature": 0.7, "cache_prompt": False}, timeout=timeout).json()
    return r.get("tokens_predicted", 0)


def reader_for(backend):
    return read_option_probs_sglang if backend == "sglang" else read_option_probs
