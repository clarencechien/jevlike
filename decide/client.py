"""Read option-letter log-probs from llama-server's native /completion endpoint."""
import math
import time

import requests


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
