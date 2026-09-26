"""Label self-check (T0-a, handoff v5): every option letter must be exactly one token that decodes back to itself.

TypeLLM does this at client start (`single_token`); we assumed A–J are single tokens. Verify against the running
server so a model/quantization change cannot silently break the first-token read.
"""
import requests

from decide.prompt import LETTERS


class LabelError(RuntimeError):
    pass


def check_labels_llama(base_url: str, letters=LETTERS, session=None, timeout=30) -> dict:
    """llama-server: POST /tokenize (add_special=false, with_pieces) and /detokenize."""
    s = session or requests
    table = {}
    for ch in letters:
        r = s.post(f"{base_url}/tokenize", json={"content": ch, "add_special": False, "with_pieces": True}, timeout=timeout).json()
        toks = r.get("tokens", [])
        if len(toks) != 1:
            raise LabelError(f"label {ch!r} tokenizes to {len(toks)} tokens: {toks}")
        tid = toks[0]["id"]
        back = s.post(f"{base_url}/detokenize", json={"tokens": [tid]}, timeout=timeout).json().get("content")
        if back != ch:
            raise LabelError(f"label {ch!r} -> id {tid} -> {back!r}: not round-trippable")
        table[ch] = tid
    if len(set(table.values())) != len(table):
        raise LabelError(f"duplicate token ids among labels: {table}")
    return table


def _check_labels_hf(model_path: str, letters) -> dict:
    """Fallback: the served model's HF tokenizer loaded locally (same files SGLang loads)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    table = {}
    for ch in letters:
        ids = tok.encode(ch, add_special_tokens=False)
        if len(ids) != 1:
            raise LabelError(f"label {ch!r} tokenizes to {len(ids)} tokens: {ids}")
        back = tok.decode(ids)
        if back != ch:
            raise LabelError(f"label {ch!r} -> id {ids[0]} -> {back!r}: not round-trippable")
        table[ch] = int(ids[0])
    if len(set(table.values())) != len(table):
        raise LabelError(f"duplicate token ids among labels: {table}")
    return table


def check_labels_sglang(base_url: str, letters=LETTERS, model=None, session=None, timeout=30) -> dict:
    """SGLang: POST /v1/tokenize (add_special_tokens=false) and /v1/detokenize; falls back to the HF tokenizer
    when the server's tokenize endpoint is unusable (sglang:gemma4-mtp's /v1/tokenize dies serialising its response)."""
    s = session or requests
    if model is None:
        model = s.get(f"{base_url}/get_model_info", timeout=timeout).json().get("model_path")
    try:
        r = s.post(f"{base_url}/v1/tokenize", json={"model": model, "prompt": letters[0], "add_special_tokens": False}, timeout=timeout)
        r.raise_for_status()
        r.json()
    except Exception:  # noqa: BLE001
        return _check_labels_hf(model, letters)
    table = {}
    for ch in letters:
        r = s.post(f"{base_url}/v1/tokenize", json={"model": model, "prompt": ch, "add_special_tokens": False}, timeout=timeout).json()
        toks = r.get("tokens", [])
        if len(toks) != 1:
            raise LabelError(f"label {ch!r} tokenizes to {len(toks)} tokens: {toks}")
        tid = int(toks[0])
        back = s.post(f"{base_url}/v1/detokenize", json={"model": model, "tokens": [tid]}, timeout=timeout).json().get("text")
        if back != ch:
            raise LabelError(f"label {ch!r} -> id {tid} -> {back!r}: not round-trippable")
        table[ch] = tid
    if len(set(table.values())) != len(table):
        raise LabelError(f"duplicate token ids among labels: {table}")
    return table


def check_labels(base_url: str, backend: str = "llama", **kw) -> dict:
    return check_labels_sglang(base_url, **kw) if backend == "sglang" else check_labels_llama(base_url, **kw)
