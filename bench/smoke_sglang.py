"""SGLang smoke: same 5 examples as bench/smoke.py, static Gemma 4 template; checks first token, missing,
prompt_tokens vs llama-server's count (tokenizer parity), cached_tokens on a repeated prompt and on a shared prefix."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.smoke import EXAMPLES  # noqa: E402
from decide.client import batch_read_option_probs_sglang, read_option_probs_sglang  # noqa: E402
from decide.labels import check_labels  # noqa: E402
from decide.prompt import TemplateRenderer, letters_for  # noqa: E402


def main(base_url, out_dir, args="", **kw):
    tr = TemplateRenderer(base_url, static=True)
    rep = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    rep["label_token_ids"] = check_labels(base_url, "sglang")  # T0-a (handoff v5)
    print("label token ids:", rep["label_token_ids"], flush=True)
    for ex in EXAMPLES:
        L = letters_for(len(ex["question"]["options"]))
        p = tr.render(ex["state"], ex["question"])
        for rep_i in range(2):
            r = read_option_probs_sglang(base_url, p, L)
            if rep_i == 1:  # T4 (handoff v5): exact token-id read must agree with the top-k read
                r2 = read_option_probs_sglang(base_url, p, L, token_ids=rep["label_token_ids"])
                r["probs_token_ids"] = r2["probs"]
                r["token_ids_max_abs_diff"] = max(abs(r["probs"].get(k, 0) - r2["probs"].get(k, 0)) for k in L)
            row = {"id": ex["id"], "rep": rep_i, "gold": ex["gold"], "chosen": max(r["probs"], key=r["probs"].get) if r["probs"] else None,
                   "probs": {k: round(v, 4) for k, v in r["probs"].items()}, "missing": r["missing"], "first_token": r["first_token"],
                   "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["tokens_cached"], "latency_ms": round(r["latency_ms"], 1),
                   "top": [t["token"] for t in r["top_tokens"][:6]],
                   "probs_token_ids": r.get("probs_token_ids"), "token_ids_max_abs_diff": r.get("token_ids_max_abs_diff")}
            rep["rows"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # v6: with the literal <bos> in the static template, SGLang's prompt_tokens must equal llama-server's (117 for s1 in v2 smoke)
    rep["bos_check"] = {"s1_prompt_tokens": rep["rows"][0]["prompt_tokens"], "llama_v2_smoke_s1": 117,
                        "ok": rep["rows"][0]["prompt_tokens"] == 117}
    print("bos_check", rep["bos_check"], flush=True)
    # shared-prefix batch: same state, 3 questions
    ex = EXAMPLES[0]
    qs = [dict(ex["question"], instructions=f"問題 {i}：" + ex["question"]["instructions"]) for i in range(3)]
    res, wall = batch_read_option_probs_sglang(base_url, [tr.render(ex["state"], q) for q in qs], [letters_for(3)] * 3, token_ids=rep["label_token_ids"])
    rep["batch"] = {"wall_ms": round(wall, 1), "cached": [r["tokens_cached"] for r in res], "prompt_tokens": [r["prompt_tokens"] for r in res], "chosen": [max(r["probs"], key=r["probs"].get) if r["probs"] else None for r in res], "missing": [r["missing"] for r in res]}
    print("batch", json.dumps(rep["batch"]), flush=True)
    # llama-server prompt token counts for the same 5 prompts (from v2 smoke) for parity
    try:
        llama = json.load(open("/root/data/../results/modal/smoke.json"))
    except Exception:  # noqa: BLE001
        llama = None
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(out_dir, "smoke.json"), "w"), ensure_ascii=False, indent=1)
    if kw.get("commit"):
        kw["commit"]()
    return rep
