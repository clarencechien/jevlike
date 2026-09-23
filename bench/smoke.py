"""M1 smoke: 5 hand-written examples; check template, tokenization of letters, first token, missing."""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decide.client import chat_json, read_option_probs  # noqa: E402
from decide.prompt import SYSTEM, TemplateRenderer, build_messages, letters_for  # noqa: E402

EXAMPLES = [
    {"id": "s1", "state": "SMT-L3 回焊爐 R-02 第 4 區溫度 245°C 超上限 10°C 持續 3 分鐘，板子仍在爐內",
     "question": {"instructions": "這則 alarm 的處置急迫度？", "options": {"A": "不急", "B": "盡快", "C": "停線"}}, "gold": "C"},
    {"id": "s2", "state": "E4021 NOZZLE VACUUM LOW HEAD2 (Mounter M-05)",
     "question": {"instructions": "這則 alarm 屬於哪一類？", "options": {"A": "機構", "B": "電控", "C": "物料", "D": "程式", "E": "環境"}}, "gold": "A"},
    {"id": "s3", "state": "工單 WO-20260923-017：AOI-01 報 Feeder 缺料 slot 23，料號 0402 電阻，線上有備料，操作員已補上",
     "question": {"instructions": "是否需要派工給設備工程師？", "options": {"A": "是", "B": "否"}}, "gold": "B"},
    {"id": "s4", "state": "DEK-02 錫膏厚度 SPC：連續 7 點在中心線下方且持續下降，尚未超出管制界限",
     "question": {"instructions": "SPC 應採取的動作？", "options": {"A": "持續監控", "B": "抽檢", "C": "停線複檢", "D": "呼叫 QE"}}, "gold": "B"},
    {"id": "s5", "state": "L2 線 UPH 過去 6 小時：1180, 1175, 1190, 1182, 1178, 1185（目標 1150）",
     "question": {"instructions": "UPH 是否異常？", "options": {"A": "是", "B": "否"}}, "gold": "B"},
]


def main(base_url, out_dir, args=""):
    os.makedirs(out_dir, exist_ok=True)
    rep = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    props = requests.get(f"{base_url}/props", timeout=30).json()
    rep["props"] = {k: props.get(k) for k in ["default_generation_settings", "total_slots", "model_path", "chat_template", "build_info", "modalities"]}

    # tokenization of letters with/without leading space
    tok = {}
    for s in ["A", " A", "B", " B", "答案：", "答案：A", "答案： A", "答案：\nA"]:
        r = requests.post(f"{base_url}/tokenize", json={"content": s, "with_pieces": True}, timeout=30).json()
        tok[s] = [(t["id"], t["piece"]) for t in r["tokens"]]
    rep["tokenize"] = tok

    variants = {
        "nothink+sys+prefill_答案：": dict(use_system=True, prefill="答案：", enable_thinking=False),
        "nothink+sys+noprefill": dict(use_system=True, prefill="", enable_thinking=False),
        "nothink+nosys+noprefill": dict(use_system=False, prefill="", enable_thinking=False),
        "nothink+sys+prefill_答案：空白": dict(use_system=True, prefill="答案： ", enable_thinking=False),
        "think+sys+prefill_答案：": dict(use_system=True, prefill="答案：", enable_thinking=True),
    }
    rep["variants"] = {}
    for vname, kw in variants.items():
        try:
            tr = TemplateRenderer(base_url, **kw)
        except Exception as e:  # noqa: BLE001
            rep["variants"][vname] = {"error": repr(e)}
            continue
        rows = []
        for ex in EXAMPLES:
            letters = letters_for(len(ex["question"]["options"]))
            prompt = tr.render(ex["state"], ex["question"])
            for rep_i in range(2):  # 2nd call shows cache effect
                r = read_option_probs(base_url, prompt, letters)
                r.update(id=ex["id"], gold=ex["gold"], rep=rep_i,
                         chosen=max(r["probs"], key=r["probs"].get) if r["probs"] else None)
                rows.append(r)
        rep["variants"][vname] = {"template": tr.template, "server_honored_kwargs": tr.server_honored_kwargs, "example_prompt": tr.render(EXAMPLES[0]["state"], EXAMPLES[0]["question"]), "rows": rows}
        print(f"== {vname}", flush=True)
        for r in rows:
            print(f"  {r['id']} rep{r['rep']} first={r['first_token']!r} chosen={r['chosen']} gold={r['gold']} "
                  f"p={ {k: round(v, 3) for k, v in r['probs'].items()} } missing={r['missing']} "
                  f"lat={r['latency_ms']:.0f}ms ptok={r['prompt_tokens']} cached={r['tokens_cached']} top={[t['token'] for t in r['top_tokens'][:5]]}", flush=True)

    # control arm: chat completion generating JSON
    ctrl = []
    for ex in EXAMPLES[:2]:
        msgs = build_messages(ex["state"], ex["question"], SYSTEM)
        msgs[-1]["content"] += '\n\n以 JSON 回答：{"answer": "<字母>"}'
        r = chat_json(base_url, msgs, max_tokens=64)
        r["id"] = ex["id"]
        ctrl.append(r)
        print(f"  chat {ex['id']}: {r['content']!r} reasoning={str(r['reasoning'])[:80]!r} lat={r['latency_ms']:.0f}ms usage={r['usage']}", flush=True)
    rep["chat_control"] = ctrl

    with open(os.path.join(out_dir, "smoke.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    return {k: v for k, v in rep.items() if k in ("tokenize",)} | {"variants": list(rep["variants"])}
