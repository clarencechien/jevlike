"""Expand template generators (data/gen/{task}.py) into easy examples, merge hard examples,
attach question definitions, write data/synthetic/{task}.jsonl and MANIFEST.md.

Usage: python3 data/gen_expand.py [--seed 7] [--easy 140] [--en-share 0.2] [task ...]
"""
import argparse
import collections
import importlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
TASKS = json.load(open(os.path.join(ROOT, "seeds", "tasks.json"), encoding="utf-8"))


def question_for(task):
    t = TASKS[task]
    return {"type": t["kind"], "instructions": t["instructions"],
            "options": {k: {"label": v["label"], "criteria": v["criteria"]} for k, v in t["options"].items()}}


def expand_task(task, n_easy, en_share, seed):
    mod = importlib.import_module(f"data.gen.{task}")
    rng = random.Random(f"{task}-{seed}")
    n_en = round(n_easy * en_share)
    n_zh = n_easy - n_en
    rows, seen = [], set()
    counter = collections.Counter()

    def draw(gens, n_target, lang):
        made = 0
        tries = 0
        i = 0
        while made < n_target and tries < n_target * 40:
            tries += 1
            g = gens[i % len(gens)]
            i += 1
            sa, ga, sb, gb = g(rng)
            if sa in seen or sb in seen or sa == sb:
                continue
            assert ga != gb, f"{task}:{g.__name__} twin has same gold"
            seen.update([sa, sb])
            pid = f"{task}-ep{counter['pair']:03d}"
            counter["pair"] += 1
            for s, gold in ((sa, ga), (sb, gb)):
                rows.append({"id": f"{task}-e{counter['row']:04d}", "task": task, "state": s, "gold": gold,
                             "difficulty": "easy", "hard_type": None, "lang": lang, "pair_id": pid,
                             "source": "expand", "generator": g.__name__})
                counter["row"] += 1
                made += 1
        if made < n_target:
            print(f"  WARN {task}/{lang}: only {made}/{n_target} unique easy rows", file=sys.stderr)

    draw(mod.PAIRS_ZH, n_zh, "zh")
    draw(mod.PAIRS_EN, n_en, "en")
    return rows


def load_hard(task):
    p = os.path.join(ROOT, "hard", f"{task}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--easy", type=int, default=140)
    ap.add_argument("--en-share", type=float, default=0.2)
    ap.add_argument("tasks", nargs="*")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    manifest = ["# MANIFEST — data/synthetic", "", f"seed={a.seed}, easy per task={a.easy}, en share={a.en_share}", "",
                "| task | kind | n | easy | hard | zh | en | pairs (easy/hard) | gold distribution | hard types |", "|---|---|---|---|---|---|---|---|---|---|"]
    for task in (a.tasks or list(TASKS)):
        easy = expand_task(task, a.easy, a.en_share, a.seed)
        hard = load_hard(task)
        q = question_for(task)
        rows = []
        for r in easy + hard:
            r = dict(r)
            r["question"] = q
            rows.append(r)
        with open(os.path.join(out_dir, f"{task}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        golds = collections.Counter(r["gold"] for r in rows)
        langs = collections.Counter(r["lang"] for r in rows)
        htypes = collections.Counter(r.get("hard_type") for r in hard)
        ep = len({r["pair_id"] for r in easy if r["pair_id"]})
        hp = len({r["pair_id"] for r in hard if r.get("pair_id")})
        manifest.append(f"| {task} | {TASKS[task]['kind']} | {len(rows)} | {len(easy)} | {len(hard)} | {langs['zh']} | {langs['en']} | {ep}/{hp} | "
                        f"{', '.join(f'{k}:{v}' for k, v in sorted(golds.items()))} | {', '.join(f'{k}:{v}' for k, v in sorted(htypes.items()))} |")
        print(f"{task}: {len(rows)} rows (easy {len(easy)}, hard {len(hard)}), gold {dict(sorted(golds.items()))}")
    manifest += ["", "## 產法", "",
                 "- easy：`data/gen/{task}.py` 的孿生模板（每個 generator 回傳一對只差一個事實、gold 不同的例子）× 隨機參數，`gen_expand.py` 展開並去重；gold 由模板規格決定。",
                 "- hard：`data/hard/{task}.jsonl`，Claude（Fable）依 `data/HARD_GUIDE.md` 手寫，先定 label 再寫情境；每筆有 `rationale`。",
                 "- 出題者（Claude/Fable）與受測者（Gemma 4 26B）不同家；Fable 不作答。",
                 "- 抽查：見 `data/synthetic/SPOTCHECK.md`。"]
    open(os.path.join(out_dir, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
