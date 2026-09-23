"""Validate data/hard/*.jsonl and data/rules/*.json against data/seeds/tasks.json."""
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS = json.load(open(os.path.join(ROOT, "seeds", "tasks.json"), encoding="utf-8"))
HARD_TYPES = {"incomplete", "borderline", "noisy", "distractor", "inverted"}


def check_hard(task, path):
    errs = []
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:  # noqa: BLE001
            errs.append(f"{path}:{i} bad json: {e}")
    letters = list(TASKS[task]["options"])
    labels_text = [o["label"] for o in TASKS[task]["options"].values()]
    ids = collections.Counter(r.get("id") for r in rows)
    pairs = collections.defaultdict(list)
    golds = collections.Counter()
    langs = collections.Counter()
    types = collections.Counter()
    for r in rows:
        rid = r.get("id", "?")
        for k in ["id", "task", "state", "gold", "difficulty", "hard_type", "lang", "pair_id", "source"]:
            if k not in r:
                errs.append(f"{rid}: missing {k}")
        if r.get("task") != task:
            errs.append(f"{rid}: task mismatch {r.get('task')}")
        if r.get("gold") not in letters:
            errs.append(f"{rid}: gold {r.get('gold')} not in {letters}")
        if r.get("difficulty") != "hard":
            errs.append(f"{rid}: difficulty must be hard")
        if r.get("hard_type") not in HARD_TYPES:
            errs.append(f"{rid}: hard_type {r.get('hard_type')} not in {sorted(HARD_TYPES)}")
        if r.get("lang") not in ("zh", "en"):
            errs.append(f"{rid}: lang {r.get('lang')}")
        if not isinstance(r.get("state"), str) or len(r["state"]) < 8:
            errs.append(f"{rid}: state too short")
        if ids[rid] > 1:
            errs.append(f"{rid}: duplicate id")
        if r.get("pair_id"):
            pairs[r["pair_id"]].append(r)
        golds[r.get("gold")] += 1
        langs[r.get("lang")] += 1
        types[r.get("hard_type")] += 1
    for pid, members in pairs.items():
        if len(members) != 2:
            errs.append(f"pair {pid}: has {len(members)} members")
        elif members[0]["gold"] == members[1]["gold"]:
            errs.append(f"pair {pid}: both members have gold {members[0]['gold']}")
    n = len(rows)
    warn = []
    if n < 60:
        warn.append(f"only {n} rows (<60)")
    if len(pairs) < 15:
        warn.append(f"only {len(pairs)} pairs (<15)")
    if n and langs["en"] / n < 0.15:
        warn.append(f"en share {langs['en']}/{n}")
    for L in letters:
        if golds[L] < (min(6, n // (2 * len(letters))) if n else 0):
            warn.append(f"label {L} only {golds[L]}")
    print(f"[{task}] hard n={n} pairs={len(pairs)} gold={dict(golds)} lang={dict(langs)} types={dict(types)}"
          + (f"  WARN: {'; '.join(warn)}" if warn else ""))
    return errs


def check_rules(task, path):
    errs = []
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"{path}: bad json: {e}"]
    letters = list(TASKS[task]["options"])
    if d.get("default") not in letters:
        errs.append(f"{path}: default {d.get('default')} invalid")
    rules = d.get("rules", [])
    if not (10 <= len(rules) <= 25):
        errs.append(f"{path}: {len(rules)} rules (want 10-25)")
    for r in rules:
        try:
            re.compile(r["pattern"], re.I)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{path}: bad regex {r.get('pattern')!r}: {e}")
        if r.get("label") not in letters:
            errs.append(f"{path}: rule label {r.get('label')} invalid")
    print(f"[{task}] rules n={len(rules)} default={d.get('default')}")
    return errs


def main():
    errs = []
    only = sys.argv[1:]
    for task in TASKS:
        if only and task not in only:
            continue
        hp = os.path.join(ROOT, "hard", f"{task}.jsonl")
        rp = os.path.join(ROOT, "rules", f"{task}.json")
        if os.path.exists(hp):
            errs += check_hard(task, hp)
        else:
            print(f"[{task}] hard: MISSING")
        if os.path.exists(rp):
            errs += check_rules(task, rp)
        else:
            print(f"[{task}] rules: MISSING")
    for e in errs:
        print("ERROR", e)
    print(f"{len(errs)} errors")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
