"""D1 perturbation sets from the D0 test split (labels unchanged).

D1-typo : zh — replace 1–2 phrases using data/seeds/typos_zh.json; en — swap/drop 1–2 characters.
D1-cue  : remove lexical overlap between state and the task's criteria (char 2–4-grams),
          replacing each overlapping span with a generic phrase; records n_cues_removed.
D1-mix  : prepend/append an unrelated sentence from another task's state on the same line.

Test split = the same group split analyze.py uses (seed 0), so D1 rows are exactly the D0 test rows.
Usage: python3 data/perturb.py [--seed 0]
"""
import argparse
import collections
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = json.load(open(os.path.join(ROOT, "data/seeds/tasks.json"), encoding="utf-8"))
TYPOS = json.load(open(os.path.join(ROOT, "data/seeds/typos_zh.json"), encoding="utf-8"))["pairs"]
OUT = os.path.join(ROOT, "data/perturbed")
GENERIC = ["狀況", "數值異常", "有狀況", "情形", "問題"]
STOP = {"機台", "產線", "工單", "料號", "操作員", "工程師", "分鐘", "小時", "目前", "已經", "沒有", "可以", "需要", "或", "的", "了", "與", "及", "和",
        "SMT", "AOI", "SPI", "DEK", "NXT", "MES", "UPH", "SPC", "QE", "ME", "PE", "PCB", "feeder", "slot", "alarm", "the", "and", "for"}


def load_rows(task):
    return [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8")]


def test_mask(rows, seed):
    groups = sorted({r.get("pair_id") or r["id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    cal = set(groups[: len(groups) // 2])
    return [(r.get("pair_id") or r["id"]) not in cal for r in rows]


def criteria_ngrams(task):
    text = TASKS[task]["instructions"] + " " + " ".join(v["label"] + " " + v["criteria"] for v in TASKS[task]["options"].values())
    zh = re.findall(r"[一-鿿]+", text)
    grams = set()
    for w in zh:
        for n in (4, 3, 2):
            for i in range(len(w) - n + 1):
                g = w[i:i + n]
                if g not in STOP:
                    grams.add(g)
    en = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)} - {s.lower() for s in STOP}
    return grams, en


def perturb_typo(state, lang, rng):
    if lang == "zh":
        cands = [(c, t) for c, t in TYPOS if c in state]
        rng.shuffle(cands)
        n = 0
        for c, t in cands:
            if n >= rng.choice([1, 2]):
                break
            state = state.replace(c, t, 1)
            n += 1
        return state, n
    # en: character swap / drop in 1–2 words of length >= 5
    words = state.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    rng.shuffle(idx)
    n = 0
    for i in idx[: rng.choice([1, 2])]:
        w = words[i]
        k = rng.randint(1, len(w) - 2)
        words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:] if rng.random() < 0.5 else w[:k] + w[k + 1:]
        n += 1
    return " ".join(words), n


def perturb_cue(state, task, grams, en, rng):
    removed = []
    # longest-first so 4-grams are replaced before their 2-gram parts
    for g in sorted(grams, key=len, reverse=True):
        if g in state:
            state = state.replace(g, rng.choice(GENERIC))
            removed.append(g)
    for w in sorted(en, key=len, reverse=True):
        pat = re.compile(r"\b" + re.escape(w) + r"\b", re.I)
        if pat.search(state):
            state = pat.sub("ISSUE", state)
            removed.append(w)
    return state, removed


def perturb_mix(state, other_states, rng):
    extra = rng.choice(other_states)
    return (extra + "；" + state) if rng.random() < 0.5 else (state + "；" + extra), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    all_rows = {t: load_rows(t) for t in TASKS}
    manifest = ["# D1 perturbation sets (from D0 test split, seed %d)" % a.seed, "",
                "| task | n | typo: mean edits | cue: mean removed / rows with ≥1 / distinct cues | mix: n |", "|---|---|---|---|---|"]
    for task in TASKS:
        rows = all_rows[task]
        mask = test_mask(rows, a.seed)
        test = [r for r, m in zip(rows, mask) if m]
        grams, en = criteria_ngrams(task)
        others = [r["state"] for t2, rs in all_rows.items() if t2 != task for r in rs if r["lang"] == "zh" and len(r["state"]) < 70]
        rng = random.Random(f"{task}-{a.seed}")
        stats = collections.defaultdict(list)
        cue_counter = collections.Counter()
        for kind in ("D1-typo", "D1-cue", "D1-mix"):
            os.makedirs(os.path.join(OUT, kind), exist_ok=True)
            with open(os.path.join(OUT, kind, f"{task}.jsonl"), "w", encoding="utf-8") as f:
                for r in test:
                    r2 = dict(r)
                    r2["source_id"] = r["id"]
                    r2["id"] = f"{r['id']}-{kind}"
                    if kind == "D1-typo":
                        r2["state"], n = perturb_typo(r["state"], r["lang"], rng)
                        r2["n_edits"] = n
                        stats["typo"].append(n)
                    elif kind == "D1-cue":
                        r2["state"], removed = perturb_cue(r["state"], task, grams, en, rng)
                        r2["n_cues_removed"] = len(removed)
                        r2["cues_removed"] = removed
                        stats["cue"].append(len(removed))
                        cue_counter.update(removed)
                    else:
                        r2["state"], extra = perturb_mix(r["state"], others, rng)
                        r2["mixed_in"] = extra
                    f.write(json.dumps(r2, ensure_ascii=False) + "\n")
        c = stats["cue"]
        manifest.append(f"| {task} | {len(test)} | {sum(stats['typo']) / len(test):.2f} | {sum(c) / len(test):.2f} / {sum(1 for x in c if x) / len(test):.0%} / {len(cue_counter)} | {len(test)} |")
        manifest.append(f"|  | top cues | | {', '.join(f'{k}×{v}' for k, v in cue_counter.most_common(8))} | |")
        print(task, "test", len(test), "typo edits", f"{sum(stats['typo']) / len(test):.2f}", "cues", f"{sum(c) / len(test):.2f}", cue_counter.most_common(5))
    manifest += ["", "D1-cue 人工抽查結果見 SPOTCHECK-D1.md。"]
    open(os.path.join(OUT, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
