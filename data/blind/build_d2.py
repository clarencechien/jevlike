"""Merge the two independent D2 label sets, compute agreement, and emit the D2 eval set.

D2 (agreed rows)  -> data/blind/D2/{task}.jsonl   (same schema as data/synthetic: id, task, state, gold, question, ...)
Disagreements     -> data/blind/D2/disagreements.jsonl (for human adjudication; kept as the borderline subset)
Summary           -> data/blind/D2/MANIFEST.md
"""
import collections
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = json.load(open(os.path.join(ROOT, "data/seeds/tasks.json"), encoding="utf-8"))
D2 = os.path.join(ROOT, "data/blind/D2")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def question_for(task):
    t = TASKS[task]
    return {"type": t["kind"], "instructions": t["instructions"],
            "options": {k: {"label": v["label"], "criteria": v["criteria"]} for k, v in t["options"].items()}}


gem = {}
for l in open(os.path.join(D2, "labels_gemini.jsonl"), encoding="utf-8"):
    r = json.loads(l); gem[r["id"]] = r
fab = {}
for p in sorted(glob.glob(os.path.join(D2, "labels_fable_*.jsonl"))):
    for l in open(p, encoding="utf-8"):
        r = json.loads(l); fab[r["id"]] = r
with open(os.path.join(D2, "labels_fable.jsonl"), "w", encoding="utf-8") as f:
    for r in fab.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

man = ["# D2 — Gemini 盲寫、雙標註", "", "出題：gemini-3.5-flash，只看 task 名稱、選項名稱與 20 筆 D0 語氣參考，**不看 criteria**。",
       "標註：Gemini（gemini-3.5-flash，temperature 0.2）與 Claude（Fable，三個獨立 subagent）各自拿完整 criteria 獨立標，不互看。",
       "一致的進 D2；不一致的存 `disagreements.jsonl` 待人裁決（同時是 borderline 子集）。", "",
       "| task | 寫出 | 兩邊都有標 | 一致 | 一致率 | κ | Gemini 低信心(≤0.6) | Fable 低信心 | D2 label 分布 |", "|---|---|---|---|---|---|---|---|---|"]
dis = open(os.path.join(D2, "disagreements.jsonl"), "w", encoding="utf-8")
tot = collections.Counter()
for task in TASKS:
    states = [json.loads(l) for l in open(os.path.join(D2, f"states_{task}.jsonl"), encoding="utf-8")]
    both = [s for s in states if s["id"] in gem and s["id"] in fab]
    a = [gem[s["id"]]["label"] for s in both]; b = [fab[s["id"]]["label"] for s in both]
    agree = [s for s in both if gem[s["id"]]["label"] == fab[s["id"]]["label"]]
    q = question_for(task)
    dist = collections.Counter()
    with open(os.path.join(D2, f"{task}.jsonl"), "w", encoding="utf-8") as f:
        for s in agree:
            g = gem[s["id"]]["label"]; dist[g] += 1
            f.write(json.dumps({"id": s["id"], "task": task, "state": s["state"], "gold": g, "difficulty": "blind", "hard_type": None,
                                "lang": "en" if not any("一" <= c <= "鿿" for c in s["state"]) else "zh", "pair_id": None,
                                "source": "gemini-blind", "question": q,
                                "conf_gemini": gem[s["id"]].get("confidence"), "conf_fable": fab[s["id"]].get("confidence")}, ensure_ascii=False) + "\n")
    for s in both:
        if gem[s["id"]]["label"] != fab[s["id"]]["label"]:
            dis.write(json.dumps({"id": s["id"], "task": task, "state": s["state"], "gemini": gem[s["id"]], "fable": fab[s["id"]], "adjudicated": None}, ensure_ascii=False) + "\n")
    k = kappa(a, b) if both else float("nan")
    lowg = sum(1 for s in both if (gem[s["id"]].get("confidence") or 1) <= 0.6)
    lowf = sum(1 for s in both if (fab[s["id"]].get("confidence") or 1) <= 0.6)
    man.append(f"| {task} | {len(states)} | {len(both)} | {len(agree)} | {len(agree) / max(1, len(both)):.2f} | {k:.2f} | {lowg} | {lowf} | {', '.join(f'{x}:{y}' for x, y in sorted(dist.items()))} |")
    tot["states"] += len(states); tot["both"] += len(both); tot["agree"] += len(agree)
    print(f"{task:18s} both={len(both)} agree={len(agree)} rate={len(agree) / max(1, len(both)):.2f} kappa={k:.2f}")
dis.close()
man.append(f"| **全部** | {tot['states']} | {tot['both']} | {tot['agree']} | {tot['agree'] / max(1, tot['both']):.2f} | | | | |")
open(os.path.join(D2, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(man) + "\n")
print("agreed total", tot["agree"], "/", tot["both"])
