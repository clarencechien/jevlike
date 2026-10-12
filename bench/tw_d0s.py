"""tw extra (docs/handoff-tw-script.md §6): the D0 factory set rewritten in Simplified characters (OpenCC t2s), to see whether
the script switch seen on the sovereignty items also moves our own production decisions.

  python3 bench/tw_d0s.py make      -> data/tw/D0_s/{task}.jsonl (state, instructions, option labels and criteria converted)
  python3 bench/tw_d0s.py compare   -> per task accuracy Trad vs Simp, argmax agreement, max |dP(gold)|  (results/tw_d0s.json)
Runs: modal_tw.py::d0s (same container reads Traditional then Simplified).
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, DST = os.path.join(REPO, "data", "synthetic"), os.path.join(REPO, "data", "tw", "D0_s")
TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "q_defect_root",
         "p_uph_anomaly", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]
RES = os.path.join(REPO, "results", "modal", "tw", "d0s", "26b")


def make():
    import opencc
    t2s = opencc.OpenCC("t2s")
    os.makedirs(DST, exist_ok=True)
    n = changed = 0
    for t in TASKS:
        out = []
        for l in open(os.path.join(SRC, f"{t}.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            s = dict(r, state=t2s.convert(r["state"]))
            q = dict(r["question"], instructions=t2s.convert(r["question"]["instructions"]))
            q["options"] = {k: ({kk: t2s.convert(vv) if isinstance(vv, str) else vv for kk, vv in v.items()} if isinstance(v, dict) else t2s.convert(v))
                            for k, v in r["question"]["options"].items()}
            s["question"] = q
            n += 1
            changed += s["state"] != r["state"]
            out.append(json.dumps(s, ensure_ascii=False))
        open(os.path.join(DST, f"{t}.jsonl"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"{n} rows, {changed} states changed by t2s")


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def compare():
    tot = {"n": 0, "agree": 0, "acc_t": 0, "acc_s": 0}
    per = {}
    for t in TASKS:
        a, b = load(os.path.join(RES, "trad", f"{t}.jsonl")), load(os.path.join(RES, "simp", f"{t}.jsonl"))
        ids = sorted(set(a) & set(b))
        arg = lambda r: max(r["probs"], key=r["probs"].get) if r.get("probs") else None  # noqa: E731
        agree = sum(arg(a[i]) == arg(b[i]) for i in ids)
        acc_t = sum(arg(a[i]) == a[i]["gold"] for i in ids)
        acc_s = sum(arg(b[i]) == b[i]["gold"] for i in ids)
        dg = max(abs(a[i]["probs"].get(a[i]["gold"], 0) - b[i]["probs"].get(b[i]["gold"], 0)) for i in ids)
        per[t] = {"n": len(ids), "acc_trad": acc_t / len(ids), "acc_simp": acc_s / len(ids), "agree": agree / len(ids), "max_dp_gold": dg}
        for k, v in (("n", len(ids)), ("agree", agree), ("acc_t", acc_t), ("acc_s", acc_s)):
            tot[k] += v
    out = {"per_task": per, "all": {"n": tot["n"], "acc_trad": tot["acc_t"] / tot["n"], "acc_simp": tot["acc_s"] / tot["n"],
                                    "agree": tot["agree"] / tot["n"]}}
    json.dump(out, open(os.path.join(REPO, "results", "tw_d0s.json"), "w"), indent=1)
    print("| task | n | acc Trad | acc Simp | argmax agree | max dP(gold) |\n|---|---|---|---|---|---|")
    for t, v in per.items():
        print(f"| {t} | {v['n']} | {v['acc_trad']:.3f} | {v['acc_simp']:.3f} | {v['agree']:.3f} | {v['max_dp_gold']:.2f} |")
    o = out["all"]
    print(f"| all | {o['n']} | {o['acc_trad']:.3f} | {o['acc_simp']:.3f} | {o['agree']:.3f} | |")


if __name__ == "__main__":
    {"make": make, "compare": compare}[sys.argv[1]]()
