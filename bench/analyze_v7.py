"""v7 P1 analysis: packed readout vs separate (docs/handoff-v7-borrowed.md §P1).

Inputs : results/modal/packed/P1/{task}.jsonl (placeholder '_', K=3, all 600 alarm states)
         results/modal/packed/P1-phA/  (placeholder 'A', 100 states/task)
         results/modal/packed/P1-K7/   (K=7 with 4 extra yes/no questions, 100 states/task; speed only)
Outputs: results/12-packed.md, results/v7.json["packed"]
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import group_split  # noqa: E402
from analyze_ladder import mcnemar  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PK = os.path.join(ROOT, "results/modal/packed")
ALARM = ["m_alarm_category", "m_alarm_severity", "m_needs_dispatch"]


def rows_of(sub, task):
    p = os.path.join(PK, sub, f"{task}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if '"per_q"' in l]


def test_ids(task):
    data = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8")]
    cal = group_split(data, 0)
    return {r["id"] for r, c in zip(data, cal) if not c}


def summarize(sub, test_only=True):
    out = {}
    for t in ALARM:
        rows = rows_of(sub, t)
        if not rows:
            continue
        te = test_ids(t)
        rr = [r for r in rows if r["id"] in te] if test_only else rows
        if not rr:
            continue
        cs = np.array([r["sep_correct"] for r in rr]); cp = np.array([r["packed_correct"] for r in rr])
        _, _, p = mcnemar(cs, cp)
        K = rr[0]["K"]
        interf = sum(r["interference"] for r in rows) / (len(rows) * K)
        # interference on own question only vs the later (position >= 1) questions
        own_flip = np.mean([r["per_q"][r["own_index"]]["sep"] != r["per_q"][r["own_index"]]["packed"] for r in rows])
        later_flip = np.mean([x["sep"] != x["packed"] for r in rows for i, x in enumerate(r["per_q"]) if i > 0])
        sep_ms = np.median([r["sep_ms"] for r in rows[1:]]); pk_ms = np.median([r["packed_ms"] for r in rows[1:]])
        out[t] = {"n_test": len(rr), "n_all": len(rows), "K": K, "acc_sep": float(cs.mean()), "acc_packed": float(cp.mean()), "p": p,
                  "ok_acc": abs(cp.mean() - cs.mean()) <= 0.01 and p >= 0.05, "interference": float(interf), "own_flip": float(own_flip),
                  "later_flip": float(later_flip), "sep_ms": float(sep_ms), "packed_ms": float(pk_ms), "tokens": float(np.median([r["n_tokens"] for r in rows])),
                  "missing_rate": float(np.mean([bool(x["missing_packed"]) for r in rows for x in r["per_q"]]))}
    return out


def main():
    L = ["# 12 — packed readout：K 題一個前向（SGLang）", "",
         "門檻預先登記於 `docs/handoff-v7-borrowed.md` §P1。alarm 家族三題打包（K=3），每個 state 只有自己 task 的那題有 gold；separate = 第一題單獨（暖前綴）+ 其餘整批；packed = 一個請求，讀每個占位符位置的字母 logprob。L40S、FP8、含 `<bos>`。", ""]
    out = {}
    main_ = summarize("P1")
    if main_:
        L += ["## 準確率與干擾（占位符 `_`，D0 test 100 筆／task；干擾率用全部 200 筆）", "",
              "| task | K | separate acc | packed acc | 差 | McNemar p | 干擾率（所有題） | 自己那題翻 | 後面的題翻 | 缺字母 | tokens | separate ms | packed ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, v in main_.items():
            L.append(f"| {t} | {v['K']} | {v['acc_sep']:.3f} | {v['acc_packed']:.3f} | {v['acc_packed'] - v['acc_sep']:+.3f} | {v['p']:.3f} | {v['interference'] * 100:.1f}% | {v['own_flip'] * 100:.1f}% | {v['later_flip'] * 100:.1f}% | {v['missing_rate']:.3f} | {v['tokens']:.0f} | {v['sep_ms']:.0f} | {v['packed_ms']:.0f} |")
        acc_ok = sum(v["ok_acc"] for v in main_.values()); interf = float(np.mean([v["interference"] for v in main_.values()]))
        speed3 = float(np.mean([v["packed_ms"] / v["sep_ms"] for v in main_.values()]))
        out["P1"] = {"tasks": main_, "acc_ok": acc_ok, "interference": interf, "speed_ratio_K3": speed3}
    k7 = summarize("P1-K7", test_only=False)
    if k7:
        L += ["", "## K=7（3 題 + 4 個一般是非題，只量速度，100 state／task）", "", "| task | tokens | separate ms | packed ms | packed/separate | 干擾率 |", "|---|---|---|---|---|---|"]
        for t, v in k7.items():
            L.append(f"| {t} | {v['tokens']:.0f} | {v['sep_ms']:.0f} | {v['packed_ms']:.0f} | {v['packed_ms'] / v['sep_ms']:.2f} | {v['interference'] * 100:.1f}% |")
        out["P1_K7"] = {"tasks": k7, "speed_ratio_K7": float(np.mean([v["packed_ms"] / v["sep_ms"] for v in k7.values()]))}
    pha = summarize("P1-phA", test_only=False)
    if pha:
        L += ["", "## 占位符換成 `A`（100 state／task，全部筆）", "", "| task | packed acc（`A`） | packed acc（`_`，同 100 筆） | 干擾率（`A`） |", "|---|---|---|---|"]
        for t, v in pha.items():
            base = rows_of("P1", t)[:v["n_all"]]
            acc_base = float(np.mean([r["packed_correct"] for r in base])) if base else float("nan")
            L.append(f"| {t} | {v['acc_packed']:.3f} | {acc_base:.3f} | {v['interference'] * 100:.1f}% |")
        out["P1_phA"] = pha
    if main_:
        s7 = out.get("P1_K7", {}).get("speed_ratio_K7", float("nan"))
        gate_acc = acc_ok == 3; gate_int = interf <= 0.05; gate_speed = speed3 <= 0.6 and (math.isnan(s7) or s7 <= 0.4)
        if gate_acc and gate_int and gate_speed:
            v = "**三項全過 → packed 進 SGLang 批次路徑的預設**"
        elif gate_acc and 0.05 < interf <= 0.10:
            v = "**準確率過、干擾率 5–10% → 只給離線批次，即時不用**"
        elif not gate_acc:
            v = "**準確率沒過 → 不採用**"
        else:
            v = "**準確率與干擾率過、速度門檻沒過 → 不採用**（省的只是請求開銷）"
        L += ["", f"門檻：每 task ±1 點且 p ≥ 0.05（{acc_ok}/3）；干擾率 ≤ 5%（{interf * 100:.1f}%）；K=3 packed ≤ separate × 0.6（{speed3:.2f}）、K=7 ≤ × 0.4（{s7:.2f}）。判定：{v}", ""]
        out["P1"]["verdict"] = v
    open(os.path.join(ROOT, "results/12-packed.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    v7p = os.path.join(ROOT, "results/v7.json")
    v7 = json.load(open(v7p)) if os.path.exists(v7p) else {}
    v7["packed"] = out
    json.dump(v7, open(v7p, "w"), ensure_ascii=False, indent=1, default=float)
    print(out.get("P1", {}).get("verdict"))


if __name__ == "__main__":
    main()
