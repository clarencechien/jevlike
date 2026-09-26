"""v5 analysis (handoff v5 §2): T1 JSON-prefill variants and T2 permutation averaging.

Inputs : results/modal/accuracy/{task}.jsonl                (V0 baseline, v2 run)
         results/modal/accuracy/T1-V1/, T1-V2/{task}.jsonl  (T1)
         results/modal/permute/T2-*/{task}.jsonl            (T2)
Outputs: results/08-typellm-followups.md, results/v5.json
Gates are the pre-registered ones in docs/handoff-v5-typellm.md.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ece, group_split  # noqa: E402
from analyze_ladder import auroc, mcnemar  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(ROOT, "results/modal/accuracy")
PERM = os.path.join(ROOT, "results/modal/permute")
WEAK = ["m_alarm_severity", "q_spc_action", "p_uph_anomaly"]
CONTROL = ["m_alarm_category"]
TASKS = WEAK + CONTROL


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def rows_of(path, key="probs"):
    if not os.path.exists(path):
        return {}
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if "gold" in r and key in r:
            out[r["id"]] = r
    return out


def test_ids(task):
    data = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8")]
    cal = group_split(data, 0)
    return {r["id"] for r, c in zip(data, cal) if not c}


def metrics(probs_list, golds):
    conf = np.array([max(p.values()) if p else 0.0 for p in probs_list])
    correct = np.array([(max(p, key=p.get) if p else None) == g for p, g in zip(probs_list, golds)], dtype=float)
    sel = conf >= 0.9
    return {"acc": float(correct.mean()), "ece": ece(conf, correct), "auroc": auroc(correct, conf),
            "sel90_acc": float(correct[sel].mean()) if sel.any() else float("nan"), "sel90_cov": float(sel.mean()),
            "conf_mean": float(conf.mean()), "correct": correct}


def main():
    out = {"T1": {}, "T2": {}}
    L = ["# 08 — 參考 TypeLLM 之後的補測（T1 JSON prefill、T2 選項順序置換平均）", "",
         "門檻預先登記於 `docs/handoff-v5-typellm.md` §2。L4、UD-Q4_K_M、`--swa-full`，D0 test（pair_id 分組 seed 0），與 `04-accuracy.md` 同一把尺。", ""]

    # ---------------- T1
    L += ["## T1 JSON 形狀 prefill（test 100 筆／task）", "",
          "V0 現行（下一 token 即字母）；V1 = V0 + prefill `{\"answer\": \"`；V2 = V1 + 指令改為「以 {\"answer\": \"<字母>\"} 回答」。", "",
          "| task | V0 acc | V1 acc | V2 acc | V1−V0 | V2−V0 | McNemar p V1 / V2 | ECE V0 / V1 / V2 | first-token 非字母 V1 / V2 |", "|---|---|---|---|---|---|---|---|---|"]
    t1 = {}
    for t in TASKS:
        base = rows_of(os.path.join(ACC, f"{t}.jsonl"))
        v1 = rows_of(os.path.join(ACC, "T1-V1", f"{t}.jsonl"))
        v2 = rows_of(os.path.join(ACC, "T1-V2", f"{t}.jsonl"))
        if not base or not (v1 or v2):
            continue
        ids = sorted(test_ids(t) & set(base) & (set(v1) or set(base)) & (set(v2) or set(base)))
        golds = [base[i]["gold"] for i in ids]
        m0 = metrics([base[i]["probs"] for i in ids], golds)
        res = {"n": len(ids), "V0": m0["acc"], "ece0": m0["ece"]}
        cells = {}
        for name, rr in (("V1", v1), ("V2", v2)):
            if not rr:
                cells[name] = (float("nan"), float("nan"), float("nan"), float("nan"))
                continue
            m = metrics([rr[i]["probs"] for i in ids], golds)
            _, _, p = mcnemar(m0["correct"], m["correct"])
            bad = sum(1 for i in ids if not (rr[i].get("first_token") or "").strip()[:1].isalpha())
            res[name] = m["acc"]; res[f"p_{name}"] = p; res[f"ece_{name}"] = m["ece"]; res[f"badfirst_{name}"] = bad
            cells[name] = (m["acc"], p, m["ece"], bad)
        t1[t] = res
        L.append(f"| {t} | {f(m0['acc'])} | {f(cells['V1'][0])} | {f(cells['V2'][0])} | {f(cells['V1'][0] - m0['acc'], 3)} | {f(cells['V2'][0] - m0['acc'], 3)} | "
                 f"{f(cells['V1'][1])} / {f(cells['V2'][1])} | {f(m0['ece'])} / {f(cells['V1'][2])} / {f(cells['V2'][2])} | {cells['V1'][3]} / {cells['V2'][3]} |")
    verdict1 = None
    if t1:
        def gate(name):
            if any(math.isnan(t1[t].get(name, float("nan"))) for t in t1):
                return None
            weak = [t for t in WEAK if t in t1]
            gain = np.mean([t1[t][name] - t1[t]["V0"] for t in weak])
            no_drop = all(t1[t][name] >= t1[t]["V0"] - 0.01 for t in t1)
            return {"weak_gain": float(gain), "no_task_drops": no_drop, "pass": bool(gain >= 0.02 and no_drop)}
        g = {name: gate(name) for name in ("V1", "V2")}
        out["T1"] = {"tasks": t1, "gates": g}
        passed = [n for n in ("V1", "V2") if g[n] and g[n]["pass"]]
        if passed:
            best = max(passed, key=lambda n: (np.mean([t1[t][n] for t in WEAK if t in t1]), n == "V1"))
            verdict1 = f"**採用 {best}**（弱 task 平均 +{g[best]['weak_gain'] * 100:.1f} 點，無 task 掉 >1 點）"
        else:
            verdict1 = "**不採用**：" + "；".join(f"{n} 弱 task 平均 {g[n]['weak_gain'] * 100:+.1f} 點" + ("" if g[n]["no_task_drops"] else "、有 task 掉 >1 點") for n in ("V1", "V2") if g[n])
        L += ["", f"門檻：弱 task 平均 ≥ +2 點且無 task（含控制）低於 V0 − 1 點。判定：{verdict1}", ""]
        out["T1"]["verdict"] = verdict1

    # ---------------- T2
    for sub in sorted(glob.glob(os.path.join(PERM, "T2-*"))):
        tag = os.path.basename(sub)
        L += [f"## T2 選項順序置換平均（{tag}，test 100 筆／task）", "",
              "單次 = 原順序一次前向；平均 = 全部順序（3 選項 6 種、其餘 8 種）機率平均。順序敏感度 = 同一題不同順序 argmax 不一致的比例。", "",
              "| task | K | 順序數 | 單次 acc | 平均 acc | 差 | McNemar p | ECE 單次 / 平均 | AUROC 單次 / 平均 | sel@0.9 acc（cov）單次 / 平均 | 順序敏感度 | 每題 ms |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        t2 = {}
        for t in TASKS:
            rr = rows_of(os.path.join(sub, f"{t}.jsonl"), key="probs_mean")
            if not rr:
                continue
            ids = sorted(test_ids(t) & set(rr))
            golds = [rr[i]["gold"] for i in ids]
            ms = metrics([rr[i]["probs_single"] for i in ids], golds)
            mm = metrics([rr[i]["probs_mean"] for i in ids], golds)
            _, _, p = mcnemar(ms["correct"], mm["correct"])
            sens = float(np.mean([rr[i]["order_sensitive"] for i in ids]))
            row_ms = float(np.median([rr[i]["row_ms"] for i in ids]))
            t2[t] = {"n": len(ids), "K": rr[ids[0]]["K"], "n_orders": rr[ids[0]]["n_orders"], "single": ms["acc"], "mean": mm["acc"], "p": p,
                     "ece_single": ms["ece"], "ece_mean": mm["ece"], "auroc_single": ms["auroc"], "auroc_mean": mm["auroc"],
                     "sel90_single": (ms["sel90_acc"], ms["sel90_cov"]), "sel90_mean": (mm["sel90_acc"], mm["sel90_cov"]),
                     "order_sensitivity": sens, "row_ms": row_ms}
            L.append(f"| {t} | {t2[t]['K']} | {t2[t]['n_orders']} | {f(ms['acc'])} | {f(mm['acc'])} | {mm['acc'] - ms['acc']:+.3f} | {f(p)} | "
                     f"{f(ms['ece'])} / {f(mm['ece'])} | {f(ms['auroc'], 2)} / {f(mm['auroc'], 2)} | "
                     f"{f(ms['sel90_acc'])}（{ms['sel90_cov']:.2f}）/ {f(mm['sel90_acc'])}（{mm['sel90_cov']:.2f}） | {sens * 100:.0f}% | {row_ms:.0f} |")
        if t2:
            weak = [t for t in WEAK if t in t2]
            gain = float(np.mean([t2[t]["mean"] - t2[t]["single"] for t in weak]))
            ece_half = all(t2[t]["ece_mean"] <= 0.5 * t2[t]["ece_single"] for t in weak) and all(t2[t]["mean"] >= t2[t]["single"] for t in weak)
            ctrl_ok = all(t2[t]["mean"] >= t2[t]["single"] - 0.01 for t in CONTROL if t in t2)
            passed = bool((gain >= 0.02 or ece_half) and ctrl_ok)
            risky = [t for t in t2 if t2[t]["order_sensitivity"] >= 0.10]
            verdict2 = ("**建議用在升級路徑／離線**" if passed else "**不建議**") + f"（弱 task 平均 {gain * 100:+.1f} 點；ECE 減半：{'是' if ece_half else '否'}；控制 task 不掉：{'是' if ctrl_ok else '否'}）"
            if risky:
                verdict2 += f"。順序敏感度 ≥10% 的 task：{', '.join(risky)}（寫進 REPORT 當風險）"
            L += ["", f"門檻：弱 task 平均 acc ≥ 單次 + 2 點，或 ECE 降一半且 acc 不掉；控制 task 不掉超過 1 點。判定：{verdict2}", ""]
            out["T2"][tag] = {"tasks": t2, "gain": gain, "ece_half": ece_half, "ctrl_ok": ctrl_ok, "pass": passed, "risky": risky, "verdict": verdict2}

    open(os.path.join(ROOT, "results/08-typellm-followups.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(ROOT, "results/v5.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({"T1": out["T1"].get("verdict"), "T2": {k: v["verdict"] for k, v in out["T2"].items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
