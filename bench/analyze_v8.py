"""v8: score the self-run JevBench public split with the vendored harness (third_party/jevbench, MIT).

Inputs : results/modal/jevbench/{tier}.jsonl (26B), results/modal/jevbench/e4b/{tier}.jsonl, optional sglang/
Arms   : raw (T=1) and temperature-scaled (T = median of results/analysis.json per-task temperatures, fitted on our
         synthetic D0 cal half, never on JevBench); fwd order only for accuracy, rev order for the flip rate.
Outputs: results/13-jevbench.md, results/v8.json, results/modal/jevbench/records_{arm}.jsonl (harness record shape)
Everything here is "self-run on the public split (231), not an official JevBench result".
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party"))
from jevbench.scoring import score_task  # noqa: E402
from jevbench.summarize import summarize  # noqa: E402
from jevbench.tasks import load_jsonl  # noqa: E402

JB = os.path.join(ROOT, "results/modal/jevbench")
TIERS = ["original", "easy", "hard"]
PEERS = [("Cygnet（凍結 Gemma-4-12B-it，vLLM，T=3.4）", "87.9%", "easy 48/48、standard 70/72、hard 85/111；官方榜第 4"),
         ("Open-Jev-27B v1.1（Qwen3.8-27B + LoRA + 決策頭，訓練）", "85.3%", "hard 72.1%"),
         ("TypeLLM（Qwen3.8-27B NVFP4，無訓練，不開思考）", "84.4%", "開思考 98.7%（每題均 919 token）"),
         ("Open-Jev-9B（訓練）", "77.5%", "hard 59.5%"),
         ("JevK5 v0.2（Qwen3.5-4B + 蒸餾 LoRA）", "—", "hard 0.739 vs 未訓練 0.613"),
         ("Open-Jev-2B（訓練）", "64.9%", "hard 41.4%")]


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def load_rows(sub):
    rows = {}
    for tier in TIERS:
        p = os.path.join(JB, sub, f"{tier}.jsonl") if sub else os.path.join(JB, f"{tier}.jsonl")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                if "fwd" in r:
                    rows[r["task_id"]] = r
    return rows


def probs_with_T(raw_logprobs, labels, T):
    lp = np.array([raw_logprobs.get(L) if raw_logprobs.get(L) is not None else -30.0 for L in labels], dtype=float) / T
    lp -= lp.max(); p = np.exp(lp); p /= p.sum()
    return {L: float(v) for L, v in zip(labels, p)}


def records_for(rows, tasks, T, order="fwd"):
    recs = []
    for t in tasks:
        r = rows.get(t.id)
        if not r:
            continue
        probs = probs_with_T(r[order]["raw_logprobs"], t.labels, T) if T != 1.0 else r[order]["probs"]
        sc = score_task(probs, t)
        recs.append({"task_id": t.id, "family": t.family, "split": t.split, "group": t.group, "ok": True, "valid": sc["valid"], "correct": sc["correct"],
                     "predicted": sc.get("predicted"), "ordinal_ev": sc.get("ordinal_ev"), "probs": sc.get("probs"), "probs_as_returned": probs,
                     "strict_valid": sc.get("strict_valid", False), "renormalized": sc.get("renormalized", False), "probs_source": "native",
                     "model": r.get("model", ""), "error": None, "status_code": 200, "latency_s": r[order]["latency_ms"] / 1000, "usage": {"input_tokens": r[order]["prompt_tokens"], "output_tokens": 1},
                     "cost_usd": None, "cost_basis": "self_hosted_gpu"})
    return recs


def main():
    tasks = {tier: load_jsonl(os.path.join(ROOT, f"data/jevbench/{tier}.jsonl")) for tier in TIERS}
    all_tasks = [t for tier in TIERS for t in tasks[tier]]
    an = json.load(open(os.path.join(ROOT, "results/analysis.json")))
    T_med = float(np.median([r["temperature"] for r in an]))
    arms = [("26B raw", "", 1.0), (f"26B T={T_med:.2f}", "", T_med), ("E4B raw", "e4b", 1.0), ("SGLang FP8 raw", "sglang", 1.0)]
    out = {"T": T_med, "arms": {}}
    L = ["# 13 — JevBench 公開子集自跑（self-run on the public split, not an official JevBench result）", "",
         f"231 題（original 72 / easy 48 / hard 111），來源與 hash 見 `data/jevbench/SOURCE.md`；評分用同 revision 的 harness（`third_party/jevbench`）。"
         f"讀法同 v2：字母第一 token logprob；英文 system prompt；每題另跑一次選項反序。溫度 T={T_med:.2f} 取自我們合成資料 D0 cal 半的每 task 溫度中位數，**不在 JevBench 上擬合**。", ""]
    table = ["| arm | 全部 acc | original | easy | hard | Brier | top-label ECE | Score 題 ordinal MAE | 嚴格有效 | 反序翻面率 | p50 s | 平均 input tokens |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sub, T in arms:
        rows = load_rows(sub)
        if not rows:
            continue
        recs = records_for(rows, all_tasks, T)
        S = summarize(all_tasks, recs)
        per = {tier: summarize(tasks[tier], [r for r in recs if r["task_id"] in {t.id for t in tasks[tier]}]) for tier in TIERS}
        recs_rev = records_for(rows, all_tasks, T, "rev")
        pred_f = {r["task_id"]: r["predicted"] for r in recs}; pred_r = {r["task_id"]: r["predicted"] for r in recs_rev}
        flip = float(np.mean([pred_f[k] != pred_r[k] for k in pred_f]))
        rev_acc = float(np.mean([bool(r["correct"]) for r in recs_rev]))
        tok = float(np.mean([r["usage"]["input_tokens"] or 0 for r in recs]))
        out["arms"][name] = {"n": S["n_attempted"], "accuracy": S["accuracy"], "per_tier": {k: v["accuracy"] for k, v in per.items()}, "brier": S["brier_mean"],
                             "ece": S["ece"]["ece"] if isinstance(S["ece"], dict) else S["ece"], "ordinal_mae": S["ordinal_mae"], "strict_valid": S["schema_validity_strict"],
                             "flip_rate": flip, "rev_accuracy": rev_acc, "latency_p50": S["latency"].get("p50"), "input_tokens": tok, "per_family": {k: v["accuracy"] for k, v in S["per_family"].items()}}
        ece = out["arms"][name]["ece"]
        table.append(f"| {name} | **{S['accuracy'] * 100:.1f}%**（{S['n_correct']}/{S['n_attempted']}） | {per['original']['accuracy'] * 100:.1f}% | {per['easy']['accuracy'] * 100:.1f}% | {per['hard']['accuracy'] * 100:.1f}% | "
                     f"{f(S['brier_mean'])} | {f(ece)} | {f(S['ordinal_mae'])} | {S['schema_validity_strict'] * 100:.0f}% | {flip * 100:.1f}%（反序 acc {rev_acc * 100:.1f}%） | {f(S['latency'].get('p50'), 2)} | {tok:.0f} |")
        if sub == "":
            with open(os.path.join(JB, f"records_{'raw' if T == 1.0 else 'T'}.jsonl"), "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    L += ["## 結果", ""] + table + ["", "## 同一子集的其他自跑數字（各自報，模型與設定不同）", "", "| 系統 | 公開 231 題 acc | 備註 |", "|---|---|---|"]
    for n, a, note in PEERS:
        L.append(f"| {n} | {a} | {note} |")
    # per family for 26B raw
    if "26B raw" in out["arms"]:
        fam = out["arms"]["26B raw"]["per_family"]
        L += ["", "## 26B raw 依題族", "", "| family | acc |", "|---|---|"] + [f"| {k} | {f(v)} |" for k, v in sorted(fam.items(), key=lambda kv: kv[1] if kv[1] is not None else 0)]
        acc = out["arms"]["26B raw"]["accuracy"]
        if acc >= 0.85:
            v = f"26B-A4B 凍結讀字母在 JevBench 公開子集 {acc * 100:.1f}%，與 Cygnet / Open-Jev 同級：**v2 的高分不是題目量身訂做**。"
        elif acc >= 0.78:
            v = f"{acc * 100:.1f}%，與 TypeLLM 無思考同級；12B dense 的 Cygnet 在英文通用題上略勝 26B-A4B MoE，產線題結論不變。"
        else:
            v = f"{acc * 100:.1f}%，明顯落後，先查 system prompt / Score 排法 / 缺字母率再重跑。"
        e4 = out["arms"].get("E4B raw", {}).get("accuracy")
        if e4 is not None:
            v += f" E4B {e4 * 100:.1f}%（低 {(acc - e4) * 100:.1f} 點，預期 5–15）。"
        fl = out["arms"]["26B raw"]["flip_rate"]
        v += f" 反序翻面率 {fl * 100:.1f}%（預期 5–10%{'，> 15% 寫進 REPORT 當風險' if fl > 0.15 else ''}）。"
        L += ["", "## 判定（`docs/handoff-v8-jevbench.md` §4）", "", v]
        out["verdict"] = v
    open(os.path.join(ROOT, "results/13-jevbench.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(ROOT, "results/v8.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(out.get("verdict"))


if __name__ == "__main__":
    main()
