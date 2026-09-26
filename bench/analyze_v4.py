"""v4 gate analysis: A0 (llama UD-Q4_K_M), A1 (llama Q8_0), B1 (SGLang FP8) on L40S.

Inputs : results/modal/v4bench/v4.json (A0), results/modal/v4bench/q8/v4.json (A1), results/modal/v4bench/sglang/v4.json (B1)
         results/modal/accuracy/{q8,sglang}/{D0,D1-cue}/{task}.jsonl (+ 26b D0/D1-cue from v2/v3)
Outputs: results/07-sglang-speed.md, results/07-sglang-accuracy.md, results/v4.json
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import TASKS, group_split  # noqa: E402
from analyze_ladder import auroc, boot_ci, conf_of, mcnemar  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, "results/modal/v4bench")
ACC = os.path.join(ROOT, "results/modal/accuracy")
ARMS = {"A0": "v4.json", "A1": "q8/v4.json", "B1": "sglang/v4.json"}
LABEL = {"A0": "A0 llama-server UD-Q4_K_M", "A1": "A1 llama-server Q8_0", "B1": "B1 SGLang FP8"}


def f(x, d=0):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def load_arm(arm):
    p = os.path.join(V4, ARMS[arm])
    if not os.path.exists(p):
        return None
    r = json.load(open(p))["results"]
    for alt in (p.replace("v4.json", "v4_L4seq.json"), p.replace("v4.json", "v4_L4warm.json")):  # alternative L4 strategies; keep the best per K
        if not os.path.exists(alt):
            continue
        r2 = json.load(open(alt))["results"]
        for K in (1, 5, 10, 16):
            k = f"L4_shared_{K}q"
            if k in r2 and (k not in r or r2[k]["summary"]["p50"] < r[k]["summary"]["p50"]):
                r[k] = dict(r2[k], note=r2[k].get("note", "") + " (seq chosen)")
        for k in ("S3_d0_test_1001", "S2_meeting_120x7"):
            if k in r2 and k not in r:
                r[k] = r2[k]
    return r


def acc_rows(arm, ds, task):
    p = {"A0": os.path.join(ACC, f"{task}.jsonl") if ds == "D0" else os.path.join(ACC, ds, f"{task}.jsonl"),
         "A1": os.path.join(ACC, "q8", ds, f"{task}.jsonl"), "B1": os.path.join(ACC, "sglang", ds, f"{task}.jsonl"),
         "A2": os.path.join(ACC, "bf16", ds, f"{task}.jsonl"), "B2": os.path.join(ACC, "sglang-bf16", ds, f"{task}.jsonl"),
         "B1w1": os.path.join(ACC, "sglang", ds + "-w1", f"{task}.jsonl"), "B1nr": os.path.join(ACC, "sglang", ds + "-noradix", f"{task}.jsonl")}[arm]
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if "correct" in json.loads(l)}


def main():
    R = {a: load_arm(a) for a in ARMS}
    have = [a for a in ARMS if R[a]]
    out = {"arms": have, "gates": {}, "speed": {}}
    L = ["# 07 — SGLang 後端評估：速度（L40S，同題、同 prompt 字串）", "",
         "門檻預先登記於 `07-sglang-prereg.md`。原始：`results/modal/v4bench/*/v4.json`。llama-server 用 `-np` 多 slot + cache_prompt + `--swa-full` 並行送出；SGLang 用 list 一次送出（RadixAttention）。", ""]

    def g(arm, key, sub="summary", field="p50"):
        r = R.get(arm) or {}
        v = r.get(key, {})
        return (v.get(sub) or {}).get(field) if sub else v.get(field)

    # ---- table L1/L2
    L += ["## L1 / L2 單題延遲 p50 / p95（ms）", "", "| 條件 | " + " | ".join(LABEL[a] for a in have) + " |", "|---|" + "---|" * len(have)]
    for key, lab in [("L1_single_100tok_2opt", "L1 單題 ~100-tok state"), ("L1_ref_same_prompt_repeated", "decode 地板（全 cache）"),
                     ("L2_state_500tok", "L2 state 500 tok"), ("L2_state_2000tok", "L2 state 2000 tok")]:
        L.append(f"| {lab} | " + " | ".join(f"{f(g(a, key))} / {f(g(a, key, field='p95'))}" for a in have) + " |")
    # ---- L4
    L += ["", "## L4 多題共用 state（~1,100 tok），K 題端到端 p50 ms（每題 cached tokens 命中率）", "", "| K | " + " | ".join(LABEL[a] for a in have) + " |", "|---|" + "---|" * len(have)]
    for K in (1, 5, 10, 16):
        L.append(f"| {K} | " + " | ".join(f"{f(g(a, f'L4_shared_{K}q'))}（{f((R[a].get(f'L4_shared_{K}q') or {}).get('prefix_hit_rate', float('nan')) * 100)}%）" for a in have) + " |")
    # ---- L5
    L += ["", "## L5 併發：decisions/s（p50 ms）", "", "| c | " + " | ".join(LABEL[a] for a in have) + " |", "|---|" + "---|" * len(have)]
    for c in (1, 8, 32):
        L.append(f"| {c} | " + " | ".join(f"{f(g(a, f'L5_concurrency_{c}', field='throughput_rps'), 1)}（{f(g(a, f'L5_concurrency_{c}'))}）" for a in have) + " |")
    # ---- L6
    L += ["", "## L6 背景生成 512 token 時的決策延遲", "", "| | " + " | ".join(LABEL[a] for a in have) + " |", "|---|" + "---|" * len(have)]
    L.append("| p50 無/有負載 | " + " | ".join(f"{f(g(a, 'L1_single_100tok_2opt'))} → {f(g(a, 'L6_with_bg_generation'))}" for a in have) + " |")
    L.append("| p95 退化倍數 | " + " | ".join(f"{f((R[a].get('L6_with_bg_generation') or {}).get('slowdown_p95', float('nan')), 2)}x" for a in have) + " |")
    L.append("| 背景生成 tokens | " + " | ".join(str((R[a].get('L6_with_bg_generation') or {}).get('bg_generation', {}).get('tokens', '—')) for a in have) + " |")
    # ---- S2/S3
    L += ["", "## S2 會議負載（120 視窗 × 7 題）與 S3 D0 test 1,001 題端到端", "", "| | " + " | ".join(LABEL[a] for a in have) + " |", "|---|" + "---|" * len(have)]
    for key, lab in [("S2_meeting_120x7", "S2 wall s（decisions/s）"), ("S3_d0_test_1001", "S3 wall s（decisions/s）")]:
        L.append(f"| {lab} | " + " | ".join(f"{f((R[a].get(key) or {}).get('wall_s', float('nan')), 1)}（{f((R[a].get(key) or {}).get('decisions_per_s', float('nan')), 1)}）" for a in have) + " |")
    # ---- gates
    gates = {}
    if "A1" in have and "B1" in have:
        a1, b1 = R["A1"], R["B1"]
        g1 = b1["L1_single_100tok_2opt"]["summary"]["p50"] <= a1["L1_single_100tok_2opt"]["summary"]["p50"] * 1.10
        r10 = a1["L4_shared_10q"]["summary"]["p50"] / b1["L4_shared_10q"]["summary"]["p50"]
        r16 = a1["L4_shared_16q"]["summary"]["p50"] / b1["L4_shared_16q"]["summary"]["p50"]
        g2 = r10 >= 1.5 and r16 >= 1.5
        g3 = all(b1[f"L5_concurrency_{c}"]["summary"]["throughput_rps"] >= a1[f"L5_concurrency_{c}"]["summary"]["throughput_rps"] for c in (8, 32))
        g4 = b1["L6_with_bg_generation"].get("slowdown_p95", 9) <= a1["L6_with_bg_generation"].get("slowdown_p95", 0)
        hit = min(b1[f"L4_shared_{K}q"]["prefix_hit_rate"] for K in (5, 10, 16))
        gates = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "L4_speedup_10": r10, "L4_speedup_16": r16, "B1_prefix_hit_min": hit,
                 "G1_ratio": b1["L1_single_100tok_2opt"]["summary"]["p50"] / a1["L1_single_100tok_2opt"]["summary"]["p50"]}
        if g1 and g2 and g3 and g4:
            verdict = "G1–G4 全過 → 進 Phase 2"
        elif g2:
            verdict = "G2 過但 G1/G3/G4 有未過 → 分工使用：SGLang 只給批次型工作"
        else:
            verdict = "G2 沒過 → 停，維持 llama-server"
        gates["verdict"] = verdict
        L += ["", "## 門檻判定（B1 vs A1）", "", "| 門檻 | 數字 | 過？ |", "|---|---|---|",
              f"| G1 單題不退步（B1 ≤ A1×1.10） | B1/A1 = {gates['G1_ratio']:.2f} | {'✓' if g1 else '✗'} |",
              f"| G2 多題有感（K=10,16 ≥1.5×） | A1/B1 = {r10:.2f}×（K=10）, {r16:.2f}×（K=16） | {'✓' if g2 else '✗'} |",
              f"| G3 併發不退步（c=8,32） | B1 {b1['L5_concurrency_8']['summary']['throughput_rps']:.1f}/{b1['L5_concurrency_32']['summary']['throughput_rps']:.1f} vs A1 {a1['L5_concurrency_8']['summary']['throughput_rps']:.1f}/{a1['L5_concurrency_32']['summary']['throughput_rps']:.1f} dec/s | {'✓' if g3 else '✗'} |",
              f"| G4 共存不更差（p95 退化） | B1 {b1['L6_with_bg_generation'].get('slowdown_p95', float('nan')):.2f}× vs A1 {a1['L6_with_bg_generation'].get('slowdown_p95', float('nan')):.2f}× | {'✓' if g4 else '✗'} |",
              f"| 前提：B1 前綴命中率 ≥90% | 最低 {hit * 100:.0f}% | {'✓' if hit >= 0.9 else '✗'} |",
              "", f"**判定：{verdict}**"]
    out["gates"] = gates
    open(os.path.join(ROOT, "results/07-sglang-speed.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    # ---- accuracy guardrail
    A = ["# 07 — SGLang 後端評估：準確率護欄", "", "B1 vs A1：每 task acc 差 ±2 點內且 McNemar p ≥ 0.05；A0 vs A1 = 量化效應，A1 vs B1 = 後端效應。", ""]
    acc = {}
    for ds in ("D0", "D1-cue"):
        A += [f"## {ds}", "", "| task | n | A0 | A1 | B1 | A1−A0（量化） | B1−A1（後端） | McNemar p（A1 vs B1） | AUROC A1 / B1 | 護欄 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in TASKS:
            rows = {a: acc_rows(a, ds, t) for a in ("A0", "A1", "B1")}
            if not rows["A1"] or not rows["B1"]:
                continue
            ids = sorted(set(rows["A1"]) & set(rows["B1"]) & (set(rows["A0"]) if rows["A0"] else set(rows["A1"])))
            if ds == "D0":
                data = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{t}.jsonl"), encoding="utf-8")]
                cal = group_split(data, 0); test = {r["id"] for r, c in zip(data, cal) if not c}
                ids = [i for i in ids if i in test]
            c = {a: [rows[a][i]["correct"] for i in ids] for a in rows if rows[a]}
            m = {a: float(np.mean(v)) for a, v in c.items()}
            n01, n10, p = mcnemar(c["A1"], c["B1"])
            au = {a: auroc(c[a], conf_of(rows[a], ids)) for a in ("A1", "B1")}
            ok = abs(m["B1"] - m["A1"]) <= 0.02 and p >= 0.05 and (math.isnan(au["B1"]) or math.isnan(au["A1"]) or au["B1"] >= au["A1"] - 0.05)
            acc[f"{ds}/{t}"] = {"n": len(ids), **m, "p": p, "auroc": au, "ok": ok}
            A.append(f"| {t} | {len(ids)} | {f(m.get('A0', float('nan')), 3)} | {m['A1']:.3f} | {m['B1']:.3f} | {f(m['A1'] - m.get('A0', float('nan')), 3)} | {m['B1'] - m['A1']:+.3f} | {p:.3f} | {f(au['A1'], 2)} / {f(au['B1'], 2)} | {'✓' if ok else '✗'} |")
        A.append("")
    # ---- bf16 disambiguation on H100: same weights (bf16), two backends
    rows2 = {a: {t: acc_rows(a, "D0", t) for t in TASKS} for a in ("A2", "B2", "B1w1", "B1nr")}
    if any(rows2["A2"].values()) and any(rows2["B2"].values()):
        A += ["## 分離權重與後端：bf16 權重、H100、兩個後端（D0 test）", "",
              "| task | n | A2 llama bf16 | B2 SGLang bf16 | B2−A2 | McNemar p | B1 SGLang FP8（L40S） | B1 workers=1 | B1 無 radix cache |", "|---|---|---|---|---|---|---|---|---|"]
        m_a2, m_b2 = [], []
        for t in TASKS:
            ra, rb = rows2["A2"][t], rows2["B2"][t]
            if not ra or not rb:
                continue
            data = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{t}.jsonl"), encoding="utf-8")]
            cal = group_split(data, 0); test = {r["id"] for r, c in zip(data, cal) if not c}
            ids = sorted(set(ra) & set(rb) & test)
            ca, cb = [ra[i]["correct"] for i in ids], [rb[i]["correct"] for i in ids]
            _, _, p = mcnemar(ca, cb)
            m_a2.append(float(np.mean(ca))); m_b2.append(float(np.mean(cb)))
            b1 = acc_rows("B1", "D0", t); b1w = rows2["B1w1"][t]; b1n = rows2["B1nr"][t]
            fx = lambda rr: f"{np.mean([rr[i]['correct'] for i in ids if i in rr]):.3f}" if rr and all(i in rr for i in ids[:5]) else "—"
            A.append(f"| {t} | {len(ids)} | {np.mean(ca):.3f} | {np.mean(cb):.3f} | {np.mean(cb) - np.mean(ca):+.3f} | {p:.3f} | {fx(b1)} | {fx(b1w)} | {fx(b1n)} |")
        A.append(f"| **平均** | | {np.mean(m_a2):.3f} | {np.mean(m_b2):.3f} | {np.mean(m_b2) - np.mean(m_a2):+.3f} | | | | |")
        A.append("")
        out["bf16"] = {"A2_mean": float(np.mean(m_a2)), "B2_mean": float(np.mean(m_b2))}
    if acc:
        n_fail = sum(1 for v in acc.values() if not v["ok"])
        A.append(f"**護欄：{len(acc) - n_fail}/{len(acc)} 通過**" + ("" if n_fail == 0 else f"，{n_fail} 項未過（見 ✗）"))
        out["accuracy"] = acc
    open(os.path.join(ROOT, "results/07-sglang-accuracy.md"), "w", encoding="utf-8").write("\n".join(A) + "\n")
    json.dump(out, open(os.path.join(ROOT, "results/v4.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(gates, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
