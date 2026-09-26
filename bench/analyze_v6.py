"""v6 analysis (docs/handoff-v6-sglang-stability.md): E1 tokenizer parity, E2 deterministic inference.

Inputs : results/modal/accuracy/q8/D0            (A1 llama-server Q8, v4)
         results/modal/accuracy/sglang/D0        (B1 SGLang FP8 text prompts, v4)
         results/modal/accuracy/sglang/D0-ids    (E1: SGLang fed llama-server ids) + _tokdiff_{task}.json
         results/modal/accuracy/sglang/D0-det-a, D0-det-b  (E2: --enable-deterministic-inference, two runs)
         results/modal/v4bench/sglang/v4_det.json (E2 speed) vs q8/v4.json + q8/v4_L4seq.json (A1)
Outputs: results/09-sglang-stability.md, results/v6.json
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import TASKS, group_split  # noqa: E402
from analyze_ladder import mcnemar  # noqa: E402
from analyze_v4 import load_arm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(ROOT, "results/modal/accuracy")
V4 = os.path.join(ROOT, "results/modal/v4bench")


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def rows_of(path):
    if not os.path.exists(path):
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8") if '"correct"' in l}


def test_ids(task):
    data = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8")]
    cal = group_split(data, 0)
    return {r["id"] for r, c in zip(data, cal) if not c}


def main():
    out = {}
    L = ["# 09 — SGLang 掉分與不穩：E1 tokenizer 對齊、E2 批次不變推理", "",
         "門檻預先登記於 `docs/handoff-v6-sglang-stability.md`。L40S、`RedHatAI` FP8、D0 test（pair_id 分組 seed 0），A1 = llama-server Q8（v4）。", ""]

    # ---------------- E1
    a1 = {t: rows_of(os.path.join(ACC, "q8/D0", f"{t}.jsonl")) for t in TASKS}
    b1 = {t: rows_of(os.path.join(ACC, "sglang/D0", f"{t}.jsonl")) for t in TASKS}
    ids = {t: rows_of(os.path.join(ACC, "sglang/D0-ids", f"{t}.jsonl")) for t in TASKS}
    if any(ids.values()):
        L += ["## E1 tokenizer 對齊：SGLang 改餵 llama-server 切好的 token id（B1-ids）", ""]
        diffs = []
        for p in sorted(glob.glob(os.path.join(ACC, "sglang/D0-ids/_tokdiff_*.json"))):
            diffs += json.load(open(p, encoding="utf-8"))
        if diffs:
            n_same = sum(1 for d in diffs if d["first_diff_pos"] is None and d["n_llama"] == d["n_hf"])
            pos = [d["first_diff_pos"] for d in diffs if d["first_diff_pos"] is not None]
            ex = [d for d in diffs if d["first_diff_pos"] is not None][:3]
            L += [f"兩個 tokenizer 的差異（每 task 前 20 題，共 {len(diffs)} 題）：完全相同 {n_same} 題；有差的題第一個不同位置中位數 {f(float(np.median(pos)) if pos else float('nan'), 0)}，"
                  f"llama 平均 {np.mean([d['n_llama'] for d in diffs]):.1f} token、HF 平均 {np.mean([d['n_hf'] for d in diffs]):.1f} token。",
                  "例：" + "；".join(f"`{d['llama_piece']!r}` vs `{d['hf_piece']!r}`（pos {d['first_diff_pos']}）" for d in ex), ""]
            out["E1_tokdiff"] = {"n": len(diffs), "n_same": n_same, "median_first_diff": float(np.median(pos)) if pos else None}
        L += ["| task | n | A1 | B1（text） | B1-ids | B1-ids−A1 | McNemar p（A1 vs B1-ids） | 護欄 | B1-ids−B1 |", "|---|---|---|---|---|---|---|---|---|"]
        n_ok = n_tot = 0
        e1 = {}
        for t in TASKS:
            if not ids[t] or not a1[t]:
                continue
            common = sorted(test_ids(t) & set(a1[t]) & set(ids[t]) & set(b1[t]))
            ca = [a1[t][i]["correct"] for i in common]; ci = [ids[t][i]["correct"] for i in common]; cb = [b1[t][i]["correct"] for i in common]
            ma, mi, mb = float(np.mean(ca)), float(np.mean(ci)), float(np.mean(cb))
            _, _, p = mcnemar(ca, ci)
            ok = abs(mi - ma) <= 0.02 and p >= 0.05
            n_ok += ok; n_tot += 1
            e1[t] = {"n": len(common), "A1": ma, "B1": mb, "B1ids": mi, "p": p, "ok": ok}
            L.append(f"| {t} | {len(common)} | {ma:.3f} | {mb:.3f} | {mi:.3f} | {mi - ma:+.3f} | {p:.3f} | {'✓' if ok else '✗'} | {mi - mb:+.3f} |")
        mA = np.mean([v["A1"] for v in e1.values()]); mB = np.mean([v["B1"] for v in e1.values()]); mI = np.mean([v["B1ids"] for v in e1.values()])
        L.append(f"| **平均** | | {mA:.3f} | {mB:.3f} | {mI:.3f} | {mI - mA:+.3f} | | {n_ok}/{n_tot} | {mI - mB:+.3f} |")
        if n_ok >= 0.8 * n_tot:
            v1 = "**掉分是 tokenizer 造成**：餵同樣的 id 後護欄過，v4「後端固有」的結論撤回"
        elif abs(mI - mB) < 0.01:
            v1 = "**tokenizer 不是原因**：餵同樣 id 分數不動，進 E3（kernel／精度）"
        else:
            v1 = "**tokenizer 是部分原因**：餵同樣 id 後差距縮小但護欄未過"
        L += ["", f"判定（護欄 ≥ 8/10 → tokenizer；B1-ids 與 B1 差 < 1 點 → 不是）：{v1}", ""]
        out["E1"] = {"tasks": e1, "pass": n_ok, "total": n_tot, "verdict": v1, "mean": {"A1": float(mA), "B1": float(mB), "B1ids": float(mI)}}

    # ---------------- E2
    da = {t: rows_of(os.path.join(ACC, "sglang/D0-det-a", f"{t}.jsonl")) for t in TASKS}
    db = {t: rows_of(os.path.join(ACC, "sglang/D0-det-b", f"{t}.jsonl")) for t in TASKS}
    if any(da.values()) and any(db.values()):
        L += ["## E2 批次不變推理：`--enable-deterministic-inference`，D0 跑兩次", "",
              "| task | n | acc a | acc b | A1 | 兩次答案一致率 | 翻掉題數 | 翻掉題中把握 >0.9 |", "|---|---|---|---|---|---|---|---|"]
        agree_all, flips_all, flips_conf_all, acc_a, acc_b = [], 0, 0, [], []
        e2 = {}
        for t in TASKS:
            if not da[t] or not db[t]:
                continue
            common = sorted(set(da[t]) & set(db[t]))
            same = [da[t][i]["chosen"] == db[t][i]["chosen"] for i in common]
            flips = [i for i in common if da[t][i]["chosen"] != db[t][i]["chosen"]]
            fc = sum(1 for i in flips if max(da[t][i]["probs"].values()) > 0.9)
            te = [i for i in common if i in test_ids(t)]
            aa = float(np.mean([da[t][i]["correct"] for i in te])); ab = float(np.mean([db[t][i]["correct"] for i in te]))
            a1m = float(np.mean([a1[t][i]["correct"] for i in te if i in a1[t]])) if a1[t] else float("nan")
            agree_all += same; flips_all += len(flips); flips_conf_all += fc; acc_a.append(aa); acc_b.append(ab)
            e2[t] = {"n": len(common), "acc_a": aa, "acc_b": ab, "A1": a1m, "agree": float(np.mean(same)), "flips": len(flips), "flips_conf": fc}
            L.append(f"| {t} | {len(common)} | {aa:.3f} | {ab:.3f} | {f(a1m)} | {np.mean(same) * 100:.1f}% | {len(flips)} | {fc} |")
        ag = float(np.mean(agree_all))
        L.append(f"| **合計** | {len(agree_all)} | {np.mean(acc_a):.3f} | {np.mean(acc_b):.3f} | | **{ag * 100:.2f}%** | {flips_all} | {flips_conf_all} |")
        L.append("（v4 無此旗標時三次同設定的兩兩一致率 97.7% / 96.9% / 97.4%，翻掉 50 題中 32 題把握 >0.9）")
        stable = ag >= 0.995 and flips_conf_all == 0
        # speed cost
        det = None
        pdet = os.path.join(V4, "sglang/v4_det.json")
        if os.path.exists(pdet):
            det = json.load(open(pdet))["results"]
            A1s = load_arm("A1"); B1s = load_arm("B1")
            L += ["", "### 速度代價（p50 ms；B1 = v4 無旗標，det = 開旗標）", "", "| 項目 | A1 llama Q8 | B1 SGLang | B1-det | 門檻 |", "|---|---|---|---|---|"]
            g = lambda r, k, fld="p50": ((r.get(k) or {}).get("summary") or {}).get(fld, float("nan"))
            l1 = g(det, "L1_single_100tok_2opt"); g1 = l1 <= g(A1s, "L1_single_100tok_2opt") * 1.10
            L.append(f"| L1 單題 | {f(g(A1s, 'L1_single_100tok_2opt'), 0)} | {f(g(B1s, 'L1_single_100tok_2opt'), 0)} | {f(l1, 0)} | G1 ≤ A1×1.1：{'✓' if g1 else '✗'} |")
            g2 = True
            for K in (10, 16):
                r = g(A1s, f"L4_shared_{K}q") / g(det, f"L4_shared_{K}q")
                g2 &= r >= 1.5
                L.append(f"| L4 共用 state K={K} | {f(g(A1s, f'L4_shared_{K}q'), 0)} | {f(g(B1s, f'L4_shared_{K}q'), 0)} | {f(g(det, f'L4_shared_{K}q'), 0)}（{r:.2f}×） | G2 ≥ 1.5×：{'✓' if r >= 1.5 else '✗'} |")
            g3 = True
            for c in (8, 32):
                rd = g(det, f"L5_concurrency_{c}", "throughput_rps"); ra = g(A1s, f"L5_concurrency_{c}", "throughput_rps")
                g3 &= rd >= ra
                L.append(f"| L5 c={c} decisions/s | {f(ra, 1)} | {f(g(B1s, f'L5_concurrency_{c}', 'throughput_rps'), 1)} | {f(rd, 1)} | G3 ≥ A1：{'✓' if rd >= ra else '✗'} |")
            out["E2_speed"] = {"G1": g1, "G2": g2, "G3": g3}
        if stable:
            v2 = "**不穩解掉**（一致率 ≥ 99.5%、翻掉的題沒有把握 >0.9 的）"
            if det is not None and not (out["E2_speed"]["G2"] and out["E2_speed"]["G3"]):
                v2 += "，但速度門檻 G2/G3 有未過 → 解掉但速度優勢不夠，SGLang 只給批次"
        else:
            v2 = f"**不穩另有原因**（一致率 {ag * 100:.2f}%，翻掉題中把握 >0.9 的有 {flips_conf_all} 題），記錄後停"
        L += ["", f"判定：{v2}", ""]
        out["E2"] = {"tasks": e2, "agree": ag, "flips": flips_all, "flips_conf": flips_conf_all, "verdict": v2}

    open(os.path.join(ROOT, "results/09-sglang-stability.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(ROOT, "results/v6.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v.get("verdict") for k, v in out.items() if isinstance(v, dict) and "verdict" in v}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
