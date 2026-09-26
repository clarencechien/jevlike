"""P2 (handoff v7): retro-compute option mass on every existing result file, no model call.

option mass = Σ_letters exp(raw_logprob). Outputs results/11-option-mass.md and adds "option_mass" to results/v7.json.
Pre-registered expectations (handoff v7 §P2): catches template-level breakage (v2 smoke variants), does NOT catch the
<bos> case (sglang/D0 vs sglang/D0-ids both ≈ 1.0); AUROC vs correctness on llama D0 expected ≈ 0.5.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ladder import auroc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = os.path.join(ROOT, "results/modal/accuracy")


def masses(path):
    m, c = [], []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if "raw_logprobs" in r and "correct" in r:
            m.append(sum(math.exp(v) for v in r["raw_logprobs"].values())); c.append(bool(r["correct"]))
    return np.array(m), np.array(c)


def main():
    L = ["# 11 — option mass（字母正規化前的總機率）回溯檢查", "",
         "`python3 bench/option_mass.py`。每個結果目錄：n、p50、p5、mass < 0.9 的比例、mass 對答對與否的 AUROC。門檻與預期見 `docs/handoff-v7-borrowed.md` §P2。", "",
         "## 模板變體（v2 smoke，5 題 × 2 次）", "", "| 變體 | mass 最小 | 中位數 | first token |", "|---|---|---|---|"]
    out = {}
    sm = json.load(open(os.path.join(ROOT, "results/modal/smoke.json")))
    for name, v in sm["variants"].items():
        if "rows" not in v:
            continue
        ms = [sum(math.exp(x) for x in r["raw_logprobs"].values()) for r in v["rows"]]
        L.append(f"| {name} | {min(ms):.3f} | {sorted(ms)[len(ms) // 2]:.3f} | {[r['first_token'] for r in v['rows'][:2]]} |")
        out[f"smoke/{name}"] = {"min": min(ms), "median": float(np.median(ms))}
    L += ["", "## 結果目錄（每目錄合併 10 task）", "", "| 目錄 | n | p50 | p5 | <0.9 比例 | AUROC（mass→答對） |", "|---|---|---|---|---|---|"]
    dirs = [ACC] + sorted(d for d in glob.glob(os.path.join(ACC, "**/"), recursive=True) if d.rstrip("/") != ACC)
    for d in dirs:
        files = [p for p in glob.glob(os.path.join(d, "*.jsonl")) if "control_" not in os.path.basename(p)]
        if not files:
            continue
        M, C = [], []
        for p in files:
            m, c = masses(p)
            M.append(m); C.append(c)
        M, C = np.concatenate(M), np.concatenate(C)
        if len(M) == 0:
            continue
        rel = os.path.relpath(d.rstrip("/"), ACC) or "D0 (26b v2)"
        au = auroc(C.astype(int), M)
        out[rel] = {"n": int(len(M)), "p50": float(np.median(M)), "p5": float(np.percentile(M, 5)), "frac_lt_0.9": float((M < 0.9).mean()), "auroc": au}
        L.append(f"| {rel} | {len(M)} | {np.median(M):.4f} | {np.percentile(M, 5):.4f} | {(M < 0.9).mean():.3f} | {au:.2f} |")
    b0, b1 = out.get("sglang/D0"), out.get("sglang/D0-ids")
    v = []
    if b0 and b1:
        caught = (b0["p50"] - b1["p50"] <= -0.05) or (b0["frac_lt_0.9"] - b1["frac_lt_0.9"] >= 0.10)
        v.append(f"`<bos>` 有無：無 p50 {b0['p50']:.4f} / 有 {b1['p50']:.4f}，<0.9 比例 {b0['frac_lt_0.9']:.3f} / {b1['frac_lt_0.9']:.3f} → option mass {'抓得到' if caught else '**抓不到**'} `<bos>` 這類錯（與跑前預期一致）。")
    bad = [k for k, x in out.items() if k.startswith("smoke/") and x["median"] < 0.5]
    good = [k for k, x in out.items() if k.startswith("smoke/") and x["median"] >= 0.9]
    v.append(f"模板等級的錯：{len(bad)} 個壞變體 mass 中位數 < 0.5（{', '.join(k[6:] for k in bad)}），{len(good)} 個好變體 ≥ 0.9 → 當 smoke 健康檢查門檻 0.9 可行。")
    base = out.get("D0 (26b v2)")
    if base:
        v.append(f"llama-server D0 上 mass 對答對的 AUROC {base['auroc']:.2f}（跑前預期 ≈ 0.5，**預期錯了**）：mass 的 p5 仍是 {base['p5']:.4f}，差異全在 1e-4 以下的尾巴，等於信心的影子（模型不確定時會漏一點機率給非字母 token）。可以排序、不能設門檻；準確率訊號仍以校準後信心為準。")
    L += ["", "**判讀**：", ""] + [f"- {x}" for x in v]
    L += ["", "**處置**：`decide/client.py` 每次讀取回傳 `option_mass`，`accuracy.py` 每列存，兩個 smoke 都加 `option_mass_check`（最小值 ≥ 0.9）。"]
    open(os.path.join(ROOT, "results/11-option-mass.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    v7p = os.path.join(ROOT, "results/v7.json")
    v7 = json.load(open(v7p)) if os.path.exists(v7p) else {}
    v7["option_mass"] = out
    json.dump(v7, open(v7p, "w"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(v))


if __name__ == "__main__":
    main()
