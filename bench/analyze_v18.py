"""v18 analysis (docs/handoff-v18-mtp.md; gates written before the run).

  python3 bench/analyze_v18.py [--mtp mtp-q8-n4]  ->  results/22-mtp.md, results/v18.json
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from verify import check_lock  # noqa: E402

V18 = os.path.join(ROOT, "results/modal/v18")
TASKS = ["m_alarm_category", "m_alarm_severity", "m_needs_dispatch", "q_spc_action", "q_defect_root", "p_uph_anomaly", "p_line_change",
         "x_ticket_route", "x_escalate", "x_10way_intent"]
WEAK = ["m_alarm_severity", "q_spc_action", "p_uph_anomaly"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def equivalence(base, mtp):
    """Per-row comparison of the letter readout: argmax flips and the largest change in the top probability."""
    n = flips = 0; dmax = 0.0; d_all = []; per = {}
    for t in TASKS:
        B = {r["id"]: r for r in load(os.path.join(base, f"{t}.jsonl"))}
        M = {r["id"]: r for r in load(os.path.join(mtp, f"{t}.jsonl"))}
        ids = sorted(set(B) & set(M)); f = 0
        for i in ids:
            pb, pm = B[i]["probs"], M[i]["probs"]
            if max(pb, key=pb.get) != max(pm, key=pm.get):
                f += 1
            d = max(abs(pb.get(k, 0.0) - pm.get(k, 0.0)) for k in set(pb) | set(pm)) if pb and pm else 1.0
            d_all.append(d); dmax = max(dmax, d)
        n += len(ids); flips += f
        per[t] = {"n": len(ids), "flips": f, "acc_base": float(np.mean([B[i]["correct"] for i in ids])) if ids else None,
                  "acc_mtp": float(np.mean([M[i]["correct"] for i in ids])) if ids else None,
                  "lat_base_p50": float(np.median([B[i]["latency_ms"] for i in ids])) if ids else None,
                  "lat_mtp_p50": float(np.median([M[i]["latency_ms"] for i in ids])) if ids else None}
    return {"n": n, "flips": flips, "dp_max": dmax, "dp_mean": float(np.mean(d_all)) if d_all else None,
            "dp_p99": float(np.percentile(d_all, 99)) if d_all else None, "per_task": per}


def think(base, mtp):
    out = {}
    gb_all, gm_all, flips, nB = [], [], 0, 0
    for t in WEAK:
        B = {r["id"]: r for r in load(os.path.join(base, f"{t}.jsonl"))}
        M = {r["id"]: r for r in load(os.path.join(mtp, f"{t}.jsonl"))}
        ids = [i for i in sorted(set(B) & set(M)) if B[i].get("B") and M[i].get("B")]
        gb = [B[i]["B"]["gen_ms"] for i in ids]; gm = [M[i]["B"]["gen_ms"] for i in ids]
        tb = [B[i]["B"]["n_tokens"] for i in ids]; tm = [M[i]["B"]["n_tokens"] for i in ids]
        f = sum(max(B[i]["B"]["probs"], key=B[i]["B"]["probs"].get) != max(M[i]["B"]["probs"], key=M[i]["B"]["probs"].get) for i in ids)
        same_thought = sum(B[i]["B"].get("thought") == M[i]["B"].get("thought") for i in ids)
        out[t] = {"n_B": len(ids), "gen_ms_p50_base": float(np.median(gb)) if gb else None, "gen_ms_p50_mtp": float(np.median(gm)) if gm else None,
                  "tok_per_s_base": float(sum(tb) / sum(gb) * 1000) if gb else None, "tok_per_s_mtp": float(sum(tm) / sum(gm) * 1000) if gm else None,
                  "letter_flips": f, "same_thought_text": same_thought}
        gb_all += gb; gm_all += gm; flips += f; nB += len(ids)
    out["all"] = {"n_B": nB, "gen_ms_p50_base": float(np.median(gb_all)) if gb_all else None, "gen_ms_p50_mtp": float(np.median(gm_all)) if gm_all else None,
                  "ratio_p50": (float(np.median(gm_all) / np.median(gb_all)) if gb_all and gm_all else None), "letter_flips": flips}
    return out


def noise_control(arm):
    """Post-hoc control (not a gate): the think arms re-read the same first-50 D0 rows per weak task on a separate container
    (stage A, no generation). base-vs-base across containers = run-to-run noise; compare with base-vs-mtp."""
    out = {}
    pairs = {"base_vs_base": ("accuracy/base", "think/base"), "mtp_vs_mtp": (f"accuracy/{arm}", f"think/{arm}"),
             "base_vs_mtp_same_rows": ("accuracy/base", f"accuracy/{arm}")}
    for name, (x, y) in pairs.items():
        d = []; flips = 0
        for t in WEAK:
            X = {r["id"]: r for r in load(os.path.join(V18, x, f"{t}.jsonl"))}
            Y = {r["id"]: r for r in load(os.path.join(V18, y, f"{t}.jsonl"))}
            ids = sorted(set(X) & set(Y))
            if name == "base_vs_mtp_same_rows":  # restrict to the rows the think arms cover, so the three numbers are comparable
                Z = {r["id"] for r in load(os.path.join(V18, "think/base", f"{t}.jsonl"))}; ids = [i for i in ids if i in Z]
            for i in ids:
                px = X[i]["probs"] if "probs" in X[i] else X[i]["A"]["probs"]; py = Y[i]["probs"] if "probs" in Y[i] else Y[i]["A"]["probs"]
                d.append(max(abs(px.get(k, 0.0) - py.get(k, 0.0)) for k in set(px) | set(py)))
                flips += max(px, key=px.get) != max(py, key=py.get)
        out[name] = {"n": len(d), "dp_max": float(max(d)) if d else None, "dp_mean": float(np.mean(d)) if d else None,
                     "n_over_0.01": int(sum(v > 0.01 for v in d)), "n_over_0.05": int(sum(v > 0.05 for v in d)), "flips": flips}
    return out


def draft_stats(path):
    """Whatever the server's /metrics says about drafting (llama-server exposes a few counters; keep the raw lines)."""
    if not os.path.exists(path):
        return None
    lines = [l.strip() for l in open(path, encoding="utf-8") if re.search(r"draft|spec|accept", l, re.I) and not l.startswith("#")]
    return lines[:20]


def main():
    a = dict(zip(*[iter(sys.argv[1:])] * 2)); arm = a.get("--mtp", "mtp-q8-n4")
    base_acc, mtp_acc = os.path.join(V18, "accuracy/base"), os.path.join(V18, f"accuracy/{arm}")
    eq = equivalence(base_acc, mtp_acc)
    print("--- check-lock on the mtp arm")
    lock_fail = check_lock(mtp_acc, os.path.join(ROOT, "results/thresholds.lock.json"))
    print("--- check-lock on the base arm (same build, same day)")
    lock_fail_base = check_lock(base_acc, os.path.join(ROOT, "results/thresholds.lock.json"))
    vb = {k: json.load(open(os.path.join(V18, f"v4bench/{k}/v18.json")))["results"] for k in ("base", arm) if os.path.exists(os.path.join(V18, f"v4bench/{k}/v18.json"))}
    lat = {}
    if len(vb) == 2:
        b, m = vb["base"], vb[arm]
        g = lambda r, k: (r.get(k) or {})  # noqa: E731
        lat = {"L1_p50": [g(b, "L1_single_100tok_2opt")["summary"]["p50"], g(m, "L1_single_100tok_2opt")["summary"]["p50"]],
               "L4_16q_p50": [g(b, "L4_shared_16q")["summary"]["p50"], g(m, "L4_shared_16q")["summary"]["p50"]],
               "L6_p50": [g(b, "L6_with_bg_generation")["summary"]["p50"], g(m, "L6_with_bg_generation")["summary"]["p50"]],
               "L6_slowdown_p50": [g(b, "L6_with_bg_generation")["slowdown_p50"], g(m, "L6_with_bg_generation")["slowdown_p50"]],
               "L6_bg_tokens": [g(b, "L6_with_bg_generation")["bg_generation"]["tokens"], g(m, "L6_with_bg_generation")["bg_generation"]["tokens"]],
               "L6_wall_s": [g(b, "L6_with_bg_generation")["summary"]["n"] * g(b, "L6_with_bg_generation")["summary"]["mean"] / 1000,
                             g(m, "L6_with_bg_generation")["summary"]["n"] * g(m, "L6_with_bg_generation")["summary"]["mean"] / 1000]}
        lat["L6_bg_tok_per_s"] = [lat["L6_bg_tokens"][0] / lat["L6_wall_s"][0], lat["L6_bg_tokens"][1] / lat["L6_wall_s"][1]]
        lat["L1_ratio"] = lat["L1_p50"][1] / lat["L1_p50"][0]; lat["L6_bg_ratio"] = lat["L6_bg_tok_per_s"][1] / lat["L6_bg_tok_per_s"][0]
    th = think(os.path.join(V18, "think/base"), os.path.join(V18, f"think/{arm}"))
    drafts = {k: draft_stats(os.path.join(V18, f"{sub}/{k}/_metrics.txt")) for sub in ("accuracy", "think") for k in (arm,)}
    runs = {k: json.load(open(os.path.join(V18, f"accuracy/{k}/_run.json"))) for k in ("base", arm) if os.path.exists(os.path.join(V18, f"accuracy/{k}/_run.json"))}
    nc = noise_control(arm)

    g1 = eq["flips"] <= 2 and lock_fail == 0 and eq["dp_max"] <= 0.05
    g2 = bool(lat) and lat["L1_ratio"] <= 1.10
    g3 = bool(lat) and lat["L6_bg_ratio"] >= 1.3 and th["all"]["ratio_p50"] is not None and th["all"]["ratio_p50"] <= 0.75
    g4 = bool(lat) and lat["L6_slowdown_p50"][1] <= lat["L6_slowdown_p50"][0] * 1.05
    if not (g1 and g2):
        rec = "判斷 server 不開 MTP（v4 規定不變）；生成 server " + ("開" if g3 else "也不開")
    elif g4:
        rec = "判斷與生成可共用一個開 MTP 的 server" if g3 else "判斷 server 開不開都行（等價、不變慢），但生成沒有感，沒必要開"
    else:
        rec = "判斷 server 單獨跑時可開（等價、不變慢），但不要與生成共用；生成 server " + ("開" if g3 else "不開")
    pred = {"A_flips_0_2": eq["flips"] <= 2 and lock_fail == 0, "B_L1_within_5pct": bool(lat) and abs(lat["L1_ratio"] - 1) <= 0.05,
            "B_L6_gen_1.5x": bool(lat) and lat["L6_bg_ratio"] >= 1.5, "B_L6_slowdown_not_worse": g4,
            "C_genms_0.6x": th["all"]["ratio_p50"] is not None and th["all"]["ratio_p50"] <= 0.6, "C_flips_le_2pct": th["all"]["n_B"] > 0 and th["all"]["letter_flips"] <= 0.02 * th["all"]["n_B"]}
    res = {"arm": arm, "equivalence": eq, "check_lock_failures": {"mtp": lock_fail, "base": lock_fail_base}, "latency": lat, "think": th, "draft_metrics": drafts,
           "runs": runs, "noise_control": nc, "gates": {"G1_equivalent": g1, "G2_single_not_slower": g2, "G3_generation_faster": g3, "G4_under_load_not_worse": g4, "recommendation": rec},
           "predictions": pred}
    json.dump(res, open(os.path.join(ROOT, "results/v18.json"), "w"), ensure_ascii=False, indent=1)

    f2 = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    L = ["# 22 — Gemma 4 MTP 草稿頭對 jevlike 有沒有影響（v18）", "",
         "計劃與門檻：`docs/handoff-v18-mtp.md`（跑前寫死）。L4、llama.cpp b11371、26B UD-Q4_K_M（lock 檔同一檔）、草稿頭 unsloth `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf`、"
         "`--spec-type draft-mtp --spec-draft-n-max 4`。兩臂同容器型號、同旗標，只差草稿頭。", "",
         "## 判定", "",
         f"- **G1 判斷等價**（翻轉 ≤ 2／1,000、check-lock 0 失敗、最大機率差 ≤ 0.05）：翻轉 {eq['flips']}／{eq['n']}、check-lock {lock_fail} 失敗（base 臂 {lock_fail_base}）、最大差 {eq['dp_max']:.4f} → {'✓' if g1 else '✗'}",
         f"- **G2 單題不變慢**（L1 p50 比 ≤ 1.10）：{f2(lat.get('L1_ratio'))} → {'✓' if g2 else '✗'}",
         f"- **G3 生成有感**（背景生成 token/s 比 ≥ 1.3 且思考生成時間比 ≤ 0.75）：{f2(lat.get('L6_bg_ratio'))}、{f2(th['all']['ratio_p50'])} → {'✓' if g3 else '✗'}",
         f"- **G4 負載下判斷不變差**（L6 slowdown 比 ≤ 1.05）：{f2(lat['L6_slowdown_p50'][0]) if lat else '—'} → {f2(lat['L6_slowdown_p50'][1]) if lat else '—'} → {'✓' if g4 else '✗'}",
         f"- **決定**：{rec}",
         "- 預測對答案：" + "、".join(f"{k} {'✓' if v else '✗'}" for k, v in pred.items()), "",
         "## A 判斷等價（D0 全部 2,000 題，逐題比）", "",
         "| 題型 | n | 翻轉 | base 準確率 | mtp 準確率 | base 單題 p50 | mtp 單題 p50 |", "|---|---|---|---|---|---|---|"]
    for t, p in eq["per_task"].items():
        L.append(f"| {t} | {p['n']} | {p['flips']} | {f2(p['acc_base'])} | {f2(p['acc_mtp'])} | {p['lat_base_p50']:.0f} ms | {p['lat_mtp_p50']:.0f} ms |" if p["n"] else f"| {t} | 0 | | | | | |")
    L += ["", f"機率差：平均 {eq['dp_mean']:.5f}、p99 {eq['dp_p99']:.5f}、最大 {eq['dp_max']:.5f}。", ""]
    if lat:
        L += ["## B 延遲（v4bench，n 100）", "", "| 項 | base | mtp | 比 |", "|---|---|---|---|",
              f"| L1 單題 p50 | {lat['L1_p50'][0]:.0f} ms | {lat['L1_p50'][1]:.0f} ms | {lat['L1_ratio']:.2f} |",
              f"| L4 共用 state 16 題 p50 | {lat['L4_16q_p50'][0]:.0f} ms | {lat['L4_16q_p50'][1]:.0f} ms | {lat['L4_16q_p50'][1] / lat['L4_16q_p50'][0]:.2f} |",
              f"| L6 背景生成時的判斷 p50 | {lat['L6_p50'][0]:.0f} ms | {lat['L6_p50'][1]:.0f} ms | {lat['L6_p50'][1] / lat['L6_p50'][0]:.2f} |",
              f"| L6 判斷被拖慢的倍數（對 L1） | {lat['L6_slowdown_p50'][0]:.2f} | {lat['L6_slowdown_p50'][1]:.2f} | — |",
              f"| L6 背景生成 token/s | {lat['L6_bg_tok_per_s'][0]:.0f} | {lat['L6_bg_tok_per_s'][1]:.0f} | {lat['L6_bg_ratio']:.2f} |", ""]
    L += ["## C 思考模式（三類弱題各 50 題，信心 < 0.9 的進階段 B，貪婪生成 ≤ 512 token）", "",
          "| 題型 | 進階段 B | base 生成 p50 | mtp 生成 p50 | base token/s | mtp token/s | 思考後字母翻轉 | 思考文字完全相同 |", "|---|---|---|---|---|---|---|---|"]
    for t in WEAK:
        p = th[t]
        L.append(f"| {t} | {p['n_B']} | {p['gen_ms_p50_base'] / 1000 if p['gen_ms_p50_base'] else 0:.1f} s | {p['gen_ms_p50_mtp'] / 1000 if p['gen_ms_p50_mtp'] else 0:.1f} s | "
                 f"{p['tok_per_s_base'] or 0:.0f} | {p['tok_per_s_mtp'] or 0:.0f} | {p['letter_flips']} | {p['same_thought_text']} |")
    L += ["", f"全部：{th['all']['n_B']} 題進階段 B，生成時間中位數比 {f2(th['all']['ratio_p50'])}，字母翻轉 {th['all']['letter_flips']}。", "",
          "## D 草稿接受率", ""]
    for k, v in drafts.items():
        L.append(f"- `{k}` /metrics：" + ("；".join(v) if v else "沒有草稿相關的計數器"))
    L += ["", "## 跑後對照：機率差有多少是 MTP、有多少是跑兩次就會有的（不是門檻）", "",
          "思考臂的階段 A 在另一個容器上重讀了同樣的題（三類弱題各前 50 題，不生成）。base 對 base 跨容器的差就是「跑兩次」的雜訊；拿它跟 base 對 mtp 比。", "",
          "| 比較 | n | 最大機率差 | 平均 | > 0.01 的題 | > 0.05 的題 | 翻轉 |", "|---|---|---|---|---|---|---|"]
    for k, v in nc.items():
        dmax = "—" if v["dp_max"] is None else "%.4f" % v["dp_max"]; dmean = "—" if v["dp_mean"] is None else "%.5f" % v["dp_mean"]
        L.append(f"| {k} | {v['n']} | {dmax} | {dmean} | {v['n_over_0.01']} | {v['n_over_0.05']} | {v['flips']} |")
    L += ["", "server 旗標與版本記在 `results/modal/v18/*/*/_run.json`；server 日誌 `results/modal/v18/_server_*.log`。"]
    open(os.path.join(ROOT, "results/22-mtp.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
