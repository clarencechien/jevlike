"""v13 analysis (docs/handoff-v13-embeddinggemma2.md; predictions and gates written before the run).

  python3 bench/analyze_v13.py   ->  results/18-embeddinggemma2.md, results/v13.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
from analyze import TASKS, ece, group_split, load_jsonl, logit_matrix, softmax  # noqa: E402
from thresholds import risk_threshold  # noqa: E402

V13 = os.path.join(ROOT, "results/modal/v13")
ACC = os.path.join(ROOT, "results/modal/accuracy")
TOPIC = ["x_10way_intent", "x_ticket_route", "m_alarm_category"]
COND = ["x_escalate", "m_needs_dispatch", "m_alarm_severity", "q_spc_action", "p_uph_anomaly"]
MID = ["q_defect_root", "p_line_change"]
GRID = np.exp(np.linspace(math.log(1e-3), math.log(5), 300))  # cosine scores live in [-1, 1]: tau is small
ZH = {"m_alarm_severity": "alarm 急迫度", "m_alarm_category": "alarm 原因分類", "m_needs_dispatch": "要不要派工", "q_spc_action": "SPC 動作",
      "q_defect_root": "不良站別", "p_uph_anomaly": "UPH 異常", "p_line_change": "停線原因", "x_ticket_route": "工單分派", "x_escalate": "要不要升級",
      "x_10way_intent": "十種意圖"}


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def fit_tau(Z, y):
    best, bt = float("inf"), 1.0
    for t in GRID:
        P = softmax(Z / t); v = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
        if v < best:
            best, bt = v, float(t)
    return bt


def llm_acc(d):
    out = {}
    for t in TASKS:
        rows = [r for r in load_jsonl(os.path.join(d, f"{t}.jsonl")) if "raw_logprobs" in r]
        if not rows:
            return None
        L = list(TASKS[t]["options"]); Z = logit_matrix(rows, L); y = np.array([L.index(r["gold"]) for r in rows])
        te = ~group_split(rows, 0)
        out[t] = float((Z.argmax(1)[te] == y[te]).mean())
    return out


def twin_pairs():
    """Exploratory (not a gate): D0 twins share a pair_id and have opposite golds. How often does a reader give both the
    same answer? Uses all 200 rows per task (cal + test)."""
    out = {}
    for t in TASKS:
        d = np.load(os.path.join(V13, f"{t}.npz")); ids = list(d["ids"]); pe = d["Z1"].argmax(1); y = d["gold"]
        pid = {r["id"]: r.get("pair_id") for r in load_jsonl(os.path.join(ROOT, f"data/synthetic/{t}.jsonl"))}
        L = list(TASKS[t]["options"])
        p26 = {r["id"]: int(logit_matrix([r], L).argmax()) for r in load_jsonl(os.path.join(ACC, f"{t}.jsonl")) if "raw_logprobs" in r}
        groups = {}
        for i, k in enumerate(ids):
            groups.setdefault(pid[k], []).append(i)
        pairs = [g for g in groups.values() if len(g) == 2 and y[g[0]] != y[g[1]]]
        out[t] = {"pairs": len(pairs), "eg2_same": int(sum(pe[a] == pe[b] for a, b in pairs)),
                  "26b_same": int(sum(p26[ids[a]] == p26[ids[b]] for a, b in pairs)),
                  "eg2_both_right": int(sum(pe[a] == y[a] and pe[b] == y[b] for a, b in pairs))}
    return out


def confidence_and_cascade():
    """Added 2026-10-07 after the run (not a gate): can Z1's calibrated confidence tell right from wrong, how much can it
    auto-handle under a 5% error budget (v9 selective risk control, cut chosen on cal), and an EG2 -> 26B cascade where
    EG2 answers above that cut and everything else goes to 26B."""
    from sklearn.metrics import roc_auc_score
    per, ok_c, sent, n = {}, 0, 0, 0
    for t in TASKS:
        d = np.load(os.path.join(V13, f"{t}.npz")); y = d["gold"]; ca = d["cal"].astype(bool); te = ~ca; Z = d["Z1"]
        tau = fit_tau(Z[ca], y[ca]); P = softmax(Z / tau); conf = P.max(1); ok = P.argmax(1) == y
        au = float(roc_auc_score(ok[te], conf[te])) if len(set(ok[te].tolist())) > 1 else None
        cut, _ = risk_threshold(conf[ca], ok[ca], 0.05); h = conf[te] >= cut
        L = list(TASKS[t]["options"]); ids = list(d["ids"])
        p26 = {r["id"]: int(logit_matrix([r], L).argmax()) for r in load_jsonl(os.path.join(ACC, f"{t}.jsonl")) if "raw_logprobs" in r}
        for i in np.where(te)[0]:
            n += 1
            if conf[i] >= cut:
                ok_c += int(ok[i])
            else:
                sent += 1; ok_c += int(p26[ids[i]] == y[i])
        per[t] = {"K": int(Z.shape[1]), "auroc": au, "coverage_eps5": float(h.mean()), "error_eps5": float((~ok[te][h]).mean()) if h.any() else 0.0}
    return {"per_task": per, "cascade_eg2_to_26b_eps5": {"acc": ok_c / n, "sent_to_26b": sent / n, "n": n}}


def main():
    from sklearn.linear_model import LogisticRegression
    run = json.load(open(os.path.join(V13, "_run.json")))
    res = {"arms": {}, "per_task": {}}
    for t in TASKS:
        d = np.load(os.path.join(V13, f"{t}.npz"))
        y, ca = d["gold"], d["cal"].astype(bool); te = ~ca; K = d["Z1"].shape[1]
        S = d["S"].astype(np.float32)
        per = {"K": int(K), "chance": 1 / K}
        for n in ("Z1", "Z2", "Z3"):
            Z = d[n]; per[n] = float((Z.argmax(1)[te] == y[te]).mean())
        tau = fit_tau(d["Z1"][ca], y[ca]); P = softmax(d["Z1"] / tau)
        per["Z1_tau"] = tau; per["Z1_ece_test"] = ece(P.max(1)[te], (P.argmax(1) == y)[te])
        per["Z1_pred_dist_test"] = np.bincount(d["Z1"].argmax(1)[te], minlength=K).tolist()
        # reference arms that use the 100 cal labels
        C = np.stack([S[ca & (y == k)].mean(0) if (ca & (y == k)).any() else np.zeros(S.shape[1]) for k in range(K)])
        C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-12, None)
        per["P1"] = float(((S[te] @ C.T).argmax(1) == y[te]).mean())
        if len(set(y[ca].tolist())) > 1:
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(S[ca], y[ca]); per["P2"] = float((lr.predict(S[te]) == y[te]).mean())
        else:
            per["P2"] = None
        res["per_task"][t] = per
    llm = {"26B": llm_acc(ACC), "12B": llm_acc(os.path.join(ROOT, "results/modal/v12/d0/12b")), "E4B": llm_acc(os.path.join(ACC, "e4b/D0")), "E2B": llm_acc(os.path.join(ACC, "e2b/D0"))}
    llm = {k: v for k, v in llm.items() if v}
    mean = lambda arm, ts: float(np.mean([res["per_task"][t][arm] for t in ts]))  # noqa: E731
    for arm in ("Z1", "Z2", "Z3", "P1", "P2"):
        res["arms"][arm] = {"all": mean(arm, TASKS), "topic": mean(arm, TOPIC), "cond": mean(arm, COND), "mid": mean(arm, MID)}
    for k, v in llm.items():
        res["arms"][k] = {"all": float(np.mean([v[t] for t in TASKS])), "topic": float(np.mean([v[t] for t in TOPIC])), "cond": float(np.mean([v[t] for t in COND])),
                          "mid": float(np.mean([v[t] for t in MID]))}
    Z1 = res["arms"]["Z1"]
    gA = Z1["topic"] >= 0.90
    replace = [t for t in TASKS if res["per_task"][t]["Z1"] >= llm["26B"][t] - 0.01]
    gB = bool(replace)
    gC = Z1["all"] >= res["arms"].get("E4B", {"all": 0.885})["all"]
    pred_topic = all(res["per_task"][t]["Z1"] >= 0.80 for t in TOPIC)
    pred_cond = all(res["per_task"][t]["Z1"] <= 0.65 for t in COND)
    tw = twin_pairs()
    cc = confidence_and_cascade()
    res.update({"confidence": cc, "twin_pairs": tw, "llm": llm, "gates": {"A_intent_router": gA, "B_replace_26b_tasks": replace, "C_beats_e4b": gC},
                "predictions": {"topic_ge_0.80": pred_topic, "cond_le_0.65": pred_cond}, "run": run})
    json.dump(res, open(os.path.join(ROOT, "results/v13.json"), "w"), ensure_ascii=False, indent=1)
    lat = run.get("latency_ms_batch1", {})
    L = ["# 18 — EmbeddingGemma 2 當零樣本決策器：D0 上到什麼程度（v13）", "",
         "計劃與門檻：`docs/handoff-v13-embeddinggemma2.md`（跑前寫死並先 commit）。模型 `google/embeddinggemma-2` @ `914f7f8`，sentence-transformers，本機 CPU（4 vCPU）。"
         "做法模仿 Google MediaPipe Decision Maker 的 embedding 後端：選項文字（label：判準）先編碼並在題內中心化，state 編碼一次，取餘弦最高者。D0 test 每類 100 筆，切分同 jevlike。", "",
         "## 判定", "",
         f"- **A 可當意圖路由前置**（主題型三類 Z1 平均 ≥ 0.90）：{Z1['topic']:.3f} → {'✓ 可以' if gA else '✗ 不行'}",
         f"- **B 可取代 26B 的類別**（Z1 ≥ 26B − 1 點）：{('、'.join(ZH[t] for t in replace)) if replace else '沒有'}",
         f"- **C 零樣本整體 ≥ E4B**（{res['arms'].get('E4B', {}).get('all', 0.885):.3f}）：Z1 十類平均 {Z1['all']:.3f} → {'✓' if gC else '✗ 手機上不能用它替代 E4B'}",
         f"- 預測對答案：主題型三類都 ≥ 0.80 {'✓ 對' if pred_topic else '✗ 錯'}；條件型五類都 ≤ 0.65 {'✓ 對' if pred_cond else '✗ 錯'}。", "",
         "## 逐類 test 準確率", "",
         "| 類別 | 型 | 亂猜 | Z1 主臂 | Z2 不中心化 | Z3 檢索式 | P1 原型（100 標註） | P2 LR（100 標註） | E2B | E4B | 12B | 26B |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in TOPIC + MID + COND:
        p = res["per_task"][t]
        typ = "主題" if t in TOPIC else ("中間" if t in MID else "條件")
        L.append(f"| {ZH[t]} | {typ} | {p['chance']:.2f} | **{p['Z1']:.2f}** | {p['Z2']:.2f} | {p['Z3']:.2f} | {p['P1']:.2f} | {f(p['P2'], 2)} | "
                 + " | ".join(f(llm[m].get(t), 2) if m in llm else "—" for m in ("E2B", "E4B", "12B", "26B")) + " |")
    L += ["", "## 平均", "", "| 臂 | 十類 | 主題型三類 | 中間兩類 | 條件型五類 |", "|---|---|---|---|---|"]
    for arm in ("Z1", "Z2", "Z3", "P1", "P2", "E2B", "E4B", "12B", "26B"):
        if arm in res["arms"]:
            v = res["arms"][arm]
            L.append(f"| {arm} | {v['all']:.3f} | {v['topic']:.3f} | {v['mid']:.3f} | {v['cond']:.3f} |")
    P = sum(v["pairs"] for v in tw.values())
    L += ["", "## 孿生題（探索，不判定）", "",
          "D0 的孿生題是同一個 pair_id、答案相反的兩題，通常只差一個數字或一個狀態（例：停了 85 分鐘還沒修好、後段在等板 vs 停了 10 分鐘、ME 已排除）。"
          f"這一對在 EmbeddingGemma 2 上的餘弦相似度是 0.922。全部 {P} 對裡：EmbeddingGemma 2 給兩題同一個答案 {sum(v['eg2_same'] for v in tw.values()) / P:.0%}，"
          f"26B {sum(v['26b_same'] for v in tw.values()) / P:.0%}；EmbeddingGemma 2 兩題都答對 {sum(v['eg2_both_right'] for v in tw.values()) / P:.0%}。", "",
          "| 類別 | 對數 | EG2 同答 | 26B 同答 | EG2 兩題都對 |", "|---|---|---|---|---|"]
    for t in TOPIC + MID + COND:
        v = tw[t]
        L.append(f"| {ZH[t]} | {v['pairs']} | {v['eg2_same'] / v['pairs']:.0%} | {v['26b_same'] / v['pairs']:.0%} | {v['eg2_both_right'] / v['pairs']:.0%} |")
    au = [v["auroc"] for v in cc["per_task"].values() if v["auroc"] is not None]
    kk = {}
    for t in TASKS:
        kk.setdefault(res["per_task"][t]["K"], []).append(res["per_task"][t]["Z1"])
    c = cc["cascade_eg2_to_26b_eps5"]
    L += ["", "## 信心與級聯（2026-10-07 跑完後補算，不判定）", "",
          f"溫度在 cal 半擬合後，信心分辨對錯的 AUROC 十類平均 {np.mean(au):.2f}（0.5 = 亂猜）；5% 錯誤預算（v9 選擇性風險控制，切點在 cal 上定）下只有 "
          f"{sum(v['coverage_eps5'] > 0 for v in cc['per_task'].values())} 類能自動處理任何題，平均自動處理 {np.mean([v['coverage_eps5'] for v in cc['per_task'].values()]):.0%}。"
          f"EmbeddingGemma 2 當第一關、沒過切點的送 26B：準確率 {c['acc']:.3f}（全 26B {llm['26B'] and np.mean(list(llm['26B'].values())):.3f}），但 {c['sent_to_26b']:.0%} 的題還是送 26B（E4B 當第一關是 34%）。", "",
          "| 類別 | 選項數 | Z1 準確率 | 信心 AUROC | 5% 預算自動處理 | 自動處理中的錯誤率 |", "|---|---|---|---|---|---|"]
    for t in TOPIC + MID + COND:
        v = cc["per_task"][t]
        L.append(f"| {ZH[t]} | {v['K']} | {res['per_task'][t]['Z1']:.2f} | {f(v['auroc'], 2)} | {v['coverage_eps5']:.0%} | {v['error_eps5']:.0%} |")
    L += ["", "依選項數平均 Z1 準確率：" + "；".join(f"{k} 選 {np.mean(v):.2f}（{len(v)} 類）" for k, v in sorted(kk.items()))
          + "。二選一並沒有比較容易：同樣是二選一，答案靠用詞的「要不要派工」0.89，靠數字的「UPH 異常」0.58。決定成敗的是答案在用詞裡還是在數字與狀態裡，不是選項數。"]
    L += ["", "## 速度與其他", "",
          f"- 本機 CPU（4 vCPU、torch {run.get('threads')} 執行緒）單一 state 編碼 p50 {f(lat.get('p50'), 1)} ms、p95 {f(lat.get('p95'), 1)} ms；模型載入 {run.get('load_s')} 秒。"
          "選項向量在 prewarm 算一次，之後每題只多 K 個內積，選項多寡幾乎不影響延遲。"
          "這個數字是 fp32 全精度、連圖片與聲音編碼器一起載入、跑在雲端容器的 4 核 CPU，**不能**拿來跟 Google 公布的手機數字比（Pixel 11 Pro TPU 純文字 8.3 ms、S26 Ultra CPU 27.1 ms，量化版 LiteRT）。",
          "- Z1 校準：擬出的 τ 與測試集 ECE 見 `v13.json` 的 `per_task`。答案不受 τ 影響。",
          "- Z1 在條件型題目上的預測分布（test）：" + "；".join(f"{ZH[t]} {res['per_task'][t]['Z1_pred_dist_test']}" for t in COND) + "（依選項 A、B、C… 計數；全擠在一格代表只是在認主題）。"]
    open(os.path.join(ROOT, "results/18-embeddinggemma2.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
