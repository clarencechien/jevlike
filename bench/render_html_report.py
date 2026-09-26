"""Build results/report.html — a self-contained imitator-style (v3) decision memo for
management: is a Jev-like typed-decision layer worth doing? TL;DR first, then each of the
eight questions in plain language. Numbers come from results/analysis.json and
results/modal/latency/*.json; the chassis CSS is pasted verbatim from a local copy.

Usage: python3 bench/render_html_report.py <path-to-report.css>
"""
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS = open(sys.argv[1], encoding="utf-8").read()
assert "imitator report chassis" in CSS
A = {r["task"]: r for r in json.load(open(os.path.join(ROOT, "results/analysis.json")))}
LAT = json.load(open(os.path.join(ROOT, "results/modal/latency/latency.json")))["results"]
SWA = json.load(open(os.path.join(ROOT, "results/modal/latency/latency_swafull.json")))["results"]
ORDER = ["x_ticket_route", "x_escalate", "x_10way_intent", "m_needs_dispatch", "p_line_change", "m_alarm_category",
         "q_defect_root", "p_uph_anomaly", "q_spc_action", "m_alarm_severity"]
ZH = {"m_alarm_severity": "機台警報有多急", "m_alarm_category": "機台警報是哪類問題", "m_needs_dispatch": "要不要派設備工程師",
      "q_spc_action": "SPC 管制圖該怎麼處置", "q_defect_root": "不良品是哪一站造成", "p_uph_anomaly": "產能是否異常",
      "p_line_change": "工單為什麼停", "x_ticket_route": "工單該派給哪個單位", "x_escalate": "要不要通知線長", "x_10way_intent": "訊息是十種意圖的哪一種"}


def e(s):
    return html.escape(str(s))


# ------------------------------------------------------------------ numbers
L1 = LAT["L1_single_100tok_2opt"]["summary"]
L1s = SWA["L1_single_100tok_2opt"]["summary"]
L7 = LAT["L7_chat_json"]["summary"]
L6 = LAT["L6_with_bg_generation"]
conc = {c: LAT[f"L5_concurrency_{c}"]["summary"] for c in (1, 4, 8)}
share10_default = LAT["L4_shared_state_10q_cache_on"]["summary"]["p50"] / LAT["L4_shared_state_1q_cache_on"]["summary"]["p50"]
share10_swa = SWA["L4_shared_state_10q_cache_on"]["summary"]["p50"] / SWA["L4_shared_state_1q_cache_on"]["summary"]["p50"]
per_q_swa = SWA["L4_shared_state_10q_cache_on"]["per_call"]["p50"]
V2 = {r["task"]: r for r in json.load(open(os.path.join(ROOT, "results/analysis-v2.json")))}
HO = {}
for t in ("m_alarm_severity", "q_spc_action"):
    for ver in ("v1", "v2"):
        rows = [json.loads(l) for l in open(os.path.join(ROOT, f"results/modal/accuracy/heldout_{ver}/{t}.jsonl"))]
        HO[(t, ver)] = sum(r["correct"] for r in rows) / len(rows)
LAD = json.load(open(os.path.join(ROOT, "results/ladder.json")))
H2 = ["m_alarm_category", "m_needs_dispatch", "q_defect_root", "m_alarm_severity", "q_spc_action", "p_uph_anomaly"]
H1 = ["p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]
d2_agree = sum(k["n"] for k in LAD["kappa"].values())
d2_rate = 584 / 600
CAS = json.load(open(os.path.join(ROOT, "results/cascade.json")))
SPD = json.load(open(os.path.join(ROOT, "results/speed_ladder.json")))
TASKDEF = json.load(open(os.path.join(ROOT, "data/seeds/tasks.json"), encoding="utf-8"))
V4 = json.load(open(os.path.join(ROOT, "results/v4.json")))
V6 = json.load(open(os.path.join(ROOT, "results/v6.json")))
V7 = json.load(open(os.path.join(ROOT, "results/v7.json")))
V8 = json.load(open(os.path.join(ROOT, "results/v8.json")))
JB26 = V8["arms"]["26B raw"]; JBE4 = V8["arms"]["E4B raw"]
THR = V7["thresholds"]["lock"]["tasks"]
STRONG7 = ["m_alarm_category", "m_needs_dispatch", "q_defect_root", "p_line_change", "x_ticket_route", "x_escalate", "x_10way_intent"]
cov5 = [THR[t]["eps"]["0.05"]["coverage_test"] for t in STRONG7]; err5 = [THR[t]["eps"]["0.05"]["error_test"] for t in STRONG7]
V4G = V4["gates"]; V4B = V4["bf16"]
V4S = {arm: json.load(open(os.path.join(ROOT, "results/modal/v4bench", f)))["results"] for arm, f in (("A1", "q8/v4.json"), ("B1", "sglang/v4.json"))}
for arm, alt in (("A1", "q8/v4_L4seq.json"), ("B1", "sglang/v4_L4warm.json")):  # best L4 mode per arm, as in analyze_v4
    r2 = json.load(open(os.path.join(ROOT, "results/modal/v4bench", alt)))["results"]
    for K in (1, 5, 10, 16):
        k = f"L4_shared_{K}q"
        if k in r2 and r2[k]["summary"]["p50"] < V4S[arm][k]["summary"]["p50"]:
            V4S[arm][k] = r2[k]
    for k in ("S3_d0_test_1001",):
        if k in r2 and k not in V4S[arm]:
            V4S[arm][k] = r2[k]
v4p = lambda arm, k: V4S[arm][k]["summary"]["p50"]
v4_guard_pass = sum(1 for v in V4["accuracy"].values() if v["ok"]); v4_guard_n = len(V4["accuracy"])
v4_acc_A1 = sum(v["A1"] for k, v in V4["accuracy"].items() if k.startswith("D0/")) / 10
v4_acc_B1 = sum(v["B1"] for k, v in V4["accuracy"].items() if k.startswith("D0/")) / 10


def example_of(task):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8")]
    zh = [r for r in rows if r["lang"] == "zh" and r["difficulty"] == "hard" and 30 <= len(r["state"]) <= 75]
    r = sorted(zh, key=lambda r: r["id"])[0] if zh else rows[0]
    return r["state"], TASKDEF[task]["options"][r["gold"]]["label"]


strong = [t for t in ORDER if A[t]["raw_test"]["acc"] >= 0.98]
weak = [t for t in ORDER if A[t]["raw_test"]["acc"] < 0.98]
sev, spc, uph = A["m_alarm_severity"], A["q_spc_action"], A["p_uph_anomaly"]
lc = spc["learning_curve"]
speed = L7["p50"] / L1["p50"]
best_rule = max(A.values(), key=lambda r: r.get("rules_acc_test", 0))


# ------------------------------------------------------------------ charts
def acc_bar_chart():
    rows = [(t, A[t]["raw_test"]["acc"]) for t in ORDER]
    h = 26 * len(rows) + 30
    x0, x1 = 190, 570
    out = [f'<svg viewBox="0 0 640 {h}" role="img" aria-label="十類產線判斷題，不做任何標註時的答對率">']
    for g in (0.5, 0.75, 1.0):
        x = x0 + (x1 - x0) * g
        out.append(f'<line class="grid" x1="{x:.0f}" y1="8" x2="{x:.0f}" y2="{h - 22}"/>')
        out.append(f'<text class="axis" x="{x:.0f}" y="{h - 6}" text-anchor="middle">{g * 100:.0f}%</text>')
    for i, (t, v) in enumerate(rows):
        y = 12 + 26 * i
        w = (x1 - x0) * v
        out.append(f'<text class="axis" x="{x0 - 8}" y="{y + 14}" text-anchor="end">{e(ZH[t])}</text>')
        out.append(f'<rect class="bar" x="{x0}" y="{y}" width="{w:.1f}" height="18" fill="var(--c1)"/>')
        out.append(f'<text class="label" x="{x0 + w + 6:.1f}" y="{y + 14}">{v * 100:.0f}%</text>')
    out.append("</svg>")
    table = "".join(f'<tr><td>{e(ZH[t])}<span class="sub">{t}</span></td><td class="num">{A[t]["raw_test"]["acc"] * 100:.0f}%</td>'
                    f'<td class="num">{A[t]["by"]["hard"]["acc"] * 100:.0f}%</td>'
                    f'<td class="num">{A[t].get("control", {}).get("acc", 0) * 100:.0f}%</td><td class="num">{A[t]["tfidf_acc_test"] * 100:.0f}%</td>'
                    f'<td class="num">{A[t].get("rules_acc_test", 0) * 100:.0f}%</td></tr>' for t in ORDER)
    return "\n".join(out), table


def learning_curve_chart():
    ns = lc["N"]
    series = [("傳統機器學習，用 N 筆標註訓練", "tfidf", "--c1"), ("Jev 式，完全不標註", "typed0", "--c2"),
              ("Jev 式，用 N 筆標註校準後只回答有把握的題", "typed_sel90", "--c3")]
    x0, x1, y0, y1 = 60, 600, 16, 240
    xs = {n: x0 + (x1 - x0) * i / (len(ns) - 1) for i, n in enumerate(ns)}
    out = ['<svg viewBox="0 0 640 280" role="img" aria-label="SPC 處置這一題：標註筆數對答對率，三個做法的比較">']
    for g in (0.4, 0.6, 0.8, 1.0):
        y = y1 - (y1 - y0) * (g - 0.4) / 0.6
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{g * 100:.0f}%</text>')
    for n in ns:
        out.append(f'<text class="axis" x="{xs[n]:.1f}" y="{y1 + 20}" text-anchor="middle">標註 {n} 筆</text>')
    for label, key, col in series:
        pts = [(xs[n], y1 - (y1 - y0) * (lc["mean"][key][str(n)] - 0.4) / 0.6) for n in ns]
        out.append('<path class="line" d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts)) + f'" stroke="var({col})"/>')
        for x, y in pts:
            out.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var({col})"/>')
        lx, ly = pts[-1]
        out.append(f'<text class="label" x="{lx + 8:.1f}" y="{ly + 4:.1f}">{lc["mean"][key][str(ns[-1])] * 100:.0f}%</text>')
    out.append("</svg>")
    legend = "".join(f'<span class="key"><span class="swatch" style="background:var({col})"></span>{e(label)}</span>' for label, _, col in series)
    table = "".join(f'<tr><td>{n} 筆</td><td class="num">{lc["mean"]["tfidf"][str(n)] * 100:.0f}%</td><td class="num">{lc["mean"]["typed0"][str(n)] * 100:.0f}%</td>'
                    f'<td class="num">{lc["mean"]["typed_sel90"][str(n)] * 100:.0f}%</td><td class="num">{lc["mean"]["typed_cov90"][str(n)] * 100:.0f}%</td></tr>' for n in ns)
    return "\n".join(out), legend, table


def ladder_chart():
    order = H2 + H1
    h = 34 * len(order) + 30
    x0, x1 = 190, 570
    out = [f'<svg viewBox="0 0 640 {h}" role="img" aria-label="十類題目在三個大小的模型上的答對率，D0 test 集">']
    for g in (0.5, 0.75, 1.0):
        x = x0 + (x1 - x0) * g
        out.append(f'<line class="grid" x1="{x:.0f}" y1="8" x2="{x:.0f}" y2="{h - 22}"/>')
        out.append(f'<text class="axis" x="{x:.0f}" y="{h - 6}" text-anchor="middle">{g * 100:.0f}%</text>')
    for i, t in enumerate(order):
        y = 10 + 34 * i
        out.append(f'<text class="axis" x="{x0 - 8}" y="{y + 16}" text-anchor="end">{e(ZH[t])}</text>')
        for j, (m, col) in enumerate((("26b", "--c1"), ("e4b", "--c2"), ("e2b", "--c3"))):
            v = LAD["results"][t]["D0"][m]["acc"]
            w = (x1 - x0) * v
            out.append(f'<rect class="bar" x="{x0}" y="{y + j * 8}" width="{w:.1f}" height="7" fill="var({col})"/>')
        v26 = LAD["results"][t]["D0"]["26b"]["acc"]; ve2 = LAD["results"][t]["D0"]["e2b"]["acc"]
        out.append(f'<text class="label" x="{x0 + (x1 - x0) * max(v26, ve2) + 6:.1f}" y="{y + 16}">{v26 * 100:.0f} / {ve2 * 100:.0f}</text>')
    out.append("</svg>")
    table = "".join(f'<tr><td>{e(ZH[t])}<span class="sub">{"26B 真的強" if t in H2 else "題目對小模型也簡單"}</span></td>'
                    + "".join(f'<td class="num">{LAD["results"][t]["D0"][m]["acc"] * 100:.0f}%</td>' for m in ("26b", "e4b", "e2b"))
                    + f'<td class="num">{LAD["results"][t]["D2"]["26b"]["acc"] * 100:.0f}%</td><td class="num">{LAD["results"][t]["D2"]["e4b"]["acc"] * 100:.0f}%</td></tr>' for t in order)
    return "\n".join(out), table


lad_svg, lad_table = ladder_chart()
cas_rows = "".join(f'<tr><td>{e(r[0])}</td><td class="num">{r[1] * 100:.1f}%</td><td class="num">{r[2] * 100:.0f}%</td><td class="num">{r[3] / 1000:.2f} 秒</td></tr>' for r in SPD["cascade"])
ex_rows = ""
for t in H2 + H1:
    st, ans = example_of(t)
    opts = " / ".join(v["label"] for v in TASKDEF[t]["options"].values())
    ex_rows += (f'<tr><td class="nowrap">{e(ZH[t])}<span class="sub">{"26B 真的強" if t in H2 else "小模型也行"}</span></td>'
                f'<td>{e(st)}</td><td>{e(opts)}</td><td class="nowrap">{e(ans)}</td></tr>')
acc_svg, acc_table = acc_bar_chart()
lc_svg, lc_legend, lc_table = learning_curve_chart()

# ------------------------------------------------------------------ the eight questions, in plain language
Q = [
    ("Q1", "上界", "一次判斷要多久？比讓模型寫一段答案快多少？",
     f"在租來的 L4 顯示卡上，一次判斷約 {L1['p50'] / 1000:.1f} 秒；同一題改讓模型寫出答案要 {L7['p50'] / 1000:.1f} 秒，Jev 式快 {speed:.1f} 倍。若一次問十題（同一份現場狀況），開對設定後平均每題只要 {per_q_swa / 1000:.2f} 秒。",
     "夠快。產線上每天幾百到幾千次的判斷，這個速度可以即時回應。GB10 的真實數字要再量一次，租來的卡只能當保守估計。"),
    ("Q2", "已答", "十類判斷題裡，哪些不用訓練就能用？哪些要再調？",
     f"{len(strong)} 類完全不標註就有 98% 以上答對率。剩下 {len(weak)} 類（{'、'.join(ZH[t] for t in weak)}）落在 {min(A[t]['raw_test']['acc'] for t in weak) * 100:.0f}–{max(A[t]['raw_test']['acc'] for t in weak) * 100:.0f}%，錯的都在相鄰等級的邊界，例如「盡快處理」和「立刻停線」之間。",
     f"不需要為這件事訓練新模型。已實測其中兩類：把「連續幾點、超限幾次、停線多久」寫成可數的邊界後，SPC 處置從 {A['q_spc_action']['raw_test']['acc'] * 100:.0f}% 升到 {V2['q_spc_action']['raw_test']['acc'] * 100:.0f}%，警報急迫度從 {A['m_alarm_severity']['raw_test']['acc'] * 100:.0f}% 升到 {V2['m_alarm_severity']['raw_test']['acc'] * 100:.0f}%；用另一批沒看過的題驗證，改善幅度相同（{HO[('q_spc_action','v1')] * 100:.0f}% → {HO[('q_spc_action','v2')] * 100:.0f}%、{HO[('m_alarm_severity','v1')] * 100:.0f}% → {HO[('m_alarm_severity','v2')] * 100:.0f}%）。"),
    ("Q3", "已答", "同一份現場狀況一次問好幾題，成本會不會倍增？",
     f"預設設定下會：問十題要 {share10_default:.0f} 倍時間。開啟一個伺服器設定後降到 {share10_swa:.1f} 倍，第二題起每題只要 {per_q_swa / 1000:.2f} 秒。",
     "可以一次問五到十題，把「這則警報有多急、是哪類問題、要不要派工」一起問掉。部署時要記得開那個設定。"),
    ("Q4", "上界", "機台同時在讓模型寫報告時，判斷會被拖慢多少？",
     f"背景持續寫長文時，判斷從 {L1['p50'] / 1000:.1f} 秒變 {L6['summary']['p50'] / 1000:.1f} 秒（{L6['slowdown_p50']:.1f} 倍）。同時來 8 個請求時每個要等 {conc[8]['p50'] / 1000:.1f} 秒，租來的卡每秒最多處理 {conc[8]['throughput_rps']:.0f} 次判斷。",
     "每秒五次以內共用一台就夠。尖峰更高的話要在 GB10 上實測，再決定要不要另外放一個小模型專門做判斷。"),
    ("Q5", "已答", "中文夾機台代碼的警報，會不會比英文差？",
     f"不會。中文普遍不輸英文；英文警報碼在四類題目略差（例如警報急迫度：英文 {sev['by']['en']['acc'] * 100:.0f}%、中文 {sev['by']['zh']['acc'] * 100:.0f}%），主要是英文警報訊息本身太短。",
     "現場的中文工單和警報可以直接用，不用先翻譯或整理格式。"),
    ("Q6", "合成版", "模型說「我有把握」的時候，可以信到什麼程度？",
     f"原始狀態不能信：模型幾乎每一題都說有 99% 把握，對錯都一樣。以警報急迫度為例，它自稱有把握的題目裡只有 {sev['raw_test']['sel90']['acc'] * 100:.0f}% 答對。用一百筆資料校準後，它說有把握的題目答對率變成 100%，但只剩 {sev['affine_test']['sel90']['coverage'] * 100:.0f}% 的題目它敢答；SPC 處置校準後是 {spc['affine_test']['sel90']['acc'] * 100:.0f}% 答對、{spc['affine_test']['sel90']['coverage'] * 100:.0f}% 敢答。",
     "「有把握就自動處理、沒把握就轉人或轉大模型」這個機制可行，但門檻一定要用真實資料校準過才能上線。"),
    ("Q7", "已答", "還需要標多少資料？",
     f"傳統機器學習標 100 筆還到不了 90%（SPC 處置只有 {lc['mean']['tfidf']['100'] * 100:.0f}%）。Jev 式七類題目零筆就 98% 以上；較弱的三類標 25 筆校準，就能在八成左右的題目上做到 95% 以上。",
     "主管版報告承諾「從每題幾千筆降到幾百筆」，實測比承諾更好：多數題目零筆，其餘幾十筆。"),
    ("Q8", "已答", "用關鍵字規則就夠的題目，有沒有？",
     f"沒有。十類題目用工程師手寫的關鍵字規則，最好的一類也只有 {best_rule['rules_acc_test'] * 100:.0f}%（{ZH[best_rule['task']]}），其餘 47–87%。",
     "規則可以留著當第一道過濾，但不能只靠規則。"),
]
CHIP = {"已答": "已答", "上界": "上界，GB10 待實測", "合成版": "方法成立，數字待真實資料"}
q_blocks = "".join(
    f'<div class="q"><p class="verdict">{q} · {e(CHIP[status])}</p><h3>{e(title)}</h3>'
    f'<p><strong>答案</strong>　{answer}</p><p class="so"><strong>這代表</strong>　{meaning}</p></div>'
    for q, status, title, answer, meaning in Q)

# ------------------------------------------------------------------ page
page = f"""<!doctype html>
<html lang="zh-Hant">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="imitator-style" content="v3">
<meta name="imitator-register" content="判定備忘 — Jev 式決策層值不值得做，給長官一頁講完">
<meta name="imitator-reference" content="1980 年代工廠 QC 課點陣印表機印在連續報表紙上的實驗紀錄 — 編號所見、判定欄、等寬欄位">
<meta name="imitator-paper" content="hsl(120 18% 94%)">
<meta name="imitator-accent" content="hsl(178 58% 26%)">
<title>Jev 式決策，值得，而且不用買</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
{CSS}
/* REGISTER:  判定備忘 — 給長官的一頁：Jev 式決策層值不值得做。先給結論與代價，再把八個問題
              一題一題用白話講「問題是什麼、答案是什麼、這代表什麼」。工程細節收進條件與附錄。
   REFERENCE: 1980 年代工廠 QC 課的實驗紀錄 — 點陣印表機印在連續報表紙（greenbar）上，
              編號的所見、判定欄、等寬體的欄位，所見與判定分開寫。
   PAPER:     連續報表紙的淡綠 — hsl(120 18% 94%)。
   VOICE:     終端機青綠只給裁決、章節號和連結；標題用 IBM Plex Sans Condensed 壓 Noto Sans TC；
              判定用等寬體標籤：已答 / 上界 / 方法成立。
   NOT:       儀表板，也不是工程報告。每個數字後面跟一句「這代表」。
   RECENT:    paper 355° 50° 20° · accent 205° 216° 282° — 同一 slug 的改版，沿用第一版的
              greenbar 120° + 青綠 178°；178° 離 205° 有 27°。 */
:root {{
  --paper: hsl(120 18% 94%);
  --card: hsl(120 22% 97%);
  --ink: hsl(140 14% 12%);
  --ink-2: hsl(140 9% 30%);
  --mist: hsl(140 7% 42%);
  --rule: hsl(120 14% 84%);
  --rule-hard: hsl(120 12% 70%);
  --accent: hsl(178 58% 26%);
  --accent-soft: hsl(178 35% 88%);
  --disp: "IBM Plex Sans Condensed", "Noto Sans TC", sans-serif;
  --sans: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    --paper: hsl(140 12% 8%);
    --card: hsl(140 11% 11%);
    --ink: hsl(110 14% 90%);
    --ink-2: hsl(110 9% 74%);
    --mist: hsl(110 6% 60%);
    --rule: hsl(140 9% 20%);
    --rule-hard: hsl(140 8% 32%);
    --accent: hsl(176 48% 58%);
    --accent-soft: hsl(178 30% 18%);
  }}
}}
:root[data-theme="dark"] {{
  --paper: hsl(140 12% 8%);
  --card: hsl(140 11% 11%);
  --ink: hsl(110 14% 90%);
  --ink-2: hsl(110 9% 74%);
  --mist: hsl(110 6% 60%);
  --rule: hsl(140 9% 20%);
  --rule-hard: hsl(140 8% 32%);
  --accent: hsl(176 48% 58%);
  --accent-soft: hsl(178 30% 18%);
}}
/* greenbar: a faint band behind every other table row, the way the paper carried it */
.table-scroll tbody tr:nth-child(even) td {{ background: color-mix(in srgb, var(--accent-soft) 45%, transparent); }}
.verdict {{ font-family: var(--mono); font-size: var(--fs-xs); letter-spacing: .08em; color: var(--accent); }}
/* the eight questions: one numbered entry each, like a findings column — not a table on a phone */
.qlist .q {{ border-top: 1px solid var(--rule); padding: var(--sp-4) 0 var(--sp-3); }}
.qlist .q:last-child {{ border-bottom: 1px solid var(--rule-hard); }}
.qlist .q h3 {{ margin: var(--sp-1) 0 var(--sp-2); font-size: var(--fs-3); }}
.qlist .q p {{ margin: 0 0 var(--sp-2); }}
.qlist .q .verdict {{ margin: 0; }}
.qlist .q .so {{ color: var(--ink-2); }}
/* the TL;DR: a ruled block, the verdict line set in the display face */
.tldr {{ border-top: 3px solid var(--accent); border-bottom: 1px solid var(--rule-hard); padding: var(--sp-4) 0 var(--sp-3); margin-block: var(--sp-6); }}
.tldr .head {{ font-family: var(--disp); font-size: var(--fs-2); font-weight: 700; margin: 0 0 var(--sp-3); }}
.tldr ol {{ margin: 0; padding-left: 1.4em; }}
.tldr li + li {{ margin-top: var(--sp-2); }}
.route table {{ min-width: 40rem; }}
</style>
<body>
<div class="progress"></div>
<button class="theme-toggle" id="theme-toggle" type="button" aria-pressed="false">theme · auto</button>

<main class="report">

  <p class="eyebrow">判定備忘 · Jev 式決策層 · 2026-09-23 起、八輪實驗 · 更新 2026-09-26</p>
  <h1 class="display">Jev 式決策，<br>值得做，<br>而且<em>不用買</em></h1>
  <p class="lede">用我們已經部署的 Gemma 4 26B，加一層「只從固定選項裡選答案、不寫文章」的決策 API。十類產線判斷題有七類不用任何標註就能上線，每次判斷約 0.2 秒。八輪驗證（含小模型對照、換引擎、追查掉分原因、借鏡他人做法、跑公開考卷）合計約十五美元的雲端 GPU。</p>
  <p class="byline">實驗於 Modal 雲端 L4 顯示卡進行 · 合成資料 10 類 × 200 題 · 工程細節與原始數據在 results/REPORT.md</p>

  <div class="tldr">
    <p class="head">TL;DR</p>
    <ol>
      <li><strong>判定：值得做，用現有模型自己做，不採購 Jev。</strong>Jev 是閉源雲端服務，資料出不了廠；我們要的能力，現有的 26B 模型加幾十行程式就有。</li>
      <li><strong>準確率：十類題目七類直接達標，而且分得出「題目簡單」和「模型真的強」。</strong>不做任何標註，{len(strong)} 類答對率 98% 以上。拿更小的 Gemma 4 E2B / E4B 跑同一批題：四類小模型也做得到（題目對這一級太簡單），六類 26B 比 E4B 高 6–13 個百分點、比 E2B 高 9–30 個百分點（是真的強）。另請 Gemini 盲寫 600 題、兩個大模型獨立標答，一致率 {d2_rate * 100:.0f}%，26B 在這批題上平均 {sum(LAD['results'][t]['D2']['26b']['acc'] for t in ORDER) / 10 * 100:.0f}%，已到兩個大模型互相一致的水準。</li>
      <li><strong>速度：一次判斷 {L1['p50'] / 1000:.1f} 秒，比讓模型寫答案快 {speed:.1f} 倍。</strong>同一份現場狀況一次問十題，開對設定後每題再降到 {per_q_swa / 1000:.2f} 秒。</li>
      <li><strong>標註：比主管報告承諾的還少。</strong>多數題目零筆；較弱的題目標 25 筆就能校準。傳統機器學習標 100 筆還到不了 90%。</li>
      <li><strong>推理引擎：SGLang 批次快五倍，第一輪掉的 {(v4_acc_A1 - v4_acc_B1) * 100:.1f} 個百分點是少一個起始符號，補上後與現行引擎同準。</strong>共用同一份現場狀況問十六題快 {V4G['L4_speedup_16']:.1f} 倍、多路併發快 6 倍。事後批次整理直接用 SGLang；即時判斷兩個引擎在 GB10 上各量一次再選。</li>
      <li><strong>公開考卷：拿 JevBench 231 題自跑，{JB26['accuracy'] * 100:.1f}%，高於同做法的 Cygnet 與訓練過的 Open-Jev。</strong>難題 {JB26['per_tier']['hard'] * 100:.1f}%；小模型 E4B {JBE4['accuracy'] * 100:.1f}%。自跑、不排名，但說明產線題的高分不是題目量身訂做。</li>
      <li><strong>四個上線條件：</strong>部署時開啟前綴快取設定、「有把握才自動處理」的門檻要用真實資料校準、選項順序固定（弱題有兩成會因順序改答案）、GB10 上重量一次速度。都是幾小時到幾天的事，不是幾個月。</li>
    </ol>
  </div>

  <div class="tiles wide stagger">
    <div class="stat"><p class="label">不標註就達 98% 的題型</p><div class="value">{len(strong)} / 10</div><div class="delta">其餘三類要調門檻</div></div>
    <div class="stat"><p class="label">一次判斷</p><div class="value">{L1['p50'] / 1000:.1f} 秒</div><div class="delta">租用 L4；GB10 待實測</div></div>
    <div class="stat"><p class="label">比模型寫答案快</p><div class="value">{speed:.1f} 倍</div><div class="delta">一次問十題可再降</div></div>
    <div class="stat"><p class="label">本次驗證的 GPU 費用</p><div class="value">≈ $15</div><div class="delta">八輪實驗，約 8.9 GPU 小時</div></div>
  </div>

  <p class="eyebrow">00 · 走到哪裡了</p>
  <h2>八輪、四天、十五美元：每一輪問一個問題，答一個問題</h2>
  <p>每一輪都先把「什麼算過、什麼算沒過」寫死再跑，跑完照規則填結論。沒過的也留著，因為「不值得做」和「值得做」一樣是答案。</p>

  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>輪</th><th>問的問題</th><th>答案</th><th class="num">GPU 費用</th></tr></thead>
      <tbody>
        <tr><td>一、二</td><td>讀選項機率當決策 API，準不準、快不快、要標多少？</td><td>七類零標註 ≥ 98%，一次 0.2 秒，多數題零筆標註；模型永遠說有把握，門檻要校準</td><td class="num">$1</td></tr>
        <tr><td>二補</td><td>弱題是不是標準沒寫清楚？</td><td>把「連續幾點、超限幾次」寫成可數規則，SPC {A['q_spc_action']['raw_test']['acc'] * 100:.0f}% → {V2['q_spc_action']['raw_test']['acc'] * 100:.0f}%，急迫度 {A['m_alarm_severity']['raw_test']['acc'] * 100:.0f}% → {V2['m_alarm_severity']['raw_test']['acc'] * 100:.0f}%；換一批題驗證等幅</td><td class="num">$0.2</td></tr>
        <tr><td>三</td><td>是題目太簡單，還是模型真的強？</td><td>拿 2B、4B 跑同一批：四類小模型也會，六類 26B 真的強；先問小模型、三分之一再問 26B，答對率不變、負載剩三分之一</td><td class="num">$1</td></tr>
        <tr><td>四</td><td>換 SGLang 引擎值不值？</td><td>批次快 3.7 到 6 倍，但答對率低 2.8 個百分點、重跑會變</td><td class="num">$7</td></tr>
        <tr><td>五</td><td>別人（TypeLLM）的技巧有沒有用？</td><td>JSON 形式與選項順序平均都沒過門檻；量到弱題兩成會因順序改答案，列為上線條件</td><td class="num">$1</td></tr>
        <tr><td>六</td><td>SGLang 掉分的真正原因？</td><td>少一個起始字元。補上後與現行引擎同準（{V6['E1']['mean']['B1ids'] * 100:.1f}% vs {V6['E1']['mean']['A1'] * 100:.1f}%），重跑一致率 99.85%；SGLang 回到候選</td><td class="num">$3.5</td></tr>
        <tr><td>八</td><td>拿別人的公開考卷（JevBench 231 題）跑同一套，站得住嗎？</td><td>{JB26['accuracy'] * 100:.1f}%，難題 {JB26['per_tier']['hard'] * 100:.1f}%；高於同做法的 Cygnet 87.9%、訓練過的 Open-Jev 85.3%。自跑、不排名</td><td class="num">$0.2</td></tr>
        <tr><td>七</td><td>三十幾個開源替代品裡，有什麼可以抄？</td><td>抄了三件：錯誤預算反推門檻（採用）、模板健康檢查（採用）、多題一個前向（不採用，後面的題答案會變）</td><td class="num">$1.4</td></tr>
      </tbody>
    </table>
  </div>

  <h2>效果多好：一張記分板</h2>
  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>指標</th><th class="num">我們</th><th>對照</th></tr></thead>
      <tbody>
        <tr><td>公開考卷 JevBench 231 題（自跑）</td><td class="num">{JB26['accuracy'] * 100:.1f}%</td><td>同做法的 Cygnet 87.9%、訓練過的 Open-Jev 85.3%、TypeLLM 84.4%；不是官方排名</td></tr>
        <tr><td>零標註就達 98% 的題型</td><td class="num">{len(strong)} / 10</td><td>Jev 在自家題約 68%、公開評測 74 分；題目不同，只看量級</td></tr>
        <tr><td>三類弱題（急迫度、SPC、產能）改寫標準後</td><td class="num">{V2['m_alarm_severity']['raw_test']['acc'] * 100:.0f}% / {V2['q_spc_action']['raw_test']['acc'] * 100:.0f}% / {A['p_uph_anomaly']['raw_test']['acc'] * 100:.0f}%</td><td>剩下的錯集中在相鄰等級，要真實資料</td></tr>
        <tr><td>給 5% 錯誤預算，強題能自動處理的比例</td><td class="num">{min(cov5) * 100:.0f}%–{max(cov5) * 100:.0f}%</td><td>實際錯誤率 {min(err5) * 100:.0f}%–{max(err5) * 100:.1f}%，保證在 8/10 類成立</td></tr>
        <tr><td>一次判斷（租用 L4）</td><td class="num">{L1['p50'] / 1000:.2f} 秒</td><td>讓模型寫答案 {speed:.1f} 倍慢；同一份狀況問十題每題 {per_q_swa / 1000:.2f} 秒</td></tr>
        <tr><td>SGLang 批次：共用狀況問十六題 / 32 路併發</td><td class="num">快 {V4G['L4_speedup_16']:.1f} 倍 / 6 倍</td><td>準確率與現行引擎同（補上起始字元後）</td></tr>
        <tr><td>先問 4B、沒把握再問 26B</td><td class="num">{CAS['mean'][6] * 100:.1f}%</td><td>全用 26B {CAS['mean'][1] * 100:.1f}%；26B 負載剩三分之一</td></tr>
        <tr><td>累計 GPU 費用</td><td class="num">≈ $15</td><td>原估 $5.5–7.5 只算前兩輪；後五輪是追加的問題</td></tr>
      </tbody>
    </table>
  </div>

  <p class="eyebrow">01 · 這是什麼</p>
  <h2>六十秒說明 Jev 式決策</h2>
  <p>今天讓大模型做產線判斷，通常是丟一段描述請它「寫出」判斷結果，再從那段文字裡撈答案。慢、貴、而且偶爾撈不到。Jev 是一家新創推出的做法：<span class="mark">不讓模型寫，只讓它從我們給的幾個選項裡選一個，並附上它有多大把握</span>。一次判斷只算一步，所以快；答案永遠是選項之一，所以不會亂答；有把握度，所以可以「有把握就自動處理、沒把握就轉人」。</p>
  <p>Jev 本身是閉源雲端服務，產線資料送不出去。這次驗證的是：<strong>用我們自己的 Gemma 4 26B 做同一件事，行不行。</strong>做法是讓模型只回答一個字母，讀它對每個字母的機率。不重新部署模型，不訓練，加一層程式而已。</p>

  <p class="eyebrow">02 · 準確率</p>
  <h2>十類題目，七類不用教就會</h2>
  <p>我們做了十類產線判斷題，每類 200 題：七成是規則展開的一般題，三成是刻意寫得資訊不全、夾錯字、放干擾資訊的難題。全部由另一家模型出題與定答案，避免自己考自己；再由獨立審核抽查 10%，錯題率為零，可爭議的 2.5%。</p>

  <figure class="reveal">
    <p class="title">十類判斷題，不做任何標註時的答對率</p>
    <p class="subtitle">同一模型改讓它寫出答案、傳統機器學習、關鍵字規則三種對照在表裡。</p>
    <div class="chart">
{acc_svg}
    </div>
    <details class="datatable">
      <summary>看數字</summary>
      <div class="table-scroll">
        <table>
          <thead><tr><th>題型</th><th class="num">Jev 式</th><th class="num">難題</th><th class="num">模型寫答案</th><th class="num">傳統 ML</th><th class="num">關鍵字規則</th></tr></thead>
          <tbody>{acc_table}</tbody>
        </table>
      </div>
    </details>
    <figcaption>三類較弱的題目，錯的集中在「盡快處理 vs 立刻停線」「加抽檢 vs 停線複檢」這種相鄰等級之間，是判斷標準寫法的問題，不是模型能力的問題。</figcaption>
  </figure>

  <div class="note warn">
    <p class="head">一個要小心的地方：模型太有自信</p>
    <p>不管對錯，模型幾乎每題都說自己有 99% 把握。所以「有把握就自動處理」的門檻不能直接用模型原始的把握度，要先用一百筆左右的資料校準。校準後，它說有把握的題目答對率可以到 96–100%，代價是它會把一到六成的題目交給人。這是可以接受的分工，但門檻要用真實資料定。</p>
  </div>

  <p class="eyebrow">02b · 把標準寫清楚</p>
  <h2>判斷標準改成可數規則，兩類弱題立刻回升</h2>
  <p>三類較弱題目裡挑兩類做了實驗。原本的標準用形容詞描述邊界（「即將造成批量問題」「已連續多點」），模型在相鄰等級之間一律往嚴重的那級靠。改成「先數什麼、數到幾就是哪一級」：SPC 先數超出管制界限的點有幾個，兩個以上就叫品保、恰好一個就停線複檢；急迫度先看有沒有不良已經產出或整線在等，有就停線，只有單站停就是盡快。</p>
  <div class="table-scroll">
    <table>
      <thead><tr><th>題型</th><th class="num">原標準</th><th class="num">可數標準</th><th class="num">沒看過的新題：原標準</th><th class="num">沒看過的新題：可數標準</th></tr></thead>
      <tbody>
        <tr><td>SPC 管制圖該怎麼處置</td><td class="num">{A['q_spc_action']['raw_test']['acc'] * 100:.0f}%</td><td class="num">{V2['q_spc_action']['raw_test']['acc'] * 100:.0f}%</td><td class="num">{HO[('q_spc_action','v1')] * 100:.0f}%</td><td class="num">{HO[('q_spc_action','v2')] * 100:.0f}%</td></tr>
        <tr><td>機台警報有多急</td><td class="num">{A['m_alarm_severity']['raw_test']['acc'] * 100:.0f}%</td><td class="num">{V2['m_alarm_severity']['raw_test']['acc'] * 100:.0f}%</td><td class="num">{HO[('m_alarm_severity','v1')] * 100:.0f}%</td><td class="num">{HO[('m_alarm_severity','v2')] * 100:.0f}%</td></tr>
      </tbody>
    </table>
  </div>
  <div class="note">
    <p class="head">這算不算先射箭再畫靶</p>
    <p>一半算。答案沒有動，改的是我們自己定的判斷規格，這是正當的；但新標準是看著錯題改出來的，同一批題上的數字偏樂觀。所以另外產了一批模型沒看過的題再測一次，改善幅度幾乎相同，證明它不是記住特定句子。真正的驗證還是要等真實警報與工單。</p>
  </div>

  <p class="eyebrow">02c · 題目簡單，還是模型強</p>
  <h2>拿小模型跑同一批題，十類分成兩種</h2>
  <p>「七類近滿分」有兩種解釋：題目太簡單，任何會讀中文的模型都會；或 26B 真的強。分辨方法是拿同一家更小的模型（Gemma 4 E2B 約 2B、E4B 約 4B）跑一模一樣的題。小模型也滿分，就是題目簡單；小模型明顯掉，就是 26B 的能力。判讀規則在跑之前就寫死（差距 ≥ 5 個百分點且統計檢定 p &lt; 0.05 才算「明顯掉」）。</p>

  <figure class="reveal">
    <p class="title">十類題目在三個大小的模型上的答對率</p>
    <p class="subtitle">前六類 26B 明顯高於小模型；後四類三個模型都在 95% 以上。標籤數字是 26B / E2B。</p>
    <div class="chart">
{lad_svg}
    </div>
    <div class="legend">
      <span class="key"><span class="swatch" style="background:var(--c1)"></span>Gemma 4 26B（現在用的）</span>
      <span class="key"><span class="swatch" style="background:var(--c2)"></span>Gemma 4 E4B（約 4B）</span>
      <span class="key"><span class="swatch" style="background:var(--c3)"></span>Gemma 4 E2B（約 2B）</span>
    </div>
    <details class="datatable">
      <summary>看數字</summary>
      <div class="table-scroll">
        <table>
          <thead><tr><th>題型</th><th class="num">26B</th><th class="num">E4B</th><th class="num">E2B</th><th class="num">盲寫題 26B</th><th class="num">盲寫題 E4B</th></tr></thead>
          <tbody>{lad_table}</tbody>
        </table>
      </div>
    </details>
    <figcaption>錯字擾動下差距不變（26B 幾乎不掉）。把題目裡和判斷標準重疊的字換掉，26B 也沒有掉，不是靠對字作答。</figcaption>
  </figure>

  <p>這對 GB10 的部署有直接意義：<span class="mark">六類要用 26B，四類（工單狀態、派給誰、要不要升級、訊息意圖）小到 2B 的模型就夠</span>，可以獨立放一個小模型做高頻判斷。另一個結果是反向的：原本想用 Gemini 盲寫的 600 題當「更難的題」，結果它比合成題容易——不看標準寫題，反而把線索寫得很完整，兩個大模型獨立標答一致率 {d2_rate * 100:.0f}%。所以真正拉開差距的是合成題裡刻意寫得資訊不全、邊界模糊的那三成難題，之後跟其他模型比要用那一批。</p>

  <p class="eyebrow">02d · 六類、四類長什麼樣</p>
  <h2>題目長什麼樣、怎麼分的、能不能先分再派</h2>
  <p>先看題目。每一類各挑一則手寫的難題，附選項與正確答案。前六類要「在幾個相鄰的等級或原因之間做判斷」，後四類是「訊息本身就說明了它是什麼」。</p>

  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>題型</th><th>一則例題</th><th>選項</th><th>答案</th></tr></thead>
      <tbody>{ex_rows}</tbody>
    </table>
  </div>

  <p><strong>怎麼分的。</strong>分的是「題型」，不是單一題目。十類題各有 100 題測試集，三個大小的模型跑同一批、同一個提問方式。規則在跑之前寫死：最小的 E2B 也答對 95% 以上的題型，判為「題目對小模型也簡單」；26B 比 E4B 高 5 個百分點以上、且統計檢定 p &lt; 0.05 的題型，判為「26B 真的強」。十類剛好分成 4 加 6，沒有落在中間的。</p>

  <p><strong>能不能先分類、再決定用哪個模型？</strong>可以，而且分兩層，第一層不需要分類器：</p>
  <ol>
    <li><strong>題型層級：不用分類器。</strong>問題是我們的系統自己發出的，發問時就知道是「要不要通知線長」還是「SPC 該怎麼處置」。四類固定走小模型、六類固定走 26B，是一張查表，不是判斷。</li>
    <li><strong>題目層級：讓小模型自己當分類器。</strong>同一類題裡也有簡單和難的。做法是先問 E4B，它的把握度夠高就採用，不夠就把同一題丟給 26B。不用另外訓練東西，把握度就是那個分類器。用十類共一千題模擬：</li>
  </ol>

  <div class="table-scroll">
    <table>
      <thead><tr><th>做法</th><th class="num">平均答對率</th><th class="num">送到 26B 的比例</th><th class="num">平均每題（L4）</th></tr></thead>
      <tbody>{cas_rows}</tbody>
    </table>
  </div>

  <p>結論：<span class="mark">先問小模型、三分之一的題再問 26B，答對率和全部用 26B 一樣</span>，而且四類簡單題幾乎不會被送上去（3–20%），六類難題會送 18–86%，等於系統自己找出了難題。兩個提醒：這裡用的是模型原始的把握度，上線要校準過；數字來自合成題，真實資料要重算門檻。</p>

  <p><strong>4B 到底多快，這樣做多值。</strong>同一張 L4 上實測：</p>
  <div class="table-scroll">
    <table>
      <thead><tr><th>一次判斷（p50）</th><th class="num">26B</th><th class="num">E4B</th><th class="num">E2B</th></tr></thead>
      <tbody>
        <tr><td>單題，每次不同的現場狀況</td><td class="num">{SPD['L1']['26b'] / 1000:.2f} 秒</td><td class="num">{SPD['L1']['e4b'] / 1000:.2f} 秒</td><td class="num">{SPD['L1']['e2b'] / 1000:.2f} 秒</td></tr>
        <tr><td>同一份狀況問十題</td><td class="num">{SPD['t10']['26b'] / 1000:.2f} 秒</td><td class="num">{SPD['t10']['e4b'] / 1000:.2f} 秒</td><td class="num">{SPD['t10']['e2b'] / 1000:.2f} 秒</td></tr>
        <tr><td>最快極限（狀況已在快取）</td><td class="num">{SPD['floor']['26b'] / 1000:.3f} 秒</td><td class="num">{SPD['floor']['e4b'] / 1000:.3f} 秒</td><td class="num">{SPD['floor']['e2b'] / 1000:.3f} 秒</td></tr>
        <tr><td>模型佔的記憶體</td><td class="num">17 GB</td><td class="num">8 GB</td><td class="num">5 GB</td></tr>
      </tbody>
    </table>
  </div>
  <p>E4B 單題快 2.2 倍、十題快 1.8 倍，但最快極限兩者一樣（26B 是「每次只動 4B」的混合專家架構），所以小模型的好處主要在記憶體和讀題的速度，不是每個字的速度。<strong>值多少：</strong>級聯後準確率 {CAS['mean'][6] * 100:.1f}%（全用 26B 是 {CAS['mean'][1] * 100:.1f}%），平均每題從 {SPD['L1']['26b'] / 1000:.2f} 秒降到 {SPD['cascade'][2][3] / 1000:.2f} 秒（省兩成），26B 的判斷負載只剩三分之一。在 GB10 上 26B 同時還要寫報告、做 RCA，把三分之二的判斷流量移走，才是這個做法真正的價值；純速度上的收益有限。四類簡單題若只用 E4B，8 GB 記憶體、單題 0.06 秒，可以放在更小的機器上。</p>

  <p class="eyebrow">02e · 換一個推理引擎值不值</p>
  <h2>SGLang 快三到六倍，掉分的原因是一個字元，補上後回到候選</h2>
  <p>現有部署用的是 llama-server。另一個常見的推理引擎 SGLang 有一項專長：多個問題共用同一段前文時，只算一次。這正是「同一份現場狀況問十題」的形狀，所以值得試。做法是同一張顯示卡（租用 L40S）、一字不差的提問、跑之前先寫死四道速度門檻與一道準確率護欄，然後兩邊各用自己最好的設定跑。</p>

  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>一次判斷（p50）</th><th class="num">llama-server</th><th class="num">SGLang</th><th>結果</th></tr></thead>
      <tbody>
        <tr><td>單題，每次不同的現場狀況</td><td class="num">{v4p('A1', 'L1_single_100tok_2opt') / 1000:.3f} 秒</td><td class="num">{v4p('B1', 'L1_single_100tok_2opt') / 1000:.3f} 秒</td><td>持平</td></tr>
        <tr><td>同一份狀況（約 1,100 字）問十六題</td><td class="num">{v4p('A1', 'L4_shared_16q') / 1000:.2f} 秒</td><td class="num">{v4p('B1', 'L4_shared_16q') / 1000:.2f} 秒</td><td><strong>SGLang 快 {V4G['L4_speedup_16']:.1f} 倍</strong></td></tr>
        <tr><td>三十二路同時問，每秒判斷數</td><td class="num">{V4S['A1']['L5_concurrency_32']['summary']['throughput_rps']:.0f}</td><td class="num">{V4S['B1']['L5_concurrency_32']['summary']['throughput_rps']:.0f}</td><td><strong>SGLang 快 {V4S['B1']['L5_concurrency_32']['summary']['throughput_rps'] / V4S['A1']['L5_concurrency_32']['summary']['throughput_rps']:.1f} 倍</strong></td></tr>
        <tr><td>一千題整批跑完</td><td class="num">{V4S['A1']['S3_d0_test_1001']['wall_s']:.0f} 秒</td><td class="num">{V4S['B1']['S3_d0_test_1001']['wall_s']:.0f} 秒</td><td><strong>SGLang 快 {V4S['A1']['S3_d0_test_1001']['wall_s'] / V4S['B1']['S3_d0_test_1001']['wall_s']:.1f} 倍</strong></td></tr>
        <tr><td>十類題平均答對率（同一批一千題）</td><td class="num">{v4_acc_A1 * 100:.1f}%</td><td class="num">{v4_acc_B1 * 100:.1f}%</td><td><strong>SGLang 低 {(v4_acc_A1 - v4_acc_B1) * 100:.1f} 個百分點</strong>，護欄 {v4_guard_n} 項只過 {v4_guard_pass} 項</td></tr>
        <tr><td>同一題重跑，答案一樣的比例</td><td class="num">≈ 100%</td><td class="num">{V4B['agree']['B1~w1'] * 100:.0f}%</td><td>SGLang 翻掉的題裡六成原本很有把握</td></tr>
      </tbody>
    </table>
  </div>

  <p><strong>第一輪的排除實驗。</strong>三個排除實驗：換成未壓縮的原版權重、同一張 H100，SGLang 仍低 {(V4B['A2_mean'] - V4B['B2_mean']) * 100:.1f} 個百分點（所以不是壓縮格式的錯）；改成一次只送一題、或關掉它的前綴快取，答對率都不變（所以不是併發或快取的錯）。剩下的解釋是引擎本身的數值路徑不同。更麻煩的是同一題跑三次答案兩兩只有 {V4B['agree']['B1~w1'] * 100:.0f}% 一致，翻掉的題有六成原本給了九成以上的把握，<span class="mark">「有把握才自動處理」的門檻在 SGLang 上會不穩</span>。</p>

  <p><strong>追查結果（第二輪）：</strong>上面的掉分不是引擎本身的問題。把現行引擎切好的字元編號直接餵給 SGLang，答對率回到 {V6['E1']['mean']['B1ids'] * 100:.1f}%（現行 {V6['E1']['mean']['A1'] * 100:.1f}%）。兩邊唯一的差別在第一個字元：現行引擎會在題目最前面加一個「開始」符號，SGLang 用的分詞器對 Gemma 4 預設不加。少這一個符號，十類題平均掉 2 到 3 個百分點。修正只有一行，已經放進程式，並用煙霧測試確認兩邊字元數一致。同一題重跑會變的問題也跟著好了一大半：補上符號並開啟批次不變模式後，兩千題只有三題重跑答案不同，代價是單題慢三成、併發少四分之一。</p>

  <p><strong>判定（更新）：</strong>SGLang 回到候選。它的批次速度是真的，準確率問題已解，穩定性接近現行引擎但還差一點。<span class="mark">即時的派工、升級、通知判斷，兩個引擎在 GB10 上各量一次再選</span>；一次幾千題的事後工作（會議記錄整理、歷史工單重標）直接交給 SGLang，快五到六倍。附帶發現：SGLang 每一次請求有約 0.06 秒的固定開銷，單題不會比 llama-server 快，它的優勢全在批次。這一段的教訓比結論重要：第一輪排除了三個原因就寫「引擎固有」，第二輪多花一美元找到真正原因，是一個字元。</p>

  <p><strong>附帶一筆：</strong>另一個開源專案 TypeLLM（用 Qwen 模型做同一件事）有兩個技巧我們也試了。把答案包成 JSON 形狀再讓模型填字母，弱題只升約一個百分點，不到門檻，不採用；把選項順序打亂多問幾次再平均，答對率沒有變好，但量到一個該知道的風險：<span class="mark">急迫度與 SPC 這兩類弱題，有兩成的題目換個選項順序答案就變</span>。上線時選項順序要固定，校準也要用同一順序。細節在 results/08-typellm-followups.md。</p>

  <p class="eyebrow">03 · 標註量</p>
  <h2>「幾百筆」的承諾，實測是「幾十筆或零筆」</h2>
  <p>主管版報告承諾 Jev 式做法能把標註需求從每題幾千筆降到幾百筆。下圖是最弱的一類題目（SPC 管制圖處置）：傳統機器學習標到 100 筆還在 {lc['mean']['tfidf']['100'] * 100:.0f}%，Jev 式零筆就 {lc['mean']['typed0']['25'] * 100:.0f}%，標 25 筆校準後在它敢答的八成題目上有 {lc['mean']['typed_sel90']['25'] * 100:.0f}%。</p>

  <figure class="reveal">
    <p class="title">SPC 處置這一題：標註筆數對答對率</p>
    <p class="subtitle">三個做法共用一條 y 軸。第三條是校準後只回答有把握的題，它敢答的比例在表裡。</p>
    <div class="chart">
{lc_svg}
    </div>
    <div class="legend">{lc_legend}</div>
    <details class="datatable">
      <summary>看數字</summary>
      <div class="table-scroll">
        <table>
          <thead><tr><th>標註</th><th class="num">傳統 ML</th><th class="num">Jev 式零筆</th><th class="num">Jev 式校準後</th><th class="num">敢答的比例</th></tr></thead>
          <tbody>{lc_table}</tbody>
        </table>
      </div>
    </details>
    <figcaption>資料集每類只有 200 題，所以標註筆數最多量到 100。</figcaption>
  </figure>

  <p class="eyebrow">04 · 八個問題</p>
  <h2>一題一題講：問題是什麼，答案是什麼</h2>
  <p>實驗設計時列了八個要回答的問題。<span class="chip">已答</span> 是在合成資料上完整回答、與硬體無關的；<span class="chip soft">上界</span> 是速度類，租用的顯示卡記憶體頻寬與 GB10 相近但算力較弱，只能當保守估計；<span class="chip soft">方法成立</span> 是門檻類，做法對了，數值要用真實資料重做。</p>

  <div class="qlist">{q_blocks}</div>

  <p class="eyebrow">05 · 怎麼分工</p>
  <h2>哪些判斷交給它，哪些不交</h2>

  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>判斷的長相</th><th>交給誰</th><th>依據</th></tr></thead>
      <tbody>
        <tr><td>答案是固定幾個選項、每天百次以上（派工、分類、要不要通知）</td><td class="nowrap"><span class="verdict">JEV 式 · 直接上</span></td><td>七類題目 98% 以上，且說有把握時 99% 以上答對</td></tr>
        <tr><td>選項多到十種以上的細分類</td><td class="nowrap"><span class="verdict">JEV 式 · 不必拆題</span></td><td>十種相近意圖 100% 答對，多選項幾乎不影響速度</td></tr>
        <tr><td>模糊、需要「以上皆非」、要能轉人</td><td class="nowrap"><span class="verdict">JEV 式 · 校準後才上</span></td><td>門檻要用真實資料校準，否則模型永遠說有把握</td></tr>
        <tr><td>純數字的判斷（產能、CT、SPC 是否超限）</td><td class="nowrap"><span class="verdict">程式公式</span></td><td>讓模型讀數字序列只有 {uph['raw_test']['acc'] * 100:.0f}%，公式是 100%</td></tr>
        <tr><td>關鍵字就能認的</td><td class="nowrap"><span class="verdict">規則 · 只當前置過濾</span></td><td>規則最好的一類 {best_rule['rules_acc_test'] * 100:.0f}%，不能單獨用</td></tr>
        <tr><td>要寫出原因、要多步推理、要查資料</td><td class="nowrap"><span class="verdict">大模型生成</span></td><td>本質上要文字；由 Jev 式判斷「沒把握」時觸發</td></tr>
      </tbody>
    </table>
  </div>

  <p class="eyebrow">06 · 上線條件</p>
  <h2>值得，前提是做到這四件事</h2>

  <div class="cols">
    <div class="card"><p class="verdict">條件一 · 部署設定</p><p>Gemma 4 的注意力機制讓伺服器預設無法重用「同一份現場狀況」的計算。要開一個設定（<code>--swa-full</code>），否則一次問十題就是十倍時間。GB10 記憶體夠，開了沒有代價。</p></div>
    <div class="card"><p class="verdict">條件二 · 真實資料校準</p><p>門檻的形式已經定了：先給可接受的錯誤率，程式反推「多有把握才自動處理」。合成資料上給 5% 錯誤預算，七類強題能自動處理 91% 到 100%、實際錯 0% 到 3%；門檻寫成鎖定檔，換模型或換引擎時自動重測，換錯了會被擋下。真實工單上要用 200–500 筆重算一次，估計一到兩天。</p></div>
    <div class="card"><p class="verdict">條件三 · 選項順序固定</p><p>急迫度與 SPC 這兩類弱題，有兩成的題目換個選項順序答案就變。上線時每類題的選項順序寫死，校準也用同一順序，否則「有把握」的門檻會飄。</p></div>
    <div class="card"><p class="verdict">條件四 · GB10 實測</p><p>速度數字全部來自租用的 L4。GB10 上重跑同一套測試約半小時，可以得到真實的秒數，以及機台同時在寫報告時的併發表現，再決定要不要另放一個小模型專做判斷。</p></div>
  </div>

  <p class="pull">不買 Jev，不換模型，不訓練。加一層程式，七類判斷今天就能用，三類再調一調。</p>

  <p class="eyebrow">07 · 下一步</p>
  <h2>建議的三個動作</h2>
  <ol>
    <li><strong>GB10 上重跑速度測試</strong>（半小時）：確認每次判斷的真實秒數與併發能力；順便量 E4B，四類簡單題可交給它。</li>
    <li><strong>收 200–500 筆真實警報與工單</strong>（一到兩天）：校準門檻，確認零標註在真實資料上的答對率。</li>
    <li><strong>推理引擎</strong>：SGLang 掉分原因已找到並修正；GB10 上兩個引擎各量一次即時判斷的併發，再決定即時層用哪個。</li>
    <li><strong>改寫兩類題目的判斷標準</strong>：已做。SPC 處置 {A['q_spc_action']['raw_test']['acc'] * 100:.0f}% → {V2['q_spc_action']['raw_test']['acc'] * 100:.0f}%，警報急迫度 {A['m_alarm_severity']['raw_test']['acc'] * 100:.0f}% → {V2['m_alarm_severity']['raw_test']['acc'] * 100:.0f}%。急迫度剩下的錯集中在「AOI 一片誤判」這類，等真實資料再調一輪。</li>
  </ol>

  <p class="eyebrow">08 · 參考</p>
  <h2>別人怎麼做、我們借鏡了什麼</h2>
  <p>Jev 在 2026 年 9 月 15 日發表後兩週，開源社群長出三十幾個替代品。我們把它們分成四類，逐一看做法、自報數字，挑出能直接用的抄回來試，試完照門檻留或不留。完整調查在 results/10-landscape.md。</p>

  <h2>先看數字：和 Jev、和別人跑同一顆模型的對照</h2>
  <p>外面有兩個可以對照的來源：Jev 官方與獨立評測（JevBench）公布的數字，以及開源專案 gemma-jev 用同一顆 Gemma 4 26B 在自家顯示卡上量到的數字。資料集、硬體、題目都不同，只能看量級，不能排名。</p>

  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>誰</th><th>硬體</th><th class="num">一次判斷</th><th>準確率</th><th>可信度</th></tr></thead>
      <tbody>
        <tr><td>Jev（TypeSafe 雲端服務）</td><td>雲端，經網路</td><td class="num">0.24–0.35 秒</td><td>獨立評測 JevBench 第 1 名（74.4 分／100）；官方自家題約 68%</td><td>公開評測，534 題</td></tr>
        <tr><td>gemma-jev（開源，同一顆 26B 模型）</td><td>RTX 3090 桌機顯示卡</td><td class="num">0.05 秒</td><td>意圖分類 87%、防注入 100%、自家控制題 100%</td><td>樣本只有 12–30 題，看方向</td></tr>
        <tr><td>同一顆 26B 模型（他人提交）</td><td>—</td><td class="num">—</td><td>JevBench 66.4 分，落後 Jev 8 分</td><td>公開評測，刻意刁難的題</td></tr>
        <tr><td><strong>我們</strong>（同一顆 26B 模型，跑 JevBench 公開 231 題）</td><td>租用 L4</td><td class="num">—</td><td>{JB26['accuracy'] * 100:.1f}%（難題 {JB26['per_tier']['hard'] * 100:.1f}%）；小模型 E4B {JBE4['accuracy'] * 100:.1f}%</td><td>自跑、不排名；同一子集 Cygnet 87.9%、Open-Jev 85.3%</td></tr>
        <tr><td><strong>我們</strong>（同一顆 26B 模型）</td><td>租用 L4；GB10 待測</td><td class="num">{L1['p50'] / 1000:.1f} 秒（共用狀況時 {per_q_swa / 1000:.2f} 秒）</td><td>十類產線題平均 {sum(A[t]['raw_test']['acc'] for t in ORDER) / 10 * 100:.0f}%，七類 ≥ 98%，難題平均 {sum(A[t]['by']['hard']['acc'] for t in ORDER) / 10 * 100:.0f}%</td><td>每類 200 題、獨立抽查；但是合成資料</td></tr>
      </tbody>
    </table>
  </div>

  <p>三個結論：第一，<span class="mark">同一顆模型在別人手上也一樣準</span>，不是我們的題目太簡單才看到高分。第二，速度差在顯示卡：3090 的記憶體頻寬是 L4 的三倍，同一顆模型量到 0.05 秒 vs 我們的 0.2 秒；GB10 的頻寬與 L4 同級，實際會落在 0.1–0.2 秒之間，靠一次問多題才會壓到 0.07 秒。第三，在刻意刁難的公開評測上，這顆模型的官方提交落後 Jev 8 分；但我們自己用同一套讀法跑公開 231 題拿到 {JB26['accuracy'] * 100:.1f}%，高於同做法的 Cygnet 與訓練過的 Open-Jev，說明產線題的高分不是題目量身訂做。真實資料上的分數仍要預期比合成資料低。</p>


  <h2>我們在公開考卷上的位置</h2>
  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>系統</th><th>做法</th><th class="num">公開 231 題</th><th class="num">難題 111 題</th></tr></thead>
      <tbody>
        <tr><td><strong>我們 · Gemma 4 26B-A4B</strong></td><td>凍結，讀字母，不訓練</td><td class="num"><strong>{JB26['accuracy'] * 100:.1f}%</strong></td><td class="num">{JB26['per_tier']['hard'] * 100:.1f}%</td></tr>
        <tr><td>Cygnet · Gemma 4 12B</td><td>凍結，讀字母，一個溫度</td><td class="num">87.9%</td><td class="num">76.6%</td></tr>
        <tr><td>Open-Jev · Qwen 27B</td><td>LoRA + 決策頭，14.9 萬筆訓練</td><td class="num">85.3%</td><td class="num">72.1%</td></tr>
        <tr><td>TypeLLM · Qwen 27B</td><td>凍結，不開思考</td><td class="num">84.4%</td><td class="num">—</td></tr>
        <tr><td>我們 · Gemma 4 E4B</td><td>凍結，讀字母</td><td class="num">{JBE4['accuracy'] * 100:.1f}%</td><td class="num">{JBE4['per_tier']['hard'] * 100:.1f}%</td></tr>
      </tbody>
    </table>
  </div>
  <p>都是各自在同一份公開子集上自跑的數字，不是官方排名（官方要跑 842 題含密封題）。我們沒有用這些題調任何參數；把合成產線題上擬合的溫度直接套上去，校準誤差從 0.093 降到 0.044。差距全在難題，最弱的是時間與數字推算，和產線題「產能該用公式判斷」的結論一致。</p>

  <h2>四類做法</h2>
  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>做法</th><th>代表專案</th><th>怎麼做</th><th>自報成績</th><th>對我們的意義</th></tr></thead>
      <tbody>
        <tr><td><strong>零樣本讀選項機率</strong>（我們這一派）</td><td>Cygnet（Gemma-4-12B）、open-alternative-jev、SemIf、openjev-sglang、verdict、gemma-jev</td><td>模型不改、不訓練，題目排成字母選項，讀下一個字的機率</td><td>Cygnet 公開評測第 4 名、公開題 87.9%，一個溫度就校準</td><td>同一條路在公開評測站得住；Gemma 4 這一派只有 Cygnet 和我們</td></tr>
        <tr><td><strong>小規模訓練</strong>（LoRA + 決策頭）</td><td>decider-4b、JevK5、Open-Jev、Kev、imajev</td><td>4B 到 27B 模型加幾千到十幾萬筆決策題訓練</td><td>decider-4b 8,000 筆訓練後榜首 64.1，贏 Jev 的 63.3</td><td>真實工單到手後的下一步：讓 26B 開思考出題標答，教 4B 一次讀出</td></tr>
        <tr><td><strong>小型非自迴歸模型</strong></td><td>Laya（421M）、von、poorjev 的 NLI 模型</td><td>一個前向對所有選項打分，CPU 跑得動</td><td>Laya 零樣本在我們的產線題接近亂猜</td><td>要標註資料才成立，CPU 是它唯一勝過我們的地方</td></tr>
        <tr><td><strong>校準與門檻工具</strong></td><td>poorjev、jevcal、jevkit</td><td>錯誤預算反推門檻、每題門檻鎖定檔、上線後漂移檢查</td><td>poorjev 校準誤差 0.170 → 0.071</td><td>把「有把握才自動處理」從報告數字變成上線機制</td></tr>
      </tbody>
    </table>
  </div>

  <h2>借鏡了什麼，結果如何</h2>
  <div class="table-scroll wide route">
    <table>
      <thead><tr><th>抄來的做法</th><th>來源</th><th>我們量到</th><th>留或不留</th></tr></thead>
      <tbody>
        <tr><td>錯誤預算反推門檻，門檻寫成鎖定檔，換模型自動重測</td><td>poorjev、jevcal</td><td>5% 預算下強題自動處理 {min(cov5) * 100:.0f}%–{max(cov5) * 100:.0f}%；重測抓到換錯引擎的 10 項偏移</td><td><span class="verdict">留</span></td></tr>
        <tr><td>模板健康檢查（字母拿到的總機率）</td><td>verdict</td><td>壞模板 0.003，好模板 1.0；抓不到起始字元那種錯</td><td><span class="verdict">留</span></td></tr>
        <tr><td>字母先過分詞器驗證、獨立驗證腳本</td><td>TypeLLM</td><td>237 個結果檔重算 0 筆不一致</td><td><span class="verdict">留</span></td></tr>
        <tr><td>指定字元編號讀機率</td><td>TypeLLM、openjev-sglang</td><td>與原做法差 0.0</td><td><span class="verdict">留</span></td></tr>
        <tr><td>答案包成 JSON 形狀</td><td>TypeLLM</td><td>弱題 +1 個百分點，不到門檻</td><td>不留</td></tr>
        <tr><td>選項順序打亂多次再平均</td><td>TypeLLM、open-alternative-jev</td><td>答對率沒變；量到弱題兩成受順序影響</td><td>不留，風險寫進上線條件</td></tr>
        <tr><td>多題排成一列、一個前向讀完</td><td>open-alternative-jev</td><td>排在後面的題答案會變 11–13%，急迫度掉 14 個百分點；只快 1.3–1.7 倍</td><td>不留</td></tr>
      </tbody>
    </table>
  </div>

  <h2>我們比別人多做的</h2>
  <ul>
    <li><strong>Gemma 4 的正確用法。</strong>TypeLLM 明說不支援 Gemma 4；多數專案綁 Qwen。我們解掉的兩個坑（思考通道要留空、起始字元要補）沒有任何專案寫到，用 HF 分詞器的引擎接 Gemma 4 都可能踩到。</li>
    <li><strong>分得出題目簡單還是模型強。</strong>別人只報一個分數；我們拿三個大小的模型跑同一批，知道四類 2B 就夠、六類要 26B。</li>
    <li><strong>兩個引擎在同一批題上對照。</strong>其他人各用一個引擎；我們因此找到一個字元值 2 到 3 個百分點。</li>
    <li><strong>每一輪先寫門檻再跑，沒過的也公開。</strong>這一派的自報數字幾乎都沒有 held-out 驗證、獨立抽查與預先登記。</li>
  </ul>

  <div class="note">
    <p class="head">這一版沒說到的</p>
    <p>Jev 官方另一個賣點是「校準過的把握度」，我們的替代方案要自己校準，這是條件二。獨立的小模型（Laya 之類）零標註接近亂猜，這次沒用。AI Studio 上的 Gemma 4 不提供選項機率，雲端對照因此沒做。實際 GPU 費用七輪合計約 8.6 小時、約十五美元（首輪 0.85 小時、約一美元），比原估低很多。</p>
  </div>

  <p class="byline">repo clarencechien/jevlike · 工程版報告 results/REPORT.md · 原始數據 results/analysis.json · results/modal/</p>

</main>

<script>
(() => {{
  const root = document.documentElement, btn = document.getElementById('theme-toggle');
  let dark = matchMedia('(prefers-color-scheme: dark)').matches;
  btn.textContent = 'theme · ' + (dark ? 'dark' : 'light');
  btn.addEventListener('click', () => {{
    dark = !dark; root.dataset.theme = dark ? 'dark' : 'light';
    btn.textContent = 'theme · ' + (dark ? 'dark' : 'light'); btn.setAttribute('aria-pressed', 'true');
  }});
}})();
</script>
</body>
</html>
"""
out = os.path.join(ROOT, "results/report.html")
open(out, "w", encoding="utf-8").write(page)
print("wrote", out, len(page.encode()), "bytes")
