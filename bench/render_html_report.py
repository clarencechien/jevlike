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

  <p class="eyebrow">判定備忘 · Jev 式決策層 · 2026-09-23</p>
  <h1 class="display">Jev 式決策，<br>值得做，<br>而且<em>不用買</em></h1>
  <p class="lede">用我們已經部署的 Gemma 4 26B，加一層「只從固定選項裡選答案、不寫文章」的決策 API。十類產線判斷題有七類不用任何標註就能上線，每次判斷約 0.2 秒。本次驗證花費約一美元的雲端 GPU。</p>
  <p class="byline">實驗於 Modal 雲端 L4 顯示卡進行 · 合成資料 10 類 × 200 題 · 工程細節與原始數據在 results/REPORT.md</p>

  <div class="tldr">
    <p class="head">TL;DR</p>
    <ol>
      <li><strong>判定：值得做，用現有模型自己做，不採購 Jev。</strong>Jev 是閉源雲端服務，資料出不了廠；我們要的能力，現有的 26B 模型加幾十行程式就有。</li>
      <li><strong>準確率：十類題目七類直接達標。</strong>不做任何標註，{len(strong)} 類答對率 98% 以上；其餘三類 {min(A[t]['raw_test']['acc'] for t in weak) * 100:.0f}–{max(A[t]['raw_test']['acc'] for t in weak) * 100:.0f}%，錯在相鄰等級的邊界。<strong>把其中兩類的判斷標準改寫成可數規則後，SPC 處置升到 {V2['q_spc_action']['raw_test']['acc'] * 100:.0f}%、警報急迫度升到 {V2['m_alarm_severity']['raw_test']['acc'] * 100:.0f}%</strong>，在模型沒看過的新題上也成立。不需要訓練新模型。</li>
      <li><strong>速度：一次判斷 {L1['p50'] / 1000:.1f} 秒，比讓模型寫答案快 {speed:.1f} 倍。</strong>同一份現場狀況一次問十題，開對設定後每題再降到 {per_q_swa / 1000:.2f} 秒。</li>
      <li><strong>標註：比主管報告承諾的還少。</strong>多數題目零筆；較弱的題目標 25 筆就能校準。傳統機器學習標 100 筆還到不了 90%。</li>
      <li><strong>三個上線條件：</strong>部署時開啟前綴快取設定、「有把握才自動處理」的門檻要用真實資料校準、GB10 上重量一次速度。都是幾小時到幾天的事，不是幾個月。</li>
    </ol>
  </div>

  <div class="tiles wide stagger">
    <div class="stat"><p class="label">不標註就達 98% 的題型</p><div class="value">{len(strong)} / 10</div><div class="delta">其餘三類要調門檻</div></div>
    <div class="stat"><p class="label">一次判斷</p><div class="value">{L1['p50'] / 1000:.1f} 秒</div><div class="delta">租用 L4；GB10 待實測</div></div>
    <div class="stat"><p class="label">比模型寫答案快</p><div class="value">{speed:.1f} 倍</div><div class="delta">一次問十題可再降</div></div>
    <div class="stat"><p class="label">本次驗證的 GPU 費用</p><div class="value">≈ $1</div><div class="delta">約 0.85 小時</div></div>
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
  <h2>值得，前提是做到這三件事</h2>

  <div class="cols">
    <div class="card"><p class="verdict">條件一 · 部署設定</p><p>Gemma 4 的注意力機制讓伺服器預設無法重用「同一份現場狀況」的計算。要開一個設定（<code>--swa-full</code>），否則一次問十題就是十倍時間。GB10 記憶體夠，開了沒有代價。</p></div>
    <div class="card"><p class="verdict">條件二 · 真實資料校準</p><p>這次全部是合成資料。零標註的答對率在真實工單上預期會降一些，「有把握才自動處理」的門檻必須用 200–500 筆真實警報與工單重新校準，估計一到兩天。</p></div>
    <div class="card"><p class="verdict">條件三 · GB10 實測</p><p>速度數字全部來自租用的 L4。GB10 上重跑同一套測試約半小時，可以得到真實的秒數，以及機台同時在寫報告時的併發表現，再決定要不要另放一個小模型專做判斷。</p></div>
  </div>

  <p class="pull">不買 Jev，不換模型，不訓練。加一層程式，七類判斷今天就能用，三類再調一調。</p>

  <p class="eyebrow">07 · 下一步</p>
  <h2>建議的三個動作</h2>
  <ol>
    <li><strong>GB10 上重跑速度測試</strong>（半小時）：確認每次判斷的真實秒數與併發能力。</li>
    <li><strong>收 200–500 筆真實警報與工單</strong>（一到兩天）：校準門檻，確認零標註在真實資料上的答對率。</li>
    <li><strong>改寫兩類題目的判斷標準</strong>：已做。SPC 處置 {A['q_spc_action']['raw_test']['acc'] * 100:.0f}% → {V2['q_spc_action']['raw_test']['acc'] * 100:.0f}%，警報急迫度 {A['m_alarm_severity']['raw_test']['acc'] * 100:.0f}% → {V2['m_alarm_severity']['raw_test']['acc'] * 100:.0f}%。急迫度剩下的錯集中在「AOI 一片誤判」這類，等真實資料再調一輪。</li>
  </ol>

  <div class="note">
    <p class="head">這一版沒說到的</p>
    <p>Jev 官方另一個賣點是「校準過的把握度」，我們的替代方案要自己校準，這是條件二。獨立的小模型（Laya 之類）零標註接近亂猜，這次沒用。AI Studio 上的 Gemma 4 不提供選項機率，雲端對照因此沒做。實際 GPU 費用約 0.85 小時、約一美元，比原估的 4–6 小時低很多。</p>
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
