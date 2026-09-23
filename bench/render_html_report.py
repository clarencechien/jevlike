"""Build results/report.html — a self-contained imitator-style (v3) report of what the
experiment has answered so far. Numbers come from results/analysis.json and
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
TASKS = json.load(open(os.path.join(ROOT, "data/seeds/tasks.json"), encoding="utf-8"))
ORDER = ["x_ticket_route", "x_escalate", "x_10way_intent", "m_needs_dispatch", "p_line_change", "m_alarm_category",
         "q_defect_root", "p_uph_anomaly", "q_spc_action", "m_alarm_severity"]
ZH = {"m_alarm_severity": "alarm 急迫度", "m_alarm_category": "alarm 分類", "m_needs_dispatch": "是否派工",
      "q_spc_action": "SPC 動作", "q_defect_root": "不良根因站別", "p_uph_anomaly": "UPH 是否異常",
      "p_line_change": "工單狀態", "x_ticket_route": "工單派給誰", "x_escalate": "是否升級線長", "x_10way_intent": "10 類意圖"}


def e(s):
    return html.escape(str(s))


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


# ------------------------------------------------------------------ charts
def acc_bar_chart():
    rows = [(t, A[t]["raw_test"]["acc"]) for t in ORDER]
    h = 26 * len(rows) + 30
    x0, x1 = 150, 560
    out = [f'<svg viewBox="0 0 640 {h}" role="img" aria-label="十個 task 在 test 集的零樣本準確率，L4 上的 Gemma 4 26B-A4B">']
    for i, g in enumerate((0.5, 0.75, 1.0)):
        x = x0 + (x1 - x0) * g
        out.append(f'<line class="grid" x1="{x:.0f}" y1="8" x2="{x:.0f}" y2="{h - 22}"/>')
        out.append(f'<text class="axis" x="{x:.0f}" y="{h - 6}" text-anchor="middle">{pct(g)}</text>')
    for i, (t, v) in enumerate(rows):
        y = 12 + 26 * i
        w = (x1 - x0) * v
        out.append(f'<text class="axis" x="{x0 - 8}" y="{y + 14}" text-anchor="end">{e(ZH[t])}</text>')
        out.append(f'<rect class="bar" x="{x0}" y="{y}" width="{w:.1f}" height="18" fill="var(--c1)"/>')
        out.append(f'<text class="label" x="{x0 + w + 6:.1f}" y="{y + 14}">{v:.2f}</text>')
    out.append("</svg>")
    table = "".join(f'<tr><td>{e(ZH[t])}<span class="sub">{t}</span></td><td class="num">{A[t]["raw_test"]["acc"]:.2f}</td>'
                    f'<td class="num">{A[t]["by"]["hard"]["acc"]:.2f}</td><td class="num">{A[t]["raw_test"]["ece"]:.3f}</td>'
                    f'<td class="num">{A[t].get("control", {}).get("acc", float("nan")):.2f}</td><td class="num">{A[t]["tfidf_acc_test"]:.2f}</td>'
                    f'<td class="num">{A[t].get("rules_acc_test", float("nan")):.2f}</td></tr>' for t in ORDER)
    return "\n".join(out), table


def shared_state_chart():
    ns = (1, 5, 10)
    d = [LAT[f"L4_shared_state_{n}q_cache_on"]["summary"]["p50"] for n in ns]
    s = [SWA[f"L4_shared_state_{n}q_cache_on"]["summary"]["p50"] for n in ns]
    top = 2500
    x0, x1, y0, y1 = 60, 600, 20, 240
    out = ['<svg viewBox="0 0 640 280" role="img" aria-label="共用一份 state 問 1、5、10 題的總時間，預設設定與開 swa-full 的比較">']
    for g in (0, 500, 1000, 1500, 2000, 2500):
        y = y1 - (y1 - y0) * g / top
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{g}</text>')
    gw = (x1 - x0) / len(ns)
    for i, n in enumerate(ns):
        cx = x0 + gw * (i + 0.5)
        for j, (v, col) in enumerate(((d[i], "--c1"), (s[i], "--c2"))):
            bx = cx - 26 + j * 28
            by = y1 - (y1 - y0) * v / top
            out.append(f'<rect class="bar" x="{bx:.1f}" y="{by:.1f}" width="24" height="{y1 - by:.1f}" fill="var({col})"/>')
            out.append(f'<text class="label" x="{bx + 12:.1f}" y="{by - 5:.1f}" text-anchor="middle">{v:.0f}</text>')
        out.append(f'<text class="axis" x="{cx:.1f}" y="{y1 + 20}" text-anchor="middle">{n} 題</text>')
    out.append(f'<text class="axis" x="{x0 - 8}" y="12" text-anchor="end">ms</text>')
    out.append("</svg>")
    table = "".join(f'<tr><td>{n} 題</td><td class="num">{d[i]:.0f}</td><td class="num">{LAT[f"L4_shared_state_{n}q_cache_on"]["per_call"]["p50"]:.0f}</td>'
                    f'<td class="num">{s[i]:.0f}</td><td class="num">{SWA[f"L4_shared_state_{n}q_cache_on"]["per_call"]["p50"]:.0f}</td>'
                    f'<td class="num">{d[i] / s[i]:.1f}x</td></tr>' for i, n in enumerate(ns))
    return "\n".join(out), table


def learning_curve_chart(task):
    lc = A[task]["learning_curve"]
    ns = lc["N"]
    series = [("TF-IDF + LR，用 N 筆訓練", "tfidf", "--c1"), ("typed decision 零樣本（N 筆不用）", "typed0", "--c2"),
              ("typed decision 用 N 筆校準後，只答信心 ≥ 0.9 的題", "typed_sel90", "--c3")]
    x0, x1, y0, y1 = 60, 600, 16, 240
    xs = {n: x0 + (x1 - x0) * i / (len(ns) - 1) for i, n in enumerate(ns)}
    out = [f'<svg viewBox="0 0 640 280" role="img" aria-label="{e(ZH[task])}：標註筆數對準確率的學習曲線，三個系列">']
    for g in (0.4, 0.6, 0.8, 1.0):
        y = y1 - (y1 - y0) * (g - 0.4) / 0.6
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{g:.1f}</text>')
    for n in ns:
        out.append(f'<text class="axis" x="{xs[n]:.1f}" y="{y1 + 20}" text-anchor="middle">N = {n}</text>')
    for label, key, col in series:
        pts = [(xs[n], y1 - (y1 - y0) * (lc["mean"][key][str(n)] - 0.4) / 0.6) for n in ns]
        out.append('<path class="line" d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts)) + f'" stroke="var({col})"/>')
        for x, y in pts:
            out.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var({col})"/>')
        lx, ly = pts[-1]
        out.append(f'<text class="label" x="{lx + 8:.1f}" y="{ly + 4:.1f}">{lc["mean"][key][str(ns[-1])]:.2f}</text>')
    out.append("</svg>")
    legend = "".join(f'<span class="key"><span class="swatch" style="background:var({col})"></span>{e(label)}</span>' for label, _, col in series)
    table = "".join(f'<tr><td>{n}</td><td class="num">{lc["mean"]["tfidf"][str(n)]:.2f}</td><td class="num">{lc["mean"]["typed0"][str(n)]:.2f}</td>'
                    f'<td class="num">{lc["mean"]["typed_sel90"][str(n)]:.2f}</td><td class="num">{lc["mean"]["typed_cov90"][str(n)]:.2f}</td></tr>' for n in ns)
    return "\n".join(out), legend, table


acc_svg, acc_table = acc_bar_chart()
ss_svg, ss_table = shared_state_chart()
lc_svg, lc_legend, lc_table = learning_curve_chart("q_spc_action")

L1 = LAT["L1_single_100tok_2opt"]["summary"]
L1s = SWA["L1_single_100tok_2opt"]["summary"]
floor = SWA["L1_ref_same_prompt_repeated"]["summary"]["p50"]
L7 = LAT["L7_chat_json"]["summary"]
L6 = LAT["L6_with_bg_generation"]
conc = {c: LAT[f"L5_concurrency_{c}"]["summary"] for c in (1, 4, 8)}
strong = [t for t in ORDER if A[t]["raw_test"]["acc"] >= 0.98]
weak = [t for t in ORDER if A[t]["raw_test"]["acc"] < 0.98]

# ------------------------------------------------------------------ question status
Q = [
    ("Q1", "GB10 上的決策延遲 p50 / p95，比生成 JSON 快幾倍", "上界",
     f"L4 上單題 p50 {L1['p50']:.0f} ms、p95 {L1['p95']:.0f} ms（開 <code>--swa-full</code> 後 {L1s['p50']:.0f} ms，decode 地板 {floor:.0f} ms）；同題生成 JSON {L7['p50']:.0f} ms，typed 快 {L7['p50'] / L1['p50']:.1f}–{L7['p50'] / L1s['p50']:.1f} 倍。",
     "GB10 重跑 L1–L7，約 30 分鐘"),
    ("Q2", "哪些 task 零樣本就夠、哪些要改題、哪些救不回", "已答",
     f"{len(strong)} 個 task test 準確率 ≥ 0.98；{len(weak)} 個（{'、'.join(ZH[t] for t in weak)}）0.82–0.93 但 top-2 / AUROC ≥ 0.99，是門檻與 criteria 問題；沒有救不回的。",
     "合成資料版；真實資料會低一些"),
    ("Q3", "多題共用 state 的成本", "已答",
     f"預設下 10 題 = {LAT['L4_shared_state_10q_cache_on']['summary']['p50'] / LAT['L4_shared_state_1q_cache_on']['summary']['p50']:.1f} 倍，前綴 cache 完全沒命中；開 <code>--swa-full</code> 後 10 題 = {SWA['L4_shared_state_10q_cache_on']['summary']['p50'] / SWA['L4_shared_state_1q_cache_on']['summary']['p50']:.1f} 倍，第 2 題起每題 {SWA['L4_shared_state_10q_cache_on']['per_call']['p50']:.0f} ms。",
     "倍數可信，絕對毫秒數以 GB10 為準"),
    ("Q4", "生成負載對決策延遲的影響", "上界",
     f"背景持續生成 512 token 時決策 p50 {L1['p50']:.0f} → {L6['summary']['p50']:.0f} ms（{L6['slowdown_p50']:.2f} 倍）；併發 4 / 8 client 的 p50 {conc[4]['p50']:.0f} / {conc[8]['p50']:.0f} ms，throughput 卡在 {conc[8]['throughput_rps']:.1f} req/s，llama-server 在 L4 上沒把 prefill 批次化。",
     "GB10 併發行為要重量，再決定要不要獨立 E4B"),
    ("Q5", "中文 vs 英文 alarm", "已答",
     f"中文夾機台代碼普遍不輸英文；英文 alarm log 在 4 個 task 略差（急迫度 {A['m_alarm_severity']['by']['en']['acc']:.2f} vs 中文 {A['m_alarm_severity']['by']['zh']['acc']:.2f}），主因是英文樣本刻意寫得極簡。不需要先正規化中文。",
     "—"),
    ("Q6", "每個 task 的信心門檻", "合成版",
     f"raw 信心平均 0.99，無鑑別力：急迫度在 raw ≥ 0.9 的 coverage {A['m_alarm_severity']['raw_test']['sel90']['coverage']:.2f} 但準確率只有 {A['m_alarm_severity']['raw_test']['sel90']['acc']:.2f}。校準後 ≥ 0.9 才可用：SPC 動作 {A['q_spc_action']['affine_test']['sel90']['acc']:.2f} @ coverage {A['q_spc_action']['affine_test']['sel90']['coverage']:.2f}，急迫度 1.00 @ 0.40。7 個強 task 用 raw ≥ 0.9 即可。",
     "用 200–500 筆真實 alarm／工單重做校準"),
    ("Q7", "標註量：傳統 ML 與 typed decision 各要幾筆到 90%", "已答",
     "TF-IDF + LR 在 100 筆內沒有一個 task 到 0.9（hard 子集 0.46–0.57）；typed decision 7 個 task 0 筆就 ≥ 0.98，弱的 3 個用 25 筆校準即可在 0.79–0.93 coverage 下達 0.95+。主管版「幾百筆」的說法成立，且多數 task 是 0 筆。",
     "資料集每 task 只有 200 筆，N > 100 量不到"),
    ("Q8", "規則基線在哪些 task 夠用", "已答",
     f"沒有 task 的關鍵字規則 ≥ 0.95；最高是工單狀態 {A['p_line_change']['rules_acc_test']:.2f}（easy 0.96）。不建議任何 task 只用規則，規則可當 easy 案例的前置過濾。",
     "—"),
]
q_blocks = "".join(
    f'<div class="q"><p class="verdict">{q} · {e(status)}</p><h3>{e(title)}</h3><p>{answer}</p>'
    f'<p class="byline">GB10 待補 · {e(todo)}</p></div>' for q, title, status, answer, todo in Q)

# ------------------------------------------------------------------ page
page = f"""<!doctype html>
<html lang="zh-Hant">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="imitator-style" content="v3">
<meta name="imitator-register" content="實驗紀錄 — 沒有 GB10，先在 L4 上把八個問題答到 80%">
<meta name="imitator-reference" content="1980 年代工廠 QC 課點陣印表機印在連續報表紙上的實驗紀錄 — 編號所見、判定欄、等寬欄位">
<meta name="imitator-paper" content="hsl(120 18% 94%)">
<meta name="imitator-accent" content="hsl(178 58% 26%)">
<title>八個問題，答了五個，三個給了上界</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
{CSS}
/* REGISTER:  實驗紀錄 — 沒有 GB10，先在 Modal 的 L4 上把 Jev 式 typed decision 的八個問題答到 80%；
              讀者是 Clarence 和要看分流建議的主管。這是「目前答到哪裡」的整理，不是最終報告。
   REFERENCE: 1980 年代工廠 QC 課的實驗紀錄 — 點陣印表機印在連續報表紙（greenbar）上，
              編號的所見、判定欄、等寬體的欄位，所見與判定分開寫。
   PAPER:     連續報表紙的淡綠 — hsl(120 18% 94%)。
   VOICE:     終端機青綠只給裁決、章節號和連結；標題用 IBM Plex Sans Condensed 壓 Noto Sans TC；
              欄位與狀態用等寬體。判定用 chip：已答 / 上界 / 合成版。
   NOT:       儀表板。數字是所見，每一個都跟著一句判定。
   RECENT:    paper 355° 50° 20° · accent 205° 216° 282° — 原本想用三聯單黃聯 52° + 紫章 275°，
              兩個都撞，改材料為 greenbar 120° + 終端機青綠 178°；178° 離 205° 有 27°。 */
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
/* the routing table carries prose: on a phone it scrolls with its first column pinned instead of squeezing */
.route table {{ min-width: 44rem; }}
</style>
<body>
<div class="progress"></div>
<button class="theme-toggle" id="theme-toggle" type="button" aria-pressed="false">theme · auto</button>

<main class="report">

  <p class="eyebrow">實驗紀錄 · GB10 前置 · 2026-09-23</p>
  <h1 class="display">八個問題，<br>答了五個，<br>三個給了<em>上界</em></h1>
  <p class="lede">在拿不到 GB10 的情況下，用 Modal 上一張 L4 跑 Gemma 4 26B-A4B，讀選項字母的第一個 token 機率當決策 API。這一頁整理目前每個問題答到哪裡、哪些數字可以直接用、哪些要等 GB10 重量。</p>
  <p class="byline">模型 gemma-4-26B-A4B-it-UD-Q4_K_M · llama-server b11118 · 合成資料 10 task × 200 筆 · GPU 約 0.85 小時 · 明細在 results/REPORT.md</p>

  <p class="eyebrow">所見 01 · 答到哪裡</p>
  <h2>八個問題的狀態</h2>
  <p>v1 交接文件列了八個實驗要回答的問題。<span class="chip">已答</span> 是在合成資料上完整回答、與 GPU 無關的；<span class="chip soft">上界</span> 是延遲類，L4 的頻寬接近 GB10 但算力較弱，數字只能當保守上界；<span class="chip soft">合成版</span> 是門檻類，方法對了但數值要用真實資料重做。</p>

  <div class="qlist">{q_blocks}</div>

  <p class="eyebrow">所見 02 · 準確率</p>
  <h2>七個 task 零樣本就過線，三個是門檻問題</h2>

  <div class="tiles wide stagger">
    <div class="stat"><p class="label">test 準確率 ≥ 0.98 的 task</p><div class="value">{len(strong)} / 10</div><div class="delta">零樣本、不校準</div></div>
    <div class="stat"><p class="label">最弱的 task</p><div class="value">{A['m_alarm_severity']['raw_test']['acc']:.2f}</div><div class="delta">alarm 急迫度，top-2 為 1.00</div></div>
    <div class="stat"><p class="label">raw 信心平均</p><div class="value">0.99</div><div class="delta down">幾乎永遠說有把握</div></div>
    <div class="stat"><p class="label">抽查不合理率</p><div class="value">2.5%</div><div class="delta">200 筆，0 筆錯、5 筆可辯</div></div>
  </div>

  <figure class="reveal">
    <p class="title">十個 task 的零樣本準確率（test 集，未校準）</p>
    <p class="subtitle">依 pair_id 分組切一半當 test；孿生例不跨集。同一模型生成 JSON 作答的對照、TF-IDF 與規則基線在表裡。</p>
    <div class="chart">
{acc_svg}
    </div>
    <details class="datatable">
      <summary>看數字</summary>
      <div class="table-scroll">
        <table>
          <thead><tr><th>task</th><th class="num">typed acc</th><th class="num">hard acc</th><th class="num">ECE raw</th><th class="num">生成 JSON</th><th class="num">TF-IDF+LR</th><th class="num">規則</th></tr></thead>
          <tbody>{acc_table}</tbody>
        </table>
      </div>
    </details>
    <figcaption>錯的集中在相鄰等級的邊界：急迫度的「盡快 vs 停線」、SPC 的「抽檢 vs 停線複檢」。這是 criteria 要寫成可數規則的問題，不是模型能力問題。</figcaption>
  </figure>

  <p>Smoke 時就看到的現象在兩千筆上重現：<span class="mark">模型幾乎永遠給 99% 的信心，對錯都一樣</span>。alarm 急迫度的 raw ECE 是 {A['m_alarm_severity']['raw_test']['ece']:.3f}，temperature 要拉到 T ≈ 6 才校準得回來。所以 Jev 式「低信心就升級 LLM」的分流，在這個模型上<strong>只能用校準後的信心</strong>，raw 信心當門檻等於沒有門檻。</p>

  <div class="note warn">
    <p class="head">判定：不需要 fine-tune</p>
    <p>三個弱 task 的 top-2 準確率與 AUROC 都 ≥ 0.99，排序能力沒問題。依 v1 的判讀順序，這是門檻與 criteria 的事：用 25–100 筆做 affine 校準，加上把「連續幾點、超限幾次、機台停多久」寫成可數的邊界，再看要不要動模型。</p>
  </div>

  <p class="eyebrow">所見 03 · 延遲</p>
  <h2>單題兩百毫秒，但前綴 cache 預設是關不掉的壞</h2>

  <p>L4 上一題約 {L1['prompt_tokens_mean']:.0f} 個 prompt token 的決策 p50 {L1['p50']:.0f} ms，同一題改讓模型生成 JSON 要 {L7['p50']:.0f} ms。這個倍數（{L7['p50'] / L1['p50']:.1f}x）在 GB10 上大致會保留；絕對值會變。</p>

  <p class="pull">Gemma 4 用 sliding-window attention，llama-server 預設的 SWA cache 只能整段命中、不能部分重用前綴——所以「多題共用 state」在預設下一毛錢都省不到。</p>

  <figure class="reveal">
    <p class="title">共用一份 300-token state，問 1 / 5 / 10 題的總時間</p>
    <p class="subtitle">兩個系列共用一條 y 軸（ms）。加 <code>--swa-full</code> 後第 2 題起只剩 decode 加問題段的 prefill。</p>
    <div class="chart">
{ss_svg}
    </div>
    <div class="legend">
      <span class="key"><span class="swatch" style="background:var(--c1)"></span>預設（cache_prompt 開或關都一樣）</span>
      <span class="key"><span class="swatch" style="background:var(--c2)"></span>加 --swa-full</span>
    </div>
    <details class="datatable">
      <summary>看數字</summary>
      <div class="table-scroll">
        <table>
          <thead><tr><th>題數</th><th class="num">預設 總 p50</th><th class="num">預設 每題</th><th class="num">swa-full 總 p50</th><th class="num">swa-full 每題</th><th class="num">省</th></tr></thead>
          <tbody>{ss_table}</tbody>
        </table>
      </div>
    </details>
    <figcaption>GB10 有 128 GB 統一記憶體，開 <code>--swa-full</code> 不是問題；不開的話 Q3 的答案就是「N 題 = N 倍」。</figcaption>
  </figure>

  <p>另外兩個延遲所見：背景持續生成 512 token 時決策慢 {L6['slowdown_p50']:.2f} 倍；併發 1 / 4 / 8 個 client 的 p50 是 {conc[1]['p50']:.0f} / {conc[4]['p50']:.0f} / {conc[8]['p50']:.0f} ms，throughput 停在 {conc[8]['throughput_rps']:.1f} req/s。llama-server 在 L4 上沒有把 prefill 批次化，併發只是排隊。這一項在 GB10 上要重量，才能決定要不要走路線 C（獨立一個 E4B 做高頻 gating）。</p>

  <p class="eyebrow">所見 04 · 標註量</p>
  <h2>「幾百筆」的說法成立，而且多數 task 是零筆</h2>

  <p>主管版報告承諾「標註需求從每題幾千筆降到幾百筆」。在三個 task 上從 calibration 集抽 N 筆、跑 3 個 seed，畫傳統 ML 和 typed decision 的學習曲線。下面是最弱的 SPC 動作；另外兩個 task 的 typed 零樣本本來就在 0.98 以上。</p>

  <figure class="reveal">
    <p class="title">SPC 動作：標註筆數 N 對 test 準確率</p>
    <p class="subtitle">三個系列共用一條 y 軸。第三條是校準後只回答信心 ≥ 0.9 的題，其 coverage 在表裡。</p>
    <div class="chart">
{lc_svg}
    </div>
    <div class="legend">{lc_legend}</div>
    <details class="datatable">
      <summary>看數字</summary>
      <div class="table-scroll">
        <table>
          <thead><tr><th>N</th><th class="num">TF-IDF+LR</th><th class="num">typed 零樣本</th><th class="num">typed 校準後 @0.9</th><th class="num">coverage</th></tr></thead>
          <tbody>{lc_table}</tbody>
        </table>
      </div>
    </details>
    <figcaption>資料集每 task 只有 200 筆，calibration 集 100 筆，所以 N 停在 100。傳統 ML 在這個範圍內沒有到 0.9。</figcaption>
  </figure>

  <p class="eyebrow">所見 05 · 分流</p>
  <h2>分流表可以先填了</h2>

  <div class="table-scroll wide route">
    <table>
      <thead><tr><th class="wrap">題目特徵</th><th class="wrap">走哪條</th><th class="wrap">實測依據</th></tr></thead>
      <tbody>
        <tr><td class="wrap">封閉 label、選項 ≤ 5、每天百次以上</td><td class="nowrap"><span class="verdict">TYPED · 零樣本</span></td><td class="wrap">7 個 task sel@0.9 ≥ 0.99、coverage ≥ 0.99</td></tr>
        <tr><td class="wrap">10 類以上細粒度分類</td><td class="nowrap"><span class="verdict">TYPED · 不必拆題</span></td><td class="wrap">10 類相近意圖 test 1.00、hard 0.98；多 8 個選項只多 40 ms</td></tr>
        <tr><td class="wrap">模糊、需要「以上皆非」、低信心升級</td><td class="nowrap"><span class="verdict">TYPED · 校準後才能 gate</span></td><td class="wrap">raw 信心無鑑別力；affine 校準後 ≥ 0.9 可用</td></tr>
        <tr><td class="wrap">中文夾機台代碼</td><td class="nowrap"><span class="verdict">TYPED · 不必正規化</span></td><td class="wrap">中文普遍不輸英文</td></tr>
        <tr><td class="wrap">關鍵字規則就能解</td><td class="nowrap"><span class="verdict">CODE · 只當前置過濾</span></td><td class="wrap">規則最高 0.885，沒有 task ≥ 0.95</td></tr>
        <tr><td class="wrap">數值進、數值出（UPH、CT、SPC）</td><td class="nowrap"><span class="verdict">CODE · 公式判定</span></td><td class="wrap">讀「數列文字」只有 0.93，hard 0.80</td></tr>
        <tr><td class="wrap">要說明為什麼、多步推理</td><td class="nowrap"><span class="verdict">LLM</span></td><td class="wrap">本質上要文字</td></tr>
      </tbody>
    </table>
  </div>

  <p class="eyebrow">所見 06 · 沒答的</p>
  <h2>這一版說不了的三件事</h2>

  <div class="cols">
    <div class="card"><p class="verdict">延遲絕對值</p><p>全部是 L4 上界。GB10 用 <code>bench/latency.py --only</code> 重跑 L1–L7 約 30 分鐘，記得開 <code>--swa-full</code>。</p></div>
    <div class="card"><p class="verdict">真實資料的門檻</p><p>校準用的是合成資料，easy 例子是模板展開、規律性高。零樣本數字在真實 alarm 上會降，Q6 的門檻要用 200–500 筆真實資料重做。</p></div>
    <div class="card"><p class="verdict">併發與 aarch64</p><p>L4 上 llama-server 不因併發加速，GB10 要重量；模型檔若不是 UD-Q4_K_M，準確率也要在 GB10 重跑一次，同格式的 raw logprobs 可直接進 analyze.py。</p></div>
  </div>

  <div class="note">
    <p class="head">還沒說到的</p>
    <p>AI Studio 上的 Gemma 4 不開放 logprobs，31B dense 的對照因此沒做；env 裡的 Gemini key 這次只用來探測能力。實際費用 GPU 約 0.85 小時、約一美元，比交接文件估的 4–6 小時低很多，因為 A4B MoE 在 L4 上每題只要兩百毫秒。</p>
  </div>

  <p class="byline">repo clarencechien/jevlike · branch claude/zen-pascal-tg5upl · results/00-env.md · 01-smoke.md · 03-latency.md · 04-accuracy.md · REPORT.md · cost.md</p>

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
