"""Render results/modal/latency/latency.json -> results/03-latency.md"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "results/modal/latency/latency.json")))
R = d["results"]


def f(x, n=0):
    return "—" if x is None else f"{x:.{n}f}"


def row(name, label, extra=""):
    s = R[name]["summary"]
    return f"| {label} | {f(s['p50'])} | {f(s['p95'])} | {f(s['p99'])} | {f(s.get('prompt_tokens_mean'))} | {f(s.get('server_prompt_ms_p50'))} | {f(s.get('cache_n_mean'))} | {extra} |"


H = "| 條件 | p50 ms | p95 ms | p99 ms | prompt tokens | server prompt_ms p50 | cache 命中 tokens | 備註 |\n|---|---|---|---|---|---|---|---|"
L = [f"# 03 — 延遲（M3）— **{d['gpu']}，GB10 保守上界**", "",
     f"每組 n={d['n']}（warm-up {d['warmup']}），client 端量測（同容器 localhost，不含 Modal 網路層）。原始：`results/modal/latency/latency.json`。",
     "模型 Gemma 4 26B-A4B UD-Q4_K_M，llama-server `-c 16384 -np 4`，`cache_prompt=true` 除非另註。", "",
     "## L1 單題基準（state ≈100 token，2 選項）", "", H,
     row("L1_single_100tok_2opt", "L1 每次不同 state", "只有模板前綴命中 cache"),
     row("L1_ref_same_prompt_repeated", "參考：同一 prompt 重複", "全 cache 命中 = 1 token decode 地板"),
     row("L1_ref_nocache", "參考：cache_prompt=false", "全部重算"), "",
     "## L2 state 長度（2 選項）", "", H,
     row("L2_state_100tok", "100 token"), row("L2_state_500tok", "500 token"), row("L2_state_2000tok", "2000 token"), "",
     "## L3 選項數（state ≈100 token）", "", H,
     row("L3_2opt", "2 選項"), row("L3_5opt", "5 選項"), row("L3_10opt", "10 選項"), "",
     "## L4 多題共用 state（state ≈300 token；總時間 = N 題全部答完）", "",
     "| N 題 | cache on 總 p50 ms | cache on 每題 p50 | cache on 每題 prompt_ms | cache off 總 p50 ms | cache off 每題 p50 | 倍數（cache off / on） | N 題 vs 1 題（cache on） |", "|---|---|---|---|---|---|---|---|"]
base_on = R["L4_shared_state_1q_cache_on"]["summary"]["p50"]
for nq in (1, 5, 10):
    on, off = R[f"L4_shared_state_{nq}q_cache_on"], R[f"L4_shared_state_{nq}q_cache_off"]
    L.append(f"| {nq} | {f(on['summary']['p50'])} | {f(on['per_call']['p50'])} | {f(on['per_call']['server_prompt_ms_p50'])} | {f(off['summary']['p50'])} | {f(off['per_call']['p50'])} | "
             f"{off['summary']['p50'] / on['summary']['p50']:.2f}x | {on['summary']['p50'] / base_on:.2f}x |")
L += ["", "## L5 併發（每次不同 state，2 選項）", "",
      "| 併發 client | p50 ms | p95 ms | p99 ms | throughput req/s |", "|---|---|---|---|---|"]
for c in (1, 4, 8):
    s = R[f"L5_concurrency_{c}"]["summary"]
    L.append(f"| {c} | {f(s['p50'])} | {f(s['p95'])} | {f(s['p99'])} | {s['throughput_rps']:.1f} |")
l6 = R["L6_with_bg_generation"]
L += ["", "## L6 背景生成負載（同容器一個執行緒持續生成 512 token）", "", H,
      row("L1_single_100tok_2opt", "無背景負載（= L1）"),
      row("L6_with_bg_generation", "有背景生成", f"背景完成 {l6['bg_generation']['n']} 次生成、{l6['bg_generation']['tokens']} tokens"),
      "", f"**拖慢倍數（p50）：{l6['slowdown_p50']:.2f}x**（L4 上的相對值；GB10 併發行為要重測）", "",
      "## L7 對照：同一題改用 `/v1/chat/completions` 生成 JSON（temperature 0，max_tokens 32）", ""]
l7 = R["L7_chat_json"]["summary"]
L += ["| 方法 | p50 ms | p95 ms | p99 ms | 輸出 tokens |", "|---|---|---|---|---|",
      f"| typed decision（L1） | {f(R['L1_single_100tok_2opt']['summary']['p50'])} | {f(R['L1_single_100tok_2opt']['summary']['p95'])} | {f(R['L1_single_100tok_2opt']['summary']['p99'])} | 1 |",
      f"| LLM 生成 JSON | {f(l7['p50'])} | {f(l7['p95'])} | {f(l7['p99'])} | {l7['completion_tokens_mean']:.1f} |",
      "", f"**JSON 生成 / typed decision（p50）= {l7['speedup_vs_L1_p50']:.1f}x**", "",
      "註：所有數字皆為 L4（300 GB/s、算力弱於 GB10）；GB10 要用 v1 §5 的 L1–L7 重跑一次（約 30 分鐘）。"]
open(os.path.join(ROOT, "results/03-latency.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote results/03-latency.md")
