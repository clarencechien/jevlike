"""v24 E3 data (docs/handoff-v24-calibration-abstain.md §3): the 'insufficient information' option.

Starts from the v23 events (data/v23/T0, same seed-12 data). Per task, one event from each of the 60 twin pairs (side a for even
pairs, b for odd). Each event gives up to three rows:
  C  complete                         gold = original
  M  one decisive fact removed        gold = the added option ('資訊不足') in Q+; still the original letter in Q- (unanswerable)
  R  one irrelevant fact removed      gold = original (control: does anything missing trigger 'insufficient'?)

Renderings: T0 (text, the clause is deleted), J0 (JSON, the key is deleted), J1n (J0 + derived fields; derived values that
depend on the removed fact are null). For C rows T0/J0/J1n are exactly the v23 T0/J0/J1 states.

Checks (assert): the clause to delete occurs exactly once in the v23 T0 text; M is valid (substituting 20+ other values for the
removed fact gives >= 2 different golds, or a value outside the rules' defined ranges); R is valid (all substitutes keep the gold).

Usage: python3 data/v24/gen_v24.py [--seed 24]
Outputs: data/v24/{Qplus,Qminus}/{T0,J0,J1n}/{task}.jsonl, data/v24/MANIFEST.md
"""
import argparse
import collections
import copy
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "data", "v23"))
sys.path.insert(0, ROOT)
import gen_v23 as G  # noqa: E402

TASKS = G.TASKS
RENDER = ["T0", "J0", "J1n"]
INSUF = {"label": "資訊不足", "criteria": "判斷這題需要的事實（數值、規格上下限、狀態、角色…）在狀態裡找不到，要先查資料或問人；事實齊全時不要選"}

FAMS = {}
for _t, lst in G.FAMILIES.items():
    for fam, _n in lst:
        FAMS[type(fam).__name__ + (f":{fam.kind}" if hasattr(fam, "kind") else "")] = fam


def s(v):
    return G.s(v)


# family -> (M fields, T0 clause template for M, R fields or None, T0 clause template for R)
# a template is (old, new): `old` must occur exactly once in the v23 T0 text and is replaced by `new`.
def spec(fam_name, f):
    if fam_name.startswith("SevReflow"):
        r = (["duration_min"], (f"，此讀值已持續 {f['duration_min']} 分鐘", ""))
        if fam_name.endswith("numeric"):
            return ["limit_hi_c"], (f"，上限 {f['limit_hi_c']}°C", ""), *r
        bz = FAMS[fam_name].bool_zh["board_in_oven"][0 if f["board_in_oven"] else 1]
        cz = FAMS[fam_name].enum_zh["conveyor"][f["conveyor"]]
        return ["board_in_oven", "conveyor"], (f"；{bz}，{cz}", ""), *r
    if fam_name == "SevNg":
        return ["inspected"], (f"最近 {f['inspected']} 片檢出", "檢出"), ["upstream"], ("，上游持續生產中", "")
    if fam_name == "SevStations":
        return ["waiting_stations"], (f"，因此停下等待的其他站數：{f['waiting_stations']}", ""), None, None
    if fam_name == "SevPaste":
        return (["paste_life_h"], (f"，使用期限 {f['paste_life_h']} 小時", ""),
                ["boards_printed_since_open"], (f"，開封後共印 {f['boards_printed_since_open']} 片", ""))
    if fam_name == "SevThrow":
        return ["control_limit_pct"], (f"，管制上限 {s(f['control_limit_pct'])}%", ""), None, None
    if fam_name == "CatCode":
        return ["code", "message"], (f" alarm {f['code']}「{f['message']}」", " alarm"), ["machine_state"], ("，機台停止", "")
    if fam_name == "CatAir":
        return (["facility_supply_bar", "facility_min_bar"], (f"；廠務主管供氣 {s(f['facility_supply_bar'])} bar（下限 {s(f['facility_min_bar'])} bar）", ""),
                None, None)
    if fam_name == "CatVolt":
        return (["plant_supply_v", "rated_v", "tolerance_pct"], (f"；同時段廠區配電盤電壓 {f['plant_supply_v']} V，額定 {f['rated_v']} V ±{f['tolerance_pct']}%", ""),
                None, None)
    if fam_name == "DispRetry":
        return (["sop_reset_limit_per_shift"], (f"SOP 規定同一 alarm 操作員每班可自行復歸 {f['sop_reset_limit_per_shift']} 次，", ""),
                ["running_min_since_last_reset"], (f"；最後一次復歸後已運轉 {f['running_min_since_last_reset']} 分鐘", ""))
    if fam_name == "DispMetric":
        return ["sop_pass_min_pct"], (f"（SOP 合格 ≥ {s(f['sop_pass_min_pct'])}%）", ""), None, None
    if fam_name == "DispAdjust":
        return ["operator_range_abs"], (f"；SOP 授權操作員可自行調整的幅度為原值 ±{s(f['operator_range_abs'])} {f['unit']}", ""), None, None
    if fam_name == "DispAction":
        return ["finding"], (f"，操作員檢查結果：{f['finding']}", ""), ["alarm_code"], (f" alarm {f['alarm_code']}，", " alarm，")
    if fam_name.startswith("Spc"):
        return ["ucl", "lcl"], (f"、UCL {f['ucl']}、LCL {f['lcl']}）", "）"), None, None
    if fam_name == "UphHourly":
        return ["target_uph"], (f"目標 UPH {f['target_uph']}；", ""), ["hours"], (f"；{f['hours']} 逐時產出", "；逐時產出")
    if fam_name == "UphAvg":
        return ["target_uph"], (f"UPH 目標 {f['target_uph']}，", "UPH "), None, None
    if fam_name == "UphCt":
        return ["target_ct_s"], (f"目標 CT {s(f['target_ct_s'])} 秒，", ""), None, None
    if fam_name.startswith("Tool"):
        r = (["context"], (f"\n情境：{f['context']}", ""))
        if fam_name == "ToolParam":
            k, (lo, hi) = next(iter(f["param_spec"].items()))
            return ["param_spec"], (f"\n參數規格範圍：{k} {s(lo)}–{s(hi)}", ""), *r
        if fam_name == "ToolScope":
            tg = "/".join(t["line"] + ":" + t["station"] for t in f["params"]["targets"])
            return ["params.targets"], (f"\n參數：targets={tg}", ""), *r
        if fam_name == "ToolRead":
            return ["caller_lines"], (f"（負責 {'、'.join(f['caller_lines'])}）", ""), *r
        if fam_name == "ToolRole":
            return ["caller_role"], (f"呼叫者：{f['caller_role']}（", "呼叫者：（"), *r
    raise KeyError(fam_name)


HIDDEN = {"CatCode": ["_meaning", "_cat"], "DispAction": ["_g"]}


def get(f, path):
    if path == "params.targets":
        return f["params"].get("targets")
    return f.get(path)


def put(f, path, v):
    if path == "params.targets":
        f["params"]["targets"] = v
    else:
        f[path] = v


def drop(f, fields):
    g = copy.deepcopy(f)
    for p in fields:
        if p == "params.targets":
            g["params"].pop("targets", None)
        else:
            g.pop(p, None)
    return g


def variants(fam_name, fam, f, fields, rng, n=24):
    """Events equal to f except the given fields take other plausible values (for validity and derived-null checks)."""
    out = []
    if fam_name.startswith("Spc") and fields == ["ucl", "lcl"]:
        half = f["ucl"] - f["cl"]
        for k in range(n):
            h = max(1, round(half * (0.5 + k / n)))
            g = copy.deepcopy(f); g["ucl"], g["lcl"] = f["cl"] + h, f["cl"] - h; out.append(g)
        return out
    if fam_name == "ToolParam" and fields == ["param_spec"]:
        k, (lo, hi) = next(iter(f["param_spec"].items())); span = hi - lo
        for i in range(n):
            d = (i / (n - 1) - 0.5) * 2 * span
            g = copy.deepcopy(f); g["param_spec"] = {k: [round(lo + d, 1) if isinstance(lo, float) else round(lo + d), round(hi + d, 1) if isinstance(hi, float) else round(hi + d)]}
            out.append(g)
        return out
    num = [p for p in fields if isinstance(get(f, p), (int, float)) and not isinstance(get(f, p), bool)]
    if num and len(num) == len(fields):  # specs, limits and sample sizes differ by product: also scale the removed numbers 0.2x-5x
        for i in range(n):
            g = copy.deepcopy(f); k = 0.2 * 25 ** (i / (n - 1))
            for p in num:
                v = get(f, p); put(g, p, round(v * k, 2) if isinstance(v, float) else max(0, round(v * k)))
            out.append(g)
    for _ in range(n):
        a, b, _k, _h = fam.sample(rng)
        for src in (a, b):
            g = copy.deepcopy(f)
            for p in fields + HIDDEN.get(fam_name, []) * any(p in ("code", "message", "finding") for p in fields):
                put(g, p, copy.deepcopy(get(src, p)))
            if fam_name == "ToolScope":  # caller must stay responsible for the substituted lines, as in the generator
                g["caller_lines"] = sorted(set(g["caller_lines"]) | {t["line"] for t in g["params"]["targets"]})
            out.append(g)
    return out


def gold_or_q(fam, g):
    try:
        return fam.gold(g)
    except (AssertionError, KeyError, ZeroDivisionError):
        return "?"


# derived fields that need a removed fact, listed explicitly (the variation test below can miss a dependency when the
# substituted values happen not to change the result; blind review round 1 found SPC 2-sigma counts and tool line checks)
DEPENDS = {("Spc", "ucl"): ["two_sigma_hi", "two_sigma_lo", "n_out_of_limits", "out_of_limit_points", "max_beyond_2sigma_in_3", "recheck_within_limits"],
           ("ToolScope", "params.targets"): ["lines_affected", "target_in_caller_lines"],
           ("ToolRead", "caller_lines"): ["target_in_caller_lines"]}


def derived_null(fam, f, vs, fam_name="", fields=()):
    d = fam.derive(f)
    if not d:
        return {}
    explicit = {k for (fn, fld), ks in DEPENDS.items() if fam_name.startswith(fn) and fld in fields for k in ks}
    dv = [fam.derive(g) for g in vs if gold_or_q(fam, g) != "?" or True]
    out = {}
    for k, v in d.items():
        changes = any(json.dumps(x.get(k), ensure_ascii=False, sort_keys=True) != json.dumps(v, ensure_ascii=False, sort_keys=True) for x in dv)
        out[k] = None if (changes or k in explicit) else v
    return out


def restore_hidden(fam_name, ev, row):
    f = copy.deepcopy(ev)
    if fam_name == "CatCode":
        f["_cat"] = row["gold"]; f["_meaning"] = row["derived"]["code_meaning"]
    if fam_name == "DispAction":
        f["_g"] = row["gold"]
    return f


def build(seed):
    rng = random.Random(seed)
    Q = G.questions()
    out = {arm: {R: {t: [] for t in TASKS} for R in RENDER} for arm in ("Qplus", "Qminus")}
    stats = collections.Counter()
    for t in TASKS:
        v23 = {R: {r["id"]: r for r in map(json.loads, open(os.path.join(ROOT, "data", "v23", R, f"{t}.jsonl"), encoding="utf-8"))} for R in ("T0", "J0", "J1")}
        ids = sorted(v23["T0"])
        pairs = collections.OrderedDict()
        for i in ids:
            pairs.setdefault(v23["T0"][i]["pair_id"], []).append(i)
        q_minus = Q[t]
        K = len(q_minus["options"]); new_letter = "ABCDEFGHIJ"[K]
        q_plus = copy.deepcopy(q_minus); q_plus["options"][new_letter] = dict(INSUF)
        for pi, (pid, members) in enumerate(pairs.items()):
            rid = sorted(members)[pi % 2]
            row = v23["T0"][rid]; fam_name = row["family"]; fam = FAMS[fam_name]
            f = restore_hidden(fam_name, row["event"], row)
            assert fam.gold(f) == row["gold"], (rid, fam.gold(f), row["gold"])
            mf, mt, rf, rt = spec(fam_name, f)
            base = {"task": t, "difficulty": row["difficulty"], "lang": "zh", "pair_id": pid, "family": fam_name, "v23_id": rid,
                    "hard_type": row["hard_type"], "needs_compare": row["needs_compare"]}
            variants_rows = [("C", None, None, row["gold"])]
            # M validity
            vs = variants(fam_name, fam, f, mf, rng)
            golds = {gold_or_q(fam, g) for g in vs}
            # the UPH criteria also count a sudden hour-to-hour drop as abnormal: such a series is decidable without the target
            sudden = fam_name == "UphHourly" and fam.derive(f)["max_hour_to_hour_drop_pct"] >= 10
            if len(golds) >= 2 and not sudden:
                variants_rows.append(("M", mf, mt, None)); stats[f"{t}:M"] += 1
            else:  # the remaining facts already decide (e.g. 1 reset is within any limit >= 1): no M row for this event
                stats[f"{t}:M_skipped"] += 1
            if rf:
                vr = variants(fam_name, fam, f, rf, rng)
                gr = {gold_or_q(fam, g) for g in vr}
                if gr == {row["gold"]}:
                    variants_rows.append(("R", rf, rt, row["gold"])); stats[f"{t}:R"] += 1
                else:
                    stats[f"{t}:R_skipped"] += 1
            stats[f"{t}:C"] += 1
            for kind, fields, tmpl, g_keep in variants_rows:
                if kind == "C":
                    st = {"T0": v23["T0"][rid]["state"], "J0": v23["J0"][rid]["state"], "J1n": v23["J1"][rid]["state"]}
                else:
                    old, new = tmpl
                    t0 = v23["T0"][rid]["state"]
                    assert t0.count(old) == 1, (rid, kind, old, t0)
                    pub = G.public(drop(f, fields))
                    vs_k = variants(fam_name, fam, f, fields, rng) if kind == "M" else vr
                    dn = derived_null(fam, f, vs_k, fam_name, fields)
                    st = {"T0": t0.replace(old, new), "J0": G.jdump(pub), "J1n": G.jdump({**pub, **dn})}
                    for p in fields:  # the removed fact must be gone from every rendering
                        v = get(f, p)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            assert f'"{p.split(".")[-1]}"' not in st["J0"], (rid, p)
                for R in RENDER:
                    for arm, q in (("Qplus", q_plus), ("Qminus", q_minus)):
                        gold = (new_letter if arm == "Qplus" else None) if kind == "M" else g_keep
                        out[arm][R][t].append({**base, "id": f"{t}-v24-{pi:03d}{kind}", "variant": kind, "state": st[R], "question": q,
                                               "gold": gold if gold is not None else "-", "insufficient_letter": new_letter if arm == "Qplus" else None,
                                               "orig_gold": row["gold"], "removed": fields})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=24)
    a = ap.parse_args()
    out, stats = build(a.seed)
    for arm in out:
        for R in RENDER:
            os.makedirs(os.path.join(HERE, arm, R), exist_ok=True)
            for t in TASKS:
                with open(os.path.join(HERE, arm, R, f"{t}.jsonl"), "w", encoding="utf-8") as fh:
                    for x in out[arm][R][t]:
                        fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    lines = ["# data/v24 MANIFEST", "", f"`python3 data/v24/gen_v24.py --seed {a.seed}`（seed 只用在有效性檢查的抽樣；事件本身是 data/v23 的 seed 12）。",
             "每類從 v23 的 60 組 twin 各取一邊（偶數組 a、奇數組 b），每個事件 C 完整／M 缺關鍵事實／R 缺無關事實（有對應欄位的家族才有 R）。",
             "Qplus 在原選項後加「資訊不足」選項（M 的 gold 是它）；Qminus 是原選項（M 的 gold 記成 `-`，只看會不會被門檻轉人）。", "",
             "| task | C | M | R | M 略過（剩下的事實已能判定） | R 略過 | 加的選項字母 |", "|---|---|---|---|---|---|---|"]
    for t in TASKS:
        lines.append(f"| {t} | {stats[t + ':C']} | {stats[t + ':M']} | {stats[t + ':R']} | {stats[t + ':M_skipped']} | {stats[t + ':R_skipped']} | {out['Qplus']['T0'][t][0]['insufficient_letter']} |")
    fam_rows = collections.Counter((x["task"], x["family"], x["variant"]) for t in TASKS for x in out["Qplus"]["T0"][t])
    lines += ["", "| task｜family | M 拿掉 | R 拿掉 | 事件數 |", "|---|---|---|---|"]
    seen = {}
    for t in TASKS:
        for x in out["Qplus"]["T0"][t]:
            if x["variant"] == "M":
                seen.setdefault((t, x["family"]), [x["removed"], None, 0])
                seen[(t, x["family"])][2] += 1
            if x["variant"] == "R":
                seen.setdefault((t, x["family"]), [None, None, 0])[1] = x["removed"]
    for (t, fn), (m, r, n) in seen.items():
        lines.append(f"| {t}｜{fn} | {', '.join(m or [])} | {', '.join(r or []) or '（無）'} | {n} |")
    lines += ["", "檢查（產生時 assert）：要刪的句段在 v23 T0 恰好出現一次；M 有效才出 M 列（被拿掉的事實換 24–72 個值：數字另外乘 0.2–5 倍，gold 至少兩種，或超出規則定義範圍；剩下的事實已能判定的事件不出 M 列，含逐時產出有 ≥ 10% 驟降的 UPH——標準把突然下滑也算異常）；"
              "R 有效才出 R 列（換值後 gold 全不變）；J1n 的衍生值凡是會隨被拿掉的事實改變的（換值測試，加上 SPC 2σ／超限計數、工具線別檢查的明列清單），一律 null。C 列三種寫法與 v23 的 T0／J0／J1 逐字相同。"]
    open(os.path.join(HERE, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
