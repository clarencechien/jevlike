"""v23 data (docs/handoff-v23-input-format.md): one event, five renderings.

  T0  Chinese text, raw values + specs (no comparison results)
  T1  T0 + derived phrases (differences, over/under, ratios, run counts, code meanings)
  J0  compact JSON of the same facts
  J1  J0 + derived fields
  JN  J0 + 6-10 irrelevant fields (ids, timestamps, firmware, PLC address...)

Every event is generated as a dict of facts; gold comes from the facts and the task rules (never from a model).
Twin pairs: the two events differ in one fact (sometimes two coupled ones) and have different gold; at least half of
each task's pairs are numeric twins (only a number moves across a limit).

Checks (assert, run on every row):
  parity  every fact value appears in T0 and J0 (numbers verbatim, booleans/enums via their Chinese phrase);
          every derived value appears in T1 and J1; T1 contains all of T0's facts
  leak    no option label / judgment word in any rendering (e.g. 停線, 允許, 異常)

Usage: python3 data/v23/gen_v23.py [--seed 12]
Outputs: data/v23/{T0,T1,J0,J1,JN}/{task}.jsonl, data/v23/MANIFEST.md
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from data.gen._common import AOIS, LINES, MOUNTERS, PRINTERS, REFLOWS, SPIS, wo  # noqa: E402

RENDERINGS = ["T0", "T1", "J0", "J1", "JN"]
TASKS = ["m_alarm_severity", "m_alarm_category", "m_needs_dispatch", "q_spc_action", "p_uph_anomaly", "t_tool_gate"]
N_PAIRS = 60


def _q(t):
    return {"type": t["kind"], "instructions": t["instructions"],
            "options": {k: {"label": v["label"], "criteria": v["criteria"]} for k, v in t["options"].items()}}


def questions():
    v2 = json.load(open(os.path.join(ROOT, "data/seeds/tasks_v2.json"), encoding="utf-8"))
    v9 = json.load(open(os.path.join(ROOT, "data/v9/tasks_v9.json"), encoding="utf-8"))
    q = {t: _q(v2[t]) for t in ("m_alarm_severity", "q_spc_action")}  # same as --criteria in the D0 runs
    for t in ("m_alarm_category", "m_needs_dispatch", "p_uph_anomaly"):
        q[t] = json.loads(open(os.path.join(ROOT, f"data/synthetic/{t}.jsonl"), encoding="utf-8").readline())["question"]
    q["t_tool_gate"] = _q(v9["t_tool_gate"])
    return q


def s(v):
    """the one number format used in every rendering (so parity can be checked verbatim)"""
    if isinstance(v, float):
        return f"{v:g}" if abs(v) >= 1 or v == 0 else f"{v:.2f}".rstrip("0")
    return str(v)


def sgn(v):
    return ("+" if v > 0 else "") + s(v)


def yn(b):
    return "是" if b else "否"


# =====================================================================================================================
# Each family: sample(rng) -> (facts_a, facts_b, twin_kind, hard_type); derive(f) -> dict; gold(f) -> letter;
# text(f, d=None) -> str (d given: T1 with derived phrases); bool_zh / enum_zh for the parity check.
# =====================================================================================================================
class Fam:
    needs_compare = True
    bool_zh: dict = {}
    enum_zh: dict = {}
    derived_bool_zh: dict = {}
    skip_parity: set = set()

    def derive(self, f):
        return {}


# ---------------------------------------------------------------- m_alarm_severity (A 不急 / B 盡快 / C 停線)
class SevReflow(Fam):
    bool_zh = {"board_in_oven": ("爐內有板", "爐內無板")}
    enum_zh = {"conveyor": {"running": "輸送帶運轉中", "stopped": "輸送帶已停止進板"}}
    derived_bool_zh = {"over_limit": ("超出上限：是", "超出上限：否")}

    def __init__(self, kind):
        self.kind = kind  # "numeric": C vs A (temp across the limit); "fact": C vs B (board in oven / auto-stopped)
        self.needs_compare = kind == "numeric"

    def sample(self, rng):
        lim = rng.choice([240, 245, 250, 255, 260])
        a = {"line": rng.choice(LINES), "device": rng.choice(REFLOWS), "zone": rng.randint(3, 8), "temp_c": lim + rng.randint(4, 15),
             "limit_hi_c": lim, "duration_min": rng.randint(2, 9), "board_in_oven": True, "conveyor": "running"}
        b = dict(a)
        if self.kind == "numeric":
            b["temp_c"] = lim - rng.randint(2, 9)
        else:
            b["board_in_oven"], b["conveyor"] = False, "stopped"
        return a, b, self.kind, None

    def derive(self, f):
        return {"over_limit": f["temp_c"] > f["limit_hi_c"], "temp_minus_limit_c": f["temp_c"] - f["limit_hi_c"]}

    def gold(self, f):
        if f["temp_c"] > f["limit_hi_c"]:
            return "C" if f["board_in_oven"] else "B"
        assert f["conveyor"] == "running"
        return "A"

    def text(self, f, d=None):
        t = f"{f['line']} 回焊爐 {f['device']} 第 {f['zone']} 區溫度 {f['temp_c']}°C，上限 {f['limit_hi_c']}°C"
        if d:
            t += f"（與上限差 {sgn(d['temp_minus_limit_c'])}°C，{self.derived_bool_zh['over_limit'][0 if d['over_limit'] else 1]}）"
        t += f"，此讀值已持續 {f['duration_min']} 分鐘；{self.bool_zh['board_in_oven'][0 if f['board_in_oven'] else 1]}，{self.enum_zh['conveyor'][f['conveyor']]}"
        return t


class SevNg(Fam):
    bool_zh = {"intercepted": ("NG 品已被 AOI 攔下", "NG 品未攔下")}
    enum_zh = {"upstream": {"running": "上游持續生產中"}}

    def sample(self, rng):
        n = rng.choice([20, 25, 40, 50, 60, 80])
        a = {"line": rng.choice(LINES), "station": rng.choice(AOIS), "inspected": n, "ng": round(n * rng.uniform(0.34, 0.55)), "intercepted": True, "upstream": "running"}
        b = dict(a, ng=max(1, round(n * rng.uniform(0.03, 0.08))))
        return a, b, "numeric", None

    def derive(self, f):
        return {"ng_rate_pct": round(100 * f["ng"] / f["inspected"], 1)}

    def gold(self, f):
        r = f["ng"] / f["inspected"]
        assert r >= 0.30 or r < 0.10
        return "C" if r >= 0.30 else "B"

    def text(self, f, d=None):
        t = f"{f['line']} {f['station']} 最近 {f['inspected']} 片檢出 NG {f['ng']} 片"
        if d:
            t += f"（NG 率 {s(d['ng_rate_pct'])}%）"
        return t + f"，{self.bool_zh['intercepted'][0]}，{self.enum_zh['upstream'][f['upstream']]}"


class SevStations(Fam):
    enum_zh = {"machine_state": {"stopped": "停機"}}
    def sample(self, rng):
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "alarm": rng.choice(["吸料失敗連續 5 次", "置件頭原點異常", "供料器卡料"]),
             "machine_state": "stopped", "waiting_stations": rng.randint(2, 5)}
        b = dict(a, waiting_stations=0)
        return a, b, "numeric", None

    def derive(self, f):
        return {"stations_affected_total": f["waiting_stations"] + 1}

    def gold(self, f):
        return "C" if f["waiting_stations"] >= 2 else "B"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} 因{f['alarm']}{self.enum_zh['machine_state'][f['machine_state']]}，因此停下等待的其他站數：{f['waiting_stations']}"
        if d:
            t += f"（連同本站共 {d['stations_affected_total']} 站停著）"
        return t


class SevPaste(Fam):
    def sample(self, rng):
        life = rng.choice([8, 12, 24])
        a = {"line": rng.choice(LINES), "printer": rng.choice(PRINTERS), "hours_since_open": life + rng.randint(1, 6), "paste_life_h": life,
             "printer_state": "printing", "boards_printed_last_hour": rng.randint(20, 60), "boards_printed_since_open": rng.randint(300, 900)}
        b = dict(a, hours_since_open=life - rng.randint(1, 5))
        return a, b, "numeric", None

    def derive(self, f):
        return {"expired": f["hours_since_open"] > f["paste_life_h"], "hours_past_life": f["hours_since_open"] - f["paste_life_h"]}

    derived_bool_zh = {"expired": ("已過使用期限：是", "已過使用期限：否")}
    enum_zh = {"printer_state": {"printing": "印刷持續中"}}

    def gold(self, f):
        return "C" if f["hours_since_open"] > f["paste_life_h"] else "A"

    def text(self, f, d=None):
        t = f"{f['line']} 印刷機 {f['printer']} 錫膏時效檢查：開封回溫後已 {f['hours_since_open']} 小時，使用期限 {f['paste_life_h']} 小時"
        if d:
            t += f"（與期限差 {sgn(d['hours_past_life'])} 小時，{self.derived_bool_zh['expired'][0 if d['expired'] else 1]}）"
        return t + f"；{self.enum_zh['printer_state'][f['printer_state']]}，最近 1 小時印 {f['boards_printed_last_hour']} 片，開封後共印 {f['boards_printed_since_open']} 片"


class SevThrow(Fam):
    enum_zh = {"machine_state": {"running": "機台運轉中"}}
    derived_bool_zh = {"over_limit": ("超出管制上限：是", "超出管制上限：否")}

    def sample(self, rng):
        lim = rng.choice([0.2, 0.3, 0.5])
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "throw_rate_pct": round(lim * rng.uniform(1.3, 2.2), 2),
             "control_limit_pct": lim, "machine_state": "running", "aoi_defects_after_reflow": 0}
        b = dict(a, throw_rate_pct=round(lim * rng.uniform(0.4, 0.8), 2))
        return a, b, "numeric", None

    def derive(self, f):
        return {"over_limit": f["throw_rate_pct"] > f["control_limit_pct"], "rate_minus_limit_pct": round(f["throw_rate_pct"] - f["control_limit_pct"], 2)}

    def gold(self, f):
        return "B" if f["throw_rate_pct"] > f["control_limit_pct"] else "A"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} 本班拋料率 {s(f['throw_rate_pct'])}%，管制上限 {s(f['control_limit_pct'])}%"
        if d:
            t += f"（差 {sgn(d['rate_minus_limit_pct'])} 個百分點，{self.derived_bool_zh['over_limit'][0 if d['over_limit'] else 1]}）"
        return t + f"；{self.enum_zh['machine_state'][f['machine_state']]}，爐後 AOI 不良 {f['aoi_defects_after_reflow']} 片"


# ---------------------------------------------------------------- m_alarm_category (A 機構 B 電控 C 物料 D 程式 E 環境)
CODES = {  # code, machine message (as the machine prints it), meaning (the code-table lookup T1/J1 add)
    "A": [("E-2107", "VAC LO HD{n}", "第 {n} 頭吸嘴真空不足"), ("E-2210", "NZL CHG FAIL ST{n}", "第 {n} 號換嘴站換嘴動作失敗"),
          ("E-2315", "CONV BELT SLIP", "輸送皮帶打滑"), ("E-2402", "CLAMP CYL TMO", "夾板氣缸動作逾時")],
    "B": [("E-5102", "SRV AMP ALM AX-Y", "Y 軸伺服驅動器警報"), ("E-5230", "ENC COMM ERR HD{n}", "第 {n} 頭編碼器通訊錯誤"),
          ("E-5311", "IO BRD {n} NO RESP", "I/O 板 {n} 無回應"), ("E-5420", "PLC LINK DOWN", "PLC 連線中斷")],
    "C": [("E-3304", "FDR EMPTY SL{n}", "第 {n} 站位料帶用完"), ("E-3310", "PN MISMATCH SL{n}", "第 {n} 站位料號與置件檔不符"),
          ("E-3402", "TRAY EMPTY", "料盤無料"), ("E-3505", "PASTE LO", "鋼網上錫膏量不足")],
    "D": [("E-6601", "PRG CHKSUM NG", "置件檔校驗碼不符"), ("E-6610", "FID NOT FOUND", "依座標找不到基準點"),
          ("E-6702", "RCP VER MISMATCH", "配方版本與工單指定版本不符"), ("E-6803", "DB SYNC FAIL", "料號資料庫同步失敗")],
    "E": [("E-7402", "AMB HUMID HI", "環境濕度超過設定"), ("E-7205", "EXH FLOW LO", "排風流量低於設定"),
          ("E-7310", "ESD GND ALM", "靜電接地監測警報"), ("E-7120", "PWR SAG DET", "偵測到供電電壓驟降")],
}


def code_pick(rng, cat, n):
    c, m, mean = rng.choice(CODES[cat])
    return c, m.format(n=n), mean.format(n=n)


class CatCode(Fam):
    needs_compare = False

    def __init__(self):
        self.queue = []

    def sample(self, rng):
        if not self.queue:  # every unordered category pair equally often (10 pairs x 3)
            import itertools
            self.queue = [list(p) for p in itertools.combinations("ABCDE", 2)] * 3
            rng.shuffle(self.queue)
            for p in self.queue:
                rng.shuffle(p)
        ca, cb = self.queue.pop()
        n = rng.randint(1, 12)
        base = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "machine_state": "stopped"}
        a, b = dict(base), dict(base)
        a["code"], a["message"], a["_meaning"], a["_cat"] = *code_pick(rng, ca, n), ca
        b["code"], b["message"], b["_meaning"], b["_cat"] = *code_pick(rng, cb, n), cb
        return a, b, "fact", None

    enum_zh = {"machine_state": {"stopped": "機台停止"}}
    skip_parity = {"_meaning", "_cat"}

    def derive(self, f):
        return {"code_meaning": f["_meaning"]}

    def gold(self, f):
        return f["_cat"]

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} alarm {f['code']}「{f['message']}」"
        if d:
            t += f"（代碼意思：{d['code_meaning']}）"
        return t + "，機台停止"


class CatAir(Fam):
    derived_bool_zh = {"facility_below_min": ("廠務端低於下限：是", "廠務端低於下限：否"),
                       "machine_below_min": ("機台端低於下限：是", "機台端低於下限：否")}

    def sample(self, rng):
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "code": "E-2120", "message": "AIR PRESS LO",
             "facility_supply_bar": round(rng.uniform(5.4, 6.0), 1), "facility_min_bar": 5.0,
             "machine_inlet_bar": round(rng.uniform(3.3, 3.9), 1), "machine_min_bar": 4.5}
        b = dict(a, facility_supply_bar=round(rng.uniform(4.0, 4.6), 1))
        return a, b, "numeric", None

    def derive(self, f):
        return {"code_meaning": "機台入口氣壓過低", "facility_below_min": f["facility_supply_bar"] < f["facility_min_bar"],
                "facility_minus_min_bar": round(f["facility_supply_bar"] - f["facility_min_bar"], 1),
                "machine_below_min": f["machine_inlet_bar"] < f["machine_min_bar"],
                "machine_minus_min_bar": round(f["machine_inlet_bar"] - f["machine_min_bar"], 1)}

    def gold(self, f):  # supply low -> utilities (E); supply fine but machine low -> the machine's own regulator/piping (A)
        return "E" if f["facility_supply_bar"] < f["facility_min_bar"] else "A"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} alarm {f['code']}「{f['message']}」"
        if d:
            t += f"（代碼意思：{d['code_meaning']}）"
        t += f"；廠務主管供氣 {s(f['facility_supply_bar'])} bar（下限 {s(f['facility_min_bar'])} bar）"
        if d:
            t += f"（差 {sgn(d['facility_minus_min_bar'])} bar，{self.derived_bool_zh['facility_below_min'][0 if d['facility_below_min'] else 1]}）"
        t += f"，機台入口 {s(f['machine_inlet_bar'])} bar（下限 {s(f['machine_min_bar'])} bar）"
        if d:
            t += f"（差 {sgn(d['machine_minus_min_bar'])} bar，{self.derived_bool_zh['machine_below_min'][0 if d['machine_below_min'] else 1]}）"
        return t


class CatVolt(Fam):
    derived_bool_zh = {"within_tolerance": ("在容許範圍內：是", "在容許範圍內：否")}

    def sample(self, rng):
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "code": "E-5102", "message": "SRV AMP ALM AX-Y",
             "plant_supply_v": rng.randint(217, 225), "rated_v": 220, "tolerance_pct": 5}
        b = dict(a, plant_supply_v=rng.randint(194, 204))
        return a, b, "numeric", None

    def derive(self, f):
        lo, hi = f["rated_v"] * (1 - f["tolerance_pct"] / 100), f["rated_v"] * (1 + f["tolerance_pct"] / 100)
        return {"code_meaning": "Y 軸伺服驅動器警報", "tolerance_range_v": f"{s(lo)}–{s(hi)}",
                "supply_dev_pct": round(100 * (f["plant_supply_v"] - f["rated_v"]) / f["rated_v"], 1), "within_tolerance": lo <= f["plant_supply_v"] <= hi}

    def gold(self, f):  # supply out of tolerance -> power quality (E); supply normal -> the drive itself (B)
        return "B" if self.derive(f)["within_tolerance"] else "E"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} alarm {f['code']}「{f['message']}」"
        if d:
            t += f"（代碼意思：{d['code_meaning']}）"
        t += f"；同時段廠區配電盤電壓 {f['plant_supply_v']} V，額定 {f['rated_v']} V ±{f['tolerance_pct']}%"
        if d:
            t += f"（容許範圍 {d['tolerance_range_v']} V，偏差 {sgn(d['supply_dev_pct'])}%，{self.derived_bool_zh['within_tolerance'][0 if d['within_tolerance'] else 1]}）"
        return t


# ---------------------------------------------------------------- m_needs_dispatch (A 是 / B 否)
class DispRetry(Fam):
    derived_bool_zh = {"over_limit": ("超過上限：是", "超過上限：否")}

    def sample(self, rng):
        lim = rng.choice([2, 3, 3, 5])
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "alarm": rng.choice(["吸料失敗", "卡板", "置件偏移", "供料器回授錯誤"]),
             "sop_reset_limit_per_shift": lim, "resets_this_shift": lim + rng.randint(1, 4), "running_min_since_last_reset": rng.randint(10, 40)}
        b = dict(a, resets_this_shift=rng.randint(1, lim - 1))
        return a, b, "numeric", None

    def derive(self, f):
        return {"over_limit": f["resets_this_shift"] > f["sop_reset_limit_per_shift"],
                "resets_minus_limit": f["resets_this_shift"] - f["sop_reset_limit_per_shift"]}

    def gold(self, f):
        return "A" if f["resets_this_shift"] > f["sop_reset_limit_per_shift"] else "B"

    def text(self, f, d=None):
        t = (f"{f['line']} {f['device']} {f['alarm']} alarm，操作員依 SOP 復歸；SOP 規定同一 alarm 操作員每班可自行復歸 {f['sop_reset_limit_per_shift']} 次，"
             f"本班已復歸 {f['resets_this_shift']} 次")
        if d:
            t += f"（與上限差 {sgn(d['resets_minus_limit'])} 次，{self.derived_bool_zh['over_limit'][0 if d['over_limit'] else 1]}）"
        return t + f"；最後一次復歸後已運轉 {f['running_min_since_last_reset']} 分鐘"


class DispMetric(Fam):
    derived_bool_zh = {"meets_standard": ("達到 SOP 合格值：是", "達到 SOP 合格值：否")}
    KINDS = [("吸嘴清潔", "吸料成功率", "%", 99.5, (99.6, 99.9), (97.8, 99.2)),
             ("鋼網擦拭", "SPI 良率", "%", 98.0, (98.3, 99.6), (93.0, 97.2)),
             ("供料器重新安裝", "吸料成功率", "%", 99.5, (99.6, 99.9), (97.5, 99.1))]

    def sample(self, rng):
        act, metric, unit, std, ok, bad = rng.choice(self.KINDS)
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS if "吸" in metric else PRINTERS), "sop_action": act, "metric": metric,
             "after_action_pct": round(rng.uniform(*bad), 1), "sop_pass_min_pct": std}
        b = dict(a, after_action_pct=round(rng.uniform(*ok), 1))
        return a, b, "numeric", None

    skip_parity = {"metric", "sop_action"}

    def derive(self, f):
        return {"meets_standard": f["after_action_pct"] >= f["sop_pass_min_pct"], "minus_standard_pct": round(f["after_action_pct"] - f["sop_pass_min_pct"], 1)}

    def gold(self, f):
        return "B" if f["after_action_pct"] >= f["sop_pass_min_pct"] else "A"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} {f['metric']}偏低，操作員依 SOP 做{f['sop_action']}，之後{f['metric']} {s(f['after_action_pct'])}%（SOP 合格 ≥ {s(f['sop_pass_min_pct'])}%）"
        if d:
            t += f"（差 {sgn(d['minus_standard_pct'])} 個百分點，{self.derived_bool_zh['meets_standard'][0 if d['meets_standard'] else 1]}）"
        return t


class DispAdjust(Fam):
    derived_bool_zh = {"within_operator_range": ("在操作員可調範圍內：是", "在操作員可調範圍內：否")}
    KINDS = [("印刷壓力", "kg", 6.0, 0.5, PRINTERS), ("回焊爐第 5 區設定溫度", "°C", 235.0, 5.0, REFLOWS), ("置件頭 Z 軸高度補償", "mm", 0.0, 0.05, MOUNTERS)]

    def sample(self, rng):
        name, unit, cur, auth, devs = rng.choice(self.KINDS)
        big = round(cur + rng.choice([-1, 1]) * auth * rng.uniform(1.6, 3.0), 2 if unit == "mm" else 1)
        small = round(cur + rng.choice([-1, 1]) * auth * rng.uniform(0.3, 0.8), 2 if unit == "mm" else 1)
        a = {"line": rng.choice(LINES), "device": rng.choice(devs), "param": name, "unit": unit, "current": cur, "required": big, "operator_range_abs": auth}
        b = dict(a, required=small)
        return a, b, "numeric", None

    skip_parity = {"param", "unit"}

    def derive(self, f):
        ch = round(abs(f["required"] - f["current"]), 2)
        return {"change_abs": ch, "within_operator_range": ch <= f["operator_range_abs"]}

    def gold(self, f):
        return "B" if self.derive(f)["within_operator_range"] else "A"

    def text(self, f, d=None):
        u = f["unit"]
        t = (f"{f['line']} {f['device']} 首件量測後，需把{f['param']}由 {s(f['current'])} {u} 改為 {s(f['required'])} {u}；"
             f"SOP 授權操作員可自行調整的幅度為原值 ±{s(f['operator_range_abs'])} {u}")
        if d:
            t += f"（本次調整幅度 {s(d['change_abs'])} {u}，{self.derived_bool_zh['within_operator_range'][0 if d['within_operator_range'] else 1]}）"
        return t


class DispAction(Fam):
    needs_compare = False
    PAIRS = [("吸嘴座斷裂、吸嘴無法固定", "吸嘴前端沾錫膏，清潔後吸料恢復正常"),
             ("供料器棘輪齒崩、送料步距不準", "料帶未卡入棘輪，重新裝料後送料正常"),
             ("輸送皮帶斷裂、碎屑卡在滾輪", "入口擋板卡住一片板子，取出後重置正常"),
             ("相機鏡頭座鬆脫、影像失焦", "相機鏡面有灰塵，依 SOP 擦拭後辨識正常"),
             ("真空泵浦異音、真空值上不去", "真空過濾棉堵塞，依 SOP 更換濾棉後真空正常")]

    def sample(self, rng):
        bad, ok = rng.choice(self.PAIRS)
        base = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "alarm_code": f"E-{rng.randint(2000, 2999)}"}
        return dict(base, finding=bad, _g="A"), dict(base, finding=ok, _g="B"), "fact", None

    skip_parity = {"_g"}

    def gold(self, f):
        return f["_g"]

    def text(self, f, d=None):
        return f"{f['line']} {f['device']} alarm {f['alarm_code']}，操作員檢查結果：{f['finding']}"


# ---------------------------------------------------------------- q_spc_action (A 持續監控 B 抽檢 C 停線複檢 D 呼叫 QE)
def spc_stats(f):
    cl, ucl, lcl, p = f["cl"], f["ucl"], f["lcl"], f["points"]
    sig = (ucl - cl) / 3
    s2h, s2l = cl + 2 * sig, cl - 2 * sig
    out = [i + 1 for i, x in enumerate(p) if x > ucl or x < lcl]
    side = [1 if x > cl else -1 if x < cl else 0 for x in p]
    best = run = 0
    for i, sd in enumerate(side):
        run = run + 1 if sd != 0 and i > 0 and sd == side[i - 1] else (1 if sd != 0 else 0)
        best = max(best, run)
    mono = run = 1
    dirn = 0
    for i in range(1, len(p)):
        dd = (p[i] > p[i - 1]) - (p[i] < p[i - 1])
        run = run + 1 if dd != 0 and dd == dirn else (2 if dd != 0 else 1)
        dirn = dd
        mono = max(mono, run)
    two3 = 0
    for i in range(len(p) - 2):
        w = p[i:i + 3]
        two3 = max(two3, sum(x > s2h for x in w), sum(x < s2l for x in w))
    return {"two_sigma_hi": round(s2h, 1), "two_sigma_lo": round(s2l, 1), "n_out_of_limits": len(out), "out_of_limit_points": out,
            "longest_same_side_run": best, "longest_monotone_run": mono, "max_beyond_2sigma_in_3": two3}


def spc_gold(f):
    st = spc_stats(f)
    if st["n_out_of_limits"] >= 2:
        return "D"
    if st["n_out_of_limits"] == 1:
        return "D" if f["recheck"] == "still_out" else "C"
    if st["longest_same_side_run"] >= 7 or st["longest_monotone_run"] >= 6 or st["max_beyond_2sigma_in_3"] >= 2:
        return "B"
    return "A"


SPC_CHARTS = [("錫膏厚度", "µm", 150, 21, SPIS), ("錫膏體積", "%", 100, 24, SPIS), ("貼片 X 偏移", "µm", 0, 45, MOUNTERS),
              ("印刷 Y 偏移", "µm", 0, 24, PRINTERS), ("回焊峰值溫度", "°C", 245, 6, REFLOWS)]


class Spc(Fam):
    enum_zh = {"recheck": {"none": "未複測", "back_in": "已複測", "still_out": "已複測"}}

    def __init__(self, kind):
        self.kind = kind  # run7 | mono6 | two3 | one_out | second_out | recheck

    def base(self, rng):
        name, unit, cl, half, stations = rng.choice(SPC_CHARTS)
        sig = half / 3
        for _ in range(500):
            pts = [round(cl + max(-1.7 * sig, min(1.7 * sig, rng.gauss(0, 0.8 * sig)))) for _ in range(25)]
            f = {"line": rng.choice(LINES), "station": rng.choice(stations), "chart": name, "unit": unit,
                 "cl": cl, "ucl": cl + half, "lcl": cl - half, "points": pts, "recheck": "none", "recheck_value": None}
            st = spc_stats(f)
            if st["longest_same_side_run"] <= 5 and st["longest_monotone_run"] <= 4 and st["max_beyond_2sigma_in_3"] <= 1 and st["n_out_of_limits"] == 0:
                return f, sig
        raise RuntimeError("no base series")

    def sample(self, rng):
        for _ in range(200):
            a, sig = self.base(rng)
            cl, ucl, lcl = a["cl"], a["ucl"], a["lcl"]
            b = json.loads(json.dumps(a))
            sd = rng.choice([1, -1])
            if self.kind == "run7":  # B: last 7 on one side (inside ±1.5σ); A: the same series with two of them flipped
                for i in range(18, 25):
                    b["points"][i] = round(cl + sd * rng.uniform(0.3, 1.4) * sig)
                a["points"] = list(b["points"])
                for i in (20, 23):
                    a["points"][i] = round(cl - sd * rng.uniform(0.3, 1.0) * sig)
                pair = (a, b)
            elif self.kind == "mono6":  # B: last 6 strictly monotone; A: one of them steps back
                start = rng.uniform(-1.4, -0.6) * sig
                vals = [round(cl + sd * (start + k * rng.uniform(0.45, 0.6) * sig)) for k in range(6)]
                b["points"][19:25] = vals
                a["points"][19:25] = vals
                a["points"][22] = vals[1]
                pair = (a, b)
            elif self.kind == "two3":  # B: two of the last 3 beyond 2σ (inside the limits); A: only one beyond
                b["points"][22] = round(cl + sd * rng.uniform(2.2, 2.8) * sig)
                b["points"][24] = round(cl + sd * rng.uniform(2.2, 2.8) * sig)
                a["points"] = list(b["points"])
                a["points"][22] = round(cl + sd * rng.uniform(0.2, 1.2) * sig)
                pair = (a, b)
            elif self.kind == "one_out":  # C: one point just beyond the limit; A: the same point just inside (numeric twin)
                i = rng.randint(14, 24)
                b["points"][i] = round(cl + sd * rng.uniform(3.15, 3.6) * sig)
                a["points"][i] = round(cl + sd * rng.uniform(2.3, 2.85) * sig)
                b["recheck"] = rng.choice(["none", "back_in"])
                if b["recheck"] == "back_in":
                    b["recheck_value"] = round(cl + rng.uniform(-1, 1) * sig)
                pair = (a, b)
            elif self.kind == "second_out":  # D: two points beyond; C: one beyond, the other just inside
                i = rng.randint(14, 22)
                for f in (a, b):
                    f["points"][i] = round(cl + sd * rng.uniform(3.15, 3.6) * sig)
                b["points"][i + 2] = round(cl + sd * rng.uniform(3.15, 3.6) * sig)
                a["points"][i + 2] = round(cl + sd * rng.uniform(1.0, 1.8) * sig)
                pair = (b, a)
            else:  # recheck: one point beyond; C recheck back inside vs D recheck still beyond
                i = rng.randint(18, 24)
                for f in (a, b):
                    f["points"][i] = round(cl + sd * rng.uniform(3.15, 3.6) * sig)
                    f["recheck"] = "back_in" if f is a else "still_out"
                a["recheck_value"] = round(cl + rng.uniform(-1, 1) * sig)
                b["recheck_value"] = round(cl + sd * rng.uniform(3.15, 3.5) * sig)
                pair = (a, b)
            want = {"run7": "AB", "mono6": "AB", "two3": "AB", "one_out": "AC", "second_out": "DC", "recheck": "CD"}[self.kind]
            def on_line(f):
                st = spc_stats(f)
                lines_ = {f["ucl"], f["lcl"], st["two_sigma_hi"], st["two_sigma_lo"]}
                return any(x in lines_ for x in f["points"] + ([f["recheck_value"]] if f["recheck_value"] is not None else []))
            if spc_gold(pair[0]) + spc_gold(pair[1]) == want and not on_line(pair[0]) and not on_line(pair[1]):
                hard = self.kind
                return pair[0], pair[1], "numeric" if self.kind != "recheck" else "fact", hard
        raise RuntimeError(f"spc {self.kind}: could not build pair")

    skip_parity = {"recheck_value"}

    def derive(self, f):
        d = spc_stats(f)
        d["out_of_limit_points"] = "、".join(f"第 {i} 點" for i in d["out_of_limit_points"]) or "無"
        if f["recheck_value"] is not None:
            d["recheck_within_limits"] = f["lcl"] <= f["recheck_value"] <= f["ucl"]
        return d

    derived_bool_zh = {"recheck_within_limits": ("複測值在管制界限內：是", "複測值在管制界限內：否")}

    def gold(self, f):
        return spc_gold(f)

    def text(self, f, d=None):
        u = f["unit"]
        t = (f"{f['line']} {f['station']} {f['chart']} SPC（CL {f['cl']}{u}、UCL {f['ucl']}、LCL {f['lcl']}），最近 25 點（舊→新）："
             + "、".join(str(x) for x in f["points"]))
        t += "；" + ("未複測" if f["recheck"] == "none" else f"已複測，複測值 {f['recheck_value']}")
        if d:
            t += (f"（2σ 線 {s(d['two_sigma_hi'])}／{s(d['two_sigma_lo'])}；超出 UCL/LCL 的點數 {d['n_out_of_limits']}（{d['out_of_limit_points']}）；"
                  f"最長同側連續 {d['longest_same_side_run']} 點；最長單向連續 {d['longest_monotone_run']} 點；任 3 點中超過 2σ 最多 {d['max_beyond_2sigma_in_3']} 點")
            if "recheck_within_limits" in d:
                t += "；" + self.derived_bool_zh["recheck_within_limits"][0 if d["recheck_within_limits"] else 1]
            t += "）"
        return t


# ---------------------------------------------------------------- p_uph_anomaly (A 是 / B 否)
class UphHourly(Fam):
    def sample(self, rng):
        tgt = rng.choice([900, 1150, 1200, 1400, 1500, 1800])
        ok = [round(tgt * rng.uniform(0.965, 1.035)) for _ in range(4)]
        h0 = rng.randint(6, 18)
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "target_uph": tgt, "hours": f"{h0:02d}:00–{h0 + 4:02d}:00", "hourly_output": ok}
        b = dict(a, hourly_output=ok[:2] + [round(tgt * rng.uniform(0.70, 0.86)) for _ in range(2)])
        return b, a, "numeric", None

    skip_parity = {"hours"}

    def derive(self, f):
        pct = [round(100 * x / f["target_uph"], 1) for x in f["hourly_output"]]
        drops = [round(100 * (f["hourly_output"][i - 1] - f["hourly_output"][i]) / f["hourly_output"][i - 1], 1) for i in range(1, len(pct))]
        return {"pct_of_target": pct, "min_pct_of_target": min(pct), "max_hour_to_hour_drop_pct": max(0.0, max(drops))}

    def gold(self, f):
        m = self.derive(f)["min_pct_of_target"]
        assert m >= 95 or m <= 90
        return "A" if m <= 90 else "B"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} 目標 UPH {f['target_uph']}；{f['hours']} 逐時產出：" + "、".join(str(x) for x in f["hourly_output"])
        if d:
            t += (f"（各小時為目標的 " + "、".join(f"{s(x)}%" for x in d["pct_of_target"]) +
                  f"；最低 {s(d['min_pct_of_target'])}%；相鄰小時最大降幅 {s(d['max_hour_to_hour_drop_pct'])}%）")
        return t


class UphAvg(Fam):
    def sample(self, rng):
        tgt = rng.choice([960, 1100, 1250, 1500, 1650])
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "target_uph": tgt, "shift_avg_uph": round(tgt * rng.uniform(0.74, 0.87))}
        b = dict(a, shift_avg_uph=round(tgt * rng.uniform(0.97, 1.04)))
        return a, b, "numeric", None

    def derive(self, f):
        return {"achievement_pct": round(100 * f["shift_avg_uph"] / f["target_uph"], 1)}

    def gold(self, f):
        return "A" if f["shift_avg_uph"] / f["target_uph"] <= 0.9 else "B"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} 貼片 UPH 目標 {f['target_uph']}，本班平均 {f['shift_avg_uph']}"
        if d:
            t += f"（達成率 {s(d['achievement_pct'])}%）"
        return t


class UphCt(Fam):
    def sample(self, rng):
        tgt = rng.choice([28.0, 32.0, 36.5, 42.0, 48.0])
        a = {"line": rng.choice(LINES), "device": rng.choice(MOUNTERS), "target_ct_s": tgt, "actual_ct_s": round(tgt * rng.uniform(1.13, 1.30), 1)}
        b = dict(a, actual_ct_s=round(tgt * rng.uniform(0.98, 1.03), 1))
        return a, b, "numeric", None

    def derive(self, f):
        return {"ct_vs_target_pct": round(100 * (f["actual_ct_s"] - f["target_ct_s"]) / f["target_ct_s"], 1)}

    def gold(self, f):
        return "A" if f["actual_ct_s"] / f["target_ct_s"] >= 1.1 else "B"

    def text(self, f, d=None):
        t = f"{f['line']} {f['device']} 目標 CT {s(f['target_ct_s'])} 秒，最近 50 片平均 CT {s(f['actual_ct_s'])} 秒"
        if d:
            t += f"（比目標 {sgn(d['ct_vs_target_pct'])}%）"
        return t


# ---------------------------------------------------------------- t_tool_gate (A 允許 / B 詢問 / C 拒絕)
ROLE_ENG = ["技術員", "製程工程師", "設備工程師"]
READ_TOOLS = [("mes.query_wo", lambda r, ln: {"line": ln, "wo": wo(r)}, "交班前確認工單完工數量"),
              ("spc.read_chart", lambda r, ln: {"line": ln, "station": r.choice(SPIS), "chart": "solder_volume_xbar", "range": "last_24h"}, "稽核前檢視錫膏體積管制圖"),
              ("aoi.export_report", lambda r, ln: {"line": ln, "station": "AOI-02", "date": "2026-11-04", "format": "csv"}, "早會要用昨天的 AOI 不良報表"),
              ("maint.list_tickets", lambda r, ln: {"line": ln, "status": "open"}, "週會前確認還沒結案的維修單")]
PARAM_TOOLS = [("reflow.set_zone_temp", "temp_c", (230, 250), lambda r, ln: {"line": ln, "oven": r.choice(REFLOWS), "zone": r.randint(3, 8)}, "首件冷焊，想把該區溫度調高"),
               ("printer.set_pressure", "pressure_kg", (5.0, 8.0), lambda r, ln: {"line": ln, "printer": r.choice(PRINTERS)}, "SPI 錫膏偏薄，想調印刷壓力"),
               ("mounter.set_speed_pct", "speed_pct", (50, 100), lambda r, ln: {"line": ln, "machine": r.choice(MOUNTERS)}, "拋料偏高，想調整置件速度"),
               ("aoi.set_threshold", "threshold", (60, 90), lambda r, ln: {"line": ln, "station": "AOI-02"}, "誤判太多，想調整判定門檻")]
REGISTRY = {"mes.query_wo": ("read", True), "spc.read_chart": ("read", True), "aoi.export_report": ("read", True), "maint.list_tickets": ("read", True),
            "reflow.set_zone_temp": ("write", True), "printer.set_pressure": ("write", True), "mounter.set_speed_pct": ("write", True),
            "aoi.set_threshold": ("write", True), "station.pause": ("write", True)}


def _inside(v, lo, hi):
    return lo <= v <= hi


class ToolFam(Fam):
    derived_bool_zh = {"reversible": ("可復原：是", "可復原：否"), "target_in_caller_lines": ("對象在呼叫者負責線別內：是", "對象在呼叫者負責線別內：否"),
                       "param_within_spec": ("參數在規格範圍內：是", "參數在規格範圍內：否")}
    skip_parity = {"context"}

    def derive(self, f):
        acc, rev = REGISTRY[f["tool"]]
        p = f["params"]
        lines = sorted({t["line"] for t in p["targets"]}) if "targets" in p else [p["line"]]
        d = {"access": "唯讀" if acc == "read" else "寫入", "reversible": rev, "lines_affected": len(lines),
             "target_in_caller_lines": all(x in f["caller_lines"] for x in lines)}
        if f.get("param_spec"):
            k, (lo, hi) = next(iter(f["param_spec"].items()))
            v = p[k]
            d["param_within_spec"] = _inside(v, lo, hi)
            d["param_vs_spec"] = "範圍內" if _inside(v, lo, hi) else ("高於上限" if v > hi else "低於下限")
            d["param_beyond_spec_by"] = 0 if _inside(v, lo, hi) else round(v - hi if v > hi else lo - v, 2)
        return d

    def gold(self, f):
        d = self.derive(f)
        if d["access"] == "唯讀":
            return "A" if d["target_in_caller_lines"] else "B"
        if f["caller_role"] == "操作員" or not d.get("param_within_spec", True) or d["lines_affected"] > 1 or not d["reversible"]:
            return "C"
        return "B"

    def text(self, f, d=None):
        p = f["params"]
        ps = "、".join(f"{k}={'/'.join(t['line'] + ':' + t['station'] for t in v) if k == 'targets' else s(v)}" for k, v in p.items())
        t = f"呼叫者：{f['caller_role']}（負責 {'、'.join(f['caller_lines'])}）\n工具：{f['tool']}\n參數：{ps}"
        if f.get("param_spec"):
            k, (lo, hi) = next(iter(f["param_spec"].items()))
            t += f"\n參數規格範圍：{k} {s(lo)}–{s(hi)}"
        if d:
            t += f"\n（工具類型：{d['access']}；{self.derived_bool_zh['reversible'][0 if d['reversible'] else 1]}；影響線數 {d['lines_affected']}；"
            t += self.derived_bool_zh["target_in_caller_lines"][0 if d["target_in_caller_lines"] else 1]
            if "param_within_spec" in d:
                t += f"；{self.derived_bool_zh['param_within_spec'][0 if d['param_within_spec'] else 1]}（{d['param_vs_spec']}，超出量 {s(d['param_beyond_spec_by'])}）"
            t += "）"
        return t + f"\n情境：{f['context']}"

    def parity_values(self, f):
        vals = [f["caller_role"], *f["caller_lines"], f["tool"]]
        for k, v in f["params"].items():
            if k == "targets":
                vals += [t["line"] for t in v] + [t["station"] for t in v]
            else:
                vals.append(s(v))
        for k, (lo, hi) in (f.get("param_spec") or {}).items():
            vals += [s(lo), s(hi)]
        return vals


class ToolParam(ToolFam):
    def sample(self, rng):
        tool, key, (lo, hi), mk, ctx = rng.choice(PARAM_TOOLS)
        ln = rng.choice(LINES)
        span = hi - lo
        isf = isinstance(lo, float)
        r = (lambda x: round(x, 1)) if isf else round
        inside = r(rng.uniform(lo + 0.15 * span, hi - 0.1 * span))
        outside = r(hi + rng.uniform(0.08, 0.35) * span) if rng.random() < 0.7 else r(lo - rng.uniform(0.08, 0.35) * span)
        base = {"caller_role": rng.choice(ROLE_ENG), "caller_lines": [ln], "tool": tool, "params": mk(rng, ln), "param_spec": {key: [lo, hi]}, "context": ctx}
        a = json.loads(json.dumps(base))
        b = json.loads(json.dumps(base))
        a["params"][key], b["params"][key] = outside, inside
        return a, b, "numeric", "param_out_of_range"


class ToolScope(ToolFam):
    def sample(self, rng):
        lines = rng.sample(LINES, rng.randint(2, 4))
        role = rng.choice(["製程工程師", "設備工程師"])
        tg = [{"line": ln, "station": rng.choice(MOUNTERS)} for ln in lines]
        ctx = rng.choice(["料號 U301 來料批異常，先把用到這顆料的機台暫停", "要做置件頭保養，先暫停相關機台"])
        base = {"caller_role": role, "caller_lines": sorted(set(LINES[:3] + lines)), "tool": "station.pause", "params": {"targets": tg}, "context": ctx}
        a = json.loads(json.dumps(base))
        b = json.loads(json.dumps(base))
        b["params"]["targets"] = tg[:1]
        return a, b, "numeric", "multi_line_scope"


class ToolRead(ToolFam):
    needs_compare = False

    def sample(self, rng):
        tool, mk, ctx = rng.choice(READ_TOOLS)
        own = sorted(rng.sample(LINES, rng.randint(1, 2)))
        other = rng.choice([x for x in LINES if x not in own])
        role = rng.choice(["操作員", "線長", "技術員", "品保工程師"])
        p = mk(rng, own[-1])
        a = {"caller_role": role, "caller_lines": own, "tool": tool, "params": p, "context": ctx}
        b = json.loads(json.dumps(a))
        b["params"]["line"] = other
        return a, b, "fact", "readonly_other_line"


class ToolRole(ToolFam):
    needs_compare = False

    def sample(self, rng):
        tool, key, (lo, hi), mk, ctx = rng.choice(PARAM_TOOLS)
        ln = rng.choice(LINES)
        v = round(rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)), 1 if isinstance(lo, float) else None)
        a = {"caller_role": "操作員", "caller_lines": [ln], "tool": tool, "params": {**mk(rng, ln), key: v}, "param_spec": {key: [lo, hi]}, "context": ctx}
        b = json.loads(json.dumps(a))
        b["caller_role"] = "技術員"
        return a, b, "fact", "role_permission"


FAMILIES = {
    "m_alarm_severity": [(SevReflow("numeric"), 12), (SevReflow("fact"), 10), (SevNg(), 10), (SevStations(), 8), (SevPaste(), 10), (SevThrow(), 10)],
    "m_alarm_category": [(CatAir(), 15), (CatVolt(), 15), (CatCode(), 30)],
    "m_needs_dispatch": [(DispRetry(), 15), (DispMetric(), 15), (DispAdjust(), 15), (DispAction(), 15)],
    "q_spc_action": [(Spc("run7"), 10), (Spc("mono6"), 8), (Spc("two3"), 8), (Spc("one_out"), 16), (Spc("second_out"), 10), (Spc("recheck"), 8)],
    "p_uph_anomaly": [(UphHourly(), 20), (UphAvg(), 20), (UphCt(), 20)],
    "t_tool_gate": [(ToolParam(), 20), (ToolScope(), 15), (ToolRead(), 13), (ToolRole(), 12)],
}

# words that would give the answer away (option labels / judgments); checked on every rendering
FORBID = {
    "m_alarm_severity": ["停線", "盡快", "不急", "嚴重", "緊急"],
    "m_alarm_category": ["機構", "電控", "物料問題", "程式問題", "環境問題", "分類", "屬於"],
    "m_needs_dispatch": ["派工", "設備工程師", "需維修", "需更換"],
    "q_spc_action": ["持續監控", "抽檢", "停線", "呼叫", "QE", "複檢"],
    "p_uph_anomaly": ["異常", "明顯", "掉產能"],
    "t_tool_gate": ["允許", "詢問", "拒絕", "權限", "不可"],
}
NOISE_KEYS = ["event_id", "uuid", "ts", "fw", "plc_addr", "op_id", "host", "seq", "shift_id", "mes_tx"]
NOISE_KEYS_TOOL = ["request_id", "trace_id", "ts", "agent_ver", "session_id", "client_ip", "seq", "model_route"]


def noise(rng, keys):
    out = {}
    for k in rng.sample(keys, rng.randint(6, min(10, len(keys)))):
        out[k] = {
            "event_id": f"EV-20261105-{rng.randint(0, 999999):06d}", "uuid": "%08x-%04x-4%03x-%04x-%012x" % (rng.getrandbits(32), rng.getrandbits(16), rng.getrandbits(12), rng.getrandbits(16), rng.getrandbits(48)),
            "ts": f"2026-11-05T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d}Z",
            "fw": f"RX-{rng.randint(3, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 30)}", "plc_addr": f"D{rng.randint(1000, 9999)}", "op_id": f"A{rng.randint(10000, 99999)}",
            "host": f"smt-edge-{rng.randint(1, 40):02d}", "seq": rng.randint(100000, 9999999), "shift_id": f"S{rng.randint(1, 3)}-1105", "mes_tx": f"TX{rng.getrandbits(40):010x}",
            "request_id": f"req_{rng.getrandbits(64):016x}", "trace_id": f"{rng.getrandbits(128):032x}", "agent_ver": f"line-agent/{rng.randint(1, 3)}.{rng.randint(0, 20)}",
            "session_id": f"sess_{rng.getrandbits(48):012x}", "client_ip": f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
            "model_route": rng.choice(["edge-a", "edge-b"]),
        }[k]
    return out


def jdump(d):
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def public(f):
    return {k: v for k, v in f.items() if not k.startswith("_")}


def parity_values(fam, f):
    if hasattr(fam, "parity_values"):
        return fam.parity_values(f)
    vals = []
    for k, v in public(f).items():
        if k in fam.skip_parity or v is None:
            continue
        if k in fam.bool_zh:
            vals.append(fam.bool_zh[k][0 if v else 1])
        elif k in fam.enum_zh:
            vals.append(fam.enum_zh[k][v])
        elif isinstance(v, list):
            vals += [s(x) for x in v]
        else:
            vals.append(s(v))
    return vals


def derived_values(fam, d):
    vals = []
    for k, v in d.items():
        if isinstance(v, bool):
            vals.append(fam.derived_bool_zh[k][0 if v else 1])
        elif isinstance(v, list):
            vals += [s(x) for x in v]
        else:
            vals.append(s(v))
    return vals


def render(task, fam, f, rng):
    d = fam.derive(f)
    pub = public(f)
    j0 = pub
    j1 = {**pub, **{k: v for k, v in d.items()}}
    nz = noise(rng, NOISE_KEYS_TOOL if task == "t_tool_gate" else NOISE_KEYS)
    items = list(nz.items())
    h = len(items) // 2
    jn = {**dict(items[:h]), **pub, **dict(items[h:])}
    out = {"T0": fam.text(f), "T1": fam.text(f, d) if d else fam.text(f), "J0": jdump(j0), "J1": jdump(j1), "JN": jdump(jn)}
    # ---- checks
    pv = parity_values(fam, f)
    for r in RENDERINGS:
        for v in pv:
            if r.startswith("T"):
                assert v in out[r], (task, r, v, out[r])
        for w in FORBID[task]:
            assert w not in out[r], (task, r, w, out[r])
    for v in pv:
        if not isinstance(v, str) or not any("一" <= c <= "鿿" for c in v):  # Chinese phrases are the text form of a JSON bool/enum
            assert v in out["J0"] or v.strip("0") in out["J0"], (task, "J0", v, out["J0"])
    for v in derived_values(fam, d):
        assert v in out["T1"], (task, "T1", v, out["T1"])
    assert out["T0"] in out["T1"].replace("（", "").replace("）", "") or all(v in out["T1"] for v in pv)
    return out, d


def build(seed):
    rng = random.Random(seed)
    Q = questions()
    rows = {r: {t: [] for t in TASKS} for r in RENDERINGS}
    meta = {}
    for task in TASKS:
        i = 0
        for fam, n in FAMILIES[task]:
            for _ in range(n):
                a, b, kind, hard = fam.sample(rng)
                ga, gb = fam.gold(a), fam.gold(b)
                assert ga != gb, (task, type(fam).__name__, ga, gb, a, b)
                pid = f"{task}-v23p{i:03d}"
                for suffix, f, g in (("a", a, ga), ("b", b, gb)):
                    rid = f"{task}-v23-{i:03d}{suffix}"
                    out, d = render(task, fam, f, rng)
                    for r in RENDERINGS:
                        rows[r][task].append({"id": rid, "task": task, "state": out[r], "gold": g, "question": Q[task],
                                              "difficulty": "hard" if fam.needs_compare else "easy", "lang": "zh", "pair_id": pid,
                                              "hard_type": hard, "needs_compare": fam.needs_compare, "family": type(fam).__name__ + (f":{fam.kind}" if hasattr(fam, "kind") else ""),
                                              "twin_kind": kind, "event": public(f), "derived": d})
                i += 1
        meta[task] = i
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12)
    a = ap.parse_args()
    rows = build(a.seed)
    for r in RENDERINGS:
        os.makedirs(os.path.join(HERE, r), exist_ok=True)
        for t in TASKS:
            with open(os.path.join(HERE, r, f"{t}.jsonl"), "w", encoding="utf-8") as fh:
                for x in rows[r][t]:
                    fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    lines = [f"# data/v23 MANIFEST\n", f"`python3 data/v23/gen_v23.py --seed {a.seed}`。gold 全部由事件與規則決定（不是模型標的）。"
             "五種寫法同一個 `id`，只差目錄。每列另有 `needs_compare`、`family`、`twin_kind`、`event`（事實）、`derived`（衍生欄位）。\n",
             "| task | n | gold 分布 | needs_compare | numeric twin 對數 | families |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        rr = rows["T0"][t]
        g = dict(sorted(collections.Counter(x["gold"] for x in rr).items()))
        nc = sum(x["needs_compare"] for x in rr)
        nt = sum(x["twin_kind"] == "numeric" for x in rr) // 2
        fams = dict(collections.Counter(x["family"] for x in rr))
        lines.append(f"| {t} | {len(rr)} | {g} | {nc} | {nt} / {len(rr) // 2} | {', '.join(f'{k} {v // 2}' for k, v in fams.items())} |")
    lines += ["", "平均字元數（state）：", "", "| task | " + " | ".join(RENDERINGS) + " |", "|---|" + "---|" * len(RENDERINGS)]
    for t in TASKS:
        lines.append(f"| {t} | " + " | ".join(str(round(sum(len(x['state']) for x in rows[r][t]) / len(rows[r][t]))) for r in RENDERINGS) + " |")
    lines += ["", "檢查（產生時 assert）：五種寫法事實一致（數字逐字、布林／列舉用中文詞）；T1 含所有衍生值；任何寫法都不含選項標籤或判斷詞（`FORBID`）；每組 twin 的 gold 不同。",
              "T1 是 T0 加上括號裡的衍生值（保留原始值），所以 T1 與 J1 資訊相同；handoff 範例的 T1 省掉原始值，這裡沒有照做，以維持資訊對等。"]
    open(os.path.join(HERE, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
