"""Easy generators for q_spc_action (A 持續監控 / B 抽檢 / C 停線複檢 / D 呼叫 QE).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair that
differs in ONE fact so the gold flips. Gold comes from the template spec, never from post-hoc judgement:
A = every point within the control limits and no run/trend rule; B = a run/trend rule fires (7 points one
side of CL, 6 points one direction, 2 of 3 beyond 2σ) but nothing beyond the limits; C = a single point
beyond UCL/LCL or USL/LSL; D = beyond the limits AND (several consecutive points, several stations or part
numbers, or still out after a recheck). Point series are built numerically from (CL, σ) so the rule that
fires is guaranteed by construction.
"""
from ._common import LINES, PRINTERS, MOUNTERS, REFLOWS, SPIS, PNS, hhmm, pick

# chart spec: (name, CL, sigma, unit, decimals, signed)
_PASTE = ("錫膏厚度", 150, 7, "µm", 0, False)
_VOL = ("錫膏體積", 100, 8, "%", 0, False)
_PEAK = ("回焊峰溫", 245, 2, "°C", 1, False)
_OFFSET = ("貼片 X 偏移", 0, 15, "µm", 0, True)


def _r(v, nd):
    return round(v, nd) if nd else int(round(v))


def _f(v, spec):
    nd, signed = spec[4], spec[5]
    return f"{v:+.{nd}f}" if signed and v != 0 else f"{v:.{nd}f}"


def _lim(spec):
    cl, sig = spec[1], spec[2]
    return cl + 3 * sig, cl - 3 * sig


def _hdr(spec):
    ucl, lcl = _lim(spec)
    return f"{spec[0]} SPC（CL {_f(spec[1], spec)}{spec[3]}、UCL {_f(ucl, spec)}、LCL {_f(lcl, spec)}）"


def _join(pts, spec):
    return "、".join(_f(p, spec) for p in pts)


def _join_en(pts, spec):
    return " ".join(_f(p, spec) for p in pts)


def _stable(rng, spec, n):
    """n points within ±1.7σ, no 4 in a row on one side, no 5 monotone: no rule fires."""
    cl, sig, nd = spec[1], spec[2], spec[4]
    while True:
        pts = [_r(cl + rng.uniform(-1.7, 1.7) * sig, nd) for _ in range(n)]
        sides = [(p > cl) - (p < cl) for p in pts]
        ok = len(set(pts)) >= n - 2
        for i in range(3, n):
            if sides[i] != 0 and sides[i - 3:i + 1] == [sides[i]] * 4:
                ok = False
        for i in range(4, n):
            w = pts[i - 4:i + 1]
            if all(w[j] < w[j + 1] for j in range(4)) or all(w[j] > w[j + 1] for j in range(4)):
                ok = False
        if ok:
            return pts


def _run(rng, spec, n, side):
    """n points all on one side of CL, 0.5σ..2.2σ: run rule, within limits."""
    cl, sig, nd = spec[1], spec[2], spec[4]
    return [_r(cl + side * rng.uniform(0.5, 2.2) * sig, nd) for _ in range(n)]


def _trend(rng, spec, n, up):
    """n strictly monotone points from ∓1.8σ, steps 0.35..0.6σ: trend rule, within limits."""
    cl, sig, nd = spec[1], spec[2], spec[4]
    v = cl - 1.8 * sig if up else cl + 1.8 * sig
    pts = []
    for _ in range(n):
        pts.append(_r(v, nd))
        v += (1 if up else -1) * rng.uniform(0.35, 0.6) * sig
    return pts


def _two_of_three(rng, spec, side):
    """5 stable points then 3 points with 2 beyond 2σ (2.15..2.7σ) on one side: rule, within limits."""
    cl, sig, nd = spec[1], spec[2], spec[4]
    tail = [_r(cl + side * rng.uniform(2.15, 2.7) * sig, nd) for _ in range(2)]
    tail.insert(rng.randint(0, 2), _r(cl + rng.uniform(-0.8, 0.8) * sig, nd))
    return _stable(rng, spec, 5) + tail


def _ooc(rng, spec, side, k=1):
    """k points 3.4..4.3σ beyond CL: outside the control limit."""
    cl, sig, nd = spec[1], spec[2], spec[4]
    return [_r(cl + side * rng.uniform(3.4, 4.3) * sig, nd) for _ in range(k)]


def _side_word(side):
    return "上方" if side > 0 else "下方"


def _lim_word(side):
    return "UCL" if side > 0 else "LCL"


# ---------------------------------------------------------------- zh pairs

def paste_run(rng):  # A / B
    line, p, spec = pick(rng, LINES), pick(rng, PRINTERS), _PASTE
    side = rng.choice([1, -1])
    return (f"{line} {p} {_hdr(spec)} 最近 8 點：{_join(_stable(rng, spec, 8), spec)}，全部在界限內，點在 CL 兩側交錯、無連續同側或單向趨勢", "A",
            f"{line} {p} {_hdr(spec)} 最近 7 點：{_join(_run(rng, spec, 7, side), spec)}，連續 7 點都在 CL {_side_word(side)}，但沒有任何一點超出 {_lim_word(side)}", "B")


def paste_ooc(rng):  # B / C
    line, p, spec = pick(rng, LINES), pick(rng, PRINTERS), _PASTE
    up = rng.choice([True, False])
    tr = _trend(rng, spec, 6, up)
    pts = _stable(rng, spec, 5) + _ooc(rng, spec, 1 if up else -1)
    return (f"{line} {p} {_hdr(spec)} 最近 6 點：{_join(tr, spec)}，連續 6 點單向{'上升' if up else '下降'}，最後一點 {_f(tr[-1], spec)} 仍在 {_lim_word(1 if up else -1)} 內", "B",
            f"{line} {p} {_hdr(spec)} 最近 6 點：{_join(pts, spec)}，前 5 點在界限內，最後一點 {_f(pts[-1], spec)} 超出 {_lim_word(1 if up else -1)}，本班第一次超限、只有這一點", "C")


def vol_multi(rng):  # C / D
    line, s, pn, spec = pick(rng, LINES), pick(rng, SPIS), pick(rng, PNS), _VOL
    side = rng.choice([1, -1])
    one = _stable(rng, spec, 5) + _ooc(rng, spec, side)
    three = _stable(rng, spec, 3) + _ooc(rng, spec, side, 3)
    return (f"{line} {s} 料號 {pn} {_hdr(spec)} 最近 6 點：{_join(one, spec)}，只有最後 1 點 {_f(one[-1], spec)} 超出 {_lim_word(side)}，其餘在界限內", "C",
            f"{line} {s} 料號 {pn} {_hdr(spec)} 最近 6 點：{_join(three, spec)}，最後 3 點連續超出 {_lim_word(side)}", "D")


def vol_trend(rng):  # A / B
    line, s, pn, spec = pick(rng, LINES), pick(rng, SPIS), pick(rng, PNS), _VOL
    up = rng.choice([True, False])
    tr = _trend(rng, spec, 6, up)
    return (f"{line} {s} 料號 {pn} {_hdr(spec)} 最近 8 點：{_join(_stable(rng, spec, 8), spec)}，全部在界限內且在 CL 兩側隨機分布、無趨勢", "A",
            f"{line} {s} 料號 {pn} {_hdr(spec)} 最近 6 點：{_join(tr, spec)}，連續 6 點單向{'上升' if up else '下降'}，全部仍在 UCL/LCL 內", "B")


def reflow_peak(rng):  # A / C
    line, r, spec = pick(rng, LINES), pick(rng, REFLOWS), _PEAK
    side = rng.choice([1, -1])
    pts = _stable(rng, spec, 5) + _ooc(rng, spec, side)
    return (f"{line} 回焊爐 {r} {_hdr(spec)} 本班 8 次測溫：{_join(_stable(rng, spec, 8), spec)}，全部在界限內、無連續同側或單向趨勢", "A",
            f"{line} 回焊爐 {r} {_hdr(spec)} 本班 6 次測溫：{_join(pts, spec)}，最後一次 {_f(pts[-1], spec)} 超出 {_lim_word(side)}，前 5 次在界限內，僅這一點超限", "C")


def recheck(rng):  # C / D
    line, p, spec, t = pick(rng, LINES), pick(rng, PRINTERS), _PASTE, hhmm(rng)
    side = rng.choice([1, -1])
    prev, x = _stable(rng, spec, 5), _ooc(rng, spec, side)[0]
    x2 = _ooc(rng, spec, side)[0]
    return (f"{line} {p} {_hdr(spec)} {t} 量測 {_f(x, spec)} 超出 {_lim_word(side)}，前 5 點 {_join(prev, spec)} 在界限內，此為本班第一次超限，尚未複測", "C",
            f"{line} {p} {_hdr(spec)} {t} 量測 {_f(x, spec)} 超出 {_lim_word(side)}，停站{'清潔鋼網' if side < 0 else '調整刮刀壓力'}後複測 {_f(x2, spec)} 仍超出 {_lim_word(side)}", "D")


def multi_station(rng):  # C / D
    line, s, pn, spec = pick(rng, LINES), pick(rng, SPIS), pick(rng, PNS), _VOL
    line2 = pick(rng, [l for l in LINES if l != line])
    s2 = pick(rng, [q for q in SPIS if q != s] or SPIS)
    side = rng.choice([1, -1])
    x, x2 = _ooc(rng, spec, side)[0], _ooc(rng, spec, side)[0]
    return (f"{line} {s} 料號 {pn} {_hdr(spec)} 最新一點 {_f(x, spec)} 超出 {_lim_word(side)}，同線其他料號與 {line2} {s2} 同料號同時段都在界限內", "C",
            f"{line} {s} 料號 {pn} {_hdr(spec)} 最新一點 {_f(x, spec)} 超出 {_lim_word(side)}，{line2} {s2} 同料號同時段也量到 {_f(x2, spec)} 超出 {_lim_word(side)}，兩站同時超限", "D")


def multi_pn(rng):  # A / D
    line, p, spec = pick(rng, LINES), pick(rng, PRINTERS), _PASTE
    pns = rng.sample(PNS, 3)
    side = rng.choice([1, -1])
    ok = [_r(spec[1] + rng.uniform(-1.5, 1.5) * spec[2], 0) for _ in range(3)]
    bad = _ooc(rng, spec, side, 3)
    names = "、".join(pns)
    return (f"{line} {p} 三個料號 {names} 的 {_hdr(spec)} 最新一點分別為 {_join(ok, spec)}，皆在界限內，各料號近 8 點無連續同側或趨勢", "A",
            f"{line} {p} 三個料號 {names} 的 {_hdr(spec)} 最新一點分別為 {_join(bad, spec)}，三個料號同時超出 {_lim_word(side)}", "D")


def two_of_three(rng):  # A / B
    line, m, spec = pick(rng, LINES), pick(rng, MOUNTERS), _OFFSET
    side = rng.choice([1, -1])
    two = _two_of_three(rng, spec, side)
    return (f"{line} {m} {_hdr(spec)} 最近 8 點：{_join(_stable(rng, spec, 8), spec)}，全部在 ±2σ（±{2 * spec[2]}µm）內、無連串", "A",
            f"{line} {m} {_hdr(spec)} 最近 8 點：{_join(two, spec)}，最後 3 點中有 2 點超過 {'+' if side > 0 else '-'}2σ（{'+' if side > 0 else '-'}{2 * spec[2]}µm）但都未超過 {_lim_word(side)}", "B")


def cpk(rng):  # A / C
    line, p, spec, n = pick(rng, LINES), pick(rng, PRINTERS), _PASTE, rng.randint(20, 40)
    k, x = rng.randint(5, n - 2), _ooc(rng, spec, -1)[0]
    return (f"{line} {p} {_hdr(spec)} 本班 {n} 點全部在界限內、無連串，Cpk {rng.choice([1.45, 1.52, 1.61, 1.67, 1.78, 1.9])}", "A",
            f"{line} {p} {_hdr(spec)} 本班 {n} 點中第 {k} 點 {_f(x, spec)} 超出 LCL，其餘 {n - 1} 點在界限內，Cpk 降至 {rng.choice([0.72, 0.8, 0.85, 0.91, 0.95])}", "C")


def trend_then_ooc(rng):  # B / D
    line, r, spec = pick(rng, LINES), pick(rng, REFLOWS), _PEAK
    tr = _trend(rng, spec, 6, True)
    head = _trend(rng, spec, 3, True)
    tail = sorted(_ooc(rng, spec, 1, 3))
    return (f"{line} 回焊爐 {r} {_hdr(spec)} 最近 6 次測溫：{_join(tr, spec)}，連續 6 點上升，最高 {_f(tr[-1], spec)} 仍低於 UCL", "B",
            f"{line} 回焊爐 {r} {_hdr(spec)} 最近 6 次測溫：{_join(head + tail, spec)}，連續 6 點上升，最後 3 點 {_join(tail, spec)} 連續超出 UCL", "D")


def run_then_ooc(rng):  # B / D
    line, s, pn, spec = pick(rng, LINES), pick(rng, SPIS), pick(rng, PNS), _VOL
    side = rng.choice([1, -1])
    run = _run(rng, spec, 7, side)
    mixed = _run(rng, spec, 4, side) + _ooc(rng, spec, side, 3)
    return (f"{line} {s} 料號 {pn} {_hdr(spec)} 最近 7 點：{_join(run, spec)}，連續 7 點在 CL {_side_word(side)}，全部未超出 {_lim_word(side)}", "B",
            f"{line} {s} 料號 {pn} {_hdr(spec)} 最近 7 點：{_join(mixed, spec)}，連續 7 點在 CL {_side_word(side)}，且最後 3 點連續超出 {_lim_word(side)}", "D")


PAIRS_ZH = [paste_run, paste_ooc, vol_multi, vol_trend, reflow_peak, recheck, multi_station, multi_pn,
            two_of_three, cpk, trend_then_ooc, run_then_ooc]


# ---------------------------------------------------------------- en pairs

def _en_hdr(spec, name):
    ucl, lcl = _lim(spec)
    unit = {"µm": "UM", "%": "%", "°C": "C"}[spec[3]]
    return f"{name} CL {_f(spec[1], spec)}{unit} UCL {_f(ucl, spec)} LCL {_f(lcl, spec)}"


def en_run(rng):  # A / B
    p, spec = pick(rng, PRINTERS), _PASTE
    side = rng.choice([1, -1])
    return (f"SPC {p} {_en_hdr(spec, 'PASTE HEIGHT')} LAST 8 PTS: {_join_en(_stable(rng, spec, 8), spec)} ALL WITHIN LIMITS NO RUN NO TREND", "A",
            f"SPC {p} {_en_hdr(spec, 'PASTE HEIGHT')} LAST 7 PTS: {_join_en(_run(rng, spec, 7, side), spec)} 7 CONSECUTIVE POINTS {'ABOVE' if side > 0 else 'BELOW'} CL WITHIN LIMITS", "B")


def en_ooc(rng):  # B / C
    s, pn, spec = pick(rng, SPIS), pick(rng, PNS), _VOL
    up = rng.choice([True, False])
    tr = _trend(rng, spec, 6, up)
    pts = _stable(rng, spec, 5) + _ooc(rng, spec, 1 if up else -1)
    return (f"SPC {s} {pn} {_en_hdr(spec, 'PASTE VOLUME')} LAST 6 PTS: {_join_en(tr, spec)} 6 CONSECUTIVE {'RISING' if up else 'FALLING'} ALL WITHIN LIMITS", "B",
            f"SPC {s} {pn} {_en_hdr(spec, 'PASTE VOLUME')} LAST 6 PTS: {_join_en(pts, spec)} LAST PT {_f(pts[-1], spec)} {'ABOVE UCL' if up else 'BELOW LCL'} SINGLE POINT FIRST OCCURRENCE", "C")


def en_multi(rng):  # C / D
    m, spec = pick(rng, MOUNTERS), _OFFSET
    side = rng.choice([1, -1])
    one = _stable(rng, spec, 5) + _ooc(rng, spec, side)
    three = _stable(rng, spec, 3) + _ooc(rng, spec, side, 3)
    return (f"SPC {m} {_en_hdr(spec, 'PLACEMENT X OFFSET')} LAST 6 PTS: {_join_en(one, spec)} 1 PT {'ABOVE UCL' if side > 0 else 'BELOW LCL'} OTHERS WITHIN LIMITS", "C",
            f"SPC {m} {_en_hdr(spec, 'PLACEMENT X OFFSET')} LAST 6 PTS: {_join_en(three, spec)} 3 CONSECUTIVE PTS {'ABOVE UCL' if side > 0 else 'BELOW LCL'}", "D")


def en_peak(rng):  # A / C
    r, spec = pick(rng, REFLOWS), _PEAK
    side = rng.choice([1, -1])
    pts = _stable(rng, spec, 5) + _ooc(rng, spec, side)
    return (f"SPC REFLOW {r} {_en_hdr(spec, 'PEAK TEMP')} SHIFT PROFILES: {_join_en(_stable(rng, spec, 8), spec)} ALL WITHIN LIMITS NO RUN", "A",
            f"SPC REFLOW {r} {_en_hdr(spec, 'PEAK TEMP')} SHIFT PROFILES: {_join_en(pts, spec)} LAST PT {_f(pts[-1], spec)} {'ABOVE UCL' if side > 0 else 'BELOW LCL'} SINGLE POINT", "C")


def en_recheck(rng):  # A / D
    p, spec, t = pick(rng, PRINTERS), _PASTE, hhmm(rng)
    ok1, ok2 = [_r(spec[1] + rng.uniform(-1.5, 1.5) * spec[2], 0) for _ in range(2)]
    x, x2 = _ooc(rng, spec, -1)[0], _ooc(rng, spec, -1)[0]
    return (f"SPC {p} {_en_hdr(spec, 'PASTE HEIGHT')} {t} PT {_f(ok1, spec)} WITHIN LIMITS RECHECK {_f(ok2, spec)} WITHIN LIMITS NO RUN", "A",
            f"SPC {p} {_en_hdr(spec, 'PASTE HEIGHT')} {t} PT {_f(x, spec)} BELOW LCL STATION STOPPED STENCIL CLEANED RECHECK {_f(x2, spec)} STILL BELOW LCL", "D")


PAIRS_EN = [en_run, en_ooc, en_multi, en_peak, en_recheck]
