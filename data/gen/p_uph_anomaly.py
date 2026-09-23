"""Easy generators for p_uph_anomaly (A 是 / B 否).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair with the same
line/machine/target where only the numeric series differs so the gold flips. Series are built from the
target so gold is guaranteed by construction: A = every point ≥12% below target, or a sudden drop ≥20%
at some point, or intermittent points ≥22% below; B = every point within ±4% of target (CT: within ±4%
of target CT). The stated percentage is computed from the generated numbers.
"""
from ._common import LINES, MOUNTERS, PRINTERS, TESTERS, hhmm, pick


def _rd(v, t):
    """UPH-scale targets round to integers; CT-scale (<100 s) keep one decimal so ±4% stays exact."""
    return round(v, 1) if t < 100 else int(round(v))


def _normal(rng, t, n):
    return [_rd(t * (1 + rng.uniform(-0.04, 0.04)), t) for _ in range(n)]


def _low(rng, t, n, lo=0.12, hi=0.25):
    return [_rd(t * (1 - rng.uniform(lo, hi)), t) for _ in range(n)]


def _long(rng, t, n, lo=0.12, hi=0.30):
    return [_rd(t * (1 + rng.uniform(lo, hi)), t) for _ in range(n)]


def _drop(rng, t, n, k):
    return _normal(rng, t, k) + _low(rng, t, n - k, 0.22, 0.38)


def _intermittent(rng, t, n):
    out = []
    for i in range(n):
        out += _normal(rng, t, 1) if i % 2 == 0 else _low(rng, t, 1, 0.22, 0.40)
    return out


def _pct_below(t, xs):
    return round((t - sum(xs) / len(xs)) / t * 100)


def _pct_diff(t, xs):
    return round((sum(xs) / len(xs) - t) / t * 100)


def _target(rng):
    return rng.choice([900, 960, 1000, 1080, 1150, 1200, 1250, 1320, 1400, 1500])


def _j(xs):
    return "、".join(str(x) for x in xs)


def _hours(rng, n):
    h = rng.randint(6, 20 - n)
    return [f"{h + i:02d}:00" for i in range(n)]


# ---------------------------------------------------------------- zh pairs

def hourly(rng):  # A / B
    line, t = pick(rng, LINES), _target(rng)
    a, b = _low(rng, t, 4), _normal(rng, t, 4)
    return (f"{line} 最近 4 小時 UPH：{_j(a)}，目標 {t}，平均低於目標 {_pct_below(t, a)}%", "A",
            f"{line} 最近 4 小時 UPH：{_j(b)}，目標 {t}，平均與目標差 {_pct_diff(t, b):+d}%", "B")


def ct_boards(rng):  # A / B
    line, m, t = pick(rng, LINES), pick(rng, MOUNTERS), rng.choice([32, 36, 40, 45, 48, 52, 60])
    a, b = _long(rng, t, 5), _normal(rng, t, 5)
    return (f"{line} {m} 每板 CT 目標 {t}s，最近 5 板：{_j(a)}s，平均較目標拉長 {abs(_pct_diff(t, a))}%", "A",
            f"{line} {m} 每板 CT 目標 {t}s，最近 5 板：{_j(b)}s，平均與目標差 {_pct_diff(t, b):+d}%", "B")


def shift_total(rng):  # A / B
    line, t, h = pick(rng, LINES), _target(rng), rng.choice([8, 10, 12])
    a, b = _low(rng, t, 1)[0], _normal(rng, t, 1)[0]
    shift = rng.choice(["早班", "中班", "夜班"])
    return (f"{line} {shift} {h} 小時累計產出 {a * h} 片，目標 {t * h} 片（平均 UPH {a} vs 目標 {t}）", "A",
            f"{line} {shift} {h} 小時累計產出 {b * h} 片，目標 {t * h} 片（平均 UPH {b} vs 目標 {t}）", "B")


def sudden_drop(rng):  # A / B
    line, t, n = pick(rng, LINES), _target(rng), 6
    k = rng.randint(2, 4)
    hs = _hours(rng, n)
    a, b = _drop(rng, t, n, k), _normal(rng, t, n)
    return (f"{line} UPH {hs[0]}–{hs[-1]} 逐時：{_j(a)}，目標 {t}，{hs[k]} 起掉到 {a[k]}（較目標低 {_pct_below(t, a[k:])}%）", "A",
            f"{line} UPH {hs[0]}–{hs[-1]} 逐時：{_j(b)}，目標 {t}，全時段在目標 ±4% 內", "B")


def intermittent(rng):  # A / B
    line, t, n = pick(rng, LINES), _target(rng), 6
    hs = _hours(rng, n)
    a, b = _intermittent(rng, t, n), _normal(rng, t, n)
    return (f"{line} UPH {hs[0]}–{hs[-1]} 逐時：{_j(a)}，目標 {t}，每隔一小時掉到 {min(a)}～{max(a[1::2])}", "A",
            f"{line} UPH {hs[0]}–{hs[-1]} 逐時：{_j(b)}，目標 {t}，逐時最低 {min(b)}", "B")


def mounter_uph(rng):  # A / B
    line, m, t = pick(rng, LINES), pick(rng, MOUNTERS), _target(rng)
    a, b = _low(rng, t, 1)[0], _normal(rng, t, 1)[0]
    return (f"{line} {m} 貼片 UPH 目標 {t}，本班平均 {a}，達成率 {round(a / t * 100)}%", "A",
            f"{line} {m} 貼片 UPH 目標 {t}，本班平均 {b}，達成率 {round(b / t * 100)}%", "B")


def daily(rng):  # A / B
    line, t = pick(rng, LINES), _target(rng) * 20
    a, b = _low(rng, t, 5), _normal(rng, t, 5)
    return (f"{line} 本週一到五日產出：{_j(a)} 片，日目標 {t} 片，五天平均低於目標 {_pct_below(t, a)}%", "A",
            f"{line} 本週一到五日產出：{_j(b)} 片，日目標 {t} 片，五天平均與目標差 {_pct_diff(t, b):+d}%", "B")


def tester_ct(rng):  # A / B
    line, tester, t = pick(rng, LINES), pick(rng, TESTERS), rng.choice([45, 60, 75, 90, 120])
    a, b = _long(rng, t, 6), _normal(rng, t, 6)
    return (f"{line} {tester} 每片測試 CT 目標 {t}s，近 6 片：{_j(a)}s，平均 {round(sum(a) / 6, 1)}s，較目標拉長 {abs(_pct_diff(t, a))}%", "A",
            f"{line} {tester} 每片測試 CT 目標 {t}s，近 6 片：{_j(b)}s，平均 {round(sum(b) / 6, 1)}s，與目標差 {_pct_diff(t, b):+d}%", "B")


def after_changeover(rng):  # A / B
    line, t = pick(rng, LINES), _target(rng)
    tm = hhmm(rng)
    a, b = _low(rng, t, 3), _normal(rng, t, 3)
    return (f"{line} {tm} 換線完成後 3 小時 UPH：{_j(a)}，目標 {t}，三小時都低於目標 {_pct_below(t, [max(a)])}% 以上", "A",
            f"{line} {tm} 換線完成後 3 小時 UPH：{_j(b)}，目標 {t}，三小時皆在目標 ±4% 內", "B")


def printer_ct(rng):  # A / B
    line, p, t = pick(rng, LINES), pick(rng, PRINTERS), rng.choice([18, 20, 22, 25, 28, 30])
    a, b = _long(rng, t, 6), _normal(rng, t, 6)
    return (f"{line} {p} 印刷 CT 目標 {t}s，最近 6 板：{_j(a)}s，每板都比目標多 {round(min(a) - t, 1)}s 以上", "A",
            f"{line} {p} 印刷 CT 目標 {t}s，最近 6 板：{_j(b)}s，最大偏差 {round(max(abs(x - t) for x in b), 1)}s", "B")


def ct_vs_prev_shift(rng):  # A / B
    line, t = pick(rng, LINES), rng.choice([35, 40, 42, 48, 55, 65])
    prev = _normal(rng, t, 1)[0]
    a, b = _long(rng, t, 1)[0], _normal(rng, t, 1)[0]
    return (f"{line} 線體 CT 目標 {t}s，前班平均 {prev}s，本班平均 {a}s，較目標拉長 {round((a - t) / t * 100)}%", "A",
            f"{line} 線體 CT 目標 {t}s，前班平均 {prev}s，本班平均 {b}s，與目標差 {round((b - t) / t * 100):+d}%", "B")


def bottleneck(rng):  # A / B
    line, t = pick(rng, LINES), _target(rng)
    p, m = pick(rng, PRINTERS), pick(rng, MOUNTERS)
    ok = _normal(rng, t, 2)
    low = _low(rng, t, 1)[0]
    ok2 = _normal(rng, t, 3)
    return (f"{line} 各站 UPH：{p} {ok[0]}、{m} {low}、回焊 {ok[1]}，線體 UPH 由最慢站決定 = {low}，目標 {t}（低 {round((t - low) / t * 100)}%）", "A",
            f"{line} 各站 UPH：{p} {ok2[0]}、{m} {ok2[1]}、回焊 {ok2[2]}，線體 UPH 由最慢站決定 = {min(ok2)}，目標 {t}（差 {round((min(ok2) - t) / t * 100):+d}%）", "B")


PAIRS_ZH = [hourly, ct_boards, shift_total, sudden_drop, intermittent, mounter_uph, daily, tester_ct,
            after_changeover, printer_ct, ct_vs_prev_shift, bottleneck]


# ---------------------------------------------------------------- en pairs

def _ln(line):
    return line.replace("SMT-", "")


def _js(xs):
    return " ".join(str(x) for x in xs)


def en_hourly(rng):  # A / B
    line, t = _ln(pick(rng, LINES)), _target(rng)
    a, b = _low(rng, t, 4), _normal(rng, t, 4)
    return (f"UPH {line} LAST 4H: {_js(a)} TARGET {t} AVG -{_pct_below(t, a)}%", "A",
            f"UPH {line} LAST 4H: {_js(b)} TARGET {t} AVG {_pct_diff(t, b):+d}%", "B")


def en_ct(rng):  # A / B
    m, t = pick(rng, MOUNTERS), rng.choice([32, 36, 40, 45, 48, 52, 60])
    a, b = _long(rng, t, 6), _normal(rng, t, 6)
    return (f"CT {m} TARGET {t}S LAST 6 BOARDS: {_js(a)}S AVG +{abs(_pct_diff(t, a))}%", "A",
            f"CT {m} TARGET {t}S LAST 6 BOARDS: {_js(b)}S AVG {_pct_diff(t, b):+d}%", "B")


def en_shift(rng):  # A / B
    line, t, h = _ln(pick(rng, LINES)), _target(rng), rng.choice([8, 10, 12])
    a, b = _low(rng, t, 1)[0], _normal(rng, t, 1)[0]
    return (f"SHIFT OUTPUT {line} {h}H {a * h} PCS TARGET {t * h} PCS UPH {a} TARGET {t}", "A",
            f"SHIFT OUTPUT {line} {h}H {b * h} PCS TARGET {t * h} PCS UPH {b} TARGET {t}", "B")


def en_drop(rng):  # A / B
    line, t, n = _ln(pick(rng, LINES)), _target(rng), 6
    k = rng.randint(2, 4)
    hs = _hours(rng, n)
    a, b = _drop(rng, t, n, k), _normal(rng, t, n)
    return (f"UPH {line} HOURLY {hs[0][:2]}-{hs[-1][:2]}: {_js(a)} TARGET {t} FROM {hs[k]} -{_pct_below(t, a[k:])}%", "A",
            f"UPH {line} HOURLY {hs[0][:2]}-{hs[-1][:2]}: {_js(b)} TARGET {t} MIN {min(b)} MAX {max(b)}", "B")


def en_intermittent(rng):  # A / B
    line, t, n = _ln(pick(rng, LINES)), _target(rng), 6
    hs = _hours(rng, n)
    a, b = _intermittent(rng, t, n), _normal(rng, t, n)
    return (f"UPH {line} HOURLY {hs[0][:2]}-{hs[-1][:2]}: {_js(a)} TARGET {t} EVERY OTHER HOUR -{_pct_below(t, a[1::2])}%", "A",
            f"UPH {line} HOURLY {hs[0][:2]}-{hs[-1][:2]}: {_js(b)} TARGET {t} MIN {min(b)} MAX {max(b)}", "B")


PAIRS_EN = [en_hourly, en_ct, en_shift, en_drop, en_intermittent]
