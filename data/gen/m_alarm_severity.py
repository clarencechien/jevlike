"""Easy generators for m_alarm_severity (A 不急 / B 盡快 / C 停線).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair that
differs in ONE fact so the gold flips. SINGLES_* entries return (state, gold).
Gold comes from the template spec, never from post-hoc judgement.
"""
from ._common import LINES, MOUNTERS, PRINTERS, REFLOWS, AOIS, hhmm, pick, slot


def reflow_temp(rng):
    line, r, zone, over = pick(rng, LINES), pick(rng, REFLOWS), rng.randint(1, 10), rng.randint(8, 15)
    base = f"{line} 回焊爐 {r} 第 {zone} 區溫度超上限 {over}°C 持續 {rng.randint(2, 6)} 分鐘"
    return base + "，板子仍在爐內", "C", base + "，爐內無板、輸送帶已自動停止", "B"


def reflow_warn(rng):
    line, r, zone = pick(rng, LINES), pick(rng, REFLOWS), rng.randint(1, 10)
    return (f"{line} 回焊爐 {r} 第 {zone} 區溫度偏差 +{rng.randint(2, 4)}°C，在警戒範圍內，機台持續運轉", "A",
            f"{line} 回焊爐 {r} 第 {zone} 區溫度偏差 +{rng.randint(12, 20)}°C 超出規格，板子仍在爐內", "C")


def feeder(rng):
    line, m, s = pick(rng, LINES), pick(rng, MOUNTERS), slot(rng)
    return (f"{line} {m} feeder slot {s} 缺料，機台暫停等待補料，備料在線邊", "B",
            f"{line} {m} feeder slot {s} 缺料，機台暫停等待補料，倉庫回覆此料號全廠無庫存，整線 {rng.randint(3, 6)} 站都等這顆料", "C")


def nozzle(rng):
    line, m, head = pick(rng, LINES), pick(rng, MOUNTERS), rng.randint(1, 8)
    return (f"{line} {m} head{head} 吸嘴吸著率降到 {rng.randint(88, 93)}%，機台持續運轉，拋料略增", "B",
            f"{line} {m} head{head} 吸嘴吸著率 {rng.randint(98, 100)}%，系統提示吸嘴已使用 {rng.randint(900, 1200)} 小時建議下次保養更換", "A")


def maintenance_reminder(rng):
    line, m = pick(rng, LINES), pick(rng, PRINTERS + MOUNTERS)
    return (f"{line} {m} 預防保養到期提醒：距上次 PM 已 {rng.randint(85, 95)} 天，請於本週安排", "A",
            f"{line} {m} 主軸馬達過熱警報，異音明顯，機台已自動停機且無法重啟", "B")


def safety(rng):
    line, m = pick(rng, LINES), pick(rng, MOUNTERS)
    return (f"{line} {m} 安全門互鎖觸發，人員在機台內部作業時機台仍有動作，急停已按下", "C",
            f"{line} {m} 安全門互鎖觸發，操作員關門後機台已正常復歸", "A")


def paste(rng):
    line, p = pick(rng, LINES), pick(rng, PRINTERS)
    return (f"{line} 印刷機 {p} 錫膏開封已超過 {rng.randint(9, 14)} 小時、超過規定 8 小時，過去 {rng.randint(30, 90)} 分鐘已印刷 {rng.randint(40, 120)} 片", "C",
            f"{line} 印刷機 {p} 錫膏開封 {rng.randint(6, 7)} 小時，系統提醒 1 小時後到期", "A")


def aoi_ng(rng):
    line, a = pick(rng, LINES), pick(rng, AOIS)
    return (f"{line} {a} 過去 {rng.randint(10, 30)} 分鐘連續 {rng.randint(8, 20)} 片同一位置 NG，NG 率 {rng.randint(35, 80)}%", "C",
            f"{line} {a} 過去 {rng.randint(10, 30)} 分鐘 NG 1 片，複判為誤判，NG 率 {rng.choice([0.5, 0.8, 1.0])}%", "A")


def air_pressure(rng):
    line = pick(rng, LINES)
    return (f"{line} 廠務空壓 {rng.choice([4.2, 4.3, 4.4])} bar 低於下限 4.5 bar，整線貼片機同時報警並停止", "C",
            f"{line} 廠務空壓 {rng.choice([5.0, 5.1, 5.2])} bar 正常，系統提示乾燥機濾芯下月到期", "A")


def belt(rng):
    line, m = pick(rng, LINES), pick(rng, MOUNTERS)
    return (f"{line} {m} 板子傳送皮帶偵測到打滑，該站暫停，前後站正常", "B",
            f"{line} {m} 板子傳送皮帶保養計數到達，建議下次換線時清潔", "A")


def spi(rng):
    line, s = pick(rng, LINES), pick(rng, ["SPI-01", "SPI-02"])
    return (f"{line} {s} 錫膏體積偏低警告，比例 {rng.randint(5, 9)}%，機台持續運轉", "B",
            f"{line} {s} 錫膏體積偏低 NG 率 {rng.randint(40, 70)}%，下游貼片機仍在持續貼件", "C")


PAIRS_ZH = [reflow_temp, reflow_warn, feeder, nozzle, maintenance_reminder, safety, paste, aoi_ng, air_pressure, belt, spi]


def en_reflow(rng):
    r, z, t, lim = pick(rng, REFLOWS), rng.randint(1, 10), rng.randint(255, 262), 245
    return (f"REFLOW {r} ZONE{z} TEMP HIGH {t}C LIMIT {lim}C DURATION {rng.randint(2, 5)}MIN BOARDS IN OVEN", "C",
            f"REFLOW {r} ZONE{z} TEMP HIGH {t}C LIMIT {lim}C CONVEYOR STOPPED OVEN EMPTY", "B")


def en_feeder(rng):
    m, s = pick(rng, MOUNTERS), slot(rng)
    return (f"F-1207 {m} FEEDER SLOT {s} EMPTY MACHINE PAUSED WAITING REFILL", "B",
            f"F-1210 {m} FEEDER SLOT {s} LOW WARNING {rng.randint(150, 400)} PCS REMAINING", "A")


def en_nozzle(rng):
    m, h = pick(rng, MOUNTERS), rng.randint(1, 8)
    return (f"E4021 NOZZLE VACUUM LOW HEAD{h} {m} PICKUP RATE {rng.randint(85, 92)}% RUNNING", "B",
            f"E4021 NOZZLE VACUUM LOW HEAD{h} {m} PICKUP RATE {rng.randint(20, 45)}% ALL HEADS AFFECTED LINE STOPPED", "C")


def en_pm(rng):
    m = pick(rng, MOUNTERS + PRINTERS)
    return (f"INFO {m} PM DUE IN {rng.randint(3, 10)} DAYS SCHEDULE MAINTENANCE", "A",
            f"ALARM {m} SERVO X-AXIS OVERLOAD MACHINE STOPPED RESTART FAILED", "B")


def en_safety(rng):
    m = pick(rng, MOUNTERS)
    return (f"SAFETY {m} EMERGENCY STOP PRESSED OPERATOR INSIDE MACHINE ENCLOSURE", "C",
            f"INFO {m} SAFETY DOOR CLOSED MACHINE RESUMED", "A")


PAIRS_EN = [en_reflow, en_feeder, en_nozzle, en_pm, en_safety]
