"""Easy generators for p_line_change (A 正常換線 / B 缺料等待 / C 設備待修).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair with the same
work order, line and time where only the reason the line is not producing is swapped so the gold flips.
Gold comes from the template spec, never from post-hoc judgement: A = previous WO completed on schedule
and the line is changing stencil/program/feeders or waiting for first-article confirmation; B = stopped
because material has not arrived, is short, or is still waiting to be issued from the warehouse;
C = stopped because a machine is broken, waiting for repair, parts or calibration.
"""
from ._common import LINES, PRINTERS, MOUNTERS, REFLOWS, SPIS, AOIS, TESTERS, PNS, NAMES, wo, hhmm, pick, slot


def _two_wo(rng):
    a = wo(rng)
    b = wo(rng)
    while b == a:
        b = wo(rng)
    return min(a, b), max(a, b)


def _head(rng, line):
    prev, cur = _two_wo(rng)
    return f"{line} 上一工單 {prev} 於 {hhmm(rng)} 依排程完成，新工單 {cur} 目前狀態：", cur


# ---------------------------------------------------------------- zh pairs

def stencil_ab(rng):  # A / B
    line, p, pn = pick(rng, LINES), pick(rng, PRINTERS), pick(rng, PNS)
    h, cur = _head(rng, line)
    return (h + f"{p} 正在更換 {cur} 鋼網並清洗刮刀，預計 {rng.randint(20, 40)} 分鐘後印首件", "A",
            h + f"鋼網已換好，但 PCB 裸板 {rng.randint(100, 600)} 片尚未從倉庫發出，線上停止等料", "B")


def program_ac(rng):  # A / C
    line, m = pick(rng, LINES), pick(rng, MOUNTERS)
    h, cur = _head(rng, line)
    return (h + f"{m} 載入新程式 {cur}-V{rng.randint(2, 6)} 並依料表核對 feeder，{pick(rng, NAMES)} 預計 {rng.randint(15, 30)} 分鐘完成", "A",
            h + f"{m} head {rng.randint(1, 8)} 真空感測器故障、機台無法啟動，維修工程師處理中", "C")


def feeder_bc(rng):  # B / C
    line, m, pn, s = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, PNS), slot(rng)
    h, cur = _head(rng, line)
    return (h + f"{m} 上料到 slot {s} 時發現 {pn} 短缺 {rng.randint(200, 2000)} pcs，倉庫回覆補料 {hhmm(rng)} 才會到，線上停止等料", "B",
            h + f"{m} feeder 全部上好，但 feeder base {rng.randint(1, 4)} 通訊故障、機台報警停機，等維修", "C")


def first_article_ab(rng):  # A / B
    line, aoi, pn = pick(rng, LINES), pick(rng, AOIS), pick(rng, PNS)
    h, cur = _head(rng, line)
    return (h + f"換線作業已完成，首件 {rng.randint(1, 3)} 片已印貼完送 {aoi} 與 QC 檢驗中，等首件確認後量產", "A",
            h + f"換線作業已完成，但 {pn} 料未到線、倉庫尚未發料，首件無法貼", "B")


def kitting_ac(rng):  # A / C
    line, m, r, z = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, REFLOWS), rng.randint(1, 10)
    h, cur = _head(rng, line)
    return (h + f"依換線 SOP 進行 {m} feeder 排程換料，共 {rng.randint(20, 60)} 站，預計 {rng.randint(25, 45)} 分鐘完成", "A",
            h + f"回焊爐 {r} 第 {z} 區加熱器燒毀，待零件到廠（預計 {rng.randint(1, 3)} 天），整線停止", "C")


def warehouse_bc(rng):  # B / C
    line, spi, pn = pick(rng, LINES), pick(rng, SPIS), pick(rng, PNS)
    h, cur = _head(rng, line)
    return (h + f"開工單已開，但主料 {pn} 倉庫發料延遲，預計 {hhmm(rng)} 才到線，線上停止等料", "B",
            h + f"料已全部到線，但 {spi} 校正片逾期、待校正完成才能開機，線上停止", "C")


def spare_ac(rng):  # A / C
    line, p = pick(rng, LINES), pick(rng, PRINTERS)
    h, cur = _head(rng, line)
    return (h + f"{p} 更換鋼網、更新印刷程式並補錫膏，{pick(rng, NAMES)} 預計 {rng.randint(15, 30)} 分鐘完成", "A",
            h + f"{p} 刮刀氣缸漏氣、無法印刷，待更換氣缸零件（廠商 {rng.randint(1, 3)} 天後到），線上停止", "C")


def calibration_bc(rng):  # B / C
    line, m, pn = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, PNS)
    h, cur = _head(rng, line)
    return (h + f"{pn} 料件短缺 {rng.randint(300, 3000)} pcs、供應商尚未交貨，線上停止等料", "B",
            h + f"料件齊全，但 {m} 視覺相機待年度校正、校正逾期無法開機，等儀校人員", "C")


def shortage_ab(rng):  # A / B
    line, m, pn, s = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, PNS), slot(rng)
    h, cur = _head(rng, line)
    return (h + f"{m} 正在依料表把 {pn} 上到 slot {s}、逐站核對條碼，換線進度 {rng.randint(40, 80)}%", "A",
            h + f"{m} 上料到 slot {s} 時 {pn} 料未到，倉庫尚未發料，線上停止等料", "B")


def tester_ac(rng):  # A / C
    line, tester = pick(rng, LINES), pick(rng, TESTERS)
    h, cur = _head(rng, line)
    return (h + f"{tester} 已切換到 {cur} 測試程式，首件 {rng.randint(1, 3)} 片正在確認中", "A",
            h + f"{tester} 針床治具 {rng.randint(3, 12)} 支探針斷裂、機台無法測試，待維修，線上停止", "C")


def supplier_bc(rng):  # B / C
    line, m, pn = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, PNS)
    h, cur = _head(rng, line)
    return (h + f"{pn} 供應商料未到，採購確認明天 {hhmm(rng)} 才交貨，線上停止等料", "B",
            h + f"{m} 伺服馬達過載停機、重啟失敗，維修工程師 {pick(rng, NAMES)} 處理中，線上停止", "C")


def reflow_ab(rng):  # A / B
    line, r = pick(rng, LINES), pick(rng, REFLOWS)
    h, cur = _head(rng, line)
    return (h + f"回焊爐 {r} 切換到 {cur} 的溫度 profile、升溫穩定中，預計 {rng.randint(10, 25)} 分鐘後可進板", "A",
            h + f"回焊爐 {r} profile 已切換完成，但 PCB 裸板 {rng.randint(100, 600)} 片尚未從倉庫發出，線上停止等料", "B")


PAIRS_ZH = [stencil_ab, program_ac, feeder_bc, first_article_ab, kitting_ac, warehouse_bc, spare_ac,
            calibration_bc, shortage_ab, tester_ac, supplier_bc, reflow_ab]


# ---------------------------------------------------------------- en pairs

def _ln(line):
    return line.replace("SMT-", "")


def _en_head(rng, line):
    prev, cur = _two_wo(rng)
    return f"{cur} {_ln(line)} PREV {prev} COMPLETED {hhmm(rng)} ON SCHEDULE STATUS: ", cur


def en_stencil(rng):  # A / B
    line, p, m = pick(rng, LINES), pick(rng, PRINTERS), pick(rng, MOUNTERS)
    h, cur = _en_head(rng, line)
    return (h + f"STENCIL CHANGE {p} PROGRAM LOAD {m} FIRST ARTICLE PENDING ETA {rng.randint(15, 40)}MIN", "A",
            h + f"STENCIL CHANGED WAITING MATERIAL {pick(rng, PNS)} NOT ISSUED FROM WAREHOUSE LINE IDLE", "B")


def en_program(rng):  # A / C
    line, m = pick(rng, LINES), pick(rng, MOUNTERS)
    h, cur = _en_head(rng, line)
    return (h + f"{m} PROGRAM {cur}-V{rng.randint(2, 6)} LOADING FEEDER VERIFY IN PROGRESS", "A",
            h + f"{m} HEAD{rng.randint(1, 8)} VACUUM SENSOR FAULT MACHINE DOWN WAITING REPAIR", "C")


def en_feeder(rng):  # B / C
    line, m, pn, s = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, PNS), slot(rng)
    h, cur = _en_head(rng, line)
    return (h + f"{m} SLOT {s} {pn} SHORT {rng.randint(200, 2000)} PCS WAREHOUSE ETA {hhmm(rng)} LINE IDLE", "B",
            h + f"{m} FEEDER BASE {rng.randint(1, 4)} COMM ERROR MACHINE DOWN WAITING REPAIR LINE IDLE", "C")


def en_first_article(rng):  # A / B
    line, aoi = pick(rng, LINES), pick(rng, ["AOI-01", "AOI-02"])
    h, cur = _en_head(rng, line)
    return (h + f"CHANGEOVER DONE FIRST ARTICLE {rng.randint(1, 3)} PCS AT {aoi} AWAITING QC CONFIRMATION", "A",
            h + f"CHANGEOVER DONE {pick(rng, PNS)} NOT ARRIVED WAITING WAREHOUSE ISSUE FIRST ARTICLE BLOCKED", "B")


def en_reflow(rng):  # A / C
    line, r = pick(rng, LINES), pick(rng, REFLOWS)
    h, cur = _en_head(rng, line)
    return (h + f"REFLOW {r} PROFILE SWITCHED TO {cur} HEATING UP ETA {rng.randint(10, 25)}MIN", "A",
            h + f"REFLOW {r} ZONE{rng.randint(1, 10)} HEATER FAILED WAITING SPARE PART ETA {rng.randint(1, 3)} DAYS LINE IDLE", "C")


PAIRS_EN = [en_stencil, en_program, en_feeder, en_first_article, en_reflow]
