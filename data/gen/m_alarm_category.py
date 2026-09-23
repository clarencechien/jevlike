"""Easy generators for m_alarm_category (A 機構 / B 電控 / C 物料 / D 程式 / E 環境).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair that
differs in ONE fact so the gold flips. Twins flip between different option pairs so every
option is gold in several generators. Gold comes from the template spec, never from post-hoc judgement.
"""
from ._common import LINES, MOUNTERS, PRINTERS, REFLOWS, AOIS, SPIS, PARTS, PNS, hhmm, pick, slot, wo


def nozzle_vs_pickup_param(rng):
    line, m, head, n = pick(rng, LINES), pick(rng, MOUNTERS), rng.randint(1, 8), rng.randint(1, 6)
    base = f"{line} {m} head{head} 吸嘴 {n} 號連續拋料 alarm，拋料率 {rng.randint(8, 20)}%"
    return (base + f"，拆下吸嘴檢查發現吸嘴頭橡膠墊破損、內孔有刮痕，真空表在 {rng.choice([-30, -35, -40])} kPa 上不去", "A",
            base + f"，吸嘴本體與真空皆正常，貼片程式中該料的 pickup 高度 Z 值被改成 {rng.choice([-0.8, -1.2, -1.5])} mm，改回標準值後不再拋料", "D")


def feeder_tape_vs_gear(rng):
    line, m, s, pn = pick(rng, LINES), pick(rng, MOUNTERS), slot(rng), pick(rng, PNS)
    base = f"{line} {m} feeder slot {s}（{pn}）取料失敗 alarm，連續 {rng.randint(3, 9)} 次 pickup miss"
    return (base + f"，檢查發現料帶蓋帶撕裂、料帶在 {rng.randint(10, 40)} 格處斷裂，元件散落在 feeder 內，feeder 本體動作正常", "C",
            base + "，料帶完好，feeder 撥料齒輪磨損打滑、料帶無法前進，換到相鄰 slot 同一捲料可正常取料", "A")


def servo_vs_facility_air(rng):
    line, m, axis = pick(rng, LINES), pick(rng, MOUNTERS), rng.choice(["X", "Y", "Z"])
    t = hhmm(rng)
    return (f"{line} {m} {t} 報 {axis} 軸伺服驅動器過電流 alarm，驅動器面板顯示 AL.{rng.randint(10, 39)}，關電重開 {rng.randint(2, 4)} 次仍無法復歸，其他機台正常", "B",
            f"{line} {t} 整線 {rng.randint(3, 6)} 台機同時報氣壓低 alarm，廠務空壓主管道壓力表 {rng.choice([4.0, 4.1, 4.2, 4.3])} bar 低於下限 4.5 bar，其他樓層產線也同時報警", "E")


def program_version_vs_barcode(rng):
    line, m, w = pick(rng, LINES), pick(rng, MOUNTERS), wo(rng)
    v1, v2 = rng.randint(3, 12), rng.randint(13, 25)
    pn = pick(rng, PNS)
    return (f"{line} {m} 工單 {w} 開線檢查 alarm：機台載入的貼片程式版本 V{v1}，MES 工單指定版本 V{v2}，程式檔 checksum 比對失敗", "D",
            f"{line} {m} 工單 {w} 上料掃描 alarm：slot {slot(rng)} 掃到的料盤條碼料號 {pn}-{rng.choice(['B', 'C', 'X'])}，與工單 BOM 料號 {pn} 不符，料盤標籤與內容為替代料", "C")


def humidity_vs_esd_comm(rng):
    line, n = pick(rng, LINES), rng.randint(3, 8)
    return (f"{line} {n} 台機台 ESD/環境監控同時報警，廠房溫濕度計顯示濕度 {rng.randint(72, 85)}% 超過上限 60%，空調除濕系統顯示故障中", "E",
            f"{line} 僅 {pick(rng, MOUNTERS)} 一台的 ESD 監控儀報通訊中斷，其他機台正常，檢查發現監控儀後方 RJ45 通訊線接頭鬆脫，重新插緊後恢復", "B")


def belt_vs_drive_power(rng):
    line, m = pick(rng, LINES), pick(rng, MOUNTERS + [pick(rng, PRINTERS)])
    base = f"{line} {m} 板子傳送 timeout alarm，板子停在入口"
    return (base + f"，開蓋檢查發現輸送皮帶已斷裂、碎屑卡在滾輪，馬達運轉正常但皮帶不動", "A",
            base + f"，皮帶與滾輪完好，輸送馬達無動作，機台內 {rng.choice(['24V', '48V'])} 驅動板保險絲燒斷、驅動器電源燈不亮", "B")


def paste_vs_print_recipe(rng):
    line, p, spi = pick(rng, LINES), pick(rng, PRINTERS), pick(rng, SPIS)
    base = f"{line} 印刷機 {p} 印後 {spi} 連續 {rng.randint(5, 15)} 片錫膏量不足 NG"
    return (base + f"，檢查錫膏罐標籤已過有效期 {rng.randint(2, 10)} 天且錫膏明顯乾硬結塊，鋼網與印刷參數皆為標準值", "C",
            base + f"，錫膏與鋼網皆正常，發現印刷機載入的配方仍是上一個機種，刮刀壓力 {rng.randint(9, 12)} kg 與本機種標準 {rng.randint(5, 7)} kg 不同", "D")


def reflow_n2_vs_fan(rng):
    line, r, zone = pick(rng, LINES), pick(rng, REFLOWS), rng.randint(1, 10)
    return (f"{line} 回焊爐 {r} 氧含量超標 alarm（{rng.randint(1500, 4000)} ppm > 1000 ppm），廠務氮氣總管壓力 {rng.choice([2.0, 2.5, 3.0])} bar 低於規格，同棟其他回焊爐同時報警", "E",
            f"{line} 回焊爐 {r} 第 {zone} 區溫度異常 alarm，開蓋檢查發現該區熱風馬達軸承卡死、風扇葉片不轉且有燒焦異音，氮氣與電源皆正常", "A")


def offset_vs_wrong_part(rng):
    line, m, a, part = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, AOIS), pick(rng, PARTS)
    base = f"{line} {a} 報 {m} 貼裝位置偏移 alarm，{part} 連續 {rng.randint(5, 20)} 片偏移"
    return (base + f"，量測偏移固定為 X+{rng.choice([0.15, 0.2, 0.25])} mm，比對發現貼片程式中該零件座標 offset 被改過，重新 teach 座標後貼裝正常", "D",
            base + f"，程式座標與機台皆正常，發現料盤內的實際元件尺寸為 {rng.choice(['0603', '0805', '1206'])} 與 BOM 的 0402 不同，料盤標籤貼錯", "C")


def voltage_sag_vs_psu(rng):
    line, t = pick(rng, LINES), hhmm(rng)
    return (f"{line} {t} 全線 {rng.randint(5, 8)} 台機台同時報電源異常後重啟，廠務電力監控記錄同一時間廠區電壓驟降 {rng.randint(15, 30)}%、持續 {rng.randint(100, 500)} ms，鄰近產線同時受影響", "E",
            f"{line} {t} 僅 {pick(rng, MOUNTERS)} 一台報電源異常，其他機台正常，檢查發現機台內 24V 電源供應器輸出僅 {rng.choice([18.5, 19.2, 20.1])} V 且有焦味", "B")


def board_sensor_vs_board_length(rng):
    line, m = pick(rng, LINES), pick(rng, MOUNTERS)
    base = f"{line} {m} 板子到位感測 alarm，板子未偵測到位"
    return (base + f"，檢查發現到位感測器固定座鬆脫、感測器位置偏離軌道 {rng.randint(5, 15)} mm，鎖回正確位置後正常", "A",
            base + f"，感測器與擋板動作皆正常，發現貼片程式中板長參數設為 {rng.randint(80, 120)} mm，實際板長 {rng.randint(180, 260)} mm，修改參數後正常", "D")


def plc_io_vs_tray(rng):
    line, m, s = pick(rng, LINES), pick(rng, MOUNTERS), rng.randint(1, 4)
    return (f"{line} {m} PLC I/O 模組錯誤 alarm，I/O 卡 slot {s} 狀態燈紅燈，PLC 診斷顯示 module {s} no response，重插卡後仍相同", "B",
            f"{line} {m} 料盤站 tray {s} 取料 alarm，tray {s} 內 {pick(rng, ['QFN-48', 'BGA-256', '連接器 CON-12'])} 已用完、剩 0 顆，機台等待補料", "C")


PAIRS_ZH = [nozzle_vs_pickup_param, feeder_tape_vs_gear, servo_vs_facility_air, program_version_vs_barcode,
            humidity_vs_esd_comm, belt_vs_drive_power, paste_vs_print_recipe, reflow_n2_vs_fan, offset_vs_wrong_part,
            voltage_sag_vs_psu, board_sensor_vs_board_length, plc_io_vs_tray]


def en_nozzle(rng):
    m, h = pick(rng, MOUNTERS), rng.randint(1, 8)
    base = f"E4021 NOZZLE VACUUM LOW HEAD{h} {m} PICKUP RATE {rng.randint(70, 90)}%"
    return (base + f" NOZZLE TIP RUBBER TORN INNER BORE SCRATCHED VACUUM STUCK AT {rng.choice([-30, -35, -40])}KPA", "A",
            base + f" NOZZLE OK VACUUM SENSOR COMM ERROR CONNECTOR CN{rng.randint(10, 29)} LOOSE RESEATED OK", "B")


def en_feeder(rng):
    m, s = pick(rng, MOUNTERS), slot(rng)
    return (f"F-1207 {m} FEEDER SLOT {s} PICKUP MISS x{rng.randint(3, 9)} TAPE BROKEN COVER TAPE TORN AT POCKET {rng.randint(10, 40)} FEEDER MECH OK", "C",
            f"F-1301 {m} FEEDER SLOT {s} PART DATA MISMATCH PROGRAM V{rng.randint(3, 12)} EXPECTS PITCH {rng.choice([2, 4])}MM FEEDER SET {rng.choice([8, 12])}MM PROGRAM DATA WRONG", "D")


def en_servo(rng):
    m, axis = pick(rng, MOUNTERS), rng.choice(["X", "Y", "Z"])
    return (f"SERVO ALARM {axis}-AXIS OVERLOAD {m} DRIVER AL.{rng.randint(10, 39)} POWER CYCLE x{rng.randint(2, 4)} FAILED OTHER MACHINES OK", "B",
            f"AIR PRESSURE LOW {m} FACILITY MAIN LINE {rng.choice([4.0, 4.1, 4.2, 4.3])}BAR LIMIT 4.5BAR ALL {rng.randint(3, 6)} MACHINES ON LINE ALARMED", "E")


def en_reflow(rng):
    r, z = pick(rng, REFLOWS), rng.randint(1, 10)
    return (f"REFLOW {r} ZONE{z} TEMP LOW BLOWER MOTOR BEARING SEIZED FAN NOT ROTATING BURNING NOISE N2 AND POWER OK", "A",
            f"REFLOW {r} O2 HIGH {rng.randint(1500, 4000)}PPM LIMIT 1000PPM FACILITY N2 SUPPLY {rng.choice([2.0, 2.5, 3.0])}BAR BELOW SPEC ALL OVENS ALARMED", "E")


def en_printer(rng):
    p, spi = pick(rng, PRINTERS), pick(rng, SPIS)
    return (f"PRINTER {p} RECIPE LOAD ERROR FILE CHECKSUM MISMATCH RECIPE V{rng.randint(3, 12)} MES EXPECTS V{rng.randint(13, 25)}", "D",
            f"PRINTER {p} {spi} PASTE VOLUME LOW x{rng.randint(5, 15)} PASTE JAR EXPIRED {rng.randint(2, 10)} DAYS PASTE DRY STENCIL AND RECIPE OK", "C")


PAIRS_EN = [en_nozzle, en_feeder, en_servo, en_reflow, en_printer]
