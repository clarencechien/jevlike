"""Easy generators for m_needs_dispatch (A 是 / B 否).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair that
differs in ONE fact so the gold flips. A = beyond operator SOP (adjust / repair / replace parts /
calibrate / ME-only parameters); B = operator SOP covers it (refill, clean, restart, consumables,
changeover) or the problem recovered by itself. Gold comes from the template spec, never from post-hoc judgement.
"""
from ._common import LINES, MOUNTERS, PRINTERS, REFLOWS, AOIS, SPIS, PNS, NAMES, hhmm, pick, slot, wo


def feeder_refill_vs_gear(rng):
    line, m, s, pn = pick(rng, LINES), pick(rng, MOUNTERS), slot(rng), pick(rng, PNS)
    base = f"{line} {m} feeder slot {s}（{pn}）取料 alarm"
    return (base + f"，料帶用完，線邊備料還有 {rng.randint(2, 6)} 捲，操作員依補料 SOP 換上新料捲即可", "B",
            base + f"，料帶還有 {rng.randint(300, 900)} 顆，feeder 撥料齒輪磨損打滑，需拆解 feeder 更換齒輪，操作員 SOP 無此項目", "A")


def nozzle_clean_vs_calibrate(rng):
    line, m, head = pick(rng, LINES), pick(rng, MOUNTERS), rng.randint(1, 8)
    base = f"{line} {m} head{head} 吸嘴拋料率 {rng.randint(6, 15)}% alarm"
    return (base + f"，吸嘴內孔有錫渣髒污，操作員依 SOP 用超音波清洗機清潔吸嘴後拋料率回到 {rng.choice([0.5, 0.8, 1.0])}%", "B",
            base + f"，吸嘴清潔與更換新吸嘴後仍拋料，量測 head{head} Z 軸原點偏移 {rng.choice([0.3, 0.4, 0.5])} mm，需做 head 校正", "A")


def reflow_pid_vs_door(rng):
    line, r, zone = pick(rng, LINES), pick(rng, REFLOWS), rng.randint(1, 10)
    base = f"{line} 回焊爐 {r} 第 {zone} 區溫度偏差 alarm"
    return (base + f"，溫度持續在設定值 ±{rng.randint(8, 15)}°C 之間震盪，該區 PID 加熱參數需重新調整，此參數操作員帳號無修改權限", "A",
            base + f"，原因是爐門開啟未關，操作員關上爐門後 {rng.randint(3, 8)} 分鐘溫度回到設定值，alarm 自動解除", "B")


def stencil_clean_vs_axis(rng):
    line, p, spi = pick(rng, LINES), pick(rng, PRINTERS), pick(rng, SPIS)
    base = f"{line} 印刷機 {p} 印後 {spi} 連續 {rng.randint(3, 8)} 片錫膏不足 NG"
    return (base + "，鋼網開孔有錫膏堵塞，操作員依 SOP 執行鋼網手動擦拭與清洗後 NG 解除", "B",
            base + f"，鋼網清洗後仍 NG，量測印刷機 Y 軸對位固定偏 {rng.choice([0.08, 0.1, 0.12])} mm，印刷機需重新做相機對位校正", "A")


def comm_restart_vs_driver(rng):
    line, m, t = pick(rng, LINES), pick(rng, MOUNTERS), hhmm(rng)
    base = f"{line} {m} {t} 報 head 通訊錯誤 alarm"
    return (base + "，操作員依 SOP 關電重開機一次後通訊恢復，之後正常生產無再發", "B",
            base + f"，操作員關電重開 {rng.randint(2, 4)} 次後仍再發，驅動器面板顯示 AL.{rng.randint(10, 39)} 硬體錯誤，驅動器需更換", "A")


def board_jam_vs_belt(rng):
    line, m = pick(rng, LINES), pick(rng, MOUNTERS + PRINTERS)
    base = f"{line} {m} 板子傳送 timeout alarm"
    return (base + "，一片板子卡在入口擋板，操作員依卡板 SOP 取出板子並重置後正常運轉", "B",
            base + f"，皮帶已磨損起毛、板子持續打滑，需更換輸送皮帶並重新調整張力", "A")


def paste_jar_vs_mixer(rng):
    line, p = pick(rng, LINES), pick(rng, PRINTERS)
    base = f"{line} 印刷機 {p} 錫膏相關 alarm"
    return (base + f"，鋼網上錫膏用完，操作員依 SOP 從回溫櫃取出已回溫 {rng.randint(4, 6)} 小時的新罐錫膏補上", "B",
            base + f"，錫膏攪拌機馬達運轉有異音且轉速不穩，攪拌機需拆開維修更換馬達", "A")


def aoi_lens_vs_camera(rng):
    line, a = pick(rng, LINES), pick(rng, AOIS)
    base = f"{line} {a} 影像亮度異常 alarm"
    return (base + "，鏡頭表面有灰塵，操作員依 SOP 用無塵布與鏡頭清潔液擦拭後影像恢復正常", "B",
            base + f"，鏡頭清潔後仍異常，相機標定板量測誤差 {rng.choice([0.05, 0.06, 0.08])} mm 超過規格，相機需重新校正", "A")


def safety_door_vs_sensor(rng):
    line, m = pick(rng, LINES), pick(rng, MOUNTERS)
    base = f"{line} {m} 安全門互鎖 alarm"
    return (base + "，前側安全門未關緊，操作員關好門後按復歸鍵機台正常啟動", "B",
            base + f"，門已關緊仍報警，門感測器磁簧開關測得斷路、需更換感測器，操作員 SOP 無此項目", "A")


def air_recovered_vs_regulator(rng):
    line, m, t = pick(rng, LINES), pick(rng, MOUNTERS), hhmm(rng)
    base = f"{line} {m} {t} 氣壓低 alarm"
    return (base + f"，壓力短暫掉到 {rng.choice([4.2, 4.3, 4.4])} bar 後 {rng.randint(10, 30)} 秒內自行回到 {rng.choice([5.0, 5.2, 5.5])} bar，alarm 自動解除，機台持續正常運轉", "B",
            base + f"，廠務主管壓力正常 {rng.choice([5.5, 6.0])} bar，但機台內調壓閥出口壓力只有 {rng.choice([3.5, 3.8, 4.0])} bar 且無法調高，調壓閥需更換", "A")


def changeover_vs_me_param(rng):
    line, m, w = pick(rng, LINES), pick(rng, MOUNTERS), wo(rng)
    base = f"{line} {m} 工單 {w} 開線 alarm"
    return (base + f"，機台載入的仍是上一工單程式，操作員依換線 SOP 載入本工單程式 V{rng.randint(3, 25)} 並掃描確認後開線", "B",
            base + f"，程式正確但 {pick(rng, PNS)} 的 pickup 延遲參數需要調整，此參數在 ME 權限頁面、操作員帳號無法修改", "A")


def filter_vs_pump(rng):
    line, m = pick(rng, LINES), pick(rng, MOUNTERS)
    base = f"{line} {m} 真空壓力低 alarm"
    return (base + f"，真空濾芯計數已達 {rng.randint(1000, 1500)} 小時，操作員依 SOP 更換新濾芯後真空回到 {rng.choice([-80, -85, -88])} kPa", "B",
            base + f"，濾芯更換後真空仍只有 {rng.choice([-50, -55, -60])} kPa，真空泵運轉異音且發熱，真空泵需拆下維修", "A")


PAIRS_ZH = [feeder_refill_vs_gear, nozzle_clean_vs_calibrate, reflow_pid_vs_door, stencil_clean_vs_axis,
            comm_restart_vs_driver, board_jam_vs_belt, paste_jar_vs_mixer, aoi_lens_vs_camera, safety_door_vs_sensor,
            air_recovered_vs_regulator, changeover_vs_me_param, filter_vs_pump]


def en_feeder(rng):
    m, s = pick(rng, MOUNTERS), slot(rng)
    return (f"F-1207 {m} FEEDER SLOT {s} EMPTY REEL USED UP {rng.randint(2, 6)} REELS AT LINESIDE OPERATOR REFILL PER SOP", "B",
            f"F-1207 {m} FEEDER SLOT {s} PICKUP MISS x{rng.randint(3, 9)} {rng.randint(300, 900)} PCS LEFT FEEDER PITCH GEAR WORN GEAR REPLACEMENT REQUIRED", "A")


def en_nozzle(rng):
    m, h = pick(rng, MOUNTERS), rng.randint(1, 8)
    base = f"E4021 NOZZLE VACUUM LOW HEAD{h} {m} PICKUP RATE {rng.randint(80, 92)}%"
    return (base + f" NOZZLE BORE DIRTY OPERATOR ULTRASONIC CLEAN PER SOP RATE {rng.randint(98, 100)}% OK", "B",
            base + f" NOZZLE CLEANED AND REPLACED STILL LOW HEAD{h} Z ORIGIN OFFSET {rng.choice([0.3, 0.4, 0.5])}MM HEAD CALIBRATION REQUIRED", "A")


def en_comm(rng):
    m, t = pick(rng, MOUNTERS), hhmm(rng)
    return (f"E7102 HEAD COMM ERROR {m} {t} OPERATOR POWER CYCLE x1 RECOVERED NO RECURRENCE", "B",
            f"E7102 HEAD COMM ERROR {m} {t} POWER CYCLE x{rng.randint(2, 4)} RECURRING DRIVER AL.{rng.randint(10, 39)} HW FAULT DRIVER REPLACEMENT REQUIRED", "A")


def en_servo(rng):
    m, axis = pick(rng, MOUNTERS), rng.choice(["X", "Y"])
    return (f"SERVO ALARM {axis}-AXIS OVERLOAD {m} {axis} BELT TENSION OUT OF SPEC TENSION ADJUSTMENT AND AXIS CALIBRATION REQUIRED", "A",
            f"SERVO ALARM {axis}-AXIS OVERLOAD {m} CAUSE BOARD JAM AT STOPPER OPERATOR CLEARED BOARD PER SOP RESET OK", "B")


def en_air(rng):
    m, t = pick(rng, MOUNTERS), hhmm(rng)
    return (f"AIR PRESSURE LOW {m} {t} DIP TO {rng.choice([4.2, 4.3, 4.4])}BAR RECOVERED TO {rng.choice([5.0, 5.2, 5.5])}BAR IN {rng.randint(10, 30)}S ALARM AUTO CLEARED RUNNING", "B",
            f"AIR PRESSURE LOW {m} {t} FACILITY {rng.choice([5.5, 6.0])}BAR OK MACHINE REGULATOR OUTPUT {rng.choice([3.5, 3.8, 4.0])}BAR CANNOT ADJUST REGULATOR VALVE REPLACEMENT REQUIRED", "A")


PAIRS_EN = [en_feeder, en_nozzle, en_comm, en_servo, en_air]
