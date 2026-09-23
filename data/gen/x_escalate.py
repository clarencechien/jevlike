"""Easy generators for x_escalate (A 是 / B 否: 是否需要立即升級通知線長).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair that
shares the same event but flips one decisive fact (duration over/under 30 min, one board vs
whole batch, injury vs none, cross-department need vs handled per SOP).
Gold comes from the template spec, never from post-hoc judgement.
"""
from ._common import LINES, MOUNTERS, PRINTERS, REFLOWS, AOIS, SPIS, PARTS, PNS, NAMES, wo, hhmm, pick, slot

FAULTS = ["X 軸 servo 過載", "吸嘴頭卡住", "板子卡在軌道", "視覺辨識失敗", "主軸馬達過熱"]
DEFECTS = ["偏移", "立碑", "空焊", "短路", "少件"]
CUSTOMERS = ["客戶 K 社", "客戶 ABC", "客戶 Delta", "客戶 NV"]


def downtime(rng):
    line, m, f, n = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, FAULTS), rng.randint(3, 6)
    head = f"{line} {m} {f}，"
    return (head + f"已經停了 {rng.randint(40, 90)} 分鐘還沒修好，後段 {n} 站都在等板", "A",
            head + f"停了 {rng.randint(8, 20)} 分鐘，ME 已排除，機台恢復生產", "B")


def ng_batch(rng):
    line, a, w, d = pick(rng, LINES), pick(rng, AOIS), wo(rng), pick(rng, DEFECTS)
    head = f"{line} {a} {w} "
    return (head + f"整批 {rng.randint(200, 600)} 片都出現{d}，全批待判無法出貨", "A",
            head + f"1 片{d} NG，已依重工 SOP 重工並複檢 OK，其餘正常", "B")


def safety(rng):
    name, line, m, s = pick(rng, NAMES), pick(rng, LINES), pick(rng, MOUNTERS), slot(rng)
    head = f"{name} 在 {line} {m} 換 slot {s} feeder 時"
    return (head + f"手被夾傷流血，已送醫務室處理", "A",
            head + f"觸發安全門，關門復歸後機台正常，沒有人受傷", "B")


def shipment(rng):
    line, w, q, t, pn = pick(rng, LINES), wo(rng), rng.randint(300, 1000), hhmm(rng), pick(rng, PNS)
    head = f"{line} {w} 今天 {t} 要出貨 {q} 片，"
    return (head + f"目前只完成 {rng.randint(30, 60)}%，{pn} 缺料，倉庫回覆要 {rng.randint(2, 4)} 天後才進料", "A",
            head + f"{q} 片已全數包裝完成，貨運已預約，等待上車", "B")


def stencil(rng):
    line, w, n = pick(rng, LINES), wo(rng), rng.randint(10, 20)
    head = f"{line} {w} 換線用的鋼網"
    return (head + f"還在廠商那邊沒回廠，需要採購、倉庫和排程一起協調改單", "A",
            head + f"已到線邊，依換線 SOP 進行中，預計 {n} 分鐘完成", "B")


def feeder(rng):
    line, m, s, pn = pick(rng, LINES), pick(rng, MOUNTERS), slot(rng), pick(rng, PNS)
    head = f"{line} {m} slot {s} {pn} 缺料，"
    return (head + f"機台停了 {rng.randint(5, 15)} 分鐘，線邊備料已補上並恢復運轉", "B",
            head + f"已停 {rng.randint(35, 60)} 分鐘，倉庫回覆全廠無庫存，整線停等", "A")


def reflow(rng):
    line, r, z, q = pick(rng, LINES), pick(rng, REFLOWS), rng.randint(1, 10), rng.randint(150, 500)
    head = f"{line} {r} 第 {z} 區溫度"
    return (head + f"超上限 {rng.randint(12, 20)}°C，一批 {q} 片已過爐，整批需判定是否報廢", "A",
            head + f"偏差 +{rng.randint(2, 4)}°C 在警戒範圍內，操作員依 SOP 微調後恢復正常", "B")


def pm(rng):
    line, m, t = pick(rng, LINES), pick(rng, MOUNTERS + PRINTERS), hhmm(rng)
    head = f"{line} {m} 例行 PM "
    return (head + f"依計畫 {t} 進行，{rng.randint(20, 30)} 分鐘完成，機台已交回生產", "B",
            head + f"時發現主軸磨損，需停機 {rng.randint(2, 4)} 小時更換，今天整線產出會少 {rng.randint(300, 800)} 片", "A")


def mes(rng):
    line, n, st = pick(rng, LINES), rng.randint(4, 6), pick(rng, ["印刷站", "貼片站", "AOI 站", "ICT 站"])
    head = f"{line} "
    return (head + f"MES 整線 {n} 站都無法過站，已持續 {rng.randint(35, 60)} 分鐘，板子堆在線邊出不去", "A",
            head + f"{st} 一台掃描器重開後恢復正常，耽誤 {rng.randint(3, 8)} 分鐘", "B")


def spi(rng):
    line, s, w, q = pick(rng, LINES), pick(rng, SPIS), wo(rng), rng.randint(150, 500)
    head = f"{line} {s} "
    return (head + f"錫膏厚度 SPC 連續 {rng.randint(5, 9)} 點超管制線，{w} 已印刷 {q} 片全部要重檢", "A",
            head + f"1 片錫膏偏薄，依 SOP 擦鋼網後重印 OK，後續量測正常", "B")


def customer(rng):
    c, w, q = pick(rng, CUSTOMERS), wo(rng), rng.randint(300, 1000)
    head = f"{c} 來電，{w} "
    return (head + f"出貨 {q} 片有{pick(rng, DEFECTS)}問題要求全批退回重驗", "A",
            head + f"詢問出貨進度，依例回覆已於 9/{rng.randint(10, 22)} 出貨並提供追蹤單號", "B")


def nozzle(rng):
    line, m, h, w = pick(rng, LINES), pick(rng, MOUNTERS), rng.randint(1, 8), wo(rng)
    head = f"{line} {m} "
    return (head + f"head{h} 拋料略增，換吸嘴後恢復正常，耗時 {rng.randint(5, 10)} 分鐘", "B",
            head + f"全部 head 拋料率 {rng.randint(30, 50)}%，{w} 已貼 {rng.randint(200, 500)} 片疑似全部偏移，要整批複檢", "A")


PAIRS_ZH = [downtime, ng_batch, safety, shipment, stencil, feeder, reflow, pm, mes, spi, customer, nozzle]


def en_downtime(rng):
    line, m, f = pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, ["X-axis servo overload", "head jam", "board stuck in conveyor", "vision error"])
    head = f"{line} {m} {f}, "
    return (head + f"down {rng.randint(40, 90)} min and still not fixed, {rng.randint(3, 6)} downstream stations waiting", "A",
            head + f"down {rng.randint(8, 20)} min, cleared by ME, machine back in production", "B")


def en_ng(rng):
    line, w, d = pick(rng, LINES), wo(rng), pick(rng, ["shift", "tombstone", "open joint", "bridge"])
    head = f"{line} AOI {w} "
    return (head + f"whole batch of {rng.randint(200, 600)} boards showing {d}, all on hold, cannot ship", "A",
            head + f"1 board {d} NG, reworked per SOP and re-inspected OK, rest normal", "B")


def en_safety(rng):
    name, line, m = pick(rng, ["Tom", "Amy", "Ken", "Lisa", "Ray"]), pick(rng, LINES), pick(rng, MOUNTERS)
    head = f"{name} changing feeder on {line} {m}, "
    return (head + f"hand caught and cut, sent to first-aid room", "A",
            head + f"safety door tripped, reset after closing door, nobody hurt", "B")


def en_shipment(rng):
    line, w, q, t = pick(rng, LINES), wo(rng), rng.randint(300, 1000), hhmm(rng)
    head = f"{line} {w} ships today {t}, {q} pcs, "
    return (head + f"only {rng.randint(30, 60)}% done, {pick(rng, PNS)} short, warehouse says {rng.randint(2, 4)} days to restock", "A",
            head + f"all packed, carrier booked, waiting for pickup", "B")


def en_feeder(rng):
    line, m, s = pick(rng, LINES), pick(rng, MOUNTERS), slot(rng)
    head = f"{line} {m} slot {s} out of parts, "
    return (head + f"paused {rng.randint(5, 15)} min, lineside spare loaded, running again", "B",
            head + f"stopped {rng.randint(35, 60)} min, warehouse confirms zero stock plant-wide, whole line waiting", "A")


PAIRS_EN = [en_downtime, en_ng, en_safety, en_shipment, en_feeder]
