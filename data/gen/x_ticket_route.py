"""Easy generators for x_ticket_route (A 產線 / B 設備 / C IT / D 品保).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair that
shares the same ticket frame but swaps the decisive clause so the receiving unit flips.
Gold comes from the template spec, never from post-hoc judgement.
"""
from ._common import LINES, MOUNTERS, PRINTERS, REFLOWS, AOIS, SPIS, TESTERS, PARTS, PNS, NAMES, wo, hhmm, pick, slot

STATIONS = ["印刷站", "貼片站", "爐後站", "AOI 站", "ICT 站", "包裝站"]
DEFECTS = ["偏移", "立碑", "空焊", "短路", "少件", "極性反"]
CUSTOMERS = ["客戶 K 社", "客戶 ABC", "客戶 Delta", "客戶 華X", "客戶 NV"]
GAUGES = ["錫膏厚度計", "推力計", "游標卡尺", "溫度記錄器", "X-ray 厚度計"]
STENCILS = ["ST-1183", "ST-2207", "ST-0912", "ST-3051"]


def scanner(rng):
    line, st, n = pick(rng, LINES), pick(rng, STATIONS), rng.randint(10, 40)
    head = f"【工單】{line} {st} 掃描器讀不到條碼，"
    return (head + f"換一台掃描器也一樣，MES 顯示該站裝置離線 {n} 分鐘，其他站正常過站", "C",
            head + f"鏡頭被推車撞歪、固定支架鬆脫，掃描頭懸空，需要拆下重新固定調整", "B")


def feeder_short(rng):
    line, m, s, pn, k = pick(rng, LINES), pick(rng, MOUNTERS), slot(rng), pick(rng, PNS), rng.randint(2, 6)
    head = f"【工單】{line} {m} slot {s} {pn} "
    return (head + f"料剩不到 {rng.randint(5, 15)} 分鐘，線邊沒備料，需要從倉庫領 {k} 盤補上", "A",
            head + f"feeder 彈片斷裂，換新料盤也送不出料，需要更換 feeder", "B")


def aoi_ng(rng):
    line, a, n, part, q = pick(rng, LINES), pick(rng, AOIS), rng.randint(6, 20), pick(rng, PARTS), rng.randint(100, 400)
    head = f"【工單】{line} {a} "
    return (head + f"連續 {n} 片 {part} 判{pick(rng, DEFECTS)}，複判仍 NG，這批 {q} 片是否可特採出貨需要判定", "D",
            head + f"相機鏡頭髒污、畫面模糊，每片都判 NG，需清潔鏡頭並重新做相機校正", "B")


def mes_upload(rng):
    line, w, n, q = pick(rng, LINES), wo(rng), rng.randint(20, 80), rng.randint(200, 900)
    head = f"【工單】{line} {w} "
    return (head + f"過站資料有 {n} 片沒上傳到 MES，看板數量停在 {q} 不動，伺服器回應 timeout", "C",
            head + f"過站漏刷 {n} 片，操作員忘記刷條碼，系統本身正常，板子已在線邊，需補刷入帳", "A")


def changeover(rng):
    line, w, m, t, n = pick(rng, LINES), wo(rng), pick(rng, MOUNTERS), hhmm(rng), rng.randint(1, 3)
    head = f"【工單】{line} {w} 換線提前到 {t}，"
    return (head + f"備料人手不夠，需要調 {n} 位人員支援備料與換 feeder", "A",
            head + f"但 {m} 從伺服器下載貼裝程式失敗，網路磁碟連不到，其他機台也抓不到程式", "C")


def spi_spc(rng):
    line, s, n, cpk, q = pick(rng, LINES), pick(rng, SPIS), rng.randint(5, 9), pick(rng, [0.7, 0.8, 0.9]), rng.randint(150, 500)
    head = f"【工單】{line} {s} "
    return (head + f"錫膏厚度 SPC 連續 {n} 點超管制上限，Cpk {cpk}，已印刷的 {q} 片是否放行需要判定", "D",
            head + f"雷射頭開機自檢失敗，校正流程跑不過，機台無法起動", "B")


def gauge_cal(rng):
    line, g, d = pick(rng, LINES), pick(rng, GAUGES), rng.randint(1, 28)
    head = f"【工單】{line} 線邊 {g} "
    return (head + f"校驗貼紙 9/{d:02d} 到期，量測值與標準片差 {rng.randint(3, 8)}%，需送校並判定期間量測結果", "D",
            head + f"量測資料無法上傳到 SPC 系統，軟體顯示連線錯誤，重開電腦仍相同", "C")


def new_hire(rng):
    name, line = pick(rng, NAMES), pick(rng, LINES)
    head = f"【工單】新進人員 {name} "
    return (head + f"MES 帳號還沒開通，無法登入刷過站，帳號申請單 {rng.randint(3, 7)} 天前已簽核", "C",
            head + f"今天報到，還沒排入班表，需安排到 {line} 跟線與教育訓練", "A")


def customer(rng):
    c, w, q, n = pick(rng, CUSTOMERS), wo(rng), rng.randint(200, 800), rng.randint(3, 20)
    head = f"【工單】{c} 來信，{w} "
    return (head + f"出貨 {q} 片中有 {n} 片{pick(rng, DEFECTS)}，要求 8D 報告與不良原因分析", "D",
            head + f"要求提前 {rng.randint(1, 3)} 天出貨，需重排產線排程與加班人力", "A")


def printer(rng):
    line, p, c = pick(rng, LINES), pick(rng, PRINTERS), rng.randint(5, 8)
    head = f"【工單】{line} {p} "
    return (head + f"刮刀壓力飄動 {c - 2}~{c + 2} kg（設定 {c} kg），機構有異音，壓力感測器疑似故障", "B",
            head + f"鋼網 {pick(rng, STENCILS)} 還沒領到線邊，備品架標示錯位置，換線等鋼網", "A")


def kanban(rng):
    line, w, y, n = pick(rng, LINES), wo(rng), rng.randint(80, 95), rng.randint(5, 30)
    head = f"【工單】{line} 看板顯示 {w} 良率 {y}%，"
    return (head + f"數字與現場點數對不起來，系統統計連續 {rng.randint(2, 5)} 小時沒更新", "C",
            head + f"現場確認 {n} 片{pick(rng, DEFECTS)}不良，這批是否放行需要判定", "D")


def reflow(rng):
    line, r, z, q = pick(rng, LINES), pick(rng, REFLOWS), rng.randint(1, 10), rng.randint(100, 400)
    head = f"【工單】{line} {r} 第 {z} 區 "
    return (head + f"加熱器無法升溫，實際 {rng.randint(150, 190)}°C 設定 {rng.randint(230, 250)}°C，heater 疑似燒毀", "B",
            head + f"profile 量測峰溫 {rng.randint(252, 260)}°C 超出規格上限 245°C，這批 {q} 片已過爐是否可用需要判定", "D")


PAIRS_ZH = [scanner, feeder_short, aoi_ng, mes_upload, changeover, spi_spc, gauge_cal, new_hire, customer, printer, kanban, reflow]


def en_scanner(rng):
    line, st, n = pick(rng, LINES), pick(rng, ["print", "SMT", "post-reflow", "AOI", "ICT"]), rng.randint(10, 40)
    head = f"TICKET {line} {st} station barcode scanner not reading, "
    return (head + f"swapped scanner same result, MES shows station offline for {n} min", "C",
            head + f"lens knocked out of alignment, bracket loose, needs remount and adjustment", "B")


def en_feeder(rng):
    line, m, s, pn = pick(rng, LINES), pick(rng, MOUNTERS), slot(rng), pick(rng, PNS)
    head = f"TICKET {line} {m} slot {s} {pn} "
    return (head + f"reel almost empty, no spare at lineside, need {rng.randint(2, 6)} reels pulled from warehouse", "A",
            head + f"feeder spring broken, new reel does not advance, feeder needs replacement", "B")


def en_aoi(rng):
    line, a, n, q = pick(rng, LINES), pick(rng, ["AOI-01", "AOI-02", "AOI-03"]), rng.randint(6, 20), rng.randint(100, 400)
    head = f"TICKET {line} {a} "
    return (head + f"{n} boards in a row flagged {pick(rng, ['shift', 'tombstone', 'open', 'bridge'])}, re-judge still NG, batch of {q} needs disposition", "D",
            head + f"camera lens dirty, image blurred, every board flagged NG, needs cleaning and camera calibration", "B")


def en_mes(rng):
    line, w, n = pick(rng, LINES), wo(rng), rng.randint(20, 80)
    head = f"TICKET {line} {w} {n} boards "
    return (head + f"not uploaded to MES, dashboard count frozen, server returns timeout", "C",
            head + f"missed scan, operator forgot to scan at outfeed, boards at lineside need re-scan", "A")


def en_customer(rng):
    c, w, q = pick(rng, ["Customer K", "Customer ABC", "Customer Delta", "Customer NV"]), wo(rng), rng.randint(200, 800)
    head = f"TICKET {c} email re {w}: "
    return (head + f"{rng.randint(3, 20)} of {q} shipped boards have solder bridges, 8D report requested", "D",
            head + f"asks to pull shipment in by {rng.randint(1, 3)} days, schedule and overtime need rework", "A")


PAIRS_EN = [en_scanner, en_feeder, en_aoi, en_mes, en_customer]
