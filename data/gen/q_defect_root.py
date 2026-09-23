"""Easy generators for q_defect_root (A 錫膏 / B 貼片 / C 回焊 / D 來料 / E 治具 / F 人為).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair with the same
defect where ONE decisive clue is swapped so the root-cause station flips. Gold comes from the template
spec, never from post-hoc judgement: A = SPI already flagged the print (volume/offset/slump/bridge) or
paste expired/under-stirred; B = placement offset/flip/missing/polarity with SPI OK and a mounter cause;
C = joint defect with SPI + pre-reflow AOI OK and a profile/belt-speed cause; D = component or PCB
itself (oxidation, lot defect, warp, pad defect, spec mismatch); E = stencil/carrier/fixture/locating pin;
F = operator error (wrong reel, wrong program, bad rework, SOP not followed).
"""
from ._common import LINES, PRINTERS, MOUNTERS, REFLOWS, AOIS, SPIS, PNS, NAMES, wo, pick, slot

_SMALL = [("R123-0402-10K", "0402 電阻"), ("C220-0603-1U", "0603 電容"), ("L4-0805-2R2", "0805 電感"), ("Q7-SOT23", "SOT-23 電晶體")]
_LEDS = [("D5-LED-0603", "LED 0603"), ("D12-LED-0805", "LED 0805"), ("C33-TANT-B", "鉭質電容 B 尺寸")]
_ICS = [("U301-QFN48", "QFN-48"), ("U105-BGA256", "BGA-256"), ("U22-QFP100", "QFP-100")]


def _lot(rng):
    return f"LOT {rng.choice(['A', 'B', 'K', 'M'])}{rng.randint(2400, 2699)}{rng.randint(10, 99)}"


def _stencil(rng):
    return f"鋼網 ST-{rng.randint(1100, 1899)}"


def _carrier(rng):
    return f"載具 CR-{rng.randint(10, 89)}"


# ---------------------------------------------------------------- zh pairs

def offset_ab(rng):  # A / B
    line, aoi, spi, m = pick(rng, LINES), pick(rng, AOIS), pick(rng, SPIS), pick(rng, MOUNTERS)
    pn, part = pick(rng, _SMALL)
    off = rng.randint(70, 140)
    base = f"{line} {aoi} 發現 {pn}（{part}）貼片後偏移約 {off}µm"
    return (base + f"，回查 {spi} 該點印刷後錫膏就已往同方向偏移 {off - rng.randint(5, 20)}µm，貼片機 {m} 校正紀錄與其他位置皆正常", "A",
            base + f"，回查 {spi} 該點錫膏位置正常（偏移小於 {rng.randint(5, 10)}µm），貼片機 {m} head {rng.randint(1, 8)} 吸嘴校正已逾期 {rng.randint(10, 40)} 天、同 head 其他元件也偏同方向", "B")


def ball_ac(rng):  # A / C
    line, aoi, spi, r = pick(rng, LINES), pick(rng, AOIS), pick(rng, SPIS), pick(rng, REFLOWS)
    pn, part = pick(rng, _SMALL)
    n = rng.randint(6, 20)
    base = f"{line} 爐後 {aoi} 連續 {n} 片在 {pn}（{part}）旁出現錫珠"
    return (base + f"，{spi} 印刷後已標記該點錫膏塌陷、超出焊墊 {rng.randint(60, 120)}µm，回焊爐 {r} 曲線在規格內", "A",
            base + f"，{spi} 與爐前 AOI 皆 OK，回焊爐 {r} 曲線量測預熱區升溫速率 {rng.choice([3.6, 3.9, 4.2, 4.5])}°C/s 超過規格上限 3.0°C/s", "C")


def tombstone_bc(rng):  # B / C
    line, spi, m, r = pick(rng, LINES), pick(rng, SPIS), pick(rng, MOUNTERS), pick(rng, REFLOWS)
    pn, part = pick(rng, _SMALL[:2])
    base = f"{line} {pn}（{part}）立碑 {rng.randint(3, 12)} 片"
    return (base + f"，{spi} 錫膏正常，爐前 AOI 顯示貼片機 {m} 貼件後元件已偏離焊墊 {rng.randint(80, 150)}µm 只壓到單側焊墊，回焊曲線在規格內", "B",
            base + f"，{spi} 錫膏正常、爐前 AOI 貼件位置置中，回焊爐 {r} 曲線量測升溫速率 {rng.choice([3.5, 3.8, 4.1, 4.4])}°C/s 超過規格上限 3.0°C/s", "C")


def insufficient_de(rng):  # D / E
    line, spi, p = pick(rng, LINES), pick(rng, SPIS), pick(rng, PRINTERS)
    pn, part = pick(rng, _ICS)
    vol = rng.randint(40, 62)
    base = f"{line} {spi} {pn}（{part}）連續多板同位置少錫，錫量 {vol}%"
    return (base + f"，{p} {_stencil(rng)} 清洗後仍少錫、開孔顯微鏡檢查乾淨，改用另一批 PCB 後恢復正常，IQC 確認原批 PCB（{_lot(rng)}）焊墊表面氧化變色", "D",
            base + f"，{p} {_stencil(rng)} 開孔顯微鏡檢查有殘錫堵塞 {rng.randint(3, 9)} 孔，清洗鋼網後立即恢復正常，PCB 批次未變", "E")


def program_ef(rng):  # E / F
    line, m, w = pick(rng, LINES), pick(rng, MOUNTERS), wo(rng)
    v = rng.randint(2, 7)
    off = rng.randint(150, 300)
    base = f"{line} {m} 工單 {w} 首件 AOI 全板元件同方向偏移 {off}µm"
    return (base + f"，程式版本正確（V{v}）、機台校正正常，{_carrier(rng)} 定位銷磨損量 {rng.choice([0.2, 0.25, 0.3, 0.35])}mm 超過 0.1mm 上限", "E",
            base + f"，查機台載入的程式為舊版 V{v - 1} 而非工單指定的 V{v}，操作員 {pick(rng, NAMES)} 換線時選錯程式，載具與定位銷正常", "F")


def polarity_df(rng):  # D / F
    line, aoi, s = pick(rng, LINES), pick(rng, AOIS), slot(rng)
    pn, part = pick(rng, _LEDS)
    n = rng.randint(20, 80)
    base = f"{line} {aoi} 發現 {pn}（{part}）連續 {n} 片極性反"
    return (base + f"，feeder slot {s} 料捲裝載方向與料表一致，拆開料帶確認供應商該批（{_lot(rng)}）元件在包裝內本身方向反向，IQC 已確認", "D",
            base + f"，查 feeder slot {s} 料捲裝載方向與料表相反，操作員 {pick(rng, NAMES)} 換料時未依 SOP 核對極性方向，元件本身與料表一致", "F")


def bga_ad(rng):  # A / D
    line, spi, r = pick(rng, LINES), pick(rng, SPIS), pick(rng, REFLOWS)
    pn, part = "U105-BGA256", "BGA-256"
    base = f"{line} X-ray 發現 {pn}（{part}）邊角多顆錫球空焊"
    return (base + f"，{spi} 該位置印刷後錫膏體積只有 {rng.randint(48, 65)}%，錫膏開封已 {rng.randint(9, 13)} 小時超過 8 小時規定，回焊曲線在規格內", "A",
            base + f"，{spi} 體積正常 {rng.randint(95, 106)}%、回焊爐 {r} 曲線在規格內，該批 PCB（{_lot(rng)}）量測板彎 {rng.choice([1.2, 1.4, 1.6, 1.8])}mm 超過 0.75% 規格，換批後正常", "D")


def missing_be(rng):  # B / E
    line, aoi, m, s = pick(rng, LINES), pick(rng, AOIS), pick(rng, MOUNTERS), slot(rng)
    pn, part = pick(rng, _SMALL)
    n = rng.randint(4, 15)
    base = f"{line} {aoi} 發現 {pn}（{part}）缺件 {n} 片"
    return (base + f"，貼片機 {m} slot {s} 拋料率 {rng.randint(8, 18)}% 遠高於 1% 上限，拋料後未補貼，載具與 PCB 正常", "B",
            base + f"，貼片機 {m} slot {s} 拋料率 {rng.choice([0.2, 0.3, 0.4])}% 且貼片紀錄有貼，{_carrier(rng)} 壓板變形、板子未壓平，該區元件進爐前被刮落", "E")


def cold_cf(rng):  # C / F
    line, spi, r = pick(rng, LINES), pick(rng, SPIS), pick(rng, REFLOWS)
    pn, part = pick(rng, _ICS)
    base = f"{line} {pn}（{part}）焊點冷焊、表面粗糙"
    return (base + f"，{spi} 與爐前 AOI 皆正常，回焊爐 {r} 量測峰溫 {rng.randint(214, 226)}°C 低於規格下限 230°C", "C",
            base + f"，回焊爐 {r} 峰溫 {rng.randint(240, 248)}°C 在規格內、同板其他元件正常，追查該點經手修站重工，手修人員 {pick(rng, NAMES)} 烙鐵設定 {rng.randint(250, 290)}°C 低於 SOP 規定 350°C 且未加助焊劑", "F")


def paste_ae(rng):  # A / E
    line, spi, p = pick(rng, LINES), pick(rng, SPIS), pick(rng, PRINTERS)
    pn = pick(rng, PNS)
    base = f"{line} {spi} {pn} 多點少錫，體積 {rng.randint(55, 70)}%"
    return (base + f"，{p} 錫膏 {_lot(rng)} 冷藏取出後未回溫直接上機、攪拌紀錄空白，鋼網開孔檢查正常", "A",
            base + f"，錫膏回溫與攪拌紀錄正常，{p} {_stencil(rng)} 已連續印 {rng.randint(300, 900)} 片未清洗，開孔顯微鏡檢查 {rng.randint(4, 12)} 處堵塞", "E")


def flip_bf(rng):  # B / F
    line, aoi, m = pick(rng, LINES), pick(rng, AOIS), pick(rng, MOUNTERS)
    pn, part = pick(rng, _SMALL + _LEDS)
    base = f"{line} {aoi} 發現 {pn}（{part}）翻件、標示面朝下"
    return (base + f"，貼片機 {m} head {rng.randint(1, 8)} 吸嘴真空值 {rng.randint(40, 55)}kPa 低於下限 60kPa，元件貼放時翻轉，同 head 拋料率偏高，該板未經手修", "B",
            base + f"，貼片後首件 AOI 該位置正常，該板經手修站更換此元件，手修人員 {pick(rng, NAMES)} 重工時裝反且未依 SOP 做重工後目檢", "F")


def void_cd(rng):  # C / D
    line, spi, r = pick(rng, LINES), pick(rng, SPIS), pick(rng, REFLOWS)
    pn, part = "U105-BGA256", "BGA-256"
    base = f"{line} X-ray {pn}（{part}）空洞率 {rng.randint(27, 42)}% 超過 25% 上限"
    return (base + f"，{spi} 正常，回焊爐 {r} 曲線 soak 時間只有 {rng.randint(28, 45)}s 低於規格下限 60s", "C",
            base + f"，{spi} 正常、回焊爐 {r} 曲線在規格內，IQC 確認該批 PCB（{_lot(rng)}）焊墊 OSP 膜異常，改用另一批 PCB 後空洞率降到 {rng.randint(4, 9)}%", "D")


PAIRS_ZH = [offset_ab, ball_ac, tombstone_bc, insufficient_de, program_ef, polarity_df, bga_ad, missing_be,
            cold_cf, paste_ae, flip_bf, void_cd]


# ---------------------------------------------------------------- en pairs

def en_offset(rng):  # A / B
    aoi, spi, m = pick(rng, ["AOI-01", "AOI-02"]), pick(rng, SPIS), pick(rng, MOUNTERS)
    pn = pick(rng, _SMALL)[0]
    off = rng.randint(70, 140)
    return (f"{aoi} DEFECT {pn} OFFSET {off}UM {spi} SAME PAD PASTE OFFSET {off - rng.randint(5, 20)}UM SAME DIRECTION FLAGGED BEFORE PLACEMENT {m} CALIBRATION OK", "A",
            f"{aoi} DEFECT {pn} OFFSET {off}UM {spi} PASTE OFFSET <{rng.randint(5, 10)}UM OK {m} HEAD{rng.randint(1, 8)} NOZZLE CALIBRATION OVERDUE {rng.randint(10, 40)} DAYS", "B")


def en_void(rng):  # C / D
    spi, r = pick(rng, SPIS), pick(rng, REFLOWS)
    v = rng.randint(27, 42)
    return (f"XRAY U105-BGA256 VOID {v}% LIMIT 25% {spi} OK REFLOW {r} SOAK {rng.randint(28, 45)}S SPEC MIN 60S", "C",
            f"XRAY U105-BGA256 VOID {v}% LIMIT 25% {spi} OK REFLOW {r} PROFILE IN SPEC PCB LOT {rng.randint(2400, 2699)} PAD OSP DEFECT CONFIRMED BY IQC NEW LOT VOID {rng.randint(4, 9)}%", "D")


def en_fixture(rng):  # E / F
    m, w = pick(rng, MOUNTERS), wo(rng)
    v, off = rng.randint(2, 7), rng.randint(150, 300)
    return (f"{m} {w} FIRST ARTICLE ALL COMPONENTS OFFSET {off}UM SAME DIRECTION PROGRAM V{v} CORRECT CARRIER CR-{rng.randint(10, 89)} LOCATING PIN WEAR {rng.choice([0.2, 0.25, 0.3])}MM LIMIT 0.1MM", "E",
            f"{m} {w} FIRST ARTICLE ALL COMPONENTS OFFSET {off}UM SAME DIRECTION PROGRAM LOADED V{v - 1} WORK ORDER SPECIFIES V{v} OPERATOR SELECTED WRONG PROGRAM CARRIER OK", "F")


def en_bridge(rng):  # A / C
    aoi, spi, r = pick(rng, ["AOI-02", "AOI-03"]), pick(rng, SPIS), pick(rng, REFLOWS)
    pn = pick(rng, _ICS)[0]
    a = rng.randint(1, 40)
    return (f"POST-REFLOW {aoi} {pn} BRIDGE PIN{a}-{a + 1} {spi} FLAGGED PASTE BRIDGE SAME PINS VOLUME {rng.randint(135, 170)}% BEFORE PLACEMENT REFLOW {r} PROFILE IN SPEC", "A",
            f"POST-REFLOW {aoi} {pn} SOLDER BALLS NEAR PIN{a}-{a + 1} {spi} OK PRE-REFLOW AOI OK REFLOW {r} PREHEAT RAMP {rng.choice([3.6, 3.9, 4.2])}C/S SPEC MAX 3.0C/S", "C")


def en_missing(rng):  # B / D
    aoi, m, s = pick(rng, ["AOI-01", "AOI-02"]), pick(rng, MOUNTERS), slot(rng)
    pn = pick(rng, _SMALL)[0]
    n = rng.randint(4, 15)
    return (f"{aoi} {pn} MISSING {n} BOARDS {m} SLOT {s} REJECT RATE {rng.randint(8, 18)}% LIMIT 1% NO RE-PICK TAPE POCKETS FULL", "B",
            f"{aoi} {pn} MISSING {n} BOARDS {m} SLOT {s} REJECT RATE {rng.choice([0.2, 0.3, 0.4])}% TAPE LOT {rng.randint(2400, 2699)} EMPTY POCKETS FOUND IQC CONFIRMED SUPPLIER LOT DEFECT", "D")


PAIRS_EN = [en_offset, en_void, en_fixture, en_bridge, en_missing]
