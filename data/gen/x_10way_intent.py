"""Easy generators for x_10way_intent (A 查詢工單進度 / B 查詢機台狀態 / C 回報機台異常 / D 回報品質異常 /
E 申請物料 / F 申請維修 / G 申請換線 / H 申請權限或帳號 / I 詢問 SOP 或參數 / J 回報完工).

Each PAIRS_* entry is a function(rng) -> (state_a, gold_a, state_b, gold_b): a twin pair of
LINE/Teams-style messages that share the same subject but flip between two confusable intents.
Gold comes from the template spec, never from post-hoc judgement.
"""
from ._common import LINES, MOUNTERS, REFLOWS, AOIS, SPIS, PARTS, PNS, NAMES, wo, hhmm, pick, slot

ALARMS = ["E4021", "F-1207", "E2310", "S-0908", "E7702"]
DEFECTS = ["偏移", "立碑", "空焊", "短路", "少件"]


def progress_done(rng):
    name, w, q = pick(rng, NAMES), wo(rng), rng.randint(200, 800)
    return (f"{name}：{w} 現在做到幾片了？客戶一直在催", "A",
            f"{name}：{w} {q} 片全部做完，已經入庫了", "J")


def eta_done(rng):
    name, w, t = pick(rng, NAMES), wo(rng), hhmm(rng)
    return (f"{name}：@線長 {w} 預計幾點可以做完？我要排下一張", "A",
            f"{name}：@線長 {w} {t} 做完了，最後一片過 AOI OK", "J")


def running_alarm(rng):
    name, line, m, a = pick(rng, NAMES), pick(rng, LINES), pick(rng, MOUNTERS), pick(rng, ALARMS)
    return (f"{name}：{line} {m} 現在有在跑嗎？看板上一直沒更新", "B",
            f"{name}：{line} {m} 跳 {a} alarm 停機了，麻煩 ME 來看一下", "C")


def util_servo(rng):
    name, line, m = pick(rng, NAMES), pick(rng, LINES), pick(rng, MOUNTERS)
    return (f"{name}：{line} {m} 今天稼動率多少？主管 {hhmm(rng)} 要看數字", "B",
            f"{name}：{line} {m} 剛剛 X 軸 servo 過載，整台卡住不動，快來一下", "C")


def reflow_now_later(rng):
    name, r, z = pick(rng, NAMES), pick(rng, REFLOWS), rng.randint(1, 10)
    return (f"{name}：{r} 第 {z} 區溫度掉了 {rng.randint(10, 20)} 度在報警，板子還在爐內，請馬上處理", "C",
            f"{name}：{r} 第 {z} 區加熱管老化，溫度偏差越來越大，請 ME 找時間安排更換", "F")


def aoi_ng_crash(rng):
    name, a, n = pick(rng, NAMES), pick(rng, AOIS), rng.randint(6, 20)
    return (f"{name}：{a} 這批連續 {n} 片{pick(rng, DEFECTS)}，NG 率 {rng.randint(20, 60)}%，請 QE 來判", "D",
            f"{name}：{a} 相機當機畫面全黑，板子卡在裡面出不來，請處理", "C")


def material_progress(rng):
    name, w, pn, k = pick(rng, NAMES), wo(rng), pick(rng, PNS), rng.randint(2, 6)
    return (f"{name}：{w} 的 {pn} 還差 {k} 盤才夠跑完，請倉庫發料到 {pick(rng, LINES)}", "E",
            f"{name}：{w} 現在做到第幾片了？下午要報產量", "A")


def nozzle_sop_perm(rng):
    name, m, pn = pick(rng, NAMES), pick(rng, MOUNTERS), pick(rng, PNS)
    return (f"{name}：{m} 貼 {pn} 要用幾號吸嘴？SOP 上沒寫", "I",
            f"{name}：MES 的換線頁面我點不進去，說帳號沒這功能，請幫我開權限", "H")


def profile_account(rng):
    name, r, w, new = pick(rng, NAMES), pick(rng, REFLOWS), wo(rng), pick(rng, NAMES)
    return (f"{name}：{r} 跑 {w} 的 profile 峰溫要設幾度？", "I",
            f"{name}：新來的 {new} 要一組 MES 帳號，刷過站用，請幫忙申請", "H")


def early_progress(rng):
    name, w, w2, t = pick(rng, NAMES), wo(rng), wo(rng), hhmm(rng)
    return (f"{name}：{w} 料到了，可以提前到 {t} 換線嗎？{w2} 想先插進來跑", "G",
            f"{name}：{w} 做到哪了？預計幾點完工？", "A")


def feeder_fixed_fix(rng):
    name, m, s = pick(rng, NAMES), pick(rng, MOUNTERS), slot(rng)
    return (f"{name}：{m} slot {s} 修好了，ME 已換新彈片，機台恢復生產", "J",
            f"{name}：{m} slot {s} 彈片鬆了拋料偏高，請 ME 排時間換 feeder", "F")


def changeover_done_delay(rng):
    name, line, w, t = pick(rng, NAMES), pick(rng, LINES), wo(rng), hhmm(rng)
    return (f"{name}：{line} 換線完成，{w} 首件 OK 開始跑了", "J",
            f"{name}：{line} 可以把 {w} 換線延到 {t} 嗎？前一張還沒做完", "G")


def zaxis_status_fix(rng):
    name, m = pick(rng, NAMES), pick(rng, MOUNTERS)
    return (f"{name}：{m} 昨天報修的 Z 軸修好了嗎？機台現在有在跑了嗎？", "B",
            f"{name}：{m} Z 軸有異音但還能跑，請 ME 安排週{rng.choice(['六', '日'])}檢修", "F")


def kanban_status_perm(rng):
    name, line, m = pick(rng, NAMES), pick(rng, LINES), pick(rng, MOUNTERS)
    return (f"{name}：{line} {m} 現在是跑中還是停機？看板沒顯示", "B",
            f"{name}：看板的機台狀態頁我沒權限看，請幫我開一下", "H")


def spec_ng(rng):
    name, w, part = pick(rng, NAMES), wo(rng), pick(rng, PARTS)
    return (f"{name}：{w} 的 {part} 偏移幾 mil 以內算 OK？檢驗標準是多少", "I",
            f"{name}：{w} 首件 {part} 偏移 {rng.randint(4, 8)} mil 超標，判 NG，請 QE 處理", "D")


def reel_bad_short(rng):
    name, pn, w = pick(rng, NAMES), pick(rng, PNS), wo(rng)
    return (f"{name}：{pn} 這盤料貼上去全部翹，來料有問題，已攔下 {rng.randint(5, 20)} 片", "D",
            f"{name}：{pn} 線邊只剩 {rng.randint(100, 500)} 顆，跑不完 {w}，請倉庫補 {rng.randint(2, 5)} 盤", "E")


def refill_feeder(rng):
    name, m, s, pn = pick(rng, NAMES), pick(rng, MOUNTERS), slot(rng), pick(rng, PNS)
    return (f"{name}：{m} slot {s} 的 {pn} 用完了，請倉庫再發 {rng.randint(2, 5)} 盤", "E",
            f"{name}：{m} slot {s} feeder 常卡料，請 ME 排時間換一支", "F")


def short_switch(rng):
    name, w, w2, pn, line = pick(rng, NAMES), wo(rng), wo(rng), pick(rng, PNS), pick(rng, LINES)
    return (f"{name}：{w} 缺 {pn} {rng.randint(2, 5)} 盤，請倉庫儘快補到 {line}", "E",
            f"{name}：{w} 的 {pn} 明天才到，可以先換 {w2} 上來跑嗎？", "G")


def changeover_sop_early(rng):
    name, w, t = pick(rng, NAMES), wo(rng), hhmm(rng)
    return (f"{name}：換到 {w} 是先換鋼網還是先切程式？SOP 順序是什麼", "I",
            f"{name}：請排程把 {w} 換線提前到 {t}，客戶急要", "G")


def spc_lock_ooc(rng):
    name, s = pick(rng, NAMES), pick(rng, SPIS)
    return (f"{name}：SPC 系統我登不進去，帳號被鎖了，請幫我解鎖", "H",
            f"{name}：{s} SPC 連續 {rng.randint(5, 9)} 點超 UCL，Cpk 掉到 {pick(rng, [0.7, 0.8, 0.9])}，請 QE 看一下", "D")


PAIRS_ZH = [progress_done, running_alarm, reflow_now_later, material_progress, nozzle_sop_perm, early_progress,
            feeder_fixed_fix, changeover_done_delay, zaxis_status_fix, kanban_status_perm, spec_ng, reel_bad_short,
            short_switch, util_servo, aoi_ng_crash, profile_account, refill_feeder, changeover_sop_early, spc_lock_ooc, eta_done]


def en_progress_done(rng):
    name, w, q = pick(rng, ["Tom", "Amy", "Ken", "Lisa", "Ray"]), wo(rng), rng.randint(200, 800)
    return (f"{name}: how many done on {w}? customer is chasing", "A",
            f"{name}: {w} all {q} pcs done, moved to stock", "J")


def en_running_alarm(rng):
    name, m, a = pick(rng, ["Tom", "Amy", "Ken", "Lisa", "Ray"]), pick(rng, MOUNTERS), pick(rng, ALARMS)
    return (f"{name}: is {m} running right now? dashboard not updating", "B",
            f"{name}: {m} threw {a} alarm and stopped, need ME here now", "C")


def en_refill_fix(rng):
    name, m, s, pn = pick(rng, ["Tom", "Amy", "Ken", "Lisa", "Ray"]), pick(rng, MOUNTERS), slot(rng), pick(rng, PNS)
    return (f"{name}: {m} slot {s} {pn} ran out, need {rng.randint(2, 5)} more reels from warehouse", "E",
            f"{name}: {m} slot {s} feeder keeps jamming, pls schedule ME to swap it", "F")


def en_sop_perm(rng):
    name, r, w = pick(rng, ["Tom", "Amy", "Ken", "Lisa", "Ray"]), pick(rng, REFLOWS), wo(rng)
    return (f"{name}: what peak temp should {r} be set to for {w}?", "I",
            f"{name}: can't open the recipe page in MES, says no permission, pls grant access", "H")


def en_ng_switch(rng):
    name, w, w2, t = pick(rng, ["Tom", "Amy", "Ken", "Lisa", "Ray"]), wo(rng), wo(rng), hhmm(rng)
    return (f"{name}: {w} first article shifted {rng.randint(4, 8)} mil, out of spec, NG, QE pls check", "D",
            f"{name}: can we move {w} changeover up to {t} and run {w2} first?", "G")


PAIRS_EN = [en_progress_done, en_running_alarm, en_refill_fix, en_sop_perm, en_ng_switch]
