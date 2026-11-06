"""v14 data (docs/handoff-v14-sop-routing.md): Gemini writes a 100-SOP catalog and 3 shop-floor messages per SOP.

Claude only writes this spec and spot-checks; Gemini writes the content (blind, like v3's D2).
  gemini_key=... python3 data/v14/gen_v14.py catalog     -> data/v14/sop_catalog.json
  gemini_key=... python3 data/v14/gen_v14.py queries     -> data/v14/queries.jsonl
"""
import json
import os
import random
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemini-3.5-flash"
CATS = ["貼片機（NXT、吸嘴、feeder、貼裝偏移）", "錫膏印刷（DEK、鋼網、刮刀、錫膏管理）", "回焊爐與波焊（溫度曲線、氮氣、輸送）",
        "SPI／AOI／X-ray 檢測（誤判、程式、校正）", "物料與倉儲（MSD 濕敏、料號、上料核對、退料）", "ICT／FCT 測試站與治具",
        "重工與維修（烙鐵、BGA 重植、修補判定）", "品質處置（異常單、隔離、首件、抽檢）", "ESD 與工安（靜電手環、化學品、急救）",
        "廠務設施（空壓、氮氣、空調、停電、廢液）", "MES／IT（條碼、工單、系統當機、標籤列印）", "換線與排程（換線準備、程式切換、交班）"]


def gemini(prompt, temperature=0.9, tries=4):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={os.environ['gemini_key']}"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:  # noqa: BLE001
            print("retry", i, repr(e)[:200], flush=True); time.sleep(5 * (i + 1))
    raise RuntimeError("gemini failed")


def catalog():
    prompt = f"""你是台灣一間 SMT（表面黏著）電子組裝廠的製程工程師。請建立一份「現場 SOP 目錄」，共 100 份 SOP，涵蓋以下 12 類（每類 7–10 份）：
{chr(10).join('- ' + c for c in CATS)}

規則：
1. 每份 SOP 給 id（S001–S100）、title（8–16 個中文字，像真實 SOP 標題）、scope（20–45 個中文字，寫「什麼情況下要用這份 SOP」，不要寫步驟）。
2. 其中刻意設計 20 組「近似孿生」：同一類、主題很接近、但適用情況明確不同（例：吸嘴堵塞清潔 vs 吸嘴磨損更換；錫膏回溫不足 vs 錫膏開封逾時）。在這 40 份上加欄位 twin_of，填另一份的 id。
3. 用台灣正體中文與台灣產線用語（可夾常見英文縮寫如 NXT、AOI、MSD、MES）。不要簡體字、不要大陸用語。
4. 100 份的 title 不可重複，scope 要能讓人分辨彼此。
只輸出 JSON：{{"sops": [{{"id": "S001", "category": "...", "title": "...", "scope": "...", "twin_of": null}}, ...]}}"""
    for attempt in range(4):
        d = gemini(prompt, temperature=0.7)
        sops = d.get("sops", [])
        ids = {s["id"] for s in sops}; titles = [s["title"] for s in sops]
        twins = [s for s in sops if s.get("twin_of")]
        ok_twins = all(s["twin_of"] in ids and next(x for x in sops if x["id"] == s["twin_of"]).get("twin_of") == s["id"] for s in twins)
        print(f"attempt {attempt}: n={len(sops)} unique_titles={len(set(titles))} twins={len(twins)} symmetric={ok_twins}", flush=True)
        if len(sops) == 100 and len(set(titles)) == 100 and 36 <= len(twins) <= 44 and ok_twins:
            json.dump({"model": MODEL, "sops": sops}, open(os.path.join(HERE, "sop_catalog.json"), "w"), ensure_ascii=False, indent=1)
            return
    raise SystemExit("catalog did not validate")


def queries():
    cat = json.load(open(os.path.join(HERE, "sop_catalog.json"), encoding="utf-8"))["sops"]
    by_id = {s["id"]: s for s in cat}
    out_path = os.path.join(HERE, "queries.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["sop_id"] for l in open(out_path, encoding="utf-8")}
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()

    def one(s):
        twin = by_id.get(s["twin_of"]) if s.get("twin_of") else None
        twin_txt = f"\n注意：還有一份很像的 SOP「{twin['title']}：{twin['scope']}」。你寫的訊息必須只適用目標 SOP、不適用這一份。" if twin else ""
        prompt = f"""你在台灣一間 SMT 電子組裝廠，請以現場人員（作業員、技術員、線長）在 LINE 工作群組裡傳訊息的口吻，寫 3 則訊息。每一則描述的狀況，都應該去參照下面這份 SOP 處理：
目標 SOP「{s['title']}：{s['scope']}」{twin_txt}

規則：
1. 不要照抄 SOP 標題裡的關鍵名詞，要描述看到的現象、數字或狀況，讓人自己判斷該看哪份 SOP。
2. 三則風格不同：第 1 則直接描述現象；第 2 則口語、可以有英文縮寫或簡寫；第 3 則夾雜一點和處理無關的資訊（例：在哪條線、誰在場、時間）。
3. 每則 15–60 個中文字，台灣正體中文、台灣產線用語，不要簡體字。
只輸出 JSON：{{"messages": ["...", "...", "..."]}}"""
        d = gemini(prompt)
        msgs = [m.strip() for m in d.get("messages", []) if isinstance(m, str) and m.strip()][:3]
        with lock, open(out_path, "a", encoding="utf-8") as f:
            for i, m in enumerate(msgs):
                f.write(json.dumps({"id": f"{s['id']}-q{i}", "sop_id": s["id"], "style": i, "state": m, "has_twin": bool(twin),
                                    "title_leak": s["title"] in m}, ensure_ascii=False) + "\n")
        print(s["id"], len(msgs), flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, [s for s in cat if s["id"] not in done]))


if __name__ == "__main__":
    {"catalog": catalog, "queries": queries}[sys.argv[1]]()
