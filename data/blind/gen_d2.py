"""D2 blind set: Gemini writes production-line states WITHOUT seeing the criteria; labels come later
from two independent annotators (Gemini with full criteria, and Claude/Fable with full criteria).

Usage:
  python3 data/blind/gen_d2.py write   [--per-task 60] [--model gemini-3.5-flash]
  python3 data/blind/gen_d2.py label   [--model gemini-3.5-flash]     # Gemini labels with criteria
Outputs: data/blind/D2/states_{task}.jsonl, data/blind/D2/labels_gemini.jsonl
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = json.load(open(os.path.join(ROOT, "data/seeds/tasks.json"), encoding="utf-8"))
OUT = os.path.join(ROOT, "data/blind/D2")
KEY = os.environ["gemini_key"]


def gemini(model, prompt, json_out=True, temperature=1.0, retries=4):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192}}
    if json_out:
        body["generationConfig"]["responseMimeType"] = "application/json"
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"])
            return text, r.get("usageMetadata")
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            print("HTTP", e.code, msg, file=sys.stderr)
            if e.code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(20 * (attempt + 1))
                continue
            raise
        except Exception as e:  # noqa: BLE001
            print("ERR", repr(e), file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(10)
                continue
            raise


def d0_examples(task, n, rng):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, f"data/synthetic/{task}.jsonl"), encoding="utf-8")]
    rng.shuffle(rows)
    return [r["state"] for r in rows[:n]]


def write_states(model, per_task, batch=20):
    os.makedirs(OUT, exist_ok=True)
    for task, t in TASKS.items():
        path = os.path.join(OUT, f"states_{task}.jsonl")
        have = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
        if len(have) >= per_task:
            print(f"[{task}] already {len(have)}")
            continue
        rng = random.Random(f"d2-{task}")
        labels = "、".join(v["label"] for v in t["options"].values())
        f = open(path, "a", encoding="utf-8")
        i = len(have)
        while i < per_task:
            n = min(batch, per_task - i)
            refs = d0_examples(task, 20, random.Random(f"ref-{task}-{i}"))
            prompt = (
                "你是台灣 EMS/SMT 電子組裝廠的產線人員。請寫出產線上真實會出現的訊息（alarm log、MES 狀態、LINE/Teams 上操作員或線長打的話、工單備註）。\n\n"
                f"題目：「{t['instructions']}」\n可能的答案只有這幾種（你不需要標答案，只要寫情境）：{labels}\n\n"
                "風格參考（只參考語氣與長度，不要抄、不要改寫這些句子）：\n" + "\n".join(f"- {s}" for s in refs) + "\n\n"
                f"請寫 {n} 則**全新**的訊息，要求：\n"
                "1. 像真實訊息：操作員口語、半句話、縮寫（ME/PE/QE/WO/MES/SPI/AOI）、可能夾錯字、可能兩件事混在一則、alarm code 可能和對照表對不上。\n"
                "2. 每一則都要讓一個有經驗的線長讀完能判斷該選哪個答案，但線索不要寫得太直白，不要出現答案的字面。\n"
                "3. 各種答案都要有，大致平均；其中約 20% 用純英文 alarm/MES log 風格，其餘台灣正體中文夾機台代碼。\n"
                "4. 機台、線別、料號、時間、數量要有變化，不要重複句型。\n"
                "5. 長度 15–80 字。\n\n"
                '只輸出 JSON：{"states": ["...", "..."]}'
            )
            text, usage = gemini(model, prompt)
            try:
                states = json.loads(text)["states"]
            except Exception:  # noqa: BLE001
                print(f"[{task}] bad json, retrying batch", file=sys.stderr)
                continue
            for s in states:
                s = s.strip()
                if len(s) < 8:
                    continue
                f.write(json.dumps({"id": f"{task}-d2-{i:03d}", "task": task, "state": s, "writer": model}, ensure_ascii=False) + "\n")
                i += 1
                if i >= per_task:
                    break
            f.flush()
            print(f"[{task}] {i}/{per_task} (usage {usage and usage.get('totalTokenCount')})", flush=True)
            time.sleep(4)  # ~15 RPM free tier
        f.close()


def label(model, batch=15):
    out_path = os.path.join(OUT, "labels_gemini.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
    f = open(out_path, "a", encoding="utf-8")
    for task, t in TASKS.items():
        path = os.path.join(OUT, f"states_{task}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if json.loads(l)["id"] not in done]
        letters = list(t["options"])
        opts = "\n".join(f"{k}. {v['label']}：{v['criteria']}" for k, v in t["options"].items())
        for b in range(0, len(rows), batch):
            chunk = rows[b:b + batch]
            prompt = (
                "你是產線判斷的標註者。依下面的判斷標準，為每一則訊息選出**唯一**最合適的選項；若訊息資訊不足以判斷，仍選最合理者並把 confidence 設低。\n\n"
                f"【問題】{t['instructions']}\n{opts}\n\n"
                "訊息：\n" + "\n".join(f"[{r['id']}] {r['state']}" for r in chunk) + "\n\n"
                '只輸出 JSON：{"labels": [{"id": "...", "label": "A", "confidence": 0.0-1.0, "reason": "一句話"}]}'
            )
            text, _ = gemini(model, prompt, temperature=0.2)
            try:
                labs = json.loads(text)["labels"]
            except Exception:  # noqa: BLE001
                print(f"[{task}] bad json at {b}", file=sys.stderr)
                continue
            for l in labs:
                if l.get("label") in letters:
                    f.write(json.dumps({"id": l["id"], "task": task, "label": l["label"], "confidence": l.get("confidence"),
                                        "reason": l.get("reason"), "annotator": model}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{task}] labeled {min(b + batch, len(rows))}/{len(rows)}", flush=True)
            time.sleep(4)
    f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "label"])
    ap.add_argument("--per-task", type=int, default=60)
    ap.add_argument("--model", default="gemini-3.5-flash")
    a = ap.parse_args()
    if a.cmd == "write":
        write_states(a.model, a.per_task)
    else:
        label(a.model)
