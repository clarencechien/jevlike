"""tw2 part B (docs/handoff-tw2-decision-apis.md §3.1): translate data/tw2/master_en.json and build data/tw2/jobs.jsonl.

- B2 languages and the Traditional-Chinese / Serbian-Cyrillic / Albanian versions: Gemini (generative model, `gemini_key`), batch per
  dispute x language, strict JSON out, meaning and stance kept, place names as customary in that language.
- Script-only pairs are deterministic: zh_cn of the B1 Chinese disputes = OpenCC t2s of the zh_tw text; sr_latn = letter-by-letter
  transliteration of sr_cyrl (same as tw: only the script differs).
- Each job also carries the mitigation texts used by bench/tw_script.py's extra arms:
    alt.norm  = script unified (zh_cn -> Traditional with OpenCC s2twp; sr_cyrl -> Latin), else identical
    alt.pivot = the job's own-language statement machine-translated back to English by Gemini (a realistic pivot, not the master)
Usage: python3 bench/tw2_translate.py [--model gemini-3.1-pro-preview] [--only dispute,dispute]
Out: data/tw2/translations/<dispute>.json (cache; re-run only fills gaps; --only a,b translates those in this process), data/tw2/jobs.jsonl, data/tw2/MANIFEST.md
"""
import argparse
import json
import os
import re
import time

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "data", "tw2")
LANG_NAME = {"zh_tw": "Traditional Chinese as used in Taiwan", "zh_cn": "Simplified Chinese as used in mainland China",
             "sr_cyrl": "Serbian in Cyrillic script", "sq": "Albanian", "ru": "Russian", "uk": "Ukrainian", "hi": "Hindi (Devanagari)",
             "ur": "Urdu (Perso-Arabic script)", "ko": "Korean", "ja": "Japanese", "vi": "Vietnamese", "fil": "Filipino (Tagalog)",
             "es": "Spanish (Argentina)", "el": "Greek", "tr": "Turkish"}
CYR2LAT = dict(zip("АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШабвгдђежзијклљмнњопрстћуфхцчџш",
                   ["A", "B", "V", "G", "D", "Đ", "E", "Ž", "Z", "I", "J", "K", "L", "Lj", "M", "N", "Nj", "O", "P", "R", "S", "T", "Ć", "U", "F", "H", "C", "Č",
                    "Dž", "Š", "a", "b", "v", "g", "d", "đ", "e", "ž", "z", "i", "j", "k", "l", "lj", "m", "n", "nj", "o", "p", "r", "s", "t", "ć", "u", "f",
                    "h", "c", "č", "dž", "š"]))
_CC = {}


def cc(conf, text):
    import opencc
    if conf not in _CC:
        _CC[conf] = opencc.OpenCC(conf)
    return _CC[conf].convert(text)


def cyr2lat(text):
    return "".join(CYR2LAT.get(ch, ch) for ch in text)


def gemini(model, prompt, tries=6):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    delay = 3
    for _ in range(tries):
        r = requests.post(url, headers={"x-goog-api-key": os.environ["gemini_key"], "Content-Type": "application/json"}, json=body, timeout=300)
        if r.status_code == 200:
            txt = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
            return json.loads(txt)
        time.sleep(delay); delay *= 2
    raise RuntimeError(f"gemini failed: {r.status_code} {r.text[:300]}")


def resolve(text, dispute, lang):
    """English master with {NAME} placeholders filled by the English name (translator is told the customary local name)."""
    for k, v in (dispute.get("names") or {}).items():
        text = text.replace("{" + k + "}", v["en"])
    return text


def translate_batch(model, sentences, lang, dispute):
    names = "; ".join(f"{v['en']} -> {v['note']}" for v in (dispute.get("names") or {}).values())
    prompt = (f"Translate each English sentence into {LANG_NAME[lang]}.\n"
              "Rules: keep the exact meaning and the exact stance; do not soften, strengthen, hedge or add words like 'allegedly'; "
              "keep each sentence a plain declarative statement (or a plain question for questions); "
              + (f"use the place name customary in {LANG_NAME[lang]} mainstream usage ({names}); " if names else "")
              + "use standard, neutral written register.\n"
              f"Return a JSON array of {len(sentences)} strings, same order, nothing else.\n\n"
              + json.dumps(sentences, ensure_ascii=False))
    out = gemini(model, prompt)
    assert isinstance(out, list) and len(out) == len(sentences), out
    return out


def to_english(model, sentences, lang):
    prompt = (f"Translate each {LANG_NAME.get(lang, lang)} sentence into English, literally and faithfully, keeping the stance. "
              f"Return a JSON array of {len(sentences)} strings, same order, nothing else.\n\n" + json.dumps(sentences, ensure_ascii=False))
    out = gemini(model, prompt)
    assert isinstance(out, list) and len(out) == len(sentences), out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--only", default="")
    ap.add_argument("--build-only", action="store_true", help="skip translation; merge the per-dispute caches and write jobs")
    a = ap.parse_args()
    master = json.load(open(os.path.join(D, "master_en.json"), encoding="utf-8"))
    # one cache file per dispute (translations/<id>.json) so disputes can be translated in parallel processes
    os.makedirs(os.path.join(D, "translations"), exist_ok=True)
    cache = {}
    for d in master["disputes"]:
        p = os.path.join(D, "translations", f"{d['id']}.json")
        if os.path.exists(p):
            cache.update(json.load(open(p, encoding="utf-8")))
    old = os.path.join(D, "translations.json")
    if os.path.exists(old):
        cache.update(json.load(open(old, encoding="utf-8")))
    q_en = master["questions_en"]

    def units(d):  # (unit id, english text) for the 4 pairs x 2 sides + 2 neutral x 2
        u = []
        for p in d["pairs"]:
            u += [(f"{p['pid']}-X", resolve(p["X"], d, "en")), (f"{p['pid']}-Y", resolve(p["Y"], d, "en"))]
        for n in d["neutral"]:
            u += [(f"{n['pid']}-true", n["true"]), (f"{n['pid']}-false", n["false"])]
        return u

    only = set(a.only.split(",")) if a.only else None
    for d in master["disputes"]:
        if a.build_only or (only and d["id"] not in only):
            continue
        us = units(d)
        for lang in d["langs"]:
            key = f"{d['id']}|{lang}"
            if key in cache:
                continue
            ids, en = [u[0] for u in us], [u[1] for u in us]
            if lang == "en":
                st, qs = en, q_en
            elif lang == "zh_cn" and d["group"] == "B1":
                base = cache.get(f"{d['id']}|zh_tw") or {}
                if not base:
                    raise SystemExit("translate zh_tw first (langs order)")
                st, qs = [cc("t2s", x) for x in base["statements"]], [cc("t2s", x) for x in master["questions_zh_tw"]]
            elif lang == "sr_latn":
                base = cache[f"{d['id']}|sr_cyrl"]
                st, qs = [cyr2lat(x) for x in base["statements"]], [cyr2lat(x) for x in base["questions"]]
            elif lang == "zh_tw":
                st, qs = translate_batch(a.model, en, lang, d), master["questions_zh_tw"]
            else:
                st, qs = translate_batch(a.model, en, lang, d), translate_batch(a.model, q_en, lang, d)
            cache[key] = {"ids": ids, "statements": st, "questions": qs}
            # pivot: own-language statement and question back to English (machine translation, as a deployment would do)
            if lang != "en":
                cache[key]["pivot_statements"] = to_english(a.model, st, lang)
                cache[key]["pivot_questions"] = to_english(a.model, qs, lang)
            json.dump({k: v for k, v in cache.items() if k.startswith(d["id"] + "|")},
                      open(os.path.join(D, "translations", f"{d['id']}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"[{key}] {len(st)} statements", flush=True)

    # ---- jobs
    jobs = []
    for d in master["disputes"]:
        for lang in d["langs"]:
            c = cache.get(f"{d['id']}|{lang}")
            if not c:
                continue
            for i, uid in enumerate(c["ids"]):
                pid, side = uid.rsplit("-", 1)
                for k in range(3):
                    if lang == "zh_cn" and d["group"] == "B1":
                        norm = {"state": cc("s2twp", c["statements"][i]), "instructions": cc("s2twp", c["questions"][k]), "lang": "zh_tw"}
                    elif lang == "sr_cyrl":
                        norm = {"state": cyr2lat(c["statements"][i]), "instructions": cyr2lat(c["questions"][k]), "lang": "sr_latn"}
                    else:
                        norm = {"state": c["statements"][i], "instructions": c["questions"][k], "lang": lang}
                    pivot = ({"state": c["pivot_statements"][i], "instructions": c["pivot_questions"][k], "lang": "en"} if lang != "en"
                             else {"state": c["statements"][i], "instructions": c["questions"][k], "lang": "en"})
                    jobs.append({"job": f"B|{d['id']}|{uid}|{lang}|p{k + 1}", "exp": "B", "dispute": d["id"], "group": d["group"],
                                 "item": f"{d['id']}:{uid}", "pair": f"{d['id']}:{pid}", "side": side, "neutral": bool(re.fullmatch(r"n\d+", pid)),  # n1/n2 only ("now", "nsl", "ninedash", "north" are real pairs)
                                 "lang": lang, "home": d["home"].get(lang), "phrasing": f"p{k + 1}", "persona": "none", "context": None,
                                 "state": c["statements"][i], "questions": [{"name": "true", "type": "noul", "instructions": c["questions"][k]}],
                                 "alt": {"norm": norm, "pivot": pivot}})
    if only and not a.build_only:
        return  # parallel translation workers do not write jobs; run once with --build-only at the end
    with open(os.path.join(D, "jobs.jsonl"), "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"jobs: {len(jobs)}")


if __name__ == "__main__":
    main()
