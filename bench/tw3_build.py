"""tw3 (docs/handoff-tw3-place-names.md): build data/tw3/jobs.jsonl from the tw2 translations and data/tw3/names.json. No model calls.

T1 (plan E1, language x name): for dokdo / scs (scarborough, spratlys pairs) / falklands, every language x name arm (X / Y / N) that tw2
   did not already run. Non-English text = the tw2 translation with only the place name swapped (asserted: the name occurs exactly once,
   and swapping it back gives the tw2 sentence). English dokdo / falklands are filled from the master templates (verb agreement for the
   singular names Dokdo / Takeshima; sentence-initial capital, which also fixes tw2's lower-case "the Falkland Islands are ..." so
   falklands en X is re-run, see the report).
T2 (plan E2, deployment pipeline): the tw2 English pivot text of every job of the three disputes (arm `pivot`, as is) and the same text
   after the English name normaliser in names.json (arm `pivot_names`). Neutral pairs included (readability check).
Out: data/tw3/jobs.jsonl, data/tw3/SPOTCHECK.md (every distinct new sentence, for the blind review)
"""
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2, D3 = os.path.join(REPO, "data", "tw2"), os.path.join(REPO, "data", "tw3")
NAMES = json.load(open(os.path.join(D3, "names.json"), encoding="utf-8"))
MASTER = {d["id"]: d for d in json.load(open(os.path.join(D2, "master_en.json"), encoding="utf-8"))["disputes"]}
DISPUTES = ["dokdo", "scs", "falklands"]
NAME_PAIRS = {"dokdo": {"status": "ISLANDS", "control": "ISLANDS", "1905": "ISLANDS", "inherent": "ISLANDS"},
              "scs": {"scarborough": "SCARBOROUGH", "spratlys": "SPRATLYS"},
              "falklands": {"status": "ISLANDS", "selfdet": "ISLANDS", "1833": "ISLANDS", "1982": "ISLANDS"}}
SINGULAR_EN = {"Dokdo", "Takeshima"}


def cap(s):
    return s[:1].upper() + s[1:]


def orig_arm(dispute, lang, key):
    """which name arm the tw2 translation already is (None = a form that is none of X / Y / N, e.g. the English dual name)"""
    t = NAMES[dispute][key][lang]
    for arm in ("X", "Y", "N"):
        if t[arm] == t["orig"] and not (dispute == "falklands" and lang == "en"):  # falklands en: tw2 text has the capitalisation bug -> re-run X
            return arm
    return None


def en_from_master(dispute, pid, side, name):
    p = next(x for x in MASTER[dispute]["pairs"] if x["pid"] == pid)
    s = p[side].replace("{ISLANDS}", name)
    if name in SINGULAR_EN:
        s = re.sub(rf"^{name} are\b", f"{name} is", s)
        s = re.sub(rf"^{name} have been\b", f"{name} has been", s)
    return cap(s)


def swap(text, old, new):
    assert text.count(old) == 1, (text, old)
    out = text.replace(old, new)
    assert out.replace(new, old) == text if new not in old and old not in new else True
    return cap(out) if text[:1].isupper() else out


def normalise_en(text):
    n = 0
    for pat, rep in NAMES["pivot_en_normalizer"]:
        text, k = re.subn(pat, rep, text, flags=re.I)
        n += k
    text = re.sub(r"\bthe the\b", "the", text, flags=re.I)
    text = re.sub(r"Liancourt Rocks is\b", "Liancourt Rocks are", text)
    text = re.sub(r"Liancourt Rocks has\b", "Liancourt Rocks have", text)
    return cap(text), n


def main():
    tr = {}
    for d in DISPUTES:
        tr.update(json.load(open(os.path.join(D2, "translations", f"{d}.json"), encoding="utf-8")))
    jobs, spot = [], []

    def add(**kw):
        jobs.append({"persona": "none", "context": None, **kw})

    # ---- T1
    for d in DISPUTES:
        m = MASTER[d]
        for lang in m["langs"]:
            c = tr[f"{d}|{lang}"]
            for i, uid in enumerate(c["ids"]):
                pid, side = uid.rsplit("-", 1)
                key = NAME_PAIRS[d].get(pid)
                if not key:
                    continue
                forms = NAMES[d][key][lang]
                done = orig_arm(d, lang, key)
                for arm in ("X", "Y", "N"):
                    if arm == done:
                        continue
                    if lang == "en" and d in ("dokdo", "falklands"):
                        text = en_from_master(d, pid, side, forms[arm])
                    else:
                        text = swap(c["statements"][i], forms["orig"], forms[arm])
                    spot.append((d, lang, arm, uid, c["statements"][i], text))
                    for k in range(3):
                        add(job=f"T1|{d}|{uid}|{lang}|{arm}|p{k + 1}", exp="T1", dispute=d, item=f"{d}:{uid}", pair=f"{d}:{pid}", side=side,
                            neutral=False, lang=lang, home=m["home"].get(lang), name=arm, phrasing=f"p{k + 1}", state=text,
                            questions=[{"name": "true", "type": "noul", "instructions": c["questions"][k]}])
    n1 = len(jobs)
    # ---- T2
    bad = []
    for l in open(os.path.join(D2, "jobs.jsonl"), encoding="utf-8"):
        j = json.loads(l)
        if j["dispute"] not in DISPUTES:
            continue
        pid = j["pair"].split(":")[1]
        pv = j["alt"]["pivot"]
        norm, n = normalise_en(pv["state"])
        has_name = pid in NAME_PAIRS[j["dispute"]]
        if (n == 0) == has_name:
            bad.append((j["job"], pv["state"], n))
        if j["phrasing"] == "p1" and has_name:
            spot.append((j["dispute"], f"pivot<-{j['lang']}", "pivot_names", j["item"].split(":")[1], pv["state"], norm))
        for arm, text in (("pivot", pv["state"]), ("pivot_names", norm)):
            add(job=f"T2|{j['dispute']}|{j['item'].split(':')[1]}|{j['lang']}|{arm}|{j['phrasing']}", exp="T2", dispute=j["dispute"],
                item=j["item"], pair=j["pair"], side=j["side"], neutral=j["neutral"], lang="en", src=j["lang"], home=j["home"], arm=arm,
                phrasing=j["phrasing"], state=text, questions=[{"name": "true", "type": "noul", "instructions": pv["instructions"]}])
    assert not bad, bad
    with open(os.path.join(D3, "jobs.jsonl"), "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    md = ["# tw3 spot-check: every new sentence (T1 = name swapped in the tw2 translation; T2 = English pivot after the name normaliser)", "",
          "| 爭議 | 語言 | 地名臂 | 句 | tw2 原句 | 新句 |", "|---|---|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {u} | {o} | {n} |" for a, b, c, u, o, n in spot]
    open(os.path.join(D3, "SPOTCHECK.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"T1 {n1} jobs, T2 {len(jobs) - n1} jobs, total {len(jobs)}; spot-check rows {len(spot)}")


if __name__ == "__main__":
    main()
