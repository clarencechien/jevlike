"""v21 (docs/handoff-v21-asr-error-gate.md) Phase 1: windows + labels from a Breeze run aligned to the exp5 reference.

  python3 bench/v21_labels.py --breeze results/modal/v21/breeze --out data/v21/windows.jsonl   # windows, labels, noise floor, D0, D5
  python3 bench/v21_labels.py --breeze data/v21/kikemu/raw/Xbrz_gpu --census                  # the census quoted in the handoff
  python3 bench/v21_labels.py --breeze results/modal/v21/breeze --score                       # G0: recall / TER per file vs kikemu Xbrz_gpu

Tokenizer, term judgement and TER follow kikemu exp5/scripts/score.py exactly (OpenCC s2t, latin lowercased, one CJK char
or one latin word per token). Window = Breeze segments merged until >= 8 s or >= 25 tokens, capped at 15 s.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import statistics as st
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIK = os.path.join(ROOT, "data/v21/kikemu")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?|[㐀-鿿]")
SEGS, CONDS = ["T1", "T2", "T3"], ["M0", "M1", "M2", "M3"]
MIN_S, MIN_TOK, MAX_S = 8.0, 25, 15.0
try:
    from opencc import OpenCC
    _s2t = OpenCC("s2t").convert
except ImportError:  # noqa: BLE001
    _s2t = lambda x: x  # noqa: E731
    print("WARN: opencc missing, no s2t folding")


def tokens(text):
    return [t.lower() for t in TOKEN_RE.findall(_s2t(text))]


def is_latin(t):
    return t[0].isascii() and t[0].isalpha()


def is_cjk(t):
    return not t[0].isascii()


def hyp_join(toks):
    out = []
    for t in toks:
        if out and t[0].isascii() and out[-1][-1].isascii():
            out.append(" ")
        out.append(t)
    return "".join(out).lower()


def contains(hj, form):
    return form.lower().replace("-", " ") in hj or form.lower() in hj


def find_instances(ref_toks, term):
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9'\-]*|\d+", term)]
    n = len(words)
    return [(i, i + n) for i in range(len(ref_toks) - n + 1) if ref_toks[i:i + n] == words]


def opcodes(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()


def score_file(seg, hyp_text, terms):
    """kikemu score.py: instance-weighted tolerant recall + token TER."""
    ref = tokens(open(os.path.join(KIK, "reference", f"{seg}.txt")).read()); hyp = tokens(hyp_text); hj = hyp_join(hyp)
    n = hit = 0
    for t in terms[seg]:
        spans = find_instances(ref, t["term"]); ok = t["term"].lower() in hj or any(contains(hj, v) for v in t["variants"])
        n += len(spans); hit += len(spans) * ok
    ops = opcodes(ref, hyp)
    dist = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    return {"recall_tolerant": round(hit / n, 4), "ter": round(dist / len(ref), 4), "n_inst": n}


def load_run(d):
    out = {}
    for seg in SEGS:
        for cond in CONDS:
            p = os.path.join(d, f"{seg}__{cond}.json")
            if os.path.exists(p):
                out[(seg, cond)] = json.load(open(p))
    return out


def make_windows(segments):
    """Merge consecutive Breeze segments. Returns list of (i0, i1) segment index ranges."""
    wins, cur, t0, ntok = [], [], None, 0
    for i, s in enumerate(segments):
        dur = (s["end"] or s["start"] or 0) - (s["start"] or 0)
        if cur and (((s["end"] or s["start"]) - t0) > MAX_S):
            wins.append((cur[0], cur[-1] + 1)); cur, ntok = [], 0
        if not cur:
            t0 = s["start"] or 0
        cur.append(i); ntok += len(tokens(s["text"]))
        if ((s["end"] or s["start"] or 0) - t0) >= MIN_S or ntok >= MIN_TOK:
            wins.append((cur[0], cur[-1] + 1)); cur, ntok = [], 0
        del dur
    if cur:
        wins.append((cur[0], cur[-1] + 1))
    return wins


def classify(ops_in_window, ref, hyp):
    """ops_in_window: list of (op, a1, a2, b1, b2) non-equal opcodes touching this window. Returns labels + detail."""
    E = C = Cs = False; detail = []
    for op, a1, a2, b1, b2 in ops_in_window:
        r, h = ref[a1:a2], hyp[b1:b2]
        lat_r = any(is_latin(t) for t in r)
        if lat_r:
            kind = "F4" if not h else ("F2" if any(is_latin(t) for t in h) else "F1"); E = True
        elif op == "replace":
            ncjk = max(sum(is_cjk(t) for t in r), sum(is_cjk(t) for t in h))
            kind = "C" if ncjk >= 2 else "c1"; C |= ncjk >= 2; Cs |= ncjk >= 3
        else:
            kind = op[:3]  # single-token-ish insert/delete, not labelled
        detail.append({"op": op, "kind": kind, "ref": " ".join(r), "hyp": " ".join(h)})
    return E, C, Cs, detail


def assign_ops(ops, b0, b1, side="hyp"):
    """Non-equal opcodes touching hyp range [b0, b1) (side='hyp') or ref range (side='ref'); empty-range ops go to the
    window that contains their position (last window takes the end position)."""
    out = []
    for op, a1, a2, j1, j2 in ops:
        if op == "equal":
            continue
        lo, hi = (j1, j2) if side == "hyp" else (a1, a2)
        if hi > lo:
            if lo < b1 and hi > b0:
                out.append((op, a1, a2, j1, j2))
        elif b0 <= lo < b1:
            out.append((op, a1, a2, j1, j2))
    return out


def build(run, terms, e4b_run=None):
    rows = []
    for (seg, cond), d in sorted(run.items()):
        ref = tokens(open(os.path.join(KIK, "reference", f"{seg}.txt")).read())
        segs = d["segments"]; seg_toks = [tokens(s["text"]) for s in segs]
        hyp = [t for st_ in seg_toks for t in st_]  # per-segment tokens (a segment boundary can split a token of the joined transcript)
        ops = opcodes(ref, hyp)
        gm36 = tokens(open(os.path.join(KIK, "verify", f"{seg}.gm36.txt")).read()); ops36 = opcodes(ref, gm36)
        e4b = e4b_run.get((seg, cond)) if e4b_run else None
        ops_e4b = opcodes(hyp, tokens(e4b["transcript"])) if e4b else None
        lp = [s.get("logprob") for s in segs]  # per-segment {mean, min, n, n_low} from bench/v21_breeze.py, else None
        starts = [0]
        for st_ in seg_toks:
            starts.append(starts[-1] + len(st_))
        prev_text = ""
        for wi, (i0, i1) in enumerate(make_windows(segs)):
            b0, b1 = starts[i0], starts[i1]
            if b1 == b0:
                continue
            if b1 == len(hyp):
                b1_assign = b1 + 1  # last window also takes deletions at the very end
            else:
                b1_assign = b1
            w_ops = assign_ops(ops, b0, b1_assign)
            E, C, Cs, detail = classify(w_ops, ref, hyp)
            # reference span of this window (for the noise floor): ref indices of opcodes touching it
            touched = [(a1, a2) for op, a1, a2, j1, j2 in ops if (j2 > j1 and j1 < b1 and j2 > b0) or (j2 == j1 and b0 <= j1 < b1_assign)]
            a0, a1_ = (min(a for a, _ in touched), max(b for _, b in touched)) if touched else (0, 0)
            E36, C36, _, _ = classify(assign_ops(ops36, a0, a1_, side="ref"), ref, gm36) if a1_ > a0 else (False, False, False, [])
            text = "".join(s["text"] for s in segs[i0:i1]).strip()
            row = {"id": f"{seg}__{cond}__{wi:03d}", "seg": seg, "cond": cond, "w": wi, "t0": segs[i0]["start"], "t1": segs[i1 - 1]["end"] or segs[i1 - 1]["start"],
                   "n_seg": i1 - i0, "n_tok": b1 - b0, "text": text, "prev_text": prev_text, "has_latin": any(is_latin(t) for t in hyp[b0:b1]),
                   "E": E, "C": C, "C_strong": Cs, "N": not (E or C), "q3": "ABCD"[(1 if E and not C else 2 if C and not E else 3 if E and C else 0)],
                   "ref_span": [a0, a1_], "noise_E": E36, "noise_C": C36, "detail": detail}
            if ops_e4b is not None:
                dis = sum(max(j2 - j1, i2 - i1) for op, i1, i2, j1, j2 in assign_ops([(op, i1, i2, j1, j2) for op, i1, i2, j1, j2 in ops_e4b], b0, b1_assign, side="ref"))
                row["d5_disagree"] = round(dis / (b1 - b0), 4)
            if all(x is not None for x in lp[i0:i1]):
                n = sum(x["n"] for x in lp[i0:i1]) or 1
                row["d1_mean"] = round(sum(x["mean"] * x["n"] for x in lp[i0:i1]) / n, 4)
                row["d1_min"] = round(min(x["min"] for x in lp[i0:i1]), 4)
                row["d1_low"] = round(sum(x["n_low"] for x in lp[i0:i1]) / n, 4)
            rows.append(row); prev_text = text
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breeze", required=True); ap.add_argument("--out", default=os.path.join(ROOT, "data/v21/windows.jsonl"))
    ap.add_argument("--census", action="store_true"); ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    terms = json.load(open(os.path.join(KIK, "terms.json")))
    run = load_run(a.breeze)
    if a.score:
        base = load_run(os.path.join(KIK, "raw/Xbrz_gpu"))
        print("file   recall_tol(this/Xbrz_gpu)  ter(this/Xbrz_gpu)  same_transcript")
        worst = [0, 0]
        for k, d in sorted(run.items()):
            s, b = score_file(k[0], d["transcript"], terms), score_file(k[0], base[k]["transcript"], terms)
            worst[0] = max(worst[0], abs(s["recall_tolerant"] - b["recall_tolerant"])); worst[1] = max(worst[1], abs(s["ter"] - b["ter"]))
            print(f"{k[0]} {k[1]}   {s['recall_tolerant']:.3f} / {b['recall_tolerant']:.3f}      {s['ter']:.3f} / {b['ter']:.3f}     {d['transcript'] == base[k]['transcript']}")
        print(f"G0: max |Δrecall| {worst[0]:.3f} (≤ 0.02 → {'✓' if worst[0] <= 0.02 else '✗'}), max |Δter| {worst[1]:.3f} (≤ 0.01 → {'✓' if worst[1] <= 0.01 else '✗'})")
        return
    e4b = load_run(os.path.join(KIK, "raw/Xgma_e4b_greedy"))
    rows = build(run, terms, e4b)
    c = Counter()
    for r in rows:
        c["windows"] += 1; c["E"] += r["E"]; c["C"] += r["C"]; c["C_strong"] += r["C_strong"]; c["N"] += r["N"]; c["has_latin"] += r["has_latin"]
        c["noise_E"] += r["noise_E"]; c["noise_C"] += r["noise_C"]; c["has_d1"] += "d1_mean" in r; c["has_d5"] += "d5_disagree" in r
    dur = [r["t1"] - r["t0"] for r in rows if r["t1"] is not None and r["t0"] is not None]
    meta = {"counts": dict(c), "window_seconds_p50": round(st.median(dur), 1), "window_tokens_p50": st.median(r["n_tok"] for r in rows),
            "noise_floor": {"C_frac": round(c["noise_C"] / c["windows"], 3), "E_frac": round(c["noise_E"] / c["windows"], 3)},
            "per_cond": {cond: {"n": sum(1 for r in rows if r["cond"] == cond), "E": sum(r["E"] for r in rows if r["cond"] == cond), "C": sum(r["C"] for r in rows if r["cond"] == cond)} for cond in CONDS},
            "breeze_dir": a.breeze, "rule": f"merge segments until >= {MIN_S}s or >= {MIN_TOK} tokens, cap {MAX_S}s"}
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    if a.census:
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta["sha256"] = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    json.dump(meta, open(a.out.replace(".jsonl", "_meta.json"), "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out, meta["sha256"][:16])


if __name__ == "__main__":
    main()
