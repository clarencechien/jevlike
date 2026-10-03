"""v10 (docs/handoff-v10-clef.md): map our rows and JevBench items to Clef records, and Clef outputs back to letters.

Clef record (Cloudflare joint_schema_model.py): {"state": str|json, "questions": {qid: {"type", "instructions", "criteria"}}}
  - choice: criteria = {option_id: description}; Clef sorts option ids, so input order does not matter.
  - score:  criteria = [description, ...] in order (index 0..K-1).
  - noul:   criteria = {"true": ..., "false": ...}.

Our rows: question = {"type": choice|score|noul, "instructions", "options": {"A": {"label", "criteria"}, ...}}.
Option text is not edited; only the format changes:
  - choice, id mode "label" (default): option_id = label, description = criteria.
  - choice, id mode "letter_fwd" / "letter_rev": option_id = a letter, description = "label：criteria"; rev assigns the
    letters in reverse, so Clef's sorted order is reversed (order-sensitivity arm; Clef has no other way to reorder).
  - score: list of "label：criteria" in A..K order.
  - noul: our options are always A=是 / B=否, so true = A's criteria, false = B's criteria.
"""
from __future__ import annotations

QID = "q"


def _letters(n):
    return [chr(ord("A") + i) for i in range(n)]


def row_question(q, id_mode="label"):
    """Our question dict -> (clef question, {clef option id: our letter})."""
    opts = q["options"]
    letters = list(opts)
    t = q["type"]
    if t == "noul":
        assert [opts[k]["label"] for k in letters] == ["是", "否"], letters
        cq = {"type": "noul", "instructions": q["instructions"],
              "criteria": {"true": opts[letters[0]]["criteria"], "false": opts[letters[1]]["criteria"]}}
        return cq, {"true": letters[0], "false": letters[1]}
    if t == "score":
        cq = {"type": "score", "instructions": q["instructions"],
              "criteria": [f"{opts[k]['label']}：{opts[k]['criteria']}" for k in letters]}
        return cq, {str(i): k for i, k in enumerate(letters)}
    assert t == "choice", t
    if id_mode == "label":
        labels = [opts[k]["label"] for k in letters]
        assert len(set(labels)) == len(labels), labels
        cq = {"type": "choice", "instructions": q["instructions"], "criteria": {opts[k]["label"]: opts[k]["criteria"] for k in letters}}
        return cq, {opts[k]["label"]: k for k in letters}
    ids = _letters(len(letters))
    if id_mode == "letter_rev":
        ids = ids[::-1]
    else:
        assert id_mode == "letter_fwd", id_mode
    cq = {"type": "choice", "instructions": q["instructions"],
          "criteria": {i: f"{opts[k]['label']}：{opts[k]['criteria']}" for i, k in zip(ids, letters)}}
    return cq, dict(zip(ids, letters))


def row_record(row, id_mode="label"):
    cq, back = row_question(row["question"], id_mode)
    return {"id": row["id"], "state": row["state"], "questions": {QID: cq}}, {QID: back}


def packed_record(state, questions, rid):
    """Several of our questions on one state (Clef scores them jointly in one pass). questions: {qid: our question}."""
    qs, backs = {}, {}
    for qid, q in questions.items():
        qs[qid], backs[qid] = row_question(q)
    return {"id": rid, "state": state, "questions": qs}, backs


def jev_record(task, id_mode="native"):
    """JevBench item -> Clef record. Labels map back to themselves (or from letters in the order arms)."""
    q = task["question"]; labels = task["labels"]; t = q["type"]
    if t == "noul":
        crit = {k: v for k, v in (q.get("criteria") or {}).items() if k in ("true", "false")}
        cq = {"type": "noul", "instructions": q["instructions"], **({"criteria": crit} if crit else {})}
        back = {"true": "yes", "false": "no"}
    elif t == "score":
        cq = {"type": "score", "instructions": q["instructions"], "criteria": list(q["criteria"])}
        back = {str(i): str(i) for i in range(len(q["criteria"]))}
    elif id_mode == "native":
        cq = {"type": "choice", "instructions": q["instructions"], "criteria": {lab: (q.get("criteria") or {}).get(lab, lab) for lab in labels}}
        back = {lab: lab for lab in labels}
    else:
        ids = _letters(len(labels))
        if id_mode == "letter_rev":
            ids = ids[::-1]
        crit = q.get("criteria") or {}
        cq = {"type": "choice", "instructions": q["instructions"],
              "criteria": {i: f"{lab}: {crit[lab]}" if crit.get(lab) else lab for i, lab in zip(ids, labels)}}
        back = dict(zip(ids, labels))
    state = task["state"]
    return {"id": task["id"], "state": state, "questions": {QID: cq}}, {QID: back}


def to_ours(option_ids, probs, back):
    """Clef option ids + softmax probs -> {our letter or label: prob}."""
    return {back[o]: float(p) for o, p in zip(option_ids, probs)}
