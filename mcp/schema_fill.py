"""Typesafe structured output from a model that cannot generate text.

Laya emits no tokens, so it cannot "write JSON". But a JSON Schema is already
a set of typed questions, and Laya answers typed questions. Walk the schema,
turn each field into the matching primitive, answer them all in ONE forward
pass, and assemble the object:

    {"type":"string","enum":[...]}   -> choice
    {"type":"boolean"}               -> noul
    {"type":"integer", min, max}     -> score, mapped back onto the range
    {"type":"object","properties":…} -> recurse

The result conforms to the schema by construction. There is no grammar to
fight, no validation-and-retry loop, and no way to invent a field or an enum
value that does not exist -- the failure modes of grammar-constrained decoding
simply cannot occur, because nothing is being decoded.

WHAT THIS CANNOT DO
-------------------
Free-text strings. A `{"type":"string"}` with no enum has no fixed option set,
so there is nothing to choose between. Those fields are skipped and reported.
This is for classification-shaped structure -- triage records, routing
decisions, config selection, form filling over known vocabularies, UI
component choice -- not for prose.

Every field also comes back with a probability, so a caller can reject or
escalate low-confidence fields instead of silently accepting a guess.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


def schema_to_questions(schema: Dict, prefix: str = "") -> Tuple[Dict[str, Dict], list]:
    """Return (questions, skipped). Keys are dotted paths into the object."""
    questions: Dict[str, Dict] = {}
    skipped: list = []

    props = schema.get("properties") or {}
    for name, spec in props.items():
        path = f"{prefix}{name}"
        desc = spec.get("description", "") or name.replace("_", " ")
        t = spec.get("type")

        if "enum" in spec:
            questions[path] = {
                "type": "choice",
                "instructions": desc,
                # an empty criterion means "the label speaks for itself"
                "criteria": {str(v): "" for v in spec["enum"]},
            }
        elif t == "boolean":
            questions[path] = {"type": "noul", "instructions": desc}
        elif t in ("integer", "number"):
            lo = spec.get("minimum", 0)
            hi = spec.get("maximum", 10)
            n = min(int(hi - lo) + 1, 11)          # score heads are ordinal, keep it small
            questions[path] = {
                "type": "score",
                "instructions": desc,
                "criteria": [str(lo + round(i * (hi - lo) / max(n - 1, 1))) for i in range(n)],
            }
            questions[path]["_range"] = (lo, hi)
        elif t == "object":
            sub_q, sub_skip = schema_to_questions(spec, prefix=f"{path}.")
            questions.update(sub_q)
            skipped.extend(sub_skip)
        else:
            # free-form strings, arrays: nothing to choose between
            skipped.append({"field": path, "reason": f"type '{t}' has no fixed option set"})

    return questions, skipped


def assemble(answers: Dict[str, Dict], questions: Dict[str, Dict]) -> Tuple[Dict, Dict]:
    """Fold flat dotted answers back into a nested object, plus a confidence map."""
    out: Dict[str, Any] = {}
    conf: Dict[str, float] = {}

    for path, a in answers.items():
        q = questions.get(path, {})
        if "choice" in a:
            value: Any = a["choice"]
            # recover ints/bools that were expressed as enum labels
            if isinstance(value, str):
                if value.isdigit():
                    value = int(value)
                elif value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        elif "noul" in a:
            value = bool(a["noul"] >= 0.5)
        elif "score" in a:
            lo, hi = q.get("_range", (0, 10))
            levels = max(len(q.get("criteria", [])) - 1, 1)
            value = round(lo + (a["score"] / levels) * (hi - lo))
        else:
            continue

        conf[path] = round(float(a.get("confidence", 0.0)), 4)

        node = out
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    return out, conf


def fill(agent_predict, state: Any, schema: Dict) -> Dict:
    """agent_predict(state, questions) -> laya result. Returns a full report."""
    questions, skipped = schema_to_questions(schema)
    if not questions:
        return {"object": {}, "confidence": {}, "skipped": skipped,
                "error": "schema contained no decidable fields"}

    # strip our private annotation before it reaches the model
    clean = {k: {kk: vv for kk, vv in q.items() if not kk.startswith("_")}
             for k, q in questions.items()}
    res = agent_predict(state, clean)
    obj, conf = assemble(res.get("answers", {}), questions)
    return {"object": obj, "confidence": conf, "skipped": skipped,
            "fields": len(questions)}
