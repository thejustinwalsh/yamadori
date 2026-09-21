#!/usr/bin/env python
"""Several propositions about one thing, judged at once, contradictions kept.

WHY THIS SHAPE

The decision model answers many typed questions in a SINGLE forward pass --
measured here at three questions in 69ms, about 23ms each. Every previous use
in this repo collapsed that to one number: "score this snippet", "is this
true". Those all failed, because a scalar has to be comparable across
different inputs and its scores are not.

Asking several questions about ONE fixed input is the opposite case, and it is
the one the model is measurably good at: the vendor's own numbers are 0.950 on
closed-label classification against 0.362 on open judgement, and every
independent evaluation found `choice` over a small option set to be the only
primitive that beat baseline.

WHY THE CONTRADICTIONS ARE THE POINT

A candidate that is correct but violates the local convention, or idiomatic but
slower, is the interesting case -- and averaging those into one score is
precisely what destroys the information. Two readings can both be true. This
returns the whole matrix and marks where the readings pull against each other,
because that tension is where an engineer's attention belongs.

WHAT IT IS NOT

Not a gate. The base rates make a single-output gate useless: at a 10% error
rate, a judge with 90% recall and 90% specificity yields nine true flags and
nine false ones -- precision 50%. This is used to SURFACE tension for a caller
who will look, never to accept or reject anything on its own.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")

# Each axis is a closed choice, never a 0-1 score, and the options are
# concrete rather than degrees of goodness. "yes/no/unclear" beats "rate 1-5"
# because the model is a classifier, not a rater.
CODE_AXES = {
    "correct": {
        "instructions": "Does this code do what the request asked?",
        "criteria": {"yes": "it does what was asked",
                     "no": "it does something else or is broken",
                     "unclear": "cannot tell from what is shown"},
    },
    "conventional": {
        "instructions": "Does this follow the conventions of the surrounding code?",
        "criteria": {"yes": "matches the existing style and structure",
                     "no": "introduces a different pattern",
                     "unclear": "no surrounding code shown"},
    },
    "complete": {
        "instructions": "Is anything required left undone?",
        "criteria": {"nothing_missing": "the change is self-contained",
                     "missing_wiring": "it needs a registration, export or binding elsewhere",
                     "missing_tests": "behaviour changed with no check"},
    },
    "risk": {
        "instructions": "What is the main risk in this code?",
        "criteria": {"none_apparent": "no obvious hazard",
                     "correctness": "it may produce wrong results",
                     "performance": "it may be slow or allocate in a hot path",
                     "breaking": "it may break existing callers"},
    },
}


def ask(state: str, axes: dict | None = None, timeout: int = 20) -> dict:
    """One state, every axis, one forward pass."""
    import code_search as cs
    axes = axes or CODE_AXES
    questions = {k: {"type": "choice", "instructions": v["instructions"],
                     "criteria": v["criteria"]}
                 for k, v in axes.items()}
    try:
        d = cs._post_json(LAYA_URL + "/decide",
                          {"state": state[:4000], "questions": questions},
                          timeout=timeout)
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out = {}
    for k, a in (d.get("answers") or {}).items():
        out[k] = {"choice": a.get("choice"),
                  "confidence": round(float(a.get("confidence") or 0), 3),
                  "probabilities": {p: round(float(v), 3)
                                    for p, v in (a.get("probabilities") or {}).items()}}
    return {"axes": out, "elapsed_ms": d.get("elapsed_ms")}


# A reading is only worth acting on if the model actually separated the
# options. Independent evaluation found it "often confidently wrong", and
# separately that confidence saturates to 1.00 above eleven options, so the
# margin between the top two is a better signal than the reported confidence.
def margin(axis: dict) -> float:
    ps = sorted((axis.get("probabilities") or {}).values(), reverse=True)
    return round(ps[0] - ps[1], 3) if len(ps) > 1 else 0.0


def tensions(reading: dict, floor: float = 0.15) -> list[str]:
    """Where the axes pull against each other, in plain words.

    Only axes that actually separated are reported. An axis whose top two
    options are within `floor` is the model failing to decide, and reporting
    that as a finding would be inventing signal.
    """
    ax = reading.get("axes") or {}
    said = {k: v["choice"] for k, v in ax.items() if margin(v) >= floor}
    out = []
    if said.get("correct") == "yes" and said.get("conventional") == "no":
        out.append("does the job but not the way this codebase does it")
    if said.get("correct") == "yes" and said.get("complete") == "missing_wiring":
        out.append("looks right in isolation but something elsewhere must be "
                   "registered or exported for it to take effect")
    if said.get("correct") == "yes" and said.get("risk") == "breaking":
        out.append("correct for the request, but may break existing callers")
    if said.get("correct") == "yes" and said.get("complete") == "missing_tests":
        out.append("behaviour changed with nothing checking it")
    if said.get("conventional") == "yes" and said.get("correct") == "no":
        out.append("fits the codebase perfectly and still does the wrong thing")
    return out


def undecided(reading: dict, floor: float = 0.15) -> list[str]:
    """Axes the model could not separate. Reported, not hidden."""
    return [k for k, v in (reading.get("axes") or {}).items()
            if margin(v) < floor]


def summarise(reading: dict) -> str:
    if reading.get("error"):
        return f"decision engine unavailable ({reading['error']})"
    ax = reading.get("axes") or {}
    parts = [f"{k}={v['choice']}" for k, v in ax.items() if margin(v) >= 0.15]
    line = "  ".join(parts) if parts else "no axis separated"
    t = tensions(reading)
    if t:
        line += "\n  tension: " + "; ".join(t)
    u = undecided(reading)
    if u:
        line += f"\n  undecided: {', '.join(u)}"
    return line


if __name__ == "__main__":
    sample = """function addDashOffset(material, value) {
  material.dashOffset = value;
}"""
    r = ask(sample)
    print(summarise(r))
    print(f"  ({r.get('elapsed_ms')} ms for {len(r.get('axes') or {})} axes)")
