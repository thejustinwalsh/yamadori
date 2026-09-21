#!/usr/bin/env python
"""Check text against controlled-English rules. No model, no network.

WHY A CHECKER AND NOT A REWRITER

Rewriting English into controlled English needs to know what the writer meant:
replacing "should verify" requires deciding whether the intent was "must",
"can" or "verifies". That is a judgement. DETECTING the violation is not --
it is a regex and a word count.

So this flags, and a person or a model fixes. That split is what makes it
free: every rule below runs in microseconds with no dependency.

WHY IT MATTERS HERE

Measured on this stack, phrasing moves results more than anything else tried:

    ordinary vs controlled-English instructions   3/6 -> 5/6 on a hedged set,
                                                  and mean decision margin
                                                  0.364 -> 0.427 on a clean one

The margin is the gate a decision has to clear before it is acted on, so
sharper phrasing means fewer decisions escalate. Since this stack writes its
own tool descriptions and instructions in source, the cheapest place to apply
that is at authoring time, on our own text.

Rules are from ASD-STE100, the aerospace controlled-language standard, whose
purpose is that a tired reader cannot misread a safety-critical instruction.
A non-autoregressive classifier has the same problem for the same reason: it
reads surface form, so ambiguous surface form is a real defect.
"""
from __future__ import annotations

import re

# Each rule: (id, pattern, why it hurts, what to do instead). The advice is
# phrased as an action, not as a prohibition, for the same reason the rules
# themselves exist.
RULES = [
    ("modal", re.compile(r"\b(should|would|may|might|could|ought to)\b", re.I),
     "a modal leaves the requirement ambiguous",
     "state the requirement directly: must, or a plain present-tense verb"),
    ("contraction", re.compile(r"\b\w+'(s|t|re|ve|ll|d|m)\b", re.I),
     "contractions add a token boundary with no meaning",
     "write the words out"),
    ("semicolon", re.compile(r";"),
     "a semicolon joins two ideas that a reader must hold at once",
     "use two sentences"),
    ("emdash", re.compile(r"—|--"),
     "a dash hides the relationship between the clauses it joins",
     "use a full stop, or name the relationship"),
    ("present_perfect", re.compile(r"\b(has|have|had)\s+\w+(ed|en)\b", re.I),
     "present perfect leaves the time of the action vague",
     "use a simple tense"),
    ("gerund_after_comma", re.compile(r",\s+\w+ing\b"),
     "a gerund after a comma attaches to an unclear subject",
     "start a new sentence with the subject named"),
    ("passive", re.compile(r"\b(is|are|was|were|be|been|being)\s+\w+(ed|en)\s+by\b", re.I),
     "the passive hides who acts",
     "name the actor and use an active verb"),
    ("vague_quantifier", re.compile(r"\b(some|several|various|a number of|appropriate|relevant)\b", re.I),
     "an unquantified word cannot be checked",
     "give the number, or the condition that selects them"),
]

# ASD-STE100 caps procedural sentences at 20 words and descriptive at 25.
MAX_WORDS = 20


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def check(text: str, max_words: int = MAX_WORDS) -> list[dict]:
    """Every violation, with the span that triggered it."""
    out: list[dict] = []
    for i, sentence in enumerate(_sentences(text)):
        words = len(sentence.split())
        if words > max_words:
            out.append({"rule": "length", "sentence": i, "found": f"{words} words",
                        "why": "a long sentence holds several claims at once",
                        "fix": f"split it; the limit is {max_words} words"})
        for rid, pat, why, fix in RULES:
            for m in pat.finditer(sentence):
                out.append({"rule": rid, "sentence": i, "found": m.group(0),
                            "why": why, "fix": fix})
    return out


def score(text: str) -> float:
    """Violations per sentence. 0.0 is clean; there is no upper bound."""
    n = max(len(_sentences(text)), 1)
    return round(len(check(text)) / n, 2)


def report(text: str, limit: int = 12) -> str:
    v = check(text)
    if not v:
        return f"clean ({len(_sentences(text))} sentences)"
    lines = [f"{len(v)} violations across {len(_sentences(text))} sentences "
             f"({score(text)} per sentence)"]
    seen: set[str] = set()
    for item in v[:limit]:
        key = f"{item['rule']}:{item['found']}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {item['rule']:<20} {item['found']!r:<24} -> {item['fix']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(report(open(sys.argv[1], encoding="utf-8").read()))
    else:
        print(report(sys.stdin.read()))
