"""Tool routing as System One: binary perception + a deterministic rule.

NOT WIRED INTO THE REQUEST PATH. This is a validated pattern, kept because the
approach generalises well beyond routing -- see "When to use this" below.

WHY THE OBVIOUS VERSION FAILS
-----------------------------
The natural thing is to ask "which of these 7 tools should handle this
request?" as one choice question. Measured on 14 labelled requests:

    7-way choice, abstract tool descriptions     50%
    7-way choice, trigger-phrased descriptions   57%

Confidence floated around 0.07-0.35 on the failures. That is not the model
being weak; it is the model being asked a System Two question. Laya does not
deliberate -- it perceives, instantly, in one forward pass. "Which tool, all
things considered" requires weighing alternatives. "Does this text name a
symbol" does not.

WHAT WORKS
----------
Ask only locally-decidable binary questions, then put the judgement in code:

    binary features + naive rule                 43%
    binary features + CALIBRATED rule            86%

The jump from 43 to 86 was entirely in the rule, not the model. Three fixes,
each a System Two habit worth unlearning:

1. ONE THRESHOLD FOR ALL FEATURES IS WRONG. Measured ranges differ sharply:
       asks_usage     fires 0.65-0.85 when true
       names_symbol   fires 0.27-0.48 when true
   A single 0.5 cutoff silently discarded the strongest signal in the set.

2. DON'T ASK THE MODEL WHAT CODE DOES BETTER. gives_path scored 0.69 on
   "parse_tool_call" -- which is fair, it does look like a filename.
   Corroborating with a one-line regex fixed it. Perception from the model,
   adjudication in code.

3. RULE ORDER IS LOAD-BEARING. Checking asks_usage before names_symbol moved
   three cases at once, because the strong feature was being gated behind the
   weak one.

WHEN TO USE THIS
----------------
Good: extracting features inside a tool ("does this request name a symbol?",
"is this snippet relevant?", "does this log show a crash?"). Cheap, ~45ms for
six features, and the answer feeds code that you can audit and test.

Bad: as the actual dispatcher in front of tool selection. A wrong route means
the correct tool is never offered to the model -- a silent, unrecoverable
failure -- and the 27B already selects correctly unaided. Reserve routing for
when the tool count grows past what a model handles well (tens, not seven).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")

# Each must be answerable by looking at the text, with no weighing of
# alternatives. If a question needs "on balance", it does not belong here.
FEATURES = {
    "names_symbol": "The text names a specific function, struct, type or variable, "
                    "such as parse_hunks or Buffer.",
    "asks_usage": "The text asks what calls, uses, depends on or references something, "
                  "or whether it is unused.",
    "gives_path": "The text contains a file path or file name, such as src/render.rs "
                  "or tool_parsing.py.",
    "wants_verdict": "The text asks whether something is true, or to classify it, "
                     "expecting a judgement rather than code.",
    "wants_summary": "The text asks to summarise, shorten or compress a long log, "
                     "transcript or conversation.",
    "about_index": "The text asks about the search index itself, such as how many "
                   "files or chunks are indexed.",
}

# Calibrated against measured firing ranges, NOT chosen a priori.
THRESHOLDS = {
    "about_index": 0.45,
    "wants_summary": 0.45,
    "asks_usage": 0.50,
    "gives_path": 0.45,
    "names_symbol": 0.25,   # real symbols only reach 0.27-0.48
    # wants_verdict is DELIBERATELY absent -- see known weakness below.
}

_PATH_RE = re.compile(
    r"[\w/\\-]+\.(rs|py|ts|tsx|js|jsx|c|h|cpp|hpp|zig|wgsl|glsl|toml|json|yaml|md)\b"
)
_DIR_RE = re.compile(r"[\w-]+/[\w-]+")


def extract(text: str, url: str = None) -> dict:
    """Return {feature: probability}. One forward pass, ~45ms for six features."""
    qs = {k: {"type": "noul", "instructions": v} for k, v in FEATURES.items()}
    body = json.dumps({"state": text, "questions": qs}).encode()
    req = urllib.request.Request((url or LAYA_URL) + "/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return {k: v["noul"] for k, v in d["answers"].items()}


def looks_like_path(text: str) -> bool:
    """Literal corroboration for gives_path. Cheap, exact, and not a model's job."""
    return bool(_PATH_RE.search(text) or _DIR_RE.search(text))


def route(text: str, features: dict | None = None) -> str:
    """Map features to a tool name. Order is load-bearing: strongest signal first."""
    f = features if features is not None else extract(text)

    if f["about_index"] > THRESHOLDS["about_index"]:
        return "index_status"
    if f["wants_summary"] > THRESHOLDS["wants_summary"]:
        return "compact"
    # Strongest feature in the set. Must NOT be gated behind names_symbol,
    # which fires far weaker -- that single mistake cost three cases.
    if f["asks_usage"] > THRESHOLDS["asks_usage"]:
        return "find_references"
    if f["gives_path"] > THRESHOLDS["gives_path"] and looks_like_path(text):
        return "read_file"
    if f["names_symbol"] > THRESHOLDS["names_symbol"]:
        return "find_definition"
    return "search_code"


# KNOWN WEAKNESS
# --------------
# `wants_verdict` does not separate: it scored 0.15 on both genuine judge
# cases AND on unrelated requests. There is no threshold that works, so the
# rule above does not consult it and "judge" is never routed to -- the two
# judge cases fall through to search_code. That is the honest 12/14.
#
# Including it at a noise-level cutoff (0.12) pushed accuracy UP to 86% on
# this set while routing "how does this handle retry backoff?" to judge, i.e.
# it was right by luck. The question needs rephrasing before the feature is
# trustworthy; a lucky threshold is not a fix.
