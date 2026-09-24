"""Q3a: a deterministic code-generation vs completion/edit detector, no model.

Operator's split: GENERATION = the prompt has no code, or code only as
reference, and the answer is a whole program; COMPLETION/EDIT = the prompt
contains partial code the answer must continue or fix.

Signals (all on the LAST fenced block at line start, and on the prompt text
OUTSIDE fenced blocks):
  P_raw    the last block fails to parse on its own (mcp/code_check.py
           syntax_errors: Python's compiler for Python, tree-sitter otherwise)
  P_stub   it still fails after a stub body is appended to its last line
           (Python: an indented `pass`; brace languages: unchanged). A bare
           signature `def f(self, x):` is a generation scaffold, not partial
           code; an open docstring or an open expression is not rescued.
  I_op     the operator's words anywhere outside code:
           continue/complete/completion/finish/fix (any inflection)
  I_narrow phrases that address the code as unfinished or broken:
           "remaining lines/part/portion", "missing portion/part/code",
           "partial completion/code/solution", "first lines of",
           "continue/complete/finish (the|this|writing) <code noun>",
           "fix (the|this|these) <code noun>"
  I_port   (added AFTER reading the domain prompts -- not validated):
           "no longer builds", "port it/this", "migrate"

Rules scored:
  R_op     = P_raw  or I_op       the rule as the operator stated it
  R_ref    = P_stub or I_narrow   refined
  R_text   = I_narrow             instruction only
  R_ref+   = R_ref or I_port      (exploratory; tuned on what it is scored on)

Ground truth:
  LiveBench      the task label: coding_completion = positive (all 128 of
                 release 2024-11-25, the 21 that ran marked)
  domain core    by construction every core/react task asks for a whole
  + react        module from a spec ("Write a ..."): negative. three_tsl
                 tasks that paste a complete r16x module and ask for a port
                 are EDITS (fix code that no longer builds): positive; the
                 five without a code block: negative. Labelled by this
                 analysis from reading the prompts (one reader, PROTOCOL 7).
  type-challenge a `= any` template to replace: reported as a flag rate only,
                 no label claimed (it is a stub, closer to LCB's starter).
Writes data/split_detector.json.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import code_check  # noqa: E402  (read-only use of its parser)

BLOCK = re.compile(r"^```([^\n`]*)\n(.*?)^```", re.M | re.S)
NOUN = r"(?:code|program|function|method|class|implementation|solution|module|snippet|bug|bugs|error|errors)"
I_OP = re.compile(r"\b(continu\w*|complet\w*|finish\w*|fix\w*)\b", re.I)
I_NARROW = re.compile(
    r"(remaining (?:lines|part|portion|code)|missing (?:portion|part|code|lines)"
    r"|partial (?:completion|code|solution|program)|first lines of"
    rf"|\b(?:continue|complete|finish) (?:the |this |writing )?(?:following |given |partial )?{NOUN}"
    rf"|\bfix (?:the |this |these )?(?:following |given )?{NOUN})", re.I)
I_PORT = re.compile(r"(no longer builds|\bport (?:it|this)\b|\bmigrate\b)", re.I)


def last_block(text: str):
    bs = BLOCK.findall(text)
    return bs[-1] if bs else None


def outside(text: str) -> str:
    return BLOCK.sub(" ", text)


def parses(code: str, lang: str | None) -> bool | None:
    lang = code_check.normalize_language(lang or "") if lang else None
    if lang is None:
        return None
    try:
        return not code_check.syntax_errors(code, lang)
    except Exception:                                            # noqa: BLE001
        return None


def stub_parses(code: str, lang: str | None) -> bool | None:
    norm = code_check.normalize_language(lang or "") if lang else None
    if norm != "python":
        return parses(code, lang)
    if parses(code, "python"):
        return True
    body = code.rstrip()
    last = body.split("\n")[-1] if body else ""
    ind = len(last) - len(last.lstrip())
    return parses(body + "\n" + " " * (ind + 4) + "pass\n", "python")


def features(text: str, default_lang: str | None = None) -> dict:
    b = last_block(text)
    f = {"has_block": b is not None, "lang": None, "p_raw": False,
         "p_stub": False}
    if b:
        tag, code = b
        lang = (tag.strip().split() or [default_lang or ""])[0] or default_lang
        f["lang"] = lang
        pr, ps = parses(code, lang), stub_parses(code, lang)
        f["p_raw"] = pr is False
        f["p_stub"] = ps is False
        f["unparsable_lang"] = pr is None
    out = outside(text)
    f["i_op"] = bool(I_OP.search(out))
    f["i_narrow"] = bool(I_NARROW.search(out))
    f["i_port"] = bool(I_PORT.search(out))
    f["R_op"] = f["p_raw"] or f["i_op"]
    f["R_ref"] = f["p_stub"] or f["i_narrow"]
    f["R_text"] = f["i_narrow"]
    f["R_ref+"] = f["R_ref"] or f["i_port"]
    return f


def main():
    items = []
    qs = json.load(open(os.path.join(HERE, "data", "lb_questions_slim.json")))
    ran = {json.loads(x)["id"] for x in open(os.path.join(
        ROOT, "bench", "livebench", "results", "lb-20260923-minp0",
        "rows_yamadori-xhigh_coding.jsonl")) if x.strip()}
    for q in qs:
        f = features(q["turns"][0], "python")
        items.append(dict(f, set="lb", id=q["question_id"][:8],
                          title=q.get("question_title"), ran=q["question_id"] in ran,
                          group=q["task"], truth=q["task"] == "coding_completion"))
    for fn in ("tasks.jsonl", "tasks_react.jsonl", "tasks_type_challenges.jsonl"):
        for line in open(os.path.join(ROOT, "bench", "domain", fn), encoding="utf-8"):
            if not line.strip():
                continue
            t = json.loads(line)
            f = features(t["prompt"])
            if t["domain"] == "typescript_tc":
                truth = None
            elif t["domain"] == "three_tsl":
                truth = f["has_block"] and bool(re.search(r"\bPort\b", t["prompt"]))
            else:
                truth = False
            items.append(dict(f, set="domain", id=t["id"], group=t["domain"],
                              contaminated=t["contaminated"], ran=None, truth=truth))
    json.dump(items, open(os.path.join(HERE, "data", "split_detector.json"), "w"), indent=1)

    def score(sub, rule):
        tp = sum(1 for x in sub if x[rule] and x["truth"])
        fp = sum(1 for x in sub if x[rule] and x["truth"] is False)
        fn = sum(1 for x in sub if not x[rule] and x["truth"])
        tn = sum(1 for x in sub if not x[rule] and x["truth"] is False)
        return f"acc {tp + tn}/{tp + fp + fn + tn}  (TP {tp} FP {fp} FN {fn} TN {tn})"

    sets = [("LiveBench, the 21 run", [x for x in items if x["set"] == "lb" and x["ran"]]),
            ("LiveBench, 107 not run", [x for x in items if x["set"] == "lb" and not x["ran"]]),
            ("domain uncontaminated (120)", [x for x in items if x["set"] == "domain" and x.get("contaminated") is False]),
            ("domain three_tsl (20, contaminated)", [x for x in items if x["set"] == "domain" and x["group"] == "three_tsl"])]
    for name, sub in sets:
        print(f"\n{name}  n={len(sub)} positives={sum(1 for x in sub if x['truth'])}")
        for rule in ("R_op", "R_ref", "R_text", "R_ref+", "p_raw", "p_stub", "i_op", "i_narrow"):
            print(f"  {rule:8} {score(sub, rule)}")
    tc = [x for x in items if x["set"] == "domain" and x["group"] == "typescript_tc"]
    print(f"\ntype-challenges n={len(tc)} (no label): " + ", ".join(
        f"{r} flags {sum(1 for x in tc if x[r])}" for r in ("R_op", "R_ref", "R_text", "p_raw", "i_op")))
    # which items R_op gets wrong on LiveBench
    for x in items:
        if x["set"] == "lb" and x["ran"]:
            print(f"  {x['id']} {x['group'][:10]:10} truth={x['truth']!s:5} p_raw={x['p_raw']!s:5} p_stub={x['p_stub']!s:5} i_op={x['i_op']!s:5} i_narrow={x['i_narrow']!s:5}")


if __name__ == "__main__":
    main()
