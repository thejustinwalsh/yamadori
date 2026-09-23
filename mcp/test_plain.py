#!/usr/bin/env python
"""The controlled-English checker, asserted. No model, no network.

WHAT THIS IS GATING

`plain.py` flags ASD-STE100 violations in text this stack writes for the
model. It is a detector, not a rewriter. Its promises:

  1. Each rule fires on its violation and reports the span that triggered it.
  2. Clean controlled English produces no violations (a checker that flags
     everything is as useless as one that flags nothing).
  3. Sentences over MAX_WORDS are flagged with the count.
  4. `score` is violations per sentence; `report` dedupes repeats.
"""
from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plain  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def rules(text: str) -> list[str]:
    return [v["rule"] for v in plain.check(text)]


# ---------------------------------------------------------------------------


def test_each_rule_fires_on_its_violation():
    cases = {"modal": "The tool should return a path.",
             "contraction": "The tool doesn't return a path.",
             "semicolon": "The tool returns a path; the caller reads it.",
             "emdash": "The tool returns a path -- the caller reads it.",
             "present_perfect": "The caller has opened the file.",
             "gerund_after_comma": "The tool returns, leaving the file open.",
             "passive": "The file is opened by the caller.",
             "vague_quantifier": "The tool returns several paths."}
    for rid, text in cases.items():
        got = rules(text)
        check(rid in got, f"{rid} fires on {text!r}", str(got))
    check("emdash" in rules("One — two."), "a real em dash is caught too")


def test_the_span_is_reported():
    v = plain.check("The tool should return a path.")
    check(v and v[0]["found"] == "should" and v[0]["sentence"] == 0,
          "the triggering word and sentence index are reported", str(v[:1]))
    check(v and v[0]["why"] and v[0]["fix"], "with why it hurts and what to do")


def test_clean_text_is_clean():
    text = ("The tool returns one path. The caller reads the file. "
            "Run the check before the commit.")
    check(plain.check(text) == [], "controlled English has no violations",
          str(rules(text)))
    check(plain.report(text) == "clean (3 sentences)", "and the report says so",
          plain.report(text))
    check(plain.score(text) == 0.0, "and scores zero")
    check(plain.check("") == [] and plain.score("") == 0.0,
          "empty text is clean and scores zero")


def test_long_sentences_are_flagged_at_the_limit():
    at = " ".join(["word"] * plain.MAX_WORDS) + "."
    over = " ".join(["word"] * (plain.MAX_WORDS + 1)) + "."
    check("length" not in rules(at), f"{plain.MAX_WORDS} words is allowed")
    v = [x for x in plain.check(over) if x["rule"] == "length"]
    check(len(v) == 1 and v[0]["found"] == f"{plain.MAX_WORDS + 1} words",
          "one more is flagged with the count", str(v))
    check([x["rule"] for x in plain.check(over, max_words=50)] == [],
          "the limit is a parameter")


def test_score_and_report_count_and_dedupe():
    text = "It should work. It should work. It works."
    check(plain.score(text) == round(2 / 3, 2), "score is violations per sentence",
          str(plain.score(text)))
    r = plain.report(text)
    check(r.startswith("2 violations across 3 sentences"), "the report totals them", r)
    check(r.count("modal") == 1, "and lists a repeated violation once", r)


def main() -> int:
    for fn in (test_each_rule_fires_on_its_violation,
               test_the_span_is_reported,
               test_clean_text_is_clean,
               test_long_sentences_are_flagged_at_the_limit,
               test_score_and_report_count_and_dedupe):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
