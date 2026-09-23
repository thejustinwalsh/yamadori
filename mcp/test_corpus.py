#!/usr/bin/env python
"""Corpus telemetry the dashboard reads, asserted. No GPU, temp database.

  1. result_strata() counts find_by_meaning hits per result tier: TAPROOT (a
     declaration with the name), BRANCH (both retrievers agree), SHOOT (one
     retriever only). A result with no tier headers is None, not zeros.
  2. log_tool_result() records those counts, and strata_totals() sums recent
     ones -- the source for the dashboard's TIER STRATA panel, which had none
     and filled its space with prose (2026-09-22).
  3. With no repository bound, the full result text is kept (a looping
     session could not be read back from "4165 chars"); with one bound, only
     the shape is, never a copy of the user's source.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_corpus_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")

import corpus  # noqa: E402

_results: list[tuple[bool, str, str]] = []

TIERED = ("== TAPROOT -- a declaration with this name is here.\n"
          "### src/a.js:1-2   [symbol table]  declares Foo\n```\nx\n```\n"
          "== BRANCH -- both independent retrievers surfaced this.\n"
          "### src/b.js:1-2   [bm25, embed]\n### src/c.js:3-4   [bm25, embed]\n"
          "== SHOOT -- one retriever only.\n### src/d.js:1-9   [bm25]\n")


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def test_the_fixture_is_not_the_real_corpus():
    db = os.environ["YAMADORI_CORPUS_DB"]
    check(db.startswith(_TMP), "the corpus is a temp file", db)


def test_strata_are_counted_per_tier():
    check(corpus.result_strata(TIERED) == {"taproot": 1, "branch": 2,
                                           "shoot": 1},
          "hits are counted under each tier header",
          json.dumps(corpus.result_strata(TIERED)))
    check(corpus.result_strata("No results.") is None
          and corpus.result_strata("src/a.js:3: foo()") is None,
          "a result with no tiers is None, not zeros")


def test_logging_feeds_the_dashboard_totals():
    before = corpus.strata_totals()
    corpus.log_tool_result("t1", None, "find_by_meaning", TIERED, 12.0)
    corpus.log_tool_result("t1", None, "find_by_pattern", "src/a.js:1: x", 3.0)
    after = corpus.strata_totals()
    check(after["searches"] == before["searches"] + 1
          and after["branch"] == before["branch"] + 2
          and after["taproot"] == before["taproot"] + 1,
          "strata_totals sums the tiered searches only", json.dumps(after))


def test_full_text_only_without_a_repository():
    import sqlite3
    corpus.log_tool_result("t2", None, "find_by_pattern", "PKG RESULT", 1.0)
    corpus.log_tool_result("t3", "C:/some/repo", "find_by_pattern",
                           "USER SOURCE", 1.0)
    con = sqlite3.connect(os.environ["YAMADORI_CORPUS_DB"])
    rows = {t: json.loads(p) for t, p in con.execute(
        "SELECT turn, payload FROM events WHERE turn IN ('t2','t3')")}
    con.close()
    check(rows.get("t2", {}).get("text") == "PKG RESULT",
          "no repository bound: the full result is kept", json.dumps(rows))
    check("text" not in rows.get("t3", {}),
          "a bound repository: only the shape, never the user's source",
          json.dumps(rows.get("t3")))


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_corpus,
               test_strata_are_counted_per_tier,
               test_logging_feeds_the_dashboard_totals,
               test_full_text_only_without_a_repository):
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
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
