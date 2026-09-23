#!/usr/bin/env python
"""The durable work log, asserted directly. No real index/rings.sqlite3.

WHAT THIS IS GATING

`rings` exists so that work done before a compaction is not redone after it.
The promises, each asserted below:

  1. What is recorded reads back, oldest first, with checks surfaced as
     PASS/FAIL evidence above the sequence.
  2. Sessions do not see each other, and a call with no session is refused
     rather than answered from somebody else's log.
  3. A bad kind or empty summary is answered with what to do, not a crash.
  4. The log never claims to be empty when it is not. (Until 2026-09-22 a
     `kind` filter that matched nothing -- or `limit=0` -- returned "Nothing
     has been recorded yet ... record it now", inviting the model to redo
     work the log exists to prove was done.)

WHAT IS COVERED ELSEWHERE

`mcp/test_tools.py` drives `read_rings` / `record_step` through the proxy's
dispatch: per-session isolation end to end, and argument validation there
(`limit` as a word, `limit` 0). This file tests the module underneath it.

NO REAL DATABASE

`RINGS_DB` is pointed at a temp file BEFORE `rings` is imported, because the
module reads it at import time.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_rings_")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")

import rings  # noqa: E402

REAL = os.path.abspath(os.path.join(HERE, "..", "index", "rings.sqlite3"))

_results: list[tuple[bool, str, str]] = []
_n = 0


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def fresh_session() -> str:
    global _n
    _n += 1
    return f"test-session-{_n}"


# ---------------------------------------------------------------------------


def test_the_fixture_is_not_the_real_database():
    db = os.path.abspath(rings.RINGS_DB)
    check(db.startswith(os.path.abspath(_TMP)), "the rings database is a temp file", db)
    check(db != REAL, "and is not index/rings.sqlite3")


def test_what_is_recorded_reads_back_in_order():
    s = fresh_session()
    out1 = rings.record("did", "changed parser.ts to accept tabs", session=s)
    out2 = rings.record("learned", "the lexer is generated", session=s)
    check("entry 1" in out1 and "entry 2" in out2 and repr(s) in out2,
          "record() reports the entry's position in its session", out2)
    log = rings.read(session=s)
    check("2 entries" in log, "the header counts the session's entries", log[:80])
    a, b = log.find("changed parser.ts"), log.find("the lexer is generated")
    check(0 < a < b, "entries read oldest first, as a narrative")
    check("Do not redo it" in log, "and the log says listed work is done")


def test_checks_are_surfaced_as_evidence():
    s = fresh_session()
    rings.record("did", "edited foo", session=s)
    rings.record("check", "npm test", outcome="pass", session=s)
    rings.record("check", "tsc --noEmit", outcome="FAIL", session=s)
    rings.record("check", "lint", outcome="flaky", session=s)
    rings.record("check", "cargo test", session=s)
    log = rings.read(session=s)
    head, _, seq = log.partition("SEQUENCE:")
    check("CHECKS RUN" in head, "checks get their own block, before the sequence")
    check("[PASS] npm test" in head, "a pass is shown as PASS")
    check("[FAIL] tsc --noEmit" in head, "an outcome is matched case-insensitively")
    check("[flaky] lint" in head, "an unusual outcome is shown verbatim")
    check("[?] cargo test" in head, "a check with no outcome is marked unknown")
    check("edited foo" not in head, "only checks are in the evidence block")


def test_detail_shows_its_first_line_only():
    s = fresh_session()
    rings.record("did", "patched", detail="first line\nsecond line", session=s)
    log = rings.read(session=s)
    check("first line" in log and "second line" not in log,
          "a multi-line detail shows its first line")


def test_limit_shows_the_most_recent():
    s = fresh_session()
    for i in range(5):
        rings.record("did", f"step-{i}", session=s)
    log = rings.read(session=s, limit=2)
    check("step-3" in log and "step-4" in log and "step-2" not in log,
          "limit keeps the most recent entries", log)
    check("5 entries" in log and "showing last 2" in log,
          "and says how many are hidden", log.splitlines()[0])


def test_the_log_never_claims_to_be_empty_when_it_is_not():
    """THE BUG THIS FILE FOUND."""
    s = fresh_session()
    rings.record("did", "refactored the loader", session=s)
    for label, kwargs in (("a kind filter that matches nothing", {"kind": "check"}),
                          ("limit=0", {"limit": 0}),
                          ("a negative limit", {"limit": -5})):
        log = rings.read(session=s, **kwargs)
        check("Nothing has been recorded" not in log and "record it now" not in log,
              f"{label} does not say the log is empty", log[:120])
    log = rings.read(session=s, kind="check")
    check("1 entries" in log or "1 entr" in log,
          "a filter miss says how many entries the session does have", log)
    log = rings.read(session=s, limit=0)
    check("refactored the loader" in log, "limit=0 still shows the entry", log[:120])
    log = rings.read(session=s, kind="DID")
    check("refactored the loader" in log, "the kind filter is case-insensitive",
          log[:120])


def test_an_empty_session_says_so_honestly():
    s = fresh_session()
    log = rings.read(session=s)
    check("No entries" in log and "not the same as nothing having been done" in log,
          "an empty log says nothing is recorded, not that nothing was done", log)


def test_bad_input_is_answered_not_crashed():
    s = fresh_session()
    out = rings.record("banana", "x", session=s)
    check("unknown kind" in out and "did" in out and "check" in out,
          "an unknown kind lists the valid ones", out)
    out = rings.record("did", "   ", session=s)
    check("summary is required" in out, "a blank summary is refused with a reason", out)
    out = rings.record("CHECK", "  mixed case kind  ", outcome=" pass ", session=s)
    check(out.startswith("recorded"), "kind is case-insensitive", out)
    try:
        out = rings.record(None, "null kind and detail", detail=None, outcome=None,
                           session=s)
        check(out.startswith("recorded"),
              "null kind, detail and outcome are treated as absent", out)
    except Exception as e:                                       # noqa: BLE001
        check(False, "null fields do not crash record()", f"{type(e).__name__}: {e}")
    try:
        out = rings.record("did", None, session=s)
        check("summary is required" in out, "a null summary is refused, not raised", out)
    except Exception as e:                                       # noqa: BLE001
        check(False, "a null summary does not crash record()",
              f"{type(e).__name__}: {e}")
    con = sqlite3.connect(rings.RINGS_DB)
    n = con.execute("SELECT COUNT(*) FROM rings WHERE session=?", (s,)).fetchone()[0]
    stored = con.execute("SELECT kind, summary, outcome FROM rings WHERE session=? "
                         "ORDER BY id LIMIT 1", (s,)).fetchone()
    con.close()
    check(n == 2, "only the two valid calls wrote a row", str(n))
    check(stored == ("check", "mixed case kind", "pass"),
          "stored values are normalised and trimmed", str(stored))


def test_handle_is_scoped_to_the_callers_session():
    out = rings.handle("record_step", {"kind": "did", "summary": "x"})
    check(out.startswith("ERROR") and "session" in out,
          "a call with no _session is refused", out[:80])
    out = rings.handle("read_rings", {"_session": "   "})
    check(out.startswith("ERROR"), "a blank _session is refused too")
    rings.handle("record_step", {"_session": "alpha", "kind": "did",
                                 "summary": "SECRET-ALPHA"})
    other = rings.handle("read_rings", {"_session": "beta"})
    check("SECRET-ALPHA" not in other, "one session cannot read another's log")
    mine = rings.handle("read_rings", {"_session": "alpha"})
    check("SECRET-ALPHA" in mine, "a session reads its own log")
    check(rings.handle("not_a_rings_tool", {"_session": "alpha"}) is None,
          "an unknown tool name returns None, for the next dispatcher")


def test_the_tool_schemas_match_the_kinds():
    names = [t["name"] for t in rings.TOOLS]
    check(names == ["record_step", "read_rings"], "two tools are declared", str(names))
    for t in rings.TOOLS:
        check(t["inputSchema"]["properties"]["kind"]["enum"] == list(rings.KINDS),
              f"{t['name']} offers exactly the accepted kinds")
    check(rings.TOOLS[0]["inputSchema"]["required"] == ["kind", "summary"],
          "record_step requires kind and summary")


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_database,
               test_what_is_recorded_reads_back_in_order,
               test_checks_are_surfaced_as_evidence,
               test_detail_shows_its_first_line_only,
               test_limit_shows_the_most_recent,
               test_the_log_never_claims_to_be_empty_when_it_is_not,
               test_an_empty_session_says_so_honestly,
               test_bad_input_is_answered_not_crashed,
               test_handle_is_scoped_to_the_callers_session,
               test_the_tool_schemas_match_the_kinds):
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
    print(f"  temp database: {os.path.abspath(rings.RINGS_DB)}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
