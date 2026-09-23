#!/usr/bin/env python
"""Session identity, asserted. No GPU, no real database.

THE BUG THIS GATES

`nebari.key_of()` hashed only the first two messages. Two callers with a
common harness system prompt and the same opening line got ONE session: one
work log (`read_rings`), one discovered-package list, one tool-offer history.
Found 2026-09-22 when a fresh conversation from the test key was logged
OFFERED_EARLIER_THIS_SESSION. The account is now part of the key, and the
proxy passes it from `accounts.identify()` through `body["_account"]`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_nebari_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = _TMP

import nebari  # noqa: E402

nebari.DB = os.path.join(_TMP, "nebari.sqlite3")

import proxy  # noqa: E402

_results: list[tuple[bool, str, str]] = []

MSGS = [{"role": "system", "content": "You are a coding harness."},
        {"role": "user", "content": "hi"}]


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def test_the_fixture_is_not_the_real_database():
    check(os.path.abspath(nebari.DB).startswith(os.path.abspath(_TMP)),
          "nebari writes to a temp database", nebari.DB)


def test_the_key_is_scoped_to_the_account():
    a, b = nebari.key_of(MSGS, "acct-a"), nebari.key_of(MSGS, "acct-b")
    check(a != b, "identical messages from two accounts are two sessions")
    check(a == nebari.key_of(MSGS, "acct-a"),
          "and one account's conversation keeps a stable key")
    check(nebari.key_of(MSGS + [{"role": "assistant", "content": "x"}],
                        "acct-a") == a,
          "later turns do not change the key (it is the first two messages)")


def test_the_proxy_threads_the_account_into_the_session():
    _, sa = proxy.session_context(MSGS, "acct-a")
    _, sb = proxy.session_context(MSGS, "acct-b")
    check(sa["_key"] != sb["_key"],
          "session_context gives two accounts two sessions",
          f"{sa['_key']} {sb['_key']}")
    ra = proxy.run_our_tool("record_step", {"kind": "did",
                                            "summary": "only account A did this"},
                            None, None, None, sa)
    check("recorded" in ra.lower(), "account A records a step", ra[:120])
    rb = proxy.run_our_tool("read_rings", {}, None, None, None, sb)
    check("only account A did this" not in rb,
          "and account B's work log does not contain it", rb[:200])
    ra2 = proxy.run_our_tool("read_rings", {}, None, None, None, sa)
    check("only account A did this" in ra2,
          "while account A's own log does", ra2[:200])


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_database,
               test_the_key_is_scoped_to_the_account,
               test_the_proxy_threads_the_account_into_the_session):
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
