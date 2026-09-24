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


def test_an_explicit_session_token_separates_identical_openings():
    """X-Yamadori-Session (2026-09-23): benchmark rows open with identical
    messages and were one session. A token keeps them apart; without one
    the key is unchanged."""
    plain = nebari.key_of(MSGS, "acct-a")
    t1 = nebari.key_of(MSGS, "acct-a", "row-0001")
    t2 = nebari.key_of(MSGS, "acct-a", "row-0002")
    check(t1 != t2, "identical requests with different tokens: different keys",
          f"{t1} {t2}")
    check(t1 == nebari.key_of(MSGS, "acct-a", "row-0001"),
          "the same token: the same key")
    check(plain == nebari.key_of(MSGS, "acct-a")
          == nebari.key_of(MSGS, "acct-a", ""),
          "no header: identical requests share one key, exactly as before")
    check(t1 != plain, "a token never collides with the tokenless key")
    for bad in ("x" * 65, "has space", "semi;colon", "", None, 7, "a/b"):
        check(nebari.session_token(bad) == "",
              f"malformed token {bad!r} is ignored")
    check(nebari.session_token(" lb-row_42 ") == "lb-row_42",
          "a well-formed token is accepted (surrounding space trimmed)")
    _, sa = proxy.session_context(MSGS, "acct-a", "row-A")
    _, sb = proxy.session_context(MSGS, "acct-a", "row-B")
    check(sa["_key"] != sb["_key"], "session_context threads the token")
    proxy.run_our_tool("record_step", {"kind": "did",
                                       "summary": "only row A did this"},
                       None, None, None, sa)
    rb = proxy.run_our_tool("read_rings", {}, None, None, None, sb)
    check("only row A did this" not in rb,
          "and row B's work log does not see row A's", rb[:200])
    seen = []
    real = proxy.session_context
    proxy.session_context = lambda m, a="", s="", **k: (seen.append(s),
                                                       real(m, a, s, **k))[1]
    try:
        proxy.prepare({"model": "yamadori", "messages": MSGS,
                       "reasoning_effort": "minimal",
                       "_session_token": "row-C"})
    finally:
        proxy.session_context = real
    check(seen == ["row-C"], "prepare() passes body['_session_token'] on",
          str(seen))
    import inspect
    src = inspect.getsource(proxy.Handler.do_POST)
    check('"X-Yamadori-Session"' in src and "session_token" in src,
          "the proxy's handler reads the header through session_token")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "server.py"), encoding="utf-8") as f:
        srv = f.read()
    check('"x-yamadori-session"' in srv and "nebari.session_token" in srv,
          "and so does the ASGI server")


def test_ledger_misses_cost_no_connection_and_no_ddl():
    """Pre-deploy review, 2026-09-24 (FIX SOON #7): every ledger miss opened
    a new sqlite connection and ran a PRAGMA plus five DDL statements under
    the global lock, on the request path (proxy.ledger_restore looks up
    every message of every request). Now the schema is made once per
    database, reads reuse one connection, and a miss is remembered."""
    import sqlite3
    import threading
    real = sqlite3.connect
    n = {"connect": 0, "ddl": 0}

    class Counting(sqlite3.Connection):
        def execute(self, sql, *a):
            if sql.lstrip().upper().startswith(("CREATE", "PRAGMA")):
                n["ddl"] += 1
            return super().execute(sql, *a)

    def counting_connect(*a, **k):
        n["connect"] += 1
        return real(*a, factory=Counting, **k)
    nebari.ledger_reset()
    nebari.ledger_put("acct", "s1", "warm-up", "inject", "x")   # schema made
    nebari.sqlite3.connect = counting_connect
    try:
        for i in range(20):
            nebari.ledger_get("acct", f"missing-{i}", "inject")
        first = dict(n)
        for i in range(20):
            nebari.ledger_get("acct", f"missing-{i}", "inject")
        second = dict(n)
        # A write after a miss is seen: the remembered miss is cleared.
        nebari.ledger_put("acct", "s1", "missing-3", "inject", "now here")
        got = nebari.ledger_get("acct", "missing-3", "inject")
        nebari.ledger_reset()            # the memory layer gone: from sqlite
        got2 = nebari.ledger_get("acct", "missing-3", "inject")
        # The reused connection works from other threads.
        out: list = []
        t = threading.Thread(target=lambda: out.append(
            nebari.ledger_get("acct", "warm-up", "inject")))
        t.start()
        t.join(5)
    finally:
        nebari.sqlite3.connect = real
    check(first["connect"] <= 1 and first["ddl"] == 0,
          "20 distinct misses: at most one connection opened (reused), and "
          "no PRAGMA or DDL", str(first))
    check(second == first,
          "the same 20 misses again: nothing touches sqlite (misses are "
          "remembered)", f"{first} -> {second}")
    check(got == "now here" and got2 == "now here" and out == ["x"],
          "a write after a remembered miss is read back, from memory and "
          "from sqlite, and from another thread", f"{got!r} {got2!r} {out}")
    nebari.ledger_reset()


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_database,
               test_the_key_is_scoped_to_the_account,
               test_the_proxy_threads_the_account_into_the_session,
               test_an_explicit_session_token_separates_identical_openings,
               test_ledger_misses_cost_no_connection_and_no_ddl):
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
