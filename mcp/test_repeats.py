#!/usr/bin/env python
"""repeats._empty: which tool results count as "nothing came back". No GPU.

    python mcp/test_repeats.py      -> "N/M checks passed"

THE BUG THIS GATES (docs/TRANSCRIPT-REVIEW-2026-09-23.md)

`_empty` matched a list of English phrases and none of the proxy's own
wordings for a caller with no repository -- "... searched instead. None
matched." and "== searched pkg@ver: no match ==" -- nor the structured
envelopes ({"ok": true, "matches": 0}, {"ok": false, "retryable": false}).
So `Turn.cached_empty` never fired for exactly the callers the repeat
breaker exists for, and an identical miss was re-run every time.

The envelopes here are produced by the real producers (code_search's
result builders and the proxy's own constants), not typed from memory
(PROTOCOL rule 7). The proxy's package-fallback miss is exercised through
the real code path in mcp/test_tools.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_TMP = tempfile.mkdtemp(prefix="yamadori_test_repeats_")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")

import code_search as cs  # noqa: E402
import proxy  # noqa: E402
import repeats  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def test_empty_recognises_every_no_result_shape():
    empty = {
        "the proxy's no-repository miss": (
            "None matched in any library the remote code-intelligence "
            "service holds. It searches library source only; the user's own "
            "project is searched with your client's own file tools. The same "
            "arguments return the same miss.\n\n== three@0.185.1 ==\n..."),
        "the proxy's named-package miss": (
            "== searched typegpu@0.12.5, paths matching 'data': no match ==\n"
            "..."),
        "the repeat notice itself": proxy._EMPTY_AGAIN,
        "a structured clean miss": cs.ok_result("find_by_pattern", matches=0,
                                                files_scanned=12),
        "NO_INDEX (cannot change with these arguments)":
            cs.no_index_error("find_by_meaning"),
        "an empty string": "",
    }
    for why, text in empty.items():
        check(repeats._empty(text), f"empty: {why}", (text or "")[:120])
    not_empty = {
        "a structured hit": cs.ok_result("find_by_pattern", matches=3),
        "bad arguments (retryable: the next call differs)":
            cs.bad_arguments_error("find_by_pattern", "missing pattern",
                                   "pass pattern"),
        "a prose hit": "src/pool.ts:1-4\nexport function sizeKvPool",
        "a check_code result": "check_code: python -- parses",
    }
    for why, text in not_empty.items():
        check(not repeats._empty(text), f"not empty: {why}", text[:120])


def test_the_breaker_now_fires_for_a_no_repository_miss():
    t = repeats.Turn()
    miss = ("None matched in any library the remote code-intelligence "
            "service holds. It searches library source only.")
    t.record("find_by_pattern", {"pattern": "fooBar"}, miss)
    check(t.cached_empty("find_by_pattern", {"pattern": "fooBar"}),
          "an identical call after that miss is served from the cache")
    check(not t.cached_empty("find_by_pattern", {"pattern": "other"}),
          "a different call is not")


def main() -> int:
    for fn in (test_empty_recognises_every_no_result_shape,
               test_the_breaker_now_fires_for_a_no_repository_miss):
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} raised", traceback.format_exc())
    passed = sum(1 for ok, _, _ in _results if ok)
    for ok, name, detail in _results:
        if not ok:
            print(f"FAIL  {name}   <- {detail}")
    print(f"\n{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
