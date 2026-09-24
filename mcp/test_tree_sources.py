#!/usr/bin/env python
"""The tokonoma's new sources, asserted: mcp/recent_turns.py (fan-out and
recall from the proxy's own x_yamadori records) and mcp/tree_sources.py
(index breadth for the roots, index freshness for the moss). No GPU, no
network, no model; every index is a temp sqlite file.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_tree_")
os.environ["YAMADORI_PACKAGES_DIR"] = os.path.join(_TMP, "packages")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")

import recent_turns as rt  # noqa: E402
import tree_sources as ts  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def near(a, b, tol=1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def fan(n=3, winner=2):
    return {"n": n, "winner_index": winner, "selection": "code_check",
            "candidates": [{"index": i, "role": "x"} for i in range(n)]}


def test_fanout():
    rt.reset()
    now = 10_000.0
    check(rt.summary(now)["fanout"] is None, "no request yet: no fan-out (inert)")
    rt.note({"fanout": fan(), "skills": {"path": "hints"}, "hints": []}, now=now - 30)
    f = rt.summary(now)["fanout"]
    check(f and f["arity"] == 3 and f["chosen"] == 2 and f["culled"] == [0, 1]
          and f["fade"] == 1.0, "arity, chosen and culled from x_yamadori.fanout", str(f))
    f = rt.summary(now + rt.FANOUT_HOLD_S - 30 + rt.FANOUT_DECAY_S / 2)["fanout"]
    check(f and near(f["fade"], 0.5, 1e-3), "after the hold it decays", str(f and f["fade"]))
    check(rt.summary(now + 1000)["fanout"] is None, "and is gone after hold + decay")
    rt.note({"fanout": None, "hints": []}, now=now)
    check(rt.summary(now + 1)["fanout"]["arity"] == 3,
          "a later request without fan-out does not cut the hold short")
    rt.note({"fanout": {"n": 0, "error": "RuntimeError"}}, now=now + 2)
    rt.note({"fanout": fan(1, 0)}, now=now + 3)
    check(rt.summary(now + 4)["fanout"]["arity"] == 3,
          "a failed or single-candidate fan-out is not a fork")
    rt.reset()
    rt.note({"fanout": fan(2, None)}, now=now)
    check(rt.summary(now)["fanout"]["chosen"] == 0, "no winner index: the original (0) was delivered")
    rt.reset()
    rt.note({"fanout": fan(), "utility": True}, now=now)
    check(rt.summary(now)["fanout"] is None, "a utility call's record is ignored")
    rt.note("not a dict")
    check(True, "note() never raises")


def test_foliage():
    rt.reset()
    now = 50_000.0
    check(rt.summary(now)["foliage"] is None, "no turns: foliage inert")
    rt.note({"skills": {"path": "hints"}, "hints": [{}, {}, {}]}, now=now - 60)
    rt.note({"skills": {"path": "hints"}, "hints": []}, now=now - 30)
    rt.note({"skills": {"path": "hints"}, "hints": [{}] * 5, "utility": True}, now=now - 10)
    f = rt.summary(now)["foliage"]
    check(f["turns"] == 2 and f["items"] == 3 and near(f["density"], 0.5)
          and f["path"] == "hints" and f["tokens"] is None,
          "hints path: items per task turn / FOLIAGE_FULL, utility excluded, no tokens", str(f))
    rt.reset()
    rt.note({"skills": {"path": "skills", "ids": ["a", "b", "c", "d"], "tokens": 900}}, now=now)
    f = rt.summary(now)["foliage"]
    check(f["path"] == "skills" and f["items"] == 4 and f["tokens"] == 900 and f["density"] == 1.0,
          "skills path: ids and tokens, density capped at 1", str(f))
    check(rt.summary(now + rt.FOLIAGE_WINDOW_S + 1)["foliage"] is None,
          "turns older than the window drop out")


def mkindex(path, chunks, defs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE chunks(id INTEGER)")
    con.execute("CREATE TABLE defs(id INTEGER)")
    con.executemany("INSERT INTO chunks VALUES(?)", [(i,) for i in range(chunks)])
    con.executemany("INSERT INTO defs VALUES(?)", [(i,) for i in range(defs)])
    con.commit()
    con.close()


def test_formulas():
    check(ts.spread_of(None) is None and ts.spread_of(0) is None, "no items: no spread (inert)")
    check(ts.spread_of(1000) == 0.0 and ts.spread_of(10**6) == 1.0 and near(ts.spread_of(31623), 0.5, 1e-3),
          "spread = (log10(items) - 3) / 3, clamped", str(ts.spread_of(31623)))
    check(ts.moss_of({"a": None}) is None, "no age: no moss (inert)")
    check(ts.moss_of({"a": 0}) == 0.0 and ts.moss_of({"a": 30 * 86400}) == 1.0
          and ts.moss_of({"a": 300 * 86400}) == 1.0, "moss: fresh 0, a month 1, capped")
    check(near(ts.moss_of({"a": 0, "b": 30 * 86400, "c": None}), 0.5),
          "moss is the mean over the ages that exist")


def test_measure():
    now = time.time()
    p = os.environ["YAMADORI_PACKAGES_DIR"]
    mkindex(os.path.join(p, "a@1.sqlite3"), 300, 200)
    mkindex(os.path.join(p, "b@2.sqlite3"), 100, 400)
    code = os.path.join(_TMP, "code.sqlite3")
    mkindex(code, 50, 50)
    ts.PACKAGES = p
    real_code, real_repos, real_arm = ts._code_index, ts._repo_dbs, ts._last_arm
    ts._code_index = lambda: code
    ts._repo_dbs = lambda: [code]              # the bound index is not counted twice
    ts._last_arm = lambda: now - 2 * 86400
    try:
        m = ts.measure(now)
        nb = m["nebari"]
        check(nb["packages"] == {"indexes": 2, "chunks": 400, "defs": 600}
              and nb["code"] == {"chunks": 50, "defs": 50} and nb["repos"]["indexes"] == 0
              and nb["items"] == 1100, "breadth: packages + code index, no double count", str(nb))
        check(near(nb["spread"], ts.spread_of(1100)), "spread from the total")
        ages = m["moss"]["ages_s"]
        check(ages["skills_last_arm"] == 2 * 86400 and ages["packages"] is not None
              and ages["code"] is not None, "moss ages: packages, code, last arm", str(ages))
        check(m["moss"]["recall"] in ("hints", "skills"), "the live recall path is labelled")
        ts._cache.clear()
        s1 = ts.snapshot(now)
        mkindex(os.path.join(p, "c@3.sqlite3"), 10_000, 0)
        s2 = ts.snapshot(now + 5)
        check(s2["nebari"]["items"] == s1["nebari"]["items"],
              "counts are cached: a new index is not read within CACHE_S")
        s3 = ts.snapshot(now + ts.CACHE_S + 1)
        check(s3["nebari"]["items"] == 11_100, "and is read after it", str(s3["nebari"]["items"]))
        check("recent" in s3 and "fanout" in s3["recent"], "the recent turns ride along")
    finally:
        ts._code_index, ts._repo_dbs, ts._last_arm = real_code, real_repos, real_arm
    ts._cache.clear()
    ts.PACKAGES = os.path.join(_TMP, "nothing")
    ts._code_index = lambda: None
    ts._repo_dbs = lambda: []
    ts._last_arm = lambda: None
    try:
        m = ts.measure(now)
        check(m["nebari"]["spread"] is None, "no index anywhere: spread None (inert)")
    finally:
        ts._code_index, ts._repo_dbs, ts._last_arm = real_code, real_repos, real_arm


def test_vitals_carries_it():
    import vitals
    t = vitals.tree()
    check(isinstance(t, dict) and "nebari" in t and "moss" in t and "recent" in t,
          "vitals.tree() is the tree payload")
    check("tree(" in open(os.path.join(HERE, "vitals.py"), encoding="utf-8").read().split("def snapshot")[1][:1600],
          "and snapshot() includes it")


def main() -> int:
    for fn in (test_fanout, test_foliage, test_formulas, test_measure, test_vitals_carries_it):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
