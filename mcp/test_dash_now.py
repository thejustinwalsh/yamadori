#!/usr/bin/env python
"""The dashboard's view of the stack as it is (the 2026-09-29 fixup), asserted.

  /dash/api/nebari   (mcp/dash_nebari.py) what the model can draw on: skill
                     counts by state and the SERVED skills along each
                     taxonomy axis; one row per held package index (name
                     from the file stem, @scope restored, counts, meta, and
                     why deep thinking treats it as unseen); cached; every
                     part fails alone; the route claims only its own path
  /dash/api/tiers    (mcp/dashboard.py) carries the model serving each tier
                     (mcp/max_mode.py model_for) and the max mode switch
  server.py          routes /dash/api/nebari through the gated catch-all

Every store is a temp path (mcp/offline_stores.py) set BEFORE any import;
the package indexes are temp sqlite files; nothing reaches a port.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_dash_now_")
PKG_DIR = os.path.join(_TMP, "packages")
os.makedirs(PKG_DIR, exist_ok=True)
os.environ["YAMADORI_PACKAGES_DIR"] = PKG_DIR

import dash_nebari  # noqa: E402
import dashboard  # noqa: E402
import tree_sources  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def _index(stem: str, chunks: int, defs: int, meta: dict) -> str:
    p = os.path.join(PKG_DIR, f"{stem}.sqlite3")
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE chunks(id INTEGER PRIMARY KEY, path TEXT, start INT, end INT, text TEXT, vec BLOB)")
    con.execute("CREATE TABLE defs(name TEXT, kind TEXT, path TEXT, start INT, end INT, line TEXT)")
    con.execute("CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT)")
    con.executemany("INSERT INTO chunks(path) VALUES (?)", [("a",)] * chunks)
    con.executemany("INSERT INTO defs(name) VALUES (?)", [("d",)] * defs)
    con.executemany("INSERT INTO meta VALUES (?, ?)", list(meta.items()))
    con.commit()
    con.close()
    return p


def test_package_names():
    check(os.path.abspath(tree_sources.PACKAGES) == os.path.abspath(PKG_DIR),
          "the package folder read is the temp one, never index/packages", tree_sources.PACKAGES)
    check(dash_nebari.package_of("react-three__fiber@10.0.0-alpha.5") == ("@react-three/fiber", "10.0.0-alpha.5"),
          "a scoped package's stem gets its @scope/ back")
    check(dash_nebari.package_of("three@0.186.0") == ("three", "0.186.0"), "an unscoped one is as written")
    check(dash_nebari.package_of("noversion") == ("noversion", ""), "a stem with no version never raises")


def test_packages_read_only():
    a = _index("react-three__fiber@10.0.0-alpha.5", 5, 7,
               {"files": "3", "embedded": "1", "complete": "1", "published": "2026-04-01"})
    _index("three@0.186.0", 50, 1, {"files": "10", "embedded": "0"})
    before = os.path.getmtime(a), os.path.getsize(a)
    import deep
    real = deep.unseen
    deep.unseen = lambda pkg, ver, db=None: ("a new major" if pkg == "@react-three/fiber" else None)
    try:
        rows = dash_nebari.packages(PKG_DIR)
    finally:
        deep.unseen = real
    check([r["package"] for r in rows] == ["three", "@react-three/fiber"],
          "one row per index, largest first", str([r["package"] for r in rows]))
    f = [r for r in rows if r["package"] == "@react-three/fiber"][0]
    check(f["chunks"] == 5 and f["defs"] == 7 and f["files"] == 3 and f["embedded"] is True
          and f["published"] == "2026-04-01", "counts and meta are read", str(f))
    check(f["unseen"] == "a new major", "deep.unseen's reason rides on the row", str(f["unseen"]))
    t = rows[0]
    check(t["embedded"] is False and t["complete"] is None and t["unseen"] is None,
          "an unembedded index says so; absent meta is None, not a guess", str(t))
    check((os.path.getmtime(a), os.path.getsize(a)) == before, "the index files are not written")


def test_skills_view():
    import skills
    real = skills.counts, skills.armed
    skills.counts = lambda: {"armed": 3, "quarantined": 1, "pipeline": 0}
    skills.armed = lambda: [
        {"id": "a", "category": {"framework": ["r3f"], "language": ["typescript"], "domain": ["gpu"]}},
        {"id": "b", "category": {"framework": ["react", "r3f"], "language": ["typescript"]}},
        {"id": "c", "category": {"language": ["rust"], "situation": ["error_output"]}},
    ]
    try:
        v = dash_nebari.skills_view()
    finally:
        skills.counts, skills.armed = real
    check(v["served"] == 3 and v["total"] == 4, "served is what selection reads; total is every state", str(v))
    check(v["served_by"]["framework"] == {"r3f": 2, "react": 1}, "frameworks counted, largest first",
          str(v["served_by"]["framework"]))
    check(list(v["served_by"]["language"]) == ["typescript", "rust"], "languages by count", str(v["served_by"]["language"]))
    check("situation" not in v["served_by"], "only the listed axes", str(list(v["served_by"])))
    check(v["labels"].get("framework", {}).get("r3f") == "React Three Fiber",
          "the taxonomy's display names ride along", str(v["labels"].get("framework")))


def test_route_cache_and_isolation():
    import skills
    real = skills.counts
    dash_nebari._cache.clear()
    skills.counts = lambda: (_ for _ in ()).throw(RuntimeError("store locked"))
    try:
        code, ctype, body = dash_nebari.handle_get("/dash/api/nebari")
    finally:
        skills.counts = real
    j = json.loads(body)
    check(code == 200 and ctype == "application/json", "the route answers JSON", f"{code} {ctype}")
    check("error" in j["skills"] and "store locked" in j["skills"]["error"],
          "a failing part reports its own error", str(j["skills"]))
    check(isinstance(j["packages"], list) and len(j["packages"]) == 2, "the other parts still answer",
          str(j["packages"])[:200])
    check(dash_nebari.handle_get("/dash/api/nebari/x") is None and dash_nebari.handle_get("/dash/api/stats") is None,
          "only its own path is claimed")
    # cached: a second call inside CACHE_S does not read the stores again
    calls = []
    skills.counts = lambda: calls.append(1) or {}
    try:
        dash_nebari.overview()
    finally:
        skills.counts = real
    check(not calls, "a second read inside CACHE_S is the cache", str(calls))
    dash_nebari._cache.clear()


def test_tiers_carry_the_serving_model():
    import max_mode
    import tiers
    # safe_effort reads the served template from llama-swap: answer it here
    # (mcp/test_tiers.py gates the real read).
    tiers._accepted = ("low", "medium", "xhigh")
    tiers._accepted_of.clear()
    tiers._accepted_of.update({max_mode.MAIN: tiers._accepted, "flash-next": tiers._accepted})
    code, _, body = dashboard.handle_get("/dash/api/tiers")
    j = json.loads(body)
    check(code == 200 and all("model" in t for t in j["tiers"].values()), "every tier names its model")
    check(j["max_mode"]["enabled"] is max_mode.ENABLED, "and max mode's switch rides along", str(j["max_mode"]))
    import tier_models
    real = max_mode.ENABLED, max_mode.MAX, max_mode.TABLE
    max_mode.ENABLED, max_mode.MAX = True, "flash-next"
    max_mode.TABLE = tier_models.Table({"tiers": {"xhigh": "mirai-s", "max": "flash-next"}}, "test")
    try:
        j = json.loads(dashboard.handle_get("/dash/api/tiers")[2])
    finally:
        max_mode.ENABLED, max_mode.MAX, max_mode.TABLE = real
    check(j["tiers"]["max"]["model"] == "flash-next" and j["tiers"]["xhigh"]["model"] == "mirai-s"
          and j["tiers"]["medium"]["model"] == max_mode.MAIN,
          "with the tier table on, max and xhigh name their models and the rest the main one",
          str({k: v["model"] for k, v in j["tiers"].items()}))
    check(all("features" in t for t in j["tiers"].values()), "the feature matrix is still carried")


def test_server_routes_it():
    src = open(os.path.join(HERE, "server.py"), encoding="utf-8").read()
    check("import dash_nebari" in src and "dash_nebari.handle_get" in src,
          "server.py dispatches /dash/api/nebari through the gated GET chain")


def main() -> int:
    for fn in (test_package_names, test_packages_read_only, test_skills_view,
               test_route_cache_and_isolation, test_tiers_carry_the_serving_model,
               test_server_routes_it):
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
