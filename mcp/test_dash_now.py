#!/usr/bin/env python
"""The dashboard's view of the stack as it is (the 2026-09-29 fixup), asserted.

  /dash/api/skill-factory/library   (mcp/dash_skills.py; what the NEBARI
                     screen showed, folded into the Skills page 2026-09-30)
                     skill counts by state and the SERVED skills along each
                     taxonomy axis; one row per held package index (name
                     from the file stem, @scope restored, counts, meta) and
                     what reads them; cached; every part fails alone
  /dash/api/tiers    (mcp/dashboard.py) carries the model serving each tier
                     (mcp/max_mode.py model_for) and the max mode switch
  server.py          NEBARI (/dash/api/nebari) and /dash/api/results are gone

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

import dash_skills  # noqa: E402
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
    check(dash_skills.package_of("react-three__fiber@10.0.0-alpha.5") == ("@react-three/fiber", "10.0.0-alpha.5"),
          "a scoped package's stem gets its @scope/ back")
    check(dash_skills.package_of("three@0.186.0") == ("three", "0.186.0"), "an unscoped one is as written")
    check(dash_skills.package_of("noversion") == ("noversion", ""), "a stem with no version never raises")


def test_packages_read_only():
    a = _index("react-three__fiber@10.0.0-alpha.5", 5, 7,
               {"files": "3", "embedded": "1", "complete": "1", "published": "2026-04-01"})
    _index("three@0.186.0", 50, 1, {"files": "10", "embedded": "0"})
    before = os.path.getmtime(a), os.path.getsize(a)
    rows = dash_skills.held_packages(PKG_DIR)
    check([r["package"] for r in rows] == ["three", "@react-three/fiber"],
          "one row per index, largest first", str([r["package"] for r in rows]))
    f = [r for r in rows if r["package"] == "@react-three/fiber"][0]
    check(f["chunks"] == 5 and f["defs"] == 7 and f["files"] == 3 and f["embedded"] is True
          and f["published"] == "2026-04-01", "counts and meta are read", str(f))
    check("unseen" not in f, "no deep-thinking unseen reason (removed 2026-09-29)", str(f))
    t = rows[0]
    check(t["embedded"] is False and t["complete"] is None,
          "an unembedded index says so; absent meta is None, not a guess", str(t))
    check((os.path.getmtime(a), os.path.getsize(a)) == before, "the index files are not written")


def test_skills_view():
    import skills
    real = skills.counts, skills.armed
    skills.counts = lambda: {"armed": 3, "quarantined": 1, "pipeline": 0}
    skills.armed = lambda *a, **k: [
        {"id": "a", "category": {"framework": ["r3f"], "language": ["typescript"], "domain": ["gpu"]}},
        {"id": "b", "category": {"framework": ["react", "r3f"], "language": ["typescript"]}},
        {"id": "c", "category": {"language": ["rust"], "situation": ["error_output"]}},
    ]
    try:
        v = dash_skills.library_by_area()
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
    dash_skills._library_cache.clear()
    skills.counts = lambda: (_ for _ in ()).throw(RuntimeError("store locked"))
    try:
        code, ctype, body = dash_skills.handle_get("/dash/api/skill-factory/library")
    finally:
        skills.counts = real
    j = json.loads(body)
    check(code == 200 and ctype == "application/json", "the route answers JSON", f"{code} {ctype}")
    check("error" in j["areas"] and "store locked" in j["areas"]["error"],
          "a failing part reports its own error", str(j["areas"]))
    check(isinstance(j["packages"], list) and len(j["packages"]) == 2, "the other parts still answer",
          str(j["packages"])[:200])
    who = [r["who"] for r in j["readers"]]
    check(any("code-search" in w for w in who) and any("type check" in w for w in who)
          and any("find_by_meaning" in r["needs_embedding"] for r in j["readers"]),
          "what reads a held package today: the code-search tools (find_by_meaning "
          "needs the index embedded) and the PROVE type check", str(j["readers"]))
    check(dash_skills.handle_get("/dash/api/skill-factory/library/x") is None,
          "only its own path is claimed")
    # cached: a second call inside LIBRARY_CACHE_S does not read the stores again
    calls = []
    skills.counts = lambda: calls.append(1) or {}
    try:
        dash_skills.library()
    finally:
        skills.counts = real
    check(not calls, "a second read inside the cache window is the cache", str(calls))
    dash_skills._library_cache.clear()


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
    check("dash_nebari" not in src and "dash_results" not in src,
          "NEBARI (/dash/api/nebari) and /dash/api/results are retired: server.py "
          "no longer dispatches them (2026-09-30)")
    check(not os.path.exists(os.path.join(HERE, "dash_nebari.py"))
          and not os.path.exists(os.path.join(HERE, "dash_results.py")),
          "and their modules are gone")


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
