#!/usr/bin/env python
"""What the model can draw on, for the dashboard's NEBARI screen.

Under /dash/api, which server.py gates with accounts.identify before
dispatching; an unauthenticated request never reaches this module.

    GET /dash/api/nebari   the skills (the one knowledge system since
                           2026-09-26: counts by state, and the ARMED skills
                           counted along each taxonomy axis), the held
                           package indexes (index/packages/*.sqlite3: the
                           package and version, chunks, definitions, files,
                           whether it is embedded, its publish date, and
                           why deep thinking treats it as unseen when it
                           does -- deep.unseen), and the bound code and
                           repository indexes (mcp/tree_sources.py).

WHY IT REPLACED /dash/api/stats ON THAT SCREEN

NEBARI drew its roots from the recipe corpus (bench/recipes, /dash/api/stats).
The corpus stopped being served on 2026-09-26: its rows were migrated into
skills (mcp/skill_migrate.py), and the recipe files stay only as provenance
and a dataset's extract output. A root flare of recipe domain tags showed
something the model no longer reads.

Read-only throughout: every sqlite read is `mode=ro`, nothing is written, no
model is asked. Cached CACHE_S, because the package indexes do not change
between polls and the screen polls once a minute.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE_S = 60.0
PATH = "/dash/api/nebari"
# The taxonomy axes counted over the armed skills, in the order the screen
# lists them (mcp/skill_classify.py taxonomy()).
AXES = ("framework", "language", "domain", "artifact", "phase")

_lock = threading.Lock()
_cache: dict = {}


def _json(code: int, payload: dict):
    return code, "application/json", json.dumps(payload, default=str).encode()


def package_of(stem: str) -> tuple[str, str]:
    """(npm name, version) from an index file's stem: deps.slug writes
    `@scope/name@1.2.3` as `scope__name@1.2.3`."""
    name, _, version = stem.rpartition("@")
    if not name:
        return stem, ""
    if "__" in name:
        name = "@" + name.replace("__", "/", 1)
    return name, version


def _meta(path: str) -> dict:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return {}
    try:
        return {k: v for k, v in con.execute("SELECT k, v FROM meta")}
    except sqlite3.Error:
        return {}
    finally:
        con.close()


def packages(folder: str | None = None) -> list[dict]:
    """One row per held package index, largest first."""
    import tree_sources
    folder = folder or tree_sources.PACKAGES
    try:
        import deep
        unseen = deep.unseen
    except Exception:                                            # noqa: BLE001
        unseen = None
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*.sqlite3"))):
        stem = os.path.basename(p)[:-len(".sqlite3")]
        name, version = package_of(stem)
        c = tree_sources._counts(p) or {}
        m = _meta(p)
        why = None
        if unseen is not None and version:
            try:
                why = unseen(name, version)
            except Exception:                                    # noqa: BLE001
                why = None
        out.append({"package": name, "version": version,
                    "chunks": c.get("chunks"), "defs": c.get("defs"),
                    "files": int(m["files"]) if str(m.get("files", "")).isdigit() else None,
                    "embedded": (m.get("embedded") == "1") if "embedded" in m else None,
                    "complete": (m.get("complete") == "1") if "complete" in m else None,
                    "published": m.get("published") or None,
                    "unseen": why})
    return sorted(out, key=lambda r: -((r["chunks"] or 0) + (r["defs"] or 0)))


def skills_view() -> dict:
    """Counts by state (skills.counts), and the SERVED skills -- what
    selection reads, skills.armed(): armed, enabled, and not all doubt --
    counted along each taxonomy axis (their category, skill_classify)."""
    import skills
    counts = skills.counts()
    served = skills.armed()
    axes: dict[str, dict[str, int]] = {a: {} for a in AXES}
    for s in served:
        cat = s.get("category") if isinstance(s.get("category"), dict) else {}
        for axis, vals in cat.items():
            if axis not in axes:
                continue
            for v in vals or []:
                axes[axis][str(v)] = axes[axis].get(str(v), 0) + 1
    labels: dict = {}
    try:
        import skill_classify
        tax = skill_classify.taxonomy()
        for axis in AXES:
            vals = tax.get(axis) if isinstance(tax, dict) else None
            if isinstance(vals, list):
                labels[axis] = {str(v.get("id")): str(v.get("name") or v.get("id"))
                                for v in vals if isinstance(v, dict)}
    except Exception:                                            # noqa: BLE001
        labels = {}
    return {"counts": counts, "total": sum(counts.values()),
            "served": len(served),
            "served_by": {a: dict(sorted(v.items(), key=lambda kv: -kv[1]))
                          for a, v in axes.items()},
            "labels": labels}


def overview(now: float | None = None) -> dict:
    now = time.time() if now is None else now
    with _lock:
        if _cache and now - _cache["measured_at"] < CACHE_S:
            return dict(_cache)
    out: dict = {"measured_at": now, "cache_s": CACHE_S}
    try:
        out["skills"] = skills_view()
    except Exception as e:                                       # noqa: BLE001
        out["skills"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    try:
        out["packages"] = packages()
    except Exception as e:                                       # noqa: BLE001
        out["packages"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    try:
        import tree_sources
        n = tree_sources.snapshot(now).get("nebari") or {}
        out["indexes"] = {"code": n.get("code"), "repos": n.get("repos"),
                          "items": n.get("items"), "spread": n.get("spread")}
    except Exception as e:                                       # noqa: BLE001
        out["indexes"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    with _lock:
        _cache.clear()
        _cache.update(out)
    return out


def handle_get(path: str):
    if path.rstrip("/") != PATH:
        return None
    try:
        return _json(200, overview())
    except Exception as e:                                       # noqa: BLE001
        return _json(500, {"error": f"nebari overview raised "
                                    f"{type(e).__name__}: {e}"})
