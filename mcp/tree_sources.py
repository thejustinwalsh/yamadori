#!/usr/bin/env python
"""What the tokonoma's roots, moss and foliage read (vitals.snapshot()
`tree`). Read-only; the sqlite counts are cached for CACHE_S.

  nebari   the breadth of what the server knows: every held package index
           (index/packages/*.sqlite3, chunks + defs), the bound code index
           (code_search.INDEX_DB) and each registered repository index
           (repos.known()). spread = clamp((log10(chunks + defs) - 3) / 3):
           a thousand items is a bare root plate, a million is full spread.
  moss     index freshness. Ages: the package indexes (median file age),
           the code index, the skill store's last arm (skill_versions.
           armed_at in the jobs DB) and, while YAMADORI_RECALL=hints, the
           recipe index (index/hints.npz). moss = mean over the ages that
           exist of clamp(log2(1 + days) / log2(1 + MOSS_FULL_DAYS)): a day
           old is a trace, a month old is full cover.
  recent   fan-out and recall from the last requests (mcp/recent_turns.py,
           in memory in the proxy process).
"""
from __future__ import annotations

import glob
import math
import os
import sqlite3
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.abspath(os.path.join(HERE, "..", "index"))
PACKAGES = os.environ.get("YAMADORI_PACKAGES_DIR", os.path.join(INDEX, "packages"))
HINTS_NPZ = os.path.join(INDEX, "hints.npz")
CACHE_S = 60.0
MOSS_FULL_DAYS = 30.0
SPREAD_LO, SPREAD_HI = 3.0, 6.0          # log10 of items: bare .. full

_lock = threading.Lock()
_cache: dict = {}


def _counts(path: str) -> dict | None:
    """chunks and defs in one index, read-only; None if unreadable."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        try:
            out = {}
            for t in ("chunks", "defs"):
                try:
                    out[t] = int(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                except sqlite3.Error:
                    out[t] = 0
            return out
        finally:
            con.close()
    except sqlite3.Error:
        return None


def _age(path: str, now: float) -> float | None:
    try:
        return max(0.0, now - os.path.getmtime(path))
    except OSError:
        return None


def spread_of(items: int | None) -> float | None:
    if not items or items <= 0:
        return None
    return max(0.0, min(1.0, (math.log10(items) - SPREAD_LO) / (SPREAD_HI - SPREAD_LO)))


def moss_of(ages_s: dict) -> float | None:
    xs = [a for a in ages_s.values() if isinstance(a, (int, float))]
    if not xs:
        return None
    full = math.log2(1 + MOSS_FULL_DAYS)
    return sum(min(1.0, math.log2(1 + a / 86400) / full) for a in xs) / len(xs)


def _code_index() -> str | None:
    try:
        import code_search
        return code_search.INDEX_DB
    except Exception:                                            # noqa: BLE001
        return None


def _repo_dbs() -> list[str]:
    try:
        import repos
        return [repos.db_path(r["root"]) for r in repos.known() if r.get("root")]
    except Exception:                                            # noqa: BLE001
        return []


def _last_arm() -> float | None:
    try:
        import jobs
        con = sqlite3.connect(f"file:{os.path.abspath(jobs.DB)}?mode=ro", uri=True, timeout=2)
        try:
            v = con.execute("SELECT MAX(armed_at) FROM skill_versions").fetchone()[0]
        finally:
            con.close()
        return float(v) if v else None
    except Exception:                                            # noqa: BLE001
        return None


def _recall_path() -> str:
    try:
        import skill_select
        return skill_select.recall_path()
    except Exception:                                            # noqa: BLE001
        return (os.environ.get("YAMADORI_RECALL") or "hints").strip().lower()


def measure(now: float | None = None) -> dict:
    """The uncached read: nebari breadth and moss ages."""
    now = time.time() if now is None else now
    pkgs = sorted(glob.glob(os.path.join(PACKAGES, "*.sqlite3")))
    pc = [c for c in (_counts(p) for p in pkgs) if c]
    code = _code_index()
    cc = _counts(code) if code and os.path.exists(code) else None
    rdbs = [p for p in _repo_dbs() if os.path.exists(p) and p != code]
    rc = [c for c in (_counts(p) for p in rdbs) if c]
    items = (sum(c["chunks"] + c["defs"] for c in pc)
             + (cc["chunks"] + cc["defs"] if cc else 0)
             + sum(c["chunks"] + c["defs"] for c in rc))
    pk_ages = [a for a in (_age(p, now) for p in pkgs) if a is not None]
    arm = _last_arm()
    recall = _recall_path()
    ages = {"packages": statistics.median(pk_ages) if pk_ages else None,
            "code": _age(code, now) if code else None,
            "skills_last_arm": (now - arm) if arm else None}
    if recall == "hints":
        ages["recipes"] = _age(HINTS_NPZ, now)
    return {
        "nebari": {"packages": {"indexes": len(pc),
                                "chunks": sum(c["chunks"] for c in pc),
                                "defs": sum(c["defs"] for c in pc)},
                   "code": cc,
                   "repos": {"indexes": len(rc), "chunks": sum(c["chunks"] for c in rc),
                             "defs": sum(c["defs"] for c in rc)},
                   "items": items, "spread": spread_of(items)},
        "moss": {"ages_s": {k: (round(v) if v is not None else None) for k, v in ages.items()},
                 "value": moss_of(ages), "recall": recall,
                 "full_days": MOSS_FULL_DAYS},
        "measured_at": now,
    }


def snapshot(now: float | None = None) -> dict:
    """measure(), at most once per CACHE_S, plus the in-memory recent turns."""
    now = time.time() if now is None else now
    with _lock:
        if not _cache or now - _cache["measured_at"] >= CACHE_S:
            _cache.clear()
            _cache.update(measure(now))
        out = dict(_cache)
    try:
        import recent_turns
        out["recent"] = recent_turns.summary(now)
    except Exception as e:                                       # noqa: BLE001
        out["recent"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out
