#!/usr/bin/env python
"""One index per repository, created the first time a repository is seen.

WHY NOT ONE BIG INDEX

Measured on the combined index: a query about `koota` scored against all
60,224 chunks when 1,351 of them were relevant. Everything else is noise
competing for the same top-5 slots, and the BM25 build cost 7.9s cold because
it covered four projects at once.

Per repository: koota searches 1,351 chunks, the build is proportionally
cheaper, and an index only has to be built once per project rather than
rebuilt whenever any project changes.

WHY IT GROWS ON ITS OWN

No one registers a repository. The proxy works out which one a conversation is
about, asks for its index, and if there is none it starts building in the
background and answers from what is available meanwhile. Work in a new project
and it indexes itself while you use it. The corpus is a side effect of working,
not a setup step.

A registry of everything ever seen also makes cross-repository search possible
later: when the current project has no answer, its siblings are already
indexed and cost nothing to consult.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sqlite3
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("YAMADORI_INDEX_DIR", os.path.join(HERE, "..", "index", "repos"))
REGISTRY = os.path.join(STORE, "registry.json")

# Indexing is IO- and GPU-bound and must never block an answer, so exactly one
# runs at a time and the caller is told to proceed without it.
_building: dict[str, float] = {}
_lock = threading.Lock()


def slug(root: str) -> str:
    """Stable, collision-resistant, and still readable in a directory listing."""
    root = os.path.abspath(root).replace("\\", "/").rstrip("/")
    name = os.path.basename(root) or "root"
    h = hashlib.sha1(root.lower().encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)[:40]
    return f"{safe}-{h}"


def db_path(root: str) -> str:
    os.makedirs(STORE, exist_ok=True)
    return os.path.join(STORE, f"{slug(root)}.sqlite3")


def _registry() -> dict:
    try:
        with open(REGISTRY, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_registry(reg: dict) -> None:
    os.makedirs(STORE, exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, REGISTRY)


def note_seen(root: str) -> None:
    reg = _registry()
    entry = reg.setdefault(slug(root), {"root": root, "first_seen": time.time()})
    entry["last_seen"] = time.time()
    entry["uses"] = entry.get("uses", 0) + 1
    _save_registry(reg)


def known() -> list[dict]:
    return sorted(_registry().values(), key=lambda e: -e.get("last_seen", 0))


def chunk_count(root: str) -> int:
    p = db_path(root)
    if not os.path.exists(p):
        return 0
    try:
        con = sqlite3.connect(p)
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        return n
    except sqlite3.Error:
        return 0


def head_commit(root: str) -> str:
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()[:12] if out.returncode == 0 else ""
    except Exception:                                            # noqa: BLE001
        return ""


def is_stale(root: str) -> bool:
    """An index built at a different commit is stale but still usable.

    Staleness is never a reason to refuse an answer -- a slightly old index
    beats no index, and the alternative is blocking on a rebuild.
    """
    reg = _registry().get(slug(root), {})
    built = reg.get("built_at_commit")
    now = head_commit(root)
    return bool(built and now and built != now)


def ensure(root: str, background: bool = True) -> dict:
    """Make sure `root` has an index. Never blocks.

    Returns what the caller should tell the model: whether an index exists,
    how big it is, and whether one is being built right now.
    """
    note_seen(root)
    n = chunk_count(root)
    with _lock:
        building = root in _building
        if n == 0 and not building and background:
            _building[root] = time.time()
            threading.Thread(target=_build, args=(root,), daemon=True).start()
            building = True
    return {"root": root, "chunks": n, "building": building,
            "stale": is_stale(root) if n else False}


def _build(root: str) -> None:
    try:
        env = dict(os.environ)
        env["CODE_INDEX_DB"] = db_path(root)
        script = os.path.join(HERE, "..", "scripts", "index_code.py")
        subprocess.run([sys.executable, script, root], env=env,
                       capture_output=True, text=True, timeout=7200)
        reg = _registry()
        e = reg.setdefault(slug(root), {"root": root})
        e["built_at_commit"] = head_commit(root)
        e["built_at"] = time.time()
        e["chunks"] = chunk_count(root)
        _save_registry(reg)
    except Exception:                                            # noqa: BLE001
        pass
    finally:
        with _lock:
            _building.pop(root, None)


def status_line(info: dict) -> str:
    """One line for the model. It should never have to reason about indexing."""
    name = os.path.basename(info["root"])
    if info["chunks"] and not info["building"]:
        s = f"Indexed {name}: {info['chunks']} chunks."
        return s + (" Index predates the current commit; recently changed code "
                    "may be missing." if info["stale"] else "")
    if info["building"] and info["chunks"]:
        return (f"Indexed {name}: {info['chunks']} chunks, and an update is "
                f"building now.")
    if info["building"]:
        return (f"{name} is being indexed for the first time right now. Search "
                f"will be thin until it finishes -- prefer find_by_pattern and "
                f"read_file_range for this turn.")
    return (f"{name} has no index and none is building. Use find_by_pattern "
            f"and read_file_range.")


if __name__ == "__main__":
    for e in known():
        print(f"  {e['root']}  chunks={e.get('chunks', '?')}  uses={e.get('uses', 0)}")
