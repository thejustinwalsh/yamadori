#!/usr/bin/env python
"""What the server has learned about one conversation, and keeps.

`nebari` is the flare of surface roots a bonsai builds up over years. It is the
part that is visible, that accumulates, and that tells you what the tree has
been doing. This is the same thing for a session.

WHY A SERVER NEEDS THIS AT ALL

Yamadori is remote. It cannot look at the caller's disk, it does not know their
project, and every request arrives as text with no memory attached. The only
things it can know are what it reads out of the conversation and what it thinks
to ask. Both are expensive to obtain and cheap to keep, which is exactly the
shape of a thing worth caching.

Without it the service re-derives the same facts every turn and, worse, asks
the same question every turn. A caller who says "three 0.185.1" once and is
asked again two messages later has learned that the service does not listen.

THE SESSION KEY

Remote clients do not reliably send a session id, so one is derived from the
opening of the conversation -- the system prompt and the first user message,
which are fixed for the life of a session and differ between sessions. It is a
hash, so no prompt text is stored to compute it.

WHAT IS AND IS NOT KEPT

Kept: package names, versions, and which questions have been asked. These are
facts about the caller's environment, they are small, and they are what the
next turn needs.

Not kept: the caller's code. It is theirs, the index already holds the public
libraries it refers to, and a service that quietly accumulates source it was
only shown in passing is a service nobody should run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("YAMADORI_NEBARI_DB",
                    os.path.join(HERE, "..", "index", "nebari.sqlite3"))
# Long enough to span a working session, short enough that a stale pin cannot
# follow someone into next week's project.
TTL_SECONDS = int(os.environ.get("YAMADORI_NEBARI_TTL", str(36 * 3600)))

_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB)), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS sessions(
        key TEXT PRIMARY KEY,
        created REAL NOT NULL,
        seen REAL NOT NULL,
        state TEXT NOT NULL)""")
    return con


def key_of(messages: list[dict], account: str = "") -> str:
    """A stable id for this conversation, derived rather than requested.

    Scoped to the ACCOUNT. Two callers whose first two messages match -- a
    common harness system prompt and the same opening line -- are different
    conversations and must never share a work log or a package list.
    """
    h = hashlib.sha256()
    h.update(("account\x00" + (account or "") + "\x00").encode("utf-8"))
    for m in messages[:2]:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        h.update((m.get("role", "") + "\x00" + (c or "")[:4000]).encode(
            "utf-8", "replace"))
    return h.hexdigest()[:20]


def load(key: str) -> dict:
    try:
        with _lock:
            con = _db()
            row = con.execute(
                "SELECT state, seen FROM sessions WHERE key=?", (key,)).fetchone()
            con.close()
    except sqlite3.Error:
        return {}
    if not row or time.time() - row[1] > TTL_SECONDS:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


def save(key: str, state: dict) -> None:
    """Never let memory break a request. A dropped write costs one re-derivation."""
    try:
        blob = json.dumps(state)[:200000]
        with _lock:
            con = _db()
            now = time.time()
            con.execute(
                "INSERT INTO sessions(key, created, seen, state) VALUES(?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET seen=?, state=?",
                (key, now, now, blob, now, blob))
            con.commit()
            con.close()
    except Exception:                                            # noqa: BLE001
        pass


def observe(key: str, found: dict) -> dict:
    """Fold this turn's discoveries into what the session already knew.

    Accumulating matters more than it looks. A caller pastes one file per
    message, so any single turn sees a slice of the dependency set and the
    union over the session is the actual answer. Counts add up for the same
    reason, which makes the ranking settle instead of lurching with each paste.
    """
    state = load(key)
    counts = state.get("counts", {})
    for pkg, n in (found.get("counts") or {}).items():
        counts[pkg] = counts.get(pkg, 0) + n

    # A version stated later wins. A caller upgrading mid-session is a real
    # thing, and the newer statement is the better evidence.
    vers = state.get("versions", {})
    vers.update(found.get("versions") or {})

    state.update({
        "counts": counts,
        "versions": vers,
        "packages": sorted(counts, key=lambda p: (-counts[p], p)),
        "has_project": bool(state.get("has_project")) or bool(found.get("has_project")),
        "turns": int(state.get("turns", 0)) + 1,
    })
    save(key, state)
    return state


def pin(key: str, package: str, version: str) -> dict:
    """Record a version the caller supplied, so it is never asked for twice."""
    state = load(key)
    state.setdefault("versions", {})[package] = version
    state.setdefault("asked", {})[package] = "answered"
    save(key, state)
    return state


def mark_asked(key: str, package: str) -> None:
    state = load(key)
    state.setdefault("asked", {}).setdefault(package, "pending")
    save(key, state)


def wants_version(state: dict, package: str) -> bool:
    """Is a version worth one question here?

    Only when it is unknown, the package matters to this session, and it has
    not already been asked for. The bar is deliberately high: a question costs
    a turn of the caller's attention, and most answers do not change with the
    version.
    """
    if (state.get("versions") or {}).get(package):
        return False
    if (state.get("asked") or {}).get(package):
        return False
    return (state.get("counts") or {}).get(package, 0) >= 2


def forget(key: str) -> None:
    try:
        with _lock:
            con = _db()
            con.execute("DELETE FROM sessions WHERE key=?", (key,))
            con.commit()
            con.close()
    except sqlite3.Error:
        pass


def stats() -> dict:
    try:
        con = _db()
        n, oldest = con.execute(
            "SELECT COUNT(*), MIN(created) FROM sessions").fetchone()
        con.close()
        return {"sessions": n,
                "oldest_hours": round((time.time() - oldest) / 3600, 1) if oldest else 0}
    except sqlite3.Error as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print(json.dumps(stats(), indent=2))
