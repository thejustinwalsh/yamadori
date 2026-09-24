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
import re
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


# The schema is created ONCE per database path in this process (pre-deploy
# review, 2026-09-24): _db() ran a PRAGMA and five DDL statements on every
# connection, and the ledger opened one per cache miss, under the global
# lock, on the request path. WAL is a property of the database file, so it
# too is set once.
_SCHEMA_DONE: set[str] = set()


def _db(shared: bool = False) -> sqlite3.Connection:
    """A connection to DB; `shared`: usable from any thread (the reused
    ledger reader, always used under _lock)."""
    path = os.path.abspath(DB)
    done = path in _SCHEMA_DONE and os.path.exists(path)
    if not done:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30, check_same_thread=not shared)
    if done:
        return con
    # WAL, as mcp/jobs.py: the ledger writes on every turn and a reader must
    # never wait on a writer.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS sessions(
        key TEXT PRIMARY KEY,
        created REAL NOT NULL,
        seen REAL NOT NULL,
        state TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS additions(
        account TEXT NOT NULL,
        session TEXT NOT NULL,
        msg_key TEXT NOT NULL,
        kind TEXT NOT NULL,
        text TEXT NOT NULL,
        ts REAL NOT NULL,
        PRIMARY KEY (account, session, msg_key, kind))""")
    con.execute("CREATE INDEX IF NOT EXISTS additions_session "
                "ON additions(account, session)")
    con.execute("CREATE INDEX IF NOT EXISTS additions_key "
                "ON additions(account, msg_key, kind)")
    con.execute("""CREATE TABLE IF NOT EXISTS ledger_sessions(
        account TEXT NOT NULL,
        session TEXT NOT NULL,
        last_seen REAL NOT NULL,
        bytes INTEGER NOT NULL,
        PRIMARY KEY (account, session))""")
    con.commit()
    _SCHEMA_DONE.add(path)
    return con


# An explicit session token from `X-Yamadori-Session`. Flat text, so a
# pattern is the right reader: at most 64 characters of [A-Za-z0-9_-].
_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def session_token(value) -> str:
    """The header's token if it is well formed, else "" (header ignored)."""
    if not isinstance(value, str):
        return ""
    v = value.strip()
    return v if _TOKEN.fullmatch(v) else ""


def key_of(messages: list[dict], account: str = "", session: str = "") -> str:
    """A stable id for this conversation, derived rather than requested.

    Scoped to the ACCOUNT. Two callers whose first two messages match -- a
    common harness system prompt and the same opening line -- are different
    conversations and must never share a work log or a package list.

    `session` is an explicit token (header `X-Yamadori-Session`, validated by
    `session_token`). Benchmark rows often open with byte-identical messages
    and were merged into one session -- one tool-offer history, one package
    list. A token keeps them apart. Without one the key is exactly what it
    was, so no existing session changes.
    """
    h = hashlib.sha256()
    h.update(("account\x00" + (account or "") + "\x00").encode("utf-8"))
    if session:
        h.update(("session\x00" + session + "\x00").encode("utf-8"))
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


# ================================================================ LEDGER ===
#
# WHAT THE PROXY ADDED TO A CONVERSATION, SO EVERY REQUEST RE-RENDERS BYTE FOR
# BYTE (docs/SELF-IMPROVEMENT-PLAN.md Phase 0.5, operator 2026-09-24).
#
# A pinned llama-server slot reuses its cached prompt only when the next
# request EXTENDS it (STEP 0 probes: an edit anywhere before the end rolls the
# slot back to a context checkpoint, 448 tokens in, however late the edit
# was). Harnesses strip what we add -- the model's reasoning, a skill block on
# a user turn, a fold-back -- and the resent history then renders
# differently from what the slot holds. The proxy (proxy.ledger_restore)
# re-adds each addition from here on every request.
#
# ONE TABLE, `additions`, keyed (account, session, msg_key, kind). What is
# stored is only what the proxy or the MODEL produced -- injections (skills,
# retrieval, the work log), a turn's delivered content when it differs from
# what the client echoes, image-tool hops (their reasoning emptied), concept
# seeds. NOT reasoning: since 2026-09-24 past reasoning passes through as the
# client sends it, and proxy.ledger_record_turn writes none. The
# caller's own messages are never stored: `msg_key` is a hash of them, and
# the client resends them every turn anyway (the module docstring: "Not
# kept: the caller's code"). Every query names the account; forgetting an
# account is one DELETE.
#
# YAMADORI_LEDGER_PERSIST_REASONING (default on) gates persisting the kinds
# that can quote the caller -- "reasoning" (no longer written: see above)
# and "hops" (image-tool hops, reasoning emptied) -- so a harness can resume
# a conversation days later with its hidden hops rendered as before.
#
# MEMORY IS THE HOT LAYER: a read-through LRU over the table. A miss reads
# sqlite; every write goes to both.
#
# EVICTION IS BY SIZE, NOT BY THE 36 h TTL above (that TTL is for pins and
# versions). `ledger_sessions` keeps (account, session, last_seen, bytes);
# past an account's byte cap, whole sessions go in least-recently-seen
# order, then the same against a global cap, and anything older than the
# max age goes regardless (a privacy bound). Pruning runs in a background
# thread at most once a minute, never on the request path. The three caps
# are CHOICES (operator, 2026-09-24), not measurements; mcp/test_ledger.py
# prints the bytes a turn actually adds so they can be set from data.
LEDGER_PERSIST_REASONING = os.environ.get(
    "YAMADORI_LEDGER_PERSIST_REASONING", "1") == "1"
LEDGER_ACCOUNT_BYTES = int(float(os.environ.get(
    "YAMADORI_LEDGER_ACCOUNT_MB", "256")) * 1024 * 1024)
LEDGER_TOTAL_BYTES = int(float(os.environ.get(
    "YAMADORI_LEDGER_TOTAL_MB", "2048")) * 1024 * 1024)
LEDGER_MAX_AGE = float(os.environ.get("YAMADORI_LEDGER_MAX_DAYS", "30")) * 86400
# The in-process LRU's size. A choice: the hot set is the live conversations'
# last few turns, far below this.
LEDGER_MEM_BYTES = int(float(os.environ.get("YAMADORI_LEDGER_MEM_MB", "64"))
                       * 1024 * 1024)
LEDGER_PRUNE_EVERY = 60.0

import collections as _collections  # noqa: E402

_mem: "_collections.OrderedDict[tuple, str]" = _collections.OrderedDict()
_mem_bytes = 0
_mem_lock = threading.Lock()
_last_prune = 0.0
_MISS = object()
# KNOWN MISSES (pre-deploy review, 2026-09-24): a key the table did not
# have, remembered so the next lookup of it costs nothing. Every write
# clears its own key; only this process writes the ledger (the proxy). A
# bound, not a measurement: the keys are ~100 bytes.
_misses: "_collections.OrderedDict[tuple, bool]" = _collections.OrderedDict()
LEDGER_MISSES_MAX = 100_000
# One connection for ledger reads, reused under _lock (a new connection
# per miss was the cost above), reopened when DB changes (tests).
_reader: list = [None, None]          # [path, connection]


def _mem_put(k: tuple, text: str) -> None:
    global _mem_bytes
    with _mem_lock:
        _misses.pop(k, None)
        old = _mem.pop(k, None)
        if old is not None:
            _mem_bytes -= len(old)
        _mem[k] = text
        _mem_bytes += len(text)
        while _mem_bytes > LEDGER_MEM_BYTES and _mem:
            _, v = _mem.popitem(last=False)
            _mem_bytes -= len(v)


def _mem_get(k: tuple):
    with _mem_lock:
        v = _mem.get(k, _MISS)
        if v is not _MISS:
            _mem.move_to_end(k)
        return v


def ledger_reset() -> None:
    """Forget the in-process layer (tests; a proxy restart does this)."""
    global _mem_bytes
    with _mem_lock:
        _mem.clear()
        _misses.clear()
        _mem_bytes = 0


def _miss_known(k: tuple) -> bool:
    with _mem_lock:
        if k in _misses:
            _misses.move_to_end(k)
            return True
        return False


def _miss_note(k: tuple) -> None:
    with _mem_lock:
        if k in _mem:
            return
        _misses[k] = True
        while len(_misses) > LEDGER_MISSES_MAX:
            _misses.popitem(last=False)


def _read_con() -> sqlite3.Connection:
    """The reused ledger-read connection. Call under _lock."""
    path = os.path.abspath(DB)
    if _reader[0] != path or _reader[1] is None:
        if _reader[1] is not None:
            try:
                _reader[1].close()
            except sqlite3.Error:
                pass
        _reader[0], _reader[1] = path, _db(shared=True)
    return _reader[1]


def _persisted(kind: str) -> bool:
    """Reasoning and image-tool hops can quote the caller; they follow the
    YAMADORI_LEDGER_PERSIST_REASONING flag. Everything else is persisted."""
    return LEDGER_PERSIST_REASONING or kind not in ("reasoning", "hops")


def ledger_put(account: str, session: str, msg_key: str, kind: str,
               text: str) -> None:
    """Record one addition. Never raises: a dropped write costs one cache
    miss, never a request."""
    if not msg_key or not kind or not isinstance(text, str):
        return
    account = account or ""
    _mem_put((account, msg_key, kind), text)
    if not _persisted(kind):
        return
    try:
        now = time.time()
        with _lock:
            con = _db()
            con.execute(
                "INSERT INTO additions(account, session, msg_key, kind, text, "
                "ts) VALUES(?,?,?,?,?,?) ON CONFLICT(account, session, msg_key,"
                " kind) DO UPDATE SET text=excluded.text, ts=excluded.ts",
                (account, session or "", msg_key, kind, text, now))
            con.execute(
                "INSERT INTO ledger_sessions(account, session, last_seen, "
                "bytes) VALUES(?,?,?,(SELECT COALESCE(SUM(LENGTH(text)),0) "
                "FROM additions WHERE account=? AND session=?)) "
                "ON CONFLICT(account, session) DO UPDATE SET "
                "last_seen=excluded.last_seen, bytes=excluded.bytes",
                (account, session or "", now, account, session or ""))
            con.commit()
            con.close()
    except Exception:                                            # noqa: BLE001
        return
    _maybe_prune()


_claim_lock = threading.Lock()


def ledger_claim(account: str, session: str, msg_key: str, kind: str,
                 text: str) -> str:
    """Record a DECISION unless a non-empty one is already recorded for this
    key, and return the decision that stands (pre-deploy review,
    2026-09-24): a duplicate in-flight request that decided "" after the
    other had recorded the library help overwrote it with "". Only this
    process writes the ledger, so a process lock is enough. The caller sends
    what this returns, so the prompt matches what every replay sends."""
    with _claim_lock:
        have = ledger_get(account, msg_key, kind)
        if have:
            return have
        ledger_put(account, session, msg_key, kind, text)
        return text


def ledger_get(account: str, msg_key: str, kind: str) -> str | None:
    """The addition recorded for this message, from memory, else the table
    (the most recent across this account's sessions: a message key is a
    hash of the conversation up to that message, so it names one turn)."""
    if not msg_key:
        return None
    k = (account or "", msg_key, kind)
    v = _mem_get(k)
    if v is not _MISS:
        return v
    if not _persisted(kind) or _miss_known(k):
        return None
    try:
        with _lock:
            row = _read_con().execute(
                "SELECT text FROM additions WHERE account=? AND msg_key=? AND "
                "kind=? ORDER BY ts DESC LIMIT 1",
                (account or "", msg_key, kind)).fetchone()
    except Exception:                                            # noqa: BLE001
        with _lock:
            _reader[1] = None           # reopened on the next miss
        return None
    if row is None:
        _miss_note(k)
        return None
    _mem_put(k, row[0])
    return row[0]


def ledger_touch(account: str, session: str) -> None:
    """A request of this session arrived: it is recently seen."""
    try:
        with _lock:
            con = _db()
            con.execute("UPDATE ledger_sessions SET last_seen=? WHERE "
                        "account=? AND session=?",
                        (time.time(), account or "", session or ""))
            con.commit()
            con.close()
    except Exception:                                            # noqa: BLE001
        pass


def _drop_session(con, account: str, session: str) -> None:
    con.execute("DELETE FROM additions WHERE account=? AND session=?",
                (account, session))
    con.execute("DELETE FROM ledger_sessions WHERE account=? AND session=?",
                (account, session))


def ledger_prune(now: float | None = None) -> dict:
    """Enforce the max age, each account's byte cap, then the global cap,
    dropping whole sessions least recently seen first. Returns counts."""
    now = time.time() if now is None else now
    out = {"aged": 0, "account_cap": 0, "total_cap": 0}
    try:
        with _lock:
            con = _db()
            old = con.execute("SELECT account, session FROM ledger_sessions "
                              "WHERE last_seen < ?",
                              (now - LEDGER_MAX_AGE,)).fetchall()
            for a, s in old:
                _drop_session(con, a, s)
                out["aged"] += 1
            for (a,) in con.execute("SELECT DISTINCT account FROM "
                                    "ledger_sessions").fetchall():
                rows = con.execute(
                    "SELECT session, bytes FROM ledger_sessions WHERE "
                    "account=? ORDER BY last_seen ASC", (a,)).fetchall()
                total = sum(b for _s, b in rows)
                for s, b in rows:
                    if total <= LEDGER_ACCOUNT_BYTES:
                        break
                    _drop_session(con, a, s)
                    total -= b
                    out["account_cap"] += 1
            rows = con.execute("SELECT account, session, bytes FROM "
                               "ledger_sessions ORDER BY last_seen ASC"
                               ).fetchall()
            total = sum(r[2] for r in rows)
            for a, s, b in rows:
                if total <= LEDGER_TOTAL_BYTES:
                    break
                _drop_session(con, a, s)
                total -= b
                out["total_cap"] += 1
            con.commit()
            con.close()
    except Exception as e:                                       # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    if any(out.get(k) for k in ("aged", "account_cap", "total_cap")):
        # The memory layer may hold what was dropped; the next miss re-reads.
        ledger_reset()
    return out


def _maybe_prune() -> None:
    """At most once a minute, in a daemon thread: never on the request path."""
    global _last_prune
    now = time.time()
    if now - _last_prune < LEDGER_PRUNE_EVERY:
        return
    _last_prune = now
    threading.Thread(target=ledger_prune, daemon=True).start()


def ledger_forget_account(account: str) -> None:
    """Everything the ledger holds for one account: one DELETE per table."""
    try:
        with _lock:
            con = _db()
            con.execute("DELETE FROM additions WHERE account=?", (account,))
            con.execute("DELETE FROM ledger_sessions WHERE account=?",
                        (account,))
            con.commit()
            con.close()
    except Exception:                                            # noqa: BLE001
        pass
    ledger_reset()


def ledger_stats() -> dict:
    try:
        con = _db()
        n, b = con.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(text)),0) "
                           "FROM additions").fetchone()
        s = con.execute("SELECT COUNT(*) FROM ledger_sessions").fetchone()[0]
        con.close()
        return {"additions": n, "bytes": b, "sessions": s,
                "memory_bytes": _mem_bytes}
    except sqlite3.Error as e:
        return {"error": str(e)}


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
