#!/usr/bin/env python
"""Tokens: every generation this stack runs, rolled up per local day.

WHY THIS EXISTS

Until 2026-09-24 no record in this repo kept token counts durably. What
existed, checked before this module was written:

  index/corpus.sqlite3    every client turn through :1234 since 2026-09-21
                          10:52, with hops and wall ms -- and no tokens
  index/power_ledger.json energy and cents per day, from 2026-09-24 only
  logs/proxy*.log         the per-request cache line, but the file is
                          truncated on every restart and has no timestamps
  x_yamadori.cache/usage  on the response, to the client; not kept here
  bench/**/rows.jsonl     client-side usage for benchmark runs only
                          (920 rows): a subset of the traffic, so not a
                          backfill source -- "all time" would then mean
                          "all benchmark time" and look like everything
  accounts.usage()        disk bytes, not tokens

So there is nothing to backfill, and the dashboard says so: the history
starts at `first_at` (the first generation this ledger recorded), and the
corpus's client turns before it are counted and labelled as the gap.

WHAT A ROW IS

One row per (local date, account, role), and its counts only ever grow:
record() is an UPSERT that adds. There is no delete and no rewrite path.

  role          main          a client conversation's generations (the
                              answer, every hop of the tool loop, fan-out's
                              candidate A)
                second_brain  deep thinking's hops (shomen, via model.post
                              with the helper share) and fan-out's B and C
                side_call     a client's utility call or compaction
                              (selection.utility_call)
                internal      the stack's own generation through
                              mcp/model.py: summarize_text, skills, the
                              worker, describe_image
                warm          proxy._warm_now's zero-token prefill; its
                              prompt tokens are real GPU work but no hosted
                              API would bill them, so savings exclude them
  prompt_processed  prompt tokens computed now (llama-server timings
                    `prompt_n`)
  prompt_cached     prompt tokens taken from the slot's cache (`cache_n`)
  prompt_unsplit    prompt tokens with no cache split reported at all --
                    counted, never guessed into either column
  completion        generated tokens (usage.completion_tokens; reasoning is
                    inside it, as llama-server counts every decoded token)
  reasoning         usage.completion_tokens_details.reasoning_tokens, only
                    where the server reported it; `reasoning_reported`
                    counts those generations, so an unreported zero is
                    never read as "no reasoning"
  no_usage          generations that came back with no token counts at all

The cache split is slots.cache_record's -- ONE definition, imported.

WHO WRITES IT

record() is the one writer. It is called from exactly two places: the
proxy's upstream reader (proxy._post_events, via record_upstream) for every
client-facing and fan-out generation, and model.post for every generation
through the one door. A generation that raises before it answers is not
recorded (there is no usage to record).

Recording is OFF until enable() is called, and only the long-running
processes call it (mcp/server.py, mcp/tools_api.py, mcp/worker.py main).
Test suites import the proxy and point it at fake upstreams; with recording
on by default every one of them would write fake tokens into the real
ledger -- which is exactly how synthetic turns once reached the corpus.

Days are machine local time, the power ledger's clock (mcp/power.py), so a
token day and an electricity day are the same day. This machine runs
America/Detroit.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("YAMADORI_TOKEN_LEDGER",
                    os.path.abspath(os.path.join(HERE, "..", "index",
                                                 "token_ledger.sqlite3")))

ROLES = ("main", "second_brain", "side_call", "internal", "warm")
# What a hosted API would have billed: every generation, as if it had been
# sent there. `warm` is our own prefill of a cache a hosted API manages
# itself, so it is paid in electricity and priced at nothing.
PRICED_ROLES = ("main", "second_brain", "side_call", "internal")
KINDS = ("generations", "prompt_processed", "prompt_cached", "prompt_unsplit",
         "completion", "reasoning", "reasoning_reported", "no_usage")

# THE WEEK. One constant: datetime.weekday() of the day a week starts, at
# 00:00 machine local time.
WEEK_START_WEEKDAY = 0                      # Monday
WEEK_START_LABEL = "MONDAY 00:00 LOCAL (AMERICA/DETROIT)"
TIMEZONE_LABEL = "machine local time (America/Detroit), the power ledger's clock"
LAST_DAYS = 30                              # "last 30 days": today and 29 before

_lock = threading.Lock()
_enabled = False
_state = {"writes": 0, "errors": 0, "last_error": None}


def enable(path: str | None = None) -> None:
    """Start recording in this process. Only a service's main() calls it."""
    global _enabled, DB
    if path:
        DB = path
    _enabled = True


def disable() -> None:
    global _enabled
    _enabled = False


def enabled() -> bool:
    return _enabled


def status() -> dict:
    return {"enabled": _enabled, "db": DB, **_state}


def _connect(path: str | None = None) -> sqlite3.Connection:
    p = path or DB
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS daily(
            day                TEXT NOT NULL,     -- local date YYYY-MM-DD
            account            TEXT NOT NULL,     -- '' = the stack itself
            role               TEXT NOT NULL,
            generations        INTEGER NOT NULL DEFAULT 0,
            prompt_processed   INTEGER NOT NULL DEFAULT 0,
            prompt_cached      INTEGER NOT NULL DEFAULT 0,
            prompt_unsplit     INTEGER NOT NULL DEFAULT 0,
            completion         INTEGER NOT NULL DEFAULT 0,
            reasoning          INTEGER NOT NULL DEFAULT 0,
            reasoning_reported INTEGER NOT NULL DEFAULT 0,
            no_usage           INTEGER NOT NULL DEFAULT 0,
            first_at           REAL,
            last_at            REAL,
            PRIMARY KEY(day, account, role));
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
    """)
    return con


def _int(x) -> int | None:
    try:
        return None if x is None else int(x)
    except (TypeError, ValueError):
        return None


def counts_of(usage: dict | None, timings: dict | None = None,
              cache: dict | None = None) -> dict:
    """The token kinds of one generation. `cache` is a slots.cache_record()
    already made for it (the proxy has one); otherwise it is made here, from
    the same function, so the split has one definition."""
    u = usage if isinstance(usage, dict) else {}
    t = timings if isinstance(timings, dict) else {}
    if not isinstance(cache, dict):
        import slots
        cache = slots.cache_record(t or None, u or None, None)
    reused, processed = _int(cache.get("reused")), _int(cache.get("processed"))
    prompt = _int(cache.get("prompt"))
    out = {k: 0 for k in KINDS}
    out["generations"] = 1
    if reused is not None and processed is not None:
        out["prompt_cached"], out["prompt_processed"] = reused, processed
    elif prompt is not None:
        out["prompt_unsplit"] = prompt
    completion = _int(u.get("completion_tokens"))
    if completion is None:
        completion = _int(t.get("predicted_n"))
    out["completion"] = completion or 0
    details = u.get("completion_tokens_details")
    r = _int(details.get("reasoning_tokens")) if isinstance(details, dict) else None
    if r is not None:
        out["reasoning"], out["reasoning_reported"] = r, 1
    if prompt is None and completion is None:
        out["no_usage"] = 1
    return out


def _day(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def record(role: str, *, account: str = "", usage: dict | None = None,
           timings: dict | None = None, cache: dict | None = None,
           when: float | None = None) -> bool:
    """Add one generation to its day's row. Never raises: a dropped record
    costs a row, never a request. Returns whether it was written."""
    if not _enabled:
        return False
    try:
        if role not in ROLES:
            role = "internal"
        c = counts_of(usage, timings, cache)
        ts = time.time() if when is None else float(when)
        cols = ", ".join(KINDS)
        marks = ", ".join("?" for _ in KINDS)
        adds = ", ".join(f"{k} = {k} + excluded.{k}" for k in KINDS)
        with _lock:
            con = _connect()
            try:
                con.execute(
                    f"INSERT INTO daily(day, account, role, {cols}, first_at, "
                    f"last_at) VALUES(?, ?, ?, {marks}, ?, ?) "
                    f"ON CONFLICT(day, account, role) DO UPDATE SET {adds}, "
                    f"last_at = MAX(COALESCE(last_at, 0), excluded.last_at)",
                    (_day(ts), (account or "")[:32], role,
                     *[c[k] for k in KINDS], ts, ts))
                con.execute("INSERT OR IGNORE INTO meta(key, value) "
                            "VALUES('first_at', ?)", (repr(ts),))
                con.commit()
            finally:
                con.close()
        _state["writes"] += 1
        return True
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"{type(e).__name__}: {e}"[:200]
        return False


def role_of(payload: dict | None) -> str:
    """Which role a PROXY upstream payload is (proxy._post_events)."""
    p = payload if isinstance(payload, dict) else {}
    util = p.get("_utility")
    slot = p.get("_slot") if isinstance(p.get("_slot"), dict) else {}
    if (isinstance(util, dict) and util.get("utility")) or slot.get("transient"):
        return "side_call"
    if p.get("_role") == "helper":
        return "second_brain"
    return "main"


def record_upstream(payload: dict | None, response: dict | None) -> bool:
    """The proxy's hook: one finished upstream generation. `response` is the
    assembled item `_post_events` yields with "done", after `_cache` is set."""
    if not _enabled:
        return False
    try:
        p = payload if isinstance(payload, dict) else {}
        r = response if isinstance(response, dict) else {}
        slot = p.get("_slot") if isinstance(p.get("_slot"), dict) else {}
        account = slot.get("account") or p.get("_account") or ""
        return record(role_of(p), account=str(account), usage=r.get("usage"),
                      cache=r.get("_cache"))
    except Exception as e:                                       # noqa: BLE001
        _state["errors"] += 1
        _state["last_error"] = f"{type(e).__name__}: {e}"[:200]
        return False


# ---------------------------------------------------------------------------
# Reading: windows.
# ---------------------------------------------------------------------------

def windows(today: _dt.date) -> list[dict]:
    """ALL TIME, LAST 30 DAYS (today and the 29 days before), THIS WEEK (from
    the most recent WEEK_START_WEEKDAY). `start` is an inclusive ISO date, or
    None for all time."""
    back = (today.weekday() - WEEK_START_WEEKDAY) % 7
    week = today - _dt.timedelta(days=back)
    d30 = today - _dt.timedelta(days=LAST_DAYS - 1)
    return [{"key": "all", "label": "ALL TIME", "start": None},
            {"key": "d30", "label": f"LAST {LAST_DAYS} DAYS",
             "start": d30.isoformat()},
            {"key": "week", "label": "THIS WEEK", "start": week.isoformat(),
             "starts": WEEK_START_LABEL}]


def _blank() -> dict:
    return {k: 0 for k in KINDS}


def totals(start: str | None = None, path: str | None = None) -> dict:
    """{by_role: {role: kinds}, total: kinds, priced: kinds} for days >= start."""
    by_role = {r: _blank() for r in ROLES}
    p = path or DB
    if os.path.exists(p):
        con = _connect(p)
        try:
            sums = ", ".join(f"SUM({k})" for k in KINDS)
            q = f"SELECT role, {sums} FROM daily"
            args: tuple = ()
            if start:
                q += " WHERE day >= ?"
                args = (start,)
            for row in con.execute(q + " GROUP BY role", args):
                role = row[0] if row[0] in by_role else "internal"
                for k, v in zip(KINDS, row[1:]):
                    by_role[role][k] += int(v or 0)
        finally:
            con.close()

    def add(roles):
        t = _blank()
        for r in roles:
            for k in KINDS:
                t[k] += by_role[r][k]
        return t
    return {"by_role": by_role, "total": add(ROLES), "priced": add(PRICED_ROLES)}


def history(path: str | None = None) -> dict:
    """When recording began, and the days and accounts it has seen."""
    p = path or DB
    out = {"first_at": None, "first_day": None, "last_at": None, "days": 0,
           "accounts": 0}
    if not os.path.exists(p):
        return out
    con = _connect(p)
    try:
        row = con.execute("SELECT value FROM meta WHERE key='first_at'").fetchone()
        if row:
            out["first_at"] = float(row[0])
            out["first_day"] = _day(out["first_at"])
        d, a, last = con.execute(
            "SELECT COUNT(DISTINCT day), COUNT(DISTINCT account), MAX(last_at) "
            "FROM daily").fetchone()
        out.update(days=int(d or 0), accounts=int(a or 0), last_at=last)
    finally:
        con.close()
    return out


_gap_cache: dict = {}


def gap(first_at: float | None, corpus_db: str | None = None) -> dict:
    """The client traffic before recording began, from the corpus: how many
    turns, from when. Tokens for them were never recorded anywhere."""
    if corpus_db is None:
        import corpus
        corpus_db = corpus.CORPUS_DB
    key = (first_at, corpus_db)
    if first_at is not None and key in _gap_cache:
        return _gap_cache[key]
    out = {"source": "index/corpus.sqlite3 (client turns through :1234)",
           "turns": None, "from": None, "until": first_at}
    try:
        if os.path.exists(corpus_db):
            con = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True, timeout=5)
            try:
                q = "SELECT COUNT(*), MIN(ts) FROM events WHERE kind='turn'"
                args: tuple = ()
                if first_at is not None:
                    q += " AND ts < ?"
                    args = (first_at,)
                n, lo = con.execute(q, args).fetchone()
            finally:
                con.close()
            out.update(turns=int(n or 0), **{"from": lo})
    except sqlite3.Error as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    if first_at is not None and "error" not in out:
        # Rows before first_at are in the past: the count cannot change.
        _gap_cache[key] = out
    return out


if __name__ == "__main__":
    h = history()
    print(json.dumps({"status": status(), "history": h,
                      "all_time": totals()["total"],
                      "gap": gap(h["first_at"])}, indent=1))
