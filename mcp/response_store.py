#!/usr/bin/env python
"""STORED RESPONSES: what `previous_response_id`, GET and DELETE
/v1/responses/{id} stand on.

THE DECISION (operator, 2026-10-06, AskUserQuestion, chosen "Store, local,
capped"): "Save each response's full input and output under its id, per
account, on this machine only. Same size and age limits as the ledger (256 MB
per account, 2 GB total, 30 days). previous_response_id rebuilds the
conversation from it, and GET and DELETE /v1/responses/{id} work like OpenAI's.
Applies when the client asks to store (OpenAI's default). This ends the rule
that client messages are only hashed, for Responses requests."

WHY: OpenAI's `store` defaults to true and a client may send only the new
input plus `previous_response_id` (the SDK's conversation-state pattern:
developers.openai.com/api/docs/guides/conversation-state). Answering that with
400 was a production trap. This is the ONE place the proxy keeps a caller's
own messages, and only for a request that did not say `store: false`.

WHAT IS KEPT, per response id (table `responses`):

  account        every query names it; another account's id is "not found",
                 never "forbidden" (existence is not leaked)
  id, created    resp_<32 hex>, unix seconds
  prev_id        the response it chained from (previous_response_id), or NULL
  root_id        the first response of its chain: the unit of eviction
  model, status  as returned
  input_json     THIS request's own input items (a string input becomes one
                 user message item), each with an id (the client's, else one
                 derived from the response id and position)
  output_json    the response's output items; an image_generation_call's
                 base64 `result` is NOT kept (it sits in the media store, whose
                 own rules apply) -- GET puts it back while the media is there
  response_json  the Response object less `output` and `x_yamadori`
                 (instructions, tools, usage, the echoed parameters)
  bytes          the three texts' UTF-8 size: what the caps count

and `chains` (account, root_id, last_seen, bytes). The conversation a
`previous_response_id` names is rebuilt by walking prev_id to the root and
concatenating, root first, each response's input items then its output items
-- exactly the items a client that resent its whole history would send, so the
chat messages, the ledger's keys and the slot's prompt come out identical
(mcp/test_responses_api.py compares the messages and the served template's
prompts). `instructions` are NOT part of it: each request carries its own
(OpenAI: "When using along with previous_response_id, the instructions from a
previous response will not be carried over to the next response."), and so are
the tools. Nothing here reads a key or a header; no key is stored.

EVICTION, the ledger's (mcp/nebari.py LEDGER; operator, 2026-09-24, applied to
stored responses by the 2026-10-06 decision): by SIZE, whole CHAINS, least
recently used first (a chain's last_seen moves when a response is added to it,
read, or continued) -- past `YAMADORI_RESPONSES_ACCOUNT_MB` (default the
ledger's 256) per account, then `YAMADORI_RESPONSES_TOTAL_MB` (2048) in all --
and any chain unused for `YAMADORI_RESPONSES_MAX_DAYS` (30) goes regardless
(OpenAI keeps responses 30 days by default). Pruning runs in a background
thread at most once a minute, never on the request path. A chain is evicted
whole because a response whose ancestor is gone cannot be continued; DELETE
removes ONE response, and a continuation that walks through it is "not found".
A single response larger than the per-account cap is not stored (its `store`
is reported false).

PRIVACY. Local sqlite under index/ (gitignored), this machine only. The corpus
and the learning paths (corpus.py, skill_learn.py) never import this module;
nothing here is read by them. `YAMADORI_RESPONSE_STORE=0` turns the store off:
every request is then answered `store: false` and `previous_response_id` is a
400 (the pre-2026-10-06 behaviour).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time

import nebari

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("YAMADORI_RESPONSES_DB",
                    os.path.join(HERE, "..", "index", "responses.sqlite3"))


def _mb(var: str, default: int) -> int:
    v = os.environ.get(var)
    return int(float(v) * 1024 * 1024) if v else default


# The ledger's numbers (nebari.LEDGER_*), one set of choices: separately
# metered, each store has its own 256 MB / 2 GB / 30 days.
ACCOUNT_BYTES = _mb("YAMADORI_RESPONSES_ACCOUNT_MB", nebari.LEDGER_ACCOUNT_BYTES)
TOTAL_BYTES = _mb("YAMADORI_RESPONSES_TOTAL_MB", nebari.LEDGER_TOTAL_BYTES)
MAX_AGE = (float(os.environ["YAMADORI_RESPONSES_MAX_DAYS"]) * 86400
           if os.environ.get("YAMADORI_RESPONSES_MAX_DAYS")
           else nebari.LEDGER_MAX_AGE)
PRUNE_EVERY = nebari.LEDGER_PRUNE_EVERY

_lock = threading.Lock()
_SCHEMA_DONE: set[str] = set()
_last_prune = 0.0


def enabled() -> bool:
    return os.environ.get("YAMADORI_RESPONSE_STORE", "1") != "0"


class NotFound(LookupError):
    """The id names nothing for this account (never stored, `store: false`,
    deleted, evicted, expired, or another account's)."""

    def __init__(self, rid: str, broken: bool = False):
        super().__init__(rid)
        self.rid = rid
        self.broken = broken        # found, but an ancestor in its chain is gone


def _db() -> sqlite3.Connection:
    path = os.path.abspath(DB)
    done = path in _SCHEMA_DONE and os.path.exists(path)
    if not done:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    if done:
        return con
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS responses(
        account TEXT NOT NULL,
        id TEXT NOT NULL,
        created REAL NOT NULL,
        prev_id TEXT,
        root_id TEXT NOT NULL,
        model TEXT,
        status TEXT,
        input_json TEXT NOT NULL,
        output_json TEXT NOT NULL,
        response_json TEXT NOT NULL,
        bytes INTEGER NOT NULL,
        PRIMARY KEY (account, id))""")
    con.execute("CREATE INDEX IF NOT EXISTS responses_root "
                "ON responses(account, root_id)")
    con.execute("""CREATE TABLE IF NOT EXISTS chains(
        account TEXT NOT NULL,
        root_id TEXT NOT NULL,
        last_seen REAL NOT NULL,
        bytes INTEGER NOT NULL,
        PRIMARY KEY (account, root_id))""")
    con.commit()
    _SCHEMA_DONE.add(path)
    return con


def reset() -> None:
    """Forget that the schema was made (tests that change DB)."""
    _SCHEMA_DONE.clear()


# ------------------------------------------------------------------ items ----

_ID_PREFIX = {"message": "msg", "function_call": "fc", "custom_tool_call":
              "ctc", "function_call_output": "fco",
              "custom_tool_call_output": "ctco", "reasoning": "rs"}


def with_ids(items: list, rid: str) -> list:
    """Copies of the input items, each carrying an `id`: the client's own,
    else one derived from the response id and its position (stable, so the
    input_items listing is paginable)."""
    out = []
    for i, it in enumerate(items):
        if isinstance(it, dict):
            it = dict(it)
            if not it.get("id"):
                kind = it.get("type") or ("message" if "role" in it else "")
                h = hashlib.sha1(f"{rid}:{i}".encode()).hexdigest()[:32]
                it["id"] = f"{_ID_PREFIX.get(kind, 'item')}_{h}"
        out.append(it)
    return out


def _dump(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


# -------------------------------------------------------------------- put ----

def put(account: str, rid: str, *, created: float, prev_id: str | None,
        model: str, status: str, input_items: list, output_items: list,
        response: dict, now: float | None = None) -> dict:
    """Store one response. Returns {stored, id, chained_from, items, bytes,
    chain}; never raises -- a store that cannot write costs the caller its
    continuation, never the request (`stored: False` with the reason)."""
    now = time.time() if now is None else now
    out = {"stored": False, "id": rid, "chained_from": prev_id}
    try:
        inp = with_ids(input_items, rid)
        t_in, t_out = _dump(inp), _dump(output_items)
        t_resp = _dump({k: v for k, v in response.items()
                        if k not in ("output", "x_yamadori")})
        size = sum(len(t.encode("utf-8")) for t in (t_in, t_out, t_resp))
        out["items"] = len(inp) + len(output_items)
        out["bytes"] = size
        if size > ACCOUNT_BYTES:
            out["reason"] = (f"{size} bytes is more than the per-account cap "
                             f"({ACCOUNT_BYTES})")
            return out
        with _lock:
            con = _db()
            try:
                root = rid
                if prev_id:
                    row = con.execute(
                        "SELECT root_id FROM responses WHERE account=? AND "
                        "id=?", (account, prev_id)).fetchone()
                    if row is not None:
                        root = row[0]
                con.execute(
                    "INSERT OR REPLACE INTO responses(account, id, created, "
                    "prev_id, root_id, model, status, input_json, output_json,"
                    " response_json, bytes) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (account, rid, created, prev_id or None, root, model,
                     status, t_in, t_out, t_resp, size))
                _chain_refresh(con, account, root, now)
                con.commit()
            finally:
                con.close()
        out["stored"] = True
        out["chain"] = root
    except Exception as e:                                       # noqa: BLE001
        out["reason"] = f"{type(e).__name__}: {e}"
        return out
    maybe_prune()
    return out


def _chain_refresh(con, account: str, root: str, now: float | None) -> None:
    """Recount a chain's bytes; touch it when `now` is given; drop its row
    when nothing is left."""
    n, b = con.execute("SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM responses "
                       "WHERE account=? AND root_id=?", (account, root)
                       ).fetchone()
    if not n:
        con.execute("DELETE FROM chains WHERE account=? AND root_id=?",
                    (account, root))
        return
    if now is None:
        con.execute("UPDATE chains SET bytes=? WHERE account=? AND root_id=?",
                    (b, account, root))
    else:
        con.execute(
            "INSERT INTO chains(account, root_id, last_seen, bytes) "
            "VALUES(?,?,?,?) ON CONFLICT(account, root_id) DO UPDATE SET "
            "last_seen=excluded.last_seen, bytes=excluded.bytes",
            (account, root, now, b))


# ------------------------------------------------------------------- read ----

def _row(r) -> dict:
    return {"id": r[0], "created": r[1], "prev_id": r[2], "root_id": r[3],
            "model": r[4], "status": r[5], "input": json.loads(r[6]),
            "output": json.loads(r[7]), "response": json.loads(r[8]),
            "bytes": r[9]}


_COLS = ("id, created, prev_id, root_id, model, status, input_json, "
         "output_json, response_json, bytes")


def chain(account: str, rid: str, now: float | None = None) -> list[dict]:
    """The responses of the chain ending at `rid`, ROOT FIRST. Raises
    NotFound(rid) when `rid` is not this account's, and NotFound(rid,
    broken=True) when an ancestor was deleted or evicted. Touches the chain."""
    now = time.time() if now is None else now
    with _lock:
        con = _db()
        try:
            rows = con.execute(
                f"""WITH RECURSIVE c(id, prev_id, depth) AS (
                      SELECT id, prev_id, 0 FROM responses
                        WHERE account=? AND id=?
                      UNION ALL
                      SELECT r.id, r.prev_id, c.depth + 1 FROM responses r
                        JOIN c ON r.account=? AND r.id=c.prev_id)
                    SELECT {', '.join('r.' + x.strip() for x in _COLS.split(','))}
                      FROM responses r JOIN c ON r.id=c.id
                     WHERE r.account=? ORDER BY c.depth DESC""",
                (account, rid, account, account)).fetchall()
            if not rows:
                raise NotFound(rid)
            out = [_row(r) for r in rows]
            if out[0]["prev_id"]:
                raise NotFound(rid, broken=True)
            root = out[-1]["root_id"]
            con.execute("UPDATE chains SET last_seen=? WHERE account=? AND "
                        "root_id=?", (now, account, root))
            con.commit()
        finally:
            con.close()
    return out


def history(account: str, rid: str) -> dict:
    """What `previous_response_id: rid` stands for: {items: the input items
    then the output items of every response of the chain, root first; chain:
    its length; prompt_cache_key: the key the last response was sent with}."""
    rows = chain(account, rid)
    items: list = []
    for r in rows:
        items.extend(r["input"])
        items.extend(r["output"])
    return {"items": items, "chain": len(rows),
            "prompt_cache_key": rows[-1]["response"].get("prompt_cache_key")}


def get(account: str, rid: str) -> dict | None:
    """The stored Response object (output restored), or None."""
    with _lock:
        con = _db()
        try:
            r = con.execute(f"SELECT {_COLS} FROM responses WHERE account=? "
                            f"AND id=?", (account, rid)).fetchone()
            if r is not None:
                con.execute("UPDATE chains SET last_seen=? WHERE account=? "
                            "AND root_id=?", (time.time(), account, r[3]))
                con.commit()
        finally:
            con.close()
    if r is None:
        return None
    row = _row(r)
    obj = dict(row["response"])
    obj["output"] = row["output"]
    return obj


def input_items(account: str, rid: str) -> list | None:
    """The response's own input items (with ids), in order, or None."""
    with _lock:
        con = _db()
        try:
            r = con.execute("SELECT input_json FROM responses WHERE account=? "
                            "AND id=?", (account, rid)).fetchone()
        finally:
            con.close()
    return json.loads(r[0]) if r else None


def delete(account: str, rid: str) -> bool:
    with _lock:
        con = _db()
        try:
            r = con.execute("SELECT root_id FROM responses WHERE account=? "
                            "AND id=?", (account, rid)).fetchone()
            if r is None:
                return False
            con.execute("DELETE FROM responses WHERE account=? AND id=?",
                        (account, rid))
            _chain_refresh(con, account, r[0], None)
            con.commit()
        finally:
            con.close()
    return True


# ------------------------------------------------------------------ prune ----

def _drop_chain(con, account: str, root: str) -> None:
    con.execute("DELETE FROM responses WHERE account=? AND root_id=?",
                (account, root))
    con.execute("DELETE FROM chains WHERE account=? AND root_id=?",
                (account, root))


def prune(now: float | None = None) -> dict:
    """Enforce the max age, each account's byte cap, then the total cap,
    dropping whole chains least recently used first."""
    now = time.time() if now is None else now
    out = {"aged": 0, "account_cap": 0, "total_cap": 0}
    try:
        with _lock:
            con = _db()
            try:
                for a, s in con.execute(
                        "SELECT account, root_id FROM chains WHERE "
                        "last_seen < ?", (now - MAX_AGE,)).fetchall():
                    _drop_chain(con, a, s)
                    out["aged"] += 1
                for (a,) in con.execute("SELECT DISTINCT account FROM chains"
                                        ).fetchall():
                    rows = con.execute(
                        "SELECT root_id, bytes FROM chains WHERE account=? "
                        "ORDER BY last_seen ASC", (a,)).fetchall()
                    total = sum(b for _s, b in rows)
                    for s, b in rows:
                        if total <= ACCOUNT_BYTES:
                            break
                        _drop_chain(con, a, s)
                        total -= b
                        out["account_cap"] += 1
                rows = con.execute("SELECT account, root_id, bytes FROM chains "
                                   "ORDER BY last_seen ASC").fetchall()
                total = sum(r[2] for r in rows)
                for a, s, b in rows:
                    if total <= TOTAL_BYTES:
                        break
                    _drop_chain(con, a, s)
                    total -= b
                    out["total_cap"] += 1
                con.commit()
            finally:
                con.close()
    except Exception as e:                                       # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def maybe_prune() -> None:
    """At most once a minute, in a daemon thread: never on the request path."""
    global _last_prune
    now = time.time()
    if now - _last_prune < PRUNE_EVERY:
        return
    _last_prune = now
    threading.Thread(target=prune, daemon=True).start()


def forget_account(account: str) -> None:
    with _lock:
        con = _db()
        try:
            con.execute("DELETE FROM responses WHERE account=?", (account,))
            con.execute("DELETE FROM chains WHERE account=?", (account,))
            con.commit()
        finally:
            con.close()


def stats() -> dict:
    try:
        with _lock:
            con = _db()
            try:
                n, b = con.execute("SELECT COUNT(*), COALESCE(SUM(bytes),0) "
                                   "FROM responses").fetchone()
                c = con.execute("SELECT COUNT(*) FROM chains").fetchone()[0]
            finally:
                con.close()
        return {"responses": n, "bytes": b, "chains": c}
    except sqlite3.Error as e:
        return {"error": str(e)}
