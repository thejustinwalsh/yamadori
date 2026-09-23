#!/usr/bin/env python
"""`rings` -- a work log that survives compaction.

THE FAILURE THIS EXISTS FOR

A long session fills its context. The harness compacts. The summary keeps the
gist and loses the specifics -- "I already changed X and the tests went green"
becomes "worked on X". The model, reading a summary that no longer proves the
work happened, re-plans and does it again. Observed on frontier models; a 27B
will be worse, not better.

WHY NOT BETTER COMPACTION

Compaction is lossy in exactly the dimension that prevents repetition. Any
summariser optimises for meaning-per-token, and "ran the type tests at step 12,
they passed" is low-meaning, high-consequence -- the first thing dropped and
the one whose loss costs the most. You cannot fix that by summarising better.

So the record does not live in the context window at all. It lives in sqlite,
it is append-only, and reading it back costs a few hundred tokens at any point
in a session no matter how long the session has run.

Named for a tree's growth rings: a durable record of what happened, readable
at any time, that the tree does not have to remember.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

RINGS_DB = os.environ.get("RINGS_DB",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "index", "rings.sqlite3"))


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(RINGS_DB)), exist_ok=True)
    con = sqlite3.connect(RINGS_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS rings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT NOT NULL,
        ts REAL NOT NULL,
        kind TEXT NOT NULL,      -- did | learned | check | decided
        summary TEXT NOT NULL,
        detail TEXT,
        outcome TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS rings_session ON rings(session, id)")
    return con


# A session that spans a compaction must resolve to the same key on both sides,
# so it cannot be anything the model remembers. The harness supplies it, or it
# falls back to the working directory, which is stable for a task.
def session_key() -> str:
    return os.environ.get("RINGS_SESSION") or os.path.basename(os.getcwd()) or "default"


KINDS = ("did", "learned", "check", "decided")


def record(kind: str, summary: str, detail: str = "", outcome: str = "",
           session: str | None = None) -> str:
    kind = str(kind or "did").strip().lower()
    if kind not in KINDS:
        return f"unknown kind {kind!r}. Use one of: {', '.join(KINDS)}."
    # A model sends `null` for an optional field as readily as it omits it,
    # and `None.strip()` would crash the call instead of recording the step.
    summary, detail, outcome = (str(x or "") for x in (summary, detail, outcome))
    if not summary.strip():
        return "summary is required -- an entry nobody can read is not a record."
    s = session or session_key()
    con = _db()
    cur = con.execute(
        "INSERT INTO rings(session, ts, kind, summary, detail, outcome) "
        "VALUES(?,?,?,?,?,?)",
        (s, time.time(), kind, summary.strip(), detail.strip(), outcome.strip()))
    con.commit()
    n = cur.lastrowid
    seq = con.execute("SELECT COUNT(*) FROM rings WHERE session=?", (s,)).fetchone()[0]
    con.close()
    return f"recorded #{n} as entry {seq} of session {s!r}."


def read(session: str | None = None, limit: int = 60, kind: str = "") -> str:
    """Render the log compactly. This is what gets read back after compaction.

    Ordered oldest-first so it reads as a narrative, and grouped so that the
    checks -- the part that actually proves work happened -- are impossible to
    skim past.
    """
    s = session or session_key()
    kind = (kind or "").strip().lower()
    # SQLite reads LIMIT 0 as "no rows" and a negative LIMIT as "all rows".
    # Either way the answer below would be wrong, so ask for at least one.
    limit = max(int(limit), 1)
    con = _db()
    q = "SELECT id, ts, kind, summary, detail, outcome FROM rings WHERE session=?"
    args: list = [s]
    if kind:
        q += " AND kind=?"
        args.append(kind)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    rows = con.execute(q, args).fetchall()
    total = con.execute("SELECT COUNT(*) FROM rings WHERE session=?", (s,)).fetchone()[0]
    con.close()
    if not rows and total:
        # A filter that matched nothing is not an empty log. Saying "nothing
        # has been recorded yet, record it now" here invites the model to
        # re-record -- and redo -- work this log exists to prove was done.
        return (f"Session {s!r} has {total} entries, none of kind {kind!r}. "
                f"Read without the kind filter to see them. Kinds: "
                f"{', '.join(KINDS)}.")
    if not rows:
        return (f"No entries for session {s!r}. Nothing has been recorded yet, "
                f"which is not the same as nothing having been done -- if you "
                f"have already worked in this session, record it now.")

    rows = list(reversed(rows))
    out = [f"WORK LOG -- session {s!r}, {total} entries"
           + (f" (showing last {len(rows)})" if total > len(rows) else "")]

    checks = [r for r in rows if r[2] == "check"]
    if checks:
        out.append("")
        out.append("CHECKS RUN -- this is the evidence work actually landed:")
        for _id, _ts, _k, summary, _d, outcome in checks:
            mark = {"pass": "PASS", "fail": "FAIL"}.get(outcome.lower(), outcome or "?")
            out.append(f"  [{mark}] {summary}")

    out.append("")
    out.append("SEQUENCE:")
    for i, (_id, ts, k, summary, detail, outcome) in enumerate(rows, 1):
        age = time.time() - ts
        when = f"{int(age // 60)}m ago" if age < 3600 else f"{age / 3600:.1f}h ago"
        line = f"  {i:>3}. [{k}] {summary}"
        if outcome:
            line += f"  -> {outcome}"
        out.append(f"{line}   ({when})")
        if detail:
            first = detail.splitlines()[0][:140]
            out.append(f"       {first}")

    out.append("")
    out.append("Work listed above is DONE. Do not redo it. If you believe an "
               "entry is wrong, verify it with a check rather than repeating it.")
    return "\n".join(out)


TOOLS = [
    {
        "name": "record_step",
        "description": (
            "Write one line into the durable work log. Do this after anything you "
            "would be annoyed to discover you had done twice: a file changed, a "
            "check run, a dead end ruled out, a decision made.\n"
            "The log lives outside the conversation, so it survives compaction. "
            "Anything you do not record here can be summarised away, and the work "
            "will look undone to whoever reads next -- including you.\n"
            "kind: 'did' for an action, 'check' for a verification and its result, "
            "'learned' for a fact worth not rediscovering, 'decided' for a choice "
            "and its reason."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS)},
                "summary": {"type": "string",
                            "description": "One line, specific. Name files and symbols."},
                "detail": {"type": "string", "description": "Optional. Diff, error, reasoning."},
                "outcome": {"type": "string",
                            "description": "For a check: pass or fail. Otherwise a short result."},
            },
            "required": ["kind", "summary"],
        },
    },
    {
        "name": "read_rings",
        "description": (
            "Read the durable work log for this session: everything already done, "
            "every check run and whether it passed.\n"
            "Read this BEFORE planning, and always immediately after the "
            "conversation has been summarised or compacted. A summary keeps the "
            "gist and drops the specifics, so it cannot tell you whether a change "
            "actually landed -- this can. Work listed here is done; redoing it is "
            "the failure this tool exists to prevent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS),
                         "description": "Optional filter, e.g. 'check' for just verifications."},
                "limit": {"type": "integer", "description": "Max entries. Default 60."},
            },
        },
    },
]


def handle(name: str, args: dict) -> str | None:
    """Dispatch, scoped to the CALLER's session rather than the server's.

    `session_key()` falls back to the basename of the server's working
    directory, which is the same string -- "llama-stack" -- for every caller
    there will ever be. So there was one shared work log: every remote user
    wrote into it and read everybody else's entries back out. Found in the
    tool audit, where a caller with no repository was handed a note about
    `EntityIndex` in packages/core from an unrelated session.

    The proxy knows the real key and passes it as `_session`, per call rather
    than through an environment variable, because this server answers several
    requests at once and a process-global would race between them.

    A missing `_session` is not quietly defaulted. Returning somebody else's
    log is worse than returning none, so an unscoped call says so.
    """
    session = (args.get("_session") or "").strip()
    if not session:
        return ("ERROR: no session is bound to this conversation, so the work "
                "log cannot be read or written. Entries are per-conversation, "
                "and answering from an unscoped log would return a different "
                "conversation's notes.")
    if name == "record_step":
        return record(args.get("kind", "did"), args.get("summary", ""),
                      args.get("detail", ""), args.get("outcome", ""),
                      session=session)
    if name == "read_rings":
        return read(session=session, kind=args.get("kind", ""),
                    limit=int(args.get("limit", 60)))
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "read":
        print(read())
    else:
        print(json.dumps([t["name"] for t in TOOLS]))
