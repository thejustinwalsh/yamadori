#!/usr/bin/env python
"""Raw event log. The training corpus for metsumi, collected by working.

WHY RAW EVENTS AND NOT LABELS

The tempting design is to score things online -- "the model called another
tool straight after this search, so the search was rejected". That signal is
inverted. Searching and then reading the top hit is exactly what SUCCESS looks
like, and it is indistinguishable from searching, failing, and falling back to
grep unless you already know the answer.

Worse, tuning against a signal like that is Goodhart with a short loop: the
cheapest way to reduce "rejections" is to return fewer results, so the model
falls back to its own tools, the metric improves, and retrieval quietly dies
with nothing raising an alarm.

So nothing is scored here. Every event is written down as it happened, with
enough context to compute any label later. Labels can always be derived from
raw events; raw events cannot be recovered from bad labels.

WHAT IT IS FOR

Three datasets fall out of this log once there is enough of it:

  routing     (request, tools offered, tool chosen) -- a closed-option choice,
              which is the one shape the decision model measurably does well
  relevance   (query, candidates returned, which the model went on to read)
  selection   (N candidates, which passed an executable check)

The first is the most promising, because tool selection is a classification
over a small fixed set and because there is a measured deficiency to fix.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

import repeats

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DB = os.environ.get("YAMADORI_CORPUS_DB",
                           os.path.join(HERE, "..", "index", "corpus.sqlite3"))
ENABLED = os.environ.get("YAMADORI_CORPUS", "1") == "1"

_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(CORPUS_DB)), exist_ok=True)
    con = sqlite3.connect(CORPUS_DB, timeout=30)
    # WAL (persistent in the file): readers no longer block the writers --
    # the rollback journal made deep-thinking records fail "database is
    # locked" 30 times during the 2026-09-24 live gate.
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn TEXT NOT NULL,       -- groups every event in one request
        ts REAL NOT NULL,
        repo TEXT,
        kind TEXT NOT NULL,       -- turn | tool_call | tool_result | answer
        name TEXT,
        payload TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS events_turn ON events(turn, id)")
    return con


def new_turn() -> str:
    return uuid.uuid4().hex[:16]


def log(turn: str, kind: str, repo: str | None = None,
        name: str | None = None, payload: dict | None = None) -> None:
    """Never let logging break a request. A dropped event costs a row."""
    if not ENABLED:
        return
    try:
        with _lock:
            con = _db()
            con.execute(
                "INSERT INTO events(turn, ts, repo, kind, name, payload) "
                "VALUES(?,?,?,?,?,?)",
                (turn, time.time(), repo, kind, name,
                 json.dumps(payload or {}, default=str)[:20000]))
            con.commit()
            con.close()
    except Exception:                                            # noqa: BLE001
        pass


# TEST TRAFFIC IS NOT PRODUCER EVIDENCE (coordinator, 2026-09-24). The live
# suite (mcp/test_live_stack.py) sends requests shaped like a real harness's --
# Hermes' reviewer, its compaction summariser -- and they landed in this log
# beside the real ones. The offline replays that treat this log as what the
# producers actually send (mcp/test_utility.py: "12 Hermes compactions") then
# counted the live gate's copy as a 13th. So each turn records its ACCOUNT
# (the proxy's 16-hex id, never the key) and `traffic`: "test" when that
# account's label marks the live suite's dev keys, else "client". Dogfood
# harness traffic (hermes-dogfood) is a real producer and stays "client",
# labelled by its account. Readers exclude test traffic with test_turns().
TEST_ACCOUNT_LABELS = ("live-test", "claude-dogfood")
# Logged BEFORE turns recorded an account: the live gate of 2026-09-24,
# 12:13:03-13:01:43, under the live-test account (created 12:12:57), is event
# ids 3787-3916 of index/corpus.sqlite3 -- read off the log, with the Hermes
# dogfood rows on either side (3783-3785 at 10:36; 3917 on, from 13:10). The
# span applies only to the file whose rows at both ends carry these turns.
LEGACY_TEST_SPANS = ({"from_id": 3787, "from_turn": "fa48a951d3164db7",
                      "to_id": 3916, "to_turn": "b8810ebe686347df",
                      "why": "live gate 2026-09-24, live-test account, "
                             "before log_turn recorded accounts"},)


def account_traffic(account: str | None) -> str:
    """"test" when `account` (a 16-hex id) is one of the live suite's dev
    accounts, by its label in the account registry; else "client". The
    dogfood harness account (`hermes-dogfood`) is a real producer: client.

    FAILS CLOSED (pre-deploy review, 2026-09-24): when the registry cannot
    be read, the row is "test" -- an unknown source is never learned from
    (mcp/deep_learn.py reads client rows only)."""
    if not account:
        return "client"
    try:
        import accounts
        reg = accounts._load()
    except Exception:                                            # noqa: BLE001
        return "test"
    for h, meta in (reg or {}).items():
        if h.startswith(account):
            label = str((meta or {}).get("label") or "").lower()
            return ("test" if label.startswith(TEST_ACCOUNT_LABELS)
                    else "client")
    return "client"


def test_turns(con: sqlite3.Connection) -> set[str]:
    """The turn ids in an open corpus that are TEST traffic: rows recorded as
    such, and the legacy spans (LEGACY_TEST_SPANS) when this is their file."""
    out: set[str] = set()
    for turn, raw in con.execute(
            "SELECT turn, payload FROM events WHERE kind='turn' AND "
            "payload LIKE '%\"traffic\": \"test\"%'"):
        out.add(turn)
    for s in LEGACY_TEST_SPANS:
        ends = dict(con.execute("SELECT id, turn FROM events WHERE id IN "
                                "(?, ?)", (s["from_id"], s["to_id"])))
        if ends.get(s["from_id"]) == s["from_turn"] and \
                ends.get(s["to_id"]) == s["to_turn"]:
            out.update(t for (t,) in con.execute(
                "SELECT DISTINCT turn FROM events WHERE id BETWEEN ? AND ?",
                (s["from_id"], s["to_id"])))
    return out


def log_turn(turn: str, repo: str | None, messages: list[dict],
             tools_offered: list[str], first_turn: bool,
             utility: bool | None = None, route: dict | None = None,
             account: str | None = None,
             client_tools: list[str] | None = None) -> None:
    """The request as it arrived.

    Only the last user message is kept, not the whole conversation. The full
    history is the user's code and belongs to them; what the routing dataset
    needs is the request and which tools were on offer when it was made.
    `account` is the proxy's account id; with it, `traffic` says whether the
    row is a producer's or the live suite's (TEST TRAFFIC above).
    """
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            last_user = (c or "")[:2000]
            break
    # The system prompt is where a harness states where it is. Recording its
    # head makes a detection failure diagnosable instead of a mystery -- the
    # first real Hermes run returned repo=None and there was no way to see why.
    system_head = ""
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            system_head = m["content"][:1200]
            break
    log(turn, "turn", repo, payload={
        "request": last_user,
        "system_head": system_head,
        "tools_offered": sorted(tools_offered),
        # The tools the CLIENT sent, apart from the upstream list above: a
        # compaction rewritten onto its conversation goes up with THAT
        # conversation's tools, so tools_offered is not what the client sent
        # (turn 11c0be7eff30, 2026-09-24). None: not recorded (older rows).
        **({"client_tools": sorted(client_tools)}
           if client_tools is not None else {}),
        "n_messages": len(messages),
        "first_turn": first_turn,
        # A client's own side call (selection.utility_call): answered by the
        # bare model, so its row is not a routing example for a task.
        "utility": utility,
        # mcp/route.py's class, and the role the conversation ends on. The
        # labelled set (bench/route/labels.jsonl) had to RECONSTRUCT whether
        # a turn ended on a client tool result from repeated requests; rows
        # from 2026-09-24 on record it.
        "route": ((route or {}).get("class") if route else None),
        "ends_on": (((route or {}).get("signals") or {}).get("ends_on")
                    if route else None),
        "account": (account or "")[:16] or None,
        "traffic": account_traffic(account),
    })


def log_tool_call(turn: str, repo: str | None, name: str, args: dict,
                  hop: int) -> None:
    log(turn, "tool_call", repo, name, {"args": args, "hop": hop})


_STRATA = (("== TAPROOT", "taproot"), ("== BRANCH", "branch"),
            ("== SHOOT", "shoot"))


def result_strata(result: str) -> dict | None:
    """Hits per result tier in a find_by_meaning answer, or None if it has no
    tiers. TAPROOT: a declaration with the name is here. BRANCH: both
    retrievers agree. SHOOT: one retriever only (code_search.py)."""
    counts = {"taproot": 0, "branch": 0, "shoot": 0}
    cur = None
    seen = False
    for line in (result or "").splitlines():
        for head, key in _STRATA:
            if line.startswith(head):
                cur, seen = key, True
                break
        else:
            if cur and line.startswith("### "):
                counts[cur] += 1
    return counts if seen else None


def strata_totals(since_seconds: float = 86400) -> dict:
    """Result-tier totals over recent searches, for the dashboard."""
    since = time.time() - since_seconds
    out = {"taproot": 0, "branch": 0, "shoot": 0, "searches": 0,
           "window_seconds": since_seconds}
    try:
        con = _db()
        rows = con.execute(
            "SELECT payload FROM events WHERE kind='tool_result' AND ts >= ? "
            "AND payload LIKE '%\"strata\"%'", (since,)).fetchall()
        con.close()
    except Exception:                                            # noqa: BLE001
        return out
    for (payload,) in rows:
        try:
            s = json.loads(payload).get("strata") or {}
        except ValueError:
            continue
        out["searches"] += 1
        for k in ("taproot", "branch", "shoot"):
            out[k] += int(s.get(k) or 0)
    return out


def log_tool_result(turn: str, repo: str | None, name: str, result: str,
                    ms: float) -> None:
    """Result SHAPE, not the code itself.

    Which paths came back is what a relevance label needs; the file contents
    are already on disk and copying them here would build a second, unmanaged
    copy of the user's source.
    """
    paths = []
    for line in result.splitlines():
        if ":" in line and ("/" in line or "\\" in line):
            paths.append(line.split(":")[0].strip()[:200])
        if len(paths) >= 12:
            break
    extra = {}
    strata = result_strata(result)
    if strata:
        # Counts only, so safe for a bound repository too. They feed the
        # dashboard's TIER STRATA panel (vitals.strata()), which had no
        # source and filled its space with prose instead.
        extra["strata"] = strata
    if repo is None:
        # No repository bound: the result came from PUBLIC package indexes or
        # our own tools, never a user's source, so it is kept in full (capped
        # like the result the model saw). Without it a looping session could
        # not be read back: the typegpu loop of 2026-09-22 showed only
        # "4165 chars" per result and the repair had to be inferred.
        extra["text"] = result[:6000]
    log(turn, "tool_result", repo, name, {
        **extra,
        "chars": len(result), "ms": round(ms, 1),
        "paths": paths,
        # ONE definition of empty, imported rather than restated.
        #
        # This used to test "no matches"/"not found" in the first 80 chars,
        # a narrower rule than the one repeats.py applies. find_by_meaning
        # says "No results.", which matches neither -- so a request that
        # searched twelve times and found nothing was logged twelve times as
        # `empty: false`. This corpus is the training data for the decision
        # model, and it was being taught that those searches succeeded.
        #
        # Two copies of a predicate drift apart. There is now one.
        "empty": repeats._empty(result),
    })


def log_answer(turn: str, repo: str | None, content: str, hops: int,
               ms: float, finish: str | None = None,
               tool_calls: int | None = None) -> None:
    """Which paths the final answer cited is the relevance label.

    A path that was returned by a search and then named in the answer was
    useful. One returned and never mentioned was not. That comparison is the
    whole point of keeping both.
    """
    import re
    cited = re.findall(r"[\w./\\-]+\.(?:ts|tsx|js|jsx|rs|c|h|cpp|py|go|zig|wgsl|glsl)",
                       content or "")
    log(turn, "answer", repo, payload={
        "chars": len(content or ""),
        "cited_paths": sorted(set(cited))[:20],
        "hops": hops,
        "ms": round(ms, 1),
        # `chars` counts content only. A turn that hands the client a tool
        # call with no preface text has chars 0 and is NOT an empty answer:
        # 28 Hermes rows were misread that way (2026-09-23). These two say
        # which it was.
        "finish": finish,
        "tool_calls": tool_calls,
    })


def stats() -> dict:
    try:
        con = _db()
        rows = dict(con.execute(
            "SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall())
        turns = con.execute("SELECT COUNT(DISTINCT turn) FROM events").fetchone()[0]
        repos_seen = con.execute(
            "SELECT COUNT(DISTINCT repo) FROM events WHERE repo IS NOT NULL").fetchone()[0]
        con.close()
        return {"turns": turns, "repos": repos_seen, **rows}
    except sqlite3.Error as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print(json.dumps(stats(), indent=2))
