#!/usr/bin/env python
"""The self-healing half of skill selection: every fallback is recorded, and
the stack learns from the records when it is idle.

WHY (operator, 2026-09-24)

Selection is a cheap cascade -- deterministic signals, then an embedding
match against each skill's triggers, then (optionally) a Laya closed
question -- and only when that cascade is unsure does it fall back to a model
call that picks among the candidates (skill_select). A fallback costs a
generation on the request path. So every fallback leaves a durable record,
and an idle-time job turns the records into what makes the next fallback
unnecessary:

  (a) the decided request becomes a LEARNED TRIGGER of each skill the
      fallback chose, so the embedding stage recognises the next request
      like it without asking;
  (b) every candidate becomes a LABELLED PAIR (request, skill, used or not)
      for training a small router head later, the way Laya's route_in head
      was trained (docs/LAYA.md). Nothing trains here.

THE MEASURE

The fallback rate -- fallbacks over requests that had any candidate, per day
-- should fall as (a) accumulates. `rate()` reports it and the dashboard
draws it. If it does not fall, the learned triggers are not doing their job,
and that is a finding.

WHEN IT RUNS

`schedule()` (the worker calls it once a minute) enqueues one `skill.learn`
job on the gpu lane -- it embeds -- only when ALL of these hold, each
established positively rather than assumed:

  - there are unlearned fallback records;
  - the gpu lane has nothing running or queued;
  - no client request has reached the proxy for IDLE_MINUTES (the newest
    `turn` event in index/corpus.sqlite3, which the proxy writes for every
    request); a corpus that cannot be read is NOT idle;
  - no learn job is already queued or running.

WHERE IT LIVES

Three tables in the jobs database, next to `skills`: the fallback records,
the learned triggers, and a per-day counter of selection outcomes. Labels go
to a jsonl under index/skills/ (YAMADORI_SKILL_LABELS). The records hold a
request EXCERPT -- the user's own words of the last user turn, at most
QUERY_CHARS, with any code or tool output it carried dropped -- and the
signals as ids, strengths, evidence kinds, names and counts, never text
(`durable()`, 2026-09-27; `scrub()` / `--scrub` purged the rows written
before it). Kept locally, never sent anywhere.

ONLY CLIENT TRAFFIC IS LEARNED FROM (2026-09-27, mirroring deep_learn).
Every fallback record, learned trigger, selection row and label carries the
request's TRAFFIC class (`traffic_of`): "test" for the live suite's dev
accounts (corpus.account_traffic, which FAILS CLOSED: an unreadable account
registry is "test"), "client" for any other account, and "unknown" when no
account came with it -- an offline suite, a bench replay, a script calling
selection directly. Test and unknown rows are KEPT for inspection and never
learned from: `pending()`, `idle_state()` and `handle_learn()` read client
rows only, `add_trigger()` refuses any other class, and
`learned_triggers()` -- what the embedding stage matches against -- returns
client-learned triggers only (a row with no class, from before the column,
is not client). Found 2026-09-27: the store's one learned trigger came from
a live-suite prompt (test account), and 21 fallback records were an offline
suite's fixture (mcp/test_domains.py through proxy.prepare, no account).
The per-day counters are kept per class too (`skill_select_stats_traffic`);
`rate()` reports the client fallback rate beside the all-traffic one.
`backfill_traffic()` / `--backfill-traffic` classifies rows written before
the column from the corpus turn the proxy logged with them (+-2 s).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402
import skill_limits as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.environ.get("YAMADORI_SKILL_LABELS") or (
    os.path.join(os.path.dirname(os.path.abspath(
        os.environ["YAMADORI_JOBS_DB"])), "router_labels.jsonl")
    if os.environ.get("YAMADORI_JOBS_DB") else
    os.path.join(HERE, "..", "index", "skills", "router_labels.jsonl"))
IDLE_MINUTES = float(os.environ.get("YAMADORI_SKILL_LEARN_IDLE_MINUTES", "15"))
QUERY_CHARS = 600
LEARN_BATCH = 200
LEARN = ("skill.learn", "gpu_a4000")  # the embedder, on the A4000: jobs.GPU_SCOPES (2026-09-30)

DDL = """
CREATE TABLE IF NOT EXISTS skill_fallbacks(
    id          TEXT PRIMARY KEY,
    created     REAL NOT NULL,
    route_class TEXT,
    query       TEXT,
    features    TEXT NOT NULL DEFAULT '{}',
    candidates  TEXT NOT NULL DEFAULT '[]',
    decision    TEXT NOT NULL DEFAULT '{}',
    reason      TEXT,
    ok          INTEGER NOT NULL DEFAULT 0,
    learned_at  REAL,
    learned     TEXT,
    traffic     TEXT,
    account     TEXT);
CREATE INDEX IF NOT EXISTS skill_fallbacks_learn ON skill_fallbacks(learned_at);
CREATE TABLE IF NOT EXISTS skill_triggers_learned(
    skill       TEXT NOT NULL,
    text        TEXT NOT NULL,
    origin      TEXT NOT NULL,
    ref         TEXT,
    created     REAL NOT NULL,
    traffic     TEXT,
    PRIMARY KEY(skill, text));
CREATE TABLE IF NOT EXISTS craft_queries(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    traffic     TEXT,
    question    TEXT,
    shortlist   TEXT NOT NULL DEFAULT '[]',
    chosen      TEXT,
    answered    INTEGER NOT NULL DEFAULT 0,
    p           REAL,
    gate_p      REAL,
    reads       TEXT,
    untuned     INTEGER NOT NULL DEFAULT 1,
    package     TEXT,
    retrieval   TEXT,
    decision_id TEXT);
CREATE INDEX IF NOT EXISTS craft_queries_ts ON craft_queries(ts);
CREATE TABLE IF NOT EXISTS skill_selections(
    ts          REAL NOT NULL,
    skill       TEXT NOT NULL,
    version     INTEGER,
    route_class TEXT,
    decided_by  TEXT,
    strength    TEXT,
    request     TEXT,
    traffic     TEXT);
CREATE INDEX IF NOT EXISTS skill_selections_skill ON skill_selections(skill, ts);
CREATE TABLE IF NOT EXISTS skill_select_stats(
    day             TEXT PRIMARY KEY,
    requests        INTEGER NOT NULL DEFAULT 0,
    with_candidates INTEGER NOT NULL DEFAULT 0,
    injected        INTEGER NOT NULL DEFAULT 0,
    fallbacks       INTEGER NOT NULL DEFAULT 0,
    fallback_errors INTEGER NOT NULL DEFAULT 0,
    laya_decided    INTEGER NOT NULL DEFAULT 0,
    cache_hits      INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS skill_select_stats_traffic(
    day             TEXT NOT NULL,
    traffic         TEXT NOT NULL,
    requests        INTEGER NOT NULL DEFAULT 0,
    with_candidates INTEGER NOT NULL DEFAULT 0,
    injected        INTEGER NOT NULL DEFAULT 0,
    fallbacks       INTEGER NOT NULL DEFAULT 0,
    fallback_errors INTEGER NOT NULL DEFAULT 0,
    laya_decided    INTEGER NOT NULL DEFAULT 0,
    cache_hits      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(day, traffic));
"""
# Columns added after a table first shipped: (table, column, type).
_ADDED = (("skill_fallbacks", "traffic", "TEXT"),
          ("skill_fallbacks", "account", "TEXT"),
          ("skill_triggers_learned", "traffic", "TEXT"),
          ("skill_selections", "traffic", "TEXT"))
TRAFFIC = ("client", "test", "unknown")
_ENSURED: set[str] = set()
STAT_KEYS = ("requests", "with_candidates", "injected", "fallbacks",
             "fallback_errors", "laya_decided", "cache_hits")


def _db() -> sqlite3.Connection:
    path = os.path.abspath(jobs.DB)
    if path not in _ENSURED:
        jobs._db().close()
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA busy_timeout=30000")
    if path not in _ENSURED:
        con.executescript(DDL)
        for table, col, typ in _ADDED:
            have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        _ENSURED.add(path)
    con.row_factory = sqlite3.Row
    return con


def today() -> str:
    return dt.date.today().isoformat()


def traffic_of(account: str | None) -> str:
    """The request's traffic class: "unknown" with no account (nothing
    says a client sent it), else corpus.account_traffic -- "test" for the
    live suite's dev accounts, "client" otherwise -- and "test" when that
    cannot be decided (FAILS CLOSED, like deep._traffic)."""
    if not account:
        return "unknown"
    try:
        import corpus
        t = corpus.account_traffic(account)
    except Exception:                                            # noqa: BLE001
        return "test"
    return t if t in TRAFFIC else "test"


def record_craft_query(q: dict, account: str | None) -> None:
    """One durable row per question the model asked yama_recall_craft
    (mcp/craft_query.py): the question (its own words, up to
    craft_query.LOG_QUESTION_CHARS), the shortlist's skill ids, what jjava
    chose and how sure, the package the conversation was on, and the
    account's TRAFFIC CLASS -- never the account or its key. These rows are
    labels for jjava (a question and the craft that answered it) and the
    evidence for gap-fill (`answered` 0: a question no craft answered). Only
    `client` rows are ever learned from (traffic_of; a test account's are
    marked `test` and skipped). Never raises."""
    try:
        con = _db()
        try:
            con.execute(
                "INSERT INTO craft_queries(ts, traffic, question, shortlist, "
                "chosen, answered, p, gate_p, untuned, package, retrieval, "
                "decision_id, reads) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), traffic_of(account),
                 str(q.get("question_full") or q.get("question") or "")[:1000],
                 json.dumps(q.get("shortlist") or []), q.get("chosen"),
                 int(bool(q.get("answered"))), q.get("p"), q.get("gate_p"),
                 int(bool(q.get("untuned", True))), q.get("package"),
                 q.get("retrieval"),
                 (q.get("choice") or {}).get("decision_id"),
                 json.dumps({"choice": q.get("choice"),
                             "gate": q.get("gate"),
                             # the state the reads saw: the goal was kept
                             # out of them; it is here for labelling
                             "state": dict(q.get("state") or {},
                                           goal_full=q.get("goal_full"))})))
        finally:
            con.close()
    except Exception as e:                                       # noqa: BLE001
        print(f"  craft query not logged: {type(e).__name__}: {e}",
              flush=True)


def _cls(traffic: str | None) -> str:
    return traffic if traffic in TRAFFIC else "unknown"


def bump(traffic: str | None = None, **counts) -> None:
    """Add to today's selection counters, all traffic and per traffic
    class. One UPSERT each per request."""
    inc = {k: int(counts.get(k) or 0) for k in STAT_KEYS}
    if not any(inc.values()):
        return
    con = _db()
    try:
        con.execute(
            f"INSERT INTO skill_select_stats(day,{','.join(STAT_KEYS)}) "
            f"VALUES(?,{','.join('?' * len(STAT_KEYS))}) "
            f"ON CONFLICT(day) DO UPDATE SET "
            + ",".join(f"{k}={k}+excluded.{k}" for k in STAT_KEYS),
            [today()] + [inc[k] for k in STAT_KEYS])
        con.execute(
            f"INSERT INTO skill_select_stats_traffic(day,traffic,"
            f"{','.join(STAT_KEYS)}) "
            f"VALUES(?,?,{','.join('?' * len(STAT_KEYS))}) "
            f"ON CONFLICT(day,traffic) DO UPDATE SET "
            + ",".join(f"{k}={k}+excluded.{k}" for k in STAT_KEYS),
            [today(), _cls(traffic)] + [inc[k] for k in STAT_KEYS])
    finally:
        con.close()


SELECTIONS_KEEP = 20000


def record_selection(matched: list[dict], *, route_class: str | None,
                     request_key: str = "", traffic: str | None = None
                     ) -> None:
    """One row per skill a DECISION injected (never on a replay or a sticky
    hit): which requests selected it recently, for the skill factory's
    detail view. `request_key` is a hash, never text. Bounded: the oldest
    rows past SELECTIONS_KEEP are dropped."""
    if not matched:
        return
    now = time.time()
    con = _db()
    try:
        con.executemany(
            "INSERT INTO skill_selections(ts,skill,version,route_class,"
            "decided_by,strength,request,traffic) VALUES(?,?,?,?,?,?,?,?)",
            [(now, m.get("id"), m.get("version"), route_class,
              m.get("decided_by"), m.get("strength"), request_key[:16],
              _cls(traffic))
             for m in matched if m.get("id")])
        n = con.execute("SELECT COUNT(*) FROM skill_selections").fetchone()[0]
        if n > SELECTIONS_KEEP + 1000:
            con.execute("DELETE FROM skill_selections WHERE rowid IN (SELECT "
                        "rowid FROM skill_selections ORDER BY ts LIMIT ?)",
                        (n - SELECTIONS_KEEP,))
    finally:
        con.close()


def selections(skill: str | None = None, limit: int = 50) -> list[dict]:
    con = _db()
    try:
        if skill:
            rows = con.execute("SELECT * FROM skill_selections WHERE skill=? "
                               "ORDER BY ts DESC LIMIT ?", (skill, limit))
        else:
            rows = con.execute("SELECT * FROM skill_selections ORDER BY ts "
                               "DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
    finally:
        con.close()


def rate(days: int = 14) -> list[dict]:
    """Per day: counts and the fallback rate (fallbacks / requests with any
    candidate), over all traffic; `by_traffic` the same per class, and
    `client_fallback_rate` the one the learned triggers should move (days
    before the per-class table: absent). None where there were no
    candidates -- not zero."""
    con = _db()
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM skill_select_stats ORDER BY day DESC LIMIT ?",
            (days,))]
        per: dict[str, dict] = {}
        for r in con.execute(
                "SELECT * FROM skill_select_stats_traffic WHERE day IN "
                "(SELECT day FROM skill_select_stats ORDER BY day DESC "
                "LIMIT ?)", (days,)):
            d = dict(r)
            n = d["with_candidates"]
            d["fallback_rate"] = round(d["fallbacks"] / n, 4) if n else None
            per.setdefault(d.pop("day"), {})[d.pop("traffic")] = d
    finally:
        con.close()
    for r in rows:
        n = r["with_candidates"]
        r["fallback_rate"] = round(r["fallbacks"] / n, 4) if n else None
        if r["day"] in per:
            r["by_traffic"] = per[r["day"]]
            r["client_fallback_rate"] = (per[r["day"]].get("client") or {}
                                         ).get("fallback_rate")
    return list(reversed(rows))


def durable(query: str | None, features: dict | None) -> tuple[str, dict]:
    """(query, features) as a durable record may hold them (2026-09-27): the
    user's words of the turn -- never the code or tool output it carries
    (skill_classify.prose_excerpt) -- and the signals as ids, strengths,
    evidence kinds, names and counts (skill_classify.durable_signals).
    Every writer of a fallback record or a label goes through this."""
    import skill_classify as C
    f = features if isinstance(features, dict) else {}
    out = {"signals": C.durable_signals(f.get("signals") or {})}
    if "route_class" in f:
        out["route_class"] = f.get("route_class")
    return C.prose_excerpt(query or "")[:QUERY_CHARS], out


def scrub(*, dry_run: bool = False) -> dict:
    """Purge text from the durable records written before durable() existed
    (2026-09-27: `fresh_text` -- the recent tool results and written code --
    rode along in the fallback's features from 2026-09-26, and a query could
    carry pasted code). Rewrites skill_fallbacks, skill_triggers_learned and
    the router labels file IN PLACE through durable(); a learned trigger
    left empty is deleted. Returns the counts."""
    counts = {"fallbacks": 0, "fallbacks_changed": 0, "triggers": 0,
              "triggers_changed": 0, "triggers_deleted": 0, "labels": 0,
              "labels_changed": 0}
    con = _db()
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT id, query, features FROM skill_fallbacks")]
        counts["fallbacks"] = len(rows)
        for r in rows:
            try:
                f = json.loads(r["features"] or "{}")
            except ValueError:
                f = {}
            q, f2 = durable(r["query"], f)
            fj = json.dumps(f2, default=list)
            if q != (r["query"] or "") or fj != (r["features"] or ""):
                counts["fallbacks_changed"] += 1
                if not dry_run:
                    con.execute("UPDATE skill_fallbacks SET query=?, "
                                "features=? WHERE id=?", (q, fj, r["id"]))
        trig = [dict(r) for r in con.execute(
            "SELECT rowid, skill, text FROM skill_triggers_learned")]
        counts["triggers"] = len(trig)
        import skill_classify as C
        for r in trig:
            t = " ".join(C.prose_excerpt(r["text"]).split())
            if t == r["text"]:
                continue
            counts["triggers_changed"] += 1
            if dry_run:
                continue
            dup = con.execute("SELECT 1 FROM skill_triggers_learned WHERE "
                              "skill=? AND text=?", (r["skill"], t)).fetchone()
            if not t or dup:
                con.execute("DELETE FROM skill_triggers_learned WHERE rowid=?",
                            (r["rowid"],))
                counts["triggers_deleted"] += 1
            else:
                con.execute("UPDATE skill_triggers_learned SET text=? WHERE "
                            "rowid=?", (t, r["rowid"]))
    finally:
        con.close()
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8") as f:
            lines = f.read().splitlines()
        out = []
        for ln in lines:
            if not ln.strip():
                continue
            counts["labels"] += 1
            try:
                d = json.loads(ln)
            except ValueError:
                counts["labels_changed"] += 1     # unreadable: dropped
                continue
            q, f2 = durable(d.get("query"), d.get("features"))
            d2 = dict(d, query=q, features=f2)
            if d2 != d:
                counts["labels_changed"] += 1
            out.append(json.dumps(d2, default=list))
        if counts["labels_changed"] and not dry_run:
            tmp = LABELS + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + ("\n" if out else ""))
            os.replace(tmp, LABELS)
    return counts


def _corpus_turns(con: sqlite3.Connection, ts: float, window: float
                  ) -> tuple[str, str | None] | None:
    """(traffic, account) of the corpus turn nearest `ts` within `window`
    seconds, or None. The proxy logs a turn event with every request it
    serves; a row with none beside it did not come through the proxy."""
    r = con.execute(
        "SELECT turn, payload FROM events WHERE kind='turn' AND ts BETWEEN "
        "? AND ? ORDER BY ABS(ts - ?) LIMIT 1",
        (ts - window, ts + window, ts)).fetchone()
    if r is None:
        return None
    try:
        p = json.loads(r[1] or "{}")
    except ValueError:
        p = {}
    acct = p.get("account") or None
    t = p.get("traffic")
    if t not in ("test", "client"):
        t = traffic_of(acct) if acct else "unknown"
    if r[0] in _corpus_test_turns(con):
        t = "test"
    return t, acct


_TEST_TURNS: dict = {}


def _corpus_test_turns(con: sqlite3.Connection) -> set[str]:
    if "set" not in _TEST_TURNS:
        import corpus
        _TEST_TURNS["set"] = corpus.test_turns(con)
    return _TEST_TURNS["set"]


def backfill_traffic(*, window: float = 2.0, dry_run: bool = False) -> dict:
    """Classify the rows written before the `traffic` column (2026-09-27)
    from the corpus: a fallback record or selection row takes the class of
    the proxy turn logged within `window` seconds of it (the live gate's
    rows sit 0.0-0.2 s from theirs); with none it is "unknown" -- it did
    not come through the proxy (an offline suite, a replay). A learned
    trigger takes its source record's class. The labels file gains each
    line's record's class. The corpus is opened READ-ONLY; unreadable, every
    row is "unknown" (never learned from). Returns counts per class."""
    _TEST_TURNS.clear()
    out = {"fallbacks": {}, "selections": {}, "triggers": {}, "labels": {},
           "corpus": "read"}
    ccon = None
    try:
        import corpus
        path = os.path.abspath(corpus.CORPUS_DB)
        if os.path.exists(path):
            ccon = sqlite3.connect(f"file:{path}?mode=ro", uri=True,
                                   timeout=5)
        else:
            out["corpus"] = "absent: every row unknown"
    except Exception as e:                                       # noqa: BLE001
        out["corpus"] = f"unreadable ({type(e).__name__}): every row unknown"
        ccon = None

    def cls(ts: float) -> tuple[str, str | None]:
        if ccon is None:
            return "unknown", None
        try:
            return _corpus_turns(ccon, ts, window) or ("unknown", None)
        except sqlite3.Error:
            return "unknown", None

    def count(bucket: str, t: str) -> None:
        out[bucket][t] = out[bucket].get(t, 0) + 1

    con = _db()
    try:
        by_id: dict[str, str] = {}
        for r in con.execute("SELECT id, created, traffic FROM "
                             "skill_fallbacks").fetchall():
            if r["traffic"]:
                by_id[r["id"]] = r["traffic"]
                continue
            t, acct = cls(r["created"])
            by_id[r["id"]] = t
            count("fallbacks", t)
            if not dry_run:
                con.execute("UPDATE skill_fallbacks SET traffic=?, "
                            "account=COALESCE(account, ?) WHERE id=?",
                            (t, (acct or "")[:16] or None, r["id"]))
        for r in con.execute("SELECT rowid, ts FROM skill_selections WHERE "
                             "traffic IS NULL").fetchall():
            t, _a = cls(r["ts"])
            count("selections", t)
            if not dry_run:
                con.execute("UPDATE skill_selections SET traffic=? WHERE "
                            "rowid=?", (t, r["rowid"]))
        for r in con.execute("SELECT rowid, ref FROM skill_triggers_learned "
                             "WHERE traffic IS NULL").fetchall():
            t = by_id.get(r["ref"] or "", "unknown")
            count("triggers", t)
            if not dry_run:
                con.execute("UPDATE skill_triggers_learned SET traffic=? "
                            "WHERE rowid=?", (t, r["rowid"]))
    finally:
        con.close()
        if ccon is not None:
            ccon.close()
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        new, changed = [], False
        for ln in lines:
            try:
                d = json.loads(ln)
            except ValueError:
                new.append(ln)
                continue
            if "traffic" not in d:
                d["traffic"] = by_id.get(d.get("fallback") or "", "unknown")
                count("labels", d["traffic"])
                changed = True
            new.append(json.dumps(d, default=list))
        if changed and not dry_run:
            tmp = LABELS + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(new) + "\n")
            os.replace(tmp, LABELS)
    return out


def purge_untrusted_triggers(*, dry_run: bool = False) -> list[dict]:
    """Delete every learned trigger that did not come from client traffic
    (back the jobs database up first: index/_archive/). learned_triggers()
    already ignores them; this removes them. Returns what was removed."""
    con = _db()
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT rowid, skill, text, origin, ref, created, traffic FROM "
            "skill_triggers_learned WHERE traffic IS NULL OR "
            "traffic != 'client'")]
        if not dry_run:
            for r in rows:
                con.execute("DELETE FROM skill_triggers_learned WHERE "
                            "rowid=?", (r["rowid"],))
    finally:
        con.close()
    return rows


def record_fallback(*, query: str, route_class: str | None, features: dict,
                    candidates: list[dict], decision: dict, reason: str,
                    ok: bool, traffic: str | None = None,
                    account: str | None = None) -> str:
    """One durable row per fallback: what the request looked like, who the
    candidates were and how they scored, what the fallback decided, why --
    and whose traffic it was (`traffic`, else decided from `account` by
    traffic_of; neither: "unknown", never learned from)."""
    fid = uuid.uuid4().hex[:12]
    query, features = durable(query, features)
    traffic = _cls(traffic) if traffic else traffic_of(account)
    con = _db()
    try:
        con.execute(
            "INSERT INTO skill_fallbacks(id,created,route_class,query,"
            "features,candidates,decision,reason,ok,traffic,account) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (fid, time.time(), route_class, query,
             json.dumps(features, default=list)[:20000],
             json.dumps(candidates, default=list)[:20000],
             json.dumps(decision, default=list)[:4000], (reason or "")[:1000],
             1 if ok else 0, traffic, (account or "")[:16] or None))
    finally:
        con.close()
    return fid


def fallbacks(limit: int = 50) -> list[dict]:
    con = _db()
    try:
        out = []
        for r in con.execute("SELECT * FROM skill_fallbacks ORDER BY created "
                             "DESC LIMIT ?", (limit,)):
            d = dict(r)
            for k in ("features", "candidates", "decision", "learned"):
                try:
                    d[k] = json.loads(d.get(k) or "null")
                except ValueError:
                    pass
            out.append(d)
        return out
    finally:
        con.close()


def pending() -> int:
    """Unlearned CLIENT records (test and unknown ones are never learned
    from, so they never make the stack due for a learn job)."""
    con = _db()
    try:
        return con.execute("SELECT COUNT(*) FROM skill_fallbacks WHERE "
                           "learned_at IS NULL AND traffic='client'"
                           ).fetchone()[0]
    finally:
        con.close()


def learned_triggers(skill_ids=None) -> dict[str, list[str]]:
    """{skill: [text]} of the triggers learned from CLIENT traffic: what
    the embedding stage matches against. A row of any other class, or with
    none (written before the column), is not served."""
    con = _db()
    try:
        out: dict[str, list[str]] = {}
        for r in con.execute("SELECT skill, text FROM skill_triggers_learned "
                             "WHERE traffic='client' ORDER BY created"):
            if skill_ids is None or r["skill"] in skill_ids:
                out.setdefault(r["skill"], []).append(r["text"])
        return out
    finally:
        con.close()


def add_trigger(skill: str, text: str, *, origin: str, ref: str = "",
                traffic: str | None = None) -> bool:
    """Add one learned trigger; the oldest learned ones beyond
    MAX_LEARNED_TRIGGERS are dropped. False if it was already there, and
    False -- nothing written -- unless `traffic` is "client": a trigger is
    learned from client traffic only."""
    # A trigger is a description sentence: the Agent Skills bound on a
    # description (skill_limits.DESCRIPTION_CHARS) bounds it.
    text = " ".join((text or "").split())[:L.DESCRIPTION_CHARS]
    if not text or traffic != "client":
        return False
    con = _db()
    try:
        cur = con.execute(
            "INSERT OR IGNORE INTO skill_triggers_learned(skill,text,origin,"
            "ref,created,traffic) VALUES(?,?,?,?,?,?)",
            (skill, text, origin, ref, time.time(), traffic))
        added = cur.rowcount > 0
        extra = con.execute(
            "SELECT rowid FROM skill_triggers_learned WHERE skill=? "
            "ORDER BY created DESC LIMIT -1 OFFSET ?",
            (skill, L.MAX_LEARNED_TRIGGERS)).fetchall()
        for (rid,) in extra:
            con.execute("DELETE FROM skill_triggers_learned WHERE rowid=?",
                        (rid,))
        return added
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Idle.
# ---------------------------------------------------------------------------
def _last_request_at() -> tuple[float | None, str]:
    """The newest client request (mcp/idle.py owns the one definition of
    idle since 2026-09-27; the package onboarding's gpu stages read it
    too)."""
    import idle
    return idle.last_request_at()


def idle_state(*, running_learn: str | None = None) -> dict:
    """{idle, why}. Every condition is checked; the first that fails is the
    reason. `running_learn` is the learn job asking about itself."""
    n = pending()
    if not n:
        return {"idle": False, "why": "no unlearned fallback records from "
                                      "client traffic", "pending": 0}
    con = jobs._db()
    try:
        busy = con.execute(
            "SELECT COUNT(*) FROM jobs WHERE lane='gpu' AND state IN "
            "('queued','running') AND id != ?", (running_learn or "",)
        ).fetchone()[0]
        learn_live = con.execute(
            "SELECT COUNT(*) FROM jobs WHERE queue=? AND state IN "
            "('queued','running') AND id != ?",
            (LEARN[0], running_learn or "")).fetchone()[0]
    finally:
        con.close()
    if busy:
        return {"idle": False, "why": f"the gpu lane has {busy} job(s) "
                                      "queued or running", "pending": n}
    if learn_live and not running_learn:
        return {"idle": False, "why": "a learn job is already queued or "
                                      "running", "pending": n}
    at, how = _last_request_at()
    if at is None:
        return {"idle": False, "why": how, "pending": n}
    quiet = (time.time() - at) / 60 if at else float("inf")
    if quiet < IDLE_MINUTES:
        return {"idle": False, "why": f"the last client request was "
                                      f"{quiet:.1f} min ago (idle means "
                                      f"{IDLE_MINUTES:g})", "pending": n}
    return {"idle": True, "why": f"{n} record(s) pending; no request for "
                                 f"{quiet:.0f} min; the gpu lane is free",
            "pending": n}


def schedule() -> str | None:
    """Enqueue one learn job when the stack is idle. The worker calls this."""
    st = idle_state()
    if not st["idle"]:
        return None
    return jobs.add(LEARN[0], {"pending": st["pending"]}, lane=LEARN[1],
                    stage="learn")


def handle_learn(job: dict, ctx) -> dict:
    """Turn pending fallback records into learned triggers and labels."""
    st = idle_state(running_learn=job["id"])
    if not st["idle"] and st.get("pending"):
        return {"deferred": st["why"]}
    import skills
    armed = {s["id"]: s for s in skills.armed()}
    con = _db()
    try:
        # CLIENT traffic only: test and unknown records stay as written.
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM skill_fallbacks WHERE learned_at IS NULL AND "
            "traffic='client' ORDER BY created LIMIT ?", (LEARN_BATCH,))]
    finally:
        con.close()
    added = labels = 0
    os.makedirs(os.path.dirname(os.path.abspath(LABELS)), exist_ok=True)
    with open(LABELS, "a", encoding="utf-8") as lab:
        for r in rows:
            decision = json.loads(r["decision"] or "{}")
            cands = json.loads(r["candidates"] or "[]")
            used = set(decision.get("use") or [])
            result: dict = {"triggers": [], "labels": 0}
            if r["ok"]:
                for sid in used:
                    if sid in armed and add_trigger(
                            sid, r["query"] or "", origin="fallback",
                            ref=r["id"], traffic=r["traffic"]):
                        result["triggers"].append(sid)
                        added += 1
                lq, lf = durable(r["query"], json.loads(r["features"]
                                                        or "{}"))
                for c in cands:
                    lab.write(json.dumps({
                        "fallback": r["id"], "at": r["created"],
                        "query": lq, "features": lf,
                        "route_class": r["route_class"],
                        "skill": c.get("id"), "version": c.get("version"),
                        "label": int(c.get("id") in used),
                        "why": decision.get("why"),
                        "traffic": r["traffic"]}, default=list) + "\n")
                    labels += 1
                    result["labels"] += 1
            else:
                result["skipped"] = "the fallback made no decision"
            c2 = _db()
            try:
                c2.execute("UPDATE skill_fallbacks SET learned_at=?, learned=? "
                           "WHERE id=?", (time.time(), json.dumps(result),
                                          r["id"]))
            finally:
                c2.close()
            ctx.beat(f"learned from {rows.index(r) + 1}/{len(rows)}")
    rebuilt = None
    if added:
        import skill_select
        rebuilt = skill_select.refresh_triggers()
    return {"records": len(rows), "triggers_added": added,
            "labels_written": labels, "trigger_index": rebuilt}


def overview() -> dict:
    st = idle_state()
    return {"rate": rate(), "pending": st.get("pending", 0),
            "idle": st, "recent": [
                {k: f.get(k) for k in ("id", "created", "route_class", "ok",
                                       "reason", "decision", "learned_at",
                                       "traffic")}
                for f in fallbacks(20)],
            "labels_file": os.path.basename(LABELS)}


if __name__ == "__main__":
    if "--scrub" in sys.argv[1:]:
        # Back the jobs database up first (index/_archive/); --dry-run counts.
        print(json.dumps(scrub(dry_run="--dry-run" in sys.argv[1:])))
        sys.exit(0)
    if "--backfill-traffic" in sys.argv[1:]:
        # Back the jobs database and the labels up first; --dry-run counts.
        print(json.dumps(backfill_traffic(dry_run="--dry-run" in sys.argv[1:])))
        sys.exit(0)
    if "--purge-untrusted-triggers" in sys.argv[1:]:
        print(json.dumps(purge_untrusted_triggers(
            dry_run="--dry-run" in sys.argv[1:]), default=str))
        sys.exit(0)
    print(json.dumps(overview(), indent=2, default=str))
