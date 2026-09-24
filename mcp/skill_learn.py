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
request EXCERPT (the first QUERY_CHARS of the last user turn) -- the same
text the corpus already logs, kept locally, never sent anywhere.
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
LABELS = os.environ.get("YAMADORI_SKILL_LABELS",
                        os.path.join(HERE, "..", "index", "skills",
                                     "router_labels.jsonl"))
IDLE_MINUTES = float(os.environ.get("YAMADORI_SKILL_LEARN_IDLE_MINUTES", "15"))
QUERY_CHARS = 600
LEARN_BATCH = 200
LEARN = ("skill.learn", "gpu")

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
    learned     TEXT);
CREATE INDEX IF NOT EXISTS skill_fallbacks_learn ON skill_fallbacks(learned_at);
CREATE TABLE IF NOT EXISTS skill_triggers_learned(
    skill       TEXT NOT NULL,
    text        TEXT NOT NULL,
    origin      TEXT NOT NULL,
    ref         TEXT,
    created     REAL NOT NULL,
    PRIMARY KEY(skill, text));
CREATE TABLE IF NOT EXISTS skill_select_stats(
    day             TEXT PRIMARY KEY,
    requests        INTEGER NOT NULL DEFAULT 0,
    with_candidates INTEGER NOT NULL DEFAULT 0,
    injected        INTEGER NOT NULL DEFAULT 0,
    fallbacks       INTEGER NOT NULL DEFAULT 0,
    fallback_errors INTEGER NOT NULL DEFAULT 0,
    laya_decided    INTEGER NOT NULL DEFAULT 0,
    cache_hits      INTEGER NOT NULL DEFAULT 0);
"""
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
        _ENSURED.add(path)
    con.row_factory = sqlite3.Row
    return con


def today() -> str:
    return dt.date.today().isoformat()


def bump(**counts) -> None:
    """Add to today's selection counters. One UPSERT per request."""
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
    finally:
        con.close()


def rate(days: int = 14) -> list[dict]:
    """Per day: counts and the fallback rate (fallbacks / requests with any
    candidate). None where there were no candidates -- not zero."""
    con = _db()
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM skill_select_stats ORDER BY day DESC LIMIT ?",
            (days,))]
    finally:
        con.close()
    for r in rows:
        n = r["with_candidates"]
        r["fallback_rate"] = round(r["fallbacks"] / n, 4) if n else None
    return list(reversed(rows))


def record_fallback(*, query: str, route_class: str | None, features: dict,
                    candidates: list[dict], decision: dict, reason: str,
                    ok: bool) -> str:
    """One durable row per fallback: what the request looked like, who the
    candidates were and how they scored, what the fallback decided, why."""
    fid = uuid.uuid4().hex[:12]
    con = _db()
    try:
        con.execute(
            "INSERT INTO skill_fallbacks(id,created,route_class,query,"
            "features,candidates,decision,reason,ok) VALUES(?,?,?,?,?,?,?,?,?)",
            (fid, time.time(), route_class, (query or "")[:QUERY_CHARS],
             json.dumps(features, default=list)[:20000],
             json.dumps(candidates, default=list)[:20000],
             json.dumps(decision, default=list)[:4000], (reason or "")[:1000],
             1 if ok else 0))
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
    con = _db()
    try:
        return con.execute("SELECT COUNT(*) FROM skill_fallbacks WHERE "
                           "learned_at IS NULL").fetchone()[0]
    finally:
        con.close()


def learned_triggers(skill_ids=None) -> dict[str, list[str]]:
    con = _db()
    try:
        out: dict[str, list[str]] = {}
        for r in con.execute("SELECT skill, text FROM skill_triggers_learned "
                             "ORDER BY created"):
            if skill_ids is None or r["skill"] in skill_ids:
                out.setdefault(r["skill"], []).append(r["text"])
        return out
    finally:
        con.close()


def add_trigger(skill: str, text: str, *, origin: str, ref: str = "") -> bool:
    """Add one learned trigger; the oldest learned ones beyond
    MAX_LEARNED_TRIGGERS are dropped. False if it was already there."""
    text = " ".join((text or "").split())[:L.TRIGGER_CHARS * 2]
    if not text:
        return False
    con = _db()
    try:
        cur = con.execute(
            "INSERT OR IGNORE INTO skill_triggers_learned(skill,text,origin,"
            "ref,created) VALUES(?,?,?,?,?)",
            (skill, text, origin, ref, time.time()))
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
    try:
        import corpus
        path = os.path.abspath(corpus.CORPUS_DB)
    except Exception as e:                                       # noqa: BLE001
        return None, f"the corpus module did not load ({e})"
    if not os.path.exists(path):
        return None, "no corpus database, so no evidence of idleness"
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            ts = con.execute("SELECT MAX(ts) FROM events WHERE kind='turn'"
                             ).fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as e:
        return None, f"the corpus could not be read ({e})"
    return (float(ts) if ts else 0.0), "read"


def idle_state(*, running_learn: str | None = None) -> dict:
    """{idle, why}. Every condition is checked; the first that fails is the
    reason. `running_learn` is the learn job asking about itself."""
    n = pending()
    if not n:
        return {"idle": False, "why": "no unlearned fallback records",
                "pending": 0}
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
        rows = [dict(r) for r in con.execute(
            "SELECT * FROM skill_fallbacks WHERE learned_at IS NULL "
            "ORDER BY created LIMIT ?", (LEARN_BATCH,))]
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
                            ref=r["id"]):
                        result["triggers"].append(sid)
                        added += 1
                for c in cands:
                    lab.write(json.dumps({
                        "fallback": r["id"], "at": r["created"],
                        "query": r["query"],
                        "features": json.loads(r["features"] or "{}"),
                        "route_class": r["route_class"],
                        "skill": c.get("id"), "version": c.get("version"),
                        "label": int(c.get("id") in used),
                        "why": decision.get("why")}, default=list) + "\n")
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
                                       "reason", "decision", "learned_at")}
                for f in fallbacks(20)],
            "labels_file": os.path.basename(LABELS)}


if __name__ == "__main__":
    print(json.dumps(overview(), indent=2, default=str))
