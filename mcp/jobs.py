#!/usr/bin/env python
"""A durable local job queue. SQLite, no broker, no daemon, no frills.

WHY NOT AN OFF-THE-SHELF QUEUE

Celery, RQ, dramatiq and arq all want a broker -- Redis or RabbitMQ -- which is
a second service to run, supervise and fail. Huey's SqliteHuey is the closest
fit and would have been a reasonable choice. procrastinate is the real Oban
analogue and is Postgres-only.

What decided it is that the three requirements here are unusual, and all three
already exist in this repo in some form:

  GPU LANES. Two jobs on one card is not slow, it is WRONG: three benchmark
  processes ran at once and degraded each other into 502s that took an hour to
  attribute to concurrency rather than to the arm that lost most. `admission.py`
  already models this with semaphores. A generic queue's "concurrency=1" cannot
  express "these jobs share a GPU and those do not".

  ERRORS ARE NOT FAILURES TO RETRY BLINDLY, AND THEY ARE NEVER "DONE". A resume
  path that counted errored rows as done permanently poisoned a benchmark file
  -- those rows could never be re-run, and the run reported a pass rate over a
  sample that had silently shrunk. `errored` is its own terminal state here,
  distinct from `done` and from `queued`, and re-running it is an explicit act.

  SQLITE IS ALREADY EVERYWHERE. corpus, rings, index, accounts. Adding a
  broker adds a failure mode to a stack whose last outage was a health check
  that knew too much.

Oban's actual insight is that the queue is a table in the database you already
have. We already have the database.

WHAT MAKES IT DURABLE

A job is a row. Claiming is `BEGIN IMMEDIATE` plus a conditional UPDATE, so two
workers cannot take the same row. A running job carries a `heartbeat`; a worker
that is killed -- which happened repeatedly today -- leaves a row whose
heartbeat goes stale, and `reclaim()` returns it to the queue with its attempt
count intact. Nothing is lost because nothing was ever held in memory.

WHAT IT DELIBERATELY DOES NOT DO

No cron, no chains, no fan-out primitives, no result backend beyond a JSON
blob, no distributed anything. Those are the frills. A pipeline stage that
wants to enqueue the next stage does it by calling `add()` in its own handler,
which is visible in the job row rather than hidden in a framework.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("YAMADORI_JOBS_DB",
                    os.path.join(HERE, "..", "index", "jobs.sqlite3"))

# A running job whose heartbeat is older than this is presumed dead and is
# returned to the queue. Long enough that a legitimate 450-second generation is
# never reclaimed out from under itself -- a model answering slowly is not a
# dead worker, and today's session mistook one for the other three times.
STALE_SECONDS = int(os.environ.get("YAMADORI_JOB_STALE", "1200"))

# Lanes are about a shared physical resource, not about speed.
#   gpu    serialised: one at a time, because they contend for one card
#   cpu    parallel: extraction, parsing, scoring
#   net    parallel: fetching sources
LANES = {"gpu": 1, "cpu": 4, "net": 4}


# ---------------------------------------------------------------------------
# PAUSING A LANE. A benchmark measures seconds and tokens per second, and a
# worker extraction on the same model in another slot moves both. So a
# measurement run pauses the gpu lane: claim() hands out nothing from a paused
# lane, a job already running finishes normally (nothing is killed), and the
# pause EXPIRES -- a benchmark that crashes without resuming must not stall
# the corpus forever. A file next to the DB rather than a table, so pausing
# never takes the queue's write lock.
# ---------------------------------------------------------------------------
def _pause_path(lane: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(DB)), f"jobs.pause.{lane}")


def pause(lane: str, by: str, why: str, ttl_seconds: float = 7200) -> dict:
    """Pause `lane` until resume() or `ttl_seconds` from now. Calling it again
    extends the pause; a long run refreshes it as it goes."""
    rec = {"lane": lane, "by": by, "why": why, "since": time.time(),
           "until": time.time() + ttl_seconds}
    p = _pause_path(lane)
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, p)
    return rec


def resume(lane: str) -> bool:
    try:
        os.remove(_pause_path(lane))
        return True
    except FileNotFoundError:
        return False


def paused(lane: str) -> dict | None:
    """The live pause record for `lane`, or None. An expired or unreadable
    record is not a pause -- a stuck file must never be able to stop work."""
    try:
        with open(_pause_path(lane), encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or float(rec.get("until") or 0) <= time.time():
        return None
    return rec


def running(lane: str) -> int:
    con = _db()
    try:
        return con.execute("SELECT COUNT(*) FROM jobs WHERE state='running' "
                           "AND lane=?", (lane,)).fetchone()[0]
    finally:
        con.close()


STATES = ("queued", "running", "done", "errored", "cancelled")


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB)), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS jobs(
            id          TEXT PRIMARY KEY,
            queue       TEXT NOT NULL,
            lane        TEXT NOT NULL DEFAULT 'cpu',
            state       TEXT NOT NULL DEFAULT 'queued',
            priority    INTEGER NOT NULL DEFAULT 0,
            payload     TEXT NOT NULL DEFAULT '{}',
            result      TEXT,
            error       TEXT,
            attempts    INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            parent      TEXT,
            dataset     TEXT,
            stage       TEXT,
            progress    TEXT,
            created     REAL NOT NULL,
            started     REAL,
            finished    REAL,
            heartbeat   REAL,
            worker      TEXT);
        CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(state, lane, priority, created);
        CREATE INDEX IF NOT EXISTS jobs_dataset ON jobs(dataset, created);
    """)
    return con


def add(queue: str, payload: dict | None = None, *, lane: str = "cpu",
        priority: int = 0, dataset: str | None = None,
        stage: str | None = None, parent: str | None = None,
        max_attempts: int = 3) -> str:
    """Enqueue one job. Returns its id."""
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; known: {sorted(LANES)}")
    jid = uuid.uuid4().hex[:12]
    con = _db()
    try:
        con.execute(
            "INSERT INTO jobs(id,queue,lane,state,priority,payload,dataset,"
            "stage,parent,max_attempts,created) "
            "VALUES(?,?,?,'queued',?,?,?,?,?,?,?)",
            (jid, queue, lane, priority, json.dumps(payload or {}),
             dataset, stage, parent, max_attempts, time.time()))
    finally:
        con.close()
    return jid


def claim(lane: str, worker: str | None = None) -> dict | None:
    """Take the next job in a lane, atomically. None if there is nothing to do.

    `BEGIN IMMEDIATE` takes the write lock before the SELECT, so two workers
    racing for the same row cannot both win it. The conditional UPDATE is a
    second guard: it only fires if the row is still `queued`.
    """
    worker = worker or f"{socket.gethostname()}:{os.getpid()}"
    if paused(lane):
        return None
    con = _db()
    try:
        con.execute("BEGIN IMMEDIATE")
        # THE LANE LIMIT IS GLOBAL, NOT PER PROCESS. A worker enforces LANES
        # with one thread per slot, which only holds while exactly one worker
        # runs. On 2026-09-22 a watchdog restart and a manual start overlapped
        # and two workers ran -- two gpu jobs on one card, the thing this
        # module exists to prevent. Counting running rows inside the same
        # write lock makes the limit hold however many workers there are. A
        # dead worker's row holds its slot until reclaim() frees it.
        limit = LANES.get(lane)
        if limit is not None:
            running = con.execute(
                "SELECT COUNT(*) FROM jobs WHERE state='running' AND lane=?",
                (lane,)).fetchone()[0]
            if running >= limit:
                con.execute("COMMIT")
                return None
        row = con.execute(
            "SELECT id, queue, payload, attempts, dataset, stage FROM jobs "
            "WHERE state='queued' AND lane=? "
            "ORDER BY priority DESC, created ASC LIMIT 1", (lane,)).fetchone()
        if row is None:
            con.execute("COMMIT")
            return None
        jid = row[0]
        now = time.time()
        changed = con.execute(
            "UPDATE jobs SET state='running', started=?, heartbeat=?, "
            "worker=?, attempts=attempts+1 WHERE id=? AND state='queued'",
            (now, now, worker, jid)).rowcount
        con.execute("COMMIT")
        if not changed:
            return None
        return {"id": jid, "queue": row[1], "payload": json.loads(row[2]),
                "attempts": row[3] + 1, "dataset": row[4], "stage": row[5]}
    except Exception:                                            # noqa: BLE001
        try:
            con.execute("ROLLBACK")
        except Exception:                                        # noqa: BLE001
            pass
        raise
    finally:
        con.close()


class UnknownJob(KeyError):
    """An id that matches no row. Raised rather than silently doing nothing.

    Every one of these functions used to run an UPDATE that matched zero rows
    and report success. `fail("typo", ...)` returned "errored" -- a terminal
    transition, indistinguishable from a real one -- while the real job stayed
    `running` until `reclaim()` eventually found it. That is the exact shape of
    bug this whole module was written to prevent: a function reporting an
    outcome for work it did not do.
    """


def _require(con: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise UnknownJob(f"no job {job_id!r}")
    return row


def beat(job_id: str, progress: str | None = None) -> None:
    """Say the job is still alive. A long job MUST call this.

    Without it a 450-second generation looks identical to a worker that was
    killed, and `reclaim` would hand its work to somebody else while it was
    still running.
    """
    con = _db()
    try:
        _require(con, job_id)
        if progress is None:
            con.execute("UPDATE jobs SET heartbeat=? WHERE id=?",
                        (time.time(), job_id))
        else:
            con.execute("UPDATE jobs SET heartbeat=?, progress=? WHERE id=?",
                        (time.time(), progress, job_id))
    finally:
        con.close()


def finish(job_id: str, result: dict | None = None) -> None:
    con = _db()
    try:
        _require(con, job_id)
        con.execute(
            "UPDATE jobs SET state='done', finished=?, result=?, error=NULL "
            "WHERE id=?", (time.time(), json.dumps(result or {}), job_id))
    finally:
        con.close()


def fail(job_id: str, error: str, *, retry: bool = True) -> str:
    """Record a failure. Returns the state the job landed in.

    An exhausted job goes to `errored`, which is TERMINAL and is not `done`.
    That distinction is the whole reason this module exists: a resume path that
    treats an errored row as finished silently shrinks the sample and reports a
    rate over what survived.
    """
    con = _db()
    try:
        # Raise on an unknown id rather than defaulting. This used to fall into
        # `(99, 0)`, compute "errored", run an UPDATE matching zero rows, and
        # RETURN "errored" -- a caller who mistyped an id got an answer shaped
        # exactly like a real terminal transition.
        row = _require(con, job_id)
        attempts, max_attempts = row["attempts"], row["max_attempts"]
        state = "queued" if (retry and attempts < max_attempts) else "errored"
        con.execute(
            "UPDATE jobs SET state=?, error=?, finished=?, heartbeat=NULL "
            "WHERE id=?",
            (state, error[:4000],
             time.time() if state == "errored" else None, job_id))
        return state
    finally:
        con.close()


def reclaim(stale_seconds: int = STALE_SECONDS) -> int:
    """Return jobs whose worker died to the queue. Returns how many.

    Attempt counts are preserved, so a job that kills its worker three times
    lands in `errored` rather than looping forever.
    """
    cutoff = time.time() - stale_seconds
    con = _db()
    try:
        n = con.execute(
            "UPDATE jobs SET state='queued', worker=NULL, heartbeat=NULL "
            "WHERE state='running' AND (heartbeat IS NULL OR heartbeat < ?)",
            (cutoff,)).rowcount
        # `finished` is stamped here too. `fail()` sets it on its terminal
        # path; this one did not, so anything ageing or sorting errored jobs
        # saw NULL from one route and a float from the other.
        con.execute(
            "UPDATE jobs SET state='errored', finished=?, "
            "error=COALESCE(error,'')||' [worker died too many times]' "
            "WHERE state='queued' AND attempts >= max_attempts",
            (time.time(),))
        return n
    finally:
        con.close()


def _pid_alive(pid: int) -> bool:
    """Whether a process exists. NEVER os.kill(pid, 0) on Windows: there it
    is TerminateProcess, and it would kill the process it was asked about."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)    # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True                          # cannot tell: assume alive
            return code.value == 259                 # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reclaim_dead_local() -> int:
    """Return running jobs whose worker on THIS host no longer exists.

    The global lane limit in claim() means a killed worker's gpu row holds the
    only gpu slot; reclaim() frees it only after STALE_SECONDS (20 minutes).
    A worker id is "host:pid:lane-slot", so a row from this host whose pid is
    gone can be returned at once, attempts intact. Rows from other hosts are
    left to the heartbeat rule, since their pids mean nothing here.
    """
    host = socket.gethostname()
    con = _db()
    try:
        rows = con.execute(
            "SELECT id, worker FROM jobs WHERE state='running'").fetchall()
        dead = []
        for jid, w in rows:
            parts = (w or "").split(":")
            if len(parts) >= 2 and parts[0] == host:
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                if pid != os.getpid() and not _pid_alive(pid):
                    dead.append(jid)
        for jid in dead:
            con.execute("UPDATE jobs SET state='queued', worker=NULL, "
                        "heartbeat=NULL WHERE id=? AND state='running'", (jid,))
        return len(dead)
    finally:
        con.close()


def last_created(queue: str) -> float | None:
    """When the newest job of `queue` was created, in any state. A schedule
    reads this to enqueue at most once per period, failures included."""
    con = _db()
    try:
        return con.execute("SELECT MAX(created) FROM jobs WHERE queue=?",
                           (queue,)).fetchone()[0]
    finally:
        con.close()


def get(job_id: str) -> dict | None:
    con = _db()
    try:
        con.row_factory = sqlite3.Row
        r = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row(r) if r else None
    finally:
        con.close()


def listing(dataset: str | None = None, state: str | None = None,
            limit: int = 200) -> list[dict]:
    con = _db()
    try:
        con.row_factory = sqlite3.Row
        q = "SELECT * FROM jobs"
        args: list = []
        where = []
        if dataset:
            where.append("dataset=?")
            args.append(dataset)
        if state:
            where.append("state=?")
            args.append(state)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY created DESC LIMIT ?"
        args.append(limit)
        return [_row(r) for r in con.execute(q, args)]
    finally:
        con.close()


def snapshot() -> dict:
    """Counts per state and lane, for the dashboard."""
    con = _db()
    try:
        by_state = {s: 0 for s in STATES}
        for s, n in con.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state"):
            by_state[s] = n
        by_lane = {}
        for lane, s, n in con.execute(
                "SELECT lane, state, COUNT(*) FROM jobs GROUP BY lane, state"):
            by_lane.setdefault(lane, {})[s] = n
        oldest = con.execute(
            "SELECT MIN(created) FROM jobs WHERE state='queued'").fetchone()[0]
        return {"states": by_state, "lanes": by_lane,
                "lane_limits": dict(LANES),
                "paused": {ln: rec for ln in LANES if (rec := paused(ln))},
                "oldest_queued_age": (time.time() - oldest) if oldest else None,
                "db": os.path.abspath(DB)}
    finally:
        con.close()


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k in ("payload", "result"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except ValueError:
                pass
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="local durable job queue")
    ap.add_argument("command", choices=["snapshot", "list", "reclaim", "add"])
    ap.add_argument("--dataset")
    ap.add_argument("--state")
    ap.add_argument("--queue", default="noop")
    ap.add_argument("--lane", default="cpu")
    a = ap.parse_args()
    if a.command == "snapshot":
        print(json.dumps(snapshot(), indent=2))
    elif a.command == "list":
        for j in listing(a.dataset, a.state):
            print(f"  {j['id']}  {j['state']:<9} {j['lane']:<4} "
                  f"{j['queue']:<22} {j.get('stage') or '':<12} "
                  f"{j.get('dataset') or ''}")
    elif a.command == "reclaim":
        print(f"  reclaimed {reclaim()} stale job(s)")
    elif a.command == "add":
        print(add(a.queue, lane=a.lane, dataset=a.dataset))
