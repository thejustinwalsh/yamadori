#!/usr/bin/env python
"""The skill store: every skill, every version, and where each has got to.

WHAT A SKILL IS

A compressed piece of guidance -- a title, an applies-when condition and a
few DO / WHEN / DO NOT items -- distilled from a source (a URL, a pasted
SKILL.md, or a group of this repo's recipes) by `skill_pipeline`, and injected
into requests whose applies-when it matches (`skill_select`). It replaces
"hints" (operator, 2026-09-24); `hints.py` stays runnable behind
YAMADORI_RECALL=hints until the skills path is validated.

THE PIPELINE, AS DURABLE JOBS (jobs.py, the same queue datasets use)

    fetch -> screen -> screen_model -> classify -> distil -> validate -> arm
     net      cpu        gpu            cpu        gpu       cpu        cpu

    watch (net, scheduled): re-fetch; a changed sha256 starts a NEW VERSION
    at `screen`. The version being served keeps serving until the new one
    arms -- or quarantines, which disarms the whole skill.

A pasted or migrated source starts at `screen` (there is nothing to fetch).
An operator's EDIT is a new version that starts at `screen` and skips
classify and distil -- the operator wrote the skill -- and the skill is
disarmed until that version re-passes the screen and validation.

NO REVIEW STAGE (operator, 2026-09-24)

A version that passes validate ARMS: it is live on the next request. Review
is optional and after the fact, on the dashboard, where a person can edit,
disable or re-enable any skill. This departs from
docs/DECISION-TREES-AND-SKILLS.md §2.5, which proposed SHADOW for new and
changed skills until a benchmark gate passed; the operator chose to trust the
pipeline. What stands in for the shadow is the screen: a failed screen
QUARANTINES the skill, never arms it, and records why.

STATES

    version:  running -> armed -> superseded
                      -> quarantined        (the screen said stop)
                      -> failed             (a stage could not produce it)
    skill:    pipeline | armed | quarantined | failed | disabled

`status` is derived in one place (`_refresh`) from the versions and the
`enabled` flag, so no two writers can leave it inconsistent.

WHERE IT LIVES

Two tables in the jobs database (`jobs.DB`), next to `datasets`, for the
reason datasets.py gives: a skill and the jobs it spawned cannot end up in
two databases that disagree, and YAMADORI_JOBS_DB moves both in a test.
Source bytes live under YAMADORI_SKILLS_DIR (default index/skills/<id>/v<n>/).
Those paths are internal: nothing here puts them on a response
(`public()`), and x_yamadori carries ids and versions only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.environ.get("YAMADORI_SKILLS_DIR",
                       os.path.join(HERE, "..", "index", "skills"))

STAGES = ("fetch", "screen", "screen_model", "classify", "distil",
          "validate", "arm")
JOBS = {
    "fetch": ("skill.fetch", "net"),
    "screen": ("skill.screen", "cpu"),
    "screen_model": ("skill.screen_model", "gpu"),
    "classify": ("skill.classify", "cpu"),
    "distil": ("skill.distil", "gpu"),
    "validate": ("skill.validate", "cpu"),
    "arm": ("skill.arm", "cpu"),
}
WATCH = ("skill.watch", "net")

# Which stages each kind of version walks.
PATHS = {
    "url": STAGES,
    "text": STAGES[1:],
    "migration": STAGES[1:],
    "watch": STAGES[1:],
    "edit": ("screen", "screen_model", "validate", "arm"),
}

VERSION_STATES = ("running", "armed", "superseded", "quarantined", "failed")
STATUSES = ("pipeline", "armed", "quarantined", "failed", "disabled")

# Re-fetch a URL skill this often unless told otherwise. Not measured: a day
# is the operator's order of "on a schedule", and any value is one field.
WATCH_HOURS = float(os.environ.get("YAMADORI_SKILL_WATCH_HOURS", "24"))

DDL = """
CREATE TABLE IF NOT EXISTS skills(
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    title           TEXT,
    source_url      TEXT,
    source_kind     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pipeline',
    reason          TEXT,
    enabled         INTEGER NOT NULL DEFAULT 1,
    served_version  INTEGER,
    latest_version  INTEGER NOT NULL DEFAULT 0,
    watch_seconds   REAL,
    next_watch      REAL,
    meta            TEXT NOT NULL DEFAULT '{}',
    created         REAL NOT NULL,
    updated         REAL NOT NULL);
CREATE INDEX IF NOT EXISTS skills_status ON skills(status, enabled);
CREATE TABLE IF NOT EXISTS skill_versions(
    skill           TEXT NOT NULL,
    version         INTEGER NOT NULL,
    origin          TEXT NOT NULL,
    author          TEXT,
    path            TEXT NOT NULL DEFAULT '[]',
    stage           TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'running',
    reason          TEXT,
    source_sha256   TEXT,
    source_bytes    INTEGER,
    fetched         TEXT NOT NULL DEFAULT '{}',
    screen          TEXT NOT NULL DEFAULT '{}',
    classify        TEXT NOT NULL DEFAULT '{}',
    distil          TEXT NOT NULL DEFAULT '{}',
    validate        TEXT NOT NULL DEFAULT '{}',
    text            TEXT,
    text_sha256     TEXT,
    meta            TEXT NOT NULL DEFAULT '{}',
    created         REAL NOT NULL,
    updated         REAL NOT NULL,
    armed_at        REAL,
    PRIMARY KEY(skill, version));
"""
_JSON_S = ("meta",)
_JSON_V = ("path", "fetched", "screen", "classify", "distil", "validate",
           "meta")
_ENSURED: set[str] = set()


def _db() -> sqlite3.Connection:
    path = os.path.abspath(jobs.DB)
    if path not in _ENSURED:
        jobs._db().close()                 # jobs.py owns the jobs schema
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA busy_timeout=30000")
    if path not in _ENSURED:
        con.executescript(DDL)
        _ENSURED.add(path)
    con.row_factory = sqlite3.Row
    return con


def _decode(r: sqlite3.Row | None, keys) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in keys:
        try:
            d[k] = json.loads(d.get(k) or ("[]" if k == "path" else "{}"))
        except ValueError:
            d[k] = [] if k == "path" else {}
    return d


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return s[:60] or "skill"


# ---------------------------------------------------------------------------
# URLs. A GitHub page is resolved to the raw file it shows, and a GitHub
# folder to its SKILL.md: the page around a file is navigation, not source.
# ---------------------------------------------------------------------------
def normalise_url(url: str) -> str:
    u = (url or "").strip()
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/(blob|tree)/([^/]+)"
                 r"(/.*)?$", u)
    if m:
        owner, repo, kind, ref, rest = m.groups()
        rest = (rest or "").rstrip("/")
        if kind == "tree" and not re.search(r"\.\w{1,5}$", rest):
            rest = f"{rest}/SKILL.md"
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}{rest}"
    return u


# ---------------------------------------------------------------------------
# Source files: internal paths, never on a response.
# ---------------------------------------------------------------------------
def version_dir(sid: str, v: int) -> str:
    return os.path.join(os.path.abspath(STORE), sid, f"v{int(v)}")


def _write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def store_source(sid: str, v: int, raw: bytes, meta: dict) -> dict:
    """Write a version's source and record its hash. Returns the fetched
    record as stored (no local path in it)."""
    sha = hashlib.sha256(raw).hexdigest()
    rec = dict(meta or {}, sha256=sha, bytes=len(raw))
    d = version_dir(sid, v)
    _write_atomic(os.path.join(d, "source.bin"), raw)
    _write_atomic(os.path.join(d, "source.json"),
                  json.dumps(rec, indent=2).encode("utf-8"))
    update_version(sid, v, source_sha256=sha, source_bytes=len(raw),
                   fetched=rec)
    return rec


def source_raw(sid: str, v: int) -> tuple[bytes, dict] | None:
    """(bytes, fetched record) of the nearest version <= v that has a
    source: an edit has none of its own and reads its parent's."""
    for n in range(int(v), 0, -1):
        p = os.path.join(version_dir(sid, n), "source.bin")
        if os.path.exists(p):
            with open(p, "rb") as f:
                raw = f.read()
            ver = version(sid, n) or {}
            return raw, dict(ver.get("fetched") or {}, version=n)
    return None


# ---------------------------------------------------------------------------
# Create, version, transition.
# ---------------------------------------------------------------------------
def create(*, url: str | None = None, text: str | None = None,
           name: str | None = None, origin: str | None = None,
           author: str = "operator", watch_hours: float | None = None,
           meta: dict | None = None) -> dict:
    """Record a skill and enqueue its first stage. Exactly one of `url` or
    `text`. `origin` is "migration" for the recipe migration; otherwise it
    follows from the source kind."""
    url = (url or "").strip()
    text = text if text is None else str(text)
    if bool(url) == bool(text and text.strip()):
        raise ValueError("give exactly one of a URL or the skill text")
    if url and not re.match(r"^https?://\S+$", url):
        raise ValueError(f"{url[:120]!r} is not one http(s) URL; paste the "
                         "text instead")
    kind = "url" if url else "text"
    vkind = origin if origin in ("migration",) else kind
    url = normalise_url(url) if url else None
    sid = uuid.uuid4().hex[:12]
    now = time.time()
    nm = slug(name or (re.sub(r"^https?://", "", url).split("?")[0]
                       if url else " ".join((text or "").split()[:6])))
    watch = None
    if url:
        hours = WATCH_HOURS if watch_hours is None else float(watch_hours)
        watch = hours * 3600 if hours > 0 else None
    con = _db()
    try:
        con.execute(
            "INSERT INTO skills(id,name,source_url,source_kind,status,"
            "latest_version,watch_seconds,next_watch,meta,created,updated) "
            "VALUES(?,?,?,?,'pipeline',1,?,?,?,?,?)",
            (sid, nm, url, vkind, watch, (now + watch) if watch else None,
             json.dumps(meta or {}), now, now))
        path = PATHS[vkind]
        con.execute(
            "INSERT INTO skill_versions(skill,version,origin,author,path,"
            "stage,state,meta,created,updated) "
            "VALUES(?,1,?,?,?,?,'running',?,?,?)",
            (sid, "ingest" if vkind in ("url", "text") else vkind, author,
             json.dumps(list(path)), path[0], json.dumps(meta or {}), now,
             now))
    finally:
        con.close()
    if text is not None and not url:
        store_source(sid, 1, text.encode("utf-8"),
                     {"kind": "text" if vkind == "text" else vkind,
                      "content_type": "text/markdown",
                      "received_at": now})
    enqueue(sid, 1, PATHS[vkind][0])
    return get(sid)


def new_version(sid: str, origin: str, *, author: str = "pipeline",
                meta: dict | None = None, **fields) -> int:
    """Add version latest+1 in state running at the first stage of its
    path. Does NOT enqueue: the caller stores the source first."""
    if origin not in PATHS:
        raise ValueError(f"unknown origin {origin!r}")
    now = time.time()
    con = _db()
    try:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT latest_version FROM skills WHERE id=?",
                        (sid,)).fetchone()
        if r is None:
            con.execute("ROLLBACK")
            raise KeyError(f"no such skill: {sid}")
        v = int(r[0]) + 1
        path = PATHS[origin]
        cols = {"skill": sid, "version": v, "origin": origin,
                "author": author, "path": json.dumps(list(path)),
                "stage": path[0], "state": "running",
                "meta": json.dumps(meta or {}), "created": now,
                "updated": now}
        for k, val in fields.items():
            cols[k] = json.dumps(val) if k in _JSON_V else val
        con.execute(f"INSERT INTO skill_versions({','.join(cols)}) VALUES("
                    f"{','.join('?' * len(cols))})", list(cols.values()))
        con.execute("UPDATE skills SET latest_version=?, updated=? WHERE id=?",
                    (v, now, sid))
        _refresh(con, sid)
        con.execute("COMMIT")
    finally:
        con.close()
    return v


def enqueue(sid: str, v: int, stage: str) -> str:
    queue, lane = JOBS[stage]
    return jobs.add(queue, {"skill": sid, "version": int(v), "stage": stage},
                    lane=lane, dataset=f"skill:{sid}", stage=stage)


def next_stage(ver: dict) -> str | None:
    path = ver.get("path") or []
    if ver.get("stage") not in path:
        return None
    i = path.index(ver["stage"])
    return path[i + 1] if i + 1 < len(path) else None


def advance(sid: str, v: int, done_stage: str) -> str | None:
    """After `done_stage`'s job finished: move a still-running version to
    its next stage and enqueue that stage's job. None when the version has
    stopped (quarantined, failed, armed) or already moved."""
    ver = version(sid, v)
    if not ver or ver["state"] != "running" or ver["stage"] != done_stage:
        return None
    nxt = next_stage(ver)
    if nxt is None:
        return None
    update_version(sid, v, stage=nxt)
    enqueue(sid, v, nxt)
    return nxt


def update_version(sid: str, v: int, **fields) -> None:
    if not fields:
        return
    sets, args = [], []
    for k, val in fields.items():
        sets.append(f"{k}=?")
        args.append(json.dumps(val) if k in _JSON_V else val)
    args += [time.time(), sid, int(v)]
    con = _db()
    try:
        con.execute(f"UPDATE skill_versions SET {','.join(sets)},updated=? "
                    "WHERE skill=? AND version=?", args)
    finally:
        con.close()


def _refresh(con: sqlite3.Connection, sid: str) -> None:
    """Derive `status` and `reason` from the versions and `enabled`. The one
    place status is written."""
    s = con.execute("SELECT enabled, served_version, latest_version FROM "
                    "skills WHERE id=?", (sid,)).fetchone()
    if s is None:
        return
    latest = con.execute("SELECT state, reason, stage FROM skill_versions "
                         "WHERE skill=? AND version=?",
                         (sid, s["latest_version"])).fetchone()
    reason = None
    if not s["enabled"]:
        status = "disabled"
        m = con.execute("SELECT meta FROM skills WHERE id=?", (sid,)).fetchone()
        reason = (json.loads(m["meta"] or "{}").get("disabled") or {}).get(
            "reason")
    elif s["served_version"]:
        status = "armed"
        if latest and latest["state"] in ("running", "failed") \
                and s["latest_version"] != s["served_version"]:
            reason = (f"v{s['latest_version']} is {latest['state']}"
                      + (f" at {latest['stage']}" if latest["state"] ==
                         "running" else f": {latest['reason']}")
                      + f"; v{s['served_version']} serves meanwhile")
    elif latest and latest["state"] == "quarantined":
        status, reason = "quarantined", latest["reason"]
    elif latest and latest["state"] == "failed":
        status, reason = "failed", latest["reason"]
    else:
        status = "pipeline"
        if latest:
            reason = f"v{s['latest_version']} is at {latest['stage']}"
    con.execute("UPDATE skills SET status=?, reason=?, updated=? WHERE id=?",
                (status, reason, time.time(), sid))


def _with(sid: str, fn) -> None:
    con = _db()
    try:
        con.execute("BEGIN IMMEDIATE")
        fn(con)
        _refresh(con, sid)
        con.execute("COMMIT")
    except Exception:
        try:
            con.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        con.close()
    _invalidate()


def quarantine(sid: str, v: int, reason: str, **fields) -> None:
    """The screen said stop. The version never arms, and the SKILL is
    disarmed: a source that turned hostile has told us something about its
    author, including about the version we are serving."""
    def fn(con):
        sets = ["state='quarantined'", "reason=?", "updated=?"]
        args = [reason[:2000], time.time()]
        for k, val in fields.items():
            sets.append(f"{k}=?")
            args.append(json.dumps(val) if k in _JSON_V else val)
        con.execute(f"UPDATE skill_versions SET {','.join(sets)} "
                    "WHERE skill=? AND version=?", args + [sid, int(v)])
        con.execute("UPDATE skill_versions SET state='superseded' WHERE "
                    "skill=? AND state='armed'", (sid,))
        con.execute("UPDATE skills SET served_version=NULL WHERE id=?", (sid,))
    _with(sid, fn)


def fail(sid: str, v: int, reason: str, **fields) -> None:
    """A stage could not produce this version. A version already serving
    keeps serving: a re-distil that failed is not evidence against it."""
    def fn(con):
        sets = ["state='failed'", "reason=?", "updated=?"]
        args = [reason[:2000], time.time()]
        for k, val in fields.items():
            sets.append(f"{k}=?")
            args.append(json.dumps(val) if k in _JSON_V else val)
        con.execute(f"UPDATE skill_versions SET {','.join(sets)} "
                    "WHERE skill=? AND version=?", args + [sid, int(v)])
    _with(sid, fn)


def arm(sid: str, v: int) -> dict:
    """Make version v the one served. Immediately: there is no review stage."""
    ver = version(sid, v)
    if ver is None:
        raise KeyError(f"no such version: {sid} v{v}")
    if ver["state"] != "running" or not ver.get("text"):
        raise ValueError(f"v{v} is {ver['state']} and "
                         f"{'has' if ver.get('text') else 'has no'} text; "
                         "only a running version with validated text arms")

    def fn(con):
        now = time.time()
        con.execute("UPDATE skill_versions SET state='superseded', updated=? "
                    "WHERE skill=? AND state='armed'", (now, sid))
        con.execute("UPDATE skill_versions SET state='armed', stage='arm', "
                    "armed_at=?, updated=? WHERE skill=? AND version=?",
                    (now, now, sid, int(v)))
        title = (ver.get("validate") or {}).get("title") or ver["text"] \
            .split("\n", 1)[0]
        con.execute("UPDATE skills SET served_version=?, title=? WHERE id=?",
                    (int(v), title[:200], sid))
    _with(sid, fn)
    return get(sid)


def disable(sid: str, *, reason: str = "", author: str = "operator") -> dict:
    _require(sid)

    def fn(con):
        m = json.loads(con.execute("SELECT meta FROM skills WHERE id=?",
                                   (sid,)).fetchone()["meta"] or "{}")
        m["disabled"] = {"by": author, "at": time.time(),
                         "reason": (reason or "disabled on the dashboard")[:300]}
        con.execute("UPDATE skills SET enabled=0, meta=? WHERE id=?",
                    (json.dumps(m), sid))
    _with(sid, fn)
    return get(sid)


def enable(sid: str, *, author: str = "operator") -> dict:
    """Re-enable. The served version already passed the screen, so it serves
    again at once; nothing is re-screened."""
    _require(sid)

    def fn(con):
        m = json.loads(con.execute("SELECT meta FROM skills WHERE id=?",
                                   (sid,)).fetchone()["meta"] or "{}")
        m["enabled"] = {"by": author, "at": time.time()}
        m.pop("disabled", None)
        con.execute("UPDATE skills SET enabled=1, meta=? WHERE id=?",
                    (json.dumps(m), sid))
    _with(sid, fn)
    return get(sid)


def set_watch(sid: str, hours: float | None) -> dict:
    s = _require(sid)
    if not s.get("source_url"):
        raise ValueError("only a skill with a source URL can be watched")
    secs = float(hours) * 3600 if hours and float(hours) > 0 else None
    con = _db()
    try:
        con.execute("UPDATE skills SET watch_seconds=?, next_watch=?, "
                    "updated=? WHERE id=?",
                    (secs, (time.time() + secs) if secs else None, time.time(),
                     sid))
    finally:
        con.close()
    return get(sid)


def edit(sid: str, text: str, *, author: str = "operator") -> dict:
    """An operator's edit: a new version, re-screened before it is re-armed.

    The skill is DISARMED now and serves nothing until the edited version
    passes screen -> screen_model -> validate -> arm. Watching stops: a
    re-fetch would otherwise distil the source again and replace what the
    operator wrote (set_watch turns it back on).
    """
    import skill_builder
    import skill_classify
    s = _require(sid)
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("the edit is empty")
    if len(text) > 4 * skill_builder.MAX_SKILL_CHARS:
        raise ValueError(f"the edit is {len(text)} characters; a skill is at "
                         f"most {skill_builder.MAX_SKILL_CHARS}")
    parsed = skill_builder.parse(text)
    served = version(sid, s["served_version"]) if s.get("served_version") \
        else latest_with_text(sid)
    old_rule = (served or {}).get("classify") or {}
    cond = parsed.get("applies_when") or ""
    if cond and skill_builder._norm(cond) == skill_builder._norm(
            old_rule.get("text") or ""):
        rule = old_rule
    else:
        rule = skill_classify.rule_of_condition(cond) if cond else {}
    known = {}
    for it in ((served or {}).get("validate") or {}).get("items") or []:
        known[skill_builder.item_line(it)] = it.get("quote") or ""
    v = new_version(sid, "edit", author=author,
                    distil={"parsed": parsed, "operator": True,
                            "submitted": text, "known_quotes": known},
                    classify=rule, meta={"edited_from": s.get(
                        "served_version") or s.get("latest_version")})

    def fn(con):
        con.execute("UPDATE skill_versions SET state='superseded' WHERE "
                    "skill=? AND state='armed'", (sid,))
        m = json.loads(con.execute("SELECT meta FROM skills WHERE id=?",
                                   (sid,)).fetchone()["meta"] or "{}")
        if s.get("watch_seconds"):
            m["watch_paused"] = {"by": author, "at": time.time(),
                                 "why": f"v{v} is an operator edit; "
                                        "re-enable watching to follow the "
                                        "source again"}
        con.execute("UPDATE skills SET served_version=NULL, watch_seconds="
                    "NULL, next_watch=NULL, meta=? WHERE id=?",
                    (json.dumps(m), sid))
    _with(sid, fn)
    enqueue(sid, v, PATHS["edit"][0])
    return get(sid)


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------
def _require(sid: str) -> dict:
    s = get(sid)
    if s is None:
        raise KeyError(f"no such skill: {sid}")
    return s


def get(sid: str) -> dict | None:
    con = _db()
    try:
        return _decode(con.execute("SELECT * FROM skills WHERE id=?",
                                   (sid,)).fetchone(), _JSON_S)
    finally:
        con.close()


def version(sid: str, v: int | None) -> dict | None:
    if v is None:
        return None
    con = _db()
    try:
        return _decode(con.execute(
            "SELECT * FROM skill_versions WHERE skill=? AND version=?",
            (sid, int(v))).fetchone(), _JSON_V)
    finally:
        con.close()


def versions(sid: str) -> list[dict]:
    con = _db()
    try:
        return [_decode(r, _JSON_V) for r in con.execute(
            "SELECT * FROM skill_versions WHERE skill=? ORDER BY version DESC",
            (sid,))]
    finally:
        con.close()


def latest_with_text(sid: str) -> dict | None:
    for ver in versions(sid):
        if ver.get("text"):
            return ver
    return None


def listing(limit: int = 500) -> list[dict]:
    con = _db()
    try:
        return [_decode(r, _JSON_S) for r in con.execute(
            "SELECT * FROM skills ORDER BY updated DESC LIMIT ?", (limit,))]
    finally:
        con.close()


def find(name: str) -> dict | None:
    con = _db()
    try:
        return _decode(con.execute("SELECT * FROM skills WHERE name=? "
                                   "ORDER BY created LIMIT 1",
                                   (name,)).fetchone(), _JSON_S)
    finally:
        con.close()


def counts() -> dict:
    con = _db()
    try:
        out = {s: 0 for s in STATUSES}
        for st, n in con.execute("SELECT status, COUNT(*) FROM skills "
                                 "GROUP BY status"):
            out[st] = n
        return out
    finally:
        con.close()


def skill_jobs(sid: str, limit: int = 50) -> list[dict]:
    return [{k: j.get(k) for k in ("id", "queue", "lane", "state", "stage",
                                   "attempts", "max_attempts", "error",
                                   "progress", "created", "finished")}
            for j in jobs.listing(dataset=f"skill:{sid}", limit=limit)]


# ---------------------------------------------------------------------------
# What is served. Cached for a few seconds: selection runs on every request,
# and one indexed SELECT per request is cheap but not free on a busy file.
# ---------------------------------------------------------------------------
CACHE_SECONDS = float(os.environ.get("YAMADORI_SKILL_CACHE_SECONDS", "5"))
_LOCK = threading.Lock()
_CACHE: dict = {"at": 0.0, "db": None, "rows": []}


def _invalidate() -> None:
    with _LOCK:
        _CACHE["at"] = 0.0


def armed() -> list[dict]:
    """Every enabled skill with a served version: id, version, title, text,
    rule, source name. What selection reads."""
    now = time.time()
    db = os.path.abspath(jobs.DB)
    with _LOCK:
        if _CACHE["db"] == db and now - _CACHE["at"] < CACHE_SECONDS:
            return list(_CACHE["rows"])
    con = _db()
    try:
        rows = []
        for r in con.execute(
                "SELECT s.id, s.name, s.source_url, s.source_kind, "
                "v.version, v.text, v.classify, v.validate "
                "FROM skills s JOIN skill_versions v "
                "ON v.skill=s.id AND v.version=s.served_version "
                "WHERE s.enabled=1 AND s.served_version IS NOT NULL "
                "AND v.state='armed' ORDER BY s.id"):
            val = json.loads(r["validate"] or "{}")
            rows.append({"id": r["id"], "version": r["version"],
                         "name": r["name"], "text": r["text"] or "",
                         "title": val.get("title") or "",
                         "items": val.get("items") or [],
                         "rule": json.loads(r["classify"] or "{}"),
                         "source_kind": r["source_kind"],
                         "source_url": r["source_url"]})
    finally:
        con.close()
    with _LOCK:
        _CACHE.update(at=now, db=db, rows=rows)
    return list(rows)


def served_texts() -> list[tuple[str, int, str]]:
    """(id, version, text) of every served skill, for a leak preflight
    (bench/domain/run.py's shingle check covers bench/recipes only)."""
    return [(s["id"], s["version"], s["text"]) for s in armed()]


# ---------------------------------------------------------------------------
# The public view: what the dashboard API returns. No filesystem path.
# ---------------------------------------------------------------------------
def public_version(ver: dict) -> dict:
    d = {k: ver.get(k) for k in ("version", "origin", "author", "path",
                                 "stage", "state", "reason", "source_sha256",
                                 "source_bytes", "fetched", "screen",
                                 "classify", "validate", "text", "created",
                                 "updated", "armed_at", "meta")}
    dist = ver.get("distil") or {}
    d["distil"] = {k: dist.get(k) for k in ("chunks", "replies_chars",
                                            "operator", "counts", "notes")
                   if k in dist}
    return d


def public(s: dict, *, detail: bool = False) -> dict:
    out = {k: s.get(k) for k in ("id", "name", "title", "source_url",
                                 "source_kind", "status", "reason", "enabled",
                                 "served_version", "latest_version",
                                 "watch_seconds", "next_watch", "created",
                                 "updated")}
    out["enabled"] = bool(out["enabled"])
    meta = s.get("meta") or {}
    out["meta"] = {k: meta[k] for k in ("disabled", "enabled", "watch_paused",
                                        "migration_group", "watch_last")
                   if k in meta}
    served = version(s["id"], s.get("served_version")) \
        if s.get("served_version") else None
    shown = served or latest_with_text(s["id"])
    out["text"] = (shown or {}).get("text")
    out["text_version"] = (shown or {}).get("version")
    rule = (shown or version(s["id"], s["latest_version"]) or {}).get(
        "classify") or {}
    out["applies_when"] = rule.get("text")
    out["applies_to"] = rule.get("applies_to")
    out["triggers"] = [t.get("text") for t in rule.get("triggers") or []
                       if isinstance(t, dict)]
    if detail:
        import skill_learn
        out["learned_triggers"] = skill_learn.learned_triggers({s["id"]}) \
            .get(s["id"], [])
    if detail:
        out["versions"] = [public_version(v) for v in versions(s["id"])]
        out["jobs"] = skill_jobs(s["id"])
    return out


if __name__ == "__main__":
    print(f"  db    {os.path.abspath(jobs.DB)}")
    print(f"  store {os.path.abspath(STORE)}")
    print(f"  {counts()}")
    for s in listing():
        print(f"  {s['id']}  {s['status']:<11} v{s['served_version'] or '-'}"
              f"/{s['latest_version']}  {(s['title'] or s['name'])[:60]}")
