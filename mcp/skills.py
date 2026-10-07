#!/usr/bin/env python
"""The skill store: every skill, every version, and where each has got to.

SKILLS ARE THE ONE KNOWLEDGE SYSTEM (operator, 2026-09-25/26): "It's time
to retire hints and build the skills system ... because skills are proven in
the greater ecosystem." "We don't have rules to fix bugs, we have skills."
The recipe corpus the old hints path served was migrated into skills
(mcp/skill_migrate.py); that path, its vector cache and its switch are
gone.

WHAT A SKILL IS

An Agent Skills folder (mcp/skill_md.py): a SKILL.md -- frontmatter with a
`name`, a `description` that IS its trigger condition, and our metadata
under `metadata.yamadori` -- whose body is a title and a handful of DO /
WHEN / DO NOT items, plus a tests.json of activation tests. ATOMIC: one
situation, one trigger, sized for this model's context (skill_limits).
Injected into the requests its applies-when rule matches (skill_select).

ONE PIPELINE (skill_pipeline.py; durable jobs on jobs.py's queue)

    fetch -> screen -> screen_model -> licence -> distil -> review ->
      net     cpu        gpu            net        gpu       gpu
    classify -> tests -> validate -> prove -> arm
      gpu        gpu       gpu        gpu     cpu

    a frontier SKILL.md:    ... -> licence -> decompose, which creates one
                            CHILD skill per atomic part; each child walks
                            review -> classify -> tests -> validate ->
                            prove -> arm
    compiled (migration, authored, a dataset's rows): the items are given;
                            screen -> review -> classify -> tests ->
                            validate -> prove -> arm (review and prove need
                            the model: an INLINE install records them as
                            not run)
    watch (net, scheduled): re-fetch; a changed sha256 starts a NEW VERSION
                            at `screen`. The version being served keeps
                            serving until the new one arms -- or
                            quarantines, which disarms the whole skill.
    edit:                   screen -> screen_model -> review -> classify ->
                            tests -> validate -> prove -> arm; the skill is
                            disarmed meanwhile

NO HUMAN REVIEW STAGE (operator, 2026-09-24); AN AUTOMATIC ONE (2026-09-28)

A version that passes validate and prove ARMS: it is live on the next
request. A person's review is optional and after the fact, on the dashboard
(NAEDOKO, the skill factory), where a person can edit, disable, archive,
quarantine, re-run a stage or re-enable any skill. The `review` stage is the
model's second pass over the draft against the definition -- "All of our
skills increase confidence and improve correctness; if it can't, then the
line doesn't need to exist" (operator, 2026-09-28) -- keeping, rewriting or
dropping each item, verified by code (skill_pipeline.handle_review). The
`prove` stage runs paired probes with and without the skill
(skill_prove.py) and quarantines a skill that makes one worse. A failed
screen, failed activation tests or a worse probe QUARANTINE the skill,
never arm it, and record why.

STATES

    version:  running -> armed -> superseded
                      -> quarantined        (the screen or the tests said stop)
                      -> failed             (a stage could not produce it)
                      -> decomposed         (a frontier source, split into
                                             child skills)
    skill:    pipeline | armed | quarantined | failed | disabled | archived
              | decomposed

`status` is derived in one place (`_refresh`) from the versions and the
`enabled` / `archived` flags, so no two writers can leave it inconsistent.

WHERE IT LIVES

Two tables in the jobs database (`jobs.DB`), next to `datasets`, for the
reason datasets.py gives: a skill and the jobs it spawned cannot end up in
two databases that disagree, and YAMADORI_JOBS_DB moves both in a test.
Files under YAMADORI_SKILLS_DIR (default index/skills/):

    library/<name>/SKILL.md, tests.json   every skill with a validated
                                          version, its state in metadata --
                                          the harness-portable form
    _sources/<id>/v<n>/source.bin, .json  what each version was made from

Those paths are internal: nothing here puts them on a response (`public()`),
and x_yamadori carries ids, versions and names only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# A test that moves the jobs DB (YAMADORI_JOBS_DB) moves the store beside
# it unless it names one: a skill row and its files cannot land in two
# places, and no test can write the live store by forgetting a variable.
STORE = os.environ.get("YAMADORI_SKILLS_DIR") or (
    os.path.join(os.path.dirname(os.path.abspath(
        os.environ["YAMADORI_JOBS_DB"])), "skills")
    if os.environ.get("YAMADORI_JOBS_DB") else
    os.path.join(HERE, "..", "index", "skills"))

STAGES = ("fetch", "screen", "screen_model", "licence", "distil",
          "decompose", "review", "classify", "tests", "validate", "prove",
          "arm")
JOBS = {
    "fetch": ("skill.fetch", "net"),
    "screen": ("skill.screen", "cpu"),
    "screen_model": ("skill.screen_model", "gpu"),
    "licence": ("skill.licence", "net"),
    "distil": ("skill.distil", "gpu"),
    "decompose": ("skill.decompose", "gpu"),
    "review": ("skill.review", "gpu"),
    "classify": ("skill.classify", "gpu"),
    "tests": ("skill.tests", "gpu"),
    "validate": ("skill.validate", "gpu"),
    "prove": ("skill.prove", "gpu"),
    "arm": ("skill.arm", "cpu"),
}
WATCH = ("skill.watch", "net")

# REVIEW (operator, 2026-09-28: "have an auto review agent in a second pass
# to ensure it doesn't happen again, and we get the best skill reduction
# without [losing] actionable information") reads the draft against the
# definition before it is tagged; PROVE ("each skill needs proof that it
# works ... a mini A/B test to ensure it doesn't make a task worse") runs
# the paired probes on the final text before it arms.
_MODEL_TAIL = ("review", "classify", "tests", "validate", "prove", "arm")
# Which stages each kind of version walks.
PATHS = {
    "url": ("fetch", "screen", "screen_model", "licence", "distil")
    + _MODEL_TAIL,
    "text": ("screen", "screen_model", "licence", "distil") + _MODEL_TAIL,
    "frontier": ("fetch", "screen", "screen_model", "licence", "decompose"),
    "frontier_text": ("screen", "screen_model", "licence", "decompose"),
    "decomposed": _MODEL_TAIL,
    "migration": ("screen",) + _MODEL_TAIL,
    "authored": ("screen",) + _MODEL_TAIL,
    "dataset": ("screen",) + _MODEL_TAIL,
    "watch": ("screen", "screen_model", "licence", "distil") + _MODEL_TAIL,
    # A watched FRONTIER source that changed goes back through decompose
    # (2026-09-27; it used to take "watch" and be distilled into one skill):
    # handle_decompose supersedes the earlier version's children.
    "watch_frontier": ("screen", "screen_model", "licence", "decompose"),
    "edit": ("screen", "screen_model") + _MODEL_TAIL,
    # A served skill rebuilt through the review (mcp/skill_rebuild.py,
    # operator 2026-09-28): its items are the draft, its source and quotes
    # stay; the served version serves until this one arms.
    "rebuild": _MODEL_TAIL,
}
# Versions whose items arrive written (the pipeline's model stages are
# skipped inside classify / tests / validate for them).
COMPILED = ("migration", "authored", "dataset")

VERSION_STATES = ("running", "armed", "superseded", "quarantined", "failed",
                  "decomposed")
STATUSES = ("pipeline", "armed", "quarantined", "failed", "disabled",
            "archived", "decomposed")

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
# Columns added after the first schema: (table, column, DDL).
_ADDED = (("skill_versions", "licence", "TEXT NOT NULL DEFAULT '{}'"),
          ("skill_versions", "tests", "TEXT NOT NULL DEFAULT '{}'"),
          # 2026-09-28: the review stage's before/after, the prove stage's
          # paired probes (skill_pipeline.handle_review, skill_prove).
          ("skill_versions", "review", "TEXT NOT NULL DEFAULT '{}'"),
          ("skill_versions", "prove", "TEXT NOT NULL DEFAULT '{}'"))
_JSON_S = ("meta",)
_JSON_V = ("path", "fetched", "screen", "classify", "distil", "validate",
           "meta", "licence", "tests", "review", "prove")
_ENSURED: set[str] = set()


def _db() -> sqlite3.Connection:
    path = os.path.abspath(jobs.DB)
    if path not in _ENSURED:
        jobs._db().close()                 # jobs.py owns the jobs schema
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA busy_timeout=30000")
    if path not in _ENSURED:
        con.executescript(DDL)
        for table, col, ddl in _ADDED:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        _ENSURED.add(path)
    con.row_factory = sqlite3.Row
    return con


def _decode(r: sqlite3.Row | None, keys) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in keys:
        if k not in d:
            continue
        try:
            d[k] = json.loads(d.get(k) or ("[]" if k == "path" else "{}"))
        except ValueError:
            d[k] = [] if k == "path" else {}
    return d


def slug(name: str) -> str:
    """The store's display slug (underscores). A SKILL.md name is
    skill_md.to_name (hyphens)."""
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


def is_skill_md(text: str) -> bool:
    """A pasted or fetched document that is itself a SKILL.md (frontmatter
    with a name and a description): the frontier path decomposes it."""
    import skill_md
    fm, _body = skill_md.split(text or "")
    return bool(fm.get("name")) and bool(fm.get("description"))


# ---------------------------------------------------------------------------
# Files: internal paths, never on a response.
# ---------------------------------------------------------------------------
def library_dir() -> str:
    return os.path.join(os.path.abspath(STORE), "library")


def version_dir(sid: str, v: int) -> str:
    return os.path.join(os.path.abspath(STORE), "_sources", sid, f"v{int(v)}")


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
                  json.dumps(rec, indent=2, sort_keys=True).encode("utf-8"))
    update_version(sid, v, source_sha256=sha, source_bytes=len(raw),
                   fetched=rec)
    return rec


def source_raw(sid: str, v: int) -> tuple[bytes, dict] | None:
    """(bytes, fetched record) of the nearest version <= v that has a
    source: an edit has none of its own and reads its parent's. A
    decomposed child reads its PARENT skill's source (meta.parent)."""
    for n in range(int(v), 0, -1):
        p = os.path.join(version_dir(sid, n), "source.bin")
        if os.path.exists(p):
            with open(p, "rb") as f:
                raw = f.read()
            ver = version(sid, n) or {}
            return raw, dict(ver.get("fetched") or {}, version=n)
    s = get(sid) or {}
    par = (s.get("meta") or {}).get("parent") or {}
    if par.get("skill") and par.get("skill") != sid:
        return source_raw(par["skill"], int(par.get("version") or 1))
    return None


# ---------------------------------------------------------------------------
# Create, version, transition.
# ---------------------------------------------------------------------------
def _insert(sid: str, name: str, kind: str, vorigin: str, path: tuple,
            *, url: str | None, author: str, meta: dict,
            watch: float | None, version_fields: dict | None = None) -> None:
    now = time.time()
    con = _db()
    try:
        con.execute(
            "INSERT INTO skills(id,name,source_url,source_kind,status,"
            "latest_version,watch_seconds,next_watch,meta,created,updated) "
            "VALUES(?,?,?,?,'pipeline',1,?,?,?,?,?)",
            (sid, name, url, kind, watch, (now + watch) if watch else None,
             json.dumps(meta or {}), now, now))
        cols = {"skill": sid, "version": 1, "origin": vorigin,
                "author": author, "path": json.dumps(list(path)),
                "stage": path[0], "state": "running",
                "meta": json.dumps(meta or {}), "created": now,
                "updated": now}
        for k, val in (version_fields or {}).items():
            cols[k] = json.dumps(val) if k in _JSON_V else val
        con.execute(f"INSERT INTO skill_versions({','.join(cols)}) VALUES("
                    f"{','.join('?' * len(cols))})", list(cols.values()))
    finally:
        con.close()


def create(*, url: str | None = None, text: str | None = None,
           name: str | None = None, origin: str | None = None,
           author: str = "operator", watch_hours: float | None = None,
           meta: dict | None = None, goal: str = "",
           frontier: bool | None = None,
           enqueue_first: bool = True) -> dict:
    """Record a skill and enqueue its first stage. Exactly one of `url` or
    `text`. A source that is itself a SKILL.md (or `frontier=True`) walks
    the frontier path and is DECOMPOSED into atomic skills; anything else is
    DISTILLED into one. `goal` ("make a skill for X") steers the model
    stages. `enqueue_first=False` queues nothing: the caller runs the stages
    itself (skill_pipeline.run_inline), so no worker ever claims them."""
    url = (url or "").strip()
    text = text if text is None else str(text)
    if bool(url) == bool(text and text.strip()):
        raise ValueError("give exactly one of a URL or the skill text")
    if url and not re.match(r"^https?://\S+$", url):
        raise ValueError(f"{url[:120]!r} is not one http(s) URL; paste the "
                         "text instead")
    kind = "url" if url else "text"
    if frontier is None:
        frontier = bool(text and is_skill_md(text)) or bool(
            url and re.search(r"/SKILL\.md$|github\.com/.+/tree/", url))
    if frontier:
        vkind = "frontier" if url else "frontier_text"
    else:
        vkind = kind
    url = normalise_url(url) if url else None
    sid = uuid.uuid4().hex[:12]
    nm = slug(name or (re.sub(r"^https?://", "", url).split("?")[0]
                       if url else " ".join((text or "").split()[:6])))
    watch = None
    if url:
        hours = WATCH_HOURS if watch_hours is None else float(watch_hours)
        watch = hours * 3600 if hours > 0 else None
    m = dict(meta or {})
    if goal:
        m["goal"] = str(goal)[:300]
    if name and str(name).strip():
        # THE NAME THE CALLER ASKED FOR (2026-10-07): the pipeline keeps it
        # as the SKILL.md name (skill_pipeline._name_for) instead of the
        # model's, and a lookup by it finds the skill under every spelling
        # (`find`, the served row's `alias`). The pitfall cases, the gap
        # lists and the dashboard name skills; a rename in the middle of the
        # pipeline made them unfindable.
        import skill_md
        m["name_requested"] = skill_md.to_name(str(name))
    _insert(sid, nm, "frontier" if frontier else kind,
            "ingest" if vkind in ("url", "text") else vkind, PATHS[vkind],
            url=url, author=author, meta=m, watch=watch)
    if text is not None and not url:
        store_source(sid, 1, text.encode("utf-8"),
                     {"kind": "frontier" if frontier else "text",
                      "content_type": "text/markdown",
                      "received_at": time.time()})
    if enqueue_first:
        enqueue(sid, 1, PATHS[vkind][0])
    return get(sid)


def create_compiled(*, skill: dict, rule: dict, tests: dict, source: str,
                    origin: str, author: str, meta: dict | None = None,
                    sid: str | None = None, run: bool = True,
                    enqueue_first: bool = False) -> dict:
    """A skill whose items are already written (a migrated recipe group,
    an authored SKILL.md, a dataset's rows): it walks screen -> classify ->
    tests -> validate -> arm. `sid` makes the id deterministic (the
    migration derives it from the name, so a re-run makes the same store).
    `run` runs the stages inline now (no worker needed); `enqueue_first`
    queues them for the worker instead."""
    if origin not in COMPILED:
        raise ValueError(f"origin must be one of {COMPILED}")
    sid = sid or uuid.uuid4().hex[:12]
    if get(sid) is not None:
        raise ValueError(f"skill {sid} already exists")
    m = dict(meta or {})
    _insert(sid, skill["name"], origin, origin, PATHS[origin], url=None,
            author=author, meta=m, watch=None,
            version_fields={"distil": {"compiled": True, "skill": skill},
                            "classify": rule, "tests": tests})
    store_source(sid, 1, source.encode("utf-8"),
                 {"kind": origin, "content_type": "text/markdown",
                  "received_at": 0 if origin == "migration" else time.time()})
    if enqueue_first:
        enqueue(sid, 1, PATHS[origin][0])
    elif run:
        import skill_pipeline
        skill_pipeline.run_inline(sid, 1)
    return get(sid)


def create_child(parent: str, pv: int, *, parsed: dict, section: str,
                 author: str = "pipeline:decompose",
                 enqueue_first: bool = True) -> dict:
    """One atomic skill decomposed out of a frontier source. Its quotes are
    checked against the PARENT's source (meta.parent)."""
    sid = uuid.uuid4().hex[:12]
    pmeta = (get(parent) or {}).get("meta") or {}
    meta = {"parent": {"skill": parent, "version": int(pv),
                       "section": section[:200]}}
    for k in ("goal",) + ONBOARDING_KEYS:
        if pmeta.get(k):
            meta[k] = pmeta[k]
    import skill_md
    name = slug(parsed.get("name") or parsed.get("title") or "skill")
    _insert(sid, name, "decomposed", "decomposed", PATHS["decomposed"],
            url=None, author=author, meta=meta, watch=None,
            version_fields={"distil": {"parsed": parsed,
                                       "name": skill_md.to_name(name),
                                       "section": section[:200]}})
    if enqueue_first:
        enqueue(sid, 1, PATHS["decomposed"][0])
    return get(sid)


def new_child_version(sid: str, parent: str, pv: int, *, parsed: dict,
                      section: str, author: str = "pipeline:decompose",
                      enqueue_first: bool = True) -> int:
    """A later version of the parent source re-proposed this child: a new
    version of it (classify -> tests -> validate -> arm), its quotes checked
    against parent version `pv` (meta.parent is repointed). The served
    version keeps serving until this one arms (arm supersedes it)."""
    import skill_md
    name = slug(parsed.get("name") or parsed.get("title") or "skill")
    v = new_version(sid, "decomposed", author=author,
                    meta={"parent_version": int(pv)},
                    distil={"parsed": parsed, "name": skill_md.to_name(name),
                            "section": section[:200]})

    pmeta = (get(parent) or {}).get("meta") or {}

    def fn(con):
        m = json.loads(con.execute("SELECT meta FROM skills WHERE id=?",
                                   (sid,)).fetchone()["meta"] or "{}")
        m["parent"] = {"skill": parent, "version": int(pv),
                       "section": section[:200]}
        # A new version of a package's source re-proposed this child: it now
        # belongs to that onboarding (whose JOIN waits for it) and version.
        for k in ONBOARDING_KEYS:
            if pmeta.get(k):
                m[k] = pmeta[k]
        con.execute("UPDATE skills SET meta=? WHERE id=?",
                    (json.dumps(m), sid))
    _with(sid, fn)
    if enqueue_first:
        enqueue(sid, v, PATHS["decomposed"][0])
    return v


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


# The meta keys a PACKAGE ONBOARDING puts on the skills it creates
# (mcp/package_sources.py; docs/PACKAGE-ONBOARDING.md 4): the onboarding
# dataset, the package it is about and the version. A decomposed child
# inherits them from its source (create_child, new_child_version).
ONBOARDING_KEYS = ("onboarding", "package", "package_version")


def enqueue(sid: str, v: int, stage: str) -> str:
    queue, lane = JOBS[stage]
    payload = {"skill": sid, "version": int(v), "stage": stage}
    m = (get(sid) or {}).get("meta") or {}
    if lane == "gpu" and (stage == "prove" or m.get("onboarding")
                          or m.get("rebuild") or m.get("replacement")):
        # An onboarding's skill model stages wait for an idle stack like its
        # other gpu stages (worker.run_one; docs/PACKAGE-ONBOARDING.md 3.2);
        # so does every PROVE (operator, 2026-09-28: "Run it as idle-gated
        # worker jobs on the gpu lane"): six or more generations a skill;
        # and the library's rebuild (mcp/skill_rebuild.py).
        payload["idle"] = True
    return jobs.add(queue, payload, lane=lane, dataset=f"skill:{sid}",
                    stage=stage)


def next_stage(ver: dict) -> str | None:
    path = ver.get("path") or []
    if ver.get("stage") not in path:
        return None
    i = path.index(ver["stage"])
    return path[i + 1] if i + 1 < len(path) else None


def advance(sid: str, v: int, done_stage: str, *,
            enqueue_next: bool = True) -> str | None:
    """After `done_stage` finished: move a still-running version to its next
    stage and (unless running inline) enqueue that stage's job. None when
    the version has stopped (quarantined, failed, armed) or already moved."""
    ver = version(sid, v)
    if not ver or ver["state"] != "running" or ver["stage"] != done_stage:
        return None
    nxt = next_stage(ver)
    if nxt is None:
        return None
    update_version(sid, v, stage=nxt)
    if enqueue_next:
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


def set_name(sid: str, name: str) -> str:
    """Give the skill its final SKILL.md name, unique in the store (a clash
    takes a suffix from the id). Returns the name set."""
    import skill_md
    base = skill_md.to_name(name)
    con = _db()
    try:
        taken = {r[0] for r in con.execute(
            "SELECT name FROM skills WHERE id != ?", (sid,))}
        nm = base
        if nm in taken:
            nm = skill_md.to_name(f"{base[:56]}-{sid[:6]}")
        con.execute("UPDATE skills SET name=?, updated=? WHERE id=?",
                    (nm, time.time(), sid))
    finally:
        con.close()
    return nm


def _refresh(con: sqlite3.Connection, sid: str) -> None:
    """Derive `status` and `reason` from the versions and `enabled`. The one
    place status is written."""
    s = con.execute("SELECT enabled, served_version, latest_version, meta "
                    "FROM skills WHERE id=?", (sid,)).fetchone()
    if s is None:
        return
    meta = json.loads(s["meta"] or "{}")
    latest = con.execute("SELECT state, reason, stage FROM skill_versions "
                         "WHERE skill=? AND version=?",
                         (sid, s["latest_version"])).fetchone()
    reason = None
    if meta.get("archived"):
        status, reason = "archived", (meta["archived"] or {}).get("reason")
    elif not s["enabled"]:
        status = "disabled"
        reason = (meta.get("disabled") or {}).get("reason")
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
    elif latest and latest["state"] == "decomposed":
        status, reason = "decomposed", latest["reason"]
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
    try:
        export(sid)
    except Exception as e:                                       # noqa: BLE001
        print(f"  skills: export of {sid} failed: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)


def quarantine(sid: str, v: int, reason: str, **fields) -> None:
    """The screen or the tests said stop. The version never arms, and the
    SKILL is disarmed: a source that turned hostile has told us something
    about its author, including about the version we are serving."""
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


def mark_decomposed(sid: str, v: int, children: list[str], reason: str
                    ) -> None:
    def fn(con):
        con.execute("UPDATE skill_versions SET state='decomposed', reason=?, "
                    "meta=?, updated=? WHERE skill=? AND version=?",
                    (reason[:2000], json.dumps({"children": children}),
                     time.time(), sid, int(v)))
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
        title = (ver.get("validate") or {}).get("title") or ""
        con.execute("UPDATE skills SET served_version=?, title=? WHERE id=?",
                    (int(v), title[:200], sid))
    _with(sid, fn)
    return get(sid)


def rearm(sid: str, v: int, **fields) -> dict:
    """Serve a version again that PROVE quarantined, after a later proof
    (the repeat rule, skill_prove.REPEATS) did not confirm the worse result.
    Only a quarantined version with text, of a skill that is enabled and not
    archived, whose reason is PROVE's. `fields` are stored with it (the new
    prove record). The served version of any other state is superseded, as
    arm does."""
    s = _require(sid)
    ver = version(sid, v)
    if ver is None:
        raise KeyError(f"no such version: {sid} v{v}")
    if ver["state"] != "quarantined" or not ver.get("text") \
            or not str(ver.get("reason") or "").startswith("prove:"):
        raise ValueError(f"v{v} is {ver['state']}"
                         f"{'' if ver.get('text') else ' with no text'}"
                         "; only a version quarantined by prove, with text, "
                         "is served again by rearm")
    if not s.get("enabled", True) or s.get("status") == "archived":
        raise ValueError(f"{sid} is disabled or archived; enable it first")

    def fn(con):
        now = time.time()
        sets = ["state='armed'", "stage='arm'", "reason=NULL", "armed_at=?",
                "updated=?"]
        args = [now, now]
        for k, val in fields.items():
            sets.append(f"{k}=?")
            args.append(json.dumps(val) if k in _JSON_V else val)
        con.execute("UPDATE skill_versions SET state='superseded', updated=? "
                    "WHERE skill=? AND state='armed'", (now, sid))
        con.execute(f"UPDATE skill_versions SET {','.join(sets)} WHERE "
                    "skill=? AND version=?", args + [sid, int(v)])
        title = (ver.get("validate") or {}).get("title") or ""
        con.execute("UPDATE skills SET served_version=?, title=? WHERE id=?",
                    (int(v), title[:200], sid))
    _with(sid, fn)
    return get(sid)


def reinstate(sid: str, v: int) -> dict:
    """Serve version v again after a later version failed to arm (an
    operator's edit that quarantined): v must be superseded and have text.
    The failed version keeps its state and its reason."""
    ver = version(sid, v)
    if ver is None or ver["state"] != "superseded" or not ver.get("text"):
        raise ValueError(f"v{v} of {sid} is not a superseded version with "
                         "text; nothing to reinstate")

    def fn(con):
        now = time.time()
        con.execute("UPDATE skill_versions SET state='armed', updated=? "
                    "WHERE skill=? AND version=?", (now, sid, int(v)))
        con.execute("UPDATE skills SET served_version=? WHERE id=?",
                    (int(v), sid))
    _with(sid, fn)
    return get(sid)


def _meta_flag(sid: str, key: str, value, *, enabled: int | None = None,
               drop: tuple = ()) -> None:
    def fn(con):
        m = json.loads(con.execute("SELECT meta FROM skills WHERE id=?",
                                   (sid,)).fetchone()["meta"] or "{}")
        if value is None:
            m.pop(key, None)
        else:
            m[key] = value
        for k in drop:
            m.pop(k, None)
        if enabled is None:
            con.execute("UPDATE skills SET meta=? WHERE id=?",
                        (json.dumps(m), sid))
        else:
            con.execute("UPDATE skills SET enabled=?, meta=? WHERE id=?",
                        (enabled, json.dumps(m), sid))
    _with(sid, fn)


def disable(sid: str, *, reason: str = "", author: str = "operator") -> dict:
    _require(sid)
    _meta_flag(sid, "disabled", {"by": author, "at": time.time(),
                                 "reason": (reason or "disabled on the "
                                            "dashboard")[:300]}, enabled=0)
    return get(sid)


def enable(sid: str, *, author: str = "operator") -> dict:
    """Re-enable (also un-archives). The served version already passed the
    screen and its tests, so it serves again at once; nothing re-runs."""
    _require(sid)
    _meta_flag(sid, "enabled", {"by": author, "at": time.time()}, enabled=1,
               drop=("disabled", "archived"))
    return get(sid)


def archive(sid: str, *, reason: str = "", author: str = "operator") -> dict:
    """Out of service and out of the default listing; nothing is deleted.
    `enable` brings it back."""
    _require(sid)
    _meta_flag(sid, "archived", {"by": author, "at": time.time(),
                                 "reason": (reason or "archived on the "
                                            "dashboard")[:300]}, enabled=0)
    return get(sid)


def quarantine_skill(sid: str, *, reason: str, author: str = "operator"
                     ) -> dict:
    """An operator's quarantine: the latest version is marked quarantined
    and the skill is disarmed, exactly as a failed screen would."""
    s = _require(sid)
    quarantine(sid, s["latest_version"],
               f"quarantined by {author}: {reason or 'no reason given'}")
    return get(sid)


def rerun(sid: str, stage: str, *, author: str = "operator") -> dict:
    """Run the LATEST version again from `stage` (which must be on its
    path): after a licence was supplied, the tests were edited, or a stage
    failed for a reason that has passed. A served version keeps serving."""
    s = _require(sid)
    v = s["latest_version"]
    ver = version(sid, v) or {}
    if stage not in (ver.get("path") or []):
        raise ValueError(f"{stage!r} is not on v{v}'s path "
                         f"{ver.get('path')}")
    if ver.get("state") in ("armed", "superseded", "decomposed"):
        raise ValueError(f"v{v} is {ver['state']}; edit the skill to make a "
                         "new version instead")

    def fn(con):
        m = json.loads(con.execute("SELECT meta FROM skill_versions WHERE "
                                   "skill=? AND version=?", (sid, v))
                       .fetchone()["meta"] or "{}")
        m.setdefault("reruns", []).append({"stage": stage, "by": author,
                                           "at": time.time()})
        con.execute("UPDATE skill_versions SET state='running', stage=?, "
                    "reason=NULL, meta=?, updated=? WHERE skill=? AND "
                    "version=?", (stage, json.dumps(m), time.time(), sid, v))
    _with(sid, fn)
    enqueue(sid, v, stage)
    return get(sid)


def set_licence(sid: str, spdx: str, quote: str, *,
                author: str = "operator") -> dict:
    """An operator's licence for a source whose licence the pipeline could
    not establish from a verbatim quote. Recorded as the operator's, with
    their quote; the licence stage accepts it."""
    _require(sid)
    spdx, quote = (spdx or "").strip(), (quote or "").strip()
    if not spdx or not quote:
        raise ValueError("give the licence and the verbatim text that "
                         "grants it")
    _meta_flag(sid, "licence", {"spdx": spdx[:80], "quote": quote[:600],
                                "where": "operator", "by": author,
                                "at": time.time()})
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


def update_meta(sid: str, **fields) -> dict:
    """Set (or, with None, drop) top-level keys of a skill's meta."""
    _require(sid)

    def fn(con):
        m = json.loads(con.execute("SELECT meta FROM skills WHERE id=?",
                                   (sid,)).fetchone()["meta"] or "{}")
        for k, v in fields.items():
            if v is None:
                m.pop(k, None)
            else:
                m[k] = v
        con.execute("UPDATE skills SET meta=? WHERE id=?",
                    (json.dumps(m), sid))
    _with(sid, fn)
    return get(sid)


def repoint(sid: str, url: str, *, onboarding: str | None = None,
            package_version: str | None = None,
            author: str = "pipeline") -> dict:
    """A package's new version (docs/PACKAGE-ONBOARDING.md 4.3, REPLACE):
    the skill's source moves to the same file at the new commit. Only
    `source_url` and the onboarding keys change here; the caller then makes
    the new version (new_version, store_source, enqueue), and the served
    version keeps serving until it arms. The move is kept in
    meta.repointed."""
    s = _require(sid)
    url = normalise_url((url or "").strip())
    if not re.match(r"^https?://\S+$", url):
        raise ValueError(f"{url[:120]!r} is not one http(s) URL")

    def fn(con):
        m = json.loads(con.execute("SELECT meta FROM skills WHERE id=?",
                                   (sid,)).fetchone()["meta"] or "{}")
        m.setdefault("repointed", []).append({
            "from": s.get("source_url"), "to": url, "by": author,
            "at": time.time(), "from_version": m.get("package_version"),
            "from_onboarding": m.get("onboarding")})
        if onboarding:
            m["onboarding"] = onboarding
        if package_version:
            m["package_version"] = package_version
        con.execute("UPDATE skills SET source_url=?, meta=? WHERE id=?",
                    (url, json.dumps(m), sid))
    _with(sid, fn)
    return get(sid)


def edit(sid: str, text: str, *, author: str = "operator",
         tests: dict | None = None, enqueue_first: bool = True) -> dict:
    """An operator's edit: a new version, re-screened and re-tested before
    it is re-armed. `enqueue_first=False`: nothing is queued; the caller
    runs the stages inline (skill_pipeline.run_inline). `text` is a whole SKILL.md (its frontmatter may change
    the name, description and applies-when) or just the body.

    The skill is DISARMED now and serves nothing until the edited version
    passes screen -> screen_model -> classify -> tests -> validate -> arm.
    Watching stops: a re-fetch would otherwise distil the source again and
    replace what the operator wrote (set_watch turns it back on).
    """
    import skill_builder
    import skill_md
    s = _require(sid)
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("the edit is empty")
    if len(text) > 8 * skill_builder.MAX_SKILL_CHARS:
        raise ValueError(f"the edit is {len(text)} characters; a skill body "
                         f"is at most {skill_builder.MAX_SKILL_CHARS}")
    served = version(sid, s["served_version"]) if s.get("served_version") \
        else latest_with_text(sid)
    prev = skill_md.parse((served or {}).get("text") or "") if served else {}
    if text.startswith("---"):
        sk = skill_md.parse(text)
    else:
        sk = dict(prev or {}, items=skill_md.parse_items(text) or
                  skill_builder.parse(text)["items"])
        t = skill_md._title_of(text) or skill_builder.parse(text)["title"]
        if t:
            sk["title"] = t
    known = {}
    for it in ((served or {}).get("validate") or {}).get("items") or []:
        known[skill_builder.item_line(it)] = it.get("quote") or ""
    parsed = {"title": sk.get("title") or (prev or {}).get("title") or "",
              "items": [dict(it, quote=it.get("quote") or "")
                        for it in sk.get("items") or []],
              "name": sk.get("name") or s.get("name"),
              "description": sk.get("description")
              or (prev or {}).get("description") or ""}
    rule = dict((served or {}).get("classify") or {})
    aw = ((sk.get("yamadori") or {}).get("applies_when"))
    legacy = None if text.startswith("---") else re.search(
        r"(?im)^\s*applies[ -]when\s*:\s*(.+)$", text)
    if isinstance(aw, dict) and aw and text.startswith("---"):
        import skill_classify
        rule = skill_classify.rule_from_metadata(aw)
        # The description IS the trigger condition (skill_classify): an
        # edited SKILL.md keeps its description sentences as triggers, as
        # an authored install does (2026-09-26: they were dropped, leaving
        # the embedding stage only the title).
        g = skill_classify.gates(rule)
        rule = skill_classify.with_gates(
            rule, phases=g["phases"], situations=g["situations"],
            all_of=g["all_of"], topics=g["topics"],
            description=parsed["description"])
    elif legacy:
        # The pre-SKILL.md form: a person's `applies when:` line.
        import skill_classify
        rule = skill_classify.rule_of_condition(legacy.group(1))
    # The served version's tests carry over only while the rule does: a
    # changed applies-when makes its near misses stale, so the tests stage
    # derives them afresh from the new rule (plus any the operator gives).
    same_rule = rule == dict((served or {}).get("classify") or {})
    old_tests = ((served or {}).get("tests") or {}) if same_rule else {}
    v = new_version(sid, "edit", author=author,
                    distil={"parsed": parsed, "operator": True,
                            "submitted": text, "known_quotes": known},
                    classify=rule, tests=tests or old_tests,
                    meta={"edited_from": s.get("served_version")
                          or s.get("latest_version")})

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
    if enqueue_first:
        enqueue(sid, v, PATHS["edit"][0])
    return get(sid)


# ---------------------------------------------------------------------------
# The portable form: library/<name>/SKILL.md for every skill with a
# validated version, its state in the metadata.
# ---------------------------------------------------------------------------
_STATE_OF = {"armed": "armed", "quarantined": "quarantined",
             "disabled": "disabled", "archived": "archived",
             "failed": "draft", "pipeline": "draft"}


def export(sid: str) -> str | None:
    """Write (or refresh) the skill's folder from the version that best
    represents it: the served one, else the latest with text. Returns the
    folder name, or None when no version has a SKILL.md yet."""
    import skill_md
    s = get(sid)
    if s is None:
        return None
    ver = version(sid, s.get("served_version")) if s.get("served_version") \
        else latest_with_text(sid)
    if not ver or not (ver.get("text") or "").startswith("---"):
        return None
    sk = skill_md.parse(ver["text"])
    ours = dict(sk.get("yamadori") or {})
    ours["state"] = _STATE_OF.get(s["status"], "draft")
    if s.get("reason") and ours["state"] != "armed":
        ours["reason"] = str(s["reason"])[:300]
    else:
        ours.pop("reason", None)
    sk["yamadori"] = ours
    root = library_dir()
    folder = os.path.join(root, sk["name"])
    # Another skill's folder under this name (a stale export) is replaced
    # only when it is ours.
    if os.path.exists(os.path.join(folder, "SKILL.md")):
        try:
            other = skill_md.read_folder(folder)[0]
            oid = (other.get("yamadori") or {}).get("id")
            if oid and oid != sid:
                return None
        except Exception:                                        # noqa: BLE001
            pass
    skill_md.write_folder(root, sk, ver.get("tests") or {})
    return sk["name"]


def export_armed(dest: str) -> list[str]:
    """Copy every ARMED skill's folder into `dest` (a harness's skills
    directory, e.g. ~/.hermes/skills/yamadori/). Returns the names."""
    out = []
    for s in armed():
        src = os.path.join(library_dir(), s["name"])
        if os.path.isdir(src):
            dst = os.path.join(dest, s["name"])
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            out.append(s["name"])
    return out


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


def listing(limit: int = 5000) -> list[dict]:
    con = _db()
    try:
        return [_decode(r, _JSON_S) for r in con.execute(
            "SELECT * FROM skills ORDER BY updated DESC LIMIT ?", (limit,))]
    finally:
        con.close()


def find(name: str) -> dict | None:
    """A skill by its name, in any spelling the store has used for it: the
    name column, its SKILL.md form (hyphens), or the name its creator asked
    for (meta.name_requested)."""
    import skill_md
    con = _db()
    try:
        row = con.execute("SELECT * FROM skills WHERE name=? "
                          "ORDER BY created LIMIT 1", (name,)).fetchone()
        if row is None:
            want = skill_md.to_name(name)
            for r in con.execute("SELECT * FROM skills ORDER BY created"):
                try:
                    req = json.loads(r["meta"] or "{}").get("name_requested")
                except ValueError:
                    req = None
                if skill_md.to_name(r["name"]) == want or req == want:
                    row = r
                    break
        return _decode(row, _JSON_S)
    finally:
        con.close()


def children(parent: str) -> list[dict]:
    return [s for s in listing() if ((s.get("meta") or {}).get("parent")
                                     or {}).get("skill") == parent]


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
                                   "progress", "result", "created",
                                   "finished")}
            for j in jobs.listing(dataset=f"skill:{sid}", limit=limit)]


# ---------------------------------------------------------------------------
# What is served. Cached for a few seconds: selection runs on every request,
# and one indexed SELECT per request is cheap but not free on a busy file.
# ---------------------------------------------------------------------------
CACHE_SECONDS = float(os.environ.get("YAMADORI_SKILL_CACHE_SECONDS", "5"))
# The serving-time ASSURED VOICE filter (_row_of; operator, 2026-09-28).
DOUBT_FILTER = os.environ.get("YAMADORI_SKILL_DOUBT_FILTER", "1") != "0"
_LOCK = threading.Lock()
_CACHE: dict = {"at": 0.0, "db": None, "rows": []}


def _invalidate() -> None:
    with _LOCK:
        _CACHE["at"] = 0.0


_ROWS: dict = {}


def row_of(sid: str, name: str, source_kind: str, source_url, ver: dict
           ) -> dict:
    """One served skill as selection and the knowledge base read it.
    Memoised on (id, version, the text's hash): parsing every SKILL.md on
    every cache refresh would cost the request path a YAML parse per
    skill."""
    key = (sid, ver.get("version"), name,
           hashlib.sha1((ver.get("text") or "").encode("utf-8")).hexdigest(),
           ver.get("package"), ver.get("package_version"))
    hit = _ROWS.get(key)
    if hit is not None:
        return dict(hit)
    row = _row_of(sid, name, source_kind, source_url, ver)
    if len(_ROWS) > 20000:
        _ROWS.clear()
    _ROWS[key] = row
    return dict(row)


def _row_of(sid: str, name: str, source_kind: str, source_url, ver: dict
            ) -> dict:
    import skill_classify
    import skill_md
    text = ver.get("text") or ""
    sk = skill_md.parse(text) if text.startswith("---") else {}
    val = ver.get("validate") or {}
    rule = ver.get("classify") or {}
    items = val.get("items") or sk.get("items") or []
    # ASSURED VOICE at serving time (operator, 2026-09-28, after pagoda-h6):
    # a skill armed before skill_limits.doubt existed may still carry a line
    # of verification homework, instability or version history; it is left
    # out of what the skill serves (body and items), and the row says which
    # lines went. The stored version is untouched: its replacement comes
    # through the pipeline (review -> validate -> prove), and
    # YAMADORI_SKILL_DOUBT_FILTER=0 serves the stored lines as they are.
    doubt_dropped: list[dict] = []
    if sk and DOUBT_FILTER:
        import skill_limits
        keep = []
        for it in sk.get("items") or []:
            d = skill_limits.doubt(skill_md.item_line(it))
            if d:
                doubt_dropped.append({"item": skill_md.item_line(it)[:200],
                                      "why": d})
            else:
                keep.append(it)
        if doubt_dropped:
            sk = dict(sk, items=keep)
            gone = {x["item"] for x in doubt_dropped}
            items = [it for it in items
                     if skill_md.item_line(it)[:200] not in gone]
    body = skill_md.injection(sk) if sk else text
    return {"id": sid, "version": ver["version"], "name": name,
            "text": text, "body": body, "doubt_dropped": doubt_dropped,
            "title": sk.get("title") or val.get("title") or "",
            "description": sk.get("description") or "",
            "items": items,
            "rule": rule, "tags": sk.get("tags") or [],
            "category": skill_classify.category(rule),
            "escalate": bool(rule.get("escalate")),
            # The package this skill LEADS (decompose/3): what the package
            # is and its core pattern, for mcp/skill_packages.py.
            "lead_for": (sk.get("yamadori") or {}).get("lead_for"),
            # The package (npm name) and version the skill is about, when a
            # package onboarding made it (meta.package, meta.package_version;
            # docs/PACKAGE-ONBOARDING.md 4.3): the selector's asked-major
            # filter reads the version here before the name.
            "package": ver.get("package") or (sk.get("yamadori") or {})
            .get("package"),
            "package_version": ver.get("package_version") or (
                sk.get("yamadori") or {}).get("package_version"),
            "source_kind": source_kind, "source_url": source_url}


def armed() -> list[dict]:
    """Every enabled skill with a served version: what selection reads."""
    now = time.time()
    db = os.path.abspath(jobs.DB)
    with _LOCK:
        if _CACHE["db"] == db and now - _CACHE["at"] < CACHE_SECONDS:
            return _serving(_CACHE["rows"])
    con = _db()
    try:
        rows = []
        for r in con.execute(
                "SELECT s.id, s.name, s.source_url, s.source_kind, s.meta, "
                "v.version, v.text, v.classify, v.validate "
                "FROM skills s JOIN skill_versions v "
                "ON v.skill=s.id AND v.version=s.served_version "
                "WHERE s.enabled=1 AND s.served_version IS NOT NULL "
                "AND v.state='armed' ORDER BY s.id"):
            try:
                sm = json.loads(r["meta"] or "{}")
            except ValueError:
                sm = {}
            ver = {"version": r["version"], "text": r["text"],
                   "classify": json.loads(r["classify"] or "{}"),
                   "validate": json.loads(r["validate"] or "{}"),
                   "package": sm.get("package"),
                   "package_version": sm.get("package_version")}
            row = row_of(r["id"], r["name"], r["source_kind"],
                         r["source_url"], ver)
            if sm.get("name_requested") and                     sm["name_requested"] != row.get("name"):
                row["alias"] = sm["name_requested"]
            rows.append(row)
    finally:
        con.close()
    with _LOCK:
        _CACHE.update(at=now, db=db, rows=rows)
    return _serving(rows)


def _serving(rows: list[dict]) -> list[dict]:
    """The rows a request may be served: a skill whose every line is doubt
    (skill_limits.doubt, _row_of) has nothing to serve."""
    return [r for r in rows if not (r.get("doubt_dropped")
                                    and not r.get("items"))]


def armed_unfiltered() -> list[dict]:
    """Every armed row, a skill whose every line is doubt included (the
    rebuild reads what it must replace: mcp/skill_rebuild.py)."""
    armed()
    with _LOCK:
        return list(_CACHE.get("rows") or [])


def served_texts() -> list[tuple[str, int, str]]:
    """(id, version, injected body) of every served skill, for a leak
    preflight (bench/domain/run.py)."""
    return [(s["id"], s["version"], s["body"]) for s in armed()]


def manifest() -> dict:
    """Every skill folder in the library with its SKILL.md hash, and one
    hash over the lot: the store's fingerprint for models/manifest.yaml."""
    import skill_md
    rows = []
    for folder in skill_md.folders(library_dir()):
        with open(os.path.join(folder, "SKILL.md"), "rb") as f:
            data = f.read()
        tp = os.path.join(folder, "tests.json")
        tsha = None
        if os.path.exists(tp):
            with open(tp, "rb") as f:
                tsha = hashlib.sha256(f.read()).hexdigest()
        sk = skill_md.parse(data.decode("utf-8"))
        rows.append({"name": os.path.basename(folder),
                     "state": (sk.get("yamadori") or {}).get("state"),
                     "skill_md_sha256": hashlib.sha256(data).hexdigest(),
                     "tests_sha256": tsha})
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['name']}\0{r['skill_md_sha256']}\0"
                 f"{r['tests_sha256']}\n".encode())
    return {"skills": rows, "n": len(rows), "sha256": h.hexdigest()}


# ---------------------------------------------------------------------------
# The public view: what the dashboard API returns. No filesystem path.
# ---------------------------------------------------------------------------
def public_version(ver: dict) -> dict:
    d = {k: ver.get(k) for k in ("version", "origin", "author", "path",
                                 "stage", "state", "reason", "source_sha256",
                                 "source_bytes", "fetched", "screen",
                                 "classify", "licence", "tests", "validate",
                                 "text", "created", "updated", "armed_at",
                                 "meta")}
    dist = ver.get("distil") or {}
    d["distil"] = {k: dist.get(k) for k in ("chunks", "replies_chars",
                                            "operator", "counts", "notes",
                                            "prompt", "compiled", "section",
                                            "children")
                   if k in dist}
    return d


def public(s: dict, *, detail: bool = False) -> dict:
    import skill_classify
    import skill_md
    out = {k: s.get(k) for k in ("id", "name", "title", "source_url",
                                 "source_kind", "status", "reason", "enabled",
                                 "served_version", "latest_version",
                                 "watch_seconds", "next_watch", "created",
                                 "updated")}
    out["enabled"] = bool(out["enabled"])
    meta = s.get("meta") or {}
    out["meta"] = {k: meta[k] for k in ("disabled", "enabled", "archived",
                                        "watch_paused", "migration_group",
                                        "watch_last", "parent", "goal",
                                        "licence")
                   if k in meta}
    served = version(s["id"], s.get("served_version")) \
        if s.get("served_version") else None
    shown = served or latest_with_text(s["id"])
    text = (shown or {}).get("text") or ""
    sk = skill_md.parse(text) if text.startswith("---") else {}
    # The SKILL.md itself only in the detail view: 525 of them are 3.5 MB.
    if detail:
        out["skill_md"] = text or None
        # `text`: the same, under the name the current SKILLS tab reads
        # (the React NAEDOKO surface is the follow-up; docs/SKILL-FACTORY.md).
        out["text"] = text or None
    out["text_version"] = (shown or {}).get("version")
    out["description"] = sk.get("description")
    out["tags"] = sk.get("tags") or []
    out["provenance"] = (sk.get("yamadori") or {}).get("provenance")
    latest = version(s["id"], s["latest_version"]) or {}
    rule = (shown or latest).get("classify") or {}
    out["applies_when"] = rule.get("text")
    out["applies_to"] = rule.get("applies_to")
    out["category"] = skill_classify.category(rule)
    out["gates"] = skill_classify.gates(rule)
    out["triggers"] = [t.get("text") for t in rule.get("triggers") or []
                       if isinstance(t, dict)]
    tres = ((shown or latest).get("validate") or {}).get("activation") or {}
    out["activation"] = {k: tres.get(k) for k in ("passed", "score", "n",
                                                  "failures")} if tres \
        else None
    out["folder"] = sk.get("name") if sk else None
    if detail:
        import skill_learn
        out["learned_triggers"] = skill_learn.learned_triggers({s["id"]}) \
            .get(s["id"], [])
        out["versions"] = [public_version(v) for v in versions(s["id"])]
        out["jobs"] = skill_jobs(s["id"])
        out["tests"] = (shown or latest).get("tests") or {}
        out["children"] = [{"id": c["id"], "name": c["name"],
                            "status": c["status"]}
                           for c in children(s["id"])]
        out["selections"] = skill_learn.selections(s["id"], limit=20)
    return out


if __name__ == "__main__":
    print(f"  db    {os.path.abspath(jobs.DB)}")
    print(f"  store {os.path.abspath(STORE)}")
    print(f"  {counts()}")
    for s in listing()[:60]:
        print(f"  {s['id']}  {s['status']:<11} v{s['served_version'] or '-'}"
              f"/{s['latest_version']}  {(s['title'] or s['name'])[:60]}")
