#!/usr/bin/env python
"""The corpus pipeline as data: what a dataset is, and where each one has got to.

WHY THIS IS A MODULE AND NOT A SCRIPT

Recipes arrived in this repo by hand. Somebody read a source, ran a collector,
and dropped a jsonl into `bench/recipes/`. That worked for 938 rows and it
leaves three facts nowhere on disk:

  WHERE IT CAME FROM     `bench/recipes/SOURCES.md` is prose, written after the
                         fact, and it is the only record. A row whose
                         provenance is a paragraph in a markdown file cannot be
                         audited, and `dashboard.save_one` refuses to let a
                         reviewer edit provenance precisely because it is the
                         thing that must not drift.
  WHAT ITS LICENCE IS    This repo has already had to flag an AGPL corpus and a
                         manual that forbids redistribution. Both were caught
                         by a person remembering. `license_if_known` is a
                         required answer here, and "unknown" is a BLOCKER, not
                         a default: a corpus whose licence nobody established
                         is one that cannot be shipped, and finding that out
                         after it is embedded into the hints cache is finding
                         out too late.
  HOW FAR IT GOT         A source that was fetched but never extracted, or
                         extracted but never indexed, is invisible. It looks
                         exactly like a source nobody started.

THE STAGES

    submitted -> clarify -> extract -> index -> (label) -> (train)

`submitted` and `clarify` are cheap and mostly human. After `clarify` the agent
runs the rest unattended: `mcp/worker.py` advances a dataset the moment the job
for its current stage is `done`. There is no review GATE -- rows are served as
soon as they are indexed, and a person reviews them afterwards if they want to,
on `/dash`. A row marked `reject` there drops out of the hints cache on the
next index (`hints._load_corpus`).

`extract`, `index` and
`train` need the card, so they are `gpu` lane jobs and they are SERIALISED --
two of them at once is not slow, it is wrong (see `jobs.py`). `label` is cpu.
A URL source gets a `net` fetch job at `submitted`; pasted text does not,
because there is nothing to fetch.

NOTHING HERE EXECUTES WORK

Every transition ENQUEUES. `advance()` writes a row to the `jobs` table and
returns; it never fetches, never runs the model and never touches the GPU.
`mcp/worker.py` claims the rows and runs them. When no worker is running, a
freshly advanced dataset sits at `queued` and stays there, and the dashboard
says so in those words rather than showing a spinner that means nothing.

A STAGE IS LEFT ONLY WHEN ITS JOB IS DONE

`blockers()` refuses to leave a job stage until that stage's job is literally
`done`, and refuses to leave `clarify` while a fetch is outstanding. Without
this, extract could be skipped past while still queued and index would embed a
file that was never written.

`errored` IS NOT `done`

`job_counts()` builds its dictionary from `jobs.STATES` and counts each state
into its own key. There is no "finished" bucket and no `state != 'queued'`
shortcut, because a resume path that conflated the two poisoned a benchmark
file once already. Re-running an errored job is `rerun()`, a separate and
explicit act, exactly as `jobs.py`'s docstring requires.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPES = os.path.join(HERE, "..", "bench", "recipes")

import domains as domain_gate  # noqa: E402
import jobs  # noqa: E402

# ---------------------------------------------------------------------------
# Stages. The tuple is the order; `label` and `train` are skipped unless the
# dataset says it is a Laya training set, because refitting a head on a corpus
# that was never labelled for it is work nobody asked for.
# ---------------------------------------------------------------------------
STAGES = ("submitted", "clarify", "extract",
          "index", "label", "train", "complete")

OPTIONAL_STAGES = ("label", "train")

# Stages a human moves the dataset out of. `clarify` is the only one, and only
# for what the source cannot establish: the assist job (ASSIST below) fills a
# licence from a verbatim quote and proposes the rest, and a dataset it
# completes advances with no human at all. A guessed licence is worse than a
# missing one, so the licence is never proposed -- quoted or asked. Review is
# deliberately NOT a stage -- the agent owns the pipeline and review is
# optional, after the fact, on /dash.
HUMAN_STAGES = ("clarify",)

# What entering a stage enqueues: (queue, lane). The lane is the physical
# resource, not the speed -- see jobs.LANES.
ENQUEUE = {
    "extract": ("dataset.extract", "gpu"),   # needs the model to read a source
    "index":   ("dataset.index", "gpu"),     # embeds non-rejected rows into hints.npz
    "label":   ("dataset.label", "cpu"),     # pair construction, no model
    "train":   ("dataset.train", "gpu"),     # scripts/train_laya.py
}

# A dataset whose source is a URL needs fetching before anything can read it,
# and fetching is network-bound, not card-bound.
FETCH = ("dataset.fetch", "net")

# The model reads the fetched source and fills what the source itself
# establishes (worker.handle_assist). It holds the card, so it is gpu-lane,
# and it runs inside `clarify`: it is enqueued when the fetch finishes, or at
# create for pasted text. It does NOT block clarify -- an operator who answers
# everything by hand is never made to wait for it, and an assist that errored
# is a fact on the job row, not a wall in front of the dataset.
ASSIST = ("dataset.assist", "gpu")
ASSIST_PRIORITY = 10       # claimed before extract (priority 0): see assist()

# The fields the assist may fill. `source_url` is not one: it is transcribed
# from what the operator typed or asked of them, never inferred. `licence` is
# filled ONLY with a verbatim quote (worker.licence_evidence); the other three
# may be the model's proposal, and each records which it is.
ASSISTED = ("source_name", "licence", "language", "domains")

# Provenance values for a field, as stored in the `assist` column and shown by
# the dashboard. `operator` is any value a person typed, including at create.
PROVENANCE = ("evidence", "proposed", "operator")

KINDS = ("recipes", "laya")


# ---------------------------------------------------------------------------
# The clarifying questions.
#
# This is the "it asks us any missing questions" requirement, and the reason
# each question exists is carried WITH the question. A form that asks for a
# licence without saying why gets "MIT" typed into it by someone who did not
# check, which is worse than an empty field because it looks answered.
# ---------------------------------------------------------------------------
FIELDS = (
    {
        "name": "source_name",
        "label": "SOURCE NAME",
        "input": "text",
        "required": True,
        "severity": "blocker",
        "why": "The review page shows this beside every row and a reviewer "
               "rejects a row whose source does not actually say it. A recipe "
               "with no attributable source cannot be reviewed, only believed.",
    },
    {
        "name": "source_url",
        "label": "SOURCE URL OR LOCATOR",
        "input": "text",
        "required": True,
        "severity": "blocker",
        "why": "Provenance is read-only once a row is written: dashboard.py "
               "refuses to let a reviewer re-attribute a recipe. If it is not "
               "on the web, give the file path or a locator -- but give "
               "something, because nothing is not a locator.",
    },
    {
        "name": "licence",
        "label": "LICENCE",
        "input": "text",
        "required": True,
        "severity": "blocker",
        "why": "This repo has already had to flag an AGPL corpus and a manual "
               "that prohibits redistribution. An unknown licence is a "
               "BLOCKER, never a default: establishing it after the rows are "
               "embedded into the hints cache is establishing it too late.",
    },
    {
        "name": "language",
        "label": "LANGUAGE OR AREA",
        "input": "text",
        "required": True,
        "severity": "required",
        "why": "domains.recipe_domains() infers a domain tag from this when "
               "the rows carry none, and an untagged recipe fails OPEN -- it "
               "becomes eligible for every task in the system.",
    },
    {
        "name": "domains",
        "label": "DOMAINS",
        "input": "multi",
        "required": True,
        "severity": "required",
        "choices": sorted(domain_gate.DOMAINS),
        "why": "The fixed vocabulary in domains.DOMAINS. Declared tags are "
               "evidence and inferred tags are this repo's guess; the review "
               "page draws them differently for that reason. Untagged rows "
               "reach every task.",
    },
)

REQUIRED = tuple(f["name"] for f in FIELDS if f["required"])

# Answers that are the absence of an answer wearing one. "unknown" in a licence
# field is the exact failure this module exists to prevent, so it is rejected
# by value rather than by emptiness.
NON_ANSWERS = {"", "-", "?", "??", "n/a", "na", "none", "null", "nil",
               "unknown", "unsure", "dunno", "tbd", "tbc", "todo", "pending",
               "not sure", "no idea", "idk"}

# Licences that are recorded and then SURFACED, because they constrain what may
# be done with the rows. Matched as substrings of the lowered answer. This does
# not block -- a local-only corpus under AGPL is legitimate -- but a page that
# did not say so would be hiding the thing somebody has to decide about.
RESTRICTED = (
    ("agpl", "AGPL: anything served from these rows carries the licence with "
             "it. This repo has flagged an AGPL corpus before."),
    ("no-redistribution", "Redistribution is prohibited: keep it local."),
    ("noredistribution", "Redistribution is prohibited: keep it local."),
    ("all rights reserved", "All rights reserved: no licence has been granted "
                            "to redistribute these rows."),
    ("proprietary", "Proprietary: no licence has been granted to redistribute "
                    "these rows."),
    ("-nc", "Non-commercial clause: check before shipping."),
    ("noncommercial", "Non-commercial clause: check before shipping."),
    ("non-commercial", "Non-commercial clause: check before shipping."),
    ("no-derivatives", "No-derivatives clause: distilling recipes out of it "
                       "may itself be the prohibited act."),
)


class Blocked(Exception):
    """A transition that was refused, carrying WHY rather than a bare False."""

    def __init__(self, stage: str, reasons: list[dict]):
        self.stage = stage
        self.reasons = reasons
        super().__init__("; ".join(r.get("what", "") for r in reasons)
                         or f"cannot leave {stage}")


# ---------------------------------------------------------------------------
# Storage. The datasets table lives in the SAME sqlite file as the jobs table,
# so a dataset and the jobs it spawned cannot end up in two databases that
# disagree, and so pointing YAMADORI_JOBS_DB at a temp path in a test moves
# both. `jobs.DB` is read at call time, not import time, for the same reason.
# ---------------------------------------------------------------------------
DDL = """
CREATE TABLE IF NOT EXISTS datasets(
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    prompt      TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'recipes',
    source      TEXT,
    source_name TEXT,
    source_url  TEXT,
    licence     TEXT,
    language    TEXT,
    domains     TEXT NOT NULL DEFAULT '[]',
    notes       TEXT,
    stage       TEXT NOT NULL DEFAULT 'submitted',
    counts      TEXT NOT NULL DEFAULT '{}',
    assist      TEXT NOT NULL DEFAULT '{}',
    created     REAL NOT NULL,
    updated     REAL NOT NULL);
CREATE INDEX IF NOT EXISTS datasets_stage ON datasets(stage, created);
"""

_ENSURED: set[str] = set()


def _db() -> sqlite3.Connection:
    path = os.path.abspath(jobs.DB)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first = path not in _ENSURED
    if first:
        # jobs.py owns the jobs schema. Calling its opener rather than copying
        # its DDL means the two can never drift, and it is idempotent.
        jobs._db().close()
    con = sqlite3.connect(path, timeout=30, isolation_level=None)
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(DDL)
    if first:
        # A database created before the `assist` column existed keeps its old
        # table: CREATE TABLE IF NOT EXISTS does not add columns.
        cols = {r[1] for r in con.execute("PRAGMA table_info(datasets)")}
        if "assist" not in cols:
            con.execute("ALTER TABLE datasets ADD COLUMN assist TEXT "
                        "NOT NULL DEFAULT '{}'")
        _ENSURED.add(path)
    con.row_factory = sqlite3.Row
    return con


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for k, empty in (("domains", "[]"), ("counts", "{}"), ("assist", "{}")):
        try:
            d[k] = json.loads(d.get(k) or empty)
        except ValueError:
            d[k] = json.loads(empty)
    return d


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return s or "dataset"


def recipe_file(ds: dict) -> str:
    """The jsonl this dataset's rows go to, which is where review reads them."""
    return f"{slug(ds.get('name') or ds.get('id') or '')}.jsonl"


# ---------------------------------------------------------------------------
# The clarifying-question logic. This is the whole of "it asks us any missing
# questions", and it is deliberately here rather than in the page: a second
# caller -- a CLI, a worker deciding whether a fetch is worth doing -- needs
# the same answer, and a rule that lives in JavaScript is a rule the tests
# cannot see.
# ---------------------------------------------------------------------------
def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, tuple, set)):
        return not [x for x in v if str(x).strip()]
    return str(v).strip().lower() in NON_ANSWERS


def missing(spec: dict) -> list[dict]:
    """Which required fields are still unanswered, and why each one matters.

    Returns one entry per unanswered field, in FIELDS order, each carrying the
    field's own `why`. An answer of "unknown" counts as unanswered -- that is
    the licence case, and it is the reason this returns a list of reasons
    rather than a list of names.
    """
    out: list[dict] = []
    for f in FIELDS:
        if not f["required"]:
            continue
        value = spec.get(f["name"])
        if not _blank(value):
            continue
        supplied = value is not None and str(value).strip() != ""
        entry = {
            "field": f["name"],
            "label": f["label"],
            "input": f["input"],
            "choices": f.get("choices"),
            "severity": f["severity"],
            "why": f["why"],
            "what": (f"{f['name']} was given as "
                     f"{str(value).strip()!r}, which is not an answer"
                     if supplied else f"{f['name']} has not been answered"),
        }
        # When the assist looked and could not establish it, say where it
        # looked. "Unanswered" alone invites the operator to repeat a search
        # that has already been done, or to type a licence nobody checked.
        searched = ((spec.get("assist") or {}).get("not_found") or {}) \
            .get(f["name"])
        if searched:
            entry["searched"] = searched
            entry["what"] += f" -- {searched}"
        out.append(entry)
    return out


def warnings(spec: dict) -> list[dict]:
    """Facts about an ANSWERED dataset that somebody still has to decide about.

    Distinct from `missing`: these do not block. A restricted licence is a
    constraint on what may be done with the rows, not a reason to refuse to
    record them, and an unknown domain tag is a typo the gate will silently
    drop.
    """
    out: list[dict] = []
    lic = str(spec.get("licence") or "").strip().lower()
    for needle, note in RESTRICTED:
        if needle in lic:
            out.append({"kind": "licence", "what": note})
            break
    bad = [d for d in (spec.get("domains") or [])
           if str(d).strip() and str(d).strip() not in domain_gate.DOMAINS]
    if bad:
        out.append({
            "kind": "domains",
            "what": f"not in the domains.DOMAINS vocabulary and will be "
                    f"ignored by the gate: {', '.join(sorted(bad))}"})
    return out


# ---------------------------------------------------------------------------
# Create, answer, advance.
# ---------------------------------------------------------------------------
def create(prompt: str, *, name: str | None = None, source: str = "",
           kind: str = "recipes", **fields) -> dict:
    """Record a submission. Enqueues a fetch if there is something to fetch.

    `prompt` is what the operator typed: pasted text, a URL, or a description
    of what to add. Nothing is inferred from it here -- in particular no
    licence and no source name, because a guessed provenance is worse than a
    missing one.
    """
    prompt = (prompt or "").strip()
    if not prompt and not source and not fields.get("source_url"):
        raise ValueError("nothing submitted: give a prompt, a URL or text")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; known: {sorted(KINDS)}")

    url = (fields.get("source_url") or "").strip()
    if not url and re.match(r"^https?://\S+$", prompt):
        # The prompt IS a URL. Copying it into source_url is transcription,
        # not inference: it is the same string the operator typed.
        url = prompt

    name = (name or "").strip() or _name_from(prompt, url)
    did = uuid.uuid4().hex[:12]
    now = time.time()
    doms = _domains_list(fields.get("domains"))

    # Whatever the operator typed at submission is recorded as theirs, so the
    # page never has to guess whether a filled field came from a person.
    given = {"source_name": (fields.get("source_name") or "").strip(),
             "licence": (fields.get("licence") or "").strip(),
             "language": (fields.get("language") or "").strip(),
             "domains": doms}
    assist = {"fields": {k: {"provenance": "operator", "value": v, "at": now}
                         for k, v in given.items() if not _blank(v)}}

    con = _db()
    try:
        con.execute(
            "INSERT INTO datasets(id,name,prompt,kind,source,source_name,"
            "source_url,licence,language,domains,notes,stage,counts,assist,"
            "created,updated) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,'submitted','{}',?,?,?)",
            (did, name, prompt, kind, source or "",
             given["source_name"], url, given["licence"], given["language"],
             json.dumps(doms), (fields.get("notes") or "").strip(),
             json.dumps(assist), now, now))
    finally:
        con.close()

    fetching = bool(url and not source)
    if fetching:
        # Fetching is network-bound. Pasted text needs no fetch, and enqueuing
        # one anyway would leave a job that can only ever be a no-op.
        jobs.add(FETCH[0], {"url": url, "dataset": did}, lane=FETCH[1],
                 dataset=did, stage="submitted")
    _set_stage(did, "clarify")
    if not fetching:
        # The text is already here, so the assist can read it now. A URL
        # source gets its assist when the fetch finishes (worker.advance_after)
        # -- enqueuing it before would hand it a file that is not there yet.
        enqueue_assist(did)
    return get(did)


def _domains_list(v) -> list[str]:
    if isinstance(v, str):
        v = [d.strip() for d in v.split(",") if d.strip()]
    return sorted({str(d).strip() for d in (v or []) if str(d).strip()})


def enqueue_assist(dataset_id: str, *, force: bool = False) -> dict:
    """Put one `dataset.assist` job in the gpu lane, if there is work for it.

    Returns {"job": id} or {"skipped": why}. Never enqueues a second assist
    while one is queued or running. Without `force` it also declines when an
    assist has already finished for this dataset (the fetch completing twice
    is not a reason to ask the model twice) and when nothing it may fill is
    missing. `force` is the operator asking again, explicitly.
    """
    ds = get(dataset_id)
    if ds is None:
        raise KeyError(f"no such dataset: {dataset_id}")
    if ds["stage"] != "clarify":
        return {"skipped": f"the dataset is in {ds['stage']}, not clarify; "
                           "the assist only fills clarifying answers"}
    open_fields = [m["field"] for m in missing(ds) if m["field"] in ASSISTED]
    if not open_fields and not force:
        return {"skipped": "every field the assist may fill is answered"}
    prior = [j for j in jobs.listing(dataset=dataset_id)
             if j["queue"] == ASSIST[0]]
    live = [j for j in prior if j["state"] in ("queued", "running")]
    if live:
        return {"skipped": f"assist job {live[0]['id']} is already "
                           f"{live[0]['state']}", "job": live[0]["id"]}
    if not force and [j for j in prior if j["state"] == "done"]:
        return {"skipped": "an assist has already run for this dataset; ask "
                           "again explicitly to re-run it"}
    # Ahead of extraction on the same gpu lane. Assist is one short call that
    # a person on the dashboard is waiting for; extract is a batch of many
    # chunk calls nobody is watching. At FIFO the first live intake waited
    # behind 26 queued extract jobs (2026-09-22).
    jid = jobs.add(ASSIST[0], {"dataset": dataset_id, "fields": open_fields},
                   lane=ASSIST[1], dataset=dataset_id, stage="clarify",
                   priority=ASSIST_PRIORITY)
    return {"job": jid}


def _name_from(prompt: str, url: str) -> str:
    if url:
        host = re.sub(r"^https?://", "", url).split("/")[0]
        tail = [p for p in re.sub(r"^https?://", "", url).split("/")[1:] if p]
        return slug(f"{host} {tail[-1] if tail else ''}")
    return slug(" ".join(prompt.split()[:6]))


def answer(dataset_id: str, fields: dict, *,
           provenance: dict | None = None, meta: dict | None = None,
           only_blank: bool = False) -> dict:
    """Store answers to the clarifying questions. Does not advance.

    Every value written to one of ASSISTED gets a provenance entry in the
    `assist` column, in the SAME update as the value, so no value can exist
    without saying where it came from. A field with no entry in `provenance`
    is recorded as `operator` -- that is every call from a page or a person.
    The assist handler passes `evidence` / `proposed` entries, `meta` (what it
    searched, what it discarded), and `only_blank`, so it never overwrites
    something a person answered while the model was thinking.
    """
    cols = {"source_name", "source_url", "licence", "language", "domains",
            "name", "notes", "source", "kind"}
    now = time.time()
    con = _db()
    try:
        con.execute("BEGIN IMMEDIATE")
        r = con.execute("SELECT * FROM datasets WHERE id=?",
                        (dataset_id,)).fetchone()
        if r is None:
            con.execute("ROLLBACK")
            raise KeyError(f"no such dataset: {dataset_id}")
        ds = _row(r)
        assist = dict(ds.get("assist") or {})
        prov = dict(assist.get("fields") or {})
        sets, args = [], []
        for k, v in (fields or {}).items():
            if k not in cols:
                continue
            if only_blank and not _blank(ds.get(k)):
                continue
            if k == "domains":
                value = _domains_list(v)
                stored = json.dumps(value)
            elif k == "kind" and v not in KINDS:
                con.execute("ROLLBACK")
                raise ValueError(f"unknown kind {v!r}")
            else:
                value = stored = str(v).strip()
            sets.append(f"{k}=?")
            args.append(stored)
            if k in ASSISTED:
                entry = dict((provenance or {}).get(k)
                             or {"provenance": "operator"})
                if entry.get("provenance") not in PROVENANCE:
                    con.execute("ROLLBACK")
                    raise ValueError(f"unknown provenance {entry!r} for {k}")
                entry.update(value=value, at=now)
                old = prov.get(k)
                if (entry["provenance"] == "operator" and old
                        and old.get("provenance") != "operator"):
                    # An override keeps what it replaced, so the page can say
                    # "you overrode the model's proposal" rather than forget.
                    entry["overrode"] = {x: old.get(x) for x in
                                         ("provenance", "value", "quote",
                                          "found_in") if old.get(x)}
                prov[k] = entry
        if meta:
            assist.update(meta)
        assist["fields"] = prov
        if not sets and not meta:
            con.execute("ROLLBACK")
            return ds
        sets.append("assist=?")
        args.append(json.dumps(assist))
        args.extend([now, dataset_id])
        con.execute(f"UPDATE datasets SET {','.join(sets)},updated=? "
                    "WHERE id=?", args)
        con.execute("COMMIT")
    finally:
        con.close()
    return get(dataset_id)


def field_states(ds: dict) -> list[dict]:
    """Each clarifying field with its value and WHERE THAT VALUE CAME FROM.

    `state` is one of:
      evidence   filled by the assist from a verbatim quote it verified
      proposed   filled by the assist from its own reading -- an inference
      operator   typed by a person (at submission, or since)
      needs_you  unanswered; `why`, `what` and `searched` say what is known

    The rule lives here rather than in the page for the reason `missing()`
    does: a second caller must get the same answer. A provenance entry whose
    recorded value no longer matches the column is not trusted -- the value
    was changed by some path that did not record it, and it is shown as the
    operator's rather than as evidence it no longer is.
    """
    prov = (ds.get("assist") or {}).get("fields") or {}
    rejected = (ds.get("assist") or {}).get("rejected") or {}
    gaps = {m["field"]: m for m in missing(ds)}
    out = []
    for f in FIELDS:
        k = f["name"]
        value = ds.get(k)
        row = {"field": k, "label": f["label"], "input": f["input"],
               "choices": f.get("choices"), "severity": f["severity"],
               "why": f["why"], "value": value,
               "assisted": k in ASSISTED}
        if k in gaps:
            row["state"] = "needs_you"
            row["what"] = gaps[k]["what"]
            if gaps[k].get("searched"):
                row["searched"] = gaps[k]["searched"]
        else:
            e = prov.get(k) or {}
            same = e.get("value") == value
            row["state"] = (e["provenance"] if same and e.get("provenance")
                            in ("evidence", "proposed") else "operator")
            if row["state"] == "evidence":
                row["quote"] = e.get("quote")
                row["found_in"] = e.get("found_in")
            if row["state"] == "proposed" and e.get("basis"):
                row["basis"] = e.get("basis")
            if row["state"] == "operator" and same and e.get("overrode"):
                row["overrode"] = e["overrode"]
        if rejected.get(k):
            row["rejected"] = rejected[k]
        out.append(row)
    return out


def assist_jobs(dataset_id: str) -> list[dict]:
    """This dataset's assist jobs, newest first, trimmed like dataset_jobs."""
    return [j for j in dataset_jobs(dataset_id) if j["queue"] == ASSIST[0]]


def next_stage(ds: dict) -> str | None:
    """The stage after this one, skipping the optional ones when they do not
    apply. None once the dataset is complete."""
    stage = ds.get("stage") or "submitted"
    if stage not in STAGES:
        return None
    i = STAGES.index(stage)
    for nxt in STAGES[i + 1:]:
        if nxt in OPTIONAL_STAGES and ds.get("kind") != "laya":
            continue
        return nxt
    return None


def _unfinished(dataset_id: str, *, queue: str | None = None,
                stage: str | None = None) -> list[dict]:
    """Jobs for this dataset, matching queue or stage, that are not `done`."""
    con = _db()
    try:
        q = ("SELECT id, queue, stage, state, error FROM jobs "
             "WHERE dataset=? AND state != 'done'")
        args: list = [dataset_id]
        if queue:
            q += " AND queue=?"
            args.append(queue)
        if stage:
            q += " AND stage=?"
            args.append(stage)
        return [dict(r) for r in con.execute(q + " ORDER BY created", args)]
    finally:
        con.close()


def _job_blocker(j: dict) -> dict:
    """One outstanding job, as a refusal reason with a remedy and an owner."""
    if j["state"] == "errored":
        return {"what": f"{j['queue']} job {j['id']} is errored",
                "why": (j.get("error") or "")[:400],
                "retryable": True,
                "remedy": "fix the cause, then re-run that job by hand "
                          "(datasets.rerun)", "owner": "operator"}
    if j["state"] == "cancelled":
        return {"what": f"{j['queue']} job {j['id']} was cancelled",
                "retryable": False,
                "remedy": "re-submit the dataset", "owner": "operator"}
    return {"what": f"{j['queue']} job {j['id']} is {j['state']}, not done",
            "retryable": True,
            "remedy": "wait for mcp/worker.py to finish it; if nothing is "
                      "running, start the worker", "owner": "worker"}


def blockers(ds: dict) -> list[dict]:
    """Why this dataset cannot leave the stage it is in. Empty means it can."""
    stage = ds.get("stage")
    if stage == "clarify":
        out = missing(ds)
        # A URL source is read by extract from what fetch saved. Leaving
        # clarify before the fetch is done hands extract a file that is not
        # there.
        out += [_job_blocker(j)
                for j in _unfinished(ds["id"], queue=FETCH[0])]
        return out
    if stage in ENQUEUE:
        return [_job_blocker(j) for j in _unfinished(ds["id"], stage=stage)]
    return []


def advance(dataset_id: str, *, to: str | None = None) -> dict:
    """Move a dataset to its next stage and ENQUEUE that stage's job.

    Never executes anything. The job row is the whole of the side effect, and
    it is visible in `jobs.listing(dataset=...)` the instant this returns.
    """
    ds = get(dataset_id)
    if ds is None:
        raise KeyError(f"no such dataset: {dataset_id}")
    target = to or next_stage(ds)
    if target is None:
        raise Blocked(ds["stage"], [{"what": "already complete"}])
    if target not in STAGES:
        raise ValueError(f"unknown stage {target!r}; known: {list(STAGES)}")
    if STAGES.index(target) <= STAGES.index(ds["stage"]):
        raise Blocked(ds["stage"],
                      [{"what": f"{target} is not after {ds['stage']}"}])

    stop = blockers(ds)
    if stop:
        raise Blocked(ds["stage"], stop)

    _set_stage(dataset_id, target)
    ds = get(dataset_id)
    if target in ENQUEUE:
        queue, lane = ENQUEUE[target]
        ds["enqueued"] = jobs.add(
            queue,
            {"dataset": ds["id"], "name": ds["name"],
             "recipe_file": recipe_file(ds), "source_url": ds["source_url"],
             "licence": ds["licence"], "domains": ds["domains"]},
            lane=lane, dataset=ds["id"], stage=target)
    return ds


def _set_stage(dataset_id: str, stage: str) -> None:
    con = _db()
    try:
        con.execute("UPDATE datasets SET stage=?, updated=? WHERE id=?",
                    (stage, time.time(), dataset_id))
    finally:
        con.close()


def record_counts(dataset_id: str, counts: dict) -> dict:
    """For the worker: store what it actually produced.

    Nothing else writes this field, which is why the page may print it. A
    count that no process measured is not in here, and the page shows the
    empty state instead of a zero.
    """
    con = _db()
    try:
        con.execute("UPDATE datasets SET counts=?, updated=? WHERE id=?",
                    (json.dumps(counts or {}), time.time(), dataset_id))
    finally:
        con.close()
    return get(dataset_id)


def rerun(job_id: str) -> dict:
    """Put one ERRORED job back in the queue. Explicitly, one at a time.

    `jobs.fail()` sends an exhausted job to `errored`, which is terminal on
    purpose. Re-running it is a decision a person makes about a specific
    failure, so there is no bulk retry here and there never should be: the
    thing that poisoned a benchmark file was a resume path that decided for
    everybody at once.
    """
    con = _db()
    try:
        row = con.execute("SELECT state, dataset FROM jobs WHERE id=?",
                          (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such job: {job_id}")
        if row["state"] != "errored":
            raise ValueError(
                f"job {job_id} is {row['state']}, not errored; only an errored "
                "job is re-run, and a running one must not be duplicated")
        con.execute(
            "UPDATE jobs SET state='queued', attempts=0, started=NULL, "
            "finished=NULL, heartbeat=NULL, worker=NULL, "
            "error='[re-run by hand] ' || COALESCE(error,'') WHERE id=?",
            (job_id,))
        return {"job": job_id, "state": "queued", "dataset": row["dataset"]}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Reporting. Every number below is counted out of a table or a file; there is
# no progress estimate anywhere, because nothing here can measure one.
# ---------------------------------------------------------------------------
def get(dataset_id: str) -> dict | None:
    con = _db()
    try:
        r = con.execute("SELECT * FROM datasets WHERE id=?",
                        (dataset_id,)).fetchone()
        return _row(r) if r else None
    finally:
        con.close()


def listing(limit: int = 200) -> list[dict]:
    con = _db()
    try:
        return [_row(r) for r in con.execute(
            "SELECT * FROM datasets ORDER BY created DESC LIMIT ?", (limit,))]
    finally:
        con.close()


def job_counts(dataset_id: str) -> dict:
    """Jobs for one dataset, counted into one bucket per state.

    Every key in `jobs.STATES` is present, including `errored`, and `done`
    counts only rows whose state is literally 'done'. There is no bucket that
    means "over".
    """
    counts = {s: 0 for s in jobs.STATES}
    con = _db()
    try:
        for state, n in con.execute(
                "SELECT state, COUNT(*) FROM jobs WHERE dataset=? "
                "GROUP BY state", (dataset_id,)):
            counts[state] = counts.get(state, 0) + n
    finally:
        con.close()
    counts["total"] = sum(counts[s] for s in jobs.STATES)
    return counts


def dataset_jobs(dataset_id: str, limit: int = 50) -> list[dict]:
    """The job rows themselves, trimmed to what the page prints."""
    out = []
    for j in jobs.listing(dataset=dataset_id, limit=limit):
        out.append({k: j.get(k) for k in
                    ("id", "queue", "lane", "state", "stage", "attempts",
                     "max_attempts", "error", "progress", "created",
                     "started", "finished", "worker")})
    return out


def review_counts(ds: dict) -> dict | None:
    """Review states for this dataset's rows, counted out of the jsonl.

    None -- not zero -- when the file does not exist. A dataset whose extract
    has not run has no rows, and reporting `0 unreviewed` for it would be a
    measurement of a file that is not there.
    """
    path = os.path.join(RECIPES, recipe_file(ds))
    if not os.path.exists(path):
        return None
    counts: dict[str, int] = {}
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                counts["unparseable"] = counts.get("unparseable", 0) + 1
                total += 1
                continue
            st = r.get("_state") or "unreviewed"
            counts[st] = counts.get(st, 0) + 1
            total += 1
    counts["total"] = total
    counts["file"] = recipe_file(ds)
    return counts


def worker_state() -> dict:
    """Has anything ever claimed a job on this database?

    The honest answer today is no, and the page prints it. A queued job with no
    worker is not "in progress"; it is waiting for a process that has not been
    written, and a dashboard that showed it as running would be inventing
    activity.
    """
    con = _db()
    try:
        row = con.execute(
            "SELECT COUNT(*) AS claimed, MAX(heartbeat) AS beat, "
            "MAX(started) AS started FROM jobs WHERE worker IS NOT NULL"
        ).fetchone()
        return {"ever_claimed": int(row["claimed"] or 0),
                "last_heartbeat": row["beat"],
                "last_started": row["started"]}
    finally:
        con.close()


def overview() -> dict:
    """Everything the pipeline view draws, in one read."""
    out = []
    for ds in listing():
        ds = dict(ds)
        ds["recipe_file"] = recipe_file(ds)
        ds["jobs"] = job_counts(ds["id"])
        ds["review"] = review_counts(ds)
        ds["missing"] = missing(ds)
        ds["warnings"] = warnings(ds)
        ds["next_stage"] = next_stage(ds)
        ds["field_states"] = field_states(ds)
        # Only the two states that carry information past the tally: an
        # errored job needs its error text and a re-run, a running one needs
        # whatever progress string its worker actually set. Queued and done
        # rows are fully described by the counts, so sending them would be
        # payload without a reader.
        rows = dataset_jobs(ds["id"])
        ds["errored_jobs"] = [j for j in rows if j["state"] == "errored"]
        ds["running_jobs"] = [j for j in rows if j["state"] == "running"]
        ds["assist_jobs"] = [j for j in rows if j["queue"] == ASSIST[0]]
        out.append(ds)
    return {
        "datasets": out,
        "stages": list(STAGES),
        "optional_stages": list(OPTIONAL_STAGES),
        "human_stages": list(HUMAN_STAGES),
        # Which stages actually put a row in the queue, so the page can say
        # "enqueue extract" and "move to review" and mean both literally.
        "enqueues": {k: {"queue": q, "lane": ln}
                     for k, (q, ln) in ENQUEUE.items()},
        "fields": list(FIELDS),
        "assist": {"queue": ASSIST[0], "lane": ASSIST[1],
                   "fields": list(ASSISTED)},
        "lanes": dict(jobs.LANES),
        "queue": jobs.snapshot(),
        "worker": worker_state(),
        "read_at": time.time(),
    }


if __name__ == "__main__":
    o = overview()
    print(f"  db {o['queue']['db']}")
    print(f"  {len(o['datasets'])} dataset(s); queue {o['queue']['states']}")
    w = o["worker"]
    if not w["ever_claimed"]:
        print("  no worker has ever claimed a job on this database -- "
              "queued jobs will not move")
    for d in o["datasets"]:
        print(f"  {d['id']}  {d['stage']:<10} {d['name'][:34]:<34} "
              f"jobs={d['jobs']['total']} errored={d['jobs']['errored']} "
              f"missing={len(d['missing'])}")
