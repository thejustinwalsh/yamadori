#!/usr/bin/env python
"""The dataset pipeline, asserted. No GPU, no worker, no real database.

WHAT THIS IS GATING

`datasets.py` decides three things that are expensive to get wrong and cheap
to check here:

  1. WHICH LANE a stage's job lands in. `gpu` is serialised because two jobs
     on one card is not slow, it is wrong; a stage that enqueued extraction on
     `cpu` would put four model calls on one card and the symptom would be
     502s an hour away from the cause.
  2. WHAT IS STILL MISSING, and that "unknown" is not an answer to the licence
     question. This repo has already had to flag an AGPL corpus and a manual
     that prohibits redistribution. A dataset that reached `extract` with no
     established licence is one whose rows are in the hints cache before
     anybody knew whether they could be.
  3. THAT `errored` IS NEVER `done`. A resume path that conflated the two
     poisoned a benchmark file: the errored rows could never be re-run and the
     run reported a pass rate over a sample that had silently shrunk. Every
     count in this module is built from `jobs.STATES` with one bucket per
     state, and this file asserts there is no bucket that means "over".

WHAT IT DOES NOT COVER

The worker. `mcp/test_worker.py` owns that. Everything here asserts the
ENQUEUE side: that a row appears in the queue, in the right lane, carrying the
right dataset -- and that a stage cannot be left until its job is `done`.
Where a test needs a stage's job finished to move on, `settle()` marks it done
directly, standing in for the worker.

NO REAL DATABASE AND NO REAL CORPUS

`YAMADORI_JOBS_DB` is pointed at a temp file BEFORE `jobs` is imported,
because that module reads the path at import time. `datasets.RECIPES` and
`dashboard.RECIPES` are redirected at a temp directory, so the review
write-back is exercised against a fixture and `bench/recipes/*.jsonl` is never
opened for writing.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# BEFORE the import, not after: jobs.DB is read at module scope. The datasets
# table lives in the same file, so this one variable moves both.
_TMP = tempfile.mkdtemp(prefix="yamadori_test_datasets_")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")

import dash_data  # noqa: E402
import dash_shell  # noqa: E402
import dashboard  # noqa: E402
import datasets  # noqa: E402
import jobs  # noqa: E402

# The review fixture. A real bench/recipes/ read would be non-deterministic
# and a real write would edit version-controlled corpus text.
FIXTURE_RECIPES = os.path.join(_TMP, "recipes")
os.makedirs(FIXTURE_RECIPES, exist_ok=True)
datasets.RECIPES = FIXTURE_RECIPES
dashboard.RECIPES = FIXTURE_RECIPES

# The databases nothing here may touch.
REAL = [os.path.abspath(os.path.join(HERE, "..", "index", n)) for n in
        ("corpus.sqlite3", "code.sqlite3", "rings.sqlite3", "jobs.sqlite3")]

ANSWERS = {
    "source_name": "Effective Modern Widgets",
    "source_url": "https://example.invalid/widgets",
    "licence": "CC-BY-4.0",
    "language": "typescript",
    "domains": ["types", "web-frontend"],
}

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


def fresh(**over) -> dict:
    """A dataset with every question answered, sitting in `clarify`."""
    fields = dict(ANSWERS)
    fields.update(over)
    kind = fields.pop("kind", "recipes")
    name = fields.pop("name", None)
    return datasets.create("https://example.invalid/widgets",
                           name=name, kind=kind, **fields)


def settle(dataset_id: str) -> int:
    """Stand in for the worker: mark this dataset's outstanding jobs done."""
    n = 0
    for j in jobs.listing(dataset=dataset_id):
        if j["state"] in ("queued", "running"):
            jobs.finish(j["id"], {"settled_by": "test_datasets"})
            n += 1
    return n


def refused(dataset_id: str) -> list[dict] | None:
    """The reasons advance() gives for refusing, or None if it advanced."""
    try:
        datasets.advance(dataset_id)
    except datasets.Blocked as e:
        return e.reasons
    return None


# ---------------------------------------------------------------------------


def test_the_fixture_is_not_the_real_database():
    db = os.path.abspath(jobs.DB)
    check(db.startswith(os.path.abspath(_TMP)),
          "the job and dataset database is a temp file", db)
    for real in REAL:
        check(db != real, f"and is not {os.path.basename(real)}")
    check(os.path.abspath(datasets.RECIPES).startswith(os.path.abspath(_TMP)),
          "and the recipe directory is a fixture, not bench/recipes",
          datasets.RECIPES)


def test_missing_fields_are_named_with_a_reason():
    m = datasets.missing({})
    names = [x["field"] for x in m]
    check(set(names) == set(datasets.REQUIRED),
          "an empty submission is missing every required field",
          json.dumps(names))
    check(all(x.get("why") for x in m),
          "and every question carries the reason it is being asked")
    check(all(x.get("what") for x in m),
          "and states what is wrong, not just which field")
    lic = [x for x in m if x["field"] == "licence"]
    check(lic and lic[0]["severity"] == "blocker",
          "licence is a blocker, not a nice-to-have",
          json.dumps(lic))

    part = dict(ANSWERS)
    part.pop("language")
    check([x["field"] for x in datasets.missing(part)] == ["language"],
          "a partly answered submission names only what is left")
    check(datasets.missing(ANSWERS) == [],
          "and a fully answered one is missing nothing",
          json.dumps(datasets.missing(ANSWERS)))


def test_unknown_is_not_an_answer():
    for bad in ("unknown", "UNKNOWN", " ? ", "tbd", "n/a", "none", ""):
        spec = dict(ANSWERS, licence=bad)
        m = [x for x in datasets.missing(spec) if x["field"] == "licence"]
        check(len(m) == 1,
              f"licence={bad!r} is still an unanswered licence",
              json.dumps(datasets.missing(spec)))
        if m and bad.strip():
            check("not an answer" in m[0]["what"],
                  f"and licence={bad!r} is refused as a non-answer, not as "
                  f"an empty field", m[0]["what"])
    check(datasets.missing(dict(ANSWERS, licence="MIT")) == [],
          "a real licence answers the question")


def test_a_restricted_licence_is_surfaced_but_does_not_block():
    w = datasets.warnings(dict(ANSWERS, licence="AGPL-3.0-or-later"))
    check(any(x["kind"] == "licence" for x in w),
          "AGPL is surfaced as something to decide about", json.dumps(w))
    check(datasets.missing(dict(ANSWERS, licence="AGPL-3.0-or-later")) == [],
          "and it is an ANSWER, so it does not block clarify")
    w = datasets.warnings(dict(ANSWERS, domains=["types", "not-a-domain"]))
    check(any(x["kind"] == "domains" for x in w),
          "a tag outside domains.DOMAINS is surfaced rather than silently "
          "dropped", json.dumps(w))
    check(datasets.warnings(ANSWERS) == [],
          "and a clean dataset warns about nothing")


def test_an_unknown_licence_cannot_advance_past_clarify():
    ds = fresh(licence="unknown")
    check(ds["stage"] == "clarify", "a new submission lands in clarify",
          ds["stage"])
    before = datasets.job_counts(ds["id"])["total"]
    raised = None
    try:
        datasets.advance(ds["id"])
    except datasets.Blocked as e:
        raised = e
    check(raised is not None,
          "advancing a dataset with an unknown licence raises Blocked")
    if raised is not None:
        check(any(r["field"] == "licence" for r in raised.reasons),
              "and the refusal names the licence as the reason",
              json.dumps(raised.reasons))
    again = datasets.get(ds["id"])
    check(again["stage"] == "clarify",
          "the dataset is still in clarify afterwards", again["stage"])
    check(datasets.job_counts(ds["id"])["total"] == before,
          "and a refused transition enqueued nothing",
          f"{before} -> {datasets.job_counts(ds['id'])['total']}")

    # Answering it is what unblocks it -- that, and the fetch finishing.
    datasets.answer(ds["id"], {"licence": "Apache-2.0"})
    settle(ds["id"])
    ds = datasets.advance(ds["id"])
    check(ds["stage"] == "extract",
          "once the licence is established the dataset advances", ds["stage"])


def test_stage_transitions_enqueue_the_right_lane():
    ds = fresh()
    check(datasets.next_stage(ds) == "extract",
          "clarify is followed by extract", str(datasets.next_stage(ds)))

    why = refused(ds["id"])
    check(why and any("dataset.fetch" in r["what"] for r in why),
          "clarify is not left while the fetch is still queued: extract "
          "would read a file that is not there", json.dumps(why))
    check(why and all(r.get("remedy") and r.get("owner") for r in why
                      if "dataset.fetch" in r["what"]),
          "and the refusal names a remedy and an owner", json.dumps(why))
    settle(ds["id"])

    ds = datasets.advance(ds["id"])
    rows = jobs.listing(dataset=ds["id"])
    ex = [j for j in rows if j["stage"] == "extract"]
    check(ds["stage"] == "extract", "the dataset moves to extract", ds["stage"])
    check(len(ex) == 1, "and exactly one extract job is enqueued",
          json.dumps([j["queue"] for j in rows]))
    if ex:
        check(ex[0]["lane"] == "gpu",
              "extract runs on the gpu lane, which is serialised at 1",
              ex[0]["lane"])
        check(ex[0]["queue"] == "dataset.extract", "named dataset.extract",
              ex[0]["queue"])
        check(ex[0]["state"] == "queued",
              "and it is queued, not running: this enqueues, it does not "
              "execute", ex[0]["state"])
        check(ex[0]["dataset"] == ds["id"],
              "the job carries the dataset it belongs to")
        check(ex[0]["payload"].get("recipe_file") == datasets.recipe_file(ds),
              "and the file the worker is expected to write",
              json.dumps(ex[0]["payload"]))

    why = refused(ds["id"])
    check(why and any("dataset.extract" in r["what"] for r in why),
          "extract is not left while its job is queued: index would embed a "
          "file that was never written", json.dumps(why))
    check(datasets.get(ds["id"])["stage"] == "extract",
          "and the refusal leaves the stage where it was")
    settle(ds["id"])

    ds = datasets.advance(ds["id"])
    idx = [j for j in jobs.listing(dataset=ds["id"]) if j["stage"] == "index"]
    check(ds["stage"] == "index",
          "extract is followed by index: review is not a gate, it is "
          "optional and after the fact", ds["stage"])
    check(len(idx) == 1 and idx[0]["lane"] == "gpu",
          "index embeds into the hints cache, so it is a gpu job",
          json.dumps([(j["stage"], j["lane"]) for j in idx]))

    settle(ds["id"])
    ds = datasets.advance(ds["id"])
    check(ds["stage"] == "complete",
          "a recipes dataset is complete after index; label and train are "
          "skipped", ds["stage"])
    check(datasets.next_stage(ds) is None,
          "and there is nothing after complete")
    raised = False
    try:
        datasets.advance(ds["id"])
    except datasets.Blocked:
        raised = True
    check(raised, "advancing a complete dataset is refused, not silently "
                  "repeated")


def test_a_laya_set_gets_label_and_train_and_a_recipes_set_does_not():
    ds = fresh(kind="laya", name="laya pairs")
    for expected in ("extract", "index", "label", "train", "complete"):
        settle(ds["id"])
        ds = datasets.advance(ds["id"])
        check(ds["stage"] == expected,
              f"a laya set reaches {expected}", ds["stage"])
    lanes = {j["stage"]: j["lane"] for j in jobs.listing(dataset=ds["id"])}
    check(lanes.get("label") == "cpu",
          "label is cpu: pair construction needs no model", str(lanes))
    check(lanes.get("train") == "gpu",
          "train is gpu: it refits the head", str(lanes))
    check("review" not in lanes, "and review never became a job")


def test_a_url_submission_enqueues_a_net_fetch_and_pasted_text_does_not():
    ds = datasets.create("https://example.invalid/manual")
    fetch = [j for j in jobs.listing(dataset=ds["id"])
             if j["queue"] == datasets.FETCH[0]]
    check(len(fetch) == 1, "a URL submission enqueues one fetch",
          str(len(fetch)))
    if fetch:
        check(fetch[0]["lane"] == "net",
              "on the net lane, because fetching is network-bound and does "
              "not touch the card", fetch[0]["lane"])
    check(ds["source_url"] == "https://example.invalid/manual",
          "and the URL is transcribed verbatim into source_url",
          ds["source_url"])
    check(ds["licence"] == "" and ds["source_name"] == "",
          "while the licence and source name are left empty rather than "
          "guessed", json.dumps([ds["licence"], ds["source_name"]]))

    pasted = datasets.create("some advice about widgets, pasted by hand",
                             source="the full pasted text")
    check(not [j for j in jobs.listing(dataset=pasted["id"])
               if j["queue"] == datasets.FETCH[0]],
          "pasted text enqueues no fetch: there is nothing to fetch")
    check(datasets.missing(pasted),
          "and it still has to answer every provenance question",
          json.dumps([m["field"] for m in datasets.missing(pasted)]))


def test_an_errored_job_is_never_counted_as_done():
    # Pasted text, so this dataset owns exactly one job and the counts below
    # are about that job rather than about a fetch that came with a URL.
    ds = datasets.create("pasted advice about widgets",
                         name="errored set", source="the full pasted text",
                         **ANSWERS)
    ds = datasets.advance(ds["id"])
    rows = jobs.listing(dataset=ds["id"])
    check(len(rows) == 1, "a pasted-text dataset at extract owns one job",
          json.dumps([j["queue"] for j in rows]))
    jid = rows[0]["id"]

    # A claim, so the database has actually seen a worker take something --
    # which is what `worker_state()` reports on. It takes the OLDEST queued
    # gpu job, not necessarily this one, which is exactly how a lane drains.
    claimed = jobs.claim("gpu", worker="test")
    check(claimed is not None and claimed["id"],
          "claiming the gpu lane returns a job", json.dumps(claimed or {}))
    if claimed and claimed["id"] != jid:
        jobs.fail(claimed["id"], "test harness released it", retry=False)

    state = jobs.fail(jid, "CUDA out of memory", retry=False)
    check(state == "errored", "an exhausted job lands in errored", state)

    c = datasets.job_counts(ds["id"])
    check(c["errored"] == 1, "job_counts reports one errored", json.dumps(c))
    check(c["done"] == 0,
          "and ZERO done: errored is terminal, and it is not done",
          json.dumps(c))
    check(set(jobs.STATES).issubset(c),
          "every state in jobs.STATES has its own bucket", json.dumps(c))
    check("finished" not in c and "over" not in c,
          "and there is no bucket that lumps the terminal states together",
          json.dumps(c))
    check(c["total"] == sum(c[s] for s in jobs.STATES),
          "the total is the sum of the per-state buckets")

    over = [d for d in datasets.overview()["datasets"] if d["id"] == ds["id"]]
    check(over and over[0]["errored_jobs"],
          "and the overview surfaces the errored job rather than hiding it")
    if over and over[0]["errored_jobs"]:
        check("CUDA out of memory" in (over[0]["errored_jobs"][0]["error"] or ""),
              "with the error text the worker recorded",
              str(over[0]["errored_jobs"][0]["error"]))

    # A re-run is an explicit act on one job, and it returns it to queued --
    # it does NOT mark it done.
    out = datasets.rerun(jid)
    check(out["state"] == "queued", "re-running puts it back in the queue",
          json.dumps(out))
    after = datasets.job_counts(ds["id"])
    check(after["errored"] == 0 and after["queued"] == 1 and after["done"] == 0,
          "and it is queued again, still not done", json.dumps(after))

    raised = None
    try:
        datasets.rerun(jid)
    except ValueError as e:
        raised = str(e)
    check(raised is not None and "errored" in raised,
          "re-running a job that is not errored is refused, so a running job "
          "is never duplicated", str(raised))
    raised = None
    try:
        datasets.rerun("nosuchjob")
    except KeyError as e:
        raised = str(e)
    check(raised is not None, "and an unknown job id is a named error",
          str(raised))


def test_counts_that_were_not_measured_are_absent_rather_than_zero():
    ds = fresh(name="unmeasured set")
    check(datasets.review_counts(ds) is None,
          "review counts are None, not 0, while the jsonl does not exist",
          json.dumps(datasets.review_counts(ds)))
    check(ds["counts"] == {},
          "and a dataset carries no counts until a worker records some",
          json.dumps(ds["counts"]))

    path = os.path.join(FIXTURE_RECIPES, datasets.recipe_file(ds))
    with open(path, "w", encoding="utf-8") as f:
        for state in ("unreviewed", "keep", "keep", "reject"):
            f.write(json.dumps({"recipe": "do the thing", "_state": state,
                                "source_name": "Effective Modern Widgets",
                                "dataset": ds["id"]}) + "\n")
    rc = datasets.review_counts(ds)
    check(rc and rc["total"] == 4 and rc["keep"] == 2 and rc["reject"] == 1,
          "once the file exists the counts are read out of it",
          json.dumps(rc))

    ds = datasets.record_counts(ds["id"], {"rows_extracted": 4})
    check(ds["counts"] == {"rows_extracted": 4},
          "and a worker's own measured counts are stored verbatim",
          json.dumps(ds["counts"]))


def test_the_review_write_back_still_works():
    ds = fresh(name="reviewable set")
    fn = datasets.recipe_file(ds)
    path = os.path.join(FIXTURE_RECIPES, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"recipe": "prefer struct of arrays",
                            "source_name": "Effective Modern Widgets",
                            "source_url": "https://example.invalid/widgets",
                            "language": "typescript"}) + "\n")
        f.write(json.dumps({"recipe": "always inline everything",
                            "source_name": "Effective Modern Widgets",
                            "language": "typescript"}) + "\n")

    rows = [r for r in dashboard.load_all() if r["_file"] == fn]
    check(len(rows) == 2, "the fixture rows load through dashboard.load_all",
          str(len(rows)))
    check(all(r["_state"] == "unreviewed" for r in rows),
          "and start unreviewed")

    rid = f"{fn}:0"
    out = dash_data.handle_post("/dash/api/recipe", {"id": rid})
    check(out is None,
          "dash_data does not claim /dash/api/recipe: the corpus page still "
          "owns the review write-back", json.dumps(out))

    hit = dashboard.handle_post("/dash/api/recipe",
                                {"id": rid, "recipe": "prefer struct of arrays",
                                 "_state": "keep"})
    check(hit and hit[0] == 200, "a keep is accepted", str(hit and hit[0]))
    body = json.loads(hit[2])
    check(body.get("ok") and body["row"]["_state"] == "keep",
          "and comes back as kept", json.dumps(body)[:200])

    hit = dashboard.handle_post(
        "/dash/api/recipe",
        {"id": f"{fn}:1", "recipe": "inline only in hot loops",
         "_state": "edited"})
    body = json.loads(hit[2])
    check(body.get("ok") and body["row"]["recipe"] == "inline only in hot loops",
          "an edit is written back", json.dumps(body)[:200])

    on_disk = [json.loads(ln) for ln in
               open(path, encoding="utf-8").read().splitlines() if ln.strip()]
    check(on_disk[0]["_state"] == "keep" and on_disk[1]["_state"] == "edited",
          "and both land in the jsonl on disk, which is what git diffs",
          json.dumps(on_disk))
    check(on_disk[0]["source_url"] == "https://example.invalid/widgets",
          "provenance is untouched by a review: it is read-only",
          json.dumps(on_disk[0]))

    rc = datasets.review_counts(datasets.get(ds["id"]))
    check(rc and rc["keep"] == 1 and rc["edited"] == 1,
          "and the dataset's review tally reflects the write-back",
          json.dumps(rc))


def test_the_page_and_its_api_answer_without_a_worker():
    hit = dash_data.handle_get("/dash/data")
    check(hit and hit[0] == 200 and b"DATASET MANAGER" in hit[2],
          "/dash/data serves the page", str(hit and hit[0]))
    check(hit and b"/dash/api/recipe" in hit[2],
          "and the page posts reviews to the existing endpoint")

    hit = dash_data.handle_get("/dash/api/datasets")
    check(hit and hit[0] == 200, "/dash/api/datasets answers",
          str(hit and hit[0]))
    d = json.loads(hit[2])
    check(d["datasets"], "with the datasets created above",
          str(len(d["datasets"])))
    check(d["worker"]["ever_claimed"] >= 1,
          "and reports, from the database, whether anything ever claimed a "
          "job", json.dumps(d["worker"]))
    literal = {s: sum(1 for j in jobs.listing(limit=10000) if j["state"] == s)
               for s in ("done", "errored")}
    check(d["queue"]["states"]["done"] == literal["done"]
          and d["queue"]["states"]["errored"] == literal["errored"] >= 1,
          "the snapshot's done counts only rows literally done, and errored "
          "is reported beside it rather than rounded up into it",
          json.dumps([d["queue"]["states"], literal]))
    check(set(d["enqueues"]) == set(datasets.ENQUEUE),
          "the page is told which stages actually enqueue, so it can say "
          "'enqueue' only where a row really goes in the queue",
          json.dumps(sorted(d["enqueues"])))

    one = d["datasets"][0]
    hit = dash_data.handle_get("/dash/api/datasets/" + one["id"])
    check(hit and hit[0] == 200, "a single dataset answers", str(hit and hit[0]))
    hit = dash_data.handle_get("/dash/api/datasets/nosuch")
    check(hit and hit[0] == 404 and b"no such dataset" in hit[2],
          "an unknown id is a named 404, not an empty 200",
          str(hit and hit[0]))

    check(dash_data.handle_get("/dash/api/stats") is None,
          "dash_data declines paths that are not its own, so the dispatch "
          "chain in server.py reaches dashboard")
    check(dash_data.handle_post("/dash/api/nothing", {}) is None,
          "and declines unknown POSTs rather than 400ing them")


def test_the_api_creates_answers_and_refuses_through_http_shapes():
    hit = dash_data.handle_post("/dash/api/dataset",
                                {"prompt": "https://example.invalid/book"})
    check(hit and hit[0] == 200, "POST /dash/api/dataset records a submission",
          str(hit and hit[0]))
    body = json.loads(hit[2])
    did = body["dataset"]["id"]
    check(body["missing"], "and answers with the questions it raises",
          json.dumps([m["field"] for m in body["missing"]]))

    hit = dash_data.handle_post("/dash/api/dataset/advance", {"id": did})
    check(hit and hit[0] == 409, "advancing it is refused with a 409",
          str(hit and hit[0]))
    body = json.loads(hit[2])
    check(body.get("ok") is False and body.get("reasons"),
          "carrying the reasons, not a bare false", json.dumps(body)[:200])

    hit = dash_data.handle_post(
        "/dash/api/dataset/answer",
        dict(ANSWERS, id=did, licence="unknown"))
    body = json.loads(hit[2])
    check(body["ok"] and [m["field"] for m in body["missing"]] == ["licence"],
          "answering all but the licence leaves exactly the licence open",
          json.dumps([m["field"] for m in body["missing"]]))

    hit = dash_data.handle_post("/dash/api/dataset/answer",
                                {"id": did, "licence": "MIT"})
    body = json.loads(hit[2])
    check(body["ok"] and not body["missing"],
          "and establishing it closes the last question", json.dumps(body["missing"]))

    settle(did)
    hit = dash_data.handle_post("/dash/api/dataset/advance", {"id": did})
    body = json.loads(hit[2])
    check(hit[0] == 200 and body["dataset"]["stage"] == "extract",
          "now it advances", json.dumps(body)[:200])
    check(body.get("enqueued"),
          "and the response names the job row it created", str(body.get("enqueued")))
    j = jobs.get(body["enqueued"])
    check(j and j["lane"] == "gpu" and j["state"] == "queued",
          "which is a queued gpu job, waiting for the worker to claim it", json.dumps({k: j[k] for k in ("lane", "state")} if j else {}))

    hit = dash_data.handle_post("/dash/api/dataset", {"prompt": ""})
    check(hit and hit[0] == 400 and b"nothing submitted" in hit[2],
          "an empty submission is refused with what was wrong with it",
          str(hit and hit[0]))


def test_the_nav_carries_the_new_page_exactly_once():
    keys = [n for n, _, _ in dash_shell.NAV]
    check("data" in keys, "dash_shell.NAV carries the data page", str(keys))
    check(keys.count("data") == 1, "exactly once")
    hrefs = [h for _, _, h in dash_shell.NAV]
    check("/dash/data" in hrefs, "at /dash/data", str(hrefs))
    check("/dash" in hrefs and "/dash/vitals" in hrefs
          and "/dash/results" in hrefs,
          "and the three existing pages are still in the nav", str(hrefs))
    markup = dash_shell.nav("data")
    check('href="/dash/data"' in markup and 'aria-current="page"' in markup,
          "and the data page marks itself current")
    check(dash_shell.nav("corpus").count('aria-current="page"') == 1,
          "while the corpus page still marks exactly one thing current")


def test_the_page_shows_no_number_it_did_not_measure():
    html = dash_data.PAGE
    check("<progress" not in html,
          "there is no <progress> element: nothing in this system measures "
          "how far through a stage a job is")
    # A bar drawn from a count would have to set its own width somewhere.
    check("style=\"width:" not in html and ".style.width" not in html
          and "width:'+" not in html,
          "and nothing sets a width from a number, which is the only way a "
          "bar could be drawn from an invented fraction")
    check("Math.round(n/" not in html and "/Math.max(" not in html,
          "no percentage is computed from a job count")
    check("jobs.claim()" in html,
          "the page names the interface a worker would use")
    check("not written yet" in html,
          "and says, in words, that the worker does not exist")
    check("j.progress" in html,
          "a worker's own progress string is printed verbatim rather than "
          "turned into a percentage")
    # The five states, per DESIGN.md. `is-stale` is the one that gets skipped.
    for cls in ("is-loading", "is-error", "state-title", "state-detail",
                "is-stale"):
        check(cls in html, f"the page ships the {cls} state")


def main() -> int:
    for fn in (test_the_fixture_is_not_the_real_database,
               test_missing_fields_are_named_with_a_reason,
               test_unknown_is_not_an_answer,
               test_a_restricted_licence_is_surfaced_but_does_not_block,
               test_an_unknown_licence_cannot_advance_past_clarify,
               test_stage_transitions_enqueue_the_right_lane,
               test_a_laya_set_gets_label_and_train_and_a_recipes_set_does_not,
               test_a_url_submission_enqueues_a_net_fetch_and_pasted_text_does_not,
               test_an_errored_job_is_never_counted_as_done,
               test_counts_that_were_not_measured_are_absent_rather_than_zero,
               test_the_review_write_back_still_works,
               test_the_page_and_its_api_answer_without_a_worker,
               test_the_api_creates_answers_and_refuses_through_http_shapes,
               test_the_nav_carries_the_new_page_exactly_once,
               test_the_page_shows_no_number_it_did_not_measure):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    print(f"  temp database: {os.path.abspath(jobs.DB)}")
    if passed < total:
        print("  The pipeline model is wrong, not the harness.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
