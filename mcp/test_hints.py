#!/usr/bin/env python
"""Bucket collapse in hints.select(), asserted. No GPU, no network.

WHAT THIS IS GATING

Live, "static array, many range-sum queries" returned prefix sums AND
"Point updates interleaved with range sums: Fenwick tree" -- two mutually
exclusive answers in front of the model at once (docs/HANDOFF.md). The fix is
bench/hint_buckets.jsonl plus a collapse to the bucket's highest-cosine member,
the strategy bench/hint_collapse.py measured best (71.9%, n=89). This file
pins the properties that fix depends on:

  1. TWO SIBLINGS NEVER BOTH REACH THE MODEL, and the one kept is the argmax
     -- including when it is not the top hint overall.
  2. THE FLOOR STILL DECIDES EMISSION. A slot freed by collapse is refilled
     only from above the floor; fewer hints, or none, is a correct output.
  3. UNBUCKETED ROWS BEHAVE EXACTLY AS BEFORE.
  4. A REJECTED ROW STAYS OUT, and its bucket passes to the next sibling.
  5. BUCKETS MATCH ON VERBATIM TEXT, and a member that does not match is
     reported loudly, not silently served unbucketed.
  6. THE HINT CACHE STAYS VALID. Buckets are outside the embedding signature,
     so editing the bucket file never re-embeds the corpus.

HOW IT IS ISOLATED

The embedder is a fake `code_search` in `sys.modules`, as in
mcp/test_worker.py. Documents embed to one-hot vectors, and a query embeds to
the exact vector of cosines the test wants, so every score below is chosen,
not approximated. Corpus, cache and bucket file are temp paths; the real
index/hints.npz is asserted untouched. One check READS the real corpus and
bucket file (never writes) to confirm all 89 members still match.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_hints_")

import numpy as np  # noqa: E402

# Recipe texts, and the one-hot slot each embeds to.
PREFIX = "Static array, many range-sum queries: prefix sums."
FENWICK = "Point updates interleaved with range sums: Fenwick tree."
LAZY = "Range updates and range queries: lazy segment tree."
CONSTRAINT = "Segment trees need an associative combining operation."
DIFF = "Many range increments read only at the end: difference array."
UNRELATED = "Prefer composition of small widgets over one mega-widget."
DIJKSTRA = "Non-negative weights: Dijkstra with a binary heap."
BELLMAN = "Negative edges allowed: Bellman-Ford, and it detects cycles."
DOCS = [PREFIX, FENWICK, LAZY, CONSTRAINT, DIFF, UNRELATED, DIJKSTRA, BELLMAN]
SLOT = {t: i for i, t in enumerate(DOCS)}

# query text -> {recipe: cosine}. Anything unlisted scores 0.
QUERIES: dict[str, dict[str, float]] = {}

_fake_cs = types.ModuleType("code_search")
_fake_cs.calls = []


def _embed(texts, is_query=False):
    _fake_cs.calls.append((len(texts), is_query))
    out = []
    for t in texts:
        v = np.zeros(len(DOCS), dtype=np.float32)
        if is_query:
            for recipe, cos in QUERIES.get(t, {}).items():
                v[SLOT[recipe]] = cos
        else:
            # A document's embedded text is _hint_text(row): recipe first line.
            v[SLOT[t.split("\n")[0]]] = 1.0
        out.append(v)
    return np.vstack(out)


_fake_cs.embed = _embed
sys.modules["code_search"] = _fake_cs

import hints  # noqa: E402

REAL_CORPUS, REAL_CACHE, REAL_BUCKETS = hints.CORPUS, hints.CACHE, hints.BUCKETS
RECIPES = os.path.join(_TMP, "recipes")
os.makedirs(RECIPES, exist_ok=True)
hints.CORPUS = RECIPES
hints.CACHE = os.path.join(_TMP, "hints.npz")
hints.BUCKETS = os.path.join(_TMP, "buckets.jsonl")

FLOOR = 0.55

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
def write_corpus(reject: tuple = ()):
    with open(os.path.join(RECIPES, "fixture.jsonl"), "w",
              encoding="utf-8") as fh:
        for t in DOCS:
            row = {"recipe": t, "source_name": "fixture"}
            if t in reject:
                row["_state"] = "reject"
            fh.write(json.dumps(row) + "\n")


BUCKET_ROWS = [
    {"bucket_id": "range_structure", "members": [
        {"member_id": "prefix", "ref": "fixture:99", "recipe": PREFIX},
        {"member_id": "fenwick", "ref": "fixture:1", "recipe": FENWICK},
        {"member_id": "lazy", "ref": "fixture:2", "recipe": LAZY}]},
    {"bucket_id": "shortest_path", "members": [
        {"member_id": "dijkstra", "ref": "fixture:6", "recipe": DIJKSTRA},
        {"member_id": "bellman", "ref": "fixture:7", "recipe": BELLMAN}]},
]


def write_buckets(rows=None, extra_lines: tuple = ()):
    with open(hints.BUCKETS, "w", encoding="utf-8") as fh:
        for b in (BUCKET_ROWS if rows is None else rows):
            fh.write(json.dumps(b) + "\n")
        for line in extra_lines:
            fh.write(line + "\n")


def reset(reject: tuple = (), buckets=None, extra_lines: tuple = ()):
    write_corpus(reject)
    write_buckets(buckets, extra_lines)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        hints.refresh()
    return err.getvalue()


def ask(name: str, scores: dict[str, float], **kw) -> list[dict]:
    QUERIES[name] = scores
    return hints.select(name, k=kw.pop("k", 3), floor=FLOOR, **kw)


def recipes(out: list[dict]) -> list[str]:
    return [h["recipe"] for h in out]


# The live failure, as cosines: three range_structure siblings on top.
RANGE_SUM = {PREFIX: 0.78, CONSTRAINT: 0.73, FENWICK: 0.72, LAZY: 0.70,
             DIFF: 0.69, UNRELATED: 0.30}


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_the_fixture_is_not_the_real_index():
    for p in (hints.CORPUS, hints.CACHE, hints.BUCKETS):
        check(os.path.abspath(p).startswith(os.path.abspath(_TMP)),
              f"redirected to temp: {os.path.basename(p)}", p)
    check(os.path.abspath(REAL_CACHE) != os.path.abspath(hints.CACHE),
          "the real index/hints.npz is not the cache under test")


def test_the_live_failure_reproduces_without_collapse():
    reset()
    out = ask("range sums", RANGE_SUM, collapse=False)
    b = [h.get("_bucket") for h in out]
    check(b.count("range_structure") == 2,
          "collapse=False still shows two range_structure siblings (the "
          "regression this fix exists for is reproducible)", str(recipes(out)))


def test_siblings_collapse_to_the_argmax():
    reset()
    out = ask("range sums", RANGE_SUM)
    got = recipes(out)
    check(got == [PREFIX, CONSTRAINT, DIFF],
          "prefix sums kept, Fenwick withheld, the constraint survives, the "
          "freed slot goes to the next hint above the floor", str(got))
    check(sum(1 for h in out if h.get("_bucket") == "range_structure") == 1,
          "exactly one range_structure member reaches the model")
    sup = [s["recipe"] for s in out[0].get("_suppressed", [])]
    check(sup == [FENWICK, LAZY],
          "the withheld siblings are recorded on the winner, in score order",
          str(sup))
    check("_bucket" not in out[1] and "_suppressed" not in out[1],
          "an unbucketed hint carries no bucket annotations")


def test_the_winner_is_the_bucket_argmax_even_when_not_first_overall():
    reset()
    out = ask("negative edges", {CONSTRAINT: 0.90, BELLMAN: 0.80,
                                 DIJKSTRA: 0.79, FENWICK: 0.60})
    got = recipes(out)
    check(got == [CONSTRAINT, BELLMAN, FENWICK],
          "Bellman-Ford (0.80) beats Dijkstra (0.79) inside shortest_path; "
          "two different buckets may each contribute one", str(got))


def test_ties_resolve_the_same_way_every_time():
    reset()
    a = recipes(ask("tie", {FENWICK: 0.7, LAZY: 0.7}))
    b = recipes(ask("tie", {FENWICK: 0.7, LAZY: 0.7}))
    check(a == b == [FENWICK],
          "equal cosines resolve by corpus order, deterministically", str(a))


def test_the_floor_still_decides_emission():
    reset()
    out = ask("two siblings only", {PREFIX: 0.70, FENWICK: 0.65, DIFF: 0.50,
                                    UNRELATED: 0.54})
    check(recipes(out) == [PREFIX],
          "the freed slot is NOT filled from below the floor -- one hint is "
          "a correct output", str(recipes(out)))
    out = ask("nothing clears", {PREFIX: 0.54, FENWICK: 0.53})
    check(out == [], "all below the floor -> silent, collapse or not")
    check(recipes(ask("nothing clears", {PREFIX: 0.54}, collapse=False)) == [],
          "and silent without collapse too")


def test_k_is_a_ceiling_not_a_target():
    reset()
    out = ask("range sums", RANGE_SUM, k=1)
    check(recipes(out) == [PREFIX], "k=1 returns the top hint only",
          str(recipes(out)))
    out = ask("range sums", RANGE_SUM, k=10)
    check(recipes(out) == [PREFIX, CONSTRAINT, DIFF],
          "k=10 still returns only what clears the floor, one per bucket",
          str(recipes(out)))


def test_unbucketed_rows_behave_exactly_as_before():
    reset()
    s = {CONSTRAINT: 0.9, DIFF: 0.8, UNRELATED: 0.6}
    on, off = ask("unbucketed", s), ask("unbucketed", s, collapse=False)
    check(recipes(on) == recipes(off) == [CONSTRAINT, DIFF, UNRELATED],
          "no bucket involved -> identical output with collapse on and off",
          f"{recipes(on)} vs {recipes(off)}")
    check([h["_score"] for h in on] == [0.9, 0.8, 0.6],
          "scores unchanged", str([h["_score"] for h in on]))


def test_a_rejected_member_stays_out_and_its_bucket_passes_on():
    err = reset(reject=(PREFIX,))
    rows, _ = hints.load()
    check(PREFIX not in [r["recipe"] for r in rows],
          "_state=reject still removes the row from the corpus")
    out = ask("range sums", RANGE_SUM)
    check(recipes(out) == [CONSTRAINT, FENWICK, DIFF],
          "with prefix sums rejected, Fenwick holds the bucket", str(recipes(out)))
    st = hints.bucket_status()
    check(st["matched"] == 4 and any("range_structure" in u
                                     for u in st["unmatched"]),
          "the rejected member is reported unmatched", str(st))
    check("buckets degraded" in err, "and the degradation is printed", err)


def test_members_match_on_verbatim_text_not_ref():
    err = reset()
    st = hints.bucket_status()
    check(st["matched"] == st["members"] == 5 and st["rows_bucketed"] == 5,
          "all 5 members matched although prefix's ref points at no row",
          str(st))
    check(err == "", "a clean bucket file prints nothing", err)
    edited = json.loads(json.dumps(BUCKET_ROWS))
    edited[0]["members"][1]["recipe"] = FENWICK + " (edited upstream)"
    err = reset(buckets=edited)
    st = hints.bucket_status()
    check(st["matched"] == 4 and len(st["unmatched"]) == 1,
          "an edited member is unmatched and named", str(st["unmatched"]))
    check("buckets degraded" in err and "can argue" in err,
          "and the warning says what it costs", err)
    out = ask("range sums", RANGE_SUM)
    check(FENWICK in recipes(out) and PREFIX in recipes(out),
          "the unmatched member is served unbucketed (today's behaviour), "
          "which is why the warning exists", str(recipes(out)))


def test_a_missing_or_malformed_bucket_file_degrades_to_today():
    write_corpus()
    if os.path.exists(hints.BUCKETS):
        os.remove(hints.BUCKETS)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        hints.refresh()
    st = hints.bucket_status()
    check(st["error"] and "buckets degraded" in err.getvalue(),
          "a missing bucket file is an error in the status and on stderr",
          f"{st['error']!r} / {err.getvalue()!r}")
    out = ask("range sums", RANGE_SUM)
    check(recipes(out) == [PREFIX, CONSTRAINT, FENWICK],
          "and select() still answers, unbucketed", str(recipes(out)))
    reset(extra_lines=("{not json", json.dumps({"members": []})))
    st = hints.bucket_status()
    check(st["buckets"] == 2 and st["matched"] == 5 and st["error"],
          "malformed lines are skipped and named; the good ones still load",
          str(st))


def test_the_cache_is_not_invalidated_by_buckets():
    reset()
    before = dict(np.load(hints.CACHE))
    _fake_cs.calls.clear()
    edited = [BUCKET_ROWS[0]]                  # drop shortest_path
    reset(buckets=edited)
    docs = [c for c in _fake_cs.calls if not c[1]]
    check(docs == [], "editing the bucket file does not re-embed documents",
          str(_fake_cs.calls))
    after = dict(np.load(hints.CACHE))
    check(str(before["sig"]) == str(after["sig"]),
          "cache signature unchanged by buckets")
    rows, _ = hints.load()
    texts = [hints._hint_text(r) for r in rows]
    check(all("range_structure" not in t for t in texts),
          "_bucket never enters the embedded text")
    out = ask("negative edges", {BELLMAN: 0.80, DIJKSTRA: 0.79})
    check(recipes(out) == [BELLMAN, DIJKSTRA],
          "refresh() picks up the bucket edit: shortest_path is no longer "
          "collapsed", str(recipes(out)))


def test_the_model_never_sees_a_withheld_sibling():
    reset()
    QUERIES["static array, many range-sum queries"] = RANGE_SUM
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "static array, many range-sum queries"}]
    out, used = hints.attach(msgs, k=3, floor=FLOOR)
    text = out[-1]["content"]
    check(PREFIX in text and FENWICK not in text and LAZY not in text,
          "attach(): the user turn carries prefix sums and neither sibling")
    check(out[0] == msgs[0], "the system message is untouched")
    check(len(used) == 3, "attach returns the hints it used", str(len(used)))


def test_the_domain_filter_runs_on_strong_evidence_only():
    """docs/SELECTION-BUILD.md step 6: domains.eligible inside select().

    A recipe outside the task's domains is ABSENT, not ranked low -- and it
    is skipped before the bucket collapse, so it can neither hold a bucket
    nor take a slot. The filter restricts only on strong evidence (an import
    or an extension); a prose task is not filtered at all, which is what
    keeps the 89-probe score where it was (see hints.task_domains).
    """
    tagged = {PREFIX: {"domains": ["algorithms"]},
              FENWICK: {"domains": ["visual-design"]}}
    with open(os.path.join(RECIPES, "fixture.jsonl"), "w",
              encoding="utf-8") as fh:
        for t in DOCS:
            if t == UNRELATED:
                continue
            fh.write(json.dumps(dict({"recipe": t, "source_name": "fixture"},
                                     **tagged.get(t, {}))) + "\n")
    # The strict corpus: closed wherever the task has evidence elsewhere.
    strict = os.path.join(RECIPES, "design_visual.jsonl")
    with open(strict, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"recipe": UNRELATED, "domains": ["visual-design"],
                             "source_name": "design"}) + "\n")
    write_buckets()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            hints.refresh()
        scores = {FENWICK: 0.90, UNRELATED: 0.85, PREFIX: 0.80, DIJKSTRA: 0.70}

        prose = "range sums over widgets, which structure"
        check(hints.task_domains([{"role": "user", "content": prose}]) is None,
              "prose with no import or extension: no filter at all")
        out = recipes(ask(prose, scores))
        check(FENWICK in out and UNRELATED in out,
              "unfiltered, a prose task gets what it got before the filter",
              str(out))

        rust = ("```rs\nuse tokio::net::TcpListener;\n```\n"
                "range sums over widgets, which structure")
        task = hints.task_domains([{"role": "user", "content": rust}])
        check(task is not None and "backend" in task,
              "an import is strong evidence: the task has domains",
              str(sorted(task or [])))
        out = recipes(ask(rust, scores))
        check(UNRELATED not in out,
              "the strict visual-design recipe is absent for backend work",
              str(out))
        check(FENWICK not in out and PREFIX not in out,
              "a tagged recipe outside the task's domains is absent", str(out))
        check(DIJKSTRA in out,
              "an untagged recipe fails open and is still served", str(out))

        # The filter runs BEFORE the collapse: with FENWICK inadmissible, the
        # bucket passes to its highest ELIGIBLE member.
        out = recipes(ask("range sums", scores, task={"algorithms"}))
        check(PREFIX in out and FENWICK not in out,
              "an ineligible sibling cannot hold the bucket; the eligible one "
              "does", str(out))

        msgs = [{"role": "user", "content": rust}]
        _, used = hints.attach(msgs, k=3, floor=FLOOR)
        check(all(h["recipe"] not in (UNRELATED, FENWICK, PREFIX) for h in used),
              "attach() applies the same filter from the conversation",
              str([h["recipe"] for h in used]))
    finally:
        os.remove(strict)
        reset()


def test_the_real_bucket_file_matches_the_real_corpus():
    """READ-ONLY against bench/. A member that stops matching is a pair of
    hints that will argue again in production."""
    saved = hints.CORPUS
    try:
        hints.CORPUS = REAL_CORPUS
        rows = hints._load_corpus()
    finally:
        hints.CORPUS = saved
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        st = hints._assign_buckets(rows, REAL_BUCKETS)
    check(st["buckets"] == 19 and st["members"] == 89,
          "the real file is 19 buckets / 89 members", str(st))
    check(st["matched"] == 89 and not st["unmatched"] and not st["error"],
          "every real member matches a served recipe verbatim",
          f"{st['matched']}/89 {st['unmatched'][:3]} {st['error']}")


def main() -> int:
    real_mtime = (os.path.getmtime(REAL_CACHE)
                  if os.path.exists(REAL_CACHE) else None)
    for fn in (test_the_fixture_is_not_the_real_index,
               test_the_live_failure_reproduces_without_collapse,
               test_siblings_collapse_to_the_argmax,
               test_the_winner_is_the_bucket_argmax_even_when_not_first_overall,
               test_ties_resolve_the_same_way_every_time,
               test_the_floor_still_decides_emission,
               test_k_is_a_ceiling_not_a_target,
               test_unbucketed_rows_behave_exactly_as_before,
               test_a_rejected_member_stays_out_and_its_bucket_passes_on,
               test_members_match_on_verbatim_text_not_ref,
               test_a_missing_or_malformed_bucket_file_degrades_to_today,
               test_the_cache_is_not_invalidated_by_buckets,
               test_the_model_never_sees_a_withheld_sibling,
               test_the_domain_filter_runs_on_strong_evidence_only,
               test_the_real_bucket_file_matches_the_real_corpus):
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

    check(real_mtime == (os.path.getmtime(REAL_CACHE)
                         if os.path.exists(REAL_CACHE) else None),
          "the real index/hints.npz was not written")
    ok, name, detail = _results[-1]
    print(("  pass  " if ok else "  FAIL  ") + name)

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    print(f"  temp dir: {_TMP}")
    if passed < total:
        print("  hints.select() is wrong, not the harness.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
