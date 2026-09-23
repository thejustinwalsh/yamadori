#!/usr/bin/env python
"""Can Laya collapse a MUTUALLY EXCLUSIVE bucket of hints better than retrieval?

THE ONE QUESTION THIS FILE ANSWERS

Laya (`mcp/laya_service.py`, POST /decide) is an option-scoring cross-encoder.
Its input is

    [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

one scalar per option marker, softmax ACROSS the markers in a single forward
pass. The options are therefore normalised AGAINST EACH OTHER. Nothing else in
this stack has that property: `code_search.embed()` scores every candidate
independently and compares afterwards, and `code_search.rerank()` cross-encodes
(query, doc) one document at a time.

Joint normalisation should only pay off where the candidates genuinely COMPETE
-- a set of alternatives of which exactly one applies. So that is the only place
it is tested here. If it loses on a mutually exclusive bucket it has no job in
this stack, and that is a completely acceptable answer.

WHAT IS MEASURED, AND AGAINST WHAT

Three headline arms, three floors and three diagnostics, on identical buckets
and identical probes:

    random          floor. Seeded per (seed, probe) so a rerun draws the same
                    picks.
    fixed           harder floor: always the bucket's first member,
                    problem-blind.
    lexical         the floor that matters most for a retrieval claim. Token
                    overlap, no model, no GPU. If a neural arm does not beat
                    this, it is string matching with extra steps.
    embedding       cosine via code_search.embed(). Qwen3-Embedding is
                    ASYMMETRIC: the probe is embedded as a QUERY (instruction
                    prefix applied inside embed()), the hints as DOCUMENTS with
                    no prefix. Getting that backwards returns near-random
                    results silently.
    rerank          code_search.rerank(), the Qwen3-Reranker cross-encoder, one
                    (probe, hint) pair at a time -- see arm_rerank for why one
                    at a time is mandatory and not a style choice.
    laya            permutation-averaged joint choice over ALL bucket members at
                    once, through bench/mechanisms/selectors.py:_laya_choice.

    embedding_cond  diagnostic: embed only each hint's applicability CONDITION.
    laya_cond       diagnostic: the same, as Laya's option text.
    rerank_batched  diagnostic: the batched rerank call, which is broken.

The three diagnostics exist because "the mechanism lost" is only a result once
the obvious repair has been tried and reported. Two of them move the number and
neither changes the verdict.

HINTS MUST NEVER HARM, SO COVERAGE IS SCORED, NOT JUST ACCURACY

Every arm can abstain. Reporting accuracy-when-it-answers rewards an arm for
abstaining, which is how docs/LAYA.md finding 1 nearly shipped a router that
declined 91.5% of questions. So the headline table here is PRECISION AT MATCHED
COVERAGE: each arm's own confidence signal is sorted and truncated so all arms
commit on the SAME fraction of probes, and precision is compared there. An arm
is only allowed to look better if it is better at the same workload.

Confidence signal per arm, all top-1 minus top-2 within one bucket:
    embedding   cosine margin
    rerank      margin after sum-normalising the bucket's scores. The rerank
                scale is meaningless in absolute terms (~1e-6 here), so an
                un-normalised gap is not comparable between buckets. This is a
                CHOICE and it is the most favourable honest one available.
    laya        the permutation-averaged probability margin, i.e. the same
                number `LAYA_MARGIN_FLOOR` / `TREE_MARGIN_GATE` are gates on.
    lexical     margin after sum-normalising the overlap scores.
    random/fixed have no confidence. They are reported flat -- a floor that
    cannot rank its own answers has no coverage curve, and pretending otherwise
    would flatter it.

RUNNING IT

    python -X utf8 bench/hint_collapse.py            # live; writes the cache
    python -X utf8 bench/hint_collapse.py --replay   # no GPU, from the cache
    python -X utf8 bench/hint_collapse.py --arms laya,embedding

Raw per-ordering Laya probabilities, raw cosines and raw rerank scores go to
bench/data/hint_collapse_runs.json so every table replays without a GPU.

RESULTS AND THE VERDICT ARE IN docs/HINTS.md. This file regenerates them.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random as _random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, os.path.join(ROOT, "mcp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SELECTORS = None


def selectors_mod():
    """bench/mechanisms/selectors.py, loaded WITHOUT shadowing the stdlib.

    Putting bench/mechanisms on sys.path would make `import selectors` resolve
    to it for every other module in the process, including anything asyncio
    touches. Loading it by path under a private name cannot do that.
    """
    global _SELECTORS
    if _SELECTORS is None:
        import importlib.util
        path = os.path.join(HERE, "mechanisms", "selectors.py")
        spec = importlib.util.spec_from_file_location("_hint_selectors", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_hint_selectors"] = mod
        spec.loader.exec_module(mod)
        _SELECTORS = mod
    return _SELECTORS

# Nothing here may write to the shared corpus. mcp/corpus.py opens CORPUS_DB on
# import of some paths and would create/touch a file another workstream owns.
os.environ.setdefault(
    "YAMADORI_CORPUS_DB",
    os.path.join(HERE, "data", "_hint_collapse_corpus.sqlite3"))

BUCKETS = os.path.join(HERE, "hint_buckets.jsonl")
PROBES = os.path.join(HERE, "hint_probes.jsonl")
RECIPES = os.path.join(HERE, "recipes")
RUNS = os.path.join(HERE, "data", "hint_collapse_runs.json")

RANDOM_SEED = 20260922

# Coverage grid for the matched-coverage table. 1.00 is "never abstain", which
# is the number that matters when a hint is going to be injected regardless.
COVERAGE_GRID = (1.00, 0.90, 0.75, 0.60, 0.50, 0.40, 0.30, 0.20)

ARMS = ("random", "fixed", "lexical", "embedding", "embedding_cond",
        "rerank", "rerank_batched", "laya", "laya_cond")

LAYA_INSTRUCTIONS = (
    "Exactly one of these applies to the situation described. Pick it.")

# Words that carry no discriminating signal in the lexical floor. Deliberately
# short: a big stoplist would be tuning the floor, and the floor exists to be
# un-tuned.
_STOP = set("""a an and are as at be but by can do does for from has have how
if in into is it its of on one only or our so that the their them then there
these this to up us use used using was we what when where which while who will
with without you your""".split())


# ================================================================== DATA =====

def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_corpus_recipes() -> set[str]:
    texts = set()
    if not os.path.isdir(RECIPES):
        return texts
    for name in sorted(os.listdir(RECIPES)):
        if not name.endswith(".jsonl"):
            continue
        for row in load_jsonl(os.path.join(RECIPES, name)):
            if row.get("recipe"):
                texts.add(row["recipe"])
    return texts


def verify_no_drift(buckets: list[dict]) -> list[str]:
    """Every member's text must still be a verbatim corpus recipe.

    The buckets were CURATED by hand but the text was COPIED from the corpus.
    If someone edits a recipe upstream, a bucket that silently keeps the old
    wording is measuring a hint that no longer ships. This fails loudly instead.
    """
    corpus = load_corpus_recipes()
    if not corpus:
        return ["bench/recipes/*.jsonl not found -- drift check skipped"]
    bad = []
    for b in buckets:
        for m in b["members"]:
            if m["recipe"] not in corpus:
                bad.append(f"{b['bucket_id']}::{m['member_id']} ({m['ref']})")
    return bad


def check_buckets(buckets: list[dict], probes: list[dict]) -> list[str]:
    """Structural invariants. A silently malformed bucket invalidates every arm."""
    problems = []
    ids = set()
    for b in buckets:
        if b["bucket_id"] in ids:
            problems.append(f"duplicate bucket_id {b['bucket_id']}")
        ids.add(b["bucket_id"])
        if len(b["members"]) < 2:
            problems.append(f"{b['bucket_id']}: fewer than 2 members")
        if b["bucket_type"] not in ("contrastive", "topical"):
            problems.append(f"{b['bucket_id']}: unknown bucket_type")
        if not b.get("why_exclusive"):
            problems.append(f"{b['bucket_id']}: no why_exclusive")
        mids = [m["member_id"] for m in b["members"]]
        if len(set(mids)) != len(mids):
            problems.append(f"{b['bucket_id']}: duplicate member_id")
        texts = [m["recipe"] for m in b["members"]]
        if len(set(texts)) != len(texts):
            problems.append(f"{b['bucket_id']}: two members share one recipe")
    by_id = {b["bucket_id"]: b for b in buckets}
    seen = set()
    for p in probes:
        if p["probe_id"] in seen:
            problems.append(f"duplicate probe_id {p['probe_id']}")
        seen.add(p["probe_id"])
        b = by_id.get(p["bucket_id"])
        if b is None:
            problems.append(f"{p['probe_id']}: unknown bucket")
            continue
        if p["correct"] not in [m["member_id"] for m in b["members"]]:
            problems.append(f"{p['probe_id']}: ground truth not in bucket")
    # every member needs a probe, or the bucket is scored on a subset of itself
    want = {(b["bucket_id"], m["member_id"])
            for b in buckets for m in b["members"]}
    got = {(p["bucket_id"], p["correct"]) for p in probes}
    for miss in sorted(want - got):
        problems.append(f"member without a probe: {miss[0]}::{miss[1]}")
    return problems


# ================================================================== ARMS =====

def _tokens(text: str) -> set[str]:
    out = set()
    word = []
    for ch in text.lower():
        if ch.isalnum() or ch in "_":
            word.append(ch)
        else:
            if word:
                out.add("".join(word))
            word = []
    if word:
        out.add("".join(word))
    return {w for w in out if len(w) > 2 and w not in _STOP}


def arm_lexical(problem: str, members: list[dict]) -> dict[str, float]:
    """Token overlap, normalised by hint length. No model, no network.

    This is the arm that decides whether any neural result means anything: the
    probes were written to avoid the hints' distinctive vocabulary, and if this
    floor still scores well, they were not written carefully enough.
    """
    q = _tokens(problem)
    out = {}
    for m in members:
        d = _tokens(m["recipe"])
        out[m["member_id"]] = len(q & d) / math.sqrt(len(d) or 1)
    return out


def arm_embedding(problem: str, members: list[dict], text=None
                  ) -> dict[str, float]:
    """Cosine, asymmetric. The probe is a QUERY (embed() applies the instruction
    prefix); the hints are DOCUMENTS and must not carry one. Reversing that puts
    the query in a different region of the space and retrieval goes near-random
    -- the measured failure was a real question matching an unrelated model at
    0.002.

    `text` selects what is embedded as the document: the whole recipe, or, for
    the `embedding_cond` arm, only its applicability condition. Cosine is a
    topical instrument, so diluting the discriminating clause in 200 characters
    of advice is the obvious suspect for the contrastive gap in Finding 5.
    """
    import code_search as cs
    text = text or (lambda m: m["recipe"])
    qv = cs.embed([problem], is_query=True)[0]
    dv = cs.embed([text(m) for m in members], is_query=False)
    return {m["member_id"]: float(sum(float(a) * float(b) for a, b in zip(qv, v)))
            for m, v in zip(members, dv)}


def arm_rerank(problem: str, members: list[dict]) -> dict[str, float]:
    """ONE DOCUMENT PER CALL. This is not a style choice, it is a bug workaround.

    The brief's definition of this arm is "cross-encodes (query, doc) one doc at
    a time", and the served /v1/rerank endpoint turns out to REQUIRE that. Sent a
    batch, it contaminates the scores across the batch: with four documents and
    the query "what is the capital of France", the sourdough document scores
    0.443 inside the batch and 1.6e-09 alone, and the Paris document -- which is
    the top-scoring document when scored alone -- comes LAST. Four identical
    batched calls inside one process return identical numbers, so it is not
    sampling noise, but the same call in a later session returned a different
    winner: it is state leaking between slots in one rerank request.

    The measurable consequence is in `arm_rerank_batched`: batched, the endpoint
    retrieves a document from its own VERBATIM text 16/89 times. Scored one at a
    time, the same endpoint gets 68/89. `bench/test_hint_collapse.py --live`
    asserts both, so this stops being a mystery if it is ever fixed.
    """
    import code_search as cs
    return {m["member_id"]: float(cs.rerank(problem, [m["recipe"]], 1)[0][1])
            for m in members}


def arm_rerank_batched(problem: str, members: list[dict]) -> dict[str, float]:
    """The batched call, kept as a DIAGNOSTIC, not as the rerank arm.

    Reported so the serving bug above is visible as a number instead of a
    footnote, and so a fix to llama-swap can be detected by this arm's score
    moving.
    """
    import code_search as cs
    docs = [m["recipe"] for m in members]
    ranked = cs.rerank(problem, docs, len(docs))
    scores = {m["member_id"]: 0.0 for m in members}
    for idx, s in ranked:
        scores[members[idx]["member_id"]] = float(s)
    return scores


def condition_text(m: dict) -> str:
    """The applicability CONDITION of a hint, not the whole hint.

    Most recipes are written "<condition>: <advice>", and several carry an
    explicit `trigger_condition`. Laya's options are rendered into the input as
    text, so a 200-character option is a passage, and docs/LAYA.md's measured
    weak regime is exactly "options are different passages". Handing it only the
    condition is the most favourable honest configuration available, and it is
    reported as its own arm rather than swapped in silently.
    """
    t = (m.get("trigger_condition") or "").strip()
    if t:
        return t
    r = m["recipe"]
    i = r.find(":")
    head = r[:i] if 0 < i < 110 else r.split(". ")[0]
    return head[:110]


@contextlib.contextmanager
def _recording_post(sink: list):
    """Wrap selectors' POST hook so _laya_choice does the real work and we still
    capture the RAW per-ordering probabilities for replay."""
    import code_search as cs
    sel = selectors_mod()

    def hook(url, payload, timeout=30):
        d = cs._post_json(url, payload, timeout=timeout)
        probs = (d.get("answers", {}).get("pick", {}).get("probabilities") or {})
        sink.append({"order": list(payload["questions"]["pick"]["criteria"]),
                     "probabilities": {k: float(v) for k, v in probs.items()}})
        return d

    with sel.mock_backends(post=hook):
        yield


@contextlib.contextmanager
def _replay_post(orders: list):
    """Serve _laya_choice from the cache, in the order it asked last time."""
    sel = selectors_mod()
    queue = list(orders)

    def hook(url, payload, timeout=30):
        want = list(payload["questions"]["pick"]["criteria"])
        for i, rec in enumerate(queue):
            if rec["order"] == want:
                return {"answers": {"pick": {
                    "probabilities": queue.pop(i)["probabilities"]}}}
        raise RuntimeError(f"no cached Laya ordering for {want}")

    with sel.mock_backends(post=hook):
        yield


def arm_laya(problem: str, members: list[dict], orders_cache=None,
             option_text=None):
    """One joint, permutation-averaged choice over the WHOLE bucket.

    Reuses bench/mechanisms/selectors.py:_laya_choice rather than reimplementing
    it: Laya is order-sensitive, a single call is not a reading, and that
    function already carries the contract guard (candidates in `criteria` only,
    never repeated into `state`) and the degenerate-order check. n_permutations
    is the member count so every option occupies a spread of positions instead
    of one option sitting in the middle every time.
    """
    sel = selectors_mod()
    option_text = option_text or (lambda m: m["recipe"])
    criteria = {m["member_id"]: option_text(m) for m in members}
    n = max(2, len(members))
    sink: list = []
    if orders_cache is None:
        with _recording_post(sink):
            v = sel._laya_choice(problem, LAYA_INSTRUCTIONS, criteria,
                                 n_permutations=n)
    else:
        with _replay_post(orders_cache):
            v = sel._laya_choice(problem, LAYA_INSTRUCTIONS, criteria,
                                 n_permutations=n)
        sink = list(orders_cache)
    return v, sink


# ============================================================== SCORING ======

def _rank(scores: dict[str, float]) -> tuple[str, float]:
    """(pick, margin). Margin is top1-top2 after sum-normalising, so it is
    comparable between buckets of different sizes and between arms."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    vals = [v for _, v in ranked]
    lo = min(vals)
    shifted = [v - lo for v in vals]          # rerank/cosine can be negative
    total = sum(shifted) or 1.0
    norm = [v / total for v in shifted]
    margin = norm[0] - norm[1] if len(norm) > 1 else 1.0
    return ranked[0][0], round(float(margin), 6)


def evaluate(records: list[dict], arm: str, subset=None) -> dict:
    rows = [r for r in records if subset is None or subset(r)]
    rows = [r for r in rows if arm in r["arms"]]
    n = len(rows)
    correct = sum(1 for r in rows if r["arms"][arm]["pick"] == r["correct"])
    return {"n": n, "correct": correct, "acc": correct / n if n else 0.0}


def coverage_curve(records: list[dict], arm: str, subset=None) -> list[tuple]:
    """(coverage, n_committed, precision) at each point of COVERAGE_GRID.

    Sorting by the arm's own confidence and truncating is exactly equivalent to
    sweeping that arm's abstention threshold, and it forces every arm onto the
    same denominator. An arm with no confidence signal is reported flat.
    """
    rows = [r for r in records if subset is None or subset(r)]
    rows = [r for r in rows if arm in r["arms"]]
    if not rows:
        return []
    flat = all(r["arms"][arm].get("margin") is None for r in rows)
    hits = [(r["arms"][arm].get("margin") or 0.0,
             1 if r["arms"][arm]["pick"] == r["correct"] else 0) for r in rows]
    hits.sort(key=lambda x: -x[0])
    base = sum(h for _, h in hits) / len(hits)
    out = []
    for cov in COVERAGE_GRID:
        k = max(1, int(round(cov * len(hits))))
        if flat:
            out.append((cov, k, base))
        else:
            out.append((cov, k, sum(h for _, h in hits[:k]) / k))
    return out


def gate_report(records: list[dict], arm: str, gate: float, subset=None) -> dict:
    rows = [r for r in records if subset is None or subset(r)]
    rows = [r for r in rows if arm in r["arms"]]
    fired = [r for r in rows
             if (r["arms"][arm].get("margin") or 0.0) >= gate]
    hit = sum(1 for r in fired if r["arms"][arm]["pick"] == r["correct"])
    return {"n": len(rows), "fired": len(fired),
            "coverage": len(fired) / len(rows) if rows else 0.0,
            "precision": hit / len(fired) if fired else float("nan"),
            "cov_acc": hit / len(rows) if rows else 0.0}


def _binom_two_sided(b: int, c: int) -> float:
    """Exact McNemar. n=b+c discordant pairs, p=0.5 under the null."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(records: list[dict], a: str, b: str, subset=None) -> dict:
    rows = [r for r in records if subset is None or subset(r)]
    rows = [r for r in rows if a in r["arms"] and b in r["arms"]]
    a_only = b_only = both = neither = 0
    for r in rows:
        ha = r["arms"][a]["pick"] == r["correct"]
        hb = r["arms"][b]["pick"] == r["correct"]
        if ha and hb:
            both += 1
        elif ha:
            a_only += 1
        elif hb:
            b_only += 1
        else:
            neither += 1
    return {"n": len(rows), "both": both, "neither": neither,
            f"{a}_only": a_only, f"{b}_only": b_only,
            "discordant": a_only + b_only,
            "p": _binom_two_sided(a_only, b_only)}


# =============================================================== RUNNING =====

def run(buckets: list[dict], probes: list[dict], arms: tuple,
        cache: dict | None, replay: bool, verbose: bool) -> tuple[list, dict]:
    by_id = {b["bucket_id"]: b for b in buckets}
    new_cache = {"meta": {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "probes": len(probes), "buckets": len(buckets),
                          "instructions": LAYA_INSTRUCTIONS},
                 "probes": {}}
    records = []
    t0 = time.time()
    for i, p in enumerate(probes):
        b = by_id[p["bucket_id"]]
        members = b["members"]
        mids = [m["member_id"] for m in members]
        cached = (cache or {}).get("probes", {}).get(p["probe_id"], {})
        store: dict = {}
        out: dict = {}

        if "random" in arms:
            rnd = _random.Random(f"{RANDOM_SEED}:{p['probe_id']}")
            out["random"] = {"pick": rnd.choice(mids), "margin": None}
        if "fixed" in arms:
            out["fixed"] = {"pick": mids[0], "margin": None}
        if "lexical" in arms:
            s = arm_lexical(p["problem"], members)
            pick, margin = _rank(s)
            out["lexical"] = {"pick": pick, "margin": margin, "scores": s}
        for name, txt in (("embedding", None),
                          ("embedding_cond", condition_text)):
            if name not in arms:
                continue
            s = cached.get(name)
            if s is None:
                if replay:
                    raise RuntimeError(f"no cached {name} for {p['probe_id']}")
                s = arm_embedding(p["problem"], members, txt)
            store[name] = s
            pick, margin = _rank(s)
            out[name] = {"pick": pick, "margin": margin, "scores": s}
        for name, fn in (("rerank", arm_rerank),
                         ("rerank_batched", arm_rerank_batched)):
            if name not in arms:
                continue
            s = cached.get(name)
            if s is None:
                if replay:
                    raise RuntimeError(f"no cached {name} for {p['probe_id']}")
                s = fn(p["problem"], members)
            store[name] = s
            pick, margin = _rank(s)
            out[name] = {"pick": pick, "margin": margin, "scores": s}
        for name, opt in (("laya", None), ("laya_cond", condition_text)):
            if name not in arms:
                continue
            key = name + "_orders"
            orders = cached.get(key)
            if orders is None and replay:
                raise RuntimeError(f"no cached {name} for {p['probe_id']}")
            v, sink = arm_laya(p["problem"], members, orders, opt)
            store[key] = sink
            if v is None:
                out[name] = {"pick": None, "margin": None, "error": "no answer"}
            else:
                out[name] = {"pick": v["choice"], "margin": v["margin"],
                             "scores": v["probabilities"],
                             "orders_used": v["orders_used"]}
        new_cache["probes"][p["probe_id"]] = store
        records.append({**p, "arms": out, "n_members": len(members)})
        if verbose and not replay and (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(probes)} probes "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return records, new_cache


# ================================================================ TABLES =====

def _pct(x) -> str:
    return "  n/a" if x != x else f"{100 * x:5.1f}%"


def _hr(title: str) -> None:
    print()
    print(title)
    print("-" * max(60, len(title)))


def report(buckets: list[dict], probes: list[dict], records: list[dict],
           arms: tuple) -> None:
    contrastive = lambda r: r["bucket_type"] == "contrastive"   # noqa: E731
    topical = lambda r: r["bucket_type"] == "topical"           # noqa: E731

    _hr("DATASET")
    for t in ("contrastive", "topical"):
        bs = [b for b in buckets if b["bucket_type"] == t]
        sizes = [len(b["members"]) for b in bs]
        ps = [p for p in probes if p["bucket_type"] == t]
        print(f"  {t:12s} {len(bs):2d} buckets  {sum(sizes):3d} members  "
              f"{len(ps):3d} probes  size {min(sizes)}-{max(sizes)} "
              f"(mean {sum(sizes) / len(sizes):.1f})")
    chance = sum(1 / r["n_members"] for r in records) / len(records)
    print(f"  {'TOTAL':12s} {len(buckets):2d} buckets  "
          f"{sum(len(b['members']) for b in buckets):3d} members  "
          f"{len(probes):3d} probes")
    print(f"  chance (1/bucket size, averaged over probes): {_pct(chance)}")

    _hr("TABLE 1 -- ACCURACY AT FULL COVERAGE (argmax, nobody abstains)")
    print(f"  {'arm':14s} {'overall':>9s} {'contrastive':>13s} {'topical':>9s}"
          f"   {'contr - top':>12s}")
    acc = {}
    for a in arms:
        o = evaluate(records, a)
        c = evaluate(records, a, contrastive)
        t = evaluate(records, a, topical)
        acc[a] = (o, c, t)
        print(f"  {a:14s} {_pct(o['acc']):>9s} {_pct(c['acc']):>13s} "
              f"{_pct(t['acc']):>9s}   {100 * (c['acc'] - t['acc']):+11.1f}pp")
    print(f"  {'chance':14s} {_pct(chance):>9s}")

    _hr("TABLE 2 -- PRECISION AT MATCHED COVERAGE (all arms commit on the same "
        "fraction)")
    for label, subset in (("ALL PROBES", None), ("CONTRASTIVE", contrastive),
                          ("TOPICAL", topical)):
        n = len([r for r in records if subset is None or subset(r)])
        print(f"\n  {label} (n={n})")
        print("  " + " " * 14 + "".join(f"{int(c * 100):>7d}%"
                                        for c in COVERAGE_GRID))
        for a in arms:
            curve = coverage_curve(records, a, subset)
            flat = all(r["arms"][a].get("margin") is None
                       for r in records
                       if (subset is None or subset(r)) and a in r["arms"])
            cells = "".join(f"{100 * pr:7.1f}" for _, _, pr in curve)
            print(f"  {a:14s}{cells}" + ("   (no ranking signal)" if flat else ""))
        print("  " + " " * 14 + "".join(
            f"{int(round(c * n)):>7d}" for c in COVERAGE_GRID) + "   <- committed")

    _hr("TABLE 3 -- EACH ARM AT ITS OWN NATIVE GATE")
    sel = selectors_mod()
    gates = {"laya": sel.LAYA_MARGIN_FLOOR,
             "laya_cond": sel.LAYA_MARGIN_FLOOR,
             "embedding": 0.05, "embedding_cond": 0.05,
             "rerank": 0.05, "lexical": 0.05}
    print("  laya gate is selectors.LAYA_MARGIN_FLOOR; the retrieval gates are "
          "an\n  arbitrary small margin and are shown only so the shape is "
          "visible.\n  Table 2 is the comparison that means something.")
    print(f"\n  {'arm':14s} {'gate':>6s} {'fired':>7s} {'coverage':>9s} "
          f"{'precision':>10s} {'cov-acc':>9s}")
    for a in arms:
        if a not in gates:
            continue
        g = gate_report(records, a, gates[a])
        print(f"  {a:14s} {gates[a]:6.2f} {g['fired']:4d}/{g['n']:<3d} "
              f"{_pct(g['coverage']):>9s} {_pct(g['precision']):>10s} "
              f"{_pct(g['cov_acc']):>9s}")
    if "laya" in arms:
        g = gate_report(records, "laya", sel.TREE_MARGIN_GATE)
        print(f"  {'laya':14s} {sel.TREE_MARGIN_GATE:6.2f} "
              f"{g['fired']:4d}/{g['n']:<3d} {_pct(g['coverage']):>9s} "
              f"{_pct(g['precision']):>10s} {_pct(g['cov_acc']):>9s}"
              "   (selectors.TREE_MARGIN_GATE)")

    _hr("TABLE 4 -- PER-BUCKET ACCURACY")
    heads = [a for a in arms
             if a in ("lexical", "embedding", "embedding_cond", "rerank",
                      "laya", "laya_cond")]
    print(f"  {'bucket':28s} {'type':12s} {'k':>2s} " +
          " ".join(f"{h:>14s}" for h in heads))
    for b in buckets:
        sub = lambda r, bid=b["bucket_id"]: r["bucket_id"] == bid   # noqa: E731
        cells = []
        for h in heads:
            e = evaluate(records, h, sub)
            cells.append(f"{e['correct']}/{e['n']}".rjust(14))
        print(f"  {b['bucket_id']:28s} {b['bucket_type']:12s} "
              f"{len(b['members']):2d} " + " ".join(cells))

    _hr("TABLE 5 -- PAIRED McNEMAR (exact, two-sided)")
    print("  n is small by construction. The DISCORDANT count is the number "
          "that\n  decides whether anything could have been detected at all.")
    pairs = [("laya", "embedding"), ("laya", "rerank"), ("laya", "lexical"),
             ("laya_cond", "embedding"), ("laya_cond", "laya"),
             ("embedding_cond", "embedding"), ("embedding", "rerank")]
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        for label, subset in (("all", None), ("contrastive", contrastive),
                              ("topical", topical)):
            m = mcnemar(records, a, b, subset)
            if m["n"] == 0:
                continue
            print(f"  {a:>9s} vs {b:<10s} [{label:11s}] "
                  f"n={m['n']:3d}  {a}_only={m[a + '_only']:2d}  "
                  f"{b}_only={m[b + '_only']:2d}  "
                  f"discordant={m['discordant']:2d}  p={m['p']:.3f}"
                  + ("  <- cannot detect anything"
                     if m["discordant"] < 6 else ""))
        print()

    _hr("TABLE 6 -- WHERE LAYA IS ALONE IN BEING RIGHT, AND ALONE IN BEING "
        "WRONG")
    if "laya" in arms and "embedding" in arms and "rerank" in arms:
        won = [r for r in records
               if r["arms"]["laya"]["pick"] == r["correct"]
               and r["arms"]["embedding"]["pick"] != r["correct"]
               and r["arms"]["rerank"]["pick"] != r["correct"]]
        lost = [r for r in records
                if r["arms"]["laya"]["pick"] != r["correct"]
                and r["arms"]["embedding"]["pick"] == r["correct"]
                and r["arms"]["rerank"]["pick"] == r["correct"]]
        print(f"  laya alone right: {len(won)}   laya alone wrong: {len(lost)}")
        for tag, rows in (("ONLY LAYA RIGHT", won), ("ONLY LAYA WRONG", lost)):
            print(f"\n  {tag}")
            for r in rows[:14]:
                print(f"    [{r['bucket_type'][:5]}] {r['probe_id']:44s} "
                      f"laya={r['arms']['laya']['pick']}"
                      f"(m={r['arms']['laya']['margin']})")
            if len(rows) > 14:
                print(f"    ... and {len(rows) - 14} more")

    _hr("TABLE 7 -- DEGENERACY: does the answer move with the STATE?")
    print("  docs/LAYA.md's cross-cutting lesson: a margin gate is only "
          "meaningful\n  after the question has been shown not to be "
          "degenerate. MODAL SHARE is the\n  fraction of a bucket's probes "
          "that got the SAME answer. The floor is 1/k --\n  a bucket whose "
          "every probe returns one member is a constant, and its\n  margin is "
          "measuring the option wordings, not the problem.")
    heads2 = [a for a in arms if a in ("lexical", "embedding",
                                       "embedding_cond", "rerank",
                                       "laya", "laya_cond")]
    print(f"\n  {'bucket':28s} {'1/k':>5s} " +
          " ".join(f"{h:>14s}" for h in heads2))
    tot = {h: [0, 0] for h in heads2}
    for b in buckets:
        rows = [r for r in records if r["bucket_id"] == b["bucket_id"]]
        cells = []
        for h in heads2:
            picks = [r["arms"][h]["pick"] for r in rows if h in r["arms"]]
            if not picks:
                cells.append(" " * 9 + "    -")
                continue
            modal = max(set(picks), key=picks.count)
            share = picks.count(modal) / len(picks)
            tot[h][0] += picks.count(modal)
            tot[h][1] += len(picks)
            cells.append(f"{share:14.2f}")
        print(f"  {b['bucket_id']:28s} {1 / len(b['members']):5.2f} "
              + " ".join(cells))
    print(f"  {'ALL (probe-weighted)':28s} {1 / 4.7:5.2f} "
          + " ".join(f"{tot[h][0] / max(1, tot[h][1]):14.2f}"
                      for h in heads2))

    _hr("THE STATED PREDICTION")
    print('  "Laya beats both on CONTRASTIVE buckets (alternatives separated by')
    print('   one condition), and ties or loses on TOPICALLY DISTINCT ones."')
    if all(a in arms for a in ("laya", "embedding", "rerank")):
        lc, le, lr = (acc["laya"][1]["acc"], acc["embedding"][1]["acc"],
                      acc["rerank"][1]["acc"])
        tc, te, tr = (acc["laya"][2]["acc"], acc["embedding"][2]["acc"],
                      acc["rerank"][2]["acc"])
        beats_c = lc > le and lc > lr
        print(f"\n  contrastive: laya {_pct(lc)}  embedding {_pct(le)}  "
              f"rerank {_pct(lr)}   -> laya "
              f"{'BEATS both' if beats_c else 'does NOT beat both'}")
        print(f"  topical    : laya {_pct(tc)}  embedding {_pct(te)}  "
              f"rerank {_pct(tr)}   -> laya "
              f"{'ahead' if tc > max(te, tr) else 'behind or tied'}")
        mc_e = mcnemar(records, "laya", "embedding", contrastive)
        mc_r = mcnemar(records, "laya", "rerank", contrastive)
        print(f"\n  paired, contrastive: vs embedding discordant="
              f"{mc_e['discordant']} p={mc_e['p']:.3f}; vs rerank discordant="
              f"{mc_r['discordant']} p={mc_r['p']:.3f}")
        print(f"\n  VERDICT: {'CONFIRMED' if beats_c else 'REFUTED'} on the "
              "first half of the prediction.")


# ================================================================== MAIN =====

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--replay", action="store_true",
                    help="recompute every table from bench/data, no GPU")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    for x in arms:
        if x not in ARMS:
            print(f"unknown arm {x!r}; known: {ARMS}")
            return 2

    buckets = load_jsonl(BUCKETS)
    probes = load_jsonl(PROBES)

    problems = check_buckets(buckets, probes)
    drift = verify_no_drift(buckets)
    if problems:
        print("BUCKET INVARIANTS FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    if drift:
        print("CORPUS DRIFT -- a member text is no longer a verbatim recipe:")
        for d in drift:
            print("  -", d)
        return 1
    print(f"buckets {len(buckets)}, probes {len(probes)}, arms {list(arms)}; "
          f"invariants OK, no corpus drift")

    cache = None
    if os.path.exists(a.runs):
        with open(a.runs, encoding="utf-8") as fh:
            cache = json.load(fh)
    if a.replay and cache is None:
        print(f"--replay needs {a.runs}; run once live first")
        return 1

    records, new_cache = run(buckets, probes, arms, cache, a.replay,
                             not a.quiet)

    if not a.replay:
        os.makedirs(os.path.dirname(a.runs), exist_ok=True)
        merged = cache or {"probes": {}}
        merged["meta"] = new_cache["meta"]
        for pid, store in new_cache["probes"].items():
            merged.setdefault("probes", {}).setdefault(pid, {}).update(store)
        with open(a.runs, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=1, sort_keys=True)
        print(f"wrote {a.runs}")

    report(buckets, probes, records, arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
