#!/usr/bin/env python
"""Recipe selectors, interchangeable behind one interface, with both bounds.

WHY THIS FILE EXISTS

`bench/recipe_oracle.py` measures the CEILING: best arm per problem, which no
selector can beat. That number is only interpretable next to a FLOOR, and the
interesting quantity is where a real mechanism lands between them. Several
mechanisms are plausible -- cosine, a cross-encoder, a Laya decision tree --
and nobody knows which wins. So none of them is built into the harness. They
all implement `select(problem_text, recipes, k)` and are swapped by name, on
identical inputs, in the same run.

The two bounds are selectors too, deliberately:

    select_random   the floor. If a mechanism does not beat a coin, selection
                    does not matter and the corpus project dies here.
    select_fixed    the second floor, and the harder one. "Always inject the
                    single best hint" needs no selector at all. A mechanism
                    that does not beat it is paying for nothing.
    select_oracle   the ceiling. A CHEAT: it reads the answer from a
                    precomputed map. NOT DEPLOYABLE, marked as such in the
                    registry, and it raises rather than degrading when the map
                    is absent -- a ceiling that silently becomes a guess is
                    worse than no ceiling (PROTOCOL rule 2).

WHY LAYA APPEARS TWICE, AND WHY THE TREE IS THE HEADLINE

`select_laya` ranks the whole recipe pool as one closed-set choice. It is
included because it is the obvious thing to try and the comparison needs it,
but it is the regime Laya is weakest in: the options are different passages,
and the measured fact is that varying both state and passage collapses the
real-vs-nonsense gap from 0.497 to ~0.

`select_tree` is the architecture that avoids that. System 2 offline, System 1
at runtime: the big model decomposes the problem space ONCE into a small tree
of simple closed questions; at runtime Laya answers one such question per node,
with the problem fixed in `state` and the answers in `criteria`. Every node is
a fixed-state, varying-option choice -- the one configuration Laya is measured
to be good at -- and no node ever compares scores across passages, so the
failure that kills the ranking approach cannot occur.

The tree's risk is different and is made visible in code rather than
discovered later: error COMPOUNDS with depth (`tree_depth_risk`), and a node
whose branches lead to the same recipe is pure added error
(`node_discrimination`).

MEASURED CONSTRAINTS ON EVERY LAYA CALL HERE
  - candidates go in `criteria` ONLY. Putting them in `state` too produced
    option "a" eight times at margins up to 0.919 -- a positional artefact.
  - not permutation invariant, so every choice is averaged over option
    orderings, as `mcp/fanout.py:_choice_averaged` does.
  - the tail of a long `state` is silently dropped, so the problem goes FIRST
    and nothing is appended after it.
  - never threshold a score across passages; only margins WITHIN one call are
    used, and only against a gate.

DRY RUN

    python selectors.py --dry-run

validates every mechanism against a mock embedder and a mock Laya endpoint
that reproduces the positional bias. No GPU, no network, no model.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)
for _p in (BENCH, os.path.join(ROOT, "mcp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")

# Laya drops the tail of a long state. 2000 chars matches what
# fanout._choice_averaged sends; a problem statement longer than this is
# truncated at the END, never at the front, because the front is the part that
# is read.
LAYA_STATE_CHARS = 2000

# Two orderings is what fanout measured as enough to make a margin honest, and
# for a 2-option node forward+reversed is a COMPLETE de-bias: each option
# occupies each position exactly once. More options need rotations to get the
# same property, which _orders() supplies when asked for more permutations.
LAYA_PERMUTATIONS = 2

# Below this, the options did not separate once position bias is removed.
# fanout.break_tie uses 0.15 as a "do not report a pick you did not make"
# floor; the tree walk uses a stricter 0.3 because its errors COMPOUND --
# measured, reordering-sensitive cases averaged down to margins of 0.012-0.184
# while a genuinely decided case stayed at 0.800, so 0.3 sits in the empty band
# between those two populations.
#
# PROVENANCE OF 0.3: UNVERIFIABLE. Nothing in this repo reproduces it.
#
# The sentence above quotes a spread (0.012-0.184 against 0.800) with no n, no
# script, and no data file. There is no run that regenerates those numbers and
# no record of how many cases went into either population, so the gate cannot
# be re-derived, re-checked, or moved on evidence. Treat it as a working
# default that someone chose, not as a measurement. The value is left alone
# deliberately -- changing it would be a behaviour change made on no better
# information than the one that set it.
#
# IT IS ALSO ARITY-DEPENDENT, AND THAT IS NOT OPTIONAL BOOKKEEPING.
# A margin gate is a distance between probabilities, so what it demands
# depends on how many options share the mass:
#
#   2 options  uniform is 0.500/0.500 -> a 0.3 margin means 0.65 vs 0.35
#   3 options  uniform is 0.333       -> the SAME 0.3 means ~0.52 vs 0.22
#
# i.e. the identical constant asks a 3-way node for far more separation, at
# identical underlying confidence, than it asks a binary one. Whatever 0.3 was
# fitted on, it was fitted on BINARY nodes: three of the four nodes in
# EXAMPLE_TREE below are binary.
#
# This has already caused one wrong verdict. The gate was lifted from here,
# applied unchanged to a three-way routing question, and the resulting
# abstentions were written up as "Laya cannot route". The shape of the question
# was the fault, not the model. Full account, with the corrected framing
# (binary cascade, `continue` as a first-class option): docs/FINDINGS.md
# section 16a. Do not import this number across an arity boundary again; a
# 3-way node needs its own gate, fitted on 3-way data.
LAYA_MARGIN_FLOOR = 0.15
TREE_MARGIN_GATE = 0.3

# A cross-encoder reads query and document together. Overflowing its context
# makes it score everything identically -- silently replacing a ranking with
# noise -- so the problem statement is truncated for reranking only.
RERANK_QUERY_CHARS = 2000
RERANK_DOC_CHARS = 1200

# PLACEHOLDER, not a result. The deployable floor is "always inject the best
# single arm", and which arm that is comes from the per-recipe table in
# recipe_oracle.report(). Use best_fixed_arm() once a run exists; until then
# this is an arbitrary choice and any comparison against it says so.
FIXED_DEFAULT_ID = "complexity"

RANDOM_SEED = 20260921        # same seed recipe_oracle samples problems with

NO_HINT = "none"              # leaf meaning "inject nothing"; not a recipe id


class SelectorError(RuntimeError):
    pass


class OracleUnavailable(SelectorError):
    """The ceiling cannot be computed. Deliberately fatal -- see PROTOCOL 2."""


class LayaContractError(SelectorError):
    """A Laya payload violated a measured constraint. Fatal by design."""


# --------------------------------------------------------------- backends ---
# Indirection, not reimplementation. The real calls are code_search.embed,
# code_search.rerank and code_search._post_json; these hooks exist so the dry
# run can substitute mocks without a GPU or a network, and for no other reason.

_EMBED_HOOK = None
_RERANK_HOOK = None
_POST_HOOK = None


def _embed(texts: list[str], is_query: bool):
    if _EMBED_HOOK is not None:
        return _EMBED_HOOK(texts, is_query)
    import code_search as cs
    return cs.embed(texts, is_query=is_query)


def _rerank(query: str, docs: list[str], top_n: int):
    if _RERANK_HOOK is not None:
        return _RERANK_HOOK(query, docs, top_n)
    import code_search as cs
    return cs.rerank(query, docs, top_n)


def _post_json(url: str, payload: dict, timeout: int = 30):
    if _POST_HOOK is not None:
        return _POST_HOOK(url, payload, timeout)
    import code_search as cs
    return cs._post_json(url, payload, timeout=timeout)


@contextlib.contextmanager
def mock_backends(embed=None, rerank=None, post=None):
    """Swap the three backends for the duration of a block. Dry run only."""
    global _EMBED_HOOK, _RERANK_HOOK, _POST_HOOK
    prev = (_EMBED_HOOK, _RERANK_HOOK, _POST_HOOK)
    _EMBED_HOOK, _RERANK_HOOK, _POST_HOOK = embed, rerank, post
    try:
        yield
    finally:
        _EMBED_HOOK, _RERANK_HOOK, _POST_HOOK = prev


# ----------------------------------------------------------------- result ---

@dataclass
class Selection:
    """What a selector returns.

    It iterates as the list of chosen ids, so `list(sel)` is the documented
    `select(...) -> list[id]` interface, while the fields carry what the
    analysis needs later: a confidence or margin where the mechanism HAS one
    (random and fixed do not, and inventing one for them would be a lie), the
    declared fallback if this is not the mechanism's own answer, and the
    per-node trace for the tree walk, where compounding error cannot be
    diagnosed without it.
    """
    ids: list[str]
    mechanism: str
    confidence: float | None = None      # absolute, comparable across problems
    margin: float | None = None          # within-call separation only
    scores: dict = field(default_factory=dict)
    fallback: str | None = None          # set => NOT this mechanism's answer
    abstained: bool = False
    trace: list = field(default_factory=list)
    notes: str = ""

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i]

    def as_dict(self) -> dict:
        return {"ids": list(self.ids), "mechanism": self.mechanism,
                "confidence": self.confidence, "margin": self.margin,
                "scores": self.scores, "fallback": self.fallback,
                "abstained": self.abstained, "trace": self.trace,
                "notes": self.notes}


# ------------------------------------------------------------------ pool ----

def default_pool(include_placebo: bool = False) -> list[dict]:
    """The recipe pool, reused from bench/recipe_oracle.py.

    `placebo` is EXCLUDED by default. It is the control arm -- same shape and
    length, no performance content -- and a selector able to pick it would be
    scored partly on how often it chooses the control, which confounds the
    mechanism comparison with the placebo comparison.
    """
    import recipe_oracle
    return [r for r in recipe_oracle.RECIPES
            if include_placebo or r["id"] != "placebo"]


def _ids(recipes: list[dict]) -> list[str]:
    return [r["id"] for r in recipes]


def _texts(recipes: list[dict]) -> list[str]:
    return [r["text"] for r in recipes]


def _pool(recipes):
    return default_pool() if recipes is None else list(recipes)


def _clamp_k(k: int, n: int) -> int:
    return max(1, min(int(k), n))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


# Used by the MOCKS only. A hashed bag of words with no idf is dominated by
# function words -- "the" and "at" alone made the mock rank a loop-hoisting
# hint above a data-structure hint for a problem about popping from a deque.
# A real embedder is not fooled by that, so the mock must not be either, or the
# assertions would be testing the mock's vocabulary rather than the selectors.
_STOPWORDS = set(
    "a an and are as at be but by can do does done each every for from had has "
    "have he i if in into is it its more most must no not of on one or other "
    "out over per she so some than that the their then they this to up we will "
    "with you your".split())


def _content_words(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOPWORDS}


# ------------------------------------------------------------- fallbacks ----
# PROTOCOL rule 2: a fallback that hides an outage is a bug. Every degraded
# return here names the mechanism it fell back to AND the reason, so a run can
# count them and fail if the rate says the backend was down rather than flaky.

def _fallback(problem: str, recipes: list[dict], k: int,
              mechanism: str, to: str, reason: str) -> Selection:
    sel = select_fixed(problem, recipes, k)
    sel.mechanism = mechanism
    sel.fallback = f"{to}: {reason}"
    sel.notes = ("DEGRADED -- this is not the mechanism's own answer. Count "
                 "these; a high rate is an outage, not resilience.")
    return sel


# ================================================================ FLOORS ====

def select_random(problem_text: str, recipes: list[dict] | None = None,
                  k: int = 1, seed: int = RANDOM_SEED) -> Selection:
    """The floor. Any mechanism that does not beat this is not a mechanism.

    Seeded on (seed, problem) rather than on a shared stream, so a rerun of the
    experiment draws the same arms for the same problems even if the problem
    ORDER changes -- an unreproducible floor cannot be compared against.

    No confidence: a random pick has none, and attaching a number to it would
    make abstention analysis meaningless.
    """
    pool = _pool(recipes)
    k = _clamp_k(k, len(pool))
    rnd = random.Random(f"{seed}:{problem_text}")
    return Selection(ids=rnd.sample(_ids(pool), k), mechanism="random")


def select_fixed(problem_text: str, recipes: list[dict] | None = None,
                 k: int = 1, recipe_id: str = FIXED_DEFAULT_ID) -> Selection:
    """The harder floor: always inject the same hint, ignoring the problem.

    This is the condition that asks whether SELECTION matters at all, as
    opposed to injecting one good hint unconditionally. For k>1 the remaining
    slots are filled in pool order, which is deterministic and equally
    problem-blind.
    """
    pool = _pool(recipes)
    ids = _ids(pool)
    k = _clamp_k(k, len(ids))
    if recipe_id not in ids:
        recipe_id = ids[0]
    chosen = [recipe_id] + [i for i in ids if i != recipe_id]
    return Selection(ids=chosen[:k], mechanism="fixed",
                     notes=f"fixed arm {recipe_id!r}")


def best_fixed_arm(results_path: str) -> tuple[str, int, int] | None:
    """(arm, passes, problems) for the best single arm in a recipe_oracle run.

    Convenience for CONFIGURING the fixed floor honestly instead of guessing.
    It is not a claim: no significance test, ties broken by pool order. Returns
    None when the run does not exist yet, which is the normal state before the
    oracle has been measured.
    """
    if not os.path.exists(results_path):
        return None
    passes: dict[str, int] = {}
    problems: dict[str, set] = {}
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            arm = r.get("arm")
            if not arm or arm in ("none", "placebo") or r.get("error"):
                continue
            problems.setdefault(arm, set()).add(r.get("question_id"))
            passes[arm] = passes.get(arm, 0) + (1 if r.get("passed") else 0)
    if not passes:
        return None
    arm = max(passes, key=lambda a: passes[a])
    return arm, passes[arm], len(problems[arm])


# =========================================================== RETRIEVAL ======

def select_embedding(problem_text: str, recipes: list[dict] | None = None,
                     k: int = 1) -> Selection:
    """Cosine between the problem and each recipe, via code_search.embed.

    Qwen3-Embedding is ASYMMETRIC: the problem is embedded as a query (the
    instruction prefix is applied inside embed()), the recipes as documents.
    Embedding a bare query puts it in a different region of the space and
    retrieval goes near-random -- the measured failure was a real question
    matching an unrelated model at 0.002.

    embed() returns unit-norm rows, so a dot product IS cosine. Cosine is the
    one score here that is comparable across problems, so it is reported as
    `confidence`; `margin` is the within-problem top-1 minus top-2 separation.
    """
    pool = _pool(recipes)
    k = _clamp_k(k, len(pool))
    try:
        qv = _embed([problem_text], True)[0]
        dv = _embed(_texts(pool), False)
    except Exception as e:                                       # noqa: BLE001
        return _fallback(problem_text, pool, k, "embedding", "fixed",
                         f"{type(e).__name__}: {e}")
    sims = {r["id"]: sum(float(a) * float(b) for a, b in zip(qv, v))
            for r, v in zip(pool, dv)}
    ranked = sorted(sims.items(), key=lambda x: -x[1])
    margin = round(ranked[0][1] - ranked[1][1], 4) if len(ranked) > 1 else None
    return Selection(ids=[i for i, _ in ranked[:k]], mechanism="embedding",
                     confidence=round(ranked[0][1], 4), margin=margin,
                     scores={i: round(s, 4) for i, s in ranked})


def select_rerank(problem_text: str, recipes: list[dict] | None = None,
                  k: int = 1) -> Selection:
    """Cross-encoder rerank of the whole pool against the problem.

    Only ORDER is meaningful from a reranker. The absolute scale is not: this
    setup returns correctly-ordered scores around 1e-13, and an earlier guard
    that rejected "too small" scores was discarding good rankings. So no
    confidence is reported, and the margin is flagged ordinal.

    The degenerate case that IS worth catching is a reranker that cannot
    separate the documents at all -- every score identical -- which is what
    context overflow looks like.
    """
    pool = _pool(recipes)
    k = _clamp_k(k, len(pool))
    docs = [f"{r['id']}: {r['text'][:RERANK_DOC_CHARS]}" for r in pool]
    try:
        ranked = _rerank(problem_text[:RERANK_QUERY_CHARS], docs, len(docs))
    except Exception as e:                                       # noqa: BLE001
        ranked = None
        reason = f"{type(e).__name__}: {e}"
    if ranked:
        scores = [s for _, s in ranked]
        if len(set(scores)) == 1:
            ranked, reason = None, ("all scores identical -- the cross-encoder "
                                    "did not separate the documents")
    elif ranked is not None:
        ranked, reason = None, "reranker returned no rows"
    if ranked is None:
        # Declared chain: rerank -> embedding -> fixed. Embedding first because
        # it is a real ranking over the same pool, not a constant.
        sel = select_embedding(problem_text, pool, k)
        sel.mechanism = "rerank"
        sel.fallback = f"embedding: {reason}"
        sel.notes = "DEGRADED -- cross-encoder unusable, embedding order used."
        return sel
    ids = [pool[i]["id"] for i, _ in ranked]
    gap = (round(float(ranked[0][1]) - float(ranked[1][1]), 12)
           if len(ranked) > 1 else None)
    return Selection(ids=ids[:k], mechanism="rerank", confidence=None,
                     margin=gap,
                     scores={pool[i]["id"]: float(s) for i, s in ranked},
                     notes="margin is ORDINAL only; rerank scale is meaningless")


# ================================================================= LAYA =====

def _orders(keys: list[str], n_permutations: int) -> list[tuple[str, ...]]:
    """Option orderings to average a choice over.

    Forward and reversed first: for two options that is a complete de-bias,
    each option sitting in each position exactly once. Beyond two options,
    rotations are added, which is what spreads a first-position boost evenly
    instead of merely symmetrising the ends.
    """
    base = list(keys)
    out = [tuple(base), tuple(reversed(base))]
    i = 1
    while len(out) < max(2, n_permutations) and i < len(base):
        rot = tuple(base[i:] + base[:i])
        if rot not in out:
            out.append(rot)
        i += 1
    return out[:max(2, n_permutations)]


def _check_laya_contract(state: str, criteria: dict) -> None:
    """Fail loudly if a payload repeats candidates inside the state.

    This is the exact configuration that produced option "a" eight times at
    margins up to 0.919. The builders below cannot violate it -- the state is
    the problem and nothing is appended -- so this guard exists to catch a
    future edit that appends "the options are ..." to the prompt, which is a
    natural thing to write and silently poisons every number downstream.
    """
    if not criteria:
        raise LayaContractError("choice with no criteria")
    for key, text in criteria.items():
        probe = (text or "")[:60].strip()
        if probe and probe in state:
            raise LayaContractError(
                f"candidate {key!r} text appears in `state`. Candidates go in "
                f"`criteria` ONLY -- see the eight-times-'a' artefact.")


def _laya_choice(state: str, instructions: str, criteria: dict,
                 n_permutations: int = LAYA_PERMUTATIONS,
                 timeout: int = 30) -> dict | None:
    """One closed-set choice, permutation-averaged. None on failure.

    Same shape as fanout._choice_averaged, with two additions: the contract
    guard above, and a per-order mass check. An order whose probabilities sum
    to ~0 is a degenerate answer, and averaging it in would quietly halve every
    margin -- the degenerate case must not look like the healthy one.
    """
    keys = list(criteria)
    if len(keys) < 2:
        return None
    state = state[:LAYA_STATE_CHARS]
    _check_laya_contract(state, criteria)
    orders = _orders(keys, n_permutations)
    acc = {k: 0.0 for k in keys}
    used = 0
    for order in orders:
        payload = {"state": state,
                   "questions": {"pick": {
                       "type": "choice",
                       "instructions": instructions,
                       "criteria": {k: criteria[k] for k in order}}}}
        try:
            d = _post_json(LAYA_URL + "/decide", payload, timeout)
            probs = (d["answers"]["pick"].get("probabilities") or {})
        except Exception:                                        # noqa: BLE001
            continue
        if sum(float(v) for v in probs.values()) < 1e-6:
            continue
        for key in keys:
            acc[key] += float(probs.get(key, 0.0))
        used += 1
    if used == 0:
        return None
    acc = {k: v / used for k, v in acc.items()}
    ranked = sorted(acc.items(), key=lambda x: -x[1])
    return {"choice": ranked[0][0],
            "margin": round(ranked[0][1] - ranked[1][1], 3),
            "confidence": round(ranked[0][1], 3),
            "probabilities": {k: round(v, 3) for k, v in acc.items()},
            "orders": len(orders), "orders_used": used}


LAYA_RANK_INSTRUCTIONS = (
    "Pick the performance hint most likely to make the difference for the "
    "program this problem requires.")


def select_laya(problem_text: str, recipes: list[dict] | None = None,
                k: int = 1, n_permutations: int = LAYA_PERMUTATIONS
                ) -> Selection:
    """The whole pool as ONE closed-set choice. Problem in `state`, recipes in
    `criteria` only.

    Honest warning, from the measurements this file was built against: this is
    Laya's WEAK regime. The options here are different passages, and its scores
    are not comparable across passages -- fixing the passage and varying the
    query separates real from nonsense by 0.497, varying both collapses the gap
    to ~0. Nothing here scores a recipe independently and thresholds across
    them, which would be the fatal version; a single choice at least keeps the
    comparison inside one forward pass. It is in the comparison because the
    question "is the tree worth its complexity" needs this arm to be answered.

    A low margin does NOT trigger a fallback: the pick is returned with its
    margin so abstention policy can be studied after the fact rather than baked
    in now. A FAILURE -- network, degenerate response -- does fall back.
    """
    pool = _pool(recipes)
    k = _clamp_k(k, len(pool))
    criteria = {r["id"]: r["text"] for r in pool}
    try:
        v = _laya_choice(problem_text, LAYA_RANK_INSTRUCTIONS, criteria,
                         n_permutations)
    except LayaContractError:
        raise
    except Exception as e:                                       # noqa: BLE001
        return _fallback(problem_text, pool, k, "laya", "fixed",
                         f"{type(e).__name__}: {e}")
    if v is None:
        return _fallback(problem_text, pool, k, "laya", "fixed",
                         "no usable response from laya")
    ranked = sorted(v["probabilities"].items(), key=lambda x: -x[1])
    return Selection(
        ids=[i for i, _ in ranked[:k]], mechanism="laya",
        confidence=v["confidence"], margin=v["margin"],
        scores=v["probabilities"],
        notes=(f"averaged over {v['orders_used']}/{v['orders']} orderings; "
               f"margin below {LAYA_MARGIN_FLOOR} means the options did not "
               f"separate once position bias was removed"))


# ======================================================== DECISION TREE =====
#
# System 2 offline, System 1 at runtime.
#
# THE TREE BELOW IS A PLACEHOLDER. It is hand-written so the walker is runnable
# and testable today. It is NOT a validated artefact: a real tree is GENERATED
# by the big model from the problem distribution and then VALIDATED against an
# oracle run -- every node checked with node_discrimination() for whether its
# branches actually lead to different best arms, and the whole tree checked
# with tree_depth_risk() for whether its depth is affordable at the per-node
# accuracy Laya actually achieves. Until that is done, no number produced with
# this tree describes anything but the walker.
#
# Node contract, and it is the whole point: each question is SELF-CONTAINED,
# answerable by looking at the problem alone, with a small set of direct
# answers. No node asks "on balance, which recipe is best" -- that is the
# System 2 question Laya scores 50-57% on. Compare laya_router.py, where the
# same move (binary local questions plus a rule in code) took routing from 43%
# to 86%, and the gain was entirely in the rule.

EXAMPLE_TREE = {
    "root": "scale",
    "nodes": {
        "scale": {
            "question": "What do the problem's stated input limits imply about "
                        "the size of the input that must be read?",
            "options": {
                "huge": "The constraints put the number of input values, "
                        "queries or lines at 100000 or more.",
                "small": "The constraints keep every quantity small, at most a "
                         "few thousand values in total.",
            },
            "next": {"huge": {"goto": "huge_shape"},
                     "small": {"goto": "small_shape"}},
        },
        "huge_shape": {
            "question": "With a large input, what dominates the work the "
                        "program must do?",
            "options": {
                "reading": "The program mostly reads and echoes or aggregates "
                           "the values, one line or token at a time.",
                "searching": "The program must search, count pairs, or match "
                             "elements against each other.",
            },
            "next": {"reading": {"leaf": "io"},
                     "searching": {"leaf": "complexity"}},
        },
        "small_shape": {
            "question": "What kind of manipulation does the problem ask for?",
            "options": {
                "sequence": "Elements are repeatedly added to or removed from "
                            "the front or back of a sequence, or membership is "
                            "tested many times.",
                "text": "The answer is built up as text, characters or "
                        "substrings joined together.",
                "arithmetic": "The answer is computed by arithmetic or simple "
                              "iteration over a small fixed range.",
            },
            "next": {"sequence": {"leaf": "structures"},
                     "text": {"leaf": "strings"},
                     "arithmetic": {"goto": "hot_loop"}},
        },
        "hot_loop": {
            "question": "Does the computation repeat the same inner work many "
                        "times over?",
            "options": {
                "repeated": "One inner block of work runs many times, inside a "
                            "loop or nested loops.",
                "once": "The work is done once, or only a handful of times.",
            },
            "next": {"repeated": {"leaf": "loops"},
                     "once": {"leaf": NO_HINT}},
        },
    },
}

TREE_INSTRUCTIONS = ("Answer this question about the programming problem "
                     "above. Choose the option that describes it best.")


def validate_tree(tree: dict, recipes: list[dict] | None = None) -> dict:
    """Structural check plus leaf-coverage. Raises on anything malformed.

    A tree arriving from a generator is untrusted input: a dangling `goto`, a
    cycle, or a leaf naming a recipe that is not in the pool would each turn
    into a confusing runtime error deep in a walk, one problem at a time.
    """
    nodes = tree.get("nodes") or {}
    root = tree.get("root")
    if root not in nodes:
        raise SelectorError(f"root {root!r} is not a node")
    known = set(_ids(_pool(recipes))) | {NO_HINT}
    leaves, edges = set(), 0
    for nid, node in nodes.items():
        opts = node.get("options") or {}
        if len(opts) < 2:
            raise SelectorError(f"node {nid!r} has fewer than two options; a "
                                f"choice needs a closed set to choose from")
        if set(node.get("next") or {}) != set(opts):
            raise SelectorError(f"node {nid!r}: every option needs a next")
        for opt, nxt in node["next"].items():
            edges += 1
            if "goto" in nxt:
                if nxt["goto"] not in nodes:
                    raise SelectorError(f"node {nid!r} option {opt!r} goes to "
                                        f"unknown node {nxt['goto']!r}")
            elif "leaf" in nxt:
                if nxt["leaf"] not in known:
                    raise SelectorError(f"node {nid!r} option {opt!r} leaves to "
                                        f"{nxt['leaf']!r}, not in the pool")
                leaves.add(nxt["leaf"])
            else:
                raise SelectorError(f"node {nid!r} option {opt!r}: need goto or leaf")

    # Cycle check by walking every path; a cycle in a tree the walker follows
    # is an infinite loop per problem.
    def walk(nid, seen):
        if nid in seen:
            raise SelectorError(f"cycle through node {nid!r}")
        for nxt in nodes[nid]["next"].values():
            if "goto" in nxt:
                walk(nxt["goto"], seen | {nid})
    walk(root, set())
    return {"nodes": len(nodes), "edges": edges, "leaves": sorted(leaves),
            "depth": tree_depth_risk(tree)["max_depth"]}


def tree_depth_risk(tree: dict, per_node_accuracy: float = 0.9) -> dict:
    """Expected end-to-end accuracy for a given per-node accuracy.

    Error compounds: a depth-5 path at 0.9 per node is 0.9**5 = 0.590. That is
    the number that decides whether a tree is affordable, and it belongs in the
    code rather than in a postmortem. Reported at max depth (the worst path)
    and at the mean leaf depth, because a tree with one deep branch is not the
    same object as a uniformly deep one.
    """
    nodes = tree["nodes"]
    depths: dict[str, int] = {}

    def walk(nid, d, path):
        for opt, nxt in nodes[nid]["next"].items():
            if "goto" in nxt:
                walk(nxt["goto"], d + 1, path + [f"{nid}={opt}"])
            else:
                depths["/".join(path + [f"{nid}={opt}"])] = d
    walk(tree["root"], 1, [])
    ds = list(depths.values()) or [0]
    mx, mean = max(ds), sum(ds) / len(ds)
    p = float(per_node_accuracy)
    return {"per_node_accuracy": p,
            "max_depth": mx, "mean_depth": round(mean, 3),
            "accuracy_at_max_depth": round(p ** mx, 4),
            "accuracy_at_mean_depth": round(p ** mean, 4),
            "per_path": {k: round(p ** v, 4) for k, v in sorted(depths.items())}}


def select_tree(problem_text: str, recipes: list[dict] | None = None,
                k: int = 1, tree: dict | None = None,
                margin_gate: float = TREE_MARGIN_GATE,
                on_abstain: str = "no_hint",
                n_permutations: int = LAYA_PERMUTATIONS) -> Selection:
    """Walk a decision tree, one permutation-averaged Laya choice per node.

    Every node is the configuration Laya is measured to be GOOD at: the state
    is fixed (the problem, first, nothing appended) and the options vary inside
    one forward pass. No node compares scores across passages, so the
    0.497-collapses-to-0 failure cannot occur here.

    ABSTENTION, not guessing. A node whose averaged margin falls below the gate
    has not made a decision -- measured, reordering-sensitive cases land at
    0.012-0.184 while a decided case sits at 0.800 -- and guessing there is how
    a depth-4 walk turns into noise. The walk stops and the declared fallback
    takes over: "no_hint" returns the no-hint leaf (an empty but FLAGGED
    selection, never a silent one), "embedding" hands the problem to the cosine
    selector.

    The per-node trace is always recorded. Compounding error is the main risk
    of this design and it cannot be diagnosed from the final id alone.
    """
    pool = _pool(recipes)
    tree = tree or EXAMPLE_TREE
    nodes = tree["nodes"]
    k = _clamp_k(k, len(pool))
    trace: list[dict] = []
    nid = tree["root"]
    visited = set()

    while True:
        if nid in visited:                       # validate_tree catches this
            return _fallback(problem_text, pool, k, "tree", "fixed",
                             f"cycle at node {nid!r}")
        visited.add(nid)
        node = nodes[nid]
        try:
            v = _laya_choice(problem_text, f"{TREE_INSTRUCTIONS}\n{node['question']}",
                             node["options"], n_permutations)
        except LayaContractError:
            raise
        except Exception as e:                                   # noqa: BLE001
            v = None
            trace.append({"node": nid, "error": f"{type(e).__name__}: {e}"})
        if v is None:
            if not trace or trace[-1].get("node") != nid:
                trace.append({"node": nid, "error": "no usable response"})
            sel = _fallback(problem_text, pool, k, "tree", "fixed",
                            f"laya unavailable at node {nid!r}")
            sel.trace = trace
            return sel

        entry = {"node": nid, "choice": v["choice"], "margin": v["margin"],
                 "confidence": v["confidence"], "probabilities": v["probabilities"],
                 "orders": v["orders"], "orders_used": v["orders_used"],
                 "abstained": v["margin"] < margin_gate}
        trace.append(entry)

        if entry["abstained"]:
            if on_abstain == "embedding":
                sel = select_embedding(problem_text, pool, k)
                sel.mechanism = "tree"
                sel.fallback = (f"embedding: node {nid!r} margin "
                                f"{v['margin']} < {margin_gate}")
                sel.abstained = True
                sel.trace = trace
                return sel
            return Selection(ids=[], mechanism="tree", margin=v["margin"],
                             confidence=v["confidence"], abstained=True,
                             fallback=f"no_hint: node {nid!r} margin "
                                      f"{v['margin']} < {margin_gate}",
                             trace=trace,
                             notes="ABSTAINED -- empty by decision, not by "
                                   "failure. Flagged so a run can count it.")

        nxt = node["next"][v["choice"]]
        if "goto" in nxt:
            nid = nxt["goto"]
            continue

        leaf = nxt["leaf"]
        margins = [t["margin"] for t in trace if "margin" in t]
        if leaf == NO_HINT:
            return Selection(ids=[], mechanism="tree", margin=min(margins),
                             abstained=True, trace=trace,
                             notes="reached the no-hint leaf: the tree decided "
                                   "no recipe applies. Not a failure.")
        ids = [leaf]
        if k > 1:
            # The tree is a k=1 mechanism by construction -- a leaf names one
            # recipe. Extra slots are filled in pool order and declared, rather
            # than pretending the walk ranked anything beyond its leaf.
            ids += [i for i in _ids(pool) if i != leaf]
        return Selection(
            ids=ids[:k], mechanism="tree",
            # The end-to-end confidence of a path is bounded by its weakest
            # node, not by its last one. Reporting the leaf's margin alone
            # would hide a walk that nearly abstained three nodes up.
            margin=min(margins), confidence=min(t["confidence"] for t in trace
                                                if "confidence" in t),
            trace=trace,
            notes=(f"depth {len(trace)}; margin reported is the weakest node "
                   f"on the path" + (", extra slots filled in pool order"
                                     if k > 1 else "")))


def tree_path(sel: Selection) -> dict:
    """{node_id: option} from a walk, for node_discrimination()."""
    return {t["node"]: t["choice"] for t in sel.trace
            if "choice" in t and not t.get("abstained")}


def node_discrimination(tree: dict, samples: list[dict]) -> dict:
    """Does each node's question actually separate different best arms?

    samples: [{"path": {node_id: option}, "best_arm": id}, ...] -- the path
    from a walk (tree_path) and the best arm from an oracle run.

    A node whose branches lead to the SAME best arm is pure added error: it
    costs a Laya call, a chance to be wrong, and one multiplication off the
    end-to-end accuracy in tree_depth_risk, and buys nothing. Those nodes are
    prunable, and finding them is the cheapest improvement available to a
    generated tree.

    Reports counts rather than a verdict where the evidence is thin: a branch
    seen twice cannot tell you anything (PROTOCOL rule 4), so `discriminates`
    is None rather than False below `min_per_branch`.
    """
    min_per_branch = 3
    out = {}
    for nid, node in tree["nodes"].items():
        per_branch: dict[str, dict[str, int]] = {o: {} for o in node["options"]}
        for s in samples:
            opt = (s.get("path") or {}).get(nid)
            if opt is None or opt not in per_branch:
                continue
            arm = s.get("best_arm")
            per_branch[opt][arm] = per_branch[opt].get(arm, 0) + 1
        populated = {o: c for o, c in per_branch.items() if c}
        majority = {o: max(c, key=lambda a: c[a]) for o, c in populated.items()}
        n = {o: sum(c.values()) for o, c in populated.items()}
        thin = len(populated) < 2 or any(v < min_per_branch for v in n.values())
        out[nid] = {
            "counts": per_branch,
            "n_per_branch": n,
            "majority_arm": majority,
            "discriminates": None if thin else len(set(majority.values())) > 1,
            "prunable": (not thin) and len(set(majority.values())) == 1,
            "why": ("too few samples per branch to tell" if thin else
                    "branches agree on the same best arm" if
                    len(set(majority.values())) == 1 else
                    "branches lead to different best arms"),
        }
    return out


# =============================================================== ORACLE =====

ORACLE_MAP = os.environ.get(
    "RECIPE_ORACLE_MAP", os.path.join(BENCH, "recipe_best_arm.jsonl"))


def problem_key(problem_text: str) -> str:
    """Stable id for a problem when the caller has no question_id."""
    norm = " ".join(_words(problem_text))[:4000]
    return hashlib.sha1(norm.encode()).hexdigest()


def _load_oracle_map(path: str) -> dict:
    """{key: [best arms]} indexed by every id the file offers.

    Accepts question_id, problem_id or sha1 as the key, and best_arm or
    best_arms as the value, because the map is produced by whatever analysis
    ran over recipe_results.jsonl and pinning one spelling now would only cause
    a silent miss later.
    """
    if not os.path.exists(path):
        raise OracleUnavailable(
            f"oracle map not found: {path}\n"
            f"The ceiling is not something this code can compute -- it comes "
            f"from a completed recipe_oracle.py run, reduced to one line per "
            f"problem: {{\"question_id\": ..., \"best_arms\": [...]}}. "
            f"Set RECIPE_ORACLE_MAP or produce the file. Refusing to guess.")
    index: dict = {}
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise OracleUnavailable(f"{path}:{ln} is not JSON: {e}")
            arms = r.get("best_arms")
            if arms is None:
                arms = [r["best_arm"]] if r.get("best_arm") else []
            for field_name in ("question_id", "problem_id", "sha1", "key"):
                if r.get(field_name):
                    index[str(r[field_name])] = list(arms)
    if not index:
        raise OracleUnavailable(f"{path} contains no usable rows")
    return index


def select_oracle(problem_text: str, recipes: list[dict] | None = None,
                  k: int = 1, problem_id: str | None = None,
                  path: str | None = None) -> Selection:
    """THE CEILING. A CHEAT. NOT DEPLOYABLE -- it reads the answer.

    It looks up the best arm per problem from a map produced by a completed
    oracle run. It exists to bound the other mechanisms: without it, a
    selector landing 4 points above random is uninterpretable, because nobody
    knows whether the available headroom was 5 points or 40.

    It RAISES when the map is missing or does not cover the problem. Falling
    back would turn the ceiling into a guess that everything else is then
    compared against -- a fallback hiding an outage, PROTOCOL rule 2.
    """
    pool = _pool(recipes)
    k = _clamp_k(k, len(pool))
    index = _load_oracle_map(path or ORACLE_MAP)
    key = problem_id if problem_id is not None else problem_key(problem_text)
    arms = index.get(str(key))
    if arms is None and problem_id is not None:
        arms = index.get(problem_key(problem_text))
    if arms is None:
        raise OracleUnavailable(
            f"no oracle row for {key!r}. The ceiling is undefined for a "
            f"problem the oracle run did not cover; exclude it from the "
            f"comparison rather than scoring it against a guess.")
    arms = [a for a in arms if a != NO_HINT]
    if not arms:
        # A real oracle answer: no hint beat every hint on this problem. The
        # only selector allowed to return fewer than k ids, and it is flagged.
        return Selection(ids=[], mechanism="oracle", abstained=True,
                         notes="oracle: no hint was the best arm")
    if len(arms) < k:
        raise OracleUnavailable(
            f"oracle row for {key!r} names {len(arms)} arm(s) but k={k} was "
            f"requested. A k={k} ceiling needs a k={k} oracle run; padding it "
            f"would invent headroom.")
    return Selection(ids=arms[:k], mechanism="oracle",
                     notes="CEILING -- cheat, not deployable")


# ============================================================= REGISTRY =====

SELECTORS = {
    "random": select_random,
    "fixed": select_fixed,
    "embedding": select_embedding,
    "rerank": select_rerank,
    "laya": select_laya,
    "tree": select_tree,
    "oracle": select_oracle,
}

# What each arm MEANS in the comparison. A middle result is only interpretable
# between the two bounds, and the ceiling is not a shippable option.
BOUND = {"random": "floor", "fixed": "floor", "oracle": "ceiling"}
DEPLOYABLE = {name: (name != "oracle") for name in SELECTORS}
select_oracle.deployable = False
for _n, _f in SELECTORS.items():
    _f.deployable = DEPLOYABLE[_n]
    _f.bound = BOUND.get(_n)


def select(mechanism: str, problem_text: str, recipes: list[dict] | None = None,
           k: int = 1, **kw) -> Selection:
    """Dispatch by name. The whole point of the file: identical inputs, one
    line changed between arms."""
    if mechanism not in SELECTORS:
        raise SelectorError(f"unknown mechanism {mechanism!r}. "
                            f"Available: {', '.join(SELECTORS)}")
    return SELECTORS[mechanism](problem_text, recipes, k, **kw)


# ============================================================== DRY RUN =====
# Mocks, not stubs: each one reproduces a MEASURED property of the thing it
# stands in for, so the logic is validated against the failure modes that
# actually occur rather than against a convenient fiction.

def mock_embed(texts: list[str], is_query: bool = False, dim: int = 1024):
    """Hashed bag-of-words, unit-normalised.

    Honours embed()'s contract -- unit-norm rows, so a dot product is cosine --
    and nothing else. It cannot model the asymmetric query/document prefix, so
    it validates the SELECTION logic and says nothing about retrieval quality.

    1024 dims, not 64: at 64 the expected hash collisions between a 13-word
    problem and a 21-word recipe (~4 dims) outweighed a genuine two-word
    overlap, and the mock ranked a loop-hoisting hint above a deque hint. That
    is collision noise, not a selector bug, and a mock that produces it tests
    nothing.
    """
    out = []
    for t in texts:
        v = [0.0] * dim
        for w in _content_words(t):
            v[int(hashlib.sha1(w.encode()).hexdigest()[:8], 16) % dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def mock_rerank(query: str, docs: list[str], top_n: int, degenerate=False):
    """Word-overlap order, at the ~1e-13 scale the real reranker returns.

    `degenerate=True` reproduces context overflow: every score identical, which
    is the case the code must catch, as opposed to "scores are small", which it
    must NOT treat as failure.
    """
    q = _content_words(query)
    rows = []
    for i, d in enumerate(docs):
        overlap = len(q & _content_words(d))
        rows.append((i, 1e-13 if degenerate else overlap * 1e-13 + 1e-15))
    rows.sort(key=lambda x: -x[1])
    return rows[:top_n]


class MockLaya:
    """A Laya /decide endpoint with the measured positional bias baked in.

    Two properties matter and both are reproduced:
      1. content signal -- word overlap between the state and each criterion.
      2. FIRST-OPTION BIAS -- the artefact that picked "a" eight times. A
         single-order call therefore gets it wrong on purpose, which is what
         makes the permutation-averaging assertions mean something.

    `fail=True` raises, for the degradation path.
    """

    def __init__(self, bias: float = 0.30, fail: bool = False,
                 uniform: bool = False):
        self.bias, self.fail, self.uniform = bias, fail, uniform
        self.calls: list[dict] = []

    def __call__(self, url: str, payload: dict, timeout: int = 30):
        self.calls.append(json.loads(json.dumps(payload)))
        if self.fail:
            raise OSError("mock laya: connection refused")
        q = payload["questions"]["pick"]
        criteria, state = q["criteria"], payload["state"]
        keys = list(criteria)
        sw = _content_words(state)
        # Cubed so a clearly-matching option lands near the measured 0.8 that a
        # DECIDED case sits at, and a near-tie stays in the 0.012-0.184 band
        # that must trigger abstention. A linear signal put every node in the
        # ambiguous band and nothing would ever have been testable.
        raw = {k: (1.0 if self.uniform else
                   (1.0 + len(sw & _content_words(criteria[k]))) ** 3)
               for k in keys}
        tot = sum(raw.values())
        probs = {k: v / tot for k, v in raw.items()}
        probs[keys[0]] += self.bias                      # the artefact
        tot = sum(probs.values())
        probs = {k: v / tot for k, v in probs.items()}
        return {"answers": {"pick": {"choice": max(probs, key=probs.get),
                                     "probabilities": probs,
                                     "confidence": max(probs.values())}},
                "elapsed_ms": 23.0}


class _Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, name, cond, detail=""):
        self.rows.append((bool(cond), name, detail))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"   [{detail}]" if detail else ""))
        return bool(cond)

    def report(self) -> int:
        bad = [r for r in self.rows if not r[0]]
        print(f"\n  {len(self.rows) - len(bad)}/{len(self.rows)} assertions passed")
        return 1 if bad else 0


IO_PROBLEM = ("You are given 200000 integers, one per line, and must print the "
              "running total after each line. The number of lines can reach "
              "100000 or more, so reading and writing the values is most of "
              "the work.")
SEQ_PROBLEM = ("A small simulation with at most 500 elements. Repeatedly remove "
               "an element from the front of the sequence, test membership, and "
               "append to the back.")
VAGUE_PROBLEM = "Solve the task."


def run_dry_run() -> int:                                        # noqa: C901
    ck = _Checks()
    pool = default_pool()
    ids = set(_ids(pool))
    print(f"\n  pool: {len(pool)} recipes {sorted(ids)}  "
          f"(placebo excluded -- it is the control arm)\n")

    laya = MockLaya()
    with mock_backends(embed=mock_embed,
                       rerank=lambda q, d, n: mock_rerank(q, d, n),
                       post=laya):

        # 1. shape contract, every deployable selector, every k
        ok, detail = True, ""
        for name in ("random", "fixed", "embedding", "rerank", "laya", "tree"):
            for k in (1, 2, 3):
                sel = select(name, IO_PROBLEM, pool, k)
                if len(sel.ids) != k or not set(sel.ids) <= ids \
                        or len(set(sel.ids)) != len(sel.ids):
                    ok, detail = False, f"{name} k={k} -> {sel.ids}"
        ck("every selector returns exactly k distinct ids from the pool", ok, detail)

        # 2. k above the pool size clamps instead of crashing
        ck("k > pool size clamps to the pool",
           len(select("embedding", IO_PROBLEM, pool, 99).ids) == len(pool))

        # 3/4. the random floor is reproducible, and seed-dependent
        a = list(select_random(IO_PROBLEM, pool, 2, seed=7))
        b = list(select_random(IO_PROBLEM, pool, 2, seed=7))
        c = list(select_random(IO_PROBLEM, pool, 2, seed=8))
        ck("random is reproducible under a fixed seed", a == b, f"{a} == {b}")
        ck("random differs under a different seed", a != c, f"{a} vs {c}")
        ck("random varies across problems at one seed",
           list(select_random(IO_PROBLEM, pool, 1, seed=7))
           != list(select_random(SEQ_PROBLEM, pool, 1, seed=7)))

        # 5. the fixed floor ignores the problem
        ck("fixed returns the same arm for every problem",
           list(select_fixed(IO_PROBLEM, pool, 1))
           == list(select_fixed(SEQ_PROBLEM, pool, 1))
           == [FIXED_DEFAULT_ID])

        # 6. embedding ranks on content and reports a margin
        sel = select_embedding(SEQ_PROBLEM, pool, 1)
        ck("embedding picks the content-matching recipe and reports cosine",
           sel.ids == ["structures"] and sel.confidence is not None
           and sel.margin is not None, f"{sel.ids} conf={sel.confidence}")

        # 7. rerank uses the cross-encoder ordering
        sel = select_rerank(SEQ_PROBLEM, pool, 1)
        ck("rerank returns cross-encoder order with an ordinal margin only",
           sel.ids == ["structures"] and sel.confidence is None
           and sel.fallback is None, f"{sel.ids}")

        # 8. degenerate reranker (context overflow) -> declared fallback
        with mock_backends(embed=mock_embed,
                           rerank=lambda q, d, n: mock_rerank(q, d, n, True),
                           post=laya):
            sel = select_rerank(SEQ_PROBLEM, pool, 1)
        ck("identical rerank scores degrade to embedding, not to empty",
           len(sel.ids) == 1 and sel.fallback and sel.fallback.startswith("embedding"),
           sel.fallback or "")

        # 9. LAYA CONTRACT: candidates in criteria only, never in state
        laya.calls.clear()
        select_laya(IO_PROBLEM, pool, 1)
        clean = bool(laya.calls)
        for call in laya.calls:
            st = call["state"]
            crit = call["questions"]["pick"]["criteria"]
            if set(crit) != ids or any(r["text"][:60] in st for r in pool):
                clean = False
        ck("laya: candidates appear in `criteria` only, never in `state`",
           clean, f"{len(laya.calls)} calls inspected")

        # 10. the problem goes FIRST and is truncated at the tail
        long_problem = IO_PROBLEM + " padding" * 4000
        laya.calls.clear()
        select_laya(long_problem, pool, 1)
        st = laya.calls[0]["state"]
        ck("laya: problem first, tail truncated (laya drops a long tail)",
           long_problem.startswith(st) and len(st) == LAYA_STATE_CHARS,
           f"state={len(st)} chars")

        # 11. permutation averaging actually happens
        laya.calls.clear()
        select_laya(IO_PROBLEM, pool, 1)
        orders = {tuple(c["questions"]["pick"]["criteria"]) for c in laya.calls}
        ck("laya: averaged over more than one option ordering",
           len(orders) > 1 and len(laya.calls) >= 2, f"{len(orders)} distinct orders")

        # 12. and it cancels the positional artefact the mock reproduces.
        # Strong bias, because the measured artefact was strong: option "a"
        # eight times at margins up to 0.919. SEQ_PROBLEM's content winner is
        # `structures`, which is NOT first in pool order, so a single-order
        # call gets it wrong and the average gets it right.
        keys = _ids(pool)
        strong = MockLaya(bias=1.0)
        fwd = strong("mock", {"state": SEQ_PROBLEM[:LAYA_STATE_CHARS],
                              "questions": {"pick": {
                                  "type": "choice", "instructions": "x",
                                  "criteria": {r["id"]: r["text"]
                                               for r in pool}}}})
        fwd_pick = max(fwd["answers"]["pick"]["probabilities"],
                       key=fwd["answers"]["pick"]["probabilities"].get)
        with mock_backends(embed=mock_embed, rerank=mock_rerank, post=strong):
            avg = select_laya(SEQ_PROBLEM, pool, 1)
        ck("laya: averaging overturns the first-option artefact",
           fwd_pick == keys[0] and avg.ids[0] == "structures",
           f"single-order={fwd_pick}, averaged={avg.ids[0]}")

        # 13. a contract violation is fatal, not silent
        try:
            _laya_choice(IO_PROBLEM, "x",
                         {"a": IO_PROBLEM[:80], "b": "something else entirely"})
            raised = False
        except LayaContractError:
            raised = True
        ck("laya: repeating a candidate inside `state` raises", raised)

        # 14. laya failure -> declared fallback, never a silent empty list
        with mock_backends(embed=mock_embed, rerank=mock_rerank,
                           post=MockLaya(fail=True)):
            sel = select_laya(IO_PROBLEM, pool, 2)
        ck("laya failure degrades to the declared fixed fallback",
           len(sel.ids) == 2 and sel.fallback and sel.fallback.startswith("fixed"),
           sel.fallback or "")

        # 15. embedding failure -> declared fallback too
        def boom(*_a, **_k):
            raise OSError("mock embeddings: connection refused")
        with mock_backends(embed=boom, rerank=mock_rerank, post=laya):
            sel = select_embedding(IO_PROBLEM, pool, 1)
        ck("embedding failure degrades to the declared fixed fallback",
           len(sel.ids) == 1 and sel.fallback and sel.fallback.startswith("fixed"),
           sel.fallback or "")

        # ------------------------------------------------------ the tree ---
        info = validate_tree(EXAMPLE_TREE, pool)
        ck("example tree validates: no dangling goto, no cycle, leaves in pool",
           info["nodes"] == 4 and info["depth"] == 3
           and set(info["leaves"]) <= (ids | {NO_HINT}), str(info))

        # 16. the walk terminates at a leaf and records every node
        laya.calls.clear()
        sel = select_tree(IO_PROBLEM, pool, 1)
        ck("tree walk terminates at a leaf with a per-node trace",
           sel.ids and len(sel.trace) >= 2 and sel.fallback is None
           and all("margin" in t for t in sel.trace),
           f"{sel.ids} via {[t['node'] + '=' + t['choice'] for t in sel.trace]}")

        # 17. every node was permutation-averaged, not just the first
        per_node: dict = {}
        for c in laya.calls:
            instr = c["questions"]["pick"]["instructions"]
            per_node.setdefault(instr, set()).add(
                tuple(c["questions"]["pick"]["criteria"]))
        ck("tree: every node averaged over >1 ordering",
           len(per_node) == len(sel.trace)
           and all(len(v) > 1 for v in per_node.values()),
           f"{ {len(v) for v in per_node.values()} } orders per node")

        # 18. every tree call obeys the criteria-only contract as well
        clean = all(
            not any(opt[:60] in c["state"]
                    for opt in c["questions"]["pick"]["criteria"].values())
            for c in laya.calls)
        ck("tree: candidate descriptions never leak into `state`", clean)

        # 19. an undecidable node abstains instead of guessing
        with mock_backends(embed=mock_embed, rerank=mock_rerank,
                           post=MockLaya(uniform=True)):
            sel = select_tree(VAGUE_PROBLEM, pool, 1)
        ck("tree: a below-gate margin abstains rather than guessing",
           sel.ids == [] and sel.abstained and sel.fallback
           and sel.fallback.startswith("no_hint") and len(sel.trace) == 1,
           f"margin={sel.margin} fallback={sel.fallback}")

        # 20. abstention can hand off to a declared mechanism instead
        with mock_backends(embed=mock_embed, rerank=mock_rerank,
                           post=MockLaya(uniform=True)):
            sel = select_tree(SEQ_PROBLEM, pool, 1, on_abstain="embedding")
        ck("tree: abstention can degrade to embedding, still flagged",
           len(sel.ids) == 1 and sel.abstained
           and sel.fallback.startswith("embedding"), sel.fallback or "")

        # 21. laya down mid-walk -> declared fallback with the trace kept
        with mock_backends(embed=mock_embed, rerank=mock_rerank,
                           post=MockLaya(fail=True)):
            sel = select_tree(IO_PROBLEM, pool, 1)
        ck("tree: laya failure degrades to fixed and keeps the trace",
           len(sel.ids) == 1 and sel.fallback and sel.trace
           and "error" in sel.trace[-1], sel.fallback or "")

        # 22. reported margin is the WEAKEST node, not the last one
        sel = select_tree(IO_PROBLEM, pool, 1)
        ck("tree: reported margin is the weakest node on the path",
           abs(sel.margin - min(t["margin"] for t in sel.trace)) < 1e-9,
           f"{sel.margin} == min({[t['margin'] for t in sel.trace]})")

    # ---------------------------------------------- depth risk (no backend) --
    # 23. hand-computed: 0.9**5 = 0.59049, 0.9**3 = 0.729
    deep = {"root": "n1", "nodes": {
        f"n{i}": {"question": "q", "options": {"y": "yes", "n": "no"},
                  "next": {"y": ({"goto": f"n{i + 1}"} if i < 5 else {"leaf": "io"}),
                           "n": ({"goto": f"n{i + 1}"} if i < 5 else {"leaf": "loops"})}}
        for i in range(1, 6)}}
    r5 = tree_depth_risk(deep, 0.9)
    r3 = tree_depth_risk(EXAMPLE_TREE, 0.9)
    ck("tree_depth_risk: depth 5 at 0.9/node = 0.5905 end to end",
       r5["max_depth"] == 5 and abs(r5["accuracy_at_max_depth"] - 0.59049) < 1e-4,
       f"{r5['accuracy_at_max_depth']}")
    ck("tree_depth_risk: example tree depth 3 at 0.9/node = 0.729",
       r3["max_depth"] == 3 and abs(r3["accuracy_at_max_depth"] - 0.729) < 1e-4,
       f"depth={r3['max_depth']} acc={r3['accuracy_at_max_depth']} "
       f"mean_depth={r3['mean_depth']}")

    # 24/25. node_discrimination flags a node whose branches agree
    samples = (
        [{"path": {"scale": "huge", "huge_shape": "reading"}, "best_arm": "io"}] * 4
        + [{"path": {"scale": "huge", "huge_shape": "searching"}, "best_arm": "io"}] * 4
        + [{"path": {"scale": "small", "small_shape": "sequence"},
            "best_arm": "structures"}] * 4)
    disc = node_discrimination(EXAMPLE_TREE, samples)
    ck("node_discrimination: flags a node whose branches agree as prunable",
       disc["huge_shape"]["prunable"] is True
       and disc["huge_shape"]["discriminates"] is False,
       disc["huge_shape"]["why"])
    ck("node_discrimination: keeps a node whose branches separate arms",
       disc["scale"]["discriminates"] is True
       and disc["scale"]["prunable"] is False, disc["scale"]["why"])
    ck("node_discrimination: refuses a verdict on a thin branch",
       disc["small_shape"]["discriminates"] is None,
       disc["small_shape"]["why"])

    # ------------------------------------------------------------ oracle ---
    import tempfile
    missing = os.path.join(tempfile.gettempdir(), "no_such_oracle_map.jsonl")
    try:
        select_oracle(IO_PROBLEM, pool, 1, path=missing)
        raised, msg = False, ""
    except OracleUnavailable as e:
        raised, msg = True, str(e)
    ck("oracle raises clearly when its map is absent",
       raised and missing in msg and "recipe_oracle" in msg,
       msg.splitlines()[0] if msg else "")

    fd, tmp = tempfile.mkstemp(suffix=".jsonl", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"question_id": "q1",
                             "best_arms": ["io", "loops"]}) + "\n")
        fh.write(json.dumps({"sha1": problem_key(SEQ_PROBLEM),
                             "best_arm": "structures"}) + "\n")
        fh.write(json.dumps({"question_id": "q3", "best_arm": NO_HINT}) + "\n")
    try:
        sel = select_oracle("", pool, 2, problem_id="q1", path=tmp)
        ck("oracle reads the precomputed best arms when the map exists",
           list(sel) == ["io", "loops"], str(list(sel)))
        sel = select_oracle(SEQ_PROBLEM, pool, 1, path=tmp)
        ck("oracle resolves a problem by content hash when no id is given",
           list(sel) == ["structures"], str(list(sel)))
        sel = select_oracle("", pool, 1, problem_id="q3", path=tmp)
        ck("oracle returns an empty FLAGGED selection when no hint won",
           sel.ids == [] and sel.abstained, sel.notes)
        try:
            select_oracle("", pool, 2, problem_id="q3x", path=tmp)
            raised = False
        except OracleUnavailable:
            raised = True
        ck("oracle raises on a problem its run did not cover", raised)
        try:
            select_oracle(SEQ_PROBLEM, pool, 2, path=tmp)
            raised = False
        except OracleUnavailable:
            raised = True
        ck("oracle refuses to pad a k=2 ceiling from a k=1 run", raised)
    finally:
        os.unlink(tmp)

    ck("oracle is marked NOT deployable in the registry",
       DEPLOYABLE["oracle"] is False and select_oracle.deployable is False
       and select_oracle.bound == "ceiling"
       and all(DEPLOYABLE[n] for n in SELECTORS if n != "oracle"))
    ck("both floors are declared as floors",
       select_random.bound == "floor" and select_fixed.bound == "floor")

    # best_fixed_arm: configure the floor from a run instead of guessing
    ck("best_fixed_arm returns None before any oracle run exists",
       best_fixed_arm(os.path.join(tempfile.gettempdir(), "nope.jsonl")) is None)
    fd, tmp = tempfile.mkstemp(suffix=".jsonl", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for qid in ("a", "b", "c"):
            fh.write(json.dumps({"question_id": qid, "arm": "io",
                                 "passed": True}) + "\n")
            fh.write(json.dumps({"question_id": qid, "arm": "loops",
                                 "passed": qid == "a"}) + "\n")
    try:
        ck("best_fixed_arm picks the strongest single arm from a run",
           best_fixed_arm(tmp) == ("io", 3, 3), str(best_fixed_arm(tmp)))
    finally:
        os.unlink(tmp)

    return ck.report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="validate every mechanism against mocks. No GPU, no network.")
    ap.add_argument("--list", action="store_true", help="list mechanisms")
    ap.add_argument("--risk", action="store_true",
                    help="print tree_depth_risk for the example tree")
    ap.add_argument("--per-node", type=float, default=0.9)
    args = ap.parse_args()

    if args.list:
        for name, fn in SELECTORS.items():
            tag = fn.bound or ("mechanism" if fn.deployable else "")
            print(f"  {name:<12}{tag:<10}"
                  f"{'' if fn.deployable else 'NOT DEPLOYABLE  '}"
                  f"{(fn.__doc__ or '').strip().splitlines()[0]}")
        return
    if args.risk:
        print(json.dumps(tree_depth_risk(EXAMPLE_TREE, args.per_node), indent=2))
        print(json.dumps(validate_tree(EXAMPLE_TREE), indent=2))
        return
    if args.dry_run:
        print("  DRY RUN -- mock embeddings, mock laya. No GPU, no network.")
        raise SystemExit(run_dry_run())
    ap.print_help()


if __name__ == "__main__":
    main()
