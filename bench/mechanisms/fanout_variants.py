#!/usr/bin/env python
"""Sampling and fan-out strategies as named arms, so they can be compared.

WHY THIS FILE EXISTS AT ALL

mcp/fanout.py already fans out, but it bundles four things into one knob:
more samples, higher temperature, different system nudges, and a consensus
selector over extracted file paths. When the 8-question koota eval came back
7/8 against 7/8 it could not say which of those four did nothing, because none
of them was ever varied alone.

Worse, the earlier LiveCodeBench-style attempt scored fan-out 0/8 for a reason
that had nothing to do with fan-out: its `_one()` issued a single completion
instead of running the tool loop, so every variant returned an empty string and
a pending tool call. fanout.py carries the comment about that bug. PROTOCOL
rule 3 is explicit -- a result that is exactly 0 is a harness bug until proven
otherwise -- and rule 9 says a mechanism that has not been measured on a
working system has not been measured. One broken configuration is not evidence
that consensus is dead.

So this file separates the knobs into arms that differ in exactly one thing:

    single_greedy      n=1, T=0.0            control
    single_temp        n=1, T=0.6            control for temperature ALONE
    fanout_n3_vote     n=3, T=0.6, majority  temperature + votes
    fanout_n5_vote     n=5, T=0.6, majority  does n buy anything past 3
    fanout_n3_seeded   n=3, T=0.6, majority, one concept seed word per sample
    fanout_n3_longest  n=3, T=0.6, longest   deliberate floor
    fanout_n3_features n=3, T=0.6, feature tree over discrete Laya labels
    fanout_n5_features n=5, T=0.6, same tree
    fanout_n3_features_gated  as n3_features, low-margin labels -> UNKNOWN

single_temp is the arm that makes the rest interpretable. Three samples at
T=0.6 differ from one sample at T=0.0 in two ways, and without the T=0.6
single-sample control, any win by fanout_n3_vote is equally well explained by
"temperature 0.6 happens to suit this model". fanout_n3_longest is the floor:
a selector that ignores agreement entirely. If voting cannot beat "pick the
longest", the selector is not doing the work and the extra samples are latency.

WHY IT TARGETS 11434 AND NOT 1234

fanout.py's UPSTREAM defaults to http://127.0.0.1:1234, which is now the PROXY,
not the model. Fanning out through the proxy would make the proxy fan out
again -- recursion, at N^2 samples. Generation here goes straight to the model
server at 11434. The proxy is a separate condition, measured by pointing
bench/livecodebench.py at it, not something this file should accidentally
include in every arm.

WHY AGREEMENT IS DEFINED OVER NORMALISED CODE

fanout.py votes on file paths and backquoted symbols, because its task is
"where is X defined". On a code-generation task there are no paths to vote on
and free prose cannot be voted on at all. What can be voted on is the artefact:
the program itself.

Two programs are the same vote when their code normalises to the same token
stream -- comments dropped, whitespace and blank lines canonicalised, block
structure kept. Tokenising rather than regex-stripping comments is PROTOCOL
rule 8: a regex cannot tell a `#` inside a string from a comment, and this repo
has already shipped one import-extraction bug for exactly that reason.

Deliberately NOT normalised: identifier names, literal values, docstrings,
statement order. Two solutions that differ only in variable naming count as
disagreeing. That is conservative in the safe direction -- it can only
UNDERSTATE agreement, so a measured win for voting is not an artefact of a
generous equivalence relation. An AST-level or behavioural equivalence (run
both, compare outputs on the public cases) would group more, and is the
obvious upgrade if these arms show that voting has any signal at all.

WHY THE TIE-BREAK IS THE MEDIAN LENGTH

With n=3 and a hard problem, the common outcome is three singleton groups: no
majority exists. Something still has to be returned, and the choice should not
be "the first one", which is arbitrary, or "the longest", which is the failure
mode this arm is supposed to beat -- fanout.py already scores `-len(content)`
into its own tie-break because the longest answer tends to be the rambling one.
The median-length candidate is the central one: it drops the truncated sample
and the padded sample, both of which are the tails that go wrong. Remaining
ties break to the lowest sample index, so the whole selector is a pure function
of the sample list and the same samples always give the same answer.

WHY THE FEATURE ARMS COMPARE LABELS AND NEVER SCORES

The obvious way to use the decision model is `laya_score(A) > laya_score(B)`.
That comparison is invalid here and it is invalid for a measured reason: hold a
passage fixed and vary the query, and a real query separates from nonsense by
0.497; vary the passage as well and the gap collapses to about zero. The scores
are calibrated WITHIN one state and mean nothing across two different states.
Ranking two candidate programs by their scores is exactly the across-state
comparison the measurement rules out.

What the model is measurably good at is a closed `choice` with a FIXED state:
one question, about one candidate, answered with a discrete label. So each
candidate is asked the same yes/no questions about ITSELF, producing a discrete
feature vector, and the candidates are then compared in FEATURE space by an
explicit tree. The only number that ever crosses candidates is a label. The
margin is used solely within a single question about a single candidate, as a
confidence gate.

Two protocol rules come from measured failures and are enforced in code, not
left to care:

  * The candidate goes in `state` and NEVER also in `criteria`. Putting the
    candidates in the options produced a positional artefact -- option "a" won
    eight times running, at margins up to 0.919. `_check_payload` raises if a
    call ever does it again.
  * Every question is asked twice, with the option order reversed, and the
    probabilities averaged. The model is not permutation invariant: 27-33%
    semantic flip rate on two options upstream, reproduced here at a margin of
    0.441 for an answer that flipped on order alone. Averaging cancels the
    positional component and, more importantly, makes the margin honest --
    cases that flip average down to 0.012-0.184 while a genuinely decided case
    stays near 0.800. That spread is where the 0.3 gate threshold comes from.
  * The candidate's code goes FIRST in `state`, because a long state is
    silently truncated at the tail.

THE FEATURE SELECTION RULE, STATED IN FULL

Features filter; votes break the residual tie.

  1. Veto. A candidate whose veto feature is confidently bad is dropped. If
     every candidate is dropped, the veto is ignored and that is recorded --
     a filter that empties the candidate set is an outage, not a decision
     (PROTOCOL rule 2).
  2. Lexicographic priority. Features are ordered once, in FEATURE_QUESTIONS.
     At each feature in turn, keep only the candidates holding the best value,
     where good (2) > UNKNOWN (1) > bad (0). UNKNOWN sits in the middle
     deliberately: no evidence of a flaw should beat evidence of a flaw, and
     should lose to a verified-good candidate. Stop as soon as one candidate
     remains.
  3. Residual tie -> select_majority over the survivors. So an all-UNKNOWN
     vector degenerates exactly to fanout_n3_vote, which is the correct
     behaviour when the features carried no information, and means the gated
     arm can never do worse than the vote arm for reasons of gating alone.

A lexicographic tree rather than a weighted sum because there is no evidence
for any weighting. A sum would need coefficients, and the only honest source
for them is a fit on data that does not exist yet.

WHAT THE FEATURE ARMS CANNOT SHOW, SAID UP FRONT

The starter questions are about PERFORMANCE idioms and LiveCodeBench scores
correctness. The only route by which a perf feature can change pass@1 is a case
that times out. So a null result for these arms is a null result for THESE
QUESTIONS on THIS grader, not for feature-based selection, and the honest next
step is correctness-shaped questions or a runtime-based outcome, not a verdict.

WHAT COUNTS AS A FAILURE AND WHAT COUNTS AS AN ERROR

PROTOCOL rule 3: never let an exception become a wrong answer. A sample that
raises is recorded as an error and excluded from the vote; the arm still
answers from the samples that worked. A sample that returns text with no
extractable code is NOT a vote either -- otherwise two apologies would outvote
one correct program, which is the 0/8 failure in a new costume. If every sample
of an arm fails, the arm raises AllSamplesFailed rather than returning "", so
the caller records a generation error instead of scoring a wrong answer.

Rule 2 applies to the per-sample tolerance as well: `error_rate` is reported on
every arm, and an arm that only ever answers on two of five samples is an
outage wearing a fan-out's clothes.

DRY RUN

`--dry-run` validates every selector against a mock generator with canned
responses and needs no GPU, no model server and no embedding index. It is the
cheapest layer that can answer "is the selection logic right" (rule 5), and it
is what caught the 0/8 harness bug class in the first place.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import inspect
import io
import json
import os
import random
import statistics
import sys
import time
import tokenize
import typing
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

import livecodebench as lcb  # noqa: E402

# The MODEL SERVER, not the proxy. See the module docstring: 1234 is the proxy
# now, and pointing fan-out at it recurses.
MODEL_URL = os.environ.get("FANOUT_MODEL_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("FANOUT_MODEL", "bonsai")

# Temperature for every multi-sample arm. One value across all of them so that
# n is the only thing that changes between fanout_n3_vote and fanout_n5_vote,
# and single_temp uses the same value so it is a real control for it.
FANOUT_TEMPERATURE = 0.6

# The decision model, on its own port and its own GPU. Never the proxy.
LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")
# Laya drops the tail of a long state silently, so the code is truncated here
# where it can be COUNTED rather than there where it cannot. dialectic.py uses
# 4000 for the same reason.
LAYA_STATE_CHARS = 4000
# Below this margin the two options did not separate once position bias is
# removed. fanout.break_tie uses 0.15 as a floor for "did it decide at all";
# 0.3 is stricter because a feature is used as a FACT here, not as a hint.
# Measured spread: ambiguous 0.012-0.184, decided about 0.800.
MARGIN_GATE = 0.3
UNKNOWN = "unknown"


class AllSamplesFailed(RuntimeError):
    """Every sample in an arm errored or produced no code.

    Raised, never swallowed. The caller records this as an ERROR, which is a
    different outcome from a wrong answer and must be counted separately.
    """


class FeatureProtocolError(RuntimeError):
    """A decision call broke the rule that produced a measured artefact.

    Loud on purpose. The positional artefact -- option "a" winning eight times
    at margins up to 0.919 -- looked exactly like a confident result, so the
    only safe handling is to make the malformed call impossible to send.
    """


class Feature(typing.NamedTuple):
    """One discrete answer about ONE candidate.

    A tuple whose first two fields are (label, margin), so it destructures the
    way a feature vector entry is normally read, while still carrying the
    averaged probabilities and the per-order raw readings. Requirement: the
    full trace is recorded, because error compounding across several questions
    is the main risk of this design and cannot be diagnosed from labels alone.
    """
    label: str
    margin: float
    probabilities: dict
    orders: list


@dataclasses.dataclass
class ArmResult:
    name: str
    answer: str                 # the raw reply text of the chosen sample
    code: str                   # the extracted code of the chosen sample
    chosen: int                 # index into samples
    samples: list
    n_requested: int
    n_voting: int               # samples that produced extractable code
    n_errors: int
    n_nocode: int
    agreement: float            # largest group / n_voting, 1.0 when n=1
    group_sizes: list
    selector: str
    ms: int
    # sample index -> {question id -> Feature}. Empty for the non-feature arms.
    features: dict = dataclasses.field(default_factory=dict)
    # Non-empty whenever the arm did not do what its name says: features
    # unavailable, veto ignored, gate swallowed everything. Never silent.
    notes: str = ""
    degraded: bool = False

    @property
    def error_rate(self) -> float:
        return self.n_errors / max(1, self.n_requested)


# ---------------------------------------------------------------- normalising

def _crude_normalise(code: str) -> str:
    """Fallback when the candidate does not tokenise: layout only.

    It deliberately does NOT strip comments with a regex. A candidate that
    fails to tokenise is exactly the case where a regex is most likely to be
    reading a `#` inside a broken string literal (PROTOCOL rule 8). Under-
    normalising costs a vote; mis-normalising invents agreement.
    """
    return "\n".join(l.rstrip() for l in code.splitlines() if l.strip())


def normalise(code: str) -> str:
    """Canonical form used for voting: comments gone, structure kept.

    INDENT/DEDENT are emitted as markers rather than as their literal
    whitespace, so `    ` and `\t` blocks are the same vote while a genuinely
    different block structure still is not.
    """
    if not (code or "").strip():
        return ""
    out: list = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING,
                          tokenize.ENDMARKER):
                continue
            if t.type == tokenize.NEWLINE:
                out.append("\n")
            elif t.type == tokenize.INDENT:
                out.append("<in>")
            elif t.type == tokenize.DEDENT:
                out.append("<out>")
            else:
                out.append(t.string)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return _crude_normalise(code)
    return " ".join(out).strip()


# ----------------------------------------------------------------- selectors

def _voting(samples: list) -> list:
    """The samples that get a vote: no exception, and code we could extract.

    A reply with no code block is not a quiet abstention that still counts --
    it is not a candidate at all. Letting empties vote is how a broken
    generation path scores 0/8 and reads as a verdict on the mechanism.
    """
    return [s for s in samples if not s["error"] and s["norm"]]


def _groups(ok: list) -> dict:
    """norm -> samples, in first-seen order. The basis of every agreement number."""
    g: dict = {}
    for s in ok:
        g.setdefault(s["norm"], []).append(s)
    return g


def _group_stats(ok: list) -> dict:
    sizes = sorted((len(g) for g in _groups(ok).values()), reverse=True)
    return {"agreement": sizes[0] / len(ok), "group_sizes": sizes}


def select_majority(samples: list) -> tuple:
    """Largest group of identical normalisations; ties to the median length.

    Returns (index into samples, stats). Deterministic: groups keep first-seen
    order, the key is total, and the final fallback is the lowest sample index,
    so the same sample list always yields the same choice regardless of the
    order the threads happened to finish in.

    Also usable on a SUBSET of the samples -- the feature arms hand it their
    survivors -- because each sample carries its own original index.
    """
    ok = _voting(samples)
    if not ok:
        raise AllSamplesFailed("no sample produced extractable code")
    med = statistics.median([len(s["norm"]) for s in ok])
    best = max(_groups(ok).values(),
               key=lambda g: (len(g),
                              -abs(len(g[0]["norm"]) - med),
                              -g[0]["i"]))
    return best[0]["i"], _group_stats(ok)


def select_longest(samples: list) -> tuple:
    """The longest extracted code. The floor, on purpose.

    No notion of agreement at all, so the n3_longest arm isolates "more samples
    plus any selector" from "more samples plus THIS selector". If voting cannot
    beat this, the selector is decoration.
    """
    ok = _voting(samples)
    if not ok:
        raise AllSamplesFailed("no sample produced extractable code")
    best = max(ok, key=lambda s: (len(s["code"]), -s["i"]))
    return best["i"], _group_stats(ok)


# ------------------------------------------------------- discrete features

# PLACEHOLDER. These five questions exist to test the MECHANISM, not because
# there is evidence they are the right questions. They were chosen for three
# properties: answerable from the code alone with no execution, yes/no rather
# than a rating, and aimed at Python performance deltas that are large and
# mechanical -- `sys.stdin.readline` against `input()`, `join` against
# print-per-line, hashing against a linear scan -- which is the same reason
# bench/recipe_oracle.py picked Python in the first place.
#
# None of them is trusted until it is validated by MEASURED DISCRIMINATION:
# build pairs of programs where the feature is true by construction and false
# by construction, ask Laya both, and keep only the questions that separate,
# recording the margin at which they do. That is a cheap retrieval-layer
# experiment (PROTOCOL rule 5) and it has not been run. Until it has, a result
# from these arms is a result about an unvalidated question set.
#
# Order is priority order for the selection tree and is part of the rule.
FEATURE_QUESTIONS = {
    "deep_recursion": {
        "instructions": "Does this program use recursion deep enough to need "
                        "sys.setrecursionlimit, without setting it?",
        "criteria": {"yes": "it recurses over the input without raising the limit",
                     "no": "it is iterative, or it raises the recursion limit"},
        # First and a veto: RecursionError is a hard wrong answer, not a
        # slowdown, so it is the one feature that can eliminate a candidate
        # outright rather than merely rank it.
        "good": "no", "veto": True,
    },
    "stdin_fast": {
        "instructions": "Does this program read its input through sys.stdin "
                        "rather than the builtin input()?",
        "criteria": {"yes": "it reads via sys.stdin",
                     "no": "it calls input(), or reads no input at all"},
        "good": "yes", "veto": False,
    },
    "joined_output": {
        "instructions": "Does this program emit its output in one write or "
                        "join, rather than calling print once per line?",
        "criteria": {"yes": "output is joined or written once",
                     "no": "it prints once per result line"},
        "good": "yes", "veto": False,
    },
    "nested_loop": {
        "instructions": "Does this program contain a loop nested inside "
                        "another loop that both range over the main input?",
        "criteria": {"yes": "there is a nested loop over the input",
                     "no": "loops over the input are not nested"},
        "good": "no", "veto": False,
    },
    "hash_membership": {
        "instructions": "Does this program test membership with a set or dict "
                        "rather than scanning a list?",
        "criteria": {"yes": "membership goes through a set or dict",
                     "no": "it scans a list, or tests no membership"},
        "good": "yes", "veto": False,
    },
}


def laya_decide(payload: dict, timeout: int = 30) -> dict:
    """POST one /decide call. Injectable so the dry run never touches a GPU."""
    _check_payload(payload)
    req = urllib.request.Request(LAYA_URL + "/decide",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _check_payload(payload: dict) -> None:
    """Refuse to send a call that repeats the measured positional artefact.

    The invariant is structural: the candidate lives in `state`, the options in
    `criteria` are fixed strings from FEATURE_QUESTIONS, and nothing from the
    candidate may appear in the options. Checked on every call rather than
    asserted in a comment, because the malformed version returned confident
    numbers -- option "a" eight times at margins up to 0.919 -- and confident
    numbers are exactly what nobody goes back to check.
    """
    state = payload.get("state") or ""
    qs = payload.get("questions") or {}
    if not state.strip():
        raise FeatureProtocolError("empty state: nothing to judge")
    if not qs:
        raise FeatureProtocolError("no questions")
    # Any 40-character window of the candidate showing up in an option means
    # the candidate has leaked into the criteria.
    windows = {state[i:i + 40] for i in range(0, max(1, len(state) - 40), 20)}
    for qid, q in qs.items():
        crit = q.get("criteria") or {}
        if len(crit) < 2:
            raise FeatureProtocolError(f"{qid}: a choice needs >= 2 options")
        for k, v in crit.items():
            blob = f"{k} {v}"
            if any(w in blob for w in windows if len(w) == 40):
                raise FeatureProtocolError(
                    f"{qid}: candidate text appears in criteria -- this is the "
                    f"configuration that produced the positional artefact")


def extract_features(code: str, questions: dict = None,
                     laya_call=laya_decide, timeout: int = 30) -> dict:
    """{question id -> Feature(label, margin, probabilities, orders)} for ONE candidate.

    One state -- this candidate's code, FIRST, because the tail of a long state
    is silently dropped -- and every question in the same payload, which is one
    forward pass for all of them. The payload is sent twice with the option
    order reversed and the probabilities averaged, because the model is not
    permutation invariant and the unaveraged margin is not an honest margin.

    Nothing here compares candidates. That happens later, over labels.
    """
    questions = questions or FEATURE_QUESTIONS
    state = (code or "")[:LAYA_STATE_CHARS]
    keys = list(questions)
    acc = {q: {} for q in keys}
    orders: dict = {q: [] for q in keys}
    for rev in (False, True):
        body = {"state": state, "questions": {}}
        for q in keys:
            crit = questions[q]["criteria"]
            names = list(crit)
            if rev:
                names = list(reversed(names))
            body["questions"][q] = {"type": "choice",
                                    "instructions": questions[q]["instructions"],
                                    "criteria": {n: crit[n] for n in names}}
        d = laya_call(body, timeout) if _takes_timeout(laya_call) else laya_call(body)
        answers = (d or {}).get("answers") or {}
        for q in keys:
            probs = (answers.get(q) or {}).get("probabilities") or {}
            orders[q].append({"order": list(body["questions"][q]["criteria"]),
                              "probabilities": {k: round(float(v), 4)
                                                for k, v in probs.items()}})
            for k in questions[q]["criteria"]:
                acc[q][k] = acc[q].get(k, 0.0) + float(probs.get(k, 0.0)) / 2.0
    out = {}
    for q in keys:
        ranked = sorted(acc[q].items(), key=lambda x: (-x[1], x[0]))
        margin = round(ranked[0][1] - ranked[1][1], 4) if len(ranked) > 1 else 0.0
        out[q] = Feature(label=ranked[0][0], margin=margin,
                         probabilities={k: round(v, 4) for k, v in acc[q].items()},
                         orders=orders[q])
    return out


def _takes_timeout(fn) -> bool:
    """Mocks in the dry run take (payload) only; the real caller takes a timeout."""
    try:
        return len(inspect.signature(fn).parameters) > 1
    except (TypeError, ValueError):
        return False


def gate(features: dict, threshold: float) -> dict:
    """Labels, with anything the model did not actually decide turned UNKNOWN.

    A margin below the threshold is the model failing to separate the options,
    and recording that as a label is inventing a fact. threshold=0.0 keeps
    every label, which is what the ungated arm uses -- so the two arms differ
    in this one number and nothing else.
    """
    return {q: (f.label if f.margin >= threshold else UNKNOWN)
            for q, f in features.items()}


def _rank(label: str, good: str) -> int:
    # good beats unknown beats bad. UNKNOWN in the middle because no evidence
    # of a flaw should not lose to evidence of one, nor beat a verified good.
    return 2 if label == good else 1 if label == UNKNOWN else 0


def select_by_features(samples: list, questions: dict = None,
                       threshold: float = 0.0, laya_call=laya_decide,
                       timeout: int = 30) -> tuple:
    """Veto, then lexicographic priority over labels, then majority vote.

    The full rule is in the module docstring. Every branch that did not run as
    advertised -- Laya down, every candidate vetoed -- is reported in `notes`
    and flagged `degraded`, because an arm that quietly became a different arm
    is worse than one that failed.
    """
    questions = questions or FEATURE_QUESTIONS
    ok = _voting(samples)
    if not ok:
        raise AllSamplesFailed("no sample produced extractable code")
    stats = _group_stats(ok)
    stats["threshold"] = threshold

    feats: dict = {}
    try:
        for s in ok:
            # Per candidate, independently. Two candidates never share a call,
            # so a score never has to mean the same thing in two states.
            feats[s["i"]] = extract_features(s["code"], questions, laya_call,
                                             timeout)
    except Exception as e:                                       # noqa: BLE001
        # Degrade to the vote, loudly. A partial feature matrix would mean
        # comparing candidates on different questions, which is worse than not
        # comparing them at all -- and returning nothing would turn a service
        # outage into an empty answer.
        idx, s2 = select_majority(samples)
        stats.update(s2)
        stats["features"] = {i: {q: f._asdict() for q, f in v.items()}
                             for i, v in feats.items()}
        stats["degraded"] = True
        stats["notes"] = (f"laya unavailable ({type(e).__name__}: {e}); "
                          f"fell back to majority vote")
        return idx, stats

    labels = {s["i"]: gate(feats[s["i"]], threshold) for s in ok}
    notes = []

    vetoes = [q for q, spec in questions.items() if spec.get("veto")]
    survivors = [s for s in ok
                 if not any(labels[s["i"]][q] not in (questions[q]["good"], UNKNOWN)
                            for q in vetoes)]
    if not survivors:
        survivors = ok
        notes.append("every candidate vetoed; veto ignored")

    for q in questions:
        if len(survivors) == 1:
            break
        good = questions[q]["good"]
        best = max(_rank(labels[s["i"]][q], good) for s in survivors)
        survivors = [s for s in survivors if _rank(labels[s["i"]][q], good) == best]

    # Residual tie -> the vote. With an all-UNKNOWN matrix nothing is ever
    # filtered and this arm is exactly fanout_n3_vote, which is the honest
    # degenerate case rather than a coin flip.
    idx, _ = select_majority(survivors)
    if len(survivors) == len(ok):
        notes.append("features separated nothing; decided by majority vote")
    stats["features"] = {i: {q: f._asdict() for q, f in v.items()}
                         for i, v in feats.items()}
    stats["labels"] = labels
    stats["survivors"] = [s["i"] for s in survivors]
    stats["notes"] = "; ".join(notes)
    stats["degraded"] = False
    return idx, stats


# ---------------------------------------------------------------- generation

def model_generate(prompt: str, temperature: float, i: int,
                   max_tokens: int = 8000, timeout: int = 900) -> tuple:
    """One completion from the model server.

    This does NOT call lcb.generate, and the reason is narrow: lcb.generate
    pins `temperature` at 0.2 in its body, and temperature is the independent
    variable these arms exist to separate. Everything else about it is kept
    identical on purpose -- same (text, meta) contract, same key source, same
    reasoning_content fallback for the empty-content failure this repo already
    documents -- so a number produced here is comparable with one produced by
    bench/livecodebench.py. lcb.generate is still used verbatim for the
    aliveness probe in main(), which is where its fixed temperature is fine.
    """
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Content-Type": "application/json"}
    key = lcb.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    req = urllib.request.Request(f"{MODEL_URL}/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip() and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    return text, {"ms": round((time.time() - t0) * 1000),
                  "finish": (d["choices"][0].get("finish_reason") or ""),
                  "empty_content": not (msg.get("content") or "").strip(),
                  "sample": i}


def default_seeds(n: int) -> list:
    """n distinct concept words, drawn from embedding space by concept_seed.

    Imported lazily. concept_seed pulls in numpy and builds or opens a sqlite
    cache of embeddings, and the dry run must not need either -- the whole
    point of the mock is that the selection logic is testable with nothing
    running.
    """
    sys.path.insert(0, os.path.join(BENCH, "..", "mcp"))
    import concept_seed  # noqa: PLC0415
    return concept_seed.draw(n)


def seed_phrase(word: str) -> str:
    """concept_seed.phrase if it is importable, else the same text inline.

    The wording is the seed's whole mechanism -- concept_seed documents that a
    seed in the SYSTEM message gets ignored and one in the USER turn does not
    -- so the dry run checks the real string, not a placeholder.
    """
    try:
        sys.path.insert(0, os.path.join(BENCH, "..", "mcp"))
        import concept_seed  # noqa: PLC0415
        return concept_seed.phrase(word)
    except Exception:                                            # noqa: BLE001
        return f"\n\nInspiration word: {word}"


def _sample_all(prompt: str, gen, n: int, temperature: float,
                seeds: list | None) -> list:
    """n concurrent samples, returned in SUBMISSION order.

    Order is submission order, not completion order, because every selector
    tie-break ends at "lowest sample index" and that guarantee is worthless if
    the indices are assigned by whichever thread finished first.

    Concurrency is the reason fan-out is affordable at all: fanout.py measures
    N=4 at 14.9s against N=1 at 5.5s on this box, 2.7x not 4x, because batching
    recovers bandwidth a single stream leaves idle.
    """
    prompts = []
    for i in range(n):
        prompts.append(prompt + seed_phrase(seeds[i]) if seeds else prompt)
    out: list = []
    with cf.ThreadPoolExecutor(max_workers=max(1, n)) as ex:
        futs = [ex.submit(gen, p, temperature, i) for i, p in enumerate(prompts)]
        for i, f in enumerate(futs):
            rec = {"i": i, "prompt": prompts[i], "temperature": temperature,
                   "text": "", "code": "", "norm": "", "error": "", "meta": {}}
            try:
                r = f.result()
            except Exception as e:                               # noqa: BLE001
                # Recorded, not swallowed. An arm with one dead sample still
                # answers from the rest; an arm with all of them dead raises.
                rec["error"] = f"{type(e).__name__}: {e}"
                out.append(rec)
                continue
            text, meta = r if isinstance(r, tuple) else (r, {})
            rec["text"] = text or ""
            rec["meta"] = meta
            rec["code"] = lcb.extract_code(rec["text"])
            rec["norm"] = normalise(rec["code"])
            out.append(rec)
    return out


def _arm(name: str, prompt: str, gen, n: int, temperature: float,
         selector, seeded: bool = False, seed_source=default_seeds) -> ArmResult:
    t0 = time.time()
    seeds = None
    if seeded:
        try:
            seeds = seed_source(n)
        except Exception as e:                                   # noqa: BLE001
            # A dead concept index must not silently turn the seeded arm into
            # plain fan-out -- that would be a fallback hiding an outage
            # (PROTOCOL rule 2), and the arm would then be measuring something
            # other than what its name says.
            raise AllSamplesFailed(f"concept seeds unavailable: "
                                   f"{type(e).__name__}: {e}") from e
        if len(set(seeds)) < n:
            raise AllSamplesFailed(f"concept_seed returned {len(set(seeds))} "
                                   f"distinct words for n={n}")
    samples = _sample_all(prompt, gen, n, temperature, seeds)
    chosen, stats = selector(samples)          # raises AllSamplesFailed if none
    s = samples[chosen]
    return ArmResult(
        name=name, answer=s["text"], code=s["code"], chosen=chosen,
        samples=samples, n_requested=n, n_voting=len(_voting(samples)),
        n_errors=sum(1 for x in samples if x["error"]),
        n_nocode=sum(1 for x in samples if not x["error"] and not x["norm"]),
        agreement=round(stats["agreement"], 3),
        group_sizes=stats["group_sizes"],
        selector=getattr(selector, "__name__", str(selector)),
        ms=round((time.time() - t0) * 1000),
        features=stats.get("features", {}),
        notes=stats.get("notes", ""),
        degraded=bool(stats.get("degraded")))


# --------------------------------------------------------------------- arms

def arm_single_greedy(prompt, gen=model_generate, **kw) -> ArmResult:
    """One sample at T=0.0. The control everything else is measured against."""
    return _arm("single_greedy", prompt, gen, 1, 0.0, select_majority, **kw)


def arm_single_temp(prompt, gen=model_generate, **kw) -> ArmResult:
    """One sample at the fan-out temperature.

    Without this arm, any win by fanout_n3_vote is confounded with temperature.
    """
    return _arm("single_temp", prompt, gen, 1, FANOUT_TEMPERATURE,
                select_majority, **kw)


def arm_fanout_n3_vote(prompt, gen=model_generate, **kw) -> ArmResult:
    """Three samples, self-consistency over normalised code."""
    return _arm("fanout_n3_vote", prompt, gen, 3, FANOUT_TEMPERATURE,
                select_majority, **kw)


def arm_fanout_n5_vote(prompt, gen=model_generate, **kw) -> ArmResult:
    """Five samples. Whether n past 3 buys anything is the question.

    Published self-consistency curves flatten well before 16 samples, and
    fanout.py's own throughput table shows the batch saturating at N=8
    (32.5s, 49.2 tok/s, below N=4's 53.8), so 5 is where the cost starts
    being felt without the batch collapsing.
    """
    return _arm("fanout_n5_vote", prompt, gen, 5, FANOUT_TEMPERATURE,
                select_majority, **kw)


def arm_fanout_n3_seeded(prompt, gen=model_generate, **kw) -> ArmResult:
    """Three samples, each with a DIFFERENT concept seed word appended.

    concept_seed's own docstring is honest that the evidence for this is
    qualitative -- ten jokes becoming ten different jokes -- with no diversity
    metric and no significance test. Comparing it against fanout_n3_vote, which
    differs from it in nothing but the seeds, is what turns that into a number.
    """
    return _arm("fanout_n3_seeded", prompt, gen, 3, FANOUT_TEMPERATURE,
                select_majority, seeded=True, **kw)


def arm_fanout_n3_longest(prompt, gen=model_generate, **kw) -> ArmResult:
    """Three samples, longest code wins. The floor for the selector.

    Kept alongside the feature arms on purpose. If a feature tree cannot beat
    "pick the longest", the tree is not earning the two extra decision calls
    per candidate, and that comparison is the whole point of having it.
    """
    return _arm("fanout_n3_longest", prompt, gen, 3, FANOUT_TEMPERATURE,
                select_longest, **kw)


def _feature_selector(threshold: float, questions=None, laya_call=laya_decide,
                      timeout: int = 30, name: str = "select_by_features"):
    def sel(samples: list) -> tuple:
        return select_by_features(samples, questions, threshold, laya_call,
                                  timeout)
    sel.__name__ = name
    return sel


def arm_fanout_n3_features(prompt, gen=model_generate, questions=None,
                           laya_call=laya_decide, threshold: float = 0.0,
                           laya_timeout: int = 30, **kw) -> ArmResult:
    """Three samples, chosen by the feature tree over discrete Laya labels.

    threshold 0.0: every label is taken at face value, however thin the margin.
    That is the point of having it next to the gated arm -- the two differ in
    one number, so whatever the gate is worth shows up as the difference.
    """
    return _arm("fanout_n3_features", prompt, gen, 3, FANOUT_TEMPERATURE,
                _feature_selector(threshold, questions, laya_call, laya_timeout),
                **kw)


def arm_fanout_n5_features(prompt, gen=model_generate, questions=None,
                           laya_call=laya_decide, threshold: float = 0.0,
                           laya_timeout: int = 30, **kw) -> ArmResult:
    """Five samples, same tree. Costs 10 decision calls per problem.

    Two calls per candidate is the permutation average and is not optional, so
    feature selection is linear in n at a steeper slope than generation, which
    batches. Whether the fifth candidate pays for its two calls is exactly the
    kind of question this arm exists to answer.
    """
    return _arm("fanout_n5_features", prompt, gen, 5, FANOUT_TEMPERATURE,
                _feature_selector(threshold, questions, laya_call, laya_timeout),
                **kw)


def arm_fanout_n3_features_gated(prompt, gen=model_generate, questions=None,
                                 laya_call=laya_decide,
                                 threshold: float = MARGIN_GATE,
                                 laya_timeout: int = 30, **kw) -> ArmResult:
    """Three samples, feature tree, low-margin answers demoted to UNKNOWN.

    The gate can only ever move a candidate from good or bad to the middle
    rank, so the arm degrades toward fanout_n3_vote as the margins get worse.
    That is the intended shape: a model that cannot separate the options should
    hand the decision back to agreement, not guess with authority.
    """
    return _arm("fanout_n3_features_gated", prompt, gen, 3, FANOUT_TEMPERATURE,
                _feature_selector(threshold, questions, laya_call, laya_timeout),
                **kw)


DETAILED = {
    "single_greedy": arm_single_greedy,
    "single_temp": arm_single_temp,
    "fanout_n3_vote": arm_fanout_n3_vote,
    "fanout_n5_vote": arm_fanout_n5_vote,
    "fanout_n3_seeded": arm_fanout_n3_seeded,
    "fanout_n3_longest": arm_fanout_n3_longest,
    "fanout_n3_features": arm_fanout_n3_features,
    "fanout_n5_features": arm_fanout_n5_features,
    "fanout_n3_features_gated": arm_fanout_n3_features_gated,
}


def _answer_only(fn):
    def call(prompt: str, gen=model_generate, **kw) -> str:
        return fn(prompt, gen, **kw).answer
    call.__name__ = fn.__name__
    call.__doc__ = fn.__doc__
    return call


# The arms as the task defines them: prompt in, one final answer string out.
# DETAILED is the same computation with the sample-level record kept, which is
# what the scoring runner writes to jsonl -- an arm's error count is not
# optional bookkeeping, it decides whether its score means anything.
ARMS = {k: _answer_only(v) for k, v in DETAILED.items()}


# ----------------------------------------------------------------- scoring

def score_row(code: str, row: dict, max_cases: int, timeout: float) -> tuple:
    """Execute the chosen code against the problem's tests. lcb does the work."""
    return lcb.run_tests(code, row, lcb.all_cases(row, max_cases), timeout)


def report(path: str) -> None:
    """Paired report across arms, baseline first.

    States the power before the verdict (PROTOCOL rule 4) and counts errors
    separately from failures (rule 3): an arm whose samples did not run has not
    been measured, and its problems are dropped from the paired set rather than
    scored as losses.
    """
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except Exception:                                        # noqa: BLE001
            continue
    by: dict = {}
    for r in rows:
        by.setdefault(r["question_id"], {})[r["arm"]] = r
    arms = [a for a in ARMS if any(a in v for v in by.values())]
    paired = {q: v for q, v in by.items()
              if len(v) == len(arms) and not any(x.get("error") for x in v.values())}
    errs = [r for r in rows if r.get("error")]
    if errs:
        kinds: dict = {}
        for r in errs:
            k = (r["arm"], r["error"].split(":")[0])
            kinds[k] = kinds.get(k, 0) + 1
        print(f"\n  {len(errs)} arm errors, excluded from scoring:")
        for (a, k), v in sorted(kinds.items(), key=lambda x: -x[1]):
            print(f"    {a:<18} {k:<24} {v}")

    n = len(paired)
    print(f"\n{'=' * 74}\n  FAN-OUT ARMS  --  {n} problems complete under all "
          f"{len(arms)} arms\n{'=' * 74}")
    if not n:
        print("  nothing paired yet")
        return
    print(f"  {'arm':<26}{'pass@1':<24}{'mean s':<9}{'agree':<8}{'err':<6}")
    print("  " + "-" * 72)
    for a in arms:
        k = sum(1 for v in paired.values() if v[a]["passed"])
        lo, hi = lcb.wilson(k, n)
        secs = sum(v[a]["ms"] for v in paired.values()) / n / 1000
        agr = sum(v[a].get("agreement", 0.0) for v in paired.values()) / n
        err = sum(v[a].get("n_errors", 0) for v in paired.values())
        print(f"  {a:<26}{k}/{n} = {k / n:5.1%} [{lo:.0%}-{hi:.0%}]".ljust(52)
              + f"{secs:<9.1f}{agr:<8.2f}{err:<6}")

    # An arm that degraded is not the arm its name claims. Reported before any
    # comparison, because a feature arm that fell back to the vote on half the
    # problems is partly a copy of the control and its McNemar cell is noise.
    deg = {a: sum(1 for v in paired.values() if v[a].get("degraded")) for a in arms}
    if any(deg.values()):
        print("\n  arms that degraded (fell back to the vote) on some problems:")
        for a, k in deg.items():
            if k:
                print(f"    {a:<26} {k}/{n}"
                      + ("   -- this arm was not measured" if k > n // 10 else ""))

    base = arms[0]
    for a in arms[1:]:
        b = sum(1 for v in paired.values() if v[a]["passed"] and not v[base]["passed"])
        c = sum(1 for v in paired.values() if v[base]["passed"] and not v[a]["passed"])
        p = lcb.mcnemar(b, c)
        verdict = (f"{a} better" if b > c and p < 0.05 else
                   f"{base} better" if c > b and p < 0.05 else
                   "no detectable difference")
        print(f"\n  McNemar {base} vs {a}: {a} solved {b} that {base} missed, "
              f"{base} solved {c} that {a} missed")
        print(f"    p = {p:.4f}  ->  {verdict}")
        if b + c < 10:
            print(f"    only {b + c} discordant pairs: this cannot detect "
                  f"anything but a large effect. Not a result yet.")


# ----------------------------------------------------------------- dry run

GOOD_A = ("Here is the solution.\n\n```python\n"
          "def solve(n):\n    return n + 1\n```\n")
# Same program, different layout and comments. Must land in the same group --
# this is the entire reason voting normalises instead of comparing strings.
GOOD_A_COMMENTED = ("```python\n# adds one to n\n\ndef solve(n):\n\n"
                    "\treturn n + 1    # off by one on purpose? no\n```\n")
# A different program, and DELIBERATELY the longest, so that "majority" and
# "longest" disagree and the two selectors are distinguishable.
GOOD_B = ("```python\n"
          "def solve(n):\n"
          "    total = 0\n"
          "    for i in range(n):\n"
          "        total += i * i\n"
          "    for j in range(n):\n"
          "        total -= j\n"
          "    return total\n```\n")
# Strictly between A and B once normalised, and NOT first in the sample list,
# so that "closest to the median length" cannot be satisfied by accident by a
# selector that just returns the first candidate. A mutation run caught exactly
# that: with an earlier fixture, A and C normalised to the same length and the
# determinism assertion passed against a deliberately broken selector.
GOOD_C = ("```python\ndef solve(n):\n    acc = n * 2\n    acc += 0\n"
          "    return acc\n    # padding comment, dropped by normalise\n```\n")
APOLOGY = "I'm sorry, I can't determine the answer to this problem."


def canned(*replies):
    """A mock generator. An Exception in the list is RAISED by that sample.

    Every prompt it sees is recorded, which is how the seeded arm is checked:
    the seeds have to reach the request, and the only way to know that is to
    look at what was sent.
    """
    seen: list = []

    def gen(prompt: str, temperature: float, i: int):
        seen.append({"i": i, "prompt": prompt, "temperature": temperature})
        r = replies[i % len(replies)]
        if isinstance(r, BaseException):
            raise r
        return r, {"ms": 0, "sample": i}

    gen.seen = seen
    return gen


def fake_laya(p_yes: dict, bias: float = 0.3, fail: BaseException = None):
    """A mock decision service with a DELIBERATE position bias.

    `p_yes` maps a marker substring of a candidate's code to {question: p}. The
    returned probabilities add `bias` to whichever option is listed FIRST, which
    is the measured failure mode being defended against -- without permutation
    averaging a 50/50 question comes back as a confident label, and with it the
    bias cancels exactly. A mock that answered symmetrically would let a
    harness that forgot to average still pass.

    Every mock call goes through the real `_check_payload`, so the protocol
    guard is exercised by the dry run and not only by production traffic.
    """
    calls: list = []

    def call(payload: dict):
        calls.append(payload)
        if fail is not None:
            raise fail
        _check_payload(payload)
        state = payload["state"]
        key = next((m for m in p_yes if m in state), None)
        if key is None:
            raise KeyError("mock laya was handed an unknown candidate")
        answers = {}
        for qid, q in payload["questions"].items():
            names = list(q["criteria"])
            p = p_yes[key].get(qid, 0.5)
            base = {"yes": p, "no": 1.0 - p}
            raw = {n: base[n] + (bias if n == names[0] else 0.0) for n in names}
            tot = sum(raw.values())
            answers[qid] = {"choice": max(raw, key=lambda k: raw[k]),
                            "probabilities": {n: raw[n] / tot for n in names}}
        return {"answers": answers, "elapsed_ms": 0}

    call.calls = calls
    return call


# Markers picked out of the fixtures above, one per distinct program, so the
# mock can tell which candidate it is being asked about.
MARK_A, MARK_B, MARK_C = "return n + 1", "total -= j", "n * 2"


def _flat(specs: dict) -> dict:
    """The same per-question probabilities for every candidate."""
    return {m: dict(specs) for m in (MARK_A, MARK_B, MARK_C)}


def dry_run() -> int:
    """Every selector, validated with no GPU, no server and no index.

    Returns the number of failed assertions. The checks are collected rather
    than raised one at a time so that a single early failure does not hide the
    state of the other seven -- a run that stops at the first problem reports
    less than it knows.
    """
    results: list = []

    def check(name: str, fn) -> None:
        try:
            fn()
            results.append((name, True, ""))
        except AssertionError as e:
            results.append((name, False, str(e) or "assertion failed"))
        except Exception as e:                                   # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {e}"))

    stub_seeds = lambda n: ["harbour", "obsidian", "lantern", "kiln", "fjord"][:n]  # noqa: E731

    def t1_all_arms_return_str() -> None:
        for name, arm in ARMS.items():
            gen = canned(GOOD_A, GOOD_A_COMMENTED, GOOD_B, GOOD_A, GOOD_B)
            kw = {}
            if "seeded" in name:
                kw["seed_source"] = stub_seeds
            if "features" in name:
                kw["laya_call"] = fake_laya(_flat({"stdin_fast": 0.9}))
            a = arm("problem", gen, **kw)
            assert isinstance(a, str), f"{name} returned {type(a).__name__}"
            assert a.strip(), f"{name} returned an empty string"

    def t2_vote_picks_majority() -> None:
        # A, A, B -> the A family is 2 of 3.
        gen = canned(GOOD_A, GOOD_A, GOOD_B)
        r = arm_fanout_n3_vote("problem", gen)
        assert r.code == lcb.extract_code(GOOD_A), f"chose {r.code!r}"
        assert r.agreement == round(2 / 3, 3), r.agreement
        assert r.group_sizes == [2, 1], r.group_sizes

    def t3_majority_beats_longest() -> None:
        # Same three samples; the two selectors must disagree, or n3_longest is
        # not a floor, it is a duplicate arm.
        gen_v = canned(GOOD_A, GOOD_A, GOOD_B)
        gen_l = canned(GOOD_A, GOOD_A, GOOD_B)
        vote = arm_fanout_n3_vote("problem", gen_v).code
        longest = arm_fanout_n3_longest("problem", gen_l).code
        assert vote == lcb.extract_code(GOOD_A), vote
        assert longest == lcb.extract_code(GOOD_B), longest
        assert vote != longest, "selectors are indistinguishable"

    def t4_normalisation_groups_comment_variants() -> None:
        # The A-with-comments sample must JOIN the A group, not form its own.
        assert normalise(lcb.extract_code(GOOD_A)) == \
            normalise(lcb.extract_code(GOOD_A_COMMENTED)), "comments split the group"
        gen = canned(GOOD_A, GOOD_A_COMMENTED, GOOD_B)
        r = arm_fanout_n3_vote("problem", gen)
        assert r.group_sizes == [2, 1], r.group_sizes
        assert r.chosen == 0, r.chosen

    def t5_seeded_prompts_differ() -> None:
        gen = canned(GOOD_A, GOOD_B, GOOD_C)
        arm_fanout_n3_seeded("problem", gen, seed_source=stub_seeds)
        prompts = [s["prompt"] for s in gen.seen]
        assert len(set(prompts)) == 3, f"{len(set(prompts))} distinct prompts"
        for p, w in zip(prompts, stub_seeds(3)):
            assert p.startswith("problem"), "base prompt was lost"
            assert w in p, f"seed {w!r} never reached the request"
        # And the unseeded arm of the same size must NOT vary its prompts, or
        # the comparison between the two arms is not about seeds.
        g2 = canned(GOOD_A, GOOD_B, GOOD_C)
        arm_fanout_n3_vote("problem", g2)
        assert len({s["prompt"] for s in g2.seen}) == 1, "unseeded arm varied"

    def t6_tie_break_is_deterministic_median() -> None:
        # Three singleton groups: no majority exists, so the tie-break decides.
        # Lengths are short / long / medium, and the median one must win from
        # any position, on every run.
        norms = [normalise(lcb.extract_code(x)) for x in (GOOD_A, GOOD_B, GOOD_C)]
        assert len(set(norms)) == 3, "fixtures are not three distinct programs"
        med = statistics.median([len(x) for x in norms])
        want_i = min(range(3), key=lambda i: abs(len(norms[i]) - med))
        assert want_i != 0, "fixture is degenerate: the median is also the first"
        picks = set()
        for _ in range(5):
            gen = canned(GOOD_A, GOOD_B, GOOD_C)
            r = arm_fanout_n3_vote("problem", gen)
            picks.add(r.chosen)
            assert r.group_sizes == [1, 1, 1], r.group_sizes
        assert picks == {want_i}, f"picked {picks}, median candidate is {want_i}"

    def t7_exception_is_not_an_empty_answer() -> None:
        # One sample explodes; the other two agree. The arm answers from them
        # and REPORTS the error -- it does not return "" and it does not let
        # the dead sample vote.
        gen = canned(RuntimeError("upstream reset"), GOOD_A, GOOD_A)
        r = arm_fanout_n3_vote("problem", gen)
        assert r.code == lcb.extract_code(GOOD_A), r.code
        assert r.answer.strip(), "returned an empty answer"
        assert r.n_errors == 1, r.n_errors
        assert r.n_voting == 2, r.n_voting
        assert r.chosen != 0, "the failed sample was chosen"
        assert round(r.error_rate, 3) == round(1 / 3, 3), r.error_rate

    def t8_total_failure_raises() -> None:
        gen = canned(RuntimeError("a"), TimeoutError("b"), RuntimeError("c"))
        try:
            arm_fanout_n3_vote("problem", gen)
        except AllSamplesFailed:
            pass
        else:
            raise AssertionError("returned an answer with every sample dead")
        # Same rule for a single-sample arm: a dead control is an error, not a
        # 0-scoring answer.
        try:
            arm_single_greedy("problem", canned(RuntimeError("x")))
        except AllSamplesFailed:
            return
        raise AssertionError("single_greedy swallowed its only failure")

    def t9_empty_replies_do_not_outvote_code() -> None:
        # Two apologies and one program. If empties could vote they would be
        # the majority, which is the 0/8 harness bug in a new costume.
        gen = canned(APOLOGY, APOLOGY, GOOD_B)
        r = arm_fanout_n3_vote("problem", gen)
        assert r.code == lcb.extract_code(GOOD_B), r.code
        assert r.n_nocode == 2, r.n_nocode
        assert r.n_voting == 1, r.n_voting
        assert r.n_errors == 0, "an apology is not an error"

    def t10_controls_use_the_right_temperature_and_n() -> None:
        g0 = canned(GOOD_A)
        arm_single_greedy("problem", g0)
        assert len(g0.seen) == 1, f"single_greedy drew {len(g0.seen)}"
        assert g0.seen[0]["temperature"] == 0.0, g0.seen[0]["temperature"]
        g1 = canned(GOOD_A)
        arm_single_temp("problem", g1)
        assert len(g1.seen) == 1, f"single_temp drew {len(g1.seen)}"
        assert g1.seen[0]["temperature"] == FANOUT_TEMPERATURE, g1.seen[0]
        g5 = canned(GOOD_A, GOOD_B, GOOD_A, GOOD_C, GOOD_A)
        r = arm_fanout_n5_vote("problem", g5)
        assert len(g5.seen) == 5, f"n5 drew {len(g5.seen)}"
        assert {s["temperature"] for s in g5.seen} == {FANOUT_TEMPERATURE}
        assert r.group_sizes == [3, 1, 1], r.group_sizes
        assert r.code == lcb.extract_code(GOOD_A), r.code

    def t11_untokenisable_code_still_votes() -> None:
        # A truncated reply (finish_reason=length) is common and must degrade
        # to a layout-only comparison rather than crash the selector.
        broken = "```python\ndef solve(n):\n    return f(\n```\n"
        assert normalise(lcb.extract_code(broken)), "truncated code normalised away"
        gen = canned(broken, broken, GOOD_A)
        r = arm_fanout_n3_vote("problem", gen)
        assert r.group_sizes == [2, 1], r.group_sizes
        assert r.chosen == 0, r.chosen

    def f1_features_are_per_candidate() -> None:
        laya = fake_laya(_flat({"stdin_fast": 0.9}))
        gen = canned(GOOD_A, GOOD_B, GOOD_C)
        r = arm_fanout_n3_features("problem", gen, laya_call=laya)
        # 3 candidates x 2 option orders. One call per candidate per order,
        # never one call holding two candidates.
        assert len(laya.calls) == 6, f"{len(laya.calls)} decision calls"
        states = [c["state"] for c in laya.calls]
        assert len(set(states)) == 3, f"{len(set(states))} distinct states"
        for st in states:
            hits = [m for m in (MARK_A, MARK_B, MARK_C) if m in st]
            assert len(hits) == 1, f"state holds {len(hits)} candidates"
        for c in laya.calls:
            assert set(c["questions"]) == set(FEATURE_QUESTIONS), \
                "questions were split across calls instead of batched"
        assert len(r.features) == 3, r.features

    def f2_candidate_never_in_criteria() -> None:
        laya = fake_laya(_flat({"stdin_fast": 0.9}))
        arm_fanout_n3_features("problem", canned(GOOD_A, GOOD_B, GOOD_C),
                               laya_call=laya)
        for c in laya.calls:
            st = c["state"]
            wins = {st[i:i + 20] for i in range(0, max(1, len(st) - 20), 10)}
            for qid, q in c["questions"].items():
                blob = " ".join(list(q["criteria"]) + list(q["criteria"].values()))
                assert not any(w in blob for w in wins if len(w) == 20), \
                    f"{qid}: candidate leaked into criteria"
                assert set(q["criteria"]) == {"yes", "no"}, q["criteria"]
        # And the guard must actually fire, or it is decoration.
        bad = {"state": "def solve(n):\n    return n + 1\n" * 4,
               "questions": {"q": {"type": "choice", "instructions": "which",
                                   "criteria": {"a": "def solve(n):\n    return n + 1\n" * 4,
                                                "b": "other"}}}}
        try:
            _check_payload(bad)
        except FeatureProtocolError:
            return
        raise AssertionError("_check_payload accepted a candidate in criteria")

    def f3_permutation_averaging() -> None:
        one = {"undecided": {"instructions": "coin flip",
                             "criteria": {"yes": "y", "no": "n"},
                             "good": "yes", "veto": False}}
        laya = fake_laya({MARK_A: {"undecided": 0.5}})
        f = extract_features(lcb.extract_code(GOOD_A), one, laya)["undecided"]
        assert len(laya.calls) == 2, f"{len(laya.calls)} calls, expected 2 orders"
        orders = [list(c["questions"]["undecided"]["criteria"]) for c in laya.calls]
        assert orders[0] == list(reversed(orders[1])), orders
        # The mock adds 0.3 to whichever option is first. One order alone would
        # have reported a confident label on a 50/50 question; averaged, the
        # positional component cancels exactly.
        single = laya.calls and orders[0][0]
        raw = fake_laya({MARK_A: {"undecided": 0.5}})
        d = raw({"state": lcb.extract_code(GOOD_A),
                 "questions": {"undecided": {"type": "choice",
                                             "instructions": "coin flip",
                                             "criteria": {"yes": "y", "no": "n"}}}})
        ps = sorted(d["answers"]["undecided"]["probabilities"].values(), reverse=True)
        assert ps[0] - ps[1] > 0.2, f"mock has no position bias to cancel: {ps}"
        assert f.margin == 0.0, f"margin {f.margin} survived averaging"
        assert len(f.orders) == 2 and single, "per-order trace not recorded"

    def f4_low_margin_becomes_unknown() -> None:
        one = {"weak": {"instructions": "weak signal",
                        "criteria": {"yes": "y", "no": "n"},
                        "good": "yes", "veto": False}}
        laya = fake_laya({MARK_A: {"weak": 0.55}})
        f = extract_features(lcb.extract_code(GOOD_A), one, laya)
        assert 0 < f["weak"].margin < MARGIN_GATE, f["weak"].margin
        assert gate(f, MARGIN_GATE)["weak"] == UNKNOWN, gate(f, MARGIN_GATE)
        assert gate(f, 0.0)["weak"] == "yes", gate(f, 0.0)
        # A decided feature must survive the same gate, or the gate is just off.
        strong = extract_features(lcb.extract_code(GOOD_A), one,
                                  fake_laya({MARK_A: {"weak": 0.98}}))
        assert strong["weak"].margin >= MARGIN_GATE, strong["weak"].margin
        assert gate(strong, MARGIN_GATE)["weak"] == "yes"

    def f5_all_unknown_degenerates_to_vote() -> None:
        # Every question a coin flip -> every label UNKNOWN -> nothing filters.
        laya = fake_laya(_flat({q: 0.5 for q in FEATURE_QUESTIONS}))
        r = arm_fanout_n3_features_gated("problem", canned(GOOD_A, GOOD_A, GOOD_B),
                                         laya_call=laya)
        vote = arm_fanout_n3_vote("problem", canned(GOOD_A, GOOD_A, GOOD_B))
        assert r.chosen == vote.chosen, f"{r.chosen} vs vote {vote.chosen}"
        assert r.code == lcb.extract_code(GOOD_A), r.code
        assert all(f["margin"] < MARGIN_GATE
                   for v in r.features.values() for f in v.values()), \
            "a coin-flip question produced a confident margin"
        assert "separated nothing" in r.notes, r.notes

    def f6_laya_failure_degrades_to_vote() -> None:
        dead = fake_laya(_flat({"stdin_fast": 0.9}),
                         fail=ConnectionRefusedError("laya is not running"))
        r = arm_fanout_n3_features("problem", canned(GOOD_A, GOOD_A, GOOD_B),
                                   laya_call=dead)
        assert r.answer.strip(), "a dead decision service produced an empty answer"
        assert r.code == lcb.extract_code(GOOD_A), r.code
        assert r.degraded is True, "degradation was not flagged"
        assert "laya unavailable" in r.notes, r.notes

    def f7_veto_can_overrule_the_majority() -> None:
        # A is the majority (2 of 3) but recurses without raising the limit,
        # which is the one feature allowed to eliminate a candidate.
        probs = {MARK_A: {"deep_recursion": 0.97},
                 MARK_B: {"deep_recursion": 0.03},
                 MARK_C: {"deep_recursion": 0.03}}
        for arm in (arm_fanout_n3_features, arm_fanout_n3_features_gated):
            r = arm("problem", canned(GOOD_A, GOOD_A, GOOD_B),
                    laya_call=fake_laya(probs))
            assert r.code == lcb.extract_code(GOOD_B), f"{arm.__name__}: {r.code}"
            assert r.notes == "", r.notes
        # The control on the same samples still picks the majority, so the
        # difference is the tree and not the samples.
        assert arm_fanout_n3_vote("problem", canned(GOOD_A, GOOD_A, GOOD_B)).code \
            == lcb.extract_code(GOOD_A)

    def f8_full_trace_is_recorded() -> None:
        laya = fake_laya(_flat({"stdin_fast": 0.9, "nested_loop": 0.2}))
        r = arm_fanout_n5_features("problem",
                                   canned(GOOD_A, GOOD_B, GOOD_C, GOOD_A, GOOD_B),
                                   laya_call=laya)
        assert len(laya.calls) == 10, f"{len(laya.calls)} calls for n=5"
        assert set(r.features) == {0, 1, 2, 3, 4}, r.features.keys()
        for i, v in r.features.items():
            assert set(v) == set(FEATURE_QUESTIONS), f"sample {i}: {list(v)}"
            for qid, f in v.items():
                assert set(f) >= {"label", "margin", "probabilities", "orders"}, f
                assert len(f["orders"]) == 2, f"{qid}: {len(f['orders'])} orders"
                assert isinstance(f["margin"], float), f
        json.dumps(r.features)      # must survive the jsonl record

    def f10_priority_order_and_unknown_rank() -> None:
        # Nothing is vetoed here, so the LEXICOGRAPHIC step decides. A mutation
        # run showed f7 passing against a _rank that ranked everything equal --
        # the veto path alone never exercises the ordering, so it is tested on
        # its own.
        safe = {"deep_recursion": 0.03}

        def probs(a, b):
            return {MARK_A: {**safe, "stdin_fast": a},
                    MARK_B: {**safe, "stdin_fast": b},
                    MARK_C: {**safe, "stdin_fast": 0.5}}

        # good (B) beats bad (A), against a 2-1 majority for A.
        r = arm_fanout_n3_features_gated(
            "problem", canned(GOOD_A, GOOD_A, GOOD_B),
            laya_call=fake_laya(probs(0.02, 0.98)))
        assert r.code == lcb.extract_code(GOOD_B), f"good lost to bad: {r.code}"
        # good (B) beats UNKNOWN (A): a verified feature outranks no evidence.
        r = arm_fanout_n3_features_gated(
            "problem", canned(GOOD_A, GOOD_A, GOOD_B),
            laya_call=fake_laya(probs(0.5, 0.98)))
        assert r.code == lcb.extract_code(GOOD_B), f"good lost to unknown: {r.code}"
        # UNKNOWN (A) beats bad (B), against a 2-1 majority for B: no evidence
        # of a flaw outranks evidence of one.
        r = arm_fanout_n3_features_gated(
            "problem", canned(GOOD_B, GOOD_B, GOOD_A),
            laya_call=fake_laya(probs(0.5, 0.02)))
        assert r.code == lcb.extract_code(GOOD_A), f"unknown lost to bad: {r.code}"
        # And with the gate off, that same UNKNOWN is a confident-looking label
        # on a coin flip, which is the failure the gate exists to prevent.
        f = extract_features(lcb.extract_code(GOOD_A),
                             {"stdin_fast": FEATURE_QUESTIONS["stdin_fast"]},
                             fake_laya({MARK_A: {"stdin_fast": 0.5}}))
        assert gate(f, 0.0)["stdin_fast"] != UNKNOWN
        assert gate(f, MARGIN_GATE)["stdin_fast"] == UNKNOWN

    def f9_universal_veto_is_ignored_and_reported() -> None:
        # Every candidate vetoed. Dropping them all would leave nothing to
        # answer with, so the veto is ignored -- and said out loud.
        probs = _flat({"deep_recursion": 0.97})
        r = arm_fanout_n3_features("problem", canned(GOOD_A, GOOD_A, GOOD_B),
                                   laya_call=fake_laya(probs))
        assert r.answer.strip(), "a universal veto emptied the candidate set"
        assert "every candidate vetoed" in r.notes, r.notes
        assert r.code == lcb.extract_code(GOOD_A), r.code

    for name, fn in [
            ("all nine arms return a non-empty string", t1_all_arms_return_str),
            ("vote picks the majority program", t2_vote_picks_majority),
            ("majority and longest选 differ on the same samples".replace("选", ""),
             t3_majority_beats_longest),
            ("comment/whitespace variants group together",
             t4_normalisation_groups_comment_variants),
            ("seeded arm sends 3 different prompts", t5_seeded_prompts_differ),
            ("no-majority tie breaks to median, deterministically",
             t6_tie_break_is_deterministic_median),
            ("one sample's exception is not an empty answer",
             t7_exception_is_not_an_empty_answer),
            ("all samples failing raises instead of answering",
             t8_total_failure_raises),
            ("replies with no code cannot outvote code",
             t9_empty_replies_do_not_outvote_code),
            ("controls use the stated temperature and n",
             t10_controls_use_the_right_temperature_and_n),
            ("untokenisable code degrades instead of crashing",
             t11_untokenisable_code_still_votes),
            ("features are extracted per candidate, independently",
             f1_features_are_per_candidate),
            ("no call puts a candidate in state AND criteria",
             f2_candidate_never_in_criteria),
            ("option order is permuted and averaged", f3_permutation_averaging),
            ("a below-threshold margin becomes UNKNOWN",
             f4_low_margin_becomes_unknown),
            ("an all-UNKNOWN vector degenerates to majority vote",
             f5_all_unknown_degenerates_to_vote),
            ("a laya failure degrades to the vote, not to empty",
             f6_laya_failure_degrades_to_vote),
            ("the feature tree can overrule the majority",
             f7_veto_can_overrule_the_majority),
            ("every feature vector and margin is recorded",
             f8_full_trace_is_recorded),
            ("a universal veto is ignored and reported",
             f9_universal_veto_is_ignored_and_reported),
            ("priority order decides, and good > unknown > bad",
             f10_priority_order_and_unknown_rank)]:
        check(name, fn)

    print(f"\n{'=' * 74}\n  DRY RUN -- selection logic only, mock generator, "
          f"no model contacted\n{'=' * 74}")
    bad = 0
    for name, ok, why in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            bad += 1
            print(f"        {why}")
    print(f"\n  {len(results) - bad}/{len(results)} assertions passed")
    if bad:
        print("  The selectors are wrong. Nothing measured with them counts.")
    return bad


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the selectors against a mock generator")
    ap.add_argument("--run", action="store_true", help="real generation")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--data", default=os.path.join(BENCH, "data", "test6.jsonl"))
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--gen-timeout", type=int, default=900)
    ap.add_argument("--test-timeout", type=float, default=12.0)
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=os.path.join(HERE, "fanout_arms.jsonl"))
    args = ap.parse_args()

    if args.report_only:
        report(args.out)
        return
    if args.dry_run or not args.run:
        raise SystemExit(dry_run())

    # Rule 1: prove the endpoint is alive with a call that does the real job,
    # before spending hours of generation on it. lcb.generate unmodified --
    # its fixed temperature does not matter for a liveness probe.
    cond = {"url": MODEL_URL, "model": MODEL}
    try:
        txt, meta = lcb.generate(cond, "Reply with the word ready.", 64, 120)
        print(f"  model server alive, {meta['ms']}ms, {txt.strip()[:40]!r}")
    except Exception as e:                                       # noqa: BLE001
        raise SystemExit(f"\n  {MODEL_URL} is not answering: "
                         f"{type(e).__name__}: {e}\n  Refusing to run. A dead "
                         f"endpoint scores 0% and reads as a result.")
    # And prove the concept index is alive before the seeded arm depends on it,
    # for the same reason -- an empty draw would quietly make that arm a
    # duplicate of fanout_n3_vote.
    if "features" in args.arms:
        # Same rule for the decision service: a 404 or a refused connection
        # would make every feature arm silently degrade into fanout_n3_vote,
        # and the run would report three arms that were all the same arm.
        probe = extract_features("import sys\nfor l in sys.stdin: print(l)\n")
        ok = [q for q, f in probe.items() if f.margin >= MARGIN_GATE]
        print(f"  laya alive, {len(ok)}/{len(probe)} starter questions "
              f"separated on the probe: "
              + ", ".join(f"{q}={probe[q].label}@{probe[q].margin:.2f}"
                          for q in probe))
        if not ok:
            raise SystemExit("  no starter question separated even on a "
                             "trivial program. The question set is not "
                             "measuring anything; fix it before running.")
    if "fanout_n3_seeded" in args.arms:
        words = default_seeds(3)
        print(f"  concept seeds alive: {words}")
        if len(set(words)) < 3:
            raise SystemExit("  concept_seed drew duplicates; seeded arm would "
                             "not be a seeded arm.")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rnd = random.Random(args.seed)
    picked: list = []
    per = max(1, args.n // 3)
    for diff in ("easy", "medium", "hard"):
        g = [r for r in rows if r.get("difficulty") == diff]
        rnd.shuffle(g)
        picked += g[:per]
    rnd.shuffle(picked)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["question_id"], r["arm"]))
            except Exception:                                    # noqa: BLE001
                continue
    names = [a.strip() for a in args.arms.split(",") if a.strip()]

    def gen(prompt: str, temperature: float, i: int):
        return model_generate(prompt, temperature, i,
                              args.max_tokens, args.gen_timeout)

    with open(args.out, "a", encoding="utf-8") as out:
        for k, row in enumerate(picked):
            qid = row["question_id"]
            functional = bool(row.get("starter_code", "").strip())
            prompt = (lcb.PROMPT_FUNCTIONAL.format(question=row["question_content"],
                                                   starter=row["starter_code"])
                      if functional else
                      lcb.PROMPT_STDIN.format(question=row["question_content"]))
            for name in names:
                if (qid, name) in done:
                    continue
                rec = {"question_id": qid, "arm": name,
                       "difficulty": row.get("difficulty")}
                try:
                    r = DETAILED[name](prompt, gen)
                except Exception as e:                           # noqa: BLE001
                    rec.update({"passed": False, "ms": 0,
                                "error": f"{type(e).__name__}: {e}"})
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"  [{k + 1}/{len(picked)}] {qid:<14} {name:<18} "
                          f"ARM FAILED {type(e).__name__}", flush=True)
                    continue
                passed, why, perf = score_row(r.code, row, args.max_cases,
                                              args.test_timeout)
                rec.update({"passed": passed, "why": why, "ms": r.ms,
                            "agreement": r.agreement, "n_errors": r.n_errors,
                            "n_nocode": r.n_nocode, "n_voting": r.n_voting,
                            "group_sizes": r.group_sizes, "chosen": r.chosen,
                            # The whole feature matrix, every margin and both
                            # option orders. Compounding error across five
                            # questions is undiagnosable without it.
                            "features": r.features, "notes": r.notes,
                            "degraded": r.degraded, **perf})
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"  [{k + 1}/{len(picked)}] {qid:<14} {name:<18} "
                      f"{'PASS' if passed else 'fail'} {r.ms / 1000:5.1f}s "
                      f"agree={r.agreement:.2f} {why[:36]}", flush=True)

    report(args.out)


if __name__ == "__main__":
    main()
