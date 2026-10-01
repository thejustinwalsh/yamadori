#!/usr/bin/env python
"""THE DECIDER: decide(state, options) -> probabilities.

Operator, 2026-09-27: after the successive filter rounds, "we will be left
with a handful of real questions that we ask and get options from". A
QUESTION is typed and closed: a STATE (this turn's evidence -- the user's
words plus a digest of the newest tool results and writes, the same text
the embedding stage reads, skill_classify.embedding_query) and a handful of
OPTIONS -- each surviving skill's distilled trigger condition (its
description) -- plus an explicit NONE ("none of these"). A decider returns
a probability per option. They are RELATIVE TO THE SET (CLM's card: "relative
to provided candidate sets"; the same rule as Laya's, AGENTS.md "Laya"): the
pick is the argmax when it beats NONE, else nothing, and no probability is
ever compared against a number or across questions.

    decider   what it reads                      when
    stub      the option's EVIDENCE (the          offline tests and the daily
              deterministic ranking the rounds    eval; today's default
              computed: prior, tier)
    lexical   the state's and the options' TEXT   TEST-ONLY (never served): a
              only (IDF-weighted word overlap)    text stand-in for a text
                                                  decider
    fallback  the model, one call (skill_select's LAST RESORT: options the
              FALLBACK_SYSTEM, thinking off)      evidence cannot settle (an
                                                  `ask` tier) and no model
                                                  decider answered

The `clm` decider (CLM-v0.1-8B, a frozen Qwen3-8B encoder + heads, served as
llama-swap's clm-encoder) was retired 2026-09-28 for the Bonsai decider and
its code removed 2026-09-29 (the way back is commit e360d37).

A decider that cannot answer returns None (the stub facing only unconfirmed
options); the orchestrator (skill_select) then tries the next.
`YAMADORI_SKILL_DECIDER` names the first (stub only today).

Every decider is order-invariant: shuffling the options changes no
probability (checked by mcp/test_skill_questions.py).
"""
from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NONE = "none"
NONE_TEXT = ("None of these: this turn is about something else, or it "
             "needs none of them.")
MAX_OPTIONS = 6          # the none option included (a CHOICE: "~6")


@dataclass
class Option:
    """One answer to a question. `text` is what a text decider reads (the
    skill's trigger condition); `evidence` is what the stub reads (never
    sent to a model): tier inject | ask | no, prior (a rank score),
    decided_by, band."""
    id: str
    text: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Question:
    """A typed closed question. `kind`: asked | evidence | step | category.
    `options` exclude NONE (every decider adds it)."""
    qid: str
    kind: str
    area: str
    state: str
    options: list[Option]
    instructions: str = ""
    meta: dict = field(default_factory=dict)

    def choice(self) -> dict[str, str]:
        out = {o.id: o.text for o in self.options}
        out[NONE] = NONE_TEXT
        return out


def softmax(logits: dict[str, float]) -> dict[str, float]:
    finite = {k: v for k, v in logits.items() if v != -math.inf}
    if not finite:
        return {k: (1.0 if k == NONE else 0.0) for k in logits}
    m = max(finite.values())
    ex = {k: (math.exp(v - m) if v != -math.inf else 0.0)
          for k, v in logits.items()}
    z = sum(ex.values()) or 1.0
    return {k: round(v / z, 6) for k, v in ex.items()}


def pick(probs: dict[str, float] | None) -> str | None:
    """The argmax when it beats NONE; None otherwise. Ties break by id (a
    fixed order, so a shuffle cannot change the pick)."""
    if not probs:
        return None
    best = max(sorted(probs), key=lambda k: probs[k])
    if best == NONE or probs[best] <= probs.get(NONE, 0.0):
        return None
    return best


class Decider:
    name = "base"

    def decide(self, state: str, options: list[Option],
               question: Question | None = None) -> dict[str, float] | None:
        raise NotImplementedError


class EvidenceStub(Decider):
    """`stub`: the deterministic ranking as a decider. An option is ELIGIBLE
    when its evidence tier is `inject` (fact or phrase strength, or a
    confirmed candidate); its logit is its prior (the rounds' rank score).
    NONE sits one below the weakest eligible option, so the best eligible
    option wins and NONE wins when nothing is eligible. Abstains (None) when
    the only candidates are unconfirmed `ask`-tier options: the evidence
    cannot settle them, a model must (the fallback)."""
    name = "stub"

    def decide(self, state, options, question=None):
        elig = {o.id: float(o.evidence.get("prior") or 0.0) for o in options
                if o.evidence.get("tier") == "inject"}
        if not elig and any(o.evidence.get("tier") == "ask" for o in options):
            return None
        logits = {o.id: (elig[o.id] if o.id in elig else -math.inf)
                  for o in options}
        logits[NONE] = (min(elig.values()) - 1.0) if elig else 0.0
        return softmax(logits)


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""the and for are but not can will should would could this
that these those what which who how why its use uses used using when whenever
with without into from your you our their there then than also only just like
such each every any all some more most much many other about over under after
before while where here none this turn needs them something else""".split())


def _words(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(str(text or "").lower()):
        if len(w) < 3 or w in _STOP or w.isdigit():
            continue
        if w.endswith("s") and len(w) > 4 and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


class LexicalStub(Decider):
    """`lexical`: TEXT ONLY -- the state and each option's text,
    nothing else. Logit = the IDF-weighted share of the option's words found
    in the state (IDF over this question's own options, so it is
    set-relative); NONE's text is scored the same way. A stand-in for a
    text decider offline; it is not the gate."""
    name = "lexical"
    SCALE = 4.0

    def decide(self, state, options, question=None):
        sw = _words(state)
        texts = {o.id: _words(o.text) for o in options}
        texts[NONE] = _words(NONE_TEXT)
        n = len(texts)
        df: dict[str, int] = {}
        for ws in texts.values():
            for w in ws:
                df[w] = df.get(w, 0) + 1
        logits = {}
        for k, ws in texts.items():
            if not ws:
                logits[k] = 0.0
                continue
            idf = {w: math.log((n + 1) / (df[w] + 0.5)) for w in ws}
            hit = sum(idf[w] for w in ws if w in sw)
            logits[k] = self.SCALE * hit / (sum(idf.values()) or 1.0)
        # NONE is scored like any option, from its own text: no floor, no
        # number outside the set.
        return softmax(logits)


DECIDERS = {"stub": EvidenceStub, "lexical": LexicalStub}


# The deciders a DEPLOYMENT may name (YAMADORI_SKILL_DECIDER). `lexical` is
# TEST-ONLY since 2026-09-27 (docs/CONSTANTS-AUDIT.md "LexicalStub SCALE /
# IDF smoothing"): its SCALE, smoothing and stop words are ours, so it is
# reachable only by a test or replay that names it (chain("lexical")).
SERVING = ("stub",)


def configured() -> str:
    name = (os.environ.get("YAMADORI_SKILL_DECIDER") or "stub").strip().lower()
    return name if name in SERVING else "stub"


def chain(first: str | None = None) -> list[Decider]:
    """The deciders in the order they are tried: the configured one, then
    the stub (the evidence) when the first abstains. The fallback model is
    the orchestrator's last resort, not in this list: it answers a batch
    of unconfirmed options in ONE call (skill_select)."""
    first = first or configured()
    out = [DECIDERS[first]()]
    if first != "stub":
        out.append(EvidenceStub())
    return out
