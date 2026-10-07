#!/usr/bin/env python
"""A TURN'S JUDGMENT QUESTIONS, answered by the main model in ONE batch.

    with decide_turn.Turn(messages, key=..., account=...) as t:
        t.facts()                       # the fixed questions, answered
        t.confirm_packages(weak)        # weak package detections only
        t.pick(area, candidates, hard_evidence=False)   # the selector's hook
    t.record()                          # x_yamadori.decider

Operator, 2026-09-27: the Bonsai decider (mcp/decider_bonsai.py) is the
DEFAULT for every judgment question. "Your comparison is hand selected, hand
tuned and not representative of random workloads." Rules keep only
MECHANICAL facts (imports, pins, unique symbols in code, exit codes, file
paths); the old hand rules become the FALLBACK when the decider cannot
answer, and every decider-vs-rule disagreement is logged -- ids and labels,
never text -- for judging on held-out traffic.

WHAT IS ASKED, AND WHERE (bench/decider/results/bonsai.json, in-sample):

  question          request       decider state    rule (fallback, compared)
  build_intent      user turn     the user turn    route.work_intent, ONLY
                                  + the previous   when the decider is
                                  assistant tail   unavailable
  phase (4-way)     every request the turn/step    skill_classify.request_
                                                   signals phases
  reads_package     agent step    RULE ONLY: a path fact (skill_select.
                                  probed_packages 0.948 vs the decider's
                                  best 0.888 on pagoda-h4)
  scratch_write     NOT ASKED: the selector's mechanical rule (a terminal
                    redirect / tee target outside the project); the decider
                    read is behind YAMADORI_DECIDER_SCRATCH=1
  package <p>       WEAK detections only: a symbol in prose or on a comment
                    line; a name, import, pin, implication or code-context
                    symbol stays a fact (the decider does not know koota's or
                    math's API: it lost 21 of 33 true detections as a veto)
  skill fit         the selector's pick(): one CHOICE per area, "none of
                    these" not always last, repeated with the pick removed

  NOT asked: a repeated failing step. The rule (deep.struggle_scan) found 5/5
  with no false alarm on pagoda-h4; the decider found 0-2 of 5 at any
  context size. It stays a rule.

THE STATE: the current turn or step only (every h4 question got worse with
more context: reads_package 0.888 on the step, 0.784 on 2,048 tokens of
history, 0.410 on the whole history). A user turn is its text; an agent step
is the newest assistant call(s) and their result(s). A state longer than
STATE_TOKENS keeps its head and its tail (half each), cut on characters in
proportion to the model's own token count. STATE_TOKENS = 2,048
(operator, 2026-09-27): the configuration every comparison measured (CLM's
trained window; the bench's `window2048` arm). Its measured cost: a first
question processing 1,024-4,095 tokens took p50 2.3 s, p90 4.5 s (the
bench's latency table), ~1,000 tokens a second of prefill beyond ~500.

A USER TURN'S STATE carries the PREVIOUS ASSISTANT TURN's tail before it
("Assistant (previous turn): ...", then "User: ..."), at most PREV_TOKENS
(255) of it, cut from its front: a bare "Go ahead." / "Ship it." was the
decider's commonest intent error (p(yes) 0.18-0.47), and what it answers is
in the turn before. The bound keeps the added prefill inside the latency
table's 64-255-processed-token band (p50 496 ms, p90 577 ms, against 857 /
1,342 ms for 256-1,023). UNMEASURED for accuracy: work_intent.jsonl's rows
carry no previous turn.

THE TYPED READOUT (the default since 2026-09-29; READOUT below): every
question is a Jev primitive -- noul, choice, score -- asked through
Turn.decide / decide() and answered by decider_bonsai.read in JEV'S SHAPE
(noul -> {noul}; choice -> {choice, probabilities, confidence}; score ->
{score, probabilities, confidence, legend}; docs/JJAVA.md), with the two
orders' diagnostics beside it. The decider returns belief and certainty
only: THIS module is the question sets' code and owns their thresholds
(THRESHOLDS below: high -> act, medium -> proceed with caution, low -> fall
back), tuned per question set per model on labelled decisions (bench/
decider/tune.py proposes, the owner accepts); NONE SHIP UNTUNED -- the
table is empty, every tier reads "untuned", and each question set acts on
the argmax as it did (a measured TIE_BAND tie still falls back where it
did). EVERY decision is logged with its confidence and its corpus join keys
(DECISIONS, join_corpus; bench/decider/label.py labels them). The wrappers
keep every old call site: facts(), confirm_packages(), pick(), choose()
(skill_match), build_intent(), judge_stop(), verify_moment.choose_tool().
YAMADORI_DECIDER_READOUT=legacy restores the forms below, unchanged, ONLY
until bench/decider/legacy_vs_typed.py has run once on the GPU; then the
legacy readout is deleted (the removal list: docs/CONSTANTS-AUDIT.md
"Decider legacy readout"; operator: "Stop keeping old shit behind
switches").

DEBIASING, LEGACY (docs/research/SKILLS-RESEARCH.md Part 4 item 2; decider_bonsai
"debiasing"): every question is a neutral-letter CHOICE (a yes/no becomes
options yes / no), asked in ORDERS option orders (the given and the
reversed, so no option is always printed last) and averaged; each template's
content-free read (decider_bonsai.NEUTRAL_STATE) divides it out when
CONTEXTUAL; the two largest within decider_bonsai.TIE_BAND are a TIE -- no
decision, the fallback rule decides and the record says so.

THE SLOT: the transient side-call slot, held for the Turn (every question
reuses the cached state), released at close (decider_bonsai.release): no
decider cell outlives the turn. Content-free reads are made once per
template per process and kept (TEMPLATE_CF), before the turn's state is
placed, so they never evict it. The Turn runs synchronously in the request's
preparation, before main's first generation.

FAILURES: a DeciderUnavailable anywhere makes every unanswered question fall
back to its rule; the record carries the failure's facts (code, retryable,
remedy). Never raises out of facts(), confirm_packages() or pick().
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decider_bonsai as D  # noqa: E402

# -------------------------------------------------------------- settings ---
# Default ON (operator, 2026-09-27) in the SERVING process: server.py calls
# enable() at startup. Anywhere else -- an offline suite exercising the
# proxy's functions, the worker -- a Turn built without on= asks nothing and
# every answer is its rule's (source "off"). YAMADORI_DECIDER=0 turns it off
# in the serving process too.
ON = os.environ.get("YAMADORI_DECIDER", "1") != "0"
_ENABLED = False


def enable(on: bool = True, prime: bool = True) -> None:
    """Turn the decider on in this (the serving) process, and start the
    background prime of the choose() label priors (prime_priors_background)
    unless `prime` is False."""
    global _ENABLED
    _ENABLED = bool(on)
    if _ENABLED and ON and prime and not typed_readout():
        # The typed readout divides no label prior out: nothing to prime.
        prime_priors_background()


# THE LABEL-PRIOR PRIME (coordinator, 2026-09-27). choose() divides out a
# label prior read once per option count per process; read lazily it costs
# the turn that first meets a count its cached state (up to ~2 s). So the
# serving process reads the priors for PRIME_COUNTS in a background thread
# after startup, ONE COUNT PER Turn (its own acquire/release of the
# transient slot, so a real request waits at most one short read), and only
# while the model is IDLE: no slot processing (/slots is_processing) and no
# request of this process in flight (slots._busy). Busy -> it waits and
# re-checks every PRIME_POLL_S; a failure stops it, and the lazy per-count
# read in choose() remains. One log line when it ends: counts and seconds.
# PRIME_COUNTS: 2..len(D.LETTERS) -- one skill and "None of these" up to
# every single-letter label, "None of these" included: it takes a letter
# too (prime_choose labels it D.LETTERS[len(opts)]), so A..Z is at most 25
# skills + None = 26 options. It read 2..27 until 2026-09-28, and 27 asked
# for a 27th letter: IndexError at every proxy start ("decider label
# priors: failed -- read [2 .. 26] ... tuple index out of range").
# PRIME_POLL_S: how often a waiting prime re-checks; a polling cadence, it
# changes no answer.
PRIME_COUNTS = tuple(range(2, len(D.LETTERS) + 1))
PRIME_POLL_S = 1.0
PRIME_STATE: dict = {"state": "not started"}


def _idle(upstream=None) -> bool:
    """No main-model slot processing and no request of this process in
    flight."""
    import max_mode
    dm = max_mode.decider_model()          # the model jjava reads (the table's helper under a locked model)
    if max_mode.blocks(dm) and max_mode.bound_model() != dm:
        # MAX MODE: the prime's model is off the card; asking its /slots would load it
        return False
    try:
        import slots
        with slots._lock:
            if any(v > 0 for v in slots._busy.values()):
                return False
        if upstream is None:
            # A READ NEVER LOADS A MODEL (2026-09-30): max_mode.blocks() is
            # False when NO main model is loaded, and /upstream/<model>/slots
            # would then start it. Not loaded: not idle (the prime waits).
            import gpu_room
            import model as _model
            loaded, _why = gpu_room.model_loaded(_model.UPSTREAM,
                                                  max_mode.decider_model(_model.MODEL))
            if not loaded:
                return False
        table = (upstream or D._upstream)("/slots", timeout=5)
        return isinstance(table, list) and not any(
            isinstance(s, dict) and s.get("is_processing") for s in table)
    except Exception:                                            # noqa: BLE001
        return False


def prime_priors(counts=PRIME_COUNTS, *, idle=None, sleep=time.sleep,
                 post=None, upstream=None, stop=None,
                 log=print) -> dict:
    """Read the label priors for `counts`, one per idle moment. Returns (and
    keeps in PRIME_STATE) {state, done, skipped, waited_s, seconds}."""
    t0 = time.time()
    idle = idle or (lambda: _idle(upstream))
    st = {"state": "running", "counts": list(counts), "done": [],
          "already": [], "waited_s": 0.0}
    PRIME_STATE.clear()
    PRIME_STATE.update(st)
    for n in counts:
        while not idle():
            if stop is not None and stop.is_set():
                break
            sleep(PRIME_POLL_S)
            st["waited_s"] += PRIME_POLL_S
        if stop is not None and stop.is_set():
            st["state"] = "stopped"
            break
        try:
            with Turn([], on=True, post=post, upstream=upstream,
                      count=len) as t:
                before = len(t.answers)
                t.prime_choose([n])
                (st["done"] if len(t.answers) > before
                 else st["already"]).append(n)
        except Exception as e:                                   # noqa: BLE001
            st["state"] = "failed"
            st["error"] = f"{type(e).__name__}: {e}"[:200]
            break
    else:
        st["state"] = "done"
    st["seconds"] = round(time.time() - t0, 1)
    PRIME_STATE.update(st)
    log(f"  decider label priors: {st['state']} -- read {st['done']}"
        f"{', already held ' + str(st['already']) if st['already'] else ''}"
        f" in {st['seconds']} s (waited {round(st['waited_s'], 1)} s for "
        f"an idle model)" + (f"; {st['error']}" if st.get("error") else "")
        + ("; the rest are read lazily by choose()"
           if st["state"] != "done" else ""), flush=True)
    return st


_PRIME_THREAD: list = []


def prime_priors_background(model: str | None = None, **kw) -> threading.Thread | None:
    """prime_priors on a daemon thread, once per process -- and, under max
    mode, once per main model (prime_for): the thread is bound to that model,
    so every read goes to it and its priors are kept under its name."""
    import max_mode
    if _PRIME_THREAD and _PRIME_THREAD[0].is_alive():
        return _PRIME_THREAD[0]

    def run():
        import cancel
        tok = cancel.Token()
        with cancel.bound(tok):
            if model:
                max_mode.set_current(model)
            prime_priors(**kw)
    th = threading.Thread(target=run, name=f"decider-prime-{model or 'main'}",
                          daemon=True)
    _PRIME_THREAD[:] = [th]
    th.start()
    return th


_PRIMED_MODELS: set = set()


def prime_for(model: str) -> threading.Thread | None:
    """MAX MODE: re-prime the label priors on `model` the first time a request
    is served by it in this process (the priors are keyed by model; the label
    check runs with them: spelling_ids reads that model's /tokenize). A no-op
    when the decider is off or the model was primed already."""
    if not enabled() or model in _PRIMED_MODELS or typed_readout():
        return None
    _PRIMED_MODELS.add(model)
    return prime_priors_background(model=model)


def _cf_key(k: str) -> str:
    """TEMPLATE_CF's key, per main model under max mode (a label prior is
    a property of the model that reads it)."""
    import max_mode
    return f"{max_mode.decider_model()}:{k}" if max_mode.ENABLED else k


def enabled() -> bool:
    return _ENABLED and ON


# Operator, 2026-09-27: the configuration every comparison measured.
STATE_TOKENS = int(os.environ.get("YAMADORI_DECIDER_STATE_TOKENS", "2048"))
# The previous assistant turn's tail on a user turn's state: the top of the
# latency table's 64-255 band (see the module docstring).
PREV_TOKENS = 255
ORDERS = 2
# HOW EACH QUESTION IS READ, from the calibration re-measurement
# (bench/decider/bonsai_decider.py `calib`, 2026-09-27, in-sample; the
# research doc's recommendations measured on OUR labels):
#   yn         the yes/no as asked, one read
#   mc_avg     a neutral-letter choice in ORDERS option orders, averaged
#   mc_avg_cc  the same, each order divided by its content-free read
# Content-free calibration HURT every yes/no-type question here: the "N/A"
# read leans "no" (p(no) 0.78-0.90), dividing it out pushes toward "yes"
# (intent 0.869 -> 0.818 accuracy on the operator's wording, reads_package
# 0.851 -> 0.582, scratch precision 0.185 -> 0.119) -- EXCEPT as a VETO of
# a detection: packages mc_avg_cc kept 33/33 true detections and vetoed the
# same 2 of 4 false ones the raw read did (raw lost 20 true). Order
# averaging helped build intent (0.848 -> 0.869) and the phase choice's
# agreement with the rule's single phase (47 -> 57 of 79); the yes/no read
# stayed best on the h4 step questions (reads_package 0.888 vs 0.851;
# scratch precision 0.227 vs 0.185).
# reads_package is a PATH fact (a read under node_modules/): the rule
# (skill_select.probed_packages) scored 0.948 on pagoda-h4, the decider's
# best read 0.888, so the rule answers it -- a mechanical fact, which the
# operator's split keeps as a rule. ASK_READS_PACKAGE asks the decider too.
ASK_READS_PACKAGE = os.environ.get("YAMADORI_DECIDER_READS_PACKAGE") == "1"
# scratch_write is the selector's MECHANICAL rule (operator, 2026-09-27: a
# terminal redirect / tee target outside the project); the decider's read
# (5/5 found, precision 0.227 on pagoda-h4's terminal steps) is off unless
# YAMADORI_DECIDER_SCRATCH=1.
ASK_SCRATCH = os.environ.get("YAMADORI_DECIDER_SCRATCH") == "1"
# BUILD INTENT is the decider's (operator, 2026-09-27, overruling the rule's
# in-sample 1.00): route.work_intent answers only when the decider is
# unavailable; a near-tie is still the decider's, recorded as a tie.
DECIDER_ON_TIE = frozenset({"build_intent"})
# MEASURED ON BONSAI ONLY (the calib numbers above, and FORM): for another
# model they are UNMEASURED -- decider_bonsai's model profile lists `readout`
# and `legacy_form` under `unmeasured` for it (record() "model_profile"), and
# the forms run as measured on Bonsai; nothing is re-chosen per model until
# bench/decider/measure_model.py has measured it (docs/JJAVA.md "Per model").
FORM = {"build_intent": "mc_avg", "phase": "mc_avg",
        "reads_package": "yn", "scratch_write": "yn",
        "package": "mc_avg_cc", "choose": "rotated_idprior"}
CONTEXTUAL = os.environ.get("YAMADORI_DECIDER_CONTEXTUAL", "1") != "0"
LOG = os.environ.get("YAMADORI_DECIDER_LOG") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs",
    "decider_disagreements.jsonl")

# THE READOUT (operator, 2026-09-29: "look to the Jev api for the answer,
# they got it right"; docs/research/DECIDER-RESEARCH.md 4.2 items 1, 3).
#   typed   (default) every question is TYPED -- noul / choice / score --
#           and answered in Jev's shape by decider_bonsai.read: letters (a
#           noul as a lettered pair), two option orders averaged, NO
#           label-prior division (choose() and judge_stop lose their
#           rotation and id prior), Jev's confidence on Choice and Score,
#           the orders' diagnostics logged, every decision logged with its
#           join keys (DECISIONS).
#   legacy  the 2026-09-27 forms (FORM above), unchanged, ONLY for the one
#           GPU side-by-side (bench/decider/legacy_vs_typed.py):
#           YAMADORI_DECIDER_READOUT=legacy. Its decisions are logged too.
#           ON THE REMOVAL LIST (docs/CONSTANTS-AUDIT.md "Decider legacy
#           readout"): deleted, with FORM, ask(), _prime(), the label-prior
#           prime and TEMPLATE_CF's legacy keys, once that run is in.
# Read at call time, so a bench can flip it per arm.
READOUT = os.environ.get("YAMADORI_DECIDER_READOUT", "typed")
# EVERY question the decider answers, by its Jev primitive (typed readout).
# Nothing asked today is a level, so no question maps to `score` yet; the
# primitive exists for the first one that is (a rubric).
MODES = {"build_intent": "noul", "phase": "choice", "reads_package": "noul",
         "scratch_write": "noul",
         "package": "noul (content-free prior kept: the measured veto)",
         "pick": "choice", "choose": "choice", "stop": "choice (judge_stop)",
         "verify": "choice (verify_moment.choose_tool, via choose)"}
# EVERY DECISION, one JSON line each (4.2 item 1: the disagreement log could
# not be joined -- 0 of 222 request ids matched the corpus -- and held no
# agreements). Ids and labels only, never text: the question's name and
# type, its option KEYS (phase names, skill ids, a client's tool names), each
# order's distribution, the averaged distribution, Jev's confidence (Choice
# and Score) or the noul, the argmax, the pick the question set acted on,
# its tier, the rule's answer where one exists, and the JOIN KEYS -- the
# conversation key, the request (the ledger's turn key), the corpus turn id
# (`corpus_turn`, or a later "join" row naming the decisions: join_corpus),
# and a `state` hash to check a reconstructed state against. The decision id
# rides in x_yamadori (the Turn's record) so a relay row joins by it.
# v2 (2026-09-29): + confidence / noul / score / tier; - abstain,
# abstain_why, thresholds (the decider's built-in abstain is gone).
DECISIONS = os.environ.get("YAMADORI_DECIDER_DECISIONS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs",
    "decider_decisions.jsonl")
DECISION_LOG_VERSION = 2

# THE QUESTION SETS' THRESHOLDS (operator, 2026-09-29: "Callers own
# thresholds"; Jev /confidence: "Three paths for using confidence in your
# code" -- high: act automatically; medium: proceed with caution; low: do
# not act, fall back -- and "Thresholds scale with risk"). Keyed by MODEL
# (a probability level is a property of the model that reads it: Bonsai,
# Flash-Next and Mirai S each get their own), then by question set: a
# question's full name ("choose:stop") or its family ("choose", "phase",
# "build_intent", "package", "pick").
#   Choice / Score  {"high": c, "medium": c} on the answer's confidence:
#                   >= high -> high; >= medium -> medium; else low.
#   Noul            {"act_yes": p, "act_no": p, "caution_yes": p,
#                   "caution_no": p} on the noul itself (Jev: a noul has no
#                   confidence; "Values in the middle can go to a person";
#                   raise the yes bar where a false yes is expensive):
#                   p >= act_yes or p <= act_no -> high; p >= caution_yes or
#                   p <= caution_no -> medium; else low.
# What each tier DOES is the question set's own (below): high acts; medium
# acts and says so in its source; low falls back -- the rule for a fact,
# no pick for pick()/choose() (the caller's fallback), keep for a package.
# EMPTY: NONE SHIP UNTUNED (operator). bench/decider/tune.py proposes a row
# from labelled decisions (bench/decider/label.py) at the owner's target
# accuracies; the owner pastes the accepted row here, with its n and date.
# Nothing writes this table.
THRESHOLDS: dict = {}


def typed_readout() -> bool:
    return READOUT != "legacy"


def tier(a: dict, model: str | None = None) -> dict:
    """The question set's tier for a typed answer (decider_bonsai.read):
    {tier: untuned | high | medium | low, question_set, model}."""
    model = model if model is not None else D.model_name()
    name = str(a.get("name") or "")
    fam = name.split(":", 1)[0]
    # PER MODEL: the serving model's own rows (an alias -- bonsai-agent --
    # reads its model's); another model's rows are never borrowed.
    table = THRESHOLDS.get(model) or THRESHOLDS.get(
        D.canonical_model(model)) or {}
    key = name if name in table else fam if fam in table else None
    if key is None:
        return {"tier": "untuned", "question_set": fam, "model": model}
    row = table[key]
    inf = float("inf")
    if a.get("type") == "noul":
        p = float(a["noul"])
        t = ("high" if p >= row.get("act_yes", inf)
             or p <= row.get("act_no", -inf) else
             "medium" if p >= row.get("caution_yes", inf)
             or p <= row.get("caution_no", -inf) else "low")
    else:
        c = float(a["confidence"])
        t = ("high" if c >= row.get("high", inf) else
             "medium" if c >= row.get("medium", inf) else "low")
    return {"tier": t, "question_set": key, "model": model}


def flat(a: dict) -> dict:
    """A typed answer as the wrappers and the log read it: {type, name,
    keys, probs (by key; a noul as true/false), argmax, tie, confidence
    (None for a noul), noul (None otherwise), score, tier, answer (what the
    question set acts on: the argmax, None on a tie or a low tier), and the
    diagnostics}."""
    d = a.get("diagnostics") or {}
    t = (a.get("tier") or {}).get("tier", "untuned")
    return {"type": a.get("type"), "name": a.get("name"),
            "keys": d.get("keys"), "probs": D.answer_probs(a),
            "argmax": d.get("argmax"), "tie": bool(d.get("tie")),
            "confidence": a.get("confidence"), "noul": a.get("noul"),
            "score": a.get("score"), "tier": t,
            "answer": None if d.get("tie") or t == "low" else d.get("argmax"),
            "margin": d.get("margin"), "orders": d.get("orders") or [],
            "disagreement": d.get("disagreement"),
            "argmax_agree": d.get("argmax_agree"),
            "label_mass_min": d.get("label_mass_min"),
            "prior": d.get("prior"), "excluded": d.get("excluded") or [],
            "readout": d.get("readout"), "reads": d.get("reads"),
            "tie_band": d.get("tie_band"),
            # the temperature read with (1.0 unless one is fitted: decider_
            # bonsai THE TEMPERATURE) and the case-variant diagnostics
            "temperature": (d.get("temperature") or {}).get("value"),
            "variant_mass_max": d.get("variant_mass_max"),
            "word_mass_max": d.get("word_mass_max"),
            "processed_tokens": d.get("processed_tokens"),
            "prompt_tokens": d.get("prompt_tokens"),
            "decision_id": a.get("decision_id"), "ms": d.get("ms") or 0.0}

# The questions (the wordings measured in bench/decider/bonsai_decider.py).
BUILD_INTENT_Q = "Is the user asking for something to be built, made or changed?"
PHASES = ["plan", "implement", "debug", "verify"]
PHASE_Q = ("Which phase is the agent's work in?",
           ["planning the work", "implementing the work",
            "debugging a failure", "verifying or testing the work"])
READS_PACKAGE_Q = ("Is the agent reading a package's installed files to "
                   "learn its API?")
SCRATCH_Q = ("Is the agent writing a throwaway file to test how something "
             "behaves?")
PACKAGE_Q = "Does this work use {name}{desc}?"
FIT_Q = "Which of these fits what the agent is doing now?"
NONE_OPTION = "None of these fits."

_CF_LOCK = threading.Lock()
TEMPLATE_CF: dict[str, list[dict]] = {}


# ------------------------------------------------------------- the state ---
def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(str(p.get("text") or "") for p in c
                         if isinstance(p, dict))
    return str(c or "")


def turn_kind(messages: list[dict]) -> str:
    """user | step | other: what the request ends on."""
    last = next((m for m in reversed(messages or []) if isinstance(m, dict)
                 and m.get("role") in ("user", "tool", "assistant")), None)
    if not last:
        return "other"
    return {"user": "user", "tool": "step"}.get(last.get("role"), "other")


def step_text(messages: list[dict]) -> str:
    """The newest step: the assistant turn that made the calls and every
    tool result after it."""
    i = len(messages) - 1
    while i >= 0 and messages[i].get("role") == "tool":
        i -= 1
    if i < 0:
        return ""
    a = messages[i]
    parts = []
    said = _text(a).strip()
    if said:
        parts.append(f"Assistant: {said}")
    names = {}
    for tc in a.get("tool_calls") or []:
        f = (tc or {}).get("function") or {}
        names[tc.get("id")] = f.get("name")
        parts.append(f"Assistant called {f.get('name')}: "
                     f"{f.get('arguments') or ''}")
    for m in messages[i + 1:]:
        parts.append(f"Result of {names.get(m.get('tool_call_id'), 'the call')}"
                     f": {_text(m)}")
    return "\n".join(parts)


def state_of(messages: list[dict], count=None) -> tuple[str, dict]:
    """(state text, info): the user turn or the step, head+tail cut to
    STATE_TOKENS by `count` (the model's tokenizer; chars/3 when absent)."""
    kind = turn_kind(messages)
    count = count or (lambda s: len(s) // 3)
    prev_info = None
    if kind == "user":
        ui = max(i for i, m in enumerate(messages)
                 if m.get("role") == "user")
        text = _text(messages[ui])
        prev = next((m for m in reversed(messages[:ui])
                     if m.get("role") == "assistant" and _text(m).strip()),
                    None)
        if prev is not None:
            pt = _text(prev).strip()
            pn = count(pt)
            if pn > PREV_TOKENS:
                pt = "..." + pt[-int(len(pt) * PREV_TOKENS / pn):]
            prev_info = {"tokens": min(pn, PREV_TOKENS),
                         "cut": pn > PREV_TOKENS}
            text = f"Assistant (previous turn): {pt}\n\nUser: {text}"
    elif kind == "step":
        text = step_text(messages)
    else:
        text = ""
    n = count(text)
    info = {"kind": kind, "tokens": n, "cut": False,
            "previous_assistant": prev_info}
    if n > STATE_TOKENS and text:
        keep = int(len(text) * STATE_TOKENS / n) // 2
        text = (text[:keep] + "\n\n[... middle left out ...]\n\n"
                + text[-keep:])
        info.update(cut=True, tokens_kept=STATE_TOKENS)
    return text, info


def _count(text: str) -> int:
    try:
        return len(D._upstream("/tokenize", {"content": text,
                                             "add_special": False},
                               timeout=10)["tokens"])
    except Exception:                                            # noqa: BLE001
        return len(text) // 3


# ------------------------------------------------------------- the turn ----
class Turn:
    """One request's questions over one state on one transient slot."""

    def __init__(self, messages: list[dict], *, key: str | None = None,
                 account: str = "", request: str | None = None,
                 client_tools=None, post=None, upstream=None, count=None,
                 on: bool | None = None, state: str | None = None,
                 state_info: dict | None = None):
        self.messages = [m for m in messages or [] if isinstance(m, dict)]
        self.key, self.account, self.request = key, account, request
        self.client_tools = list(client_tools or [])
        self.on = enabled() if on is None else bool(on)
        self._post = post or D._post_default
        self._upstream = upstream or D._upstream
        # Off: no request of any kind (chars/3 stands in for the count).
        # `state` given: the caller built it (judge_stop: a generated turn,
        # not the request's last message).
        if state is not None:
            self.state = state
            self.state_info = dict(state_info or {"kind": "other"})
        else:
            self.state, self.state_info = state_of(
                self.messages, count or (_count if self.on else None))
        self.kind = self.state_info["kind"]
        # The decision log's check on a reconstructed state (never the text).
        self.state_sha = (hashlib.sha1(self.state.encode("utf-8"))
                          .hexdigest()[:16] if self.state else None)
        self.decision_ids: list[str] = []
        um = next((m for m in reversed(self.messages)
                   if m.get("role") == "user"), None)
        self.user_text = _text(um) if um else ""
        self.slot = None
        self.grant = None
        self.answers: list[dict] = []
        self.failure: dict | None = None
        self.disagreements: list[dict] = []
        self.release_rec: dict | None = None
        self.t0 = time.time()
        self.ms_questions = 0.0
        self._facts: dict | None = None
        # choose(): each question's earlier rounds [(question, label)],
        # keyed by area + its options (the rendering extends them).
        self._rounds: dict[str, list] = {}

    # -- slot
    def __enter__(self):
        # The selector reaches the Turn through choose() (module
        # __getattr__, below) while it is this thread's current one.
        self._prev = getattr(_LOCAL, "turn", None)
        _LOCAL.turn = self
        return self

    def __exit__(self, *exc):
        _LOCAL.turn = getattr(self, "_prev", None)
        self.close()
        return False

    def _open(self):
        if self.slot is not None or self.grant is not None:
            return
        import slots
        self.grant = slots.acquire(None, transient=True)
        self.slot = self.grant.get("slot")

    def close(self) -> None:
        b = getattr(self, "_burst", None)
        if b is not None:                  # the batched reads' resident prefix
            self._burst = None
            self.batch_release = b.release()
        if self.grant is None:
            return
        import slots
        slots.release(self.grant)
        # LAYOUT V2 (operator, 2026-09-29): the decider's slot is THE LANE,
        # kept between turns (slots THE LANE) -- its head stays cached, and
        # the release it no longer pays cost median 395 ms, p90 824 ms a turn
        # (logs/proxy.log, the last 200 decider turns).
        self.release_rec = (slots.lane_kept_note(self.slot, "decider turn",
                                                 log=[])
                            if slots.lane_kept() else
                            D.release(self.slot, "decider turn"))
        self.grant = None

    # -- asking
    def _cf(self, c: dict) -> dict | None:
        """The content-free read of one ordered choice (its template: the
        question and options, not the state), made once per process. It is
        made BEFORE the turn's state is placed on the slot (facts() asks
        every template it needs first)."""
        if not CONTEXTUAL:
            return None
        k = _cf_key(hashlib.sha1(json.dumps([c["question"], c["options"]]).encode()
                                 ).hexdigest())
        with _CF_LOCK:
            if k in TEMPLATE_CF:
                return TEMPLATE_CF[k]
        a = D.ask_one(D.NEUTRAL_STATE, c, slot=self.slot, post=self._post,
                      upstream=self._upstream)
        p = D.meaning_probs(a, c)
        with _CF_LOCK:
            TEMPLATE_CF[k] = p
        return p

    def ask(self, name: str, q: dict) -> dict:
        """One semantic question, read in its FORM: {name, form, probs (by
        meaning), answer | None, tie, reads, ms}. Raises
        DeciderUnavailable."""
        self._open()
        t0 = time.time()
        form = _form(name)
        if form == "yn" and q["kind"] == "yes_no":
            a = D.ask_one(self.state or D.NEUTRAL_STATE, q, slot=self.slot,
                          post=self._post, upstream=self._upstream)
            cs, dists = [q], [dict(a["probs"])]
        else:
            cs = [D.as_choice(q, o) for o in D.orders_for(q, ORDERS)]
            cfs = [self._cf(c) if form == "mc_avg_cc" else None for c in cs]
            dists = []
            for c, cf in zip(cs, cfs):
                a = D.ask_one(self.state or D.NEUTRAL_STATE, c,
                              slot=self.slot, post=self._post,
                              upstream=self._upstream)
                p = D.meaning_probs(a, c)
                dists.append(D.contextual(p, cf) if cf else p)
        probs = D.average(dists)
        dec = D.decision(probs)
        rec = {"name": name, "form": form, "probs": {k: round(v, 4)
                                       for k, v in probs.items()},
               "answer": dec["answer"], "tie": dec["tie"],
               "reads": len(cs), "ms": round((time.time() - t0) * 1000, 1)}
        self.ms_questions += rec["ms"]
        self.answers.append(rec)
        return rec

    def _prime(self, named: list[tuple[str, dict]]) -> None:
        """Every content-free read these (name, question)s need, before the
        state is placed."""
        self._open()
        for n, q in named:
            if _form(n) == "mc_avg_cc":
                for o in D.orders_for(q, ORDERS):
                    self._cf(D.as_choice(q, o))

    # -- THE TYPED API (the default readout)
    def decide(self, questions: list[dict], *, rules: dict | None = None,
               render=None, exclude=(), extra: dict | None = None
               ) -> list[dict]:
        """Typed questions ({type: noul | choice | score, name, text, ...};
        decider_bonsai.q_noul / q_choice / q_score) over this Turn's state
        on its slot -- the state prefix placed once, each question appended
        after it and read at its answer position (decider_bonsai.read: two
        orders, letters, no label-prior division unless the question asks
        for the measured content-free veto). Each answer is Jev's
        (decider_bonsai.read) plus `tier` (this module's THRESHOLDS for its
        question set) and `decision_id`; every decision is logged
        (DECISIONS) with `rules[name]` beside it. Raises
        DeciderUnavailable."""
        self._open()
        out = []
        if D.batch_on():
            # BATCHED READS (mcp/decider_batch.py): every question of this
            # call in ONE engine request, the prefix kept resident for the
            # Turn's next call and freed at close(). A question that needs the
            # content-free prior keeps the old loop (its prior is read BEFORE
            # the state is placed, question by question).
            qs = [D.typed(q) for q in questions]
            if not any(q.get("prior") == "content_free" and CONTEXTUAL
                       for q in qs):
                rs = self._batch().read_many(
                    self.state or D.NEUTRAL_STATE, qs, slot=self.slot,
                    post=self._post, upstream=self._upstream, render=render,
                    exclude=exclude)
                return [self._after_read(q, r, rules, extra)
                        for q, r in zip(qs, rs)]
        for q in questions:
            q = D.typed(q)
            pre = [c for c in D.rendered_orders(q)] \
                if q.get("prior") == "content_free" and CONTEXTUAL else []
            for c in pre:                       # before the state is placed
                self._typed_cf(c)
            r = D.read(self.state or D.NEUTRAL_STATE, q, slot=self.slot,
                       post=self._post, upstream=self._upstream,
                       render=render, exclude=exclude,
                       prior_for=self._typed_cf if pre else None)
            out.append(self._after_read(q, r, rules, extra))
        return out

    def _after_read(self, q: dict, r: dict, rules, extra) -> dict:
        """One typed read's answer: its tier, the decision logged, the Turn's
        record."""
        r["tier"] = tier(r)
        rule = (rules or {}).get(q["name"])
        f = flat(r)
        r["decision_id"] = self._log_decision(f, rule, extra)
        f["decision_id"] = r["decision_id"]
        self.ms_questions += f["ms"]
        self.answers.append({k: f[k] for k in (
            "name", "type", "answer", "argmax", "tie", "tier",
            "confidence", "noul", "score", "probs", "disagreement",
            "argmax_agree", "label_mass_min", "prior", "reads",
            "readout", "decision_id", "ms")})
        return r

    def _batch(self):
        """This Turn's burst of batched reads (decider_batch): its calls keep
        the state's prefix resident, close() frees it."""
        if getattr(self, "_burst", None) is None:
            import decider_batch as B
            self._burst = B.burst(self.state, upstream=self._upstream)
        return self._burst

    def _typed_cf(self, c: dict) -> dict | None:
        """The content-free read of one printed order (typed keys), once per
        process -- the package veto's prior, kept as measured."""
        if not CONTEXTUAL:
            return None
        k = _cf_key("typed:" + hashlib.sha1(json.dumps(
            [c["question"], c["options"]]).encode()).hexdigest())
        with _CF_LOCK:
            if k in TEMPLATE_CF:
                return TEMPLATE_CF[k]
        a = D.ask_one(D.NEUTRAL_STATE, c, slot=self.slot, post=self._post,
                      upstream=self._upstream)
        p = D.meaning_probs(a, c)
        with _CF_LOCK:
            TEMPLATE_CF[k] = p
        return p

    def _log_decision(self, f: dict, rule=None, extra: dict | None = None
                      ) -> str:
        """One DECISIONS row for a decision `f` (flat(): a typed read, or
        the legacy readout's equivalent): ids and labels, never text.
        Returns the decision id."""
        did = uuid.uuid4().hex[:16]
        row = {"row": "decision", "v": DECISION_LOG_VERSION, "id": did,
               "ts": round(time.time(), 3), "model": D.model_name(),
               "template": D.TEMPLATE_VERSION,
               "readout": f.get("readout") or READOUT,
               "account": self.account or "",
               "conversation": self.key or "", "request": self.request or "",
               "corpus_turn": _corpus_turn_of(self.request),
               "kind": self.kind,
               "state": {"sha": self.state_sha,
                         "tokens": self.state_info.get("tokens"),
                         "cut": bool(self.state_info.get("cut"))},
               "question": {"type": f.get("type"), "name": f.get("name"),
                            "keys": f.get("keys")},
               # + raw / variant_mass / word_mass (2026-09-30, additive):
               # each order's distribution as read, before the temperature
               # (bench/decider/fit_temperature.py refits from it), and
               # where the rest of its top K went (decider_bonsai.
               # case_variants)
               "orders": [{k: v for k, v in (
                   ("printed", o.get("printed")), ("p", o.get("probs")),
                   ("raw", o.get("raw")), ("label_mass", o.get("label_mass")),
                   ("variant_mass", o.get("variant_mass")),
                   ("word_mass", o.get("word_mass")))
                   if k in ("printed", "p", "label_mass") or v is not None}
                   for o in f.get("orders") or []],
               "p": f.get("probs"), "pick": f.get("answer"),
               "argmax": f.get("argmax"), "tie": f.get("tie"),
               "tier": f.get("tier") or "untuned",
               "margin": f.get("margin"),
               "disagreement": f.get("disagreement"),
               "argmax_agree": f.get("argmax_agree"),
               "label_mass_min": f.get("label_mass_min"),
               "prior": f.get("prior"),
               "excluded": f.get("excluded") or [],
               "rule": _jsonable(rule)}
        # + ms / reads / processed_tokens / prompt_tokens (2026-09-30, the
        # JJAVA page's latency per read and per burst): what this decision
        # cost, from its own reads. Additive: v2 readers ignore them.
        for k in ("confidence", "noul", "score", "tie_band", "ms", "reads",
                  "processed_tokens", "prompt_tokens", "temperature",
                  "variant_mass_max", "word_mass_max"):
            if f.get(k) is not None:
                row[k] = f[k]
        row.update(extra or {})
        self.decision_ids.append(did)
        _pending(self.request, did)
        _append_to(DECISIONS, row)
        return did

    def _log_legacy(self, name: str, qtype: str, probs: dict, pick,
                    rule=None, extra: dict | None = None) -> str:
        """The legacy readout's decisions, logged the same way (one read's
        distribution, no orders' diagnostics), so the two readouts compare
        on one log."""
        yn = {"yes": "true", "no": "false"}
        if qtype == "noul":                     # the typed keys
            probs = {yn.get(k, k): v for k, v in (probs or {}).items()}
            pick = yn.get(pick, pick)
        probs = dict(probs or {})
        return self._log_decision({
            "type": qtype, "name": name, "keys": sorted(probs),
            "probs": probs, "answer": pick, "argmax": pick,
            "confidence": (round(D.confidence(probs), 6)
                           if qtype != "noul" and probs else None),
            "noul": probs.get("true") if qtype == "noul" else None,
            "readout": "legacy/" + _form(name)}, rule, extra)

    # -- the fixed questions
    def facts(self) -> dict:
        """{build_intent, phase, reads_package, scratch_write}: each
        {value, source: decider | rule | tie->rule | off, p?}. Rules are the
        fallback and the comparison; disagreements are logged."""
        if self._facts is not None:
            return self._facts
        rules = self._rules()
        qs: list[tuple[str, dict]] = [("phase", D.choice(*PHASE_Q))]
        if self.kind == "user":
            qs.insert(0, ("build_intent", D.yes_no(BUILD_INTENT_Q)))
        if self.kind == "step":
            if ASK_READS_PACKAGE:
                qs.append(("reads_package", D.yes_no(READS_PACKAGE_Q)))
            if ASK_SCRATCH and self._terminal_step():
                qs.append(("scratch_write", D.yes_no(SCRATCH_Q)))
        out: dict = {}
        if not self.on or self.kind == "other":
            for n, _ in qs:
                out[n] = {"value": rules.get(n), "source": "off"
                          if not self.on else "rule"}
            self._facts = out
            return out
        try:
            if typed_readout():
                self._facts_typed([n for n, _ in qs], rules, out)
                qs_legacy: list = []
            else:
                self._prime(qs)
                qs_legacy = qs
            for n, q in qs_legacy:
                r = self.ask(n, q)
                lp, la = r["probs"], r["answer"]
                if n == "phase":                 # letters -> phase names
                    lp = {PHASES[D.LETTERS.index(k)]: v
                          for k, v in lp.items()}
                    la = PHASES[D.LETTERS.index(la)] if la else None
                self._log_legacy(n, "noul" if q["kind"] == "yes_no"
                                 else "choice", lp, la, rules.get(n))
                if r["tie"] and n in DECIDER_ON_TIE:
                    # The decider stays the answer (operator: the rule is
                    # the fallback only when the decider is unavailable):
                    # the larger side of the averaged distribution, the
                    # tie recorded.
                    out[n] = {"value": max(r["probs"], key=r["probs"].get)
                              == "yes", "source": "decider (tie)",
                              "p": r["probs"]}
                    continue
                if r["tie"]:
                    out[n] = {"value": rules.get(n), "source": "tie->rule",
                              "p": r["probs"]}
                    continue
                v = r["answer"]
                if n == "phase":
                    v = PHASES[D.LETTERS.index(v)] if v in D.LETTERS else v
                else:
                    v = v == "yes"
                out[n] = {"value": v, "source": "decider", "p": r["probs"]}
        except D.DeciderUnavailable as e:
            self.failure = e.facts()
        for n, _ in qs:
            if n not in out:
                out[n] = {"value": rules.get(n), "source": "rule"}
        if self.kind == "step" and "reads_package" not in out:
            out["reads_package"] = {"value": rules.get("reads_package"),
                                    "source": "rule (a path fact)"}
        for n, v in out.items():
            if v["source"] == "decider" and n in rules and \
                    rules[n] is not None and _disagree(n, v["value"],
                                                       rules[n]):
                self._log(n, v["value"], rules[n], v.get("p"))
        self._facts = out
        return out

    def _fact_q(self, n: str) -> dict:
        """The fact questions, typed (MODES)."""
        if n == "phase":
            return D.q_choice(n, PHASE_Q[0], PHASE_Q[1], keys=PHASES)
        return D.q_noul(n, {"build_intent": BUILD_INTENT_Q,
                            "reads_package": READS_PACKAGE_Q,
                            "scratch_write": SCRATCH_Q}[n])

    def _facts_typed(self, names: list[str], rules: dict, out: dict) -> None:
        """facts() on the typed readout: each answer's value, its source,
        its distribution, its confidence (a choice) or noul, and its tier
        into `out`. A tie (TIE_BAND) falls back to the rule, except for a
        DECIDER_ON_TIE question, which keeps the decider's larger side (the
        operator: the rule is the fallback only when the decider is
        unavailable); a LOW tier (THRESHOLDS, once tuned) falls back to the
        rule; a medium one acts and says so. Raises DeciderUnavailable."""
        for n in names:
            f = flat(self.decide([self._fact_q(n)], rules=rules)[0])
            if f["type"] == "noul":
                p = {"yes": f["probs"]["true"], "no": f["probs"]["false"]}

                def val(k):
                    return k == "true"
            else:
                p = dict(f["probs"])

                def val(k):
                    return k
            if f["tie"] and n in DECIDER_ON_TIE:
                out[n] = {"value": val(f["argmax"]),
                          "source": "decider (tie)", "p": p}
            elif f["tie"] or f["tier"] == "low":
                what = "tie" if f["tie"] else "low"
                out[n] = {"value": rules.get(n), "source": f"{what}->rule",
                          "p": p}
            else:
                out[n] = {"value": val(f["answer"]), "source": "decider"
                          + (" (medium)" if f["tier"] == "medium" else ""),
                          "p": p}
            out[n].update(decision_id=f["decision_id"], tier=f["tier"],
                          disagreement=f["disagreement"])
            if f["confidence"] is not None:
                out[n]["confidence"] = f["confidence"]

    def _terminal_step(self) -> bool:
        i = len(self.messages) - 1
        while i >= 0 and self.messages[i].get("role") == "tool":
            i -= 1
        calls = (self.messages[i].get("tool_calls") or []) if i >= 0 else []
        return any(((c or {}).get("function") or {}).get("name") in
                   ("terminal", "bash", "shell", "run_command",
                    "execute_command", "exec_command")
                   for c in calls)

    def _rules(self) -> dict:
        """The fallback rules' answers (the same functions the bench
        compared). Each is None when it raised."""
        out: dict = {}
        try:
            import route
            if self.kind == "user":
                out["build_intent"] = bool(route.work_intent(
                    self.user_text))
        except Exception:                                        # noqa: BLE001
            out["build_intent"] = None
        try:
            import skill_classify
            ph = set((skill_classify.request_signals(
                self.messages, None, self.client_tools).get("phases")
                or {}).keys()) & set(PHASES)
            out["phase"] = sorted(ph)[0] if len(ph) == 1 else (
                sorted(ph) if ph else None)
        except Exception:                                        # noqa: BLE001
            out["phase"] = None
        if self.kind == "step":
            try:
                import skill_select
                i = len(self.messages) - 1
                while i >= 0 and self.messages[i].get("role") == "tool":
                    i -= 1
                calls = self.messages[i].get("tool_calls") or []
                out["reads_package"] = any(
                    skill_select.probed_packages(c) for c in calls)
            except Exception:                                    # noqa: BLE001
                out["reads_package"] = None
            out["scratch_write"] = None      # no rule sees a terminal write
        return out

    # -- packages: weak detections only
    def confirm_packages(self, detected: dict, descriptions: dict | None
                         = None) -> dict:
        """{package: {keep, source, p?}} for every detection; only WEAK ones
        (weak_detections) are asked; a strong one is kept as a fact."""
        weak = weak_detections(self.messages, detected)
        out = {p: {"keep": True, "source": "fact"} for p in detected
               if p not in weak}
        for p in weak:
            if not self.on:
                out[p] = {"keep": True, "source": "off"}
                continue
            try:
                d = (descriptions or {}).get(p)
                text = PACKAGE_Q.format(name=p, desc=f" ({d})" if d else "")
                if typed_readout():
                    # noul with the content-free prior: the one place it was
                    # measured to help (a veto; decider_bonsai PRIORS).
                    f = flat(self.decide([D.q_noul(
                        f"package:{p}", text, prior="content_free")],
                        rules={f"package:{p}": True})[0])
                    pp = {"yes": f["probs"]["true"],
                          "no": f["probs"]["false"]}
                    if f["answer"] is None:     # a tie, or a low tier
                        out[p] = {"keep": True, "source": (
                            "tie->rule" if f["tie"] else "low->rule"),
                            "p": pp, "decision_id": f["decision_id"],
                            "tier": f["tier"]}
                    else:
                        out[p] = {"keep": f["answer"] == "true",
                                  "source": "decider" + (
                                      " (medium)" if f["tier"] == "medium"
                                      else ""), "p": pp,
                                  "decision_id": f["decision_id"],
                                  "tier": f["tier"]}
                        if not out[p]["keep"]:
                            self._log(f"package:{p}", False, True, pp)
                    continue
                q = D.yes_no(text)
                self._prime([(f"package:{p}", q)])
                r = self.ask(f"package:{p}", q)
                self._log_legacy(f"package:{p}", "noul", r["probs"],
                                 r["answer"], True)
                if r["tie"]:
                    out[p] = {"keep": True, "source": "tie->rule",
                              "p": r["probs"]}
                else:
                    out[p] = {"keep": r["answer"] == "yes",
                              "source": "decider", "p": r["probs"]}
                    if not out[p]["keep"]:
                        self._log(f"package:{p}", False, True, r["probs"])
            except D.DeciderUnavailable as e:
                self.failure = e.facts()
                out[p] = {"keep": True, "source": "rule"}
        return out

    # -- the selector's hook
    def pick(self, area: str, candidates: list[dict],
             hard_evidence: bool = False) -> dict:
        """Ordered picks among `candidates` ([{name, description}]): one
        CHOICE "Which of these fits what the agent is doing now?" with the
        candidates and, unless `hard_evidence`, "None of these fits." --
        asked in ORDERS orders so none is always last -- repeated with the
        pick removed until none wins, a tie, or the candidates run out.
        Returns {picks: [name], rounds: [...], source}. The content-free
        read of a remaining set is the full set's, restricted to the
        remaining options and renormalised (the per-option prior; one read
        per template, not per round -- an approximation, stated)."""
        names = [c["name"] for c in candidates]
        texts = {c["name"]: (c.get("description") or c["name"])
                 for c in candidates}
        rec = {"area": area, "picks": [], "rounds": [], "source": "decider",
               "hard_evidence": bool(hard_evidence)}
        if not self.on or not names:
            rec["source"] = "off" if not self.on else "empty"
            return rec
        left = list(names)
        try:
            while left:
                opts = [texts[n] for n in left]
                keys = list(left)
                if not hard_evidence:
                    opts.append(NONE_OPTION)
                    keys.append(None)
                if typed_readout() and len(opts) >= 2:
                    tk = [k if k is not None else "none" for k in keys]
                    f = flat(self.decide([D.q_choice(
                        f"pick:{area}", FIT_Q, opts, keys=tk,
                        none=None if hard_evidence else len(tk) - 1)])[0])
                    rnd = {"options": tk, "probs": f["probs"],
                           "confidence": f["confidence"], "tie": f["tie"],
                           "tier": f["tier"],
                           "disagreement": f["disagreement"],
                           "decision_id": f["decision_id"], "ms": f["ms"]}
                    rec["rounds"].append(rnd)
                    if f["answer"] is None:     # a tie, or a low tier
                        rnd["pick"] = None
                        break
                    rnd["pick"] = f["answer"]
                    if f["answer"] == "none":
                        break
                    rec["picks"].append(f["answer"])
                    left.remove(f["answer"])
                    continue
                if typed_readout():
                    # One candidate under hard evidence: nothing to choose.
                    rec["rounds"].append({"options": keys, "pick": keys[0],
                                          "only": True})
                    rec["picks"].append(keys[0])
                    left.remove(keys[0])
                    continue
                q = D.choice(FIT_Q, opts)
                full = D.choice(FIT_Q, [texts[n] for n in names]
                                + ([] if hard_evidence else [NONE_OPTION]))
                self._prime([("pick", full)])
                t0 = time.time()
                dists = []
                for o in D.orders_for(q, ORDERS):
                    c = D.as_choice(q, o)
                    a = D.ask_one(self.state or D.NEUTRAL_STATE, c,
                                  slot=self.slot, post=self._post,
                                  upstream=self._upstream)
                    p = D.meaning_probs(a, c)
                    cf = self._restricted_cf(full, q)
                    dists.append(D.contextual(p, cf) if cf else p)
                probs = D.average(dists)
                dec = D.decision(probs)
                ms = round((time.time() - t0) * 1000, 1)
                self.ms_questions += ms
                byname = {(keys[D.LETTERS.index(k)] or "none"): round(v, 4)
                          for k, v in probs.items()}
                rnd = {"options": [k or "none" for k in keys],
                       "probs": byname, "tie": dec["tie"], "ms": ms}
                rec["rounds"].append(rnd)
                if dec["tie"]:
                    rnd["pick"] = None
                    break
                got = keys[D.LETTERS.index(dec["answer"])]
                rnd["pick"] = got or "none"
                if got is None:
                    break
                rec["picks"].append(got)
                left.remove(got)
        except D.DeciderUnavailable as e:
            self.failure = e.facts()
            rec["source"] = "unavailable"
        self.answers.append({"name": f"pick:{area}", "picks": rec["picks"],
                             "rounds": len(rec["rounds"])})
        return rec

    def _restricted_cf(self, full: dict, q: dict) -> dict | None:
        """The full option set's content-free prior (averaged over orders),
        restricted to q's options, in q's letters."""
        if not CONTEXTUAL:
            return None
        per = []
        for o in D.orders_for(full, ORDERS):
            c = D.as_choice(full, o)
            k = _cf_key(hashlib.sha1(json.dumps([c["question"], c["options"]])
                                     .encode()).hexdigest())
            with _CF_LOCK:
                cf = TEMPLATE_CF.get(k)
            if cf:
                per.append(cf)
        if not per:
            return None
        prior = D.average(per)                    # by full's letters
        by_text = {full["options"][D.LETTERS.index(k)]: v
                   for k, v in prior.items()}
        return {lab: by_text.get(t, 1.0 / len(q["options"]))
                for lab, t in zip(q["labels"], q["options"])}

    # -- the selector's numbered question (mcp/skill_match.py ask_rounds)
    def choose(self, area: str, options: list[dict], turn: dict
               ) -> tuple[str | None, dict]:
        """skill_match's decider contract: (label, {label: p}) for one round
        of one question. `options` [{letter, id, name, text}] are IDENTICAL
        in every round of the question; `turn` carries the round's question
        (with its "already chosen" suffix) and `exclude` (labels not to be
        answered: those chosen, and "none" under hard evidence's first
        round).

        THE RENDERING, EACH ROUND EXTENDING THE LAST (2026-09-27 fix):
        system / the turn's state / the OPTIONS in their own user message /
        then, for every earlier round of THIS question (same area, same
        options), its question and an assistant turn "Answer: <label>" (the
        label the caller took: the argmax of the returned distribution over
        the allowed labels) / the round's question / the "Answer:" prefill.
        Round k+1's prompt is round k's with " <label>", an end of turn and
        the new question appended (the served template keeps a past
        assistant turn's empty think block, so it renders byte for byte as
        the prefill did: mcp/test_decide_turn.py renders two rounds through
        mcp/fixtures/bonsai_chat_template.jinja). Before this, each round
        REPLACED the last user message, and live every round after the first
        re-processed ~1,440 tokens (~2.5 s): the hybrid model can only roll
        back to a context checkpoint, and none covered the options. Now the
        slot's sequence is extended; at worst the one sampled token differs
        from the label and the server restores its checkpoint 4 tokens
        before the prompt's end (server-context.cpp checkpoint_offsets).

        The options are PRINTED in a rotation (never leaving "None of these"
        last; research doc 2.5) that depends on the NUMBER of options only,
        so the label prior below is one read per option count per process.
        Each option keeps its own label. Labels are skill_match.LABELS.

        CALIBRATION (FORM["choose"] = "rotated_idprior", 2026-09-27; the
        diagnosis: flat distributions on agent steps picked "A" over "None
        of these", 0.28 vs 0.24): the LABEL PRIOR -- the same template, the
        content-free state (NEUTRAL_STATE) AND every option's text replaced
        by NEUTRAL_STATE, in the same printed order -- is divided out over
        the allowed labels (Calibrate Before Use, W = diag(p_cf)^-1; the
        prior is over the option-ID tokens, which PriDe 2309.03882 found to
        be the bias's cause). The option texts are blanked on purpose: with
        "None of these" still readable, an empty state makes NONE the right
        answer, and dividing by that would push every question away from
        NONE -- the opposite of the fix. One read per printed label order
        per process (TEMPLATE_CF), made BEFORE this turn's state is placed
        when the caller primes it (prime_choose) -- otherwise it costs this
        slot's state prefix once. "rotated" turns it off. MEASURED: nothing
        yet (the pagoda run records both picks: raw_pick and pick).

        Within TIE_BAND of each other the two best are a tie: the label
        returned is None (the distribution is still returned; the caller
        decides what a tie does). Excluded labels get 0."""
        t0 = time.time()
        if not self.on:
            raise DeciderUnavailable_off()
        if typed_readout():
            return self._choose_typed(area, options, turn)
        self._open()
        labels = [o["letter"] for o in options]
        ex = set(turn.get("exclude") or ())
        printed = _rotation(options)
        opt_msg = _options_msg(printed)
        q = {"kind": "choice", "question": turn.get("question") or FIT_Q,
             "labels": labels, "options": [o["text"] for o in options]}
        key = hashlib.sha1((area + "\n" + opt_msg).encode()).hexdigest()
        history = self._rounds.setdefault(key, [])
        prior = (self._id_prior(printed)
                 if FORM["choose"] == "rotated_idprior" and CONTEXTUAL
                 else None)
        msgs = _choose_msgs(self.state or D.NEUTRAL_STATE, opt_msg, history,
                            q["question"])
        a = D.ask_one(self.state or D.NEUTRAL_STATE, q, slot=self.slot,
                      post=self._post, upstream=self._upstream, msgs=msgs)
        allowed = {lab: p for lab, p in a["probs"].items() if lab not in ex}
        z = sum(allowed.values()) or 1.0
        raw = {lab: p / z for lab, p in allowed.items()}
        cal = (D.contextual(raw, {lab: prior.get(lab, 0.0) for lab in raw})
               if prior else raw)
        dec = D.decision(cal) if cal else {"answer": None, "tie": False}
        dist = {lab: round(cal.get(lab, 0.0), 6) for lab in labels}
        took = max(cal, key=cal.get) if cal else None
        history.append((q["question"], took))
        ms = round((time.time() - t0) * 1000, 1)
        self.ms_questions += ms
        self.answers.append({
            "name": f"choose:{area}", "kind": turn.get("kind"),
            "options": len(options), "excluded": len(ex),
            "round": len(history), "answer": dec["answer"],
            "tie": dec["tie"], "pick": took,
            "raw_pick": max(raw, key=raw.get) if raw else None,
            "calibration": FORM["choose"] if prior else "none",
            "ms": ms, "processed_tokens": a.get("processed_tokens"),
            "prompt_tokens": a.get("prompt_tokens")})
        self.answers[-1]["decision_id"] = self._log_legacy(
            f"choose:{area}", "choice", dist, dec["answer"], None,
            {"option_ids": _option_ids(options), "round": len(history),
             "raw_pick": self.answers[-1]["raw_pick"]})
        return dec["answer"], dist

    def _choose_typed(self, area: str, options: list[dict], turn: dict
                      ) -> tuple[str | None, dict]:
        """choose() on the typed readout: one CHOICE (MODES) read in TWO
        ORDERS -- the options' own message printed with POSITIONAL letters
        in each order ("None of these" in the middle of both where there are
        three or more options), averaged -- and NO label prior divided out
        (DECIDER-RESEARCH.md 4.2 item 3; decider_bonsai READOUT).

        THE ROUNDS STILL EXTEND THE SLOT, per order: each order's rendering
        is system / state / its options / every earlier round of the
        question (its question, "Answer: <that order's letter for the label
        taken>") / this round's question / the prefill, so round k+1 in an
        order extends round k in that order. The two orders share the state
        prefix; each switch between them re-reads its options block and its
        rounds from the state's end (the cost of the second order: the
        options message plus the rounds, per read -- UNMEASURED on the GPU).

        The caller's letters are the KEYS (the distribution and `exclude`
        are in them); a question's letter references -- skill_match's
        "Already chosen: B; choose another, or D (none of these)." -- are
        rewritten into each order's letters (_reletter). The label returned
        is None on a tie (TIE_BAND) or a LOW tier (THRESHOLDS for choose /
        choose:<area>, once tuned): the caller's fallback decides."""
        labels = [o["letter"] for o in options]
        ex = [lab for lab in (turn.get("exclude") or ()) if lab in labels]
        question = turn.get("question") or FIT_Q
        none = next((i for i, o in enumerate(options)
                     if "id" in o and o.get("id") is None), None)
        q = D.q_choice(f"choose:{area}", question,
                       [o["text"] for o in options], keys=labels, none=none)
        hkey = hashlib.sha1((area + "\n" + json.dumps(
            [[o["letter"], o["text"]] for o in options])).encode()
        ).hexdigest()
        history = self._rounds.setdefault(hkey, [])
        state = self.state or D.NEUTRAL_STATE

        def render(c: dict) -> list[dict]:
            to_printed = {key: lab for lab, key in c["meaning"].items()}
            printed = [{"letter": lab, "text": t}
                       for lab, t in zip(c["labels"], c["options"])]
            return _choose_msgs(
                state, _options_msg(printed),
                [(_reletter(qq, to_printed), to_printed.get(took))
                 for qq, took in history],
                _reletter(question, to_printed))
        f = flat(self.decide([q], render=render, exclude=ex, extra={
            "option_ids": _option_ids(options),
            "round": len(history) + 1, "choose_kind": turn.get("kind")})[0])
        took = f["argmax"]
        history.append((question, took))
        dist = {lab: round(f["probs"].get(lab, 0.0), 6) for lab in labels}
        rec = self.answers[-1]
        rec.update({
            "name": f"choose:{area}", "kind": turn.get("kind"),
            "options": len(options), "excluded": len(ex),
            "round": len(history), "pick": took, "raw_pick": took,
            "calibration": "none (two orders averaged)",
            "orders": [o["printed"] for o in f["orders"]],
            "processed_tokens": f["processed_tokens"],
            "prompt_tokens": max((o.get("prompt_tokens") or 0)
                                 for o in f["orders"])})
        return f["answer"], dist

    def _id_prior(self, printed: list[dict]) -> dict | None:
        """The label prior for this printed label order (see choose), read
        once per process."""
        blank = [dict(o, text=D.NEUTRAL_STATE) for o in printed]
        opt_msg = _options_msg(blank)
        k = _cf_key("idprior:" + hashlib.sha1(opt_msg.encode()).hexdigest())
        with _CF_LOCK:
            if k in TEMPLATE_CF:
                return TEMPLATE_CF[k]
        labels = [o["letter"] for o in printed]
        q = {"kind": "choice", "question": FIT_Q, "labels": labels,
             "options": [D.NEUTRAL_STATE] * len(labels)}
        t0 = time.time()
        a = D.ask_one(D.NEUTRAL_STATE, q, slot=self.slot, post=self._post,
                      upstream=self._upstream,
                      msgs=_choose_msgs(D.NEUTRAL_STATE, opt_msg, [], FIT_Q))
        p = dict(a["probs"])
        with _CF_LOCK:
            TEMPLATE_CF[k] = p
        self.answers.append({"name": "choose:label_prior",
                             "options": len(labels),
                             "ms": round((time.time() - t0) * 1000, 1)})
        return p

    def prime_choose(self, counts) -> None:
        """Read the label priors for these option counts (the last option
        "None of these") now, before the state is placed -- for a caller
        that knows its question sizes early."""
        if not self.on or FORM["choose"] != "rotated_idprior" \
                or not CONTEXTUAL:
            return
        self._open()
        for n in counts:
            opts = [{"letter": lab, "id": i, "text": ""} for i, lab in
                    enumerate(D.LETTERS[:max(int(n) - 1, 1)])]
            opts.append({"letter": D.LETTERS[len(opts)], "id": None,
                         "text": ""})
            self._id_prior(_rotation(opts))

    # -- the record
    def _log(self, question: str, decider, rule, p) -> None:
        rec = {"ts": round(time.time(), 3), "account": self.account or "",
               "conversation": self.key or "", "request": self.request or "",
               "kind": self.kind, "question": question,
               "decider": decider, "rule": rule, "p": p}
        self.disagreements.append(rec)
        _append(rec)

    def record(self) -> dict:
        typed = typed_readout()
        return {"on": self.on, "template": D.TEMPLATE_VERSION,
                "readout": D.READOUT_VERSION if typed else "legacy",
                "kind": self.kind, "state": self.state_info,
                "slot": self.slot, "orders": ORDERS,
                "form": MODES if typed else FORM,
                "decisions": list(self.decision_ids),
                "contextual": CONTEXTUAL, "tie_band": D.tie_band(),
                # PER MODEL (decider_bonsai THE MODEL PROFILE): what was
                # measured on the model that answered, and what was not.
                "model_profile": D.profile_status(),
                "thresholds": sorted(THRESHOLDS.get(D.model_name())
                                     or THRESHOLDS.get(D.canonical_model())
                                     or {}) or "untuned",
                "facts": self._facts, "answers": self.answers,
                "disagreements": len(self.disagreements),
                "failure": self.failure, "release": self.release_rec,
                "ms_questions": round(self.ms_questions, 1),
                "ms_total": round((time.time() - self.t0) * 1000, 1)}


def _form(name: str) -> str:
    """FORM of a question name (package:<p> -> package)."""
    return FORM.get(name.split(":", 1)[0], "mc_avg")


def _disagree(name: str, decider, rule) -> bool:
    if name == "phase":
        return decider not in (rule if isinstance(rule, list) else [rule])
    return bool(decider) != bool(rule)


_LOG_LOCK = threading.Lock()


def _append(rec: dict) -> None:
    """Disagreements, one JSON line each: ids and labels, never text."""
    _append_to(LOG, rec)


def _append_to(path: str, rec: dict) -> None:
    """One JSON line; a failed write costs the row, never the request."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOG_LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


def _jsonable(v):
    """A rule's answer as logged: bool, str, number, a list of them, None."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple, set)):
        return sorted(str(x) for x in v)
    return str(v)


def _option_ids(options: list[dict]) -> list[str]:
    """A choose() question's option ids (skill ids, a client's tool names,
    judge_stop's keys; "none" for the none option): labels, not text."""
    return [str(o.get("id")) if o.get("id") is not None else "none"
            for o in options]


_LETTER_REF = re.compile(r"\b([A-Z])\b")


def _reletter(question: str, to_printed: dict) -> str:
    """The question with its single-letter option references (after its
    first "?": skill_match's round suffixes) rewritten into one printed
    order's letters, all at once."""
    head, sep, tail = question.partition("?")
    if not sep:
        return question
    return head + sep + _LETTER_REF.sub(
        lambda m: to_printed.get(m.group(1), m.group(1)), tail)


# ------------------------------------------------------------ the join -----
# THE CORPUS JOIN (DECIDER-RESEARCH.md 4.2 item 1). The decider runs in the
# proxy's prepare(), BEFORE _run_turn mints the corpus turn id; both know the
# request by the ledger's turn key (the Turn's `request`). So each decision
# is remembered under its request key here, and join_corpus -- ONE call in
# proxy._run_turn right after corpus.new_turn() -- writes a "join" row naming
# the corpus turn and every decision of that request so far; a decision made
# after the join (judge_stop, later in the same request) carries corpus_turn
# itself. JOIN_MEMORY bounds the two maps (a request that never reaches
# _run_turn -- a refusal after the decider ran -- would otherwise be kept
# forever); it is a memory guard, it changes no decision.
JOIN_MEMORY = 4096
_JOIN_LOCK = threading.Lock()
_PENDING: "collections.OrderedDict[str, list[str]]" = collections.OrderedDict()
_CORPUS: "collections.OrderedDict[str, str]" = collections.OrderedDict()


def _pending(request: str | None, did: str) -> None:
    if not request:
        return
    with _JOIN_LOCK:
        _PENDING.setdefault(request, []).append(did)
        _PENDING.move_to_end(request)
        while len(_PENDING) > JOIN_MEMORY:
            _PENDING.popitem(last=False)


def _corpus_turn_of(request: str | None) -> str | None:
    if not request:
        return None
    with _JOIN_LOCK:
        return _CORPUS.get(request)


def join_corpus(request: str | None, corpus_turn: str | None, *,
                account: str = "", conversation: str | None = None) -> dict:
    """Bind this request's decisions to the corpus turn id (the proxy's
    call site: proxy._run_turn, right after `turn = corpus.new_turn()`:
        decide_turn.join_corpus((payload.get("_ledger") or {}).get(
            "turn_key"), turn, account=body.get("_account") or "")
    ). Writes one "join" row when the request has decisions pending;
    decisions made later in the request carry `corpus_turn` themselves.
    Returns {joined: n, corpus_turn}. Never raises."""
    try:
        if not request or not corpus_turn:
            return {"joined": 0, "why": "no request key or corpus turn"}
        with _JOIN_LOCK:
            ids = _PENDING.pop(request, [])
            _CORPUS[request] = corpus_turn
            _CORPUS.move_to_end(request)
            while len(_CORPUS) > JOIN_MEMORY:
                _CORPUS.popitem(last=False)
        if ids:
            _append_to(DECISIONS, {
                "row": "join", "v": DECISION_LOG_VERSION,
                "ts": round(time.time(), 3), "request": request,
                "corpus_turn": corpus_turn, "account": account or "",
                "conversation": conversation or "", "decisions": ids})
        return {"joined": len(ids), "corpus_turn": corpus_turn}
    except Exception as e:                                       # noqa: BLE001
        return {"joined": 0, "why": f"{type(e).__name__}: {e}"[:200]}


# ---------------------------------------------------------- weak packages --
_COMMENT_LINE = ("//", "#", "*", "/*", "<!--", "--")


def weak_detections(messages: list[dict], detected: dict) -> set[str]:
    """The detections resting ONLY on weak evidence: every reason is a
    `symbol`, and every occurrence of those symbols is in prose (not
    code-like, skill_packages._symbols_in) or on a comment line. A name,
    import, pin or implication, or one code-like symbol on a code line,
    makes the detection a fact."""
    import skill_packages as SP
    pieces = SP.evidence_pieces(messages)
    out = set()
    for pkg, e in (detected or {}).items():
        whys = e.get("why") or []
        if not whys or any(w.get("how") != "symbol" for w in whys):
            continue
        strong = False
        for w in whys:
            sym = w.get("what")
            for where, text in pieces:
                for s, code_like in SP._symbols_in(where, text):
                    if s != sym or not code_like:
                        continue
                    lines = [ln for ln in text.splitlines() if sym in ln]
                    if any(not ln.strip().startswith(_COMMENT_LINE)
                           for ln in lines):
                        strong = True
                        break
                if strong:
                    break
            if strong:
                break
        if not strong:
            out.add(pkg)
    return out


# ------------------------------------------------------------ build intent --
def build_intent(messages: list[dict], *, account: str = "",
                 key: str | None = None, request: str | None = None,
                 post=None, upstream=None, count=None,
                 on: bool | None = None) -> dict:
    """The build-intent question alone for the user turn `messages` end on
    (the server-tool trigger `implement`: a user turn after an answer that
    asks for something to be built runs yama_plan). {value: bool | None,
    source: decider | decider (tie) | off | unavailable, p?, failure?}; the
    caller's rule answers when value is None. Its own Turn: the transient
    slot, released at the end. Never raises."""
    t = Turn(messages, key=key, account=account, request=request,
             post=post, upstream=upstream, count=count, on=on)
    if not t.on or t.kind != "user":
        return {"value": None, "source": "off" if not t.on else t.kind}
    try:
        with t:
            if typed_readout():
                rule = None
                try:
                    import route
                    rule = bool(route.work_intent(t.user_text))
                except Exception:                                # noqa: BLE001
                    pass
                r = t.decide([t._fact_q("build_intent")],
                             rules={"build_intent": rule})[0]
            else:
                r = t.ask("build_intent", D.yes_no(BUILD_INTENT_Q))
                t._log_legacy("build_intent", "noul", r["probs"],
                              r["answer"])
    except D.DeciderUnavailable as e:
        return {"value": None, "source": "unavailable",
                "failure": e.facts()}
    except Exception as e:                                       # noqa: BLE001
        return {"value": None, "source": "unavailable",
                "failure": {"code": type(e).__name__,
                            "situation": str(e)[:200], "retryable": False}}
    if "type" in r:                                   # the typed readout
        f = flat(r)
        p = {"yes": f["probs"]["true"], "no": f["probs"]["false"]}
        base = {"p": p, "noul": f["noul"], "decision_id": f["decision_id"],
                "tier": f["tier"], "disagreement": f["disagreement"],
                "label_mass_min": f["label_mass_min"]}
        if f["tie"]:
            # build intent is the decider's on a near-tie (DECIDER_ON_TIE):
            # its larger side, recorded as such.
            return dict(base, value=f["argmax"] == "true",
                        source="decider (tie)")
        if f["tier"] == "low":
            # A tuned LOW tier: the caller's rule answers (value None).
            return dict(base, value=None, source="low")
        return dict(base, value=f["answer"] == "true", source="decider" + (
            " (medium)" if f["tier"] == "medium" else ""))
    if r["tie"]:
        return {"value": max(r["probs"], key=r["probs"].get) == "yes",
                "source": "decider (tie)", "p": r["probs"]}
    return {"value": r["answer"] == "yes", "source": "decider",
            "p": r["probs"]}


# ------------------------------------------------------- the typed door ----
def decide(state: str, questions: list[dict], *, key: str | None = None,
           account: str = "", request: str | None = None,
           rules: dict | None = None, state_info: dict | None = None,
           post=None, upstream=None, on: bool | None = None) -> dict:
    """THE TYPED API for serving code, in Jev's response shape: typed
    questions (decider_bonsai.q_noul / q_choice / q_score, or plain dicts in
    Jev's request form {type, name, instructions | text, criteria}) over one
    state -- the state prefix placed once on the transient slot, each
    question read at its answer position in two orders, the slot released
    at the end -- every decision logged with its join keys (key = the
    conversation, request = the ledger turn key).

    Returns {on, model, answers: {name: Jev's answer (noul | choice +
    probabilities + confidence | score + probabilities + confidence +
    legend) + diagnostics + tier + decision_id}, usage: {input_tokens,
    output_tokens}, record, failure?, why?}. Thresholds are the CALLER's
    (THRESHOLDS; `tier` says where an answer falls, "untuned" until one is
    accepted). Never raises: a malformed question, a repeated name, the
    decider off or unavailable leave `answers` short and say why."""
    on = enabled() if on is None else bool(on)
    out: dict = {"on": on, "model": D.model_name(), "answers": {},
                 "usage": {"input_tokens": 0, "output_tokens": 0}}
    if not on:
        out["why"] = ("the decider is off in this process (decide_turn."
                      "enabled(), YAMADORI_DECIDER)")
        return out
    t = Turn([], key=key, account=account, request=request, post=post,
             upstream=upstream, on=True, state=state or "",
             state_info=state_info or {"kind": "other",
                                       "tokens": len(state or "") // 3})
    try:
        qs = [D.typed(q) for q in questions or []]
        names = [q["name"] for q in qs]
        if len(set(names)) != len(names):
            raise D._refuse("DUPLICATE_NAME", "every question in a batch "
                            "needs its own name (the answers are keyed by "
                            "it)")
        with t:
            for q in qs:
                a = t.decide([q], rules=rules)[0]
                out["answers"][q["name"]] = a
                d = a["diagnostics"]
                out["usage"]["input_tokens"] += d.get("prompt_tokens") or 0
                out["usage"]["output_tokens"] += d.get("reads") or 0
    except D.DeciderUnavailable as e:
        out.update(why="the decider could not answer", failure=e.facts())
    except Exception as e:                                       # noqa: BLE001
        out.update(why="the decider raised",
                   failure={"code": type(e).__name__,
                            "situation": str(e)[:200], "retryable": False})
    out["record"] = t.record()
    return out


# ---------------------------------------------------- a stop with no call ---
# THE STATED STEP (operator-approved, 2026-09-27; proxy CONTINUE A STATED
# STEP). An agent step that ended finish=stop with no tool call, while the
# client offered tools, is judged here: ONE choice over what the generated
# turn DOES, in choose()'s form -- the options in their own message; on the
# typed readout read in two orders with "None of these" in the middle of
# both and no label prior (DECIDER-RESEARCH.md 4.1: the one live judge_stop
# record showed the prior flipping next_step to other); on the legacy one
# printed in the count-only rotation with the label prior divided out
# (FORM["choose"]) -- on the transient slot, released when the Turn closes. The live case: deploy_check's agent loop [echo]
# step 3 and pagoda-h3 ended on "Now let me understand the task..." with
# nothing done, and the harness ended the run.
# THE STATE: the turn's last REASONING tail (at most PREV_TOKENS, cut from
# its front: the bound decide_turn already uses for the turn before a user
# turn, from the latency table's 64-255 band), then its visible text, the
# whole cut head+tail to STATE_TOKENS like every state (the operator's
# 2,048). UNMEASURED for accuracy: no labelled set of stopped steps exists
# yet; x_yamadori.continued records raw_pick and pick for one.
STOP_Q = "What does the assistant's last message do?"
# (key, option text); the last is the "none of these" option (id None).
STOP_OPTIONS = (
    ("finished", "It reports that the task is finished."),
    ("next_step", "It says what it is about to do next, without doing it "
                  "yet."),
    ("asking", "It asks the user something."),
    ("other", NONE_OPTION),
)


def stop_state(text: str, reasoning: str = "", count=None
               ) -> tuple[str, dict]:
    """(state, info) for judge_stop: the reasoning's tail, then the text."""
    count = count or (lambda s: len(s) // 3)
    parts, rinfo = [], None
    r = (reasoning or "").strip()
    if r:
        rn = count(r)
        if rn > PREV_TOKENS:
            r = "..." + r[-int(len(r) * PREV_TOKENS / rn):]
        rinfo = {"tokens": min(rn, PREV_TOKENS), "cut": rn > PREV_TOKENS}
        parts.append(f"Assistant (the end of its thinking): {r}")
    parts.append(f"Assistant (its message): {(text or '').strip()}")
    state = "\n\n".join(parts)
    n = count(state)
    info = {"kind": "stop", "tokens": n, "cut": False, "reasoning": rinfo}
    if n > STATE_TOKENS:
        keep = int(len(state) * STATE_TOKENS / n) // 2
        state = (state[:keep] + "\n\n[... middle left out ...]\n\n"
                 + state[-keep:])
        info.update(cut=True, tokens_kept=STATE_TOKENS)
    return state, info


def judge_stop(text: str, reasoning: str = "", *, account: str = "",
               key: str | None = None, request: str | None = None,
               post=None, upstream=None, count=None,
               on: bool | None = None) -> dict:
    """What a turn that stopped with no call does: {judged, pick (a
    STOP_OPTIONS key, None on a tie), raw_pick (before the label prior),
    distribution {key: p}, tie, ms, state, release, failure?}. Never raises:
    `judged` False says why (the decider off or unavailable)."""
    t0 = time.time()
    on = enabled() if on is None else bool(on)
    out: dict = {"judged": False, "template": D.TEMPLATE_VERSION}
    if not on:
        out["why"] = ("the decider is off in this process (decide_turn."
                      "enabled(), YAMADORI_DECIDER)")
        return out
    state, info = stop_state(text, reasoning,
                             count or (lambda s: _count(s)))
    opts = [{"letter": D.LETTERS[i], "id": k if k != "other" else None,
             "name": k, "text": t}
            for i, (k, t) in enumerate(STOP_OPTIONS)]
    key_of = {o["letter"]: o["name"] for o in opts}
    turn = Turn([], key=key, account=account, request=request, post=post,
                upstream=upstream, on=True, state=state, state_info=info)
    try:
        with turn:
            lab, dist = turn.choose("stop", opts, {"kind": "stop",
                                                   "question": STOP_Q})
        rec = next((a for a in reversed(turn.answers)
                    if a.get("name") == "choose:stop"), {})
        out.update(judged=True,
                   pick=key_of.get(lab) if lab else None,
                   raw_pick=key_of.get(rec.get("raw_pick")),
                   took=key_of.get(rec.get("pick")),
                   tie=bool(rec.get("tie")),
                   distribution={key_of[k]: v for k, v in dist.items()},
                   calibration=rec.get("calibration"),
                   processed_tokens=rec.get("processed_tokens"),
                   decision_id=rec.get("decision_id"),
                   readout=rec.get("readout") or "legacy")
        if rec.get("readout"):
            # The typed readout: Jev's confidence, the question set's tier,
            # and the diagnostics (two orders, no prior).
            out.update(confidence=rec.get("confidence"),
                       tier=rec.get("tier"),
                       disagreement=rec.get("disagreement"),
                       argmax_agree=rec.get("argmax_agree"),
                       label_mass_min=rec.get("label_mass_min"),
                       orders=[[key_of.get(x) for x in o]
                               for o in rec.get("orders") or []])
    except D.DeciderUnavailable as e:
        out.update(why="the decider could not answer", failure=e.facts())
    except Exception as e:                                       # noqa: BLE001
        out.update(why="the decider raised",
                   failure={"code": type(e).__name__,
                            "situation": str(e)[:200], "retryable": False})
    out.update(state=info, slot=turn.slot, release=turn.release_rec,
               ms=round((time.time() - t0) * 1000, 1))
    return out


# ------------------------------------------------------ the selector's door --
_LOCAL = threading.local()


def current() -> "Turn | None":
    """This thread's open Turn (inside `with Turn(...)`), or None."""
    return getattr(_LOCAL, "turn", None)


def DeciderUnavailable_off() -> D.DeciderUnavailable:
    return D.DeciderUnavailable(
        "DECIDER_OFF", "no turn decider is open on this thread", False,
        "the caller: fall back to its own decider (skill_match's stub)")


def _options_msg(printed: list[dict]) -> str:
    return "OPTIONS:\n" + "\n".join(f"{o['letter']}) {o['text']}"
                                     for o in printed)


CHOOSE_TAIL = "\n\nAnswer with the label of one option."


def _choose_msgs(state: str, opt_msg: str, history: list, question: str
                 ) -> list[dict]:
    """The choose() rendering: each earlier round of the question as its
    question and "Answer: <label>", then this round's question and the
    prefill -- so a round's prompt EXTENDS the previous one."""
    out = [{"role": "system", "content": D.SYSTEM},
           {"role": "user", "content": D.STATE_HEAD + state},
           {"role": "user", "content": opt_msg}]
    for q, lab in history:
        out.append({"role": "user", "content": "QUESTION: " + q
                    + CHOOSE_TAIL})
        out.append({"role": "assistant",
                    "content": f"{D.ANSWER_LEAD} {lab}" if lab
                    else D.ANSWER_LEAD})
    out.append({"role": "user", "content": "QUESTION: " + question
                + CHOOSE_TAIL})
    out.append({"role": "assistant", "content": D.ANSWER_LEAD})
    return out


def _rotation(options: list[dict]) -> list[dict]:
    """The printed order: rotated by an amount that depends on the NUMBER
    of options only (the same in every round and for every option set of
    that size, so the label prior is one read per size), one step further
    when that would print "None of these" (id None) last."""
    n = len(options)
    if n < 2:
        return list(options)
    h = int(hashlib.sha1(str(n).encode()).hexdigest(), 16)
    r = h % n
    out = options[r:] + options[:r]
    if out[-1].get("id") is None and "id" in out[-1]:
        out = out[1:] + out[:1]
        if out[-1].get("id") is None and "id" in out[-1]:
            out = out[-1:] + out[:-1]
    return out


def __getattr__(name: str):
    """skill_match.decider() looks up `choose`: it exists only while this
    thread has an OPEN, ENABLED Turn (the proxy's skills path), so anywhere
    else -- an offline suite, the decider off -- the lookup fails and the
    selector falls back to its stub, as its contract says."""
    if name == "choose":
        t = current()
        if t is not None and t.on:
            return t.choose
    raise AttributeError(name)
