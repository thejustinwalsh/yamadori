#!/usr/bin/env python
"""A STUBBED jjava for OFFLINE suites: the real decider code (decide_turn.
Turn, decider_bonsai.read, the typed readout, the decision log) over a FAKE
model server that answers by a deterministic evidence rule.

    import decider_stub
    with decider_stub.installed():                  # pass_all (default)
        ... proxy.prepare(...) / skill_select.decide(...) ...
    with decider_stub.installed(want=decider_stub.evidence):
    with decider_stub.installed(want=my_rule):      # a suite's own rule

Operator, 2026-09-29: jjava is always in scope where skills serve; "Offline
suites use a stubbed decider" (the coordinator's brief) -- never a path that
picks skills without it. Nothing here reaches a port: the fake answers
/tokenize and the chat body in process, and the slot release is a no-op.

THE RULES, for the injector's questions only. Every OTHER question -- the
turn's facts, stage 1's per-area choice (skill_match through Turn.choose),
judge_stop -- is answered as the model server being down (a DeciderUnavailable
in the real readout), so each falls back to its own rule exactly as it did
offline before the injector: stage 1 stays the evidence stub's, and only the
injector's two questions are the stub's.

`pass_all` (the DEFAULT): every item at the top level, "yes" to inject --
jjava transparent, so a suite that tests STAGE 1 (the selection: which
skills are offered, as a body or a recall) sees stage 1's choice go in,
within the composer's caps.

`evidence`:
  skill_item Score  the TOP level when a code-shaped token of the fact
                    (bench/skills/inject_labels.code_tokens' test:
                    skill_classify.code_shaped) or a word of 5+ letters of it
                    appears in the state, else level 0
  skill_inject noul "yes"
A stub, not a model: it exercises the wiring, never a measurement.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decider_bonsai as D  # noqa: E402

P = 0.9
IDS = {}
for _i, _L in enumerate(D.LETTERS):
    IDS[_L] = 600 + _i
    IDS[" " + _L] = 700 + _i


def upstream(path, payload=None, timeout=30):
    if path == "/tokenize":
        s = (payload or {}).get("content") or ""
        return {"tokens": [IDS[s]] if s in IDS else
                list(range(max(len(s) // 3, 1)))}
    if path == "/slots":
        return []
    raise ConnectionRefusedError(f"decider_stub: no {path} offline")


def _printed(msgs):
    q = next((m["content"] for m in reversed(msgs) if m["role"] == "user"
              and "OPTIONS:\n" in str(m.get("content"))), "")
    out = []
    for ln in q.split("OPTIONS:\n")[1].split("\n\n")[0].splitlines() \
            if q else []:
        sep = ". " if ln[1:3] == ". " else ") "
        if sep in ln:
            lab, text = ln.split(sep, 1)
            out.append((lab, text))
    return q, out


_WORD = re.compile(r"[A-Za-z_$][\w$.]*")


NOT_MINE = object()


def pass_all(state: str, question: str, options: list[str]):
    """The default rule (module docstring)."""
    import skill_inject as I
    if "FACT: " in question and len(options) == len(I.ITEM_LEVELS):
        return I.ITEM_LEVELS[-1]
    if "more correct with these facts" in question:
        return "yes" if "yes" in options else None
    return NOT_MINE


def evidence(state: str, question: str, options: list[str]):
    """The `evidence` rule (module docstring)."""
    import skill_inject as I
    if "FACT: " in question and len(options) == len(I.ITEM_LEVELS):
        fact = question.split("FACT: ", 1)[1].split("\n")[0]
        import skill_classify as C
        low = state.lower()
        for w in _WORD.findall(fact):
            w = w.rstrip(".")
            if (C.code_shaped(w) and w in state) or (
                    len(w) >= 5 and w.isalpha() and w.lower() in low):
                return I.ITEM_LEVELS[-1]
        return I.ITEM_LEVELS[0]
    if "more correct with these facts" in question:
        return "yes" if "yes" in options else None
    return NOT_MINE


class Fake:
    """The model server: the option `want` favours gets P, the rest share
    1 - P; None -> uniform."""

    def __init__(self, want=pass_all):
        self.want, self.bodies = want, []

    def __call__(self, body, timeout):
        self.bodies.append(body)
        m = body["messages"]
        state = str(m[1]["content"])[len(D.STATE_HEAD):]
        q, opts = _printed(m)
        fav = (self.want(state, q, [t for _l, t in opts]) if opts
               else NOT_MINE)
        if fav is NOT_MINE:
            raise ConnectionRefusedError("decider_stub: answers the "
                                         "injector's questions only")
        n = max(len(opts), 1)
        top = []
        for lab, text in opts:
            pr = (P if text == fav else (1 - P) / max(n - 1, 1)) \
                if fav is not None else 1.0 / n
            top.append({"id": IDS[" " + lab], "logprob": math.log(pr)})
        top.sort(key=lambda t: -t["logprob"])
        head = top[0] if top else {"id": 0, "logprob": 0.0}
        return {"choices": [{"logprobs": {"content": [
            {"id": head["id"], "logprob": head["logprob"],
             "top_logprobs": top}]}}], "usage": {"prompt_tokens": 60},
            "timings": {"prompt_n": 15, "cache_n": 45, "prompt_ms": 1.0}}


@contextlib.contextmanager
def installed(want=pass_all):
    """The decider on, answering through Fake(want); restored after. A
    caller of skill_select.decide with no Turn of its own (a suite calling
    the selector directly, where the proxy would have opened one) gets a
    Turn on its messages for the call, as the proxy's _decider_turn does."""
    import decide_turn as T
    import skill_select as S
    fake = Fake(want)
    saved = (D._post_default, D._upstream, D.release, T._ENABLED, S.decide)
    real_decide = S.decide

    def decide(messages, *a, **kw):
        if T.current() is not None:
            return real_decide(messages, *a, **kw)
        with T.Turn(messages or [], on=True, key=kw.get("key")):
            return real_decide(messages, *a, **kw)
    import slots
    import tempfile
    # The slot count: a suite that set none gets the THREE-SLOT layout's
    # (slots.py; -np 3), so acquiring the decider's slot never reads /props.
    saved_slots = (slots._n, slots._n_source)
    if slots._n is None:
        slots._n, slots._n_source = 3, "decider_stub"
    # The decision logs: never the live logs/ (a suite that isolated its
    # stores already points them at its temp dir).
    live = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "logs")
    saved_logs = (T.DECISIONS, T.LOG)
    tmp = tempfile.mkdtemp(prefix="decider_stub_")
    if os.path.abspath(T.DECISIONS).startswith(live):
        T.DECISIONS = os.path.join(tmp, "decider_decisions.jsonl")
    if os.path.abspath(T.LOG).startswith(live):
        T.LOG = os.path.join(tmp, "decider_disagreements.jsonl")
    # Stage 1's per-area choice: skill_match's own evidence stub, as offline
    # before the injector (skill_match.decider's fallback) -- the stub
    # decider answers the injector's questions, not stage 1's.
    import skill_match
    saved_match = skill_match.decider
    skill_match.decider = lambda prefer: ("stub",
                                          skill_match.stub_decider(prefer))
    # pass_all is TRANSPARENT: the composer's size cap goes too (every
    # candidate item of stage 1's first MAX_SKILLS skills goes in), so a
    # suite that judges stage 1 sees each skill stage 1 offered delivered
    # and GIVEN, as its expectations were written. Any other rule keeps the
    # served profile.
    import skill_inject
    saved_profile = skill_inject.profile_for
    if want is pass_all:
        def profile_for(model):
            p = dict(saved_profile(model))
            p["max_items"] = skill_inject.MAX_SKILLS * \
                skill_inject.L.MAX_ITEMS
            return p
        skill_inject.profile_for = profile_for
    D._post_default = fake
    D._upstream = upstream
    D.release = lambda slot, why="": {"released": False, "slot": slot,
                                      "skipped": "decider_stub"}
    T._ENABLED = True
    S.decide = decide
    try:
        yield fake
    finally:
        (D._post_default, D._upstream, D.release, T._ENABLED,
         S.decide) = saved
        skill_match.decider = saved_match
        skill_inject.profile_for = saved_profile
        slots._n, slots._n_source = saved_slots
        T.DECISIONS, T.LOG = saved_logs
