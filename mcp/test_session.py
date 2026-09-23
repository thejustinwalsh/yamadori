#!/usr/bin/env python
"""The held-open Laya decision session, asserted against a fake socket.

WHAT THIS IS GATING

`session.py` turns Laya's probability distributions into decisions. What it
promises, each asserted below:

  1. Always a CHOICE over named options, never a score -- the margin between
     the top two options decides whether the answer counts.
  2. Below a 0.15 margin the answer is `decided: False`. Acting on it would be
     inventing a decision the model did not make.
  3. `triage()` is adaptive: the second question depends on the first answer,
     and an undecided first answer stops the chain rather than guessing.
  4. The state goes FIRST in every frame (AGENTS.md: Laya drops the tail of a
     long state, so the thing judged must lead).

NO LAYA, NO NETWORK

A fake `websockets` module is installed in `sys.modules` before any session
opens. It records every frame sent and replies from a script, so nothing here
can reach :1238. The first test asserts that the fake is what gets connected.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import session  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


class FakeWS:
    """Replies to each frame with `script(frame) -> answers dict`."""

    def __init__(self, url, script):
        self.url = url
        self.script = script
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def recv(self) -> str:
        reply = self.script(self.sent[-1])
        if "__raw__" in reply:          # a whole frame, not just answers
            return json.dumps(reply["__raw__"])
        return json.dumps({"answers": reply})

    async def close(self) -> None:
        self.closed = True


_sockets: list[FakeWS] = []
_script = {"fn": lambda frame: {}}


async def _connect(url, **kw):                                   # noqa: ARG001
    ws = FakeWS(url, lambda frame: _script["fn"](frame))
    _sockets.append(ws)
    return ws


sys.modules["websockets"] = types.SimpleNamespace(connect=_connect)


def answer(choice: str, probs: dict) -> dict:
    return {"choice": choice, "probabilities": probs}


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------


def test_the_fixture_is_a_fake_socket():
    async def go():
        async with session.Session("state") as s:
            return s
    s = run(go())
    check(isinstance(s._ws, FakeWS), "the session connected to the fake socket")
    check(_sockets and _sockets[-1].closed, "and closed it on exit")
    check(session.LAYA_WS.startswith("ws://"), "the default URL is only read, "
          "never dialled", session.LAYA_WS)


def test_ask_returns_a_choice_margin_and_distribution():
    _script["fn"] = lambda f: {"kind": answer("io", {"pure": 0.1, "stateful": 0.2,
                                                     "io": 0.7})}

    async def go():
        async with session.Session("async function f(){}") as s:
            return await s.ask("kind", "What kind?",
                               {"pure": "p", "stateful": "s", "io": "i"}), s
    out, s = run(go())
    check(out["choice"] == "io", "the choice is Laya's", str(out["choice"]))
    check(out["margin"] == 0.5, "margin is top minus second", str(out["margin"]))
    check(out["decided"] is True, "0.5 clears the 0.15 gate")
    check(out["probabilities"] == {"pure": 0.1, "stateful": 0.2, "io": 0.7},
          "the full distribution is returned")
    check(isinstance(out["ms"], float), "with its latency")
    check(s.asked == [out], "and it is kept in the session's history")
    frame = _sockets[-1].sent[-1]
    check(list(frame)[0] == "state", "the state is the first field of the frame",
          str(list(frame)))
    q = frame["questions"]["kind"]
    check(q["type"] == "choice" and q["criteria"] == {"pure": "p", "stateful": "s",
                                                       "io": "i"},
          "the question is a closed choice over the named options", json.dumps(q))


def test_the_margin_gate_is_at_015():
    cases = [({"a": 0.575, "b": 0.425}, True, "exactly 0.15 counts as decided"),
             ({"a": 0.57, "b": 0.43}, False, "0.14 does not"),
             ({"a": 1.0}, False, "a single option has no margin"),
             ({}, False, "no distribution is undecided")]
    for probs, want, label in cases:
        _script["fn"] = lambda f, p=probs: {"q": answer("a", p)}

        async def go():
            async with session.Session("x") as s:
                return await s.ask("q", "?", {"a": "a", "b": "b"})
        out = run(go())
        check(out["decided"] is want, label, json.dumps(out))


def test_a_missing_answer_is_undecided_not_a_crash():
    _script["fn"] = lambda f: {}

    async def go():
        async with session.Session("x") as s:
            return await s.ask("q", "?", {"a": "a", "b": "b"})
    out = run(go())
    check(out["choice"] is None and out["decided"] is False,
          "no answer for the question comes back undecided", json.dumps(out))


def test_a_laya_error_is_raised_not_read_as_undecided():
    """THE BUG THIS FILE FOUND. laya_service.py answers a failed frame with
    {"id", "error"} and no answers; that used to come back as `decided: False`,
    and triage() reported an outage as "could not classify the code"."""
    _script["fn"] = lambda f: {"__raw__": {"id": None,
                                           "error": "RuntimeError: CUDA OOM"}}

    async def one():
        async with session.Session("x") as s:
            return await s.ask("q", "?", {"a": "a", "b": "b"})

    async def many():
        async with session.Session("x") as s:
            return await s.ask_many({"q": {"type": "choice", "criteria": {}}})

    for label, coro in (("ask", one), ("ask_many", many), ("triage", None)):
        try:
            run(coro() if coro else session.triage("x"))
            err = None
        except session.LayaError as e:
            err = str(e)
        except Exception as e:                                   # noqa: BLE001
            err = f"WRONG TYPE {type(e).__name__}: {e}"
        check(err is not None and not err.startswith("WRONG") and "CUDA OOM" in err,
              f"{label}: a service error raises LayaError carrying its message",
              str(err))


def test_ask_many_sends_one_frame():
    _script["fn"] = lambda f: {k: answer("y", {"y": 0.9, "n": 0.1})
                               for k in f["questions"]}
    qs = {"q1": {"type": "choice", "criteria": {"y": "y", "n": "n"}},
          "q2": {"type": "choice", "criteria": {"y": "y", "n": "n"}}}

    async def go():
        async with session.Session("st") as s:
            return await s.ask_many(qs)
    out = run(go())
    check(len(_sockets[-1].sent) == 1, "two questions go in one frame")
    check(set(out) == {"q1", "q2", "_ms"}, "each is answered, plus timing",
          str(sorted(out)))
    check(out["q1"]["decided"] and out["q1"]["margin"] == 0.8,
          "with the same margin gate as ask()", json.dumps(out["q1"]))


def test_triage_branches_on_the_first_answer():
    def script(frame):
        name = next(iter(frame["questions"]))
        if name == "kind":
            return {"kind": answer("stateful", {"pure": 0.05, "stateful": 0.9,
                                                "io": 0.05})}
        return {name: answer("shared", {"none": 0.1, "shared": 0.8,
                                        "ordering": 0.1})}
    _script["fn"] = script
    r = run(session.triage("function f(a){ a.push(1) }"))
    names = [x["name"] for x in r["trail"]]
    check(names == ["kind", "mutation_risk"],
          "a stateful answer leads to the mutation question, not the others",
          str(names))
    check(r["conclusion"] == "stateful, risk: shared",
          "the conclusion names both answers", r["conclusion"])
    sent = [next(iter(f["questions"])) for f in _sockets[-1].sent]
    check(sent == ["kind", "mutation_risk"], "only two frames were sent", str(sent))


def test_triage_stops_when_the_first_answer_is_undecided():
    _script["fn"] = lambda f: {"kind": answer("pure", {"pure": 0.34, "stateful": 0.33,
                                                       "io": 0.33})}
    r = run(session.triage("x"))
    check(len(r["trail"]) == 1, "an undecided classification asks nothing more",
          str(len(r["trail"])))
    check("could not classify" in r["conclusion"],
          "and says so instead of guessing", r["conclusion"])


def test_triage_reports_an_undecided_risk_honestly():
    def script(frame):
        name = next(iter(frame["questions"]))
        if name == "kind":
            return {"kind": answer("io", {"pure": 0.0, "stateful": 0.0, "io": 1.0})}
        return {name: answer("unawaited", {"none": 0.4, "unawaited": 0.45,
                                           "unhandled": 0.15})}
    _script["fn"] = script
    r = run(session.triage("x"))
    check(r["conclusion"] == "io, risk undetermined",
          "a risk below the margin is 'undetermined', not the top option",
          r["conclusion"])


def main() -> int:
    for fn in (test_the_fixture_is_a_fake_socket,
               test_ask_returns_a_choice_margin_and_distribution,
               test_the_margin_gate_is_at_015,
               test_a_missing_answer_is_undecided_not_a_crash,
               test_a_laya_error_is_raised_not_read_as_undecided,
               test_ask_many_sends_one_frame,
               test_triage_branches_on_the_first_answer,
               test_triage_stops_when_the_first_answer_is_undecided,
               test_triage_reports_an_undecided_risk_honestly):
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
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
