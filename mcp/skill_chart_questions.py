#!/usr/bin/env python
"""THE QUESTION STATECHART: which bank questions are LEGAL at this turn.

SHELVED (operator, 2026-09-27): the question pipeline is not on any path;
the package rule (mcp/skill_packages.py) is the selection path. Kept, with
its test (mcp/test_skill_questions_bank.py), unused.

Operator, 2026-09-27: the question match (mcp/skill_question_match.py) runs
"with a statechart deciding which questions are legal". This extends
mcp/skill_chart.py's model (its top region and its phase map are read from
there, unchanged) with the question side, in its own module:

  top     the turn's main PHASE (skill_classify's six; skill_chart's top
          region). The phase decides which KIND of bank question is legal
          this turn -- the operator's table for this pipeline (2026-09-27:
          "planning: design/choice questions; building: how-to for that
          area; debugging: what-explains-this-error; verifying:
          how-to-check"), stated per fine phase:

              plan                  choose   design and choice questions
              implement, refactor   how      how-to questions
              debug                 error    what-explains-this-error
              verify, review        check    how-to-check questions

          No phase read: every kind is legal.

  areas   a bank question is legal only in an OPEN area (the request's
          patterns: skill_select.open_areas / open_artifacts -- asked, a
          fence, an import, a pin, a path, a tool call, an error line).

  answered
          a question already answered by a GIVEN skill is not asked again:
          every question of a given skill is illegal -- unless this turn
          raises one of the selection chart's recall events for the skill's
          area (skill_chart.AREA_MACHINE: given --ASKED | ERROR |
          PHASE--> recall_eligible): the user asks for the area again, an
          error line names it, or the phase changed. Then its questions are
          legal once more and a pick is a RECALL. (The operator's rule was
          "given and not faded"; the fade -- FADE_TOKENS -- was removed from
          skill_chart on 2026-09-27, docs/CONSTANTS-AUDIT.md, so the events
          are the only way back, exactly as in skill_chart.)

  COMPACTED (the conversation shrank to under 0.7 of its size, the
          selection chart's own test in skill_select.decide) forgets what
          was given: every question is askable again.

Pure: no I/O. The state is a small dict the caller keeps per
conversation:

    {"v": 1, "given": {skill id: {"chars", "req", "question"}},
     "req": n, "chars": n, "top": str | None}
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skill_chart  # noqa: E402

# The operator's table (above), per fine phase.
KIND_OF_PHASE = {"plan": "choose", "implement": "how", "refactor": "how",
                 "debug": "error", "verify": "check", "review": "check"}
ALL_KINDS = frozenset(KIND_OF_PHASE.values())
# skill_select._PHASE_ORDER: when a turn shows several phases, the first of
# these is the main one (read there; restated so this module stays pure).
PHASE_ORDER = ("debug", "verify", "refactor", "review", "implement", "plan")
# The selection chart's compaction test (skill_select.decide).
SHRINK = 0.7


def new_state() -> dict:
    return {"v": 1, "given": {}, "req": 0, "chars": 0, "top": None}


def main_phase(phases) -> str | None:
    have = set(phases or ())
    return next((p for p in PHASE_ORDER if p in have), None)


def top_of(phases) -> str | None:
    return skill_chart.top_of(main_phase(phases))


def legal_kinds(top: str | None) -> frozenset:
    k = KIND_OF_PHASE.get(top or "")
    return frozenset({k}) if k else ALL_KINDS


class Turn:
    """The chart at one request: `legal(question_row)` -> (ok, why)."""

    def __init__(self, state: dict | None, *, chars: int, phases,
                 open_areas, events: dict | None = None):
        st = dict(state or new_state())
        st["given"] = dict(st.get("given") or {})
        self.compacted = bool(st.get("chars")) and \
            chars < SHRINK * int(st["chars"])
        if self.compacted:
            st["given"] = {}
        self.state = st
        self.chars = int(chars)
        self.top = top_of(phases)
        self.kinds = legal_kinds(self.top)
        self.areas = set(open_areas or ())
        # {area: {"ASKED", "ERROR"}} this turn; PHASE when the phase moved.
        self.events = {a: set(e) for a, e in (events or {}).items()}
        if st.get("top") and self.top and st["top"] != self.top:
            for a in self.areas:
                self.events.setdefault(a, set()).add("PHASE")

    def recall_event(self, area: str) -> str | None:
        """The event that makes a given skill of `area` legal again, if
        this turn raises one (skill_chart: given --EVENT--> recall_eligible).
        """
        ev = self.events.get(area) or set()
        for e in ("ASKED", "ERROR", "PHASE"):
            if e in ev and skill_chart.transition("given", e) is not None:
                return e
        return None

    def answered(self, sid: str, area: str | None = None) -> bool:
        return sid in self.state["given"] and not (
            area and self.recall_event(area))

    def form(self, sid: str) -> str:
        return "recall" if sid in self.state["given"] else "body"

    def legal(self, q: dict) -> tuple[bool, str]:
        if q["area"] not in self.areas:
            return False, f"area {q['area']} is not open"
        if q["kind"] not in self.kinds:
            return False, f"a {q['kind']} question is not legal while " \
                          f"{self.top}"
        if self.answered(q["skill"], q["area"]):
            return False, "answered by a given skill (no recall event)"
        return True, "legal"

    def commit(self, picks: list[dict]) -> dict:
        """Record this turn's picks ([{skill, question}]) as given; returns
        the new state."""
        st = dict(self.state)
        st["req"] = int(st.get("req") or 0) + 1
        st["chars"] = self.chars
        st["top"] = self.top
        given = dict(st["given"])
        for p in picks:
            given[p["skill"]] = {"chars": self.chars, "req": st["req"],
                                 "question": p.get("question")}
        st["given"] = given
        return st

    def record(self) -> dict:
        return {"top": self.top, "kinds": sorted(self.kinds),
                "areas": sorted(self.areas),
                "given": sorted(self.state["given"]),
                "events": {a: sorted(e) for a, e in sorted(
                    self.events.items())},
                "compacted": self.compacted}
