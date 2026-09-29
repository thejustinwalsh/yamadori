#!/usr/bin/env python
"""THE SELECTION STATECHART: which skill questions are LEGAL at this turn.

Operator, 2026-09-27: the per-conversation selection state is an explicit
small statechart, in XState's sense (Stately's agent machines: the machine
declares the legal events per state; a guard is a transition that returns
undefined when it does not apply, and then the event is simply not legal
there). The selector (skill_select) asks the decider ONLY the questions
this chart says are legal, and every option of a question is one the chart
allows at this turn.

TWO REGIONS

  top     where the work is: the FINE phase the evidence raises (one of
          skill_classify.PHASES: plan, implement, debug, verify, refactor,
          review), entered on the PHASE event, with `area` -- the area the
          newest evidence is about (the first asked area, else the first
          framework, else the first language of the fresh evidence) --
          while the work is implement, refactor or debug. (Until 2026-09-27
          the six phases were grouped into four coarse states; the grouping
          was ours: docs/CONSTANTS-AUDIT.md "PHASE_TOP mapping".)

  area    one substate per area the conversation has touched:

              unseen --BODY--> given --ASKED | ERROR | PHASE--> recall_eligible
                                 ^                                   |
                                 +------------- RECALLED ------------+

          `unseen`: nothing of the area was given -- a BODY question is
          legal on a trigger (asked, new evidence, an error, a phase gate).
          `given`: a body (or a recall) went in; another skill of the area
          may still come in as a body on new evidence of its own, and the
          given one comes back only on an EVENT that makes it matter (asked,
          an error, a phase change). `recall_eligible` is entered on this
          turn's event and left on RECALLED (or settles back when the
          decider picked nothing). A COMPACTED conversation returns every
          area to `unseen`. REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md, D):
          the fade (a `faded` substate after FADE_TOKENS of conversation)
          and the recall cooldown (RECALL_EVERY_STEPS requests); the one
          repeat rule left is skill_select's: never the identical line twice
          in a row.

THE GUARDS (a transition returning None is illegal, never an error)

  ASKED     the area was asked for (or implied) in this user turn
  EVIDENCE  a skill of the area has new evidence in the fresh messages
  ERROR     ... and the fresh evidence shows an error
  PHASE     the fine phase changed this turn

These are the rules skill_select.decide applied after selection before
2026-09-27, stated as a machine and moved IN FRONT of the questions: an
option that is not legal is never shown to the decider.

THE CONTEXT (persisted in the conversation's skill state -- one ledger row,
kind `skills`, beside `given`, `recalls`, `phase`, `req`, `chars`):

    st["chart"] = {"v": 1, "top": {"state", "area"},
                   "areas": {area: substate}, "last": [transition, ...]}

Pure: no I/O. skill_select builds a View per request and commits it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TOP_STATES = ("plan", "implement", "debug", "verify", "refactor", "review")
# The top states that carry the area the work is about.
FOCUSED = ("implement", "refactor", "debug")
AREA_STATES = ("unseen", "given", "recall_eligible")
# The events the machine knows. Anything else is illegal everywhere.
EVENTS = ("ASKED", "EVIDENCE", "ERROR", "PHASE", "BODY", "RECALLED",
          "SETTLE", "COMPACTED")
TOP_EVENTS = ("PHASE", "FOCUS")


# ---------------------------------------------------------------------------
# The machine: state -> event -> target (a string) or a guard (a function of
# the context returning the target, or None when the transition does not
# apply -- XState's "undefined": the event is not legal here).
# ---------------------------------------------------------------------------
def _settle(ctx: dict):
    return ctx.get("before") or "given"


AREA_MACHINE: dict = {
    "unseen": {"BODY": "given", "COMPACTED": "unseen"},
    "given": {"BODY": "given", "ASKED": "recall_eligible",
              "ERROR": "recall_eligible", "PHASE": "recall_eligible",
              "COMPACTED": "unseen"},
    "recall_eligible": {"RECALLED": "given", "BODY": "recall_eligible",
                        "SETTLE": _settle, "COMPACTED": "unseen"},
}


def transition(state: str, event: str, ctx: dict | None = None
               ) -> str | None:
    """The area machine's next state for `event`, or None when the event is
    not legal in `state` (no transition declared, or its guard returned
    None). Pure."""
    t = (AREA_MACHINE.get(state) or {}).get(event)
    if t is None:
        return None
    return t(ctx or {}) if callable(t) else t


def legal_events(state: str, ctx: dict | None = None) -> list[str]:
    """The events that have a transition from `state` under `ctx` (what a
    `decide` over this machine could choose among)."""
    return [e for e in EVENTS if transition(state, e, ctx) is not None]


def top_of(phase: str | None) -> str | None:
    """The top state for the work's phases ("debug", or "debug+implement":
    every phase the evidence raises, no priority among them)."""
    parts = [p for p in str(phase or "").split("+") if p in TOP_STATES]
    return "+".join(parts) or None


def new_chart() -> dict:
    return {"v": 1, "top": {"state": None, "area": None}, "areas": {},
            "last": []}


# ---------------------------------------------------------------------------
# The per-request view: the chart's context for THIS request, and the
# legality of each option the selector would show the decider.
# ---------------------------------------------------------------------------
class View:
    """The chart as of this request. `st` is the conversation's skill state
    (skill_select.new_state's shape); `area_of(skill)` names a skill's area.

    `option(skill, kind, ...)` -> (trigger, form, why): the trigger that
    makes the option legal and the form it would take (body / recall), or
    (None, None, why) when it is not legal at this turn."""

    def __init__(self, st: dict, *, chars: int, req: int,
                 phase_now: str | None, phase_changed: bool, first: bool,
                 area_of, kind: str = "user", asked=(), focus=None):
        self.st = st
        self.chars = int(chars)
        self.req = int(req)
        self.phase_now = phase_now
        self.phase_changed = bool(phase_changed)
        self.first = bool(first)
        self.kind = kind
        self.area_of = area_of
        self.asked = list(asked or [])
        self.focus = focus
        # Copies: the chart reads the state AS OF THE START of this request
        # (decide updates `given` and `recalls` as it injects).
        self.given = json.loads(json.dumps(st.get("given") or {}))
        self.recalls = json.loads(json.dumps(st.get("recalls") or {}))
        chart = st.get("chart") if isinstance(st.get("chart"), dict) \
            else None
        self.chart = json.loads(json.dumps(chart)) if chart else new_chart()
        # Areas the conversation has touched: the stored substates, and the
        # areas of what was given (a state written before the chart existed).
        self.areas: dict[str, str] = dict(self.chart.get("areas") or {})
        self._skill_area = dict((st.get("chart") or {}).get("skill_area")
                                or {})
        self.transitions: list[dict] = []
        self.events: dict[str, set] = {}

    def fresh(self) -> bool:
        """Nothing given and nothing recalled: every option is legal as a
        body on its trigger, so a selection needs no chart."""
        return not self.given and not self.recalls

    def substate(self, area: str) -> str:
        """The area's substate BEFORE this turn's events."""
        s = self.areas.get(area)
        given_here = [sid for sid in self.given
                      if self._skill_area.get(sid) == area]
        if s is None:
            s = "given" if given_here else "unseen"
        if s not in ("unseen", "given"):
            # recall_eligible settles; a `faded` stored before 2026-09-27
            # is given.
            s = "given"
        return s

    def note_skill(self, sid: str, area: str) -> None:
        self._skill_area[sid] = area

    # --- legality of one option ---------------------------------------
    def option(self, s: dict, kind: str, *, fresh_hit: bool = False,
               err: bool = False, phase_gated: bool = False,
               step_trigger: str | None = None
               ) -> tuple[str | None, str | None, str]:
        """(trigger, form, why). `kind`: the question it would appear in --
        `asked` (the user named its area, or the implication table implies
        it), `evidence` (a user turn's evidence question), `step` (an agent
        step's), `category`. `fresh_hit`: the skill matches the newest
        evidence (skill_select.turn_match); `phase_gated`: the skill is
        gated on the phase the work just entered."""
        sid = s["id"]
        area = self.area_of(s)
        self.note_skill(sid, area)
        # The trigger the evidence gives (the same order as before).
        if kind == "asked":
            trig = "asked"
            self._event(area, "ASKED")
        elif kind == "step":
            trig = step_trigger or ("error" if err else "new_area")
            self._event(area, "ERROR" if trig == "error" else "EVIDENCE")
        elif self.first:
            trig = "new_area"
            self._event(area, "EVIDENCE")
        elif fresh_hit:
            trig = "error" if err else "new_area"
            self._event(area, "ERROR" if err else "EVIDENCE")
        elif self.phase_changed and phase_gated:
            trig = "phase"
            self._event(area, "PHASE")
        else:
            return None, None, "no new evidence in this turn"
        if sid not in self.given:
            return trig, "body", "new to the conversation"
        # Given: only an EVENT that makes it matter brings it back.
        sub = self.substate(area)
        if trig in ("asked", "error"):
            ev = "ASKED" if trig == "asked" else "ERROR"
        elif self.phase_changed:
            trig, ev = "phase", "PHASE"
        else:
            return None, None, "already given; nothing new makes it " \
                               "matter now"
        if transition(sub if sub in AREA_MACHINE else "given", ev) is None:
            return None, None, f"no {ev} transition from {sub}"
        return trig, "recall", f"{sub} --{ev}--> recall_eligible"

    def _event(self, area: str, ev: str) -> None:
        self.events.setdefault(area, set()).add(ev)
        if self.phase_changed:
            self.events[area].add("PHASE")

    # --- the sticky key ------------------------------------------------
    def signature(self) -> str:
        """What of the chart a selection depends on: the given skills,
        the phase change and the first-request flag."""
        body = json.dumps({
            "given": sorted(self.given),
            "phase": [self.phase_now, self.phase_changed],
            "first": self.first, "kind": self.kind}, sort_keys=True)
        return hashlib.sha1(body.encode()).hexdigest()[:16]

    # --- commit ---------------------------------------------------------
    def commit(self, picks: list[dict]) -> dict:
        """Apply this turn's picks ([{id, area, form}]) to the chart: BODY
        (unseen/given -> given), RECALLED (recall_eligible -> given),
        SETTLE for an area whose event made it recall_eligible but nothing
        was picked. Returns the chart record for x_yamadori.skills."""
        top_before = dict(self.chart.get("top") or {})
        top_state = top_of(self.phase_now) or top_before.get("state")
        focus = self.focus or top_before.get("area")
        top_after = {"state": top_state,
                     "area": focus if set(str(top_state or "").split(
                         "+")) & set(FOCUSED) else None}
        if top_after != top_before:
            # PHASE: the work moved (implement -> debug); FOCUS: the same
            # state, about another area (implement(typescript) ->
            # implement(koota)).
            self.transitions.append({
                "region": "top", "from": top_before.get("state"),
                "event": "PHASE" if top_state != top_before.get("state")
                else "FOCUS", "to": top_state, "area": top_after["area"]})
        picked_areas: dict[str, str] = {}
        for p in picks:
            self.note_skill(p["id"], p["area"])
            picked_areas.setdefault(p["area"], p["form"])
        touched = set(self.events) | set(picked_areas)
        areas_out = dict(self.areas)
        for area in sorted(touched):
            before = self.substate(area)
            evs = self.events.get(area) or set()
            cur = before
            # The event that made it recall_eligible, if any (an ask, an
            # error, a phase change).
            for ev in ("ASKED", "ERROR", "PHASE"):
                if ev in evs:
                    nxt = transition(cur, ev, {"before": before})
                    if nxt is not None and nxt != cur:
                        self.transitions.append({"region": "area",
                                                 "area": area, "from": cur,
                                                 "event": ev, "to": nxt})
                        cur = nxt
                        break
            form = picked_areas.get(area)
            if form == "recall" and cur == "recall_eligible":
                nxt = transition(cur, "RECALLED")
                self.transitions.append({"region": "area", "area": area,
                                         "from": cur, "event": "RECALLED",
                                         "to": nxt})
                cur = nxt
            elif form == "body":
                nxt = transition(cur, "BODY") or cur
                if nxt != cur or before == "unseen":
                    self.transitions.append({"region": "area", "area": area,
                                             "from": cur, "event": "BODY",
                                             "to": nxt})
                cur = "given" if nxt == "recall_eligible" else nxt
            elif cur == "recall_eligible":
                nxt = transition(cur, "SETTLE", {"before": before})
                self.transitions.append({"region": "area", "area": area,
                                         "from": cur, "event": "SETTLE",
                                         "to": nxt})
                cur = nxt
            if cur != "unseen":
                areas_out[area] = cur
        self.chart = {"v": 1, "top": top_after, "areas": areas_out,
                      "skill_area": dict(sorted(self._skill_area.items())),
                      "last": self.transitions[-12:]}
        return self.record()

    def record(self) -> dict:
        """x_yamadori.skills.chart: the top state and its area, every area's
        substate, this turn's transitions, and the events of the areas the
        conversation holds (an event on an area never touched changes
        nothing and is not listed)."""
        areas = dict(self.chart.get("areas") or {})
        moved = {t.get("area") for t in self.transitions}
        return {"state": (self.chart.get("top") or {}).get("state"),
                "area": (self.chart.get("top") or {}).get("area"),
                "areas": areas,
                "transitions": list(self.transitions[-12:]),
                "events": {a: sorted(e) for a, e in sorted(self.events.items())
                           if a in areas or a in moved}}


def compacted(st: dict) -> dict:
    """COMPACTED: every area back to unseen (the conversation shrank)."""
    ch = st.get("chart") if isinstance(st.get("chart"), dict) else new_chart()
    before = dict(ch.get("areas") or {})
    ch = dict(ch, areas={}, last=[{"region": "area", "area": a, "from": s,
                                   "event": "COMPACTED", "to": "unseen"}
                                  for a, s in sorted(before.items())][-12:])
    return ch
