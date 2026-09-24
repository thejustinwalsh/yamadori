"""Offline checks for mechanisms.py on x_yamadori shapes copied from real answers
(bench/livebench/results/lb-20260923/smoke_yamadori_arm, bonsai rows) plus
constructed failure cases. Prints "N/M checks passed"; exits 1 on any failure.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechanisms as M  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name, ok):
    checks.append((name, bool(ok)))
    if not ok:
        print("FAIL:", name)


# bonsai arm, captured 2026-09-23 (everything forced off)
bonsai = {"tier": "medium", "effort_sent": "medium", "tools_gate": None, "hints": [], "suppressed_hints": [],
          "selection": {"hints": False, "investigate": False, "fanout_n": 1,
                        "signals": {"forced": ["fanout", "hints", "investigate"],
                                    "allowed": {"hints": False, "investigate": False, "fanout": 1}}},
          "fanout": None, "investigate": None, "hops": 1, "images": [],
          "budget": {"max_tokens_sent": 101983, "reasoning_budget_tokens": 97887}}
# header full-stack smoke answer, captured 01:36 (investigate ran and legitimately found nothing)
full = {"tier": "medium", "effort_sent": "medium", "tools_gate": {"offer": True, "why": "NO_DOMAIN_EVIDENCE: ..."},
        "hints": [], "suppressed_hints": [],
        "selection": {"hints": True, "investigate": True, "fanout_n": 3,
                      "signals": {"forced": ["fanout", "hints", "investigate"],
                                  "allowed": {"hints": True, "investigate": True, "fanout": 3}}},
        "fanout": {"n": 3, "asked": 3, "seeds": ["a", "b", "c"], "agreement": None, "votes": 0,
                   "winner": "direct", "dissent_noted": False},
        "investigate": {"ran": True, "hops": 0, "handle": "x", "injected": False, "ok": True, "seconds": 8.7,
                        "cited": 0, "unsupported": 0, "why": "it made no search"},
        "hops": 1}

rb = M.record(bonsai)
check("bonsai: no stack error", rb["stack_error"] == [])
check("bonsai: fan-out not allowed, not run", rb["fanout"]["allowed_n"] == 1 and not rb["fanout"]["ran"])
check("bonsai: deep thinking not allowed", not rb["deep_thinking"]["allowed"] and rb["deep_thinking"]["ran"] is None)
check("bonsai: retrieval 0 tool rounds, produced unknown", rb["retrieval"]["tool_rounds"] == 0
      and rb["retrieval"]["produced"] is None)
rf = M.record(full)
check("full: ran-and-found-nothing is DATA, not a stack error", rf["stack_error"] == [])
check("full: deep thinking ran, not injected", rf["deep_thinking"]["ran"] and rf["deep_thinking"]["injected"] is False)
check("full: fan-out ran n=3", rf["fanout"]["ran"] and rf["fanout"]["n"] == 3)

# post-fix fan-out record with selection + delivery
fix = dict(full, fanout=dict(full["fanout"], selection="code_medoid", winner_index=2, replaced=True, appended=False,
                             candidates=[{"variant": "original", "parses": True}, {"variant": "a", "parses": False}]))
rx = M.record(fix)
check("post-fix: method and replaced recorded", rx["fanout"]["method"] == "code_medoid" and rx["fanout"]["replaced"] is True)
check("post-fix: candidates parse list", rx["fanout"]["candidates_parse"] == [True, False])

# broken mechanisms -> stack_error
check("fan-out exception is a stack error", M.record(dict(full, fanout={"n": 0, "asked": 3, "error": "Timeout"}))["stack_error"])
check("fan-out asked 3, got 0 is a stack error",
      M.record(dict(full, fanout={"n": 0, "asked": 3, "seeds": []}))["stack_error"])
check("deep thinking failed is a stack error",
      M.record(dict(full, investigate={"ran": True, "ok": False, "why": "helper 502"}))["stack_error"])
check("deep thinking selected but no record is a stack error",
      M.record(dict(full, investigate=None))["stack_error"])
check("check_code tool error is a stack error",
      M.record(dict(bonsai, check_code={"offered": True, "calls": 1, "results": [{"ok": False, "error": "parser crashed"}]}))["stack_error"])
check("check_code problems found is DATA",
      M.record(dict(bonsai, check_code={"offered": True, "calls": 1, "results": [{"ok": True, "problems": 2}]}))["stack_error"] == [])
check("missing x_yamadori is flagged", M.record(None)["stack_error"])

# sequential fan-out (schema of mcp/fanout.py selection_record + proxy rec, 2026-09-23 ~10:15)
seq_base = {"n": 2, "asked": 3, "seeds": ["kw1"], "agreement": None, "votes": 0, "winner": "original",
            "dissent_noted": False, "replaced": False, "appended": False, "mode": "sequential", "steps": 2,
            "stop_reason": "clear: agreement", "similarity_ab": 0.91, "selection": "code_grade", "winner_index": 0,
            "why": "clear: both parse and agree", "candidates": [
                {"variant": "original", "role": "main", "seed": None, "parses": True, "errors": 0},
                {"variant": "direct", "role": "helper", "seed": "kw1", "parses": True, "errors": 0}]}
rs = M.record(dict(full, fanout=seq_base))
check("sequential: code_grade clear agreement is data", rs["stack_error"] == [] and rs["fanout"]["ran"])
check("sequential: mode, steps, seeds recorded", rs["fanout"]["mode"] == "sequential" and rs["fanout"]["steps"] == 2
      and rs["fanout"]["candidate_seeds"] == [None, "kw1"] and rs["fanout"]["method"] == "code_grade")
tb = dict(seq_base, n=3, steps=3, stop_reason="tie-breaker: both parse but disagree", selection="code_medoid",
          winner="tiebreak", winner_index=2, replaced=True)
rtb = M.record(dict(full, fanout=tb))
check("sequential: tie-breaker with delivered winner is data", rtb["stack_error"] == [] and rtb["fanout"]["replaced"])
check("sequential: helper busy is a stack error",
      M.record(dict(full, fanout=dict(seq_base, n=1, steps=1, skipped="helper busy",
                                      stop_reason="skipped: helper busy")))["stack_error"])
check("sequential: empty second candidate is a stack error",
      M.record(dict(full, fanout=dict(seq_base, stop_reason="second candidate failed")))["stack_error"])
check("sequential: unresolved without tie-breaker room is data",
      M.record(dict(full, fanout=dict(seq_base, stop_reason="unresolved: neither parses, no tie-breaker allowed")))["stack_error"] == [])
hs = M.health([rs, rtb])
check("health: sequential mode and steps counted", hs["mechanisms"]["fanout"].get("mode:sequential") == 2
      and hs["mechanisms"]["fanout"].get("steps:3") == 1)

# tool-turn cap (x_yamadori.tool_turns)
rcap = M.record(dict(full, tool_turns={"limit": 10, "turns": 10, "hit": True}))
check("tool_turns: hit is data, not a stack error", rcap["stack_error"] == [] and rcap["tool_turns"]["hit"] is True)
check("tool_turns: absent before the deploy -> None", M.record(full)["tool_turns"] is None)
hc = M.health([rcap, M.record(dict(full, tool_turns={"limit": 10, "turns": 2, "hit": False})), M.record(full)])
check("health: cap hits counted per row", hc["tool_turn_cap"]["hit"] == 1
      and hc["tool_turn_cap"]["answers_with_record"] == 2 and hc["tool_turn_cap"]["limits"] == [10])

# post-deploy x_yamadori.tools
tl = dict(full, hops=3, tools=[{"name": "find_by_meaning", "empty": False, "error": None, "chars": 900},
                               {"name": "find_definition_opt", "empty": True, "error": None, "chars": 0}])
rt = M.record(tl)
check("tools: calls, nonempty, produced", rt["retrieval"]["calls"] == 2 and rt["retrieval"]["nonempty"] == 1
      and rt["retrieval"]["produced"] is True and rt["stack_error"] == [])
check("tools: an empty result is data", M.record(dict(full, tools=[{"name": "x", "empty": True, "error": None}]))["stack_error"] == [])
check("tools: a tool error is a stack error",
      M.record(dict(full, tools=[{"name": "find_by_pattern", "empty": True, "error": "index locked"}]))["stack_error"])
ht = M.health([rt, rb])
check("health mixes answers with and without tools: produced counted only where recorded",
      ht["mechanisms"]["retrieval"]["produced"] == 1
      and ht["mechanisms"]["retrieval"]["answers_without_tool_record"] == 1)

h = M.health([rb, rf, rx])
check("health counts answers", h["answers"] == 3 and h["stack_errors"] == 0)
check("health: fan-out ran on 2, replaced on 1", h["mechanisms"]["fanout"]["ran"] == 2
      and h["mechanisms"]["fanout"]["winner_replaced"] == 1)
check("health: deep thinking allowed 2, ran 2, produced 0", h["mechanisms"]["deep_thinking"]["allowed"] == 2
      and h["mechanisms"]["deep_thinking"]["ran"] == 2 and h["mechanisms"]["deep_thinking"]["produced"] == 0)
check("health: retrieval produced unknown", h["mechanisms"]["retrieval"]["produced"] is None)

passed = sum(ok for _, ok in checks)
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
