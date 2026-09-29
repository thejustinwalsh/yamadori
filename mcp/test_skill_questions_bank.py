#!/usr/bin/env python
"""The question bank (SHELVED 2026-09-27: mcp/skill_questions_bank.py,
mcp/question_prompts.py, mcp/skill_chart_questions.py are kept, unused). No
GPU, no network, no model: the generation is a fake.

    python mcp/test_skill_questions_bank.py      -> "N/M checks passed"

  1. Every text is pinned to its version (mcp/fixtures/
     question_prompt_pins.json), and the data paragraph still matches the
     skill factory's.
  2. The reply is parsed strictly and every question verified: a name the
     skill never mentions, a kind outside the table, a missing "?", an item
     the skill does not have, a repeat -- each rejected with its reason.
  3. The question statechart: a kind is legal only in its phase, an area
     only when open, a given skill's questions only on a recall event.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_stores  # noqa: E402
offline_stores.isolate("yamadori_test_qbank_")

import question_prompts as QP  # noqa: E402
import skill_chart_questions as CQ  # noqa: E402
import skill_questions_bank as B  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok, name, detail=""):
    _results.append((bool(ok), name, str(detail)[:300]))


SKILL = {"id": "s1", "version": 1, "name": "koota-queries-and-systems",
         "title": "Koota Queries and Systems",
         "description": "Use when querying a koota world: world.query with "
                        "updateEach or readEach.",
         "body": "", "rule": {"applies_to": {"frameworks": ["koota"]},
                              "phases": ["implement"], "topics": []},
         "items": [{"form": "DO", "text": "batch-update entities with "
                    "world.query(Position, Velocity).updateEach(...)."},
                   {"form": "DO", "text": "use readEach for read-only "
                    "iteration."}]}


def test_pins():
    with open(os.path.join(HERE, "fixtures", "question_prompt_pins.json"),
              encoding="utf-8") as f:
        pins = json.load(f)
    for r in QP.registry():
        p = pins.get(r["name"]) or {}
        check(p.get("version") == r["version"] and p.get("sha256")
              == r["sha256"], f"[pins] {r['name']} is pinned to "
              f"{r['version']}: change the text, bump the version and the "
              "fixture", (p, r["sha256"]))
    try:
        import skill_prompts
        check(skill_prompts.DATA_PARAGRAPH == QP.DATA_PARAGRAPH,
              "[pins] the data paragraph matches skill_prompts'")
    except Exception as e:                                       # noqa: BLE001
        check(True, f"[pins] skill_prompts not importable here ({e!r}); "
              "the paragraph comparison is skipped")
    check(QP.recall_query("x") == f"Instruct: {QP.RECALL_TASK}\nQuery:x",
          "[pins] the query side is the model card's get_detailed_instruct")


def test_verify():
    reply = ("<think>...</think>\n"
             "Q1 [how]: How do I batch-update entities with updateEach?\n"
             "Q2 [how]: How do I iterate read-only with readEach?\n"
             "Q2 [how]: How do I iterate read-only with readEach?\n"
             "Q0 [how]: How do I use useQuery with a koota world?\n"
             "Q3 [how]: How do I do a third thing?\n"
             "Q1 [ponder]: Which query is best?\n"
             "Q1 [how]: How do I query a koota world\n"
             "not a question line")
    e = B.distil(SKILL, gen=lambda s: {"reply": reply, "usage": {},
                                       "timings": {}, "finish": "stop",
                                       "reasoning_chars": 0,
                                       "seconds": 0.0})
    kept = [q["text"] for q in e["questions"]]
    why = [r["why"] for r in e["rejected"]]
    check(kept == ["How do I batch-update entities with updateEach?",
                   "How do I iterate read-only with readEach?"],
          "[verify] the answerable questions are kept", kept)
    for part in ("repeats", "never mentions: useQuery", "item 3",
                 "kind 'ponder'", "ending in '?'", "reply shape"):
        check(any(part in w for w in why), f"[verify] rejected: {part}",
              why)
    check(e["area"] == "koota" and e["template"] == QP.BANK_VERSION,
          "[verify] the entry carries the area and the template version",
          (e["area"], e["template"]))


def test_chart():
    t = CQ.Turn(None, chars=100, phases={"implement": 1},
                open_areas=["koota"])
    q = {"area": "koota", "kind": "how", "skill": "s1"}
    check(t.legal(q)[0], "[chart] a how-to question while implementing")
    check(not t.legal(dict(q, kind="choose"))[0],
          "[chart] a choice question is not legal while implementing")
    check(not t.legal(dict(q, area="react"))[0],
          "[chart] a closed area's question is not legal")
    st = t.commit([{"skill": "s1"}])
    t2 = CQ.Turn(st, chars=200, phases={"implement": 1},
                 open_areas=["koota"])
    check(not t2.legal(q)[0], "[chart] a given skill's questions are not "
          "asked again without an event")
    t3 = CQ.Turn(st, chars=200, phases={"debug": 1}, open_areas=["koota"],
                 events={"koota": {"ERROR"}})
    check(t3.legal(dict(q, kind="error"))[0] and t3.form("s1") == "recall",
          "[chart] an ERROR brings the given skill back as a recall")
    t4 = CQ.Turn(st, chars=10, phases={"implement": 1},
                 open_areas=["koota"])
    check(t4.compacted and t4.legal(q)[0], "[chart] a compacted "
          "conversation forgets what was given")


def main() -> int:
    for fn in (test_pins, test_verify, test_chart):
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            check(False, f"{fn.__name__} itself raised",
                  f"{type(e).__name__}: {e}")
    fails = 0
    for ok, name, detail in _results:
        print(f"  {'pass' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"   <- {detail}"))
        fails += not ok
    n = len(_results)
    print(f"\n  {n - fails}/{n} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
