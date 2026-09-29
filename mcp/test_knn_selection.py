#!/usr/bin/env python
"""THE EXAMPLE kNN VOTE IN THE SELECTOR (docs/PACKAGE-ONBOARDING.md 6.5,
item J), offline.

    python mcp/test_knn_selection.py      -> "N/M checks passed"

  1. A prose turn never queries the index; a turn whose newest evidence is
     JS/TS code the model read or wrote does, with that code (newest first,
     within skill_match.rank_all's 4,000-character bound).
  2. The package at the top of the tally that detect() did not put in play
     opens a WEAK area ("knn: m of k neighbours"), all of them on a tie.
  3. With the STUB decider a vote alone delivers nothing: no lead is pushed,
     and the package's first appearance is not spent.
  4. A package detect() already put in play is not opened twice.
  5. Inside a package question, a skill the neighbours' chunks label (its
     code-shaped topics among their identifiers: example_knn.derived_skills)
     ranks after the lead and before the rest.
  6. x_yamadori.skills.knn records group ranks, packages, the tally and
     what opened -- never example text; a missing index is recorded, not an
     error.
"""
from __future__ import annotations

import os
import sys
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import offline_stores  # noqa: E402

_TMP = offline_stores.isolate("yamadori_test_knn_selection_")

import skill_match as M  # noqa: E402
import skill_select  # noqa: E402

FAILS: list[str] = []
N = [0]


def check(ok, name, detail=""):
    N[0] += 1
    if ok:
        print(f"  pass  {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL  {name}" + (f"\n        <- {str(detail)[:600]}"
                                   if detail != "" else ""))


def _sk(sid, name, fw, topics=(), lead=None):
    return {"id": sid, "name": name, "version": 1, "description": "Use when",
            "title": name, "body": "b", "items": [],
            "rule": {"applies_to": {"frameworks": [fw] if fw else [],
                                    "languages": ["typescript"],
                                    "artifacts": ["code"]},
                     "topics": list(topics)},
            "lead_for": lead}


def _area(rule):
    a = (rule or {}).get("applies_to") or {}
    return (a.get("frameworks") or a.get("languages") or ["code"])[0]


POOL = [_sk("k1", "koota-lead", "koota", lead="koota"),
        _sk("k2", "koota-queries", "koota", topics=["world.query"]),
        _sk("k3", "koota-update", "koota", topics=["updateEach"]),
        _sk("m1", "math-lead", "pmndrs_math", lead="math"),
        _sk("m2", "math-noise", "pmndrs_math", topics=["createNoise2D"])]
LEADS = {"koota": "koota-lead", "math": "math-lead"}


def plan(pkg_play, knn, state=None, old=()):
    return M.plan_turn(pool=POOL, old=list(old), pkg_play=pkg_play,
                       pkg_state=state, given=set(), cos=None,
                       legal_whys={}, phase_changed=False,
                       leads_for=LEADS.get, area_of=_area,
                       subject_areas=lambda s: set(), knn=knn)


def vote(top, tally, groups, ok=True):
    return {"ok": ok, "index": "sig1", "k": 3, "top": top, "tally": tally,
            "groups": groups, "why": "test"}


def play(*pk, events=("ASKED",)):
    return {p: {"why": [{"how": "name", "what": p, "where": "user"}],
                "version": None, "events": list(events),
                "strength": "strong"} for p in pk}


def test_weak_area():
    k = vote(["koota"], {"koota": 2, "math": 1},
             [{"rank": 1, "group": "g1", "packages": ["koota"],
               "identifiers": ["updateEach", "spawn"]},
              {"rank": 2, "group": "g2", "packages": ["koota"],
               "identifiers": []},
              {"rank": 3, "group": "g3", "packages": ["math"],
               "identifiers": ["createNoise2D"]}])
    out, rec, st = plan({}, k)
    q = [x for x in rec["questions"] if x["area"] == "koota"]
    check(q and not q[0]["hard"] and q[0]["strength"] == "weak"
          and q[0]["evidence"] == ["knn: 2 of 3 neighbours"],
          "[knn] the tally's top package opens a WEAK area, with m of k",
          q[:1])
    check(not [x for x in rec["questions"] if x["area"] == "pmndrs_math"],
          "[knn] only the top of the tally opens (math had 1 vote)",
          [x["area"] for x in rec["questions"]])
    check(out == [] and "koota" not in (st.get("seen") or {}),
          "[knn] the stub delivers nothing on a vote alone; the package's "
          "first appearance is not spent", (out, st))
    check(q and q[0]["leads"] == [], "[knn] no lead is pushed on a vote "
          "alone", q[0].get("leads") if q else None)
    r = rec.get("knn") or {}
    check(r.get("opened_weak") == ["koota"] and r.get("k") == 3
          and r.get("tally") == {"koota": 2, "math": 1}
          and all(set(g) == {"rank", "packages"} for g in r["groups"]),
          "[record] x_yamadori.skills.knn: ranks, packages, tally, opened -- "
          "no text, no identifiers", r)
    check(r.get("labelled") == ["koota-update", "math-noise"],
          "[record] the skills the neighbours label (derived at load)",
          r.get("labelled"))


def test_tie_and_in_play():
    k = vote(["koota", "math"], {"koota": 1, "math": 1},
             [{"rank": 1, "group": "g1", "packages": ["koota"]},
              {"rank": 2, "group": "g2", "packages": ["math"]}])
    _o, rec, _s = plan({}, k)
    check(sorted((rec.get("knn") or {}).get("opened_weak") or []) ==
          ["koota", "math"], "[knn] a tie opens every package at the top",
          rec.get("knn"))
    _o, rec, _s = plan(play("koota"), k)
    q = [x for x in rec["questions"] if x["area"] == "koota"]
    check(q and q[0]["hard"] and (rec.get("knn") or {}).get("opened_weak")
          == ["math"], "[knn] a package detect() put in play is not opened "
          "again (its question stays HARD)", (q[:1], rec.get("knn")))


def test_rank_tier():
    k = vote(["koota"], {"koota": 1},
             [{"rank": 1, "group": "g1", "packages": ["koota"],
               "identifiers": ["updateEach"]}])
    _o, rec, _s = plan(play("koota"), k)
    q = next(x for x in rec["questions"] if x["area"] == "koota")
    names = [c["name"] for c in q["candidates"]]
    check(names[:3] == ["koota-lead", "koota-update", "koota-queries"],
          "[rank] the lead first, then the skill the neighbours label, then "
          "the rest", names)
    _o, rec, _s = plan(play("koota"), None)
    q = next(x for x in rec["questions"] if x["area"] == "koota")
    check([c["name"] for c in q["candidates"]][:3] ==
          ["koota-lead", "koota-queries", "koota-update"]
          and "knn" not in rec, "[rank] without a vote the order and the "
          "record are as before", [c["name"] for c in q["candidates"]])


def test_no_index():
    k = vote([], {}, [], ok=False)
    out, rec, _s = plan({}, k)
    check(out == [] and (rec.get("knn") or {}).get("ok") is False,
          "[knn] a vote that could not run (no index) is recorded and "
          "changes nothing", rec.get("knn"))


def test_query():
    calls = []
    fake = types.ModuleType("example_knn")
    fake.vote = lambda code, embed=None: calls.append(code) or {
        "ok": True, "top": [], "tally": {}, "groups": [], "k": 1}
    real = sys.modules.get("example_knn")
    sys.modules["example_knn"] = fake
    try:
        prose = [{"role": "user", "content": "make the particles faster"}]
        got = skill_select.knn_vote(prose, {"typescript": "asked"})
        check(got is None and not calls, "[query] a prose turn (a language "
              "only named) never queries the index", got)
        code = [{"role": "user", "content": "fix this\n```ts\nconst w = "
                 "createWorld()\nw.spawn(Position)\n```"}]
        got = skill_select.knn_vote(code, {"typescript": "fence"})
        check(got and got.get("ok") and calls and "createWorld()" in
              calls[-1] and "fix this" not in calls[-1],
              "[query] a turn with JS/TS code queries with that code only",
              calls[-1:])
        big = [{"role": "user", "content": "```ts\n" + "a" * 9000 + "\n```"}]
        skill_select.knn_vote(big, {"typescript": "fence"})
        check(len(calls[-1]) <= 4000, "[query] within the 4,000-character "
              "state bound rank_all applies", len(calls[-1]))
        py = [{"role": "user", "content": "```python\nx = 1\n```"}]
        got = skill_select.knn_vote(py, {"python": "fence"})
        check(got is None, "[query] only the examples' own languages "
              "(JS/TS) query it", got)
    finally:
        if real is not None:
            sys.modules["example_knn"] = real
        else:
            sys.modules.pop("example_knn", None)


def test_vote_identifiers():
    import example_knn
    src = open(example_knn.__file__, encoding="utf-8").read()
    check('"identifiers": list(row.get("identifiers")' in src,
          "[vote] each voting group carries its chunk's code-shaped "
          "identifiers (never text), for the skill labels")


def main() -> int:
    for t in (test_weak_area, test_tie_and_in_play, test_rank_tier,
              test_no_index, test_query, test_vote_identifiers):
        try:
            t()
        except Exception:                                        # noqa: BLE001
            check(False, f"{t.__name__} raised", traceback.format_exc())
    print(f"\n  {N[0] - len(FAILS)}/{N[0]} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
