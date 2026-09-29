#!/usr/bin/env python
"""WHERE ONE SKILL ENDS AND ANOTHER BEGINS: recorded, never gating.

    python mcp/skill_boundaries.py black-holes [--json FILE] [--top N]

The operator's plan (2026-09-27), from docs/research/SKILLS-RESEARCH.md
Part 4.4: write "NOT when" boundaries into descriptions, especially between
near-twin skills, and list generic "black-hole" skills for the operator to
retire. The evidence: confusability is the main selection failure -- one
similar competitor per skill costs 7-30% (Single-Agent Skills 2601.04748),
95% of helpful top-3 hits carry the risky sibling (Right Family, Wrong Skill
2606.10388), and boundary rewriting lifted routing +12.8 while removing
abstract "black-hole" skills added +4.9 (Scaling Laws of Skills
2605.16508).

THREE RECORDS, NONE OF THEM A GATE (no threshold is known for this model;
these rows are how one would be measured):

  boundary_of(description)   whether the description states a "Not for" /
                             "Not when" boundary (distil/4, decompose/3 ask
                             for one where a nearby situation needs other
                             advice); validate records it
  sibling_cases(rule, id)    every SIBLING's own should-cases -- armed skills
                             keyed on the same framework (else language) --
                             run against this skill's rule: a sibling's
                             situation that selects this skill too is a
                             blurred boundary; validate records each
  black_holes(pool)          every armed skill that is a candidate for other
                             skills' should-cases from AREAS it does not
                             share, most areas first: candidates for the
                             operator to rewrite or retire. Never retired
                             automatically.

Deterministic stages only (skill_tests.run_case / skill_classify.match), as
the activation tests run. Read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_BOUNDARY = re.compile(r"(?:^|(?<=[.;:!?]))\s*Not (?:for|when)\b[^.]*\.?")


def boundary_of(description: str) -> dict:
    m = _BOUNDARY.search(description or "")
    return {"stated": bool(m), "text": m.group(0).strip()[:300] if m
            else None}


def area_of(rule: dict) -> frozenset:
    """What a skill is ABOUT: its frameworks and languages, else its
    non-code artifacts, else its domains."""
    import skill_classify as C
    a = C.applies_to(rule)
    terms = a["frameworks"] + a["languages"]
    if terms:
        return frozenset(terms)
    arts = [x for x in a["artifacts"] if x != "code"]
    if arts:
        return frozenset(arts)
    return frozenset((rule or {}).get("domains") or [])


def _should_cases(sid: str, served: int | None) -> list[dict]:
    import skills
    ver = skills.version(sid, served) if served else None
    act = ((ver or {}).get("tests") or {}).get("activation") or {}
    return [c for c in act.get("should") or [] if isinstance(c, dict)]


def _served_rows() -> list[dict]:
    import skills
    return [dict(r, served=r.get("version")) for r in skills.armed()]


def sibling_cases(rule: dict, sid: str, pool: list[dict] | None = None
                  ) -> dict:
    """{siblings, cases, fires, rows[{sibling, case, verdict, why}]}: every
    should-case of every sibling against `rule`."""
    import skill_classify as C
    import skill_tests
    a = C.applies_to(rule)
    key = set(a["frameworks"]) or set(a["languages"])
    rows: list[dict] = []
    sibs = 0
    if key:
        for s in pool if pool is not None else _served_rows():
            if s["id"] == sid:
                continue
            sa = C.applies_to(s.get("rule") or {})
            if not key & (set(sa["frameworks"]) if a["frameworks"]
                          else set(sa["languages"])):
                continue
            sibs += 1
            for c in _should_cases(s["id"], s.get("served")):
                r = skill_tests.run_case(c, rule)
                rows.append({"sibling": s["name"],
                             "case": str(c.get("text") or "")[:120],
                             "verdict": r["verdict"], "why": r["why"][:3]})
    return {"siblings": sibs, "cases": len(rows),
            "fires": sum(1 for r in rows if r["verdict"] != "none"),
            "rows": rows}


def black_holes(pool: list[dict] | None = None) -> list[dict]:
    """Armed skills that are candidates for OTHER areas' should-cases, the
    most areas first. Each: {id, name, area, captured, areas, examples}."""
    import skill_classify as C
    import skill_select
    import skill_tests
    pool = pool if pool is not None else _served_rows()
    areas = {s["id"]: area_of(s.get("rule") or {}) for s in pool}
    got: dict[str, dict] = {}
    for owner in pool:
        cases = _should_cases(owner["id"], owner.get("served"))
        for c in cases:
            sig = C.request_signals(skill_tests.messages_of(c),
                                    c.get("route_class"), c.get("tools"))
            for s in pool:
                if s["id"] == owner["id"]:
                    continue
                if areas[s["id"]] & areas[owner["id"]]:
                    continue
                det = C.match(s.get("rule") or {}, sig)
                if skill_select.verdict(det["strength"], None) == "none":
                    continue
                g = got.setdefault(s["id"], {
                    "id": s["id"], "name": s["name"],
                    "area": sorted(areas[s["id"]]), "captured": 0,
                    "areas": set(), "examples": []})
                g["captured"] += 1
                g["areas"].add(",".join(sorted(areas[owner["id"]])) or "-")
                if len(g["examples"]) < 3:
                    g["examples"].append({"owner": owner["name"],
                                          "case": str(c.get("text") or "")
                                          [:100]})
    out = []
    for g in got.values():
        g["areas"] = sorted(g["areas"])
        out.append(g)
    out.sort(key=lambda g: (-len(g["areas"]), -g["captured"], g["name"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("black-holes",))
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(argv)
    rows = black_holes()
    print(f"  {len(rows)} armed skill(s) are candidates for another area's "
          "own should-case (most areas first; for the operator, never "
          "retired automatically)")
    for g in rows[:a.top]:
        print(f"  {g['name']:<48} area {'/'.join(g['area']) or '-':<22} "
              f"{g['captured']:>3} case(s) from {len(g['areas'])} area(s)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
