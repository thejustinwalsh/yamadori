#!/usr/bin/env python
"""A frontier skill through the frontier path, offline: the worked example.

    python bench/skills/frontier_example.py

Takes the local fixture bench/skills/fixtures/frontier/systematic-debugging/
SKILL.md (Hermes Agent's bundled skill, MIT; SOURCE.md says where it came
from) through the one pipeline in a TEMP store: screen -> licence ->
decompose -> (each child) classify -> tests -> validate -> arm. No network,
no live model: the model's replies are stand-ins -- the decompose reply is
bench/skills/fixtures/frontier/decompose_reply.txt, HAND-WRITTEN by reading
the skill (not model output); tag, tests and faithful return minimal
replies. What it demonstrates is everything AROUND the model: the screen,
the licence from the frontmatter line, every quote verified against the
source, the caps, the taxonomy, the activation tests and arming. Prints the
children's SKILL.md bodies and what was dropped.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIX = os.path.join(HERE, "fixtures", "frontier")
_TMP = tempfile.mkdtemp(prefix="yamadori_frontier_example_")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["YAMADORI_GPU_ROOM"] = "0"
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import skill_md  # noqa: E402
import skill_pipeline  # noqa: E402
import skill_prompts  # noqa: E402
import skill_screen  # noqa: E402
import skills  # noqa: E402


def stand_in(system, user, *, max_tokens, temperature=0.2, purpose=""):
    if system == skill_screen.SCREEN_SYSTEM:
        return '{"verdict": "clean", "findings": []}'
    if system == skill_prompts.DECOMPOSE_SYSTEM:
        with open(os.path.join(FIX, "decompose_reply.txt"),
                  encoding="utf-8") as f:
            return f.read()
    if system == skill_prompts.TESTS_SYSTEM:
        return '{"should": [], "should_not": [], "behaviour": []}'
    if system == skill_prompts.REVIEW_SYSTEM:
        return '{"items": []}'
    if system == skill_prompts.FAITHFUL_SYSTEM:
        # One verdict per item: silence is not a verdict (2026-09-27).
        import re
        return json.dumps({"items": [
            {"i": int(n), "faithful": True, "why": "example"}
            for n in re.findall(r"(?m)^(\d+)\. ", user)]})
    return json.dumps({"artifacts": ["code"], "languages": [],
                       "frameworks": [], "situations": []})


def main() -> int:
    skill_pipeline.ask_model = stand_in
    with open(os.path.join(FIX, "systematic-debugging", "SKILL.md"),
              encoding="utf-8") as f:
        src = f.read()
    s = skills.create(text=src, author="example", goal="compact debugging "
                      "habits for a small model")
    out = skill_pipeline.run_inline(s["id"], 1)
    parent = skills.get(s["id"])
    ver = skills.version(s["id"], 1)
    print(f"source: {len(src):,} characters, {parent['source_kind']}; "
          f"status {parent['status']}: {parent['reason']}")
    print(f"screen: {'ok' if (ver['screen'].get('deterministic') or {}).get('ok') else ver['screen']}")
    print(f"licence: {json.dumps(ver['licence'])}")
    print(f"stages: {list(out)}")
    for c in skills.children(s["id"]):
        v = skills.version(c["id"], 1)
        print(f"\n=== {c['name']}: {c['status']}"
              + (f" -- {c['reason']}" if c.get("reason") else ""))
        if v.get("text"):
            sk = skill_md.parse(v["text"])
            print(f"description: {sk['description']}")
            print(f"category: {json.dumps(sk['yamadori'].get('category'))}")
            print(f"provenance: {json.dumps(sk['yamadori'].get('provenance'))}")
            print(f"activation: {json.dumps(sk['yamadori'].get('tests'))}")
            print(skill_md.injection(sk))
        for d in (v.get("validate") or {}).get("dropped") or []:
            print(f"  dropped: {d['item'][:90]} -- {d['why']}")
    print(f"\ntemp store: {_TMP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
