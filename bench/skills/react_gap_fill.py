#!/usr/bin/env python
"""FILL THE GAPS THE PITFALL HARNESS FOUND, through the ONE skills pipeline
(GPU: its model stages -- distil, review, the faithfulness check, prove --
run on the loaded model).

    python bench/skills/react_gap_fill.py --list            # offline
    python bench/skills/react_gap_fill.py --run --model bonsai

The operator (2026-09-29): "Even something as simple as recommending the react
compiler, and where to get the info for setting it up is a skill, we should
be steering toward the react compiler by default." And (via the coordinator,
same day): feed the missing pages through the ONE pipeline to fill the gaps --
the React Compiler setup skill; ref-as-prop and <Context> as provider in the
assured voice ("pass ref as a prop; React 19 needs no forwardRef"), not
dropped by the doubt filter; never a hand edit.

WHAT THE GAPS ARE (bench/skills/pitfall_harness.py --list, 2026-09-29):
  react-setup-*, react-compiler-first-plugin   no skill carries react.dev's
        React Compiler installation page (the install, the Vite forms, the
        compiler first in the Babel plugins)
  react-ref-as-prop, react-context-as-provider  the library's lines are NOT
        served: both mention a future deprecation, which the ASSURED VOICE
        filter (skill_limits.doubt) drops at serving
  react-modal-open-prop                        no skill carries the
        forwardRef page's Pitfall ("If you can express something as a prop,
        you should not use a ref")

HOW: each source is one react.dev page at the PINNED commit (the same bytes
the harness cites: bench/skills/pitfalls/sources/react.dev@75ef18a9172e/,
MANIFEST.json holds each sha256; licence CC BY 4.0, LICENSE-DOCS.md), given
to skills.create(url=..., goal=...) -- the pipeline's own fetch (robots.txt
obeyed), screen, model screen, licence, distil, review, classify, tests,
validate (the faithfulness check), prove, arm -- run INLINE here
(skill_pipeline.run_inline) so its model stages run inside the granted GPU
window and nowhere else. `goal` steers the distil and review stages
(skills.create's own field); the item text is the model's, from the page,
checked by the pipeline. Every GOAL below is UNMEASURED WORDING.

REFUSES without --run, and with --run unless --model is loaded (GET
/running). Writes the live skill store (that is the pipeline's job).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

SHA = "75ef18a9172e4c100b5d5650ae14c395c5c8ef42"
RAW = f"https://raw.githubusercontent.com/reactjs/react.dev/{SHA}/src/content/"
SOURCES = [
    {"name": "react-dev-compiler-setup-vite",
     "page": "learn/react-compiler/installation.md",
     "cases": ["react-setup-new-vite-app", "react-setup-add-feature",
               "react-compiler-first-plugin"],
     "goal": "Steer every new or existing React + Vite project onto the "
             "React Compiler by default: install babel-plugin-react-compiler "
             "as a devDependency and enable it in vite.config the way this "
             "page shows for the project's @vitejs/plugin-react version, "
             "with the compiler first in the Babel plugins. With the "
             "compiler on, components are written without manual useMemo, "
             "useCallback or memo."},
    {"name": "react-dev-ref-as-prop",
     "page": "reference/react/forwardRef.md",
     "cases": ["react-ref-as-prop", "react-modal-open-prop"],
     "goal": "State the current React 19 way as the default: function "
             "components take `ref` as a regular prop, so new components are "
             "written without forwardRef; state such as open/closed is "
             "expressed as props, and refs are kept for imperative DOM work "
             "(focus, scroll, selection)."},
    {"name": "react-dev-context-as-provider",
     "page": "reference/react/createContext.md",
     "cases": ["react-context-as-provider"],
     "goal": "State the current React 19 way as the default: render "
             "`<SomeContext value={...}>` as the provider (not "
             "`<SomeContext.Provider>`), read it with useContext or use."},
]


def running(upstream: str) -> list[str]:
    with urllib.request.urlopen(f"{upstream}/running", timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = d.get("running") if isinstance(d, dict) else d
    return [str((x or {}).get("model")) for x in rows or []
            if isinstance(x, dict)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="bonsai")
    a = ap.parse_args(argv)
    if a.list or not a.run:
        for s in SOURCES:
            print(json.dumps({"name": s["name"], "url": RAW + s["page"],
                              "cases": s["cases"]}))
        return 0
    import model as M
    have = running(M.UPSTREAM)
    if a.model not in have:
        print(f"REFUSED: {a.model} is not loaded (GET /running: {have})")
        return 2
    import skill_pipeline
    import skills
    out = []
    for s in SOURCES:
        rec = skills.create(url=RAW + s["page"], name=s["name"],
                            goal=s["goal"], author="react_gap_fill",
                            meta={"gap_fill": {"cases": s["cases"],
                                               "commit": SHA}},
                            enqueue_first=False)
        stages = skill_pipeline.run_inline(rec["id"], 1, model=True,
                                           max_steps=30)
        got = skills.get(rec["id"]) or {}
        out.append({"name": s["name"], "id": rec["id"],
                    "status": got.get("status"), "reason": got.get("reason"),
                    "stages": sorted(stages)})
        print(json.dumps(out[-1]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
