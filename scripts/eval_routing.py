#!/usr/bin/env python
"""A/B the system prompt on FIRST-CALL tool selection.

Only the first call is scored. It is the one made with no information, it sets
the trajectory for the rest of the turn, and it is where the observed mistake
was: the agent led with search_code (182 tokens, imprecise) for a symbol it
could name, where find_definition (19 tokens, exact) was correct.

Each task has one defensible first tool. Ambiguous phrasings are deliberately
excluded -- this measures whether the prompt steers, not whether the model can
read minds.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import code_search as cs  # noqa: E402

API = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")
MODEL = os.environ.get("AGENT_MODEL", "bonsai-agent")

TASKS = [
    ("Where is LineDashedMaterial defined?", "find_definition"),
    ("Show me the Material class.", "find_definition"),
    ("What uses materialLineDashOffset?", "find_references"),
    ("I want to rename gapSize -- what depends on it?", "find_references"),
    ("Find every TODO comment in the renderers.", "grep"),
    ("Where is the uniform totalSize assigned?", "grep"),
    ("Show me lines 40 to 60 of materials/LineDashedMaterial.js", "read_file"),
    ("Open src/materials/Material.js around line 900.", "read_file"),
    ("How does this library decide when to recompile a shader program?", "search_code"),
]

ROUTING = """  You know the identifier            -> find_definition
  You are changing or deleting it    -> find_references FIRST
  It appears verbatim somewhere      -> grep
  The code may use OTHER words       -> search_code  (last resort, costs most)
  You have a path and line range     -> read_file
  A yes/no you will act on           -> judge"""

BASE = """You are a principal engineer working in an unfamiliar codebase.

Work in sub-tasks. For each, pick the single cheapest tool that can answer.

""" + ROUTING + """

Never state what code does without having read it."""

FIRST_CALL = """You are a principal engineer working in an unfamiliar codebase.

YOUR FIRST TOOL CALL MATTERS MOST. You make it knowing least, and it sets the
direction for everything after. Before calling anything, ask: does the request
already contain an identifier or a file path? If it does, you do NOT need to
search for it.

""" + ROUTING + """

search_code runs two GPU models and returns ~10x the tokens of a symbol
lookup. It is correct ONLY when the codebase might use different words than
the request does. If the request names something, find_definition or grep
will answer it exactly, and cheaper.

Never state what code does without having read it."""

COSTED = """You are a principal engineer working in an unfamiliar codebase.

Pick the cheapest tool that can answer. Measured cost per answer:

  find_definition    19 tokens   exact, instant     request names a symbol
  grep               78 tokens   exact, fast        it appears verbatim
  read_file           -          exact              you have a path
  search_code       182 tokens   fuzzy, 2 GPU models  ONLY if the code might
                                                     use other words

""" + ROUTING + """

Never state what code does without having read it."""

REFINED = COSTED.replace(
    "  You are changing or deleting it    -> find_references FIRST",
    "  What USES / CALLS / DEPENDS ON it  -> find_references\n"
    "    (also before renaming or deleting -- check dependents first)")

# Tool SELECTION is a classification, not creative writing. The stack default
# is temp 1.0 (Qwen's thinking-mode value) which is right for prose and wrong
# for picking one of eight labels -- repeated identical prompts gave different
# tools. Test the sampler as well as the wording.
VARIANTS = {
    "refined @ t=1.0": (REFINED, 1.0),
    "refined @ t=0.3": (REFINED, 0.3),
    "refined @ t=0.0": (REFINED, 0.0),
}


def tools_spec():
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["inputSchema"]}}
            for t in cs.TOOLS]


def first_tool(system, task, tools, temp=None):
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": task}],
        "tools": tools, "max_tokens": 400,
    }
    if temp is not None:
        payload["temperature"] = temp
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    tc = d["choices"][0]["message"].get("tool_calls") or []
    return (tc[0]["function"]["name"] if tc else None,
            d.get("usage", {}).get("completion_tokens", 0))


def main():
    tools = tools_spec()
    results = {}
    REPS = 3
    for vname, (system, temp) in VARIANTS.items():
        ok = 0
        toks = 0
        rows = []
        for task, want in TASKS:
            hits = 0
            got = None
            for _ in range(REPS):
                got, n = first_tool(system, task, tools, temp)
                toks += n
                hits += (got == want)
            hit = hits == REPS
            ok += hits / REPS
            if hits not in (0, REPS):
                rows.append((task, want, f"{got} (unstable {hits}/{REPS})", False))
            else:
                rows.append((task, want, got, hit))
        results[vname] = (ok, toks, rows)
        print(f"\n=== {vname}  ({ok}/{len(TASKS)}) ===")
        for task, want, got, hit in rows:
            if not hit:
                print(f"   MISS {task[:46]:<48} want={want:<16} got={got}")

    print("\n" + "=" * 62)
    print(f"  {'variant':<14}{'first-call correct':<22}{'tokens'}")
    for v, (ok, toks, _) in results.items():
        print(f"  {v:<14}{f'{ok:.1f}/{len(TASKS)}':<22}{toks}")


if __name__ == "__main__":
    main()
