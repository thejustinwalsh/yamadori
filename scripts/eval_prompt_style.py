#!/usr/bin/env python
"""Does a decision-router prompt beat a prose prompt on THIS model?

The published guidance is consistent: positive instructions outperform
negative ones, because "never do X" fires the model's attention on X and
requires suppressing an already-activated path, while a positive instruction
just activates the right one. Short unambiguous prohibitions survive; piles of
them do not.

But that guidance is derived from frontier models. Bonsai is a 27B ternary
quantisation of a Qwen3-family model, and quantisation degrades exactly the
kind of fine-grained instruction-following this depends on. So it gets tested
here rather than assumed.

Only the FIRST tool call is scored. It is made with no information, it sets
the trajectory for the whole turn, and it is where a routing prompt should
show up if it shows up anywhere.

    python scripts/eval_prompt_style.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import code_search as cs  # noqa: E402

API = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")
MODEL = os.environ.get("AGENT_MODEL", "bonsai-agent")
REPS = int(os.environ.get("EVAL_REPS", "3"))
TEMP = float(os.environ.get("EVAL_TEMP", "0.3"))

# One defensible first tool each. Ambiguous phrasings are excluded on purpose:
# this measures whether the prompt steers, not whether the model reads minds.
TASKS = [
    ("Where is LineDashedMaterial defined?",                      "find_definition"),
    ("Show me the Material class.",                               "find_definition"),
    ("Open the definition of WebGLPrograms.",                     "find_definition"),
    ("What uses materialLineDashOffset?",                         "find_references"),
    ("I want to rename gapSize -- what depends on it?",           "find_references"),
    ("Is refreshUniformsDash dead code?",                         "find_references"),
    ("Find every TODO comment in the renderers.",                 "grep"),
    ("Where is the uniform totalSize assigned?",                  "grep"),
    ("Which files import WebGLState?",                            "grep"),
    ("Show me lines 40 to 60 of materials/LineDashedMaterial.js", "read_file"),
    ("Open src/materials/Material.js around line 900.",           "read_file"),
    ("How does this library decide when to recompile a shader?",  "search_code"),
    ("How is dashing actually implemented end to end?",           "search_code"),
    ("What is actually in the index?",                            "index_status"),
]

ROUTER = """  The request names a symbol            -> find_definition
  You want what USES or CALLS it        -> find_references
  The text appears verbatim somewhere   -> grep
  The code may use OTHER words          -> search_code
  You have a path and a line range      -> read_file
  You changed something                 -> verify
  A search came back empty              -> index_status"""

# --- A: prose, no routing table, no prohibitions ------------------------------
PROSE = """You are a principal engineer working in an unfamiliar codebase. You
have tools for searching and reading code. Think about which tool best fits
what is being asked, and use the one that will answer most directly and
cheaply. Prefer precise lookups over broad searches when the request gives you
something precise to look up."""

# --- B: the decision router, purely positive ---------------------------------
ROUTER_ONLY = """You are a principal engineer working in an unfamiliar codebase.

Match the situation to the tool:

""" + ROUTER

# --- C: router + two short, sharp prohibitions -------------------------------
# The owner's proposal. Published guidance says short unambiguous negatives
# survive where piles of them do not, so this keeps exactly two, each naming a
# specific observed failure.
ROUTER_SELECT_NEVER = ROUTER_ONLY + """

Two things, and only two:
- Never search for a location you already have. Read it.
- Never state what code does without having read it."""

# --- D: router + the full HARD RULES stack (what we shipped) -----------------
ROUTER_MANY_NEVER = ROUTER_ONLY + """

HARD RULES
- Never state what code does without having read it.
- Never invent an API. If a tool returns nothing useful, try a different one.
- Never search again for a location you already have.
- Never claim something works because it looks right.
- Never change a signature before checking its references.
- Never guess at a convention; read a neighbouring file."""

# --- E: router + costs, no prohibitions --------------------------------------
ROUTER_COSTED = """You are a principal engineer working in an unfamiliar codebase.

Match the situation to the tool. Measured cost per answer:

  find_definition    19 tok   exact, instant    the request names a symbol
  grep               78 tok   exact, fast       it appears verbatim
  read_file           -       exact             you have a path
  search_code       182 tok   two GPU models    the code may use other words

""" + ROUTER

VARIANTS = {
    "A prose":            PROSE,
    "B router":           ROUTER_ONLY,
    "C router+2never":    ROUTER_SELECT_NEVER,
    "D router+6never":    ROUTER_MANY_NEVER,
    "E router+costs":     ROUTER_COSTED,
}


def tools_spec():
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["inputSchema"]}}
            for t in cs.TOOLS]


def first_tool(system: str, task: str, tools) -> tuple[str | None, int]:
    payload = {"model": MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": task}],
               "tools": tools, "max_tokens": 400, "temperature": TEMP}
    req = urllib.request.Request(f"{API}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    tc = d["choices"][0]["message"].get("tool_calls") or []
    return (tc[0]["function"]["name"] if tc else None,
            d.get("usage", {}).get("completion_tokens", 0))


def main() -> None:
    tools = tools_spec()
    print(f"model={MODEL}  temp={TEMP}  reps={REPS}  tasks={len(TASKS)}  "
          f"tools={len(tools)}\n")
    results = {}
    for vname, system in VARIANTS.items():
        score = 0.0
        toks = 0
        no_call = 0
        misses = []
        for task, want in TASKS:
            got = Counter()
            for _ in range(REPS):
                try:
                    g, n = first_tool(system, task, tools)
                except Exception as e:                           # noqa: BLE001
                    print(f"  request failed: {e}")
                    return
                toks += n
                got[g] += 1
                if g is None:
                    no_call += 1
            hits = got[want]
            score += hits / REPS
            if hits < REPS:
                top = got.most_common(1)[0][0]
                misses.append(f"    {task[:40]:<42} want={want:<16} "
                              f"got={top} ({hits}/{REPS} correct)")
        results[vname] = (score, toks, no_call)
        print(f"=== {vname}  {score:.1f}/{len(TASKS)}   "
              f"tokens={toks}  no-call={no_call}")
        for m in misses:
            print(m)
        print()

    print("=" * 70)
    print(f"  {'variant':<20}{'first-call correct':<22}{'tokens':<10}{'no call'}")
    for v, (s, t, nc) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {v:<20}{f'{s:.1f}/{len(TASKS)}':<22}{t:<10}{nc}")
    print("\nNote: n is small. Treat gaps under ~1.5 tasks as noise.")


if __name__ == "__main__":
    main()
