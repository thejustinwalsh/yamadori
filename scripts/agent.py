#!/usr/bin/env python
"""Minimal agent loop for testing the stack end to end.

Not a replacement for Hermes -- a harness to observe what the model actually
does: which tools it reaches for, in what order, how many tokens each step
costs, and whether it answers from the code or from memory.

    python scripts/agent.py "add a dashOffset property to LineDashedMaterial"

Every tool call is logged with its argument and result size, so tool-selection
behaviour is visible rather than inferred.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

# Windows consoles default to cp1252, which raises UnicodeEncodeError on the
# arrows and box characters models routinely emit -- crashing the harness
# AFTER a correct answer was produced. Errors are replaced rather than raised.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import code_search as cs  # noqa: E402

API = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")
MODEL = os.environ.get("AGENT_MODEL", "bonsai-agent")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "12"))
EFFORT = os.environ.get("AGENT_EFFORT")   # low | medium | xhigh
# A final answer that cites files and shows a diff does not fit in 1500 tokens.
# Too low a cap truncates the answer mid-sentence, which reads like a model
# failure and is not one.
MAX_TOK = int(os.environ.get("AGENT_MAX_TOKENS", "3000"))

SYSTEM = """You ship with a code-intelligence stack. It is not something your
harness had to provide and not something you should hesitate to spend: it runs
on this machine, against an index of this codebase, over a warm GPU. No
network, no rate limit, no quota. Calling it is close to free; guessing is not.

WHAT YOU HAVE, and what each is best at

  find_definition   ~19 tok   exact, instant. The request names a symbol.
  grep              ~78 tok   exact, fast. It appears verbatim somewhere.
  read_file          cheap    you already have a path and a line range.
  find_references    cheap    what USES / CALLS / DEPENDS ON this. Ask before
                              renaming, deleting, or changing a signature.
  search_code      ~182 tok   embeddings + cross-encoder reranker over the
                              index. Finds code that means what you asked even
                              when it says it differently. The expensive one,
                              and the only one that survives not knowing the
                              vocabulary.
  judge              cheap    a yes/no you are about to act on.
  compact            cheap    squeeze a long log or file down to what matters,
                              identifiers and errors kept verbatim.
  verify             varies   run this project's own checks. No argument lists
                              them.
  index_status       free     what is actually indexed. Worth one call when a
                              search comes back empty.

Reach for the cheapest one that can answer. They compose: find_definition to
land, read_file to widen, find_references to see the blast radius.

WHAT THEY DO NOT COVER

They read; they do not write. Editing files and running arbitrary commands
come from your harness -- use those, and use these to know what to change
before you change it and what moved after.

The index has edges. It covers specific roots, and a tool that finds nothing
will tell you what it does cover rather than leaving you to guess. Read that
and adapt; do not retry the same call with the arguments permuted.

HONESTY

- Never state what code does without having read it. You can read it cheaply,
  so there is no excuse to infer.
- Never invent an API. If a tool returns nothing useful, try a different one.
- Follow the conventions already in the file you are editing.
- A result you did not check is a guess, however good the reasoning behind it.
  Say which is which."""


def tools_spec():
    """Reuse the MCP tool definitions so the harness cannot drift from them."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["inputSchema"]}}
            for t in cs.TOOLS]


def call_model(messages, tools):
    payload = {"model": MODEL, "messages": messages,
               "tools": tools, "max_tokens": MAX_TOK}
    if EFFORT:
        payload["reasoning_effort"] = EFFORT
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d, time.time() - t0


def run_tool(name, args):
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": name, "arguments": args}}
    resp = cs.handle(req)
    try:
        return resp["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return json.dumps(resp)[:500]


def main():
    task = " ".join(sys.argv[1:]) or "describe this codebase"
    tools = tools_spec()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]

    print(f"TASK: {task}\n" + "=" * 78)
    used, total_tok, t_start = [], 0, time.time()

    for step in range(1, MAX_STEPS + 1):
        d, el = call_model(messages, tools)
        ch = d["choices"][0]
        msg = ch["message"]
        total_tok += d.get("usage", {}).get("completion_tokens", 0)

        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        if content and not calls:
            print(f"\n--- step {step}: ANSWER ({el:.1f}s) ---")
            print(content)
            break

        if content:
            print(f"\n[{step}] reasoning: {content[:200]}")

        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": calls})

        for tc in calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            used.append(fn)
            brief = json.dumps(args)[:90]
            out = run_tool(fn, args)
            print(f"[{step}] {fn:<16} {brief:<92} -> {len(out):>6} chars  ({el:.1f}s)")
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": out[:6000]})
    else:
        print(f"\n(hit max steps {MAX_STEPS})")

    print("\n" + "=" * 78)
    print(f"steps with tools : {len(used)}")
    print(f"tool sequence    : {' -> '.join(used) if used else '(none)'}")
    print(f"completion tokens: {total_tok}")
    print(f"wall clock       : {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
