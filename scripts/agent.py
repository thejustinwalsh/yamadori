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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))
import code_search as cs  # noqa: E402

API = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")
MODEL = os.environ.get("AGENT_MODEL", "bonsai-agent")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "12"))

SYSTEM = """You are a principal engineer working in an unfamiliar codebase.

Work in sub-tasks. For each, pick the single cheapest tool that can answer,
and say in one short line why that tool and not another.

  You know the identifier            -> find_definition
  You are changing or deleting it    -> find_references FIRST
  It appears verbatim somewhere      -> grep
  The code may use OTHER words       -> search_code  (last resort, costs most)
  You have a path and line range     -> read_file
  A yes/no you will act on           -> judge

HARD RULES
- Never state what code does without having read it.
- Never invent an API. If a tool returns nothing useful, try a different one.
- Follow the conventions already in the file you are editing.

When you have read enough, produce the actual edit as a diff or a complete
replacement function. Do not describe what you would do -- do it."""


def tools_spec():
    """Reuse the MCP tool definitions so the harness cannot drift from them."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["inputSchema"]}}
            for t in cs.TOOLS]


def call_model(messages, tools):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "tools": tools, "max_tokens": 1500}).encode()
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
