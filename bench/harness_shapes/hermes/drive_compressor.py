#!/usr/bin/env python
"""Hermes' REAL context compressor, driven offline on synthetic agentic
conversations, to capture the shapes a compaction takes on the wire.

    HERMES_AGENT_DIR=C:/Users/jwals/AppData/Local/hermes/hermes-agent \\
      <hermes venv python> bench/harness_shapes/hermes/drive_compressor.py \\
      [mcp/fixtures/hermes_compactions.json]

Written 2026-10-07 for "Why do Hermes' flattened compactions not map onto the
stored conversation?" (mcp/compaction.py, THE MAPPING'S FAILURE). The pagoda
run's transcripts were not kept (the corpus keeps 2,000 characters of a
request), so the compressor itself is run: `ContextCompressor.compress()` on
a conversation we build, its summariser (`call_llm`) replaced by a stub that
records the prompt it was handed and answers with a short synthetic summary
that opens with our session line, as the proxy's does. Every compaction is
fed the history the previous one produced, grown by more steps -- what Hermes
does -- so the carriers (the summary row, merged into the first row of the
kept tail, with the restated unfinished request) take every shape the real
code gives them. Nothing here is the operator's: the tasks, files and tool
outputs are stand-ins.

Per scenario and compaction the fixture keeps `history` (the messages Hermes
held when it compacted: what the client sent the proxy last, plus the tool
results it appended since), `prompt` (the summariser's request, which is the
flattened compaction the proxy receives) and `summary`.

The tail budget is set small so a short history has a middle to summarise;
nothing else about the compressor is changed. Hermes commit: the install's
(`git rev-parse HEAD`), recorded in the fixture.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import types

HERMES = os.environ.get("HERMES_AGENT_DIR",
                        r"C:\Users\jwals\AppData\Local\hermes\hermes-agent")
SESSION = "98f09f5a5a17"

sys.path.insert(0, HERMES)
import agent.context_compressor as cc                      # noqa: E402

PROMPTS: list[str] = []


def fake_call_llm(**kw):
    PROMPTS.append(kw["messages"][0]["content"])
    n = len(PROMPTS)
    msg = types.SimpleNamespace(
        content=(f"yamadori session {SESSION}\n\n## Historical Task Snapshot\n"
                 f"stand-in\n\n## Goal\nSynthetic summary {n}.\n\n"
                 "## Completed Actions\n1. Ran the stand-in steps."),
        reasoning_content=None, tool_calls=None)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
        usage=None, model="stub")


cc.call_llm = fake_call_llm


def scenario(name: str, user_every: int, final_answers: bool, big: int,
             tail: int, rounds: int = 4, steps: int = 9) -> dict:
    comp = cc.ContextCompressor(model="yamadori", threshold_percent=0.5,
                                config_context_length=141824, quiet_mode=True,
                                protect_last_n=20, summary_target_ratio=0.20)
    comp.tail_token_budget = tail
    n = [0]

    def step(content: str) -> list[dict]:
        n[0] += 1
        tid = "call_%s_%08x" % (SESSION, 0xc247b623 + n[0])
        return [{"role": "assistant", "content": content,
                 "tool_calls": [{"id": tid, "type": "function", "function": {
                     "name": "terminal",
                     "arguments": json.dumps({"command": f"step {n[0]}"})}}]},
                {"role": "tool", "tool_call_id": tid,
                 "content": f"output of step {n[0]} " + "x" * big}]

    cur = [{"role": "system", "content": "You are a stand-in agent."},
           {"role": "user", "content": "Build the stand-in scene."}]
    for i in range(14):
        cur += step(f"Working on part {i}." if i % 3 else "")
    out = {"name": name, "compactions": []}
    for rnd in range(1, rounds + 1):
        if rnd > 1:
            for i in range(steps + rnd):
                cur += step(f"Round {rnd} step {i}." if i % 2 else "")
                if user_every and i % user_every == user_every - 1:
                    if final_answers:
                        cur.append({"role": "assistant",
                                    "content": f"Finished part {i}."})
                    cur.append({"role": "user", "content": "Continue"})
        before = copy.deepcopy(cur)
        n0 = len(PROMPTS)
        res = comp.compress(copy.deepcopy(cur), current_tokens=75000)
        if len(PROMPTS) == n0:
            continue
        out["compactions"].append({
            "round": rnd, "history": before, "prompt": PROMPTS[-1],
            "after": len(res)})
        cur = res
    return out


def main() -> None:
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "mcp",
        "fixtures", "hermes_compactions.json")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERMES,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    # name, a user turn every N steps (0: the task is the only one), an
    # assistant answer before it, a tool output's size, the tail budget.
    # The first is the pagoda run's shape: ONE request and a long tool loop.
    scenarios = [scenario("one task, a long tool loop", 0, False, 100, 300),
                 scenario("one task, a longer tail", 0, False, 100, 700),
                 scenario("the user says continue", 3, False, 100, 300),
                 scenario("answers, then continue", 3, True, 100, 300)]
    data = {"hermes_commit": commit, "session": SESSION,
            "scenarios": scenarios}
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(dest, sum(len(s["compactions"]) for s in scenarios), "compactions")


if __name__ == "__main__":
    main()
