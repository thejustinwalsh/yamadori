#!/usr/bin/env python
"""Server-sent events for the proxy, including what happens during tool hops.

THE PROBLEM THIS SOLVES

The proxy runs a tool loop server-side, which is what makes it zero-config:
the client never learns the tools exist. But it also means the client sees
nothing at all until the whole loop finishes. Measured, a simple lookup took
5.4 seconds of complete silence, and a long generation at large context can
take minutes. Silence is indistinguishable from a hang, and a user who cannot
tell the difference will kill it.

Worse, the previous version did not merely pause -- it stripped `stream` from
the request entirely, so a client that asked for streaming got a single blob
whatever it asked for.

WHAT GETS STREAMED

  1. the preamble, immediately, so something appears within milliseconds
  2. a line per tool call as it runs, which is the part that was silent
  3. the final answer, forwarded token by token from upstream

The tool narration is real information rather than a spinner: which tool, what
it was asked, how much came back. It is the same trace that would otherwise be
invisible, and it is what makes a slow turn legible instead of alarming.

FORMAT

OpenAI's chat completion chunk format, because the point of this proxy is that
a client needs no special handling. Anything that can read a streamed
completion can read this.
"""
from __future__ import annotations

import json
import time
import urllib.request
import uuid


def chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> bytes:
    payload = {
        "id": cid, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def text_chunk(cid: str, model: str, text: str) -> bytes:
    return chunk(cid, model, {"content": text})


DONE = b"data: [DONE]\n\n"


def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def describe_call(name: str, args: dict) -> str:
    """One line naming the tool and its most informative argument.

    Showing the argument matters: 'searching' is a spinner, while
    'find_by_pattern "createWorld"' tells the reader whether the agent
    understood the question, which is the thing they actually want to know
    while they wait.
    """
    for key in ("symbol", "pattern", "query", "path", "check"):
        if args.get(key):
            return f"{name} {json.dumps(args[key])[:80]}"
    return name


def stream_upstream(upstream: str, payload: dict, cid: str, model: str,
                    timeout: int = 1800):
    """Forward a streamed completion, rewriting ids so the client sees one call.

    Upstream emits its own completion id per request. Since a single client
    request may involve several upstream calls, passing those through would
    look like several different completions to anything tracking ids.
    """
    body = dict(payload)
    body["stream"] = True
    req = urllib.request.Request(f"{upstream}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            # A thinking model streams reasoning_content before any answer.
            # Passing it through as content would splice its private reasoning
            # into the reply; dropping it silently would make the wait look
            # like a hang again. It is surfaced as a heartbeat instead.
            if delta.get("content"):
                yield text_chunk(cid, model, delta["content"])
            elif delta.get("reasoning_content"):
                yield chunk(cid, model, {})
