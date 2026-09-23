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
  2. every hop's reasoning, live, as `reasoning_content` deltas
  3. a line per tool call as it runs, which is the part that was silent
  4. the final answer, forwarded token by token as it is generated -- ONCE.

Since 2026-09-22 the proxy streams every hop through its own reader
(`proxy._post_events` + `forward_delta` here) and the answer is the hop that
made no tool call. `stream_upstream` and `stream_hop` below are no longer on
the proxy's path; `stream_upstream` regenerated an answer the loop had already
produced (docs/CONSTRAINTS.md item 10b). Nothing in the stack calls them now;
they are kept, with their tests, until someone decides to delete them.

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


def chunk(cid: str, model: str, delta: dict, finish: str | None = None,
          usage: dict | None = None, extra: dict | None = None) -> bytes:
    """One SSE chunk. `extra` adds top-level keys (the proxy's `x_yamadori`
    on the final chunk); OpenAI clients ignore keys they do not know."""
    payload = {
        "id": cid, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        payload["usage"] = usage
    if extra:
        payload.update(extra)
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def text_chunk(cid: str, model: str, text: str) -> bytes:
    return chunk(cid, model, {"content": text})


def forward_delta(cid: str, model: str, delta: dict) -> bytes | None:
    """An upstream delta re-issued under OUR id: reasoning and content only.

    Reasoning goes out as `reasoning_content`, the channel clients render as
    thinking and do not echo back, so the wait is visible without the
    reasoning ever entering the reply. Tool-call deltas are NOT forwarded
    here: until the hop ends nobody knows whether a call is ours (run
    server-side, never shown) or the client's (forwarded whole by the proxy
    once the hop resolves).
    """
    out = {}
    if delta.get("reasoning_content"):
        out["reasoning_content"] = delta["reasoning_content"]
    if delta.get("content"):
        out["content"] = delta["content"]
    return chunk(cid, model, out) if out else None


DONE = b"data: [DONE]\n\n"


class UpstreamError(RuntimeError):
    """The model server reported an error INSIDE a stream that began with 200."""


def _raise_if_error(d: dict) -> None:
    """Turn an in-stream error event into an exception.

    llama-server can fail after the headers are sent -- context exceeded,
    slot killed -- and says so as `data: {"error": {...}}` with no `choices`.
    Both stream readers used to read that as an empty delta and carry on, so
    the caller received a stream that ended cleanly with nothing in it: an
    outage delivered as an empty answer. Raising puts it on the proxy's
    existing `[stream error: ...]` path instead.
    """
    if isinstance(d, dict) and d.get("error") and not d.get("choices"):
        e = d["error"]
        msg = e.get("message") if isinstance(e, dict) else str(e)
        raise UpstreamError(f"upstream reported an error mid-stream: {msg}")


def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def describe_call(name: str, args: dict) -> str:
    """One line naming the tool and its most informative argument.

    Showing the argument matters: 'searching' is a spinner, while
    'find_by_pattern "createWorld"' tells the reader whether the agent
    understood the question, which is the thing they actually want to know
    while they wait.
    """
    # The proxy decodes the model's arguments with json.loads, which returns
    # a list or a string as readily as an object. `args.get` on either raised
    # AttributeError inside the stream, killing the reply over a label.
    if not isinstance(args, dict):
        return name
    for key in ("symbol", "pattern", "query", "path", "check", "prompt"):
        if args.get(key):
            return f"{name} {json.dumps(args[key])[:80]}"
    return name


def stream_hop(upstream: str, payload: dict, cid: str, model: str,
               timeout: int = 3600):
    """Stream a TOOL HOP, yielding events while it runs and the message at the end.

    A hop cannot be forwarded verbatim, because until it finishes nobody knows
    whether it is an answer or a tool call, and a token streamed from a hop
    that turns out to be a call would have to be retracted. The previous
    version drew the obvious conclusion and ran hops non-streamed -- which
    meant the connection stayed open but sat silent for the whole hop, and on
    a long one that is indistinguishable from a hang.

    The resolution is that the two things arriving on this connection are on
    different clocks. The hop's own text has to be held back; everything ELSE
    about the hop is known immediately and is exactly what a waiting reader
    wants:

      - the model has started thinking            (reasoning tokens arriving)
      - it has decided to call `find_definition_opt "Trait"`
                                                  (the name, the moment it
                                                   appears in the delta, long
                                                   before the hop returns)

    So the stream stays open and those are merged in live, while the content
    accumulates privately until the hop resolves. Yields ("event", bytes) for
    anything to send now and ("done", message) once, at the end.
    """
    body = dict(payload)
    body["stream"] = True
    req = urllib.request.Request(f"{upstream}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    announced: set[int] = set()
    finish = None

    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("error:"):
                # llama.cpp's other in-stream error shape: an `error:` field
                # rather than `data: {"error": ...}`.
                raise UpstreamError("upstream reported an error mid-stream: "
                                    + line[6:].strip()[:500])
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                continue
            _raise_if_error(d)
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            finish = ch.get("finish_reason") or finish

            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
                # An empty delta is a heartbeat: it keeps the connection
                # demonstrably alive without leaking the model's reasoning
                # into the reply.
                yield "event", chunk(cid, model, {})

            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                slot = calls.setdefault(
                    i, {"id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
                # Announce as soon as the NAME is known. The arguments are
                # still streaming in and the hop has not returned, but the
                # reader learns what is happening now rather than afterwards.
                if i not in announced and slot["function"]["name"]:
                    announced.add(i)
                    yield "event", text_chunk(
                        cid, model, f"`{slot['function']['name']}` ")

    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    yield "done", {"message": msg, "finish_reason": finish}


def stream_upstream(upstream: str, payload: dict, cid: str, model: str,
                    timeout: int = 3600):
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
            if line.startswith("error:"):
                # llama.cpp's other in-stream error shape: an `error:` field
                # rather than `data: {"error": ...}`.
                raise UpstreamError("upstream reported an error mid-stream: "
                                    + line[6:].strip()[:500])
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                continue
            _raise_if_error(d)
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}

            # TOOL CALLS MUST BE FORWARDED. The model can call a tool the
            # CLIENT declared, and those are the client's to execute. An
            # earlier version forwarded only content, so a client tool call
            # vanished and the client saw a stream that ended with nothing.
            # Hermes reads that as "cut off by a network error mid-stream" and
            # retries -- four identical calls in the first real run, which read
            # as the model looping when it was the proxy dropping its replies.
            if delta.get("tool_calls"):
                yield chunk(cid, model, {"tool_calls": delta["tool_calls"]},
                            finish=ch.get("finish_reason"))
                continue
            if delta.get("role"):
                yield chunk(cid, model, {"role": delta["role"]})
            # A thinking model streams reasoning_content before any answer.
            # Passing it through as content would splice its private reasoning
            # into the reply; dropping it silently would make the wait look
            # like a hang again. It is surfaced as a heartbeat instead.
            if delta.get("content"):
                yield text_chunk(cid, model, delta["content"])
            elif delta.get("reasoning_content"):
                yield chunk(cid, model, {})
            if ch.get("finish_reason") and ch.get("finish_reason") != "stop":
                yield chunk(cid, model, {}, finish=ch["finish_reason"])
