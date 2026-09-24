#!/usr/bin/env python
"""An OpenAI endpoint that is one standard model, with a second one beside it.

THE POINT

A client adds one OpenAI-compatible model and gets the whole stack. It declares
no tools, configures no MCP server, and never learns any of this exists.

    client --/v1/chat/completions--> [proxy] --> llama-swap --> model (main)
                                        |
                                        +-- the second brain (shomen.run):
                                            our tools, deep thinking,
                                            fan-out, repair -- on the
                                            helper lane

ONE MODEL, ONE CACHE (operator, 2026-09-24; docs/SELF-IMPROVEMENT-PLAN.md
Phase 0.5, AGENTS.md "One model, one cache"). Main -- the model the client
talks to -- gets the client's tools untouched, plus only the image tools. Our
code tools are the second brain's; what it does folds back into main's own
turn in fixed phrases (shomen.PHRASES): prefilled into a main generation, or
written as a one-line note and then warmed into the slot (_warm).

WHAT IT ADDS, AND WHY THE CACHE NEVER NOTICES

  addendum        a short fixed table at the end of the client's system text
                  (ADDENDUM), where the fixup runs
  per-turn        skills, library definitions, the work log after a
                  compaction -- on the user turn they served, decided once
  reasoning       the model's own, restored on every assistant turn
  fold-backs      "After thinking deeply," (prefilled), "Verified",
                  "Repaired", "Compared two approaches", the seed line

KV CACHE. llama-server reuses a slot's prompt only when the next request
EXTENDS it (STEP 0: an edit anywhere earlier rolls the slot back to a
checkpoint). So everything above is recorded in the LEDGER and re-added byte
for byte on every request, whatever the client stripped; `_run_turn` is the
one turn both paths run.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import code_search as cs  # noqa: E402
import repos  # noqa: E402
import corpus  # noqa: E402
import accounts  # noqa: E402
import admission  # noqa: E402
import streaming  # noqa: E402
import packages  # noqa: E402
import repeats  # noqa: E402
import discover  # noqa: E402
import nebari  # noqa: E402
import fanout  # noqa: E402
import shomen  # noqa: E402
import tiers  # noqa: E402
import dashboard  # noqa: E402
import catalog  # noqa: E402
import selection  # noqa: E402
import images  # noqa: E402
import vision  # noqa: E402
import gpu_room  # noqa: E402
import code_check  # noqa: E402
import route as router  # noqa: E402
import tool_code  # noqa: E402
import slots  # noqa: E402
import recent_turns  # noqa: E402
import token_ledger  # noqa: E402
import compaction  # noqa: E402
import power  # noqa: E402
import deep  # noqa: E402
import research_tools  # noqa: E402

# The defaults ARE the configuration. llama-swap listens on 11434 and the
# proxy takes 1234, because 1234 is the port every OpenAI client is already
# pointed at -- putting the proxy there is what makes it need no setup. These
# two lines were stale for a whole session: the proxy listened on 1233 and
# called an upstream on 1234 that nothing was serving, so it only ever worked
# when launched with environment variables that were not written down.
UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
PORT = int(os.environ.get("YAMADORI_PROXY_PORT", "1234"))
PREAMBLE = os.environ.get("YAMADORI_PREAMBLE", "1") == "1"
# Seconds between empty deltas while a streamed answer's fan-out variants run.
FANOUT_HEARTBEAT = float(os.environ.get("YAMADORI_FANOUT_HEARTBEAT", "5"))

# A thinking model spends tokens reasoning BEFORE it writes anything. If the
# budget runs out first, the reply is empty content with finish_reason
# "length" -- which looks to a caller exactly like the model failing.
#
# Measured: at max_tokens=300 this model returns 0 characters of content after
# ~1000 characters of reasoning, with a short instruction AND a long one. At
# 1200 both return a complete answer. The reasoning grows with instruction
# complexity (994 vs 3572 characters for the same task), so the floor has to
# cover the reasoning, not just the answer.
#
# Clients commonly default to 256 or 512, so without this floor the stack
# looks broken to anyone who does not know to raise it.
#
# SUPERSEDED 2026-09-22 by the one budget rule in tiers.budget(): a client's
# max_tokens is an ANSWER allowance (never below tiers.A_MIN) and a thinking
# breaker is added on top. A 1500-token TOTAL floor was below the model's
# natural thinking length (682-2,826 tokens, docs/CONSTRAINTS.md 1b). Kept as a
# name for anything that still reads it.
MIN_BUDGET = tiers.A_MIN

# THE STATIC ADDENDUM (operator, 2026-09-24). One short, FIXED paragraph at
# the end of the client's system text -- identical on every turn, so it is
# part of the cached prefix and never a cause of a miss. It says what the
# service does beside the model and how that work reaches it: the fold-back
# phrases (shomen.PHRASES; AGENTS.md has the same table). A decision table,
# not prose, and no prohibition (AGENTS.md "Prompting this model": a router
# table measured 10.7 vs 10.0; prohibitions degrade monotonically).
#
# Added only where every row is true: where the tool-call check fixes client
# writes (tiers.repair_on -- `high`, `xhigh`, `max`; operator 2026-09-24).
# Its wording is a CHOICE; nothing has measured it.
#
# The capability block it replaces described our code tools on main. Those
# tools are the second brain's now (see OUR TOOLS below); its first-call
# routing numbers (docs/CONSTRAINTS.md #31) describe a surface that no longer
# exists.
ADDENDUM = """

---
A second model works beside you on this service. What it does, and how its
work reaches you:

| when | what it does | what you see |
|---|---|---|
| you write or patch a file with a tool | checks the code before the call leaves, and repairs it if it does not parse | a line before the call: "Verified <file> ..." or "Repaired <file> ..." |
| the user asks about a library's API | reads that library's source | definitions after the user's message, or your answer opens "After thinking deeply," with the facts, each with file:line, in your thinking |
| your answer contains code | checks it, and may write an independent second answer | a line after your answer: "Verified ...", "Repaired ..." or "Compared two approaches ..." |
| it drew a concept seed for its work | names the word | "Today I was inspired by <word>." |

Text that opens with these phrases was written for you by that model:
continue from it, and cite what it cites."""

# THE think_deeply ROW (Phase 0.6, operator 2026-09-24): one more row, only
# where main has the tool (deep.think_tool_offered: xhigh and max, kept for
# the whole conversation), so every row stays true and the text is still the
# same on every turn of a conversation. Its wording is a CHOICE.
ADDENDUM_THINK_ROW = (
    "| you call think_deeply: stuck, unsure of an API or version, a fix "
    "failed twice, or the user says it is still broken | researches the "
    "question in library source, skills, notes and the web | its hand-off as "
    "the tool result, then your answer opens \"After thinking deeply,\" |\n")


def addendum_text(think: bool = False) -> str:
    """ADDENDUM, with the think_deeply row before the seed row when main
    has the tool."""
    if not think:
        return ADDENDUM
    at = ADDENDUM.index("| it drew a concept seed")
    return ADDENDUM[:at] + ADDENDUM_THINK_ROW + ADDENDUM[at:]


def add_addendum(messages: list[dict], think: bool = False) -> list[dict]:
    """Append the addendum to the client's system message, or add one.

    At the end of the system text, never at the end of the conversation:
    this model's chat template raises "System message must be at the
    beginning" outright, so a trailing system message is a hard 500."""
    text = addendum_text(think)
    out = list(messages)
    for i, m in enumerate(out):
        if isinstance(m, dict) and m.get("role") == "system"                 and isinstance(m.get("content"), str):
            out[i] = dict(m, content=m["content"] + text)
            return out
    # No system message: add one rather than prepending to the user's turn,
    # which would put our text in their words.
    return [{"role": "system", "content": text.strip()}] + out


def _post(path: str, payload: dict, timeout: int = 3600,
          retries: int = 1) -> dict:
    """One upstream call, STREAMED, returned in the non-streaming shape.

    WHY THIS STREAMS EVEN WHEN NOBODY ASKED FOR A STREAM

    Measured, three ways, on the same prompt in the same minute:

        direct to llama-server :10001   OK    447.67s   7,379 tokens
        through llama-swap     :11434   FAIL  215.30s   ConnectionReset
        through llama-swap     :11434   FAIL  103.96s   ConnectionReset  (earlier)

    The model needs 447 seconds for that answer and produces it correctly.
    A short request through the same hop returns in 2.90s, so llama-swap is
    not broken -- the failure is a function of DURATION. A non-streaming
    request moves zero bytes between the request line and the complete JSON,
    so for seven minutes the socket looks idle to anything in the path that
    reaps idle connections. The cut time varies (104s, 215s), which is what
    rules out a configured timeout and points at a reaper.

    Streaming moves bytes every few tokens, so the connection is never idle.

    That cost us roughly 60% of a LiveCodeBench run, recorded as model
    failures. They were not model failures. The model was still working.

    WHAT IS RETURNED WHEN IT DIES ANYWAY

    Whatever arrived, shaped as a normal response with finish_reason
    "incomplete". A partial answer the caller can see beats a 502 that throws
    away seven minutes of correct generation, and the marker means nothing
    downstream mistakes it for a finished one. Only a drop with NOTHING
    accumulated raises, because there is then genuinely nothing to hand back.

    `usage` is requested explicitly via stream_options. Without it a streamed
    reply carries no token counts at all, which is why 43 of 46 benchmark
    rows had no cost figures to report.

    This drains `_post_events`, which is the same reader exposed as a
    generator so the streamed path can forward deltas while they arrive.
    """
    for kind, item in _post_events(path, payload, timeout, retries):
        if kind == "done":
            return item
    raise RuntimeError("upstream stream ended without a response")  # unreachable


def _post_events(path: str, payload: dict, timeout: int = 3600,
                 retries: int = 1):
    """`_post_events_raw` on a chosen llama-server slot, with the cache record.

    SLOT. `payload["_slot"]` ({key, transient}, set by prepare) picks the slot
    (mcp/slots.py): a conversation goes back to the slot holding its prefix, a
    client utility call to one no conversation holds. A second-brain payload
    (`_role` "helper", fan-out's candidates) pins to slots.HELPER. The choice
    rides upstream as `id_slot`; a payload with no `_slot` is left to the
    server, as before.

    CACHE. The response carries `_cache` -- prompt tokens, how many came from
    the slot's cache and how many were processed now (llama-server `timings`)
    -- and the record is appended to `payload["_cache_log"]` when the caller
    keeps one, which x_yamadori.cache and the per-request log line read.
    """
    want = payload.get("_slot")
    if payload.get("_role") == "helper":
        want = {"key": slots.HELPER, "transient": False}
    log = payload.get("_cache_log")
    # What this prompt looks like to the slot that serves it (slots.
    # fingerprint): a compaction is placed by it (`prefix`, COMPACTION
    # AFFINITY), and every answered request records it for the next one.
    fp = (slots.fingerprint(payload)
          if isinstance(want, dict) and slots.ENABLED else None)
    grant = (slots.acquire(want.get("key"), bool(want.get("transient")),
                           prefix=fp if want.get("prefix") else None)
             if isinstance(want, dict) else None)
    send = payload
    if grant and grant.get("slot") is not None:
        send = dict(payload, id_slot=grant["slot"], cache_prompt=True)
    try:
        for kind, item in _post_events_raw(path, send, timeout, retries):
            if kind == "done":
                rec = slots.cache_record(item.pop("_timings", None),
                                         item.get("usage"), grant)
                item["_cache"] = rec
                if isinstance(log, list):
                    log.append(rec)
                # The token ledger (mcp/token_ledger.py): every upstream
                # generation of the proxy's own, once. Never raises.
                token_ledger.record_upstream(payload, item)
                slots.remember(grant, fp)
                if isinstance(want, dict) and want.get("record") \
                        and want.get("key"):
                    # A conversation turn: kept for a later compaction of it
                    # (mcp/compaction.py), exactly as it went upstream.
                    compaction.record(
                        want.get("account") or "", want["key"],
                        payload.get("_client_messages"),
                        {k: v for k, v in send.items()
                         if not k.startswith("_")},
                        ((item.get("choices") or [{}])[0] or {}).get("message"))
            yield kind, item
    finally:
        slots.release(grant)


def _http_error_detail(e) -> str:
    """The status and the first 300 characters of an upstream error body.

    llama-server says exactly what it rejected ("Field 'x': ...") and that
    text was being discarded: minimal-tier requests failed with HTTP 400 in
    under 60 ms on 2026-09-23 and the proxy logged only "dropped with nothing
    in hand", so the trigger could not be found from the log."""
    try:
        body = e.read()[:300].decode("utf-8", "replace")
    except Exception:                                            # noqa: BLE001
        body = ""
    return f"HTTP {getattr(e, 'code', '?')}: {' '.join(body.split()) or '(no body)'}"


def _request_shape(payload: dict) -> str:
    """What a rejected request looked like, without its text: the fields sent,
    the message roles, and the values that change how the server parses it."""
    msgs = payload.get("messages") or []
    roles = ",".join(str(m.get("role")) for m in msgs if isinstance(m, dict))
    parts = sorted({p.get("type") for m in msgs if isinstance(m, dict)
                    and isinstance(m.get("content"), list)
                    for p in m["content"] if isinstance(p, dict)})
    keys = sorted(k for k in payload if k not in ("messages", "tools"))
    pick = {k: payload.get(k) for k in (
        "enable_thinking", "chat_template_kwargs", "reasoning_effort",
        "response_format", "tool_choice", "stop", "n", "logit_bias",
        "max_tokens", "id_slot") if k in payload}
    return (f"keys={keys} roles=[{roles}] content_parts={parts} "
            f"tools={len(payload.get('tools') or [])} "
            f"{json.dumps(pick, default=str)[:400]}")


def _post_events_raw(path: str, payload: dict, timeout: int = 3600,
                     retries: int = 1):
    """`_post`, as a generator: ("delta", delta) live, then ("done", response).

    ONE READER FOR BOTH PATHS. The streamed path needs each upstream delta the
    moment it arrives (reasoning shown live); the blocking path needs the
    assembled response. Two parsers of the same SSE would drift -- the
    streamed path already had a second one (`streaming.stream_upstream`) that
    silently discarded reasoning while this one kept it.

    A delta is yielded only once something has arrived, so the retry-on-empty
    below never repeats anything a consumer has already forwarded.
    """
    # `_tier` and `_client_ip` are this proxy's own bookkeeping. llama-server
    # rejects unknown fields on some builds, and shipping them upstream would
    # also make them part of the cached prompt.
    payload = {k: v for k, v in payload.items() if not k.startswith("_")}
    payload = dict(payload, stream=True,
                   stream_options={"include_usage": True})
    req = urllib.request.Request(f"{UPSTREAM}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})

    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    usage = None
    timings = None
    finish = None
    head = {"id": "", "model": payload.get("model", ""), "created": 0}
    t0 = time.time()
    first = None
    gap = 0.0
    last = t0

    def assemble(reason: str, note: str = "") -> dict:
        msg: dict = {"role": "assistant",
                     "content": (note + "".join(content)) if note else "".join(content)}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if calls:
            msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        out = {"id": head["id"] or f"chatcmpl-{int(t0)}",
               "object": "chat.completion",
               "created": head["created"] or int(t0),
               "model": head["model"],
               "choices": [{"index": 0, "message": msg,
                            "finish_reason": reason}]}
        if usage:
            out["usage"] = usage
        # Telemetry the vitals page can show, and the number that proves the
        # connection was never idle. Underscored so it is stripped before it
        # reaches a client or the next upstream payload.
        out["_transport"] = {"seconds": round(time.time() - t0, 2),
                             "ttfb": round(first - t0, 2) if first else None,
                             "max_chunk_gap": round(gap, 2)}
        # llama-server's prompt accounting (cache_n / prompt_n), on the final
        # chunk; `_post_events` turns it into the cache record.
        if timings:
            out["_timings"] = timings
        return out

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                now = time.time()
                if first is None:
                    first = now
                gap = max(gap, now - last)
                last = now
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("error:"):
                    # llama.cpp's other in-stream error shape. Both shapes are
                    # checked here because this reader now feeds the streamed
                    # path too, which `streaming.stream_upstream` used to
                    # guard; an outage must not arrive as an empty answer.
                    raise streaming.UpstreamError(
                        "upstream reported an error mid-stream: "
                        + line[6:].strip()[:500])
                if not line.startswith("data:"):
                    continue
                blob = line[5:].strip()
                if blob == "[DONE]":
                    break
                try:
                    d = json.loads(blob)
                except ValueError:
                    continue
                if d.get("id"):
                    head["id"] = d["id"]
                if d.get("model"):
                    head["model"] = d["model"]
                if d.get("created"):
                    head["created"] = d["created"]
                if d.get("usage"):
                    usage = d["usage"]
                if isinstance(d.get("timings"), dict):
                    timings = d["timings"]
                streaming._raise_if_error(d)
                for ch in d.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if delta.get("content") or delta.get("reasoning_content"):
                        yield "delta", delta
                    for tc in delta.get("tool_calls") or []:
                        i = tc.get("index", 0)
                        slot = calls.setdefault(
                            i, {"id": "", "type": "function",
                                "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
    except Exception as e:                                       # noqa: BLE001
        if isinstance(e, urllib.error.HTTPError):
            # The server ANSWERED, with a refusal. Its body says what it
            # refused, and it is logged with the request's shape (never its
            # text). A 4xx is the same answer every time, so it is not
            # retried: it is raised with the server's words.
            detail = _http_error_detail(e)
            print(f"  upstream refused after {time.time() - t0:.2f}s: "
                  f"{detail}\n    request shape: {_request_shape(payload)}",
                  flush=True)
            if 400 <= int(getattr(e, "code", 0) or 0) < 500:
                raise streaming.UpstreamError(
                    f"the model server rejected the request ({detail})") from e
        if not (content or reasoning or calls):
            # Nothing arrived, so nothing was generated and nothing is lost by
            # asking again -- and the prompt is still in the prefix cache, so
            # the retry skips prefill. Exactly one: a second failure with an
            # empty stream is a real fault, and retrying a fault in a loop
            # turns one bad request into sustained load on a single-GPU box.
            if retries > 0 and not isinstance(e, streaming.UpstreamError):
                print(f"  upstream dropped with nothing in hand after "
                      f"{time.time() - t0:.1f}s ({type(e).__name__}: "
                      f"{str(e)[:160]}); retrying once", flush=True)
                yield from _post_events_raw(path, payload, timeout,
                                            retries - 1)
                return
            raise
        took = time.time() - t0
        print(f"  upstream dropped after {took:.1f}s with "
              f"{len(''.join(content))} chars in hand: "
              f"{type(e).__name__}: {e}", flush=True)
        out = assemble("incomplete",
                       f"[the connection to the model dropped after "
                       f"{took:.0f}s; what follows is the part that "
                       f"arrived]\n\n")
        out["_transport"]["dropped_after"] = round(took)
        yield "done", out
        return
    yield "done", assemble(finish or "stop")


# OUR TOOLS LIVE ON THE SECOND BRAIN, NOT ON MAIN (operator, 2026-09-24).
#
# Main -- the model the client talks to -- gets the client's tools untouched,
# plus only generate_image and describe_image where they are offered. The
# code-intelligence tools (find_*, read_file_range, describe_index, ...) are
# the second brain's (mcp/shomen.py): it searches library source in its own
# context and folds a short result back (the hand-off, prefilled as main's
# reasoning). It never gets the client's tools: what the harness read is in
# the conversation already, and a file it would need becomes the hand-off's
# NEXT STEP, for main to read with the harness's own tool.
#
# What went with them: `bind_project_context` (deleted -- it pinned versions
# for tools main no longer has), the `check_code` tool (client writes are
# checked by the proxy itself, mcp/tool_code.py), `record_step` /
# `read_rings` (the proxy writes the work log from the turns it sees,
# `_log_turn`) and the capability block that described them all.
#
# DEEP THINKING HAS ONE TOOL ON MAIN SINCE PHASE 0.6 (operator, 2026-09-24):
# `think_deeply` (deep.THINK_TOOL) at xhigh and max, the model-chosen trigger
# -- the one non-image tool of ours on main -- run by _think_deeply as a
# hidden hop. The other three triggers run before main (mcp/deep.py).
# `delegate_investigation` is KEPT only as a header-forced benchmark arm
# (docs/SELECTION-BUILD.md step 8; PROTOCOL rule 9):
# `X-Yamadori-Features: {"delegate": true}` or YAMADORI_DELEGATE_TOOL=1 puts
# it on main, and the main loop runs it through the one second-brain runner
# (shomen.run).
DELEGATE_TOOL = os.environ.get("YAMADORI_DELEGATE_TOOL", "0") == "1"


def image_tools() -> list[dict]:
    """`generate_image` and `describe_image`, only where an image server is
    configured.

    Gated on YAMADORI_IMAGEGEN_URL so no request pays prompt tokens for a tool
    that cannot run. Read per request, so the gate follows the environment.
    `describe_image` (mcp/vision.py) goes wherever `generate_image` goes, so
    the model can look at what it drew; YAMADORI_VISION=0 withholds it. A
    request carrying an attached image gets it even without an image server
    (prepare, `vision_tools`).
    """
    if not images.configured():
        return []
    return [images.TOOL] + ([vision.TOOL] if vision.enabled() else [])


def vision_tools(att: dict | None) -> list[dict]:
    """`describe_image` for a request that carries something to look at: a
    readable attached image, or a signed link to one this server made."""
    if not vision.enabled() or not att:
        return []
    readable = any(not e.get("error") for e in (att.get("images") or {}).values())
    return [vision.TOOL] if readable or att.get("media") else []


def main_tools(client_tools: list | None, att: dict | None = None,
               delegate: bool = False, images_on: bool = True,
               think: bool = False) -> tuple[list, set]:
    """(the tool list main is sent, the names of ours in it).

    The client's own list first and untouched, so its rendering -- the
    template prints tools first in the system block -- never depends on what
    we add. Ours after it, dropped on a name the client already uses: the
    client's version is the one with side effects the client can handle.
    `think`: think_deeply (Phase 0.6, deep.THINK_TOOL), the one non-image
    tool of ours on main, where deep.think_tool_offered says so."""
    client_tools = list(client_tools or [])
    taken = {t.get("function", {}).get("name") for t in client_tools
             if isinstance(t, dict)}
    mine: list[dict] = []
    for t in ((image_tools() if images_on else []) + vision_tools(att)
              + ([deep.THINK_TOOL] if think else [])
              + ([shomen.TOOL] if delegate else [])):
        name = t["function"]["name"]
        if name not in taken:
            taken.add(name)
            mine.append(t)
    return client_tools + mine, {t["function"]["name"] for t in mine}


def deep_thinking_tools(att: dict | None = None) -> list[dict]:
    """The second brain's tools: OURS, read-only, always.

    The MCP surface and the work-log tools (code_search.ALL_TOOLS), never the
    request's `tools` -- those are the CLIENT's, which only the client can
    execute. Not `delegate_investigation`: it would recurse, unbounded.

    `generate_image` IS included (operator, 2026-09-23): deep thinking makes
    mockups, designs and sketches while it works, and the image's markdown
    crosses back in the hand-off. `describe_image` comes with it: Bonsai is
    text-only, and the vision copy (mcp/vision.py) is how it looks at what it
    drew and refines it. `att`, the request's attachment register, adds
    `describe_image` when the user attached an image and no image server is
    configured.
    """
    tools = [{"type": "function",
              "function": {"name": t["name"], "description": t["description"],
                           "parameters": t["inputSchema"]}}
             for t in cs.ALL_TOOLS] + image_tools()
    # Phase 0.6: skills, the knowledge base, the web (mcp/research_tools.py).
    # Never think_deeply: it would recurse.
    tools += list(research_tools.TOOLS)
    names = {t["function"]["name"] for t in tools}
    return tools + [t for t in vision_tools(att)
                    if t["function"]["name"] not in names]


# Every name this proxy executes itself: the second brain's tools, the image
# tools and the delegate arm.
OUR_NAMES = ({t["name"] for t in cs.ALL_TOOLS}
             | {"delegate_investigation", images.TOOL_NAME, vision.TOOL_NAME,
                deep.TOOL_NAME} | set(research_tools.NAMES))

# Tools that read the code index, and therefore cannot work without one.
# Everything else -- the work log, summarisation -- is independent of it and
# must keep working when no repository is bound, which is the normal
# condition for a remote caller.
INDEX_TOOLS = {"find_by_meaning", "find_by_pattern", "find_definition_opt",
               "find_references", "read_file_range", "describe_index"}

# Tools that need a REPOSITORY ON DISK, which is a different requirement from
# an index. run_check shells out to lint/test/build in the project root; with
# no repository it was reading `SELECT path FROM roots` off a database that
# has no such table, throwing OperationalError, and handing the agent
# "search failed: OperationalError: no such table: roots" as though that were
# an answer to "run the linter".
ROOT_TOOLS = {"run_check"}


def tool_gate(messages: list[dict], root: str | None,
              state: dict | None = None) -> dict:
    """Are the code tools admissible for this request at all?

    NOT a prediction about whether they would help -- a FACT about whether any
    index this caller can reach could answer. The rule, the measurement and
    the real-input replay live in `domains.tool_admission`; this wrapper only
    supplies what the proxy already knows about the session.

    REVERSED, AND WHY. This gate used to be `root or code_search.has_index()`,
    with domain rejected outright. `has_index()` reads the server's own index
    (index/code.sqlite3, this repository's source), which exists on every
    deployment and which no caller without a repository is ever searched
    against -- `run_our_tool` redirects them to `_no_repository_bound.sqlite3`.
    So it returned True for every request, and a LiveCodeBench puzzle paid
    2,729 prompt tokens (87%) for tools guaranteed to return NO_INDEX.

    The objection to domain was that a misread domain could withhold
    `find_references` from someone asking about their own code. It cannot now:
    a bound repository is offered unconditionally, and without one the
    caller's own code is not indexed and the tool would return NO_INDEX
    anyway. Domain only withholds when the task carries positive domain
    evidence that no held package serves and nothing names one.

    Returns the structured decision: `offer`, `situation`, `because`,
    `evidence`, and for a withheld request `retryable` and `remedies`, each
    remedy with the party that can apply it.
    """
    import domains

    st = state or {}
    return domains.tool_admission(
        messages, root,
        discovered=list(st.get("packages") or []),
        offered_before=bool(st.get("tools_offered")))


def should_offer_tools(messages: list[dict], root: str | None,
                       state: dict | None = None) -> tuple[bool, str]:
    """`tool_gate` reduced to (offer, "SITUATION: because"), for callers that log."""
    g = tool_gate(messages, root, state)
    return g["offer"], f"{g['situation']}: {g['because']}"


def _remember_offered(state: dict | None) -> None:
    """Record that this conversation has had the tools, so it keeps them.

    See the "offered earlier" rule in `domains.tool_admission`. Stored in the
    session's nebari row by load-modify-save rather than by saving `state`
    whole, so a field another writer added since `session_context` ran is not
    overwritten with a stale copy. A dropped write costs one re-decision.

    KNOWN GAP: `nebari.key_of` hashes the first two messages. With a system
    prompt that is (system, first user) for the whole session. Without one it
    is (user) on turn 1 and (user, assistant) from turn 2, so a conversation
    with no system message can lose the flag once, between its first and
    second turns. Found by mcp/test_domains.py; nebari owns the key.
    """
    if not state or state.get("tools_offered") or not state.get("_key"):
        return
    cur = nebari.load(state["_key"])
    cur["tools_offered"] = True
    nebari.save(state["_key"], cur)
    state["tools_offered"] = True


def is_first_turn(messages: list[dict]) -> bool:
    return not any(m.get("role") == "assistant" for m in messages)


def preamble_for(info: dict | None, checks: list[str], how: str = "") -> str:
    """A line prepended to the first answer, or nothing at all.

    THE BAR: it must tell the reader something they can act on.

    This used to open every first answer with

        `yamadori` · no repository detected in this conversation ·
        code tools available, retrieval limited

    on every request that did not carry a repo. For a remote service that is
    not a warning, it is the NORMAL state -- most callers never bind a repo --
    and there is nothing in it for the reader to do: binding project context
    is the MODEL's job, through bind_project_context, not theirs. So it spent
    a line at the top of every answer to report that nothing was wrong.

    The other branches stay, because each one names something that changes
    what you would do next:

        indexing now              results are thin this turn; ask again later
        predates current commit   the index is behind; reindex
        not indexed               search will find nothing; index it
        checks: lint, test        these commands exist in this repo

    A healthy, current, bound index with no checks says nothing either. If
    the answer to "what should I do differently" is "nothing", the correct
    length for this line is zero.
    """
    if not info:
        return ""
    name = os.path.basename(info["root"])
    state = ""
    if info["building"]:
        state = "indexing now, search will be thin this turn"
    elif not info["chunks"]:
        state = "not indexed"
    elif info.get("stale"):
        state = f"{info['chunks']:,} chunks indexed, predates current commit"

    head = f"`yamadori` · **{name}**"
    bits = []
    if state:
        bits.append(f"{head} · {state}")
    if checks:
        # Worth a line on its own when it is the only actionable fact: these
        # commands exist in this repo and can be run.
        bits.append("checks: " + ", ".join(checks) if bits
                    else head + " · checks: " + ", ".join(checks))
    if not bits:
        return ""
    return " · ".join(bits) + "\n\n"

def available_checks(root: str) -> list[str]:
    pkg = os.path.join(root, "package.json")
    found = []
    try:
        with open(pkg, encoding="utf-8") as f:
            scripts = (json.load(f).get("scripts") or {})
        for want in ("lint", "test", "typecheck", "build"):
            if want in scripts:
                found.append(want)
    except (OSError, json.JSONDecodeError):
        pass
    if os.path.exists(os.path.join(root, "Cargo.toml")):
        found.append("cargo test")
    return found


# Tools whose result is not a pure function of their arguments. These always
# execute, however many times they are called.
_STATEFUL = {"bind_project_context", "record_step", "run_check",
             "delegate_investigation", images.TOOL_NAME, vision.TOOL_NAME,
             code_check.TOOL_NAME}

# What a repeated empty search returns instead of running again. It says the
# same thing the search said, so the model sees no inconsistency, and it says
# plainly that nothing was re-run so the wording is not mistaken for a fresh
# negative result.
_EMPTY_AGAIN = ("No results -- this exact search was already run this turn and "
                "returned nothing, so it was not repeated. The index has not "
                "changed since.")


def _route_package_glob(glob: str | None, held_names: list[str]):
    """(package, version-or-None, remaining glob) when `glob` names a held
    package, else None.

    A model searching a library passes the library as the glob -- measured on
    a typegpu task: glob="typegpu@0.12.5/data", "typegpu/data". Inside a
    package index the paths are "src/data/...", so that glob matched NOTHING
    in any package, every search returned a wall of misses from all 14
    indexes, and the model looped to the 12-hop breaker over 33 minutes before
    inventing an API (`d.typeOf`) the package does not have. So a glob that
    names a package routes the search to that package, and whatever follows
    the name filters paths inside it.
    """
    if not glob:
        return None
    g = glob.replace("\\", "/").strip().lstrip("./")
    if g.startswith("node_modules/"):
        g = g[len("node_modules/"):]
    for pkg in sorted(held_names, key=len, reverse=True):   # longest first
        if g == pkg or g.startswith(pkg + "/") or g.startswith(pkg + "@"):
            rest = g[len(pkg):]
            version = None
            if rest.startswith("@"):
                version, _, rest = rest[1:].partition("/")
                version = version or None
            else:
                rest = rest[1:] if rest.startswith("/") else rest
            return pkg, version, rest.strip("/")
    return None


def _run_on_package(db: str, name: str, args: dict) -> str:
    # The database is PASSED, bound for this thread only (code_search.
    # bound_index): the process-wide index is never swapped (pre-deploy
    # review, 2026-09-24 -- concurrent requests swapped each other's index).
    resp = cs.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": name, "arguments": args}}, db=db)
    return resp["result"]["content"][0]["text"]


def _held_labels() -> list[str]:
    """Every library index held, as "name@version", for a failure return to
    name what IS available. Empty when nothing is held or the store is
    unreadable."""
    import domains
    try:
        held = domains.held_sources()
    except Exception:                                            # noqa: BLE001
        return []
    return [f"{p}@{v}" for p in sorted(held) for v, _db in held[p]]


def _describe_held() -> str | None:
    """describe_index with no repository bound: the libraries held."""
    labels = _held_labels()
    if not labels:
        return None
    return ("Library source held by the remote code-intelligence service "
            f"({len(labels)} indexes, read-only):\n"
            + "\n".join(f"  {x}" for x in labels)
            + "\n\nTo search one library, pass its name as `glob` "
              "(glob=\"typegpu\"); to read a file, start the path with it "
              "(read_file_range path=\"typegpu/src/index.ts\"). The user's "
              "own project is not indexed here; use your client's own file "
              "tools for it.")


def _search_packages_without_repo(state: dict | None, probe: str, name: str,
                                  args: dict) -> str | None:
    """An index tool's answer from the package indexes, or None.

    1. A glob that NAMES a package searches that package only.
    2. Otherwise the libraries this conversation's imports named, then every
       package index this server holds.
    3. A miss says what was searched -- compactly, with near-miss names from
       the most relevant package only -- never NO_INDEX (which claims nothing
       ran), and never a 4 KB wall of every package's misses.
    """
    import domains
    # A READ names a file, not a query, so it has no probe. A path that starts
    # with a held package ("typegpu@0.12.5/data/struct.d.ts") is read from
    # that package's index. Measured: the model asked for exactly the right
    # file on a typegpu task and got an error envelope, twice, while the
    # package was indexed (bench/domain smoke tg01, 2026-09-22).
    if name == "read_file_range" and args.get("path"):
        try:
            held = domains.held_sources()
            routed = _route_package_glob(args["path"], sorted(held))
            if routed:
                pkg, version, rest = routed
                versions = held.get(pkg) or []
                pick = next((v for v in versions
                             if version and v[0] == version),
                            versions[0] if versions else None)
                if pick and rest:
                    ver, db = pick
                    text = _run_on_package(db, name, dict(args, path=rest))
                    note = (f" (asked for {version}; {ver} is what is "
                            f"indexed)" if version and version != ver else "")
                    return f"== {pkg}@{ver}{note} ==\n{text}"
        except Exception as e:                                   # noqa: BLE001
            print(f"  package read failed: {type(e).__name__}: {e}",
                  flush=True)
        return None
    if not probe:
        return None
    try:
        held = domains.held_sources()
        routed = _route_package_glob(args.get("glob"), sorted(held))
        if routed:
            pkg, version, rest = routed
            inner = dict(args)
            if rest:
                inner["glob"] = rest
            else:
                inner.pop("glob", None)
            versions = held.get(pkg) or []
            pick = next((v for v in versions if version and v[0] == version),
                        versions[0] if versions else None)
            if pick is None:
                return None
            ver, db = pick
            alt = packages.search_discovered(
                {"packages": [pkg], "versions": {pkg: ver}}, probe, name, inner)
            if alt:
                return alt
            text = _run_on_package(db, name, inner)
            note = (f" (asked for {version}; {ver} is what is indexed)"
                    if version and version != ver else "")
            return (f"== searched {pkg}@{ver}{note}"
                    + (f", paths matching {rest!r}" if rest else "")
                    + f": no match ==\n{text[:1500]}")

        alt = packages.search_discovered(state or {}, probe, name, args)
        if alt:
            return alt
        if not held:
            return None
        alt = packages.search_discovered({"packages": sorted(held)}, probe,
                                         name, args)
        if alt:
            return alt
        # Searched everything, nothing matched. One package's near-misses --
        # the one the conversation names, else the first -- and the rest by
        # name, so the model sees what exists without a wall of text.
        import re as _re
        said = (probe or "") + " " + str(args.get("glob") or "")
        named = [p for p in sorted(held)
                 if domains.HELD_ALIASES.get(p)
                 and _re.search(domains.HELD_ALIASES[p], said, _re.I)]
        first = (named or sorted(held))[0]
        ver, db = held[first][0]
        text = _run_on_package(db, name, args)
        if not packages._is_empty(text):
            # The direct run found it after all (search_discovered resolves
            # packages by name and can miss a held index). This used to be
            # headed "None matched." above the match itself, and with
            # repeats._empty now reading that phrase, a hit would have been
            # cached as a miss.
            return f"== {first}@{ver} ==\n{text[:3000]}"
        others =", ".join(f"{p}@{held[p][0][0]}" for p in sorted(held)
                           if p != first)
        # "None matched" stays first: repeats._empty reads the head of a
        # result, and this is the miss its breaker exists for.
        return ("None matched in any library the remote code-intelligence "
                "service holds. It searches library source only; the user's "
                "own project is searched with your client's own file tools. "
                "The same arguments return the same miss."
                f"\n\n== {first}@{ver} ==\n{text[:1500]}"
                f"\n\nAlso searched, no match: {others}.\nTo search one "
                "library, pass its name as `glob` (e.g. glob=\"typegpu\" or "
                "\"typegpu/src/data\").")
    except Exception as e:                                       # noqa: BLE001
        print(f"  package fallback failed: {type(e).__name__}: {e}",
              flush=True)
    return None


def run_our_tool(name: str, args: dict, db: str | None,
                 root: str | None = None,
                 turn: "repeats.Turn | None" = None,
                 state: dict | None = None) -> str:
    # Every A4000 decision a tool causes (a search loading embeddings, a draw,
    # a look: mcp/gpu_room.py) lands in this request's x_yamadori.gpu_room.
    # Set here, in the thread the tool runs in.
    with gpu_room.recording((state or {}).get("_gpu_room")):
        return _run_our_tool(name, args, db, root, turn, state)


def _run_our_tool(name: str, args: dict, db: str | None,
                  root: str | None = None,
                  turn: "repeats.Turn | None" = None,
                  state: dict | None = None) -> str:
    # An identical search that already returned nothing is not run again. The
    # index does not change within a turn, so the second answer IS the first
    # answer -- and re-deriving it cost about 47 seconds each of the eleven
    # times one request did exactly this. The model still receives a result and
    # still decides what to do; nothing is refused, only not recomputed.
    #
    # Tools with effects are excluded: for record_step and
    # bind_project_context, "same arguments" does not mean "same outcome".
    if (turn is not None and name not in _STATEFUL
            and turn.cached_empty(name, args)):
        turn.record(name, args, _EMPTY_AGAIN)
        return _EMPTY_AGAIN + turn.guidance(name, args, OUR_NAMES)

    if name == images.TOOL_NAME:
        # The proxy runs it, on CUDA1, in its own lane (admission.image_lane).
        # The result carries a signed /media URL built on the address the
        # client reaches us on, and each call is recorded for x_yamadori.
        st = state if state is not None else {}
        out = images.run_tool(args, st.get("_public_base") or images.public_base(),
                              st.setdefault("_images", []),
                              account=st.get("_account") or None)
        # What it drew may now be looked at (describe_image), by url or sha.
        vision.note_generated(out, st.setdefault("_attached", vision.empty()))
        return out

    if name == vision.TOOL_NAME:
        # The proxy runs it on the A4000, in the image lane. It reads ONLY this
        # request's attached images and our own media store (mcp/vision.py,
        # "WHERE AN IMAGE MAY COME FROM"); each call is recorded for
        # x_yamadori.vision.
        st = state if state is not None else {}
        return vision.run_tool(args, st.setdefault("_attached", vision.empty()),
                               st.setdefault("_vision", []))

    if name == deep.TOOL_NAME:
        # think_deeply runs inside a chat turn (_run_turn -> _think_deeply),
        # where its hand-off becomes the next hop's context; anywhere else
        # there is no turn to hand it to.
        return cs.error_result(
            name, "NOT_IN_A_TURN",
            "think_deeply runs only inside a chat turn, where its hand-off "
            "is folded into the answer. Nothing was run.", retryable=False,
            remedies=[{"fixable_by": "agent",
                       "action": "ask the question in a chat turn at "
                                 "reasoning_effort xhigh or max",
                       "effect": "the model can call think_deeply there"}])

    if name in research_tools.NAMES:
        # The second brain's other sources (Phase 0.6): skills, the knowledge
        # base (docs + this conversation's work log), the web. Never on main.
        # One deep-thinking run's search budget (research_tools.
        # SEARCHES_PER_RUN), reset by each run (_deep_thinking,
        # _think_deeply).
        return research_tools.run(
            name, args, session_lineage(state) if state else "",
            budget=(state.setdefault("_research_budget", {})
                    if state is not None else None))

    if name == "delegate_investigation":
        # THE ARGUMENT GATE COMES FIRST, BEFORE ANY GPU IS COMMITTED.
        #
        # This is the most expensive call in the stack: a whole second context,
        # minutes of generation, and the one helper lane held for all of it.
        # It took `args.get("question", "")` and started regardless, so a
        # malformed call launched a real investigation into an empty string.
        # Found when a test suite triggered one by accident.
        #
        # Validating is free and refusing is instant. An empty question cannot
        # produce a finding however long it runs.
        q = args.get("question")
        if not isinstance(q, str) or len(q.strip()) < 8:
            return cs.error_result(
                name, "BAD_ARGUMENTS",
                ("`question` must be a non-empty string of at least 8 "
                 "characters saying what to investigate. Nothing was run."),
                retryable=True,
                remedies=[{"fixable_by": "agent",
                           "action": ("call again with a specific question, "
                                      "e.g. 'how is the KV pool sized and "
                                      "where is that set'"),
                           "effect": "deep thinking runs when the lane frees"}])

        # A second context with the SAME index tools, whose searching never
        # enters this conversation. Only the conclusion crosses back, so a
        # six-search investigation costs the caller a few hundred tokens
        # instead of several thousand.
        #
        # The investigator gets the read-only search tools and NOT this one:
        # letting it delegate again would recurse, and nothing bounds the
        # depth of that.
        sub_tools = deep_thinking_tools((state or {}).get("_attached"))

        def sub_run(fn: str, a: dict) -> str:
            return run_our_tool(fn, a, db, root, None, state)

        # At most HELPER_LANES second brains (one, with 3/8 of the pool --
        # mcp/budget.py says why it is not two). The
        # lane is the mechanism; the count was once only a statement in a
        # comment, because this call runs in a worker thread and went
        # straight to the model -- two requests in flight produced two second
        # brains when one was the limit, on a card with 3.9 GB free.
        #
        # A refusal is REPORTED, not swallowed. An investigation that quietly
        # did not happen looks to the model exactly like one that found
        # nothing, and it will reason from an absence we manufactured.
        # The one second-brain runner (shomen.run) holds the helper lane.
        res = shomen.run("investigate", question=args.get("question", ""),
                         tools=sub_tools, run_tool=sub_run,
                         context=args.get("context", ""))
        if res.get("skipped"):
            return cs.error_result(
                "delegate_investigation", "HELPER_BUSY",
                ("Deep thinking is already in progress for another request. "
                 "Only one runs at a time. Nothing was thought about here."),
                retryable=True,
                remedies=[{"fixable_by": "agent",
                           "action": ("answer from what is already in "
                                      "context, or ask again shortly"),
                           "effect": ("the lane frees when the other "
                                      "investigation finishes")}])
        # The cost is reported to the caller because context economy is the
        # entire justification for this tool, and an unmeasured saving is a
        # claim rather than a result.
        return (res["finding"] + "\n\n"
                + f"[thought about deeply: {res['hops']} tool "
                  f"calls, {res['helper_tokens']} tokens spent there, "
                  f"{res['seconds']}s. None of that entered this "
                  f"conversation. Trace handle {res['handle']}.]")

    # WHICH TOOLS ACTUALLY NEED AN INDEX.
    #
    # This used to be `if db:` around the whole dispatch, so with no repository
    # bound -- the normal case for a remote service -- NOTHING ran. Not the
    # searches, and not record_step, read_rings or summarize_text, which never
    # needed an index in the first place. Every one of them returned an empty
    # string, and the agent had no way to tell that from a tool that simply
    # found nothing.
    #
    # Two separate decisions were tangled in that one `if`: whether a corpus
    # exists, and whether this particular tool reads one.
    needs_index = name in INDEX_TOOLS
    if db is None and needs_index:
        # No repository bound -- the normal remote case. The package indexes
        # are the only corpus this caller can reach, so try them BEFORE
        # declaring NO_INDEX. This used to return NO_INDEX right here, which
        # made the fallback below unreachable: measured live, a three.js
        # question offered the tools looped 12 times over 331 s and 53,785
        # prompt tokens against NO_INDEX while three@0.185.1 sat indexed.
        if name == "describe_index":
            # Its description says it lists the libraries held, and with no
            # repository that is the only corpus there is. It used to return
            # NO_INDEX here -- "nothing is indexed" from a service holding
            # nineteen library indexes.
            listing = _describe_held()
            if listing:
                return listing
        probe = (args.get("query") or args.get("symbol")
                 or args.get("pattern") or "")
        alt = _search_packages_without_repo(state, probe, name, args)
        if alt:
            if turn is not None:
                turn.record(name, args, alt)
                alt += turn.guidance(name, args, OUR_NAMES)
            return alt
        return cs.no_index_error(name, _held_labels())
    if root is None and name in ROOT_TOOLS:
        return cs.error_result(
            name, "NO_REPOSITORY",
            ("This tool runs a check inside a project set up on the remote "
             "code-intelligence service, and none is set up for this "
             "conversation. Nothing was run. The service holds library "
             "source only and has no copy of the user's project."),
            retryable=False,
            remedies=[{"fixable_by": "agent",
                       "applies_when": "the check is for the user's own project",
                       "action": ("run its linter, tests or build with your "
                                  "client's own terminal tool"),
                       "effect": "the output is then in context"}])

    # With no repository, point the index somewhere that cannot exist. The
    # server's own index holds 7,742 chunks of THIS codebase; letting a
    # caller's search fall through to it would answer their question with our
    # source, which is both wrong and a disclosure.
    target = db or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "index", "_no_repository_bound.sqlite3")
    # PASSED to the call, bound for this thread only -- never swapped into
    # the process (pre-deploy review, 2026-09-24).
    # The work-log tools are per-CONVERSATION. Passed in the arguments rather
    # than an environment variable: several requests are served at once and a
    # process-global would hand one caller another's session.
    call_args = args
    if name in {"record_step", "read_rings"}:
        # The conversation's LINEAGE, so the log written before a compaction
        # is the one read after it (session_context).
        call_args = dict(args, _session=session_lineage(state))
    try:
        resp = cs.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": call_args}},
                         db=target)
        text = resp["result"]["content"][0]["text"]
    except Exception as e:                                       # noqa: BLE001
        return cs.error_result(
            name, "TOOL_RAISED", f"{type(e).__name__}: {e}",
            retryable=False,
            remedies=[{"fixable_by": "operator",
                       "action": "check the server log for the traceback",
                       "why_not_the_agent": ("the tool threw; no arguments "
                                             "change that")}])

    probe = (args.get("query") or args.get("symbol")
             or args.get("pattern") or "")

    # The repository had no answer. Its dependencies might -- and until now
    # they were indexed and unreachable. Tried only on a miss, because
    # `three` alone is 15,021 chunks against koota's 1,351 and merging them
    # would bury the user's own code under library internals.
    if root and probe and packages._is_empty(text):
        alt = packages.search(root, probe, name, args)
        if alt:
            text = text.rstrip() + "\n\n" + alt

    # No repository at all -- the normal case for a remote caller. Serve from
    # the libraries their own imports named. This is the path that makes the
    # service useful to someone whose disk we will never see, and it is why
    # failing to detect a directory is no longer fatal.
    if not root and probe and packages._is_empty(text) and state:
        alt = packages.search_discovered(state, probe, name, args)
        if alt:
            text = (text.rstrip() + "\n\n" + alt) if text.strip() else alt

    # A repeat still runs. Refusing it would be a negation, and this model
    # family reads negation as topic rather than constraint -- measured,
    # describing what failed made the decision model pick retry at margin
    # 0.288, while asking which action progresses answered correctly at 0.493.
    # So the result carries what has NOT been tried instead.
    if turn is not None:
        turn.record(name, args, text)
        text += turn.guidance(name, args, OUR_NAMES)

    # A TOOL NEVER RETURNS NOTHING.
    #
    # This function opened with `text = ""` and only filled it inside
    # `if db:`. With no repository bound -- which is the NORMAL case for a
    # remote service -- the tool was never executed and the agent received an
    # empty string. Not an error. Not a message. Nothing at all.
    #
    # MEASURED: the corpus recorded `chars: 0` on the first tool results of
    # the run that then issued the same search twelve times over 842 seconds.
    # The agent was not ignoring our guidance; there was no guidance, because
    # nothing ran. Every careful word written into these results was
    # unreachable code.
    #
    # An empty result is indistinguishable from a broken transport, a silent
    # exception, or a tool that does not exist. It is the least informative
    # thing a tool can say, so it is now never said.
    if not (text or "").strip():
        # Only an index tool with no index is a NO_INDEX. summarize_text has
        # nothing to do with the index, and labelling its empty output that way
        # sends the agent to look for a repository it never needed.
        return cs.no_index_error(name, _held_labels()) if (db is None and needs_index)             else cs.error_result(
            name, "EMPTY_RESULT",
            ("The tool ran and produced no output. This is a fault in the "
             "tool, not a statement about the query."),
            retryable=False,
            remedies=[{"fixable_by": "operator",
                       "action": "check the server log for this tool",
                       "why_not_the_agent": ("no arguments change this; the "
                                             "tool returned nothing at all")}])
    return text



def resolve_repo(messages, client_ip: str = ""):
    """Locate the caller's repository, if this deployment can see one at all.

    A repository is usable only when the harness states where it is AND that
    path exists here. That is the local case, and it is the less common one:
    Yamadori is a remote service, so usually there is no filesystem to look at
    and no path that would mean anything if there were.

    Two failures got us here. Detection scored 8/8 against prompts written by
    the same person who wrote the detector, then returned None on the first
    real Hermes run, because Hermes prints its working directory only from a
    container-backend probe and a local run states no location at all. The
    repair was worse: a loopback fallback that guessed the most recently
    indexed repository. It answered `glyph` while Hermes ran in `koota`, and
    more fundamentally it assumed the client shares a machine with the server,
    which for a remote service is never true.

    So there is no guess. With no repository, the caller is served from what
    they actually sent -- see `session_context`.

    Returns (root, trusted, how). `root` is always None now.

    CLOSED 2026-09-22 -- A SECURITY HOLE, NOT A FEATURE. This used to take a
    path from the system or user message, and if that path existed on THIS
    machine, treat it as the caller's repository: approve it on first sight,
    index it, and serve find_* / read_file_range / run_check from it. Both of
    those messages are written by whoever holds a key, so any caller could
    name any directory on the server and read it back through the tools. The
    contract is the opposite: the caller's source reaches the model only
    through the caller's own harness tools; the server serves only what it
    holds (package indexes, hints, deep thinking, fan-out). Nothing in a
    request may select a directory on this disk.
    """
    return None, False, "none"


def session_context(messages: list[dict], account: str = "",
                    session: str = "",
                    utility: dict | None = None) -> tuple[str, dict]:
    """What this caller is working with, learned from the code they sent.

    Code always arrives, and code names its own libraries. Imports are parsed
    with tree-sitter (`discover`), accumulated across the session (`nebari`)
    and matched against the dependency indexes this server already holds. No
    path, no configuration and no question needed -- and it works for a caller
    the server will never share a disk with, which a repository path never
    could.

    A CLIENT UTILITY CALL HAS NO SESSION (`utility`, selection.utility_call).
    The key hashes the first two messages, so a side call is keyed by its OWN
    system prompt and instruction -- and an identical one repeats that key:
    Hermes re-asked its approval classifier about the same command (corpus
    854938/38fb6f, 649358/c7690e/9bea36, same text, same key), the first call
    was offered the tools by the domain gate and stamped `tools_offered`, and
    every repeat was logged OFFERED_EARLIER_THIS_SESSION. So a utility call
    gets an empty state that is neither read from nor written to nebari: it
    cannot inherit or change an offered-tools flag, a work log or a pin.

    A CONVERSATION CONTINUES ACROSS A COMPACTION (`_continue_after_compaction`).
    Hermes' compaction rewrites the head of the conversation, so the key
    changes: nebari shows the live session's main key renewed at each
    compaction (8122f27e at 21:26, e64fb66a at 21:56, 6cc061d0 at 22:14), and
    everything recorded under the old key -- the work log read_rings exists to
    bring back, the pinned versions, the offered tools -- was unreachable from
    the new one.
    """
    # The ACCOUNT is part of the key. It was not: two callers whose first
    # two messages matched -- a shared harness system prompt and "hi" -- got
    # one session, and so one work log (read_rings), one package list and one
    # tool-offer history. Found 2026-09-22 when a fresh conversation was
    # logged OFFERED_EARLIER_THIS_SESSION.
    # `session`: the X-Yamadori-Session token, when a caller sends one.
    if utility and utility.get("utility"):
        if (utility.get("signals") or {}).get("form") == "summarise_conversation":
            _note_compaction(account)
        return "", {"_key": "", "_utility": True}
    key = nebari.key_of(messages, account, session)
    fresh = not nebari.load(key)
    state = nebari.observe(key, discover.scan(messages))
    if fresh and not is_first_turn(messages):
        state = _continue_after_compaction(key, state, account)
    state["_key"] = key
    with _SESSIONS_LOCK:
        _LAST_SESSION[account] = (key, time.time())
    return key, state


# COMPACTION CONTINUITY. A conversation whose key is NEW but which already has
# answers in it cannot be a new conversation -- a new one starts with no
# assistant turn. Arriving right after the same account's harness asked for a
# summary of a conversation (a summarise_conversation utility call), it is
# that conversation, compacted. Both halves are required: a fresh key with
# history alone is also a proxy restart or an expired session, and a summary
# alone says nothing about which request comes next. In-process only; a
# restart between the compaction and the next turn loses the link (the old
# behaviour, not a wrong one).
COMPACTION_LINK_SECONDS = int(os.environ.get("YAMADORI_COMPACTION_LINK_S", "1800"))
_SESSIONS_LOCK = threading.Lock()
_LAST_SESSION: dict[str, tuple[str, float]] = {}      # account -> (key, when)
_PENDING_COMPACTION: dict[str, tuple[str, float]] = {}  # account -> (key, when)


def _note_compaction(account: str) -> None:
    """A summarise-the-conversation call: remember whose conversation it was."""
    with _SESSIONS_LOCK:
        last = _LAST_SESSION.get(account)
        if last and time.time() - last[1] <= COMPACTION_LINK_SECONDS:
            _PENDING_COMPACTION[account] = (last[0], time.time())


def _continue_after_compaction(key: str, state: dict, account: str) -> dict:
    """Carry a compacted conversation's session over to its new key: the work
    log (`lineage`, which record_step / read_rings and the slot pin use), the
    offered-tools flag, versions and packages. One link per compaction."""
    with _SESSIONS_LOCK:
        pend = _PENDING_COMPACTION.pop(account, None)
    if not pend or pend[0] == key or time.time() - pend[1] > COMPACTION_LINK_SECONDS:
        return state
    prev = nebari.load(pend[0])
    if not prev:
        return state
    cur = nebari.load(key)
    cur["lineage"] = prev.get("lineage") or pend[0]
    cur["continues"] = pend[0]
    if prev.get("tools_offered"):
        cur["tools_offered"] = True
    for field in ("versions", "asked", "counts"):
        merged = dict(prev.get(field) or {})
        merged.update(cur.get(field) or {})
        cur[field] = merged
    counts = cur.get("counts") or {}
    cur["packages"] = sorted(counts, key=lambda p: (-counts[p], p))
    nebari.save(key, cur)
    print(f"  session {key[:8]} continues {pend[0][:8]} after a compaction "
          f"(work log {cur['lineage'][:8]})", flush=True)
    return cur


def session_lineage(state: dict | None) -> str:
    """The key a conversation's work log and slot pin live under: its first
    session's, across compactions."""
    st = state or {}
    return st.get("lineage") or st.get("_key") or ""


def strip_thinking(messages: list[dict]) -> list[dict]:
    """Remove reasoning that a client echoed back, before it can reach the KV.

    The second hemisphere's work is shown to the PERSON as a thinking block and
    must never be paid for by the MODEL. Those are different channels: a slot's
    KV is built from the `messages` of each incoming request, so streamed
    output costs nothing -- unless the client sends it back next turn, at which
    point the saving the server was careful to make is undone by the client.

    Conventionally `reasoning_content` is display-only and is not echoed, but
    that is a convention, not a guarantee, and it varies by client. Stripping
    on the way IN makes it a property of this server instead of a hope about
    someone else's.

    Cheap when there is nothing to strip: the list is returned unchanged, so
    the KV prefix stays byte-identical and no prefill is triggered.
    """
    if not any(isinstance(m, dict) and m.get("reasoning_content")
               for m in messages):
        return messages
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("reasoning_content"):
            m = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(m)
    return out


# THE LEDGER (docs/SELF-IMPROVEMENT-PLAN.md Phase 0.5, operator 2026-09-24).
#
# One model, one cache: everything the proxy adds to a conversation is
# recorded per message and re-added, byte for byte, on every later request,
# so the rendering of turn N+1 EXTENDS the rendering of turn N and the pinned
# slot reuses all of it. STEP 0 measured why it matters on this build: a
# request that diverges anywhere before the end of the slot's sequence rolls
# back to a context checkpoint (448 tokens in, wherever the edit was), so one
# missing hint on an early user turn re-prefills everything after it. The
# compaction agent measured exactly that on Hermes: 87% of the prompt shared
# on an ordinary turn, because the hint attached to the previous user turn was
# not in the client's resent copy.
#
# What is recorded, and under which key (storage: nebari's `additions`
# table, memory in front of it -- see nebari.py LEDGER):
#
#   user turn        "inject"     skills / hints, library definitions, the
#                                 work log after a compaction: decided ONCE,
#                                 on the request whose last message is that
#                                 user turn, and replayed after its content
#   tool result      "inject"     LIBRARY USE (#19): the same, on the tool
#                                 result a request ended on
#   assistant turn   (reasoning)  NOT RECORDED, NOT RESTORED (design change,
#                                 coordinator/operator 2026-09-24). What a
#                                 client sends is what the model sees: a
#                                 client that drops past reasoning (Hermes
#                                 does, for any provider that does not
#                                 require the echo: agent/message_sanitization
#                                 .py apply_reasoning_content_policy) gets no
#                                 past reasoning -- the reference run
#                                 (sudoingX: plain llama-server + Hermes +
#                                 Bonsai 2, 5 h, 125k context) ran exactly so
#                                 -- and one that echoes it gets its echo,
#                                 unchanged. Restoring it cost 6-10k tokens of
#                                 context per step (V0 pilot, rows of 25-41k
#                                 chars). The price: the next request diverges
#                                 at the previous turn's think block, which the
#                                 slot generated in full; the reuse then rests
#                                 on the checkpoint at the previous prompt's
#                                 end -- to be measured live.
#                    "content"    a turn with tool calls whose delivered
#                                 content (the check note) a client may drop
#                    "hops"       the image-tool hops the proxy ran inside the
#                                 turn, which the client never sees -- with
#                                 their reasoning EMPTIED: reasoning lives
#                                 within the one request whose hops these are
#                                 (a prefilled hand-off included), never
#                                 replayed
#   the request      "seed:<job>" the concept seed a second-brain job drew,
#                                 so a retry or replay uses the same word
#
# Keys are hashes, never text. A user turn's key hashes the conversation up
# to and including it (roles, text, tool-call ids and arguments; reasoning
# excluded), so the same words at two points of a conversation -- "continue"
# -- are two keys, and two conversations cannot share one. An assistant
# turn's key is its tool-call ids, else its content: what the client echoes.
# The key does not use the session key, which changes between turn 1 and 2
# for a conversation with no system message (nebari.key_of, KNOWN GAP); the
# `session` column is for pruning only.
import hashlib as _hashlib  # noqa: E402


def _args_norm(raw) -> str:
    """Tool-call arguments in one canonical form: clients re-serialise them."""
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


def _msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = "\n".join(p.get("text") or "" for p in c
                      if isinstance(p, dict) and p.get("type") == "text")
    return c if isinstance(c, str) else ""


def _canon(m: dict) -> str:
    """A message as the chain hashes it: what the CLIENT controls."""
    return json.dumps({
        "role": m.get("role"), "text": _msg_text(m),
        "calls": [[c.get("id"), (c.get("function") or {}).get("name"),
                   _args_norm((c.get("function") or {}).get("arguments"))]
                  for c in (m.get("tool_calls") or []) if isinstance(c, dict)],
        "tool_call_id": m.get("tool_call_id")}, sort_keys=True,
        ensure_ascii=False)


def chain_keys(messages: list[dict]) -> list[str]:
    """One key per message: the hash of the conversation up to and including
    it, as the client sent it."""
    out, h = [], ""
    for m in messages:
        m = m if isinstance(m, dict) else {}
        h = _hashlib.sha256((h + _canon(m)).encode("utf-8", "replace")
                            ).hexdigest()[:32]
        out.append("u:" + h)
    return out


def _memo_keys(msg: dict, prev: str = "") -> list[str]:
    """An assistant turn's ledger keys: its tool-call ids, else its text.

    A TEXT key also hashes `prev`, the chain key of the message before the
    turn (pre-deploy review, 2026-09-24): keyed by the text alone, two of an
    account's conversations whose assistant said the same words ("Done.")
    shared one key, and one conversation's recorded content and hidden hops
    were restored into the other. The chain key names one conversation up
    to that point and, unlike the session key, does not change between turn
    1 and 2 (nebari.key_of's KNOWN GAP)."""
    keys = [f"call:{c.get('id')}" for c in (msg.get("tool_calls") or [])
            if isinstance(c, dict) and c.get("id")]
    if not keys:
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            keys = ["text:" + _hashlib.sha256(
                (prev + "\x00" + content.strip()).encode()).hexdigest()]
    return keys


def ledger_scope(payload: dict | None) -> tuple[str, str]:
    """(account, session) a request's additions are recorded under."""
    sl = (payload or {}).get("_slot") or {}
    return sl.get("account") or "", sl.get("key") or ""


def ledger_restore(messages: list[dict], account: str) -> tuple[list, dict]:
    """The client's messages with every recorded addition put back.

    Returns (messages, counts). Messages the ledger has nothing for are the
    client's own objects, unchanged."""
    keys = chain_keys(messages)
    out: list = []
    n = {"inject": 0, "content": 0, "hops": 0, "echoed_reasoning": 0,
         "markers_in_echo": 0}
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        # A user turn's injection, or a tool result's (LIBRARY USE, #19).
        if role in ("user", "tool") and isinstance(m.get("content"), str):
            add = nebari.ledger_get(account, keys[i], "inject")
            if add:
                m = dict(m, content=m["content"] + add)
                n["inject"] += 1
        elif role == "assistant":
            mk = _memo_keys(m, keys[i - 1] if i else "")
            # REASONING PASSES THROUGH (see the block above): the client's
            # echo, or nothing. Counted, and a template marker inside an echo
            # is counted too (#12) -- not scrubbed: it is what the client
            # sent.
            r = m.get("reasoning_content")
            if isinstance(r, str) and r.strip():
                n["echoed_reasoning"] += 1
                n["markers_in_echo"] += sum(template_markers(r).values())
            # A call turn's delivered content (the check note a client may
            # drop), or a text turn whose client copy carries more than the
            # slot rendered (an earlier hop's streamed content, #10).
            for k in mk:
                c = nebari.ledger_get(account, k, "content")
                if c is not None:
                    if (m.get("content") or "") != c:
                        m = dict(m, content=c)
                    n["content"] += 1
                    break
            for k in mk:
                hops = nebari.ledger_get(account, k, "hops")
                if hops:
                    try:
                        out.extend(json.loads(hops))
                        n["hops"] += 1
                    except ValueError:
                        pass
                    break
        out.append(m)
    return out, n


def ledger_record_turn(account: str, session: str, msg: dict,
                       hops: list[dict] | None = None,
                       stored: dict | None = None, prev: str = "") -> int:
    """Record what the proxy delivered for one assistant turn: its content
    when it carries tool calls (or differs from what the client stores), and
    the image-tool hops before it, their reasoning emptied. NOT its
    reasoning: past reasoning is the client's to keep or drop (the block
    above). Returns the bytes written, for the size record
    (mcp/test_ledger.py).

    `stored`: the turn as the CLIENT will store it, when that differs from
    `msg` -- a stream that carried an earlier hop's content (#10). The keys
    are then that message's, and the content to render (`msg`'s) is recorded
    for a text turn too, so ledger_restore puts back what the slot holds."""
    if not isinstance(msg, dict):
        return 0
    written = 0
    keyed = stored if isinstance(stored, dict) else msg
    keys = _memo_keys(keyed, prev)
    restore_content = bool(msg.get("tool_calls")) or (
        (keyed.get("content") or "").strip()
        != (msg.get("content") or "").strip())
    if hops:
        hops = [dict(h, reasoning_content="") if isinstance(h, dict)
                and h.get("role") == "assistant" else h for h in hops]
    for k in keys:
        if restore_content:
            nebari.ledger_put(account, session, k, "content",
                              msg.get("content") or "")
            written += len(msg.get("content") or "")
        if hops:
            blob = json.dumps(hops, ensure_ascii=False)
            nebari.ledger_put(account, session, k, "hops", blob)
            written += len(blob)
    return written


def scope_reasoning(messages: list[dict], account: str = "") -> list[dict]:
    """The client's messages with the ledger's additions put back
    (ledger_restore, messages only). The name is historical: since
    2026-09-24 it restores no reasoning -- past reasoning passes through as
    the client sent it."""
    return ledger_restore(messages, account)[0]


def ledger_seed(payload: dict | None, job: str,
                prompt: str | None = None) -> dict | None:
    """The concept seed for one second-brain job of this request (operator,
    2026-09-24): drawn once (concept_seed.seed_for, away from `prompt`),
    recorded under the request's own key and the job, and returned again
    for any replay of the same request -- a retry, a re-render, a warm, a
    compaction's splice -- so the job's prompt is byte-identical. None when
    the embedding matrix is not extracted: a missing seed costs nothing."""
    lg = (payload or {}).get("_ledger") or {}
    key = lg.get("turn_key")
    account = lg.get("account") or ""
    if key:
        got = nebari.ledger_get(account, key, "seed:" + job)
        if got:
            try:
                return json.loads(got)
            except ValueError:
                pass
    seed = _draw_seed(prompt)
    if seed and key:
        nebari.ledger_put(account, lg.get("session") or "", key, "seed:" + job,
                          json.dumps(seed))
    return seed


def _draw_seed(prompt: str | None) -> dict | None:
    """One fresh seed. Split out so a test can pin it."""
    import concept_seed
    return (concept_seed.seed_for(prompt, 1) or [None])[0]


# Names this proxy injected on main before 2026-09-24. The corpus recorded
# the UPSTREAM tool list, ours included, so a replay of it (mcp/test_utility.py)
# must not count them as the client's.
_LEGACY_NAMES = {"bind_project_context", code_check.TOOL_NAME}


def client_tool_names(body: dict) -> list[str]:
    """The CLIENT's own tools, never ours -- even when a client re-sends ours
    by name."""
    return [n for n in (t.get("function", {}).get("name")
                        for t in (body.get("tools") or [])
                        if isinstance(t, dict))
            if n and n not in OUR_NAMES and n not in _LEGACY_NAMES]


# X-Yamadori-Features flags that, forced ON, mean a benchmark asked for an
# augmentation -- and then a utility-shaped request still gets it.
_AUGMENTATIONS = ("retrieval", "hints", "investigate", "check_code", "repair",
                  "delegate")


def utility_of(body: dict, messages: list[dict] | None = None) -> dict:
    """Is this request a client's own side call (selection.utility_call)?

    {utility, because, signals}. The header decides when it says so:
    X-Yamadori-Features {"utility": true|false} forces it, and a header that
    forces an augmentation ON (or fan-out above one) overrides the rule,
    because a benchmark that forced it meant it."""
    msgs = messages if messages is not None else (body.get("messages") or [])
    d = selection.utility_call(msgs, client_tool_names(body),
                               body.get("response_format"))
    if (d.get("signals") or {}).get("image"):
        # Never a utility call, whatever a header says: the bare text model
        # cannot see (selection.utility_call).
        return d
    feats = tiers.from_header(body.get("_features")) or {}
    if "utility" in feats:
        on = bool(feats["utility"])
        return dict(d, utility=on, because=(
            f"forced {'on' if on else 'off'} by X-Yamadori-Features"
            + (f"; the rule said: {d['because']}" if on != d["utility"] else "")))
    forced = [k for k in _AUGMENTATIONS if feats.get(k) is True]
    if int(feats.get("fanout") or 1) > 1:
        forced.append("fanout")
    if d["utility"] and forced:
        return dict(d, utility=False, because=(
            f"{d['because']} -- but X-Yamadori-Features forces "
            f"{', '.join(forced)} on, which wins"))
    return d


def _utility_selection(util: dict, requested_tier: str) -> dict:
    """The selection record for a utility call: every system off, and why."""
    why = ("a client utility call gets the bare model (tier minimal, from "
           f"{requested_tier}): {util['because']}")
    print(f"  utility call: tier {requested_tier} -> minimal, no block, no "
          f"tools, no hints -- {util['because']}", flush=True)
    return {"hints": False, "investigate": False, "fanout_n": 1,
            "utility": True,
            "because": {"utility": util["because"], "hints": why,
                        "investigate": why, "fanout": why},
            "signals": {"utility": util.get("signals"),
                        "tier_requested": requested_tier}}


def prepare(body: dict) -> dict:
    """Resolve the ledger, tools, addendum and per-turn injection without
    calling the model.

    Split out so an experiment can fan the SAME resolved request out several
    ways; otherwise a comparison measures prompt differences rather than the
    thing being tested.
    """
    raw = body.get("messages") or []
    account = body.get("_account") or ""
    root, trusted, how = resolve_repo(raw, body.get("_client_ip", ""))
    # A client's own side call (an approval check, a title, a compaction):
    # the bare model, no session. selection.utility_call has the rule.
    util = body.get("_utility") or utility_of(body, strip_thinking(raw))
    utility = bool(util.get("utility"))
    _key, state = session_context(raw, account,
                                   body.get("_session_token") or "",
                                   utility=util)
    lineage = "" if utility else session_lineage(state)
    # THE LEDGER (see above): everything this proxy added to the
    # conversation on earlier requests -- reasoning, injections, the
    # delivered content of checked calls, image-tool hops -- put back, so this
    # request's rendering EXTENDS what the pinned slot holds. A utility call
    # is not a conversation and gets the client's messages as sent.
    keys = chain_keys(raw)
    if utility:
        messages, restored = list(raw), {}
    else:
        messages, restored = ledger_restore(raw, account)
        nebari.ledger_touch(account, lineage)
    # ATTACHED IMAGES. The chat model is text-only, and llama-server refuses an
    # image part for a model without a projector, so the turn used to fail.
    # Each image part becomes a text placeholder naming an id; the image stays
    # in `att` for describe_image (mcp/vision.py), which sends it to the
    # vision copy. After session_context on purpose: the session key hashes
    # the messages as the client sent them.
    # An image part carrying OUR signed /media link (a client looking at an
    # image we drew, e.g. Hermes' vision_analyze) is read from the media
    # store as an attachment when its host is ours: YAMADORI_PUBLIC_BASE, the
    # address this request reached us on, or loopback. Nothing is fetched.
    messages, att = vision.extract(messages, vision.our_hosts(
        body.get("_public_base") or ""))
    # `reasoning_effort` doubles as the product dial: it decides the thinking
    # budget AND which augmentations run. See mcp/tiers.py for why an existing
    # field is overloaded rather than a new one invented -- a new parameter is
    # configuration, and the premise here is that a client configures nothing.
    # A model name may carry a tier, for the many clients that expose a model
    # picker and nothing else. An explicit reasoning_effort still wins: a
    # caller who set it meant it.
    requested = body.get("model")
    internal, tier_hint, _known = catalog.resolve(requested)
    if tier_hint and not body.get("reasoning_effort") and not (
            isinstance(body.get("reasoning"), dict)
            and body["reasoning"].get("effort")):
        body = dict(body, reasoning_effort=tier_hint)

    tier = tiers.resolve(body, tiers.from_header(body.get("_features")))
    requested_tier = tier["name"]
    if utility:
        # The CLIENT cannot choose per situation: Hermes sends its approval
        # checks, titles and compactions to the same model name as its main
        # turns. So the proxy decides (operator, 2026-09-23): a utility call
        # runs at `minimal` -- thinking off, the vendor's instruct sampling
        # (presence 1.5 counters the repetition seen in a 35,425-character
        # compaction), no augmentation. A one-word probe at minimal answered
        # in 1.97 s against 409 s for the classifier at medium.
        tier = dict(tiers.TIERS["minimal"], name="minimal")

    # THE GATE: is anything the caller can reach indexed? A fact, read from
    # the package store (domains.tool_admission). It no longer puts tools on
    # main; it decides whether library help (the definitions injection, deep
    # thinking) can have anything to read. Logged one line per request.
    gate = tool_gate(messages, root, state) if tier["retrieval"] else None
    if gate and gate["offer"]:
        _remember_offered(state)
        print(f"  library source reachable: {gate['situation']} -- "
              f"{gate['because']}", flush=True)
    elif gate:
        print(f"  library source NOT reachable: {gate['situation']} -- "
              f"{gate['because']}", flush=True)
    # MAIN'S TOOLS: the client's, untouched, plus generate_image /
    # describe_image where offered (a capability on every tier, operator
    # 2026-09-23), plus the delegate benchmark arm when a header forces it.
    # A utility call gets the client's list and nothing of ours.
    delegate = (bool(tier.get("delegate")) or DELEGATE_TOOL) and not utility
    # think_deeply (Phase 0.6, trigger 1): where deep thinking is allowed and
    # not forced off, decided on the conversation's first request and kept
    # (deep.think_tool_offered: a tool list that changes between turns would
    # change the system block the slot caches).
    think_on = deep.think_tool_offered(
        tier, account, lineage, utility,
        continuing=any(isinstance(m, dict) and m.get("role") == "assistant"
                       for m in raw))
    tools, ours = main_tools(body.get("tools"), None if utility else att,
                             delegate=delegate,
                             images_on=tiers.images_offered(tier)
                             and not utility, think=think_on)

    # SELECTION: which of the ALLOWED systems fire for this request.
    #
    # The tier says what the caller allows; `selection.select` decides what
    # runs -- skills, deep thinking (the regex + symbol lookup, with Laya's
    # trained route_in head as a second signal), and how wide fan-out goes.
    # It never exceeds the tier, and a flag set in X-Yamadori-Features is
    # forced on or off. One log line per request, and the whole decision
    # rides along as `_selection` (and on the response, in `x_yamadori`).
    # The CLIENT's own tools (never ours, even when a client re-sends ours by
    # name): a harness with its own file and terminal tools that is asked to
    # act on the user's machine gets neither deep thinking nor fan-out
    # (selection.acts_locally).
    client_tools = client_tool_names(body)
    # THE ROUTE (mcp/route.py): one class per request, decided here, once.
    # Fan-out and repair read it (code_generation / code_edit only), deep
    # thinking and the definitions injection read it (library_question
    # only). A tier without retrieval computed no gate; the router still gets
    # one, so a request's class does not depend on its tier.
    route_gate = gate
    if route_gate is None and not utility:
        route_gate = tool_gate(messages, root, state)
    try:
        route_dbs = selection.symbol_dbs(repos.db_path(root) if root else None)
    except Exception:                                            # noqa: BLE001
        route_dbs = {}
    route = router.classify(messages, client_tools=client_tools, util=util,
                            gate=route_gate, dbs=route_dbs)
    print("  " + router.log_line(route), flush=True)
    # A compaction that resends the conversation (compaction.in_place): part
    # of that conversation -- its session, tools and slot -- but nothing is
    # added to it and nothing escalates. _serve_compaction shapes the rest.
    inplace = not utility and compaction.in_place(messages)
    # DEEP THINKING'S TRIGGERS (Phase 0.6, mcp/deep.py): struggle, a task
    # kickoff, a known-hard area -- decided here from what the client sent,
    # on any route class; recorded whether or not one fires. Laya is not
    # consulted. The model's own think_deeply call is the fourth, at
    # generation time (_think_deeply).
    trig = _deep_trigger(raw, tier, route, util, account, lineage,
                         keys[-1] if keys else None, inplace, state)
    if utility:
        sel = _utility_selection(util, requested_tier)
    elif inplace:
        why = ("an in-place compaction: served on the conversation's own "
               "prompt and slot, nothing added (mcp/compaction.py)")
        sel = {"hints": False, "investigate": False, "fanout_n": 1,
               "utility": False, "compaction": True,
               "because": {"utility": util.get("because"), "hints": why,
                           "investigate": why, "fanout": why},
               "signals": {"tier_requested": requested_tier}}
    else:
        sel = selection.select(messages, tier, gate, state,
                               root_db=repos.db_path(root) if root else None,
                               client_tools=client_tools, route=route,
                               trigger=trig)
        sel["utility"] = False
        sel.setdefault("because", {})["utility"] = util.get("because")

    # THE STATIC ADDENDUM, where every row of it is true (tiers.repair_on:
    # the tool-call check fixes client writes). Fixed text, so part of the
    # cached prefix; an in-place compaction gets it too, or its system block
    # would not match the conversation's.
    fix_on = tiers.repair_on(tier) and not utility
    augmented = add_addendum(messages, think=think_on) if fix_on else messages

    # THE PER-TURN INJECTION, decided ONCE per user turn and replayed from
    # the ledger ever after (ledger_restore put it back above). Decided on
    # the request whose last message IS that user turn -- a request that
    # ends on a tool result is a step of a task whose user turn the slot
    # already holds, and adding to it now would change a prefix it has
    # cached. One exception: the first request after a compaction, whose
    # whole prefix is new, may carry the work log on its last user turn.
    used_hints: list[dict] = []
    skills_rec: dict | None = None
    inject_rec = {"decided": False, "chars": 0, "parts": []}
    use_rec: dict | None = None         # LIBRARY USE (#19), None when off
    # Template markers scrubbed from OUR injected text (library source,
    # skills, the work log) when it is decided -- a replay sends what was
    # recorded, byte for byte (pre-deploy review, 2026-09-24).
    scrub_note: dict = {}
    li = next((i for i in range(len(raw) - 1, -1, -1)
               if isinstance(raw[i], dict) and raw[i].get("role") == "user"),
              None)
    speaking = li is not None and li == len(raw) - 1
    continued = bool(state.get("continues")) and not state.get(
        "rings_reinjected")
    recorded = (None if li is None or utility else
                nebari.ledger_get(account, keys[li], "inject"))
    if recorded is not None:
        inject_rec.update(replayed=True, chars=len(recorded))
        skills_rec = {"path": None, "on": False, "ids": [], "versions": [],
                      "why": "decided on the request this user turn arrived "
                             "with; replayed from the ledger"}
    elif (li is not None and not utility and not inplace
          and isinstance(raw[li].get("content"), str)
          and (speaking or continued)):
        parts: list[str] = []
        if speaking:
            tail, used_hints, skills_rec = _skills_tail(messages, sel, route)
            if tail:
                parts.append(tail)
                inject_rec["parts"].append("skills")
            # A skill injected here that declares `escalate` (known-hard
            # area, Phase 0.6): deep thinking runs for it, once per skill per
            # conversation, when nothing else fired.
            if trig is not None and not trig.get("fire"):
                trig = deep.escalate_skills(trig, skills_rec, account,
                                            lineage, raw)
                forced = "investigate" in (tier.get("overridden") or [])
                if trig.get("fire") and not sel.get("investigate") \
                        and not forced:
                    sel["investigate"] = True
                    sel.setdefault("because", {})["investigate"] = \
                        trig["because"]
                    sel.setdefault("signals", {})["trigger"] = trig["kind"]
            defs = _library_definitions(route, tier, gate, sel, messages,
                                        route_dbs, state)
            if defs:
                parts.append(defs)
                inject_rec["parts"].append("definitions")
            if tier.get("retrieval") and not sel.get("investigate"):
                # LIBRARY USE (#19): whatever the class. Read from the
                # client's own messages -- never from what this service
                # injected, whose definitions carry imports of their own.
                use, use_rec = _library_use(raw, account, lineage, state)
                if use:
                    parts.append(use)
                    inject_rec["parts"].append("library_use")
        if continued:
            log = _work_log_block(state)
            if log:
                parts.append(log)
                inject_rec["parts"].append("work_log")
            _mark_rings_reinjected(state)
        text, n_scrub = scrub_markers("".join(parts))
        _note_scrub(scrub_note, "injection", n_scrub)
        # Never over a non-empty decision a duplicate request recorded
        # meanwhile; what stands is what is sent (nebari.ledger_claim).
        text = nebari.ledger_claim(account, lineage, keys[li], "inject", text)
        inject_rec.update(decided=True, chars=len(text))
        if text:
            ai = _index_of_user(augmented, raw[li])
            if ai is not None:
                augmented = list(augmented)
                augmented[ai] = dict(augmented[ai],
                                     content=augmented[ai]["content"] + text)
    # LIBRARY USE on a request that ENDS ON A TOOL RESULT (#19): the tool
    # result is the one message the slot does not hold yet, so a package the
    # conversation just started using (a file the harness read, a file the
    # model wrote) is covered there, decided once and recorded under that
    # message's key (ledger_restore replays it on every later request).
    last_raw = raw[-1] if raw and isinstance(raw[-1], dict) else {}
    if (not utility and not inplace and tier.get("retrieval")
            and not sel.get("investigate")
            and last_raw.get("role") == "tool"
            and isinstance(last_raw.get("content"), str)
            and augmented and isinstance(augmented[-1], dict)
            and augmented[-1].get("role") == "tool"):
        got = nebari.ledger_get(account, keys[-1], "inject")
        if got is None:
            use, use_rec = _library_use(raw, account, lineage, state)
            use, n_scrub = scrub_markers(use)
            _note_scrub(scrub_note, "injection", n_scrub)
            use = nebari.ledger_claim(account, lineage, keys[-1], "inject",
                                      use)
            use_rec["decided"] = True
            if use:
                augmented = list(augmented)
                augmented[-1] = dict(augmented[-1],
                                     content=augmented[-1]["content"] + use)
        else:
            use_rec = {"replayed": True, "chars": len(got)}

    # The token budget is set inside tiers.apply by the one rule: the
    # client's max_tokens as the answer allowance, plus the thinking breaker.
    # The prompt estimate reads the messages AFTER vision.extract: a 1 MB
    # base64 image counted as text is ~450,000 "tokens", which would floor the
    # thinking room at MIN_THINKING for any turn that attached a picture.
    out = tiers.apply(dict(body, messages=messages), tier)
    out.update(scrub_note)
    out["messages"] = augmented
    out["tools"] = tools
    # Our tools on main (image tools, the delegate arm): the only calls the
    # main loop runs itself.
    out["_ours"] = sorted(ours)
    # The attachment register (JSON-safe; the turn moves it into the session
    # state before anything deep-copies the payload), and the messages as the
    # text model reads them, for deep thinking's question.
    out["_attached"] = att
    out["_seen_messages"] = messages
    # 120 characters of each recipe and no more: this goes back to the client
    # in `x_yamadori`, and a recipe snippet is the only corpus text that may.
    out["_hints"] = [{"score": h.get("_score"),
                      "recipe": (h.get("recipe") or "")[:120],
                      "source": h.get("source_name") or h.get("_file"),
                      "bucket": h.get("_bucket")}
                     for h in used_hints]
    out["_suppressed_hints"] = [
        {"score": x.get("score"), "recipe": (x.get("recipe") or "")[:120],
         "bucket": h.get("_bucket"), "held_by": (h.get("recipe") or "")[:60]}
        for h in used_hints for x in (h.get("_suppressed") or [])]
    # x_yamadori.skills: path, ids, versions, why -- never text or a path.
    out["_skills"] = skills_rec
    out["_selection"] = sel
    # x_yamadori.deep: the trigger decision (or the recorded non-decision),
    # and think_deeply's offer and calls on this request.
    out["_deep"] = trig
    out["_think_tool"] = {"offered": think_on, "calls": []}
    # THE CODE CHECKS (mcp/tool_code.py, code_check.review_answer).
    #   _tool_code  client writes are checked (tiers.check_code_offered:
    #               `medium` and up); never a utility call
    #   _fixup      and repaired by the second brain (tiers.repair_on:
    #               `high` and up; operator 2026-09-24)
    #   _repair     a final answer's code is checked and repaired, where the
    #               route is code work or a header forces it
    forced_repair = "repair" in (tier.get("overridden") or [])
    out["_repair"] = fix_on and (forced_repair or router.is_code(route))
    out["_tool_code"] = tiers.check_code_offered(tier) and not utility
    out["_fixup"] = fix_on
    out["_route"] = route
    out["model"] = internal            # what llama-swap actually routes on
    out["_tools_gate"] = gate
    out["_tier"] = tier
    out["_public_model"] = requested
    out["_utility"] = util
    out["_tier_requested"] = requested_tier
    # The ledger's scope and this request's key (the chain hash of its last
    # message: a concept seed is recorded under it, so a replay of this
    # request draws the same word), and what was restored.
    out["_ledger"] = {"account": account, "session": lineage,
                      "turn_key": keys[-1] if keys else None,
                      "restored": restored, "inject": inject_rec}
    if (restored or {}).get("markers_in_echo"):
        print(f"  template markers: {restored['markers_in_echo']} inside "
              f"reasoning the client echoed; passed through as sent",
              flush=True)
    # x_yamadori.library_use: the packages used and held, what was injected
    # (package:name pairs, overviews, chars) -- never the text (#19).
    out["_library_use"] = use_rec
    # WHICH KIND of side call (selection.utility_kind). A compaction -- a
    # flattened one (utility) or one that resends the conversation (in
    # place) -- is served on the conversation's stored prompt and slot by
    # _serve_compaction, below the slot choice it may override.
    kind = selection.utility_kind(util) or ("compaction" if inplace else None)
    out["_utility_kind"] = kind
    if utility and kind != "compaction" and not body.get("max_tokens"):
        # Thinking is off, so the whole answer is content, and a compaction
        # summary is thousands of tokens: the A_MIN floor alone (2,048) would
        # cut it and append a budget notice into the client's own history. A
        # client that set no limit gets what a thinking request gets -- the
        # main share less the prompt -- minus the room context_full keeps, so
        # a request with no tools is never "landed".
        room = (tiers._shares()["main"]
                - tiers.estimate_prompt_tokens({"messages": messages,
                                                "tools": tools})
                - tiers.MIN_THINKING - 1)
        out["max_tokens"] = max(room, tiers.A_MIN)
    # WHICH SLOT (mcp/slots.py): a conversation's turns go back to the slot
    # holding its prefix -- keyed by its lineage, so a compacted conversation
    # keeps its slot -- and a utility call to a slot no conversation holds.
    # `record`: a conversation turn's generations are kept for a later
    # compaction of it (compaction.record, from _post_events), with the
    # messages as the client sent them; a side call's and a compaction's are
    # not.
    out["_slot"] = {"key": None if utility else (lineage or None),
                    "transient": utility, "prefix": kind == "compaction",
                    "account": account,
                    "record": not utility and kind != "compaction"}
    if out["_slot"]["record"]:
        out["_client_messages"] = list(raw)
    if kind == "compaction":
        _serve_compaction(out, body, messages, inplace)
    # One record per upstream generation (prompt, reused, processed, slot),
    # appended by _post_events; x_yamadori.cache and the log line read it.
    out["_cache_log"] = []
    out.pop("stream", None)
    return out


def _deep_trigger(raw: list[dict], tier: dict, route: dict, util: dict,
                  account: str, lineage: str, turn_key: str | None,
                  inplace: bool, state: dict | None) -> dict | None:
    """deep.decide for one request, with the packages the conversation USES
    (library_uses, the same reading library help makes) and the held
    version each maps to. The package reading is done only where deep
    thinking is allowed; below that the decision is recorded without it.
    Never raises: a fault is a recorded non-decision."""
    uses: dict = {}
    heldv: dict = {}
    if tier.get("investigate") and not inplace \
            and not (util or {}).get("utility"):
        try:
            import domains
            held_all = domains.held_sources()
            uses = library_uses(raw)
            stated = conversation_versions(raw, state)
            for pkg in uses:
                if pkg in held_all:
                    ver, db, _label = held_version(held_all[pkg],
                                                   stated.get(pkg))
                    heldv[pkg] = (ver, db)
        except Exception as e:                                   # noqa: BLE001
            print(f"  deep: package reading failed ({type(e).__name__}: "
                  f"{e}); the area trigger is off for this request",
                  flush=True)
            uses, heldv = {}, {}
    try:
        trig = deep.decide(raw=raw, tier=tier, route=route, util=util,
                           account=account, lineage=lineage,
                           turn_key=turn_key, uses=uses, held=heldv,
                           inplace=inplace,
                           continues=bool((state or {}).get("continues")))
    except Exception as e:                                       # noqa: BLE001
        print(f"  deep: trigger decision raised ({type(e).__name__}: {e}); "
              f"no trigger", flush=True)
        return {"fire": False, "kind": None, "allowed":
                bool(tier.get("investigate")), "forced": None,
                "because": f"the trigger decision raised "
                           f"{type(e).__name__}: {e}"[:300],
                "signals": {}, "thresholds": {}, "cooldown": None}
    if trig.get("fire") or trig.get("signals", {}).get("struggle", {}).get(
            "count"):
        print(f"  deep: {trig.get('kind') or 'none'} -- "
              f"{trig.get('because', '')[:200]}", flush=True)
    return trig


def _index_of_user(messages: list, original: dict) -> int | None:
    """Where the client's last user turn sits in `messages` (the addendum
    may have added a system message in front, and restored image hops add
    messages before assistant turns)."""
    text = original.get("content")
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user" \
                and isinstance(m.get("content"), str) \
                and m["content"].startswith(text or ""):
            return i
    return None


def _skills_tail(messages: list[dict], sel: dict, route: dict
                 ) -> tuple[str, list[dict], dict | None]:
    """The skills / hints block for the last user turn, as the text to
    append (mcp/skill_select.py decides; YAMADORI_RECALL picks the path).

    Selection is by embedding similarity with a FLOOR, not a rank cutoff, so
    "the best of a bad set" stays silent (19 buckets / 89 probes: embeddings
    71.9% against Laya's 33.7% and a 21.3% floor). A user turn, not the
    system block: the system prefix must stay byte-identical, and
    concept_seed.py records that guidance in the system message "was
    sometimes ignored" while the same text in the user message "couldn't"
    be. A failure degrades to silence and says so."""
    try:
        import skill_select
        base = [dict(m) if isinstance(m, dict) else m for m in messages]
        out, used, rec = skill_select.attach(base, messages, sel,
                                             {"route": route})
    except Exception as e:                                       # noqa: BLE001
        print(f"  skills unavailable, continuing without: "
              f"{type(e).__name__}: {e}", flush=True)
        return "", [], {"path": None, "on": False, "ids": [], "versions": [],
                        "why": f"recall raised {type(e).__name__}: {e}"[:300]}
    li = next((i for i in range(len(base) - 1, -1, -1)
               if isinstance(base[i], dict) and base[i].get("role") == "user"),
              None)
    if li is None:
        return "", used, rec
    before = base[li].get("content") or ""
    after = (out[li].get("content") if li < len(out) else before) or ""
    tail = after[len(before):] if after.startswith(before) else ""
    return tail, used, rec


# LIBRARY DEFINITIONS (operator decision 2026-09-24, an UNMEASURED choice).
# With our tools off main, a library question at a tier where deep thinking
# does not run (`medium`, `high`) would get no library source at all. So a
# user turn classified library_question, where the gate says something held
# can answer, gets the definitions of the names in it that a held source
# DEFINES -- the router's own lookup (selection.defined_symbols), then
# find_definition_opt against the package indexes -- appended as a capped
# tail, recorded in the ledger and replayed identically. Where deep thinking
# runs it does the reading and this stays silent. The caps are choices.
DEFINITIONS_MAX_NAMES = 3
DEFINITIONS_MAX_CHARS_EACH = 1200
DEFINITIONS_MAX_CHARS = 3000
DEFINITIONS_HEAD = ("\n\n---\nLibrary definitions for names in this message, "
                    "read from the library source this service holds (cite "
                    "them as path:line):\n\n")


def _library_definitions(route: dict, tier: dict, gate: dict | None,
                         sel: dict, messages: list[dict], dbs: dict,
                         state: dict | None) -> str:
    if (route or {}).get("class") != "library_question" \
            or not tier.get("retrieval") or not (gate or {}).get("offer") \
            or sel.get("investigate"):
        return ""
    try:
        q, _ctx, _speaking = selection.question_of(messages)
        held = selection.defined_symbols(q, dbs)
    except Exception:                                            # noqa: BLE001
        return ""
    names: list[str] = []
    for ns in held.values():
        for n in ns:
            if n not in names:
                names.append(n)
    parts = []
    for n in names[:DEFINITIONS_MAX_NAMES]:
        try:
            out = run_our_tool("find_definition_opt", {"symbol": n}, None,
                               None, None, state)
        except Exception:                                        # noqa: BLE001
            continue
        if not out or out.lstrip().startswith("{") or packages._is_empty(out):
            continue
        parts.append(f"== {n} ==\n{out[:DEFINITIONS_MAX_CHARS_EACH]}")
    if not parts:
        return ""
    return DEFINITIONS_HEAD + "\n\n".join(parts)[:DEFINITIONS_MAX_CHARS]


# LIBRARY USE (#19 in docs/SELF-IMPROVEMENT-LOG.md; coordinator/operator,
# 2026-09-24; an UNMEASURED choice). Harness traffic never got library help:
# every Hermes request routes agent_step, and the definitions above are
# injected on library_question only, so the Octopus pilot's three-flatland and
# @pmndrs/glyph tasks -- libraries the model has never seen -- got no source
# at all. So, WHATEVER THE CLASS, the packages a conversation USES are read
# from what it carries:
#   imports      in tool results (a file the harness read) and user turns
#   written code in the client's write/edit calls (tool_code.detect)
#   manifests    package.json-style "name": "version" pairs in tool results
# (discover.imported_names / discover.versions: parsed, not matched). For a
# HELD package it has not covered yet, the service appends, to the message
# this request ENDS on (a user turn or a tool result: the only text the slot
# does not already hold), the definitions of the names imported from it
# (find_definition_opt on that package's own index) or, for a package used
# with no names yet, a short list of its exported classes and functions.
# Decided once per message and recorded in the ledger under that message's
# key (ledger_restore replays it, byte for byte); what was covered is recorded
# per conversation, so each package and each name is injected once. Every cap
# is a choice.
USE_MAX_CHARS = 3000                # one injection
USE_MAX_NAMES = 4                   # definitions in one injection
USE_MAX_CHARS_EACH = 1000
USE_OVERVIEW_ITEMS = 20             # exports listed for a package with no names
USE_CONVERSATION_MAX_CHARS = 12000  # everything this injects in a conversation
# Imports sit at the top of a file; a harness can return a megabyte bundle.
# Only the head of each message or written file is parsed.
USE_SCAN_CHARS = 20000
USE_HEAD = ("\n\n---\nLibrary source for packages this conversation uses, "
            "read from the source this service holds (cite as path:line):\n")


def library_uses(messages: list[dict]) -> dict[str, dict]:
    """{package: {"names": [...], "via": [...]}} the conversation USES (see
    LIBRARY USE)."""
    import discover
    uses: dict[str, dict] = {}

    def add(pkg: str, names: list[str], via: str) -> None:
        u = uses.setdefault(pkg, {"names": [], "via": []})
        u["names"].extend(n for n in names if n not in u["names"])
        if via not in u["via"]:
            u["via"].append(via)

    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role in ("tool", "user"):
            text = _msg_text(m)[:USE_SCAN_CHARS]
            if not text:
                continue
            for pkg, names in discover.imported_names(text).items():
                add(pkg, names, "import")
            if role == "tool":
                for pkg in discover.versions(text):
                    if discover.package_of(pkg):
                        add(discover.package_of(pkg), [], "manifest")
        elif role == "assistant":
            for c in m.get("tool_calls") or []:
                try:
                    units = tool_code.detect(c).get("units") or []
                except Exception:                                # noqa: BLE001
                    units = []
                for u in units:
                    code = (u.get("code") or "")[:USE_SCAN_CHARS]
                    if not code:
                        continue
                    lang = u.get("language") or ""
                    for pkg, names in discover.imported_names(
                            f"```{lang}\n{code}\n```").items():
                        add(pkg, names, "written")
    return uses


def _package_overview(db: str) -> str:
    """A package's exported classes and functions, shallowest files first:
    for a package used before any name from it is."""
    import sqlite3 as _sq
    try:
        con = _sq.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT name, kind, path, start, line FROM defs WHERE kind IN "
                "('class', 'function') AND line LIKE 'export %' AND path NOT "
                "LIKE '%test%' AND path NOT LIKE '%spec%' AND path NOT LIKE "
                "'%internal%' ORDER BY (LENGTH(path) - LENGTH(REPLACE(path, "
                "'/', ''))), path, start LIMIT ?",
                (USE_OVERVIEW_ITEMS,)).fetchall()
        finally:
            con.close()
    except _sq.Error:
        return ""
    return "\n".join(f"{p}:{s}  {(ln or n).strip()[:140]}"
                     for n, _k, p, s, ln in rows)


def conversation_versions(messages: list[dict],
                          state: dict | None = None) -> dict[str, str]:
    """{package: version} this conversation states: the session's recorded
    versions (nebari.observe accumulates them), then any manifest version in
    these messages, the later statement winning (nebari's rule)."""
    import discover
    out = dict((state or {}).get("versions") or {})
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") in ("user", "tool",
                                                     "system"):
            for name, ver in discover.versions(
                    _msg_text(m)[:USE_SCAN_CHARS]).items():
                pkg = discover.package_of(name) or name
                out[pkg] = ver
    return out


def held_version(have: list[tuple[str, str]], stated: str | None
                 ) -> tuple[str, str, str | None]:
    """(version, db, label) of the held index for a package the conversation
    uses at `stated` (pre-deploy review, 2026-09-24: library help used the
    newest held version whatever the conversation used). Exact match first,
    then the same major.minor; else the newest held, with a label saying
    why, which goes into the injected header. `have` is newest first
    (domains.held_sources)."""
    if stated:
        s = stated.strip().lstrip("^~=>v ")
        for ver, db in have:
            if ver == s:
                return ver, db, None
        mm = ".".join(s.split(".")[:2])
        for ver, db in have:
            if mm and ".".join(ver.split(".")[:2]) == mm:
                return ver, db, (f"the nearest held to the conversation's "
                                 f"{stated}")
        ver, db = have[0]
        return ver, db, (f"the newest held; the conversation's {stated} is "
                         f"not held")
    ver, db = have[0]
    return ver, db, ("the newest held; the conversation's version is not "
                     "known")


def _library_use(messages: list[dict], account: str, lineage: str,
                 state: dict | None = None) -> tuple[str, dict]:
    """(the text to append to the message this request ends on, the record).
    Updates the conversation's covered set in the ledger; the caller records
    the text under the message's key."""
    import domains
    rec: dict = {"packages": [], "names": [], "overview": [], "chars": 0,
                 "versions": {}}
    try:
        held = domains.held_sources()
        uses = library_uses(messages)
    except Exception as e:                                       # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
        return "", rec
    used = {p: u for p, u in uses.items() if p in held}
    rec["used_held"] = sorted(used)
    if not used or not lineage:
        return "", rec
    key = "libuse:" + lineage
    try:
        done = json.loads(nebari.ledger_get(account, key, "libuse") or "{}")
    except ValueError:
        done = {}
    spent = int(done.get("_chars") or 0)
    if spent >= USE_CONVERSATION_MAX_CHARS:
        rec["why"] = "the conversation's cap is spent"
        return "", rec
    parts: list[str] = []
    n_defs = 0
    stated = conversation_versions(messages, state)
    for pkg in sorted(used):
        ver, db, label = held_version(held[pkg], stated.get(pkg))
        rec["versions"][pkg] = {"used": ver, "stated": stated.get(pkg),
                                "label": label}
        # The header names the version, and says so when it is not the
        # conversation's own.
        vtag = f"{ver} ({label})" if label else ver
        covered = done.setdefault(pkg, [])
        names = [n for n in used[pkg]["names"] if n not in covered]
        if names:
            for n in names:
                if n_defs >= USE_MAX_NAMES:
                    break
                try:
                    out = _run_on_package(db, "find_definition_opt",
                                          {"symbol": n})
                except Exception:                                # noqa: BLE001
                    out = ""
                covered.append(n)
                if not out or out.lstrip().startswith("{") \
                        or packages._is_empty(out):
                    continue
                parts.append(f"== {pkg}@{vtag}: {n} ==\n"
                             f"{out[:USE_MAX_CHARS_EACH]}")
                rec["names"].append(f"{pkg}:{n}")
                n_defs += 1
        elif "*" not in covered:
            covered.append("*")
            ov = _package_overview(db)
            if ov:
                parts.append(f"== {pkg}@{vtag}: exported classes and "
                             f"functions ==\n{ov}")
                rec["overview"].append(pkg)
        if parts and pkg not in rec["packages"]:
            rec["packages"].append(pkg)
    if not parts:
        nebari.ledger_put(account, lineage, key, "libuse", json.dumps(done))
        return "", rec
    room = min(USE_MAX_CHARS, USE_CONVERSATION_MAX_CHARS - spent)
    text = (USE_HEAD + "\n\n".join(parts))[:room]
    done["_chars"] = spent + len(text)
    nebari.ledger_put(account, lineage, key, "libuse", json.dumps(done))
    rec["chars"] = len(text)
    return text, rec


def _work_log_block(state: dict | None) -> str:
    """The conversation's work log (mcp/rings.py), for the first user turn
    after a compaction. The proxy writes the log itself (_log_turn); the
    model no longer has record_step / read_rings."""
    try:
        import rings
        text = rings.read(session_lineage(state), limit=40)
    except Exception:                                            # noqa: BLE001
        return ""
    if not text or "nothing has been recorded" in text.lower():
        return ""
    return ("\n\n---\nWork log of this conversation before it was "
            "summarised, recorded by the service:\n\n" + text)


def _mark_rings_reinjected(state: dict | None) -> None:
    if not state or not state.get("_key"):
        return
    cur = nebari.load(state["_key"])
    cur["rings_reinjected"] = True
    nebari.save(state["_key"], cur)
    state["rings_reinjected"] = True


def _serve_compaction(out: dict, body: dict, messages: list[dict],
                      inplace: bool) -> None:
    """Shape a compaction on the conversation's stored prompt (mcp/
    compaction.py): the stored prompt byte for byte, the answer as generated,
    one user turn; its slot; thinking off only where that leaves the rendered
    prefix alone; tool_choice none; the compaction budget. Falls back to the
    request as sent -- a flattened one to the transient slot by affinity, as
    before -- and x_yamadori.compaction says which and why."""
    account = body.get("_account") or ""
    rec: dict = {"shape": "in_place" if inplace else "flattened"}
    stored = None
    text = selection._text(messages[-1]) if messages else ""
    parsed = None if inplace else compaction.parse_flattened(text)
    if inplace:
        # THE LEDGER'S RENDERING (2026-09-24). prepare() already put back
        # everything this proxy added to the conversation -- injections, a
        # fixed call's content, image hops with their reasoning emptied; past
        # reasoning passes through as the client sent it -- so the request's
        # own messages render as the slot holds them. The stored prompt is
        # the CHECK: when it (and the answer after it) is a prefix of this
        # rendering, the compaction is served as prepared; when not, the old
        # splice replaces the resent history with it.
        _note_compaction(account)
        key = out["_slot"].get("key")
        e = next((x for x in compaction.entries(account) if x["key"] == key),
                 None)
        if e is None:
            rec.update(mode="ledger", why="no stored prompt to compare (a "
                       "restart, an eviction, or its first turn): the "
                       "ledger's rendering, as prepared")
        else:
            stored_msgs = list(e["upstream"]["messages"]) + (
                [dict(e["response"], role="assistant")]
                if e.get("response") else [])
            fields = {k: e["upstream"][k] for k in
                      ("chat_template_kwargs", "enable_thinking",
                       "reasoning_effort") if k in e["upstream"]}
            fp_s = slots.fingerprint(dict(fields, messages=stored_msgs,
                                          tools=e["upstream"]["tools"]))
            fp_o = slots.fingerprint(dict(fields, messages=out["messages"],
                                          tools=out.get("tools")))
            if fp_s["chars"] and slots.shared_prefix(fp_o, fp_s) \
                    == fp_s["chars"][-1]:
                stored = e
                rec.update(mode="ledger", why="the ledger's rendering extends "
                           f"the stored prompt ({len(stored_msgs)} messages) "
                           "byte for byte")
            else:
                sp = compaction.splice_in_place(e, body.get("messages") or [])
                rec["why"] = ("the ledger's rendering differs from the stored "
                              "prompt; " + sp["why"])
                if sp["ok"]:
                    out["messages"] = sp["messages"]
                    out["tools"] = e["upstream"]["tools"]
                    stored, rec["mode"] = e, "spliced"
                else:
                    rec["mode"] = "as_sent"
    elif not parsed:
        rec.update(mode="as_sent", why="not a flattened transcript this proxy "
                   "can map (no TURNS TO SUMMARIZE block of [ROLE]: records)")
    else:
        best, why = None, "no stored conversation for this account"
        for e in compaction.entries(account):
            stored_msgs = list(e["upstream"]["messages"]) + (
                [dict(e["response"], role="assistant")] if e.get("response")
                else [])
            m = compaction.map_records(parsed["records"], stored_msgs)
            if m["mapped"] and (best is None or m["matched"] > best[1]["matched"]):
                best = (e, m, stored_msgs)
            elif best is None:
                why = m["why"]
        rec["records"] = len(parsed["records"])
        if best:
            e, m, stored_msgs = best
            prev = compaction.find_previous(text, parsed, stored_msgs)
            rec["previous_summary"] = (None if not parsed.get("previous") else
                                       "referenced" if prev is not None else
                                       "kept: not found in the conversation")
            out["messages"] = compaction.continue_messages(
                e, compaction.instruction_for(text, parsed, m, stored_msgs,
                                              prev))
            out["tools"] = e["upstream"]["tools"]
            out["_slot"] = dict(out["_slot"], key=e["key"], transient=False,
                                prefix=False)
            stored = e
            rec.update(mode="rewritten", why=m["why"], mapped=m["matched"],
                       span=[m["first"], m["last"]], conversation=e["key"][:8])
        else:
            rec.update(mode="as_sent", why=why)
    src = stored["upstream"] if stored else out
    client_max = (body.get("max_tokens") or body.get("max_completion_tokens")
                  or (parsed or {}).get("target_tokens"))
    comp = tiers.compaction_budget(
        client_max, tiers.estimate_prompt_tokens({"messages": out["messages"],
                                                  "tools": out.get("tools")}),
        helper_active=admission.helper_active())
    fields = compaction.prefix_fields(src, comp["answer"])
    rec["thinking"] = fields.pop("_thinking")
    thinks = bool(fields["enable_thinking"])
    if not thinks:
        for k in ("reasoning_effort", "reasoning_budget_tokens",
                  "reasoning_budget_message"):
            out.pop(k, None)
    out.update(fields)
    sampling, out["_sampling"] = tiers.enforce_sampling(body, thinks)
    out.update(sampling)
    # tool_choice none: the model cannot call a tool, and the render is the
    # same (llama-server gives the template the tools whatever tool_choice
    # says; mcp/compaction.py cites the lines). No tools, no field.
    if out.get("tools"):
        out["tool_choice"] = "none"
    else:
        out.pop("tool_choice", None)
    out["_fixed_budget"] = True           # tiers.rebudget leaves it alone
    out["_repair"] = out["_tool_code"] = False
    comp.update(rec, target_tokens=(parsed or {}).get("target_tokens"))
    out["_compaction"] = comp
    print(f"  compaction ({rec['shape']}, {rec['mode']}): {rec.get('why')}; "
          f"answer {comp['answer']}, thinking {rec['thinking']}, room "
          f"{comp['room']}", flush=True)


def _budget_note(finish: str | None, content: str,
                 usage: dict | None) -> str | None:
    """What to put in `content` when the model stopped on the token limit.

    None when the finish was normal. The text states the fact -- a limit was
    reached, and whether any answer was written -- so it can never be read as
    the model's answer or as the model returning nothing.
    """
    notice = _budget_notice(finish, content, usage)
    if notice is None:
        return None
    return content.rstrip() + notice if content.strip() else notice


def _budget_notice(finish: str | None, content: str,
                   usage: dict | None) -> str | None:
    """Only the text `_budget_note` adds -- what the streamed path appends.

    A stream has already delivered the partial answer, so it sends the notice
    alone as a final content delta; the blocking path sends answer + notice.
    One function, so the two paths cannot word the event differently.
    """
    if finish != "length":
        return None
    n = (usage or {}).get("completion_tokens")
    spent = f" after {n} tokens" if n else ""
    if not content.strip():
        return (f"[no answer: the model reached its token limit{spent} while "
                f"still thinking (finish_reason=length). This is a budget "
                f"event, not the model's answer. Raise max_tokens, which is "
                f"the answer allowance; thinking is budgeted separately.]")
    return (f"\n\n[answer cut off at the token limit"
            f"{spent} (finish_reason=length); raise max_tokens for the rest]")


# What the landing asks for. Shared by both tool loops so they land the same.
LANDING_PROMPT = ("Stop searching and answer now from what you have. If the "
                  "searches found nothing, answer from your own knowledge -- "
                  "that is expected here and is not a failure. Give the "
                  "complete answer in full.")


def context_full(payload: dict, convo: list[dict], role: str = "main",
                 share_n: int = 1) -> bool:
    """Would one more hop no longer fit in this request's share of the pool?

    One of two things that end a tool loop besides the model stopping; the
    other is the tool-turn cap (tiers.tool_turn_limit, 2026-09-23), which lands the same way. The old
    hop count was removed, and rightly: MAX_TOOL_HOPS=12
    turned a tool bug (package globs matching nothing, 2026-09-22) into a
    33-minute failure instead of exposing it, and a count says nothing about
    whether the model is making progress. The share is the real limit: the
    conversation, its tools and room to think and answer must fit in the part
    of the KV pool this request owns (mcp/budget.py). When it no longer does,
    the tools are withdrawn and the model writes its answer from what it has.

    tool_choice "none" (a compaction, proxy._serve_compaction): no hop can
    follow, so there is nothing to land -- and withdrawing the tools would
    change the rendered prefix the compaction exists to reuse.
    """
    if payload.get("tool_choice") == "none":
        return False
    shares = tiers._shares()
    share = shares["helper" if role == "helper" else "main"] // max(share_n, 1)
    answer = max(int(payload.get("max_tokens") or 0)
                 - int(payload.get("reasoning_budget_tokens") or 0),
                 tiers.A_MIN)
    prompt = tiers.estimate_prompt_tokens({"messages": convo,
                                           "tools": payload.get("tools")})
    return prompt + answer + tiers.MIN_THINKING >= share


# TOOL-TURN CAP (operator, 2026-09-23): PrismML's Bonsai-demo bounds the tool
# loop at agenticMaxTurns = 10, and we follow the vendor. This is NOT the old
# MAX_TOOL_HOPS=12 the operator removed on 2026-09-22: that one ended a loop by
# returning the last tool request as the answer, hiding a broken tool behind a
# 33-minute failure. This one lands -- tools withdrawn, the model answers from
# what it has, exactly as context_full does -- and x_yamadori.tool_turns
# records that it was hit, so a loop caused by a tool shows up in the record.
# A tool turn is one generation whose calls to OUR tools were executed.
# The limit is per tier and per context: tiers.tool_turn_limit (10; 20 at
# `max`).


def _turn_cap(payload: dict | None = None) -> dict:
    return {"limit": tiers.tool_turn_limit((payload or {}).get("_tier")),
            "turns": 0, "hit": False}


# How much content a hop may produce before the proxy decides it is the
# answer and streams it live (see CHANNEL ORDER in stream_body). A preface
# to a tool call is a sentence or two; an answer passes this quickly.
HOLD_CONTENT_CHARS = 400


def _land(payload: dict, convo: list[dict],
          why: str = "context_full") -> dict:
    """THE LANDING: the breaker's last hop, with the tools withdrawn.

    Without it a tool loop has no ending, only an edge. The model -- which
    has just requested a search on every hop -- requests another, and that
    request is returned to the caller AS the answer. On the streamed path it
    was worse: the final call went out with our tools still attached, so a
    call to one of OUR tools could be forwarded to a client that has never
    heard of it (docs/CONSTRAINTS.md item 10a).

    shomen.py has done this correctly all along: withdraw the tools so the
    request is unambiguous, and say what is wanted now. Reaching here is a
    defect report (PROTOCOL rule 15), so it is logged as a breaker trip.
    """
    if why == "tool_turn_cap":
        cap = payload.get("_turn_cap") or {}
        print(f"  tool_turn_cap: {cap.get('limit')} tool turns reached; tools "
              "withdrawn for the landing", flush=True)
    elif not payload.get("tools"):
        # NO TOOLS, NOTHING TO WITHDRAW. A request that carries no tool
        # cannot loop, and "stop searching" appended to it -- a client's
        # compaction, a puzzle with the tools withheld -- is an instruction
        # about searches that never happened, written into the client's own
        # conversation. It is sent as it is.
        print("  context_full: a request with no tools; sent as it is",
              flush=True)
        return dict(payload, messages=convo)
    else:
        print("  context_full: this request's share of the KV pool cannot fit "
              "another tool hop; tools withdrawn for the landing", flush=True)
    convo.append({"role": "user", "content": LANDING_PROMPT})
    out = dict(payload, tools=[], messages=convo)
    # A forced tool_choice with no tools is a contradiction the server may
    # reject; the landing wants text.
    out.pop("tool_choice", None)
    return out


# What crosses the callosum: deep thinking's hand-off (shomen.SECTIONS),
# PREFILLED AS MAIN'S OWN REASONING (operator decision 2, 2026-09-24): the
# second brain's thinking becomes main's thinking for THIS request (its
# hidden hops included, _run_turn); on later turns it is whatever the client
# echoes back -- the ledger restores no reasoning (pass-through,
# 2026-09-24) -- and the user sees only the conclusion -- main's visible
# answer, which opens with the fold-back phrase (shomen.opening). A run that
# SEARCHED crosses under FINDINGS_HEAD; one that made no search under
# REASONING_HEAD, which says no source was checked (operator, 2026-09-23).
FINDINGS_HEAD = ("I investigated this in the library source before "
                 "answering. A fact ending in path:line was read there; a "
                 "fact labelled otherwise was not checked.\n\n")
REASONING_HEAD = ("I thought this through before answering, without "
                  "searching: reasoning, no sources checked. Nothing below "
                  "was read from a file.\n\n")


def _deep_thinking(payload: dict, messages: list[dict], db: str | None = None,
                   root: str | None = None, state: dict | None = None):
    """DEEP THINKING. A generator: yields the investigation's trace lines
    while it runs, and RETURNS the record for `x_yamadori` (None when the
    selection engine did not choose it).

    WHETHER it runs is `payload["_selection"]["investigate"]`: the tier
    allows and a Phase 0.6 trigger fired before main (mcp/deep.py:
    struggle, a known-hard area, a task kickoff), or a header forces it --
    mcp/selection.py records which. It runs as the trigger's job of the one
    second-brain runner (shomen.run: `investigate`, or `plan` for a
    kickoff), with the concept seed the ledger recorded for this request and
    job, and the second brain's tools (deep_thinking_tools), never the
    request's. The model's own trigger, think_deeply, is _think_deeply.

    WHAT CROSSES: the hand-off, ALWAYS, once it ran -- shomen's four
    sections, labelled per fact -- as `payload["_prefill"]`: an assistant
    message whose reasoning_content is the hand-off and whose content is the
    opening phrase. The main generation continues it (STEP 0 (a): the next
    request reused 976 of 995 prompt tokens after such a prefill). No
    written hand-off (the helper returned nothing, failed or raised):
    shomen.machine_handoff() builds one from the trace, labelled.
    x_yamadori.investigate.handoff carries the counts, never the text.
    """
    sel = payload.get("_selection") or {}
    if not sel.get("investigate"):
        return None
    # WHAT TO THINK ABOUT. A Phase 0.6 trigger (mcp/deep.py) states its own
    # question -- the struggle with its failing output, the unseen package,
    # the task to plan -- and its job (`plan` for a kickoff). A header that
    # forces deep thinking, or nothing, asks the last user turn, as before.
    trig = payload.get("_deep") or {}
    own = bool(trig.get("fire") and trig.get("question")
               and trig.get("kind") in ("struggle", "kickoff", "area"))
    job = (trig.get("job") or "investigate") if own else "investigate"
    if own:
        q, ctx = trig["question"], trig.get("context") or ""
    else:
        # The messages as the text model reads them: an attached image is
        # its placeholder, so the question carries the id describe_image
        # takes.
        q, ctx, _speaking = selection.question_of(
            payload.get("_seen_messages") or messages)
    rec: dict = {"ran": False, "hops": 0, "handle": None, "injected": False,
                 "trigger": trig.get("kind") if trig.get("fire") else None,
                 "job": job}
    if len(q.strip()) < selection.MIN_QUESTION_CHARS:
        rec["why"] = "no question to think about"
        return rec
    import queue as _queue
    import threading as _threading

    seed = ledger_seed(payload, job, q)
    ctx_run = _research_context(payload, messages)
    if state is not None:
        state["_research_budget"] = ctx_run
    box: dict = {}
    trace_q: _queue.Queue = _queue.Queue()

    def _watched(fn, a):
        out = run_our_tool(fn, a, db, root, None, state)
        trace_q.put("  " + streaming.describe_call(fn, a))
        return out

    tier = payload.get("_tier") or {}

    def _work():
        try:
            box["res"] = shomen.run(
                job, question=q,
                tools=deep_thinking_tools((state or {}).get("_attached")),
                run_tool=_watched, context=ctx,
                on_think=lambda t: trace_q.put(t.rstrip()),
                # The request's tier (medium on `xhigh`, xhigh on `max`); an
                # effort override (the domain benchmark's header) wins.
                tier=tier.get("name") or "max",
                effort=(tier.get("effort") if "effort" in
                        (tier.get("overridden") or []) else None),
                seed=seed, lane_timeout=trig.get("lane_timeout"))
        except Exception as e:                                   # noqa: BLE001
            box["err"] = f"{type(e).__name__}: {e}"
        finally:
            trace_q.put(None)

    # The investigation runs in a thread and its tool calls are streamed AS
    # THEY HAPPEN: the searches ARE the thinking. A thread because it blocks
    # for minutes and this is a generator.
    th = _threading.Thread(target=_work, daemon=True)
    th.start()
    while True:
        line = trace_q.get()
        if line is None:
            break
        yield line
    th.join(timeout=5)
    res = box.get("res") or {}
    if res.get("skipped"):
        # Reported, not swallowed: an investigation that quietly did not
        # happen looks exactly like one that found nothing.
        rec["why"] = "the helper lane is busy with another request"
        rec["skipped"] = True
        print(f"  deep thinking skipped: {rec['why']}", flush=True)
        return rec
    text, stats, searches = _handoff_of(res, q, box.get("err"), rec, seed)
    rec["web_refused"] = len(ctx_run.get("refused") or [])
    if searches == 0:
        rec.setdefault("why", "no search ran: handed off as reasoning, no "
                              "sources checked")
    head = (deep.PLAN_HEAD if job == "plan" else
            FINDINGS_HEAD if searches else REASONING_HEAD)
    seeds = [seed] if seed else []
    # The second brain wrote the hand-off, but OUR path puts it inside main's
    # think block: a marker in it would close that block early (#12).
    text, n_scrub = scrub_markers(text)
    _note_scrub(payload, "hand-off", n_scrub)
    # Then the screen on the way to main (FETCHED CONTENT IS DATA): after
    # the template markers are scrubbed and counted, so a marker is counted
    # as ours, not mistaken for a role token.
    text = _screen_handoff(payload, text, ctx_run, rec)
    payload["_prefill"] = {"role": "assistant",
                           "reasoning_content": head + text,
                           "content": shomen.opening(seeds)}
    payload.setdefault("_fold_back", []).append(
        {"job": job, "phrase": "investigate", "into": "prefill",
         "trigger": rec["trigger"],
         "seeds": [s.get("word") for s in seeds]})
    rec["injected"] = True
    rec["into"] = "prefill"
    rec["searches"] = searches
    rec["handoff"] = _handoff_record(stats)
    # What main should act on next, for the outcome check (deep.observe:
    # a hand-off none of whose names main then uses was wasted).
    payload["_deep_terms"] = deep.handoff_terms(text)
    h = rec["handoff"]
    print(f"  deep thinking: ran, {searches} searches, hand-off prefilled as "
          f"reasoning: {h['facts']} facts ({h['unverified']} unverified), "
          f"{h['searched_empty']} searched-empty, {h['open']} open, "
          f"{h['chars']} chars"
          + (", MACHINE-BUILT" if h["machine_built"] else "")
          + (f" -- {rec['why']}" if rec.get("why") else "")
          + f" (handle {rec['handle']})", flush=True)
    return rec


def _handoff_record(stats: dict | None) -> dict:
    return {k: (stats or {}).get(k) for k in (
        "facts", "verified", "unverified", "searched_empty", "open", "chars",
        "machine_built", "cut",
        # Did the helper write the four sections? The rate of this is how
        # well the format is followed -- measure it before relying on it.
        "structured", "plan")}


def _handoff_of(res: dict, q: str, err: str | None, rec: dict,
                seed: dict | None) -> tuple[str, dict | None, int]:
    """(hand-off text, its stats, searches) from one second-brain run, the
    record `rec` updated. A raise, or a result carrying no hand-off, is
    rebuilt from the trace (a written `finding` is still used): an empty
    hand-off is impossible (operator, 2026-09-23)."""
    rec.update(ran=True, ok=bool(res.get("ok")), hops=int(res.get("hops") or 0),
               handle=res.get("handle"), seconds=res.get("seconds"),
               seed=res.get("seed") or shomen.concept_summary(seed),
               effort=res.get("effort"),
               cited=len(res.get("cited") or []),
               unsupported=len(res.get("unsupported") or []))
    # Real searches, not the "(turn cap)" / "(budget)" trace markers.
    searches = int(res.get("searches", rec["hops"]) or 0)
    text = (res.get("handoff") or "").strip()
    stats = res.get("handoff_stats")
    if err or not text:
        tr = shomen.get_trace(res.get("handle") or "") or {}
        seen = set(tr.get("retrieved") or res.get("cited") or [])
        written = (res.get("finding") or "").strip() if res.get("ok") else ""
        if err:
            rec["why"] = "the investigation raised: " + err[:200]
        if written:
            text, stats = shomen.handoff(written, tr.get("trace") or [], seen,
                                         rec["handle"] or "")
        else:
            text, stats = shomen.machine_handoff(
                q, tr.get("trace") or [], seen,
                rec.get("why") or res.get("error") or "no hand-off came back",
                rec["handle"] or "")
    return text, stats, searches


# THINK_DEEPLY, THE MODEL-CHOSEN TRIGGER (Phase 0.6, operator 2026-09-24).
# Main calls it; the proxy runs the question through the one second-brain
# runner (shomen.run("investigate")) in the helper lane and returns the
# hand-off as the TOOL RESULT. The call and its result are a hidden hop: the
# client never sees them, and the ledger records and replays them
# (ledger_record_turn / ledger_restore) so the next request extends the
# slot. The next hop is prefilled: reasoning deep.THINK_REASONING, content
# the fold-back opening (seed line + "After thinking deeply,").
THINK_CALLS_PER_REQUEST = 1


def _think_deeply(payload: dict, args: dict, db: str | None,
                  root: str | None, state: dict | None,
                  n_messages: int) -> str:
    """Run one think_deeply call. Returns the tool result; the record goes
    to payload["_think_tool"]["calls"]. Refusals are structured."""
    tt = payload.setdefault("_think_tool", {"offered": True, "calls": []})
    call: dict = {"ran": False}
    tt["calls"].append(call)
    tier = payload.get("_tier") or {}
    q = args.get("question")
    if not isinstance(q, str) or len(q.strip()) < selection.MIN_QUESTION_CHARS:
        call["refused"] = "BAD_ARGUMENTS"
        return cs.error_result(
            deep.TOOL_NAME, "BAD_ARGUMENTS",
            "`question` must be a string of at least 8 characters naming "
            "what to think about. Nothing was run.", retryable=True,
            remedies=[{"fixable_by": "agent",
                       "action": "call again with one self-contained "
                                 "question naming the files, symbols and "
                                 "error",
                       "effect": "deep thinking runs"}])
    over = tier.get("overridden") or []
    if not tier.get("investigate"):
        call["refused"] = "DEEP_THINKING_OFF"
        return cs.error_result(
            deep.TOOL_NAME, "DEEP_THINKING_OFF",
            f"Deep thinking is off for this request (tier "
            f"{tier.get('name', '?')}"
            + (", forced off by X-Yamadori-Features" if "investigate" in over
               else "") + "). Nothing was run.", retryable=False,
            remedies=[{"fixable_by": "agent",
                       "action": "answer from what is in the conversation, "
                                 "or read the files with your own tools",
                       "effect": "the task goes on without it"}])
    ran_before = [c for c in tt["calls"][:-1] if c.get("ran")]
    if (payload.get("_think_pre") or {}).get("ran") \
            or len(ran_before) >= THINK_CALLS_PER_REQUEST:
        call["refused"] = "ALREADY_THOUGHT"
        return cs.error_result(
            deep.TOOL_NAME, "ALREADY_THOUGHT",
            "Deep thinking already ran for this request; its hand-off is "
            "above (in your thinking, or the earlier think_deeply result). "
            "Nothing new was run.", retryable=False,
            remedies=[{"fixable_by": "agent",
                       "action": "act on that hand-off's NEXT STEP",
                       "effect": "the task moves on"}])
    tried = args.get("tried")
    question = q.strip() + ("\n\nAlready tried, and how it failed:\n"
                            + tried.strip()[:2000]
                            if isinstance(tried, str) and tried.strip()
                            else "")
    _q, ctx, _sp = selection.question_of(
        payload.get("_seen_messages") or payload.get("messages") or [])
    seed = ledger_seed(payload, "think_deeply", question)
    ctx_run = _research_context(payload, payload.get("_client_messages")
                                or [])
    if state is not None:
        state["_research_budget"] = ctx_run
    lg = payload.get("_ledger") or {}
    dst = deep.load_state(lg.get("account") or "", lg.get("session") or "")
    # Main waits for a busy lane at most once per episode (deep.DEFER_...).
    lane_timeout = (0 if dst.get("waited_episode") == int(
        dst.get("episode") or 0) and "waited_episode" in dst else None)
    try:
        res = shomen.run(
            "investigate", question=question,
            tools=deep_thinking_tools((state or {}).get("_attached")),
            run_tool=lambda fn, a: run_our_tool(fn, a, db, root, None, state),
            context=ctx, tier=tier.get("name") or "max",
            effort=(tier.get("effort") if "effort" in over else None),
            seed=seed, lane_timeout=lane_timeout)
        err = None
    except Exception as e:                                       # noqa: BLE001
        res, err = {}, f"{type(e).__name__}: {e}"
    if res.get("skipped"):
        call["refused"] = "HELPER_BUSY"
        deep.mark_deferred(lg.get("account") or "", lg.get("session") or "",
                           "model")
        return cs.error_result(
            deep.TOOL_NAME, "HELPER_BUSY",
            "Deep thinking is already running for another request; only one "
            "runs at a time. Nothing was thought about here.",
            retryable=True,
            remedies=[{"fixable_by": "agent",
                       "action": "go on from what is in context, or call "
                                 "again on a later step",
                       "effect": "the lane frees when the other run ends"}])
    text, stats, searches = _handoff_of(res, question, err, call, seed)
    call["web_refused"] = len(ctx_run.get("refused") or [])
    head = FINDINGS_HEAD if searches else REASONING_HEAD
    text, n_scrub = scrub_markers(head + text)
    _note_scrub(payload, "think_deeply", n_scrub)
    text = _screen_handoff(payload, text, ctx_run, call)
    call.update(searches=searches, handoff=_handoff_record(stats))
    call["_seed"] = seed
    payload["_deep_terms"] = deep.handoff_terms(text)
    deep.mark_ran(lg.get("account") or "", lg.get("session") or "",
                  n_messages, "model")
    print(f"  think_deeply: ran, {searches} searches, "
          f"{(stats or {}).get('chars')} chars handed back as the tool "
          f"result (handle {call.get('handle')})", flush=True)
    return text


def _drain(gen):
    """Run a generator to the end; its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def _fan_out(payload: dict, msg: dict, finish: str | None):
    """FAN-OUT: (dissent note, x_yamadori record, winner).

    Runs only when the selection engine chose N > 1 and the answer is a
    finished text answer -- not a budget event, not a hand-off of tool calls
    to the client. The original answer is candidate A; B (and, when the code
    check does not separate A and B, the tie-breaker C) are jobs of the one
    second-brain runner (fanout.run -> shomen.run), each with the concept
    seed the ledger recorded for this request and job.

    MEASURED, and a null where it was tried (the old four-way concurrent
    design): 7/8 against 7/8 on file-location questions, ZERO discordant
    pairs, at 3.2x wall clock. Nothing here claims it helps.

    WHAT CROSSES BACK (_finish_text, the fold-back "Compared two
    approaches"): a code winner that is not A is delivered -- in place of A
    on the blocking path, after it on a stream -- with one line saying why
    and what the others used; prose keeps A, and B's differing points
    (fanout.handback) are prefilled after it for main to weigh in its own
    turn. The weigh turn this replaces (a new user turn in main's context
    the client never saw) is gone (operator, 2026-09-24).
    """
    n = int((payload.get("_selection") or {}).get("fanout_n") or 1)
    if (n <= 1 or finish != "stop" or msg.get("tool_calls")
            or not (msg.get("content") or "").strip()):
        return "", None, None
    original = {"variant": ORIGINAL, "seed": None,
                "content": msg.get("content") or "",
                "raw": {"choices": [{"message": {"content": msg.get("content")},
                                     "finish_reason": finish}]}}
    try:
        v = fanout.run(dict(payload, tools=[]), original=original, n=n,
                       seed_for=lambda job, prompt: ledger_seed(
                           payload, job, prompt))
    except Exception as e:                                       # noqa: BLE001
        print(f"  fan-out unavailable, answering once: "
              f"{type(e).__name__}: {e}", flush=True)
        return "", {"mode": "sequential", "n": 0, "asked": n, "seeds": [],
                    "agreement": None, "error": type(e).__name__,
                    "replaced": False, "appended": False}, None
    results = list(v.get("results") or [])
    others = results[1:]
    note = fanout.dissent_note(v)
    win = v.get("winner") or None
    rec = {"n": sum(1 for r in results if r.get("content")), "asked": n,
           "seeds": [r.get("seed") for r in others if r.get("seed")],
           "agreement": v.get("agreement"),
           "votes": v.get("votes", 0),
           "winner": (win or {}).get("variant"),
           "dissent_noted": bool(note),
           "replaced": False, "appended": False, "handback": None}
    rec.update(fanout.selection_record(v))
    try:
        hb = fanout.handback(v)
    except Exception as e:                                       # noqa: BLE001
        print(f"  fan-out hand-back unavailable: {type(e).__name__}: {e}",
              flush=True)
        hb = None
    if hb:
        rec["_handback"] = hb
        rec["handback"] = {"kind": hb["kind"], "from": hb["from"],
                           "points": hb["points"], "chars": 0, "into": None}
    print(f"  fan-out (sequential): {rec.get('steps')} steps, "
          f"{rec.get('stop_reason')}, similarity A-B "
          f"{rec.get('similarity_ab')}, selection {rec.get('selection')}, "
          f"winner {rec['winner']}", flush=True)
    return note, rec, win


# The name the original answer carries among the fan-out candidates.
ORIGINAL = "original"

# The selections whose winner is delivered: fanout.run's grade after two, and
# the code medoid after the tie-breaker.
DELIVERED_SELECTIONS = ("code_grade", "code_medoid")


def _delivers_winner(fan: dict | None, win: dict | None) -> bool:
    """Does the fan-out winner REPLACE (or, streamed, follow) the answer?

    Only for a code answer -- chosen by the grade after two (`code_grade`) or
    as the medoid after the tie-breaker (`code_medoid`) -- and only when the
    winner is not the original. A `fallback` (nothing parsed) is recorded,
    never delivered. Prose answers chosen by the path vote keep
    the original: there is no measured basis for swapping prose.
    """
    return bool(fan and win and fan.get("selection") in DELIVERED_SELECTIONS
                and win.get("variant") != ORIGINAL
                and (win.get("content") or "").strip())


def _code_of(content: str) -> str:
    """The winner's fenced code blocks, re-fenced, for the streamed append."""
    out = []
    for b in code_check.fenced_blocks(content or ""):
        if b["code"].strip():
            fence = "````" if "```" in b["code"] else "```"
            out.append(f"{fence}{b['lang']}\n{b['code'].rstrip()}\n{fence}")
    return "\n\n".join(out)


def _repair_state(payload: dict) -> dict | None:
    """The x_yamadori.repair record for a final answer's code, or None when
    the check is off (not a code route, or below `high`)."""
    if not payload.get("_repair"):
        return None
    return {"enabled": True, "rounds": 0, "errors_before": None,
            "errors_after": None, "blocks_checked": 0, "stopped": None,
            "check_mode": None, "into": None}


def _tool_evidence(name: str, out: str) -> dict:
    """One proxy-executed tool call, for x_yamadori.tools: its name, whether
    it came back empty or as an error, and its size. No argument or result
    text -- the corpus has those; the response carries only the facts."""
    err = False
    s = (out or "").lstrip()
    if s.startswith("{"):
        try:
            d = json.loads(s)
            err = isinstance(d, dict) and d.get("ok") is False
        except ValueError:
            pass
    return {"name": name, "empty": repeats._empty(out), "error": err,
            "chars": len(out or "")}


def _research_context(payload: dict, messages: list[dict]) -> dict:
    """One deep-thinking run's context for the second brain's web tools
    (research_tools.run's `budget`): its search count, the URLs its
    searches return and the USER's own URLs (the only ones read_web_page
    may read), and the conversation's other text -- tool results and
    answers -- for read_web_page's leak check. In memory, for the run."""
    user_urls: list[str] = []
    other: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        t = _msg_text(m)
        if m.get("role") == "user":
            user_urls += research_tools.urls_in(t)
        elif m.get("role") in ("tool", "function", "assistant"):
            other.append(t)
            for c in m.get("tool_calls") or []:
                other.append(str((c.get("function") or {}).get("arguments")))
    own = "\n".join(_msg_text(m) for m in messages or []
                    if isinstance(m, dict))
    return {"search_web": 0, "urls": [], "user_urls": user_urls[-100:],
            "conversation": "\n".join(other)[-200000:],
            "own_text": own[-400000:], "web_text": "", "refused": [],
            "screened": []}


def _screen_handoff(payload: dict, text: str, ctx_run: dict,
                    rec: dict) -> str:
    """FETCHED CONTENT IS DATA, on the way out (skill_screen.screen_handoff):
    the hand-off is the path to the user. What was removed or labelled, and
    what the run's fetches stripped, go to rec["screen"] and
    x_yamadori.deep.screen."""
    import skill_screen
    text, scr = skill_screen.screen_handoff(
        text, web_text=ctx_run.get("web_text") or "",
        own_text=ctx_run.get("own_text") or "")
    summary = {"removed": len(scr["removed"]), "labelled": scr["labelled"],
               "removed_rules": sorted({x["rule"] for x in scr["removed"]}),
               "fetched": list(ctx_run.get("screened") or [])[:20],
               "urls_refused": list(ctx_run.get("refused") or [])[:20]}
    rec["screen"] = summary
    if summary["removed"] or summary["labelled"] or summary["fetched"] \
            or summary["urls_refused"]:
        payload.setdefault("_deep_screen", []).append(summary)
        print(f"  deep: hand-off screen removed {summary['removed']} "
              f"line(s) {summary['removed_rules']}, labelled "
              f"{summary['labelled']} as from the web", flush=True)
    return text


def _deep_record(payload: dict, messages: list[dict],
                 think: dict | None) -> str | None:
    """deep.observe for this conversation, then deep.record for this
    request, OFF the response path (deep.submit: one background thread, a
    bounded queue, one transaction). Returns the row id at once
    (x_yamadori.deep.record). Never raises."""
    trig = payload.get("_deep")
    if not trig or trig.get("skip_record"):
        return None
    lg = payload.get("_ledger") or {}
    account, conv = lg.get("account") or "", lg.get("session") or ""
    try:
        calls = [c for c in (payload.get("_think_tool") or {}).get("calls")
                 or [] if c.get("ran")]
        ran_pre = bool(think and think.get("ran"))
        handoff = ((think or {}).get("handoff") if ran_pre else
                   calls[-1].get("handoff") if calls else None)
        rid = deep.new_id()
        ok = deep.submit(
            deep.observe_and_record, account=account, conversation=conv,
            messages=list(messages), rid=rid,
            tier=(payload.get("_tier") or {}).get("name"),
            route=(payload.get("_route") or {}).get("class"),
            rec=json.loads(json.dumps(deep.public(trig) | {
                "epoch": trig.get("epoch"), "episode": trig.get("episode")},
                default=str)),
            n_messages=len(messages), ran=ran_pre or bool(calls),
            kind=(trig.get("kind") if trig.get("fire") else None),
            model_calls=len(calls), handoff=handoff,
            terms=list(payload.get("_deep_terms") or []))
        return rid if ok else None
    except Exception as e:                                       # noqa: BLE001
        print(f"  deep: recording failed ({type(e).__name__}: {e})",
              flush=True)
        return None


def _deep_public(payload: dict) -> dict | None:
    """x_yamadori.deep: the trigger decision (never the question text), the
    thresholds in force, think_deeply's offer and calls, the record id."""
    trig = payload.get("_deep")
    if trig is None:
        return None
    out = deep.public(trig) or {}
    tt = payload.get("_think_tool") or {}
    out["think_tool"] = {"offered": bool(tt.get("offered")),
                         "calls": [{k: v for k, v in c.items()
                                    if not k.startswith("_")}
                                   for c in tt.get("calls") or []]}
    out["record"] = payload.get("_deep_record")
    # FETCHED CONTENT IS DATA: what the screen stripped from fetched text and
    # removed from (or labelled in) the hand-off, per run.
    out["screen"] = list(payload.get("_deep_screen") or [])
    return out


def _x_yamadori(payload: dict, *, hops: int, fan: dict | None,
                think: dict | None, repair: dict | None = None,
                tool_check: dict | None = None) -> dict:
    """Every decision this request took, on the response, as data.

    `x_yamadori` is a top-level extension key: OpenAI clients ignore keys
    they do not know. It exists so that a live check can read what happened
    instead of asking the model to quote it (the hints check had to). It
    carries decisions and numbers only -- no message text beyond 120
    characters of each recipe, and never the account or its key.

    `hops` is the number of upstream generations in the main loop, the same
    number as `usage.hops`, on both paths.
    """
    tier = payload.get("_tier") or {}
    gate = payload.get("_tools_gate")
    lg = payload.get("_ledger") or {}
    return {
        "tier": tier.get("name"),
        "effort_sent": payload.get("reasoning_effort"),
        "tools_gate": ({"offer": bool(gate.get("offer")),
                        "why": f"{gate.get('situation')}: {gate.get('because')}"}
                       if gate else None),
        # Skills (mcp/skill_select.py): path, on, route_class, ids, versions,
        # why, matched. `hints` and `suppressed_hints` are DEPRECATED aliases,
        # kept one release: the legacy recipe rows on the hints path, one
        # entry per injected skill (recipe = its title) on the skills path.
        "skills": payload.get("_skills"),
        "hints": list(payload.get("_hints") or []),
        "suppressed_hints": list(payload.get("_suppressed_hints") or []),
        "selection": payload.get("_selection"),
        # The hand-back's text rides in the record as `_handback` and never
        # leaves: only its counts (`handback`) are data.
        "fanout": ({k: v for k, v in fan.items() if not k.startswith("_")}
                   if isinstance(fan, dict) else fan),
        "investigate": think,
        # Deep thinking's triggers (Phase 0.6, mcp/deep.py): which fired or
        # why none did, the signals counted, the thresholds in force (and
        # whether each is the default, pinned or learned), the cooldown, and
        # think_deeply's offer and calls; `record` names the durable row.
        "deep": _deep_public(payload),
        "hops": hops,
        # One entry per generate_image call: ok, id prefix, size, seed,
        # seconds -- or the error code. Never the prompt or the URL.
        "images": list(payload.get("_images") or []),
        # One entry per describe_image call: ok, which image (attached id or
        # sha prefix), source, format, bytes, seconds, token usage -- or the
        # error code. Never the image or the question.
        "vision": list(payload.get("_vision") or []),
        # The images this request carried and what became of each: id,
        # source, format, bytes, error. Never the bytes.
        "attachments": list(payload.get("_attachments") or []),
        # Every A4000 decision this request caused (mcp/gpu_room.py): model,
        # action (loaded / fit / evicted / busy / no_room / uncoordinated),
        # need, free before/after, what was unloaded. Numbers and model ids.
        "gpu_room": list(payload.get("_gpu_room") or []),
        # One entry per tool call the proxy executed in THIS conversation's
        # loop (image tools, the delegate arm): name, empty, error, chars.
        "tools": list(payload.get("_tool_calls") or []),
        # The tool-turn cap (tiers.tool_turn_limit): limit, tool turns
        # executed, and whether the loop landed because of it.
        "tool_turns": dict(payload.get("_turn_cap") or _turn_cap(payload)),
        # The check_code TOOL left main on 2026-09-24 (operator): client
        # writes are checked by the proxy (`tool_code`), an answer's code by
        # the repair pass (`repair`). Kept, always unoffered, for readers of
        # older rows (bench/domain/analyse.py).
        "check_code": {"offered": False, "calls": 0, "results": []},
        # A final answer's code: None when off; otherwise errors in the
        # answer, the fixup job's rounds, and where the result went ("note"
        # for Verified, "in_place" / "appended" for Repaired).
        "repair": ({k: v for k, v in repair.items() if not k.startswith("_")}
                   if repair is not None else None),
        # The route (mcp/route.py): {class, because, signals}, decided once
        # in prepare. What fan-out, repair and deep thinking read.
        "route": payload.get("_route"),
        # The tool-call code check (mcp/tool_code.py): None when off;
        # otherwise the files checked (path, language, errors before and
        # after, formatted), the fixup job, why it stopped, and client calls
        # that carried code-sized strings nothing recognised. Never the code.
        "tool_code": tool_code.public(tool_check),
        # The fold-backs this turn carried (shomen.PHRASES): job, phrase,
        # where it went (prefill / note / continuation / in_place /
        # appended), and the seed words named.
        "fold_back": list(payload.get("_fold_back") or []),
        # The ledger: what was restored on this request (counts per kind)
        # and whether this request decided its user turn's injection.
        # `keyed_by_shown`: the client was streamed more content than the
        # turn renders (an earlier hop's), so the turn is keyed by what it
        # stores and its content restored as delivered (#10).
        "ledger": {"restored": lg.get("restored") or {},
                   "inject": lg.get("inject"),
                   "keyed_by_shown": bool(payload.get("_stored_differs"))},
        # Whether the slot is being warmed with the delivered turn, and why.
        "warm": payload.get("_warm"),
        # The warm that ran after this conversation's PREVIOUS response:
        # reused / processed / slot, and `short` when it reused less than
        # the prompt that slot had just generated on (#11).
        "warm_before": payload.get("_warm_before"),
        # The chat template's own markers (<think>, </think>, <|im_start|>,
        # <|im_end|>, <tool_call>) in the delivered content -- counts and
        # whose (`model`: left as written; `ours`: a defect) -- and what our
        # own path scrubbed before it reached a prompt. None when clean (#12).
        "template_markers": payload.get("_template_markers"),
        # LIBRARY USE (#19): held packages this conversation uses, and the
        # definitions or overview injected for them on this request (names,
        # never text), or that a recorded injection was replayed.
        "library_use": payload.get("_library_use"),
        # Vendor sampling as enforced by tiers.apply, and what the client
        # had asked for where it differed.
        "sampling": payload.get("_sampling"),
        "budget": {"max_tokens_sent": payload.get("max_tokens"),
                   "reasoning_budget_tokens":
                       payload.get("reasoning_budget_tokens")},
        # A client utility call runs at `minimal` whatever the client's tier
        # (proxy.prepare); the rule's reason is selection.because.utility.
        "utility": bool((payload.get("_utility") or {}).get("utility")),
        "tier_requested": payload.get("_tier_requested") or tier.get("name"),
        "tier_overridden": ("minimal" if (payload.get("_utility") or {})
                            .get("utility") else None),
        # compaction | classifier | structured | other, None for a task turn
        # (selection.utility_kind). A compaction also carries its budget
        # record (tiers.compaction_budget).
        "utility_kind": payload.get("_utility_kind"),
        "compaction": payload.get("_compaction"),
        # Prompt tokens processed vs reused from the slot's cache, and which
        # slot (mcp/slots.py), summed over this request's generations.
        "cache": _cache_summary(payload),
        # Electricity for this request (mcp/power.py).
        "energy": power.request_energy(payload.get("_t_start")),
    }


def _cache_summary(payload: dict, log_line: bool = True) -> dict | None:
    """x_yamadori.cache: {prompt, reused, processed, slot, mode, hops} over
    the request's generations, from `_cache_log`; None when nothing was
    generated. One proxy log line per request, from the same numbers."""
    recs = list(payload.get("_cache_log") or [])
    if not recs:
        return None

    def total(k):
        vals = [r.get(k) for r in recs]
        return None if any(v is None for v in vals) else sum(int(v) for v in vals)

    last = recs[-1]
    out = {"prompt": total("prompt"), "reused": total("reused"),
           "processed": total("processed"), "slot": last.get("slot"),
           "mode": last.get("mode"), "generations": len(recs),
           "evicted": next((r.get("evicted") for r in recs if r.get("evicted")),
                           None),
           "first": {k: recs[0].get(k) for k in ("prompt", "reused",
                                                  "processed")}}
    aff = next((r["affinity"] for r in recs if r.get("affinity")), None)
    if aff:
        out["affinity"] = aff
    if log_line:
        f = out["first"]
        print(f"  cache: slot {out['slot']} ({out['mode']}) "
              f"first prompt {f['prompt']} reused {f['reused']} processed "
              f"{f['processed']}; {len(recs)} generation(s) reused "
              f"{out['reused']} of {out['prompt']}"
              + (f"; evicted {out['evicted']}" if out["evicted"] else ""),
              flush=True)
    return out


def _empty_notice(finish: str | None, content: str, msg: dict) -> str | None:
    """What an answer says when the model wrote NOTHING: no content, no tool
    call, and a normal stop. None otherwise.

    Found 2026-09-23: 28 Hermes turns were logged `chars: 0` and read as empty
    answers. They were not -- every one was followed by the same user request
    with one assistant message and its tool results appended (corpus replay in
    mcp/test_utility.py), i.e. a client tool call with no preface text, and
    corpus.log_answer counted only content. That record now says so
    (finish, tool_calls). What was never explained is the real case, a stop
    with nothing written, which reached the client as a blank answer: that
    is what this says instead (AGENTS.md, "Failure returns carry the next
    step")."""
    if (content or "").strip() or msg.get("tool_calls") or finish != "stop":
        return None
    thought = len(msg.get("reasoning_content") or "")
    return ("[no answer: the model stopped (finish_reason=stop) without "
            "writing an answer or calling a tool"
            + (f", after {thought} characters of reasoning, which are in "
               f"reasoning_content" if thought else "")
            + ". Nothing was cut off and nothing failed. Retryable: yes -- "
            "the same request samples again and normally answers. If it "
            "repeats, the operator can read this turn in the corpus.]")


# ============================================================ THE TURN =====
#
# ONE implementation of a turn, for both paths: `_run_turn` is a generator
# of events -- ("reasoning", text), ("content", text), ("heartbeat", None),
# ("calls", [client calls]) -- that RETURNS the response. complete() drains
# it (streamed=False: nothing is presented, and a repaired answer is
# delivered in place); stream_body() turns each event into an SSE chunk.
# Two copies of this loop had drifted three times (tool lists, deep
# thinking's tools, fan-out on one path only).
#
# WHAT MAIN'S CONTEXT HOLDS: the client's messages, the ledger's additions,
# the addendum, main's own generations -- and nothing else. No repair turn,
# no tool-call round, no weigh turn, no hand-off as a user turn: those were
# turns in main's context the client never received, so the next request
# could not extend what the slot held. The second brain does that work
# (shomen.run) and it folds back in fixed phrases (shomen.PHRASES), either
# PREFILLED into a main generation (the slot processed exactly those
# tokens) or written by the proxy as a note -- and then the slot is WARMED
# with the delivered turn (STEP 0 (b): the next request processed only its
# 23-token tail).

# Seconds between empty deltas while the second brain works on a stream.
HEARTBEAT = FANOUT_HEARTBEAT


def _in_thread(fn):
    """Run fn() in a daemon thread; yield heartbeats until it ends; return
    ("ok", value) or ("error", exception). Silence of a minute is
    indistinguishable from a hang (streaming.py)."""
    import threading as _threading
    box: dict = {}

    def _go():
        try:
            box["v"] = fn()
        except Exception as e:                                   # noqa: BLE001
            box["e"] = e

    th = _threading.Thread(target=_go, daemon=True)
    th.start()
    while th.is_alive():
        th.join(timeout=HEARTBEAT)
        if th.is_alive():
            yield ("heartbeat", None)
    if "e" in box:
        return ("error", box["e"])
    return ("ok", box.get("v"))


# THE TEMPLATE'S OWN MARKERS (#12, docs/SELF-IMPROVEMENT-LOG.md). The served
# chat template (mcp/fixtures/bonsai_chat_template.jinja) builds every turn
# out of these; text that carries one, rendered back into a prompt, opens or
# closes a block the template did not (a `</think>` inside reasoning_content
# ends the think block early and leaves the template's own `</think>` dangling).
#
# WHO WROTE IT decides what happens (operator, 2026-09-24):
#   OUR path     -- anything the proxy moves into a prompt or into content:
#                   the hand-off it prefills as reasoning, the hand-back it
#                   prefills after an answer, a seed word, the text it
#                   injects (skills, library source, the work log) -- is
#                   SCRUBBED, and counted. A client's echo of the reasoning
#                   channel is NOT ours: it passes through as sent and is
#                   only counted (ledger_restore, markers_in_echo).
#   the MODEL    -- markers in content the model generated are left as
#                   written and RECORDED (x_yamadori.template_markers, one
#                   log line): hiding them would hide a model defect.
# Evidence for the live case: the gate's echo run, step 5, delivered
# 'Done.\n</think>\n\nDone.\n</think>\n\nDone.'; that request reused 3964 of
# 3988 prompt tokens (it extended what the slot held, so the model saw the
# ledger's rendering), the ledger renders an echo client's session
# byte-identically to a stripping client's with one balanced think block per
# turn (mcp/test_ledger.py, echo test), and no proxy path writes these
# strings. The model wrote them.
TEMPLATE_MARKERS = ("<think>", "</think>", "<|im_start|>", "<|im_end|>",
                    "<tool_call>", "</tool_call>")


def template_markers(text: str | None) -> dict:
    """{marker: count} of the template's markers in `text` ({} when none)."""
    if not isinstance(text, str) or "<" not in text:
        return {}
    return {m: text.count(m) for m in TEMPLATE_MARKERS if m in text}


def scrub_markers(text: str | None) -> tuple[str, int]:
    """(`text` without the template's markers, how many were removed). For
    OUR path only -- see TEMPLATE_MARKERS."""
    if not isinstance(text, str) or "<" not in text:
        return (text or "") if isinstance(text, str) else "", 0
    n = 0
    for m in TEMPLATE_MARKERS:
        k = text.count(m)
        if k:
            n += k
            text = text.replace(m, "")
    return text, n


def _note_scrub(payload: dict, where: str, n: int) -> None:
    if n:
        payload.setdefault("_markers_scrubbed", {})
        payload["_markers_scrubbed"][where] = \
            payload["_markers_scrubbed"].get(where, 0) + n
        print(f"  template markers: {n} scrubbed from {where} (our path)",
              flush=True)


def _markers_record(payload: dict, delivered: str, upstream: str) -> dict | None:
    """x_yamadori.template_markers: the markers in the delivered content, and
    whose they are -- `model` when the upstream generations (main's hops,
    its continuation, a second-brain winner delivered in its place) carried
    at least as many, else `ours` (a defect in this file: our path should
    have scrubbed them). Plus what our path scrubbed. None when clean."""
    found = template_markers(delivered)
    scrubbed = dict(payload.get("_markers_scrubbed") or {})
    if not found and not scrubbed:
        return None
    up = template_markers(upstream)
    model = {m: min(k, up.get(m, 0)) for m, k in found.items()
             if up.get(m, 0)}
    ours = {m: k - model.get(m, 0) for m, k in found.items()
            if k - model.get(m, 0) > 0}
    rec = {"in_content": found, "model": model, "ours": ours,
           "scrubbed": scrubbed,
           "source": ("ours" if ours else "model" if model else None)}
    if found:
        print(f"  template markers in the delivered content: {found} -- "
              f"written by {'the MODEL' if not ours else 'OUR PATH (defect)'}"
              f"; left as written, recorded", flush=True)
    return rec


class _Out:
    """What a turn has presented, and CHANNEL ORDER (2026-09-24, live SSE
    diagnostic): a client closes its thinking block at the first `content`
    delta, so every byte of content must come after all reasoning. Once any
    content has gone out, later reasoning is not forwarded."""

    def __init__(self, streamed: bool, shown_prefix: str = ""):
        self.streamed = streamed
        self.content_sent = False
        # Every content byte the client was sent, in order: what a streaming
        # client STORES as this turn's content, and so the text the ledger
        # must key the turn by (_run_turn, #10 in docs/SELF-IMPROVEMENT-LOG).
        self.shown: list[str] = [shown_prefix] if shown_prefix else []
        # Every reasoning byte the client was sent: what an ECHOING client
        # sends back as this turn's reasoning (pass-through, 2026-09-24).
        self.reasoning_shown: list[str] = []

    def reasoning(self, text: str):
        if self.streamed and text and not self.content_sent:
            self.reasoning_shown.append(text)
            yield ("reasoning", text)

    def content(self, text: str):
        if self.streamed and text:
            self.content_sent = True
            self.shown.append(text)
            yield ("content", text)


class TurnRefused(RuntimeError):
    """A turn that cannot start, as a STRUCTURED error (pre-deploy review,
    2026-09-24): `yamadori-vision`'s gpu_room.NoRoom escaped as a bare 502
    on the blocking path and a broken stream on the streamed one. It carries
    the situation, whether retrying helps (as a fact) and remedies with an
    owner; server.py answers it with its status and `body()`, and
    stream_body with an SSE error event."""

    def __init__(self, status: int, code: str, reason: str, retryable: bool,
                 remedies: list, facts: dict | None = None):
        super().__init__(reason)
        self.status, self.code, self.reason = status, code, reason
        self.retryable, self.remedies = retryable, remedies
        self.facts = dict(facts or {})

    @classmethod
    def of_no_room(cls, e: "gpu_room.NoRoom") -> "TurnRefused":
        return cls(429 if e.retryable else 507, e.code,
                   e.reason + " Nothing was generated.", e.retryable,
                   e.remedies, e.facts)

    def body(self) -> dict:
        kind = "rate_limit_error" if self.status == 429 else "api_error"
        return {"error": {"message": self.reason, "type": kind,
                          "code": self.code, "param": None,
                          "retryable": self.retryable,
                          "remedies": self.remedies, **self.facts}}


def _run_turn(body: dict, streamed: bool):
    messages = body.get("messages") or []
    root, trusted, how = resolve_repo(messages, body.get("_client_ip", ""))
    db = repos.db_path(root) if root else None
    # Decided once, here, and handed to prepare(): a utility call has no
    # session in either place (session_context).
    body = dict(body, _utility=utility_of(body, strip_thinking(messages)))
    _key, state = session_context(messages, body.get("_account") or "",
                                   body.get("_session_token") or "",
                                   utility=body["_utility"])
    # x_yamadori.gpu_room: every A4000 decision THIS request causes
    # (mcp/gpu_room.py) -- prepare's own embeddings (skills, library help),
    # then every tool run_our_tool executes. Fresh per request.
    room_log: list = []
    # FAIL FAST (pre-deploy review, 2026-09-24): prepare's own embeddings
    # (skill / hint selection) never wait for the A4000's room lock, which
    # an image draw holds for its whole run (up to ROOM_WAIT_S, 300 s). A
    # loaded search model still takes its lease; one that is not loaded is
    # skipped at once, and the skip is recorded (x_yamadori.skills /
    # x_yamadori.gpu_room).
    with gpu_room.recording(room_log), gpu_room.fail_fast(
            "a chat turn's own embedding (skill / hint selection) does not "
            "wait for the A4000"):
        payload = prepare(body)
    state["_gpu_room"] = room_log
    payload["_gpu_room"] = room_log
    # A client that names the vision copy itself (`yamadori-vision`) loads
    # it with this turn's generation, which cannot be wrapped from here: room
    # is made once, now, and no lock is held through the turn (the KNOWN GAP
    # in mcp/gpu_room.py). A turn that cannot fit fails before it starts.
    if gpu_room.on_card(payload.get("model")):
        with gpu_room.recording(room_log):
            try:
                gpu_room.ensure_room(payload["model"], upstream=UPSTREAM)
            except gpu_room.NoRoom as e:
                raise TurnRefused.of_no_room(e) from None
    # THE WARM RACE (see _WARM_PENDING): this conversation's own pending warm
    # finishes before anything of this request reaches its slot.
    waited = wait_for_warm((payload.get("_slot") or {}).get("key"))
    if waited is not None:
        payload["_warm_waited"] = waited
    ours = set(payload.get("_ours") or [])
    # generate_image links its result on the address the client used, and
    # picks the image model from this caller's account; each call is
    # recorded for x_yamadori.
    state["_public_base"] = body.get("_public_base") or ""
    state["_account"] = body.get("_account") or ""
    payload["_images"] = state.setdefault("_images", [])
    # describe_image reads this request's attached images from the session
    # state, not the payload: fan-out deep-copies payloads through JSON.
    state["_attached"] = payload.pop("_attached", None) or vision.empty()
    payload["_attachments"] = vision.summary(state["_attached"])
    payload["_vision"] = state.setdefault("_vision", [])
    payload["_tool_calls"] = []
    payload.setdefault("_fold_back", [])
    rep = _repair_state(payload)
    tcheck = tool_code.state(bool(payload.get("_tool_code")),
                             fix=bool(payload.get("_fixup")))
    # The question as the client sent it, for the checks: `convo` can be the
    # very same list object, and the loop appends to it.
    question = list(messages)
    turn = corpus.new_turn()
    t_start = time.time()
    payload["_t_start"] = t_start        # x_yamadori.energy's wall clock
    corpus.log_turn(turn, root, messages,
                    [t.get("function", {}).get("name")
                     for t in (payload.get("tools") or [])],
                    is_first_turn(messages),
                    utility=bool(body["_utility"].get("utility")),
                    route=payload.get("_route"),
                    account=body.get("_account") or "",
                    client_tools=client_tool_names(body))
    tracker = repeats.Turn()
    out = _Out(streamed, body.get("_shown_prefix") or "")

    # DEEP THINKING BEFORE MAIN, by a trigger (mcp/deep.py: struggle, a task
    # kickoff, a known-hard area) or a header. Its searches go out as
    # reasoning while it runs; its hand-off becomes this turn's prefill. The
    # model's own trigger, think_deeply, runs inside the loop below.
    gen = _deep_thinking(payload, messages, db, root, state)
    think = None
    while True:
        try:
            line = next(gen)
        except StopIteration as stop:
            think = stop.value
            break
        yield from out.reasoning(line + "\n")
    payload["_think_pre"] = think
    if think and think.get("skipped"):
        lg = payload.get("_ledger") or {}
        # The trigger fired and nothing ran: deferred, with its own cooldown
        # (deep.DEFER_REQUESTS), and this episode's one wait is spent.
        deep.mark_deferred(lg.get("account") or "", lg.get("session") or "",
                           (payload.get("_deep") or {}).get("kind"))
    if think and think.get("ran"):
        trig = payload.get("_deep") or {}
        lg = payload.get("_ledger") or {}
        # The struggle episode ends here, the cooldown starts, and an area or
        # a kickoff is marked done (deep.mark_ran).
        deep.mark_ran(lg.get("account") or "", lg.get("session") or "",
                      len(messages), trig.get("kind") or "forced",
                      packages=trig.get("packages"), skills=trig.get("skills"),
                      turn_key=lg.get("turn_key"))

    # THE MAIN LOOP. It runs more than once only for OUR tools on main --
    # generate_image / describe_image, and the delegate benchmark arm. It
    # ends when the model stops calling them, at context_full, or at the
    # tool-turn cap (tiers.tool_turn_limit), which LAND: tools withdrawn,
    # the answer asked for. ONE upstream generation per answer: every hop
    # is streamed, and the hop that makes no call of ours IS the answer.
    convo = payload["messages"]
    # The turn's own tools, before a landing withdraws them: what the next
    # request renders with, and so what the compaction store keeps.
    tools0 = list(payload.get("tools") or [])
    prefill = payload.pop("_prefill", None)
    spent = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    n_calls = 0
    hops_added: list[dict] = []
    # Every generation's content as the model server returned it: what the
    # model WROTE, for attributing template markers (#12).
    upstream_text: list[str] = []
    cap = payload["_turn_cap"] = _turn_cap(payload)
    d: dict = {}
    send: dict = payload
    landed = False
    pending = ""
    hop = -1
    while True:
        hop += 1
        pending = ""
        payload = tiers.rebudget(dict(payload, messages=convo), role="main")
        last = context_full(payload, convo)
        if not last and cap["turns"] >= cap["limit"]:
            cap["hit"] = True
            payload = _land(payload, convo, "tool_turn_cap")
            last = landed = True
        elif last:
            payload = _land(payload, convo)
            landed = bool(convo) and convo[-1].get("content") == \
                LANDING_PROMPT
        send = payload
        # A prefill opens this hop: deep thinking's hand-off on hop 0, or the
        # fold-back after a think_deeply result on the hop that follows it.
        if prefill is not None:
            send = dict(payload, messages=list(convo) + [prefill])
        d = {}
        # CHANNEL ORDER within a hop: content is held until it passes
        # HOLD_CONTENT_CHARS (it is the answer: stream it live) or the hop
        # ends (a hop that calls one of OUR tools sends its held preface as
        # reasoning; the final hop flushes it as content).
        held = ""
        live = False
        try:
            for kind, item in _post_events("/v1/chat/completions", send):
                if kind == "done":
                    d = item
                    continue
                if item.get("reasoning_content"):
                    yield from out.reasoning(item["reasoning_content"])
                piece = item.get("content") or ""
                if not piece:
                    continue
                if live:
                    yield from out.content(piece)
                    continue
                held += piece
                if len(held) > HOLD_CONTENT_CHARS:
                    live = True
                    yield from out.content(held)
                    held = ""
        except Exception as e:                                   # noqa: BLE001
            if not streamed:
                raise
            yield ("content", f"\n[upstream error: {type(e).__name__}: {e}]")
            corpus.log_answer(turn, root, "", hop,
                              (time.time() - t_start) * 1000)
            return {"_error": True}
        n_calls += 1
        u = d.get("usage") or {}
        for k in spent:
            spent[k] += int(u.get(k) or 0)
        d["usage"] = dict(spent, hops=n_calls)
        msg = d["choices"][0]["message"]
        upstream_text.append(msg.get("content") or "")
        calls = [c for c in (msg.get("tool_calls") or [])
                 if c.get("function", {}).get("name") in ours]
        if held:
            if calls and not last:
                yield from out.reasoning(held)
            elif tcheck is not None and any(
                    c.get("function", {}).get("name") not in ours
                    for c in (msg.get("tool_calls") or [])):
                # A preface to CLIENT calls waits for the code check: the
                # fix-up's reasoning line must go out before any content.
                pending = held
            else:
                yield from out.content(held)
            held = ""
        # The landing is the last generation whatever it asked for: a call
        # made there is not run (nothing would read its result).
        if not calls or last:
            break
        # The hop's reasoning stays with it (the template renders it back),
        # and the ledger records the hops for the next request.
        hop_msg = {"role": "assistant", "content": msg.get("content") or "",
                   "reasoning_content": msg.get("reasoning_content") or "",
                   "tool_calls": msg.get("tool_calls") or []}
        if prefill is not None:
            # THE HAND-OFF CARRIES INTO HOPS 1+ (pre-deploy review,
            # 2026-09-24). The prefill is sent on hop 0 only; hops 1+ see
            # this hop in its place. llama-server re-sends a prefill's
            # reasoning and content as its first deltas (STEP 0), so they
            # are normally here already -- where they are not, deep
            # thinking's hand-off was lost the moment an image tool ran in
            # the same request. Put back, exactly once.
            pr = prefill.get("reasoning_content") or ""
            if pr and not hop_msg["reasoning_content"].startswith(pr.strip()):
                hop_msg["reasoning_content"] = (
                    pr + "\n" + hop_msg["reasoning_content"]
                    if hop_msg["reasoning_content"] else pr)
            pc = prefill.get("content") or ""
            if pc and not hop_msg["content"].startswith(pc):
                hop_msg["content"] = pc + hop_msg["content"]
        convo.append(hop_msg)
        hops_added.append(hop_msg)
        next_prefill = None
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield from out.reasoning(f"`{streaming.describe_call(fn, args)}`\n")
            corpus.log_tool_call(turn, root, fn, args, hop)
            t0 = time.time()
            # An image takes a minute or more, and so can looking at one: a
            # thread with heartbeats. So can deep thinking (think_deeply).
            if fn == deep.TOOL_NAME:
                n_before = sum(1 for x in payload["_think_tool"]["calls"]
                               if x.get("ran"))
                status, res = yield from _in_thread(
                    lambda args=args: _think_deeply(payload, args, db, root,
                                                    state, len(messages)))
                ran_now = [x for x in payload["_think_tool"]["calls"]
                           if x.get("ran")]
                if status == "ok" and len(ran_now) > n_before:
                    # THE FOLD-BACK: the next hop opens with the seed line and
                    # "After thinking deeply," (prefilled; it ends on a comma,
                    # never a space), its reasoning a fixed line that points
                    # at the hand-off in the tool result.
                    seed = ran_now[-1].pop("_seed", None)
                    seeds = [seed] if seed else []
                    next_prefill = {"role": "assistant",
                                    "reasoning_content": deep.THINK_REASONING,
                                    "content": shomen.opening(seeds)}
                    payload.setdefault("_fold_back", []).append(
                        {"job": "investigate", "phrase": "investigate",
                         "into": "prefill", "trigger": "model",
                         "seeds": [x.get("word") for x in seeds]})
            else:
                status, res = yield from _in_thread(
                    lambda fn=fn, args=args: run_our_tool(fn, args, db, root,
                                                          tracker, state))
            res = res if status == "ok" and res else json.dumps(
                images.ImageError(
                    "TOOL_RAISED", f"the {fn} call died without a result",
                    retryable=False,
                    remedies=[{"fixable_by": "operator",
                               "action": "check the proxy log"}]).envelope(fn))
            corpus.log_tool_result(turn, root, fn, res,
                                   (time.time() - t0) * 1000)
            payload["_tool_calls"].append(_tool_evidence(fn, res))
            tmsg = {"role": "tool", "tool_call_id": c["id"],
                    "content": repeats.cap_tool_result(res, fn, args)}
            convo.append(tmsg)
            hops_added.append(tmsg)
        cap["turns"] += 1
        payload["messages"] = convo
        prefill = next_prefill

    # ---------------------------------------------------------------- finish
    choice = d["choices"][0]
    msg = choice["message"]
    fin = choice.get("finish_reason") or "stop"
    # What the slot holds after this turn: the prompt of the last generation
    # and what it generated. Anything the client is delivered beyond that is
    # warmed into the slot before the next request (_warm).
    slot_msg = {"content": msg.get("content") or "",
                "reasoning_content": msg.get("reasoning_content") or "",
                "tool_calls": json.loads(json.dumps(msg.get("tool_calls")
                                                    or []))}
    base_messages = list(send.get("messages") or [])
    if prefill is not None and base_messages and \
            base_messages[-1] is prefill:
        base_messages = base_messages[:-1]
    content = msg.get("content") or ""
    client_calls = [c for c in (msg.get("tool_calls") or [])
                    if c.get("function", {}).get("name") not in ours]
    stray = [c for c in (msg.get("tool_calls") or [])
             if c.get("function", {}).get("name") in ours]
    fan = None
    if stray:
        # Withheld (the client does not have them), so the finish must not
        # claim tool calls the client will never receive.
        fin = "stop" if fin == "tool_calls" else fin
        if not content.strip():
            t = ("[no answer: the tool loop reached its breaker and the "
                 "landing still asked for a tool. This is a defect report, "
                 "not the model's answer.]")
            yield from out.content(t)
            content += t
    if fin == "incomplete":
        took = (d.get("_transport") or {}).get("dropped_after")
        t = (f"\n\n[the connection to the model dropped"
             f"{f' after {took}s' if took is not None else ''}; "
             f"the answer above is the part that arrived]")
        yield from out.content(t)
        content += t
    # A `length` finish is a BUDGET EVENT, reported with the same words on
    # both paths, as content -- never the reasoning.
    notice = _budget_notice(fin, content, d.get("usage"))
    if notice:
        yield from out.content(notice)
        content = content.rstrip() + notice if content.strip() else notice
    blank = _empty_notice(fin, content, msg)
    if blank:
        print("  empty answer: finish=stop, no content, no tool call; the "
              "client is told so", flush=True)
        yield from out.content(blank)
        content = blank

    if client_calls:
        content = yield from _finish_calls(payload, tcheck, msg, client_calls,
                                           content, question, ours, out,
                                           pending)
    elif not notice and not blank and fin == "stop":
        content, fan, slot_msg = yield from _finish_text(
            payload, rep, msg, fin, content, question, send, slot_msg, out)
        # A continuation, and a second-brain winner delivered in place, are
        # model-written too.
        upstream_text.append(slot_msg.get("content") or "")
        if fan and isinstance(fan.get("_winner"), dict):
            upstream_text.append(fan["_winner"].get("content") or "")
    msg["content"] = content
    if client_calls:
        msg["tool_calls"] = client_calls
        fin = "tool_calls"
        yield ("calls", client_calls)
    elif stray:
        msg.pop("tool_calls", None)
    choice["finish_reason"] = fin
    # The reasoning the next request must render for this turn: what the
    # slot holds (a prefill's hand-off included).
    msg["reasoning_content"] = slot_msg.get("reasoning_content") or \
        msg.get("reasoning_content") or ""
    if fan is not None:
        d["_fanout"] = dict(fan, winner=(fan.get("_winner")))
        fan.pop("_winner", None)
    delivered = {"role": "assistant", "content": content,
                 "reasoning_content": msg["reasoning_content"]}
    if client_calls:
        delivered["tool_calls"] = client_calls
    # WHAT THE CLIENT STORES (#10, docs/SELF-IMPROVEMENT-LOG.md). A streaming
    # client keeps every content byte it was sent. When a hop that called one
    # of OUR tools had already streamed content (a prefill's opening phrase,
    # or a preface past HOLD_CONTENT_CHARS), that is more than `content` --
    # the last hop's answer -- and a ledger keyed by `content` alone never
    # matched the turn the client sent back: the hand-off reasoning and the
    # hidden hops were not restored, and the next request re-read the whole
    # turn (live gate 2026-09-24: reused 2513 of 3955 after a 7-generation
    # deep-thinking turn). The turn is keyed by what the client stores; the
    # content it renders is the delivered one (ledger_restore).
    stored = dict(delivered)
    if streamed:
        shown = "".join(out.shown)
        if shown.strip() != (content or "").strip():
            stored["content"] = shown
            payload["_stored_differs"] = True
    payload["_template_markers"] = _markers_record(
        payload, stored.get("content") or "", "".join(upstream_text))
    # THE TURN AS THE NEXT REQUEST WILL RENDER IT (reasoning pass-through,
    # 2026-09-24): its reasoning is whatever this client sends back -- the
    # echo of what it was shown if it echoes (seen on its past turns), else
    # nothing -- and the hidden hops before it come back with their
    # reasoning emptied (ledger_record_turn). The warm and the compaction
    # store use that form; the ledger records no reasoning at all.
    # Whether this client echoes: seen on this request's past assistant
    # turns; on a first turn (none to see), what the ACCOUNT's client did
    # last time -- echoing is a property of the harness, not the
    # conversation. Unknown: it strips (Hermes does, for this provider).
    acct = ((payload.get("_ledger") or {}).get("account")) or ""
    if any(isinstance(m, dict) and m.get("role") == "assistant"
           for m in messages):
        echoes = bool(((payload.get("_ledger") or {}).get("restored") or {})
                      .get("echoed_reasoning"))
        flag = "1" if echoes else "0"
        if nebari.ledger_get(acct, "client", "echoes_reasoning") != flag:
            nebari.ledger_put(acct, "", "client", "echoes_reasoning", flag)
    else:
        echoes = nebari.ledger_get(acct, "client", "echoes_reasoning") == "1"
    client_turn = {k: v for k, v in delivered.items()
                   if k != "reasoning_content"}
    if echoes:
        client_turn["reasoning_content"] = (
            "".join(out.reasoning_shown) if streamed
            else delivered["reasoning_content"])
    hop_ids = {id(h) for h in hops_added}
    warm_base = [dict(m, reasoning_content="") if id(m) in hop_ids
                 and m.get("role") == "assistant" else m
                 for m in base_messages]
    account, session = ledger_scope(payload)
    if session:
        ledger_record_turn(account, session, delivered, hops_added or None,
                           stored=stored,
                           prev=chain_keys(messages)[-1] if messages else "")
        # The compaction store keeps the prompt AS THE LEDGER RENDERS IT
        # (pre-deploy review, 2026-09-24): hidden hops with their reasoning
        # emptied and no landing request, with the turn's own tools. It held
        # the last generation's prompt as sent upstream -- hop reasoning and
        # LANDING_PROMPT included -- so a compaction's extension check never
        # matched and the splice/continue fallback REINJECTED that hidden
        # reasoning into the conversation.
        ledger_prompt = list(warm_base)
        if ledger_prompt and ledger_prompt[-1].get("role") == "user" and \
                ledger_prompt[-1].get("content") == LANDING_PROMPT:
            ledger_prompt.pop()
        compaction.update_response(account, session, client_turn,
                                   prompt=ledger_prompt, tools=tools0)
    # The previous warm's record first: the one scheduled below replaces it.
    payload["_warm_before"] = warm_before((payload.get("_slot") or {})
                                          .get("key"))
    if payload["_warm_before"] is not None and "_warm_waited" in payload:
        payload["_warm_before"] = dict(payload["_warm_before"],
                                       waited_s=payload["_warm_waited"])
    payload["_warm"] = _warm(payload, warm_base, client_turn, slot_msg,
                             landed)
    _log_turn(state, delivered, tcheck, think, fan, rep)
    # THE SELF-IMPROVEMENT LOOP (Phase 0.6): this conversation's earlier
    # decisions are observed against what the client sent back, then this
    # request's decision -- or non-decision -- is recorded.
    payload["_deep_record"] = _deep_record(payload, messages, think)
    for u in payload.pop("_extra_usage", None) or []:
        n_calls += 1
        for k in spent:
            spent[k] += int(u.get(k) or 0)
    d["x_yamadori"] = _x_yamadori(payload, hops=n_calls, fan=fan,
                                  think=think, repair=rep, tool_check=tcheck)
    recent_turns.note(d["x_yamadori"])
    corpus.log_answer(turn, root, content, hop,
                      (time.time() - t_start) * 1000,
                      finish=fin, tool_calls=len(client_calls))
    d["usage"] = dict(spent, hops=n_calls)
    return d


def _finish_calls(payload: dict, tcheck: dict | None, msg: dict,
                  calls: list[dict], content: str, question: list[dict],
                  ours: set[str], out: _Out, pending: str = ""):
    """CLIENT CALLS that write code (mcp/tool_code.py): checked before they
    leave; at `high` and up the second brain repairs what does not parse
    (the fixup job gets ONLY the code, its errors and the user's request);
    the note -- "Verified ...", "Repaired ...", or at `medium` "Checked
    ..." -- goes out as content just before the calls. Returns the content
    delivered."""
    rv = tool_code.check(tcheck, msg, "tool_calls", ours)
    todo = tool_code.fixable(rv) if (tcheck and tcheck.get("fix")) else []
    if todo:
        yield from out.reasoning(
            "`fixing " + ", ".join(sorted({str(t["path"]) for t in todo}))
            + " before the call is sent`\n")
        request, _ctx, _sp = selection.question_of(question)
        tier = payload.get("_tier") or {}
        seed = ledger_seed(payload, "fixup", request)
        status, job = yield from _in_thread(lambda: shomen.run(
            "fixup", units=todo, request=request, check=tool_code.recheck,
            tier=tier.get("name") or "max",
            effort=(tier.get("effort") if "effort" in
                    (tier.get("overridden") or []) else None),
            seed=seed))
        if status != "ok":
            print(f"  fixup failed: {job}", flush=True)
            job = {"ok": False, "units": [], "rounds": 0,
                   "error": str(job)[:200]}
        tool_code.apply(tcheck, rv, calls, job.get("units") or [], job)
    # The model's own preface, held back for the check (CHANNEL ORDER).
    yield from out.content(pending)
    tnote = tool_code.finish(tcheck, msg, ours)
    if tnote:
        if tcheck and tcheck.get("rounds"):
            payload["_fold_back"].append(
                {"job": "fixup", "phrase": "repaired", "into": "note",
                 "seeds": [(tcheck.get("seed") or {}).get("word")]
                 if tcheck.get("seed") else []})
        elif tnote.startswith(shomen.PHRASES["verified"]):
            payload["_fold_back"].append({"job": None, "phrase": "verified",
                                          "into": "note", "seeds": []})
        # CHANNEL ORDER: content, after every reasoning delta and before the
        # calls. A blank line after the model's own text, if any.
        t = ("\n\n" if content.strip() else "") + tnote
        yield from out.content(t)
        content += t
    return content


def _finish_text(payload: dict, rep: dict | None, msg: dict, fin: str,
                 content: str, question: list[dict], send: dict,
                 slot_msg: dict, out: _Out):
    """A final TEXT answer: the code check ("Verified" / "Repaired"), then
    fan-out ("Compared two approaches"). Returns (content, fan record,
    what the slot holds)."""
    ph = shomen.PHRASES
    if rep is not None:
        review = code_check.review_answer(content, question)
        rep["errors_before"] = rep["errors_after"] = review["count"]
        rep["blocks_checked"] = review["checked"]
        rep["check_mode"] = review.get("check_mode")
        langs = ", ".join(review.get("langs") or [])
        if not review["checked"]:
            rep["stopped"] = "no_code"
        elif not review["count"]:
            # VERIFIED costs no generation (operator decision 3): a note.
            rep["stopped"] = rep["into"] = "clean"
            t = f"\n\n{ph['verified']}: the {langs} code above parses."
            yield from out.content(t)
            content += t
            payload["_fold_back"].append({"job": None, "phrase": "verified",
                                          "into": "note", "seeds": []})
        else:
            content = yield from _repair_answer(payload, rep, review, content,
                                                question, out)
    fan = None
    note, fan, win = "", None, None
    status, res = yield from _in_thread(
        lambda: _fan_out(payload, dict(msg, content=content), fin))
    if status == "ok" and res:
        note, fan, win = res
    elif status != "ok":
        fan = {"n": 0, "error": type(res).__name__}
    if fan is not None and not fan.get("error") and not fan.get("skipped") \
            and (fan.get("steps") or 0) >= 2:
        seeds = [s for s in (fan.get("seeds") or []) if s]
        line = shomen.seed_line(seeds)
        lead = (line + " " if line else "") + ph["compared"]
        hb = fan.get("_handback")
        if fan.get("selection") in DELIVERED_SELECTIONS or (
                hb and hb.get("kind") == "code"):
            why = (fan.get("why") or "").rstrip(".")
            extra = (" " + hb["text"]) if hb and hb.get("kind") == "code" \
                else ""
            if _delivers_winner(fan, win):
                if not out.streamed:
                    # Nothing has gone out: the winner is delivered in place.
                    content = win["content"]
                    fan["replaced"] = True
                    t = f"\n\n{lead}: {why}. The code above is the one " \
                        f"selected.{extra}"
                else:
                    code = _code_of(win["content"])
                    fan["appended"] = True
                    t = (f"\n\n{lead}: {why}. The selected code:{extra}\n\n"
                         + code)
            else:
                t = f"\n\n{lead}: {why}.{extra}"
            yield from out.content(t)
            content += t
            payload["_fold_back"].append(
                {"job": "alternative" if (fan.get("steps") or 0) < 3
                 else "tiebreak", "phrase": "compared", "into": "note",
                 "seeds": seeds})
            if hb:
                fan["handback"].update(into="note", chars=len(t))
        elif hb and hb.get("kind") == "prose":
            # PROSE: B's differing points are PREFILLED after main's own
            # answer, in its own turn, and main continues -- it weighs them
            # itself. Ends on a letter (rule 4 of the operator's decisions).
            # B wrote the points; OUR path prefills them into main's turn.
            points, n_scrub = scrub_markers(hb["text"])
            _note_scrub(payload, "hand-back", n_scrub)
            fold = (f"\n\n{lead}: a second answer, written independently, "
                    f"makes these points that mine does not.\n\n"
                    f"{points}\n\nWeighing them")
            new, slot2 = yield from _continue(payload, send, slot_msg,
                                              content, fold, out)
            if new is not None:
                content, slot_msg = new, slot2
                fan["handback"].update(into="continuation", chars=len(fold))
                payload["_fold_back"].append(
                    {"job": "alternative", "phrase": "compared",
                     "into": "continuation", "seeds": seeds})
            else:
                t = fold[:-len("\n\nWeighing them")]
                yield from out.content(t)
                content += t
                fan["handback"].update(into="note", chars=len(t))
                payload["_fold_back"].append(
                    {"job": "alternative", "phrase": "compared",
                     "into": "note", "seeds": seeds})
        fan["_winner"] = ({"variant": win.get("variant"),
                           "seed": win.get("seed"),
                           "content": win.get("content")} if win else None)
    if note:
        yield from out.content(note)
        content += note
    return content, fan, slot_msg


def _repair_answer(payload: dict, rep: dict, review: dict, content: str,
                   question: list[dict], out: _Out):
    """REPAIRED (operator decision 3): the flagged blocks go to the second
    brain's fixup job with the user's request; non-streamed, the corrected
    code replaces the broken code IN PLACE; streamed, it follows the answer
    (which has gone out). Either way the note opens "Repaired", after the
    seed line. Returns the content delivered."""
    ph = shomen.PHRASES
    request, _ctx, _sp = selection.question_of(question)
    pcode = code_check.prompt_code(question)
    units = [{"path": f"code block {f['index']}", "language": f["lang"],
              "kind": "block", "code": f["code"], "original": f["code"],
              "errors": f["errors"], "index": f["index"]}
             for f in review.get("flagged") or []]
    tier = payload.get("_tier") or {}
    seed = ledger_seed(payload, "fixup", request)
    yield from out.reasoning("`repairing the answer's code`\n")
    status, job = yield from _in_thread(lambda: shomen.run(
        "fixup", units=units, request=request,
        check=lambda u, code: code_check.block_problems(code, u["language"],
                                                        pcode),
        tier=tier.get("name") or "max",
        effort=(tier.get("effort") if "effort" in
                (tier.get("overridden") or []) else None),
        seed=seed))
    if status != "ok":
        job = {"ok": False, "units": [], "rounds": 0}
    rep["rounds"] = int(job.get("rounds") or 0)
    fixed = [u for u in job.get("units") or [] if u.get("changed")
             and not u.get("errors_after")]
    left = sum(int(u.get("errors_after") or 0) for u in job.get("units") or [])
    rep["errors_after"] = left if job.get("units") else review["count"]
    langs = ", ".join(sorted({u["language"] for u in units}))
    # A repair note is mechanical: no seed line (the seed stays in the fix-up
    # job's own user message; coordinator, 2026-09-24).
    lead = ""
    if not fixed:
        rep["stopped"] = rep["into"] = "not_fixed"
        t = (f"\n\n{ph['checked']}: the {langs} code above does not parse, "
             f"and the repair did not fix it.")
        yield from out.content(t)
        return content + t
    what = (f"{len(fixed)} code block{'s' if len(fixed) != 1 else ''} that "
            f"did not parse, fixed in {rep['rounds']} "
            f"round{'s' if rep['rounds'] != 1 else ''}")
    payload["_fold_back"].append(
        {"job": "fixup", "phrase": "repaired", "seeds":
         [seed.get("word")] if seed else [],
         "into": "appended" if out.streamed else "in_place"})
    rep["stopped"] = "fixed"
    if not out.streamed:
        for u in fixed:
            content = content.replace(u["original"].rstrip("\n"),
                                      u["code"].rstrip("\n"), 1)
        rep["into"] = "in_place"
        return content + f"\n\n{lead}{ph['repaired']}: {what}; the code " \
                         f"above is the corrected version."
    rep["into"] = "appended"
    blocks = "\n\n".join(shomen._fence(u["code"], u["language"])
                         for u in fixed)
    t = f"\n\n{lead}{ph['repaired']}: {what}. The corrected code:\n\n{blocks}"
    yield from out.content(t)
    return content + t


def _continue(payload: dict, send: dict, slot_msg: dict, content: str,
              fold: str, out: _Out):
    """Continue main's own answer after a PREFILLED fold-back: the last
    generation's prompt, then an assistant message whose content is the
    answer so far plus `fold` (reasoning as generated). The slot holds the
    prompt and the answer, so only `fold` and the continuation are new.
    Returns (content, what the slot now holds), or (None, None) when the
    continuation failed and the caller should write `fold` as a note."""
    pre = {"role": "assistant", "content": content + fold,
           "reasoning_content": slot_msg.get("reasoning_content") or ""}
    body = tiers.rebudget(dict(send, messages=list(send.get("messages")
                                                   or []) + [pre]),
                          role="main")
    yield from out.content(fold)
    got = ""
    d = {}
    try:
        for kind, item in _post_events("/v1/chat/completions", body):
            if kind == "done":
                d = item
                continue
            piece = item.get("content") or ""
            if not piece:
                continue
            got += piece
            # llama-server re-sends the prefilled content as the first
            # delta(s) (STEP 0): forward only what lies beyond it.
            if len(got) > len(pre["content"]):
                new = got[max(len(got) - len(piece), len(pre["content"])):]
                yield from out.content(new)
    except Exception as e:                                       # noqa: BLE001
        print(f"  fold-back continuation failed: {type(e).__name__}: {e}",
              flush=True)
        return None, None
    # Banked like every generation (usage accumulates across a turn's
    # generations; _run_turn adds it).
    payload.setdefault("_extra_usage", []).append(d.get("usage") or {})
    full = ((d.get("choices") or [{}])[0].get("message") or {}).get(
        "content") or got
    if not full.startswith(pre["content"]):
        full = pre["content"] + full
    return full, {"content": full,
                  "reasoning_content": pre["reasoning_content"],
                  "tool_calls": []}


# ============================================================== WARMING ====
#
# When the turn the client stores differs from what the slot generated -- a
# repaired call, a note, a fold-back written by the proxy, a notice -- the
# next request would diverge inside that turn and roll the slot back to a
# checkpoint. So, while the harness runs its tool (or the user reads), the
# proxy loads the turn AS THE CLIENT WILL SEND IT into the conversation's
# own slot: a zero-token /completion of the prompt the template renders for
# it, cut at the end of the assistant turn (STEP 0 (b): the next request
# processed 23 tokens, exactly its tail, against 93 without). A trailing
# assistant message on the chat endpoint cannot do this: prefill mode drops
# its tool calls. The render is llama-server's own /apply-template, rendered
# twice with two different placeholders for the next message; the common
# prefix, backed off to the last end-of-turn token, is the cut. The
# conversation's next request waits for the warm on the same slot
# (slots.acquire, `warm`).
WARM = os.environ.get("YAMADORI_WARM", "1") == "1"
# The served template's end-of-turn token (llama-server /props eos_token).
END_OF_TURN = os.environ.get("YAMADORI_END_OF_TURN", "<|im_end|>")


def _same_turn(a: dict, b: dict) -> bool:
    def calls(m):
        return [(c.get("id"), (c.get("function") or {}).get("name"),
                 _args_norm((c.get("function") or {}).get("arguments")))
                for c in (m.get("tool_calls") or [])]
    return ((a.get("content") or "") == (b.get("content") or "")
            and calls(a) == calls(b))


def _warm(payload: dict, base: list[dict], delivered: dict, slot_msg: dict,
          landed: bool) -> dict:
    """Schedule the warm; the record says whether and why."""
    sl = payload.get("_slot") or {}
    if not sl.get("key") or sl.get("transient"):
        return {"sent": False, "why": "not a conversation turn"}
    if _same_turn(delivered, slot_msg):
        return {"sent": False, "why": "the slot already holds the turn "
                                      "as delivered"}
    if landed:
        return {"sent": False, "why": "the turn landed; its prompt carries "
                                      "the landing request, which the "
                                      "client does not keep"}
    if not WARM:
        return {"sent": False, "why": "YAMADORI_WARM=0"}
    import threading as _threading
    last = (payload.get("_cache_log") or [{}])[-1] or {}
    # What the warm should reuse at the least: the last generation's whole
    # prompt, which that slot processed moments ago (the delivered turn
    # diverges only after it). A warm that reuses less says the slot's
    # content was replaced or its checkpoints were not there (#11); the
    # record reaches the client on the conversation's NEXT response
    # (x_yamadori.warm_before), since this one has gone out before the warm
    # ends.
    rec = {"sent": True, "why": "the delivered turn differs from what the "
                                "slot generated", "state": "scheduled",
           "expect_reused_at_least": last.get("prompt"),
           "generated_on_slot": last.get("slot")}
    msgs = list(base) + [delivered]
    fields = {k: payload[k] for k in ("tools", "chat_template_kwargs",
                                      "enable_thinking", "reasoning_effort")
              if k in payload}
    model = payload.get("model") or "bonsai"
    # Registered BEFORE the thread starts, so a fast client's next request
    # already finds it pending (wait_for_warm) and its record (warm_before).
    ev = _WARM_PENDING[sl["key"]] = _threading.Event()
    _WARMS_DONE.pop(sl["key"], None)
    _WARMS_DONE[sl["key"]] = rec
    while len(_WARMS_DONE) > 512:     # conversations that never came back
        _WARMS_DONE.pop(next(iter(_WARMS_DONE)), None)
    _threading.Thread(target=_warm_now,
                      args=(sl["key"], model, msgs, fields, rec, ev),
                      daemon=True).start()
    return rec


def _upstream_json(path: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(f"{UPSTREAM}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def warm_prompt(model: str, msgs: list[dict], fields: dict,
                timeout: float = 120) -> str | None:
    """The rendered prompt up to the end of the last assistant turn, or None
    when the template cannot be asked."""
    last = msgs[-1]
    if last.get("tool_calls"):
        ph = [[{"role": "tool", "tool_call_id": c.get("id"), "content": x}
               for c in last["tool_calls"]] for x in ("\x01a", "\x02b")]
    else:
        ph = [[{"role": "user", "content": x}] for x in ("\x01a", "\x02b")]
    ra = _upstream_json(f"/upstream/{model}/apply-template",
                        dict(fields, messages=msgs + ph[0]),
                        timeout=timeout).get("prompt")
    rb = _upstream_json(f"/upstream/{model}/apply-template",
                        dict(fields, messages=msgs + ph[1]),
                        timeout=timeout).get("prompt")
    if not isinstance(ra, str) or not isinstance(rb, str):
        return None
    n = 0
    for x, y in zip(ra, rb):
        if x != y:
            break
        n += 1
    cut = ra.rfind(END_OF_TURN, 0, n)
    if cut < 0:
        return None
    cut += len(END_OF_TURN)
    if ra[cut:cut + 1] == "\n" and cut < n:
        cut += 1
    return ra[:cut]


_WARMS_DONE: dict[str, dict] = {}     # conversation key -> its last warm
# THE WARM RACE (#11, the V0 pilot on slot 1, 2026-09-24). The warm thread
# acquires the slot, asks /apply-template twice, and only then sends its
# /completion. llama-server defers a request pinned to a slot only while that
# slot is BUSY server-side, so a fast client's next request could reach the
# server first, be served, and then be overwritten by the warm's older, shorter
# prefix -- or meet a warm mid-load. So the conversation's next request waits,
# in this process, for its own pending warm to finish (WARM_WAIT, a CHOICE:
# a warm of a 70k prompt re-read from zero is ~20-40 s on this card), and
# x_yamadori.warm_before says how long it waited.
_WARM_PENDING: dict[str, "threading.Event"] = {}
WARM_WAIT = float(os.environ.get("YAMADORI_WARM_WAIT", "180"))


def wait_for_warm(key: str | None, timeout: float | None = None
                  ) -> float | None:
    """Block until this conversation's pending warm has finished. Returns the
    seconds waited, or None when no warm was pending.

    RE-CHECKS (pre-deploy review, 2026-09-24): when the warm waited on ends
    and ANOTHER is now pending for the key (a newer turn scheduled one), it
    waits for that one too, within the same budget. A warm itself never
    outlives WARM_WAIT (_warm_now bounds its own requests by it), so a
    waiter that returns on the budget has not left a warm running."""
    ev = _WARM_PENDING.get(key) if key else None
    if ev is None:
        return None
    t0 = time.time()
    end = t0 + (WARM_WAIT if timeout is None else timeout)
    while ev is not None:
        ev.wait(max(0.0, end - time.time()))
        nxt = _WARM_PENDING.get(key)
        if nxt is ev or nxt is None or time.time() >= end:
            break
        ev = nxt
    return round(time.time() - t0, 3)


def warm_before(key: str | None) -> dict | None:
    """The finished record of the warm that ran for this conversation since
    its last response (x_yamadori.warm_before), once."""
    return _WARMS_DONE.pop(key, None) if key else None


def _warm_now(key: str, model: str, msgs: list[dict], fields: dict,
              rec: dict, ev: "threading.Event | None" = None) -> None:
    # Every request this warm makes is bounded by WARM_WAIT, which a waiter
    # (wait_for_warm) waits at most: the warm can never still be running
    # when its waiter gives up (pre-deploy review, 2026-09-24; its render and
    # /completion timeouts were 120 + 120 + 600 s against a 180 s wait).
    end = time.time() + WARM_WAIT

    def left(cap: float) -> float:
        return max(1.0, min(cap, end - time.time()))
    grant = slots.acquire(key, warm=True)
    try:
        if grant.get("slot") is None:
            rec.update(state="skipped", why=grant.get("how"))
            return
        if rec.get("generated_on_slot") not in (None, grant["slot"]):
            # The pin moved between the generation and the warm: the warm
            # loads a slot that never held this prefix (a full prefill,
            # which the next request would pay anyway). Recorded.
            rec["moved_from"] = rec["generated_on_slot"]
        prompt = warm_prompt(model, msgs, fields, timeout=left(120))
        if not prompt:
            rec.update(state="skipped", why="the template render did not "
                                            "give a cut point")
            return
        r = _upstream_json(f"/upstream/{model}/completion",
                           {"prompt": prompt, "n_predict": 0,
                            "id_slot": grant["slot"], "cache_prompt": True},
                           timeout=left(600))
        t = r.get("timings") or {}
        token_ledger.record("warm", timings=t)
        rec.update(state="done", reused=t.get("cache_n"),
                   processed=t.get("prompt_n"), slot=grant["slot"])
        exp = rec.get("expect_reused_at_least")
        short = (exp is not None and t.get("cache_n") is not None
                 and int(t["cache_n"]) < int(exp))
        rec["short"] = bool(short)
        print(f"  warm: slot {grant['slot']} reused {t.get('cache_n')} "
              f"processed {t.get('prompt_n')}"
              + (f" -- SHORT: the slot generated on a {exp}-token prompt "
                 f"moments ago" if short else ""), flush=True)
    except Exception as e:                                       # noqa: BLE001
        rec.update(state="failed", error=f"{type(e).__name__}: {e}"[:200])
        print(f"  warm failed: {rec['error']}", flush=True)
    finally:
        slots.release(grant)
        # Only ITS OWN event (pre-deploy review, 2026-09-24): a newer warm
        # for the same key may have registered while this one ran, and
        # popping that would let its waiter through before it finished.
        if ev is None:
            ev = _WARM_PENDING.get(key)
        if ev is not None:
            if _WARM_PENDING.get(key) is ev:
                _WARM_PENDING.pop(key, None)
            ev.set()


# ============================================================ WORK LOG =====
#
# The model no longer has record_step / read_rings (operator, 2026-09-24):
# the proxy writes the work log (mcp/rings.py) from the turns it sees, under
# the conversation's lineage, and re-injects it on the first user turn after
# a compaction (prepare, _work_log_block). What is logged is a CHOICE: the
# client calls the model made, what the checks found and fixed, what deep
# thinking handed back, what fan-out decided.
def _log_turn(state: dict | None, msg: dict, tcheck: dict | None,
              think: dict | None, fan: dict | None,
              rep: dict | None) -> None:
    s = session_lineage(state)
    if not s:
        return
    try:
        import rings
        for c in (msg.get("tool_calls") or [])[:8]:
            fn = (c.get("function") or {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            target = next((str(args[k]) for k in ("path", "file_path",
                           "filePath", "command", "cmd", "query", "url")
                           if isinstance(args, dict) and args.get(k)), "")
            rings.record("did", f"{fn.get('name')} {target}"[:160], session=s)
        for f in (tcheck or {}).get("files") or []:
            if f.get("errors_before") or f.get("errors_after"):
                rings.record("check", f"{f['path']}: {f['errors_before']} "
                             f"problem(s) found, {f['errors_after']} left",
                             session=s)
        if rep and rep.get("errors_before"):
            rings.record("check", f"answer code: {rep['errors_before']} "
                         f"problem(s), {rep.get('stopped')}", session=s)
        if think and think.get("ran"):
            h = think.get("handoff") or {}
            rings.record("learned", f"deep thinking: {h.get('facts')} facts "
                         f"from {think.get('searches')} searches (handle "
                         f"{think.get('handle')})", session=s)
        if fan and fan.get("why"):
            rings.record("decided", f"fan-out: {fan['why']}"[:160], session=s)
    except Exception as e:                                       # noqa: BLE001
        print(f"  work log not written: {type(e).__name__}: {e}", flush=True)


def complete(body: dict) -> dict:
    """The blocking path: the one turn implementation, drained."""
    return _drain(_run_turn(body, streamed=False))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                            # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            # The PUBLIC catalogue, not whatever llama-swap happens to have
            # loaded. Proxying the real list advertised `bonsai`,
            # `critic-disabled`, `embeddings` and `reranker` -- a bypass, a
            # footgun and an implementation detail, all selectable by name.
            return self._send(200, catalog.public_list())
        # The HTML shell is public; every route that carries DATA is gated.
        # A browser cannot set an Authorization header on a page load, so
        # gating the page itself would make the dashboard unreachable from a
        # browser at all. The page holds no data -- it asks for the key and
        # sends it with each fetch.
        if self.path.startswith("/dash/api"):
            who, why = accounts.identify(self.headers.get("Authorization"))
            if who is None:
                return self._send(401, {"error": f"{why}."})
        hit = dashboard.handle_get(self.path)
        if hit:
            code, ctype, payload = hit
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.rstrip("/") == "/health":
            return self._send(200, {"ok": True, "repos": len(repos.known()),
                                    "corpus": corpus.stats()})
        self._send(404, {"error": "not found"})

    def do_POST(self):                                           # noqa: N802
        if self.path.startswith("/dash/"):
            who, why = accounts.identify(self.headers.get("Authorization"))
            if who is None:
                return self._send(401, {"error": f"{why}."})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            hit = dashboard.handle_post(self.path, body)
            if hit:
                code, ctype, payload = hit
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            return self._send(404, {"error": "not found"})

        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})

        # The API key a client already sends is the account. Nothing extra to
        # configure: every OpenAI client has the field and already fills it.
        account, why = accounts.identify(self.headers.get("Authorization"))
        if account is None:
            return self._send(401, {"error": {
                "message": f"{why}. Set an API key in your client.",
                "type": "invalid_request_error", "code": "invalid_api_key"}})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._send(400, {"error": f"bad request: {e}"})
        body["_account"] = account

        # Experiment overrides ride on a header so normal traffic and the
        # cached prompt are untouched. See tiers.from_header.
        feats = self.headers.get("X-Yamadori-Features")
        if feats:
            body["_features"] = feats
        # An explicit session (nebari.key_of); a malformed token is ignored.
        body["_session_token"] = nebari.session_token(
            self.headers.get("X-Yamadori-Session"))

        if body.get("stream"):
            return self._stream(body)

        t0 = time.time()
        try:
            d = complete(body)
        except TurnRefused as e:
            return self._send(e.status, e.body())
        except Exception as e:                                   # noqa: BLE001
            return self._send(502, {"error": f"{type(e).__name__}: {e}"})

        # Echo the name the caller used. Clients write this field into logs,
        # UIs and saved transcripts, so leaving the internal name in it leaks
        # the implementation one layer later and shows a user a model name
        # they cannot select.
        d = catalog.rewrite_response(d, body.get("model") or "yamadori")

        if PREAMBLE and is_first_turn(body.get("messages") or []):
            msgs = body.get("messages") or []
            root, trusted, how = resolve_repo(msgs, self.client_address[0])
            info = repos.ensure(root, from_trusted=trusted) if root else None
            note = preamble_for(info, available_checks(root) if root else [], how)
            try:
                m = d["choices"][0]["message"]
                m["content"] = note + (m.get("content") or "")
            except (KeyError, IndexError, TypeError):
                pass

        print(f"{self.path} {time.time() - t0:.1f}s", flush=True)
        self._send(200, d)


def main() -> None:
    # BINDS WIDE, AND THAT IS ONLY SAFE BECAUSE EVERY ROUTE AUTHENTICATES.
    #
    # An earlier build bound to every interface with NO authentication, which
    # was a straightforward hole: anyone reachable could run tools and read
    # the indexed source. The fix then was to retreat to loopback.
    #
    # Loopback is useless for the actual deployment -- this is a remote
    # service reached over ZeroTier, and a server nobody can reach serves
    # nobody. So the protection is the API key on every route, not the
    # interface, and that includes /dash, which writes to the corpus.
    #
    # Set YAMADORI_PROXY_HOST to a specific address to narrow it, e.g. the
    # ZeroTier address alone rather than every interface.
    HOST = os.environ.get("YAMADORI_PROXY_HOST", "0.0.0.0")

    # REFUSE TO START IF THE PORT IS TAKEN.
    #
    # ThreadingHTTPServer inherits allow_reuse_address = 1, which on Windows
    # lets a second process bind a port another process is already listening
    # on. It does not fail, it does not warn, and the OS then hands incoming
    # connections to one or the other nondeterministically.
    #
    # Every "restart" during development therefore added a listener instead of
    # replacing one. Two proxies ended up serving :1234 -- one of them nine
    # minutes stale, pointed at an upstream that had moved -- and a benchmark
    # running against that port got 502s from the old one and real answers
    # from the new one, at random. Results measured through a port with two
    # servers on it mean nothing, and nothing in the logs said so.
    #
    # Binding must fail loudly instead.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        srv = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        raise SystemExit(
            f"cannot bind {HOST}:{PORT} ({e.strerror or e}).\n"
            f"Something is already serving that port -- almost certainly an "
            f"older proxy.\nStop it first; two servers on one port answer "
            f"requests at random.")
    print(f"yamadori proxy on {HOST}:{PORT} -> {UPSTREAM}", flush=True)
    if HOST == "0.0.0.0":
        print("  reachable on every interface; every route requires an API key",
              flush=True)
    print("point any OpenAI client here; it needs no tool configuration.",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()


def stream_body(body: dict, public_name: str = "yamadori"):
    """The streamed path: the one turn implementation (`_run_turn`), its
    events as SSE bytes.

    Extracted from the request handler so it is not welded to one server. It
    YIELDS rather than writing to a socket, which is what lets the ASGI app
    drive it from a worker thread while the event loop stays free. A
    consumer that stops iterating raises GeneratorExit in here, which
    unwinds the turn and stops the work.

    What it promises, each asserted by mcp/test_stream.py against a fake
    upstream:

      - ONE upstream generation per answer: the hop with no tool call is
        streamed as it is generated and is never regenerated.
      - reasoning goes out live as `reasoning_content` deltas, never content,
        and never after the first content delta (CHANNEL ORDER).
      - the breaker lands exactly like `complete()`: tools withdrawn (`_land`).
      - a `length` finish ends with the same notice `complete()` writes.
      - the corpus gets the real hop count, and usage is summed over hops.
      - `x_yamadori` rides on the final chunk, the same object `complete()`
        puts on its response.
    """
    cid = streaming.new_id()
    model = public_name
    messages = body.get("messages") or []
    if PREAMBLE and is_first_turn(messages):
        root, trusted, how = resolve_repo(messages, body.get("_client_ip", ""))
        info = repos.ensure(root, from_trusted=trusted) if root else None
        note = preamble_for(info, available_checks(root) if root else [], how)
        if note:
            yield streaming.text_chunk(cid, model, note)
            # Part of the content the client stores for this turn (#10).
            body = dict(body, _shown_prefix=note)
    gen = _run_turn(body, streamed=True)
    while True:
        try:
            kind, item = next(gen)
        except StopIteration as stop:
            d = stop.value or {}
            break
        except TurnRefused as e:
            # The stream is already a 200: the refusal goes as an SSE error
            # event (OpenAI's shape, which its clients raise on), then DONE.
            yield b"data: " + json.dumps(e.body()).encode() + b"\n\n"
            yield streaming.DONE
            return
        if kind == "reasoning":
            yield streaming.reasoning_chunk(cid, model, item)
        elif kind == "content":
            yield streaming.text_chunk(cid, model, item)
        elif kind == "heartbeat":
            yield streaming.chunk(cid, model, {})
        elif kind == "calls":
            # The CLIENT's tools are the client's to execute. Forwarded
            # whole, in the streamed shape, which needs an index per call.
            yield streaming.chunk(cid, model, {"tool_calls": [
                dict(c, index=i) for i, c in enumerate(item)]})
    if d.get("_error"):
        yield streaming.DONE
        return
    fin = d["choices"][0].get("finish_reason") or "stop"
    yield streaming.chunk(cid, model, {}, finish=fin, usage=d.get("usage"),
                          extra={"x_yamadori": d.get("x_yamadori")})
    yield streaming.DONE


def log_message(self, *a):                                   # noqa: D102
    pass
