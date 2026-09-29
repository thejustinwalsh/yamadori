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
import re
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
import skill_prompts  # noqa: E402
import cancel  # noqa: E402
import max_mode  # noqa: E402
import session_id  # noqa: E402
import system_roles  # noqa: E402
import message_text  # noqa: E402
import image_input  # noqa: E402
import api_errors  # noqa: E402
import progress  # noqa: E402
import mcp_host  # noqa: E402

# The defaults ARE the configuration. llama-swap listens on 11434 and the
# proxy takes 1234, because 1234 is the port every OpenAI client is already
# pointed at -- putting the proxy there is what makes it need no setup. These
# two lines were stale for a whole session: the proxy listened on 1233 and
# called an upstream on 1234 that nothing was serving, so it only ever worked
# when launched with environment variables that were not written down.
UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
PORT = int(os.environ.get("YAMADORI_PROXY_PORT", "1234"))
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
| the user asks about a library's API, or you use a library newer than your training | reads that library's source | definitions after the user's message, or your answer opens "After thinking deeply," with the facts, each with file:line, in your thinking |
| your answer contains code | checks it, and may write an independent second answer | a line after your answer: "Verified ...", "Repaired ..." or "Compared two approaches ..." |
| it drew a concept seed for its work | names the word | "Today I was inspired by <word>." |

Text that opens with these phrases was written for you by that model:
continue from it, and cite what it cites."""

# WHERE MAIN HAS THE SERVER TOOLS (deep.think_tool_offered: xhigh and max,
# kept for the whole conversation, so the text is the same on every turn of a
# conversation). Two parts:
#   ADDENDUM_THINK_ROW -- the automatic row: the first task is planned.
#   ADDENDUM_TOOLS -- a table that NAMES the server tools and says when to call
#     each, at the END of the system text (the part nearest the conversation).
# Operator, 2026-09-27: "we need to steer them a bit ... the statement names
# them and helps the agent use them". Evidence: pagoda-h4 (n=1 run), 0 calls
# of any yama_* tool in 84 requests, while the model spent ~15 minutes writing
# scratch files and reading node_modules to learn how pmndrs `math` resolves
# -- the case yama_think_deeply exists for. Before this, the addendum only
# said what happens WHEN the model calls one. A decision table, no
# prohibition (AGENTS.md "Prompting this model"). The wording is UNMEASURED.
ADDENDUM_THINK_ROW = (
    "| the conversation's first task | plans the files, the order of the "
    "steps, the key decisions and the constraints | the plan as a "
    "yama_plan tool "
    "result; you carry it out with your own tools |\n")
ADDENDUM_TOOLS = """

Server tools: call these yourself when the work needs them. Each runs on the
Yamadori server, takes minutes, and leaves your workspace alone; you go on with
your own tools from what it returns.

| when you are | call | what you get back |
|---|---|---|
| about to start a new app, a multi-file feature, a migration or a rewrite; or reaching a large piece of the work the plan you have does not cover | yama_plan | the files, the order of the steps, the key decisions, the constraints the code holds |
| unsure of a library's API or version; reading a package's installed files, or writing scratch files, to find out how it works; on a fix that has failed twice | yama_think_deeply | facts from the library's own source, this service's knowledge base and the web, each with its source, and a next step; your answer then opens "After thinking deeply," |"""


def addendum_text(think: bool = False) -> str:
    """ADDENDUM; where main has the server tools, the first-task row before
    the seed row and the server-tools table at the end."""
    if not think:
        return ADDENDUM
    at = ADDENDUM.index("| it drew a concept seed")
    return ADDENDUM[:at] + ADDENDUM_THINK_ROW + ADDENDUM[at:] + ADDENDUM_TOOLS


def add_addendum(messages: list[dict], think: bool = False) -> list[dict]:
    """Append the addendum to the client's system message, or add one.

    At the end of the system text, never at the end of the conversation:
    this model's chat template raises "System message must be at the
    beginning" outright, so a trailing system message is a hard 500."""
    return add_system_tail(messages, addendum_text(think))


def add_system_tail(messages: list[dict], text: str) -> list[dict]:
    """`text` at the end of the client's system message (or a system
    message of its own when there is none): the addendum, and the craft
    index after it (skill_select PROGRESSIVE DISCLOSURE)."""
    out = list(messages)
    # The FIRST message only, and `developer` as well as `system` (Pi and any
    # OpenAI-SDK client with a reasoning model send `developer`; 2026-09-26,
    # docs/HARNESS-PI.md gap 1: a second system message in front of it was a
    # deterministic 502 at high and up). system_roles.one_system already
    # made the client's own a `system` message; this is the belt to that
    # brace.
    m = out[0] if out else None
    if isinstance(m, dict) and m.get("role") in INSTRUCTION_ROLES:
        c = m.get("content")
        if isinstance(c, str):
            out[0] = dict(m, role="system", content=c + text)
            return out
        if isinstance(c, list):
            # Text parts render back to back (the template's render_content),
            # so one more part is the same text as a string would be.
            out[0] = dict(m, role="system",
                          content=list(c) + [{"type": "text", "text": text}])
            return out
    # No system message: add one rather than prepending to the user's turn,
    # which would put our text in their words.
    return [{"role": "system", "content": text.strip()}] + out


# ONE SYSTEM MESSAGE (2026-09-26, docs/HARNESS-PI.md gap 1). OpenAI's
# `developer` role is the system message for reasoning models; Pi sends it
# whenever a model has `reasoning: true`. The served template knows only
# `system`, and only first ("System message must be at the beginning").
# mcp/system_roles.one_system -- the ONE helper both wires use (the
# Responses adapter too) -- makes the leading system/developer run one
# `system` message and a later one a user turn. The chat path applies it ON
# THE WAY IN (_run_turn, prepare), once, before anything reads the messages
# -- the addendum, skills, the utility rule, compaction, the session, the
# ledger's chain keys -- so every request of a conversation is mapped alike
# and the prefix is the same on every turn.
INSTRUCTION_ROLES = system_roles.SYSTEM_ROLES


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
    # `prompt`: a conversation key with no pin adopts the slot whose prompt
    # this one continues (slots ADOPTION, #38).
    grant = (slots.acquire(want.get("key"), bool(want.get("transient")),
                           prefix=fp if want.get("prefix") else None,
                           prompt=(fp if want.get("key") and not
                                   want.get("transient") else None),
                           inherits=want.get("inherits"))
             if isinstance(want, dict) else None)
    send = payload
    if grant and grant.get("slot") is not None:
        # id_slot, cache_prompt, and the slots' KV ranks (slots RANKS)
        send = dict(payload, **slots.upstream_fields(grant))
        # IDLE CLEAR (mcp/slots.py): this generation is about to run, so
        # other conversations' slots idle past IDLE_CLEAR_S go now. A no-op
        # when none qualifies.
        ic = payload.get("_idle_clear") or {}
        slots.clear_idle(
            grant, on=ic.get("on"), log=payload.get("_slots_cleared"),
            key=(want.get("key") if want.get("key") != slots.HELPER
                 else None), model=payload.get("model"),
            after_s=ic.get("override_s"),
            account=want.get("account") or None)
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
        rel = payload.get("_slot_release") or {}
        if grant and grant.get("mode") == "transient" \
                and isinstance(want, dict) and want.get("transient"):
            # A side call on the child slot. LAYOUT V2: that slot is THE
            # LANE, kept between the decider's turns and side calls (its
            # head stays cached; slots THE LANE) -- recorded, not released.
            # (slots.LANE_KEEP off, the offline suites' hook: released --
            # slots RELEASE.)
            if slots.lane_kept() and not want.get("prefix"):
                slots.lane_kept_note(grant["slot"], "side call ended",
                                     log=payload.get("_slots_released"))
            else:
                slots.release_idle(
                    grant["slot"], "side call ended"
                    if not want.get("prefix") else "compaction sent as is ended",
                    model=payload.get("model"), on=rel.get("on"),
                    log=payload.get("_slots_released"))
        elif grant and grant.get("mode") == "compaction":
            # LAYOUT V2: a compaction sent up as is ran on the least recently
            # used conversation slot (slots._compaction_slot). Its transcript
            # is reused by nothing (the continuation opens with the tools and
            # the summary) and the conversation pinned there lost its cache
            # to it, so the slot is emptied, pin or not.
            slots.release_idle(
                grant["slot"], "compaction sent as is ended",
                model=payload.get("model"), on=rel.get("on"),
                log=payload.get("_slots_released"), pinned_ok=True)


def _http_error_detail(e) -> str:
    """The status and the first 300 characters of an upstream error body.

    llama-server says exactly what it rejected ("Field 'x': ...") and that
    text was being discarded: minimal-tier requests failed with HTTP 400 in
    under 60 ms on 2026-09-23 and the proxy logged only "dropped with nothing
    in hand", so the trigger could not be found from the log.

    The body is read ONCE and kept on the exception (`_yamadori_body`), so
    mcp/api_errors.py can hand the client the server's own error object."""
    raw = getattr(e, "_yamadori_body", None)
    if raw is None:
        try:
            raw = e.read()[:8192]
        except Exception:                                        # noqa: BLE001
            raw = b""
        try:
            e._yamadori_body = raw
        except Exception:                                        # noqa: BLE001
            pass
    body = raw[:300].decode("utf-8", "replace")
    return f"HTTP {getattr(e, 'code', '?')}: {' '.join(body.split()) or '(no body)'}"


def _upstream_error_of(e) -> dict:
    """An upstream HTTPError's error object (after _http_error_detail)."""
    raw = getattr(e, "_yamadori_body", b"") or b""
    try:
        d = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return {"message": " ".join(raw.decode("utf-8", "replace").split())
                [:500]} if raw else {}
    if isinstance(d, dict) and isinstance(d.get("error"), dict):
        return d["error"]
    return d if isinstance(d, dict) else {"message": str(d)[:500]}


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
    # A request whose client went away starts nothing more (mcp/cancel.py).
    cancel.check()
    # THE IMAGE GUARD (tool_code, #46): main's generations only (_run_turn
    # sets it); a call whose IMAGE argument opens as image data is stopped
    # here, at its first characters, and handed back marked `_image_arg`.
    guard = payload.get("_image_guard")
    watch = tool_code.ImageArgWatch(payload.get("tools")) if guard else None
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
    beat_at = t0
    stopped = None                  # the image guard's hit, when it fired
    arg_deltas = 0                  # tool-call argument deltas read (~tokens)

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
                        beat_at = now
                    elif delta.get("tool_calls") and \
                            now - beat_at >= HEARTBEAT:
                        # A CALL BEING WRITTEN IS NOT SILENCE (#44). Its
                        # arguments are assembled here and go to the client
                        # whole, at the end -- which took 40 minutes when
                        # the model transcribed a screenshot's base64 into
                        # one (Octopus v0e-V0-xhigh-1): no byte reached
                        # Hermes, its 900 s stale detector killed the
                        # request, and the relay in between never learned
                        # the client had gone, so the abandoned turn kept
                        # its main lane. A "beat" is an empty delta the
                        # streamed turn forwards (_run_turn); every other
                        # reader skips it (it carries no text).
                        beat_at = now
                        yield "beat", {}
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
                            if watch is not None:
                                arg_deltas += 1
                                stopped = watch.feed(
                                    i, slot["function"]["name"],
                                    fn["arguments"])
                                if stopped is not None:
                                    break
                    if stopped is not None:
                        break
                if stopped is not None:
                    # THE IMAGE GUARD: stop reading. Leaving the `with`
                    # closes the upstream connection now, and llama-server
                    # cancels the generation when it sees it closed (the
                    # same path as a client hang-up, mcp/cancel.py) -- it
                    # does not run to completion.
                    break
    except Exception as e:                                       # noqa: BLE001
        if cancel.cancelled():
            # Its client went away and the socket was shut on purpose
            # (mcp/cancel.py): not a drop to retry or to report as one.
            raise cancel.Cancelled(cancel.current().why) from e
        if isinstance(e, urllib.error.HTTPError):
            # The server ANSWERED, with a refusal. Its body says what it
            # refused, and it is logged with the request's shape (never its
            # text). A 4xx is the same answer every time, so it is not
            # retried: it is raised with the server's words.
            detail = _http_error_detail(e)
            print(f"  upstream refused after {time.time() - t0:.2f}s: "
                  f"{detail}\n    request shape: {_request_shape(payload)}",
                  flush=True)
            # 501 (llama-server's not_supported_error) is the request's too.
            if 400 <= int(getattr(e, "code", 0) or 0) < 500 or \
                    int(getattr(e, "code", 0) or 0) == 501:
                raise streaming.UpstreamError(
                    f"the model server rejected the request ({detail})",
                    status=int(e.code), error=_upstream_error_of(e)) from e
        if isinstance(e, streaming.UpstreamError):
            # The model server SAID it failed (an HTTP refusal, or its
            # in-stream error event): not a drop to paper over with the
            # partial text. Raised with its error object, for the one error
            # path (mcp/api_errors.py) -- whatever had arrived.
            raise
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
                yield from _post_events_raw(
                    path, dict(payload, _image_guard=guard) if guard
                    else payload, timeout, retries - 1)
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
    if cancel.cancelled():
        # A shut socket can also read as a clean end of the stream.
        raise cancel.Cancelled(cancel.current().why)
    if stopped is not None:
        # The generation as far as it went, the stopped call's image value
        # replaced (valid JSON: it is rendered back into the next hop), and
        # no call after it. `_image_arg` tells _run_turn to hand it back as
        # NOT EXECUTED.
        idx = stopped["index"]
        c = calls[idx]
        stop = tool_code.image_stopped(c["function"]["name"],
                                       c["function"]["arguments"], stopped)
        c["function"]["arguments"] = stop.pop("arguments")
        for k in [k for k in calls if k > idx]:
            del calls[k]
        for k in calls:
            # Each call is answered by a tool result, which names its id.
            calls[k]["id"] = calls[k]["id"] or f"call_img_{int(t0)}_{k}"
        stop.update(index=idx, call_id=c["id"], deltas=arg_deltas,
                    seconds=round(time.time() - t0, 2))
        print(f"  image guard: stopped the {stop['tool']!r} call after "
              f"{arg_deltas} argument deltas ({stop['seconds']}s): its "
              f"{stop['argument']!r} argument opened as {stop['kind']}; the "
              f"upstream connection was closed", flush=True)
        out = assemble("tool_calls")
        out["_image_arg"] = stop
        yield "done", out
        return
    yield "done", assemble(finish or "stop")


# OUR TOOLS LIVE ON THE SECOND BRAIN, NOT ON MAIN (operator, 2026-09-24).
#
# Main -- the model the client talks to -- gets the client's tools untouched,
# plus only yama_generate_image and yama_describe_image where they are offered. The
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
# `yama_think_deeply` (deep.THINK_TOOL) at xhigh and max, the model-chosen trigger
# -- the one non-image tool of ours on main -- run by _think_deeply as a
# hidden hop. The other three triggers run before main (mcp/deep.py).
# `delegate_investigation` is KEPT only as a header-forced benchmark arm
# (docs/SELECTION-BUILD.md step 8; PROTOCOL rule 9):
# `X-Yamadori-Features: {"delegate": true}` or YAMADORI_DELEGATE_TOOL=1 puts
# it on main, and the main loop runs it through the one second-brain runner
# (shomen.run).
DELEGATE_TOOL = os.environ.get("YAMADORI_DELEGATE_TOOL", "0") == "1"


def image_tools(main: bool = False) -> list[dict]:
    """`yama_generate_image` and `yama_describe_image`, only where an image server is
    configured. `main`: main's copy of yama_generate_image (images.MAIN_TOOL),
    whose description says the proxy shows the picture itself (IMAGES REACH
    THE CHAT, _run_turn); deep thinking's copy asks for the markdown line in
    its hand-off.

    Gated on YAMADORI_IMAGEGEN_URL so no request pays prompt tokens for a tool
    that cannot run. Read per request, so the gate follows the environment.
    `yama_describe_image` (mcp/vision.py) goes wherever `yama_generate_image` goes, so
    the model can look at what it drew; YAMADORI_VISION=0 withholds it. A
    request carrying an attached image gets it even without an image server
    (prepare, `vision_tools`).
    """
    if not images.configured():
        return []
    return ([images.MAIN_TOOL if main else images.TOOL]
            + ([vision.TOOL] if vision.enabled() else []))


def vision_tools(att: dict | None) -> list[dict]:
    """`yama_describe_image` for a request that carries something to look at: a
    readable attached image, or a signed link to one this server made."""
    if not vision.enabled() or not att:
        return []
    readable = any(not e.get("error") for e in (att.get("images") or {}).values())
    return [vision.TOOL] if readable or att.get("media") else []


# NO CONFLICTS WITH THE HARNESS'S TOOLS (operator, 2026-09-27: "Make sure
# proxy tools do not conflict with harness tools"). Before a tool of ours
# goes on main it is compared with the client's list; it is WITHHELD on
#   - the same name, normalised (case, `-`/`_`, a plural s), or
#   - a DECLARED OVERLAP: a client tool that answers the same question.
# Each row names the harness tool and why (from bench/harness_shapes and the
# harnesses' own sources); rows decided NOT to overlap are listed too, so
# the decision is visible. The withheld set is kept for the conversation
# with the craft offer (skill_select's state), like yama_think_deeply's offer:
# the tool list never changes mid-conversation.
TOOL_OVERLAPS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    # Since the `yama_*` rename (2026-09-27) no harness tool can share one
    # of our names; a client tool under our OLD name (describe_image,
    # generate_image) is the same question, so it is a declared overlap.
    (vision.TOOL_NAME, ("vision_analyze", vision.LEGACY_TOOL_NAME),
     "Hermes' vision_analyze answers the same question (what is in this "
     "image) with its own vision call, and reads our signed /media links "
     "(AGENTS.md yama_describe_image); two tools for one question split the "
     "model's choice"),
    (images.TOOL_NAME, ("image_generation", "create_image", "image_gen",
                        "text_to_image", "generate_images",
                        images.LEGACY_TOOL_NAME),
     "the client generates images itself (Codex's hosted image_generation)"),
    ("delegate_investigation", ("task", "multi_agent_v1", "spawn_agent",
                                "subagent"),
     "the client delegates to its own sub-agent (OpenCode task, Codex "
     "multi_agent_v1): one delegation tool, the harness's"),
)
# Decided NOT to overlap (kept on main):
#   yama_describe_image vs Codex view_image -- view_image ATTACHES a local image
#     to the conversation; this text-only model sees it only through
#     yama_describe_image (proxy.prepare's placeholder names its id).
#   yama_recall_craft vs Hermes skills_list / skill_view / skill_manage and
#     OpenCode `skill` -- those read the HARNESS's skills; ours are this
#     service's craft, named apart so neither is mistaken for the other.
#   yama_think_deeply vs OpenCode task / Codex multi_agent_v1 -- a sub-agent
#     works in the user's project with the harness's tools; yama_think_deeply
#     researches library source, the craft library and the web.


def _norm_tool(name: str) -> str:
    n = re.sub(r"[-\s]+", "_", str(name or "").strip().lower())
    return n[:-1] if n.endswith("s") and len(n) > 3 else n


def tool_conflicts(ours: list[dict], client_tools: list | None
                   ) -> tuple[list[dict], list[dict]]:
    """(the tools of ours that may go on main, [{ours, because,
    client_tool}] withheld)."""
    names = [((t.get("function") or {}).get("name") or t.get("name") or "")
             for t in client_tools or [] if isinstance(t, dict)]
    norm = {_norm_tool(n): n for n in names if n}
    keep, withheld = [], []
    for t in ours:
        n = t["function"]["name"]
        hit = norm.get(_norm_tool(n))
        if hit:
            withheld.append({"ours": n, "because": "same name",
                             "client_tool": hit})
            continue
        # The MCP-backed tools' rows come from their server's configuration
        # (mcp_config: each tool's `overlaps`, e.g. a client that hosts
        # PackageLens itself).
        row = next(((cl, why) for o, cl, why in
                    (*TOOL_OVERLAPS, *mcp_host.overlap_rows())
                    if o == n and set(cl) & set(names)), None)
        if row:
            withheld.append({"ours": n, "because": row[1],
                             "client_tool": sorted(set(row[0])
                                                   & set(names))[0]})
            continue
        keep.append(t)
    return keep, withheld


def main_tools(client_tools: list | None, att: dict | None = None,
               delegate: bool = False, images_on: bool = True,
               think: bool = False, craft: bool = False,
               withheld: list | None = None,
               keep_withheld: list | None = None,
               mcp: list | None = None) -> tuple[list, set]:
    """(the tool list main is sent, the names of ours in it).

    The client's own list first and untouched, so its rendering -- the
    template prints tools first in the system block -- never depends on what
    we add. Ours after it, WITHHELD on a conflict with the client's
    (tool_conflicts: the same normalised name, or a declared overlap); the
    client's version is the one with side effects the client can handle.
    `think`: yama_think_deeply (Phase 0.6, deep.THINK_TOOL), where
    deep.think_tool_offered says so. `craft`: yama_recall_craft
    (skill_select.READ_TOOL), where the conversation's craft offer says so.
    `withheld` collects what was withheld; `keep_withheld` names tools of
    ours withheld earlier in the conversation, kept out. `mcp`: the
    MCP-backed tools the conversation was offered (mcp_host; the offer is
    decided on its first request and kept by name, _mcp_offer)."""
    import skill_select
    client_tools = list(client_tools or [])
    taken = {t.get("function", {}).get("name") for t in client_tools
             if isinstance(t, dict)}
    cand = ((image_tools(main=True) if images_on else []) + vision_tools(att)
            + ([deep.THINK_TOOL, deep.PLAN_TOOL] if think else [])
            + ([skill_select.READ_TOOL] if craft else [])
            + list(mcp or [])
            + ([shomen.TOOL] if delegate else []))
    ok, held = tool_conflicts(cand, client_tools)
    # A name stored before the `yama_*` rename is read as its new name.
    for n in [canonical_tool_name(x) for x in keep_withheld or []]:
        if any(t["function"]["name"] == n for t in ok):
            ok = [t for t in ok if t["function"]["name"] != n]
            held.append({"ours": n, "because": "withheld earlier in this "
                         "conversation", "client_tool": None})
    if withheld is not None:
        withheld.extend(held)
    mine: list[dict] = []
    for t in ok:
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

    `yama_generate_image` IS included (operator, 2026-09-23): deep thinking makes
    mockups, designs and sketches while it works, and the image's markdown
    crosses back in the hand-off. `yama_describe_image` comes with it: Bonsai is
    text-only, and the vision copy (mcp/vision.py) is how it looks at what it
    drew and refines it. `att`, the request's attachment register, adds
    `yama_describe_image` when the user attached an image and no image server is
    configured.
    """
    tools = [{"type": "function",
              "function": {"name": t["name"], "description": t["description"],
                           "parameters": t["inputSchema"]}}
             for t in cs.ALL_TOOLS] + image_tools()
    # Phase 0.6: the knowledge base (the armed skills), the web
    # (mcp/research_tools.py).
    # Never yama_think_deeply: it would recurse.
    tools += list(research_tools.TOOLS)
    names = {t["function"]["name"] for t in tools}
    return tools + [t for t in vision_tools(att)
                    if t["function"]["name"] not in names]


# Every name this proxy executes itself: the second brain's tools, the image
# tools and the delegate arm.
OUR_NAMES = ({t["name"] for t in cs.ALL_TOOLS}
             | {"delegate_investigation", images.TOOL_NAME, vision.TOOL_NAME,
                deep.TOOL_NAME, deep.PLAN_TOOL_NAME,
                skill_prompts.CRAFT_TOOL_NAME}
             | set(research_tools.NAMES)
             # The MCP-backed tools (mcp_host; every name a configured
             # server backs, read at import).
             | mcp_host.mcp_config.all_tool_names())

# THE `yama_*` RENAME (operator, 2026-09-27): every tool the proxy adds to
# main is named `yama_*`, a name no harness offers. The names before it, as
# stored ledger rows (hidden hops), the craft state's withheld list and old
# records still carry them: each is read as its new name (the hops replay
# byte for byte as stored; a call the model makes by an old name -- copied
# from a replayed hop -- runs as the new tool).
LEGACY_TOOL_NAMES = {
    deep.LEGACY_TOOL_NAME: deep.TOOL_NAME,
    images.LEGACY_TOOL_NAME: images.TOOL_NAME,
    vision.LEGACY_TOOL_NAME: vision.TOOL_NAME,
    skill_prompts.LEGACY_CRAFT_TOOL_NAME: skill_prompts.CRAFT_TOOL_NAME,
}
LEGACY_TOOL_NAMES.update(mcp_host.mcp_config.legacy_names())


def canonical_tool_name(name: str) -> str:
    """A tool name as this proxy runs it: an old name of ours
    (LEGACY_TOOL_NAMES) read as its `yama_*` name, anything else as is."""
    return LEGACY_TOOL_NAMES.get(name, name) if isinstance(name, str) \
        else name


def is_ours(name: str, ours: set | frozenset) -> bool:
    """Is a call by `name` one of OUR tools offered on this request (`ours`,
    prepare's list) -- by its name, or by the old name of one of them?"""
    return name in ours or (name in LEGACY_TOOL_NAMES
                            and LEGACY_TOOL_NAMES[name] in ours)

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


# preamble_for / available_checks / PREAMBLE (YAMADORI_PREAMBLE): a status
# line prepended to the first answer. REMOVED 2026-09-27
# (docs/CONSTANTS-AUDIT.md): dead since resolve_repo always returns no
# repository (2026-09-22), so it never wrote anything.


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
    # An old name of ours runs as its `yama_*` tool (LEGACY_TOOL_NAMES).
    name = canonical_tool_name(name)
    if (turn is not None and name not in _STATEFUL
            and turn.cached_empty(name, args)):
        turn.record(name, args, _EMPTY_AGAIN)
        return _EMPTY_AGAIN

    if name == images.TOOL_NAME:
        # The proxy runs it, on CUDA1, in its own lane (admission.image_lane).
        # The result carries a signed /media URL built on the address the
        # client reaches us on, and each call is recorded for x_yamadori.
        st = state if state is not None else {}
        out = images.run_tool(args, st.get("_public_base") or images.public_base(),
                              st.setdefault("_images", []),
                              account=st.get("_account") or None,
                              # A Responses image_generation tool's size
                              # (mcp/responses_api.py), for this request.
                              options=st.get("_image_options"))
        # What it drew may now be looked at (yama_describe_image), by url or sha.
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

    if mcp_host.is_mcp_tool(name):
        # An MCP-backed tool (mcp/mcp_host.py): the proxy is the MCP client
        # of the server behind it (PackageLens, in its gated container). The
        # result is fetched content, rendered, screened and framed as data
        # there; each call is recorded for x_yamadori.mcp.
        st = state if state is not None else {}
        return mcp_host.run_tool(name, args, st.setdefault("_mcp_calls", []))

    if name == skill_prompts.CRAFT_TOOL_NAME:
        # yama_recall_craft (skill_select PROGRESSIVE DISCLOSURE): one craft in
        # full, by name or topic, as a hidden hop the ledger replays. What
        # it returned counts as GIVEN for the per-turn engine (a later need
        # gets a recall line, not the body again).
        import skill_select
        st = state if state is not None else {}
        text, rec = skill_select.read_craft(args)
        st.setdefault("_craft_reads", []).append(rec)
        lineage = session_lineage(st) if st else ""
        if rec.get("found") and lineage:
            account = st.get("_account") or ""
            with skill_select_lock(account, lineage):
                sk = _skill_state(account, lineage)
                sk.setdefault("given", {}).setdefault(rec["found"], {
                    "chars": int(sk.get("chars") or 0),
                    "req": int(sk.get("req") or 0), "via": "read"})
                _save_skill_state(account, lineage, sk)
        return text

    if name in (deep.TOOL_NAME, deep.PLAN_TOOL_NAME):
        # yama_think_deeply and yama_plan run inside a chat turn (_run_turn
        # -> _think_deeply / _plan_task), where the result becomes the next
        # hop's context; anywhere else there is no turn to hand it to.
        return cs.error_result(
            name, "NOT_IN_A_TURN",
            f"{name} runs only inside a chat turn, where its result is the "
            f"next step's context. Nothing was run.", retryable=False,
            remedies=[{"fixable_by": "agent",
                       "action": "ask in a chat turn at reasoning_effort "
                                 "xhigh or max",
                       "effect": f"the model can call {name} there"}])

    if name in research_tools.NAMES:
        # The second brain's other sources (Phase 0.6): the knowledge base
        # (the armed skills, nothing else: operator 2026-09-25/26), the web.
        # Never on main.
        # One deep-thinking run's web budget (research_tools.
        # SEARCHES_PER_RUN, READS_PER_RUN) and seen URLs, reset by each run
        # (_deep_thinking, _think_deeply).
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

    # A repeat still runs, and is recorded (the empty-search cache above).
    # The "what has NOT been tried" guidance appended here was removed
    # 2026-09-27 (docs/CONSTANTS-AUDIT.md: its evidence was Laya's margins,
    # not this model's).
    if turn is not None:
        turn.record(name, args, text)

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
    holds (package indexes, skills, deep thinking, fan-out). Nothing in a
    request may select a directory on this disk.
    """
    return None, False, "none"


def session_context(messages: list[dict], account: str = "",
                    session: str = "",
                    utility: dict | None = None, cache_key: str = "",
                    minted: str | None = None) -> tuple[str, dict]:
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
    # WHICH CONVERSATION (#41, mcp/session_id.py): an explicit id, never
    # inferred -- prompt_cache_key, the header, or our own id (carried in
    # the tool-call ids we return, or on a compaction summary's line).
    key, ses = session_identity(messages, account, session, cache_key, minted)
    fresh = not nebari.load(key)
    state = nebari.observe(key, discover.scan(messages))
    if fresh and not is_first_turn(messages):
        state = _continue_after_compaction(key, state, account, messages)
    # Our id -- or, for a conversation the client named by prompt_cache_key,
    # the CARRIER of its key (session_id.carrier_of), so its compaction
    # summary carries a line too and a continuation under a new key is
    # recognised by it (the alias rule, session_identity).
    keep = ses["id"] if ses["source"] in session_id.OURS else \
        ses.get("carrier")
    if keep and state.get("session_id") != keep:
        # Kept with the conversation, so a compaction of it (mapped by
        # _serve_compaction to its lineage) can carry the same id on its
        # summary line.
        cur = nebari.load(key)
        cur.update(session_id=keep, session_source="token")
        nebari.save(key, cur)
        state.update(session_id=keep, session_source="token")
    state["_key"] = key
    state["_session"] = ses
    if ses.get("forked_from") and not state.get("continues"):
        # A FORK -- or a compaction whose summary line never reached the
        # client (pagoda-h6: Hermes hung up on its summariser after 600 s,
        # compacted on its own, and its prompt_cache_key changed): its
        # history is the other conversation's, so what that conversation
        # did once per key stays done (deep.INHERITED). Once per fork.
        _inherit_once_per_key(account, state, ses["forked_from"])
    with _SESSIONS_LOCK:
        pend = _PENDING_COMPACTION.get(account)
        if pend and pend.get("lineage") == session_lineage(state):
            # The compacted conversation came back BY ITS ID (its session
            # line): the pending link is spent -- it must not attach a later
            # conversation that happens to arrive inside the window.
            _PENDING_COMPACTION.pop(account, None)
    # The ledger's chain keys are salted with an explicit conversation's key
    # (chain_keys): two conversations that open alike hash their opening
    # alike, and one's decisions on it were replayed into the other. The
    # fallback (no id) keeps the unsalted keys it was recorded under.
    state["_chain_salt"] = key if ses["source"] != "none" else ""
    with _SESSIONS_LOCK:
        _LAST_SESSION[account] = (key, time.time(), session_lineage(state))
    return key, state


def session_identity(messages: list[dict], account: str = "",
                     header: str = "", cache_key: str = "",
                     minted: str | None = None) -> tuple[str, dict]:
    """(key, {id, source, why}) of the conversation this request belongs
    to, from the first explicit id present (mcp/session_id.py):

      prompt_cache_key  the body's field
      header            X-Yamadori-Session (the key as it always was:
                        nebari.key_of with the token)
      tool_call_id      our id, carried in a tool-call id we returned (an
                        assistant tool_calls[].id or a tool_call_id of our
                        form anywhere in the history; the first found wins)
      summary_line      our id on the line of a compaction summary we wrote
      answer_record     our id, RECORDED with an answer of ours that carried
                        nothing (no client tool call), keyed by the whole
                        conversation through that answer as the client
                        stores it (_session_of_answers) -- our own bytes,
                        never the opening alone
      minted            none of these: a NEW conversation. Its id rides out
                        in this turn's tool-call ids (or its answer's
                        record). `minted` is the id when this request
                        already minted one (session_context runs twice per
                        request: _run_turn, then prepare).
    `id` is ours in full; a client's own id is shown as a hash prefix. Our
    sources key a conversation alike (conversation_key with "token"), so it
    keeps its key whichever carrier brought the id back. The old fallback
    -- `none`, nebari.key_of on the first two messages -- is no longer
    reached: nothing is inferred."""
    if cache_key:
        # A key that CHANGES mid-conversation (session_id.py, Hermes'
        # Responses key): a key seen before keeps the conversation it named
        # first; a NEW key whose history carries one of our ids continues
        # that conversation (no session header: an explicit per-conversation
        # header, e.g. OpenCode's fork, keeps its own); else its own.
        ref = session_id.cache_key_ref(cache_key)
        own = session_id.conversation_key(account, "prompt_cache_key",
                                          cache_key)
        info = {"id": session_id.short(cache_key),
                "source": "prompt_cache_key",
                "why": "the request's prompt_cache_key"}
        key = session_id.alias_get(account, ref)
        # Only COMPACTION evidence joins a new key to a conversation: our
        # summary line. Our id in tool-call ids alone is a fork (a fork
        # keeps the full history): a new conversation, `forked_from`.
        summ = None if header else session_id.summary_of(messages)
        if key is None:
            key = session_id.conversation_of_id(account, summ) if summ \
                else own
            session_id.alias_put(account, key, ref)
        if key != own:
            # (Also the second session_context of a request, which sees the
            # alias the first recorded.)
            info.update(aliased_to=summ or True, why=(
                f"a prompt_cache_key new to this server, whose history "
                f"carries our summary line ({summ}): the compacted "
                f"conversation continues under the new key" if summ else
                "the request's prompt_cache_key, an alias of the "
                "conversation it continued"))
        elif not header:
            sid, where = session_id.from_messages(messages)
            if sid and where == "tool_call_id" and \
                    session_id.conversation_of_id(account, sid) != own:
                info.update(forked_from=sid, why=(
                    f"the request's prompt_cache_key; its history carries "
                    f"our id {sid} in tool-call ids only (no summary line): "
                    f"a fork, a new conversation"))
        # The id of ours its tool-call ids carry, so a later key change can
        # find it (_carry_session).
        info["carrier"] = session_id.carrier_of(key)
        cref = session_id.carrier_ref(info["carrier"])
        if session_id.alias_get(account, cref) != key:
            session_id.alias_put(account, key, cref)
        return key, info
    if header:
        return (nebari.key_of(messages, account, header),
                {"id": session_id.short(header), "source": "header",
                 "why": "the X-Yamadori-Session header"})
    sid, where = session_id.from_messages(messages)
    if sid:
        # The carrier of a client-named conversation names THAT one.
        return (session_id.conversation_of_id(account, sid),
                {"id": sid, "source": where,
                 "why": ("our id, carried in a tool-call id of the history"
                         if where == "tool_call_id" else
                         "our id, on the session line of a compaction "
                         "summary")})
    sid = None if is_first_turn(messages) else \
        _session_of_answers(messages, account)
    if sid:
        return (session_id.conversation_key(account, "token", sid),
                {"id": sid, "source": "answer_record",
                 "why": "our id, recorded with an answer of ours in the "
                        "history that made no tool call (nothing carried "
                        "it)"})
    sid = minted or session_id.mint()
    return (session_id.conversation_key(account, "token", sid),
            {"id": sid, "source": "minted",
             "why": ("no prompt_cache_key, no header and no id of ours in "
                     "the history: a new conversation"
                     + ("" if is_first_turn(messages) else
                        " (it has answers, none of them ours on record)")
                     + "; the id rides in this turn's tool-call ids")})


# THE ANSWER RECORD (#41, 2026-09-25). A conversation whose answers make no
# client tool call has no carrier, and the literal rule -- "its next request
# is a new session" -- broke the slot's cache on every such turn
# (mcp/test_ledger.py, run on the carrier alone: the second request salted
# its chain with a new id, so the first turn's recorded skill injection was
# not replayed and the prompt diverged at turn 1; at xhigh the new
# conversation decided yama_think_deeply's offer afresh and the tool list itself
# changed; the known-hard-area trigger would re-fire on every such request).
# So the proxy records, with every answer it delivers for a conversation with
# our id that carries nothing, that id under the UNSALTED chain key of the
# conversation through that answer as the client stores it (roles, text,
# call ids; reasoning excluded -- chain_keys). A later request with no
# carrier looks its assistant messages up, first found wins. The key covers
# our own generated answer, not only the opening, so two runs that open
# alike are two conversations unless the model answered them byte for byte
# alike (the same argument as slots._adopt "through an answer").
SESSION_KIND = "session"


def _answer_key(prev: str, text: str) -> str | None:
    """The record's key for an answer: the unsalted chain key of the message
    before it and the answer's text, stripped (a client may trim it)."""
    text = (text or "").strip()
    if not text:
        return None
    return "session:" + _hashlib.sha256(
        (prev + "\x00" + text).encode("utf-8", "replace")).hexdigest()


def _session_of_answers(messages: list[dict], account: str) -> str | None:
    """Our id recorded with an assistant message of this history, or None;
    the first found wins."""
    keys = None
    for i, m in enumerate(messages or []):
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        keys = keys or chain_keys(messages, "")
        k = _answer_key(keys[i - 1] if i else "", _msg_text(m))
        sid = nebari.ledger_get(account, k, SESSION_KIND) if k else None
        if isinstance(sid, str) and re.fullmatch(r"[0-9a-f]{12}", sid):
            return sid
    return None


def _record_answer_session(payload: dict, messages: list[dict],
                           text: str, calls: list[dict]) -> bool:
    """Record our id with a delivered answer that carries nothing (see THE
    ANSWER RECORD). `text`: the answer's content as the client stores it."""
    ses = (payload.get("_ledger") or {}).get("conversation") or {}
    if calls or ses.get("source") not in session_id.OURS \
            or not ses.get("id"):
        return False
    k = _answer_key(chain_keys(messages, "")[-1] if messages else "", text)
    if not k:
        return False
    account, session = ledger_scope(payload)
    nebari.ledger_put(account, session, k, SESSION_KIND, ses["id"])
    return True


def _carry_session(payload: dict, calls: list[dict],
                   slot_msg: dict | None = None) -> list[dict]:
    """The client calls this turn returns, their ids rewritten to carry the
    conversation's id (#41, session_id.carry: `call_<id>_<8 hex>`), when the
    id is ours. A client's own id (prompt_cache_key, the header) needs no
    carrier: the client sends it again. Every OpenAI-compatible client
    echoes a call's id and its result's tool_call_id verbatim, and the
    served template renders neither, so the model never sees them and the
    slot's prompt is the same whatever they are.

    `slot_msg` (what the slot generated) gets the same ids, so the warm's
    comparison (_same_turn) does not mistake a renamed call for a changed
    turn."""
    ses = (payload.get("_ledger") or {}).get("conversation") or {}
    # Our own id; or, for a conversation the client named with
    # prompt_cache_key, its carrier (session_id.carrier_of), so a history
    # replayed under a CHANGED key still names it (Hermes' Responses key).
    sid = ses.get("id") if ses.get("source") in session_id.OURS else \
        ses.get("carrier")
    if not calls or not sid:
        return calls
    new, ids = session_id.carry(calls, sid)
    if isinstance(slot_msg, dict) and slot_msg.get("tool_calls"):
        slot_msg["tool_calls"] = [
            dict(c, id=ids.get(c.get("id"), c.get("id")))
            if isinstance(c, dict) else c for c in slot_msg["tool_calls"]]
    payload["_session_carried"] = len(new)
    return new


# COMPACTION CONTINUITY. A conversation whose key is NEW but which already has
# answers in it cannot be a new conversation -- a new one starts with no
# assistant turn. Arriving right after the same account's harness asked for a
# summary of a conversation (a summarise_conversation utility call), it is
# that conversation, compacted. Both halves are required: a fresh key with
# history alone is also a proxy restart or an expired session, and a summary
# alone says nothing about which request comes next. In-process only; a
# restart between the compaction and the next turn loses the link (the old
# behaviour, not a wrong one).
#
# WHOSE COMPACTION IT WAS, AND THE LINK (Octopus v0b-V0-xhigh-1, 2026-09-25,
# docs/SELF-IMPROVEMENT-LOG.md #38). No guess:
#   - a compaction this proxy MAPPED to a stored conversation (flattened:
#     "136 of 136 turns map"; in place: its own slot key) names that
#     conversation's lineage itself;
#   - the continuation is recognised by CONTENT: it carries the summary the
#     compaction produced (Hermes puts it in a message behind its prefix);
#   - or it names the conversation explicitly (session_id: our id in its
#     tool-call ids, the summary line, a key or header), which
#     session_identity resolves before this is reached.
# REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md): COMPACTION_LINK_SECONDS
# (1,800 s, YAMADORI_COMPACTION_LINK_S), the recency guess that linked a
# new-key request to the account's last conversation, and linked an unmapped
# compaction to it -- a 2,903 s turn had already defeated it (#38), and the
# operator: "we can't assume a new session is a resumable one" (2026-09-25).
# The summary probes are held in memory only, like the compaction store.
_SESSIONS_LOCK = threading.Lock()
# account -> (key, when, lineage): the account's last conversation request.
_LAST_SESSION: dict[str, tuple[str, float, str]] = {}
# account -> {key, lineage, when, probes, how}: a compaction awaiting its
# continuation.
_PENDING_COMPACTION: dict[str, dict] = {}
# Summary probes: whitespace-normalised windows of the compaction's answer.
PROBE_CHARS = 96
PROBE_MIN_SUMMARY = 200


def _norm_ws(text: str) -> str:
    return " ".join((text or "").split())


def _summary_probes(summary: str) -> list[str]:
    """Up to three windows from inside the summary (a quarter, half and three
    quarters in), whitespace-normalised: a harness that prefixes it, appends a
    footer or re-wraps it still carries them."""
    s = _norm_ws(summary)
    if len(s) < PROBE_MIN_SUMMARY:
        return []
    out = []
    for f in (0.25, 0.5, 0.75):
        i = min(int(len(s) * f), len(s) - PROBE_CHARS)
        p = s[i:i + PROBE_CHARS]
        if p and p not in out:
            out.append(p)
    return out


def _carries_summary(messages: list[dict], probes: list[str] | None) -> bool:
    if not probes:
        return False
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        t = _norm_ws(_msg_text(m))
        if t and any(p in t for p in probes):
            return True
    return False


def _note_compaction(account: str, lineage: str | None = None) -> None:
    """A summarise-the-conversation call: remember whose conversation it was.
    `lineage`: the conversation the compaction was MAPPED to (the stored
    prompt it was served on) -- exact, so no recency guess is needed."""
    if not lineage:
        # Unmapped: whose conversation it summarised is unknown, and nothing
        # is guessed (no recency link, 2026-09-27).
        return
    now = time.time()
    with _SESSIONS_LOCK:
        last = _LAST_SESSION.get(account)
        key = last[0] if last and last[2] == lineage else None
        _PENDING_COMPACTION[account] = {
            "key": key, "lineage": lineage, "when": now, "probes": [],
            "how": "mapped"}


def _compaction_done(account: str, lineage: str | None, summary: str) -> None:
    """The compaction answered: its window starts NOW (a 12k-token summary
    can take most of an hour), and the continuation will carry `summary`.
    The conversation's next request carries the work log, whatever its key
    (progress.note_compaction; #54)."""
    _compaction_counted(account, lineage)
    with _SESSIONS_LOCK:
        pend = _PENDING_COMPACTION.get(account)
        if pend is None or (lineage and pend.get("lineage") != lineage):
            if not lineage:
                return
            last = _LAST_SESSION.get(account)
            pend = {"key": last[0] if last and last[2] == lineage else None,
                    "lineage": lineage, "how": "mapped"}
        pend = dict(pend, when=time.time(), probes=_summary_probes(summary))
        _PENDING_COMPACTION[account] = pend


def _compaction_counted(account: str, lineage: str | None) -> None:
    """Count a finished compaction against the conversation it summarised:
    the mapped lineage, else the one the pending link names (an unmapped
    flattened compaction of the account's last conversation)."""
    lin = lineage
    if not lin:
        with _SESSIONS_LOCK:
            lin = (_PENDING_COMPACTION.get(account) or {}).get("lineage")
    if lin:
        try:
            progress.note_compaction(account, lin)
        except Exception as e:                                   # noqa: BLE001
            print(f"  work log: compaction not counted ({type(e).__name__}: "
                  f"{e})", flush=True)


def _session_touch(account: str, key: str) -> None:
    """A conversation turn ENDED: the account's last activity is now. A turn
    can run most of an hour (deep thinking, a long generation)."""
    if not key:
        return
    with _SESSIONS_LOCK:
        last = _LAST_SESSION.get(account)
        if last and last[0] == key:
            _LAST_SESSION[account] = (key, time.time(), last[2])


def _continue_after_compaction(key: str, state: dict, account: str,
                               messages: list[dict] | None = None) -> dict:
    """Carry a compacted conversation's session over to its new key: the work
    log (`lineage`, which record_step / read_rings and the slot pin use), the
    offered-tools flag, versions and packages. One link per compaction.

    Linked only when the request CARRIES the compaction's summary (the
    recency window was removed 2026-09-27). A request that does not carry
    it leaves the link pending for the continuation that does."""
    with _SESSIONS_LOCK:
        pend = _PENDING_COMPACTION.get(account)
        if not pend or pend.get("key") == key:
            return state
        carried = _carries_summary(messages or [], pend.get("probes"))
        if not carried:
            return state
        _PENDING_COMPACTION.pop(account, None)
    lineage = pend["lineage"]
    prev = nebari.load(pend["key"]) if pend.get("key") else {}
    if not prev:
        prev = nebari.load(lineage)
    cur = nebari.load(key)
    cur["lineage"] = prev.get("lineage") or lineage
    cur["continues"] = pend.get("key") or lineage
    if prev.get("tools_offered"):
        cur["tools_offered"] = True
    for field in ("versions", "asked", "counts"):
        merged = dict(prev.get(field) or {})
        merged.update(cur.get(field) or {})
        cur[field] = merged
    counts = cur.get("counts") or {}
    cur["packages"] = sorted(counts, key=lambda p: (-counts[p], p))
    cur["linked_by"] = "summary"
    nebari.save(key, cur)
    print(f"  session {key[:8]} continues {cur['continues'][:8]} after a "
          f"compaction (work log {cur['lineage'][:8]}; linked by "
          f"{cur['linked_by']})", flush=True)
    return cur


def _inherit_once_per_key(account: str, state: dict, sid: str) -> None:
    """The deep-thinking state a conversation did once per key (the
    server-tool triggers' fired keys, the areas researched, the verify
    moments passed) carried from the conversation our id `sid` names into
    this one's lineage (deep.inherit_state; idempotent)."""
    parent_key = session_id.conversation_of_id(account, sid)
    lineage = session_lineage(state)
    if not parent_key or parent_key == state.get("_key"):
        return
    parent = session_lineage(dict(nebari.load(parent_key) or {},
                                  _key=parent_key))
    try:
        got = deep.inherit_state(account, lineage, parent)
    except Exception as e:                                       # noqa: BLE001
        print(f"  deep: once-per-key state not inherited ({type(e).__name__}"
              f": {e})", flush=True)
        return
    if got:
        print(f"  session {lineage[:8]} inherits {parent[:8]}'s once-per-key "
              f"state ({len(got['fired'])} trigger keys, "
              f"{len(got['areas'])} areas)", flush=True)


def _fork_parent_lineage(account: str, state: dict | None,
                         ses: dict | None) -> str | None:
    """The lineage key of the conversation this one FORKED from (session
    `forked_from`), or None. Never raises."""
    sid = (ses or {}).get("forked_from")
    if not sid:
        return None
    try:
        parent_key = session_id.conversation_of_id(account, sid)
        if not parent_key or parent_key == (state or {}).get("_key"):
            return None
        return session_lineage(dict(nebari.load(parent_key) or {},
                                    _key=parent_key)) or None
    except Exception:                                            # noqa: BLE001
        return None


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
# missing injection on an early user turn re-prefills everything after it. The
# compaction agent measured exactly that on Hermes: 87% of the prompt shared
# on an ordinary turn, because the text attached to the previous user turn was
# not in the client's resent copy.
#
# What is recorded, and under which key (storage: nebari's `additions`
# table, memory in front of it -- see nebari.py LEDGER):
#
#   user turn        "inject"     skills, library definitions, the
#                                 work log after a compaction: decided ONCE,
#                                 on the request whose last message is that
#                                 user turn, and replayed after its content
#   tool result      "inject"     LIBRARY USE (#19): the same, on the tool
#                                 result a request ended on
#   assistant turn   "reasoning"  RESTORED (operator, 2026-09-27, reversing
#                                 the 2026-09-24 pass-through: "Keeping
#                                 thinking across turns seems useful, fuck
#                                 Hermes, Hermes can do whatever it wants.").
#                                 The slot's own reasoning for the delivered
#                                 turn (a prefilled hand-off included), keyed
#                                 like its content, put back into every past
#                                 turn the client sent without reasoning
#                                 (Hermes strips it: agent/
#                                 message_sanitization.py
#                                 apply_reasoning_content_policy); a client's
#                                 echo is kept as sent. Bonsai 2's base,
#                                 Qwen3.8-27B, keeps thinking across turns by
#                                 default (its card: `preserve_thinking`
#                                 "enabled by default", for consistency and
#                                 "improved KV cache utilization"), and the
#                                 served template renders every past turn's
#                                 think block (`preserve_thinking` undefined),
#                                 so each request EXTENDS the slot again. The
#                                 cost is context: every past turn's
#                                 reasoning, on every request (V0 pilot: 6-10k
#                                 tokens a step), counted by the window check
#                                 (check_client_prompt). Switch
#                                 `restore_reasoning` (tiers.BEHAVIOURS,
#                                 default on); off, the pass-through.
#                    "content"    a turn with tool calls whose delivered
#                                 content (the check note) a client may drop
#                    "hops"       the hidden hops the proxy ran inside the
#                                 turn, which the client never sees, with the
#                                 reasoning they were generated with: kept
#                                 when past reasoning is restored, emptied
#                                 when it is not (and the client does not
#                                 echo)
#   the request      "seed:<job>" the concept seed a second-brain job drew,
#                                 so a retry or replay uses the same word
#
# Keys are hashes, never text. A user turn's key hashes the conversation up
# to and including it (roles, text, tool-call ids and arguments; reasoning
# excluded), so the same words at two points of a conversation -- "continue"
# -- are two keys, and two conversations cannot share one. An assistant
# turn's key is its tool-call ids, else its content: what the client echoes.
# A conversation with an explicit id (#41, session_identity) salts the chain
# with its key, which is fixed from its first request (the id is minted
# there); one with no id keeps the unsalted chain, which does not depend on
# nebari.key_of (it changes between turn 1 and 2 for a conversation with no
# system message). The `session` column is for pruning only.
import hashlib as _hashlib  # noqa: E402


def _args_norm(raw) -> str:
    """Tool-call arguments in one canonical form: clients re-serialise them."""
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


# ONE ORDER FOR TOOL-CALL ARGUMENTS (Octopus v0e-V0-xhigh-1, 2026-09-25,
# step 6: reused 16187 of 25487 right after a warm that had loaded 25258). The
# served template renders a call's arguments in KEY ORDER
# (`tool_call.arguments|items`; llama-server keeps the object's order), and
# Hermes re-serialises every historical call with sorted keys on its send
# path (agent/conversation_loop.py _canonicalize_tool_call_arguments:
# json.dumps(..., sort_keys=True)). The model wrote patch as path, old_string,
# new_string; the warm rendered that order, the next request rendered
# new_string first, and the slot fell back past the divergence to a
# checkpoint 6.3K tokens earlier still. So every copy the proxy renders -- the
# request (prepare), the warm and the compaction store -- sorts the keys
# (recursively, as Hermes does): a client that sorts and one that keeps the
# model's order render alike, and the warm renders what the next request
# will. The chain keys already hash the sorted form (_args_norm).
def _args_sorted(raw):
    """`raw` (a JSON string or an object) with its keys sorted; the input
    unchanged when it is not a JSON object."""
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return raw
    if not isinstance(v, dict):
        return raw
    s = json.dumps(v, sort_keys=True, ensure_ascii=False)
    return s if isinstance(raw, str) else json.loads(s)


def sort_call_arguments(messages: list) -> list:
    """The messages with every assistant tool call's arguments in sorted key
    order (see ONE ORDER FOR TOOL-CALL ARGUMENTS). Messages that need no
    change are the same objects."""
    out = []
    for m in messages or []:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if isinstance(calls, list) and calls:
            new = []
            for c in calls:
                fn = c.get("function") if isinstance(c, dict) else None
                if isinstance(fn, dict) and "arguments" in fn:
                    a = _args_sorted(fn["arguments"])
                    if a != fn["arguments"] or (
                            isinstance(a, dict) and list(a) != list(
                                fn["arguments"])):
                        c = dict(c, function=dict(fn, arguments=a))
                new.append(c)
            if any(x is not y for x, y in zip(new, calls)):
                m = dict(m, tool_calls=new)
        out.append(m)
    return out


def _args_as_rendered(raw) -> str:
    """Tool-call arguments as the template renders them: parsed, in THEIR
    key order (unlike _args_norm)."""
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return json.dumps(v, ensure_ascii=False)
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


def chain_keys(messages: list[dict], salt: str = "") -> list[str]:
    """One key per message: the hash of the conversation up to and including
    it, as the client sent it.

    `salt`: the conversation's key when it has an explicit id (#41,
    session_context). Two conversations that open alike hash their opening
    alike; unsalted, the second replayed the first one's decisions on it
    (its skills, its concept seeds). "" -- a conversation with no id --
    keeps the keys it was always recorded under."""
    out, h = [], salt or ""
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


def ledger_restore(messages: list[dict], account: str,
                   skip: frozenset | set = frozenset(),
                   salt: str = "", restore_reasoning: bool = False
                   ) -> tuple[list, dict]:
    """The client's messages with every recorded addition put back.

    Returns (messages, counts). Messages the ledger has nothing for are the
    client's own objects, unchanged. `skip`: message keys whose recorded
    injection is NOT put back -- a retryable decision this request decides
    again (_inject_retry). `salt`: the conversation's (chain_keys).

    `restore_reasoning` (switch `restore_reasoning`, PAST REASONING IS
    RESTORED, operator 2026-09-27): a past assistant turn whose reasoning the
    client dropped gets the slot's own back (kind `reasoning`), and hidden
    hops keep theirs; a turn the client sent with reasoning keeps it as sent.
    Off: the 2026-09-24 pass-through.

    OUR SESSION LINE (#41, mcp/session_id.py) is never rendered. Since
    2026-09-25 only a compaction summary we wrote carries it (answers carry
    the id in their tool-call ids, which the template never renders): the
    summary turn is restored to what the slot generated (its recorded
    content), and a copy the ledger has no record for -- the summary in a
    continuation, a pruned record, an answer stored while answers carried
    the line -- has the line taken out. The model never reads it, so it
    cannot imitate or alter it."""
    keys = chain_keys(messages, salt)
    out: list = []
    n = {"inject": 0, "content": 0, "hops": 0, "echoed_reasoning": 0,
         "markers_in_echo": 0, "session_line": 0}
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        # The ledger's keys are the client's copy, line included.
        mk = _memo_keys(m, keys[i - 1] if i else "") \
            if role == "assistant" else []
        if role in ("user", "assistant") and isinstance(m.get("content"), str) \
                and session_id.PREFIX in m["content"]:
            bare = session_id.strip(m["content"])
            if bare != m["content"]:
                m = dict(m, content=bare)
                n["session_line"] += 1
        # A user turn's injection, or a tool result's (LIBRARY USE, #19).
        # Its text may be a list of text parts (Pi, Responses input_text):
        # the addition is one more part, as it was decided (message_text).
        if role in ("user", "tool") and message_text.has_text(m) \
                and keys[i] not in skip:
            add = nebari.ledger_get(account, keys[i], "inject")
            if add:
                m = message_text.append_text(m, add)
                n["inject"] += 1
        elif role == "assistant":
            # A CLIENT'S ECHO IS KEPT AS SENT (see the block above): counted,
            # and a template marker inside an echo is counted too (#12) --
            # not scrubbed: it is what the client sent. A turn with no echo
            # gets the slot's own reasoning back below (restore_reasoning).
            r = m.get("reasoning_content")
            echoed = isinstance(r, str) and bool(r.strip())
            if echoed:
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
                        hl = json.loads(hops)
                    except ValueError:
                        break
                    # HIDDEN HOPS AND AN ECHOING CLIENT (2026-09-26, the
                    # cache after a yama_describe_image hop, docs/HARNESS-PI.md
                    # gap 5). When this turn's reasoning is EXACTLY the
                    # echo of what the client was shown (the hops'
                    # reasoning, their call lines and the answer's, run
                    # together), the turn is rendered as the slot holds it:
                    # each hop with its own reasoning, the answer with its
                    # own -- the same text, split where it was generated,
                    # as #10 does for content. Anything else: the hops keep
                    # their own reasoning when past reasoning is restored
                    # (the slot generated it), else it is emptied; the
                    # client's copy of the turn as sent.
                    er = _hop_echo(account, k)
                    r = m.get("reasoning_content")
                    if er and isinstance(r, str) and r.strip() and \
                            er.get("shown") == _echo_hash(r):
                        m = dict(m, reasoning_content=er.get("slot") or "")
                        n["hop_reasoning"] = n.get("hop_reasoning", 0) + 1
                    elif not restore_reasoning:
                        hl = [dict(h, reasoning_content="")
                              if isinstance(h, dict)
                              and h.get("role") == "assistant" else h
                              for h in hl]
                    out.extend(hl)
                    n["hops"] += 1
                    break
            # PAST REASONING IS RESTORED (operator, 2026-09-27): the turn the
            # client sent without reasoning gets what the slot generated for
            # it, so the rendering is the slot's sequence again. A turn the
            # ledger has no reasoning for (recorded before the switch, pruned,
            # another server's) is counted: it still diverges there.
            if restore_reasoning and not echoed:
                got = next((g for g in (nebari.ledger_get(
                    account, k, "reasoning") for k in mk) if g), None)
                if got:
                    m = dict(m, reasoning_content=got)
                    n["reasoning"] = n.get("reasoning", 0) + 1
                    n["reasoning_chars"] = n.get("reasoning_chars", 0) \
                        + len(got)
                elif not (m.get("reasoning_content") or "").strip():
                    n["reasoning_missing"] = n.get("reasoning_missing", 0) + 1
        out.append(m)
    return out, n


def _echo_hash(text: str) -> str:
    """The reasoning an echoing client sends back, whitespace-normalised."""
    import hashlib
    return hashlib.sha1(" ".join((text or "").split()).encode(
        "utf-8")).hexdigest()


def _hop_echo(account: str, key: str) -> dict | None:
    raw = nebari.ledger_get(account, key, "hop_echo")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def ledger_record_turn(account: str, session: str, msg: dict,
                       hops: list[dict] | None = None,
                       stored: dict | None = None, prev: str = "",
                       echo: str | None = None,
                       slot_reasoning: str | None = None,
                       reasoning: str | None = None) -> int:
    """Record what the proxy delivered for one assistant turn: its content
    when it carries tool calls (or differs from what the client stores), and
    the hidden hops before it, with the reasoning they were generated with
    (ledger_restore empties it unless the client echoes or past reasoning is
    restored). `reasoning`: the turn's own reasoning as the slot holds it (a
    prefilled hand-off included), recorded as kind `reasoning` when past
    reasoning is restored (switch `restore_reasoning`; None when it is off).
    Returns the bytes written, for the size record (mcp/test_ledger.py).

    `echo`: the reasoning this client will send back for the turn when it
    echoes (what it was shown), with `slot_reasoning`, what the slot
    generated for it: recorded with the hops (kind `hop_echo`, a hash of the
    echo and the slot's text) so an echoed turn renders as the slot holds
    it (2026-09-26, HIDDEN HOPS AND AN ECHOING CLIENT in ledger_restore).

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
    echo_rec = (json.dumps({"shown": _echo_hash(echo),
                            "slot": slot_reasoning or ""},
                           ensure_ascii=False)
                if hops and echo and echo.strip() else None)
    for k in keys:
        # Under the FIRST key only (ledger_restore reads any of the turn's
        # keys): a turn of three parallel calls would store it three times,
        # and reasoning is the ledger's largest kind.
        if reasoning and reasoning.strip() and k == keys[0]:
            nebari.ledger_put(account, session, k, "reasoning", reasoning)
            written += len(reasoning)
        if echo_rec:
            nebari.ledger_put(account, session, k, "hop_echo", echo_rec)
            written += len(echo_rec)
        if restore_content:
            nebari.ledger_put(account, session, k, "content",
                              msg.get("content") or "")
            written += len(msg.get("content") or "")
        if hops:
            blob = json.dumps(hops, ensure_ascii=False)
            nebari.ledger_put(account, session, k, "hops", blob)
            written += len(blob)
    return written


def scope_reasoning(messages: list[dict], account: str = "",
                    features=None) -> list[dict]:
    """The client's messages with the ledger's additions put back
    (ledger_restore, messages only), past reasoning included when switch
    `restore_reasoning` is on (the default; `features`: a request's
    X-Yamadori-Features)."""
    on = tiers.behaviour_of_header(features, "restore_reasoning")[0]
    return ledger_restore(messages, account, restore_reasoning=on)[0]


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
_LEGACY_NAMES = {"bind_project_context", code_check.TOOL_NAME} | set(
    # And the names ours had before the `yama_*` rename (2026-09-27): the
    # corpus recorded them in upstream tool lists.
    LEGACY_TOOL_NAMES)


def client_tool_names(body: dict) -> list[str]:
    """The CLIENT's own tools, never ours -- even when a client re-sends ours
    by name."""
    return [n for n in (t.get("function", {}).get("name")
                        for t in (body.get("tools") or [])
                        if isinstance(t, dict))
            if n and n not in OUR_NAMES and n not in _LEGACY_NAMES]


# X-Yamadori-Features flags that, forced ON, mean a benchmark asked for an
# augmentation -- and then a utility-shaped request still gets it.
_AUGMENTATIONS = ("retrieval", "skills", "investigate", "check_code", "repair",
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
          f"tools, no skills -- {util['because']}", flush=True)
    return {"skills": False, "investigate": False, "fanout_n": 1,
            "utility": True,
            "because": {"utility": util["because"], "skills": why,
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
    # Idempotent: _run_turn mapped them already; a direct caller (a bench
    # arm, a test) gets the same mapping (system_roles.one_system).
    raw, roles_rec = system_roles.one_system(body.get("messages") or [])
    if raw is not body.get("messages"):
        body = dict(body, messages=raw, _roles=roles_rec)
    account = body.get("_account") or ""
    root, trusted, how = resolve_repo(raw, body.get("_client_ip", ""))
    # A client's own side call (an approval check, a title, a compaction):
    # the bare model, no session. selection.utility_call has the rule.
    util = body.get("_utility") or utility_of(body, strip_thinking(raw))
    utility = bool(util.get("utility"))
    _key, state = session_context(raw, account,
                                   body.get("_session_token") or "",
                                   utility=util,
                                   cache_key=session_id.cache_key_of(body),
                                   minted=body.get("_session_minted"))
    lineage = "" if utility else session_lineage(state)
    # THE CONVERSATION'S ID (#41): where it came from, and the salt of the
    # ledger's chain keys (session_context).
    ses = None if utility else state.get("_session")
    salt = "" if utility else (state.get("_chain_salt") or "")
    # THE LEDGER (see above): everything this proxy added to the
    # conversation on earlier requests -- reasoning, injections, the
    # delivered content of checked calls, image-tool hops -- put back, so this
    # request's rendering EXTENDS what the pinned slot holds. A utility call
    # is not a conversation and gets the client's messages as sent.
    keys = chain_keys(raw, salt)
    # A RETRYABLE decision on the message this request ends on (a part came
    # out empty because something was unavailable) is decided again below,
    # so it is not put back here (_inject_retry).
    retry_key, retry_meta = (None, {}) if utility else _inject_retry(
        raw, keys, account)
    # PAST REASONING IS RESTORED (switch `restore_reasoning`, operator
    # 2026-09-27): read before the tier, from the header and the environment.
    rr_on, rr_src = tiers.behaviour_of_header(body.get("_features"),
                                              "restore_reasoning")
    if utility:
        messages, restored = list(raw), {}
    else:
        messages, restored = ledger_restore(
            raw, account, skip={retry_key} if retry_key else frozenset(),
            salt=salt, restore_reasoning=rr_on)
        # ONE ORDER FOR TOOL-CALL ARGUMENTS: rendered sorted, whatever order
        # this client keeps (the warm renders the same).
        messages = sort_call_arguments(messages)
        nebari.ledger_touch(account, lineage)
    # ATTACHED IMAGES. When the main model has its projector
    # (vision.main_sees(), its /props: `bonsai` from 2026-09-27 until layout
    # v2, the max-mode model), a readable image in a user turn goes to it as
    # an image part -- a data: URI vision builds from the bytes, never a
    # client's URL -- with a label naming its id. Any other image (a tool
    # result's, one printed as text, and EVERY image while the main model has
    # no projector: layout v2, operator 2026-09-29, "Vision can go to second
    # card and swap in and out") becomes a text placeholder naming its id,
    # for yama_describe_image (mcp/vision.py), which asks `bonsai-vision` on
    # the A4000 (or the main model when it sees). After session_context on
    # purpose: the session key hashes the messages as the client sent them.
    # An image part carrying OUR signed /media link (a client looking at an
    # image we drew, e.g. Hermes' vision_analyze) is read from the media
    # store as an attachment when its host is ours: YAMADORI_PUBLIC_BASE, the
    # address this request reached us on, or loopback. Nothing is fetched.
    messages, att = vision.extract(messages, vision.our_hosts(
        body.get("_public_base") or ""), see=vision.main_sees())
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
    # RELEASE (mcp/slots.py): read before a utility call's tier is replaced
    # by `minimal`, so the header's switch holds for side calls too.
    slot_release = dict(zip(("on", "source"),
                            tiers.behaviour_source(tier, "slot_release")))
    idle_clear = dict(zip(("on", "source"),
                          tiers.behaviour_source(tier, "idle_clear")))
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
    # MAIN'S TOOLS: the client's, untouched, plus yama_generate_image /
    # yama_describe_image where offered (a capability on every tier, operator
    # 2026-09-23), plus the delegate benchmark arm when a header forces it.
    # A utility call gets the client's list and nothing of ours.
    delegate = (bool(tier.get("delegate")) or DELEGATE_TOOL) and not utility
    # yama_think_deeply (Phase 0.6, trigger 1): where deep thinking is allowed and
    # not forced off, decided on the conversation's first request and kept
    # (deep.think_tool_offered: a tool list that changes between turns would
    # change the system block the slot caches).
    continuing = any(isinstance(m, dict) and m.get("role") == "assistant"
                     for m in raw)
    think_on = deep.think_tool_offered(
        tier, account, lineage, utility, continuing=continuing)
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
    # consulted. The model's own yama_think_deeply call is the fourth, at
    # generation time (_think_deeply).
    # THE SERVER-TOOL TRIGGERS (operator, 2026-09-27): where main has
    # yama_think_deeply / yama_plan, the moments they were recalled at now
    # run them (deep.decide kind "auto"); build intent on a user turn is
    # the decider's (decide_turn.build_intent), the rule its fallback.
    # THE VERIFY DIRECTIVE (mcp/verify_moment.py) is decided with them:
    # its moment reads the same plan tracking. Each has its switch
    # (tiers.BEHAVIOURS auto_triggers, verify_directive).
    auto_on = tiers.behaviour(tier, "auto_triggers")
    verify_on = tiers.behaviour(tier, "verify_directive")
    auto_in = None if (utility or inplace or not (auto_on or verify_on)) \
        else {"think_ok": auto_on and deep.TOOL_NAME in ours,
              "plan_ok": auto_on and deep.PLAN_TOOL_NAME in ours,
              "plan_files": _plan_files(ours, messages),
              "verify": verify_on,
              "intent": lambda _instr: _build_intent(
                  raw, account, lineage, keys[-1] if keys else None)}
    trig = _deep_trigger(raw, tier, route, util, account, lineage,
                         keys[-1] if keys else None, inplace, state,
                         auto=auto_in)
    verify_rec = _verify_decision((trig or {}).get("verify"), raw,
                                  body.get("tools"), account, lineage,
                                  keys[-1] if keys else None)
    if utility:
        sel = _utility_selection(util, requested_tier)
    elif inplace:
        why = ("an in-place compaction: served on the conversation's own "
               "prompt and slot, nothing added (mcp/compaction.py)")
        sel = {"skills": False, "investigate": False, "fanout_n": 1,
               "utility": False, "compaction": True,
               "because": {"utility": util.get("because"), "skills": why,
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
    # THE CRAFT OFFER (skill_select PROGRESSIVE DISCLOSURE; operator,
    # 2026-09-27): on the conversation's first request, an index of the
    # craft relevant to it at the end of the system text (after the
    # addendum) and yama_recall_craft on main; decided ONCE and kept with the
    # tools withheld for a conflict with the client's (tool_conflicts), so
    # the system block and the tool list never change mid-conversation. A
    # kept offer applies to an in-place compaction too (its system block
    # must match the conversation's).
    craft_offer: dict = {"tool": False, "index": ""}
    tools_withheld: list[dict] = []
    mcp_rec: dict | None = None
    mcp_line = ""
    if not utility:
        with skill_select_lock(account, lineage):
            sk_state = _skill_state(account, lineage)
            try:
                craft_offer, sk_state = _craft_offer(
                    raw, sel, route, client_tools, account, sk_state,
                    inplace=inplace, continuing=continuing)
            except Exception as e:                               # noqa: BLE001
                craft_offer = {"tool": False, "index": "",
                               "why": f"the offer raised "
                                      f"{type(e).__name__}: {e}"[:200]}
            # THE MCP TOOLS (mcp/mcp_host.py; switch `mcp_tools`): decided
            # on the conversation's first request and kept by name, like the
            # craft offer, so the tool list and the system line never change
            # mid-conversation.
            mcp_rec = _mcp_offer(tier, sk_state, continuing)
            kept = list(sk_state.get("tools_withheld") or [])
            tools, ours = main_tools(
                body.get("tools"), None if utility else att,
                delegate=delegate, images_on=tiers.images_offered(tier)
                and not utility, think=think_on,
                craft=bool(craft_offer.get("tool")),
                withheld=tools_withheld, keep_withheld=kept,
                mcp=mcp_host.definitions(mcp_rec["offered"]))
            if not continuing:
                sk_state["tools_withheld"] = sorted(
                    {w["ours"] for w in tools_withheld})
            _save_skill_state(account, lineage, sk_state)
        # The one system line where they are offered (a conflict with the
        # client's tools withholds a tool, and the line names only the rest).
        mcp_rec["on_main"] = [n for n in mcp_rec["offered"] if n in ours]
        mcp_line = mcp_host.line_for(mcp_rec["on_main"])
        mcp_rec["line_chars"] = len(mcp_line)
    if mcp_line:
        augmented = add_system_tail(augmented, mcp_line)
    if craft_offer.get("index"):
        augmented = add_system_tail(augmented, craft_offer["index"])

    # THE PER-TURN INJECTION, decided ONCE per user turn and replayed from
    # the ledger ever after (ledger_restore put it back above). Decided on
    # the request whose last message IS that user turn -- a request that
    # ends on a tool result is a step of a task whose user turn the slot
    # already holds, and adding to it now would change a prefix it has
    # cached. One exception: the first request after a compaction, whose
    # whole prefix is new, may carry the work log on its last user turn.
    skills_rec: dict | None = None
    inject_rec = {"decided": False, "chars": 0, "parts": []}
    use_rec: dict | None = None         # LIBRARY USE (#19), None when off
    # Template markers scrubbed from OUR injected text (library source,
    # skills, the work log) when it is decided -- a replay sends what was
    # recorded, byte for byte (pre-deploy review, 2026-09-24).
    scrub_note: dict = {}
    # The last user turn the USER wrote: a harness's synthetic tool-media
    # turn carries a tool's image and is looked past (image_input), as
    # route, deep and selection.question_of do.
    li = next((i for i in range(len(raw) - 1, -1, -1)
               if isinstance(raw[i], dict) and raw[i].get("role") == "user"
               and not image_input.is_tool_media(raw, i)), None)
    speaking = li is not None and li == len(raw) - 1
    continued = bool(state.get("continues")) and not state.get(
        "rings_reinjected")
    # THE WORK LOG AFTER A COMPACTION OF A CONVERSATION WHOSE KEY DID NOT
    # CHANGE (#54; 2026-09-26). The link above (_continue_after_compaction)
    # fires only when the compacted conversation comes back under a NEW
    # session key; since #41 a conversation with an explicit id keeps its
    # key across a compaction, so the work log was never re-injected for
    # Hermes. _compaction_done counts the conversation's compactions
    # (progress.note_compaction); the first request after one carries it.
    wl_on, _wl_src = tiers.behaviour_source(tier, "work_log_reinject")
    reinject = (not utility and not inplace and not continued and wl_on
                and progress.reinject_due(account, lineage))
    continued = continued or reinject
    recorded = (None if li is None or utility else
                nebari.ledger_get(account, keys[li], "inject"))
    # RETRYABLE DECISIONS (live gate 2026-09-24: a decision made while the
    # embeddings were not loaded was replayed, forever, as final). A part
    # that came out empty BECAUSE something was unavailable is recorded under
    # `retry` in the decision's meta; the next request that ends on this
    # same user turn -- nothing after it is cached -- decides again, keeping
    # the parts that did not fail. Once the conversation has moved past the
    # turn, the slot holds what was sent, and that is what replays.
    retrying = (li is not None and retry_key is not None
                and retry_key == keys[li])
    if recorded is not None and not retrying:
        rmeta = nebari.ledger_meta(account, keys[li], "inject")
        inject_rec.update(replayed=True, chars=len(recorded),
                          parts=list(rmeta.get("parts") or []))
        # A replay reports the skills its text carries (recorded in the
        # decision's meta), without deciding again.
        skills_rec = dict(rmeta.get("skills") or {"ids": [], "versions": [],
                                                   "names": [], "chars": 0},
                          on=False, replayed=True,
                          why="decided on the request this user turn arrived "
                              "with; replayed from the ledger")
    elif (li is not None and not utility and not inplace
          and message_text.has_text(raw[li])
          and (speaking or continued)):
        keep = dict(retry_meta.get("texts") or {}) if retrying else {}
        failed = set(retry_meta.get("retry") or {}) if retrying else set()
        texts: dict[str, str] = {}
        unavailable: dict[str, str] = {}
        if speaking:
            if "skills" in keep and "skills" not in failed:
                texts["skills"] = keep["skills"]
                skills_rec = dict(retry_meta.get("skills") or {
                    "ids": [], "versions": [], "names": [], "chars": 0},
                    on=False, why="kept from this turn's retryable "
                                  "decision (skills did not fail)")
            else:
                # Selection reads the CLIENT's messages (`raw`), never the
                # ledger-restored ones: what we injected before is not
                # evidence (pagoda-h4: a TypeGPU craft body restored into
                # the conversation picked the next TypeGPU craft).
                _SKILL_CTX.value = {"lineage": lineage, "key": keys[li],
                                    "raw": raw,
                                    "server_tools": _server_tools(ours),
                                    "plan_files": _plan_files(ours,
                                                              messages)}
                try:
                    tail, skills_rec = _skills_tail(messages, sel, route,
                                                    client_tools, account)
                finally:
                    _SKILL_CTX.value = None
                texts["skills"] = tail
                # Retryable only when it came out EMPTY for that reason: a
                # fallback that still chose is a decision.
                why = None if tail else _skills_unavailable(skills_rec)
                if why:
                    unavailable["skills"] = why
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
            if "definitions" in keep and "definitions" not in failed:
                texts["definitions"] = keep["definitions"]
            else:
                why_defs: list[str] = []
                texts["definitions"] = _library_definitions(
                    route, tier, gate, sel, messages, route_dbs, state,
                    unavailable=why_defs)
                if why_defs:
                    unavailable["definitions"] = "; ".join(why_defs)[:300]
            if tier.get("retrieval") and not sel.get("investigate"):
                # LIBRARY USE (#19): whatever the class. Read from the
                # client's own messages -- never from what this service
                # injected, whose definitions carry imports of their own.
                if "library_use" in keep and "library_use" not in failed:
                    texts["library_use"] = keep["library_use"]
                else:
                    use, use_rec = _library_use(
                        raw, account, lineage, state,
                        room=_injection_room(augmented, body.get("tools"), sum(
                            len(v or "") for v in texts.values())))
                    texts["library_use"] = use
                    if use_rec.get("unavailable"):
                        unavailable["library_use"] = "; ".join(
                            use_rec["unavailable"])[:300]
        if continued:
            texts["work_log"] = _work_log_block(state)
            _mark_rings_reinjected(state)
            progress.mark_reinjected(account, lineage)
            inject_rec["work_log"] = {"reinjected": True,
                                      "same_key": bool(reinject),
                                      "chars": len(texts["work_log"])}
        elif keep.get("work_log"):
            texts["work_log"] = keep["work_log"]
        order = [p for p in INJECT_PARTS if texts.get(p)]
        clean: dict[str, str] = {}
        for p in order:
            clean[p], n_scrub = scrub_markers(texts[p])
            _note_scrub(scrub_note, "injection", n_scrub)
        text = "".join(clean[p] for p in order)
        meta = {"parts": order, "skills": _skills_meta(skills_rec)}
        if unavailable:
            # The parts that did NOT fail are kept verbatim for the retry.
            meta.update(retry=unavailable, texts=clean)
        # What stands is what is sent: never over a final non-empty decision
        # a duplicate request recorded meanwhile (nebari.ledger_decide).
        text, meta = nebari.ledger_decide(account, lineage, keys[li],
                                          "inject", text, meta)
        inject_rec.update(decided=True, chars=len(text),
                          parts=list(meta.get("parts") or []))
        if retrying:
            inject_rec["retried"] = sorted(failed)
        if meta.get("retry"):
            inject_rec["retryable"] = dict(meta["retry"])
            print("  injection RETRYABLE (decided again on the next request "
                  "that ends on this turn): "
                  + "; ".join(f"{k}: {v}" for k, v in meta["retry"].items()),
                  flush=True)
        if text:
            ai = _index_of_user(augmented, raw[li])
            if ai is not None:
                augmented = list(augmented)
                augmented[ai] = message_text.append_text(augmented[ai], text)
    # LIBRARY USE on a request that ENDS ON A TOOL RESULT (#19): the tool
    # result is the one message the slot does not hold yet, so a package the
    # conversation just started using (a file the harness read, a file the
    # model wrote) is covered there, decided once and recorded under that
    # message's key (ledger_restore replays it on every later request).
    #
    # THE PROJECT (mcp/progress.py): the step's successful writes are read
    # into the conversation's project (the named files, the inferred working
    # directory) for the fix-up's scope (#56) and deep thinking's label rule
    # (#52). State only, no text. The tool-result SITUATIONS that used to be
    # appended here (#50's progress line, #54's unchanged-read line) were
    # REMOVED 2026-09-27 (operator: task-targeted steering in prompts; skills
    # are the channel); tool results that already carry one replay it from
    # the ledger byte for byte (ledger_restore), so their slots' prefixes
    # hold.
    last_raw = raw[-1] if raw and isinstance(raw[-1], dict) else {}
    ends_on_tool = (not utility and not inplace
                    and last_raw.get("role") == "tool"
                    and message_text.has_text(last_raw)
                    and augmented and isinstance(augmented[-1], dict)
                    and augmented[-1].get("role") == "tool")
    lib_on = bool(tier.get("retrieval")) and not sel.get("investigate")
    proj_rec = (progress.learn(account, lineage, raw)
                if ends_on_tool and lineage else None)
    # SKILLS ON AN AGENT STEP (skill_select PER-TURN INJECTION, operator
    # 2026-09-27): the newest evidence of the step decides, and what it
    # brings is appended LAST to the tool result -- the end of what the
    # model reads next -- decided once under the result's key and replayed.
    step_skills_on = (bool(sel.get("skills")) and bool(lineage)
                      and not utility and not inplace)
    if ends_on_tool and (lib_on or step_skills_on):
        got = nebari.ledger_get(account, keys[-1], "inject")
        # A retryable decision on this tool result is decided again (see
        # RETRYABLE DECISIONS above); ledger_restore did not put it back.
        if got is None or (retry_key is not None and retry_key == keys[-1]):
            use = ""
            parts: list[str] = []
            meta: dict = {}
            if lib_on:
                use, use_rec = _library_use(
                    raw, account, lineage, state,
                    room=_injection_room(augmented, body.get("tools")))
                use, n_scrub = scrub_markers(use)
                _note_scrub(scrub_note, "injection", n_scrub)
                if use:
                    parts.append("library_use")
                if use_rec.get("unavailable"):
                    meta.update(retry={"library_use": "; ".join(
                        use_rec["unavailable"])[:300]}, texts={})
            sk_text = ""
            if step_skills_on:
                sk_text, skills_rec = _skills_step(
                    raw, sel, route, client_tools, account, lineage,
                    keys[-1], server_tools=_server_tools(ours),
                    plan_files=_plan_files(ours, messages))
                sk_text, n_scrub = scrub_markers(sk_text)
                _note_scrub(scrub_note, "injection", n_scrub)
                if sk_text:
                    sk_text = "\n" + sk_text
                    parts.append("skills")
                    meta["skills"] = _skills_meta(skills_rec)
                # THE VERIFY DIRECTIVE's craft (mcp/verify_moment.py): the
                # armed craft for checking this kind of work rides with the
                # line, in the same injection (recorded, replayed).
                vsk = (verify_rec or {}).pop("_skill", None)
                if vsk is not None and vsk.get("id") not in (
                        (skills_rec or {}).get("ids") or []):
                    import skill_select
                    vt, n_scrub = scrub_markers(skill_select.render([vsk]))
                    _note_scrub(scrub_note, "injection", n_scrub)
                    if vt:
                        sk_text += "\n" + vt
                        parts.append("verify_skill")
                        verify_rec["skill"]["injected"] = True
            meta["parts"] = parts
            text, meta = nebari.ledger_decide(account, lineage, keys[-1],
                                              "inject", use + sk_text,
                                              meta)
            if use_rec is not None and lib_on:
                use_rec["decided"] = True
                if got is not None:
                    use_rec["retried"] = True
                if meta.get("retry"):
                    use_rec["retryable"] = dict(meta["retry"])
            if text:
                augmented = list(augmented)
                augmented[-1] = message_text.append_text(augmented[-1], text)
        else:
            if lib_on:
                use_rec = {"replayed": True, "chars": len(got)}
            rmeta_t = nebari.ledger_meta(account, keys[-1], "inject")
            rparts = rmeta_t.get("parts") or []
            if "skills" in rparts:
                skills_rec = dict(rmeta_t.get("skills") or {}, on=False,
                                  replayed=True, why="decided on the request "
                                  "this tool result arrived with; replayed "
                                  "from the ledger")

    # The token budget is set inside tiers.apply by the one rule: the
    # client's max_tokens as the answer allowance, plus the thinking breaker.
    # The prompt estimate reads the messages AFTER vision.extract: a 1 MB
    # base64 image counted as text is ~450,000 "tokens", which would floor the
    # thinking room at MIN_THINKING for any turn that attached a picture.
    # EVERY thinking request has a cap (operator, 2026-09-25): a step in the
    # client's own tool loop thinks at most tiers.AGENT_STEP_THINKING, any
    # other turn tiers.USER_TURN_THINKING. Without one an uncapped question
    # got the whole main share (108,570 live) and the 60% nudge never fired.
    # A utility call (and so a compaction) keeps its own budget. An agent
    # step's nudge names the action (tiers.AGENT_STEP_NUDGE_MESSAGE, switch
    # step_nudge). The cap by what the step answers (#53: 2,048 after a
    # read, 6,144 after an error) was REMOVED 2026-09-27
    # (docs/CONSTANTS-AUDIT.md: chosen from one Octopus run).
    step_rec: dict | None = None
    nudge = None
    if not utility and route.get("class") == "agent_step":
        step_cap = tiers.AGENT_STEP_THINKING
        if tiers.behaviour(tier, "step_nudge"):
            nudge = tiers.AGENT_STEP_NUDGE_MESSAGE
        step_rec = {"thinking_cap": step_cap,
                    "why": "an agent step (tiers.AGENT_STEP_THINKING)",
                    "nudge": "agent_step" if nudge else "general"}
    else:
        step_cap = None if utility else tiers.USER_TURN_THINKING
    out = tiers.apply(dict(body, messages=messages), tier, step_cap=step_cap,
                      nudge=nudge)
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
    # x_yamadori.skills: ids, versions, names, chars, why -- never skill text
    # or a path. A REPLAYED decision reports the skills it carries.
    out["_skills"] = skills_rec
    # The craft offer (kept for the conversation) and the tools of ours
    # withheld for a conflict with the client's (x_yamadori.craft,
    # x_yamadori.tools_withheld).
    out["_craft"] = {k: craft_offer.get(k) for k in ("tool", "why", "kept")
                     if k in craft_offer} | {
        "listed": len(craft_offer.get("ids") or []),
        "index_chars": len(craft_offer.get("index") or "")}
    out["_tools_withheld"] = tools_withheld
    # x_yamadori.mcp: the switch, the offer (kept for the conversation), the
    # servers' states; the calls are added by the turn (_mcp_calls).
    out["_mcp"] = mcp_rec
    out["_selection"] = sel
    # x_yamadori.deep: the trigger decision (or the recorded non-decision),
    # and yama_think_deeply's and yama_plan's offer and calls on this
    # request (one offer: deep.think_tool_offered).
    out["_deep"] = trig
    # x_yamadori.deep.auto.verify: the verify moment's line and why.
    if verify_rec is not None:
        verify_rec.pop("_skill", None)
    out["_verify"] = verify_rec
    out["_think_tool"] = {"offered": think_on, "calls": []}
    out["_plan_tool"] = {"offered": think_on, "calls": []}
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
    if body.get("_upstream_model") and internal == max_mode.MAIN:
        # MAX MODE (mcp/max_mode.py): the main model server._serve_turn chose for this request
        out["model"] = body["_upstream_model"]
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
                      "restored": restored, "inject": inject_rec,
                      "salt": salt, "conversation": ses,
                      "restore_reasoning": {"on": bool(rr_on and not utility),
                                            "source": rr_src if not utility
                                            else "utility call"}}
    # OUR SESSION LINE (#41), only on a COMPACTION SUMMARY we write (operator,
    # 2026-09-25: no visible line in answers -- answers carry the id in
    # their tool-call ids, _carry_session): an in-place compaction of a
    # conversation with our id opens with the line, so the continuation
    # (which carries the summary) keeps the conversation.
    # _serve_compaction sets it for a flattened compaction it maps.
    lead_id = (ses.get("id") if ses and ses.get("source") in session_id.OURS
               else (ses or {}).get("carrier"))
    if lead_id and inplace:
        out["_session_lead"] = session_id.line(lead_id)
    if (restored or {}).get("markers_in_echo"):
        print(f"  template markers: {restored['markers_in_echo']} inside "
              f"reasoning the client echoed; passed through as sent",
              flush=True)
    # x_yamadori.library_use: the packages used and held, what was injected
    # (package:name pairs, overviews, chars) -- never the text (#19).
    out["_library_use"] = use_rec
    # x_yamadori.progress (mcp/progress.py, tiers.BEHAVIOURS): the agent
    # step's thinking cap and nudge, every switch and its source, and the
    # project as far as the conversation says (fix-up scope, deep labels),
    # with the writes this request's tool result added to it.
    out["_project"] = (progress.project_hint(account, lineage, raw)
                       if not utility else None)
    out["_progress"] = {"step": step_rec,
                        "switches": tiers.behaviours(tier),
                        "project": ({"root": out["_project"].get("root"),
                                     "root_by": out["_project"].get(
                                         "root_by"),
                                     "named": len(out["_project"].get(
                                         "named") or []),
                                     "writes": ({k: proj_rec.get(k) for k in (
                                         "project_writes", "scratch_writes",
                                         "error") if k in proj_rec}
                                         if proj_rec else None)}
                                    if out["_project"] else None)}
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
                    "record": not utility and kind != "compaction",
                    # A FORK takes over its parent's primacy (slots RANKS):
                    # pagoda-h6's post-compaction session was one.
                    "inherits": _fork_parent_lineage(account, state, ses)}
    # Whether this request empties the slots it leaves nobody will reuse
    # (slots RELEASE), and what it released (x_yamadori.slots).
    out["_slot_release"] = slot_release
    out["_slots_released"] = []
    # And whether it clears OTHER conversations' slots idle past
    # slots.IDLE_CLEAR_S before it generates (slots IDLE CLEAR).
    out["_idle_clear"] = dict(idle_clear)
    if isinstance(tier.get("idle_clear_s"), int) \
            and "idle_clear_s" in (tier.get("overridden") or []):
        # The live test's threshold override: a TEST account only, and it
        # clears only that account's own conversations (slots.clear_idle).
        test = corpus.account_traffic(account) == "test"
        out["_idle_clear"].update(
            override_s=tier["idle_clear_s"] if test else None,
            override_refused=None if test else "not a test account")
    out["_slots_cleared"] = []
    if out["_slot"]["record"]:
        out["_client_messages"] = list(raw)
    if kind == "compaction":
        _serve_compaction(out, body, messages, inplace)
    # One record per upstream generation (prompt, reused, processed, slot),
    # appended by _post_events; x_yamadori.cache and the log line read it.
    out["_cache_log"] = []
    out.pop("stream", None)
    return out


def _build_intent(raw: list[dict], account: str, lineage: str,
                  key: str | None) -> bool | None:
    """The decider's build intent for the user turn a request ends on
    (decide_turn.build_intent), or None (off, unavailable) -- the caller's
    rule then answers."""
    try:
        import decide_turn
        return decide_turn.build_intent(
            _text_messages(raw), account=account, key=lineage or None,
            request=key).get("value")
    except Exception as e:                                       # noqa: BLE001
        print(f"  build intent: the decider raised ({type(e).__name__}: "
              f"{e}); the rule answers", flush=True)
        return None


def _verify_decision(vm: dict | None, raw: list[dict], client_tools,
                     account: str, lineage: str, key: str | None
                     ) -> dict | None:
    """THE VERIFY DIRECTIVE for a verify moment deep.decide found
    (mcp/verify_moment.py): the decider's pick among the CLIENT's tools,
    the line (named or generic, and why), and the armed craft for checking
    this kind of work (or the coverage gap). None when this request is no
    verify moment. Never raises: a fault is the generic line, said why."""
    if not vm:
        return None
    import verify_moment as V
    rec: dict = {"trigger": "verify", "moment": vm.get("moment"),
                 "key": vm.get("key"), "kind": (vm.get("kind") or {}).get(
                     "kind"), "signal": (vm.get("kind") or {}).get("signal"),
                 "path": vm.get("path") or (vm.get("kind") or {}).get("path")}
    if vm.get("error"):
        rec.update(line=None, form=None, why="the moment raised: "
                   + vm["error"])
        return rec
    try:
        opts = V.tool_options([t for t in client_tools or []
                               if not is_ours(((t or {}).get("function")
                                               or {}).get("name"), OUR_NAMES)])
        rec["options"] = [o["name"] for o in opts]
        kind = vm.get("kind") or {}
        choice = (V.choose_tool(_text_messages(raw), kind.get("label"), opts,
                                account=account, key=lineage or None,
                                request=key)
                  if kind.get("kind") else {"judged": False,
                                            "why": "the kind is unknown"})
        vst = deep.load_state(account, lineage).get("verify") or {}
        line, form, why = V.directive(kind, choice, vst)
        rec.update(pick=choice.get("pick"), tie=choice.get("tie"),
                   raw_pick=choice.get("raw_pick"),
                   distribution=choice.get("distribution"),
                   rounds=len(choice.get("rounds") or []),
                   ms=choice.get("ms"), line=line, form=form, why=why,
                   **({"failure": choice["failure"]}
                      if choice.get("failure") else {}))
        sk = V.verify_skill(kind.get("kind"))
        rec["skill"] = ({"id": sk.get("id"), "name": sk.get("name"),
                         "version": sk.get("version")} if sk else
                        {"gap": f"no armed craft checks a "
                                f"{kind.get('label') or 'work of this kind'}"
                                f" (phase verify)"})
        if sk:
            rec["_skill"] = sk
    except Exception as e:                                       # noqa: BLE001
        rec.update(line=V.GENERIC, form="generic",
                   why=f"the verify decision raised {type(e).__name__}: "
                       f"{e}"[:200])
    print(f"  verify: {rec.get('moment')} ({rec.get('kind')}) -> "
          f"{rec.get('form')}: {rec.get('why')}", flush=True)
    return rec


def _deep_trigger(raw: list[dict], tier: dict, route: dict, util: dict,
                  account: str, lineage: str, turn_key: str | None,
                  inplace: bool, state: dict | None,
                  auto: dict | None = None) -> dict | None:
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
                           continues=bool((state or {}).get("continues")),
                           auto=auto)
    except Exception as e:                                       # noqa: BLE001
        print(f"  deep: trigger decision raised ({type(e).__name__}: {e}); "
              f"no trigger", flush=True)
        return {"fire": False, "kind": None, "allowed":
                bool(tier.get("investigate")), "forced": None,
                "because": f"the trigger decision raised "
                           f"{type(e).__name__}: {e}"[:300],
                "signals": {}, "thresholds": {}, "last_run": None}
    if trig.get("fire") or trig.get("signals", {}).get("struggle", {}).get(
            "count"):
        print(f"  deep: {trig.get('kind') or 'none'} -- "
              f"{trig.get('because', '')[:200]}", flush=True)
    return trig


# The per-turn injection's parts, in the order they are appended.
INJECT_PARTS = ("skills", "definitions", "library_use", "work_log")


def _skills_meta(rec: dict | None) -> dict:
    """What the ledger keeps about a decision's skills (ids, versions,
    names, chars), so a replay can report what its text carries."""
    rec = rec or {}
    out = {k: list(rec.get(k) or []) for k in ("ids", "versions", "names")} \
        | {"chars": int(rec.get("chars") or 0)}
    if rec.get("tool_recall"):
        # SERVER-TOOL RECALL lines the text carries (skill_select).
        out["tool_recall"] = [{k: d.get(k) for k in ("name", "trigger",
                                                      "key")}
                              for d in rec["tool_recall"]]
    return out


def _skills_unavailable(rec: dict | None) -> str | None:
    """Why the skills part came out empty because something was
    UNAVAILABLE (the embedder not loaded, the A4000 busy, a raise), or None
    when it was a real decision -- nothing cleared the floor, skills not
    allowed, no skill armed."""
    rec = rec or {}
    why = str(rec.get("why") or "")
    if why.startswith("skills raised"):
        return why[:200]
    emb = rec.get("embedding") or {}
    w = str(emb.get("why") or "")
    if emb.get("ok") is False and w.startswith(("embedding failed",
                                                "the query vector is zero")):
        return w[:200]
    return None


def _tool_unavailable(out) -> str | None:
    """A find_* result that failed for a reason that can pass (its envelope
    says `retryable`: NoRoom, a busy lane, an unreadable card), or None --
    a real answer, a real miss, or a failure retrying cannot fix."""
    if not isinstance(out, str) or not out.lstrip().startswith("{"):
        return None
    try:
        d = json.loads(out)
    except ValueError:
        return None
    if isinstance(d, dict) and d.get("ok") is False and d.get("retryable"):
        return (f"{d.get('error') or 'error'}: "
                f"{str(d.get('reason') or '')[:160]}")
    return None


def _inject_retry(raw: list[dict], keys: list[str],
                  account: str) -> tuple[str | None, dict]:
    """(key, meta) of a RETRYABLE injection on the message this request ends
    on -- the user turn it is speaking, or the tool result it ends on --
    else (None, {}). Only there: nothing after that message is cached, so
    deciding it again changes no prefix a slot holds."""
    if not raw or not keys:
        return None, {}
    last = raw[-1] if isinstance(raw[-1], dict) else {}
    if last.get("role") not in ("user", "tool"):
        return None, {}
    meta = nebari.ledger_meta(account, keys[-1], "inject")
    if meta.get("retry") and nebari.ledger_get(account, keys[-1],
                                               "inject") is not None:
        return keys[-1], meta
    return None, {}


def _index_of_user(messages: list, original: dict) -> int | None:
    """Where the client's last user turn sits in `messages` (the addendum
    may have added a system message in front, and restored image hops add
    messages before assistant turns)."""
    text = message_text.first_text(original)
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user" \
                and message_text.has_text(m) \
                and message_text.first_text(m).startswith(text or ""):
            return i
    return None


# THE CONVERSATION'S SKILL STATE (skill_select PER-TURN INJECTION and the
# craft offer): which skills were given and when, the last recall per area,
# the phase, the craft index and the tools withheld -- one ledger row per
# conversation (kind `skills`), changed under a per-conversation lock.
def skill_select_lock(account: str, lineage: str):
    import skill_select
    return skill_select.state_lock(account, lineage)


def _skill_state(account: str, lineage: str) -> dict:
    import skill_select
    try:
        return skill_select.load_state(account, lineage)
    except Exception:                                            # noqa: BLE001
        return skill_select.new_state()


def _save_skill_state(account: str, lineage: str, st: dict) -> None:
    import skill_select
    try:
        skill_select.save_state(account, lineage, st)
    except Exception as e:                                       # noqa: BLE001
        print(f"  skill state not saved: {type(e).__name__}: {e}", flush=True)


def _text_messages(messages: list) -> list[dict]:
    """The client's messages with every turn's text as a string (a list of
    text parts -- Pi, Responses input_text -- joined), as skill_select
    reads them."""
    return [dict(m, content=message_text.text_of(m, ""))
            if isinstance(m, dict) and isinstance(m.get("content"), list)
            and message_text.has_text(m)
            else (dict(m) if isinstance(m, dict) else m)
            for m in messages if isinstance(m, dict)]


def _mcp_offer(tier: dict, st: dict, continuing: bool) -> dict:
    """The conversation's MCP tools (mcp/mcp_host.py), decided on its FIRST
    request and kept by name in the skill state (`st["mcp"]`, saved by the
    caller): the switch `mcp_tools` (tiers.BEHAVIOURS: medium and up, a
    header or YAMADORI_MCP_TOOLS) and the host's offer at that moment. A
    request that continues a conversation with no decision gets none --
    adding tools then would change the system block the slot caches (the
    craft offer's rule). Returns x_yamadori.mcp's record."""
    on, src = tiers.behaviour_source(tier, "mcp_tools")
    got = st.get("mcp")
    if isinstance(got, dict) and "offered" in got:
        return {"switch": {"on": on, "source": src},
                "offered": [mcp_host.mcp_config.canonical(n)
                            for n in got.get("offered") or []],
                "kept": True,
                "why": got.get("why"), "calls": []}
    if continuing:
        rec = {"offered": [], "why": "the conversation started without an "
                                     "MCP tool offer"}
    elif not on:
        rec = {"offered": [], "why": f"switch mcp_tools off ({src})"}
    else:
        defs, why = mcp_host.offer()
        rec = {"offered": [t["function"]["name"] for t in defs],
               "why": why}
    st["mcp"] = rec
    return {"switch": {"on": on, "source": src}, "offered": rec["offered"],
            "kept": False, "why": rec["why"], "calls": []}


def _craft_offer(raw: list, sel: dict, route: dict, client_tools, account: str,
                 st: dict, *, inplace: bool, continuing: bool
                 ) -> tuple[dict, dict]:
    """The conversation's craft offer (skill_select.craft_offer), decided on
    its first request and kept."""
    import skill_select
    allowed = (bool((sel or {}).get("skills")) and not inplace
               and (route or {}).get("class") not in skill_select.SKIP_CLASSES)
    try:
        import skill_learn
        traffic = skill_learn.traffic_of(account or None)
    except Exception:                                            # noqa: BLE001
        traffic = "unknown"
    return skill_select.craft_offer(
        _text_messages(raw), (route or {}).get("class"),
        list(client_tools or []), st, allowed=allowed,
        continuing=continuing, traffic=traffic, account=account or None)


# The per-request context the real _skills_tail reads (the conversation and
# the user turn's ledger key): set by prepare, so the tail's signature --
# which the offline suites replace -- stays as it is.
_SKILL_CTX = threading.local()


def _skills_tail(messages: list[dict], sel: dict, route: dict,
                 client_tools: list[str] | None = None, account: str = ""
                 ) -> tuple[str, dict | None]:
    """The skills block for the last user turn, as the text to append
    (mcp/skill_select.py decides; the tier's `skills` flag allows it).

    A user turn, not the system block: the system prefix must stay
    byte-identical, and concept_seed.py records that guidance in the system
    message "was sometimes ignored" while the same text in the user message
    "couldn't" be. Recorded in the ledger with the turn, so every later
    request replays it byte for byte. A failure degrades to silence and says
    so (and is RETRYABLE, _skills_unavailable)."""
    try:
        import skill_select
        # skill_select reads a turn's text as a string: a list of text
        # parts (Pi, Responses input_text) is handed over as its text, and
        # the caller appends the tail as a part (message_text).
        base = [dict(m, content=message_text.text_of(m, ""))
                if isinstance(m, dict) and isinstance(m.get("content"), list)
                and message_text.has_text(m)
                else (dict(m) if isinstance(m, dict) else m)
                for m in messages]
        # The client's tool names: a skill may be keyed on the harness
        # (tools_any / tools_all / tools_none), never on our own tools.
        # THE CONVERSATION'S STATE (skill_select PER-TURN INJECTION): what
        # was given, when, and the last recall per area -- loaded, decided
        # and saved under the conversation's lock.
        ctx = getattr(_SKILL_CTX, "value", None) or {}
        lineage = ctx.get("lineage") or ""
        # MATCHING reads the client's own messages (prepare hands them over
        # as `raw`); the block is appended to `base`, what will be sent.
        own = (_text_messages(ctx["raw"]) if ctx.get("raw") is not None
               else [dict(m) for m in base if isinstance(m, dict)])
        with skill_select_lock(account, lineage), \
                _decider_turn(own, account, lineage, ctx.get("key"),
                              client_tools, sel, route) as turn:
            st = _skill_state(account, lineage) if lineage else None
            out, rec = skill_select.attach(
                base, own, sel,
                {"route": route, "client_tools": list(client_tools or []),
                 # whose traffic: skill learning learns from client
                 # traffic only
                 "account": account, "skill_state": st,
                 "key": ctx.get("key"),
                 "server_tools": ctx.get("server_tools"),
                 "plan_files": ctx.get("plan_files"),
                 # the conversation's compactions (skill_select: a moved
                 # count resets what was given)
                 "compactions": _compactions(account, lineage),
                 "decider": turn})
            new_st = (rec or {}).pop("_state", None)
            if lineage and new_st is not None:
                _save_skill_state(account, lineage, new_st)
        _decider_record(rec, turn)
    except Exception as e:                                       # noqa: BLE001
        print(f"  skills unavailable, continuing without: "
              f"{type(e).__name__}: {e}", flush=True)
        return "", {"on": False, "ids": [], "versions": [], "names": [],
                    "chars": 0,
                    "why": f"skills raised {type(e).__name__}: {e}"[:300]}
    li = next((i for i in range(len(base) - 1, -1, -1)
               if isinstance(base[i], dict) and base[i].get("role") == "user"),
              None)
    if li is None:
        return "", rec
    before = base[li].get("content") or ""
    after = (out[li].get("content") if li < len(out) else before) or ""
    tail = after[len(before):] if after.startswith(before) else ""
    return tail, rec


# The real tail, for a suite that replaces _skills_tail and wants it back.
_skills_tail_real = _skills_tail


def _server_tools(ours) -> list[str]:
    """Our server tools on main this request (SERVER-TOOL RECALL names only
    these): yama_think_deeply and yama_plan where deep.think_tool_offered
    put them there and no conflict withheld them."""
    have = set(ours or ())
    return [n for n in (deep.TOOL_NAME, deep.PLAN_TOOL_NAME) if n in have]


def _compactions(account: str, lineage: str) -> int | None:
    """How many compactions this conversation has had (progress's count,
    kept by _compaction_done), for the skill state's reset; None when it
    cannot be read."""
    if not lineage:
        return None
    try:
        return int(progress.load(account, lineage).get("compactions") or 0)
    except Exception:                                            # noqa: BLE001
        return None


def _plan_files(ours, messages: list) -> list[str] | None:
    """The newest plan's FILES, read from the ledger-restored messages (the
    yama_plan hop the client never saw), where main has yama_plan."""
    if deep.PLAN_TOOL_NAME not in set(ours or ()):
        return None
    try:
        import skill_select
        return skill_select.plan_files_of(_text_messages(messages)) or None
    except Exception:                                            # noqa: BLE001
        return None


def _skills_step(raw: list, sel: dict, route: dict, client_tools,
                 account: str, lineage: str, key: str | None,
                 server_tools: list[str] | None = None,
                 plan_files: list[str] | None = None
                 ) -> tuple[str, dict | None]:
    """The skills this AGENT STEP's newest evidence brings (skill_select
    PER-TURN INJECTION): a body the first time an area or API appears in
    what the step read or wrote, a recall line when an error or a phase
    change brings a given one back -- appended to the tool result the
    request ends on, decided once under its key and replayed from the
    ledger. A failure degrades to silence and says so."""
    try:
        import skill_select
        msgs = _text_messages(raw)
        with skill_select_lock(account, lineage), \
                _decider_turn(msgs, account, lineage, key,
                              client_tools, sel, route) as turn:
            st = _skill_state(account, lineage) if lineage else None
            text, rec = skill_select.attach_step(
                msgs, sel,
                {"route": route, "client_tools": list(client_tools or []),
                 "account": account, "skill_state": st, "key": key,
                 "server_tools": server_tools, "plan_files": plan_files,
                 "compactions": _compactions(account, lineage),
                 "decider": turn})
            new_st = (rec or {}).pop("_state", None)
            if lineage and new_st is not None:
                _save_skill_state(account, lineage, new_st)
        _decider_record(rec, turn)
        return text, rec
    except Exception as e:                                       # noqa: BLE001
        print(f"  step skills unavailable, continuing without: "
              f"{type(e).__name__}: {e}", flush=True)
        return "", {"on": False, "ids": [], "versions": [], "names": [],
                    "chars": 0,
                    "why": f"skills raised {type(e).__name__}: {e}"[:300]}


def _decider_turn(messages: list, account: str, lineage: str,
                  key: str | None, client_tools, sel: dict | None = None,
                  route: dict | None = None):
    """THE TURN DECIDER (mcp/decide_turn.py; operator 2026-09-27: the
    default for every judgment question): this request's fixed questions
    answered in one batch on the transient slot, before main's first
    generation, the Turn handed to the selector (ctx "decider": its pick()
    and confirm_packages() reuse the cached state) and its slot released
    when the block ends. Off (answers are the rules') outside the serving
    process, with YAMADORI_DECIDER=0, and where skills do not run (the
    selection did not allow them; a client's side call); never raises."""
    import decide_turn
    on = None
    try:
        import skill_select
        if not bool((sel or {}).get("skills")) or                 skill_select.route_class_of(None, sel or {}) in                 skill_select.SKIP_CLASSES or                 (route or {}).get("class") in skill_select.SKIP_CLASSES:
            on = False
    except Exception:                                            # noqa: BLE001
        on = False
    turn = decide_turn.Turn(messages, key=lineage or None, account=account,
                            request=key, client_tools=client_tools, on=on)
    try:
        turn.facts()
    except Exception as e:                                       # noqa: BLE001
        turn.failure = {"code": type(e).__name__, "situation": str(e)[:160],
                        "retryable": False}
    return turn


def _decider_record(rec: dict | None, turn) -> None:
    """The Turn's record on the skills record, x_yamadori.skills.turn
    (`decider` there is the selector's own: the decider's name)."""
    if isinstance(rec, dict) and turn is not None:
        try:
            rec["turn"] = turn.record()
        except Exception:                                        # noqa: BLE001
            pass


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
                         state: dict | None,
                         unavailable: list | None = None) -> str:
    """The definitions tail, or "". `unavailable` collects why a lookup that
    should have run could not (a raise, a retryable failure envelope): the
    caller then records the decision as RETRYABLE, never as final."""
    if (route or {}).get("class") != "library_question" \
            or not tier.get("retrieval") or not (gate or {}).get("offer") \
            or sel.get("investigate"):
        return ""
    miss = unavailable if unavailable is not None else []
    try:
        q, _ctx, _speaking = selection.question_of(messages)
        held = selection.defined_symbols(q, dbs)
    except Exception as e:                                       # noqa: BLE001
        miss.append(f"symbol lookup raised {type(e).__name__}: {e}"[:200])
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
        except Exception as e:                                   # noqa: BLE001
            miss.append(f"find_definition_opt({n}) raised "
                        f"{type(e).__name__}: {e}"[:200])
            continue
        why = _tool_unavailable(out)
        if why:
            miss.append(f"find_definition_opt({n}): {why}")
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
# (find_definition_opt on that package's own index), each WHOLE. Decided once
# per message and recorded in the ledger under that message's key
# (ledger_restore replays it, byte for byte); what was covered is recorded
# per conversation, so each name is injected once.
# THE ONE BOUND IS THE WINDOW (2026-09-27, docs/CONSTANTS-AUDIT.md): an
# injection never takes the request past what the main share leaves after
# the request itself and the generation floor (_injection_room); a
# definition that does not fit is left uncovered, to be tried again on a
# later message. REMOVED with the invented caps USE_MAX_CHARS 3,000,
# USE_MAX_NAMES 4, USE_MAX_CHARS_EACH 1,000 and USE_CONVERSATION_MAX_CHARS
# 12,000: the package OVERVIEW for a package used before any name from it
# (USE_OVERVIEW_ITEMS 20 exports, lines cut at 140, test/spec/internal paths
# skipped) -- it cannot exist without a picked number, so it is gone.
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


def _injection_room(messages: list[dict], tools=None,
                    already: int = 0) -> int:
    """Characters an injection may add to this request: the main share
    (window_limit's) less the request's high estimate and the generation
    floor at a thinking tier, in the high estimate's own unit (3 characters
    a token), less `already` characters decided for it. A real constraint:
    the addition must not push the prompt past the window."""
    est = high_estimate({"messages": messages, "tools": tools or None})
    left = (int(tiers._shares()["main"]) - est - tiers.A_MIN
            - tiers.MIN_THINKING)
    return max(0, left * 3 - int(already))


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
                 state: dict | None = None, room: int | None = None
                 ) -> tuple[str, dict]:
    """(the text to append to the message this request ends on, the record).
    Updates the conversation's covered set in the ledger; the caller records
    the text under the message's key. `room`: the characters the window
    leaves (_injection_room); None computes it from `messages`."""
    import domains
    rec: dict = {"packages": [], "names": [], "chars": 0, "versions": {}}
    try:
        held = domains.held_sources()
        uses = library_uses(messages)
    except Exception as e:                                       # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
        rec["unavailable"] = [f"reading the held packages raised "
                              f"{rec['error']}"]
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
    if room is None:
        room = _injection_room(messages)
    rec["room"] = room
    parts: list[str] = []
    size = len(USE_HEAD)
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
        for n in names:
            try:
                out = _run_on_package(db, "find_definition_opt",
                                      {"symbol": n})
            except Exception as e:                               # noqa: BLE001
                rec.setdefault("unavailable", []).append(
                    f"find_definition_opt({pkg}:{n}) raised "
                    f"{type(e).__name__}: {e}"[:200])
                continue              # not covered: decided again later
            why = _tool_unavailable(out)
            if why:
                rec.setdefault("unavailable", []).append(
                    f"find_definition_opt({pkg}:{n}): {why}")
                continue
            if not out or out.lstrip().startswith("{") \
                    or packages._is_empty(out):
                covered.append(n)
                continue
            part = f"== {pkg}@{vtag}: {n} ==\n{out}"
            if size + len(part) + 2 > room:
                # Does not fit what the window leaves: not covered, so a
                # later message tries it again.
                rec.setdefault("did_not_fit", []).append(f"{pkg}:{n}")
                continue
            covered.append(n)
            parts.append(part)
            size += len(part) + 2
            rec["names"].append(f"{pkg}:{n}")
            if pkg not in rec["packages"]:
                rec["packages"].append(pkg)
    # A lookup that could not run (rec["unavailable"]) makes this decision
    # RETRYABLE (prepare): nothing is marked covered, so the retry decides
    # the whole of it again.
    commit = not rec.get("unavailable")
    if not parts:
        if commit:
            nebari.ledger_put(account, lineage, key, "libuse",
                              json.dumps(done))
        return "", rec
    text = USE_HEAD + "\n\n".join(parts)
    if commit:
        nebari.ledger_put(account, lineage, key, "libuse", json.dumps(done))
    rec["chars"] = len(text)
    return text, rec


def _work_log_block(state: dict | None) -> str:
    """The conversation's work log (mcp/rings.py), for the first user turn
    after a compaction. The proxy writes the log itself (_log_turn); the
    model no longer has record_step / read_rings. (The files-read listing
    #54 appended here was removed 2026-09-27 with the unchanged-read line:
    task-targeted steering; skills are the channel.)"""
    try:
        import rings
        text = rings.read(session_lineage(state), limit=40)
    except Exception:                                            # noqa: BLE001
        text = ""
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
    # The compaction call's own system prompt (Pi's summariser; Hermes and
    # OpenCode send none), and whose form this is (x_yamadori.compaction
    # .harness: hermes | pi | opencode | None).
    own_system = selection._text(messages[0]) if len(messages) > 1 and \
        messages[0].get("role") == "system" else ""
    rec["harness"] = (parsed or {}).get("harness") or (
        None if inplace else compaction.harness_of(text, own_system))
    if inplace:
        # THE LEDGER'S RENDERING (2026-09-24). prepare() already put back
        # everything this proxy added to the conversation -- injections, a
        # fixed call's content, hidden hops, and (switch restore_reasoning,
        # 2026-09-27) the slot's own past reasoning -- so the request's
        # own messages render as the slot holds them. The stored prompt is
        # the CHECK: when it (and the answer after it) is a prefix of this
        # rendering, the compaction is served as prepared; when not, the old
        # splice replaces the resent history with it.
        key = out["_slot"].get("key")
        _note_compaction(account, lineage=key)
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
                   "can map (none of Hermes' TURNS TO SUMMARIZE, Pi's "
                   "<conversation> / # Conversation or OpenCode's "
                   "<conversation> blocks of [Role]: records)")
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
                                              prev, system=own_system))
            out["tools"] = e["upstream"]["tools"]
            out["_slot"] = dict(out["_slot"], key=e["key"], transient=False,
                                prefix=False)
            # WHOSE compaction this is, exactly: the conversation it maps to
            # (the recency guess in session_context missed it after a
            # 2,903 s turn, #38).
            _note_compaction(account, lineage=e["key"])
            # The conversation's OWN session line on the summary (#41): the
            # continuation carries the summary, so it keeps the conversation
            # by its id. Only for a compaction mapped to it exactly, and only
            # when the id is ours (a client with its own id sends it again).
            st = nebari.load(e["key"])
            if st.get("session_source") == "token" and st.get("session_id"):
                out["_session_lead"] = session_id.line(st["session_id"])
                rec["session_line"] = True
                out.setdefault("_ledger", {})["conversation"] = {
                    "id": st["session_id"], "source": "compaction_map",
                    "why": "the conversation this compaction maps to; its "
                           "summary carries the id on its line"}
            stored = e
            rec.update(mode="rewritten", why=m["why"], mapped=m["matched"],
                       span=[m["first"], m["last"]], conversation=e["key"][:8])
        else:
            rec.update(mode="as_sent", why=why)
    src = stored["upstream"] if stored else out
    if not stored and (out.get("_utility") or {}).get("utility"):
        # Nothing stored to take the conversation's effort from, and the
        # utility rule set `minimal`: the effort the CLIENT sent is the
        # conversation's (compaction.client_fields; AGENTS.md "A compaction
        # THINKS at the conversation's own effort"). No effort sent: off.
        eff = body.get("reasoning_effort") or (
            body["reasoning"].get("effort")
            if isinstance(body.get("reasoning"), dict) else None)
        src = compaction.client_fields(out.get("_tier_requested")
                                       if eff else None)
    client_max = (body.get("max_tokens") or body.get("max_completion_tokens")
                  or (parsed or {}).get("target_tokens"))
    comp = tiers.compaction_budget(
        client_max, tiers.estimate_prompt_tokens({"messages": out["messages"],
                                                  "tools": out.get("tools")}),
        helper_active=admission.helper_active())
    fields = compaction.prefix_fields(src, comp["answer"],
                                      comp["thinking_tokens"])
    rec["thinking"] = fields.pop("_thinking")
    thinks = bool(fields["enable_thinking"])
    if not thinks:
        for k in ("reasoning_effort", "reasoning_budget_tokens",
                  "reasoning_budget_message", "reasoning_budget_nudge",
                  "reasoning_budget_nudge_at"):
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


# _budget_note / _budget_notice REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md):
# they appended "[no answer: the model reached its token limit ...]" or
# "[answer cut off at the token limit ...]" to the client's content, where it
# entered the client's stored history. finish_reason "length" carries the
# fact (the OpenAI contract); the content is what was generated.


# What the landing asks for. Shared by both tool loops so they land the same.
LANDING_PROMPT = ("Stop searching and answer now from what you have. If the "
                  "searches found nothing, answer from your own knowledge -- "
                  "that is expected here and is not a failure. Give the "
                  "complete answer in full.")


# THE CLIENT'S OWN PROMPT (C1, docs/OPENAI-CONFORMANCE.md; 2026-09-25,
# "before V0 fixes should land"). A conversation past its window used to be
# LANDED: the client's tools withdrawn and LANDING_PROMPT -- a user message
# the client never sent -- appended, on a chars/3 estimate that fired before
# the advertised context_length. An agent loop saw its tools vanish and the
# task end, and no client ever saw `context_length_exceeded`, so none
# compacted. Now the request as the proxy would send it (the client's
# messages plus what the ledger and prepare put back, and the tools) is
# measured BEFORE anything runs, and a prompt that does not fit is refused:
# 400 context_length_exceeded, in OpenAI's wording (mcp/api_errors.py), with
# the real count. The landing stays only for OUR hops (context_full, hop 1+).
#
# THE LIMIT is the window /v1/models advertises (catalog.context_window: the
# main share, budget.budgets()["main"]) -- for a compaction, the window its
# budget record was given (tiers.compaction_budget: the pool less a running
# second brain's share). It covers prompt AND generation, as OpenAI's
# context_length does, so a prompt is refused when it leaves less than
# GENERATION_FLOOR: A_MIN (the smallest answer allowance the proxy sends)
# plus MIN_THINKING when the request thinks -- 3,072 tokens at a thinking
# tier, 2,048 without. A CHOICE (the smallest generation the budget rule
# ever sends), not a measurement of what an answer needs.
#
# THE COUNT is llama-server's own: the chat template renders the request
# (/apply-template) and the model's tokenizer counts it (/tokenize), the same
# two calls the warm already makes (warm_prompt, _prefill_floor). It costs
# two round trips, so it is made only NEAR THE EDGE: when a high estimate
# (chars / 3 over messages and tools, plus half the extra UTF-8 bytes, so
# text of non-ASCII scripts is not under-counted) says the prompt might not
# fit. A prompt the estimate clears cannot overflow by more than the
# estimate's error. When the count cannot be made the request is sent, and
# llama-server's own exceed_context_size_error (at the slot's n_ctx) is
# mapped to the same 400.
COUNT_TIMEOUT = float(os.environ.get("YAMADORI_COUNT_TIMEOUT", "30"))


def generation_floor(payload: dict) -> int:
    thinks = bool(payload.get("enable_thinking", True)) and \
        "reasoning_budget_tokens" in payload
    return tiers.A_MIN + (tiers.MIN_THINKING if thinks else 0)


def window_limit(payload: dict) -> int:
    comp = payload.get("_compaction") or {}
    if comp.get("window"):
        return int(comp["window"])
    return int(tiers._shares()["main"])


def high_estimate(payload: dict, messages: list | None = None) -> int:
    """chars / 3 over messages and tools, plus half the bytes non-ASCII text
    adds in UTF-8 (a CJK character is ~1 token, not 1/3)."""
    n = extra = 0
    for key, v in (("messages", messages if messages is not None
                    else payload.get("messages")),
                   ("tools", payload.get("tools"))):
        if key == "messages" and v:
            # images passed to the main model: the projector's tokens for each
            # (image_input.image_tokens), never their base64
            v, image_tok = image_input.without_image_bytes(v)
            n += 3 * image_tok
        if not v:
            continue
        try:
            s = json.dumps(v, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(v)
        n += len(s)
        extra += len(s.encode("utf-8", "replace")) - len(s)
    return n // 3 + extra // 2


def count_prompt_tokens(payload: dict, messages: list | None = None,
                        timeout: float | None = None) -> int | None:
    """The request's prompt in the model's own tokens: the served template's
    render, tokenized by the model server. None when either call gives no
    answer."""
    model = payload.get("model") or "bonsai"
    fields = {k: payload[k] for k in ("tools", "chat_template_kwargs",
                                      "enable_thinking", "reasoning_effort")
              if k in payload}
    t = COUNT_TIMEOUT if timeout is None else timeout
    # Image parts are counted as the projector counts them (image_input.
    # image_tokens, from each image's own size), not rendered: the template
    # would put a media marker there, whose few tokens are not the image's.
    msgs, image_tok = image_input.without_image_bytes(
        messages if messages is not None else payload.get("messages"))
    prompt = _upstream_json(f"/upstream/{model}/apply-template",
                            dict(fields, messages=msgs),
                            timeout=t).get("prompt")
    if not isinstance(prompt, str):
        return None
    toks = _upstream_json(f"/upstream/{model}/tokenize",
                          {"content": prompt, "add_special": True},
                          timeout=t).get("tokens")
    return (len(toks) + image_tok if isinstance(toks, list) else None)


def check_client_prompt(payload: dict) -> dict:
    """Refuse (api_errors.context_length_exceeded) a request whose prompt
    leaves less than the generation floor in its window; else the record for
    x_yamadori.context: {limit, floor, estimate, tokens, counted, why}."""
    limit, floor = window_limit(payload), generation_floor(payload)
    est = high_estimate(payload)
    rec = {"limit": limit, "floor": floor, "estimate": est, "tokens": None,
           "counted": False}
    if est + floor < limit:
        rec["why"] = "fits by the high estimate; not counted"
        return rec
    try:
        n = count_prompt_tokens(payload)
    except cancel.Cancelled:
        raise
    except Exception as e:                                       # noqa: BLE001
        n = None
        rec["count_error"] = f"{type(e).__name__}: {e}"[:200]
    if n is None:
        rec["why"] = ("near the edge and the model server could not count "
                      "it: sent; its own context limit decides")
        print(f"  context: ~{est} tokens (estimate) near the {limit}-token "
              f"window and not counted ({rec.get('count_error')}); sent",
              flush=True)
        return rec
    rec.update(tokens=n, counted=True)
    if n + floor > limit:
        print(f"  context: {n} tokens + {floor} to answer > the {limit}-token "
              f"window: refused (context_length_exceeded)", flush=True)
        raise api_errors.context_length_exceeded(n, limit, floor)
    rec["why"] = "counted near the edge: fits"
    return rec


def fit_window(payload: dict, n_prompt: int, limit: int) -> dict:
    """Hop 0's token fields cut to the window: max_tokens never asks for
    more than the prompt leaves (the budget rule adds MIN_THINKING and the
    answer allowance on top of an estimate, which could), thinking keeps
    what the answer allowance leaves. Unchanged when it already fits."""
    room = int(limit) - int(n_prompt)
    mt = int(payload.get("max_tokens") or 0)
    if not mt or mt <= room:
        return payload
    out = dict(payload, max_tokens=max(room, 1))
    rbt = payload.get("reasoning_budget_tokens")
    if rbt is not None:
        ans = int(payload.get("_answer") or tiers.A_MIN)
        out["reasoning_budget_tokens"] = max(1, min(
            int(rbt), max(room - ans, min(tiers.MIN_THINKING, room // 2))))
    out["_fit_window"] = {"room": room, "max_tokens_was": mt}
    return out


def context_full(payload: dict, convo: list[dict], role: str = "main",
                 share_n: int = 1) -> bool:
    """Would one more hop no longer fit in this request's share of the pool?

    OUR HOPS ONLY (hop 1+, after one of our tools ran; 2026-09-25): the
    client's own prompt is measured before the turn starts
    (check_client_prompt) and refused with context_length_exceeded, never
    landed.

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
    elif why == "image_guard":
        print(f"  image guard: image data in a tool call again after "
              f"{tool_code.IMAGE_REGENERATIONS} regenerations; tools "
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
# hidden hops included, _run_turn); on later turns the ledger restores it
# with the turn (PAST REASONING IS RESTORED, 2026-09-27; with switch
# restore_reasoning off, whatever the client echoes) -- and the user sees
# only the conclusion -- main's visible
# answer, which opens with the fold-back phrase (shomen.opening). A run that
# SEARCHED crosses under FINDINGS_HEAD; one that made no search under
# REASONING_HEAD, which says no source was checked (operator, 2026-09-23).
FINDINGS_HEAD = ("I investigated this in the library source before "
                 "answering. A fact ending in path:line was read "
                 "there.\n\n")
REASONING_HEAD = ("I thought this through before answering, without "
                  "searching: reasoning, no sources checked. Nothing below "
                  "was read from a file.\n\n")
# THE HAND-OFF'S LAST LINE (live gate 2026-09-24). After a 28-search, 638 s
# investigation the visible answer was one line -- "After thinking deeply,
# the deprecation chain is pinned to r179 with the hand-off above." -- while
# the hand-off sat in the prefilled REASONING, which the user never sees. So
# the prefilled reasoning ends by saying who sees what, and that the answer
# it leads into carries the findings itself. Positive wording on purpose
# (AGENTS.md "Prompting this model"). Ends on a letter or a period, never a
# space (rule 4). The wording is a choice, unmeasured beyond the live check.
# REVISED after the second live run the same night: 1,337 characters of
# findings, but opened "the answer below was assembled from the hand-off
# facts" and repeated "**After thinking deeply,**" as a heading. The two
# prohibitions name exactly those observed failures (AGENTS.md allows two).
FOLD_BACK_TAIL = ("\n\nThe user sees only my answer, not this thinking and "
                  "not this hand-off. So my answer stands on its own: it "
                  "states the findings in full -- each fact it relies on "
                  "with its path:line and the source lines that show it, "
                  "and the next step. It opens with "
                  "\"After thinking deeply,\" and that same sentence goes "
                  "straight on into the answer itself. It never mentions "
                  "the hand-off, the investigation or anything \"above\", "
                  "and it never writes the opening phrase a second time.")
# A MACHINE-BUILT hand-off (the second brain wrote no conclusion) gets its
# own tail (deploy check 2026-09-26, handle d28941fb): under FOLD_BACK_TAIL
# main was told to state "the findings in full" when there were none, wrote
# "The hand-off didn't finish, so I'll continue from its NEXT STEP" into its
# visible content and told the user "the deep-thinking pass ... cut off
# before finishing". This one says what IS there -- a search log, no
# conclusion -- and what the answer does with it. Same two prohibitions, the
# observed leak's words added to the first. UNMEASURED WORDING. (2026-09-27:
# "from the source lines shown here" dropped -- the machine-built hand-off
# carries no source lines since the MACHINE EVIDENCE excerpts were removed.)
FOLD_BACK_TAIL_MACHINE = (
    "\n\nThe user sees only my answer, not this thinking and not this search "
    "log. No conclusion was written here, so my answer works the question "
    "out itself from what I know and acts on it. It opens with \"After "
    "thinking "
    "deeply,\" and that same sentence goes straight on into the answer "
    "itself. It never mentions the search log, the investigation, deep "
    "thinking or anything \"above\", and it never writes the opening phrase "
    "a second time.")
PLAN_TAIL = ("\n\nThe user sees only my answer, not this plan. So my answer "
             "says in a sentence or two what I will do and in what order, "
             "then starts on the first step.")
# THE PLAN AS A TOOL RESULT (operator, 2026-09-27; pagoda-h2): the initial
# prompt's plan and a yama_plan call both come back as the yama_plan TOOL
# RESULT of a hidden hop, and main goes on acting from it. PLAN_HEAD /
# PLAN_TAIL above were written as main's own prefilled reasoning ("I
# planned ..."); these are the result's own words: what it is, who sees it,
# and the next action. No prohibition. UNMEASURED WORDING, a CHOICE.
# plan/3 (operator, 2026-09-28, after pagoda-h6: "plans don't sow doubt"):
# the head no longer says which decisions were "not checked" -- the
# decisions are made, and the tail says to carry them out.
PLAN_RESULT_HEAD = ("The plan for this task, written by a second model on "
                    "the Yamadori server. Its decisions are made.\n\n")
PLAN_RESULT_TAIL = ("\n\nThe user sees neither this result nor the plan. "
                    "Carry it out step by step with your own tools, "
                    "starting with the first ORDER step now.")
# A hand-off delivered as an inserted yama_think_deeply result (switch
# deep_tool_hop, OFF by default): no prefilled "After thinking deeply,"
# follows it, so the result itself says who sees what. UNMEASURED WORDING.
THINK_RESULT_TAIL = ("\n\nThe user sees neither this result nor your "
                     "thinking. Your next step acts on the NEXT STEP above, "
                     "and a reply to the user states the findings it relies "
                     "on in full, each with its path:line.")
THINK_RESULT_TAIL_MACHINE = (
    "\n\nThe user sees neither this result nor your thinking. No conclusion "
    "was written here: work the question out from the source lines shown, "
    "if any, citing each by its path:line, and act on it.")


# THE SERVER-TOOL TRIGGERS' DIRECTIVE (operator, 2026-09-27, after pagoda-h5:
# "this model isn't taking gentle hints, we need to tell it what to do ... at
# the right time"). When the proxy runs deep thinking or a plan FOR the model
# (deep.decide kind "auto": a package probe or scratch test, a new piece, a
# finished plan, a build request after an answer), the result is inserted as
# a hidden yama_think_deeply / yama_plan hop and main's turn after it OPENS
# with one fixed line in the model's own voice saying what it does next --
# generic, never task-specific, never "server tool". Only on the request the trigger fired: the line nearest the
# generation, never repeated. Each ends on a LETTER (AGENTS.md's prefill
# rule: "ends on a letter, or on an ending measured safe ... never a
# space"). THE LAST LINE OF MAIN'S REASONING, the think block LEFT OPEN
# (coordinator, 2026-09-27: "so the model keeps thinking toward the action
# and then acts, instead of writing code with zero reasoning on that step"):
# a prefill whose reasoning_content is the line and whose content is empty
# (directive_prefill). That llama-server leaves the block open for such a
# prefill is INFERRED from STEP 0's probe (docs/SELF-IMPROVEMENT-PLAN.md: a
# trailing assistant message with reasoning and tool calls, no text,
# rendered through /apply-template as `<think>` + the reasoning, and
# stopped), not measured for a reasoning-only prefill -- confirm on the
# live /apply-template before trusting it. The reasoning is the slot's own text: the client is
# streamed it as reasoning (llama-server re-sends a prefill first), the
# ledger restores it (restore_reasoning). The wording follows the
# operator's examples; UNMEASURED.
#
# REWORDED 2026-09-28 (operator, after pagoda-h6: "don't say 'I have it from
# source' or anything that confuses the model into overthinking its way out
# of using the findings"). The 2026-09-27 lines said "I have <P>'s API from
# its own source now, so I'll stop reading its files ..." and after every
# one the model kept probing node_modules and writing scratch scripts: a
# line that names WHERE the findings came from (and which files to stop
# reading) gives the reasoning something to weigh -- where, how, whether to
# trust it, whether to check. Each line now says only that the result just
# given answers the question, to USE it, and the next concrete action: no
# provenance, no package name, no prohibition. One line per job (the
# investigate triggers -- probe, scratch -- share one; the plan triggers --
# next_piece, plan_done, implement -- share one). The operator's candidates,
# as given.
AUTO_DIRECTIVE_THINK = ("The findings above answer this, so I'll use them "
                        "and write the code now")
AUTO_DIRECTIVE_PLAN = "Next I'll do the first step of this plan"


def directive_prefill(line: str) -> dict:
    """A directive as the opening of main's REASONING, the think block left
    open: no content, so the model thinks on from the line, then acts."""
    return {"role": "assistant", "content": "", "reasoning_content": line}


def auto_directive(auto: dict, machine: bool = False) -> str:
    """The directive line for a fired server-tool trigger (deep.decide's
    `auto` record). `machine`: the run wrote no conclusion -- the same line
    (a CHOICE: the model reads the result either way)."""
    if auto.get("job") == "plan":
        return AUTO_DIRECTIVE_PLAN
    return AUTO_DIRECTIVE_THINK


def synthetic_hop(name: str, args: dict, result: str,
                  key: str | None = None) -> list[dict]:
    """A call of ours the PROXY makes on main's behalf, and its result: the
    assistant turn that calls `name` with `args` (no content, no reasoning)
    and the tool turn that answers it -- a hidden hop, recorded and replayed
    by the ledger like one the model made (_run_turn). The id is not our
    session form (session_id.carry) and never reaches the client."""
    import hashlib
    h = hashlib.sha1(f"{name}\x00{key or ''}\x00{result}".encode(
        "utf-8")).hexdigest()[:12]
    cid = f"call_{name}_{h}"
    return [{"role": "assistant", "content": "", "reasoning_content": "",
             "tool_calls": [{"id": cid, "type": "function", "function": {
                 "name": name,
                 "arguments": json.dumps(args, ensure_ascii=False,
                                         sort_keys=True)}}]},
            {"role": "tool", "tool_call_id": cid, "content": result}]
# x_yamadori.fold_back_answer: how many characters of answer followed the
# deep-thinking opening -- a number, recorded. REMOVED 2026-09-27
# (docs/CONSTANTS-AUDIT.md): FOLD_BACK_MIN_CHARS (200, a log threshold), the
# _ABOVE_REF / _HIDDEN_REF regexes that flagged an answer pointing at text
# the user never saw (hand-written from two live-gate answers), and
# dedup_opening, which removed later copies of "After thinking deeply," from
# the model's answer (one live-gate run). The answer is delivered as written.


def fold_back_answer(content: str) -> dict:
    """{chars_after_opening, opened} for an answer that opened with the
    deep-thinking fold-back ("... After thinking deeply,")."""
    phrase = shomen.PHRASES["investigate"]
    text = content or ""
    i = text.find(phrase)
    after = text[i + len(phrase):] if i >= 0 else text
    return {"chars_after_opening": len(after.strip()), "opened": i >= 0}


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
    request's. The model's own trigger, yama_think_deeply, is _think_deeply.

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
               and trig.get("kind") in ("struggle", "kickoff", "area",
                                        "auto"))
    auto = trig.get("auto") if trig.get("kind") == "auto" else None
    t_run = time.time()
    job = (trig.get("job") or "investigate") if own else "investigate"
    if own:
        q, ctx = trig["question"], trig.get("context") or ""
    else:
        # The messages as the text model reads them: an attached image is
        # its placeholder, so the question carries the id yama_describe_image
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
    # THE REQUEST'S CANCELLATION (mcp/cancel.py, #39): the job's upstream
    # sockets register with it, so a client that goes away stops the job and
    # frees the helper lane. A caller with no token (complete(), a test) gets
    # one of its own, cancelled if this generator is closed early.
    tok = cancel.current() or cancel.Token()

    def _work():
        try:
            with cancel.bound(tok):
                box["res"] = _job()
        except Exception as e:                                   # noqa: BLE001
            box["err"] = f"{type(e).__name__}: {e}"
        finally:
            trace_q.put(None)

    def _job():
        return shomen.run(
            job, question=q,
            tools=deep_thinking_tools((state or {}).get("_attached")),
            run_tool=_watched, context=ctx,
            on_think=lambda t: trace_q.put(t.rstrip()),
            # The request's tier (medium on `xhigh`, xhigh on `max`); an
            # effort override (the domain benchmark's header) wins.
            tier=tier.get("name") or "max",
            effort=(tier.get("effort") if "effort" in
                    (tier.get("overridden") or []) else None),
            seed=seed,
            # #52: the research seed line, behind its switch
            # (tiers.BEHAVIOURS).
            seed_frame=tiers.behaviour(tier, "seed_frame"),
            # #60: the plan job's tools, budget and prompt switches.
            plan_switches=tiers.plan_switches(tier),
            # A citation of the bound repository's own file is read from
            # it (shomen THE EVIDENCE); with none bound, held packages
            # only.
            **({"source_root": root} if root else {}))

    # The investigation runs in a thread and its tool calls are streamed AS
    # THEY HAPPEN: the searches ARE the thinking. A thread because it blocks
    # for minutes and this is a generator.
    #
    # HEARTBEATS (#39): between trace lines this yields None every HEARTBEAT
    # seconds, which _run_turn sends as an empty delta. The helper's first
    # line came 350 s in and the last ~900 s before the end on the Octopus
    # run, and Hermes kills a stream that sends no CHUNK for its stale
    # timeout (900 s for a local endpoint, agent/chat_completion_helpers.py
    # _local_stream_stale_timeout_default; an SSE comment is not a chunk to
    # the OpenAI SDK, an empty delta is). It also gives a closed stream a
    # yield point to be closed at, within HEARTBEAT seconds.
    th = _threading.Thread(target=_work, daemon=True)
    th.start()
    try:
        while True:
            try:
                line = trace_q.get(timeout=HEARTBEAT)
            except _queue.Empty:
                yield None
                continue
            if line is None:
                break
            yield line
    except GeneratorExit:
        # The stream was closed mid-investigation: stop the job (its
        # sockets are shut, shomen.run fails fast and frees the lane).
        tok.cancel("the stream was closed during deep thinking")
        print("  deep thinking cancelled: the stream was closed", flush=True)
        raise
    th.join(timeout=5)
    if tok.cancelled:
        raise cancel.Cancelled(tok.why)
    res = box.get("res") or {}
    if res.get("skipped"):
        # Reported, not swallowed: an investigation that quietly did not
        # happen looks exactly like one that found nothing.
        rec["why"] = "the helper lane is busy with another request"
        rec["skipped"] = True
        print(f"  deep thinking skipped: {rec['why']}", flush=True)
        return rec
    text, stats, searches = _handoff_of(res, q, box.get("err"), rec, seed,
                                        root)
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
    machine = bool((stats or {}).get("machine_built"))
    # HOW IT REACHES MAIN. The initial prompt's PLAN always, and any other
    # run when the deep_tool_hop switch is on (tiers.BEHAVIOURS, OFF by
    # default; operator deciding, 2026-09-27): as an INSERTED CALL of ours
    # and its result -- yama_plan, or yama_think_deeply -- a hidden hop
    # before main's first generation, as if main had called it. Main's
    # generation continues from a tool result: no visible opening, no
    # prefilled reasoning, and it can go straight on to a tool call
    # (pagoda-h2, 2026-09-27: the plan prefilled under "After thinking
    # deeply," was continued as prose about the second model, finish=stop,
    # no call, and the harness ended the run). Otherwise, as before: the
    # hand-off prefilled as main's reasoning under the fold-back opening.
    # A run whose result would not leave main room in its window is the
    # window check's to handle, as for any prompt: hop 0's max_tokens is cut
    # to what is left (fit_window), a `length` finish is a budget event, and
    # the client's next request is refused context_length_exceeded
    # (check_client_prompt), which a harness compacts on.
    if job == "plan" or auto is not None or \
            tiers.behaviour(tier, "deep_tool_hop"):
        name = deep.PLAN_TOOL_NAME if job == "plan" else deep.TOOL_NAME
        result = (PLAN_RESULT_HEAD + text.rstrip() + PLAN_RESULT_TAIL
                  if job == "plan" else
                  head + text.rstrip() + (THINK_RESULT_TAIL_MACHINE if machine
                                          else THINK_RESULT_TAIL))
        payload["_pre_hops"] = synthetic_hop(
            name, deep.synthetic_args(trig, job), result,
            (payload.get("_ledger") or {}).get("turn_key"))
        payload.setdefault("_fold_back", []).append(
            {"job": job, "phrase": None, "into": "tool_result", "tool": name,
             "trigger": rec["trigger"],
             "seeds": [s.get("word") for s in seeds]})
        rec["into"] = "tool_result"
        rec["tool"] = name
        if auto is not None:
            # THE DIRECTIVE (operator, 2026-09-27): main's turn after the
            # inserted result opens with one fixed line in its own voice
            # saying what it does next -- on this request only, the line
            # nearest the generation.
            line = auto_directive(auto, machine)
            payload["_prefill"] = directive_prefill(line)
            payload["_prefill_auto"] = True
            auto.update(seconds=round(time.time() - t_run, 1),
                        delivered_as="tool_result", directive=line)
    else:
        payload["_prefill"] = {"role": "assistant",
                               "reasoning_content": head + text.rstrip() + (
                                   PLAN_TAIL if job == "plan" else
                                   FOLD_BACK_TAIL_MACHINE if machine else
                                   FOLD_BACK_TAIL),
                               "content": shomen.opening(seeds)}
        payload.setdefault("_fold_back", []).append(
            {"job": job, "phrase": "investigate", "into": "prefill",
             "trigger": rec["trigger"],
             "seeds": [s.get("word") for s in seeds]})
        rec["into"] = "prefill"
    rec["injected"] = True
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
        "machine_built",
        # Did the helper write the four sections? The rate of this is how
        # well the format is followed -- measure it before relying on it.
        "structured", "plan",
        # THE EVIDENCE (shomen): excerpts inlined from the held source, and
        # citations removed because the verifier could not read them.
        "excerpts", "excerpt_chars", "excerpts_skipped",
        "citations_removed")}


def _handoff_of(res: dict, q: str, err: str | None, rec: dict,
                seed: dict | None, root: str | None = None
                ) -> tuple[str, dict | None, int]:
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
    if res.get("error") and not err:
        # Why the run wrote no conclusion (shomen._fail), in the record and
        # on the log line: "MACHINE-BUILT" alone left the 2026-09-26 cause
        # to be dug out of the ledger.
        rec.setdefault("why", "no conclusion: " + str(res["error"])[:300])
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
                                         rec["handle"] or "", root=root)
        else:
            text, stats = shomen.machine_handoff(
                q, tr.get("trace") or [], seen,
                rec.get("why") or res.get("error") or "no hand-off came back",
                rec["handle"] or "", root=root)
    return text, stats, searches


# THINK_DEEPLY, THE MODEL-CHOSEN TRIGGER (Phase 0.6, operator 2026-09-24).
# Main calls it; the proxy runs the question through the one second-brain
# runner (shomen.run("investigate")) in the helper lane and returns the
# hand-off as the TOOL RESULT. The call and its result are a hidden hop: the
# client never sees them, and the ledger records and replays them
# (ledger_record_turn / ledger_restore) so the next request extends the
# slot. The next hop is prefilled: reasoning deep.think_reasoning(concluded)
# (THINK_REASONING, or THINK_REASONING_MACHINE after a machine-built result),
# content
# the fold-back opening (seed line + "After thinking deeply,").
THINK_CALLS_PER_REQUEST = 1


# ALREADY_THOUGHT: a yama_think_deeply call refused because deep thinking
# already ran for this reply (THINK_CALLS_PER_REQUEST; one helper lane). A
# failure return carries the situation, whether it is retryable, and a
# remedy with an owner (AGENTS.md "Failure returns carry the next step"):
# here, the facts -- that it ran, when, whether it concluded, how many
# searches, and where its result already is in this reply. The steering the
# return used to carry ("make the next call now" / "answer now", "cite each
# by its path:line", the first 6 excerpt labels and 6-12 files read) was
# REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md: unmeasured wording from one
# deploy check, n=1). Whether a second run should be ALLOWED after a failed
# one is the operator's call.
def already_thought(payload: dict, ran_before: list[dict]) -> str:
    pre = payload.get("_think_pre") or {}
    first = pre if pre.get("ran") else (ran_before[-1] if ran_before else {})
    h = first.get("handoff") or {}
    machine = bool(h.get("machine_built"))
    n = int(first.get("searches") or 0)
    plan = bool(pre.get("ran") and pre.get("job") == "plan")
    when = ("before this reply began" if pre.get("ran")
            else "earlier in this reply, as your yama_think_deeply call")
    what = "plan" if plan else "hand-off"
    where = (f"Its {what} is the {pre.get('tool')} result before this reply"
             if pre.get("ran") and pre.get("into") == "tool_result"
             else f"Its {what} opens this reply's thinking" if pre.get("ran")
             else f"Its {what} is that call's result")
    searched = f"{n} search{'es' if n != 1 else ''}"
    how = ("wrote no conclusion; the result was built from its search log"
           if machine else "planned the task" if plan else "concluded")
    return cs.error_result(
        deep.TOOL_NAME, "ALREADY_THOUGHT",
        f"Deep thinking already ran for this request, {when} ({searched}), "
        f"and {how}. {where}. It runs once per reply; another "
        f"{deep.TOOL_NAME} call returns this same result.",
        retryable=False,
        ran="before this reply" if pre.get("ran") else "earlier in this reply",
        concluded=not machine, searches=n,
        remedies=[{"fixable_by": "agent",
                   "action": f"use the {what} already in this reply "
                             f"({where[0].lower() + where[1:]})",
                   "effect": "what deep thinking found is there"}])


def _think_deeply(payload: dict, args: dict, db: str | None,
                  root: str | None, state: dict | None,
                  n_messages: int) -> str:
    """Run one yama_think_deeply call. Returns the tool result; the record goes
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
        return already_thought(payload, ran_before)
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
    # A busy lane: the standard helper-lane wait (admission.WAIT_SECONDS),
    # then skipped. The deferral counter and the one-wait-per-episode rule
    # (deep.DEFER_REQUESTS) were removed 2026-09-27 (docs/CONSTANTS-AUDIT.md).
    try:
        res = shomen.run(
            "investigate", question=question,
            tools=deep_thinking_tools((state or {}).get("_attached")),
            run_tool=lambda fn, a: run_our_tool(fn, a, db, root, None, state),
            context=ctx, tier=tier.get("name") or "max",
            effort=(tier.get("effort") if "effort" in over else None),
            seed=seed,
            seed_frame=tiers.behaviour(tier, "seed_frame"),
            **({"source_root": root} if root else {}))
        err = None
    except Exception as e:                                       # noqa: BLE001
        res, err = {}, f"{type(e).__name__}: {e}"
    if res.get("skipped"):
        call["refused"] = "HELPER_BUSY"
        return cs.error_result(
            deep.TOOL_NAME, "HELPER_BUSY",
            "Deep thinking is already running for another request; only one "
            "runs at a time. Nothing was thought about here.",
            retryable=True,
            remedies=[{"fixable_by": "agent",
                       "action": "go on from what is in context, or call "
                                 "again on a later step",
                       "effect": "the lane frees when the other run ends"}])
    text, stats, searches = _handoff_of(res, question, err, call, seed,
                                        root)
    call["web_refused"] = len(ctx_run.get("refused") or [])
    head = FINDINGS_HEAD if searches else REASONING_HEAD
    text, n_scrub = scrub_markers(head + text)
    _note_scrub(payload, "yama_think_deeply", n_scrub)
    text = _screen_handoff(payload, text, ctx_run, call)
    call.update(searches=searches, handoff=_handoff_record(stats))
    call["_seed"] = seed
    payload["_deep_terms"] = deep.handoff_terms(text)
    deep.mark_ran(lg.get("account") or "", lg.get("session") or "",
                  n_messages, "model")
    print(f"  yama_think_deeply: ran, {searches} searches, "
          f"{(stats or {}).get('chars')} chars handed back as the tool "
          f"result (handle {call.get('handle')})", flush=True)
    return text


# YAMA_PLAN, THE MODEL'S OWN PLAN CALL (operator, 2026-09-27). Main calls it
# before a long implementation task; the proxy runs the task through the one
# second-brain runner (shomen.run("plan"), its own budget, #60) in the
# helper lane and returns the plan as the TOOL RESULT (PLAN_RESULT_HEAD +
# the four sections + PLAN_RESULT_TAIL). A hidden hop, like
# yama_think_deeply's, but nothing is prefilled after it: main goes on
# acting from the result. One per request (PLAN_CALLS_PER_REQUEST), and
# none after a run before main in the same request (the initial prompt's
# inserted plan, or a trigger's run).
PLAN_CALLS_PER_REQUEST = 1


def _plan_task(payload: dict, args: dict, db: str | None,
               root: str | None, state: dict | None) -> str:
    """Run one yama_plan call. Returns the tool result; the record goes to
    payload["_plan_tool"]["calls"]. Refusals are structured."""
    pt = payload.setdefault("_plan_tool", {"offered": True, "calls": []})
    call: dict = {"ran": False}
    pt["calls"].append(call)
    tier = payload.get("_tier") or {}
    name = deep.PLAN_TOOL_NAME
    task = args.get("task") if isinstance(args, dict) else None
    if not isinstance(task, str) or \
            len(task.strip()) < selection.MIN_QUESTION_CHARS:
        call["refused"] = "BAD_ARGUMENTS"
        return cs.error_result(
            name, "BAD_ARGUMENTS",
            "`task` must be a string of at least 8 characters naming what "
            "to build or change. Nothing was planned.", retryable=True,
            remedies=[{"fixable_by": "agent",
                       "action": "call again with the task: what to build, "
                                 "the libraries it names, where it goes",
                       "effect": "the plan is written"}])
    over = tier.get("overridden") or []
    if not tier.get("investigate"):
        call["refused"] = "DEEP_THINKING_OFF"
        return cs.error_result(
            name, "DEEP_THINKING_OFF",
            f"Planning on the server is off for this request (tier "
            f"{tier.get('name', '?')}"
            + (", forced off by X-Yamadori-Features" if "investigate" in over
               else "") + "). Nothing was planned.", retryable=False,
            remedies=[{"fixable_by": "agent",
                       "action": "plan the steps yourself and start on the "
                                 "first one with your own tools",
                       "effect": "the task goes on without it"}])
    pre = payload.get("_think_pre") or {}
    ran_before = [c for c in pt["calls"][:-1] if c.get("ran")]
    if pre.get("ran") or len(ran_before) >= PLAN_CALLS_PER_REQUEST:
        call["refused"] = "ALREADY_PLANNED"
        where = ("before this reply began, and its result is in this "
                 "conversation" if pre.get("ran") and pre.get("job") == "plan"
                 else "earlier in this reply" if ran_before else
                 "before this reply began (deep thinking on this step)")
        return cs.error_result(
            name, "ALREADY_PLANNED",
            f"The second model already ran for this request, {where}; it "
            f"runs once per request, so nothing more was planned.",
            retryable=False,
            remedies=[{"fixable_by": "agent",
                       "action": "carry out the plan or hand-off you have, "
                                 "starting with its first step, with your "
                                 "own tools",
                       "effect": "the task moves on"}])
    question = "Plan this task.\n\nTASK:\n" + task.strip()
    _q, ctx, _sp = selection.question_of(
        payload.get("_seen_messages") or payload.get("messages") or [])
    seed = ledger_seed(payload, "plan_call", question)
    ctx_run = _research_context(payload, payload.get("_client_messages")
                                or [])
    if state is not None:
        state["_research_budget"] = ctx_run
    # A busy lane: the standard helper-lane wait (admission.WAIT_SECONDS),
    # then skipped. The deferral counter and the one-wait-per-episode rule
    # (deep.DEFER_REQUESTS) were removed 2026-09-27 (docs/CONSTANTS-AUDIT.md).
    try:
        res = shomen.run(
            "plan", question=question,
            tools=deep_thinking_tools((state or {}).get("_attached")),
            run_tool=lambda fn, a: run_our_tool(fn, a, db, root, None, state),
            context=ctx, tier=tier.get("name") or "max",
            effort=(tier.get("effort") if "effort" in over else None),
            seed=seed,
            seed_frame=tiers.behaviour(tier, "seed_frame"),
            plan_switches=tiers.plan_switches(tier),
            **({"source_root": root} if root else {}))
        err = None
    except Exception as e:                                       # noqa: BLE001
        res, err = {}, f"{type(e).__name__}: {e}"
    if res.get("skipped"):
        call["refused"] = "HELPER_BUSY"
        return cs.error_result(
            name, "HELPER_BUSY",
            "The second model is already working for another request; only "
            "one runs at a time. Nothing was planned.", retryable=True,
            remedies=[{"fixable_by": "agent",
                       "action": "plan the first steps yourself and start, "
                                 "or call again on a later step",
                       "effect": "the lane frees when the other run ends"}])
    text, stats, searches = _handoff_of(res, question, err, call, seed, root)
    call["web_refused"] = len(ctx_run.get("refused") or [])
    text, n_scrub = scrub_markers(text)
    _note_scrub(payload, name, n_scrub)
    text = _screen_handoff(payload, text, ctx_run, call)
    call.update(searches=searches, handoff=_handoff_record(stats))
    payload["_deep_terms"] = deep.handoff_terms(text)
    print(f"  {name}: ran, {searches} searches, "
          f"{(stats or {}).get('chars')} chars handed back as the tool "
          f"result (handle {call.get('handle')})", flush=True)
    return PLAN_RESULT_HEAD + text.rstrip() + PLAN_RESULT_TAIL


def _drain(gen):
    """Run a generator to the end; its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def _fan_out(payload: dict, msg: dict, finish: str | None):
    """FAN-OUT: (x_yamadori record, winner).

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
        return None, None
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
        return {"mode": "sequential", "n": 0, "asked": n, "seeds": [],
                "agreement": None, "error": type(e).__name__,
                "replaced": False, "appended": False}, None
    results = list(v.get("results") or [])
    others = results[1:]
    win = v.get("winner") or None
    rec = {"n": sum(1 for r in results if r.get("content")), "asked": n,
           "seeds": [r.get("seed") for r in others if r.get("seed")],
           "agreement": v.get("agreement"),
           "votes": v.get("votes", 0),
           "winner": (win or {}).get("variant"),
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
    return rec, win


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
    (research_tools.run's `budget`): its search and page-read counts, the
    SEEN URLs -- its searches' results, the USER's own URLs, the links of
    the pages it reads -- which read_web_page reads with their query (any
    other URL only by its path), the conversation's text -- tool results,
    answers, and all of it in own_text -- for read_web_page's leak check,
    and every fetch. In memory, for the run."""
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
    return {"search_web": 0, "read_web_page": 0, "urls": [], "links": [],
            "user_urls": user_urls[-100:],
            "conversation": "\n".join(other)[-200000:],
            "own_text": own[-400000:], "web_text": "", "refused": [],
            "screened": [], "fetches": []}


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
               "urls_refused": list(ctx_run.get("refused") or [])[:20],
               # Every web fetch of the run (research_tools._record_fetch):
               # host, provenance (search | user | link | memory), whether
               # the query was stripped, status, bytes; never the path.
               "searches": int(ctx_run.get("search_web") or 0),
               "reads": int(ctx_run.get("read_web_page") or 0),
               "fetches": list(ctx_run.get("fetches") or [])[:60]}
    rec["screen"] = summary
    if summary["removed"] or summary["labelled"] or summary["fetched"] \
            or summary["urls_refused"] or summary["fetches"] \
            or summary["searches"]:
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
                "epoch": trig.get("epoch"), "episode": trig.get("episode"),
                # #52 / remedy 7: how a run of this request is labelled
                # (deep._outcome: `helped` needs a project file changed
                # after it), and the project that rule reads.
                "helped_rule": deep.LABEL_RULE if tiers.behaviour(
                    payload.get("_tier") or {}, "helped_needs_change")
                else 1,
                "project": _project_brief(payload.get("_project"))},
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


def _project_brief(project: dict | None) -> dict | None:
    """The project as a deep_decisions row keeps it: the root and at most
    20 named files -- paths only, never content."""
    if not project:
        return None
    return {"root": project.get("root"),
            "named": list(project.get("named") or [])[:20]}


def _deep_public(payload: dict) -> dict | None:
    """x_yamadori.deep: the trigger decision (never the question text), the
    thresholds in force, yama_think_deeply's offer and calls, the record id."""
    trig = payload.get("_deep")
    if trig is None:
        return None
    out = deep.public(trig) or {}
    tt = payload.get("_think_tool") or {}
    out["think_tool"] = {"offered": bool(tt.get("offered")),
                         "calls": [{k: v for k, v in c.items()
                                    if not k.startswith("_")}
                                   for c in tt.get("calls") or []]}
    # yama_plan (2026-09-27): the model's own calls; the initial prompt's
    # inserted plan is x_yamadori.investigate (into: tool_result).
    pt = payload.get("_plan_tool") or {}
    out["plan_tool"] = {"offered": bool(pt.get("offered")),
                        "calls": [{k: v for k, v in c.items()
                                   if not k.startswith("_")}
                                  for c in pt.get("calls") or []]}
    out["record"] = payload.get("_deep_record")
    # THE VERIFY DIRECTIVE (mcp/verify_moment.py): under auto.verify -- the
    # moment, the kind, the options, the pick, the line, its form (named or
    # generic) and why -- beside a server-tool trigger's record, if one
    # fired on the same request.
    vf = payload.get("_verify")
    if vf is not None:
        out["auto"] = dict(out.get("auto") or {"trigger": "verify"},
                           verify={k: v for k, v in vf.items()
                                   if not k.startswith("_")})
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
    instead of asking the model to quote it (the old recipe check had to). It
    carries decisions and numbers only -- no message text beyond 120
    characters of each recipe, and never the account or its key.

    `hops` is the number of upstream generations of main in the turn, the
    same number as `x_yamadori.usage.generations` (it was `usage.hops`
    until 2026-09-25, U2), on both paths.
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
        # Skills (mcp/skill_select.py): on, route_class, ids, versions,
        # names, chars, why, matched.
        "skills": payload.get("_skills"),
        # The craft index and yama_recall_craft (skill_select PROGRESSIVE
        # DISCLOSURE): the offer kept for the conversation, and this
        # request's reads (name, version, how it was found -- never the
        # query's text).
        "craft": dict(payload.get("_craft") or {},
                      reads=list(payload.get("_craft_reads") or [])),
        # Tools of ours withheld for a conflict with the client's
        # (tool_conflicts): [{ours, because, client_tool}].
        "tools_withheld": list(payload.get("_tools_withheld") or []),
        "selection": payload.get("_selection"),
        # The hand-back's text rides in the record as `_handback` and never
        # leaves: only its counts (`handback`) are data.
        "fanout": ({k: v for k, v in fan.items() if not k.startswith("_")}
                   if isinstance(fan, dict) else fan),
        "investigate": think,
        # Deep thinking's triggers (Phase 0.6, mcp/deep.py): which fired or
        # why none did, the signals counted, the thresholds in force (and
        # whether each is the default, pinned or learned), the cooldown, and
        # yama_think_deeply's offer and calls; `record` names the durable row.
        "deep": _deep_public(payload),
        "hops": hops,
        # CONTINUE A STATED STEP: an agent step that stopped with no call,
        # judged by the decider and continued once when it only said what
        # it was about to do ({judged, distribution, raw_pick, pick,
        # continued, line, call_followed, ms, why, ...}); None when the
        # trigger did not hold.
        "continued": payload.get("_continued"),
        # One entry per yama_generate_image call: ok, id prefix, size, seed,
        # seconds -- or the error code. Never the prompt or the URL.
        "images": list(payload.get("_images") or []),
        # One entry per yama_describe_image call: ok, which image (attached id or
        # sha prefix), source, format, bytes, seconds, token usage -- or the
        # error code. Never the image or the question.
        "vision": list(payload.get("_vision") or []),
        # The MCP-backed tools (mcp/mcp_host.py): the switch, the tools the
        # conversation was offered (kept by name), which of them are on main
        # this request, the servers' states at the offer, and one entry per
        # call: tool, server, upstream, ms, ok, bytes, error, screen,
        # args (120 chars each), names (the package names it returned).
        "mcp": (dict(payload["_mcp"],
                     calls=list(payload.get("_mcp_calls") or []))
                if payload.get("_mcp") is not None else None),
        # The images this request carried and what became of each: id,
        # source, format, bytes, error. Never the bytes.
        "attachments": list(payload.get("_attachments") or []),
        # Every A4000 decision this request caused (mcp/gpu_room.py): model,
        # action (loaded / fit / evicted / busy / no_room / uncoordinated),
        # need, free before/after, what was unloaded. Numbers and model ids.
        "gpu_room": list(payload.get("_gpu_room") or []),
        # Max mode (mcp/max_mode.py): the main model that served this request and why; how long it waited for the
        # other model's work to drain. Absent when max mode is off.
        **({"capacity": payload["_capacity"]} if payload.get("_capacity") else {}),
        # One entry per tool call the proxy executed in THIS conversation's
        # loop (image tools, the delegate arm): name, empty, error, chars.
        "tools": list(payload.get("_tool_calls") or []),
        # The tool-turn cap (tiers.tool_turn_limit): limit, tool turns
        # executed, and whether the loop landed because of it.
        "tool_turns": dict(payload.get("_turn_cap") or _turn_cap(payload)),
        # The client's own prompt against its window (check_client_prompt,
        # C1): limit, generation floor, the high estimate, and the model
        # server's own count when it was near enough the edge to be made.
        "context": payload.get("_context"),
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
        # THE IMAGE GUARD (tool_code, #46): None when no call was stopped;
        # else each stopped call -- tool, argument name, kind of image data
        # (a data:image/ URL, base64 PNG data, ...), the argument characters
        # and deltas (~tokens) read and the seconds before the stop, the
        # hop, `withheld` on a last hop -- how many times the model was asked
        # again, and whether the turn landed. Never the argument's text.
        "image_guard": payload.get("_image_guard_rec"),
        # The fold-backs this turn carried (shomen.PHRASES): job, phrase,
        # where it went (prefill / note / continuation / in_place /
        # appended), and the seed words named.
        "fold_back": list(payload.get("_fold_back") or []),
        # After a deep-thinking fold-back: how much answer followed the
        # opening, and whether it pointed at text the user never saw
        # (fold_back_answer).
        "fold_back_answer": payload.get("_fold_back_answer"),
        # The ledger: what was restored on this request (counts per kind)
        # and whether this request decided its user turn's injection.
        # `keyed_by_shown`: the client was streamed more content than the
        # turn renders (an earlier hop's), so the turn is keyed by what it
        # stores and its content restored as delivered (#10).
        # `restore_reasoning` {on, source}: past reasoning restored (restored
        # counts `reasoning`, `reasoning_chars`, `reasoning_missing`) and
        # `reasoning_recorded`, the chars this turn's own reasoning added.
        "ledger": {"restored": lg.get("restored") or {},
                   "inject": lg.get("inject"),
                   "keyed_by_shown": bool(payload.get("_stored_differs")),
                   "restore_reasoning": lg.get("restore_reasoning"),
                   "reasoning_recorded": lg.get("reasoning_recorded")},
        # WHICH CONVERSATION (#41, mcp/session_id.py): {id, source, why,
        # carried, line}. source: prompt_cache_key | header | tool_call_id
        # (our id, read back from a tool-call id) | summary_line (from a
        # compaction summary we wrote) | minted (a new conversation) |
        # compaction_map (a flattened compaction mapped to a conversation).
        # `carried`: how many of this turn's client tool calls went out with
        # an id carrying ours (_carry_session). `line`: this answer is a
        # compaction summary that opens with the session line. `id` is ours
        # in full, a client's own as a hash prefix. Never the key.
        "session": (dict(lg["conversation"],
                         carried=int(payload.get("_session_carried") or 0),
                         line=bool(payload.get("_session_lead")))
                    if lg.get("conversation") else None),
        # Whether the slot is being warmed with the delivered turn, and why.
        "warm": payload.get("_warm"),
        # The warm that ran after this conversation's PREVIOUS response:
        # reused / processed / slot, and `short` when it reused less than
        # the prompt that slot had just generated on (#11).
        "warm_before": payload.get("_warm_before"),
        # How many earlier attempts of this very request (a client's retry
        # of it) were still running and were cancelled for it (#44).
        "superseded": payload.get("_superseded"),
        # The chat template's own markers (<think>, </think>, <|im_start|>,
        # <|im_end|>, <tool_call>) in the delivered content -- counts and
        # whose (`model`: left as written; `ours`: a defect) -- and what our
        # own path scrubbed before it reached a prompt. None when clean (#12).
        "template_markers": payload.get("_template_markers"),
        # LIBRARY USE (#19): held packages this conversation uses, and the
        # definitions injected for them on this request (names, never
        # text), or that a recorded injection was replayed.
        "library_use": payload.get("_library_use"),
        # THE OVERTHINKING SWITCHES (mcp/progress.py, tiers.BEHAVIOURS;
        # docs/research/OVERTHINKING.md): `project` -- the step's project and
        # scratch writes read into the project on this request's tool
        # result, the root and how it was found, how many files the task
        # names -- `step` (the agent step's thinking
        # cap and why, which nudge) and `switches` (each switch, on/off and
        # its source: header / env / tier / default).
        "progress": payload.get("_progress"),
        # Vendor sampling as enforced by tiers.apply, and what the client
        # had asked for where it differed.
        "sampling": payload.get("_sampling"),
        "budget": {"max_tokens_sent": payload.get("max_tokens"),
                   "reasoning_budget_tokens":
                       payload.get("reasoning_budget_tokens"),
                   # the thinking nudge's fraction (tiers.NUDGE_AT), None
                   # when it was not sent; whether it FIRED is only in the
                   # reasoning text, which the fork does not report
                   "nudge_at": payload.get("reasoning_budget_nudge_at")},
        # A client utility call runs at `minimal` whatever the client's tier
        # (proxy.prepare); the rule's reason is selection.because.utility.
        "utility": bool((payload.get("_utility") or {}).get("utility")),
        "tier_requested": payload.get("_tier_requested") or tier.get("name"),
        "tier_overridden": ("minimal" if (payload.get("_utility") or {})
                            .get("utility") else None),
        # compaction | title | classifier | structured | other, None for a
        # task turn (selection.utility_kind). A compaction also carries its
        # budget record (tiers.compaction_budget) and the harness whose form
        # it is (mcp/compaction.py).
        "utility_kind": payload.get("_utility_kind"),
        "compaction": payload.get("_compaction"),
        # How the client's instruction messages were mapped
        # (system_roles.one_system: {merged, developer_as_user}); None when
        # nothing was.
        "roles": payload.get("_roles"),
        # Prompt tokens processed vs reused from the slot's cache, and which
        # slot (mcp/slots.py), summed over this request's generations.
        "cache": _cache_summary(payload),
        # RELEASE (mcp/slots.py): the switch for this request (header / env
        # / default) and each slot it emptied -- the second brain's after a
        # run, the transient one after a side call -- or kept, and why:
        # {slot, why, released, cells_before, ms, method, skipped?}.
        "slots": {"release": dict(payload.get("_slot_release") or {}),
                  "released": list(payload.get("_slots_released") or []),
                  # IDLE CLEAR: the switch and threshold, the other
                  # conversations' idle slots this request cleared before
                  # it generated, and -- on a conversation that came back
                  # to a cleared slot -- that its prompt was re-processed.
                  "idle_clear": dict(
                      payload.get("_idle_clear") or {},
                      after_s=((payload.get("_idle_clear") or {})
                               .get("override_s") or slots.IDLE_CLEAR_S)),
                  "cleared_idle": list(payload.get("_slots_cleared") or []),
                  "resumed_cold": next(
                      (r["resumed_cold"] for r in
                       payload.get("_cache_log") or []
                       if r.get("resumed_cold")), None)},
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
           "processed": total("processed"),
           # The model server's own milliseconds over this request's
           # generations (slots.cache_record); None when a generation did not
           # report them. Wall clock minus this is the stack's overhead.
           "model_ms": total("model_ms"), "slot": last.get("slot"),
           # The prefill's own milliseconds over the request, and the LAST
           # generation's decode rate (slots.cache_record).
           "prompt_ms": total("prompt_ms"),
           "decode_tps": last.get("decode_tps"),
           "mode": last.get("mode"), "generations": len(recs),
           "evicted": next((r.get("evicted") for r in recs if r.get("evicted")),
                           None),
           "first": {k: recs[0].get(k) for k in ("prompt", "reused",
                                                  "processed")}}
    aff = next((r["affinity"] for r in recs if r.get("affinity")), None)
    if aff:
        out["affinity"] = aff
    adopted = next((r["adopted"] for r in recs if r.get("adopted")), None)
    if adopted:
        out["adopted"] = adopted
    # RANKS the engine was told for the last generation (layout v2: the
    # lane's rank among them), and where a compaction sent as is went.
    if last.get("kv_ranks") is not None:
        out["kv_rank"], out["kv_ranks"] = last.get("kv_rank"), last["kv_ranks"]
    for k in ("displaced", "how", "stray_pins_dropped"):
        v = next((r[k] for r in recs if r.get(k)), None)
        if v:
            out[k] = v
    if log_line:
        f = out["first"]
        print(f"  cache: slot {out['slot']} ({out['mode']}) "
              f"first prompt {f['prompt']} reused {f['reused']} processed "
              f"{f['processed']}; {len(recs)} generation(s) reused "
              f"{out['reused']} of {out['prompt']}"
              + (f"; evicted {out['evicted']}" if out["evicted"] else "")
              + (f"; adopted slot {adopted['slot']} from {adopted['key']} "
                 f"({adopted['why']})" if adopted else ""),
              flush=True)
    return out


# _empty_notice REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md): a stop with
# nothing written (no content, no tool call) used to be replaced by "[no
# answer: the model stopped ...]". It is delivered as it is: blank content,
# finish_reason stop. corpus.log_answer records finish and tool calls, so a
# client tool call with no preface is not read as an empty answer.


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
    # The request's cancellation reaches the job's sockets (mcp/cancel.py);
    # closing this generator early cancels the job (#39).
    tok = cancel.current() or cancel.Token()

    def _go():
        try:
            with cancel.bound(tok):
                box["v"] = fn()
        except Exception as e:                                   # noqa: BLE001
            box["e"] = e

    th = _threading.Thread(target=_go, daemon=True)
    th.start()
    try:
        while th.is_alive():
            th.join(timeout=HEARTBEAT)
            if th.is_alive():
                yield ("heartbeat", None)
    except GeneratorExit:
        tok.cancel("the stream was closed while a job ran")
        raise
    if tok.cancelled:
        raise cancel.Cancelled(tok.why)
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
#   the MODEL    -- a stray marker in the ANSWER is removed from what the
#                   client gets, with the text after it when that repeats
#                   what was already sent, and RECORDED
#                   (x_yamadori.template_markers.stripped, one log line):
#                   _StrayMarkers. Operator, 2026-09-25, REVERSING the
#                   2026-09-24 "delivered as written": the model does it at a
#                   rate (7 of 40 replays of one fixed context with reasoning
#                   echoed, 3 of 40 stripped), not once, and every client
#                   showed it. The record keeps the defect visible.
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


def _markers_record(payload: dict, delivered: str, upstream: str,
                    stray: "_StrayMarkers | None" = None) -> dict | None:
    """x_yamadori.template_markers: the markers the answer carried -- those
    still in the delivered content (`in_content`: inside code, or a
    `<tool_call>`) and those _StrayMarkers removed from it (`stripped`, with
    `repeat_chars_dropped` and `new_text_after`) -- and whose they are:
    `model` when the upstream generations (main's hops, its continuation, a
    second-brain winner delivered in its place) carried at least as many,
    else `ours` (a defect in this file: our path should have scrubbed them).
    Plus what our path scrubbed. None when clean."""
    found = template_markers(delivered)
    stripped = dict(stray.stripped) if stray is not None else {}
    scrubbed = dict(payload.get("_markers_scrubbed") or {})
    if not found and not scrubbed and not stripped:
        return None
    total = dict(found)
    for m, k in stripped.items():
        total[m] = total.get(m, 0) + k
    up = template_markers(upstream)
    model = {m: min(k, up.get(m, 0)) for m, k in total.items()
             if up.get(m, 0)}
    ours = {m: k - model.get(m, 0) for m, k in total.items()
            if k - model.get(m, 0) > 0}
    rec = {"in_content": found, "stripped": stripped,
           "repeat_chars_dropped": stray.repeat_chars if stray else 0,
           "new_text_after": stray.new_after if stray else 0,
           "model": model, "ours": ours, "scrubbed": scrubbed,
           "source": ("ours" if ours else "model" if model else None)}
    if total:
        print(f"  template markers in the answer: {total} -- written by "
              f"{'the MODEL' if not ours else 'OUR PATH (defect)'}; "
              f"stripped {stripped or 'none'} (a repeat of "
              f"{rec['repeat_chars_dropped']} chars dropped), left in code "
              f"{found or 'none'}; recorded", flush=True)
    return rec


# IMAGES REACH THE CHAT THE MOMENT THEY EXIST (operator, 2026-09-25: "when
# you ask for an image [Claude/ChatGPT] emit the image to the harness when
# they make it"). yama_generate_image ran as a hidden hop and its picture reached
# the user only inside the final answer, after the model had thought again.
# Now the proxy streams the image's markdown line as CONTENT the moment the
# tool returns (_run_turn); the blocking path puts it at the start of the
# answer -- the same content either way. The model
# is told the picture is shown (images.shown_on_main) and writes around it;
# a copy it writes anyway is removed from what the client gets (_ImageDedup)
# but kept in what the slot renders. The line is content the SLOT did not
# generate: the ledger keys the turn by the client's copy and renders the
# slot's text (#10), as for a compaction's session line, so the next
# request extends the slot (mcp/test_image_emit.py, through the served
# template).
#
# After the first content byte, reasoning is not forwarded (CHANNEL ORDER):
# it goes out as an empty delta at most every REASONING_BEAT_S seconds, so a
# client still sees the turn alive while the model thinks after the image.
REASONING_BEAT_S = 1.0
# The describe line (_describe_line): the description's first characters.
DESCRIBE_LINE_CHARS = 100


class _ImageDedup:
    """Removes, from the model's content, an exact duplicate of an image the
    proxy already showed: a markdown image whose target is that url (any alt
    text), or the bare url alone on a line -- with the blank line that set it
    apart. Streamed, text that might still become a duplicate is held back
    until it cannot (feed / flush); `strip` does a whole text at once.
    `removed`: {url: count}."""

    _PARTIAL = re.compile(r"!(?:\[[^\]\n]{0,400}(?:\](?:\(([^\s)]*)(\))?"
                          r"[ \t]*\n?)?)?)?")

    def __init__(self):
        self.urls: list[str] = []
        self.removed: dict[str, int] = {}
        self.buf = ""
        self._last = "\n"            # the last character released

    def add(self, url: str) -> None:
        if url and url not in self.urls:
            self.urls.append(url)

    def _rx(self) -> "re.Pattern":
        if getattr(self, "_rx_for", None) != tuple(self.urls):
            alts = "|".join(re.escape(u) for u in
                            sorted(self.urls, key=len, reverse=True))
            self._rx_c = re.compile(r"!\[[^\]\n]*\]\((%s)\)|(?m:^[ \t]*(%s)"
                                    r"[ \t]*(?=\n|\Z))" % (alts, alts))
            self._rx_for = tuple(self.urls)
        return self._rx_c

    def _strip(self, text: str, final: bool, before: str = "\n") -> str:
        if not self.urls or not text:
            return text
        out, pos = [], 0
        for m in self._rx().finditer(text):
            s, e = m.start(), m.end()
            if not final and e + 2 > len(text):
                break                 # what follows it is not known yet
            url = m.group(1) or m.group(2)
            s2 = s
            while s2 > pos and text[s2 - 1] in " \t":
                s2 -= 1
            prev = text[s2 - 1] if s2 > 0 else (before or "\n")
            e2 = e
            while e2 < len(text) and text[e2] in " \t":
                e2 += 1
            if prev == "\n":
                # A line of its own: the line goes, with the blank line after.
                k = 0
                while e2 < len(text) and text[e2] == "\n" and k < 2:
                    e2 += 1
                    k += 1
            else:
                s2 = s                # mid-line: keep the space before it
            out.append(text[pos:s2])
            pos = e2
            self.removed[url] = self.removed.get(url, 0) + 1
        out.append(text[pos:])
        return "".join(out)

    def _hold_from(self, text: str) -> int:
        """The first index from which `text` could still become (or end in)
        a duplicate; len(text) when none."""
        n = len(text)
        if not self.urls:
            return n
        best = n
        # A bare url at the end, whole (what follows it is not known yet) or
        # begun: the longest suffix that is a url's prefix.
        body = text.rstrip()
        for u in self.urls:
            if body.endswith(u):
                best = min(best, len(body) - len(u))
            for k in range(min(len(u), n), 0, -1):
                if text.endswith(u[:k]):
                    best = min(best, n - k)
                    break
        # A markdown image begun (or ended, its line not yet known) whose
        # target is, or may still become, one of the urls.
        lo = max(0, n - (max(len(u) for u in self.urls) + 440))
        i = text.find("!", lo)
        while 0 <= i < best:
            m = self._PARTIAL.fullmatch(text, i)
            if m:
                u = m.group(1)
                if u is None or (any(x.startswith(u) for x in self.urls)
                                 and (not m.group(2) or u in self.urls)):
                    return i
            i = text.find("!", i + 1)
        return best

    def feed(self, piece: str) -> str:
        """Streamed: what may be released now."""
        if not self.urls:
            return piece
        self.buf += piece
        text = self._strip(self.buf, final=False, before=self._last)
        cut = self._hold_from(text)
        out, self.buf = text[:cut], text[cut:]
        if out:
            self._last = out[-1]
        return out

    def flush(self) -> str:
        text = self._strip(self.buf, final=True, before=self._last)
        self.buf = ""
        if text:
            self._last = text[-1]
        return text

    def strip(self, text: str) -> str:
        """A whole text (the blocking path)."""
        return self._strip(text, final=True)


# STRAY TEMPLATE MARKERS IN THE ANSWER (#12; operator, 2026-09-25). The model
# thinks, the FIRST `</think>` ends its reasoning (llama-server's parser
# takes it; reasoning_content is whole), and then, in the ANSWER, it writes
# a SECOND `</think>` and usually repeats itself: 'Done.\n</think>\n\nDone.',
# 'Done.\n\nVerified stats.py ...\n</think>\n\nDone.', or '<sentence>\n
# </think>\n\n' before its tool calls (live replays, 2026-09-25: 7 of 40
# draws of one fixed context with reasoning echoed, 3 of 40 stripped). What
# the CLIENT gets: the marker removed, with the whitespace around it; the
# text after it dropped while it repeats what was already sent (a
# whitespace-normalised prefix of it: the whole of it, or whole lines of
# it); text after it that is NEW goes through, one blank line at most
# between. A marker inside code (a ``` fence, or an odd number of backticks
# on its line) is literal and stays (a program about templates). What the
# SLOT holds is untouched: the ledger keys the turn by the client's copy
# and renders the slot's text (#10), so the next request extends the slot.
STRAY_MARKERS = ("</think>", "<think>", "<|im_end|>", "<|im_start|>")
_FENCE_LINE = re.compile(r"(?m)^[ \t]{0,3}(?:```|~~~)")


class _StrayMarkers:
    """The answer's stray template markers out (see above). Streamed, text
    that might still be (or precede) a marker -- a trailing '<', '</th', the
    whitespace before them -- and text after a marker that might still be a
    repeat are held until they cannot (feed / flush); fed a character at a
    time or all at once, the output is the same (`clean` does a whole text).
    Only the model's content goes through it: the session line and image
    lines never do (_Out). `stripped`: {marker: n}; `repeat_chars`: the
    repeated characters dropped; `new_after`: how often new text followed a
    marker."""

    _MAXLEN = max(len(m) for m in STRAY_MARKERS)
    _ROLES = ("assistant\n", "user\n", "system\n", "tool\n")

    def __init__(self):
        self.buf = ""
        self.role_next = False  # a `<|im_start|>`'s role line may follow
        self.sent = ""          # every character released: what a repeat is
        self.after = False      # just past a stray marker
        self.sep = ""           # the whitespace around it, held
        self.stripped: dict[str, int] = {}
        self.repeat_chars = 0
        self.new_after = 0

    @staticmethod
    def _literal(ctx: str) -> bool:
        """A marker after `ctx` is inside code: an open fence, or an odd
        number of backticks on its line."""
        if len(_FENCE_LINE.findall(ctx)) % 2:
            return True
        return ctx[ctx.rfind("\n") + 1:].count("`") % 2 == 1

    def _find(self, buf: str) -> tuple[int, str | None]:
        """The first stray marker in `buf`: (index, marker), or (-1, None)."""
        pos = 0
        while True:
            hits = [(i, m) for m in STRAY_MARKERS
                    for i in (buf.find(m, pos),) if i >= 0]
            if not hits:
                return -1, None
            i, m = min(hits)
            if not self._literal(self.sent + buf[:i]):
                return i, m
            pos = i + 1

    def _hold_from(self, buf: str) -> int:
        """Where the held tail starts: a suffix that may still become a
        marker, and the whitespace before it (it goes with the marker)."""
        n = len(buf)
        k = n
        for L in range(min(n, self._MAXLEN - 1), 0, -1):
            if any(m.startswith(buf[n - L:]) for m in STRAY_MARKERS):
                k = n - L
                break
        while k > 0 and buf[k - 1].isspace():
            k -= 1
        return k

    @staticmethod
    def _repeat(t: str, ref: str, final: bool) -> tuple[str, int]:
        """Is `t` (it opens on a non-space) a repeat of the start of `ref`,
        whitespace-normalised? ("repeat", k): t[:k] is -- all of ref, or
        whole lines of it; ("new", 0): it is not; ("need", 0): not known
        until more of t arrives."""
        n, m = len(t), len(ref)
        j = 0
        while j < m and ref[j].isspace():
            j += 1
        if j == m:
            return "new", 0
        i = last_t = 0
        last_r = j
        crossed = False       # just crossed a line break in both
        while True:
            i2, j2 = i, j
            while i2 < n and t[i2].isspace():
                i2 += 1
            while j2 < m and ref[j2].isspace():
                j2 += 1
            t_ws, r_ws = i2 > i, j2 > j
            if j2 == m:
                # All of ref matched: a repeat, when it ends a word in t.
                if i2 < n:
                    return ("repeat", last_t) if t_ws else ("new", 0)
                if t_ws or final:
                    return "repeat", last_t
                return "need", 0
            if i2 == n:
                if t_ws and not r_ws:
                    return "new", 0
                if not final:
                    return "need", 0
                # The end of the answer is a line end: whole lines repeated.
                k = last_r
                while k < m and ref[k].isspace():
                    k += 1
                return (("repeat", last_t) if "\n" in ref[last_r:k]
                        else ("new", 0))
            if t_ws != r_ws:
                return "new", 0
            if t_ws:
                crossed = "\n" in t[i:i2] and "\n" in ref[j:j2]
                i, j = i2, j2
                continue
            if t[i] != ref[j]:
                return ("repeat", last_t) if crossed and last_t else ("new", 0)
            i += 1
            j += 1
            last_t, last_r, crossed = i, j, False

    @staticmethod
    def _sep_of(ws: str) -> str:
        k = ws.count("\n")
        return "\n\n" if k >= 2 else "\n" if k == 1 else (" " if ws else "")

    def _run(self, final: bool) -> str:
        out: list[str] = []

        def emit(t: str) -> None:
            if t:
                out.append(t)
                self.sent += t

        while True:
            if not self.after:
                i, mk = self._find(self.buf)
                if mk is not None:
                    j = i
                    while j > 0 and self.buf[j - 1].isspace():
                        j -= 1
                    emit(self.buf[:j])
                    self.sep = self.buf[j:i]
                    self.buf = self.buf[i + len(mk):]
                    self.stripped[mk] = self.stripped.get(mk, 0) + 1
                    self.role_next = mk == "<|im_start|>"
                    self.after = True
                    continue
                h = len(self.buf) if final else self._hold_from(self.buf)
                emit(self.buf[:h])
                self.buf = self.buf[h:]
                return "".join(out)
            # Just past a stray marker. `<|im_start|>`'s role line is part
            # of it (the template writes "<|im_start|>assistant\n").
            if self.role_next:
                if any(self.buf.startswith(r) for r in self._ROLES):
                    r = next(r for r in self._ROLES if self.buf.startswith(r))
                    self.buf = self.buf[len(r):]
                elif not final and any(r.startswith(self.buf)
                                       for r in self._ROLES):
                    return "".join(out)
                self.role_next = False
            # Its whitespace is held.
            k = 0
            while k < len(self.buf) and self.buf[k].isspace():
                k += 1
            self.sep += self.buf[:k]
            self.buf = self.buf[k:]
            if not self.buf:
                if final:                 # nothing followed: it all goes
                    self.sep, self.after = "", False
                return "".join(out)
            mk = next((x for x in STRAY_MARKERS if self.buf.startswith(x)),
                      None)
            if mk is not None:
                self.buf = self.buf[len(mk):]
                self.stripped[mk] = self.stripped.get(mk, 0) + 1
                self.role_next = mk == "<|im_start|>"
                continue
            if not final and any(x.startswith(self.buf) for x in
                                 STRAY_MARKERS):
                return "".join(out)
            how, k = self._repeat(self.buf, self.sent, final)
            if how == "need":
                return "".join(out)
            if how == "repeat":
                self.repeat_chars += k
                self.buf = self.buf[k:]
                self.sep = ""
                continue
            # New text: it goes through, one blank line at most before it.
            self.new_after += 1
            if self.sent and not self.sent[-1].isspace():
                emit(self._sep_of(self.sep))
            self.sep, self.after = "", False

    def feed(self, piece: str) -> str:
        """Streamed: what may be released now."""
        self.buf += piece or ""
        return self._run(final=False)

    def flush(self) -> str:
        """The answer's end (or an image line about to go out)."""
        return self._run(final=True)

    @classmethod
    def clean(cls, text: str) -> tuple[str, "_StrayMarkers"]:
        """A whole text (the blocking path): (cleaned, the filter's record)."""
        f = cls()
        return f.feed(text or "") + f.flush(), f


def strip_stray_markers(text: str) -> str:
    """`text` as the client gets it (_StrayMarkers)."""
    return _StrayMarkers.clean(text)[0]


# REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md), the answer is delivered as
# the model wrote it:
#   _ToolMarkup / tool_markup_notice -- removed written-out tool-call markup
#     from a LANDED answer and, when nothing else was written, replaced it
#     with a "[no answer: ...]" notice (built from one run, pagoda-r3f-1).
#   _ImitatedNotes / NOTE_HEAD_MAX / note_heads -- removed note-shaped lines
#     ("Verified ...", "Repaired ...") from the model's own content (#36; no
#     operator decision found, and it altered model output).


def _describe_line(result: str) -> str | None:
    """ONE reasoning line when a yama_describe_image call returns (operator,
    2026-09-25): which image, and the first DESCRIBE_LINE_CHARS characters
    of what the vision model saw. SCREENED like any tool text shown to the
    user -- the template's markers, markup, links and invisible characters
    out; a credential, AI-directed text or exfiltration withholds the
    excerpt (mcp/skill_screen.py). None for a failure (the error is in the
    tool result the model reads)."""
    try:
        d = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not (isinstance(d, dict) and d.get("ok")
            and isinstance(d.get("answer"), str) and d["answer"].strip()):
        return None
    import skill_screen
    label = str(d.get("image") or "the image")
    if not re.fullmatch(r"image-[0-9a-f]{10}", label):
        label = ("generated image " + label[:10]
                 if re.fullmatch(r"[0-9a-f]{10,64}", label) else "the image")
    t, _n = scrub_markers(d["answer"])
    t = re.sub(r"<!--.*?(?:-->|\Z)", " ", t, flags=re.S)
    t = re.sub(r"</?[A-Za-z][^>\n]{0,200}>", " ", t)
    t = re.sub(r"!?\[([^\]\n]*)\]\([^)\s]*\)", r"\1", t)
    t = re.sub(r"(?i)\b(?:https?|ftp|file|data):\S+|\bwww\.\S+", "[link]", t)
    t = t.replace("`", "'")
    t = skill_screen.visible(t, limit=4000)
    hits = (skill_screen.check_credentials(t)
            + skill_screen.check_ai_directed(t)
            + skill_screen.check_exfiltration(t))
    if hits:
        t = f"(the description is withheld from this line: " \
            f"{hits[0].get('rule') or 'screened'})"
    elif len(t) > DESCRIBE_LINE_CHARS:
        t = t[:DESCRIBE_LINE_CHARS - 1].rstrip() + "…"
    return f"`looked at {label}: {t}`\n"


class _Out:
    """What a turn has presented, and CHANNEL ORDER (2026-09-24, live SSE
    diagnostic): a client closes its thinking block at the first `content`
    delta, so every byte of content must come after all reasoning. Once any
    content has gone out, later reasoning is not forwarded: it becomes an
    empty delta at most every REASONING_BEAT_S (IMAGES REACH THE CHAT)."""

    def __init__(self, streamed: bool, shown_prefix: str = "",
                 lead: str = ""):
        self.streamed = streamed
        self.content_sent = False
        # Image lines shown this turn, in order (IMAGES REACH THE CHAT), and
        # the filter that removes the model's own copy of one.
        self.images: list[str] = []
        self.dedup = _ImageDedup()
        # The model's stray template markers (#12), filtered AFTER the
        # duplicate filter: a repeat is judged against what the client was
        # actually sent (a copy of a shown image, already gone, cannot hide
        # one). The same order on the blocking path (_run_turn).
        self.markers = _StrayMarkers()
        self.reasoning_suppressed = 0
        self._beat_at = 0.0
        # OUR SESSION LINE (#41, mcp/session_id.py), on a COMPACTION SUMMARY
        # only (answers carry the id in their tool-call ids): the first
        # content byte, after all reasoning (CHANNEL ORDER).
        self.lead = lead or ""
        # Every content byte the client was sent, in order: what a streaming
        # client STORES as this turn's content, and so the text the ledger
        # must key the turn by (_run_turn, #10 in docs/SELF-IMPROVEMENT-LOG).
        self.shown: list[str] = [shown_prefix] if shown_prefix else []
        # Every reasoning byte the client was sent: what an ECHOING client
        # sends back as this turn's reasoning (pass-through, 2026-09-24).
        self.reasoning_shown: list[str] = []

    def reasoning(self, text: str):
        if not (self.streamed and text):
            return
        if not self.content_sent:
            self.reasoning_shown.append(text)
            yield ("reasoning", text)
            return
        # CHANNEL ORDER: after content, reasoning is a heartbeat, throttled.
        self.reasoning_suppressed += len(text)
        now = time.time()
        if now - self._beat_at >= REASONING_BEAT_S:
            self._beat_at = now
            yield ("heartbeat", None)

    def content(self, text: str):
        if self.streamed and text:
            yield from self.flush_lead()
            self.content_sent = True
            if self.dedup.urls:
                text = self.dedup.feed(text)
            text = self.markers.feed(text)
            if not text:
                return
            self.shown.append(text)
            yield ("content", text)

    def note(self, text: str, sep: str = "\n\n"):
        """OUR note (tool_code's): after the model's content, which is
        released through its filters first, and never filtered itself (it IS
        the note). `sep` before it when the client was shown any text."""
        if not (self.streamed and text):
            return
        yield from self.flush_lead()
        yield from self.flush_held()
        t = (sep if "".join(self.shown).strip() else "") + text
        self.content_sent = True
        self.shown.append(t)
        yield ("content", t)

    def image(self, block: str, url: str):
        """An image line the proxy shows the moment it exists: content, after
        whatever content is held, bypassing the
        duplicate filter (it IS the image). Blocking: recorded, and placed at
        the start of the answer when the turn ends (_run_turn)."""
        self.images.append(block)
        if self.streamed:
            yield from self.flush_lead()
            yield from self.flush_held()
            self.content_sent = True
            self.shown.append(block)
            yield ("content", block)
        self.dedup.add(url)
        self.dedup._last = "\n"

    def flush_held(self):
        """Content the filters still hold, released (turn end, or an image
        line about to go out): the duplicate filter's, through the marker
        filter, then the marker filter's."""
        if not self.streamed:
            return
        t = self.dedup.flush() if self.dedup.urls else ""
        t = self.markers.feed(t) + self.markers.flush()
        if t:
            self.shown.append(t)
            yield ("content", t)

    def flush_lead(self):
        if self.streamed and self.lead:
            t, self.lead = self.lead, ""
            self.content_sent = True
            self.shown.append(t)
            yield ("content", t)


class TurnRefused(api_errors.ApiError):
    """A turn that cannot start, as a STRUCTURED error (pre-deploy review,
    2026-09-24): `yamadori-vision`'s gpu_room.NoRoom escaped as a bare 502
    on the blocking path and a broken stream on the streamed one. It carries
    the situation, whether retrying helps (as a fact) and remedies with an
    owner. One of the ApiErrors (mcp/api_errors.py): raised before the first
    byte it is an HTTP status and `body()` on both paths (the streamed path
    commits nothing until the turn has started, server.chat); after it, the
    SSE error event."""

    def __init__(self, status: int, code: str, reason: str, retryable: bool,
                 remedies: list, facts: dict | None = None):
        super().__init__(status, reason, code=code,
                         headers={"Retry-After": "60"} if retryable else None)
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


# =================================================== CONTINUE A STATED STEP ==
#
# "PLANNING WITHOUT ACTION" (operator-approved fix, 2026-09-27). An agent
# step ends finish=stop with no tool call and text that only states its next
# action -- "Now let me understand the task..." -- with nothing done, and
# the harness, seeing an answer, ends the run (live: deploy_check's agent
# loop [echo] step 3; pagoda-h3). So:
#   TRIGGER (structural): an agent step (route agent_step, or the
#     conversation carries client tool calls), the generation ended "stop",
#     it made NO tool call, it wrote visible text, and the client offered
#     tools. Only then:
#   JUDGE: the Bonsai decider, ONE choice over that text and the tail of
#     its reasoning (decide_turn.judge_stop: finished / about to do next /
#     asking the user / none of these), on the transient slot, released.
#   CONTINUE, when it said what it is about to do: main's own turn is
#     PREFILLED -- its text, a blank line, CONTINUE_LINE -- and generated
#     again with the client's tools still offered and the same budget rule
#     (tiers.rebudget, as every hop). The client receives ONE turn: the
#     text, the line and what the continuation wrote (normally a call).
#     Nothing is hidden from the client and nothing is rewritten: the slot
#     holds exactly the turn delivered (prompt, the prefill, the
#     continuation), so the ledger records it like any turn and no warm is
#     needed unless something else changed it (a tool_code note).
#   ONCE per request: a continuation that stops again is delivered as is.
# CONTINUE_LINE ends on a LETTER, never a space or a period: AGENTS.md's
# prefill rule ("A prefilled phrase ends on a letter, or on an ending
# measured safe (`After thinking deeply,` -- STEP 0), never a space"); a
# period was never measured. "Let me ..." is the voice of the operator's
# approved agent-step nudge (tiers.AGENT_STEP_NUDGE_MESSAGE, "Let me make
# the call I've already worked out"). UNMEASURED WORDING; not yet run live.
# x_yamadori.continued: {judged, distribution, raw_pick, pick, continued,
# line, call_followed, ms, ...}; None when the trigger did not hold.
CONTINUE_LINE = "Let me do that now"
CONTINUE_PICK = "next_step"


def _continue_separator(text: str) -> str:
    """What goes between the model's text and CONTINUE_LINE: one blank line,
    counting the newlines the text already ends on (its bytes are kept as
    generated: the prefill must extend what the slot holds)."""
    if text.endswith("\n\n"):
        return ""
    return "\n" if text.endswith("\n") else "\n\n"


def _stated_step_gate(payload: dict, messages: list[dict], msg: dict,
                      fin: str | None, ours: set, landed: bool) -> dict | None:
    """None when the structural trigger does not hold (nothing to decide);
    else {"eligible": bool, "why": ...} -- eligible only when the switch,
    the tier and the turn allow the judgment."""
    if fin != "stop" or msg.get("tool_calls"):
        return None
    client_tools = [t for t in (payload.get("tools") or [])
                    if not is_ours(((t or {}).get("function") or {})
                                   .get("name"), ours)]
    if not client_tools:
        return None
    route = payload.get("_route") or {}
    step = route.get("class") == "agent_step" or any(
        isinstance(m, dict) and m.get("role") == "assistant"
        and m.get("tool_calls") for m in messages)
    if not step:
        return None
    tier = payload.get("_tier") or {}
    on, src = tiers.behaviour_source(tier, "continue_stated_step")
    rec = {"eligible": False, "judged": False, "continued": False,
           "switch": {"on": on, "source": src}}
    if not on:
        rec["why"] = f"switch continue_stated_step off ({src})"
    elif tier.get("name") in ("minimal", "low"):
        rec["why"] = f"tier {tier.get('name')}: the model as it ships"
    elif (payload.get("_utility") or {}).get("utility"):
        rec["why"] = "a client's side call"
    elif landed:
        rec["why"] = "the turn landed (tools withdrawn)"
    elif not (msg.get("content") or "").strip():
        rec["why"] = "no visible text to judge"
    else:
        rec["eligible"] = True
    return rec


def _judge_stop(payload: dict, msg: dict) -> dict:
    """decide_turn.judge_stop over the stopped turn (never raises)."""
    import decide_turn
    lg = payload.get("_ledger") or {}
    return decide_turn.judge_stop(
        msg.get("content") or "", msg.get("reasoning_content") or "",
        account=lg.get("account") or "", key=lg.get("session"),
        request=lg.get("turn_key"))


def _stated_step(payload: dict, convo: list[dict], msg: dict, rec: dict):
    """Judge a stopped agent step; (prefill, the line as the client gets
    it) when it only said what it is about to do, else None. `rec` is
    filled in (x_yamadori.continued). A generator: the judgment runs in a
    thread, with heartbeats (_in_thread)."""
    status, j = yield from _in_thread(lambda: _judge_stop(payload, msg))
    if status != "ok" or not isinstance(j, dict):
        j = {"judged": False, "why": "the judgment raised",
             "failure": {"code": type(j).__name__, "situation": str(j)[:200],
                         "retryable": False}}
    for k in ("judged", "pick", "raw_pick", "tie", "distribution", "ms",
              "why", "failure", "calibration", "state", "slot", "release",
              "processed_tokens"):
        if k in j:
            rec[k] = j[k]
    if not j.get("judged"):
        return None
    if j.get("pick") != CONTINUE_PICK:
        rec["why"] = ("a tie: no decision, delivered as is" if j.get("tie")
                      else f"judged {j.get('pick')!r}: delivered as is")
        return None
    text = msg.get("content") or ""
    sep = _continue_separator(text)
    prefill = {"role": "assistant", "content": text + sep + CONTINUE_LINE,
               "reasoning_content": msg.get("reasoning_content") or ""}
    if context_full(payload, list(convo) + [prefill]):
        rec["why"] = ("judged next_step, but the continuation does not fit "
                      "this request's share: delivered as is")
        return None
    rec.update(continued=True, line=CONTINUE_LINE,
               why="judged next_step: continued once from its own text")
    print(f"  stated step: continued (p={j.get('distribution', {}).get(CONTINUE_PICK)}"
          f", {j.get('ms')} ms)", flush=True)
    return prefill, sep + CONTINUE_LINE


class _Resent:
    """A prefill llama-server re-sends as its first deltas (STEP 0) that the
    client has ALREADY been shown (a stated step's text): skipped once per
    channel. Where the deltas do not open with it (no re-send), what was
    held is released as it came."""

    def __init__(self, reasoning: str, content: str):
        self.want = {"reasoning": reasoning or "", "content": content or ""}
        self.got = {"reasoning": "", "content": ""}
        self.done = {k: not v for k, v in self.want.items()}

    def feed(self, ch: str, text: str | None) -> str:
        if not text or self.done[ch]:
            return text or ""
        g, w = self.got[ch] + text, self.want[ch]
        if len(g) < len(w) and w.startswith(g):
            self.got[ch] = g
            return ""
        self.done[ch] = True
        return g[len(w):] if g.startswith(w) else g


def _image_stop_record(stop: dict, hop: int) -> dict:
    """One call the image guard stopped, for x_yamadori.image_guard: names
    and numbers, never the argument's text."""
    return {"tool": stop.get("tool"), "argument": stop.get("argument"),
            "kind": stop.get("kind"), "chars_seen": stop.get("chars_seen"),
            "deltas": stop.get("deltas"), "seconds": stop.get("seconds"),
            "hop": hop}


def _run_turn(body: dict, streamed: bool):
    # MAX MODE (mcp/max_mode.py): every call this request makes -- the second brain, the decider, summaries, warms --
    # goes to the model server._serve_turn chose; a max request waits here (never cancelling it) until the other
    # model's work in flight has ended, and its first upstream call swaps the card.
    if body.get("_upstream_model"):
        max_mode.set_current(body["_upstream_model"])
        waited = max_mode.wait_ready(body["_upstream_model"])
        body = dict(body, _capacity=dict(body.get("_capacity") or {}, **waited))
        if body["_upstream_model"] == max_mode.MAX:
            # the decider's label priors (and its label check) are per model: read on the max model once it serves
            import decide_turn
            decide_turn.prime_for(max_mode.MAX)
    # ONE SYSTEM MESSAGE, `developer` included (system_roles.one_system):
    # before anything reads the messages, so every reader sees what renders.
    messages, roles_rec = system_roles.one_system(body.get("messages") or [])
    if messages is not body.get("messages"):
        body = dict(body, messages=messages, _roles=roles_rec)
    root, trusted, how = resolve_repo(messages, body.get("_client_ip", ""))
    db = repos.db_path(root) if root else None
    # Decided once, here, and handed to prepare(): a utility call has no
    # session in either place (session_context).
    body = dict(body, _utility=utility_of(body, strip_thinking(messages)))
    _key, state = session_context(messages, body.get("_account") or "",
                                   body.get("_session_token") or "",
                                   utility=body["_utility"],
                                   cache_key=session_id.cache_key_of(body))
    # A NEW conversation's id (#41), minted here once: prepare() resolves
    # the session again and must find the same one.
    if (state.get("_session") or {}).get("source") == "minted":
        body["_session_minted"] = state["_session"]["id"]
    # x_yamadori.gpu_room: every A4000 decision THIS request causes
    # (mcp/gpu_room.py) -- prepare's own embeddings (skills, library help),
    # then every tool run_our_tool executes. Fresh per request.
    room_log: list = []
    # FAIL FAST (pre-deploy review, 2026-09-24; revised after the live gate
    # the same night): prepare's own embeddings (skills, library
    # help, E1) wait at most gpu_room.FAIL_FAST_WAIT_S (2 s) for the A4000's
    # room lock, which an image draw holds for its whole run (up to
    # ROOM_WAIT_S, 300 s). A loaded search model takes its lease; one that is
    # not loaded is LOADED when the room comes free within the wait (after a
    # restart nothing else loads it), and skipped otherwise -- recorded
    # (x_yamadori.skills / x_yamadori.gpu_room) and decided again on the next
    # request (a RETRYABLE injection, prepare).
    with gpu_room.recording(room_log), gpu_room.fail_fast(
            "a chat turn's own embedding (skills, library help, E1) waits "
            "for the A4000 only briefly"):
        payload = prepare(body)
    state["_gpu_room"] = room_log
    payload["_gpu_room"] = room_log
    # The second brain's runs happen in threads bound to this request's
    # cancel token: the release at the end of each (admission.helper_lane)
    # finds this request's switch and record through it.
    rel = payload.get("_slot_release") or {}
    payload.setdefault("_slots_released", [])
    payload.setdefault("_slots_cleared", [])
    slots.bind_request(cancel.current(), rel.get("on", True),
                       rel.get("source", "default"),
                       payload["_slots_released"],
                       key=(payload.get("_slot") or {}).get("key"),
                       idle_on=(payload.get("_idle_clear") or {}).get("on"),
                       idle_log=payload["_slots_cleared"],
                       account=body.get("_account") or "",
                       idle_after_s=(payload.get("_idle_clear") or {})
                       .get("override_s"))
    # Earlier attempts of this same request that the server cancelled on
    # its arrival (server._supersede, #44).
    payload["_superseded"] = body.get("_superseded")
    # THE CLIENT'S OWN PROMPT MUST FIT (C1): measured here, before deep
    # thinking or any generation, and refused with context_length_exceeded
    # (an ApiError: a real HTTP 400 on both paths, since nothing has been
    # sent yet) -- never landed.
    payload["_context"] = check_client_prompt(payload)
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
    # yama_generate_image links its result on the address the client used, and
    # picks the image model from this caller's account; each call is
    # recorded for x_yamadori.
    state["_public_base"] = body.get("_public_base") or ""
    # A Responses request's hosted image_generation tool (its size), for
    # THIS request only: set every request, so it never outlives one.
    state["_image_options"] = body.get("_image_options")
    state["_account"] = body.get("_account") or ""
    # yama_recall_craft's reads in THIS request (x_yamadori.craft.reads).
    state["_craft_reads"] = []
    payload["_craft_reads"] = state["_craft_reads"]
    payload["_images"] = state.setdefault("_images", [])
    # yama_describe_image reads this request's attached images from the session
    # state, not the payload: fan-out deep-copies payloads through JSON.
    state["_attached"] = payload.pop("_attached", None) or vision.empty()
    payload["_attachments"] = vision.summary(state["_attached"])
    payload["_vision"] = state.setdefault("_vision", [])
    # The MCP-backed tools' calls in THIS request (x_yamadori.mcp.calls).
    state["_mcp_calls"] = []
    payload["_mcp_calls"] = state["_mcp_calls"]
    payload["_tool_calls"] = []
    payload.setdefault("_fold_back", [])
    rep = _repair_state(payload)
    tcheck = tool_code.state(bool(payload.get("_tool_code")),
                             fix=bool(payload.get("_fixup")))
    # The question as the client sent it, for the checks: `convo` can be the
    # very same list object, and the loop appends to it.
    question = list(messages)
    turn = corpus.new_turn()
    # The decider's rows of this request join the corpus turn (decide_turn:
    # decisions pending before this point; later ones carry it themselves).
    # Never raises.
    import decide_turn
    decide_turn.join_corpus(
        (payload.get("_ledger") or {}).get("turn_key"), turn,
        account=body.get("_account") or "",
        conversation=(payload.get("_ledger") or {}).get("session"))
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
    out = _Out(streamed, body.get("_shown_prefix") or "",
               lead=payload.get("_session_lead") or "")

    # DEEP THINKING BEFORE MAIN, by a trigger (mcp/deep.py: struggle, a task
    # kickoff, a known-hard area) or a header. Its searches go out as
    # reasoning while it runs; its hand-off becomes this turn's prefill. The
    # model's own trigger, yama_think_deeply, runs inside the loop below.
    gen = _deep_thinking(payload, messages, db, root, state)
    think = None
    while True:
        try:
            line = next(gen)
        except StopIteration as stop:
            think = stop.value
            break
        if line is None:
            # The second brain is still working: an empty delta (#39).
            yield ("heartbeat", None)
            continue
        yield from out.reasoning(line + "\n")
    payload["_think_pre"] = think
    if think and think.get("ran"):
        trig = payload.get("_deep") or {}
        lg = payload.get("_ledger") or {}
        # The struggle episode ends here, and an area or
        # a kickoff is marked done (deep.mark_ran).
        deep.mark_ran(lg.get("account") or "", lg.get("session") or "",
                      len(messages), trig.get("kind") or "forced",
                      packages=trig.get("packages"), skills=trig.get("skills"),
                      turn_key=lg.get("turn_key"),
                      auto_key=(trig.get("key") if trig.get("kind") == "auto"
                                else None))

    # THE MAIN LOOP. It runs more than once only for OUR tools on main --
    # yama_generate_image / yama_describe_image, and the delegate benchmark arm. It
    # ends when the model stops calling them, at context_full, or at the
    # tool-turn cap (tiers.tool_turn_limit), which LAND: tools withdrawn,
    # the answer asked for. ONE upstream generation per answer: every hop
    # is streamed, and the hop that makes no call of ours IS the answer.
    convo = payload["messages"]
    # The turn's own tools, before a landing withdraws them: what the next
    # request renders with, and so what the compaction store keeps.
    tools0 = list(payload.get("tools") or [])
    prefill = payload.pop("_prefill", None)
    # THE VERIFY DIRECTIVE (mcp/verify_moment.py): its line opens main's
    # turn -- in place of a server-tool trigger's directive when both fell
    # on this request (checking the work comes before going further; the
    # job's result is still inserted), never over deep thinking's visible
    # opening.
    vf = payload.get("_verify")
    if vf and vf.get("line"):
        if prefill is None or payload.get("_prefill_auto"):
            if prefill is not None:
                vf["replaced"] = prefill.get("reasoning_content")
                auto = (payload.get("_deep") or {}).get("auto")
                if isinstance(auto, dict):
                    auto["directive_replaced"] = auto.get("directive")
                    auto["directive"] = vf["line"]
            prefill = directive_prefill(vf["line"])
            vf["delivered"] = True
            if vf.get("form") == "named":
                lg = payload.get("_ledger") or {}
                deep.note_verify(lg.get("account") or "",
                                 lg.get("session") or "", vf["line"],
                                 vf.get("pick"), vf.get("key"),
                                 at=len(messages))
        else:
            vf["delivered"] = False
            vf["why_not"] = ("deep thinking's visible opening opens this "
                             "turn")
    spent = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    # Each main generation of the turn, in order: {usage, cache, reasoning,
    # prefill_reasoning}. usage reports the FINAL one (_usage_of, U1).
    gens: list[dict] = []
    n_calls = 0
    hops_added: list[dict] = []
    # (url, x_yamadori.images record) per image shown (IMAGES REACH THE CHAT).
    shown_images: list[tuple[str, dict]] = []
    # Every generation's content as the model server returned it: what the
    # model WROTE, for attributing template markers (#12).
    upstream_text: list[str] = []
    cap = payload["_turn_cap"] = _turn_cap(payload)
    d: dict = {}
    send: dict = payload
    landed = False
    # Whether any generation of this turn opened with a prefill (_warm).
    prefilled = False
    pending = ""
    # THE IMAGE GUARD (tool_code, #46): calls stopped because an image
    # argument opened as image data, each handed back as a hidden hop; past
    # IMAGE_REGENERATIONS the turn lands. x_yamadori.image_guard.
    img_rec: dict | None = None
    img_land = False
    # A CALL OF OURS THE PROXY MADE ON MAIN'S BEHALF (_deep_thinking): the
    # initial prompt's yama_plan -- or, behind the deep_tool_hop switch, a
    # yama_think_deeply -- and its result, before main's first generation.
    # A hidden hop like one the model made: the client never sees it, the
    # ledger records it with the turn (hops_added) and replays it, and hop 0
    # continues from its tool result.
    pre_hops = payload.pop("_pre_hops", None) or []
    if pre_hops:
        pc = pre_hops[0]["tool_calls"][0]
        p_name = pc["function"]["name"]
        try:
            p_args = json.loads(pc["function"]["arguments"] or "{}")
        except ValueError:
            p_args = {}
        yield from out.reasoning(
            f"`{streaming.describe_call(p_name, p_args)}`\n")
        corpus.log_tool_call(turn, root, p_name, p_args, -1)
        corpus.log_tool_result(turn, root, p_name, pre_hops[1]["content"], 0)
        payload["_tool_calls"].append(
            dict(_tool_evidence(p_name, pre_hops[1]["content"]),
                 inserted=True))
        for h in pre_hops:
            convo.append(h)
            hops_added.append(h)
    # CONTINUE A STATED STEP (above _Resent): the record (None until a stop
    # with no call is seen), and -- on the one continuation hop -- the text
    # the client already has, which the server re-sends first.
    cont_rec: dict | None = None
    cont_checked = False
    continuing = False
    resend: _Resent | None = None
    cont_lead = ""
    hop = -1
    while True:
        hop += 1
        pending = ""
        payload = tiers.rebudget(dict(payload, messages=convo), role="main")
        if continuing:
            # Measured before the continuation was chosen (context_full
            # with its prefill): the same tools, nothing appended.
            last = False
        elif hop == 0:
            # THE CLIENT'S OWN REQUEST IS NEVER LANDED (C1): it was measured
            # before the turn (check_client_prompt) and fits, so its tools
            # stay and nothing is appended. Its token fields are cut to the
            # window it was measured against (a prefill -- deep thinking's
            # hand-off -- counted by estimate on top).
            ctx = payload.get("_context") or {}
            n0 = ctx.get("tokens") if ctx.get("counted") else \
                tiers.estimate_prompt_tokens(payload)
            if prefill is not None:
                n0 = int(n0 or 0) + tiers.estimate_prompt_tokens(
                    {"messages": [prefill]})
            if pre_hops and ctx.get("counted"):
                # The inserted call and its result (the plan) on top of the
                # client's counted prompt; an estimate already has them.
                n0 = int(n0 or 0) + tiers.estimate_prompt_tokens(
                    {"messages": pre_hops})
            payload = fit_window(payload, int(n0 or 0), window_limit(payload))
            last = False
        else:
            last = context_full(payload, convo)
        if not last and hop and cap["turns"] >= cap["limit"]:
            cap["hit"] = True
            payload = _land(payload, convo, "tool_turn_cap")
            last = landed = True
        elif not last and img_land:
            payload = _land(payload, convo, "image_guard")
            last = landed = True
            if img_rec is not None:
                img_rec["landed"] = True
        elif last:
            payload = _land(payload, convo)
            landed = bool(convo) and convo[-1].get("content") == \
                LANDING_PROMPT
        send = payload
        # A prefill opens this hop: deep thinking's hand-off on hop 0, or the
        # fold-back after a yama_think_deeply result on the hop that follows it.
        if prefill is not None:
            send = dict(payload, messages=list(convo) + [prefill])
            # A stated step's continuation prefills the model's OWN turn,
            # which the client keeps: not a prefill it drops (_warm).
            prefilled = prefilled or not continuing
        if (tool_code.IMAGE_GUARD and payload.get("tools")
                and (payload.get("_tier") or {}).get("name")
                not in ("minimal", "low")):
            # Main's generation, with tools: the reader stops a call whose
            # image argument opens as image data (_post_events_raw). Not at
            # `minimal`/`low` (operator, 2026-09-26): those are the model as
            # it ships, with nothing of ours.
            send = dict(send, _image_guard=True)
        d = {}
        # CHANNEL ORDER within a hop: content is held until it passes
        # HOLD_CONTENT_CHARS (it is the answer: stream it live) or the hop
        # ends (a hop that calls one of OUR tools sends its held preface as
        # reasoning; the final hop flushes it as content).
        # A stated step's continuation opens with its line (CONTINUE_LINE),
        # held like any preface: after the fix-up's reasoning line, before
        # the calls.
        held = cont_lead if continuing else ""
        live = False
        try:
            for kind, item in _post_events("/v1/chat/completions", send):
                if kind == "done":
                    d = item
                    continue
                if kind == "beat":
                    # A tool call is being written (#44): an empty delta,
                    # so the client -- and any relay between -- sees a live
                    # stream. Nothing is added to reasoning or content.
                    if streamed:
                        yield ("heartbeat", None)
                    continue
                rpiece = item.get("reasoning_content") or ""
                piece = item.get("content") or ""
                if resend is not None:
                    # The stated step's reasoning and text, re-sent: the
                    # client has them already.
                    rpiece = resend.feed("reasoning", rpiece)
                    piece = resend.feed("content", piece)
                if rpiece:
                    yield from out.reasoning(rpiece)
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
            if isinstance(e, cancel.Cancelled) or cancel.cancelled():
                # The client went away (mcp/cancel.py): nothing to tell it.
                raise cancel.Cancelled(str(e)) from e
            # ONE ERROR PATH (E1): raised on both paths. stream_body turns it
            # into an HTTP error when nothing has been sent yet, else into
            # the SSE error event -- never into assistant content.
            # finish "error": a failed turn, never read as an empty answer
            # (mcp/test_utility.py, the chars=0 replay).
            corpus.log_answer(turn, root, "", hop,
                              (time.time() - t_start) * 1000, finish="error")
            raise
        n_calls += 1
        u = d.get("usage") or {}
        for k in spent:
            spent[k] += int(u.get(k) or 0)
        # This generation's own usage (the length notice names its tokens);
        # the turn's usage is decided at the end (_usage_of).
        d["usage"] = dict(u)
        msg = d["choices"][0]["message"]
        if continuing:
            # The prefill in the message exactly once (llama-server re-sends
            # it, STEP 0; where it did not, it is put back): the turn is the
            # stated step's text, the line and the continuation.
            pc = prefill.get("content") or ""
            pr = prefill.get("reasoning_content") or ""
            if not (msg.get("content") or "").startswith(pc):
                msg["content"] = pc + (msg.get("content") or "")
            got_r = msg.get("reasoning_content") or ""
            if pr.strip() and not got_r.lstrip().startswith(pr.strip()):
                msg["reasoning_content"] = pr + ("\n" + got_r if got_r
                                                 else "")
            cont_rec["call_followed"] = (bool(msg.get("tool_calls"))
                                         or d.get("_image_arg") is not None)
            cont_rec["finish"] = d["choices"][0].get("finish_reason")
            continuing, resend, cont_lead = False, None, ""
        gens.append({"usage": dict(u), "cache": d.get("_cache"),
                     "reasoning": msg.get("reasoning_content") or "",
                     "prefill_reasoning": (prefill or {}).get(
                         "reasoning_content") or "",
                     "which": f"hop {hop}"})
        upstream_text.append(msg.get("content") or "")
        # A call stopped by the image guard: nothing of this generation runs
        # or reaches the client; it is handed back below as NOT EXECUTED.
        img = d.get("_image_arg")
        calls = [c for c in (msg.get("tool_calls") or [])
                 if is_ours(c.get("function", {}).get("name"), ours)] \
            if img is None else []
        if held:
            if (calls or img is not None) and not last:
                yield from out.reasoning(held)
            elif img is None and tcheck is not None and any(
                    not is_ours(c.get("function", {}).get("name"), ours)
                    for c in (msg.get("tool_calls") or [])):
                # A preface to CLIENT calls waits for the code check: the
                # fix-up's reasoning line must go out before any content.
                pending = held
            else:
                yield from out.content(held)
            held = ""
        # The landing is the last generation whatever it asked for: a call
        # made there is not run (nothing would read its result).
        if (not calls and img is None) or last:
            if not cont_checked and img is None:
                # CONTINUE A STATED STEP: once per request, on the first
                # generation that ends the turn.
                cont_checked = True
                cont_rec = _stated_step_gate(
                    payload, messages, msg,
                    d["choices"][0].get("finish_reason"), ours, last)
                nxt = None
                if cont_rec is not None and cont_rec.pop("eligible"):
                    nxt = yield from _stated_step(payload, convo, msg,
                                                  cont_rec)
                if nxt is not None:
                    prefill, cont_lead = nxt
                    resend = _Resent(prefill["reasoning_content"],
                                     prefill["content"])
                    continuing = True
                    payload["_continued"] = cont_rec
                    continue
                payload["_continued"] = cont_rec
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
            # An old name of ours (copied from a replayed hop) runs as its
            # `yama_*` tool; the call itself stays as the model wrote it.
            fn = canonical_tool_name(c["function"]["name"])
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield from out.reasoning(f"`{streaming.describe_call(fn, args)}`\n")
            corpus.log_tool_call(turn, root, fn, args, hop)
            t0 = time.time()
            # An image takes a minute or more, and so can looking at one: a
            # thread with heartbeats. So can deep thinking (yama_think_deeply),
            # and a plan (yama_plan): its result is the next hop's context,
            # nothing prefilled after it -- main goes on acting.
            if fn == deep.PLAN_TOOL_NAME:
                status, res = yield from _in_thread(
                    lambda args=args: _plan_task(payload, args, db, root,
                                                 state))
            elif fn == deep.TOOL_NAME:
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
                    # A machine-built result (no conclusion) gets its own
                    # line (deep.THINK_REASONING_MACHINE).
                    concluded = not (ran_now[-1].get("handoff") or {}).get(
                        "machine_built")
                    next_prefill = {"role": "assistant",
                                    "reasoning_content":
                                        deep.think_reasoning(concluded),
                                    "content": shomen.opening(seeds)}
                    payload.setdefault("_fold_back", []).append(
                        {"job": "investigate", "phrase": "investigate",
                         "into": "prefill", "trigger": "model",
                         "seeds": [x.get("word") for x in seeds]})
            else:
                n_img0 = len(payload["_images"])
                status, res = yield from _in_thread(
                    lambda fn=fn, args=args: run_our_tool(fn, args, db, root,
                                                          tracker, state))
                if fn == images.TOOL_NAME and status == "ok" and res:
                    # IMAGES REACH THE CHAT: the picture goes to the client
                    # now, as content; the model reads that it is shown.
                    shown = images.shown_on_main(
                        res, looks=vision.TOOL_NAME in ours)
                    if shown:
                        res, md, url = shown
                        yield from out.image(md + "\n\n", url)
                        new = payload["_images"][n_img0:]
                        rec = new[-1] if new else None
                        if rec is None:
                            rec = {"ok": True}
                            payload["_images"].append(rec)
                        rec["emitted"] = {
                            "ms": round((time.time() - t_start) * 1000),
                            "after_hop": hop, "before_hop": hop + 1,
                            "at": "stream" if streamed else "answer_start",
                            "order": len(out.images)}
                        shown_images.append((url, rec))
                elif fn == vision.TOOL_NAME and status == "ok" and res:
                    # One screened line of what the vision model saw: a
                    # heartbeat once content has started (CHANNEL ORDER).
                    seen = _describe_line(res)
                    if seen:
                        yield from out.reasoning(seen)
            res = res if status == "ok" and res else json.dumps(
                images.ImageError(
                    "TOOL_RAISED", f"the {fn} call died without a result",
                    retryable=False,
                    remedies=[{"fixable_by": "operator",
                               "action": "check the proxy log"}]).envelope(fn))
            corpus.log_tool_result(turn, root, fn, res,
                                   (time.time() - t0) * 1000)
            payload["_tool_calls"].append(_tool_evidence(fn, res))
            # A hand-off or a plan crosses WHOLE (operator, 2026-09-27: the
            # 6,000-character cut and its marker broke pagoda-h2): its
            # length is bounded by its job's own generation budget, and the
            # window check handles one that does not fit (context_full lands
            # the next hop). Other results keep repeats.RESULT_CAP.
            tmsg = {"role": "tool", "tool_call_id": c["id"],
                    "content": res if fn in (deep.TOOL_NAME,
                                             deep.PLAN_TOOL_NAME)
                    else repeats.cap_tool_result(res, fn, args)}
            convo.append(tmsg)
            hops_added.append(tmsg)
        if img is not None:
            # THE IMAGE GUARD'S HAND-BACK (#46): the stopped call -- its
            # image value replaced, so the model sees what it did without
            # the data -- and a NOT EXECUTED result that says why and what
            # to pass instead (tool_code.image_result); a complete call
            # before it in the same generation is not run either. A hidden
            # hop like the image tools' (the ledger replays it before the
            # delivered turn), counted in tool_turns; the model then writes
            # the turn again.
            for c in hop_msg["tool_calls"]:
                tmsg = {"role": "tool", "tool_call_id": c.get("id") or "",
                        "content": (tool_code.image_result(
                            img, generated=images.TOOL_NAME in ours)
                            if c.get("id") == img["call_id"]
                            else tool_code.image_other_result(img))}
                convo.append(tmsg)
                hops_added.append(tmsg)
            if img_rec is None:
                img_rec = {"stopped": [], "regenerated": 0, "landed": False}
            img_rec["stopped"].append(_image_stop_record(img, hop))
            if len(img_rec["stopped"]) > tool_code.IMAGE_REGENERATIONS:
                img_land = True
            else:
                img_rec["regenerated"] += 1
            yield from out.reasoning(
                f"`stopped a {img['tool']} call: its {img['argument']} "
                f"argument was image data ({img['kind']}); "
                + ("asking for the answer`\n" if img_land
                   else "asking for a file path`\n"))
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
                    if not is_ours(c.get("function", {}).get("name"), ours)]
    stray = [c for c in (msg.get("tool_calls") or [])
             if is_ours(c.get("function", {}).get("name"), ours)]
    img_last = d.get("_image_arg")
    if img_last is not None:
        # THE IMAGE GUARD on the last generation (a landing, or a full
        # context): the stopped call is never forwarded, and there is no
        # hop left to hand it back in.
        stray, client_calls = list(msg.get("tool_calls") or []), []
        if img_rec is None:
            img_rec = {"stopped": [], "regenerated": 0, "landed": landed}
        img_rec["stopped"].append(dict(_image_stop_record(img_last, hop),
                                       withheld=True))
    fan = None
    if stray:
        # Withheld (the client does not have them), so the finish must not
        # claim tool calls the client will never receive. Nothing is written
        # in their place: the "[no answer: ...]" defect notices were REMOVED
        # 2026-09-27 (docs/CONSTANTS-AUDIT.md); an empty answer with
        # finish_reason stop is what the client gets.
        fin = "stop" if fin == "tool_calls" else fin
    if fin == "incomplete":
        took = (d.get("_transport") or {}).get("dropped_after")
        t = (f"\n\n[the connection to the model dropped"
             f"{f' after {took}s' if took is not None else ''}; "
             f"the answer above is the part that arrived]")
        yield from out.content(t)
        content += t
    # A `length` finish is a BUDGET EVENT: finish_reason "length" says so,
    # and the content is what was generated (the "[no answer: ...]" /
    # "[answer cut off ...]" notice and the empty-answer notice were REMOVED
    # 2026-09-27, docs/CONSTANTS-AUDIT.md). Neither is treated as an answer
    # (no code check, no fan-out): _finish_text runs on a "stop" with text.
    if fin == "length" or (fin == "stop" and not content.strip()
                           and not client_calls):
        print(f"  finish={fin}, {len(content.strip())} chars of content; "
              f"delivered as generated", flush=True)

    if client_calls:
        content = yield from _finish_calls(payload, tcheck, msg, client_calls,
                                           content, question, ours, out,
                                           pending)
    elif fin == "stop" and content.strip():
        content, fan, slot_msg = yield from _finish_text(
            payload, rep, msg, fin, content, question, send, slot_msg, out)
        # A continuation, and a second-brain winner delivered in place, are
        # model-written too.
        upstream_text.append(slot_msg.get("content") or "")
        if fan and isinstance(fan.get("_winner"), dict):
            upstream_text.append(fan["_winner"].get("content") or "")
    msg["content"] = content
    # THE CONVERSATION'S ID RIDES IN THE CALL IDS (#41, _carry_session): from
    # here on -- the streamed delta, the blocking answer, the ledger's keys,
    # the compaction store and the warm -- every copy has the rewritten ids.
    client_calls = _carry_session(payload, client_calls, slot_msg)
    # THE WARM RACE, streamed (live gate 2026-09-24, #5): the client has the
    # calls the moment they are yielded, and a harness that runs the tool
    # then and there can send its next request before this turn reaches
    # _warm below. A HOLD is registered first, so that request waits for the
    # warm this turn is about to schedule (_warm takes the hold over, or
    # releases it when nothing is warmed).
    hold = _warm_hold(payload) if client_calls else None
    # What the duplicate filter still holds goes out now (IMAGES REACH THE
    # CHAT), and a compaction summary's session line if nothing streamed it
    # yet (#41; answers carry no line).
    yield from out.flush_held()
    yield from out.flush_lead()
    if client_calls:
        msg["tool_calls"] = client_calls
        fin = "tool_calls"
        try:
            yield ("calls", client_calls)
        except BaseException:
            _warm_release(payload, hold)
            raise
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
    if any(f.get("phrase") == "investigate"
           for f in payload.get("_fold_back") or []):
        payload["_fold_back_answer"] = fold_back_answer(content)
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
    # STRAY TEMPLATE MARKERS (#12, _StrayMarkers): the client's copy has them
    # out -- the stream filtered them as it went; the blocking path cleans
    # the whole answer here, the same filters in the same order (image
    # duplicates, then markers). `delivered` keeps the slot's text: the
    # ledger renders it, keyed by the client's copy (#10), so the next
    # request extends the slot. `clean`: the model's answer as the client
    # gets it, without the lines the proxy put before it.
    tnote = payload.get("_tool_note") or {}
    if streamed:
        stray = out.markers
        clean = strip_stray_markers(content or "")
    else:
        # The model's text is filtered WITHOUT our note (the marker filter
        # never sees the proxy's own lines), which is put back after it as
        # the stream sends it (_Out.note).
        body_text = content or ""
        ours = tnote.get("appended") or ""
        if ours and body_text.endswith(ours):
            body_text = body_text[:-len(ours)]
        else:
            ours = ""
        clean, stray = _StrayMarkers.clean(
            out.dedup.strip(body_text) if out.images else body_text)
        if ours:
            clean += ("\n\n" if clean.strip() else "") + tnote["note"]
    if streamed:
        shown = "".join(out.shown)
        if shown.strip() != (content or "").strip():
            stored["content"] = shown
            payload["_stored_differs"] = True
    elif payload.get("_session_lead") or out.images or clean != content:
        # A COMPACTION SUMMARY'S SESSION LINE on the blocking path (#41; no
        # answer carries one): the client receives and stores the line and
        # the summary; the slot generated the summary. Keyed by the client's
        # copy, rendered as the slot's (#10 path), so the model never reads
        # it. IMAGES REACH THE CHAT: the image lines open the answer, in
        # the order they were made -- where the stream puts them -- and the
        # model's own copy of one is removed from the client's copy only.
        stored["content"] = ((payload.get("_session_lead") or "")
                             + "".join(out.images) + clean)
        msg["content"] = stored["content"]
        payload["_stored_differs"] = True
    for url, rec in shown_images:
        rec["duplicates_stripped"] = int(out.dedup.removed.get(url, 0))
    payload["_template_markers"] = _markers_record(
        payload, stored.get("content") or "", "".join(upstream_text), stray)
    # THE TURN AS THE NEXT REQUEST WILL RENDER IT: its reasoning is the echo
    # of what this client was shown if it echoes (seen on its past turns),
    # else the slot's own, which the ledger records and restores (switch
    # restore_reasoning, 2026-09-27; off: nothing, the pass-through, and the
    # hidden hops' reasoning emptied). The warm and the compaction store use
    # that form.
    # Whether this client echoes: seen on this request's past assistant
    # turns; on a first turn (none to see), what the ACCOUNT's client has
    # been seen doing -- echoing is a property of the harness, not the
    # conversation (_echo_guess). Unknown: it strips (Hermes does, for this
    # provider).
    acct = ((payload.get("_ledger") or {}).get("account")) or ""
    if any(isinstance(m, dict) and m.get("role") == "assistant"
           for m in messages):
        echoes = bool(((payload.get("_ledger") or {}).get("restored") or {})
                      .get("echoed_reasoning"))
        _echo_note(acct, echoes)
    else:
        echoes = _echo_guess(acct)
    client_turn = {k: v for k, v in delivered.items()
                   if k != "reasoning_content"}
    # PAST REASONING IS RESTORED (switch `restore_reasoning`, operator
    # 2026-09-27): a client that strips gets the slot's reasoning back on its
    # next request, so the warm and the compaction store render it too.
    rr_on = bool(((payload.get("_ledger") or {}).get("restore_reasoning")
                  or {}).get("on"))
    if echoes:
        client_turn["reasoning_content"] = (
            "".join(out.reasoning_shown) if streamed
            else delivered["reasoning_content"])
    elif rr_on:
        client_turn["reasoning_content"] = delivered["reasoning_content"]
    hop_ids = {id(h) for h in hops_added}
    # HIDDEN HOPS AND AN ECHOING CLIENT (ledger_restore, 2026-09-26): the
    # next request renders the hops with their own reasoning and this turn
    # with the slot's, so the warm and the compaction store do too; for a
    # client that strips, the hops' reasoning is emptied -- unless past
    # reasoning is restored, which keeps them as the slot holds them.
    echo_shown = client_turn.get("reasoning_content") if echoes else None
    # What an echoing client WILL send back for this turn is what it was
    # shown -- recorded as the hop echo whenever the turn has hops, whatever
    # `echoes` guessed. On a conversation's FIRST request `echoes` is only
    # _echo_guess (the account's history), and the kickoff's inserted
    # yama_plan hop sits exactly there: an echoing client on an account
    # guessed "strips" sent the planner's streamed reasoning (9,544 chars;
    # the slot's own was 67) back as main's, which passed through, so the
    # request diverged inside the turn and re-read it (deploy check
    # 2026-09-27, h5: agent loop [echo] step 2 reused 4,131 of 7,133,
    # processed 3,002). The record is a hash; it matches only an echo.
    hop_echo_shown = echo_shown
    if hops_added and not echoes:
        hop_echo_shown = ("".join(out.reasoning_shown) if streamed
                          else delivered.get("reasoning_content")) or None
    keep_hops = bool(hops_added) and (bool((echo_shown or "").strip())
                                      or rr_on)
    if keep_hops:
        client_turn["reasoning_content"] = slot_msg.get(
            "reasoning_content") or ""
    warm_base = [dict(m, reasoning_content="") if id(m) in hop_ids
                 and m.get("role") == "assistant" and not keep_hops else m
                 for m in base_messages]
    # ONE ORDER FOR TOOL-CALL ARGUMENTS: the next request renders every call
    # with sorted keys (prepare), so the warm and the compaction store do too
    # -- the delivered turn and any hidden hop of ours, which the slot
    # generated in the model's own order.
    client_turn = sort_call_arguments([client_turn])[0]
    warm_base = sort_call_arguments(warm_base)
    account, session = ledger_scope(payload)
    if session:
        salt = (payload.get("_ledger") or {}).get("salt") or ""
        own = (delivered.get("reasoning_content") or "") if rr_on else ""
        ledger_record_turn(account, session, delivered, hops_added or None,
                           stored=stored,
                           prev=chain_keys(messages, salt)[-1] if messages
                           else "", echo=hop_echo_shown,
                           slot_reasoning=slot_msg.get("reasoning_content"),
                           reasoning=own or None)
        if isinstance(payload.get("_ledger"), dict):
            payload["_ledger"]["reasoning_recorded"] = len(own.strip() and own)
        # An answer that carries no id (no client call) is recorded with it
        # (#41, THE ANSWER RECORD), so the next request finds its session.
        _record_answer_session(payload, messages,
                               stored.get("content") or "", client_calls)
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
                             landed, hold=hold, prefilled=prefilled)
    _log_turn(state, delivered, tcheck, think, fan, rep)
    # THE COMPACTION LINK (#38): a compaction's window starts when it ENDS,
    # and its continuation is recognised by the summary it carries; any
    # other conversation turn refreshes the account's last activity.
    if payload.get("_utility_kind") == "compaction":
        # The summary as the client will carry it (stray markers out).
        _compaction_done(body.get("_account") or "",
                         (payload.get("_slot") or {}).get("key"), clean)
    elif not body["_utility"].get("utility"):
        _session_touch(body.get("_account") or "", state.get("_key") or "")
    # THE SELF-IMPROVEMENT LOOP (Phase 0.6): this conversation's earlier
    # decisions are observed against what the client sent back, then this
    # request's decision -- or non-decision -- is recorded.
    payload["_deep_record"] = _deep_record(payload, messages, think)
    for u in payload.pop("_extra_usage", None) or []:
        n_calls += 1
        for k in spent:
            spent[k] += int(u.get(k) or 0)
    # A fold-back continuation is main's own last generation of the answer.
    if payload.get("_final_gen"):
        gens.append(payload.pop("_final_gen"))
    usage, usage_rec = _usage_of(gens, spent, n_calls, payload)
    payload["_image_guard_rec"] = img_rec
    d["x_yamadori"] = _x_yamadori(payload, hops=n_calls, fan=fan,
                                  think=think, repair=rep, tool_check=tcheck)
    d["x_yamadori"]["usage"] = usage_rec
    recent_turns.note(d["x_yamadori"])
    corpus.log_answer(turn, root, content, hop,
                      (time.time() - t_start) * 1000,
                      finish=fin, tool_calls=len(client_calls))
    d["usage"] = usage
    return d


# USAGE (U1, U2; docs/OPENAI-CONFORMANCE.md, 2026-09-25). `usage` describes
# the FINAL main generation of the turn -- the one the client's answer came
# from (the last hop, or a fold-back continuation when one ran):
#   prompt_tokens      that generation's prompt: the context the conversation
#                      occupies now. It was the SUM over the turn's hops, and
#                      Hermes reads it as the context size
#                      (context_compressor.py: last_prompt_tokens =
#                      usage.prompt_tokens), so every multi-hop turn read as a
#                      context 2-3x its size and compacted early.
#   completion_tokens  what that generation produced (reasoning included, as
#                      the spec counts it), which is what the client received
#                      from it.
#   total_tokens       their sum.
#   prompt_tokens_details.cached_tokens     prompt tokens reused from the
#                      slot's cache (llama-server's own usage field, else
#                      its timings' cache_n: slots.cache_record).
#   completion_tokens_details.reasoning_tokens   llama-server does not report
#                      it (server-task.cpp usage_json_oaicompat), so the
#                      reasoning the generation wrote is counted with the
#                      model's own tokenizer (/tokenize; a prefill's echoed
#                      reasoning is not counted); ABSENT when it cannot be
#                      counted -- never guessed.
# HIDDEN HOPS -- a generation that called one of OUR tools (images,
# yama_think_deeply, the delegate arm) -- have no field in the spec. Their output
# is in the final generation's prompt (the hop and its tool result are part
# of what the model read), so prompt_tokens already covers them as context;
# their own generation is the proxy's work, not the client's answer. Every
# generation's numbers, and the sums the old `usage` carried, are in
# x_yamadori.usage; `usage.hops` moved there (and x_yamadori.hops is the
# same count).
REASONING_COUNT_TIMEOUT = 10.0


def _count_text_tokens(model: str, text: str) -> int | None:
    try:
        toks = _upstream_json(f"/upstream/{model}/tokenize",
                              {"content": text, "add_special": False},
                              timeout=REASONING_COUNT_TIMEOUT).get("tokens")
    except cancel.Cancelled:
        raise
    except Exception:                                            # noqa: BLE001
        return None
    return len(toks) if isinstance(toks, list) else None


def _usage_of(gens: list[dict], spent: dict, n_calls: int,
              payload: dict) -> tuple[dict, dict]:
    """(usage for the client, x_yamadori.usage)."""
    final = gens[-1] if gens else {}
    u = final.get("usage") or {}
    cache = final.get("cache") or {}
    prompt = u.get("prompt_tokens")
    if prompt is None:
        prompt = cache.get("prompt")
    prompt = int(prompt or 0)
    completion = int(u.get("completion_tokens") or 0)
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "total_tokens": prompt + completion}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = cache.get("reused")
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached)}
    rec = {"final": final.get("which"), "generations": n_calls,
           "summed": dict(spent), "reasoning_tokens": None}
    text = final.get("reasoning") or ""
    pre = (final.get("prefill_reasoning") or "").strip()
    if pre and text.lstrip().startswith(pre):
        text = text.lstrip()[len(pre):]
    if not text.strip():
        usage["completion_tokens_details"] = {"reasoning_tokens": 0}
        rec["reasoning_tokens"] = "none written"
    else:
        n = _count_text_tokens(payload.get("model") or "bonsai", text)
        if n is not None:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": min(n, completion) if completion else n}
            rec["reasoning_tokens"] = "tokenized"
        else:
            rec["reasoning_tokens"] = "absent: the tokenizer did not answer"
    rec["per_generation"] = [
        {"which": g.get("which"),
         "prompt_tokens": (g.get("usage") or {}).get("prompt_tokens"),
         "completion_tokens": (g.get("usage") or {}).get("completion_tokens"),
         "cached_tokens": (g.get("cache") or {}).get("reused")}
        for g in gens]
    return usage, rec


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
    # Every blocking unit is repaired. The fix-up SCOPE (#56: project files
    # only, scratch-name table) was REMOVED 2026-09-27
    # (docs/CONSTANTS-AUDIT.md: its table named one run's files).
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
        # OUR note goes past the imitation filter (IMITATED NOTES, #36): the
        # model's text before it is released through the filters first. The
        # blocking path splits it off the same way (_run_turn).
        yield from out.note(tnote)
        payload["_tool_note"] = {"note": tnote, "appended": t}
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
    fan, win = None, None
    status, res = yield from _in_thread(
        lambda: _fan_out(payload, dict(msg, content=content), fin))
    if status == "ok" and res:
        fan, win = res
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
    # The answer's last generation: what `usage` reports (_usage_of, U1).
    payload["_final_gen"] = {
        "usage": dict(d.get("usage") or {}), "cache": d.get("_cache"),
        "reasoning": ((d.get("choices") or [{}])[0].get("message") or {})
        .get("reasoning_content") or "",
        "prefill_reasoning": pre["reasoning_content"],
        "which": "continuation"}
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
    # Arguments compared as the template RENDERS them, key order included
    # (ONE ORDER FOR TOOL-CALL ARGUMENTS): a call the model wrote in another
    # order than the sorted one the next request renders is not in the slot
    # as delivered.
    def calls(m):
        return [(c.get("id"), (c.get("function") or {}).get("name"),
                 _args_as_rendered((c.get("function") or {}).get("arguments")))
                for c in (m.get("tool_calls") or [])]
    return ((a.get("content") or "") == (b.get("content") or "")
            and calls(a) == calls(b))


def _same_reasoning(a: dict, b: dict) -> bool:
    """The two turns' reasoning as the template renders it (trimmed)."""
    return ((a.get("reasoning_content") or "").strip()
            == (b.get("reasoning_content") or "").strip())


def _warm(payload: dict, base: list[dict], delivered: dict, slot_msg: dict,
          landed: bool, hold: "threading.Event | None" = None,
          prefilled: bool = False) -> dict:
    """Schedule the warm; the record says whether and why. `hold`: the
    pending event _warm_hold registered before the calls went out -- taken
    over by the warm, or released when nothing is warmed. `prefilled`: a
    generation of this turn opened with a prefill (PREFILLED TURNS)."""
    sl = payload.get("_slot") or {}
    why = None
    if not sl.get("key") or sl.get("transient"):
        why = "not a conversation turn"
    elif _same_turn(delivered, slot_msg) and (
            not prefilled or _same_reasoning(delivered, slot_msg)):
        # A PREFILLED turn whose reasoning comes back (restored, or echoed
        # exactly) renders as the slot holds it: nothing to warm.
        why = "the slot already holds the turn as delivered"
    elif landed:
        why = ("the turn landed; its prompt carries the landing request, "
               "which the client does not keep")
    elif not WARM:
        why = "YAMADORI_WARM=0"
    elif image_input.without_image_bytes(base)[1]:
        # the warm renders a TEXT prompt (/apply-template, /completion): an
        # image part would become its media marker as text, a prefix the
        # multimodal request never has
        why = "the conversation carries images passed to the model"
    if why:
        _warm_release(payload, hold)
        return {"sent": False, "why": why}
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
    if prefilled:
        # PREFILLED TURNS: the expectation is counted by the warm itself
        # (_prefill_floor), not the generation's prompt.
        rec.update(prefilled=True, expect_reused_at_least=None,
                   generated_prompt=last.get("prompt"),
                   generated_reused=last.get("reused"))
        if _same_turn(delivered, slot_msg):
            rec["why"] = ("the turn opened with a prefill the client does "
                          "not send back")
    msgs = list(base) + [delivered]
    fields = {k: payload[k] for k in ("tools", "chat_template_kwargs",
                                      "enable_thinking", "reasoning_effort")
              if k in payload}
    model = payload.get("model") or "bonsai"
    # Registered BEFORE the thread starts, so a fast client's next request
    # already finds it pending (wait_for_warm) and its record (warm_before).
    # A hold registered before the calls went out becomes this warm's event:
    # a request already waiting on it keeps waiting, now for the warm.
    ev = hold if hold is not None else _threading.Event()
    ev.hold_until = None
    _WARM_PENDING[sl["key"]] = ev
    _WARMS_DONE.pop(sl["key"], None)
    _WARMS_DONE[sl["key"]] = rec
    while len(_WARMS_DONE) > 512:     # conversations that never came back
        _WARMS_DONE.pop(next(iter(_WARMS_DONE)), None)
    _threading.Thread(target=_warm_now,
                      args=(sl["key"], model, msgs, fields, rec, ev),
                      daemon=True).start()
    return rec


def _upstream_json(path: str, body: dict, timeout: int = 120) -> dict:
    if path.startswith("/upstream/"):
        # MAX MODE's backstop (mcp/max_mode.py): never a request that loads a model max mode holds off the card
        max_mode.guard(path.split("/")[2])
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


def _idle_guard(key: str) -> str | None:
    """slots IDLE CLEAR's veto: a conversation whose warm is pending (held
    or running) is not idle, whatever its slot's clock says."""
    ev = _WARM_PENDING.get(key)
    return "a warm is pending" if ev is not None and not ev.is_set() else None
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
# A HOLD (_warm_hold) that no warm took over by then is abandoned: the turn
# that registered it died between sending its calls and scheduling the warm.
# A breaker, not a measurement: that stretch is in-process work taking ms.
WARM_HOLD_S = float(os.environ.get("YAMADORI_WARM_HOLD", "30"))
slots.idle_guard = _idle_guard


# THE ECHO HABIT on a conversation's FIRST turn, where the conversation has
# shown nothing yet (live gate 2026-09-24, second run, the repair test): the
# account's LAST observation said "echoes" -- the agent-loop test's echoing
# client had just run on the same key -- so the warm rendered the turn WITH
# its reasoning, the stripping client sent it back without, and the next
# request diverged at the turn's think block. So a first turn guesses
# "echoes" only when this account has been seen echoing and NEVER seen
# stripping; anything mixed or unknown is "strips", the documented default.
# The account keeps the SET of what it was seen doing ("0" strips, "1"
# echoes), not a window: ECHO_WINDOW (the last 8 observations, "a choice:
# 8") was REMOVED 2026-09-27 (docs/CONSTANTS-AUDIT.md). A legacy
# `echo_history` string reads as the set of its characters.


def _echo_note(account: str, echoes: bool) -> None:
    """Record that this account's client was seen echoing (or stripping)
    past reasoning (a request that carries assistant turns)."""
    seen = set(nebari.ledger_get(account, "client", "echo_history") or "")
    new = "".join(sorted(seen | {"1" if echoes else "0"}))
    if new != "".join(sorted(seen)):
        nebari.ledger_put(account, "", "client", "echo_history", new)


def _echo_guess(account: str) -> bool:
    """A first turn's guess: echoes only when this account was only ever
    seen echoing (a record from before the history, the single last flag,
    counts as one observation)."""
    hist = nebari.ledger_get(account, "client", "echo_history")
    if hist is None:
        hist = nebari.ledger_get(account, "client", "echoes_reasoning") or ""
    return bool(hist) and set(hist) == {"1"}


# A warm that diverges inside the turn the slot just generated restores from
# the checkpoint llama-server makes 4 tokens before a prompt's end
# (server-context.cpp checkpoint_offsets {4 + n_ubatch, 4}; live gate
# 2026-09-24: reused 4452 of 4456, 2886 of 2890), so it re-reads those 4
# tokens by design. `short` allows that and a little more: 8 is a choice.
WARM_CHECKPOINT_SLACK = 8

# PREFILLED TURNS (Octopus v0b-V0-xhigh-1, step 1, 2026-09-25: a kickoff, the
# plan prefilled as main's reasoning; prompt 10660, then the warm "reused
# 7526 processed 3055 -- SHORT", and the next request reused 10581 of 10611).
# The warm was RIGHT: it rendered the turn stripped, as Hermes sent it. What
# was wrong was the expectation. The generation's prompt carries the prefill
# -- `<|im_start|>assistant\n<think>\n` + the plan -- which no client sends
# back, so the warm diverges from it where the plan begins, and the server
# restores the latest context checkpoint at or before that point. The ones
# it makes at a prompt's end (4 and 4 + n_ubatch tokens before it, 512 in
# config.yaml) lie INSIDE any prefill longer than that; the one it makes at
# the start of the LAST USER MESSAGE (llama.cpp server-context.cpp
# `last_user_pos`; a tool result is not a user message there, common/chat.cpp
# message_delimiters) is the floor. 7526 can only have been that one (the
# slot started empty, reused 0, and the prompt held one user message), and
# 3055 ~ the spec (~2245 tokens, x_yamadori.deep) + the delivered turn
# (inferred from the server source and the counts, not tokenized). So a
# prefilled turn's expectation is the prompt BEFORE the last user message,
# counted by the model server's own /tokenize; the re-read of that message
# is the server's checkpoint placement, not a miss. When an EARLIER request
# processed that message (a struggle prefill), its checkpoint may since
# have been evicted (checkpoint_min_step 8192), and a SHORT then says so,
# as for any other warm. And a prefilled turn is warmed even when the
# delivered turn equals what the slot generated: the slot still holds the
# prefill's reasoning, which the client never sends back, so without a warm
# the NEXT request pays that same re-read (offline, test_ledger [prefill
# warm]: no warm was sent for a plain terminal call); with one it happens
# while the harness runs its tool. The same tokens either way.
USER_TURN = "<|im_start|>user"
TOOL_TURN = "<|im_start|>user\n<tool_response>"


def _last_user_start(prompt: str) -> int:
    """Where the last user message (not a tool result) opens in a rendered
    prompt; -1 when there is none."""
    i = len(prompt)
    while True:
        i = prompt.rfind(USER_TURN, 0, i)
        if i < 0 or not prompt.startswith(TOOL_TURN, i):
            return i


def _prefill_floor(model: str, prompt: str, timeout: float) -> int | None:
    """Tokens before the last user message of the warmed prompt (the
    server's checkpoint there), or None when they cannot be counted."""
    i = _last_user_start(prompt)
    if i < 0:
        return None
    toks = _upstream_json(f"/upstream/{model}/tokenize",
                          {"content": prompt[:i], "add_special": True},
                          timeout=timeout).get("tokens")
    return len(toks) if isinstance(toks, list) else None


# A PREFILL MID-CONVERSATION (Octopus v0f-V0-xhigh-1 step 25, 2026-09-26: a
# struggle-triggered hand-off, ~3.5k chars, prefilled as main's reasoning;
# prompt 63,397 reused 59,343; the warm then "reused 54,604 processed 8,280"
# for a step that wrote 503 tokens, and the next request extended the warm,
# reused 62,884 processed 1,628). The divergence is the prefill's, as above:
# the client drops it, so the warm diverges where the hand-off begins. What
# the server restores there is decided by its checkpoint THINNING
# (server-context.cpp create_checkpoint): every task that makes a checkpoint
# first erases each OTHER task's checkpoint lying within checkpoint_min_step
# (8192, common.h; config.yaml does not set it) of the previous one kept.
# The prefilled generation made its own two near its prompt's end -- inside
# the hand-off -- and in doing so erased the previous request's (58,831 and
# 59,343, both within 8,192 of 54,604 = step 23's prompt 55,120 - 4 - 512),
# so the nearest checkpoint before the hand-off was 54,604: 7.8k tokens of
# the conversation re-read before the delivered turn (15.9 s the next
# request waited). Nothing the proxy sends can keep one closer: a pre-warm's
# checkpoint at the hand-off's start is another task's too, and is erased
# the same way unless it lies more than 8,192 past the one before it. So the
# re-read is INHERENT to a client that drops the prefill (the hand-off lives
# in one request by design: "Reasoning lives within one request only",
# AGENTS.md) at this server's checkpoint spacing, and the warm -- which does
# it while the harness runs its tool -- is the cheapest correct place for it.
# It is BOUNDED: whatever the generation restored from (`generated_reused`,
# the checkpoint it resumed at, or the whole cached prompt) either survived
# or was erased for lying within CHECKPOINT_MIN_STEP of one that did, so the
# warm reuses at least generated_reused - CHECKPOINT_MIN_STEP - 4. That is
# the expectation now (never below the last user message's floor), so a
# warm that restored less -- a real miss -- is SHORT; the old floor alone
# (7,552 here) would have passed anything. `reread_before_turn` records the
# measured cost: the tokens before the delivered turn that were re-read.
# The two ways to remove it are recorded in docs/SELF-IMPROVEMENT-LOG.md
# (#37): replay the hand-off as that turn's reasoning (an operator decision,
# it reverses the pass-through for our own prefill), or a smaller server
# --checkpoint-min-step (needs the checkpoint size measured). The first was
# taken on 2026-09-27 for all past reasoning (switch restore_reasoning): a
# prefilled turn then renders as the slot holds it and is not warmed (_warm,
# _same_reasoning); all of this applies with the switch off.
CHECKPOINT_MIN_STEP = int(os.environ.get("YAMADORI_CHECKPOINT_MIN_STEP",
                                         "8192"))
ASSISTANT_TURN = "<|im_start|>assistant"


def _count_prefix(model: str, prompt: str, i: int,
                  timeout: float) -> int | None:
    """Tokens in prompt[:i] by the model server's own /tokenize."""
    if i < 0:
        return None
    toks = _upstream_json(f"/upstream/{model}/tokenize",
                          {"content": prompt[:i], "add_special": True},
                          timeout=timeout).get("tokens")
    return len(toks) if isinstance(toks, list) else None


def _prefill_expectation(model: str, prompt: str, rec: dict,
                         timeout: float) -> None:
    """A prefilled turn's warm: expect_reused_at_least (the last user
    message's checkpoint, or the thinning bound below the checkpoint the
    generation resumed at, whichever is later), why, and what it re-read
    before the delivered turn (turn_starts_at, reread_before_turn)."""
    floor = _prefill_floor(model, prompt, timeout)
    gr = rec.get("generated_reused")
    bound = (int(gr) - CHECKPOINT_MIN_STEP - 4) if gr else None
    rec["checkpoint_min_step"] = CHECKPOINT_MIN_STEP
    if bound is not None and (floor is None or bound > floor):
        rec["expect_reused_at_least"] = bound
        rec["expect_why"] = (
            f"the turn opened with a prefill the client drops: the server "
            f"restores its nearest checkpoint before it, at most "
            f"{CHECKPOINT_MIN_STEP} + 4 tokens before the {gr} the "
            f"generation resumed at (checkpoint thinning)")
    else:
        rec["expect_reused_at_least"] = floor
        rec["expect_why"] = (
            "the turn opened with a prefill: the server restores its "
            "checkpoint at the last user message" if floor is not None else
            "the turn opened with a prefill and the tokens before the last "
            "user message could not be counted: not judged")
    at = _count_prefix(model, prompt, prompt.rfind(ASSISTANT_TURN), timeout)
    if at is not None:
        rec["turn_starts_at"] = at
        if rec.get("reused") is not None:
            rec["reread_before_turn"] = max(0, at - int(rec["reused"]))


def _warm_hold(payload: dict) -> "threading.Event | None":
    """Register a pending warm for this conversation BEFORE its calls reach
    the client (the streamed race, #5); None when no warm could follow (not
    a conversation turn, warming off)."""
    sl = payload.get("_slot") or {}
    if not WARM or not sl.get("key") or sl.get("transient"):
        return None
    ev = threading.Event()
    ev.hold_until = time.time() + WARM_HOLD_S
    _WARM_PENDING[sl["key"]] = ev
    return ev


def _warm_release(payload: dict, ev: "threading.Event | None") -> None:
    """A hold no warm took over: dropped, and its waiters let through."""
    if ev is None or getattr(ev, "hold_until", None) is None:
        return
    key = (payload.get("_slot") or {}).get("key")
    if key and _WARM_PENDING.get(key) is ev:
        _WARM_PENDING.pop(key, None)
    ev.set()


def wait_for_warm(key: str | None, timeout: float | None = None
                  ) -> float | None:
    """Block until this conversation's pending warm has finished -- whether
    it is running, scheduled, or only HELD (its turn's calls went out and
    the warm is about to be scheduled). Returns the seconds waited, or None
    when no warm was pending.

    RE-CHECKS (pre-deploy review, 2026-09-24): when the warm waited on ends
    and ANOTHER is now pending for the key (a newer turn scheduled one), it
    waits for that one too, within the same budget. A warm itself never
    outlives WARM_WAIT (_warm_now bounds its own requests by it), so a
    waiter that returns on the budget has not left a warm running. A hold
    never taken over is given up at its WARM_HOLD_S."""
    ev = _WARM_PENDING.get(key) if key else None
    if ev is None:
        return None
    t0 = time.time()
    end = t0 + (WARM_WAIT if timeout is None else timeout)
    while ev is not None:
        hu = getattr(ev, "hold_until", None)
        ev.wait(max(0.0, (end if hu is None else min(end, hu))
                    - time.time()))
        if not ev.is_set():
            hu = getattr(ev, "hold_until", None)
            if hu is not None and time.time() >= hu:
                # Abandoned: nothing will be warmed.
                if _WARM_PENDING.get(key) is ev:
                    _WARM_PENDING.pop(key, None)
                break
            if time.time() >= end:
                break
            continue          # the hold became a warm meanwhile: wait on
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
    if max_mode.blocks(model):
        # MAX MODE: this conversation's model is off the card; a warm would load it (mcp/max_mode.py)
        rec.update(state="skipped", why=max_mode.blocked_reason(model))
        if ev is None:
            ev = _WARM_PENDING.get(key)
        if ev is not None:
            if _WARM_PENDING.get(key) is ev:
                _WARM_PENDING.pop(key, None)
            ev.set()
        return

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
                            **slots.upstream_fields(grant)},
                           timeout=left(600))
        t = r.get("timings") or {}
        token_ledger.record("warm", timings=t)
        rec.update(state="done", reused=t.get("cache_n"),
                   processed=t.get("prompt_n"), slot=grant["slot"])
        # What the slot holds now (the warm popped the old claim): the
        # delivered turn, which the next request extends. Read by slots
        # ADOPTION and COMPACTION AFFINITY (#38).
        slots.remember(grant, slots.fingerprint(dict(fields, messages=msgs)))
        if rec.get("prefilled"):
            # PREFILLED TURNS / A PREFILL MID-CONVERSATION.
            try:
                _prefill_expectation(model, prompt, rec, left(30))
            except Exception as e:                               # noqa: BLE001
                rec["expect_error"] = f"{type(e).__name__}: {e}"[:200]
                rec.setdefault("expect_why", "the turn opened with a prefill "
                               "and its expectation could not be counted: "
                               "not judged")
        exp = rec.get("expect_reused_at_least")
        short = (exp is not None and t.get("cache_n") is not None
                 and int(t["cache_n"]) < int(exp) - WARM_CHECKPOINT_SLACK)
        rec["short"] = bool(short)
        rec["slack"] = WARM_CHECKPOINT_SLACK
        pre = (f"a checkpoint at or after {exp} (the turn opened with a "
               f"prefill the client drops; {rec.get('reread_before_turn')} "
               f"tokens before the turn re-read)") if rec.get("prefilled") \
            else ""
        print(f"  warm: slot {grant['slot']} reused {t.get('cache_n')} "
              f"processed {t.get('prompt_n')}"
              + ((f" -- SHORT: below {pre}" if pre else
                  f" -- SHORT: the slot generated on a {exp}-token prompt "
                  f"moments ago") if short else
                 f" -- from {pre}" if pre else ""), flush=True)
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
    """The blocking path: the one turn implementation, drained.

    Under a cancel token like the streamed path (never cancelled here): job
    threads started under it (_in_thread) share it, which is how a release
    at the end of a second-brain run finds this request (slots.bind_request).
    """
    with cancel.bound(cancel.current() or cancel.Token()):
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


def stream_body(body: dict, public_name: str = "yamadori",
                token: "cancel.Token | None" = None):
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
        and never after the first content delta (CHANNEL ORDER): after it,
        reasoning goes out as an empty delta (a heartbeat, at most one per
        REASONING_BEAT_S).
      - an image yama_generate_image makes goes out as content the moment the
        tool returns, before the model's next generation is requested
        (IMAGES REACH THE CHAT), and a copy the
        model writes of it is removed from the content (_ImageDedup).
      - a stray template marker the model writes in its answer never goes
        out: it is removed, with the text after it while that repeats what
        was already sent; new text after it goes through; only the minimal
        tail that could still be a marker (or a repeat) is held
        (_StrayMarkers, after _ImageDedup; complete() gives the same
        content; mcp/test_stray_markers.py). Recorded in
        x_yamadori.template_markers.stripped.
      - a yama_describe_image result leaves ONE screened reasoning line
        (_describe_line), or a heartbeat once content has started.
      - the breaker lands exactly like `complete()`: tools withdrawn (`_land`),
        on OUR hops only; the client's own prompt is never landed (C1).
      - a `length` finish is finish_reason "length", as `complete()`
        reports it (no notice text is added).
      - a tool call whose IMAGE argument opens as image data (a
        data:image/ URL, base64 image bytes) is never forwarded: its
        upstream connection is closed at the value's first characters, the
        model gets it back as NOT EXECUTED in a hidden hop and writes the
        turn again; the client sees one reasoning line (tool_code IMAGE
        GUARD, mcp/test_image_guard.py).
      - the corpus gets the real hop count; usage is the final generation's
        (U1), sent -- when the client asks with stream_options.include_usage
        -- as its own last chunk with `choices: []`, as the spec says.
      - `x_yamadori` rides on the finish chunk, the same object `complete()`
        puts on its response.
      - NOTHING IS YIELDED UNTIL THE TURN HAS STARTED (E1, 2026-09-25): the
        first byte waits for the turn's first event -- prepare(), the
        context check and the first upstream generation (or deep thinking's
        first heartbeat) behind it -- so server.chat can still answer a
        failure before it with a real HTTP status: this generator RAISES the
        ApiError then. After the first byte a failure is ONE SSE error event
        (`data: {"error": {...}}`) and [DONE]: never assistant content, and
        never a last chunk without finish_reason.
    """
    cid = streaming.new_id()
    model = public_name
    include_usage = bool((body.get("stream_options") or {})
                         .get("include_usage")) if isinstance(
                             body.get("stream_options"), dict) else False
    lead = b""
    # THE REQUEST'S CANCELLATION (mcp/cancel.py, #39). Bound to the thread
    # around every resume of the turn, so each upstream socket the turn opens
    # -- main's generation, and (through the threads they start under it)
    # deep thinking's and the jobs' -- registers with it. The server cancels
    # it the moment the client disconnects (server.py _CancellingStream); a
    # consumer that simply stops iterating cancels it here.
    token = token or cancel.Token()
    gen = _run_turn(body, streamed=True)
    finished = False
    started = False
    try:
        while True:
            try:
                with cancel.bound(token):
                    kind, item = next(gen)
            except StopIteration as stop:
                d = stop.value or {}
                break
            except cancel.Cancelled as e:
                # Its client is gone: nothing is sent, nothing is logged as
                # an answer. The work below it has already stopped.
                finished = True
                print(f"  turn cancelled: {e or token.why}", flush=True)
                return
            except Exception as e:                               # noqa: BLE001
                finished = True
                err = api_errors.of_exception(e)
                print(f"  turn failed ({err.status} {err.code}): "
                      f"{type(e).__name__}: {str(e)[:300]}", flush=True)
                if not started:
                    # Nothing sent: the caller (server.chat) answers it with
                    # its HTTP status and the error object.
                    if err is e:
                        raise
                    raise err from e
                # The stream is committed: OpenAI's mid-stream error form,
                # which its SDKs raise on, then DONE.
                yield err.sse()
                yield streaming.DONE
                return
            if not started:
                started = True
                if lead:
                    yield lead
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
        finished = True
    finally:
        if not finished:
            # Closed before the turn ended (GeneratorExit): stop its work.
            token.cancel("the stream was closed before the turn ended")
            with cancel.bound(token):
                gen.close()
    if not started and lead:
        yield lead
    fin = d["choices"][0].get("finish_reason") or "stop"
    yield streaming.chunk(cid, model, {}, finish=fin,
                          extra={"x_yamadori": d.get("x_yamadori")})
    if include_usage and d.get("usage"):
        # stream_options.include_usage: the spec's own last chunk, usage
        # with an EMPTY choices array (U2). Not sent unasked.
        yield streaming.usage_chunk(cid, model, d["usage"])
    yield streaming.DONE


def log_message(self, *a):                                   # noqa: D102
    pass
