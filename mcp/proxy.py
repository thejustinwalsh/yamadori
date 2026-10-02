#!/usr/bin/env python
"""An OpenAI endpoint that is one standard model.

THE POINT

A client adds one OpenAI-compatible model and gets the whole stack. It declares
no tools, configures no MCP server, and never learns any of this exists.

    client --/v1/chat/completions--> [proxy] --> llama-swap --> model (main)

ONE MODEL, ONE CACHE (operator, 2026-09-24; docs/SELF-IMPROVEMENT-PLAN.md
Phase 0.5, AGENTS.md "One model, one cache"). Main -- the model the client
talks to -- gets the client's tools untouched, plus only the server tools of
ours that are offered (images, vision, the MCP host's package lookups, the
craft tool where the skills offer it), each run as a hidden hop.

The second brain (deep thinking, fan-out, the fix-up repair, their triggers
and fold-back phrases) and the library-definitions injection were REMOVED
2026-09-29 (docs/REMOVED.md; the way back is commit e360d37).

WHAT IT ADDS, AND WHY THE CACHE NEVER NOTICES

  per-turn        skills, the concept seed (a conversation's first user turn
                  only), the work log after a compaction -- on the user turn
                  they served, decided once
  reasoning       the model's own, restored on every assistant turn
  hidden hops     our tools' calls and results, which the client never sees

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
import repeats  # noqa: E402
import discover  # noqa: E402
import nebari  # noqa: E402
import tiers  # noqa: E402
import dashboard  # noqa: E402
import catalog  # noqa: E402
import selection  # noqa: E402
import images  # noqa: E402
import vision  # noqa: E402
import gpu_room  # noqa: E402
import route as router  # noqa: E402
import tool_code  # noqa: E402
import slots  # noqa: E402
import recent_turns  # noqa: E402
import token_ledger  # noqa: E402
import stage_timing  # noqa: E402
import compaction  # noqa: E402
import power  # noqa: E402
import skill_prompts  # noqa: E402
import cancel  # noqa: E402
import max_mode  # noqa: E402
import tier_models  # noqa: E402
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
# Seconds between empty deltas while a streamed turn waits on work of its own
# (a tool of ours running in a thread, _in_thread; #39: Hermes kills a stream
# that sends no chunk for its stale timeout). The variable keeps the name it
# had when fan-out was what the stream waited on.
HEARTBEAT = float(os.environ.get("YAMADORI_FANOUT_HEARTBEAT", "5"))

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

# THE STATIC ADDENDUM -- a fixed table at the end of the client's system
# text describing the second model's fold-backs -- and its server-tools rows
# (ADDENDUM, ADDENDUM_THINK_ROW, ADDENDUM_TOOLS) were REMOVED 2026-09-29 with
# the second brain (docs/REMOVED.md).


def add_system_tail(messages: list[dict], text: str) -> list[dict]:
    """`text` at the end of the client's system message (or a system
    message of its own when there is none): the MCP host's line, and the
    craft index after it (skill_select PROGRESSIVE DISCLOSURE).

    At the end of the system text, never at the end of the conversation:
    this model's chat template raises "System message must be at the
    beginning" outright, so a trailing system message is a hard 500."""
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
    client utility call to one no conversation holds. A payload with
    `_role` "helper" pins to slots.HELPER (the child slot). The choice
    rides upstream as `id_slot`; a payload with no `_slot` is left to the
    server, as before.

    CACHE. The response carries `_cache` -- prompt tokens, how many came from
    the slot's cache and how many were processed now (llama-server `timings`)
    -- and the record is appended to `payload["_cache_log"]` when the caller
    keeps one, which x_yamadori.cache and the per-request log line read.
    """
    want = payload.get("_slot")
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
    if finish is None:
        # THE STREAM ENDED WITH NO FINISH CHUNK: the model server died behind llama-swap, which answers 200 and
        # closes the body (2026-10-01: flash-next faulted on its first request and the client got an EMPTY answer,
        # finish "stop", which a harness reads as the turn ending; llama-swap logged "recovered from upstream
        # disconnection during streaming"). A generation that ended always carries a finish_reason; one that did
        # not end is a failure, never an answer.
        took = time.time() - t0
        if not (content or reasoning or calls):
            if retries > 0:
                print(f"  upstream ended the stream with nothing and no finish after {took:.1f}s (the model "
                      f"server exited?); retrying once", flush=True)
                yield from _post_events_raw(
                    path, dict(payload, _image_guard=guard) if guard
                    else payload, timeout, retries - 1)
                return
            raise streaming.UpstreamError(
                "the model server ended the stream without generating anything (it exited or was restarted); "
                "retry the request", status=503)
        print(f"  upstream ended the stream after {took:.1f}s with {len(''.join(content))} chars in hand and no "
              f"finish: reported as a dropped connection", flush=True)
        out = assemble("incomplete",
                       f"[the connection to the model dropped after "
                       f"{took:.0f}s; what follows is the part that "
                       f"arrived]\n\n")
        out["_transport"]["dropped_after"] = round(took)
        yield "done", out
        return
    yield "done", assemble(finish)


# OUR TOOLS ON MAIN (operator, 2026-09-24; AGENTS.md "The surface").
#
# Main -- the model the client talks to -- gets the client's tools untouched,
# plus only the server tools of ours that are offered: yama_generate_image
# and yama_describe_image, the MCP host's package lookups (mcp/mcp_host.py)
# and yama_recall_craft where the skills' craft offer says so. The proxy runs
# each as a hidden hop (run_our_tool); the client never sees it.
#
# REMOVED 2026-09-29 with the second brain (docs/REMOVED.md): the second
# brain's own tools (the code-intelligence tools, the work-log tools and the
# research tools find_in_knowledge_base / read_web_page / search_web),
# yama_think_deeply, yama_plan and the `delegate_investigation` benchmark arm
# (YAMADORI_DELEGATE_TOOL). The code-intelligence tools stay on the MCP
# server (:1235, code_search.TOOLS).


def image_tools() -> list[dict]:
    """`yama_generate_image` and `yama_describe_image`, only where an image
    server is configured: main's copy of yama_generate_image
    (images.MAIN_TOOL), whose description says the proxy shows the picture
    itself (IMAGES REACH THE CHAT, _run_turn).

    Gated on YAMADORI_IMAGEGEN_URL so no request pays prompt tokens for a tool
    that cannot run. Read per request, so the gate follows the environment.
    `yama_describe_image` (mcp/vision.py) goes wherever `yama_generate_image`
    goes, so the model can look at what it drew; YAMADORI_VISION=0 withholds
    it. A request carrying an attached image gets it even without an image
    server (prepare, `vision_tools`).
    """
    if not images.configured():
        return []
    return ([images.MAIN_TOOL]
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
# with the craft offer (skill_select's state): the tool list never changes
# mid-conversation.
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
)
# Decided NOT to overlap (kept on main):
#   yama_describe_image vs Codex view_image -- view_image ATTACHES a local image
#     to the conversation; this text-only model sees it only through
#     yama_describe_image (proxy.prepare's placeholder names its id).
#   yama_recall_craft vs Hermes skills_list / skill_view / skill_manage and
#     OpenCode `skill` -- those read the HARNESS's skills; ours are this
#     service's craft, named apart so neither is mistaken for the other.


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
               images_on: bool = True, craft: bool = False,
               withheld: list | None = None,
               keep_withheld: list | None = None,
               mcp: list | None = None) -> tuple[list, set]:
    """(the tool list main is sent, the names of ours in it).

    The client's own list first and untouched, so its rendering -- the
    template prints tools first in the system block -- never depends on what
    we add. Ours after it, WITHHELD on a conflict with the client's
    (tool_conflicts: the same normalised name, or a declared overlap); the
    client's version is the one with side effects the client can handle.
    `craft`: yama_recall_craft (skill_select.READ_TOOL), where the
    conversation's craft offer says so. `withheld` collects what was
    withheld; `keep_withheld` names tools of ours withheld earlier in the
    conversation, kept out. `mcp`: the MCP-backed tools the conversation was
    offered (mcp_host; the offer is decided on its first request and kept by
    name, _mcp_offer)."""
    import skill_select
    client_tools = list(client_tools or [])
    taken = {t.get("function", {}).get("name") for t in client_tools
             if isinstance(t, dict)}
    cand = ((image_tools() if images_on else []) + vision_tools(att)
            + ([skill_select.READ_TOOL] if craft else [])
            + list(mcp or []))
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


# Every name this proxy executes itself: the image tools, the craft tool and
# the MCP-backed tools (mcp_host; every name a configured server backs, read
# at import).
OUR_NAMES = ({images.TOOL_NAME, vision.TOOL_NAME,
              skill_prompts.CRAFT_TOOL_NAME}
             | mcp_host.mcp_config.all_tool_names())

# THE `yama_*` RENAME (operator, 2026-09-27): every tool the proxy adds to
# main is named `yama_*`, a name no harness offers. The names before it, as
# stored ledger rows (hidden hops), the craft state's withheld list and old
# records still carry them: each is read as its new name (the hops replay
# byte for byte as stored; a call the model makes by an old name -- copied
# from a replayed hop -- runs as the new tool).
LEGACY_TOOL_NAMES = {
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


def tool_gate(messages: list[dict], root: str | None,
              state: dict | None = None) -> dict:
    """Can any library source this server holds bear on this request?

    The route's gate (mcp/route.py: a library_question needs one), since the
    tools it once admitted left main (2026-09-24) and the injections that
    read it were removed (2026-09-29). NOT a prediction about whether reading
    would help -- a FACT about whether any index this caller can reach could
    answer. The rule, the measurement and
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


def run_our_tool(name: str, args: dict, state: dict | None = None) -> str:
    """Run one server tool of ours that main called (a hidden hop). Every
    A4000 decision a tool causes (a draw, a look: mcp/gpu_room.py) lands in
    this request's x_yamadori.gpu_room -- set here, in the thread the tool
    runs in."""
    with gpu_room.recording((state or {}).get("_gpu_room")):
        return _run_our_tool(name, args, state)


def _run_our_tool(name: str, args: dict, state: dict | None = None) -> str:
    # An old name of ours runs as its `yama_*` tool (LEGACY_TOOL_NAMES).
    name = canonical_tool_name(name)
    st = state if state is not None else {}

    if name == images.TOOL_NAME:
        # The proxy runs it, on CUDA1, in its own lane (admission.image_lane).
        # The result carries a signed /media URL built on the address the
        # client reaches us on, and each call is recorded for x_yamadori.
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
        return vision.run_tool(args, st.setdefault("_attached", vision.empty()),
                               st.setdefault("_vision", []))

    if mcp_host.is_mcp_tool(name):
        # An MCP-backed tool (mcp/mcp_host.py): the proxy is the MCP client
        # of the server behind it (PackageLens, in its gated container). The
        # result is fetched content, rendered, screened and framed as data
        # there; each call is recorded for x_yamadori.mcp.
        return mcp_host.run_tool(name, args, st.setdefault("_mcp_calls", []))

    if name == skill_prompts.CRAFT_TOOL_NAME:
        # yama_recall_craft (skill_select PROGRESSIVE DISCLOSURE): one craft in
        # full, by name or topic, as a hidden hop the ledger replays. What
        # it returned counts as GIVEN for the per-turn engine (a later need
        # gets a recall line, not the body again).
        import skill_select
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

    # Not a tool this server runs (the main loop runs only the names of ours
    # offered on the request; this is a caller that asked by name).
    return cs.error_result(
        name, "NOT_A_SERVER_TOOL",
        f"{name} is not a tool this server runs. Nothing was run.",
        retryable=False,
        remedies=[{"fixable_by": "agent",
                   "action": "call one of the tools offered in this request",
                   "effect": "the call runs"}])


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
    holds. Nothing in a request may select a directory on this disk.
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
# conversation decided its server-tool offer afresh and the tool list itself
# changed; the triggers of the time would re-fire on every such request).
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
#   user turn        "inject"     skills, the concept seed (the
#                                 conversation's first user turn), the work
#                                 log after a compaction: decided ONCE, on
#                                 the request whose last message is that
#                                 user turn, and replayed after its content
#   tool result      "inject"     skills on an agent step: the same, on the
#                                 tool result a request ended on (and, byte
#                                 for byte, what an older one carried:
#                                 LIBRARY USE #19, the situations)
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
#   the conversation "seed"       its concept seed, by lineage
#                                 (x_yamadori.session.seed; SEED_KEY)
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


def _draw_seed(prompt: str | None) -> dict | None:
    """One fresh seed. Split out so a test can pin it."""
    import concept_seed
    return (concept_seed.seed_for(prompt, 1) or [None])[0]


# Names this proxy put on main (or the second brain's upstream lists) before
# they were removed: the code-intelligence and work-log tools, check_code and
# bind_project_context (2026-09-24), yama_think_deeply, yama_plan, the
# delegate arm and the research tools (2026-09-29). The corpus recorded the
# UPSTREAM tool list, ours included, so a replay of it (mcp/test_utility.py)
# must not count them as the client's.
_LEGACY_NAMES = ({t["name"] for t in cs.ALL_TOOLS}
                 | {"bind_project_context", "check_code",
                    "delegate_investigation", "think_deeply",
                    "yama_think_deeply", "yama_plan", "find_skills",
                    "find_in_knowledge_base", "read_web_page", "search_web"}
                 # And the names ours had before the `yama_*` rename
                 # (2026-09-27): the corpus recorded them in upstream tool
                 # lists.
                 | set(LEGACY_TOOL_NAMES))


def client_tool_names(body: dict) -> list[str]:
    """The CLIENT's own tools, never ours -- even when a client re-sends ours
    by name."""
    return [n for n in (t.get("function", {}).get("name")
                        for t in (body.get("tools") or [])
                        if isinstance(t, dict))
            if n and n not in OUR_NAMES and n not in _LEGACY_NAMES]


# X-Yamadori-Features flags that, forced ON, mean a benchmark asked for an
# augmentation -- and then a utility-shaped request still gets it.
_AUGMENTATIONS = ("skills",)


def utility_of(body: dict, messages: list[dict] | None = None) -> dict:
    """Is this request a client's own side call (selection.utility_call)?

    {utility, because, signals}. The header decides when it says so:
    X-Yamadori-Features {"utility": true|false} forces it, and a header that
    forces an augmentation ON overrides the rule, because a benchmark that
    forced it meant it."""
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
    return {"skills": False, "utility": True,
            "because": {"utility": util["because"], "skills": why},
            "signals": {"utility": util.get("signals"),
                        "tier_requested": requested_tier}}


def prepare(body: dict) -> dict:
    """Resolve the ledger, tools and per-turn injection without calling the
    model.

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
    # conversation on earlier requests -- reasoning, injections, hidden
    # hops -- put back, so this request's rendering EXTENDS what the pinned
    # slot holds. A utility call is not a conversation and gets the client's
    # messages as sent.
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

    # THE ROUTE'S GATE: can any library source this server holds bear on
    # this request? A fact, read from the package store
    # (domains.tool_admission); mcp/route.py reads it (a library_question
    # needs one). Logged one line per request.
    gate = None if utility else tool_gate(messages, root, state)
    if gate and gate["offer"]:
        _remember_offered(state)
    # MAIN'S TOOLS: the client's, untouched, plus our server tools where
    # offered (images and vision on every tier, operator 2026-09-23; the
    # MCP tools and the craft tool below, decided on the conversation's
    # first request). A utility call gets the client's list and nothing of
    # ours.
    continuing = any(isinstance(m, dict) and m.get("role") == "assistant"
                     for m in raw)
    tools, ours = main_tools(body.get("tools"), None if utility else att,
                             images_on=tiers.images_offered(tier)
                             and not utility)

    # SELECTION: whether skills may run for this request (the tier allows,
    # a header forces; mcp/selection.py). The CLIENT's own tools (never
    # ours, even when a client re-sends ours by name) ride along for the
    # skills' tool gates.
    client_tools = client_tool_names(body)
    # THE ROUTE (mcp/route.py): one class per request, decided here, once.
    # Skills and the agent step's thinking cap read it.
    try:
        route_dbs = selection.symbol_dbs(repos.db_path(root) if root else None)
    except Exception:                                            # noqa: BLE001
        route_dbs = {}
    route = router.classify(messages, client_tools=client_tools, util=util,
                            gate=gate, dbs=route_dbs)
    print("  " + router.log_line(route), flush=True)
    # A compaction that resends the conversation (compaction.in_place): part
    # of that conversation -- its session, tools and slot -- but nothing is
    # added to it. _serve_compaction shapes the rest.
    inplace = not utility and compaction.in_place(messages)
    if utility:
        sel = _utility_selection(util, requested_tier)
    elif inplace:
        why = ("an in-place compaction: served on the conversation's own "
               "prompt and slot, nothing added (mcp/compaction.py)")
        sel = {"skills": False, "utility": False, "compaction": True,
               "because": {"utility": util.get("because"), "skills": why},
               "signals": {"tier_requested": requested_tier}}
    else:
        sel = selection.select(messages, tier, client_tools=client_tools,
                               route=route)
        sel["utility"] = False
        sel.setdefault("because", {})["utility"] = util.get("because")

    augmented = messages
    # THE CRAFT OFFER (skill_select PROGRESSIVE DISCLOSURE; operator,
    # 2026-09-27): on the conversation's first request, an index of the
    # craft relevant to it at the end of the system text and
    # yama_recall_craft on main; decided ONCE and kept with the tools
    # withheld for a conflict with the client's (tool_conflicts), so the
    # system block and the tool list never change mid-conversation. A kept
    # offer applies to an in-place compaction too (its system block must
    # match the conversation's).
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
                body.get("tools"), att,
                images_on=tiers.images_offered(tier),
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
    # Template markers scrubbed from OUR injected text (skills, the work
    # log) when it is decided -- a replay sends what was recorded, byte for
    # byte (pre-deploy review, 2026-09-24).
    scrub_note: dict = {}
    # The last user turn the USER wrote: a harness's synthetic tool-media
    # turn carries a tool's image and is looked past (image_input), as
    # route and selection.question_of do.
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
    # THE CONCEPT SEED (operator, 2026-09-29): drawn once, on a
    # conversation's FIRST user turn -- the first request, no answer in it,
    # not a compaction's continuation -- where the tier's `seed` flag
    # allows it, and carried in that turn's injection, so the ledger replays
    # it byte for byte on every later request (and a compaction served on
    # the ledger's rendering summarises it with the rest).
    seed_turn = (not utility and not inplace and not continuing
                 and not continued and bool(tier.get("seed")))
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
        seed_meta: dict | None = None
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
                                    "raw": raw}
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
            if seed_turn:
                if "seed" in keep:
                    texts["seed"] = keep["seed"]
                    seed_meta = retry_meta.get("seed")
                elif conversation_seed(account, lineage) is None:
                    # Once per conversation: one that already drew its seed
                    # (a continuation that opens with no answer in it, under
                    # the same lineage) draws none.
                    texts["seed"], seed_meta = _seed_text(
                        message_text.text_of(raw[li], ""))
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
        if seed_meta and "seed" in order:
            meta["seed"] = seed_meta
        if unavailable:
            # The parts that did NOT fail are kept verbatim for the retry.
            meta.update(retry=unavailable, texts=clean)
        # What stands is what is sent: never over a final non-empty decision
        # a duplicate request recorded meanwhile (nebari.ledger_decide).
        text, meta = nebari.ledger_decide(account, lineage, keys[li],
                                          "inject", text, meta)
        if meta.get("seed") and lineage:
            # The conversation's seed, by its lineage (it outlives the user
            # turn that carries it: x_yamadori.session.seed on every
            # request, a compaction included).
            nebari.ledger_put(account, lineage, SEED_KEY + lineage, "seed",
                              json.dumps(meta["seed"]))
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
    # SKILLS ON AN AGENT STEP (skill_select PER-TURN INJECTION, operator
    # 2026-09-27): a request that ENDS ON A TOOL RESULT -- the one message
    # the slot does not hold yet -- gets what the step's newest evidence
    # brings, appended LAST to the tool result (the end of what the model
    # reads next), decided once under the result's key and replayed from the
    # ledger (ledger_restore). Tool results that carry an older injection
    # (LIBRARY USE, #19, removed 2026-09-29; the situations, removed
    # 2026-09-27) replay it byte for byte, so their slots' prefixes hold.
    last_raw = raw[-1] if raw and isinstance(raw[-1], dict) else {}
    ends_on_tool = (not utility and not inplace
                    and last_raw.get("role") == "tool"
                    and message_text.has_text(last_raw)
                    and augmented and isinstance(augmented[-1], dict)
                    and augmented[-1].get("role") == "tool")
    step_skills_on = (bool(sel.get("skills")) and bool(lineage)
                      and not utility and not inplace)
    if ends_on_tool and step_skills_on:
        got = nebari.ledger_get(account, keys[-1], "inject")
        # A retryable decision on this tool result is decided again (see
        # RETRYABLE DECISIONS above); ledger_restore did not put it back.
        if got is None or (retry_key is not None and retry_key == keys[-1]):
            parts: list[str] = []
            meta: dict = {}
            sk_text, skills_rec = _skills_step(
                raw, sel, route, client_tools, account, lineage, keys[-1])
            sk_text, n_scrub = scrub_markers(sk_text)
            _note_scrub(scrub_note, "injection", n_scrub)
            if sk_text:
                sk_text = "\n" + sk_text
                parts.append("skills")
                meta["skills"] = _skills_meta(skills_rec)
            meta["parts"] = parts
            text, meta = nebari.ledger_decide(account, lineage, keys[-1],
                                              "inject", sk_text, meta)
            if text:
                augmented = list(augmented)
                augmented[-1] = message_text.append_text(augmented[-1], text)
        else:
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
    # (docs/CONSTANTS-AUDIT.md: chosen from one Octopus run). With the
    # tier -> model table on (mcp/tier_models.py) the MODEL's token profile
    # supplies the cap for the route the request names (`turn`); the cap
    # applied is what tiers.apply kept (`_step_cap`).
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
                      nudge=nudge,
                      turn=("agent_step" if step_rec is not None
                            else (None if utility else "user_turn")))
    if step_rec is not None:
        step_rec["thinking_cap"] = out.get("_step_cap")
    out.update(scrub_note)
    out["messages"] = augmented
    out["tools"] = tools
    # Our tools on main: the only calls the main loop runs itself.
    out["_ours"] = sorted(ours)
    # The attachment register (JSON-safe; the turn moves it into the session
    # state before anything deep-copies the payload).
    out["_attached"] = att
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
    out["_route"] = route
    out["model"] = internal            # what llama-swap actually routes on
    if body.get("_upstream_model") and internal == max_mode.MAIN:
        # MAX MODE (mcp/max_mode.py): the main model server._serve_turn chose for this request
        out["model"] = body["_upstream_model"]
    out["_tier"] = tier
    out["_public_model"] = requested
    out["_utility"] = util
    out["_tier_requested"] = requested_tier
    # The ledger's scope and this request's key (the chain hash of its last
    # message), and what was restored. `seed`: the conversation's concept
    # seed (x_yamadori.session.seed).
    out["_ledger"] = {"account": account, "session": lineage,
                      "turn_key": keys[-1] if keys else None,
                      "restored": restored, "inject": inject_rec,
                      "salt": salt, "conversation": ses,
                      "seed": (None if utility
                               else conversation_seed(account, lineage)),
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
    # x_yamadori.progress (tiers.BEHAVIOURS): the agent step's thinking cap
    # and nudge, and every switch and its source.
    out["_progress"] = {"step": step_rec,
                        "switches": tiers.behaviours(tier)}
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
    if isinstance(tier.get("primary_hold_s"), int) \
            and "primary_hold_s" in (tier.get("overridden") or []):
        # The live suite's ONE CONVERSATION hold override: a TEST account
        # only, and only against that account's own owner (slots._hold_for).
        test = corpus.account_traffic(account) == "test"
        if test:
            out["_slot"]["hold_s"] = tier["primary_hold_s"]
        out["_slot"]["hold_override"] = {"hold_s": tier["primary_hold_s"],
                                         "honoured": test,
                                         "refused": None if test else "not a test account"}
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


# THE CONCEPT SEED ON THE FIRST USER TURN (operator, 2026-09-29). It rode in
# every second-brain job's user message (fan-out's candidates, deep
# thinking, the fix-up) until those jobs were removed; now it is drawn ONCE
# per conversation (concept_seed.seed_for, away from the user's words) and
# appended to the conversation's first user turn with the rest of that
# turn's injection (prepare). The ledger records the injection, so every
# later request replays it byte for byte; the seed itself is recorded in the
# injection's meta and by the conversation's lineage (SEED_KEY), for
# x_yamadori.session.seed. None -- and no line -- when the embedding matrix
# is not extracted: a missing seed costs nothing.
SEED_KEY = "seed:"


def _seed_text(prompt: str | None) -> tuple[str, dict | None]:
    """(the line to append to the first user turn, the seed's record)."""
    import concept_seed
    seed = _draw_seed(prompt)
    if not seed or not seed.get("word"):
        return "", None
    return concept_seed.user_turn_line(seed["word"]), concept_seed.summary(seed)


def conversation_seed(account: str, lineage: str) -> dict | None:
    """The conversation's concept seed ({word, token_id, u32}), or None."""
    if not lineage:
        return None
    raw = nebari.ledger_get(account, SEED_KEY + lineage, "seed")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


# The per-turn injection's parts, in the order they are appended.
INJECT_PARTS = ("skills", "seed", "work_log")


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
    """Where the client's last user turn sits in `messages` (a system
    message of ours may have been added in front, and restored image hops
    add messages before assistant turns)."""
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


def _skills_step(raw: list, sel: dict, route: dict, client_tools,
                 account: str, lineage: str, key: str | None
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


# LIBRARY DEFINITIONS (a library question's held-symbol definitions) and
# LIBRARY USE (#19: the definitions of the names a conversation imports from
# a held package, appended to the message a request ended on) were REMOVED
# 2026-09-29 (docs/REMOVED.md): the MCP host's package lookups
# (yama_find_package, yama_read_package_readme, ...) answer on the model's
# call instead. A message that carries one replays it from the ledger.


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
    # With the tier -> model table on (mcp/tier_models.py), the compaction
    # samples with its MODEL's profile, like every other request of that
    # model (tiers.apply); off, the vendor's values.
    prof = None
    if tier_models.profiles_on():
        m = out.get("model")
        p = tier_models.profile(max_mode.current(
            m if max_mode.is_main(m) else None))
        prof = (p.get("sampling_thinking" if thinks
                      else "sampling_instruct") or {}).get("value")
    sampling, out["_sampling"] = tiers.enforce_sampling(body, thinks, prof)
    out.update(sampling)
    # tool_choice none: the model cannot call a tool, and the render is the
    # same (llama-server gives the template the tools whatever tool_choice
    # says; mcp/compaction.py cites the lines). No tools, no field.
    if out.get("tools"):
        out["tool_choice"] = "none"
    else:
        out.pop("tool_choice", None)
    out["_fixed_budget"] = True           # tiers.rebudget leaves it alone
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

    So: withdraw the tools so the request is unambiguous, and say what is
    wanted now. Reaching here is a
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


# A DIRECTIVE PREFILL (operator, 2026-09-27; KEPT 2026-09-29 as a skill
# delivery channel, docs/REMOVED.md): a line in the model's own voice as the
# LAST LINE OF MAIN'S REASONING with the think block LEFT OPEN -- a prefill
# whose reasoning_content is the line and whose content is empty -- so the
# model keeps thinking toward the action and then acts. A payload carrying
# `_prefill` opens hop 0 with it (_run_turn); the reasoning is the slot's own
# text, which the ledger restores (restore_reasoning), so no warm is needed.
# The triggers that fired it (the server-tool triggers, the verify
# directive, continue_stated_step) were REMOVED 2026-09-29; the skills
# renderer measures it against injected text per model. That llama-server
# leaves the block open for a reasoning-only prefill is INFERRED from STEP
# 0's /apply-template probe (docs/SELF-IMPROVEMENT-PLAN.md), not measured:
# confirm it on the live /apply-template before a result that uses it
# counts. A line ends on a LETTER, never a space (the prefill rule).


def directive_prefill(line: str) -> dict:
    """A directive as the opening of main's REASONING, the think block left
    open: no content, so the model thinks on from the line, then acts."""
    return {"role": "assistant", "content": "", "reasoning_content": line}


def _drain(gen):
    """Run a generator to the end; its return value."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


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


def _x_yamadori(payload: dict, *, hops: int) -> dict:
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
    lg = payload.get("_ledger") or {}
    return {
        "tier": tier.get("name"),
        "effort_sent": payload.get("reasoning_effort"),
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
        "hops": hops,
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
        # loop (images, vision, the MCP lookups, the craft tool): name,
        # empty, error, chars.
        "tools": list(payload.get("_tool_calls") or []),
        # The tool-turn cap (tiers.tool_turn_limit): limit, tool turns
        # executed, and whether the loop landed because of it.
        "tool_turns": dict(payload.get("_turn_cap") or _turn_cap(payload)),
        # The client's own prompt against its window (check_client_prompt,
        # C1): limit, generation floor, the high estimate, and the model
        # server's own count when it was near enough the edge to be made.
        "context": payload.get("_context"),
        # The route (mcp/route.py): {class, because, signals}, decided once
        # in prepare. What skills and the agent step's thinking cap read.
        "route": payload.get("_route"),
        # THE IMAGE GUARD (tool_code, #46): None when no call was stopped;
        # else each stopped call -- tool, argument name, kind of image data
        # (a data:image/ URL, base64 PNG data, ...), the argument characters
        # and deltas (~tokens) read and the seconds before the stop, the
        # hop, `withheld` on a last hop -- how many times the model was asked
        # again, and whether the turn landed. Never the argument's text.
        "image_guard": payload.get("_image_guard_rec"),
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
        # carried, line, seed}. source: prompt_cache_key | header | tool_call_id
        # (our id, read back from a tool-call id) | summary_line (from a
        # compaction summary we wrote) | minted (a new conversation) |
        # compaction_map (a flattened compaction mapped to a conversation).
        # `carried`: how many of this turn's client tool calls went out with
        # an id carrying ours (_carry_session). `line`: this answer is a
        # compaction summary that opens with the session line. `seed`: the
        # conversation's concept seed {word, token_id, u32}, drawn on its
        # first user turn (proxy.conversation_seed), or None. `id` is ours
        # in full, a client's own as a hash prefix. Never the key.
        "session": (dict(lg["conversation"],
                         carried=int(payload.get("_session_carried") or 0),
                         line=bool(payload.get("_session_lead")),
                         seed=lg.get("seed"))
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
        # THE SWITCHES (tiers.BEHAVIOURS; docs/research/OVERTHINKING.md):
        # `step` (the agent step's thinking cap and why, which nudge) and
        # `switches` (each switch, on/off and its source: header / env /
        # tier / default).
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
        # / default) and each slot it emptied -- the transient one after a
        # side call or an as-sent compaction -- or kept, and why:
        # {slot, why, released, cells_before, ms, method, skipped?}.
        "slots": {"release": dict(payload.get("_slot_release") or {}),
                  "released": list(payload.get("_slots_released") or []),
                  "routed": (payload.get("_slot") or {}).get("routed"),
                  "switch": (payload.get("_slot") or {}).get("switch"),
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
# it (streamed=False: nothing is presented); stream_body() turns each
# event into an SSE chunk. Two copies of this loop had drifted three times
# (tool lists, deep thinking's tools, fan-out on one path only).
#
# WHAT MAIN'S CONTEXT HOLDS: the client's messages, the ledger's additions,
# main's own generations (our tools' hidden hops included) -- and nothing
# else: a turn in main's context the client never received would keep the
# next request from extending what the slot held. Where the turn the client
# stores differs from what the slot generated, the slot is WARMED with the
# delivered turn (STEP 0 (b): the next request processed only its 23-token
# tail).

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
#                   a line it prefills as reasoning, a seed word, the text it
#                   injects (skills, the work log) -- is
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
    `model` when the upstream generations (main's hops) carried at least
    as many,
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


def _image_stop_record(stop: dict, hop: int) -> dict:
    """One call the image guard stopped, for x_yamadori.image_guard: names
    and numbers, never the argument's text."""
    return {"tool": stop.get("tool"), "argument": stop.get("argument"),
            "kind": stop.get("kind"), "chars_seen": stop.get("chars_seen"),
            "deltas": stop.get("deltas"), "seconds": stop.get("seconds"),
            "hop": hop}


def _run_turn(body: dict, streamed: bool):
    # x_yamadori.timing (mcp/stage_timing.py): this request's per-stage wall clock, from here to the record
    stage_timing.begin()
    # MAX MODE (mcp/max_mode.py): every call this request makes -- the decider, summaries, warms -- goes to the
    # model server._serve_turn chose; a request for another model waits here (never cancelling it) until the other
    # model's work in flight has ended, and its first upstream call swaps the card.
    if body.get("_upstream_model"):
        max_mode.set_current(body["_upstream_model"])
        waited = max_mode.wait_ready(body["_upstream_model"])
        body = dict(body, _capacity=dict(body.get("_capacity") or {}, **waited))
        if body["_upstream_model"] != max_mode.MAIN:
            # the decider's label priors (and its label check) are per model: read on this model once it serves
            import decide_turn
            decide_turn.prime_for(body["_upstream_model"])
    # ONE SYSTEM MESSAGE, `developer` included (system_roles.one_system):
    # before anything reads the messages, so every reader sees what renders.
    messages, roles_rec = system_roles.one_system(body.get("messages") or [])
    if messages is not body.get("messages"):
        body = dict(body, messages=messages, _roles=roles_rec)
    root, trusted, how = resolve_repo(messages, body.get("_client_ip", ""))
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
    # (mcp/gpu_room.py) -- prepare's own embeddings (skills, E1), then every
    # tool run_our_tool executes. Fresh per request.
    room_log: list = []
    # FAIL FAST (pre-deploy review, 2026-09-24; revised after the live gate
    # the same night): prepare's own embeddings (skills, E1) wait at most
    # gpu_room.FAIL_FAST_WAIT_S (2 s) for the A4000's room lock, which an
    # image draw holds for its whole run (up to ROOM_WAIT_S, 300 s). A loaded
    # search model takes its lease; one that is not loaded is LOADED when the
    # room comes free within the wait (after a restart nothing else loads
    # it), and skipped otherwise -- recorded (x_yamadori.skills /
    # x_yamadori.gpu_room) and decided again on the next request (a
    # RETRYABLE injection, prepare).
    with gpu_room.recording(room_log), gpu_room.fail_fast(
            "a chat turn's own embedding (skills, E1) waits for the A4000 "
            "only briefly"):
        payload = prepare(body)
    state["_gpu_room"] = room_log
    payload["_gpu_room"] = room_log
    # ONE CONVERSATION (mcp/slots.py, layout v3): a second conversation is refused before any generation
    routed = slots.check_owner(payload.get("_slot"), model=payload.get("model"))
    if routed:
        max_mode.set_current(routed["model"])
        payload = tiers.rebudget(dict(payload, model=routed["model"]))
    # Jobs this request runs in threads (_in_thread) are bound to its cancel
    # token: a slot release at the end of one finds this request's switch
    # and record through it.
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
    # THE CLIENT'S OWN PROMPT MUST FIT (C1): measured here, before any
    # generation, and refused with context_length_exceeded (an ApiError: a
    # real HTTP 400 on both paths, since nothing has been sent yet) -- never
    # landed.
    payload["_context"] = check_client_prompt(payload)
    # THE PRE-FLIGHT IS OVER: every refusal that must be a real HTTP status (validation, the one-conversation
    # rule, the window check) has had its chance. From here the stream may send heartbeats (_TurnPump: THE WAIT IS
    # HEARD) -- not before: a prompt past the window took 9.6 s to count (deploy check 2026-10-02) and a heartbeat
    # at 5 s turned its 400 context_length_exceeded, which harnesses compact on, into a mid-stream error.
    _tok = cancel.current()
    if _tok is not None:
        _tok.preflight_done = True
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
    # yama_describe_image reads this request's attached images from the
    # session state, not the payload.
    state["_attached"] = payload.pop("_attached", None) or vision.empty()
    payload["_attachments"] = vision.summary(state["_attached"])
    payload["_vision"] = state.setdefault("_vision", [])
    # The MCP-backed tools' calls in THIS request (x_yamadori.mcp.calls).
    state["_mcp_calls"] = []
    payload["_mcp_calls"] = state["_mcp_calls"]
    payload["_tool_calls"] = []
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
    out = _Out(streamed, body.get("_shown_prefix") or "",
               lead=payload.get("_session_lead") or "")

    # THE MAIN LOOP. It runs more than once only for OUR tools on main --
    # yama_generate_image / yama_describe_image, the MCP host's lookups,
    # yama_recall_craft. It ends when the model stops calling them, at
    # context_full, or at the tool-turn cap (tiers.tool_turn_limit), which
    # LAND: tools withdrawn, the answer asked for. ONE upstream generation
    # per answer: every hop is streamed, and the hop that makes no call of
    # ours IS the answer.
    convo = payload["messages"]
    # The turn's own tools, before a landing withdraws them: what the next
    # request renders with, and so what the compaction store keeps.
    tools0 = list(payload.get("tools") or [])
    # A DIRECTIVE PREFILL (directive_prefill) opens hop 0 when the request
    # carries one (`_prefill`).
    prefill = payload.pop("_prefill", None)
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
    # THE IMAGE GUARD (tool_code, #46): calls stopped because an image
    # argument opened as image data, each handed back as a hidden hop; past
    # IMAGE_REGENERATIONS the turn lands. x_yamadori.image_guard.
    img_rec: dict | None = None
    img_land = False
    hop = -1
    while True:
        hop += 1
        payload = tiers.rebudget(dict(payload, messages=convo), role="main")
        if hop == 0:
            # THE CLIENT'S OWN REQUEST IS NEVER LANDED (C1): it was measured
            # before the turn (check_client_prompt) and fits, so its tools
            # stay and nothing is appended. Its token fields are cut to the
            # window it was measured against (a prefill counted by estimate
            # on top).
            ctx = payload.get("_context") or {}
            n0 = ctx.get("tokens") if ctx.get("counted") else \
                tiers.estimate_prompt_tokens(payload)
            if prefill is not None:
                n0 = int(n0 or 0) + tiers.estimate_prompt_tokens(
                    {"messages": [prefill]})
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
        # A prefill opens hop 0 only; hops 1+ see hop 0 in its place.
        if prefill is not None:
            send = dict(payload, messages=list(convo) + [prefill])
            prefilled = True
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
        held = ""
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
        # This generation's own usage; the turn's usage is decided at the
        # end (_usage_of).
        d["usage"] = dict(u)
        msg = d["choices"][0]["message"]
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
            else:
                yield from out.content(held)
            held = ""
        # The landing is the last generation whatever it asked for: a call
        # made there is not run (nothing would read its result).
        if (not calls and img is None) or last:
            break
        # The hop's reasoning stays with it (the template renders it back),
        # and the ledger records the hops for the next request.
        hop_msg = {"role": "assistant", "content": msg.get("content") or "",
                   "reasoning_content": msg.get("reasoning_content") or "",
                   "tool_calls": msg.get("tool_calls") or []}
        if prefill is not None:
            # THE PREFILL CARRIES INTO HOPS 1+ (pre-deploy review,
            # 2026-09-24). The prefill is sent on hop 0 only; hops 1+ see
            # this hop in its place. llama-server re-sends a prefill's
            # reasoning and content as its first deltas (STEP 0), so they
            # are normally here already -- where they are not, put back,
            # exactly once.
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
            # thread with heartbeats.
            n_img0 = len(payload["_images"])
            status, res = yield from _in_thread(
                lambda fn=fn, args=args: run_our_tool(fn, args, state))
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
            tmsg = {"role": "tool", "tool_call_id": c["id"],
                    "content": repeats.cap_tool_result(res, fn, args)}
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
        prefill = None

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
    # 2026-09-27, docs/CONSTANTS-AUDIT.md).
    if fin == "length" or (fin == "stop" and not content.strip()
                           and not client_calls):
        print(f"  finish={fin}, {len(content.strip())} chars of content; "
              f"delivered as generated", flush=True)
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
    # slot holds (a prefill's line included).
    msg["reasoning_content"] = slot_msg.get("reasoning_content") or \
        msg.get("reasoning_content") or ""
    delivered = {"role": "assistant", "content": content,
                 "reasoning_content": msg["reasoning_content"]}
    if client_calls:
        delivered["tool_calls"] = client_calls
    # WHAT THE CLIENT STORES (#10, docs/SELF-IMPROVEMENT-LOG.md). A streaming
    # client keeps every content byte it was sent. When a hop that called one
    # of OUR tools had already streamed content (a preface past
    # HOLD_CONTENT_CHARS, an image line), that is more than `content` -- the
    # last hop's answer -- and a ledger keyed by `content` alone never
    # matched the turn the client sent back: the hidden hops were not
    # restored, and the next request re-read the whole turn (live gate
    # 2026-09-24: reused 2513 of 3955 after a 7-generation turn). The turn is
    # keyed by what the client stores; the content it renders is the
    # delivered one (ledger_restore).
    stored = dict(delivered)
    # STRAY TEMPLATE MARKERS (#12, _StrayMarkers): the client's copy has them
    # out -- the stream filtered them as it went; the blocking path cleans
    # the whole answer here, the same filters in the same order (image
    # duplicates, then markers). `delivered` keeps the slot's text: the
    # ledger renders it, keyed by the client's copy (#10), so the next
    # request extends the slot. `clean`: the model's answer as the client
    # gets it, without the lines the proxy put before it.
    if streamed:
        stray = out.markers
        clean = strip_stray_markers(content or "")
    else:
        clean, stray = _StrayMarkers.clean(
            out.dedup.strip(content or "") if out.images else content or "")
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
    # _echo_guess (the account's history): an echoing client on an account
    # guessed "strips" sends back what it was streamed, which must match
    # (deploy check 2026-09-27, h5: agent loop [echo] step 2 reused 4,131 of
    # 7,133 before this). The record is a hash; it matches only an echo.
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
    _log_turn(state, delivered)
    # THE COMPACTION LINK (#38): a compaction's window starts when it ENDS,
    # and its continuation is recognised by the summary it carries; any
    # other conversation turn refreshes the account's last activity.
    if payload.get("_utility_kind") == "compaction":
        # The summary as the client will carry it (stray markers out).
        _compaction_done(body.get("_account") or "",
                         (payload.get("_slot") or {}).get("key"), clean)
    elif not body["_utility"].get("utility"):
        _session_touch(body.get("_account") or "", state.get("_key") or "")
    usage, usage_rec = _usage_of(gens, spent, n_calls, payload)
    payload["_image_guard_rec"] = img_rec
    d["x_yamadori"] = _x_yamadori(payload, hops=n_calls)
    d["x_yamadori"]["usage"] = usage_rec
    # where the wall clock went: every timed stage from the top of the turn to here, and what no stage covers
    d["x_yamadori"]["timing"] = stage_timing.record()
    stage_timing.log_if_slow(d["x_yamadori"]["timing"],
                             "side call" if (payload.get("_utility") or {}).get("utility") else "request")
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
# HIDDEN HOPS -- a generation that called one of OUR tools (images, vision,
# the MCP lookups, the craft tool) -- have no field in the spec. Their output
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


# ============================================================== WARMING ====
#
# When the turn the client stores differs from what the slot generated -- a
# line the proxy wrote, a call's rewritten ids, a prefill the client drops --
# the next request would diverge inside that turn and roll the slot back to a
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
    ok_touch, why_locked = max_mode.touch_allowed("warm", model)
    if max_mode.blocks(model) or not ok_touch:
        # MAX MODE: this conversation's model is off the card (a warm would load it), or LOCKED (nothing but its
        # own conversation's generation touches it: operator 2026-09-30)
        rec.update(state="skipped", why=max_mode.blocked_reason(model) if max_mode.blocks(model) else why_locked)
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
        token_ledger.record("warm", timings=t, model=model)
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
# The model does not have record_step / read_rings (operator, 2026-09-24):
# the proxy writes the work log (mcp/rings.py) from the turns it sees, under
# the conversation's lineage, and re-injects it on the first user turn after
# a compaction (prepare, _work_log_block). What is logged is a CHOICE: the
# client calls the model made. (What the code checks found, what deep
# thinking handed back and what fan-out decided were logged too until those
# were removed, 2026-09-29.)
def _log_turn(state: dict | None, msg: dict) -> None:
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
    except Exception as e:                                       # noqa: BLE001
        print(f"  work log not written: {type(e).__name__}: {e}", flush=True)


def complete(body: dict) -> dict:
    """The blocking path: the one turn implementation, drained.

    Under a cancel token like the streamed path (never cancelled here): job
    threads started under it (_in_thread) share it, which is how a slot
    release at the end of one finds this request (slots.bind_request).
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
            # `critic-disabled`, `embeddings` and (until 2026-10-01) `reranker` -- a bypass, a
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


# THE WAIT IS HEARD (2026-10-02). The stream said nothing until the turn's first event -- and the first event of
# a max request waits for the Flash-Next load (131-272 s measured) and then its prompt (a cold 30K prompt: 135-321
# s). The operator's live session, 2026-10-01 ET, VS Copilot agents: the client hung up at exactly 180.0 s with no
# byte received (17:28), again at 20:25, and each hang-up cancelled the load or the prefill it was waiting for.
# So the turn runs on its own thread (_TurnPump) and whenever it has produced nothing for HEARTBEAT seconds the
# stream sends a heartbeat -- the empty delta stream_body already sends for "heartbeat" events, which Responses
# turns into `response.in_progress` and Messages into `ping`. HEARTBEAT (5 s) is the proxy's existing beat, under
# the shortest client silence limit we know (Hermes' codex transport: 12 s without a parsed event). E1 KEEPS ITS
# PROMISE: no heartbeat goes out until the turn's PRE-FLIGHT is over (_run_turn marks the request's token after
# the window check), so validation, 503 at capacity and 400 context_length_exceeded are still real HTTP statuses
# however long they take to decide; a failure after the first heartbeat is the committed stream's ONE error event,
# as after any first byte.
class _TurnPump:
    """`_run_turn`'s events, pulled on ONE thread of its own (every resume on the same thread, the request's
    cancellation bound to it, the caller's contextvars copied), with a ("heartbeat", None) from next() whenever the
    turn has produced nothing for HEARTBEAT seconds. next() raises what the turn raised (StopIteration with the
    turn's value at its end)."""

    def __init__(self, gen, token: "cancel.Token"):
        import contextvars
        import queue
        self._gen, self._token = gen, token
        self._q: "queue.Queue" = queue.Queue()
        self._empty = queue.Empty
        self._done = False
        # THE TURN ADVANCES ONLY WHEN THE STREAM PULLS, as it did when the stream called next(gen) itself: one
        # permit per pull (mcp/test_ledger.py [race]: a turn that ran ahead of its consumer finished its warm
        # before the client had the calls). A heartbeat leaves the pull outstanding and takes no new permit.
        self._want = threading.Semaphore(0)
        self._pulling = False
        self._closed = False
        ctx = contextvars.copy_context()
        self._th = threading.Thread(target=lambda: ctx.run(self._run), daemon=True, name="yamadori-turn")
        self._th.start()

    def _run(self) -> None:
        try:
            with cancel.bound(self._token):
                while True:
                    self._want.acquire()
                    if self._closed:
                        return
                    try:
                        ev = next(self._gen)
                    except StopIteration as stop:
                        self._q.put(("stop", stop.value))
                        return
                    self._q.put(("event", ev))
        except BaseException as e:                                   # noqa: BLE001
            self._q.put(("error", e))

    def next(self):
        if self._done:
            raise StopIteration(None)
        if not self._pulling:
            self._pulling = True
            self._want.release()
        while True:
            try:
                what, item = self._q.get(timeout=HEARTBEAT)
                break
            except self._empty:
                # heartbeats once the turn's pre-flight is over (_run_turn sets it after the window check): until
                # then a refusal must still be a real HTTP status (E1), however long the check takes -- EXCEPT
                # while the request waits for the card or loads its model (max_mode.wait_ready's `card_wait`:
                # minutes, and it comes BEFORE the pre-flight). A refusal after such a wait is then the committed
                # stream's one error event: only a request that both swaps a model in and fails its pre-flight.
                if getattr(self._token, "preflight_done", False) or getattr(self._token, "card_wait", False):
                    return ("heartbeat", None)
        self._pulling = False
        if what == "event":
            return item
        self._done = True
        if what == "stop":
            raise StopIteration(item)
        raise item

    def close(self) -> None:
        """The stream ended. If the turn is still running its token was cancelled by the caller: wait for its
        thread to unwind (the cancel shuts its upstream sockets), then close the generator from here."""
        self._closed = True
        self._want.release()
        if self._th.is_alive():
            self._th.join(30)
        if not self._th.is_alive():
            try:
                with cancel.bound(self._token):
                    self._gen.close()
            except Exception:                                        # noqa: BLE001
                pass


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
    pump = _TurnPump(gen, token)
    try:
        while True:
            try:
                kind, item = pump.next()
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
        pump.close()
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
