#!/usr/bin/env python
"""The Anthropic Messages API (`POST /v1/messages`, `POST
/v1/messages/count_tokens`), as a TRANSLATION LAYER.

WHY (operator, 2026-09-29: "Yes add anthropic api endpoints for claude").
Claude Code speaks only the Anthropic Messages format to a custom
ANTHROPIC_BASE_URL (code.claude.com/docs/en/llm-gateway-protocol, "API
formats": `/v1/messages`, `/v1/messages/count_tokens` optional). With this
module Claude Code can use Yamadori as its model.

WHAT IT IS, AND WHAT IT IS NOT. Not a second pipeline -- the same rule as
mcp/responses_api.py. A Messages request is translated into the chat body
`proxy._run_turn` already takes (`to_chat`), the SAME turn runs (server.
_serve_turn: admission, the tier -> model table, supersede, the E1 commit
point, cancel on disconnect; proxy: tiers, the ledger and reasoning restore,
sessions, the window check, hidden hops, the MCP host's tools, stray-marker
filtering, the code check, usage) and its result is translated back: a
Message object (`of_chat`), or Anthropic's SSE event grammar (`Stream`,
which reads `proxy.stream_body`'s own chunks, so every stream promise of the
chat path holds).

THE MAPPING (request -> chat)

  system (a string, or text blocks)  the leading system message: blocks
                                joined by a blank line, cache_control
                                dropped and counted (x_yamadori.messages.
                                cache_control); a `role: "system"` entry in
                                `messages` (Claude Code appends them mid-
                                conversation) goes through system_roles.
                                one_system: leading ones join the system
                                message, later ones become user turns
  user text blocks              text; all-text content becomes ONE string
                                ("".join, as the template renders a text-part
                                list)
  image (base64 | url | file)   the chat image path: a data: URL image_url
                                part, an http(s) URL (never fetched: vision.
                                extract refuses to download), a file id kept
                                as a Responses input_image part (named and
                                explained by vision.extract: no files API)
  document / other user blocks  400 invalid_request_error naming the block
  tool_result                   a `tool` message: content a string or blocks
                                (text; images as image_url parts; a document
                                / search_result / tool_reference becomes a
                                text note -- refusing would wedge the
                                conversation, it is in every later request);
                                `is_error` counted (the content already says
                                what failed; the chat wire has no flag)
  assistant text                content (consecutive assistant messages merge
                                into ONE turn, as Anthropic merges them)
  tool_use                      tool_calls, the id KEPT (our session carrier
                                rides in it, exactly like chat's call ids),
                                input serialized in its own key order (the
                                server parses it back before the template)
  thinking                      the turn's reasoning_content, ONLY when its
                                `signature` verifies as ours (an HMAC over
                                the account and the text: `sign_thinking`);
                                an unverified block (another model's, a
                                forged one) is dropped and counted -- the
                                ledger then puts back the slot's own
                                reasoning (restore_reasoning), as for a
                                client that strips. A verified echo is kept
                                as sent (AGENTS.md "Past reasoning is
                                restored": an echo is kept as sent)
  redacted_thinking, server_tool_use, *_tool_result  ignored, counted
  tools (name, description, input_schema)  function tools; `strict` kept;
                                `defer_loading` ignored (every tool is
                                offered) and counted; cache_control dropped
  server tools (web_search_*, web_fetch_*, code_execution_*,
  tool_search_tool_*, advisor_*, mcp_toolset)  ignored and recorded: the
                                model never sees them (as Responses ignores
                                hosted tools)
  Anthropic-schema client tools (bash_*, text_editor_*, computer_*,
  memory_*) and unknown types   400: their schema is Anthropic's, not sent
  tool_choice                   what llama-server serves (common/chat.cpp
                                common_chat_tool_choice_parse_oaicompat:
                                auto, none, required ONLY; an object falls
                                back to auto with a warning,
                                server-common.h json_value):
                                  auto -> "auto", none -> "none",
                                  any -> "required",
                                  tool {name} -> the tools narrowed to that
                                  one + "required" (the same constraint in
                                  the form the server has; recorded); a name
                                  not among the tools -> 400
                                disable_parallel_tool_use -> parallel_tool_calls
  max_tokens (required)         max_tokens -- the tier's model profile owns
                                the turn's size (tiers.apply, `_client`)
  stop_sequences                stop
  temperature / top_p / top_k   the same fields as chat (the vendor sampling
                                is enforced and the client's recorded)
  output_config.format          response_format (json_schema)
  output_config.effort, thinking, the model name   the TIER (below)
  metadata.user_id              a session source only when it carries a
                                session id (JSON `session_id`, or the legacy
                                `..._session_<uuid>` form); a user id alone
                                names a person, not a conversation: ignored
  context_management, container, mcp_servers, service_tier, inference_geo,
  speed                         accepted, ignored, recorded
  headers                       anthropic-version / anthropic-beta recorded;
                                the Claude Code session and hint headers
                                below

EFFORT -> TIER (deterministic; x_yamadori.messages.effort records the inputs,
the rule that fired, the tier and the model that serves it). First match:

  1. output_config.effort                  low -> low, medium -> medium,
                                           high -> high, xhigh -> xhigh,
                                           max -> max: the names are the
                                           same scale (Anthropic's effort
                                           levels, code.claude.com/docs/en/
                                           model-config "Available Levels":
                                           low, medium, high, xhigh, max; our
                                           tiers.ORDER after `minimal`). Any
                                           other value: 400
  2. thinking {type: "disabled"} or
     {type: "between_tools"}              minimal (tiers.TIERS: thinking off
                                           is `minimal`; between_tools is
                                           Anthropic's "no up-front
                                           thinking")
  3. otherwise                             no effort sent: the server's
                                           default tier (tiers.DEFAULT), or
                                           a tier variant named by `model`
                                           (catalog: yamadori-max, ...)
  X-Yamadori-Features applies on top, as on every wire (tiers.resolve).
  thinking {enabled, budget_tokens} and {adaptive}: no tier of their own
  (the model's profile owns the thinking budget, operator 2026-09-29: "This
  is stuff the user and the harness will just get wrong"); recorded.

  Evidence, Claude Code's side: for a model id it does not recognise it
  sends "Everything current Claude models accept on the Claude API,
  including adaptive reasoning, effort, and context management" -- effort in
  `output_config` (gateway protocol, "Requests and defaults by connection
  method" and "Feature pass-through"). `/effort` and `effortLevel` set it
  (model-config, "How to Set Effort Level"). The tier then picks the model
  (mcp/tier_models.yaml: minimal..high -> bonsai, xhigh -> mirai-s, max ->
  flash-next).

SESSIONS (AGENTS.md "A conversation is named by an explicit session id").
`x-claude-code-session-id` ("A unique identifier for the current Claude Code
session", gateway protocol "Request headers") is the conversation's key,
sent as the chat body's `prompt_cache_key` (source 1), with
`x-claude-code-agent-id` appended for a subagent's own turns (each subagent
is its own conversation; a fork's history carries our ids, so it is
recorded forked_from, as for Codex). An X-Yamadori-Session header wins (it is
ours). Otherwise metadata.user_id's session, otherwise our id in the
tool_use ids (the chat carrier, unchanged).

CLAUDE CODE'S HINT HEADERS (sent with CLAUDE_CODE_GATEWAY_HINT_HEADERS=1,
which the harness kit sets): recorded; `x-claude-code-request-class:
auxiliary` ("side requests such as session titles, classifiers, and
summaries") on a request that offers no tools is a client side call --
X-Yamadori-Features {"utility": true} (the bare model), unless a header of
the caller's says otherwise.

THE OUTPUT (Message): content blocks thinking (with our signature; empty
text with display "omitted"; none when the client turned thinking off), text,
tool_use (id, name, input as an object); stop_reason tool_use / max_tokens /
end_turn (llama-server's chat route does not say WHICH stop ended a turn, so
stop_sequence is never claimed; recorded); usage input_tokens (the prompt
LESS the reused cache, as Anthropic counts it), cache_read_input_tokens
(chat's cached_tokens), cache_creation_input_tokens 0, output_tokens.
`x_yamadori` is the chat one plus `messages`.

THE STREAM (code.claude.com gateway protocol "Streaming"): message_start,
then per block content_block_start / content_block_delta (thinking_delta,
signature_delta, text_delta, input_json_delta) / content_block_stop, then
message_delta (stop_reason, usage, x_yamadori) and message_stop. `ping`:
each chat heartbeat (throttled to KEEPALIVE_S), and -- since Claude Code
"aborts a stream that goes silent for 300 seconds by default" and counts
every byte -- a ping whenever the turn has sent nothing for PING_S (the
proxy's own HEARTBEAT cadence), from a pump thread, once the stream has
started. Nothing is sent before the turn's first chunk (E1): a failure
before it is a real HTTP status. After it, an `error` event.

ERRORS: Anthropic's envelope {type: "error", error: {type, message},
request_id} (platform.claude.com/docs/en/api/errors), mapped from our one
error path (api_errors) by `anthropic_error`; the context-overflow message is
Anthropic's own wording, "prompt is too long: N tokens > M maximum", which
Claude Code recognises and compacts on (code.claude.com/docs/en/errors,
"Prompt is too long").
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import queue
import re
import threading
import time
import uuid

import api_errors
import system_roles

KEEPALIVE_S = 1.0       # the Responses translation's own throttle (responses_api.KEEPALIVE_S)
EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING_TYPES = ("enabled", "adaptive", "disabled", "between_tools")
THINKING_OFF = ("disabled", "between_tools")

# Server tools Anthropic runs: nothing here runs them; the model never sees
# them (web search, fetch, code execution, tool search, the advisor, the MCP
# connector's toolsets).
SERVER_TOOL_PREFIXES = ("web_search_", "web_fetch_", "code_execution_",
                        "tool_search_tool_", "advisor_", "mcp_toolset")
# Client tools whose schema is Anthropic's own (never sent in the request):
# nothing to render them from.
SCHEMA_TOOL_PREFIXES = ("bash_", "text_editor_", "computer_", "memory_")

IGNORED_FIELDS = ("context_management", "container", "mcp_servers",
                  "service_tier", "inference_geo", "speed")

HINT_HEADERS = ("x-claude-code-request-class", "x-claude-code-agent-type",
                "x-claude-code-compaction", "x-claude-code-context-compacted",
                "x-claude-code-prompt-id")
_SESSION_OK = re.compile(r"[A-Za-z0-9._:\-]{1,128}")
_LEGACY_USER_SESSION = re.compile(r"_session_([0-9A-Za-z\-]{8,64})$")


def _err(message: str, param: str | None = None,
         code: str = "invalid_value") -> api_errors.ApiError:
    return api_errors.invalid(message, param=param, code=code)


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# ------------------------------------------------------------- signatures --

SIG_PREFIX = "ymt1."


def _sig_key() -> bytes:
    """The thinking-signature key: derived from the media-link secret
    (images._secret, created once, never logged), domain-separated, so no
    second secret file exists."""
    import images
    return hmac.new(images._secret(), b"yamadori/messages/thinking/1",
                    hashlib.sha256).digest()


def sign_thinking(text: str, account: str) -> str:
    """An opaque, verifiable signature for a thinking block we produced:
    HMAC-SHA256 over the account and the block's text."""
    mac = hmac.new(_sig_key(), (account or "").encode("utf-8") + b"\0"
                   + (text or "").encode("utf-8"), hashlib.sha256).digest()
    return SIG_PREFIX + base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def verify_thinking(text: str, signature, account: str) -> bool:
    if not isinstance(signature, str) or not signature.startswith(SIG_PREFIX):
        return False
    return hmac.compare_digest(sign_thinking(text, account), signature)


# ----------------------------------------------------------------- errors --

_TYPES = {400: "invalid_request_error", 401: "authentication_error",
          402: "billing_error", 403: "permission_error",
          404: "not_found_error", 405: "invalid_request_error",
          409: "conflict_error", 413: "request_too_large",
          422: "invalid_request_error", 429: "rate_limit_error",
          500: "api_error", 502: "api_error", 503: "overloaded_error",
          504: "timeout_error", 529: "overloaded_error"}


def anthropic_type(status: int, code: str | None = None) -> str:
    """Anthropic's error type for one of our errors. Capacity refusals --
    the lanes full (429 server_busy), another tier's model on the card or
    the model server loading (503 model_at_capacity / model_unavailable) --
    are `overloaded_error` (the brief, 2026-09-29); the rest by status."""
    if code in ("server_busy", "model_at_capacity", "model_unavailable"):
        return "overloaded_error"
    s = int(status)
    return _TYPES.get(s, "invalid_request_error" if 400 <= s < 500
                      else "api_error")


def anthropic_message(e: api_errors.ApiError) -> str:
    """The message, in Anthropic's wording where a client matches on it: a
    prompt past the window is "prompt is too long: N tokens > M maximum"
    (Claude Code's reactive compaction), M the tokens a prompt may take here
    (the window less the generation floor)."""
    if e.code == "context_length_exceeded":
        n = int(e.extra.get("n_prompt_tokens") or 0)
        limit = int(e.extra.get("context_length") or 0)
        floor = int(e.extra.get("floor") or 0)
        return (f"prompt is too long: {n} tokens > {max(limit - floor, 0)} "
                f"maximum")
    return e.message


def anthropic_error(e: api_errors.ApiError) -> tuple[int, dict, dict]:
    """(status, body, headers) of one of our errors in Anthropic's envelope.
    Our code and the facts ride in `error.x_yamadori` (clients ignore it)."""
    rid = _new("req")
    err = {"type": anthropic_type(e.status, e.code),
           "message": anthropic_message(e)}
    x = {"code": e.code}
    if e.param:
        x["param"] = e.param
    for k, v in e.extra.items():
        x.setdefault(k, v)
    err["x_yamadori"] = x
    headers = dict(e.headers or {})
    headers["request-id"] = rid
    return e.status, {"type": "error", "error": err, "request_id": rid}, headers


def unauthorised(why: str) -> api_errors.ApiError:
    return api_errors.ApiError(401, f"{why}. Set an API key: x-api-key "
                               f"(ANTHROPIC_API_KEY) or Authorization: Bearer "
                               f"(ANTHROPIC_AUTH_TOKEN).",
                               code="invalid_api_key")


def sse_error(e: api_errors.ApiError) -> bytes:
    _status, body, _h = anthropic_error(e)
    return _sse("error", {"type": "error", "error": body["error"]})


# ---------------------------------------------------------------- blocks ---

def _text_of_blocks(content, where: str) -> str:
    """System content: a string, or text blocks (cache_control dropped)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _err(f"{where}: must be a string or an array of text blocks.",
                   where, "invalid_type")
    out = []
    for j, b in enumerate(content):
        if not isinstance(b, dict) or b.get("type") != "text" \
                or not isinstance(b.get("text"), str):
            raise _err(f"{where}.{j}: system blocks must be text blocks.",
                       f"{where}.{j}")
        out.append(b["text"])
    return "\n\n".join(out)


def _image_part(b: dict, where: str, *, in_tool: bool = False) -> dict:
    src = b.get("source")
    if not isinstance(src, dict):
        raise _err(f"{where}.source: Field required.", f"{where}.source",
                   "missing_required_parameter")
    kind = src.get("type")
    if kind == "base64":
        mt, data = src.get("media_type"), src.get("data")
        if not isinstance(mt, str) or not isinstance(data, str) or not data:
            raise _err(f"{where}.source: base64 images need media_type and "
                       f"data.", f"{where}.source")
        return {"type": "image_url",
                "image_url": {"url": f"data:{mt};base64,{data}"}}
    if kind == "url" and isinstance(src.get("url"), str) and src["url"]:
        return {"type": "image_url", "image_url": {"url": src["url"]}}
    if kind == "file" and isinstance(src.get("file_id"), str):
        # No files API here: vision.extract names it and says so.
        return {"type": "input_image", "file_id": src["file_id"]}
    raise _err(f"{where}.source.type {json.dumps(kind)} is not served; send "
               f"base64 or url images.", f"{where}.source.type")


def _note(text: str) -> dict:
    return {"type": "text", "text": text}


def _tool_result_content(content, where: str, rec: dict):
    """A tool_result's content as chat tool content: a string when all text,
    else a list of text and image_url parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _err(f"{where}: must be a string or an array of blocks.",
                   where, "invalid_type")
    parts = []
    for j, b in enumerate(content):
        w = f"{where}.{j}"
        if not isinstance(b, dict):
            raise _err(f"{w}: must be an object.", w, "invalid_type")
        t = b.get("type")
        if t == "text":
            parts.append(_note(str(b.get("text") or "")))
        elif t == "image":
            parts.append(_image_part(b, w, in_tool=True))
        elif t == "search_result":
            body = "".join(str(c.get("text") or "") for c in
                           b.get("content") or [] if isinstance(c, dict))
            parts.append(_note(f"[{b.get('title') or 'result'}]"
                               f"({b.get('source') or ''})\n{body}"))
        elif t == "tool_reference":
            parts.append(_note(f"[tool reference: {b.get('tool_name') or b.get('name') or '?'}]"))
            rec["tool_references"] = rec.get("tool_references", 0) + 1
        else:
            # A tool produced it: refusing would wedge the conversation for
            # good (it is in every later request), so the model is told.
            parts.append(_note(f"[a {t or 'block'} was returned here. This "
                               f"server passes only text and images to the "
                               f"model, so its content is unavailable; tell "
                               f"the user if it matters.]"))
            rec.setdefault("tool_blocks_noted", []).append(str(t))
    if all(p.get("type") == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _user_parts(content, where: str) -> tuple[list, list]:
    """(chat parts in order, [(index, tool_result block)]) of a user
    message's content blocks; each entry of the first list is ("part", p)
    or ("tool", block)."""
    seq = []
    for j, b in enumerate(content):
        w = f"{where}.{j}"
        if not isinstance(b, dict):
            raise _err(f"{w}: must be an object.", w, "invalid_type")
        t = b.get("type")
        if t == "text":
            if not isinstance(b.get("text"), str):
                raise _err(f"{w}.text: must be a string.", f"{w}.text",
                           "invalid_type")
            seq.append(("part", _note(b["text"])))
        elif t == "image":
            seq.append(("part", _image_part(b, w)))
        elif t == "tool_result":
            seq.append(("tool", b, w))
        else:
            raise _err(f"{w}.type {json.dumps(t)} cannot be read by this "
                       f"model; send text, image or tool_result blocks.",
                       f"{w}.type", "invalid_content_part")
    return seq


def _flush_user(parts: list, msgs: list) -> None:
    if not parts:
        return
    if all(p.get("type") == "text" for p in parts):
        msgs.append({"role": "user",
                     "content": "".join(p["text"] for p in parts)})
    else:
        msgs.append({"role": "user", "content": list(parts)})
    parts.clear()


def _tool_call(b: dict, where: str) -> dict:
    cid, name = b.get("id"), b.get("name")
    if not isinstance(cid, str) or not cid:
        raise _err(f"{where}.id: Field required.", f"{where}.id",
                   "missing_required_parameter")
    if not isinstance(name, str) or not name:
        raise _err(f"{where}.name: Field required.", f"{where}.name",
                   "missing_required_parameter")
    inp = b.get("input")
    if inp is None:
        inp = {}
    args = inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _messages(body: dict, rec: dict, account: str) -> list[dict]:
    raw = body.get("messages")
    if raw is None:
        raise _err("messages: Field required.", "messages",
                   "missing_required_parameter")
    if not isinstance(raw, list) or not raw:
        raise _err("messages: must be a non-empty array.", "messages",
                   "invalid_type")
    msgs: list[dict] = []
    system = _text_of_blocks(body.get("system"), "system")
    if system:
        msgs.append({"role": "system", "content": system})
    counts = rec.setdefault("thinking_in", {})

    def assistant() -> dict:
        if msgs and msgs[-1].get("role") == "assistant" and \
                msgs[-1].get("_from_messages"):
            return msgs[-1]
        a = {"role": "assistant", "content": "", "_from_messages": True}
        msgs.append(a)
        return a

    for i, m in enumerate(raw):
        where = f"messages.{i}"
        if not isinstance(m, dict):
            raise _err(f"{where}: must be an object.", where, "invalid_type")
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant", "system"):
            raise _err(f"{where}.role: must be user or assistant; got "
                       f"{json.dumps(role)}.", f"{where}.role")
        if content is not None and not isinstance(content, (str, list)):
            raise _err(f"{where}.content: must be a string or an array of "
                       f"content blocks.", f"{where}.content", "invalid_type")
        if role == "system":
            msgs.append({"role": "system",
                         "content": _text_of_blocks(content,
                                                    f"{where}.content")})
            continue
        if role == "user":
            if content is None or isinstance(content, str):
                msgs.append({"role": "user", "content": content or ""})
                continue
            pending: list = []
            for item in _user_parts(content, f"{where}.content"):
                if item[0] == "part":
                    pending.append(item[1])
                    continue
                _flush_user(pending, msgs)
                b, w = item[1], item[2]
                tid = b.get("tool_use_id")
                if not isinstance(tid, str) or not tid:
                    raise _err(f"{w}.tool_use_id: Field required.",
                               f"{w}.tool_use_id",
                               "missing_required_parameter")
                if b.get("is_error"):
                    rec["tool_errors"] = rec.get("tool_errors", 0) + 1
                msgs.append({"role": "tool", "tool_call_id": tid,
                             "content": _tool_result_content(
                                 b.get("content"), f"{w}.content", rec)})
            _flush_user(pending, msgs)
            continue
        # assistant: one turn, merged with an assistant message right before
        a = assistant()
        if isinstance(content, str) or content is None:
            a["content"] = (a.get("content") or "") + (content or "")
            continue
        for j, b in enumerate(content):
            w = f"{where}.content.{j}"
            if not isinstance(b, dict):
                raise _err(f"{w}: must be an object.", w, "invalid_type")
            t = b.get("type")
            if t == "text":
                a["content"] = (a.get("content") or "") + str(b.get("text") or "")
            elif t == "tool_use":
                a.setdefault("tool_calls", []).append(_tool_call(b, w))
            elif t == "thinking":
                text = b.get("thinking") if isinstance(b.get("thinking"),
                                                       str) else ""
                if not text:
                    counts["empty"] = counts.get("empty", 0) + 1
                elif verify_thinking(text, b.get("signature"), account):
                    a["reasoning_content"] = (a.get("reasoning_content")
                                              or "") + text
                    counts["verified"] = counts.get("verified", 0) + 1
                else:
                    counts["unverified_dropped"] = counts.get(
                        "unverified_dropped", 0) + 1
            elif t == "redacted_thinking":
                counts["redacted_ignored"] = counts.get("redacted_ignored", 0) + 1
            else:
                # server_tool_use, web_search_tool_result, ...: records of a
                # server tool this server never ran; nothing the model reads.
                rec.setdefault("blocks_ignored", []).append(str(t))
    if not counts:
        rec.pop("thinking_in", None)
    if rec.get("blocks_ignored"):
        rec["blocks_ignored"] = sorted(set(rec["blocks_ignored"]))
    for m in msgs:
        m.pop("_from_messages", None)
    msgs, moved = system_roles.one_system(msgs)
    if moved.get("developer_as_user"):
        rec["system_as_user"] = moved["developer_as_user"]
    if not any(m.get("role") != "system" for m in msgs):
        raise _err("messages: at least one user message is required.",
                   "messages")
    return msgs


# ----------------------------------------------------------------- tools ---

def _tools(body: dict, rec: dict) -> list[dict]:
    tools = body.get("tools")
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise _err("tools: must be an array.", "tools", "invalid_type")
    out = []
    for i, t in enumerate(tools):
        w = f"tools.{i}"
        if not isinstance(t, dict):
            raise _err(f"{w}: must be an object.", w, "invalid_type")
        kind = t.get("type")
        if kind in (None, "custom"):
            name, schema = t.get("name"), t.get("input_schema")
            if not isinstance(name, str) or not name:
                raise _err(f"{w}.name: Field required.", f"{w}.name",
                           "missing_required_parameter")
            if not isinstance(schema, dict):
                raise _err(f"{w}.input_schema: Field required.",
                           f"{w}.input_schema", "missing_required_parameter")
            fn = {"name": name}
            if isinstance(t.get("description"), str):
                fn["description"] = t["description"]
            fn["parameters"] = schema
            if isinstance(t.get("strict"), bool):
                fn["strict"] = t["strict"]
            if t.get("defer_loading"):
                rec["deferred_tools_offered"] = rec.get(
                    "deferred_tools_offered", 0) + 1
            if t.get("cache_control") is not None:
                rec["cache_control"] = rec.get("cache_control", 0) + 1
            out.append({"type": "function", "function": fn})
        elif isinstance(kind, str) and kind.startswith(SERVER_TOOL_PREFIXES):
            rec.setdefault("tools_ignored", []).append(kind)
        elif isinstance(kind, str) and kind.startswith(SCHEMA_TOOL_PREFIXES):
            raise _err(f"'yamadori' does not support tool types: {kind}. "
                       f"Define it as a custom tool with an input_schema.",
                       f"{w}.type")
        else:
            raise _err(f"{w}.type {json.dumps(kind)} cannot be served here: "
                       f"send custom tools (name, description, "
                       f"input_schema).", f"{w}.type")
    return out


def _tool_choice(tc, tools: list, rec: dict):
    """(chat tool_choice or None, parallel_tool_calls or None, tools)."""
    if tc is None:
        return None, None, tools
    if not isinstance(tc, dict):
        raise _err("tool_choice: must be an object.", "tool_choice",
                   "invalid_type")
    kind = tc.get("type")
    par = None
    if isinstance(tc.get("disable_parallel_tool_use"), bool):
        par = not tc["disable_parallel_tool_use"]
    if kind == "auto":
        return "auto", par, tools
    if kind == "none":
        return "none", par, tools
    if kind == "any":
        return "required", par, tools
    if kind == "tool":
        name = tc.get("name")
        keep = [t for t in tools if t["function"]["name"] == name]
        if not keep:
            raise _err(f"tool_choice.name {json.dumps(name)} is not one of "
                       f"the tools.", "tool_choice.name")
        rec["tool_choice_narrowed"] = name
        return "required", par, keep
    raise _err(f"tool_choice.type must be auto, any, tool or none; got "
               f"{json.dumps(kind)}.", "tool_choice.type")


# ---------------------------------------------------------------- effort ---

def _effort(body: dict, rec: dict) -> str | None:
    """The reasoning_effort to send (a tier name), or None for the default;
    rec['effort'] says which rule fired (see the module doc)."""
    oc = body.get("output_config")
    if oc is not None and not isinstance(oc, dict):
        raise _err("output_config: must be an object.", "output_config",
                   "invalid_type")
    eff = (oc or {}).get("effort")
    th = body.get("thinking")
    if th is not None and not isinstance(th, dict):
        raise _err("thinking: must be an object.", "thinking", "invalid_type")
    ttype = (th or {}).get("type")
    if th is not None and ttype not in THINKING_TYPES:
        raise _err(f"thinking.type must be one of "
                   f"{', '.join(THINKING_TYPES)}; got {json.dumps(ttype)}.",
                   "thinking.type")
    e = {"output_config.effort": eff, "thinking": ttype,
         "budget_tokens": (th or {}).get("budget_tokens"),
         "display": (th or {}).get("display"), "model": body.get("model")}
    if eff is not None:
        if eff not in EFFORTS:
            raise _err(f"output_config.effort must be one of "
                       f"{', '.join(EFFORTS)}; got {json.dumps(eff)}.",
                       "output_config.effort")
        e["rule"] = "output_config.effort names the tier (the same scale)"
        e["sent"] = eff
    elif ttype in THINKING_OFF:
        e["rule"] = f"thinking {ttype}: the thinking-off tier"
        e["sent"] = "minimal"
    else:
        e["rule"] = ("no effort: the server's default tier, or the tier a "
                     "yamadori-* model name carries")
        e["sent"] = None
    rec["effort"] = e
    return e["sent"]


def _record_tier(chat: dict, rec: dict, header_features: str = "") -> None:
    """The tier this request resolves to and the model that serves it, for
    x_yamadori.messages.effort (pure: the catalog, tiers and the table)."""
    try:
        import max_mode
        import tier_models
        feats = merge_features(chat.get("_features"), header_features or None)
        tier = max_mode.requested_tier(dict(chat, _features=feats)
                                       if feats else chat)
        rec["effort"]["tier"] = tier
        rec["effort"]["serves"] = tier_models.model_for(tier)
    except Exception as ex:                                      # noqa: BLE001
        rec["effort"]["tier_error"] = f"{type(ex).__name__}: {ex}"[:200]


# --------------------------------------------------------------- session ---

def _hdr(headers, name: str) -> str:
    if headers is None:
        return ""
    try:
        v = headers.get(name)
    except Exception:                                            # noqa: BLE001
        v = None
    return v.strip() if isinstance(v, str) else ""


def _session_of_user_id(uid) -> str:
    if not isinstance(uid, str) or not uid:
        return ""
    try:
        d = json.loads(uid)
        if isinstance(d, dict) and isinstance(d.get("session_id"), str):
            return d["session_id"]
    except ValueError:
        pass
    m = _LEGACY_USER_SESSION.search(uid)
    return m.group(1) if m else ""


def _session(body: dict, headers, rec: dict) -> str:
    """The prompt_cache_key for this request, or "" (see SESSIONS)."""
    if _hdr(headers, "x-yamadori-session"):
        rec["session_source"] = "x-yamadori-session (ours; wins)"
        return ""
    sid = _hdr(headers, "x-claude-code-session-id")
    src = "x-claude-code-session-id"
    if not sid:
        md = body.get("metadata") if isinstance(body.get("metadata"),
                                                dict) else {}
        sid = _session_of_user_id(md.get("user_id"))
        src = "metadata.user_id"
    if not sid or not _SESSION_OK.fullmatch(sid):
        rec["session_source"] = "none: our id in the tool_use ids"
        return ""
    agent = _hdr(headers, "x-claude-code-agent-id")
    key = f"claude-code:{sid}"
    if agent and _SESSION_OK.fullmatch(agent):
        key += f":{agent}"
        src += " + x-claude-code-agent-id"
    rec["session_source"] = src
    return key


# --------------------------------------------------------------- request ---

class Ctx:
    """What the translation back needs."""

    def __init__(self, body: dict, public_name: str, account: str):
        self.id = _new("msg")
        self.model = public_name
        self.account = account or ""
        self.body = body
        self.rec: dict = {}
        th = body.get("thinking") if isinstance(body.get("thinking"),
                                                dict) else {}
        self.thinking_off = th.get("type") in THINKING_OFF
        self.omit_thinking = th.get("display") == "omitted"
        self.stream = bool(body.get("stream"))


def _features(extra: dict) -> str:
    return json.dumps(extra)


def to_chat(body, headers=None, account: str = "",
            public_name: str = "yamadori", *, count_only: bool = False
            ) -> tuple[dict, Ctx]:
    """A Messages request -> (the chat body `_run_turn` takes, Ctx).
    Raises api_errors.ApiError (400) for what cannot be served."""
    if not isinstance(body, dict):
        raise _err("The request body must be a JSON object.", None,
                   "invalid_body")
    ctx = Ctx(body, body.get("model") or public_name, account)
    rec = ctx.rec
    ver = _hdr(headers, "anthropic-version")
    beta = _hdr(headers, "anthropic-beta")
    if ver:
        rec["anthropic_version"] = ver
    if beta:
        rec["anthropic_beta"] = [b.strip() for b in beta.split(",")
                                 if b.strip()]
    if not count_only:
        mt = body.get("max_tokens")
        if mt is None:
            raise _err("max_tokens: Field required.", "max_tokens",
                       "missing_required_parameter")
        if isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
            raise _err("max_tokens: must be a positive integer.",
                       "max_tokens")
    cc = json.dumps(body.get("system")).count('"cache_control"') + \
        json.dumps(body.get("messages")).count('"cache_control"')
    tools = _tools(body, rec)
    if cc:
        rec["cache_control"] = rec.get("cache_control", 0) + cc
    msgs = _messages(body, rec, account)
    chat: dict = {"model": body.get("model") or public_name,
                  "messages": msgs, "stream": bool(body.get("stream"))}
    tc, par, tools = _tool_choice(body.get("tool_choice"), tools, rec)
    if tools:
        chat["tools"] = tools
        if tc is not None:
            chat["tool_choice"] = tc
        if par is not None:
            chat["parallel_tool_calls"] = par
    eff = _effort(body, rec)
    if eff:
        chat["reasoning_effort"] = eff
    if not count_only:
        chat["max_tokens"] = body["max_tokens"]
    stops = body.get("stop_sequences")
    if stops is not None:
        if not isinstance(stops, list) or not all(isinstance(s, str)
                                                  for s in stops):
            raise _err("stop_sequences: must be an array of strings.",
                       "stop_sequences", "invalid_type")
        if stops:
            chat["stop"] = list(stops)
            rec["stop_sequences"] = len(stops)
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            chat[k] = body[k]
    fmt = (body.get("output_config") or {}).get("format") if isinstance(
        body.get("output_config"), dict) else None
    if isinstance(fmt, dict):
        if fmt.get("type") == "json_schema" and isinstance(fmt.get("schema"),
                                                          dict):
            chat["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "output", "schema": fmt["schema"]}}
        else:
            raise _err(f"output_config.format of type "
                       f"{json.dumps(fmt.get('type'))} is not served; send "
                       f"json_schema with a schema.", "output_config.format")
    ignored = [k for k in IGNORED_FIELDS if body.get(k) not in (None, [], {})]
    if ignored:
        rec["fields_ignored"] = ignored
    key = _session(body, headers, rec)
    if key:
        chat["prompt_cache_key"] = key
    hints = {h[len("x-claude-code-"):]: _hdr(headers, h) for h in HINT_HEADERS
             if _hdr(headers, h)}
    if _hdr(headers, "x-claude-code-agent-id"):
        hints["agent-id"] = "present"
    if hints:
        rec["hints"] = hints
    if hints.get("request-class") == "auxiliary" and not tools:
        # Claude Code's own side request (a title, a classifier, a
        # summary): the bare model, as every client side call gets it.
        chat["_features"] = _features({"utility": True})
        rec["utility_from_hint"] = True
    if chat["stream"]:
        chat["stream_options"] = {"include_usage": True}
    if rec.get("tools_ignored"):
        rec["tools_ignored"] = sorted(set(rec["tools_ignored"]))
    _record_tier(chat, rec, _hdr(headers, "x-yamadori-features"))
    return chat, ctx


def merge_features(ours, header) -> str | None:
    """The translation's X-Yamadori-Features and the caller's header, as one
    header value: the caller's keys win."""
    def as_dict(v):
        try:
            d = json.loads(v) if isinstance(v, str) else v
        except ValueError:
            return {}
        return d if isinstance(d, dict) else {}
    a, b = as_dict(ours), as_dict(header)
    if not a:
        return header
    return json.dumps(dict(a, **b))


# ---------------------------------------------------------------- output ---

def usage_of(u: dict | None) -> dict:
    """Chat usage -> Anthropic usage. input_tokens EXCLUDES the reused cache
    (Anthropic: total input = input_tokens + cache_read_input_tokens +
    cache_creation_input_tokens), so a client summing the three gets the
    prompt, as Claude Code does for its context meter."""
    u = u if isinstance(u, dict) else {}
    prompt = int(u.get("prompt_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens")
                 or 0)
    cached = min(cached, prompt)
    return {"input_tokens": prompt - cached,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached,
            "output_tokens": int(u.get("completion_tokens") or 0)}


def _input_of(args) -> tuple[dict, bool]:
    """A chat call's arguments as the tool_use input object: (input, ok).
    Arguments that are not a JSON object are carried whole under
    `_unparsed_arguments` (the client's schema check then tells the model)."""
    if isinstance(args, dict):
        return args, True
    try:
        v = json.loads(args) if isinstance(args, str) and args.strip() else {}
    except ValueError:
        v = None
    if isinstance(v, dict):
        return v, True
    return {"_unparsed_arguments": args}, False


def _tool_use(c: dict, rec: dict) -> dict:
    fn = c.get("function") or {}
    inp, ok = _input_of(fn.get("arguments"))
    if not ok:
        rec["tool_input_unparsed"] = rec.get("tool_input_unparsed", 0) + 1
    return {"type": "tool_use", "id": c.get("id") or _new("toolu"),
            "name": fn.get("name") or "", "input": inp}


def _stop_reason(finish: str | None, has_calls: bool) -> str:
    if has_calls:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "end_turn"


DROPPED = api_errors.ApiError(
    502, "The model server's connection dropped mid-answer. Send the request "
         "again.", code="upstream_dropped")


def _thinking_block(text: str, ctx: Ctx) -> dict | None:
    if ctx.thinking_off or not (text or "").strip():
        return None
    shown = "" if ctx.omit_thinking else text
    return {"type": "thinking", "thinking": shown,
            "signature": sign_thinking(shown, ctx.account)}


def of_chat(d: dict, ctx: Ctx) -> dict:
    """A blocking chat completion -> the Message object. Raises ApiError for
    a turn that did not complete (the model server dropped mid-answer)."""
    ch = (d.get("choices") or [{}])[0]
    finish = ch.get("finish_reason")
    if finish == "incomplete":
        raise DROPPED
    m = ch.get("message") or {}
    content: list = []
    tb = _thinking_block(m.get("reasoning_content") or "", ctx)
    if tb:
        content.append(tb)
    text = m.get("content") or ""
    if text:
        content.append({"type": "text", "text": text})
    calls = m.get("tool_calls") or []
    for c in calls:
        content.append(_tool_use(c, ctx.rec))
    if ctx.rec.get("stop_sequences"):
        ctx.rec["stop_sequence_reported"] = (
            "never: the model server's chat route does not say which stop "
            "ended the turn")
    x = d.get("x_yamadori") or {}
    return {"id": ctx.id, "type": "message", "role": "assistant",
            "model": ctx.model, "content": content,
            "stop_reason": _stop_reason(finish, bool(calls)),
            "stop_sequence": None, "usage": usage_of(d.get("usage")),
            "x_yamadori": dict(x, messages=ctx.rec)}


# ---------------------------------------------------------------- stream ---

def _sse(event: str, data: dict) -> bytes:
    return (b"event: " + event.encode() + b"\ndata: "
            + json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n\n")


class Stream:
    """proxy.stream_body's chat chunks -> Anthropic's event sequence.

      message_start                                    on the first chunk
      thinking: content_block_start {thinking, signature ""}, thinking_delta
        per chunk, then signature_delta and content_block_stop when the
        answer starts (after content has started, reasoning is a ping:
        CHANNEL ORDER, as the chat stream does)
      text: content_block_start {text ""}, text_delta per chunk, stop
      calls: per call content_block_start {tool_use, input {}}, ONE
        input_json_delta (the whole input: calls are checked whole), stop
      heartbeat: ping, at most every KEEPALIVE_S
      the end: message_delta {stop_reason, stop_sequence, usage, x_yamadori}
        and message_stop; a mid-stream failure: `error`, nothing after it.
    """

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.started = False
        self.done = False
        self.index = -1
        self.block: dict | None = None      # {kind, index, text}
        self.text_started = False
        self.calls = 0
        self.finish: str | None = None
        self.x: dict = {}
        self.usage: dict | None = None
        self.last_event = 0.0

    def _ev(self, out: list, event: str, **data) -> None:
        data = dict({"type": event}, **data)
        out.append(_sse(event, data))
        self.last_event = time.time()

    def _begin(self, out: list) -> None:
        self.started = True
        self._ev(out, "message_start", message={
            "id": self.ctx.id, "type": "message", "role": "assistant",
            "content": [], "model": self.ctx.model, "stop_reason": None,
            "stop_sequence": None, "usage": usage_of(None)})

    def _open(self, out: list, kind: str, block: dict) -> None:
        self.index += 1
        self.block = {"kind": kind, "index": self.index, "text": ""}
        self._ev(out, "content_block_start", index=self.index,
                 content_block=block)

    def _close(self, out: list) -> None:
        b = self.block
        if b is None:
            return
        self.block = None
        if b["kind"] == "thinking":
            shown = "" if self.ctx.omit_thinking else b["text"]
            self._ev(out, "content_block_delta", index=b["index"],
                     delta={"type": "signature_delta",
                            "signature": sign_thinking(shown,
                                                       self.ctx.account)})
        self._ev(out, "content_block_stop", index=b["index"])

    def ping(self, out: list | None = None) -> bytes:
        o = [] if out is None else out
        self._ev(o, "ping")
        return b"".join(o) if out is None else b""

    def _keepalive(self, out: list) -> None:
        if time.time() - self.last_event >= KEEPALIVE_S:
            self._ev(out, "ping")

    def _reason(self, out: list, text: str) -> None:
        if self.ctx.thinking_off or self.text_started or self.calls:
            self._keepalive(out)
            return
        if self.block is None or self.block["kind"] != "thinking":
            self._close(out)
            self._open(out, "thinking", {"type": "thinking", "thinking": "",
                                         "signature": ""})
        self.block["text"] += text
        if self.ctx.omit_thinking:
            self._keepalive(out)
            return
        self._ev(out, "content_block_delta", index=self.block["index"],
                 delta={"type": "thinking_delta", "thinking": text})

    def _text(self, out: list, text: str) -> None:
        if self.block is None or self.block["kind"] != "text":
            self._close(out)
            self._open(out, "text", {"type": "text", "text": ""})
            self.text_started = True
        self.block["text"] += text
        self._ev(out, "content_block_delta", index=self.block["index"],
                 delta={"type": "text_delta", "text": text})

    def _calls(self, out: list, calls: list) -> None:
        self._close(out)
        for c in calls:
            tu = _tool_use(c, self.ctx.rec)
            self._open(out, "tool_use", {"type": "tool_use", "id": tu["id"],
                                         "name": tu["name"], "input": {}})
            self._ev(out, "content_block_delta", index=self.index,
                     delta={"type": "input_json_delta",
                            "partial_json": json.dumps(tu["input"],
                                                       ensure_ascii=False)})
            self._ev(out, "content_block_stop", index=self.index)
            self.block = None
            self.calls += 1

    def _terminal(self, out: list) -> None:
        if self.finish == "incomplete":
            self._close(out)
            self._failed(out, DROPPED)
            return
        self._close(out)
        if self.ctx.rec.get("stop_sequences"):
            self.ctx.rec["stop_sequence_reported"] = (
                "never: the model server's chat route does not say which "
                "stop ended the turn")
        self._ev(out, "message_delta",
                 delta={"stop_reason": _stop_reason(self.finish,
                                                    bool(self.calls)),
                        "stop_sequence": None},
                 usage=usage_of(self.usage),
                 x_yamadori=dict(self.x or {}, messages=self.ctx.rec))
        self._ev(out, "message_stop")
        self.done = True

    def _failed(self, out: list, e: api_errors.ApiError) -> None:
        out.append(sse_error(e))
        self.last_event = time.time()
        self.done = True

    def fail(self, e: BaseException) -> bytes:
        """An exception after the first byte that stream_body did not turn
        into its error event (it always does; this is the backstop)."""
        out: list = []
        if not self.started:
            self._begin(out)
        self._failed(out, api_errors.of_exception(e))
        return b"".join(out)

    def feed(self, raw: bytes) -> bytes:
        """One chat SSE chunk (as proxy.stream_body yields it) -> bytes."""
        out: list = []
        if self.done:
            return b""
        for line in raw.decode("utf-8", "replace").split("\n"):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not self.started:
                self._begin(out)
            if data == "[DONE]":
                if not self.done:
                    self._terminal(out)
                continue
            try:
                d = json.loads(data)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("error") and \
                    not d.get("choices"):
                err = d["error"] if isinstance(d["error"], dict) else {
                    "message": str(d["error"])}
                self._failed(out, api_errors.ApiError(
                    _status_of_error(err), str(err.get("message") or "error"),
                    code=err.get("code"),
                    extra={k: v for k, v in err.items() if k in (
                        "n_prompt_tokens", "context_length", "floor")}))
                break
            if d.get("x_yamadori") is not None:
                self.x = d["x_yamadori"]
            if d.get("usage"):
                self.usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
                if delta.get("reasoning_content"):
                    self._reason(out, delta["reasoning_content"])
                if delta.get("content"):
                    self._text(out, delta["content"])
                if delta.get("tool_calls"):
                    self._calls(out, delta["tool_calls"])
                if not delta and not ch.get("finish_reason"):
                    self._keepalive(out)
        return b"".join(out)


def _status_of_error(err: dict) -> int:
    """The status an in-stream chat error object stands for (its type is
    OpenAI's; api_errors.type_of maps statuses to those types)."""
    code, kind = err.get("code"), err.get("type")
    if code == "context_length_exceeded" or kind == "invalid_request_error":
        return 400
    if kind == "rate_limit_error":
        return 429
    if kind == "service_unavailable_error" or code in (
            "model_unavailable", "model_at_capacity"):
        return 503
    return 502


_END = object()


def stream(chat_gen, ctx: Ctx, ping_s: float | None = None):
    """The Messages stream over `proxy.stream_body(...)`.

    The FIRST chunk is pulled in the caller's thread, so a failure before
    the turn starts still RAISES here and server.py answers it with a real
    HTTP status (E1). After it a pump thread reads the rest into a queue,
    and a `ping` goes out whenever the turn has sent nothing for `ping_s`
    (Claude Code aborts a stream silent for 300 s). Closing this closes
    stream_body (the turn's cancel)."""
    if ping_s is None:
        try:
            import proxy
            ping_s = float(getattr(proxy, "HEARTBEAT", 5.0))
        except Exception:                                        # noqa: BLE001
            ping_s = 5.0
    s = Stream(ctx)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def _close_gen():
        try:
            close = getattr(chat_gen, "close", None)
            if close is not None:
                close()
        except ValueError:
            pass            # executing in another thread; its token is cancelled

    def pump():
        try:
            for raw in chat_gen:
                q.put(("chunk", raw))
                if stop.is_set():
                    break
            q.put(("end", None))
        except BaseException as e:                               # noqa: BLE001
            q.put(("error", e))
        finally:
            _close_gen()

    th = None
    try:
        first = next(chat_gen, _END)        # raises before the first byte (E1)
        if first is _END:
            b = s.feed(b"data: [DONE]\n\n")
            if b:
                yield b
            return
        b = s.feed(first)
        if b:
            yield b
        if s.done:
            return
        th = threading.Thread(target=pump, name="messages-stream-pump",
                              daemon=True)
        th.start()
        while True:
            try:
                kind, item = q.get(timeout=ping_s)
            except queue.Empty:
                yield s.ping()
                continue
            if kind == "chunk":
                b = s.feed(item)
                if b:
                    yield b
                if s.done:
                    break
            elif kind == "end":
                if not s.done:
                    b = s.feed(b"data: [DONE]\n\n")
                    if b:
                        yield b
                break
            else:
                if isinstance(item, GeneratorExit):
                    break
                yield s.fail(item)
                break
    except GeneratorExit:
        raise
    except Exception as e:                                       # noqa: BLE001
        if not s.started:
            raise
        yield s.fail(e)
    finally:
        stop.set()
        if th is None:
            _close_gen()


# ----------------------------------------------------------- count_tokens --

def count_tokens(body, headers=None, account: str = "") -> dict:
    """POST /v1/messages/count_tokens: {input_tokens} -- the model server's
    own count (the served template's render of the client's messages and
    tools, tokenized by the model: proxy.count_prompt_tokens, the counted
    path of the window check), on the model that serves the request's tier.
    What the proxy adds on the turn itself (restored reasoning, injections,
    our tools) is not counted: counting must not decide or record anything.
    When that model is not the one on the card (it would be loaded to
    count), or the server cannot count, the window check's high estimate
    answers, and x_yamadori says so."""
    import proxy
    chat, ctx = to_chat(body, headers, account, count_only=True)
    tier = ctx.rec.get("effort", {}).get("tier")
    model = ctx.rec.get("effort", {}).get("serves") or "bonsai"
    payload = {"model": model, "messages": chat["messages"]}
    if chat.get("tools"):
        payload["tools"] = chat["tools"]
    x: dict = {"tier": tier, "model": model, "counted": False,
               "messages": ctx.rec}
    n = None
    try:
        import max_mode
        on_card = (not max_mode.ENABLED) or max_mode.card_model() in (
            None, model)
    except Exception:                                            # noqa: BLE001
        on_card = True
    if on_card:
        try:
            n = proxy.count_prompt_tokens(payload)
        except Exception as e:                                   # noqa: BLE001
            x["count_error"] = f"{type(e).__name__}: {e}"[:200]
    else:
        x["why"] = (f"{model} is not on the card; counting would load it: the "
                    f"high estimate instead")
    if n is None:
        n = proxy.high_estimate(payload)
        x.setdefault("why", "the model server did not count it: the high "
                            "estimate (chars / 3)")
    else:
        x["counted"] = True
    return {"input_tokens": int(n), "x_yamadori": x}
