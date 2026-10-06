#!/usr/bin/env python
"""The OpenAI Responses API (`POST /v1/responses`), as a TRANSLATION LAYER.

WHY (operator, 2026-09-26: "If everyone supports responses api... both need
to exist." docs/OPENAI-CONFORMANCE.md R1, section 10). Codex CLI speaks only
Responses (`wire_api = "chat"` was removed), Hermes has a `codex_responses`
transport, and the SDKs are moving there. Chat Completions stays.

WHAT IT IS, AND WHAT IT IS NOT. Not a second pipeline. A Responses request is
translated into the chat body `proxy._run_turn` already takes (`to_chat`), the
SAME turn runs -- tiers, caps, skills, deep thinking, the code check and
repair, the image guard, sessions, the ledger, warms, usage -- and what it
produced is translated back: a Response object on the blocking path
(`of_chat`), the Responses SSE event sequence on the streamed one (`Stream`,
which reads `proxy.stream_body`'s own chunks, so every stream promise --
CHANNEL ORDER, IMAGES REACH THE CHAT, the E1 commit point, cancel on
disconnect -- is the chat path's, unchanged).

STORED STATE (operator, 2026-10-06, AskUserQuestion, chosen "Store, local,
capped": "Save each response's full input and output under its id, per account,
on this machine only. Same size and age limits as the ledger (256 MB per
account, 2 GB total, 30 days). previous_response_id rebuilds the conversation
from it, and GET and DELETE /v1/responses/{id} work like OpenAI's. Applies when
the client asks to store (OpenAI's default). This ends the rule that client
messages are only hashed, for Responses requests."). Until then the module was
stateless and refused previous_response_id -- a production trap, since OpenAI's
`store` defaults to true and a client may send only the new input plus
previous_response_id.

  The store is mcp/response_store.py (schema, caps, eviction). Here:
  - `store` absent or true (OpenAI's default) -> the response is stored when it
    ends (`persist`: blocking, before the body is returned; streamed, before
    the terminal event goes out, so a client that chains the moment it sees
    `response.completed` finds it). `store: false` -> nothing is kept, as
    before. The Response's `store` field reports the truth.
  - `previous_response_id` -> the chain's stored input and output items come
    FIRST, this request's `input` after them, and the same translation runs on
    the whole list: the messages are exactly those of a client that resent its
    history (`_messages(prefix=)`), so the ledger's chain keys, the sessions
    and the slot's prompt cache work as they do for a full resend. Not carried
    over: `instructions` (OpenAI: "When using along with previous_response_id,
    the instructions from a previous response will not be carried over to the
    next response") and `tools` -- both are the current request's. An id that
    is not this account's, was stored with `store: false`, was deleted or
    evicted is 400 `previous_response_not_found`, param `previous_response_id`,
    "Previous response with id '<id>' not found." (OpenAI's status, type,
    code and message: reported by clients of the live API, e.g.
    github.com/dotnet/extensions/issues/7704,
    github.com/microsoft/semantic-kernel/issues/13128; the platform docs
    (developers.openai.com/api/docs/guides/conversation-state) do not print
    the body). A continuation that sends no prompt_cache_key keeps the one the
    previous response was sent with (an explicit id: the chain).
  - Still refused (400 `unsupported_parameter`): `conversation` (needs the
    Conversations API), `prompt` (templates stored at OpenAI), `background:
    true`, and `item_reference` items: resolving one needs an index of every
    stored item id and its own ordering rules, and no harness we serve
    (Codex, Hermes, Pi, OpenCode) sends one -- previous_response_id covers the
    stored-state need. `store: true` with no account or with
    YAMADORI_RESPONSE_STORE=0 is answered `store: false`, and a
    previous_response_id is then refused.
  - A streamed request whose client hangs up before the terminal event is not
    stored (its `response.created` said it would be); a `failed` or
    `incomplete` response is stored like any other (GET shows it; a client
    that chains from it gets what it was sent).

THE MAPPING (request -> chat)

  instructions                  the leading system message, joined (blank
                                line) with any system/developer items that
                                come before the first other item: the served
                                template takes ONE system message, first
  a later system/developer item a user message (the template refuses a
                                system message anywhere but first); counted
                                in x_yamadori.responses.developer_as_user
  input: a string               one user message
  message items                 chat messages; parts via `chat_part_of`
                                (input_text/output_text -> text, input_image
                                -> image_url, a file_id image kept as the
                                Responses part, which vision.extract names
                                and explains). All-text content becomes ONE
                                string ("".join -- the template renders a
                                text-part list exactly so)
  consecutive assistant items   ONE assistant message: reasoning, text and
                                function_call items of one turn merge, as
                                chat has them (deterministic, or the
                                ledger's chain keys change between turns)
  function_call / custom_tool_call   tool_calls on that assistant message,
                                `call_id` as the id (our session carrier
                                rides in it when the client names no session)
  function_call_output /        `tool` messages; an ARRAY output keeps its
  custom_tool_call_output       images (Codex's view_image) as image_url parts
  reasoning items               PASS-THROUGH: `content[].reasoning_text` is
                                the next assistant message's
                                reasoning_content (the client echoes what the
                                model wrote, and the model reads its echo).
                                `summary[]` is display text and
                                `encrypted_content` is opaque: neither is
                                given to the model; both are counted
  function tools (flat)         the nested chat form
  custom tools (freeform)       a function with one string parameter
                                `input`; a call comes back as a
                                `custom_tool_call` item
  image_generation (hosted)     our image capability for this request
                                (yama_generate_image, mcp/images.py): its size
                                is the request's (`_image_options`), a call
                                comes back as an `image_generation_call`
                                item. Unsupported values: 400
  other hosted tools            web_search, file_search, code_interpreter,
                                computer_use, mcp, tool_search: accepted and
                                IGNORED (the model never sees them; Codex
                                offers web_search by default and a 400 would
                                end every Codex session), listed in
                                x_yamadori.responses.tools_ignored
  namespace tools               FLATTENED into functions (Codex sends
                                `multi_agent_v1` on every request); a call
                                comes back with `namespace` set, and a
                                replayed one renders as the flat name
  client-run built-ins we cannot render (local_shell, shell, apply_patch,
  anything unknown)             400 naming the tool type
  tool_choice                   the chat form; a hosted type -> "auto"
  reasoning.effort              reasoning_effort (the tier dial)
  max_output_tokens             max_tokens
  text.format                   response_format (json_schema, json_object)
  prompt_cache_key              kept: the conversation id (session source 1)
  metadata, user, store, include, service_tier, truncation, safety_identifier,
  client_metadata, stream_options, text.verbosity, reasoning.summary,
  max_tool_calls                accepted and ignored (metadata echoed)

THE OUTPUT (Response items, in order): a reasoning item (when the turn
reasoned; see REASONING_MODE), any image_generation_call items (hosted tool
declared), a message item with the answer, then one function_call (or
custom_tool_call) item per client call. `usage` maps the chat usage (U1/U2:
the final generation's). `x_yamadori` is the chat one plus `responses`.

REASONING_MODE (YAMADORI_RESPONSES_REASONING; a CHOICE, the operator's to
change): how the turn's reasoning is shown.
  summary (default)  as `summary: [{summary_text}]` and the
                     reasoning_summary_* events: what Codex and Hermes
                     DISPLAY by default. Codex echoes summaries, which are
                     display text (above), so the model does not read its
                     past reasoning back -- the reference run's form (Hermes
                     over chat strips it; AGENTS.md "The ledger").
  content            as `content: [{reasoning_text}]` and reasoning_text
                     events: Codex then ECHOES it (it serializes reasoning
                     text), and the model reads it back (6-10k tokens of
                     context per step in the V0 pilot).
  off                no reasoning item; the wait is kept alive (below).

KEEP-ALIVE. A Responses consumer's idle timer resets only on a PARSED event:
Hermes' codex transport gives up after 12 s between events on a small
context (HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS, codex_runtime.py), Codex
after 300 s (stream_idle_timeout_ms). SSE comment lines are dropped by the
decoders (openai-python _streaming.SSEDecoder, eventsource-stream) and reset
neither. So a chat heartbeat (an empty delta while the second brain works,
or reasoning suppressed after content) becomes `response.in_progress`,
carrying a compact snapshot (identity and status; the SDKs construct events
without validating them, Codex and Hermes ignore its body), at most once
per KEEPALIVE_S.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.parse
import uuid

import api_errors
import response_store
import system_roles

REASONING_MODE = (os.environ.get("YAMADORI_RESPONSES_REASONING") or
                  "summary").strip().lower()
if REASONING_MODE not in ("summary", "content", "off"):
    REASONING_MODE = "summary"
KEEPALIVE_S = 1.0

# Hosted tools the SERVER would run at OpenAI. Nothing here runs them: the
# model never sees them. `image_generation` is served (below).
HOSTED_IGNORED = frozenset({
    "web_search", "web_search_preview", "web_search_preview_2025_03_11",
    "web_search_2025_08_26", "file_search", "code_interpreter",
    "computer_use_preview", "computer_use", "computer", "mcp",
    "tool_search"})
IMAGE_TOOL = "image_generation"

# Refused request fields: each needs a feature this server lacks.
# (previous_response_id is served by the response store since 2026-10-06.)
_STATEFUL = {
    "conversation": "this server has no Conversations API; chain responses "
                    "with previous_response_id, or send the whole "
                    "conversation in `input`",
    "prompt": "prompt templates are stored server-side at OpenAI; send "
              "`instructions` and `input` instead",
}

# The hosted image tool's values we can serve (docs: the ImageGenTool
# schema; our sizes are images.parse_size's). quality is accepted and does
# not pick the image model: the caller's saved preference does (operator,
# 2026-09-24: turbo default), recorded in x_yamadori.images.
IMAGE_QUALITY = ("low", "medium", "high", "xhigh", "max", "auto")
IMAGE_ACTIONS = ("generate", "auto")


def _err(message: str, param: str | None, code: str = "invalid_value"
         ) -> api_errors.ApiError:
    return api_errors.invalid(message, param=param, code=code)


def _new(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# ---------------------------------------------------------------- parts ----

def chat_part_of(part, where: str, *, in_tool: bool = False) -> dict:
    """ONE Responses content part -> the chat content part the turn reads.

    The single place a Responses part becomes a chat part, for messages and
    for function_call_output arrays alike. Images go to the chat image path
    (vision.extract: placeholders naming an id, yama_describe_image), never
    fetched: an http(s) URL stays a URL, which extract refuses to download.
    The vision build extends image handling inside vision.extract, which
    reads the parts this returns."""
    if not isinstance(part, dict):
        raise _err(f"{where} must be an object.", where, "invalid_type")
    t = part.get("type")
    if t in ("input_text", "output_text", "text", "summary_text",
             "reasoning_text"):
        text = part.get("text")
        if not isinstance(text, str):
            raise _err(f"{where}.text must be a string.", f"{where}.text",
                       "invalid_type")
        return {"type": "text", "text": text}
    if t == "refusal":
        return {"type": "text", "text": str(part.get("refusal") or "")}
    if t == "input_image":
        url = part.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(url, str) and url and in_tool:
            # A function_call_output's image (Codex's view_image) goes on as
            # the Responses part it is, so vision.normalise records its form
            # as `responses_output` (image_input.FORMS), not `tool_part`; the
            # placeholder's words are the same for both (2026-09-26,
            # mcp/test_harness_decisions.py codex/responses/view-image).
            keep = {"type": "input_image", "image_url": url}
            if part.get("detail"):
                keep["detail"] = part["detail"]
            return keep
        if isinstance(url, str) and url:
            iu = {"url": url}
            if part.get("detail"):
                iu["detail"] = part["detail"]
            return {"type": "image_url", "image_url": iu}
        if isinstance(part.get("file_id"), str):
            # No files API here: vision.extract names it and says so.
            keep = {"type": "input_image", "file_id": part["file_id"]}
            if part.get("detail"):
                keep["detail"] = part["detail"]
            return keep
        raise _err(f"{where} is an input_image with neither image_url nor "
                   f"file_id.", where, "invalid_content_part")
    if t == "input_file":
        if in_tool:
            # A tool produced it: refusing would wedge the conversation for
            # good (it is in every later request), so the model is told.
            name = part.get("filename") or part.get("file_id") or "a file"
            return {"type": "text", "text": (
                f"[a file ({name}) was returned here. This server passes "
                f"only text and images to the model, so its content is "
                f"unavailable; tell the user if it matters.]")}
        raise _err(f"{where}.type \"input_file\" cannot be read by this "
                   f"model; send text or images.", f"{where}.type",
                   "invalid_content_part")
    raise _err(f"{where}.type {json.dumps(t)} cannot be read by this model; "
               f"send input_text or input_image parts.", f"{where}.type",
               "invalid_content_part")


def _content_of(content, where: str, *, in_tool: bool = False):
    """A message's (or a tool output's) content as chat content: a string
    when it is all text (the template renders a text list as its "".join),
    else the list of chat parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _err(f"{where} must be a string or an array of content parts.",
                   where, "invalid_type")
    parts = [chat_part_of(p, f"{where}[{j}]", in_tool=in_tool)
             for j, p in enumerate(content)]
    if all(p.get("type") == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def _text_of(content) -> str:
    """Plain text of chat content (a system message or an assistant turn)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text") or "" for p in content or []
                   if isinstance(p, dict))


def _reasoning_text(item: dict) -> str:
    """What a reasoning item gives the model: its reasoning TEXT only."""
    out = []
    for c in item.get("content") or []:
        if isinstance(c, dict) and c.get("type") in ("reasoning_text", "text") \
                and isinstance(c.get("text"), str):
            out.append(c["text"])
    return "".join(out)


# ---------------------------------------------------------------- tools ----

def _custom_function(t: dict) -> dict:
    desc = str(t.get("description") or "")
    fmt = t.get("format") if isinstance(t.get("format"), dict) else {}
    note = "The input is free text"
    if fmt.get("type") == "grammar" and fmt.get("definition"):
        note += (f" in this {fmt.get('syntax') or 'grammar'} grammar:\n"
                 f"{fmt['definition']}")
    else:
        note += "."
    return {"type": "function", "function": {
        "name": t["name"],
        "description": (desc + "\n\n" if desc else "") + note,
        "parameters": {"type": "object", "properties": {"input": {
            "type": "string", "description": "The tool's whole input."}},
            "required": ["input"]}}}


def _image_options(t: dict, i: int) -> dict:
    """The hosted image tool's values, checked against what we can make."""
    import images
    p = f"tools[{i}]"
    size = t.get("size")
    if size not in (None, "", "auto"):
        try:
            images.parse_size(size)
        except images.ImageError as e:
            raise _err(f"{p}.size {json.dumps(size)}: {e.reason}",
                       f"{p}.size") from None
    q = t.get("quality")
    if q is not None and q not in IMAGE_QUALITY:
        raise _err(f"{p}.quality must be one of {', '.join(IMAGE_QUALITY)}.",
                   f"{p}.quality")
    bad = images.unsupported_option(t)
    if bad:
        raise _err(f"{p}.{bad[0]}: {bad[1]}", f"{p}.{bad[0]}")
    act = t.get("action")
    if act is not None and act not in IMAGE_ACTIONS:
        raise _err(f"{p}.action {json.dumps(act)} is not served: this server "
                   f"generates new images and does not edit them.",
                   f"{p}.action")
    if t.get("input_image_mask"):
        raise _err(f"{p}.input_image_mask is not served: this server does not "
                   f"edit images.", f"{p}.input_image_mask")
    return {"size": None if size in (None, "", "auto") else size,
            "quality": q, "partial_images": t.get("partial_images"),
            "model_requested": t.get("model")}


def _function_tool(t: dict, where: str, name: str | None = None,
                   note: str = "") -> dict:
    name = name or t.get("name")
    if not isinstance(t.get("name"), str) or not t.get("name"):
        raise _err(f"{where}.name is required for a function tool.",
                   f"{where}.name", "missing_required_parameter")
    fn = {"name": name}
    for k in ("description", "parameters", "strict"):
        if t.get(k) is not None:
            fn[k] = t[k]
    if note:
        fn["description"] = (fn.get("description") or "") + note
    return {"type": "function", "function": fn}


def _tools(body: dict, rec: dict) -> tuple[list, set, dict | None, dict]:
    """(chat tools, custom tool names, hosted image options or None,
    {flat name: (namespace, name)} for namespaced functions).

    A `namespace` tool (Codex sends `multi_agent_v1` on every request;
    docs/HARNESS-RESPONSES.md) is FLATTENED: each of its functions is a chat
    function under its own name -- or `<namespace>__<name>` when that name
    is taken -- and a call of it comes back with `namespace` set, since
    Codex routes by (namespace, name)."""
    tools = body.get("tools")
    if tools is None:
        return [], set(), None, {}
    if not isinstance(tools, list):
        raise _err("'tools' must be an array.", "tools", "invalid_type")
    taken = {t.get("name") for t in tools if isinstance(t, dict)
             and t.get("type") in ("function", "custom")}
    out, custom, image, ns_of = [], set(), None, {}
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            raise _err(f"tools[{i}] must be an object.", f"tools[{i}]",
                       "invalid_type")
        kind = t.get("type")
        if kind == "function":
            out.append(_function_tool(t, f"tools[{i}]"))
        elif kind == "namespace":
            ns = t.get("name")
            if not isinstance(ns, str) or not ns:
                raise _err(f"tools[{i}].name is required for a namespace.",
                           f"tools[{i}].name", "missing_required_parameter")
            inner = t.get("tools")
            if not isinstance(inner, list):
                raise _err(f"tools[{i}].tools must be an array.",
                           f"tools[{i}].tools", "invalid_type")
            nd = t.get("description")
            note = (f"\n\n(Part of {ns}: {nd})" if isinstance(nd, str) and nd
                    else "")
            for j, f in enumerate(inner):
                where = f"tools[{i}].tools[{j}]"
                if not isinstance(f, dict) or f.get("type") not in (
                        "function", None):
                    raise _err(f"{where}: a namespace's tools must be "
                               f"functions.", f"{where}.type")
                name = f.get("name")
                flat = name if name not in taken else f"{ns}__{name}"
                taken.add(flat)
                out.append(_function_tool(f, where, flat, note))
                ns_of[flat] = (ns, name)
        elif kind == "custom":
            if not isinstance(t.get("name"), str) or not t.get("name"):
                raise _err(f"tools[{i}].name is required for a custom tool.",
                           f"tools[{i}].name", "missing_required_parameter")
            out.append(_custom_function(t))
            custom.add(t["name"])
        elif kind == IMAGE_TOOL:
            opts = _image_options(t, i)
            import images
            if images.configured():
                image = opts
            else:
                rec.setdefault("tools_ignored", []).append(
                    f"{IMAGE_TOOL} (no image server: YAMADORI_IMAGEGEN_URL "
                    f"is not set)")
        elif kind in HOSTED_IGNORED or (isinstance(kind, str) and (
                kind.startswith("web_search") or
                kind.startswith("computer_use"))):
            rec.setdefault("tools_ignored", []).append(kind)
        else:
            raise _err(f"tools[{i}] of type {json.dumps(kind)} cannot be "
                       f"served here: send function, custom or namespace "
                       f"tools (hosted tools such as web_search are accepted "
                       f"and not offered to the model; image_generation is "
                       f"served).", f"tools[{i}].type")
    if ns_of:
        rec["namespaces"] = sorted({v[0] for v in ns_of.values()})
    return out, custom, image, ns_of


def _tool_choice(tc, custom: set):
    if tc is None or isinstance(tc, str):
        if tc not in (None, "auto", "none", "required"):
            raise _err("tool_choice must be none, auto, required or a tool "
                       "object.", "tool_choice")
        return tc
    if not isinstance(tc, dict):
        raise _err("tool_choice must be a string or an object.",
                   "tool_choice", "invalid_type")
    kind = tc.get("type")
    if kind in ("function", "custom"):
        if not tc.get("name"):
            raise _err("tool_choice.name is required.", "tool_choice.name",
                       "missing_required_parameter")
        return {"type": "function", "function": {"name": tc["name"]}}
    if kind == IMAGE_TOOL or kind in HOSTED_IGNORED or (
            isinstance(kind, str) and kind.startswith("web_search")):
        return "auto"
    raise _err(f"tool_choice of type {json.dumps(kind)} cannot be served "
               f"here; send none, auto, required or {{\"type\": "
               f"\"function\", \"name\": ...}}.", "tool_choice")


# ---------------------------------------------------------------- input ----

def _call_of(item: dict, where: str, custom: bool,
             flat_of: dict | None = None) -> dict:
    cid = item.get("call_id")
    name = item.get("name")
    ns = item.get("namespace")
    if isinstance(ns, str) and ns and isinstance(name, str):
        # A namespaced call renders as the flat name the model was given.
        name = (flat_of or {}).get((ns, name), name)
    if not isinstance(cid, str) or not cid:
        raise _err(f"{where}.call_id is required.", f"{where}.call_id",
                   "missing_required_parameter")
    if not isinstance(name, str) or not name:
        raise _err(f"{where}.name is required.", f"{where}.name",
                   "missing_required_parameter")
    if custom:
        args = json.dumps({"input": item.get("input") or ""},
                          ensure_ascii=False)
    else:
        args = item.get("arguments")
        if args is None:
            args = "{}"
        elif not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def own_items(body: dict) -> list:
    """This request's own input items: `input` as a list (a string is one
    user message). Raises the 400 for a missing or mistyped `input`."""
    raw = body.get("input")
    if raw is None:
        raise _err("Missing required parameter: 'input'.", "input",
                   "missing_required_parameter")
    if isinstance(raw, str):
        return [{"type": "message", "role": "user", "content": raw}]
    if isinstance(raw, list):
        return raw
    raise _err("'input' must be a string or an array of input items.",
               "input", "invalid_type")


def _messages(body: dict, rec: dict, flat_of: dict | None = None,
              prefix: list | None = None) -> list[dict]:
    """The chat messages of `input`; `prefix` is a stored chain's items
    (previous_response_id), translated first, as if the client had sent them."""
    own = own_items(body)
    prefix = prefix or []
    items = prefix + own
    system: list[str] = []
    ins = body.get("instructions")
    if isinstance(ins, str):
        if ins:
            system.append(ins)
    elif isinstance(ins, list):
        # The typed form: a list of input messages (the spec allows it).
        for j, m in enumerate(ins):
            if isinstance(m, dict):
                system.append(_text_of(_content_of(
                    m.get("content"), f"instructions[{j}].content")))
    elif ins is not None:
        raise _err("'instructions' must be a string.", "instructions",
                   "invalid_type")

    msgs: list[dict] = []
    counts: dict[str, int] = {}
    ignored: list[str] = []
    state = {"open": None}

    def close():
        if state["open"] is not None:
            msgs.append(state["open"])
            state["open"] = None

    def opened() -> dict:
        if state["open"] is None:
            state["open"] = {"role": "assistant", "content": ""}
        return state["open"]

    for i, it in enumerate(items):
        where = (f"input[{i - len(prefix)}]" if i >= len(prefix)
                 else "previous_response_id")
        if not isinstance(it, dict):
            raise _err(f"{where} must be an object.", where, "invalid_type")
        kind = it.get("type") or ("message" if "role" in it else None)
        counts[str(kind)] = counts.get(str(kind), 0) + 1
        if kind == "message":
            role = it.get("role")
            if role == "assistant":
                text = _text_of(_content_of(it.get("content"),
                                            f"{where}.content"))
                a = state["open"]
                if a is not None and a.get("tool_calls"):
                    close()
                a = opened()
                a["content"] = (a.get("content") or "") + text
            elif role in ("system", "developer"):
                # Kept in place; system_roles.one_system (below) makes ONE
                # system message of the leading run and a user message of a
                # later one.
                close()
                msgs.append({"role": role, "content": _text_of(_content_of(
                    it.get("content"), f"{where}.content"))})
            elif role == "user":
                close()
                msgs.append({"role": "user", "content": _content_of(
                    it.get("content"), f"{where}.content")})
            else:
                raise _err(f"{where}.role must be user, assistant, system "
                           f"or developer; got {json.dumps(role)}.",
                           f"{where}.role")
        elif kind == "reasoning":
            text = _reasoning_text(it)
            if it.get("encrypted_content"):
                rec["encrypted_reasoning_ignored"] = rec.get(
                    "encrypted_reasoning_ignored", 0) + 1
            if any(isinstance(s, dict) and s.get("text")
                   for s in it.get("summary") or []):
                rec["reasoning_summaries_not_read"] = rec.get(
                    "reasoning_summaries_not_read", 0) + 1
            if not text:
                continue
            a = state["open"]
            if a is not None and (a.get("content") or a.get("tool_calls")):
                close()
            a = opened()
            a["reasoning_content"] = (a.get("reasoning_content") or "") + text
            rec["reasoning_restored"] = rec.get("reasoning_restored", 0) + 1
        elif kind in ("function_call", "custom_tool_call"):
            opened().setdefault("tool_calls", []).append(
                _call_of(it, where, kind == "custom_tool_call", flat_of))
        elif kind in ("function_call_output", "custom_tool_call_output"):
            close()
            cid = it.get("call_id")
            if not isinstance(cid, str) or not cid:
                raise _err(f"{where}.call_id is required.",
                           f"{where}.call_id", "missing_required_parameter")
            msgs.append({"role": "tool", "tool_call_id": cid,
                         "content": _content_of(it.get("output"),
                                                f"{where}.output",
                                                in_tool=True)})
        elif kind == "item_reference":
            raise api_errors.invalid(
                f"{where} is an item_reference: this server does not resolve "
                f"item references; send the item itself, or chain "
                f"responses with previous_response_id.",
                param=where, code="unsupported_parameter")
        else:
            # A hosted call's record (web_search_call, image_generation_call
            # -- Codex replays the image it was given --, compaction, ...):
            # nothing the model can read. It does not end the turn it sits in.
            ignored.append(str(kind))
    close()
    if ignored:
        rec["items_ignored"] = sorted(set(ignored))
    rec["items"] = counts
    if system:
        msgs.insert(0, {"role": "system", "content": "\n\n".join(system)})
    msgs, moved = system_roles.one_system(msgs)
    if moved.get("developer_as_user"):
        rec["developer_as_user"] = moved["developer_as_user"]
    if not any(m.get("role") != "system" for m in msgs):
        raise _err("'input' must contain at least one message or item.",
                   "input")
    return msgs


# ---------------------------------------------------------------- request --

class Ctx:
    """What the translation back needs: the echoes, the tool kinds, ids."""

    def __init__(self, body: dict, public_name: str):
        self.id = _new("resp")
        # Stored state (response_store): who it is kept for (None: nothing is
        # kept), whether THIS response will be, the response it chains from,
        # and this request's own input items, as sent.
        self.account: str | None = None
        self.store_on = False
        self.prev_id: str | None = None
        self.input_items: list = []
        self.cache_key: str | None = None   # the prompt_cache_key it ran with
        self.created_at = int(time.time())
        self.model = public_name
        self.body = body
        self.custom: set = set()
        self.ns_of: dict = {}           # flat name -> (namespace, name)
        self.image: dict | None = None
        self.rec: dict = {}

    def echo(self) -> dict:
        b = self.body
        reasoning = b.get("reasoning") if isinstance(b.get("reasoning"),
                                                     dict) else {}
        text = b.get("text") if isinstance(b.get("text"), dict) else {}
        return {
            "instructions": b.get("instructions"),
            "max_output_tokens": b.get("max_output_tokens"),
            "max_tool_calls": b.get("max_tool_calls"),
            "metadata": b.get("metadata") if isinstance(b.get("metadata"),
                                                        dict) else {},
            "parallel_tool_calls": b.get("parallel_tool_calls", True),
            "previous_response_id": self.prev_id,
            "prompt_cache_key": self.cache_key or b.get("prompt_cache_key"),
            "reasoning": {"effort": reasoning.get("effort"),
                          "summary": reasoning.get("summary")},
            "safety_identifier": b.get("safety_identifier"),
            "service_tier": "default",
            "store": self.store_on,
            "temperature": b.get("temperature"),
            "text": {"format": text.get("format") or {"type": "text"},
                     "verbosity": text.get("verbosity")},
            "tool_choice": b.get("tool_choice") or "auto",
            "tools": b.get("tools") if isinstance(b.get("tools"),
                                                  list) else [],
            "top_logprobs": b.get("top_logprobs") or 0,
            "top_p": b.get("top_p"),
            "truncation": b.get("truncation") or "disabled",
            "user": b.get("user"),
            "access_programs": None,
        }


def previous_not_found(rid: str, broken: bool = False) -> api_errors.ApiError:
    """OpenAI's error for a previous_response_id that names nothing: 400,
    invalid_request_error, param previous_response_id, code
    previous_response_not_found, "Previous response with id '<id>' not
    found." (see the module doc for the sources)."""
    msg = f"Previous response with id '{rid}' not found."
    if broken:
        msg += " An earlier response in its chain was deleted or evicted."
    return api_errors.invalid(msg, param="previous_response_id",
                              code="previous_response_not_found")


def to_chat(body, public_name: str = "yamadori", account: str | None = None
            ) -> tuple[dict, Ctx]:
    """A Responses request -> (the chat body `_run_turn` takes, Ctx).
    `account` is who the response is stored for and whose stored responses
    `previous_response_id` may name (None: no store, as before 2026-10-06).
    Raises api_errors.ApiError (400) for what cannot be served."""
    if not isinstance(body, dict):
        raise _err("The request body must be a JSON object.", None,
                   "invalid_body")
    for k, why in _STATEFUL.items():
        if body.get(k) not in (None, "", {}, []):
            raise api_errors.invalid(f"'{k}' is not supported: {why}.",
                                     param=k, code="unsupported_parameter")
    if body.get("background"):
        raise api_errors.invalid(
            "'background' is not supported: this server answers in the "
            "request (send stream: true for progress).", param="background",
            code="unsupported_parameter")
    if body.get("top_logprobs"):
        raise _err("Log probabilities cannot be returned: the server streams "
                   "from the model with tools attached, where the model "
                   "server refuses them. Leave out top_logprobs.",
                   "top_logprobs")
    store = body.get("store")
    if store is not None and not isinstance(store, bool):
        raise _err("'store' must be a boolean.", "store", "invalid_type")
    ctx = Ctx(body, body.get("model") or public_name)
    rec = ctx.rec
    available = bool(account) and response_store.enabled()
    # OpenAI's default is store: true.
    ctx.store_on = available and store is not False
    if available:
        ctx.account = account
    elif store:
        rec["store_requested"] = True
    prev = body.get("previous_response_id")
    history = None
    if prev not in (None, ""):
        if not isinstance(prev, str):
            raise _err("'previous_response_id' must be a string.",
                       "previous_response_id", "invalid_type")
        if not available:
            raise api_errors.invalid(
                "'previous_response_id' is not supported: this server is not "
                "storing responses (YAMADORI_RESPONSE_STORE=0); send the "
                "whole conversation in `input`.",
                param="previous_response_id", code="unsupported_parameter")
        try:
            history = response_store.history(account, prev)
        except response_store.NotFound as e:
            raise previous_not_found(e.rid, e.broken) from None
        ctx.prev_id = prev
        rec["previous_response"] = {"id": prev, "chain": history["chain"],
                                    "items": len(history["items"])}
    ctx.input_items = own_items(body)
    tools, custom, image, ns_of = _tools(body, rec)
    ctx.custom, ctx.image, ctx.ns_of = custom, image, ns_of
    msgs = _messages(body, rec, {v: k for k, v in ns_of.items()},
                     prefix=history["items"] if history else None)
    chat: dict = {"model": body.get("model") or public_name,
                  "messages": msgs, "stream": bool(body.get("stream"))}
    if tools:
        chat["tools"] = tools
    tc = _tool_choice(body.get("tool_choice"), custom)
    if tc is not None and tools:
        chat["tool_choice"] = tc
    if isinstance(body.get("parallel_tool_calls"), bool) and tools:
        chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    r = body.get("reasoning")
    if isinstance(r, dict) and isinstance(r.get("effort"), str) and \
            r["effort"]:
        chat["reasoning_effort"] = r["effort"]
    mot = body.get("max_output_tokens")
    if mot is not None:
        if isinstance(mot, bool) or not isinstance(mot, int) or mot < 1:
            raise _err("'max_output_tokens' must be a positive integer.",
                       "max_output_tokens")
        chat["max_tokens"] = mot
    fmt = (body.get("text") or {}).get("format") if isinstance(
        body.get("text"), dict) else None
    if isinstance(fmt, dict):
        kind = fmt.get("type")
        if kind == "json_schema":
            js = {"name": fmt.get("name") or "response",
                  "schema": fmt.get("schema") or {}}
            for k in ("strict", "description"):
                if fmt.get(k) is not None:
                    js[k] = fmt[k]
            chat["response_format"] = {"type": "json_schema",
                                       "json_schema": js}
        elif kind == "json_object":
            chat["response_format"] = {"type": "json_object"}
        elif kind not in (None, "text"):
            raise _err(f"text.format of type {json.dumps(kind)} is not "
                       f"served; send text, json_object or json_schema.",
                       "text.format")
    for k in ("prompt_cache_key", "temperature", "top_p"):
        if body.get(k) is not None:
            chat[k] = body[k]
    if (history and not chat.get("prompt_cache_key")
            and history.get("prompt_cache_key")):
        # A chain is one conversation, named by an explicit id (the previous
        # response); its key goes on when the request sends none.
        chat["prompt_cache_key"] = history["prompt_cache_key"]
        rec["prompt_cache_key_from_chain"] = True
    ctx.cache_key = chat.get("prompt_cache_key")
    if chat["stream"]:
        # The terminal event carries usage: always asked of the chat stream.
        chat["stream_options"] = {"include_usage": True}
    if image is not None:
        # yama_generate_image's size for THIS request (proxy._run_our_tool reads
        # it from the session state; images.run_tool applies it).
        chat["_image_options"] = {"size": image["size"],
                                  "source": IMAGE_TOOL}
        rec["image_generation"] = {k: v for k, v in image.items()
                                   if v is not None}
    return chat, ctx


# --------------------------------------------------------------- output ----

_IMAGE_LINE = re.compile(
    r"!\[[^\]\n]*\]\((https?://[^\s)]+/media/([0-9a-f]{64})\.png\?([^\s)]*))\)")


def image_of_line(text: str):
    """(sha, url) when `text` is exactly one image line the proxy showed
    (IMAGES REACH THE CHAT): a signed /media link of ours that verifies and
    names a stored image. Anything else -- a model-written link, a stale
    signature -- is None."""
    m = _IMAGE_LINE.fullmatch((text or "").strip())
    if not m:
        return None
    import images
    q = urllib.parse.parse_qs(m.group(3) or "")
    ok, _why = images.verify(m.group(2), (q.get("exp") or [""])[0],
                             (q.get("sig") or [""])[0])
    if not ok or images.media_path(m.group(2)) is None:
        return None
    return m.group(2), m.group(1)


def image_item(sha: str, url: str, ctx: Ctx, with_result: bool = True
               ) -> dict:
    """The standard `image_generation_call` item for one image we made,
    with our signed link as an extension field."""
    import images
    meta = images.metadata(sha) or {}
    png = images.png_bytes(sha) if with_result else None
    item = {"type": "image_generation_call", "id": "ig_" + sha[:32],
            "status": "completed" if with_result else "in_progress",
            "result": base64.b64encode(png).decode("ascii") if png else None,
            "revised_prompt": meta.get("prompt"),
            "output_format": "png", "size": meta.get("size"),
            "background": "opaque",
            "quality": (ctx.image or {}).get("quality") or "auto",
            "action": "generate",
            "x_yamadori": {"url": url, "id": sha[:16],
                           "model": meta.get("image_model"),
                           "seed": meta.get("seed"),
                           "steps": meta.get("steps")}}
    return item


def usage_of(u: dict | None) -> dict:
    """Chat usage -> Responses usage, ALWAYS an object of integers (the AI
    SDK drops a terminal event whose usage does not parse, and OpenCode then
    records finish `other`; docs/HARNESS-RESPONSES.md): zeros when the turn
    reported none. `output_tokens_details` is left out when the reasoning
    count is unknown (the tokenizer did not answer): never guessed
    (proxy._usage_of); the spec marks it required, Codex and the AI SDK
    read it as optional."""
    if not isinstance(u, dict):
        u = {}
    inp = int(u.get("prompt_tokens") or 0)
    out = int(u.get("completion_tokens") or 0)
    res = {"input_tokens": inp,
           "input_tokens_details": {"cached_tokens": int(
               (u.get("prompt_tokens_details") or {}).get("cached_tokens")
               or 0), "cache_write_tokens": 0},
           "output_tokens": out,
           "total_tokens": int(u.get("total_tokens") or inp + out)}
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if rt is not None:
        res["output_tokens_details"] = {"reasoning_tokens": int(rt)}
    return res


def _status_of(finish: str | None) -> tuple[str, dict | None]:
    if finish == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if finish == "incomplete":
        # F1: the model server dropped mid-answer (the partial is in the
        # output). A FAILURE, not an incomplete: Codex treats every
        # response.incomplete as an error anyway, Pi every one but
        # max_output_tokens, and Hermes would CONTINUE it as if cut by the
        # length cap (docs/HARNESS-RESPONSES.md gap 3).
        return "failed", None
    return "completed", None


DROPPED = {"code": "server_error", "message": (
    "The model server's connection dropped mid-answer; the partial answer is "
    "in the output. Send the request again."),
    "x_yamadori_code": "upstream_dropped"}


def _reasoning_item(text: str, rid: str | None = None) -> dict | None:
    if REASONING_MODE == "off" or not (text or "").strip():
        return None
    item = {"type": "reasoning", "id": rid or _new("rs"), "summary": [],
            "encrypted_content": None, "status": "completed"}
    if REASONING_MODE == "content":
        item["content"] = [{"type": "reasoning_text", "text": text}]
    else:
        item["summary"] = [{"type": "summary_text", "text": text}]
    return item


def _message_item(text: str, mid: str | None = None,
                  status: str = "completed") -> dict:
    return {"type": "message", "id": mid or _new("msg"), "role": "assistant",
            "status": status, "content": [{
                "type": "output_text", "text": text, "annotations": [],
                "logprobs": []}]}


def _call_item(c: dict, ctx: Ctx, status: str = "completed") -> dict:
    fn = c.get("function") or {}
    name = fn.get("name") or ""
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args or {}, ensure_ascii=False)
    if name in ctx.custom:
        try:
            v = json.loads(args)
            inp = v.get("input") if isinstance(v, dict) else None
        except ValueError:
            inp = None
        return {"type": "custom_tool_call", "id": _new("ctc"),
                "call_id": c.get("id") or _new("call"), "name": name,
                "input": inp if isinstance(inp, str) else args,
                "status": status}
    item = {"type": "function_call", "id": _new("fc"),
            "call_id": c.get("id") or _new("call"), "name": name,
            "arguments": args, "status": status}
    if name in ctx.ns_of:
        # A flattened namespace function: Codex routes by (namespace, name).
        item["namespace"], item["name"] = ctx.ns_of[name]
    return item


def response_object(ctx: Ctx, *, status: str, output: list, usage=None,
                    error=None, incomplete=None, x=None,
                    compact: bool = False) -> dict:
    d = {"id": ctx.id, "object": "response", "created_at": ctx.created_at,
         "status": status, "background": False, "error": error,
         "incomplete_details": incomplete, "model": ctx.model,
         "output": output}
    if compact:
        return d
    d.update(ctx.echo())
    samp = ((x or {}).get("sampling") or {}).get("enforced") or {}
    for k in ("temperature", "top_p"):
        if samp.get(k) is not None:
            d[k] = samp[k]
    d["usage"] = usage
    if x is not None:
        d["x_yamadori"] = dict(x, responses=ctx.rec)
    return d


_MEDIA_URL = re.compile(r"/media/([0-9a-f]{64})\.png")


def _elide_images(items: list) -> list:
    """Copies of the output items without an image_generation_call's base64
    `result` (the PNG sits in the media store, and the item's own signed link
    names its sha, so GET can put it back while the media is there)."""
    out = []
    for it in items:
        if (isinstance(it, dict) and it.get("type") == "image_generation_call"
                and it.get("result")):
            it = dict(it, result=None)
        out.append(it)
    return out


def rehydrate_images(obj: dict) -> dict:
    """GET: an image_generation_call whose `result` was left out of the store
    gets its PNG back from the media store when it is still there."""
    import images
    out = []
    for it in obj.get("output") or []:
        if (isinstance(it, dict) and it.get("type") == "image_generation_call"
                and not it.get("result")):
            m = _MEDIA_URL.search(str((it.get("x_yamadori") or {}).get("url")
                                      or ""))
            png = images.png_bytes(m.group(1)) if m else None
            if png:
                it = dict(it, result=base64.b64encode(png).decode("ascii"))
        out.append(it)
    return dict(obj, output=out)


def persist(ctx: Ctx, resp: dict) -> dict:
    """Store the finished response (response_store) when this request is
    stored, BEFORE it is returned or its terminal event goes out. Sets
    `ctx.rec["stored"]` {id, chained_from, items, bytes} (the Response's
    x_yamadori.responses holds that same dict, so it shows in it) and
    reports the truth in `store`. Never raises."""
    if not ctx.store_on or ctx.account is None:
        return resp
    info = response_store.put(
        ctx.account, ctx.id, created=ctx.created_at, prev_id=ctx.prev_id,
        model=ctx.model, status=resp.get("status") or "",
        input_items=ctx.input_items,
        output_items=_elide_images(resp.get("output") or []), response=resp)
    ctx.rec["stored"] = {k: v for k, v in info.items() if k != "chain"}
    if info.get("stored"):
        print(f"responses: stored {ctx.id} (chained from "
              f"{ctx.prev_id or 'none'}, {info.get('items')} items, "
              f"{info.get('bytes')} bytes)", flush=True)
    else:
        resp["store"] = False
        print(f"responses: {ctx.id} NOT stored: {info.get('reason')}",
              flush=True)
    return resp


def of_chat(d: dict, ctx: Ctx) -> dict:
    """A blocking chat completion -> the Response object."""
    ch = (d.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    output: list = []
    r = _reasoning_item(m.get("reasoning_content") or "")
    if r:
        output.append(r)
    content = m.get("content") or ""
    if ctx.image is not None:
        for line in content.splitlines():
            hit = image_of_line(line)
            if hit:
                output.append(image_item(hit[0], hit[1], ctx))
    calls = m.get("tool_calls") or []
    if content or not calls:
        status, _inc = _status_of(ch.get("finish_reason"))
        output.append(_message_item(
            content, status="incomplete" if status != "completed"
            else "completed"))
    for c in calls:
        output.append(_call_item(c, ctx))
    status, inc = _status_of(ch.get("finish_reason"))
    return persist(ctx, response_object(
        ctx, status=status, output=output, usage=usage_of(d.get("usage")),
        incomplete=inc, error=DROPPED if status == "failed" else None,
        x=d.get("x_yamadori") or {}))


# -------------------------------------------------------------- stream -----

def _sse(ev: dict) -> bytes:
    return (b"event: " + ev["type"].encode() + b"\ndata: "
            + json.dumps(ev, ensure_ascii=False).encode("utf-8") + b"\n\n")


def _failure_of(err: dict) -> dict:
    """Our error object -> the Response `error` {code, message}, in the codes
    Codex maps (codex-rs sse.rs: context_length_exceeded -> compaction,
    rate_limit_exceeded / server_is_overloaded -> retry with backoff,
    invalid_prompt -> not retried); the rest are server_error. Our own code
    rides along as `x_yamadori_code`."""
    code = err.get("code")
    kind = err.get("type") or ""
    if code == "context_length_exceeded":
        out = "context_length_exceeded"
    elif kind == "rate_limit_error" or code in ("server_busy",):
        out = "rate_limit_exceeded"
    elif kind == "service_unavailable_error" or code == "model_unavailable":
        out = "server_is_overloaded"
    elif kind == "invalid_request_error":
        out = "invalid_prompt"
    else:
        out = "server_error"
    return {"code": out, "message": str(err.get("message") or "error"),
            "x_yamadori_code": code}


class Stream:
    """proxy.stream_body's chat chunks -> the Responses event sequence.

    Fed one chat SSE chunk at a time (`feed`), it returns the bytes to send:
      response.created, response.in_progress        on the first chunk
      reasoning: output_item.added, reasoning_summary_part.added (or
        content_part.added), *_text.delta per chunk, *_text.done,
        *_part.done, output_item.done               while reasoning arrives
      text: output_item.added (message), content_part.added, output_text.delta
        per chunk (logprobs: []), then output_text.done, content_part.done,
        output_item.done                            from the first content
      an image the proxy showed (hosted image tool declared): the open
        message closed, image_generation_call added, image_generation_call
        .completed, output_item.done (result: the PNG) -- the image line
        itself stays in the text, which opens a new message item (the
        client's replay merges them back: the ledger's key)
      calls: per call output_item.added, function_call_arguments.delta (the
        whole string: calls are checked whole), .done, output_item.done
        (custom tools: custom_tool_call_input.delta/.done)
      heartbeat: response.in_progress (compact), at most every KEEPALIVE_S
      terminal: response.completed | response.incomplete, with usage and
        x_yamadori; a mid-stream error: response.failed {error: {code,
        message}}. No [DONE]: clients end on the terminal event.
    """

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.seq = 0
        self.started = False
        self.done = False
        self.output: list = []
        self.reasoning: dict | None = None      # open reasoning item state
        self.message: dict | None = None        # open message item state
        self.finish: str | None = None
        self.x: dict = {}
        self.usage: dict | None = None
        self.last_event = 0.0

    # -- plumbing
    def _ev(self, out: list, **ev) -> None:
        ev["sequence_number"] = self.seq
        self.seq += 1
        self.last_event = time.time()
        out.append(_sse(ev))

    def _snapshot(self, status: str, compact: bool = False) -> dict:
        return response_object(self.ctx, status=status,
                               output=[] if compact else list(self.output),
                               compact=compact)

    def _begin(self, out: list) -> None:
        self.started = True
        base = response_object(self.ctx, status="in_progress", output=[])
        self._ev(out, type="response.created", response=base)
        self._ev(out, type="response.in_progress", response=base)

    # -- reasoning
    def _reason(self, out: list, text: str) -> None:
        if REASONING_MODE == "off" or self.message is not None:
            self._keepalive(out)
            return
        if self.reasoning is None:
            item = {"type": "reasoning", "id": _new("rs"), "summary": [],
                    "encrypted_content": None, "status": "in_progress"}
            if REASONING_MODE == "content":
                item["content"] = []
            idx = len(self.output)
            self.output.append(item)
            self.reasoning = {"item": item, "index": idx, "text": ""}
            self._ev(out, type="response.output_item.added",
                     output_index=idx, item=item)
            if REASONING_MODE == "content":
                self._ev(out, type="response.content_part.added",
                         item_id=item["id"], output_index=idx,
                         content_index=0,
                         part={"type": "reasoning_text", "text": ""})
            else:
                self._ev(out, type="response.reasoning_summary_part.added",
                         item_id=item["id"], output_index=idx,
                         summary_index=0,
                         part={"type": "summary_text", "text": ""})
        r = self.reasoning
        r["text"] += text
        if REASONING_MODE == "content":
            self._ev(out, type="response.reasoning_text.delta",
                     item_id=r["item"]["id"], output_index=r["index"],
                     content_index=0, delta=text)
        else:
            self._ev(out, type="response.reasoning_summary_text.delta",
                     item_id=r["item"]["id"], output_index=r["index"],
                     summary_index=0, delta=text)

    def _close_reasoning(self, out: list) -> None:
        r = self.reasoning
        if r is None:
            return
        self.reasoning = None
        item, idx, text = r["item"], r["index"], r["text"]
        part = ({"type": "reasoning_text", "text": text}
                if REASONING_MODE == "content"
                else {"type": "summary_text", "text": text})
        if REASONING_MODE == "content":
            self._ev(out, type="response.reasoning_text.done",
                     item_id=item["id"], output_index=idx, content_index=0,
                     text=text)
            self._ev(out, type="response.content_part.done",
                     item_id=item["id"], output_index=idx, content_index=0,
                     part=part)
            item["content"] = [part]
        else:
            self._ev(out, type="response.reasoning_summary_text.done",
                     item_id=item["id"], output_index=idx, summary_index=0,
                     text=text)
            self._ev(out, type="response.reasoning_summary_part.done",
                     item_id=item["id"], output_index=idx, summary_index=0,
                     part=part)
            item["summary"] = [part]
        item["status"] = "completed"
        self._ev(out, type="response.output_item.done", output_index=idx,
                 item=item)

    # -- text
    def _text(self, out: list, text: str) -> None:
        if self.ctx.image is not None:
            hit = image_of_line(text)
            if hit:
                self._image(out, *hit)
        self._close_reasoning(out)
        if self.message is None:
            item = _message_item("", status="in_progress")
            item["content"] = []
            idx = len(self.output)
            self.output.append(item)
            self.message = {"item": item, "index": idx, "text": ""}
            self._ev(out, type="response.output_item.added",
                     output_index=idx, item=item)
            self._ev(out, type="response.content_part.added",
                     item_id=item["id"], output_index=idx, content_index=0,
                     part={"type": "output_text", "text": "",
                           "annotations": [], "logprobs": []})
        m = self.message
        m["text"] += text
        self._ev(out, type="response.output_text.delta",
                 item_id=m["item"]["id"], output_index=m["index"],
                 content_index=0, delta=text, logprobs=[])

    def _close_message(self, out: list, status: str = "completed") -> None:
        m = self.message
        if m is None:
            return
        self.message = None
        item, idx, text = m["item"], m["index"], m["text"]
        part = {"type": "output_text", "text": text, "annotations": [],
                "logprobs": []}
        self._ev(out, type="response.output_text.done", item_id=item["id"],
                 output_index=idx, content_index=0, text=text, logprobs=[])
        self._ev(out, type="response.content_part.done", item_id=item["id"],
                 output_index=idx, content_index=0, part=part)
        item["content"] = [part]
        item["status"] = status
        self._ev(out, type="response.output_item.done", output_index=idx,
                 item=item)

    def _image(self, out: list, sha: str, url: str) -> None:
        """The image exists (the proxy shows it the moment the tool
        returns): its item, whole. No in_progress / generating /
        partial_image events -- the stream learns of the call when it has
        completed, and a partial it never had is not invented."""
        self._close_reasoning(out)
        self._close_message(out)
        added = image_item(sha, url, self.ctx, with_result=False)
        idx = len(self.output)
        self.output.append(added)
        self._ev(out, type="response.output_item.added", output_index=idx,
                 item=added)
        self._ev(out, type="response.image_generation_call.completed",
                 item_id=added["id"], output_index=idx)
        done = image_item(sha, url, self.ctx)
        self.output[idx] = done
        self._ev(out, type="response.output_item.done", output_index=idx,
                 item=done)
        self.ctx.rec["images_emitted"] = self.ctx.rec.get(
            "images_emitted", 0) + 1

    # -- calls
    def _calls(self, out: list, calls: list) -> None:
        self._close_reasoning(out)
        self._close_message(out)
        for c in calls:
            item = _call_item(c, self.ctx, status="in_progress")
            idx = len(self.output)
            custom = item["type"] == "custom_tool_call"
            body = item["input"] if custom else item["arguments"]
            added = dict(item, **({"input": ""} if custom
                                  else {"arguments": ""}))
            self.output.append(added)
            self._ev(out, type="response.output_item.added",
                     output_index=idx, item=added)
            if custom:
                self._ev(out, type="response.custom_tool_call_input.delta",
                         item_id=item["id"], output_index=idx,
                         call_id=item["call_id"], delta=body)
                self._ev(out, type="response.custom_tool_call_input.done",
                         item_id=item["id"], output_index=idx,
                         call_id=item["call_id"], input=body)
            else:
                self._ev(out, type="response.function_call_arguments.delta",
                         item_id=item["id"], output_index=idx, delta=body)
                self._ev(out, type="response.function_call_arguments.done",
                         item_id=item["id"], output_index=idx,
                         name=item["name"], arguments=body)
            item["status"] = "completed"
            self.output[idx] = item
            self._ev(out, type="response.output_item.done",
                     output_index=idx, item=item)

    def _keepalive(self, out: list) -> None:
        if time.time() - self.last_event >= KEEPALIVE_S:
            self._ev(out, type="response.in_progress",
                     response=self._snapshot("in_progress", compact=True))

    # -- the end
    def _terminal(self, out: list) -> None:
        status, inc = _status_of(self.finish)
        self._close_reasoning(out)
        self._close_message(out, "completed" if status == "completed"
                            else "incomplete")
        resp = persist(self.ctx, response_object(
            self.ctx, status=status, output=list(self.output),
            usage=usage_of(self.usage), incomplete=inc,
            error=DROPPED if status == "failed" else None, x=self.x))
        self._ev(out, type={"completed": "response.completed",
                            "failed": "response.failed"}.get(
                                status, "response.incomplete"),
                 response=resp)
        self.done = True

    def _failed(self, out: list, err: dict) -> None:
        resp = persist(self.ctx, response_object(
            self.ctx, status="failed", output=list(self.output),
            error=_failure_of(err), x=self.x or {}))
        self._ev(out, type="response.failed", response=resp)
        self.done = True

    def fail(self, e: BaseException) -> bytes:
        """An exception after the first byte that stream_body did not turn
        into its error event (it always does; this is the backstop)."""
        out: list = []
        if not self.started:
            self._begin(out)
        self._failed(out, api_errors.of_exception(e).body()["error"])
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
                self._failed(out, d["error"] if isinstance(
                    d["error"], dict) else {"message": str(d["error"])})
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


def stream(chat_gen, ctx: Ctx):
    """The Responses stream over `proxy.stream_body(...)`. Nothing is
    yielded before stream_body's first chunk, so a failure before the turn
    starts still RAISES here and server.py answers it with a real HTTP
    status (E1). Closing this closes stream_body, which cancels the turn."""
    s = Stream(ctx)
    try:
        for raw in chat_gen:
            b = s.feed(raw)
            if b:
                yield b
            if s.done:
                break
    except GeneratorExit:
        raise
    except Exception as e:                                       # noqa: BLE001
        if not s.started:
            raise
        yield s.fail(e)
    finally:
        try:
            close = getattr(chat_gen, "close", None)
            if close is not None:
                close()
        except ValueError:
            pass            # executing in its thread; its token is cancelled
