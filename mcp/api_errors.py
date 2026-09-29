#!/usr/bin/env python
"""ONE ERROR PATH: every failure a chat client can see, in OpenAI's shape.

WHY (docs/OPENAI-CONFORMANCE.md, E1 and E2, 2026-09-25). Before this module
the proxy had four error shapes: the OpenAI object on 401/404/429, a bare
string (`{"error": "HTTPError: ..."}`) on a bad body and on every turn
failure (always 502, an upstream 400 included), assistant CONTENT on the
streamed path (`"\\n[upstream error: ...]"` on a 200, then `[DONE]` with no
finish_reason), and a cut socket for anything that escaped the turn. Hermes
read the streamed one as a mid-stream drop and retried a deterministic
error; Pi, Cline and the current AI SDK raise "ended without finish_reason";
no client could branch on a type or a code, and a context overflow was never
recognised, so nobody compacted.

THE CONTRACT

  - Before the first byte: an HTTP status and `{"error": {"message", "type",
    "param", "code"}}` (all four keys always present; `param`/`code` may be
    null). `ApiError.body()`.
  - After it (a stream already committed): ONE SSE event `data: {"error":
    {...}}`, then `data: [DONE]`. The OpenAI SDKs raise on an `error` key in
    an event (openai-python `_streaming.py` Stream.__stream__: `if
    is_mapping(data) and data.get("error"): raise APIError(message,
    body=data["error"])`); the AI SDK turns a top-level `error` into a
    stream error. Never assistant content. `ApiError.sse()`.
  - Status: 400 for the client's own mistakes (bad JSON, bad `messages`, a
    field we must refuse, an upstream 4xx, a prompt that does not fit); 401,
    403 and 429 as before; 404 for a route we do not serve; 503 when the
    model server is unreachable or loading; 5xx otherwise only for a real
    fault. SDKs retry 5xx (openai-python: twice), so a client error sent as
    one is sent three times.

CONTEXT OVERFLOW. `context_length_exceeded` uses OpenAI's own wording --
"This model's maximum context length is M tokens. However, your messages
resulted in N tokens." -- because clients match on it: Hermes classifies a
400 by message (agent/error_classifier.py `_CONTEXT_OVERFLOW_PATTERNS`:
"maximum context", "context length", "reduce the length"), a status-less
streamed error by `code` (`_ERROR_CODE_VERDICTS`: `context_length_exceeded`),
and reads the window back from "maximum context length is (\\d+)"
(agent/model_metadata.py parse_context_limit_from_error). The message avoids
the phrases Hermes reads as an OUTPUT-cap error ("requested", "in the
completion", "must be", "should be", "limited to", "available tokens",
"max_tokens") so it compresses rather than shrinking max_tokens, and the
phrases it reads as a request-shape error ("not supported", "unsupported",
"unknown parameter"), which would fail fast instead.
"""
from __future__ import annotations

import json
import re

# OpenAI's `type` per status. 503 is the spec's service_unavailable_error.
_TYPES = {400: "invalid_request_error", 401: "invalid_request_error",
          403: "permission_error", 404: "invalid_request_error",
          405: "invalid_request_error", 409: "invalid_request_error",
          413: "invalid_request_error", 422: "invalid_request_error",
          429: "rate_limit_error", 500: "server_error", 502: "server_error",
          503: "service_unavailable_error", 504: "server_error",
          507: "server_error"}


def type_of(status: int) -> str:
    return _TYPES.get(int(status), "invalid_request_error"
                      if 400 <= int(status) < 500 else "server_error")


class ApiError(RuntimeError):
    """A failure a client sees: status, OpenAI's four fields, and extras
    (ignored by clients: `retryable`, `remedies`, facts) inside `error`."""

    def __init__(self, status: int, message: str, *, type: str | None = None,
                 code: str | None = None, param: str | None = None,
                 headers: dict | None = None, extra: dict | None = None):
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.type = type or type_of(self.status)
        self.code = code
        self.param = param
        self.headers = dict(headers or {})
        self.extra = dict(extra or {})

    def body(self) -> dict:
        err = {"message": self.message, "type": self.type,
               "param": self.param, "code": self.code}
        for k, v in self.extra.items():
            err.setdefault(k, v)
        return {"error": err}

    def sse(self) -> bytes:
        """The mid-stream form: one event, the error object at the top."""
        return b"data: " + json.dumps(self.body()).encode() + b"\n\n"


def invalid(message: str, param: str | None = None,
            code: str | None = None) -> ApiError:
    return ApiError(400, message, code=code, param=param)


# ------------------------------------------------------------------ routes --

def unknown_route(method: str, path: str) -> ApiError:
    return ApiError(404, f"Unknown request URL: {method} {path}. This server "
                    "serves POST /v1/chat/completions, POST /v1/responses, "
                    "POST /v1/messages, POST /v1/messages/count_tokens, "
                    "GET /v1/models, GET /v1/models/{model} and POST "
                    "/v1/images/generations.",
                    code="unknown_url")


def method_not_allowed(method: str, path: str, allow: list[str]) -> ApiError:
    return ApiError(405, f"{method} is not allowed on {path}; use "
                    f"{', '.join(sorted(allow))}.", code="method_not_allowed",
                    headers={"Allow": ", ".join(sorted(allow))})


# ------------------------------------------- one model per tier ----

def at_capacity(holder: str, retry_after: int, why: str = "", model: str | None = None,
                tier: str | None = None) -> ApiError:
    """A request whose model is not on the card while another tier's model holds it (mcp/max_mode.py; operator,
    2026-09-28: "If a medium tier comes in on max mode, we reject it. Plain and simple, model is at capacity
    error."; generalised to the tier -> model table 2026-09-29). 503 with Retry-After: every harness in
    docs/FLASH-NEXT.md section 5 retries a 5xx, and OpenAI SDKs honour Retry-After; a 429 would read as the
    caller's own rate. The message says it plainly: which tier asked, that another tier's model holds the card, and
    when to retry."""
    try:
        import max_mode
        full = bool(max_mode.FULL)
    except Exception:                                    # noqa: BLE001
        full = False
    if not full:
        # the 2026-09-28 max-mode message, word for word (no table file: that day's behaviour exactly)
        return ApiError(
            503, f"The model is at capacity: the stack is serving max mode ({holder}) and refuses other requests "
                 f"until it frees -- retry in about {int(retry_after)} s. Ask for reasoning_effort \"max\" to be "
                 f"served now.", code="model_at_capacity", headers={"Retry-After": str(int(retry_after))},
            extra={"retryable": True, "max_model": holder, "why": why})
    asked = f"reasoning_effort \"{tier}\"" if tier else "this request's effort tier"
    return ApiError(
        503, f"The model is at capacity: another effort tier's model is serving right now (one model per tier, "
             f"one on the card at a time), and {asked} is served by a different model. Retry in about "
             f"{int(retry_after)} s; this conversation keeps its model.", code="model_at_capacity",
        headers={"Retry-After": str(int(retry_after))},
        extra={"retryable": True, "holder": holder, "max_model": holder, "model": model, "tier": tier,
               "why": why})


# ------------------------------------------------------ context overflow ----

def context_length_exceeded(n_prompt: int, limit: int,
                            floor: int = 0) -> ApiError:
    """400, code context_length_exceeded, in OpenAI's wording (see the
    module doc for why each phrase is there and which are avoided)."""
    msg = (f"This model's maximum context length is {int(limit)} tokens. "
           f"However, your messages resulted in {int(n_prompt)} tokens")
    if floor and n_prompt <= limit:
        msg += (f", which leaves {int(limit) - int(n_prompt)} tokens, fewer "
                f"than the {int(floor)} an answer needs here")
    msg += ". Please reduce the length of the messages."
    return ApiError(400, msg, code="context_length_exceeded",
                    param="messages",
                    extra={"n_prompt_tokens": int(n_prompt),
                           "context_length": int(limit),
                           # what an answer needs here: the Messages API
                           # states the prompt's maximum as limit - floor
                           "floor": int(floor)})


# ------------------------------------------------------------- upstream -----

def _parse_error(raw) -> dict:
    """llama-server's (or llama-swap's) error body as a dict; {} if not JSON."""
    if isinstance(raw, dict):
        return raw.get("error") if isinstance(raw.get("error"), dict) else raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    try:
        d = json.loads(raw or "")
    except (TypeError, ValueError):
        return {"message": " ".join(str(raw or "").split())[:500]} if raw else {}
    if isinstance(d, dict) and isinstance(d.get("error"), dict):
        return d["error"]
    if isinstance(d, dict):
        return d
    return {"message": str(d)[:500]}


_JINJA = re.compile(r"(?:Jinja Exception|raise_exception)[^:\n]*:\s*(.+)")


def template_refusal(message: str | None) -> str | None:
    """The chat template's own refusal sentence, when `message` is one (a
    raise_exception in the served template, as llama-server reports it:
    '... Error: Jinja Exception: System message must be at the
    beginning.'), else None."""
    m = _JINJA.search(message or "")
    return " ".join(m.group(1).split())[:300] if m else None


def of_upstream(status: int | None, error: dict | None,
                limit: int | None = None) -> ApiError:
    """What the model server said, as the client's error.

    llama-server's types (tools/server/server-common.cpp format_error_response):
    invalid_request_error 400, authentication_error 401, not_found_error 404,
    server_error 500, permission_error 403, not_supported_error 501,
    unavailable_error 503, exceed_context_size_error 400 (with n_prompt_tokens
    and n_ctx). A 4xx is the CLIENT's error and stays a 4xx (400 unless it is
    one of the statuses a client acts on); not_supported (501) is also the
    request's, so 400; 503 is 503; anything else is 502."""
    e = dict(error or {})
    msg = str(e.get("message") or "").strip() or "the model server refused " \
        "the request"
    if "image input is not supported" in msg.lower():
        # the main server has no projector after all (vision.main_sees said
        # it had): stop passing images to it until this process restarts
        try:
            import vision
            vision.note_no_projector(msg)
        except Exception:                                            # noqa: BLE001
            pass
    kind = str(e.get("type") or "")
    try:
        st = int(status if status is not None else e.get("code") or 0)
    except (TypeError, ValueError):
        st = 0
    if kind == "exceed_context_size_error" or (
            "context size" in msg.lower() and 400 <= st < 500):
        n = int(e.get("n_prompt_tokens") or 0)
        return context_length_exceeded(n, int(limit or e.get("n_ctx") or 0))
    if kind == "not_supported_error" or st == 501:
        return ApiError(400, f"The model server cannot serve this request: "
                        f"{msg}", code="unsupported_request")
    tpl = template_refusal(msg)
    if tpl:
        # THE CHAT TEMPLATE REFUSED THE MESSAGES (2026-09-26, docs/HARNESS-
        # PI.md gap 1). llama-server answers a template's raise_exception with
        # a 500 server_error, which was sent on as a 502 -- and a client
        # retries a 5xx (Pi: 3 times, 2/4/8 s) a request that fails the same
        # way every time. It is the request's shape: 400, `invalid_prompt`
        # (the code Codex branches on), the template's own sentence without
        # the Jinja trace.
        return ApiError(400, f"The chat template cannot render these "
                        f"messages: {tpl}", code="invalid_prompt",
                        param="messages")
    if 400 <= st < 500:
        keep = st if st in (400, 401, 403, 404, 409, 413, 422) else 400
        return ApiError(keep, msg, code=(kind or None)
                        if kind not in ("invalid_request_error", "") else None)
    if st == 503 or kind == "unavailable_error":
        return ApiError(503, f"The model is not available right now (loading "
                        f"or restarting): {msg}", code="model_unavailable",
                        headers={"Retry-After": "30"},
                        extra={"retryable": True})
    return ApiError(502, f"The model server failed: {msg}",
                    code="upstream_error")


def of_exception(e: BaseException, limit: int | None = None) -> ApiError:
    """Any exception from a turn, as the client's error (the one mapping
    server.py and proxy.stream_body both use)."""
    if isinstance(e, ApiError):
        return e
    if type(e).__name__ == "ModelAtCapacity":            # mcp/max_mode.py, the one door's backstop
        import max_mode
        return at_capacity(getattr(e, "holder", None) or max_mode.MAX or "another model",
                           int(getattr(e, "retry_after", 0) or 30), str(e), model=getattr(e, "model", None))
    import socket
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        raw = getattr(e, "_yamadori_body", None)
        if raw is None:
            try:
                raw = e.read()
            except Exception:                                    # noqa: BLE001
                raw = b""
        return of_upstream(e.code, _parse_error(raw), limit)
    # streaming.UpstreamError: the model server's own refusal, before or
    # inside a stream, with its status and error object when it gave them.
    if hasattr(e, "upstream_error") or hasattr(e, "upstream_status"):
        err = getattr(e, "upstream_error", None)
        status = getattr(e, "upstream_status", None)
        if not isinstance(err, dict):
            err = {"message": str(e)}
        if status is None and not err.get("type") and not err.get("code") \
                and not template_refusal(str(err.get("message") or e)):
            # An in-stream failure with no status: the model server broke
            # mid-generation. Not the client's error.
            return ApiError(502, f"The model server failed: {e}",
                            code="upstream_error")
        return of_upstream(status, err, limit)
    if isinstance(e, (urllib.error.URLError, ConnectionError)) and not \
            isinstance(e, (socket.timeout, TimeoutError)):
        return ApiError(503, f"The model server is not reachable "
                        f"({type(e).__name__}: {getattr(e, 'reason', e)}).",
                        code="model_unavailable",
                        headers={"Retry-After": "30"},
                        extra={"retryable": True})
    if isinstance(e, (socket.timeout, TimeoutError)):
        return ApiError(504, f"The model server did not answer in time "
                        f"({type(e).__name__}).", code="upstream_timeout")
    return ApiError(500, f"Internal error: {type(e).__name__}: {e}",
                    code="internal_error")


# ------------------------------------------------------------ validation ----

ROLES = ("system", "developer", "user", "assistant", "tool", "function")
# Content parts this server can use: text, and images (read by yama_describe_image,
# mcp/vision.py). `input_audio` and `file` would reach a text model that
# cannot read them.
PARTS = ("text", "image_url")

# Fields we REFUSE, and why: each would otherwise be silently wrong.
#   n > 1               the reader merges the choices into one message ("pongpong")
#   logprobs / top_logprobs  llama-server refuses them with tools + stream, and
#                       the proxy always streams upstream and adds its tools
#   modalities ["audio"], audio   text-only model
#   content parts input_audio / file   the model cannot read them
#   tools of a type other than "function", tool_choice allowed_tools / custom
#                       llama-server does not know those forms
# Fields we ACCEPT AND IGNORE (safe: nothing the client relies on changes):
#   user, safety_identifier, metadata, store, service_tier, prediction,
#   verbosity, web_search_options, seed (best effort by spec; fan-out may
#   deliver another candidate), functions/function_call (deprecated), and
#   temperature/top_p/presence_penalty (overridden by the vendor sampling,
#   recorded in x_yamadori.sampling). Everything else passes through.
REFUSED_NOTE = ("n>1, logprobs/top_logprobs, audio output, input_audio/file "
                "parts, non-function tool forms")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_chat(body) -> ApiError | None:
    """A chat body's shape, BEFORE admission: the first error, or None."""
    if not isinstance(body, dict):
        return invalid("The request body must be a JSON object.",
                       code="invalid_body")
    msgs = body.get("messages")
    if msgs is None:
        return invalid("Missing required parameter: 'messages'.",
                       param="messages", code="missing_required_parameter")
    if not isinstance(msgs, list) or not msgs:
        return invalid("'messages' must be a non-empty array of message "
                       "objects.", param="messages", code="invalid_type")
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            return invalid(f"messages[{i}] must be an object.",
                           param=f"messages[{i}]", code="invalid_type")
        role = m.get("role")
        if role not in ROLES:
            return invalid(f"messages[{i}].role must be one of "
                           f"{', '.join(ROLES)}; got {json.dumps(role)}.",
                           param=f"messages[{i}].role", code="invalid_value")
        c = m.get("content")
        if c is not None and not isinstance(c, (str, list)):
            return invalid(f"messages[{i}].content must be a string or an "
                           f"array of content parts.",
                           param=f"messages[{i}].content", code="invalid_type")
        if isinstance(c, list):
            for j, p in enumerate(c):
                t = p.get("type") if isinstance(p, dict) else None
                if t not in PARTS:
                    return invalid(
                        f"messages[{i}].content[{j}].type {json.dumps(t)} "
                        f"cannot be read by this model; send text or "
                        f"image_url parts.",
                        param=f"messages[{i}].content[{j}].type",
                        code="invalid_content_part")
    n = body.get("n")
    if n is not None and (not _is_int(n) or n != 1):
        return invalid("This server returns one choice per request: send n=1 "
                       "or leave it out.", param="n", code="invalid_value")
    for k in ("max_tokens", "max_completion_tokens"):
        v = body.get(k)
        if v is not None and (not _is_int(v) or v < 1):
            return invalid(f"'{k}' must be a positive integer.", param=k,
                           code="invalid_value")
    if body.get("logprobs") or body.get("top_logprobs"):
        k = "logprobs" if body.get("logprobs") else "top_logprobs"
        return invalid("Log probabilities cannot be returned: the server "
                       "streams from the model with tools attached, where "
                       "the model server refuses them. Leave out logprobs "
                       "and top_logprobs.", param=k, code="invalid_value")
    mods = body.get("modalities")
    if body.get("audio") is not None or (isinstance(mods, list)
                                         and "audio" in mods):
        return invalid("This model produces text only; leave out 'audio' "
                       "and 'modalities'.", param="modalities",
                       code="invalid_value")
    tools = body.get("tools")
    if tools is not None and not isinstance(tools, list):
        return invalid("'tools' must be an array.", param="tools",
                       code="invalid_type")
    for i, t in enumerate(tools or []):
        if not isinstance(t, dict) or t.get("type") != "function" or \
                not isinstance(t.get("function"), dict) or \
                not t["function"].get("name"):
            return invalid(f"tools[{i}] must be {{\"type\": \"function\", "
                           f"\"function\": {{\"name\": ...}}}}; other tool "
                           f"forms cannot be served here.",
                           param=f"tools[{i}]", code="invalid_value")
    tc = body.get("tool_choice")
    if isinstance(tc, dict) and tc.get("type") not in (None, "function"):
        return invalid(f"tool_choice of type {json.dumps(tc.get('type'))} "
                       f"cannot be served here; send none, auto, required or "
                       f"{{\"type\": \"function\", ...}}.",
                       param="tool_choice", code="invalid_value")
    return None
