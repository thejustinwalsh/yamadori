#!/usr/bin/env python
"""OpenAI conformance fixes 1-3: one error path, the client's own overflow,
usage. No GPU.

    python mcp/test_openai_conformance.py      -> "N/M checks passed"

WHAT THIS GATES (docs/OPENAI-CONFORMANCE.md, items E1, E2, C1, U1, U2;
operator 2026-09-25: "Before V0 fixes should land" -- V0 is Hermes, chat
completions, streamed, xhigh):

  1. ONE ERROR PATH. Every failure a client can see is OpenAI's object
     {"error": {message, type, param, code}} with the right status: 400 for
     bad JSON, a bad body, a field we must refuse and an upstream 4xx (never
     502: SDKs retry 5xx); 404 for a /v1 route we do not serve (it was 405);
     503 for an unreachable or loading model server. The fields we accept and
     ignore still get a 200.
  2. STREAMED: the 200 is not committed before the turn has started. A
     failure before the first byte is a real HTTP error (it was assistant
     content on a 200); a failure after it is ONE SSE `data: {"error": ...}`
     event and [DONE] -- never content, never a last chunk without
     finish_reason -- and that event is what openai-python raises on.
  3. CONTEXT: a client prompt that does not fit its window is refused with
     400 context_length_exceeded, counted by the model server's own
     tokenizer, in the wording Hermes matches ("maximum context length");
     one that fits keeps its tools and gets no appended message; our own
     hops still land; /v1/models advertises the limit enforced.
  4. USAGE: the final generation's prompt, not the sum over hops; cached and
     reasoning token details; the streamed usage chunk with `choices: []`
     only when include_usage asks for it.

Every database is a temp file set BEFORE proxy is imported (test_stream's
fake upstream and stubs are reused; this suite adds its own scriptable fake
for error statuses).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_conformance_")
for _k, _v in (("YAMADORI_CORPUS_DB", "corpus.sqlite3"),
               ("YAMADORI_NEBARI_DB", "nebari.sqlite3"),
               ("RINGS_DB", "rings.sqlite3"),
               ("CODE_INDEX_DB", "code.sqlite3"),
               ("YAMADORI_SLOTS_STATE", "slots_state.json"),
               ("CONCEPT_SEED_LAST", "seed_last.json"),
               ("YAMADORI_ACCOUNTS_DIR", "accounts")):
    os.environ[_k] = os.path.join(_TMP, _v)
os.environ.setdefault("YAMADORI_GPU_ROOM", "0")

import test_stream as S  # noqa: E402  (its own temp paths + fake upstream + stubs)
import api_errors  # noqa: E402
import budget  # noqa: E402
import catalog  # noqa: E402
import proxy  # noqa: E402
import tiers  # noqa: E402
import served_fixture  # noqa: E402
# The served model's /props, pinned (mcp/served_fixture.py): budget and
# tiers would otherwise ask the live stack (llama-swap reloads `bonsai`).
served_fixture.pin()

_tmp_root = os.path.abspath(tempfile.gettempdir())
for _k in ("YAMADORI_CORPUS_DB", "YAMADORI_NEBARI_DB", "RINGS_DB",
           "CODE_INDEX_DB", "YAMADORI_ACCOUNTS_DIR"):
    assert os.path.abspath(os.environ[_k]).startswith(_tmp_root), _k

# The pool is pinned: budget.pool_size() would otherwise ask a live
# llama-server on this machine for its n_ctx.
budget._POOL = 163840

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------ a scriptable fake upstream
# Each chat request pops the next action: a test_stream reply (streamed as
# SSE), {"status": N, "json": {...}} (an HTTP error before any byte, as
# llama-server answers a refusal: server-context.cpp "the first error must be
# treated as non-stream response"), or {"raw": bytes} (an SSE body verbatim,
# for an error INSIDE a stream). /upstream/<model>/apply-template renders the
# messages as JSON; /tokenize counts ONE TOKEN PER WHITESPACE WORD.

_acts: list[dict] = []
_chat: list[dict] = []
_upstream_calls: list[str] = []


class _Up(BaseHTTPRequestHandler):
    def _send(self, status: int, data: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):                                           # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.startswith("/upstream/"):
            _upstream_calls.append(self.path)
            if self.path.endswith("/apply-template"):
                out = {"prompt": json.dumps(body.get("messages"))
                       + json.dumps(body.get("tools") or [])}
            elif self.path.endswith("/tokenize"):
                out = {"tokens": [1] * len((body.get("content") or "")
                                           .split())}
            else:
                out = {"timings": {"cache_n": 10, "prompt_n": 2}}
            return self._send(200, json.dumps(out).encode(),
                              "application/json")
        _chat.append(body)
        if not _acts:
            return self._send(500, b'{"error": {"code": 500, "message": '
                              b'"script empty", "type": "server_error"}}',
                              "application/json")
        a = _acts.pop(0)
        if "status" in a:
            return self._send(a["status"], json.dumps(a["json"]).encode(),
                              "application/json")
        if "raw" in a:
            return self._send(200, a["raw"], "text/event-stream")
        return self._send(200, S._sse(a), "text/event-stream")

    def log_message(self, *a):                                   # noqa: D102
        pass


_srv = ThreadingHTTPServer(("127.0.0.1", 0), _Up)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{_srv.server_address[1]}"


class upstream:
    """Point the proxy at this suite's fake for the block, with a script."""

    def __init__(self, acts: list[dict], url: str | None = None):
        self.acts, self.url = acts, url

    def __enter__(self):
        S.reset([])
        _acts[:] = list(self.acts)
        _chat.clear()
        _upstream_calls.clear()
        self.saved = proxy.UPSTREAM
        proxy.UPSTREAM = self.url or URL
        return self

    def __exit__(self, *a):
        proxy.UPSTREAM = self.saved
        return False


def err_obj(status: int, typ: str, msg: str, **extra) -> dict:
    return {"status": status, "json": {"error": dict(
        {"code": status, "message": msg, "type": typ}, **extra)}}


def mid_stream_error(msg: str = "slot killed") -> bytes:
    """Reasoning deltas, then llama-server's in-stream error event."""
    ev = [{"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"},
                        "finish_reason": None}]},
          {"error": {"code": 500, "message": msg, "type": "server_error"}}]
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in ev)


# ------------------------------------------------------------- the server
import accounts  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
import server  # noqa: E402

KEY = accounts.create("conformance-tests")
H = {"Authorization": f"Bearer {KEY}"}
CLIENT = TestClient(server.app)


def post(body, stream: bool = False, raw: bytes | None = None,
         path: str = "/v1/chat/completions", headers: dict | None = None):
    """(status, headers, parsed JSON or None, SSE events or None)."""
    h = dict(H, **(headers or {}))
    if raw is not None:
        r = CLIENT.post(path, content=raw, headers=dict(
            h, **{"Content-Type": "application/json"}))
    elif stream:
        with CLIENT.stream("POST", path, json=body, headers=h) as r:
            text = "".join(r.iter_text())
            if r.headers.get("content-type", "").startswith(
                    "text/event-stream"):
                return r.status_code, r.headers, None, sse_events(text)
            return r.status_code, r.headers, _json(text), None
    else:
        r = CLIENT.post(path, json=body, headers=h)
    return r.status_code, r.headers, _json(r.text), None


def _json(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None


def sse_events(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        data = block[6:]
        out.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return out


def is_error_object(d, status: int, typ: str | None = None,
                    code: str | None = None) -> bool:
    e = (d or {}).get("error") if isinstance(d, dict) else None
    return (isinstance(e, dict)
            and all(k in e for k in ("message", "type", "param", "code"))
            and isinstance(e["message"], str) and e["message"]
            and (typ is None or e["type"] == typ)
            and (code is None or e["code"] == code))


def user(text: str = "hello") -> dict:
    return {"model": "yamadori",
            "messages": [{"role": "user", "content": text}]}


# ============================================================ 1. errors =====

def test_request_validation_is_a_400_object():
    cases = [
        ("bad JSON", None, b"{not json", "invalid_json", None),
        ("a JSON array body", None, b"[1, 2]", "invalid_body", None),
        ("missing messages", {"model": "yamadori"}, None,
         "missing_required_parameter", "messages"),
        ("empty messages", {"model": "yamadori", "messages": []}, None,
         "invalid_type", "messages"),
        ("a message that is not an object",
         {"messages": ["hi"]}, None, "invalid_type", "messages[0]"),
        ("an unknown role", {"messages": [{"role": "robot",
                                           "content": "x"}]}, None,
         "invalid_value", "messages[0].role"),
        ("content of the wrong type", {"messages": [{"role": "user",
                                                     "content": 5}]}, None,
         "invalid_type", "messages[0].content"),
        ("an input_audio part", {"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "", "format":
                                                    "wav"}}]}]}, None,
         "invalid_content_part", "messages[0].content[0].type"),
        ("n = 2", dict(user(), n=2), None, "invalid_value", "n"),
        ("logprobs", dict(user(), logprobs=True), None, "invalid_value",
         "logprobs"),
        ("audio output", dict(user(), modalities=["text", "audio"]), None,
         "invalid_value", "modalities"),
        ("a custom tool", dict(user(), tools=[{"type": "custom", "name":
                                               "apply_patch"}]), None,
         "invalid_value", "tools[0]"),
        ("tool_choice allowed_tools", dict(user(), tool_choice={
            "type": "allowed_tools", "allowed_tools": {}}), None,
         "invalid_value", "tool_choice"),
        ("max_tokens 0", dict(user(), max_tokens=0), None, "invalid_value",
         "max_tokens"),
    ]
    with upstream([]):
        for label, body, raw, code, param in cases:
            st, _h, d, _ = post(body, raw=raw)
            check(st == 400 and is_error_object(d, 400,
                                                "invalid_request_error", code)
                  and d["error"]["param"] == param,
                  f"{label}: 400 invalid_request_error, code {code}, param "
                  f"{param}", f"{st} {json.dumps(d)[:300]}")
        check(not _chat, "none of them reached the model server",
              str(len(_chat)))


def test_fields_we_ignore_are_accepted():
    body = dict(user(), user="u-1", metadata={"a": "b"}, store=True, seed=7,
                service_tier="auto", safety_identifier="s",
                prediction={"type": "content", "content": "x"},
                verbosity="low", web_search_options={},
                temperature=0.2, top_p=0.5, n=1, parallel_tool_calls=True)
    with upstream([S.reply("pong")]):
        st, _h, d, _ = post(body)
    check(st == 200 and (d["choices"][0]["message"]["content"] == "pong"),
          "user, metadata, store, seed, service_tier, safety_identifier, "
          "prediction, verbosity, web_search_options, sampling and n=1 are "
          "accepted (ignored or overridden), not refused",
          f"{st} {json.dumps(d)[:300]}")


def test_routes_and_auth():
    # /v1/responses is served since 2026-09-26 (mcp/test_responses_api.py).
    for path in ("/v1/embeddings", "/v1/completions"):
        st, _h, d, _ = post({"input": "x"}, path=path)
        check(st == 404 and is_error_object(d, 404, "invalid_request_error",
                                            "unknown_url"),
              f"POST {path}: 404 unknown_url in the error object (it was "
              f"405 from the dashboard's GET catch-all)",
              f"{st} {json.dumps(d)[:200]}")
    r = CLIENT.get("/v1/nope", headers=H)
    check(r.status_code == 404 and is_error_object(r.json(), 404),
          "GET /v1/nope: 404 in the error object", f"{r.status_code} "
          f"{r.text[:200]}")
    r = CLIENT.get("/v1/chat/completions", headers=H)
    check(r.status_code == 405 and "POST" in r.headers.get("allow", "")
          and is_error_object(r.json(), 405),
          "GET on the chat route keeps its 405, with Allow: POST",
          f"{r.status_code} {r.headers.get('allow')} {r.text[:200]}")
    r = CLIENT.get("/v1/models")
    check(r.status_code == 200 and r.json()["data"],
          "GET /v1/models still served", f"{r.status_code}")
    r = CLIENT.post("/v1/chat/completions", json=user(),
                    headers={"Authorization": "Bearer nope"})
    check(r.status_code == 401 and is_error_object(
        r.json(), 401, "invalid_request_error", "invalid_api_key"),
        "a bad key: 401 with all four fields", r.text[:200])


def test_upstream_errors_keep_their_class():
    cases = [
        ("llama-server 400 invalid_request_error",
         [err_obj(400, "invalid_request_error", "Field 'x' is wrong")], 400,
         "invalid_request_error", None, "Field 'x' is wrong"),
        ("llama-server 501 not_supported_error",
         [err_obj(501, "not_supported_error", "no logprobs with tools")], 400,
         "invalid_request_error", "unsupported_request", "no logprobs"),
        ("llama-server 400 exceed_context_size_error",
         [err_obj(400, "exceed_context_size_error",
                  "request (170000 tokens) exceeds the available context "
                  "size (163840 tokens), try increasing it",
                  n_prompt_tokens=170000, n_ctx=163840)], 400,
         "invalid_request_error", "context_length_exceeded",
         "maximum context length is"),
        ("llama-swap 503 (model loading), twice",
         [err_obj(503, "unavailable_error", "Loading model"),
          err_obj(503, "unavailable_error", "Loading model")], 503,
         "service_unavailable_error", "model_unavailable", "Loading model"),
        ("llama-server 500, twice",
         [err_obj(500, "server_error", "boom"),
          err_obj(500, "server_error", "boom")], 502, "server_error",
         "upstream_error", "boom"),
    ]
    for label, acts, status, typ, code, text in cases:
        for stream in (False, True):
            with upstream(acts):
                st, h, d, ev = post(dict(user(), stream=stream),
                                    stream=stream)
            check(st == status and ev is None
                  and is_error_object(d, status, typ, code)
                  and text in d["error"]["message"]
                  and "application/json" in h.get("content-type", ""),
                  f"{label} ({'streamed' if stream else 'blocking'}): HTTP "
                  f"{status} {typ} {code} before any byte, never a 200 with "
                  f"the error in the content",
                  f"{st} {json.dumps(d or ev)[:300]}")
    with upstream([err_obj(503, "unavailable_error", "Loading model")] * 2):
        st, h, d, _ = post(user())
    check(h.get("retry-after") == "30",
          "a 503 carries Retry-After", str(dict(h)))
    with upstream([], url="http://127.0.0.1:9"):
        st, h, d, _ = post(user())
    check(st == 503 and is_error_object(d, 503, "service_unavailable_error",
                                        "model_unavailable"),
          "an unreachable model server: 503 service_unavailable_error",
          f"{st} {json.dumps(d)[:300]}")


# ======================================================= 2. the stream ======

def _no_error_in_content(ev: list) -> bool:
    text = "".join(((e.get("choices") or [{}])[0].get("delta") or {})
                   .get("content") or "" for e in ev if isinstance(e, dict)
                   and e.get("choices"))
    return "error" not in text.lower() and "[upstream" not in text


def test_a_failure_after_the_first_byte_is_one_error_event():
    # Hop 0 calls OUR tool (its reasoning and the tool line commit the
    # stream), hop 1 is refused by the model server.
    acts = [S.reply(reasoning="need a picture", calls=[S.img()]),
            err_obj(400, "invalid_request_error", "Field 'y' is wrong")]
    with upstream(acts):
        st, _h, _d, ev = post(dict(user(), stream=True), stream=True)
    ev = ev or []
    errs = [e for e in ev if isinstance(e, dict) and e.get("error")]
    check(st == 200 and len(errs) == 1 and ev[-1] == "[DONE]"
          and ev[-2] is errs[0]
          and is_error_object(errs[0], 400, "invalid_request_error")
          and "Field 'y'" in errs[0]["error"]["message"],
          "after the first byte: ONE `data: {\"error\": ...}` event, then "
          "[DONE]", json.dumps(ev)[-500:])
    check(_no_error_in_content(ev),
          "and the error is never assistant content",
          json.dumps(ev)[:300])
    chunks = [e for e in ev if isinstance(e, dict) and e.get("choices")]
    check(not any(c["choices"][0].get("finish_reason") for c in chunks),
          "no finish_reason is claimed for a turn that failed "
          "(the error event is the last word)", json.dumps(chunks)[-300:])
    # llama-server's in-stream error event, after reasoning went out.
    with upstream([{"raw": mid_stream_error("slot killed")}]):
        st, _h, _d, ev = post(dict(user(), stream=True), stream=True)
    ev = ev or []
    errs = [e for e in ev if isinstance(e, dict) and e.get("error")]
    check(st == 200 and len(errs) == 1 and ev[-1] == "[DONE]"
          and is_error_object(errs[0], 502, "server_error", "upstream_error")
          and "slot killed" in errs[0]["error"]["message"]
          and _no_error_in_content(ev),
          "an in-stream upstream error after reasoning went out: the error "
          "event (502 server_error), not content", json.dumps(ev)[-400:])


def test_the_sdk_raises_on_the_error_event():
    """openai-python's Stream.__stream__ (oaispec py_streaming.py) raises
    APIError when an event's data carries `error`. Replayed here on our
    bytes: the same rule, applied to what we send."""
    acts = [S.reply(reasoning="r", calls=[S.img()]),
            err_obj(400, "exceed_context_size_error",
                    "request (170000 tokens) exceeds the available context "
                    "size (163840 tokens)", n_prompt_tokens=170000,
                    n_ctx=163840)]
    with upstream(acts):
        raw = b"".join(proxy.stream_body(dict(user(), stream=True)))
    raised = None
    for block in raw.decode().split("\n\n"):
        if not block.startswith("data: ") or block[6:] == "[DONE]":
            continue
        data = json.loads(block[6:])
        if isinstance(data, dict) and data.get("error"):
            raised = data["error"]
            break
    check(raised is not None and raised.get("code") == "context_length_exceeded"
          and "maximum context length is" in raised.get("message", ""),
          "the SDK's rule finds the error event; Hermes classifies a "
          "status-less error by its code (context_length_exceeded -> "
          "compress)", json.dumps(raised)[:300])


def test_a_committed_stream_during_deep_thinking_still_ends_in_an_event():
    """Deep thinking's heartbeats commit the stream (the long pre-main work
    keeps its keep-alive); main's refusal after them is the error event."""
    srv, url = S._slow_server()
    saved = (S._model.UPSTREAM, proxy.HEARTBEAT)
    S._model.UPSTREAM, proxy.HEARTBEAT = url, 0.05
    S._Slow.HOLD = 0.4
    try:
        with upstream([err_obj(400, "invalid_request_error", "bad field")]):
            S.SELECTION["investigate"] = True
            body = {"model": "yamadori", "stream": True, "messages": [
                {"role": "user", "content": S.Q_LATHE}]}
            chunks = list(proxy.stream_body(body, "yamadori"))
    finally:
        S._model.UPSTREAM, proxy.HEARTBEAT = saved
        S._Slow.HOLD = 4.0
        srv.shutdown()
        S.SELECTION.clear()
    beats = sum(1 for b in chunks if S._is_heartbeat(b))
    ev = sse_events(b"".join(chunks).decode())
    errs = [e for e in ev if isinstance(e, dict) and e.get("error")]
    check(beats >= 1 and len(errs) == 1 and ev[-1] == "[DONE]"
          and "bad field" in errs[0]["error"]["message"],
          "heartbeats first, then one error event and [DONE]",
          f"{beats} heartbeats; {json.dumps(ev)[-300:]}")


def test_a_normal_stream_ends_on_finish_then_done():
    with upstream([S.reply("pong")]):
        st, _h, _d, ev = post(dict(user(), stream=True), stream=True)
    ev = ev or []
    chunks = [e for e in ev if isinstance(e, dict)]
    check(st == 200 and ev and ev[-1] == "[DONE]" and chunks
          and chunks[-1]["choices"][0]["finish_reason"] == "stop"
          and not any("usage" in c for c in chunks),
          "no include_usage: the last chunk before [DONE] carries "
          "finish_reason, and no usage is sent unasked",
          json.dumps(chunks[-1:])[:300])


# ======================================================== 3. the context ====

class window:
    """tiers._shares() pinned: main = `main` tokens."""

    def __init__(self, main: int):
        self.main = main

    def __enter__(self):
        self.saved = tiers._shares
        tiers._shares = lambda: {"main": self.main, "helper": 1000,
                                 "pool": self.main + 1000}

    def __exit__(self, *a):
        tiers._shares = self.saved
        return False


CLIENT_TOOL = {"type": "function", "function": {
    "name": "read_file", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}}}}


def test_the_client_prompt_past_the_window_is_refused():
    words = " ".join(["word"] * 4000)
    with window(5000), upstream([S.reply("never")]):
        try:
            proxy.complete(dict(user(words), tools=[CLIENT_TOOL]))
            got = None
        except api_errors.ApiError as e:
            got = e
        counted = list(_upstream_calls)
        gens = len(_chat)
    b = got.body()["error"] if got else {}
    n = b.get("n_prompt_tokens")
    check(got is not None and got.status == 400
          and b.get("code") == "context_length_exceeded"
          and b.get("type") == "invalid_request_error"
          and b.get("param") == "messages" and n and n > 4000,
          "400 context_length_exceeded, param messages, with the model "
          "server's own count", json.dumps(b)[:400])
    check(any(p.endswith("/apply-template") for p in counted)
          and any(p.endswith("/tokenize") for p in counted) and gens == 0,
          "counted by the served template + tokenizer; nothing was generated",
          json.dumps({"calls": counted, "generations": gens}))
    msg = (b.get("message") or "").lower()
    avoid = ("requested", "in the completion", "must be", "should be",
             "limited to", "available tokens", "max_tokens", "not supported",
             "unsupported", "unknown parameter", "think", "reasoning")
    check("maximum context length is 5000 tokens" in msg
          and f"resulted in {n} tokens" in msg
          and "reduce the length" in msg
          and not any(a in msg for a in avoid),
          "OpenAI's wording, which Hermes matches as an overflow (and parses "
          "the window from); none of the phrases it reads as an output cap "
          "or a bad parameter", b.get("message", ""))
    with window(5000), upstream([S.reply("never")]):
        st, _h, d, ev = post(dict(user(words), stream=True), stream=True)
    check(st == 400 and ev is None and is_error_object(
        d, 400, "invalid_request_error", "context_length_exceeded"),
        "streamed: a real HTTP 400 before any byte", f"{st} "
        f"{json.dumps(d)[:200]}")


def test_the_threshold_is_the_count_plus_the_floor():
    # Long enough that the high estimate sends it to the count.
    body = dict(user(" ".join(["w"] * 5000)), tools=[CLIENT_TOOL])
    floor = tiers.A_MIN          # the stub payload does not think
    real = proxy.count_prompt_tokens
    out = {}
    for n in (5000 - floor, 5000 - floor + 1):
        proxy.count_prompt_tokens = lambda payload, n=n, **k: n
        try:
            with window(5000), upstream([S.reply("ok")]):
                try:
                    d = proxy.complete(dict(body))
                    out[n] = ("ok", d["x_yamadori"]["context"],
                              _chat[0] if _chat else {})
                except api_errors.ApiError as e:
                    out[n] = ("refused", e.body()["error"], None)
        finally:
            proxy.count_prompt_tokens = real
    fit, over = out[5000 - floor], out[5000 - floor + 1]
    check(fit[0] == "ok" and fit[1].get("counted") is True
          and fit[1].get("tokens") == 5000 - floor
          and over[0] == "refused" and over[1]["code"] ==
          "context_length_exceeded",
          f"prompt + {floor} (the generation floor) == window: served; one "
          f"token more: refused", json.dumps({k: v[:2] for k, v in
                                              out.items()})[:400])
    sent = fit[2] or {}
    check(sent.get("tools") and sent["tools"][0]["function"]["name"] ==
          "read_file"
          and not any(m.get("content") == proxy.LANDING_PROMPT
                      for m in sent.get("messages") or [])
          and int(sent.get("max_tokens") or 0) <= floor,
          "at the edge the client's tools go up untouched, nothing is "
          "appended, and max_tokens is cut to what the window leaves",
          json.dumps({"tools": [t["function"]["name"] for t in
                                sent.get("tools") or []],
                      "max_tokens": sent.get("max_tokens")}))


def test_far_from_the_edge_nothing_is_counted():
    with upstream([S.reply("ok")]):
        d = proxy.complete(user("short"))
    c = d["x_yamadori"]["context"]
    check(c["counted"] is False and not any(
        p.endswith("/apply-template") for p in _upstream_calls)
          and c["limit"] == catalog.context_window(),
          "a prompt the high estimate clears costs no count; the limit is "
          "the advertised context_length", json.dumps(c))


def test_hop_zero_is_never_landed_but_our_hops_are():
    prev = proxy.context_full
    proxy.context_full = lambda *a, **k: True       # "always full"
    try:
        with upstream([S.reply(calls=[S.img()]), S.reply("answered")]):
            d = proxy.complete(dict(user("draw"), tools=[CLIENT_TOOL]))
            seen = list(_chat)
    finally:
        proxy.context_full = prev
    h0 = seen[0] if seen else {}
    h1 = seen[1] if len(seen) > 1 else {}
    check(any(t["function"]["name"] == "read_file"
              for t in h0.get("tools") or [])
          and not any(m.get("content") == proxy.LANDING_PROMPT
                      for m in h0.get("messages") or []),
          "hop 0 (the client's own request) keeps the client's tools and "
          "gets no appended user message, even when context_full says full",
          json.dumps([t["function"]["name"] for t in h0.get("tools") or []]))
    check(h1.get("tools") in (None, []) and any(
        m.get("content") == proxy.LANDING_PROMPT
        for m in h1.get("messages") or [])
          and d["choices"][0]["message"]["content"] == "answered",
          "hop 1, after OUR tool ran, still lands (our additions overflowed)",
          json.dumps({"tools": h1.get("tools")})[:200])


def test_the_window_advertised_is_the_window_enforced():
    card = catalog.public_list()["data"][0]
    check(card.get("context_length") == proxy.window_limit({})
          == int(budget.budgets()["main"]),
          "/v1/models context_length == the limit check_client_prompt "
          "enforces == the main share", json.dumps(
              {"card": card.get("context_length"),
               "enforced": proxy.window_limit({})}))


# ========================================================== 4. usage ========

def test_usage_is_the_final_generation():
    acts = [S.reply(reasoning="need it", calls=[S.img()],
                    usage={"prompt_tokens": 900, "completion_tokens": 40,
                           "total_tokens": 940}),
            S.reply("done", reasoning="one two three four",
                    usage={"prompt_tokens": 1000, "completion_tokens": 30,
                           "total_tokens": 1030,
                           "prompt_tokens_details": {"cached_tokens": 950}})]
    with upstream(acts):
        d = proxy.complete(user("draw it"))
    u = d.get("usage") or {}
    x = (d.get("x_yamadori") or {}).get("usage") or {}
    check(u.get("prompt_tokens") == 1000 and u.get("completion_tokens") == 30
          and u.get("total_tokens") == 1030 and "hops" not in u,
          "prompt_tokens is the FINAL generation's (1000), not the sum "
          "(1900) Hermes read as the context size; no usage.hops",
          json.dumps(u))
    check(u.get("prompt_tokens_details") == {"cached_tokens": 950}
          and u.get("completion_tokens_details") == {"reasoning_tokens": 4},
          "cached_tokens from the model server; reasoning_tokens counted by "
          "its tokenizer (4 words in the fake)", json.dumps(u))
    check(x.get("generations") == 2 and x.get("summed") == {
        "prompt_tokens": 1900, "completion_tokens": 70, "total_tokens": 1970}
          and x.get("final") == "hop 1"
          and [g["prompt_tokens"] for g in x.get("per_generation") or []]
          == [900, 1000] and d["x_yamadori"]["hops"] == 2,
          "x_yamadori.usage keeps the per-hop numbers and the sums",
          json.dumps(x)[:400])


def test_usage_details_fall_back_and_stay_honest():
    gens = [{"usage": {"prompt_tokens": 500, "completion_tokens": 20},
             "cache": {"reused": 480, "prompt": 500},
             "reasoning": "HANDOFF TEXT\nmine now", "prefill_reasoning":
             "HANDOFF TEXT", "which": "hop 0"}]
    real = proxy._count_text_tokens
    seen = []
    proxy._count_text_tokens = lambda model, text: seen.append(text) or \
        len(text.split())
    try:
        u, rec = proxy._usage_of(gens, {"prompt_tokens": 500,
                                        "completion_tokens": 20,
                                        "total_tokens": 520}, 1, {})
    finally:
        proxy._count_text_tokens = real
    check(u["prompt_tokens_details"] == {"cached_tokens": 480}
          and u["completion_tokens_details"] == {"reasoning_tokens": 2}
          and seen == ["\nmine now"],
          "cached_tokens from the slot's timings when the server's usage "
          "has none; a prefill's echoed reasoning is not counted as "
          "generated", json.dumps({"usage": u, "counted": seen}))
    proxy._count_text_tokens = lambda model, text: None
    try:
        u, rec = proxy._usage_of(gens, {}, 1, {})
    finally:
        proxy._count_text_tokens = real
    check("completion_tokens_details" not in u
          and "absent" in (rec.get("reasoning_tokens") or ""),
          "reasoning_tokens is ABSENT (never guessed) when the tokenizer "
          "does not answer, and x_yamadori says why", json.dumps(rec)[:200])


def test_the_streamed_usage_chunk():
    acts = [S.reply(reasoning="need it", calls=[S.img()],
                    usage={"prompt_tokens": 900, "completion_tokens": 40,
                           "total_tokens": 940}),
            S.reply("done", usage={"prompt_tokens": 1000,
                                   "completion_tokens": 30,
                                   "total_tokens": 1030})]
    with upstream(acts):
        st, _h, _d, ev = post(dict(user(), stream=True,
                                   stream_options={"include_usage": True}),
                              stream=True)
    ev = ev or []
    fin = [e for e in ev if isinstance(e, dict) and e.get("choices")
           and e["choices"][0].get("finish_reason")]
    last = ev[-2] if len(ev) >= 2 else {}
    check(st == 200 and ev[-1] == "[DONE]" and isinstance(last, dict)
          and last.get("choices") == [] and last.get("usage", {}).get(
              "prompt_tokens") == 1000 and fin and "usage" not in fin[-1]
          and fin[-1].get("x_yamadori"),
          "include_usage: finish chunk (with x_yamadori), then a chunk with "
          "choices [] and the final generation's usage, then [DONE]",
          json.dumps(ev[-3:])[:500])


TESTS = [test_request_validation_is_a_400_object,
         test_fields_we_ignore_are_accepted,
         test_routes_and_auth,
         test_upstream_errors_keep_their_class,
         test_a_failure_after_the_first_byte_is_one_error_event,
         test_the_sdk_raises_on_the_error_event,
         test_a_committed_stream_during_deep_thinking_still_ends_in_an_event,
         test_a_normal_stream_ends_on_finish_then_done,
         test_the_client_prompt_past_the_window_is_refused,
         test_the_threshold_is_the_count_plus_the_floor,
         test_far_from_the_edge_nothing_is_counted,
         test_hop_zero_is_never_landed_but_our_hops_are,
         test_the_window_advertised_is_the_window_enforced,
         test_usage_is_the_final_generation,
         test_usage_details_fall_back_and_stay_honest,
         test_the_streamed_usage_chunk]


def main() -> int:
    for fn in TESTS:
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail[:400]}" if not ok and detail else ""))
    _srv.shutdown()
    S._srv.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}\n  {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
