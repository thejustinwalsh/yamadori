#!/usr/bin/env python
"""The production HTTP surface. ASGI, not the stdlib demo server.

WHY THIS REPLACES THE HAND-ROLLED SERVER

The proxy ran on `BaseHTTPRequestHandler` + `ThreadingHTTPServer`, which is
Python's teaching example, not a server. What that cost, measured rather than
supposed:

  keep-alive           It advertises HTTP/1.1, so clients reuse the socket.
                       Between two generations that is minutes, the server
                       drops the connection, and the reused socket fails as a
                       502 the caller reads as a model error. 8 of 16 runs
                       died this way while the identical request on a fresh
                       connection succeeded every time.
  HEAD                 `501 Unsupported method ('HEAD')`. Health checkers and
                       load balancers use HEAD.
  identity             `Server: BaseHTTP/0.6 Python/3.13.15` in every reply.
  thread per socket    One OS thread per connection, no backpressure, and a
                       slow client occupies a thread for the whole generation.
  shutdown             No graceful drain. Restarting mid-run killed in-flight
                       requests, which is how two benchmark runs were lost.

None of that is exotic. It is what a real server does for free, and every hour
spent debugging it was an hour not spent measuring the product.

WHAT IS REUSED, AND WHY

All of it. `prepare`, `complete`, the tool loop, tiers, the catalogue, the
dashboard and the secret filtering are unchanged and imported. Those are the
parts that took a day to get right; the transport is the part that was wrong.
Rewriting the loop as async at the same time would mean changing two things at
once and not knowing which broke.

The loop stays synchronous and runs in a worker thread. It spends its life
blocked on a 30-to-200-second upstream call, so the event loop is free either
way, and `run_in_threadpool` keeps it off the loop without touching it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, Response  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

import accounts  # noqa: E402
import admission  # noqa: E402
import api_errors  # noqa: E402
import catalog  # noqa: E402
import dash_data  # noqa: E402
import dash_skills  # noqa: E402
import dash_mcp  # noqa: E402
import dash_harness  # noqa: E402
import dash_jjava  # noqa: E402
import dash_perf  # noqa: E402
import dash_tokens  # noqa: E402
import dash_static  # noqa: E402
import dash_vitals  # noqa: E402
import dashboard  # noqa: E402
import cancel  # noqa: E402
import images  # noqa: E402
import jev_api  # noqa: E402
import max_mode  # noqa: E402
import nebari  # noqa: E402
import messages_api  # noqa: E402
import proxy  # noqa: E402
import response_store  # noqa: E402
import responses_api  # noqa: E402

app = FastAPI(title="yamadori", version="1", docs_url=None, redoc_url=None)
# A request body over image_input.max_body_bytes() (64 MB, a choice) is a 413
# in OpenAI's envelope, before a byte of it is read (mcp/body_limit.py;
# docs/VISION.md 5e). Every POST route, so /v1/responses too.
import body_limit  # noqa: E402
app.add_middleware(body_limit.BodyLimit)

HOST = os.environ.get("YAMADORI_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("YAMADORI_PROXY_PORT", "1234"))

# Which dashboard a browser gets: the committed React build (web/dist) at the
# site root, the Python pages at /dash, or a pointer to Vite in dev. See
# mcp/dash_static.py. When React owns the root the Python pages move to
# /dash/classic*, where the two write flows React does not have yet (recipe
# review, dataset submission) stay reachable, and old /dash bookmarks
# redirect to the root.
DASH_UI = dash_static.ui_mode()
_PY = "" if DASH_UI == "python" else "/classic"


@app.exception_handler(StarletteHTTPException)
async def openai_shaped_errors(request: Request, exc: StarletteHTTPException):
    """Errors in OpenAI's envelope, not FastAPI's.

    A 404 on `/` or `/v1` is correct -- neither is an endpoint in the spec --
    but the BODY was FastAPI's default `{"detail": "Not Found"}`. Clients
    written against the OpenAI API parse `error.message` / `error.type`, so
    ours read as a malformed response rather than a clean 404, and a client
    that surfaces `error.message` showed nothing at all.
    """
    detail = exc.detail if isinstance(exc.detail, str) else "error"
    kind = {401: "invalid_request_error", 403: "permission_error",
            404: "invalid_request_error", 429: "rate_limit_error"}.get(
                exc.status_code, "api_error")
    return JSONResponse(status_code=exc.status_code, content={"error": {
        "message": detail, "type": kind, "param": None,
        "code": None if exc.status_code >= 500 else exc.status_code}})


@app.get("/")
async def root(request: Request) -> Response:
    """Not in the spec, but a bare GET is the first thing anyone tries.

    Content-negotiated. A browser (Accept: text/html) gets the dashboard,
    which lives at the root; curl, SDKs and scripts send no text/html and get
    this JSON descriptor exactly as before.
    """
    if DASH_UI != "python" and dash_static.wants_html(request.headers.get("accept")):
        return dash_static.index_response(
            DASH_UI, build="ok" if DASH_BUILD.get("ok") else "stale")
    ui = "/" if DASH_UI != "python" else "/dash"
    return JSONResponse({
        "service": "yamadori",
        "endpoints": ["/v1/models", "/v1/models/{id}", "/v1/chat/completions",
                      "/v1/responses", "/v1/responses/{id}",
                      "/v1/responses/{id}/input_items", "/v1/messages",
                      "/v1/messages/count_tokens", "/v1/images/generations",
                      "/v1/systemone", "/jev/v1/systemone", "/jev/v1/models",
                      "/health",
                      ui,
                      f"/dash{_PY}/data", f"/dash{_PY}/vitals"],
        "dashboard": DASH_UI,
        "note": "OpenAI-compatible. Point any client here; it needs no tool "
                "configuration.",
    }, headers={"Vary": "Accept"})


def _too_many(e: "admission.Full", messages=None) -> JSONResponse:
    """A refusal, not a timeout.

    Three benchmark processes once ran against this proxy at once because
    nothing stopped them, and they degraded each other into 502s that took an
    hour to attribute to concurrency rather than to the arm that happened to
    lose most. One model over one shared KV pool does not go faster when it is
    given more work; it goes wrong in a way that looks like a different bug.
    """
    if messages is not None:
        return _anthropic_error(api_errors.ApiError(
            429, str(e), code="server_busy", headers={"Retry-After": "30"}))
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "30"},
        content={"error": {"message": str(e), "type": "rate_limit_error",
                           "param": None, "code": "server_busy"}})


def _unauthorised(why: str) -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": {
        "message": f"{why}. Set an API key in your client.",
        "type": "invalid_request_error", "param": None,
        "code": "invalid_api_key"}})


def _api_error(e: "api_errors.ApiError") -> JSONResponse:
    """ONE ERROR PATH (mcp/api_errors.py): the status and OpenAI's object."""
    return JSONResponse(status_code=e.status, content=e.body(),
                        headers=e.headers or None)


def _anthropic_error(e: "api_errors.ApiError") -> JSONResponse:
    """The same error in Anthropic's envelope (mcp/messages_api.py), for the
    /v1/messages routes."""
    status, body, headers = messages_api.anthropic_error(e)
    return JSONResponse(status_code=status, content=body, headers=headers)


def _identify_anthropic(request: Request) -> tuple[str | None, str]:
    """An Anthropic client's key: `Authorization: Bearer` (Claude Code's
    ANTHROPIC_AUTH_TOKEN) or `x-api-key` (ANTHROPIC_API_KEY), either one an
    account key of ours. Tried in that order; the first that names an
    account wins."""
    auth = request.headers.get("authorization")
    who, why = accounts.identify(auth) if auth else (None, "no API key supplied")
    if who is not None:
        return who, why
    xk = (request.headers.get("x-api-key") or "").strip()
    if xk:
        return accounts.identify(f"Bearer {xk}")
    return who, why


@app.get("/health")
@app.head("/health")
async def health() -> JSONResponse:
    """GET and HEAD. The old server answered HEAD with a 501.

    Deliberately trivial. This is a liveness route, and the watchdog restarts
    the stack on what it says, so it must not depend on the model, the
    catalogue, the GPU or anything that can be slow or busy. It answers in
    microseconds or the process is gone, and those are the only two outcomes
    worth reporting here.
    """
    import repos
    return JSONResponse({"ok": True, "repos": len(repos.known()),
                         "server": "yamadori"})


@app.get("/v1/models")
async def models() -> JSONResponse:
    # In a thread: the context window reads the pool size from llama-server
    # (/props, once, then cached -- mcp/budget.py), which must not stall the
    # event loop on the first call.
    return JSONResponse(await run_in_threadpool(catalog.public_list))


@app.get("/v1/models/{model_id:path}")
async def model_detail(model_id: str) -> JSONResponse:
    # OpenAI's retrieve-model route; harnesses probe it for one model's
    # window. Never a 404: see catalog.model_card. There is deliberately no
    # llama.cpp /props, Ollama /api/show or LM Studio route beside it -- this
    # is an OpenAI-compatible proxy, and those would describe the raw server
    # (the whole KV pool, the model path) rather than a conversation's share.
    return JSONResponse(await run_in_threadpool(catalog.model_card, model_id))


@app.get(f"/dash{_PY}")
async def dash_page() -> Response:
    # The shell is public; it holds no data and a browser cannot set an
    # Authorization header on a page load. Everything under /dash/api is gated.
    return Response(content=dashboard.PAGE, media_type="text/html; charset=utf-8")


@app.get(f"/dash{_PY}/data")
async def dash_data_page() -> Response:
    # Same deal as /dash: the shell is public and carries no data of its own.
    # Every dataset, job count and recipe row on it arrives from the gated
    # /dash/api/datasets and /dash/api/recipes.
    code, ctype, payload = dash_data.handle_get("/dash/data")
    return Response(content=payload, status_code=code, media_type=ctype)


@app.get(f"/dash{_PY}/vitals")
async def dash_vitals_page() -> Response:
    # Same deal as /dash: the shell is public and carries no telemetry, the
    # JSON behind it is gated. Declared before the /dash/api catch-all only
    # for readability; the paths do not overlap.
    code, ctype, payload = dash_vitals.handle_get("/dash/vitals")
    return Response(content=payload, status_code=code, media_type=ctype)


# The classic benchmark page (/dash/classic/results) was retired 2026-09-30
# with the React one: PERFORMANCE (/performance, /dash/api/perf) replaces it.


# The caller's own settings. Declared BEFORE the /dash/api catch-all, which
# would otherwise own the GET (Starlette matches in registration order). The
# account is the one the caller's key resolves to -- never a field of the
# body or the path -- so a key reads and writes only its own account.
@app.get("/dash/api/settings/image")
async def settings_image_get(request: Request) -> Response:
    who, why = accounts.identify(request.headers.get("authorization"))
    if who is None:
        return _unauthorised(why)
    return JSONResponse(await run_in_threadpool(images.settings, who))


@app.put("/dash/api/settings/image")
async def settings_image_put(request: Request) -> Response:
    who, why = accounts.identify(request.headers.get("authorization"))
    if who is None:
        return _unauthorised(why)
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return JSONResponse(status_code=400, content={"ok": False,
                                                      "error": f"bad request: {e}"})
    code, out = await run_in_threadpool(images.set_setting, who, body)
    return JSONResponse(status_code=code, content=out)


@app.get("/dash/api/{rest:path}")
async def dash_get(rest: str, request: Request) -> Response:
    who, why = accounts.identify(request.headers.get("authorization"))
    if who is None:
        return _unauthorised(why)
    path = f"/dash/api/{rest}"
    # One gate, two handlers: the catch-all above already owns every path
    # under /dash/api, so a second @app.get for /dash/api/vitals would never
    # be reached. Dispatch by asking each module in turn instead.
    hit = await run_in_threadpool(dashboard.handle_get, path)
    if not hit:
        hit = await run_in_threadpool(dash_vitals.handle_get, path)
    if not hit:
        hit = await run_in_threadpool(dash_data.handle_get, path)
    if not hit:
        hit = await run_in_threadpool(dash_skills.handle_get, path)
    if not hit:
        # Tokens, electricity and the hosted-API comparison (mcp/dash_tokens.py).
        hit = await run_in_threadpool(dash_tokens.handle_get, path)
    if not hit:
        # The MCP servers the proxy hosts (mcp/dash_mcp.py): read-only.
        hit = await run_in_threadpool(dash_mcp.handle_get, path)
    if not hit:
        # HARNESS TOOLS: the harness kit entries and the per-harness export
        # (mcp/dash_harness.py).
        hit = await run_in_threadpool(dash_harness.handle_get, path)
    if not hit:
        # JJAVA and PERFORMANCE (mcp/dash_jjava.py, mcp/dash_perf.py):
        # read-only, model-free (2026-09-30).
        hit = await run_in_threadpool(dash_jjava.handle_get, path)
    if not hit:
        hit = await run_in_threadpool(dash_perf.handle_get, path)
    if not hit:
        return JSONResponse(status_code=404, content={"error": "not found"})
    code, ctype, payload = hit
    return Response(content=payload, status_code=code, media_type=ctype)


@app.post("/dash/api/{rest:path}")
async def dash_post(rest: str, request: Request) -> Response:
    who, why = accounts.identify(request.headers.get("authorization"))
    if who is None:
        return _unauthorised(why)
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return JSONResponse(status_code=400, content={"error": f"bad request: {e}"})
    path = f"/dash/api/{rest}"
    # Same dispatch chain as the GET, and for the same reason: the catch-all
    # above owns every path under /dash/api, so a second @app.post for a
    # nested api path would never be reached.
    hit = await run_in_threadpool(dashboard.handle_post, path, body)
    if not hit:
        hit = await run_in_threadpool(dash_data.handle_post, path, body)
    if not hit:
        # The skills API records the caller's account (a hash prefix, never
        # the key) as the author of an edit or a disable.
        hit = await run_in_threadpool(dash_skills.handle_post, path, body, who)
    if not hit:
        hit = await run_in_threadpool(dash_harness.handle_post, path, body, who)
    if not hit:
        return JSONResponse(status_code=404, content={"error": "not found"})
    code, ctype, payload = hit
    return Response(content=payload, status_code=code, media_type=ctype)


# Session headers, first present wins: ours, then the harness's own session
# id (OpenCode: X-Session-Id / x-session-affinity, docs/HARNESS-OPENCODE.md).
SESSION_HEADERS = ("x-yamadori-session", "x-session-id", "x-session-affinity")


def session_of_headers(headers) -> tuple[str, str]:
    """(token, header name) of the first well-formed session header, else
    ("", "")."""
    for name in SESSION_HEADERS:
        tok = nebari.session_token(headers.get(name))
        if tok:
            return tok, name
    return "", ""

@app.post("/v1/chat/completions")
async def chat(request: Request) -> Response:
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _unauthorised(why)
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return _api_error(api_errors.invalid(
            f"The request body is not valid JSON: {e}", code="invalid_json"))
    # The body's shape, before admission (mcp/api_errors.validate_chat): a
    # client's mistake is its 400, never our 5xx after a queue wait.
    bad = api_errors.validate_chat(body)
    if bad is not None:
        return _api_error(bad)
    return await _serve_turn(request, account, body,
                             body.get("model") or "yamadori")


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    """The OpenAI Responses API (mcp/responses_api.py): the request is
    translated to the chat body, the SAME turn runs (_serve_turn ->
    proxy.complete / proxy.stream_body), and its result is translated back --
    a Response object, or the Responses event stream. A request that did not
    say `store: false` is stored under the account (mcp/response_store.py)
    and `previous_response_id` rebuilds a conversation from it."""
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _unauthorised(why)
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return _api_error(api_errors.invalid(
            f"The request body is not valid JSON: {e}", code="invalid_json"))
    try:
        # In a thread: a previous_response_id reads the response store.
        chat_body, ctx = await run_in_threadpool(
            responses_api.to_chat, body, "yamadori", account)
    except Exception as e:                                       # noqa: BLE001
        # A request we cannot serve is its 400 (ApiError); anything else is
        # our fault, and says so (500 internal_error), in the one object.
        return _api_error(api_errors.of_exception(e))
    return await _serve_turn(request, account, chat_body, ctx.model,
                             responses=ctx)


# STORED RESPONSES (mcp/response_store.py; operator, 2026-10-06): OpenAI's
# retrieve, delete and list-input-items routes. Another account's id is as
# "not found" as an unknown one. Errors: 404, invalid_request_error,
# "Response with id '<id>' not found." (OpenAI's wording, as its SDKs print it;
# the reference does not document the body).
def _response_not_found(rid: str) -> JSONResponse:
    return _api_error(api_errors.ApiError(
        404, f"Response with id '{rid}' not found."))


def _storing() -> bool:
    return response_store.enabled()


@app.get("/v1/responses/{response_id}")
async def response_get(response_id: str, request: Request) -> Response:
    """GET /v1/responses/{id}: the stored Response object. `stream=true` (a
    replay of the event stream) is not served."""
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _unauthorised(why)
    if request.query_params.get("stream", "").lower() in ("1", "true"):
        return _api_error(api_errors.invalid(
            "'stream' is not supported on retrieve: this server does not "
            "replay a stored response as events; read the object.",
            param="stream", code="unsupported_parameter"))
    obj = await run_in_threadpool(response_store.get, account,
                                  response_id) if _storing() else None
    if obj is None:
        return _response_not_found(response_id)
    return JSONResponse(responses_api.rehydrate_images(obj))


@app.delete("/v1/responses/{response_id}")
async def response_delete(response_id: str, request: Request) -> Response:
    """DELETE /v1/responses/{id}: {id, object: "response.deleted", deleted:
    true}."""
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _unauthorised(why)
    gone = await run_in_threadpool(response_store.delete, account,
                                   response_id) if _storing() else False
    if not gone:
        return _response_not_found(response_id)
    print(f"responses: deleted {response_id}", flush=True)
    return JSONResponse({"id": response_id, "object": "response.deleted",
                         "deleted": True})


@app.get("/v1/responses/{response_id}/input_items")
async def response_input_items(response_id: str, request: Request) -> Response:
    """GET /v1/responses/{id}/input_items: the response's own input items,
    newest first by default (`order` desc, OpenAI's default), `limit` 1-100
    (default 20), `after` an item id: {object: "list", data, first_id, last_id,
    has_more}."""
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _unauthorised(why)
    q = request.query_params
    try:
        limit = int(q.get("limit", "20"))
        if not 1 <= limit <= 100:
            raise ValueError
    except ValueError:
        return _api_error(api_errors.invalid(
            "'limit' must be an integer from 1 to 100.", param="limit",
            code="invalid_value"))
    order = q.get("order", "desc")
    if order not in ("asc", "desc"):
        return _api_error(api_errors.invalid(
            "'order' must be 'asc' or 'desc'.", param="order",
            code="invalid_value"))
    items = await run_in_threadpool(response_store.input_items, account,
                                    response_id) if _storing() else None
    if items is None:
        return _response_not_found(response_id)
    if order == "desc":
        items = items[::-1]
    after = q.get("after")
    if after:
        ids = [i.get("id") if isinstance(i, dict) else None for i in items]
        if after not in ids:
            return _api_error(api_errors.invalid(
                f"Item with id '{after}' not found.", param="after",
                code="invalid_value"))
        items = items[ids.index(after) + 1:]
    page = items[:limit]
    ids = [i.get("id") if isinstance(i, dict) else None for i in page]
    return JSONResponse({"object": "list", "data": page,
                         "first_id": ids[0] if ids else None,
                         "last_id": ids[-1] if ids else None,
                         "has_more": len(items) > limit})


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    """The Anthropic Messages API (mcp/messages_api.py), for Claude Code: the
    request is translated to the chat body, the SAME turn runs (_serve_turn
    -> proxy.complete / proxy.stream_body), and its result is translated
    back -- a Message, or Anthropic's event stream. Errors in Anthropic's
    envelope. Claude Code posts to /v1/messages?beta=true (the path
    matches)."""
    account, why = _identify_anthropic(request)
    if account is None:
        return _anthropic_error(messages_api.unauthorised(why))
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return _anthropic_error(api_errors.invalid(
            f"The request body is not valid JSON: {e}", code="invalid_json"))
    try:
        chat_body, ctx = messages_api.to_chat(body, request.headers, account)
    except Exception as e:                                       # noqa: BLE001
        return _anthropic_error(api_errors.of_exception(e))
    return await _serve_turn(request, account, chat_body, ctx.model,
                             messages=ctx)


@app.post("/v1/messages/count_tokens")
async def messages_count_tokens(request: Request) -> Response:
    """Anthropic's token count (Claude Code's /context): the model server's
    own count of the client's messages and tools (messages_api.count_tokens).
    Takes no main lane: it generates nothing."""
    account, why = _identify_anthropic(request)
    if account is None:
        return _anthropic_error(messages_api.unauthorised(why))
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return _anthropic_error(api_errors.invalid(
            f"The request body is not valid JSON: {e}", code="invalid_json"))
    try:
        out = await run_in_threadpool(messages_api.count_tokens, body,
                                      request.headers, account)
    except Exception as e:                                       # noqa: BLE001
        return _anthropic_error(api_errors.of_exception(e))
    return JSONResponse(out)


# THE JEV API (mcp/jev_api.py; operator, 2026-09-29: "the jjava api exact
# public api endpoints that match Jev exposed through our proxy"). TypeSafe's
# POST /v1/systemone and GET /v1/models, under /jev so a TypeSafe SDK works
# with base_url "<public base>/jev" -- our GET /v1/models stays OpenAI's --
# and POST /v1/systemone at the root too (it collides with nothing). Jev's
# statuses and the error body its SDK reads ({"detail": ...}); proxy.py is
# not involved. Declared before the /v1 and /jev catch-alls.
def _jev_response(status: int, content: dict, headers: dict) -> JSONResponse:
    return JSONResponse(status_code=status, content=content,
                        headers=headers or None)


def _jev_refusal(e: "jev_api.JevError", rid: str) -> JSONResponse:
    return _jev_response(e.status, e.body(),
                         dict(e.headers, **{jev_api.REQUEST_ID_HEADER: rid}))


@app.post("/jev/v1/systemone")
@app.post("/v1/systemone")
async def jev_systemone(request: Request) -> Response:
    rid = jev_api.new_request_id()
    route = request.url.path
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        # recorded (jev_api.refused): the JJAVA page counts 401s too
        return _jev_refusal(jev_api.refused(route, rid,
                                            jev_api.unauthorised(why)), rid)
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return _jev_refusal(jev_api.refused(route, rid, jev_api.json_invalid(e),
                                            account), rid)
    status, content, headers = await run_in_threadpool(
        jev_api.systemone, body, account, route=request.url.path,
        request_id=rid)
    return _jev_response(status, content, headers)


@app.get("/jev/v1/models")
async def jev_models(request: Request) -> Response:
    rid = jev_api.new_request_id()
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _jev_refusal(jev_api.refused(request.url.path, rid,
                                            jev_api.unauthorised(why)), rid)
    content = await run_in_threadpool(jev_api.models_call, account, rid)
    return _jev_response(200, content, {jev_api.REQUEST_ID_HEADER: rid})


@app.api_route("/jev/{rest:path}",
               methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
               include_in_schema=False)
async def jev_unknown(rest: str, request: Request) -> Response:
    """Every /jev path not served: 404 (or 405 for a served path by another
    method) in the Jev API's body, {"detail": ...}, not OpenAI's."""
    served = {"/jev/v1/systemone": "POST", "/jev/v1/models": "GET"}
    path = request.url.path.rstrip("/") or "/"
    rid = jev_api.new_request_id()
    if path in served:
        return _jev_response(405, {"detail": "Method Not Allowed"},
                             {"Allow": served[path],
                              jev_api.REQUEST_ID_HEADER: rid})
    return _jev_response(404, {"detail": "Not Found"},
                         {jev_api.REQUEST_ID_HEADER: rid})


def _complete_on(token, body: dict) -> dict:
    """proxy.complete under `token` (it binds the current token, or a new one): the blocking path's lease rides on
    it (THE OTHER CARD: a routed turn releases its main-model lease, max_mode.set_current)."""
    with cancel.bound(token):
        return proxy.complete(body)


async def _serve_turn(request: Request, account: str, body: dict,
                      public_name: str, responses=None,
                      messages=None) -> Response:
    """One turn, blocking or streamed, for /v1/chat/completions, (with
    `responses`, a responses_api.Ctx) /v1/responses and (with `messages`, a
    messages_api.Ctx) /v1/messages: admission, the retry supersede, the E1
    commit point and cancel-on-disconnect are the same for all three; only
    the rendering of the result -- and, for Messages, of the errors --
    differs."""
    error = _anthropic_error if messages is not None else _api_error
    # Experiment overrides ride on a header, so the body stays the OpenAI
    # schema and the override never becomes part of the cached prompt. A
    # translation's own (Messages: a Claude Code side request is a utility
    # call) merges under it: the caller's header wins, key by key.
    feats = request.headers.get("x-yamadori-features")
    if messages is not None and body.get("_features"):
        feats = messages_api.merge_features(body["_features"], feats)
    if feats:
        body["_features"] = feats
    # An explicit session token (nebari.key_of): benchmark rows whose first
    # messages are identical get independent sessions. Malformed -> ignored.
    # A harness's OWN session id is as explicit as ours: OpenCode sends
    # X-Session-Id / x-session-affinity with its `ses_...` id on every
    # request (docs/HARNESS-OPENCODE.md, 2026-09-26: without it an
    # OpenCode --fork carried the original's tool-call ids and merged into
    # the original's session). First present wins; utility calls stay
    # sessionless whatever they carry (proxy.session_context).
    body["_session_token"], body["_session_header"] = session_of_headers(
        request.headers)
    client = request.client
    body["_client_ip"] = client.host if client else ""
    # Sessions are per account: the key feeds nebari.key_of, which otherwise
    # merges two callers whose first messages match.
    body["_account"] = account
    # Where this client reaches us, for the signed /media links yama_generate_image
    # puts in an answer. Underscored: stripped before anything goes upstream.
    body["_public_base"] = _public_base(request)
    route = ("/v1/responses" if responses is not None else
             "/v1/messages" if messages is not None else
             "/v1/chat/completions")

    # ONE MODEL PER EFFORT TIER (mcp/max_mode.py, the table mcp/tier_models.py): which main model serves this
    # request, or a 503 model_at_capacity while a higher tier's model holds the card. Decided before admission; off
    # (one model) unless the table is set (YAMADORI_TIER_MODELS, or the older YAMADORI_MAX_MODEL).
    lease = None
    # A CLIENT's request (the token profile does not trust its max_tokens to size the turn: tiers.apply). Internal
    # callers (model.shape) never carry it.
    body["_client"] = True
    if max_mode.ENABLED:
        tier_asked = max_mode.requested_tier(body)
        decision = max_mode.decide(tier_asked, max_mode.is_utility(body), kind=max_mode.utility_kind(body))
        if decision.refuse:
            print(f"{route} refused: model_at_capacity ({decision.why}; retry after {decision.retry_after} s)",
                  flush=True)
            return error(api_errors.at_capacity(decision.holder or max_mode.MAX, decision.retry_after or 30,
                                                decision.why, model=decision.model, tier=tier_asked))
        body["_upstream_model"] = decision.model
        body["_capacity"] = decision.record()
        lease = max_mode.Lease(decision)

    if body.get("stream"):
        # ONE REQUEST, ONE ATTEMPT (#44): a request identical to one still
        # running is its client's retry; the attempt it repeats is cancelled
        # -- on arrival once it has run SUPERSEDE_AFTER_S, and at any age
        # when the lanes are full (below).
        key = _retry_key(body)
        superseded = _supersede(key, SUPERSEDE_AFTER_S) if key else 0
        # The lane is taken HERE, not inside the generator. A StreamingResponse
        # is returned before its body ever runs, so a lane acquired in the
        # generator could not turn a refusal into a 429 -- the client would
        # already be reading a 200 with an error buried in the stream. It is
        # given back by the request's _Attempt: in the body generator's
        # finally (the client hanging up mid-answer included) or, when the
        # body never ran, when the response ends (_CancellingStream).
        lane = admission.admit("main")
        try:
            await lane.__aenter__()
        except admission.Full as e:
            # THE SELF-LOCK (Octopus v0e-V0-xhigh-1, #44): both lanes were
            # held by this very request's earlier attempts, abandoned by
            # Hermes' stale detector and kept open by a relay, and the retry
            # was refused 429 three times, which ended the run. A refusal is
            # right against OTHER work; against its own orphans it is a lock.
            n = _supersede(key, 0.0) if key else 0
            if not n:
                max_mode.release(lease)
                return _too_many(e, messages)
            superseded += n
            try:
                await lane.__aenter__()
            except admission.Full as e2:
                max_mode.release(lease)
                return _too_many(e2, messages)
        if superseded:
            body["_superseded"] = superseded
        token = cancel.Token()
        # the max-mode lease ends with the attempt (_Attempt.close)
        token.yamadori_lease = lease
        # THE 200 IS NOT COMMITTED UNTIL THE TURN HAS STARTED (E1,
        # 2026-09-25). It used to go out before prepare(), so every failure
        # -- a refusal, a prompt that does not fit, an upstream 400 -- reached
        # the client as assistant content on a 200. Now the stream's first
        # chunk is pulled HERE: proxy.stream_body yields nothing until the
        # turn's first event (the first upstream generation has answered),
        # and raises the ApiError when the turn fails before
        # it. A client that hangs up while it waits cancels the token, as
        # _CancellingStream does once the stream is open.
        gen = proxy.stream_body(body, public_name, token=token)
        if responses is not None:
            # The same chunks, as the Responses event sequence; closing it
            # closes stream_body (the turn's cancel).
            gen = responses_api.stream(gen, responses)
        elif messages is not None:
            # The same chunks, as Anthropic's event stream (pings while the
            # turn is silent); closing it closes stream_body.
            gen = messages_api.stream(gen, messages)
        attempt = _Attempt(lane, key, token, gen)
        try:
            first = await _first_chunk(gen, token, request)
        except BaseException as e:
            await attempt.close()
            if not isinstance(e, Exception):
                raise
            if isinstance(e, cancel.Cancelled):
                return Response(status_code=499)
            err = api_errors.of_exception(e)
            print(f"{route} (stream) refused before the first "
                  f"byte: {err.status} {err.code}", flush=True)
            return error(err)
        if first is _GONE:
            await attempt.close()
            return Response(status_code=499)
        if first is _END and token.cancelled:
            # Superseded (or otherwise cancelled) before its first byte: a
            # client still listening is told why, not handed an empty 200.
            await attempt.close()
            if messages is not None:
                return error(api_errors.ApiError(
                    409, f"This request was cancelled before it started: "
                         f"{token.why}.", code="superseded"))
            return JSONResponse(status_code=409, content={"error": {
                "message": f"This request was cancelled before it started: "
                           f"{token.why}.", "type": "invalid_request_error",
                "param": None, "code": "superseded"}})
        return _CancellingStream(_stream(gen, first, attempt),
                                 token, attempt=attempt,
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    t0 = time.time()
    # The lease rides on the turn's token, as on the streamed path (token.yamadori_lease): a turn routed to the
    # other card releases it there (max_mode.set_current), so it never delays a swap of the main card
    btok = cancel.Token()
    btok.yamadori_lease = lease
    try:
        async with admission.admit("main"):
            d = await run_in_threadpool(_complete_on, btok, body)
    except admission.Full as e:
        max_mode.release(lease)
        return _too_many(e, messages)
    except Exception as e:                                       # noqa: BLE001
        # ONE ERROR PATH (mcp/api_errors.py): a refusal keeps its status
        # (TurnRefused: the vision copy with no room on the A4000; a prompt
        # past the window: 400 context_length_exceeded), an upstream 4xx is
        # the client's 4xx, an unreachable model server a 503 -- never a
        # bare-string 502.
        max_mode.release(lease)
        err = api_errors.of_exception(e)
        print(f"{route} failed: {err.status} {err.code}: "
              f"{type(e).__name__}: {str(e)[:300]}", flush=True)
        return error(err)
    max_mode.release(lease)

    d = catalog.rewrite_response(d, public_name)
    print(f"{route} {time.time() - t0:.1f}s", flush=True)
    if responses is not None:
        # In a thread: it stores the response (a sqlite write) before it
        # returns, and the event loop must not wait on the disk.
        return JSONResponse(await run_in_threadpool(responses_api.of_chat, d,
                                                    responses))
    if messages is not None:
        try:
            return JSONResponse(messages_api.of_chat(d, messages))
        except api_errors.ApiError as e:
            return error(e)
    return JSONResponse(d)


class _CancellingStream(StreamingResponse):
    """A stream whose client hanging up STOPS the turn's work (#39).

    Starlette notices the disconnect (http.disconnect) and cancels the body
    task -- but the task is parked in `run_in_threadpool(next, gen)`, and a
    thread cannot be cancelled: the cancellation waits until the turn yields
    again, which during a long prefill or a second-brain job was minutes to
    never (Octopus v0b-V0-xhigh-1: the abandoned request's deep thinking ran
    on and held the helper lane). So the disconnect also cancels the
    request's token (mcp/cancel.py), which shuts every upstream socket the
    turn opened: the blocked read returns, the turn raises
    cancel.Cancelled, the lane is freed and llama-server stops generating."""

    def __init__(self, content, token: "cancel.Token",
                 attempt: "_Attempt | None" = None, **kw):
        super().__init__(content, **kw)
        self._token = token
        self._finished = False
        self._attempt = attempt

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # THE LANE COMES BACK HOWEVER THE RESPONSE ENDED (#44). It was
            # given back only in the body generator's finally -- and a
            # client that hangs up between the first chunk and Starlette's
            # first pull of the body gets that body cancelled before it ever
            # runs, so its finally never ran: a main lane lost until the
            # next restart (mcp/test_stream.py reproduces it). Idempotent.
            if self._attempt is not None:
                await self._attempt.close()

    async def stream_response(self, send) -> None:
        await super().stream_response(send)
        # The whole reply was sent. A client closing the connection after
        # this is an ordinary close, not a hang-up (2026-09-25: every
        # finished stream logged "client disconnected" and cancelled the
        # token, which post-reply work such as the warm is bound to).
        self._finished = True

    async def listen_for_disconnect(self, receive) -> None:
        await super().listen_for_disconnect(receive)
        if self._finished:
            return
        if not self._token.cancelled:
            print("  client disconnected: cancelling the turn's upstream "
                  "work", flush=True)
        self._token.cancel("the client disconnected")


_GONE = object()        # the client hung up before the stream's first byte
_END = object()         # the generator ended


async def _wait_disconnect(request: Request) -> None:
    """Return when the client hangs up (ASGI http.disconnect). The body was
    read already, so the next message the server has for this request is
    the disconnect."""
    while True:
        msg = await request.receive()
        if msg.get("type") == "http.disconnect":
            return


async def _first_chunk(gen, token: "cancel.Token", request: Request):
    """The stream's first chunk, pulled BEFORE the response starts: the
    chunk, _END, or _GONE when the client hung up first (its turn is
    cancelled through the token, as _CancellingStream does later). Raises
    what stream_body raised: a failure before the first byte."""
    import asyncio
    pull = asyncio.ensure_future(run_in_threadpool(next, gen, _END))
    watch = asyncio.ensure_future(_wait_disconnect(request))
    try:
        done, _ = await asyncio.wait({pull, watch},
                                     return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        token.cancel("the request was cancelled before the stream started")
        watch.cancel()
        raise
    if pull in done:
        watch.cancel()
        return pull.result()
    print("  client disconnected before the stream started: cancelling the "
          "turn's upstream work", flush=True)
    token.cancel("the client disconnected before the stream started")
    try:
        await pull
    except BaseException:                                        # noqa: BLE001
        pass
    return _GONE


async def _stream(gen, first=None, lane=None):
    """Bridge the synchronous streaming generator onto the event loop.

    `proxy.stream_body` yields bytes and blocks on the upstream socket. Pulling
    it directly here would stall every other request on this worker, so each
    `next()` is taken in a thread and the loop stays free. A queue would buy
    nothing: the generator is strictly sequential. `first` is the chunk
    server.chat pulled before the response started (_first_chunk).
    """
    sentinel = _END
    try:
        if first is not None and first is not _END:
            yield first
        elif first is _END:
            return
        while True:
            chunk = await run_in_threadpool(next, gen, sentinel)
            if chunk is sentinel:
                return
            yield chunk
    finally:
        # Covers the ordinary end, a client that hangs up mid-answer, and an
        # exception in the generator. A lane that is not released is a lane
        # this server never gets back -- so it is released even when closing
        # the generator raises. `lane`: an admission.admit or the request's
        # _Attempt (both release on __aexit__, once).
        try:
            gen.close()
        except ValueError:
            # Still executing in its thread: its token was cancelled, so it
            # ends on its own.
            pass
        finally:
            if lane is not None:
                await lane.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# ONE REQUEST, ONE ATTEMPT (Octopus v0e-V0-xhigh-1, 2026-09-26, #44 in
# docs/SELF-IMPROVEMENT-LOG.md).
#
# Hermes' stale detector killed a request after 900 s without a chunk and
# sent it again; so did the retry. Neither attempt's turn ever learned its
# client had gone -- a recording relay between them kept both connections to
# this server open -- so the two attempts of ONE request held both main
# lanes, and the third attempt was refused 429 three times, which ended the
# run. A second request byte-identical to one still running (same account,
# same session token, same messages, same tools) is its client's retry: the
# earlier attempt's answer has nobody to go to, and leaving it running
# halves the retry's decode speed beside it (#39).
#
# So the earlier attempt is cancelled (its token: upstream sockets shut, the
# lane freed within a second): on arrival once it has run SUPERSEDE_AFTER_S
# (30, a CHOICE: far above any client's immediate reconnect, far below the
# 600-900 s after which SDKs and Hermes give up), and at any age when the
# lanes are full and the only alternative is refusing the retry. The KNOWN
# COST: two clients sending byte-identical requests under one key and no
# session header, at once, with the lanes full, lose the older one -- the
# X-Yamadori-Session header keeps such arms apart.
# ---------------------------------------------------------------------------
SUPERSEDE_AFTER_S = float(os.environ.get("YAMADORI_SUPERSEDE_AFTER", "30"))

# retry key -> [(token, started)]; touched only on the event loop.
_attempts: dict[str, list] = {}


def _retry_key(body: dict) -> str | None:
    """What makes two requests the same request: the caller's account and
    session token, the messages and the tools, exactly. None when there is
    nothing to compare."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    h = hashlib.sha256()
    for part in (body.get("_account") or "", body.get("_session_token") or ""):
        h.update(str(part).encode("utf-8", "replace") + b"\0")
    h.update(json.dumps([msgs, body.get("tools") or []], sort_keys=True,
                        ensure_ascii=False, default=str)
             .encode("utf-8", "replace"))
    return h.hexdigest()


def _supersede(key: str | None, older_than: float) -> int:
    """Cancel the running attempts of `key` that started at least
    `older_than` seconds ago; how many were cancelled."""
    now = time.time()
    n = 0
    for tok, t0 in list(_attempts.get(key or "") or []):
        if tok.cancelled or now - t0 < older_than:
            continue
        print(f"  superseded: the same request arrived again (a client "
              f"retry); cancelling the attempt it repeats, running "
              f"{now - t0:.0f}s", flush=True)
        tok.cancel("superseded by a retry of the same request")
        n += 1
    return n


class _Attempt:
    """One streamed request's hold: its main lane and its entry in
    _attempts, given back ONCE, from wherever the request ends."""

    def __init__(self, lane, key: str | None, token: "cancel.Token",
                 gen=None):
        self.lane, self.key, self.token, self.gen = lane, key, token, gen
        self.closed = False
        if key:
            _attempts.setdefault(key, []).append((token, time.time()))

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.gen is not None:
                try:
                    self.gen.close()
                except ValueError:
                    pass            # executing in its thread; token cancelled
        finally:
            if self.key:
                left = [(t, s) for t, s in _attempts.get(self.key, [])
                        if t is not self.token]
                if left:
                    _attempts[self.key] = left
                else:
                    _attempts.pop(self.key, None)
            await self.lane.__aexit__(None, None, None)
            max_mode.release(getattr(self.token, "yamadori_lease", None))

    async def __aexit__(self, *exc) -> bool:
        await self.close()
        return False


def _public_base(request: Request) -> str:
    return images.public_base(
        request_base=str(request.base_url),
        forwarded_proto=request.headers.get("x-forwarded-proto", ""))


def _image_error(e: "images.ImageError") -> JSONResponse:
    """An image failure in OpenAI's envelope, carrying the next step."""
    kind = {400: "invalid_request_error", 429: "rate_limit_error"}.get(
        e.status, "api_error")
    headers = {"Retry-After": "60"} if e.status == 429 else None
    return JSONResponse(status_code=e.status, headers=headers, content={
        "error": {"message": e.reason, "type": kind, "code": e.code,
                  "param": None, "retryable": e.retryable,
                  "remedies": e.remedies}})


@app.post("/v1/images/generations")
async def images_generations(request: Request) -> Response:
    """OpenAI Images API: {prompt, n?, size?, response_format?, seed?, model?}.

    `model` "yamadori-image" or "yamadori-image-turbo" picks the image model
    for this request; otherwise the caller's saved choice (GET/PUT
    /dash/api/settings/image), else YAMADORI_IMAGEGEN_DEFAULT.

    Authenticated like every /v1 route. It holds the IMAGE lane, never a chat
    lane: generation runs on CUDA1 and must not make a conversation wait
    (admission.image_lane). `url` answers with a signed /media link, which a
    browser can open with no header; `b64_json` inlines the PNG and is only
    ever sent when asked for.
    """
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _unauthorised(why)
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return _api_error(api_errors.invalid(
            f"The request body is not valid JSON: {e}", code="invalid_json"))
    if not isinstance(body, dict):
        return _image_error(images.bad_args(
            "the body must be a JSON object.", "send {\"prompt\": \"...\"}"))
    fmt = body.get("response_format") or "url"
    if fmt not in ("url", "b64_json"):
        return _image_error(images.bad_args(
            f"response_format {fmt!r} is not supported.",
            "send response_format 'url' or 'b64_json'"))
    # The spec's other options (docs/OPENAI-CONFORMANCE.md section 6):
    # quality accepted (it does not pick the model), output_format png only,
    # background opaque/auto; style, moderation, output_compression, user
    # and partial_images accepted and ignored; streaming is not served.
    if body.get("stream"):
        return _api_error(api_errors.invalid(
            "Streaming image generation is not served; leave out 'stream'.",
            param="stream", code="unsupported_parameter"))
    bad = images.unsupported_option(body)
    if bad:
        return _api_error(api_errors.invalid(bad[1], param=bad[0],
                                             code="invalid_value"))
    n = body.get("n", 1)
    # Which image model: `model` naming one of ours (catalog.IMAGE_MODELS)
    # overrides for this request; anything else -- an SDK's own default such
    # as "dall-e-3" -- is not a choice, and the caller's preference applies.
    choice, source = images.resolve_model(
        account, catalog.resolve_image(body.get("model")))
    try:
        recs = await run_in_threadpool(
            images.generate, body.get("prompt"), size=body.get("size"),
            seed=body.get("seed"), n=n if n is not None else 1, model=choice)
    except images.ImageError as e:
        return _image_error(e)
    base = _public_base(request)
    data = []
    for r in recs:
        if fmt == "b64_json":
            import base64
            data.append({"b64_json": base64.b64encode(
                images.png_bytes(r["id"]) or b"").decode(),
                "revised_prompt": r["prompt"]})
        else:
            data.append({"url": images.signed_url(r["id"], base),
                         "revised_prompt": r["prompt"]})
    print(f"/v1/images/generations {len(recs)} image(s) {choice} "
          f"{recs[0]['size']} {recs[0]['seconds']}s", flush=True)
    out = {"created": int(time.time()), "data": data,
           "output_format": "png", "size": recs[0]["size"],
           "background": "opaque"}
    if body.get("quality") in ("low", "medium", "high", "xhigh", "max"):
        out["quality"] = body["quality"]      # echoed; it picked nothing
    out["x_yamadori"] = {"images": [
        {"id": r["id"][:16], "size": r["size"], "seed": r["seed"],
         "steps": r["steps"], "model": r["image_model"],
         "model_source": source, "seconds": r["seconds"]} for r in recs]}
    return JSONResponse(out)


@app.get("/media/{name}")
@app.head("/media/{name}")
async def media(name: str, exp: str = "", sig: str = "") -> Response:
    """A generated image, by CAPABILITY URL -- no Authorization header.

    A chat client shows `![...](url)` by having the browser fetch it, and that
    fetch carries no key, so a Bearer gate would be a broken image everywhere.
    The signature is the gate instead (images.verify). The name must be
    `<64 hex>.png` -- an id, never a path; anything else is a 404 before the
    filesystem is touched. A bad or expired signature is a 403 whether or not
    the image exists, so a guess learns nothing.
    """
    import re as _re
    m = _re.fullmatch(r"([0-9a-f]{64})\.png", name)
    if not m:
        return JSONResponse(status_code=404, content={"error": {
            "message": "no such image", "type": "invalid_request_error",
            "code": 404, "param": None}})
    ok, reason = images.verify(m[1], exp, sig)
    if not ok:
        return JSONResponse(status_code=403, content={"error": {
            "message": f"{reason}; ask for the image again for a fresh link",
            "type": "permission_error", "code": 403, "param": None}})
    png = images.png_bytes(m[1])
    if png is None:
        return JSONResponse(status_code=404, content={"error": {
            "message": "no such image", "type": "invalid_request_error",
            "code": 404, "param": None}})
    return Response(content=png, media_type="image/png", headers={
        "Cache-Control": "private, max-age=86400, immutable",
        "X-Content-Type-Options": "nosniff"})


@app.api_route("/v1/{rest:path}",
               methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
               include_in_schema=False)
async def v1_unknown(rest: str, request: Request) -> Response:
    """Every /v1 path this server does not serve: 404 in OpenAI's object.

    Registered after every real /v1 route and before the dashboard's SPA
    catch-all, which is GET-only and so answered an unknown POST
    (`/v1/embeddings`, `/v1/responses`, `/v1/completions`) with 405 -- "wrong
    method" where the truth is "no such endpoint" (docs/OPENAI-CONFORMANCE.md
    section 4). A path a real route serves by another method keeps its 405
    (GET /v1/chat/completions)."""
    from starlette.routing import Match
    allow: set[str] = set()
    for route in request.app.router.routes:
        if getattr(route, "endpoint", None) is v1_unknown:
            continue
        path = getattr(route, "path", "")
        if not path.startswith("/v1"):
            continue
        match, _ = route.matches(request.scope)
        if match == Match.PARTIAL:
            allow |= set(getattr(route, "methods", None) or ())
    if allow:
        return _api_error(api_errors.method_not_allowed(
            request.method, request.url.path, sorted(allow)))
    return _api_error(api_errors.unknown_route(request.method,
                                               request.url.path))


# The React dashboard. Registered AFTER every other route: Starlette matches in
# registration order and the SPA fallback is a GET catch-all, so a GET route
# declared below this line would never be reached.
DASH_BUILD = (dash_static.mount(app, DASH_UI) if DASH_UI != "python"
              else {"ok": True, "problems": [], "note": "Python pages"})


def main() -> None:
    import uvicorn
    print(f"yamadori on {HOST}:{PORT} -> {proxy.UPSTREAM}", flush=True)
    print("  uvicorn/h11, real keep-alive, HEAD, graceful drain", flush=True)
    print(f"  dashboard: {DASH_UI}"
          + (f" ({DASH_BUILD['note']})" if DASH_BUILD.get("note") else ""), flush=True)
    for problem in DASH_BUILD.get("problems", []):
        # Loud, not fatal: a stale bundle still serves, and
        # mcp/test_dash_static.py is what fails.
        print(f"  WARNING dashboard bundle is STALE: {problem}", flush=True)
    # GPU power, sampled every second in THIS process (mcp/power.py): the
    # dashboard's electricity panel and x_yamadori.energy read its samples,
    # and it keeps the daily ledger in index/power_ledger.json. Started here,
    # not on import, so tests that import the app start no thread.
    try:
        import power
        power.start()
        print(f"  power sampler: every {power.SAMPLE_SECONDS:g}s, ledger "
              f"{power.LEDGER_PATH}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"  WARNING power sampler not started: {type(e).__name__}: {e}",
              flush=True)
    # Token ledger (mcp/token_ledger.py): every generation this process runs
    # -- client turns, hidden hops, internal calls -- is added to
    # index/token_ledger.sqlite3. Enabled here, not on import, so test suites
    # that import the proxy never write it.
    import token_ledger
    token_ledger.enable()
    print(f"  token ledger: {token_ledger.DB}", flush=True)
    # Slot pins survive a restart (mcp/slots.py, docs/SELF-IMPROVEMENT-LOG.md
    # #38): llama-server still holds each conversation's prefix on its slot.
    # Here, not on import, so no test suite writes the file.
    import slots
    st = slots.persist(os.environ.get("YAMADORI_SLOTS_STATE", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "index", "slots_state.json")))
    print(f"  slot pins: {st['pins']} restored, {st['prompts']} prompts "
          f"({st['path']})", flush=True)
    # This process empties the transient slot after each side call
    # (mcp/slots.py RELEASE). Here, not
    # on import: the worker and the tools API never do, and no test suite
    # sends a release to a model server unless it asks.
    slots.enable_release()
    # ONE PROCESS SWAPS THE CARD: the proxy (mcp/max_mode.py enable_swaps). Here, not on import: an offline suite
    # that imports max_mode can never make llama-swap load a model (2026-09-29, an unstubbed test started
    # flash-next on the running stack).
    max_mode.enable_swaps()
    # x_yamadori.timing: the proxy times its turn's stages (mcp/stage_timing.py); never on import, so no offline
    # suite or other process is wrapped
    import stage_timing
    stage_timing.install()
    # THE POOL IS READ ONCE THE MODEL SERVER ANSWERS (mcp/budget.py pool_size, "THE FALLBACK IS NOT THE LAST
    # WORD"): asked on the main model's own port (never llama-swap's /upstream, which would load a model), every
    # budget.RETRY_S, until one read succeeds. Off the request path.
    import threading

    def _read_pool_when_ready():
        import budget
        while not budget.read_ok():
            pool, how = budget.refresh_direct(timeout=3)
            if budget.read_ok():
                slots.count(refresh=True)
                print(f"  pool: {pool} cells, {budget._SLOTS} slot(s) ({how})", flush=True)
                return
            time.sleep(budget.RETRY_S)
    threading.Thread(target=_read_pool_when_ready, daemon=True, name="yamadori-pool").start()
    print(f"  tier models: {'on' if max_mode.ENABLED else 'off'} ({max_mode.TABLE.source}); "
          f"{', '.join(f'{t}={m}' for t, m in max_mode.snapshot()['tiers'].items())}", flush=True)
    print(f"  slot release: on after second-brain runs and side calls "
          f"(default {'on' if slots._release_default() else 'OFF'}, "
          f"YAMADORI_SLOT_RELEASE / X-Yamadori-Features slot_release)",
          flush=True)
    # THE TURN DECIDER (mcp/decide_turn.py): this process answers each
    # request's judgment questions on the main model. Here, not on import:
    # no test suite sends a decider batch to a model server unless it asks.
    import decide_turn
    decide_turn.enable()
    print(f"  turn decider: {'on' if decide_turn.enabled() else 'OFF'} "
          f"(YAMADORI_DECIDER)", flush=True)
    # THE MCP HOST (mcp/mcp_host.py): this process is the MCP client of the
    # servers it hosts for the model (PackageLens first), each started once,
    # in its gated container, in the background. Here, not on import: no
    # test suite starts a container, and the worker and the tools API offer
    # no MCP tool.
    import mcp_host
    mcp_host.enable()
    started = mcp_host.start(background=True)
    print(f"  mcp host: {'starting ' + ', '.join(started) if started else 'no server started'}"
          f" (YAMADORI_MCP_HOST; {mcp_host.mcp_config.path()}; switch "
          f"mcp_tools / YAMADORI_MCP_TOOLS)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning",
                # A generation runs for minutes. The default 5s keep-alive
                # would drop idle sockets between problems, which is the exact
                # failure the stdlib server had.
                timeout_keep_alive=300,
                # Give in-flight generations time to finish on shutdown
                # instead of killing them, which cost two benchmark runs.
                timeout_graceful_shutdown=120)


if __name__ == "__main__":
    main()
