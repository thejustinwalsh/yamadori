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
import catalog  # noqa: E402
import dash_data  # noqa: E402
import dash_results  # noqa: E402
import dash_static  # noqa: E402
import dash_vitals  # noqa: E402
import dashboard  # noqa: E402
import images  # noqa: E402
import proxy  # noqa: E402

app = FastAPI(title="yamadori", version="1", docs_url=None, redoc_url=None)

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
        "endpoints": ["/v1/models", "/v1/chat/completions",
                      "/v1/images/generations", "/health", ui,
                      f"/dash{_PY}/data", f"/dash{_PY}/vitals", f"/dash{_PY}/results"],
        "dashboard": DASH_UI,
        "note": "OpenAI-compatible. Point any client here; it needs no tool "
                "configuration.",
    }, headers={"Vary": "Accept"})


def _too_many(e: "admission.Full") -> JSONResponse:
    """A refusal, not a timeout.

    Three benchmark processes once ran against this proxy at once because
    nothing stopped them, and they degraded each other into 502s that took an
    hour to attribute to concurrency rather than to the arm that happened to
    lose most. One model over one shared KV pool does not go faster when it is
    given more work; it goes wrong in a way that looks like a different bug.
    """
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "30"},
        content={"error": {"message": str(e), "type": "rate_limit_error",
                           "code": "server_busy"}})


def _unauthorised(why: str) -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": {
        "message": f"{why}. Set an API key in your client.",
        "type": "invalid_request_error", "code": "invalid_api_key"}})


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
    return JSONResponse(catalog.public_list())


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


@app.get(f"/dash{_PY}/results")
async def dash_results_page() -> Response:
    # Same deal again: the shell is public and holds no measurement, and every
    # number on it arrives from the gated /dash/api/results.
    code, ctype, payload = dash_results.handle_get("/dash/results")
    return Response(content=payload, status_code=code, media_type=ctype)


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
        hit = await run_in_threadpool(dash_results.handle_get, path)
    if not hit:
        hit = await run_in_threadpool(dash_data.handle_get, path)
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
        return JSONResponse(status_code=404, content={"error": "not found"})
    code, ctype, payload = hit
    return Response(content=payload, status_code=code, media_type=ctype)


@app.post("/v1/chat/completions")
async def chat(request: Request) -> Response:
    account, why = accounts.identify(request.headers.get("authorization"))
    if account is None:
        return _unauthorised(why)
    try:
        body = json.loads(await request.body() or b"{}")
    except (ValueError, json.JSONDecodeError) as e:
        return JSONResponse(status_code=400, content={"error": f"bad request: {e}"})

    # Experiment overrides ride on a header, so the body stays the OpenAI
    # schema and the override never becomes part of the cached prompt.
    feats = request.headers.get("x-yamadori-features")
    if feats:
        body["_features"] = feats
    client = request.client
    body["_client_ip"] = client.host if client else ""
    # Sessions are per account: the key feeds nebari.key_of, which otherwise
    # merges two callers whose first messages match.
    body["_account"] = account
    # Where this client reaches us, for the signed /media links generate_image
    # puts in an answer. Underscored: stripped before anything goes upstream.
    body["_public_base"] = _public_base(request)
    public_name = body.get("model") or "yamadori"

    if body.get("stream"):
        # The lane is taken HERE, not inside the generator. A StreamingResponse
        # is returned before its body ever runs, so a lane acquired in the
        # generator could not turn a refusal into a 429 -- the client would
        # already be reading a 200 with an error buried in the stream. It is
        # released in the generator's finally, which also covers the client
        # hanging up mid-answer.
        lane = admission.admit("main")
        try:
            await lane.__aenter__()
        except admission.Full as e:
            return _too_many(e)
        return StreamingResponse(_stream(body, public_name, lane),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    t0 = time.time()
    try:
        async with admission.admit("main"):
            d = await run_in_threadpool(proxy.complete, body)
    except admission.Full as e:
        return _too_many(e)
    except Exception as e:                                       # noqa: BLE001
        return JSONResponse(status_code=502,
                            content={"error": f"{type(e).__name__}: {e}"})

    if proxy.PREAMBLE and proxy.is_first_turn(body.get("messages") or []):
        import repos
        msgs = body.get("messages") or []
        root, trusted, how = proxy.resolve_repo(msgs, body["_client_ip"])
        info = repos.ensure(root, from_trusted=trusted) if root else None
        note = proxy.preamble_for(
            info, proxy.available_checks(root) if root else [], how)
        try:
            m = d["choices"][0]["message"]
            m["content"] = note + (m.get("content") or "")
        except (KeyError, IndexError, TypeError):
            pass

    d = catalog.rewrite_response(d, public_name)
    print(f"/v1/chat/completions {time.time() - t0:.1f}s", flush=True)
    return JSONResponse(d)


async def _stream(body: dict, public_name: str, lane=None):
    """Bridge the synchronous streaming generator onto the event loop.

    `proxy.stream_body` yields bytes and blocks on the upstream socket. Pulling
    it directly here would stall every other request on this worker, so each
    `next()` is taken in a thread and the loop stays free. A queue would buy
    nothing: the generator is strictly sequential.
    """
    gen = proxy.stream_body(body, public_name)
    sentinel = object()
    try:
        while True:
            chunk = await run_in_threadpool(next, gen, sentinel)
            if chunk is sentinel:
                return
            yield chunk
    finally:
        # Covers the ordinary end, a client that hangs up mid-answer, and an
        # exception in the generator. A lane that is not released is a lane
        # this server never gets back.
        gen.close()
        if lane is not None:
            await lane.__aexit__(None, None, None)


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
    """OpenAI Images API: {prompt, n?, size?, response_format?, seed?}.

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
        return JSONResponse(status_code=400, content={"error": {
            "message": f"bad request: {e}", "type": "invalid_request_error",
            "code": "bad_json", "param": None}})
    if not isinstance(body, dict):
        return _image_error(images.bad_args(
            "the body must be a JSON object.", "send {\"prompt\": \"...\"}"))
    fmt = body.get("response_format") or "url"
    if fmt not in ("url", "b64_json"):
        return _image_error(images.bad_args(
            f"response_format {fmt!r} is not supported.",
            "send response_format 'url' or 'b64_json'"))
    n = body.get("n", 1)
    try:
        recs = await run_in_threadpool(
            images.generate, body.get("prompt"), size=body.get("size"),
            seed=body.get("seed"), n=n if n is not None else 1)
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
    print(f"/v1/images/generations {len(recs)} image(s) "
          f"{recs[0]['size']} {recs[0]['seconds']}s", flush=True)
    return JSONResponse({"created": int(time.time()), "data": data,
                         "x_yamadori": {"images": [
                             {"id": r["id"][:16], "size": r["size"],
                              "seed": r["seed"], "steps": r["steps"],
                              "seconds": r["seconds"]} for r in recs]}})


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
