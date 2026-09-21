#!/usr/bin/env python
"""Laya decision engine: HTTP + WebSocket in one process.

Laya is a non-autoregressive "System 1" model (421M, ModernBERT-large) that
answers TYPED questions about a state in a single forward pass -- no token
generation. Measured here: 3 questions in 69 ms, ~23 ms/question.

    choice  pick one of N labelled options, with probabilities
    score   ordinal level against a rubric
    noul    boolean, returned as a probability

WHY IT IS HERE
--------------
An agent loop makes hundreds of small judgements -- "is this patch worth
benchmarking?", "did that output indicate a regression?", "is this retrieved
snippet actually relevant?". Asking a 27B to generate an answer for each costs
seconds and a slot. Laya answers in tens of milliseconds with a calibrated
probability, which is also a better signal than prose you then have to parse.

It is NOT a replacement for the chat model. It cannot write, explain or plan.
It classifies.

TRANSPORTS
----------
    POST /decide          one-shot HTTP, for scripts and CI
    WS   /ws              persistent socket, for hot loops
    GET  /health, /schema, /openapi.json

The WebSocket matters more than it looks: at 23 ms of compute, HTTP connection
setup and JSON headers are a large fraction of wall time. A held-open socket
removes that per-call overhead, which is the difference between "fast enough to
call in a loop" and "fast enough to call on every step".

Runs in its own venv (.venv-laya) because laya pulls a newer transformers than
the rest of the stack uses.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Pin to the A4000 BEFORE torch is imported -- once CUDA initialises this has
# no effect. laya defaults to cuda:0, which is the 5060 Ti running the primary
# model at 208k context; adding ~1 GiB there took that card to 98% and it is
# the card that dies first under a long prefill. The A4000 has ~5 GiB spare.
# PCI_BUS_ID matches the ordering the rest of the stack assumes.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("LAYA_GPU", "1"))

import laya  # noqa: E402

MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
HOST = os.environ.get("LAYA_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("LAYA_HTTP_PORT", "1237"))
WS_PORT = int(os.environ.get("LAYA_WS_PORT", "1238"))

_agent = None
_lock = threading.Lock()          # the model is not thread-safe; serialise it


def agent():
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                t0 = time.time()
                _agent = laya.load(MODEL)
                print(f"laya loaded in {time.time()-t0:.1f}s", flush=True)
    return _agent


SCHEMA = {
    "state": "string | object | list  -- the thing being judged",
    "questions": {
        "<id>": {
            "choice": {"type": "choice", "instructions": "...",
                       "criteria": {"option_a": "what it means",
                                    "option_b": "what it means"}},
            "score": {"type": "score", "instructions": "...",
                      "criteria": ["level 0", "level 1", "level 2"]},
            "noul": {"type": "noul", "instructions": "a statement to judge true/false"},
        }
    },
    "note": "criteria are REQUIRED for choice and score; optional for noul.",
}


def decide(payload: dict) -> dict:
    state = payload.get("state")
    questions = payload.get("questions")
    if state is None:
        raise ValueError("'state' is required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("'questions' must be a non-empty object")

    t0 = time.time()
    with _lock:                    # single GPU model, one caller at a time
        res = agent().predict(state, questions)
    ms = (time.time() - t0) * 1000
    return {"answers": res.get("answers", res),
            "elapsed_ms": round(ms, 1),
            "questions": len(questions)}


# ---------------------------------------------------------------- HTTP ------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                   # noqa: N802
        p = urlparse(self.path).path
        if p == "/health":
            return self._send(200, {"ok": True, "model": MODEL,
                                    "loaded": _agent is not None,
                                    "ws": f"ws://<host>:{WS_PORT}/ws"})
        if p == "/schema":
            return self._send(200, SCHEMA)
        if p == "/openapi.json":
            return self._send(200, {
                "openapi": "3.1.0",
                "info": {"title": "Laya decision engine", "version": "1.0.0",
                         "description": "Typed decisions (choice/score/noul) in ~23ms per "
                                        "question. Non-generative classifier."},
                "paths": {"/decide": {"post": {
                    "operationId": "decide",
                    "summary": "Answer typed questions about a state.",
                    "responses": {"200": {"description": "answers with probabilities"}}}}},
            })
        return self._send(404, {"error": "try /health, /schema, /openapi.json, POST /decide"})

    def do_POST(self):                                  # noqa: N802
        if urlparse(self.path).path != "/decide":
            return self._send(404, {"error": "POST /decide"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        try:
            return self._send(200, decide(payload))
        except ValueError as e:
            return self._send(400, {"error": str(e), "schema": SCHEMA})
        except Exception as e:                          # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *a):
        pass


# ------------------------------------------------------------ WebSocket -----

async def ws_handler(conn):
    """One JSON message in, one JSON message out, connection held open.

    Each frame is an independent request; there is no session state. An `id`
    field, if supplied, is echoed back so a client can pipeline requests.
    """
    async for raw in conn:
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            await conn.send(json.dumps({"error": "invalid JSON"}))
            continue
        rid = req.get("id")
        try:
            # decide() blocks on the GPU; keep the event loop responsive
            out = await asyncio.to_thread(decide, req)
            if rid is not None:
                out["id"] = rid
            await conn.send(json.dumps(out, default=str))
        except ValueError as e:
            await conn.send(json.dumps({"id": rid, "error": str(e)}))
        except Exception as e:                          # noqa: BLE001
            await conn.send(json.dumps({"id": rid, "error": f"{type(e).__name__}: {e}"}))


async def ws_main():
    import websockets
    async with websockets.serve(ws_handler, HOST, WS_PORT, max_size=8 * 1024 * 1024):
        print(f"laya ws   on ws://{HOST}:{WS_PORT}/ws", flush=True)
        await asyncio.Future()


def main():
    agent()                                             # load before serving
    threading.Thread(
        target=lambda: ThreadingHTTPServer((HOST, HTTP_PORT), Handler).serve_forever(),
        daemon=True).start()
    print(f"laya http on http://{HOST}:{HTTP_PORT}", flush=True)
    asyncio.run(ws_main())


if __name__ == "__main__":
    main()
