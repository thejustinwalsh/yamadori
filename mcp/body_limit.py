#!/usr/bin/env python
"""The request-body limit: 413 in OpenAI's error envelope (docs/VISION.md 5e).

server.py had none: a request of any size was read whole into memory
before anything looked at it. A request resends every image of its
history (a 10 MB image is ~13.4 MB of base64), so the limit is
image_input.max_body_bytes() -- 64 MB by default, YAMADORI_MAX_BODY_BYTES,
a CHOICE, not a measurement.

Pure ASGI, so it sees the bytes before FastAPI reads them: a declared
Content-Length over the limit is refused before a byte is read, and a body
without one (chunked) is counted as it arrives and refused the moment it
passes the limit. Only methods that carry a body are checked.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_input  # noqa: E402

METHODS = frozenset({"POST", "PUT", "PATCH"})


class _TooLarge(Exception):
    pass


def envelope(limit: int, seen: int | None = None) -> dict:
    size = (f"{seen:,} bytes or more" if seen is not None
            else "larger than that")
    return {"error": {
        "message": (f"The request body is {size}; this server accepts at most "
                    f"{limit:,} bytes (about {limit // (1024 * 1024)} MB). A "
                    f"request resends every image in its history: send fewer "
                    f"or smaller images, or start a new conversation."),
        "type": "invalid_request_error", "param": None,
        "code": "request_too_large", "retryable": False,
        "remedies": [{"fixable_by": "user",
                      "action": "send fewer or smaller images, or start a "
                                "new conversation",
                      "effect": "the request fits the limit"}]}}


async def _send_413(send, limit: int, seen: int | None = None) -> None:
    body = json.dumps(envelope(limit, seen)).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"connection", b"close")]})
    await send({"type": "http.response.body", "body": body})


class BodyLimit:
    """ASGI middleware: app.add_middleware(BodyLimit)."""

    def __init__(self, app, limit: int | None = None):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") not in METHODS:
            await self.app(scope, receive, send)
            return
        limit = self.limit or image_input.max_body_bytes()
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.strip().isdigit() \
                and int(declared) > limit:
            await _send_413(send, limit, int(declared))
            return
        seen = 0
        started = False

        async def counted():
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body") or b"")
                if seen > limit:
                    raise _TooLarge()
            return message

        async def tracked(message):
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted, tracked)
        except _TooLarge:
            if not started:
                await _send_413(send, limit, seen)
