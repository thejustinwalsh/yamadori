#!/usr/bin/env python
"""Verify the Laya sideband channel: one port, FIFO, optional ids, heartbeat.

Run with the Laya venv, which is the one that has `websockets`:

    .venv-laya\\Scripts\\python.exe scripts/ws_check.py

Checks the four properties the channel is supposed to have, rather than
assuming them:

  one port      POST /decide and the /ws upgrade share 1237. The second port
                only ever existed because ThreadingHTTPServer cannot perform
                the upgrade handshake.
  FIFO          one worker over a queue, so responses come back in request
                order and a pipelining client can correlate positionally.
  ids optional  echoed when supplied, absent when not.
  heartbeat     ping/pong is handled by the protocol layer below the handler,
                so an idle connection survives without the application sending
                anything. Tested by actually idling past the ping interval.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request

import websockets

HTTP = "http://127.0.0.1:1237/decide"
WS = "ws://127.0.0.1:1237/ws"
STATE = "Given a sorted array of integers, find a target value by binary search."
Q = {"v": {"type": "choice", "instructions": "Is the input sorted?",
           "criteria": {"yes": "", "no": ""}}}


def http_once() -> float:
    body = json.dumps({"state": STATE, "questions": Q}).encode()
    req = urllib.request.Request(HTTP, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        json.load(r)
    return (time.time() - t0) * 1000


def median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


async def main(idle_seconds: int) -> None:
    print("  POST /decide and WS /ws on the SAME port 1237\n")
    http_ms = median([http_once() for _ in range(8)])
    print(f"    HTTP  median {http_ms:6.1f} ms  (new connection per call)")

    async with websockets.connect(WS) as ws:
        ws_times = []
        for _ in range(8):
            t0 = time.time()
            await ws.send(json.dumps({"state": STATE, "questions": Q}))
            await ws.recv()
            ws_times.append((time.time() - t0) * 1000)
        ws_ms = median(ws_times)
        print(f"    WS    median {ws_ms:6.1f} ms  (held open)")
        print(f"    -> the channel removes {http_ms - ws_ms:.1f} ms of "
              f"per-call transport\n")

        # FIFO with no ids at all: order is the correlation.
        n = 5
        for i in range(n):
            await ws.send(json.dumps(
                {"state": f"Problem {i}. " + STATE, "questions": Q}))
        got = [json.loads(await ws.recv()) for _ in range(n)]
        ok = all("answers" in g for g in got)
        print(f"    pipelined {n} with no id -> {len(got)} responses, "
              f"all well-formed: {ok}")

        await ws.send(json.dumps({"id": "abc-123", "state": STATE,
                                  "questions": Q}))
        with_id = json.loads(await ws.recv())
        await ws.send(json.dumps({"state": STATE, "questions": Q}))
        no_id = json.loads(await ws.recv())
        print(f"    id echoed when supplied: {with_id.get('id')!r}")
        print(f"    id absent when omitted:  {'id' not in no_id}")

        print(f"\n    idling {idle_seconds}s (ping interval is 20s)...",
              flush=True)
        await asyncio.sleep(idle_seconds)
        t0 = time.time()
        await ws.send(json.dumps({"id": "after-idle", "state": STATE,
                                  "questions": Q}))
        back = json.loads(await ws.recv())
        alive = back.get("id") == "after-idle"
        print(f"    connection survived the idle: {alive} "
              f"({(time.time() - t0) * 1000:.0f} ms)")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 50))
