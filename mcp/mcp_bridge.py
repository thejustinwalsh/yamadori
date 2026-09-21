#!/usr/bin/env python
"""stdio <-> remote MCP bridge. Run this LOCALLY; the tools stay on the GPU box.

THE PROBLEM
-----------
An MCP client's `"command"` field spawns a LOCAL subprocess and talks to it
over stdin/stdout. There is no mechanism for it to execute a CLI on another
machine -- the CLI is always local. So when the agent runs on your laptop and
the tools live on the GPU box, a stdio config cannot reach them, and it fails
silently: no tools appear, which looks exactly like the model ignoring them.

THE SHAPE OF THE FIX
--------------------
Keep the stdio contract, move the work. The client spawns THIS script locally;
it forwards every JSON-RPC frame to the remote server and writes the replies
back to stdout. The client cannot tell the difference.

    [client] --stdio--> [mcp_bridge.py (local)] --ws or http--> [GPU box]

TRANSPORT
---------
MCP defines stdio and Streamable HTTP. WebSocket is NOT in the spec, so no
compliant client will offer it directly -- but it is a better fit for a
long-lived bridge than HTTP: one handshake instead of a connection per call,
which matters when a single agent turn makes several tool calls.

Both are supported here. --url ws://...  uses WebSocket;  http://... uses POST.
WebSocket needs the `websockets` package; HTTP needs nothing.

USAGE (in the client's MCP config, on the REMOTE machine)

    {
      "mcpServers": {
        "code-search": {
          "command": "python",
          "args": ["mcp_bridge.py", "--url", "ws://ai.thejustinwalsh.me:1236/mcp"]
        }
      }
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def log(msg: str) -> None:
    # stdout is the protocol channel and must carry ONLY JSON-RPC frames.
    # Anything diagnostic goes to stderr or it corrupts the stream.
    sys.stderr.write(f"[mcp-bridge] {msg}\n")
    sys.stderr.flush()


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def run_http(url: str) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            r = urllib.request.Request(url, data=line.encode(),
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=1800) as resp:
                if resp.status == 202:      # notification: no reply is correct
                    continue
                body = resp.read()
            if body:
                emit(json.loads(body))
        except Exception as e:                      # noqa: BLE001
            # A transport failure must still answer the client, or it hangs
            # forever waiting on a response that will never come.
            if req.get("id") is not None:
                emit({"jsonrpc": "2.0", "id": req["id"],
                      "error": {"code": -32000, "message": f"bridge: {e}"}})


def run_ws(url: str) -> None:
    import asyncio
    import websockets

    async def main() -> None:
        async with websockets.connect(url, max_size=32 * 1024 * 1024) as ws:
            log(f"connected {url}")
            loop = asyncio.get_running_loop()
            pending: set = set()          # request ids still awaiting a reply
            stdin_done = asyncio.Event()

            async def pump_stdin():
                while True:
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if not line:
                        stdin_done.set()
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rid = json.loads(line).get("id")
                        if rid is not None:
                            pending.add(rid)
                    except json.JSONDecodeError:
                        pass
                    await ws.send(line)

            async def pump_ws():
                async for msg in ws:
                    try:
                        obj = json.loads(msg)
                    except json.JSONDecodeError:
                        log(f"dropped non-JSON frame: {msg[:80]}")
                        continue
                    pending.discard(obj.get("id"))
                    emit(obj)

            reader = asyncio.create_task(pump_stdin())
            writer = asyncio.create_task(pump_ws())

            # Closing the socket the instant stdin hits EOF discards replies
            # that are still in flight -- piping three requests in returned only
            # the first. Drain outstanding ids before shutting down.
            await stdin_done.wait()
            deadline = loop.time() + 1800
            while pending and loop.time() < deadline:
                await asyncio.sleep(0.05)
            if pending:
                log(f"giving up with {len(pending)} unanswered: {sorted(pending)}")

            writer.cancel()
            reader.cancel()
            await ws.close()

    try:
        asyncio.run(main())
    except Exception as e:                           # noqa: BLE001
        log(f"websocket failed: {e}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True,
                    help="ws://host:1236/mcp  or  http://host:1235/mcp")
    a = ap.parse_args()
    if a.url.startswith(("ws://", "wss://")):
        run_ws(a.url)
    elif a.url.startswith(("http://", "https://")):
        run_http(a.url)
    else:
        log("url must start with ws://, wss://, http:// or https://")
        sys.exit(2)


if __name__ == "__main__":
    main()
