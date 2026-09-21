#!/usr/bin/env python
"""MCP over WebSocket, for remote clients reached via mcp_bridge.py.

Why this exists alongside POST /mcp: a single agent turn often makes several
tool calls, and HTTP pays a connection setup on each one. A held-open socket
pays it once. For find_definition -- a sqlite lookup measured in single-digit
milliseconds -- that overhead is a large fraction of the total.

Every frame is one JSON-RPC message, handled by exactly the same
code_search.handle() the stdio and HTTP servers use. Three transports, one
implementation, nothing to keep in sync.

WebSocket is NOT part of the MCP spec, so no compliant client will connect
here directly. It is reached through mcp_bridge.py, which presents a normal
stdio server to the client and forwards frames over this socket.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import code_search as cs  # noqa: E402

HOST = os.environ.get("MCP_WS_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_WS_PORT", "1236"))


async def handler(conn):
    peer = getattr(conn, "remote_address", ("?", 0))
    print(f"client connected: {peer}", flush=True)
    try:
        async for raw in conn:
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                # handle() touches sqlite and the GPU services, so keep it off
                # the event loop or one slow search blocks every other client.
                resp = await asyncio.to_thread(cs.handle, req)
            except Exception as e:                       # noqa: BLE001
                resp = {"jsonrpc": "2.0", "id": req.get("id"),
                        "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
            if resp is not None:                         # notifications get no reply
                await conn.send(json.dumps(resp))
    finally:
        print(f"client disconnected: {peer}", flush=True)


async def main() -> None:
    import websockets
    async with websockets.serve(handler, HOST, PORT, max_size=32 * 1024 * 1024):
        print(f"MCP websocket on ws://{HOST}:{PORT}/mcp", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
