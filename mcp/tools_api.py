#!/usr/bin/env python
"""HTTP API for the code-intelligence tools.

Same logic as the MCP server, different transport. MCP is for agents; this is
for your own software -- scripts, build steps, editors, CI, anything that would
rather make an HTTP call than speak JSON-RPC over stdio.

    GET  /                  service description (also what /openapi.json documents)
    GET  /health            liveness + index size
    GET  /openapi.json      OpenAPI 3.1 spec, so tools can self-discover
    POST /search            {"query": str, "top_k": int}
    POST /definition        {"symbol": str}
    POST /references        {"symbol": str, "calls_only": bool}
    GET  /status            index coverage

GET variants are accepted for the POST routes too (?query=, ?symbol=), because
being able to curl a search without composing JSON matters more than REST
purity.

Stdlib only -- no framework, no extra dependency.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import code_search as cs          # noqa: E402
import symbols as sym             # noqa: E402

# MCP over HTTP.
#
# MCP's default transport is stdio: the client SPAWNS the server as a
# subprocess. That only works when the client runs on this machine. Driving
# the stack from a laptop or phone means the agent cannot reach the tools at
# all -- and it fails silently, looking like the model ignoring them.
#
# code_search.handle() is already a pure JSON-RPC function, so exposing it
# over HTTP costs one route and shares every code path with the stdio server.
# No second implementation to drift.
MCP_PATH = "/mcp"

HOST = os.environ.get("TOOLS_API_HOST", "0.0.0.0")
PORT = int(os.environ.get("TOOLS_API_PORT", "1235"))

DESCRIPTION = {
    "service": "llama-stack code intelligence",
    "version": "1.0.0",
    "transports": {
        "http": f"this API (port {PORT})",
        "mcp": "mcp/code_search.py over stdio, same tools",
    },
    "endpoints": {
        "POST /search": "semantic search: embeddings + cross-encoder rerank. "
                        "Use when you CANNOT name what you want.",
        "POST /definition": "exact symbol definition lookup. Instant, no GPU. "
                            "Use when you KNOW the identifier.",
        "POST /references": "call sites and references for a symbol.",
        "GET /status": "index coverage",
    },
}


def openapi_spec() -> dict:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "llama-stack code intelligence",
            "version": "1.0.0",
            "description": (
                "Code search over a local index. Semantic search is backed by "
                "Qwen3-Embedding plus a Qwen3-Reranker-4B cross-encoder; symbol "
                "lookups are exact sqlite queries built with tree-sitter."
            ),
        },
        "servers": [{"url": f"http://ai.thejustinwalsh.me:{PORT}"}],
        "paths": {
            "/search": {
                "post": {
                    "operationId": "searchCode",
                    "summary": "Semantic code search. Use when you cannot name the symbol.",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Plain-language question."},
                            "top_k": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    }}}},
                    "responses": {"200": {"description": "ranked snippets"}},
                }
            },
            "/definition": {
                "post": {
                    "operationId": "findDefinition",
                    "summary": "Where a symbol is defined. Exact match, instant.",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {"symbol": {"type": "string"}},
                        "required": ["symbol"],
                    }}}},
                    "responses": {"200": {"description": "definition sites"}},
                }
            },
            "/references": {
                "post": {
                    "operationId": "findReferences",
                    "summary": "Call sites and references. Use before changing a signature.",
                    "requestBody": {"required": True, "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "calls_only": {"type": "boolean", "default": False},
                        },
                        "required": ["symbol"],
                    }}}},
                    "responses": {"200": {"description": "reference sites"}},
                }
            },
            "/status": {"get": {"operationId": "indexStatus", "summary": "Index coverage",
                                "responses": {"200": {"description": "counts"}}}},
        },
    }


def do_search(args: dict) -> dict:
    q = args.get("query")
    if not q:
        raise ValueError("'query' is required")
    hits = cs.search(q, int(args.get("top_k", cs.DEFAULT_TOP_K)))
    return {"query": q, "count": len(hits), "results": hits}


def do_definition(args: dict) -> dict:
    s = args.get("symbol")
    if not s:
        raise ValueError("'symbol' is required")
    con = cs._db()
    rows = sym.find_definition(con, s)
    con.close()
    return {"symbol": s, "count": len(rows), "results": [
        {"name": n, "kind": k, "path": p, "start": st, "end": en, "line": ln}
        for n, k, p, st, en, ln in rows]}


def do_references(args: dict) -> dict:
    s = args.get("symbol")
    if not s:
        raise ValueError("'symbol' is required")
    calls_only = str(args.get("calls_only", False)).lower() in ("1", "true", "yes")
    con = cs._db()
    rows = sym.find_references(con, s, calls_only)
    con.close()
    return {"symbol": s, "calls_only": calls_only, "count": len(rows), "results": [
        {"name": n, "kind": k, "path": p, "line": ln, "text": tx}
        for n, k, p, ln, tx in rows]}


def do_status(_args: dict) -> dict:
    con = cs._db()
    nc = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    nf = con.execute("SELECT COUNT(DISTINCT path) FROM chunks").fetchone()[0]
    nd = con.execute("SELECT COUNT(*) FROM defs").fetchone()[0]
    nr = con.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
    con.close()
    return {"chunks": nc, "files": nf, "definitions": nd, "references": nr}


ROUTES = {
    "/search": do_search,
    "/definition": do_definition,
    "/references": do_references,
    "/status": do_status,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, path: str, args: dict) -> None:
        try:
            if path in ("/", ""):
                return self._send(200, DESCRIPTION)
            if path == "/openapi.json":
                return self._send(200, openapi_spec())
            if path == "/health":
                st = do_status({})
                return self._send(200, {"ok": True, **st})
            fn = ROUTES.get(path)
            if fn is None:
                return self._send(404, {"error": f"no such endpoint: {path}",
                                        "endpoints": sorted(ROUTES) + ["/openapi.json", "/health"]})
            return self._send(200, fn(args))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:                       # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):                                # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        self._dispatch(u.path, q)

    def do_POST(self):                               # noqa: N802
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            args = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "body is not valid JSON"})

        # MCP over HTTP: one JSON-RPC request in, one response out, handled by
        # exactly the same code the stdio server uses. Notifications legally
        # produce no response, so return 202 rather than an empty body.
        if u.path == MCP_PATH:
            try:
                resp = cs.handle(args)
            except Exception as e:                   # noqa: BLE001
                return self._send(200, {"jsonrpc": "2.0", "id": args.get("id"),
                                        "error": {"code": -32603,
                                                  "message": f"{type(e).__name__}: {e}"}})
            if resp is None:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            return self._send(200, resp)

        self._dispatch(u.path, args)

    def log_message(self, fmt, *a):                  # quieter default logging
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))


def main() -> None:
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"tools API on http://{HOST}:{PORT}  (spec at /openapi.json)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
