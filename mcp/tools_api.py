#!/usr/bin/env python
"""HTTP API for the code-intelligence tools.

Same logic as the MCP server, different transport. MCP is for agents; this is
for your own software -- scripts, build steps, editors, CI, anything that would
rather make an HTTP call than speak JSON-RPC over stdio.

    GET  /health            liveness only                        (no key)
    GET  /.well-known/oauth-protected-resource[/mcp]  RFC 9728      (no key)
    GET  /                  service description                  (key)
    GET  /openapi.json      OpenAPI 3.1 spec                     (key)
    POST /search            {"query": str, "top_k": int}         (key)
    POST /definition        {"symbol": str}                      (key)
    POST /references        {"symbol": str, "calls_only": bool}  (key)
    GET  /status            index coverage                       (key)
    POST /mcp               MCP, one JSON-RPC message per POST   (key)

GET variants are accepted for the POST routes too (?query=, ?symbol=), because
being able to curl a search without composing JSON matters more than REST
purity. The key never goes in the query string.

THE DOOR (operator, 2026-09-26; SELF-IMPROVEMENT-LOG #48). This binds
0.0.0.0 on purpose -- remote MCP clients reach it over ZeroTier -- and until
2026-09-26 it had no auth at all: the LAN, ZeroTier and, through DNS
rebinding, any web page in the operator's browser could call /mcp's
read_file_range and run_check. Every request now passes two checks, in the
MCP spec's own terms (revision 2026-07-28):

  1. ORIGIN (basic/transports/streamable-http, "Security & Endpoint": "If
     the Origin header is present and invalid, servers MUST respond with
     HTTP 403 Forbidden"). No Origin passes (a non-browser client); a
     present one must be in YAMADORI_TOOLS_ALLOWED_ORIGINS -- empty by
     default, because nothing in this repo calls this port from a browser.
     Every route, /health and the metadata included.
  2. BEARER (basic/authorization, "Access Token Usage"): `Authorization:
     Bearer <key>` on every request but /health and the metadata, checked
     by accounts.identify() -- the SAME keys as :1234. Missing or wrong is
     401 with `WWW-Authenticate: Bearer resource_metadata="..."` (RFC 9728
     s5.1). With no accounts.json the proxy is single-user and open; THIS
     door is closed instead (a door that reads files is not open because
     nobody made a key).

The key is an operator-issued API key, not an OAuth token: there is no
authorization server, so the metadata names none (docs/TOOLS-API.md, "The
gap"). The access log carries neither the key nor any query string.

Stdlib only -- no framework, no extra dependency.
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import accounts                   # noqa: E402
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
    "auth": "Authorization: Bearer <account key> (the :1234 keys) on every "
            "route but /health and /.well-known/oauth-protected-resource",
    "transports": {
        "http": f"this API (port {PORT})",
        "mcp": "POST /mcp here, or mcp/code_search.py over stdio, same tools",
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
        "components": {"securitySchemes": {"accountKey": {
            "type": "http", "scheme": "bearer",
            "description": "An account key from mcp/accounts.py -- the same "
                           "keys as the proxy on :1234."}}},
        "security": [{"accountKey": []}],
        "paths": {
            "/health": {"get": {"operationId": "health", "security": [],
                                "summary": "Liveness only. No key.",
                                "responses": {"200": {"description": "ok"}}}},
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


# ----------------------------------------------------------------- the door
PRM_PATH = "/.well-known/oauth-protected-resource"
# Served without a key: liveness (the watchdog and deploy_check read the
# status code only) and the RFC 9728 metadata a client reads to learn how to
# authenticate. Everything else needs the key.
OPEN_PATHS = frozenset({"/health", PRM_PATH})
# A body larger than this is refused unread (413): the MCP websocket's frame
# cap (mcp_ws.py max_size), the largest message any in-repo client sends.
MAX_BODY = 32 * 1024 * 1024
# A refused request's body is drained up to this before the connection
# closes: closing a socket with unread bytes resets it, and the client then
# loses the 401 it was sent.
DRAIN_LIMIT = 1024 * 1024
CORS_ALLOW_HEADERS = ("Authorization, Content-Type, Accept, "
                      "MCP-Protocol-Version, Mcp-Method, Mcp-Name")
_HOST_OK = re.compile(r"^(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$")


def _origin_of(url: str) -> str | None:
    """scheme://host[:port], lower-cased, default port dropped; None when it
    is not an http(s) origin (the literal "null", garbage)."""
    try:
        u = urlsplit(url.strip())
        port = u.port
    except ValueError:
        return None
    if u.scheme not in ("http", "https") or not u.hostname:
        return None
    host = u.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port and port != {"http": 80, "https": 443}[u.scheme]:
        host = f"{host}:{port}"
    return f"{u.scheme}://{host}"


def allowed_origins() -> frozenset[str]:
    """YAMADORI_TOOLS_ALLOWED_ORIGINS, comma-separated, read per request.
    Empty by default: the dashboard is served by :1234 and fetches only
    :1234's own relative paths (web/src/api/client.ts), so no browser page
    in this repo calls this port."""
    raw = os.environ.get("YAMADORI_TOOLS_ALLOWED_ORIGINS", "")
    return frozenset(o for o in (_origin_of(x) for x in raw.split(",")
                                 if x.strip()) if o)


def origin_allowed(origin: str | None) -> bool:
    """No Origin header: a non-browser client, allowed. Present: it must be
    on the list -- "null" (a sandboxed frame, a file:// page) never is."""
    if origin is None:
        return True
    o = _origin_of(origin)
    return o is not None and o in allowed_origins()


def public_base(host_header: str | None) -> str:
    """The canonical URL of this server, no trailing slash (MCP
    basic/authorization, "Canonical Server URI"): YAMADORI_TOOLS_PUBLIC_BASE
    when set, else http://<Host> -- validated, so a crafted Host cannot put
    anything but a host into a header -- else loopback."""
    env = os.environ.get("YAMADORI_TOOLS_PUBLIC_BASE", "").strip().rstrip("/")
    if env:
        return env
    h = (host_header or "").strip()
    if not _HOST_OK.match(h):
        h = f"127.0.0.1:{PORT}"
    return f"http://{h}"


def metadata_url(base: str, resource_path: str = "") -> str:
    """RFC 9728 s3.1: the well-known suffix goes BETWEEN the host and any
    path component of the resource identifier."""
    u = urlsplit(base)
    return f"{u.scheme}://{u.netloc}{PRM_PATH}{u.path.rstrip('/')}{resource_path}"


def protected_resource_metadata(resource: str) -> dict:
    """RFC 9728 s2. `authorization_servers` is OMITTED. RFC 9728 makes it
    OPTIONAL; MCP 2026-07-28 ("Authorization Server Location") says the
    document MUST list at least one -- and there is none: the key is an
    operator-issued API key. Naming a server that does not exist would send
    an OAuth-capable client into a flow that cannot finish, so the document
    says only what is true. The gap is written down in docs/TOOLS-API.md."""
    return {
        "resource": resource,
        "bearer_methods_supported": ["header"],
        "resource_name": "Yamadori tools API",
    }


def authenticate(auth_header: str | None) -> tuple[str | None, str]:
    """(account, reason); account None means refuse. The proxy's own check
    (accounts.identify), except that no registry is CLOSED here."""
    if not accounts.multi_tenant():
        return None, ("no accounts exist on this server, so no key can be "
                      "checked and the tools API refuses every request. "
                      "Operator: python mcp/accounts.py create <label>")
    return accounts.identify(auth_header)


_REDACT = (
    # A query string: search text, and any key a client put there by mistake.
    (re.compile(r"(\s/[^\s?]*)\?\S*"), r"\1?<query>"),
    (re.compile(r"(?i)\b(bearer|basic)\s+\S+"), r"\1 <redacted>"),
    (re.compile(r"\bym-[A-Za-z0-9_\-]{8,}"), "<redacted>"),
)


def redact(line: str) -> str:
    """A log line with no query string and nothing key-shaped in it."""
    for pat, rep in _REDACT:
        line = pat.sub(rep, line)
    return line


ROUTES = {
    "/search": do_search,
    "/definition": do_definition,
    "/references": do_references,
    "/status": do_status,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self) -> None:
        """A CORS grant for an ALLOWED Origin only, never '*' (it was '*'
        until 2026-09-26, so any page could read any answer)."""
        origin = self.headers.get("Origin")
        if origin is not None and origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin.strip())
            self.send_header("Access-Control-Expose-Headers",
                             "WWW-Authenticate")
            self.send_header("Vary", "Origin")

    def _send(self, code: int, payload: dict,
              headers: dict | None = None) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self._cors()
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    # -------------------------------------------------------------- the door
    def _content_length(self) -> int | None:
        """The declared body length; None when it is not a number >= 0."""
        raw = self.headers.get("Content-Length")
        if raw is None:
            return 0
        try:
            n = int(raw)
        except ValueError:
            return None
        return n if n >= 0 else None

    def _drain(self) -> None:
        """Read and discard a refused request's body (up to DRAIN_LIMIT) so
        the refusal reaches the client; the connection then closes."""
        self.close_connection = True
        n = self._content_length() or 0
        if 0 < n <= DRAIN_LIMIT:
            try:
                self.rfile.read(n)
            except OSError:
                pass

    def _refuse(self, code: int, path: str, why: str, *,
                challenge: str | None = None) -> None:
        """403 (Origin) or 401 (key). /mcp gets a JSON-RPC error with no id
        (streamable-http, "Security & Endpoint"); a REST route the tools'
        own situation / retryable / remedy shape. `why` never holds a key
        (accounts.identify's reasons name the situation, not the value)."""
        sys.stderr.write(redact(f"{self.address_string()} - refused {code} "
                                f"{self.command} {path}: {why}") + "\n")
        self._drain()
        if code == 403:
            remedy = ("A browser page may call this API only from an origin "
                      "the operator lists in YAMADORI_TOOLS_ALLOWED_ORIGINS; "
                      "a non-browser client sends no Origin header.")
        else:
            remedy = ("Send `Authorization: Bearer <key>` with an account key "
                      "from the operator (python mcp/accounts.py create "
                      "<label>) -- the same key as the proxy on :1234, never "
                      "in the query string.")
        if path == MCP_PATH:
            payload = {"jsonrpc": "2.0", "error": {
                "code": -32001 if code == 401 else -32003,
                "message": f"{why}. {remedy}"}}
        else:
            payload = {"error": "unauthorized" if code == 401 else "forbidden",
                       "message": why, "retryable": False, "remedy": remedy}
        self._send(code, payload,
                   {"WWW-Authenticate": challenge} if challenge else None)

    def _origin_ok(self, path: str) -> bool:
        if origin_allowed(self.headers.get("Origin")):
            return True
        self._refuse(403, path, "Origin not allowed")
        return False

    def _gate(self, path: str) -> bool:
        """True when the request may go on; else a refusal has been sent.
        Origin first, on every path; then the key, on all but OPEN_PATHS."""
        if not self._origin_ok(path):
            return False
        if path in OPEN_PATHS or path.startswith(PRM_PATH + "/"):
            return True
        auth = self.headers.get("Authorization")
        who, why = authenticate(auth)
        if who is not None:
            return True
        meta = metadata_url(public_base(self.headers.get("Host")),
                            MCP_PATH if path == MCP_PATH else "")
        challenge = f'Bearer resource_metadata="{meta}"'
        # RFC 6750 s3.1: an error code only when a credential WAS presented.
        if (auth or "").lower().startswith("bearer ") and \
                accounts.multi_tenant():
            challenge += ', error="invalid_token"'
        self._refuse(401, path, why, challenge=challenge)
        return False

    def _metadata(self, path: str) -> None:
        """The RFC 9728 documents: the root one (resource = the base) and the
        path-inserted one for the MCP endpoint (resource = base + /mcp), each
        also under the base's own path prefix when there is one (s3.1)."""
        base = public_base(self.headers.get("Host"))
        prefix = urlsplit(base).path.rstrip("/")
        docs = {}
        for pre in {"", prefix}:
            docs[PRM_PATH + pre] = base
            docs[PRM_PATH + pre + MCP_PATH] = base + MCP_PATH
        resource = docs.get(path.rstrip("/"))
        if resource is None:
            return self._send(404, {"error": f"no metadata at {path}",
                                    "documents": sorted(docs)})
        return self._send(200, protected_resource_metadata(resource))

    def _dispatch(self, path: str, args: dict) -> None:
        try:
            if path == "/health":
                # LIVENESS ONLY (2026-09-26). It used to hand the index
                # statistics to anyone who asked. The watchdog and
                # deploy_check read the status code and nothing else
                # (PROTOCOL rule 12); the statistics are /status, keyed.
                return self._send(200, {"ok": True})
            if path == PRM_PATH or path.startswith(PRM_PATH + "/"):
                return self._metadata(path)
            if path in ("/", ""):
                return self._send(200, DESCRIPTION)
            if path == "/openapi.json":
                return self._send(200, openapi_spec())
            fn = ROUTES.get(path)
            if fn is None:
                return self._send(404, {"error": f"no such endpoint: {path}",
                                        "endpoints": sorted(ROUTES) + ["/openapi.json", "/health"]})
            return self._send(200, fn(args))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:                       # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_OPTIONS(self):                            # noqa: N802
        """A CORS preflight carries no credentials by design, so it needs no
        key -- but it is answered for an allowed Origin only."""
        u = urlparse(self.path)
        if not self._origin_ok(u.path):
            return
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        if self.headers.get("Origin") is not None:
            self.send_header("Access-Control-Allow-Methods",
                             "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers",
                             CORS_ALLOW_HEADERS)
            self.send_header("Access-Control-Max-Age", "600")
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):                                # noqa: N802
        u = urlparse(self.path)
        if not self._gate(u.path):
            return
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        self._dispatch(u.path, q)

    def do_POST(self):                               # noqa: N802
        u = urlparse(self.path)
        # The door BEFORE the body: an unauthenticated caller's bytes are
        # drained (capped) and never parsed.
        if not self._gate(u.path):
            return
        n = self._content_length()
        if n is None:
            self.close_connection = True
            return self._send(400, {"error": "bad Content-Length"})
        if n > MAX_BODY:
            self.close_connection = True
            return self._send(413, {"error": f"body over {MAX_BODY} bytes"})
        raw = self.rfile.read(n) if n else b"{}"
        try:
            args = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "body is not valid JSON"})
        if not isinstance(args, dict):
            return self._send(400, {"error": "the body must be one JSON object"})

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
                self._cors()
                self.end_headers()
                return
            return self._send(200, resp)

        self._dispatch(u.path, args)

    def log_message(self, fmt, *a):                  # quieter default logging
        # Never the key and never a query string (search text, or a key a
        # client put there by mistake): every line goes through redact().
        sys.stderr.write(redact("%s - %s" % (self.address_string(), fmt % a))
                         + "\n")


def main() -> None:
    # summarize_text generates through mcp/model.py in THIS process; its
    # tokens go to the token ledger (mcp/token_ledger.py) like every other.
    import token_ledger
    token_ledger.enable()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"tools API on http://{HOST}:{PORT}  (spec at /openapi.json)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
