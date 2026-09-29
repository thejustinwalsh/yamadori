#!/usr/bin/env python
"""The tools API's door (:1235), asserted over a real socket. No model, no
real registry, no real index.

WHAT THIS IS GATING (operator, 2026-09-26; SELF-IMPROVEMENT-LOG #48)

`mcp/tools_api.py` listens on 0.0.0.0:1235 on purpose -- remote MCP clients
on the operator's laptop and phone reach it over ZeroTier. Until 2026-09-26
it had NO auth: anything on the LAN, on ZeroTier, and (through DNS
rebinding) any web page open in the operator's browser could call its REST
routes and /mcp, including `read_file_range` and `run_check`. The rules
asserted here, each from the MCP spec revision 2026-07-28:

  1. ORIGIN (basic/transports/streamable-http, "Security & Endpoint"): an
     Origin header that is present and not allowed is 403 -- on /mcp with a
     JSON-RPC error that has no id -- on EVERY route, a valid key included.
     No Origin (a non-browser client) passes this step.
  2. BEARER (basic/authorization, "Access Token Usage"): `Authorization:
     Bearer <key>` on every request to /mcp and the REST routes, checked
     against the SAME account keys as :1234 (mcp/accounts.py). Missing or
     invalid is 401 with `WWW-Authenticate: Bearer resource_metadata="..."`
     (RFC 9728 s5.1). Never a key in the query string, never X-API-Key.
  3. PROTECTED RESOURCE METADATA (RFC 9728) is served unauthenticated.
  4. /health is liveness only: no index statistics without a key.
  5. The key never reaches the log.

A NO-REGISTRY SERVER IS CLOSED HERE. accounts.identify() admits everyone as
`anonymous` when no accounts.json exists (single-user mode, the proxy's
choice). The tools API refuses instead: it exists to be reached from other
machines, and "no keys exist" cannot mean "no key needed" for a door that
reads files.

Everything runs against a ThreadingHTTPServer on 127.0.0.1:<ephemeral>, with
YAMADORI_ACCOUNTS_DIR, CODE_INDEX_DB and the other stores pointed at temp
paths BEFORE any import, so the real index/accounts/accounts.json is never
opened and no real key exists in this process.
"""
from __future__ import annotations

import http.client
import io
import json
import os
import sys
import tempfile
import threading
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_TMP = tempfile.mkdtemp(prefix="yamadori_test_tools_api_auth_")
os.environ["YAMADORI_ACCOUNTS_DIR"] = os.path.join(_TMP, "accounts")
os.environ["CODE_INDEX_DB"] = os.path.join(_TMP, "code.sqlite3")
os.environ["RINGS_DB"] = os.path.join(_TMP, "rings.sqlite3")
os.environ["YAMADORI_CORPUS_DB"] = os.path.join(_TMP, "corpus.sqlite3")
os.environ["YAMADORI_NEBARI_DB"] = os.path.join(_TMP, "nebari.sqlite3")
os.environ["YAMADORI_JOBS_DB"] = os.path.join(_TMP, "jobs.sqlite3")
os.environ["YAMADORI_SKILLS_DIR"] = os.path.join(_TMP, "skills")
os.environ["YAMADORI_PKG_DIR"] = os.path.join(_TMP, "pkgs")
for _k in ("YAMADORI_TOOLS_ALLOWED_ORIGINS", "YAMADORI_TOOLS_PUBLIC_BASE"):
    os.environ.pop(_k, None)

import accounts  # noqa: E402
import tools_api  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

REAL_STORE = os.path.abspath(os.path.join(HERE, "..", "index", "accounts"))
PRM = "/.well-known/oauth-protected-resource"

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    _results.append((bool(ok), name, detail))
    return bool(ok)


# ------------------------------------------------------------------ fixture
_SRV = ThreadingHTTPServer(("127.0.0.1", 0), tools_api.Handler)
_SRV.daemon_threads = True
PORT = _SRV.server_address[1]
BASE = f"http://127.0.0.1:{PORT}"
threading.Thread(target=_SRV.serve_forever, daemon=True).start()

# The server's log goes to sys.stderr, looked up on every write, so swapping
# it for a buffer captures every line the handler threads write.
_LOG = io.StringIO()
_REAL_STDERR = sys.stderr


def no_registry() -> None:
    for p in (accounts.REGISTRY,):
        if os.path.exists(p):
            os.remove(p)


KEY = ""


def with_registry() -> str:
    """A fresh registry holding one account; its key (test-only)."""
    no_registry()
    return accounts.create("tools-api-test")


def req(method: str, path: str, *, key: str | None = None,
        body: dict | bytes | None = None, headers: dict | None = None
        ) -> tuple[int, dict, dict | str]:
    """(status, lower-cased headers, parsed JSON or text)."""
    h = dict(headers or {})
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h.setdefault("Content-Type", "application/json")
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    try:
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read().decode("utf-8", "replace")
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        status = r.status
    finally:
        c.close()
    try:
        return status, hdrs, json.loads(raw) if raw else ""
    except ValueError:
        return status, hdrs, raw


LIST = {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}}


# -------------------------------------------------------------------- tests
def test_isolation():
    check(os.path.abspath(accounts.STORE) != REAL_STORE,
          "the real account store is never touched", accounts.STORE)
    check(os.path.abspath(os.environ["CODE_INDEX_DB"]).startswith(
        os.path.abspath(_TMP)), "the index is a temp file")


def test_no_key_is_401_with_the_challenge():
    global KEY
    KEY = with_registry()
    for method, path, body in (("POST", "/mcp", LIST),
                               ("GET", "/status", None),
                               ("POST", "/definition", {"symbol": "x"}),
                               ("GET", "/definition?symbol=x", None),
                               ("POST", "/search", {"query": "x"}),
                               ("GET", "/", None),
                               ("GET", "/openapi.json", None)):
        st, h, d = req(method, path, body=body)
        check(st == 401, f"no key: {method} {path} is 401 (was 200 before "
              f"2026-09-26)", f"HTTP {st} {str(d)[:120]}")
        www = h.get("www-authenticate", "")
        check(www.startswith("Bearer ") and "resource_metadata=\"" in www,
              f"no key: {method} {path} carries WWW-Authenticate: Bearer "
              f"resource_metadata", www)
        check("error=" not in www,
              f"no key: {method} {path}: no error code when no credential "
              f"was sent (RFC 6750 s3.1)", www)
    st, h, d = req("POST", "/mcp", body=LIST)
    www = h.get("www-authenticate", "")
    check(f'resource_metadata="{BASE}{PRM}/mcp"' in www,
          "on /mcp the challenge names the path-inserted metadata, whose "
          "`resource` is the MCP endpoint's own URL (RFC 9728 s3.3)", www)
    check(isinstance(d, dict) and d.get("jsonrpc") == "2.0"
          and "error" in d and "id" not in d and "result" not in d,
          "on /mcp the 401 body is a JSON-RPC error with no id, no result",
          str(d)[:200])
    st, h, d = req("GET", "/status")
    check(f'resource_metadata="{BASE}{PRM}"' in h.get("www-authenticate", ""),
          "on a REST route the challenge names the root metadata",
          h.get("www-authenticate", ""))
    check(isinstance(d, dict) and d.get("retryable") is False
          and d.get("remedy"),
          "a REST 401 says it is not retryable as is, and the remedy",
          str(d)[:200])


def test_bad_key_is_401_invalid_token():
    for bad in ("ym-not-a-real-key", KEY + "x", ""):
        st, h, d = req("POST", "/mcp", key=bad, body=LIST)
        check(st == 401, f"a wrong key is 401 ({len(bad)} chars)",
              f"HTTP {st}")
        if bad:
            check('error="invalid_token"' in h.get("www-authenticate", ""),
                  "a presented, invalid key gets error=\"invalid_token\"",
                  h.get("www-authenticate", ""))
    st, h, d = req("POST", "/mcp", body=LIST,
                   headers={"Authorization": f"Basic {KEY}"})
    check(st == 401, "a non-Bearer scheme is refused", f"HTTP {st}")


def test_key_elsewhere_is_not_accepted():
    for q in (f"/status?access_token={KEY}", f"/status?key={KEY}",
              f"/definition?symbol=x&api_key={KEY}"):
        st, _, _ = req("GET", q)
        check(st == 401, "a key in the query string is never accepted",
              f"HTTP {st} {q.split('?')[0]}")
    st, _, _ = req("POST", "/mcp", body=LIST, headers={"X-API-Key": KEY})
    check(st == 401, "X-API-Key is not accepted (no in-repo client needs it)",
          f"HTTP {st}")


def test_good_key_is_200():
    st, h, d = req("POST", "/mcp", key=KEY, body=LIST)
    names = [t.get("name") for t in ((d or {}).get("result") or {})
             .get("tools", [])] if isinstance(d, dict) else []
    check(st == 200 and "read_file_range" in names and d.get("id") == 7,
          "good key: /mcp tools/list is 200 and lists the tools",
          f"HTTP {st} {names[:4]}")
    st, _, d = req("POST", "/mcp", key=KEY,
                   body={"jsonrpc": "2.0", "method":
                         "notifications/initialized"})
    check(st == 202, "good key: a notification is 202", f"HTTP {st}")
    st, _, d = req("POST", "/definition", key=KEY, body={"symbol": "nope"})
    check(st == 200 and isinstance(d, dict) and d.get("count") == 0,
          "good key: POST /definition is 200", f"HTTP {st} {str(d)[:120]}")
    st, _, d = req("GET", "/definition?symbol=nope", key=KEY)
    check(st == 200, "good key: GET /definition?symbol= is 200", f"HTTP {st}")
    st, _, d = req("GET", "/status", key=KEY)
    check(st == 200 and isinstance(d, dict) and "chunks" in d,
          "good key: /status (the index statistics) is 200", str(d)[:120])
    st, _, d = req("GET", "/openapi.json", key=KEY)
    sec = (((d or {}).get("components") or {}).get("securitySchemes")
           if isinstance(d, dict) else None) or {}
    check(st == 200 and any(v.get("scheme") == "bearer"
                            for v in sec.values()),
          "the OpenAPI spec declares the bearer scheme", str(sec)[:120])


def test_prm_is_served_unauthenticated():
    for path, resource in ((PRM, BASE), (PRM + "/mcp", BASE + "/mcp")):
        st, h, d = req("GET", path)
        check(st == 200 and isinstance(d, dict), f"{path} is 200 without a "
              f"key", f"HTTP {st}")
        if not isinstance(d, dict):
            continue
        check(d.get("resource") == resource,
              f"{path}: `resource` is the URL it describes (RFC 9728 s3.3)",
              str(d.get("resource")))
        check(d.get("bearer_methods_supported") == ["header"],
              f"{path}: the key goes in the header only",
              str(d.get("bearer_methods_supported")))
        check("authorization_servers" not in d,
              f"{path}: no authorization server is claimed (there is none "
              f"yet; the gap is in docs/TOOLS-API.md)", str(d)[:200])
        check(h.get("content-type", "").startswith("application/json"),
              f"{path}: JSON", h.get("content-type", ""))
    os.environ["YAMADORI_TOOLS_PUBLIC_BASE"] = "https://ai.example.test/tools/"
    try:
        _, h, d = req("GET", PRM)
        check(isinstance(d, dict)
              and d.get("resource") == "https://ai.example.test/tools",
              "YAMADORI_TOOLS_PUBLIC_BASE sets the canonical URL (no "
              "trailing slash)", str(d)[:160])
        _, h, _ = req("POST", "/mcp", body=LIST)
        check('resource_metadata="https://ai.example.test/.well-known/'
              'oauth-protected-resource/tools/mcp"'
              in h.get("www-authenticate", ""),
              "behind a prefix the metadata URL inserts the well-known "
              "suffix between host and path (RFC 9728 s3.1)",
              h.get("www-authenticate", ""))
        st, _, d = req("GET", PRM + "/tools/mcp")
        check(st == 200 and isinstance(d, dict)
              and d.get("resource") == "https://ai.example.test/tools/mcp",
              "and that path-inserted document is served", f"HTTP {st}")
    finally:
        os.environ.pop("YAMADORI_TOOLS_PUBLIC_BASE", None)
    st, h, _ = req("GET", "/status", headers={"Host": 'evil", error="x'})
    www = h.get("www-authenticate", "")
    check(st == 401 and "evil" not in www and www.count('"') == 2,
          "a malformed Host cannot write into the challenge (it falls back "
          "to loopback)", www)


def test_bad_origin_is_403_everywhere():
    for method, path, body in (("POST", "/mcp", LIST),
                               ("GET", "/status", None),
                               ("GET", "/health", None),
                               ("GET", PRM, None),
                               ("POST", "/search", {"query": "x"})):
        for origin in ("https://evil.example", "null",
                       "http://127.0.0.1:1235.evil.example"):
            st, h, d = req(method, path, key=KEY, body=body,
                           headers={"Origin": origin})
            check(st == 403, f"Origin {origin}: {method} {path} is 403, "
                  f"even with a valid key", f"HTTP {st}")
            check(h.get("access-control-allow-origin") is None,
                  f"Origin {origin}: {method} {path}: no CORS grant",
                  str(h.get("access-control-allow-origin")))
    st, _, d = req("POST", "/mcp", key=KEY, body=LIST,
                   headers={"Origin": "https://evil.example"})
    check(isinstance(d, dict) and d.get("jsonrpc") == "2.0"
          and "error" in d and "id" not in d,
          "on /mcp the 403 body is a JSON-RPC error with no id "
          "(streamable-http, Security & Endpoint)", str(d)[:200])
    st, _, _ = req("OPTIONS", "/mcp", headers={
        "Origin": "https://evil.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type"})
    check(st == 403, "a preflight from a foreign origin is 403", f"HTTP {st}")


def test_allowed_origin_passes_and_is_configurable():
    os.environ["YAMADORI_TOOLS_ALLOWED_ORIGINS"] = (
        "https://Dash.Example.test/, http://10.0.0.44:5173")
    try:
        st, h, d = req("POST", "/mcp", key=KEY, body=LIST,
                       headers={"Origin": "https://dash.example.test"})
        check(st == 200, "an allowed Origin with a key is 200 (case and a "
              "trailing slash in the setting do not matter)", f"HTTP {st}")
        check(h.get("access-control-allow-origin")
              == "https://dash.example.test"
              and "origin" in h.get("vary", "").lower(),
              "an allowed Origin gets its own CORS grant, never '*'",
              str(h.get("access-control-allow-origin")))
        st, h, _ = req("POST", "/mcp", body=LIST,
                       headers={"Origin": "http://10.0.0.44:5173"})
        check(st == 401, "an allowed Origin still needs the key", f"HTTP {st}")
        check("www-authenticate" in (h.get("access-control-expose-headers")
                                     or "").lower(),
              "and a browser may read the challenge",
              str(h.get("access-control-expose-headers")))
        st, h, _ = req("OPTIONS", "/mcp", headers={
            "Origin": "https://dash.example.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"})
        check(st == 204 and "authorization" in (
            h.get("access-control-allow-headers") or "").lower(),
            "a preflight from an allowed origin is 204 and allows "
            "Authorization", f"HTTP {st} {h.get('access-control-allow-headers')}")
        st, _, _ = req("GET", "/status", key=KEY,
                       headers={"Origin": "https://dash.example.test:444"})
        check(st == 403, "another port of an allowed host is not allowed",
              f"HTTP {st}")
    finally:
        os.environ.pop("YAMADORI_TOOLS_ALLOWED_ORIGINS", None)
    st, h, _ = req("GET", "/status", key=KEY)
    check(st == 200 and h.get("access-control-allow-origin") is None,
          "no Origin: no CORS header at all (it used to be '*')",
          str(h.get("access-control-allow-origin")))


def test_health_is_liveness_only():
    st, _, d = req("GET", "/health")
    check(st == 200 and isinstance(d, dict) and d.get("ok") is True,
          "/health answers 200 without a key (watchdog, deploy_check)",
          f"HTTP {st} {str(d)[:120]}")
    leaked = sorted(set(d) & {"chunks", "files", "definitions", "references"}
                    ) if isinstance(d, dict) else []
    check(not leaked, "/health carries no index statistics", str(leaked))


def test_no_registry_is_closed():
    global KEY
    no_registry()
    try:
        st, _, d = req("POST", "/mcp", key="ym-anything", body=LIST)
        check(st == 401, "no accounts.json: refused, not single-user open",
              f"HTTP {st}")
        st, _, d = req("GET", "/status")
        check(st == 401 and "accounts.py create" in json.dumps(d),
              "and the refusal names the remedy", str(d)[:200])
        st, _, _ = req("GET", "/health")
        check(st == 200, "/health still answers", f"HTTP {st}")
        with open(accounts.REGISTRY, "w", encoding="utf-8") as f:
            f.write("{not json")
        st, _, _ = req("POST", "/mcp", key=KEY, body=LIST)
        check(st == 401, "an unreadable registry is closed too", f"HTTP {st}")
    finally:
        KEY = with_registry()


def test_bodies_and_lengths():
    st, _, _ = req("POST", "/mcp", body=b"x" * 4096)
    check(st == 401, "an unauthenticated body is refused before it is "
          "parsed", f"HTTP {st}")
    st, _, d = req("POST", "/mcp", key=KEY, body=b"{not json")
    check(st == 400, "with a key, bad JSON is 400", f"HTTP {st}")
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    try:
        c.putrequest("POST", "/mcp")
        c.putheader("Authorization", f"Bearer {KEY}")
        c.putheader("Content-Length", "-5")
        c.endheaders()
        st = c.getresponse().status
    finally:
        c.close()
    check(st == 400, "a negative Content-Length is 400", f"HTTP {st}")
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    try:
        c.putrequest("POST", "/mcp")
        c.putheader("Authorization", f"Bearer {KEY}")
        c.putheader("Content-Length", str(tools_api.MAX_BODY + 1))
        c.endheaders()
        st = c.getresponse().status
    finally:
        c.close()
    check(st == 413, "a body over MAX_BODY is 413, never read", f"HTTP {st}")


def test_the_key_never_reaches_the_log():
    before = len(_LOG.getvalue())
    req("POST", "/mcp", key=KEY, body=LIST)
    req("POST", "/mcp", key=KEY + "wrong", body=LIST)
    req("GET", f"/status?access_token={KEY}")
    req("GET", f"/definition?symbol=x&key={KEY}", key=KEY)
    req("POST", "/mcp", body=LIST, headers={"X-API-Key": KEY})
    req("GET", "/status", key=KEY, headers={"Origin": "https://evil.example"})
    log = _LOG.getvalue()[before:]
    check(bool(log.strip()), "the requests were logged", repr(log[:80]))
    check(KEY not in log and KEY[3:20] not in log,
          "the key (or a fragment of it) appears nowhere in the log",
          "<redacted>" if KEY in log else "")
    check("access_token=" not in log and "key=" not in log
          and "symbol=x" not in log,
          "query strings are not logged", log[:200].replace(KEY, "<KEY>"))
    check("refused 401" in log and "refused 403" in log,
          "refusals are logged, with the reason and never the key",
          log[:300].replace(KEY, "<KEY>"))


def test_the_bridge_sends_the_key():
    """mcp/mcp_bridge.py (the stdio relay a remote client spawns) against
    this server: YAMADORI_API_KEY becomes the Bearer header; without it the
    client gets a JSON-RPC error that names the remedy, not a hang."""
    import subprocess
    bridge = os.path.join(HERE, "mcp_bridge.py")
    line = json.dumps(LIST) + "\n"
    for key, want in ((KEY, "result"), ("", "error")):
        env = {k: v for k, v in os.environ.items() if k != "YAMADORI_API_KEY"}
        if key:
            env["YAMADORI_API_KEY"] = key
        p = subprocess.run([sys.executable, bridge, "--url", f"{BASE}/mcp"],
                           input=line, capture_output=True, text=True,
                           timeout=60, env=env)
        try:
            d = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            d = {}
        check(want in d, f"bridge {'with' if key else 'without'} "
              f"YAMADORI_API_KEY: a JSON-RPC {want}",
              (p.stdout + p.stderr)[-200:].replace(KEY, "<KEY>"))
        if not key:
            check("YAMADORI_API_KEY" in json.dumps(d),
                  "and the error names the remedy", str(d)[:200])
        check(KEY not in p.stderr, "the bridge never prints the key")


def test_the_websocket_door():
    """mcp/mcp_ws.py binds 0.0.0.0 too. Nothing starts it today, but its
    gate is the tools API's: foreign Origin 403, no key 401, key 101."""
    import asyncio
    try:
        import websockets
        import mcp_ws
    except ImportError as e:
        check(False, "the websocket door could not be tested", str(e))
        return

    async def run() -> dict:
        out = {}
        async with websockets.serve(mcp_ws.handler, "127.0.0.1", 0,
                                    process_request=mcp_ws.gate) as srv:
            port = srv.sockets[0].getsockname()[1]
            url = f"ws://127.0.0.1:{port}/mcp"
            for name, hdrs in (("nokey", {}),
                               ("origin", {"Authorization": f"Bearer {KEY}",
                                           "Origin": "https://evil.example"})):
                try:
                    async with websockets.connect(url, extra_headers=hdrs):
                        out[name] = 101
                except websockets.exceptions.InvalidStatusCode as e:
                    out[name] = e.status_code
            async with websockets.connect(
                    url, extra_headers={"Authorization": f"Bearer {KEY}"}) as ws:
                await ws.send(json.dumps(LIST))
                out["list"] = json.loads(await ws.recv())
        return out

    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf        # mcp_ws prints connects and refusals
    try:
        out = asyncio.run(run())
    finally:
        sys.stdout = real
    check(out.get("nokey") == 401, "websocket without a key: 401",
          str(out.get("nokey")))
    check(out.get("origin") == 403, "websocket from a foreign Origin: 403",
          str(out.get("origin")))
    check("result" in (out.get("list") or {}),
          "websocket with the key: tools/list answers", str(out)[:160])


TESTS = [test_isolation, test_no_key_is_401_with_the_challenge,
         test_bad_key_is_401_invalid_token, test_key_elsewhere_is_not_accepted,
         test_good_key_is_200, test_prm_is_served_unauthenticated,
         test_bad_origin_is_403_everywhere,
         test_allowed_origin_passes_and_is_configurable,
         test_health_is_liveness_only, test_no_registry_is_closed,
         test_bodies_and_lengths, test_the_key_never_reaches_the_log,
         test_the_bridge_sends_the_key, test_the_websocket_door]


def main() -> int:
    sys.stderr = _LOG
    try:
        for fn in TESTS:
            n0 = len(_results)
            try:
                fn()
            except Exception:                                    # noqa: BLE001
                check(False, f"{fn.__name__} raised",
                      traceback.format_exc().strip().split("\n")[-1])
            for ok, name, detail in _results[n0:]:
                line = ("  pass  " if ok else "  FAIL  ") + name
                if not ok and detail:
                    line += f"   <- {detail}"
                print(line.replace(KEY, "<KEY>") if KEY else line)
    finally:
        sys.stderr = _REAL_STDERR
        _SRV.shutdown()
    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total and total else 1


if __name__ == "__main__":
    sys.exit(main())
