#!/usr/bin/env python
"""The pinned GET: how the proxy itself fetches ONE named document.

What is left of mcp/research_tools.py (deep thinking's knowledge-base and web
tools, find_in_knowledge_base / read_web_page / search_web, REMOVED
2026-09-29 with deep thinking: docs/REMOVED.md; the way back is commit
e360d37). Its fetch layer stays because the MCP host reads through it
(mcp/mcp_host.py: the README fallback on GitHub and npm's abbreviated
packument for the versions' peer dependencies, 2026-09-29).

  fetch_named_file  GET one document the proxy asks for by name: every hop
                    (each redirect included) resolved ONCE, refused unless
                    every address is `ip.is_global`, and connected to at that
                    pinned address; GET only, exactly REQUEST_HEADERS (no
                    Cookie, no Authorization, no Referer); one deadline and a
                    byte cap; only text, markdown, HTML or JSON (or the one
                    NAMED_ACCEPT type asked for); a 429 or a CAPTCHA page
                    pauses the host (Refused RATE_LIMITED, retryable).

Raises Refused with a code, whether a retry can help, and the HTTP status
when the server answered (AGENTS.md "Failure returns carry the next step").
"""
from __future__ import annotations

import http.client
import ipaddress
import os
import re
import socket
import ssl
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# A site's own 429 pauses that host for its Retry-After as the server sent it
# (RFC 9110 10.2.3), else HOST_BACKOFF_S (a choice, carried from
# research_tools).
HOST_BACKOFF_S = 60.0

# THE FETCH (pre-deploy review, 2026-09-24; BLOCKS DEPLOY #1). The first
# version called skill_pipeline.fetch_source, a plain urlopen: a public URL
# could 302 to 127.0.0.1:11434 or a ZeroTier peer (10.242.x) and the GET
# happened before any check; the host was re-resolved inside urlopen (DNS
# rebinding); 100.64.0.0/10 passed (`is_private` is False for it). Now every
# hop -- each redirect included -- is resolved ONCE, refused unless every address is
# `ip.is_global`, and connected to at that pinned address (the Host header
# and TLS SNI carry the name); one deadline covers the whole fetch and a byte
# cap stops a server that never ends. All numbers are choices.
FETCH_DEADLINE = float(os.environ.get("YAMADORI_WEB_DEADLINE", "30"))
FETCH_MAX_BYTES = int(os.environ.get("YAMADORI_WEB_MAX_BYTES",
                                     str(1024 * 1024)))
MAX_REDIRECTS = 5
# THE REQUEST (operator, 2026-09-25: "be careful of request params and
# headers, and keep it on get requests"). Every request this module sends --
# the document, a redirect hop -- goes through _send: GET, and exactly
# these headers, the same on every hop. http.client adds only Host (the
# name, over the pinned address). Nothing is derived from the conversation:
# no Cookie (Set-Cookie is never read, so nothing carries across hops), no
# Authorization, no Referer. The User-Agent names the service plainly.
UA_TOKEN = "yamadori-research"
UA = (f"{UA_TOKEN}/2 (documentation reader of Yamadori, a self-hosted "
      f"coding assistant; one page at a time; honours robots.txt)")
ACCEPT = ("text/html, application/xhtml+xml, text/markdown, "
          "text/plain;q=0.9, application/json;q=0.8")
METHOD = "GET"
REQUEST_HEADERS = (("User-Agent", UA), ("Accept", ACCEPT),
                   ("Accept-Language", "en"),
                   ("Accept-Encoding", "identity"))
# A NAMED DOCUMENT (fetch_named_file) may ask for one other representation,
# and only these: the Accept value replaces ACCEPT for that one request, every
# other header is REQUEST_HEADERS, and the answer may carry that media type.
#   npm's ABBREVIATED packument (npm registry API docs, "Package metadata":
#   `Accept: application/vnd.npm.install-v1+json` returns every version's
#   install fields -- dependencies, peerDependencies, peerDependenciesMeta,
#   dist -- without the readme and per-version metadata of the full one).
NAMED_ACCEPT = ("application/vnd.npm.install-v1+json",)
PAGE_TYPES = {"text/plain", "text/markdown", "text/x-markdown", "text/html",
              "application/xhtml+xml", "text/x-rst", "application/markdown",
              "application/json"}
# A block page rather than an answer: a 403/503 whose body says CAPTCHA or
# challenge is a rate limit, not a refusal (2026-09-25).
_CHALLENGE = re.compile(r"(?i)captcha|cf-chl|challenge-platform|are you a "
                        r"robot|unusual traffic|too many requests")
_LOCAL_NAMES = (".localhost", ".local", ".internal", ".lan", ".home",
                ".zt", ".ts.net", ".home.arpa")


class Refused(Exception):
    """A fetch that must not happen, or that the server refused for good."""

    def __init__(self, code: str, why: str, retryable: bool = False,
                 after: float | None = None, status: int | None = None):
        super().__init__(why)
        self.code, self.why, self.retryable = code, why, retryable
        self.after = after          # seconds until a retry can help
        self.status = status        # the HTTP status, when the server answered


def _resolve(host: str) -> list[str]:
    """Every address `host` resolves to. A seam: tests replace it."""
    return [i[4][0].split("%")[0] for i in socket.getaddrinfo(
        host, None, type=socket.SOCK_STREAM)]


def _pin(host: str) -> str:
    """The ONE address to connect to for `host`, resolved once. Refused
    unless every address it resolves to is globally routable
    (`ip.is_global`: not loopback, private, CGNAT 100.64/10 -- ZeroTier's
    and Tailscale's -- link-local, reserved or documentation)."""
    h = (host or "").strip("[]").lower()
    if not h or h == "localhost" or h.endswith(_LOCAL_NAMES):
        raise Refused("REFUSED_ADDRESS", f"{host!r} is a local name")
    try:
        addrs = _resolve(h)
    except OSError as e:
        raise Refused("FETCH_FAILED", f"{host!r} does not resolve ({e})",
                      retryable=True) from e
    if not addrs:
        raise Refused("FETCH_FAILED", f"{host!r} does not resolve",
                      retryable=True)
    for a in addrs:
        ip = ipaddress.ip_address(a)
        if not ip.is_global or ip.is_multicast:
            raise Refused("REFUSED_ADDRESS",
                          f"{host!r} resolves to {ip}, a non-public address")
    return addrs[0]


class _PinnedHTTP(http.client.HTTPConnection):
    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self):
        self.sock = socket.create_connection((self._ip, self.port),
                                             self.timeout)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout,
                         context=ssl.create_default_context())
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        # SNI and certificate verification against the NAME, over the
        # pinned address.
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _send(conn: http.client.HTTPConnection, method: str, path: str,
          accept: str | None = None) -> None:
    """The ONE way a request leaves (THE REQUEST): GET only, exactly
    REQUEST_HEADERS -- with Accept replaced only by a NAMED_ACCEPT value.
    Anything else is refused before a byte is sent."""
    if method != METHOD:
        raise Refused("REFUSED_METHOD", f"{method!r} is not sent: the web "
                      f"tools only GET")
    if accept is not None and accept not in NAMED_ACCEPT:
        raise Refused("REFUSED_HEADER", f"Accept {accept[:80]!r} is not one "
                      f"the fetch sends")
    conn.putrequest(METHOD, path, skip_accept_encoding=True)
    for k, v in REQUEST_HEADERS:
        conn.putheader(k, accept if (k == "Accept" and accept) else v)
    conn.endheaders()


def _get(url: str, deadline: float, accept: str | None = None) -> dict:
    """GET `url`, every hop pinned and re-validated. A redirect's Location
    is followed as the server sent it (its query included) and checked like
    the first hop. Returns {status, headers, body, url, cut}. Raises
    Refused."""
    for _hop in range(MAX_REDIRECTS + 1):
        p = urllib.parse.urlsplit(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise Refused("REFUSED_ADDRESS",
                          f"a redirect led to {url[:120]!r}, not http(s)")
        if p.username or p.password:
            raise Refused("REFUSED_ADDRESS", "a URL with credentials in it")
        ip = _pin(p.hostname)
        left = deadline - time.time()
        if left <= 0:
            raise Refused("FETCH_FAILED", f"the fetch passed its "
                          f"{FETCH_DEADLINE:g} s deadline", retryable=True)
        port = p.port or (443 if p.scheme == "https" else 80)
        cls = _PinnedHTTPS if p.scheme == "https" else _PinnedHTTP
        conn = cls(p.hostname, port, ip, timeout=min(15.0, left))
        path = (p.path or "/") + (f"?{p.query}" if p.query else "")
        try:
            _send(conn, METHOD, path, accept)
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.getheader("Location") or ""
                resp.close()
                if not loc:
                    raise Refused("FETCH_REFUSED", f"HTTP {resp.status} "
                                  f"with no Location")
                url = urllib.parse.urljoin(url, loc)
                continue
            body = bytearray()
            cut = False
            while True:
                left = deadline - time.time()
                if left <= 0:
                    raise Refused("FETCH_FAILED", f"the page did not finish "
                                  f"within the {FETCH_DEADLINE:g} s "
                                  f"deadline", retryable=True)
                if conn.sock is not None:
                    conn.sock.settimeout(min(15.0, left))
                block = resp.read1(64 * 1024) if hasattr(resp, "read1") \
                    else resp.read(64 * 1024)
                if not block:
                    break
                body += block
                if len(body) >= FETCH_MAX_BYTES:
                    body = body[:FETCH_MAX_BYTES]
                    cut = True
                    break
            return {"status": resp.status, "headers": resp.headers,
                    "body": bytes(body), "url": url, "cut": cut}
        except Refused:
            raise
        except (OSError, http.client.HTTPException) as e:
            raise Refused("FETCH_FAILED", f"GET {url[:120]} failed: "
                          f"{type(e).__name__}: {str(e)[:120]}",
                          retryable=True) from e
        finally:
            conn.close()
    raise Refused("FETCH_REFUSED", f"more than {MAX_REDIRECTS} redirects")


def _refuse_ext(url: str) -> None:
    ext = os.path.splitext(urllib.parse.urlsplit(url).path)[1].lower()
    import skill_pipeline
    if ext in skill_pipeline.REFUSED_EXT:
        raise Refused("FETCH_REFUSED", f"{ext} files are scripts or binaries")


def fetch_named_file(url: str, deadline: float | None = None,
                     accept: str | None = None) -> tuple[bytes, dict]:
    """ONE NAMED DOCUMENT the proxy itself asks for -- not a page the model
    picked: THE FETCH above (every hop pinned and `ip.is_global`, GET only,
    exactly REQUEST_HEADERS, no cookie, the byte cap) and the same answer
    checks, under the caller's one `deadline`, WITHOUT robots.txt. robots.txt
    is the crawler exclusion protocol (RFC 9309: rules for crawlers); a
    request for one file by its exact name, made because a tool call asked
    for that package's README, is not crawling -- the package onboarding
    reads raw.githubusercontent.com files the same way (package_net.raw,
    no robots.txt), and raw.githubusercontent.com/robots.txt answers 404
    anyway (the coordinator checked it, 2026-09-29). `accept`: one of
    NAMED_ACCEPT, sent in place of ACCEPT, and the answer may be that type.
    Used by mcp_host (the README fallback and the versions' peer
    dependencies, 2026-09-29). A seam: tests replace it."""
    _refuse_ext(url)
    return _checked(_get(url, deadline or time.time() + FETCH_DEADLINE,
                         accept), url, (accept,) if accept else ())


def _checked(got: dict, url: str, extra_types: tuple = ()
             ) -> tuple[bytes, dict]:
    """What a fetch may pass on: a 429 or a CAPTCHA page pauses the host,
    another 4xx/5xx is refused (with its status), and only text, markdown,
    HTML or JSON without NUL bytes is returned."""
    st = got["status"]
    if st == 429 or (st in (403, 503) and _CHALLENGE.search(
            got["body"][:65536].decode("utf-8", "replace"))):
        host = urllib.parse.urlsplit(got["url"]).hostname or ""
        after = _retry_after(got["headers"])
        _HOST_PAUSED[host.lower()] = time.time() + after
        kind = "rate limit" if st == 429 else "a CAPTCHA or bot check"
        raise Refused("RATE_LIMITED", f"{host} answered HTTP {st} ({kind}); "
                      f"this host is paused for {after:.0f} s",
                      retryable=True, after=after, status=st)
    if st >= 400:
        raise Refused("FETCH_FAILED" if st >= 500 or st in (408, 429)
                      else "FETCH_REFUSED", f"GET {url[:120]} answered HTTP "
                      f"{st}", retryable=st >= 500 or st in (408, 429),
                      status=st)
    ctype = (got["headers"].get_content_type()
             if got["headers"] is not None else "text/plain")
    if ctype not in PAGE_TYPES and ctype not in extra_types:
        raise Refused("FETCH_REFUSED", f"the page is {ctype}, not text, "
                      f"markdown or HTML")
    if b"\x00" in got["body"][:8192]:
        raise Refused("FETCH_REFUSED", "the page contains NUL bytes: a "
                      "binary, not a document")
    return got["body"], {"content_type": ctype, "final_url": got["url"],
                         "charset": got["headers"].get_content_charset(),
                         "cut_bytes": got["cut"], "status": st}


# Hosts that answered 429 or a CAPTCHA page: host -> time a retry may help.
# Process-wide, so a later fetch does not hammer a host that just pushed back.
_HOST_PAUSED: dict[str, float] = {}


def _retry_after(headers, default: float = HOST_BACKOFF_S) -> float:
    """Seconds from a Retry-After header (seconds form only), as sent (at
    least 1 s); `default` when there is none."""
    try:
        v = float((headers.get("Retry-After") if headers is not None
                   else None) or "")
    except (TypeError, ValueError):
        v = default
    return max(1.0, v)
