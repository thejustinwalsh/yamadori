#!/usr/bin/env python
"""GET ONLY: every network read a package onboarding makes.

docs/PACKAGE-ONBOARDING.md 2.3. Nothing is POSTed, nothing is installed, no
lifecycle script or example is run.

| what | where | how |
|---|---|---|
| registry metadata | registry.npmjs.org/<name>, pypi.org/pypi/<name>/json | documented APIs |
| GitHub refs and trees | api.github.com/repos/<o>/<r>/... | unauthenticated (operator, 2026-09-27, decision 8: "No GitHub token: unauthenticated GETs, with backoff on rate limits") |
| files at a commit | raw.githubusercontent.com/<o>/<r>/<sha>/<path> | a commit SHA in the URL makes the file immutable |
| tarballs | the packument's own `dist.tarball` | verified by deps.fetch_verified |

The fixed headers are the skill pipeline's (`skill_pipeline.UA`,
`yamadori-skill-worker/1`), an Accept line, and nothing else: no cookie is
kept (urllib keeps none), no Authorization, no Referer.

RATE LIMITS. GitHub answers 403 or 429 when the unauthenticated allowance is
spent and says when it resets (`X-RateLimit-Remaining: 0` with
`X-RateLimit-Reset`, or `Retry-After`). That is raised as `RateLimited`
carrying the SERVER's own time; the stage handler turns it into a deferral
(worker.Deferred -> jobs.defer: no attempt is spent), so the job waits
exactly as long as the server asked. No wait time here is ours.

TESTS replace `TRANSPORT` (a function url, headers -> Response) with a fake;
the offline suites never reach the network.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UA = "yamadori-skill-worker/1"          # skill_pipeline.UA, the one UA
TIMEOUT = 60                            # skill_pipeline.FETCH_TIMEOUT is 30 for
# a document; a packument of a long-lived package (three: ~20 MB of JSON) is
# larger, and deps.registry_history already reads it with 60.
REGISTRY = os.environ.get("NPM_REGISTRY", "https://registry.npmjs.org")
PYPI = "https://pypi.org/pypi"
GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"


@dataclass
class Response:
    status: int
    body: bytes = b""
    headers: dict = field(default_factory=dict)
    url: str = ""

    def json(self):
        return json.loads(self.body.decode("utf-8", "replace") or "null")

    def text(self) -> str:
        return self.body.decode("utf-8", "replace")


class RateLimited(Exception):
    """The server said wait; `until` is the server's own reset time."""

    def __init__(self, url: str, until: float, why: str):
        self.url, self.until, self.why = url, until, why
        super().__init__(why)


class TooLarge(Exception):
    """The body passed the cap the caller named (an existing limit)."""


def _urllib(url: str, headers: dict, max_bytes: int | None) -> Response:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read(64 * 1024)
        except Exception:                                        # noqa: BLE001
            pass
        return Response(e.code, body, {k.lower(): v for k, v in
                                       (e.headers or {}).items()}, url)
    with resp:
        buf = bytearray()
        while True:
            block = resp.read(1 << 16)
            if not block:
                break
            buf += block
            if max_bytes is not None and len(buf) > max_bytes:
                raise TooLarge(f"{url} is larger than {max_bytes:,} bytes")
        return Response(resp.status, bytes(buf),
                        {k.lower(): v for k, v in resp.headers.items()},
                        resp.geturl())


# Tests replace this: (url, headers, max_bytes) -> Response.
TRANSPORT = _urllib
# Every GET made in this process, for the stage records: (url, status, bytes).
LOG: list[tuple[str, int, int]] = []


def get(url: str, *, accept: str = "application/json",
        max_bytes: int | None = None) -> Response:
    """One GET with the fixed headers. Raises RateLimited when the server
    says to wait; any other status is returned for the caller to judge."""
    if not url.startswith("https://"):
        raise ValueError(f"{url[:120]!r}: only https GETs are made")
    headers = {"User-Agent": UA, "Accept": accept}
    r = TRANSPORT(url, headers, max_bytes)
    LOG.append((url, r.status, len(r.body)))
    if r.status in (403, 429):
        h = r.headers or {}
        until = None
        if str(h.get("x-ratelimit-remaining", "")).strip() == "0" and \
                h.get("x-ratelimit-reset"):
            try:
                until = float(h["x-ratelimit-reset"])
            except ValueError:
                until = None
        if until is None and h.get("retry-after"):
            try:
                until = time.time() + float(h["retry-after"])
            except ValueError:
                until = None
        if until is not None or r.status == 429:
            # A 429 with neither header: the server said "too many" without
            # a time; the stage looks again at the worker's next tick.
            import idle
            raise RateLimited(url, until if until is not None
                              else idle.next_tick(),
                              f"{urllib.parse.urlsplit(url).hostname} "
                              f"answered HTTP {r.status} (rate limited"
                              + (f"; the server's reset is "
                                 f"{time.strftime('%H:%M:%S', time.localtime(until))}"
                                 if until else "") + ")")
    return r


def get_json(url: str):
    r = get(url)
    if r.status != 200:
        return None, r
    try:
        return r.json(), r
    except ValueError:
        return None, r


# ---------------------------------------------------------------------------
# The documented endpoints.
# ---------------------------------------------------------------------------
def npm_packument(name: str):
    return get_json(f"{REGISTRY}/{name.replace('/', '%2F')}")


def pypi_project(name: str, version: str | None = None):
    tail = f"{name}/{version}/json" if version else f"{name}/json"
    return get_json(f"{PYPI}/{tail}")


def gh(path: str):
    return get_json(f"{GITHUB_API}/{path.lstrip('/')}")


def raw_url(owner: str, repo: str, ref: str, path: str) -> str:
    return (f"{GITHUB_RAW}/{owner}/{repo}/{ref}/"
            + urllib.parse.quote(path.lstrip("/")))


def raw(owner: str, repo: str, ref: str, path: str, *,
        max_bytes: int | None = None) -> Response:
    return get(raw_url(owner, repo, ref, path), accept="*/*",
               max_bytes=max_bytes)
