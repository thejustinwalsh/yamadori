#!/usr/bin/env python
"""A request's work stops when its client goes away.

THE FAILURE THIS EXISTS FOR (Octopus v0b-V0-xhigh-1, 2026-09-25,
docs/SELF-IMPROVEMENT-LOG.md #39). A struggle trigger ran deep thinking
before main. Hermes' stream went quiet for 900 s -- the second brain
streamed nothing while it generated -- so Hermes' stale-stream detector
killed the request and retried. The first request's second-brain job ran on:
it was a daemon thread in shomen.run holding the one helper lane, so the
retry found "helper lane busy after 20s; investigate skipped", and main's
retry decoded at 1.2-3.4 tok/s beside the orphan until it died too.

Nothing stopped the work because nothing COULD: every generation is a
blocking read on an upstream socket (proxy._post_events_raw for main,
model.post for the second brain), in a thread the request no longer owns.
So a request carries a Token, and cancelling it SHUTS DOWN every upstream
socket opened on its behalf. The blocked read returns at once, the call
raises, shomen.run's `with helper_lane` releases the lane, and llama-server
sees the connection close and cancels the task: server_response_reader
polls `should_stop` (the connection's is_connection_closed) every
HTTP_POLLING_SECONDS = 1 and `stop()` cancels the remaining tasks
(llamacpp-sudoingx-bonsai2 tools/server/server-context.cpp:38, 4224, 4256;
server-queue.cpp:548-560, 602). llama-swap in between is a Go reverse proxy,
which cancels its upstream request when its client's goes (inferred, not read
from its source).

HOW A SOCKET IS FOUND. A token is BOUND to a thread (`bound`): the thread
that resumes the request's generator (proxy.stream_body binds it around every
next()), and each worker thread the request starts (deep thinking, the
_in_thread jobs) binds the token it was started under. http.client's
HTTPConnection.connect is wrapped ONCE, here: a connection opened on a thread
with a bound token registers its socket with that token. Nothing else
changes -- urllib.request.urlopen is still what every caller uses (the test
suites replace it with stubs, and those keep working), and on a thread with
no token the wrapper does nothing at all. A warm (proxy._warm) runs on its
own thread with no token, so a finished turn's warm is never cut.
"""
from __future__ import annotations

import contextlib
import http.client
import socket
import threading


class Cancelled(RuntimeError):
    """The request this work was for was cancelled (its client went away)."""


class Token:
    """One request's cancellation: an event, and the sockets to shut."""

    def __init__(self) -> None:
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self._socks: list = []
        self.why = ""

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def cancel(self, why: str = "cancelled") -> None:
        with self._lock:
            if self._ev.is_set():
                return
            self.why = why
            self._ev.set()
            socks, self._socks = list(self._socks), []
        for s in socks:
            _shut(s)

    def register(self, sock) -> None:
        with self._lock:
            if not self._ev.is_set():
                # Only live sockets are kept: one closed after its request
                # ended is dropped the next time one registers.
                self._socks = [s for s in self._socks if s.fileno() != -1]
                self._socks.append(sock)
                return
        _shut(sock)         # opened after the cancel: shut at once

    def check(self) -> None:
        if self._ev.is_set():
            raise Cancelled(self.why or "cancelled")

    def wait(self, timeout: float | None = None) -> bool:
        return self._ev.wait(timeout)


def _shut(sock) -> None:
    """shutdown(), never close(): safe from another thread, and the blocked
    reader returns (or raises) instead of waiting out its timeout. A socket
    already closed raises EBADF, which is fine: its request is over."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


_local = threading.local()


def current() -> Token | None:
    return getattr(_local, "token", None)


def cancelled() -> bool:
    t = current()
    return bool(t and t.cancelled)


def check() -> None:
    t = current()
    if t is not None:
        t.check()


@contextlib.contextmanager
def bound(token: Token | None):
    """Bind `token` to this thread for the block (None: leave it unbound)."""
    prev = getattr(_local, "token", None)
    _local.token = token
    try:
        yield token
    finally:
        _local.token = prev


# The unwrapped connect, even if this module is loaded twice.
_orig_connect = getattr(http.client.HTTPConnection.connect, "_yamadori_orig",
                        http.client.HTTPConnection.connect)


def _connect(self):
    _orig_connect(self)
    t = getattr(_local, "token", None)
    # Plain HTTP only (the upstreams are loopback HTTP): HTTPSConnection
    # wraps the socket after this and detaches the one registered here.
    if t is not None and self.sock is not None:
        t.register(self.sock)


_connect._yamadori_orig = _orig_connect
http.client.HTTPConnection.connect = _connect
