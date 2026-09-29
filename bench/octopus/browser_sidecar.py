#!/usr/bin/env python3
"""The sandboxed browser for the Octopus `--tools browser` arm. Runs INSIDE
the grader's Playwright image (octo-playwright:1.63.0, playwright.Dockerfile;
no other image), started by run.py before Hermes and removed after it.

    python3 /octo/browser_sidecar.py            (the container's command)

THE TOPOLOGY. This container owns a network namespace; Hermes' terminal
container joins it (`--network container:<this>`, toolset_arms.py), so
`localhost` is the same place for the model's shell, its game server and this
browser. The host publishes ONE port of it, 127.0.0.1:<CDP_HOST_PORT> -> this
forwarder, for Hermes' agent-browser client (browser.cdp_url).

  - chrome-headless-shell (Playwright's build 1243 = Chromium 153) with
    --remote-debugging-port=CHROME_PORT. It binds 127.0.0.1 inside the netns
    whatever --remote-debugging-address says (seen 2026-09-26: "DevTools
    listening on ws://127.0.0.1:..."), so a published port cannot reach it
    directly; FORWARD_PORT relays 0.0.0.0 -> 127.0.0.1:CHROME_PORT.
  - --proxy-server to a closed loopback port: Chromium sends loopback
    requests direct (its implicit bypass) and everything else to a proxy that
    refuses, so the page can reach the sandbox's own loopback -- the game --
    and nothing else: not host.docker.internal (which reaches even the host's
    127.0.0.1-bound services from a Docker Desktop container), not the bridge,
    not the internet. WebRTC's non-proxied UDP is disabled for the same reason.
  - software GL (ANGLE SwiftShader), like the grader: the container has no
    GPU anyway.
  - Chrome is restarted if it exits (an agent-browser `close` over CDP may
    close the browser): this process, not Chrome, holds the namespace the
    terminal container lives in, so a closed browser never takes the model's
    shell offline. Each start is printed (`chrome start n=<k>`) -- run.py
    records the count.
"""
from __future__ import annotations

import glob
import os
import socket
import subprocess
import sys
import threading
import time

CHROME_PORT = 9322        # Chrome's own DevTools port (netns loopback only)
FORWARD_PORT = 9323       # what the host's published port reaches
DEAD_PROXY = "http://127.0.0.1:9"   # discard port: nothing listens
# The Octopus sidecar keeps DEAD_PROXY (the page reaches its loopback only).
# The harness box (bench/sandbox/harness_box.py) sets SIDECAR_PROXY to the
# run's gate, http://egress:3128: the page then also reaches what the gate
# allows (global addresses on 80/443), never the host. Loopback stays direct
# either way (Chromium's implicit bypass).
PROXY = os.environ.get("SIDECAR_PROXY") or DEAD_PROXY

FLAGS = [
    "--no-sandbox", "--disable-dev-shm-usage",
    f"--remote-debugging-port={CHROME_PORT}",
    "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
    f"--proxy-server={PROXY}",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--no-first-run", "--no-default-browser-check",
    "--window-size=1280,720",
    "about:blank",
]


def chrome_binary() -> str:
    hits = sorted(glob.glob("/ms-playwright/chromium_headless_shell-*/"
                            "chrome-headless-shell-linux64/chrome-headless-shell"))
    if not hits:
        sys.exit("no chrome-headless-shell under /ms-playwright")
    return hits[-1]


def _pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def forward() -> None:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", FORWARD_PORT))
    srv.listen(64)
    while True:
        client, _ = srv.accept()
        try:
            upstream = socket.create_connection(("127.0.0.1", CHROME_PORT), timeout=10)
            upstream.settimeout(None)
        except OSError:
            client.close()
            continue
        for x, y in ((client, upstream), (upstream, client)):
            threading.Thread(target=_pipe, args=(x, y), daemon=True).start()


# ------------------------------------------------------------------ error mirror
# WHY (found 2026-09-26, Docker, no model; docs/HARNESSES.md): Hermes ee5ee84's
# browser_console reads each uncaught exception's `message` from agent-browser
# (tools/browser_tool.py browser_console), and agent-browser 0.26.0 -- the only
# release in Hermes' pin ^0.26.0 -- names that field `text`: the model gets
# `js_errors: [{"message": ""}]`, an exception with no words (#47's "PLAYER.hit
# is not a function" would arrive empty). Neither is ours to patch. The
# sidecar's own CDP client therefore adds, to every page before its scripts
# run, a listener that repeats each uncaught error and unhandled rejection as
# a console.error line ("Uncaught ReferenceError: ..."), which Hermes does
# pass through. SIDECAR_ERROR_MIRROR=1 turns it on (toolset_arms.sidecar_argv
# sets it; the harness box does not: Playwright MCP reports page errors).

MIRROR_JS = r"""(() => {
  if (window.__yamadoriErrorMirror) return; window.__yamadoriErrorMirror = true;
  const ce = console.error.bind(console);
  const text = (x) => { try { return (x && x.stack) ? String(x.stack) : String(x); } catch (e) { return '?'; } };
  window.addEventListener('error', (e) => { if (!(e instanceof ErrorEvent)) return;
    ce('Uncaught ' + (e.error ? text(e.error) : e.message)); });
  window.addEventListener('unhandledrejection', (e) => ce('Unhandled promise rejection: ' + text(e.reason)));
})();"""


class _WS:
    """A minimal RFC 6455 client (text frames, client masking) over stdlib
    sockets: the Playwright image has no websocket package to rely on."""

    def __init__(self, host: str, port: int, path: str):
        import base64
        self.s = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                        "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.s.recv(4096)
            if not chunk:
                raise OSError("websocket handshake: closed")
            head += chunk
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise OSError("websocket handshake: " + head.split(b"\r\n", 1)[0].decode(errors="replace"))
        self.buf = head.split(b"\r\n\r\n", 1)[1]
        self.s.settimeout(None)
        self.lock = threading.Lock()

    def send(self, text: str) -> None:
        data = text.encode()
        n = len(data)
        hdr = bytearray([0x81])
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + n.to_bytes(2, "big")
        else:
            hdr += bytes([0x80 | 127]) + n.to_bytes(8, "big")
        mask = os.urandom(4)
        with self.lock:
            self.s.sendall(bytes(hdr) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _read(self, k: int) -> bytes:
        while len(self.buf) < k:
            chunk = self.s.recv(65536)
            if not chunk:
                raise OSError("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:k], self.buf[k:]
        return out

    def recv(self) -> str:
        msg = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = int.from_bytes(self._read(2), "big")
            elif n == 127:
                n = int.from_bytes(self._read(8), "big")
            payload = self._read(n)
            op = b0 & 0x0F
            if op == 0x8:
                raise OSError("websocket closed by peer")
            if op in (0x9, 0xA):          # ping/pong: Chrome does not ping; ignore
                continue
            msg += payload
            if b0 & 0x80:
                return msg.decode(errors="replace")


def error_mirror() -> None:
    """Attach to Chrome's browser target, auto-attach to every page (paused
    until the listener is in), add MIRROR_JS to each. Reconnects when Chrome
    restarts."""
    import json
    import urllib.request
    while True:
        try:
            v = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CHROME_PORT}/json/version", timeout=2))
            path = "/" + v["webSocketDebuggerUrl"].split("/", 3)[3]
            ws = _WS("127.0.0.1", CHROME_PORT, path)
            n = [0]

            def call(method: str, params: dict | None = None, session: str | None = None) -> None:
                n[0] += 1
                m = {"id": n[0], "method": method, "params": params or {}}
                if session:
                    m["sessionId"] = session
                ws.send(json.dumps(m))

            call("Target.setDiscoverTargets", {"discover": True})
            call("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True,
                                          "flatten": True})
            print("error mirror attached", flush=True)
            while True:
                ev = json.loads(ws.recv())
                meth = ev.get("method")
                if meth == "Target.targetCreated":
                    info = ev["params"]["targetInfo"]
                    if info.get("type") == "page" and not info.get("attached"):
                        call("Target.attachToTarget", {"targetId": info["targetId"], "flatten": True})
                elif meth == "Target.attachedToTarget":
                    sid = ev["params"]["sessionId"]
                    if ev["params"]["targetInfo"].get("type") == "page":
                        call("Page.enable", session=sid)
                        call("Page.addScriptToEvaluateOnNewDocument", {"source": MIRROR_JS}, sid)
                        call("Runtime.evaluate", {"expression": MIRROR_JS}, sid)
                    call("Runtime.runIfWaitingForDebugger", session=sid)
        except (OSError, ValueError, KeyError, IndexError):
            time.sleep(1)


def main() -> int:
    exe = chrome_binary()
    threading.Thread(target=forward, daemon=True).start()
    if os.environ.get("SIDECAR_ERROR_MIRROR") == "1":
        threading.Thread(target=error_mirror, daemon=True).start()
    n = 0
    while True:
        n += 1
        print(f"chrome start n={n} {exe}", flush=True)
        rc = subprocess.run([exe, *FLAGS]).returncode
        print(f"chrome exit n={n} rc={rc}", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    os.environ.setdefault("HOME", "/root")
    sys.exit(main())
