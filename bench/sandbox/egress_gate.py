#!/usr/bin/env python3
"""The egress gate for the model sandboxes (bench/sandbox/sandbox_net.py:
Octopus, SWE-bench, harness_box). Runs INSIDE a container on the pinned node
image, attached to TWO networks: Docker's default bridge (the only way out)
and the run's `--internal` network (no route anywhere), where it answers as
`egress`. The model's containers live on the internal network only, so this
process is the one path out of them.

    python3 /sandbox/egress_gate.py [--forward LISTEN:HOST:PORT ...]

WHY (SELF-IMPROVEMENT-LOG #48, checked 2026-09-26): on Docker Desktop
(WSL2 engine, gvisor networking) a container on the default bridge reaches
the Windows host through host.docker.internal = 192.168.65.254 -- including
services bound to the host's 127.0.0.1 (llama-swap :11434, llama-server
:10001, SearXNG :8888, Caddy's admin API :2019) -- and the host's LAN,
ZeroTier and WSL addresses reach its 0.0.0.0 services (:1234, :1235, SMB).
Blocking the NAME does nothing: the address answers directly.

WHAT IT DOES
  - an HTTP proxy on :3128 (the containers get HTTP(S)_PROXY): CONNECT
    host:port for HTTPS, absolute-URI requests for plain HTTP. The name is
    resolved HERE and the connection goes to an address that passed the
    check, never to a name, so a rebinding answer cannot slip through.
    ALLOWED: an address whose `ipaddress.is_global` is true (no loopback,
    private, link-local, CGNAT, Docker Desktop's 192.168.65.0/24, multicast,
    reserved), on a port in ALLOWED_PORTS. Everything else: 403, logged.
  - fixed TCP forwards (`--forward 9223:octo-browser:9323`): how the host
    reaches the browser sidecar's CDP (the sidecar is on the internal network,
    which Docker does not publish), and the game port for the host browser.
    Outward, harness_box.py's ONE forward to the host (`--forward
    1234:host.docker.internal:1234`): the harness's proxy or recording relay,
    the only non-global target any sandbox reaches (#49).
    The target is fixed by the runner; nothing a client sends chooses it.
  - one log line per decision (`ALLOW` / `DENY` host port -> address, why),
    which sandbox_net.down() counts into the run row.
"""
from __future__ import annotations

import argparse
import ipaddress
import socket
import sys
import threading
import time

PROXY_PORT = 3128
# A CHOICE (2026-09-26): npm, pip, git-over-https and CDNs need 443 and 80;
# nothing in the Octopus tasks needs another port.
ALLOWED_PORTS = (80, 443)
MAX_HEADER = 64 * 1024
CONNECT_TIMEOUT = 15

_lock = threading.Lock()


def log(msg: str) -> None:
    with _lock:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def allowed(ip: str) -> bool:
    """The destination rule: a global unicast address, nothing else."""
    try:
        a = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_global and not a.is_multicast


def resolve(host: str, port: int) -> tuple[str | None, str]:
    """(address to connect to, why). Only an allowed address is ever returned."""
    if port not in ALLOWED_PORTS:
        return None, f"port {port} not in {ALLOWED_PORTS}"
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        return None, f"resolve failed: {e}"
    addrs = [i[4][0] for i in infos]
    ok = [a for a in addrs if allowed(a)]
    if not ok:
        return None, "not a global address: " + ",".join(sorted(set(addrs)))[:120]
    # IPv4 first: the bridge has no IPv6 route.
    ok.sort(key=lambda a: ":" in a)
    return ok[0], "global"


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


def splice(a: socket.socket, b: socket.socket) -> None:
    t = threading.Thread(target=_pipe, args=(b, a), daemon=True)
    t.start()
    _pipe(a, b)
    t.join()
    a.close()
    b.close()


def read_head(c: socket.socket) -> tuple[bytes, bytes] | None:
    buf = b""
    while b"\r\n\r\n" not in buf:
        if len(buf) > MAX_HEADER:
            return None
        chunk = c.recv(65536)
        if not chunk:
            return None
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def reply(c: socket.socket, code: int, why: str) -> None:
    body = f"egress gate: {why}\n".encode()
    reason = {403: "Forbidden", 400: "Bad Request", 502: "Bad Gateway"}.get(code, "Error")
    try:
        c.sendall(f"HTTP/1.1 {code} {reason}\r\nContent-Type: text/plain\r\n"
                  f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    except OSError:
        pass
    c.close()


def split_hostport(s: str, default: int) -> tuple[str, int]:
    if s.startswith("["):                           # [v6]:port
        host, _, rest = s[1:].partition("]")
        return host, int(rest[1:]) if rest.startswith(":") else default
    host, sep, port = s.rpartition(":")
    if sep and port.isdigit():
        return host, int(port)
    return s, default


def handle(c: socket.socket, peer) -> None:
    try:
        c.settimeout(60)
        got = read_head(c)
        if not got:
            return reply(c, 400, "no request head")
        head, rest = got
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) != 3:
            return reply(c, 400, "bad request line")
        method, target, version = parts
        if method.upper() == "CONNECT":
            host, port = split_hostport(target, 443)
        else:
            if not target.lower().startswith("http://"):
                return reply(c, 400, "only CONNECT and absolute http:// URIs")
            hostport, _, path = target[7:].partition("/")
            host, port = split_hostport(hostport, 80)
            target = "/" + path
        ip, why = resolve(host, port)
        if ip is None:
            log(f"DENY {method} {host}:{port} from {peer[0]}: {why}")
            return reply(c, 403, f"{host}:{port} refused ({why})")
        try:
            up = socket.create_connection((ip, port), timeout=CONNECT_TIMEOUT)
        except OSError as e:
            log(f"FAIL {method} {host}:{port} -> {ip}: {e}")
            return reply(c, 502, f"connect to {host}:{port} failed")
        up.settimeout(None)
        c.settimeout(None)
        log(f"ALLOW {method} {host}:{port} -> {ip}")
        if method.upper() == "CONNECT":
            c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            if rest:
                up.sendall(rest)
        else:
            keep = [ln for ln in lines[1:] if ln and not ln.lower().startswith(
                ("proxy-", "connection:", "keep-alive:"))]
            up.sendall(("\r\n".join([f"{method} {target} {version}", *keep,
                                     "Connection: close"]) + "\r\n\r\n").encode("latin-1") + rest)
        splice(c, up)
    except (OSError, ValueError) as e:
        log(f"ERROR from {peer[0]}: {e}")
        try:
            c.close()
        except OSError:
            pass


def serve(port: int, handler) -> None:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(128)
    while True:
        c, peer = srv.accept()
        threading.Thread(target=handler, args=(c, peer), daemon=True).start()


def forwarder(host: str, port: int):
    def handler(c: socket.socket, peer) -> None:
        try:
            up = socket.create_connection((host, port), timeout=10)
        except OSError as e:
            log(f"FORWARD-FAIL {host}:{port} from {peer[0]}: {e}")
            c.close()
            return
        log(f"FORWARD {host}:{port} from {peer[0]}")
        up.settimeout(None)
        splice(c, up)
    return handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="append", default=[], metavar="LISTEN:HOST:PORT")
    a = ap.parse_args()
    for spec in a.forward:
        listen, host, port = spec.split(":")
        threading.Thread(target=serve, args=(int(listen), forwarder(host, int(port))),
                         daemon=True).start()
        log(f"forward :{listen} -> {host}:{port}")
    threading.Thread(target=serve, args=(PROXY_PORT, handle), daemon=True).start()
    time.sleep(0.5)
    log(f"gate ready: proxy :{PROXY_PORT}, ports {ALLOWED_PORTS}, global addresses only")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
