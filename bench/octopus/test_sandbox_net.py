#!/usr/bin/env python
"""The Octopus sandbox network (sandbox_net.py, egress_gate.py, and how run.py
and grade.py use them; SELF-IMPROVEMENT-LOG #47), offline: the gate's
destination rule, the proxy protocol over a socket pair, the docker argv, the
profile line every arm gets, the lifecycle (docker faked) and fail-closed
paths. No GPU, no network, no docker, no Hermes. The live check is
`python bench/octopus/sandbox_net.py verify` (Docker, no model).

    python bench/octopus/test_sandbox_net.py      -> "N/M checks passed"
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import egress_gate as eg  # noqa: E402
import sandbox_net as sn  # noqa: E402

_results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def test_destination_rule() -> None:
    # Every address the 2026-09-26 probe reached the host through, and its kin.
    host = ["192.168.65.254", "192.168.65.1", "172.17.0.1", "10.0.0.44", "10.242.120.152",
            "172.29.32.1", "192.168.96.1", "127.0.0.1", "0.0.0.0", "169.254.1.1",
            "100.64.0.1", "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1",
            "::ffff:192.168.65.254", "224.0.0.1", "255.255.255.255", "not-an-ip"]
    bad = [a for a in host if eg.allowed(a)]
    check(not bad, "no loopback, private, link-local, CGNAT, Docker Desktop, mapped or "
                   "multicast address is allowed", json.dumps(bad))
    good = ["104.16.0.1", "8.8.8.8", "2606:4700::1111", "::ffff:8.8.8.8"]
    check(all(eg.allowed(a) for a in good), "global unicast addresses are allowed")
    real = socket.getaddrinfo
    table = {"registry.npmjs.org": ["104.16.0.1"],
             "host.docker.internal": ["192.168.65.254"],
             "rebind.example": ["127.0.0.1"],
             "mixed.example": ["10.0.0.44", "104.16.0.2", "2606:4700::1"]}

    def fake(host, port, *a, **k):
        if host not in table:
            raise socket.gaierror("no such host")
        return [(0, 0, 0, "", (ip, port)) for ip in table[host]]
    socket.getaddrinfo = fake
    try:
        check(eg.resolve("registry.npmjs.org", 443) == ("104.16.0.1", "global"),
              "a public name on 443 resolves to its global address")
        ip, why = eg.resolve("host.docker.internal", 443)
        check(ip is None and "192.168.65.254" in why, "host.docker.internal is refused", why)
        ip, why = eg.resolve("rebind.example", 80)
        check(ip is None and "not a global" in why, "a public name that answers loopback is refused")
        ip, _ = eg.resolve("mixed.example", 443)
        check(ip == "104.16.0.2", "a mixed answer connects to a global IPv4 address only", str(ip))
        ip, why = eg.resolve("registry.npmjs.org", 11434)
        check(ip is None and "port" in why, "a port outside 80/443 is refused", why)
        ip, why = eg.resolve("nowhere.example", 443)
        check(ip is None and "resolve failed" in why, "an unresolvable name is refused", why)
    finally:
        socket.getaddrinfo = real
    check(eg.split_hostport("[2606:4700::1]:443", 80) == ("2606:4700::1", 443)
          and eg.split_hostport("a.b:8080", 80) == ("a.b", 8080)
          and eg.split_hostport("a.b", 80) == ("a.b", 80), "host:port parsing (v4, v6, default)")


def _ask(request: bytes) -> bytes:
    """One request through eg.handle over a socket pair."""
    a, b = socket.socketpair()
    t = threading.Thread(target=eg.handle, args=(b, ("172.18.0.3", 1)), daemon=True)
    t.start()
    a.sendall(request)
    a.settimeout(5)
    out = b""
    try:
        while True:
            d = a.recv(4096)
            if not d:
                break
            out += d
    except OSError:
        pass
    a.close()
    t.join(5)
    return out


def test_proxy_protocol() -> None:
    real_resolve, real_log = eg.resolve, eg.log
    logged: list[str] = []
    eg.log = logged.append
    eg.resolve = lambda h, p: (None, "not a global address: 192.168.65.254")
    try:
        r = _ask(b"CONNECT host.docker.internal:11434 HTTP/1.1\r\nHost: x\r\n\r\n")
        check(r.startswith(b"HTTP/1.1 403") and b"refused" in r,
              "CONNECT to the host gets 403 and nothing is connected", r[:80].decode())
        r = _ask(b"GET http://192.168.65.254:1234/health HTTP/1.1\r\nHost: x\r\n\r\n")
        check(r.startswith(b"HTTP/1.1 403"), "a plain-HTTP request to the host gets 403")
        check(any(ln.startswith("DENY CONNECT host.docker.internal:11434") for ln in logged)
              and any(ln.startswith("DENY GET 192.168.65.254:1234") for ln in logged),
              "each refusal is logged with method, target and why", json.dumps(logged))
        r = _ask(b"GET /health HTTP/1.1\r\nHost: 192.168.65.254\r\n\r\n")
        check(r.startswith(b"HTTP/1.1 400"),
              "an origin-form request (a client that is not using it as a proxy) gets 400")
        r = _ask(b"garbage\r\n\r\n")
        check(r.startswith(b"HTTP/1.1 400"), "a malformed request line gets 400")
    finally:
        eg.resolve, eg.log = real_resolve, real_log
    # ALLOW path: a local upstream stands in for the internet; the rule is
    # patched to say it is global, and the request is rewritten to origin-form.
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    seen: list[bytes] = []

    def upstream():
        c, _ = srv.accept()
        data = b""
        while b"\r\n\r\n" not in data:
            data += c.recv(4096)
        seen.append(data)
        c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        c.close()
    threading.Thread(target=upstream, daemon=True).start()
    eg.resolve, eg.log = (lambda h, p: ("127.0.0.1", "global")), (lambda m: None)
    try:
        r = _ask(f"GET http://registry.example:{port}/is-number HTTP/1.1\r\nHost: registry.example\r\n"
                 f"Proxy-Authorization: x\r\nConnection: keep-alive\r\n\r\n".encode())
    finally:
        eg.resolve, eg.log = real_resolve, real_log
        srv.close()
    check(r.startswith(b"HTTP/1.1 200") and r.endswith(b"ok"), "an allowed request is relayed",
          r[:60].decode())
    up = seen[0].decode() if seen else ""
    check(up.startswith("GET /is-number HTTP/1.1") and "Proxy-Authorization" not in up
          and "Connection: close" in up and "keep-alive" not in up,
          "upstream gets origin-form, no Proxy-* header, Connection: close", up)


def test_argv_and_env() -> None:
    tag = "run/1 p2"
    n = sn.names(tag)
    check(n == {"network": "octo-net-run-1-p2", "gate": "octo-gate-run-1-p2"},
          "names are docker-safe and per tag", json.dumps(n))
    g = sn.gate_argv(tag)
    check(g[g.index("--network") + 1] == "bridge" and "-p" not in g
          and "/sandbox/egress_gate.py" in g and sn.GATE_IMAGE in g and "--rm" in g,
          "the gate: default bridge, pinned image, publishes nothing unless asked", json.dumps(g))
    g = sn.gate_argv(tag, [(9223, "octo-browser", 9323, 9222)])
    pubs = [g[i + 1] for i, a in enumerate(g) if a == "-p"]
    check(pubs == ["127.0.0.1:9222:9223"], "a forward is published on host loopback only",
          json.dumps(pubs))
    env = sn.proxy_env()
    check(env["HTTPS_PROXY"] == env["https_proxy"] == env["HTTP_PROXY"] == env["http_proxy"]
          == f"http://egress:{eg.PROXY_PORT}" and "localhost" in env["NO_PROXY"]
          and env["no_proxy"] == env["NO_PROXY"],
          "proxy env in both cases (curl reads lowercase), loopback bypassed", json.dumps(env))
    check(sn.GATE_PORT == eg.PROXY_PORT and sn.GATE_ALIAS == "egress", "one port, one alias")
    import run as runmod
    check(sn.GATE_IMAGE == runmod.IMAGE, "the gate runs on run.py's pinned image (no download)")


PROFILE = "\n".join([
    "terminal:",
    '  backend: "docker"',
    '  docker_shared_container_key: ""   # OCTO-RUN-KEY (run.py rewrites per run)',
    "  docker_volumes: []   # OCTO-RUN-VOLUMES (run.py rewrites per run)",
    "",
    "browser:",
    "  inactivity_timeout: 120",
    "agent:",
    "  disabled_toolsets:",
    "    - browser",
    "skills:",
    "  creation_nudge_interval: 0",
    "platform_toolsets:",
    "  cli: [file, skills, terminal, vision]",
    "",
])


def test_profile_line() -> None:
    import yaml
    import run as runmod
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.yaml")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(PROFILE)
        rows = {}
        for arm, host in (("default", "sandbox"), ("browser", "sandbox"), ("browser", "host"),
                          ("default", "sandbox")):
            xa = runmod.set_profile_run("t-V0-xhigh-1", r"C:\r", arm, host,
                                        "octo-browser-t-p1", "t-V0-xhigh-1-p1", path=p)
            text = open(p, encoding="utf-8").read()
            cfg = yaml.safe_load(text)
            rows[f"{arm}/{host}"] = (xa, cfg["terminal"].get("docker_extra_args"),
                                     text.count("docker_extra_args"))
        first = open(p, encoding="utf-8").read()
    ok = all(xa == written and count == 1 for xa, written, count in rows.values())
    check(ok, "every arm: exactly one docker_extra_args line, equal to what run.py passes in env",
          json.dumps(rows))
    check(rows["default/sandbox"][0][:2] == ["--network", "octo-net-t-V0-xhigh-1-p1"],
          "control: the terminal is on the run's internal network")
    check(rows["browser/sandbox"][0][:2] == ["--network", "container:octo-browser-t-p1"],
          "sandbox browser: the sidecar's namespace")
    check("OCTO-RUN-NET" in first and "OCTO-ARM" not in first,
          "back to control: the network line stays, no arm line is left", first)
    # make_profile writes the line fail-closed: no network until run.py sets it
    src = open(os.path.join(HERE, "make_profile.py"), encoding="utf-8").read()
    check('docker_extra_args: ["--network", "none"]' in src and "OCTO-RUN-NET" in src,
          "make_profile.py writes --network none (a profile used outside run.py has no network)")


class FakeDocker:
    def __init__(self, fail: str | None = None, logs: str = "gate ready"):
        self.calls: list[list[str]] = []
        self.fail, self.logs = fail, logs

    def __call__(self, args, timeout=120):
        self.calls.append(list(args))
        verb = " ".join(args[:2])
        rc = 1 if self.fail and verb.startswith(self.fail) else 0
        out = self.logs if args[0] == "logs" else ("abcdef0123456789" if args[0] == "run" else "")
        return subprocess.CompletedProcess(["docker", *args], rc, out, "boom" if rc else "")


def test_lifecycle() -> None:
    real = sn._docker
    try:
        fd = FakeDocker()
        sn._docker = fd
        rec = sn.up("t-p1", wait_s=1)
        verbs = [" ".join(c[:2]) for c in fd.calls]
        create = next(c for c in fd.calls if c[:2] == ["network", "create"])
        conn = next(c for c in fd.calls if c[:2] == ["network", "connect"])
        check(rec["ok"] and "--internal" in create and conn[2:4] == ["--alias", "egress"]
              and verbs.index("network create") < verbs.index("run -d")
              < verbs.index("network connect"),
              "up: internal network, then the gate, then the gate joins it as `egress`",
              json.dumps(verbs))
        check("t-p1" in sn._LIVE, "a network that is up is tracked for atexit cleanup")
        fd.logs = ("10:00:00 gate ready\n10:00:01 ALLOW CONNECT registry.npmjs.org:443 -> 104.16.0.1\n"
                   "10:00:02 DENY CONNECT host.docker.internal:11434 from 172.18.0.3: port\n")
        st = sn.down("t-p1")
        check(st["allowed"] == 1 and st["denied"] == 1
              and st["denied_sample"][0].startswith("DENY CONNECT host.docker.internal")
              and ["network", "rm", "octo-net-t-p1"] in fd.calls and "t-p1" not in sn._LIVE,
              "down: counts what the model tried, removes gate and network", json.dumps(st))
        for fail, why in (("network create", "network create"), ("run -d", "gate run"),
                          ("network connect", "gate connect")):
            sn._docker = FakeDocker(fail=fail)
            rec = sn.up("t-p2", wait_s=1)
            check(rec["ok"] is False and rec["error"].startswith(why),
                  f"up fails closed when `{fail}` fails", json.dumps(rec)[:200])
        sn._docker = FakeDocker(logs="")
        rec = sn.up("t-p3", wait_s=0)
        check(rec["ok"] is False and "not ready" in rec["error"], "a gate that never says ready fails")
        sn._LIVE.clear()
    finally:
        sn._docker = real


def test_runner_wiring() -> None:
    """run.py and grade.py: no gate, no model container (read from source)."""
    run_src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read()
    i_up = run_src.index("sandbox_net.up(net_tag")
    i_notrun = run_src.index('"why": "sandbox_net"')
    i_hermes = run_src.index("p = subprocess.Popen(cmd")
    check(i_up < i_notrun < i_hermes, "run.py: the gate is up before Hermes, or the prompt is not_run")
    check('"TERMINAL_DOCKER_EXTRA_ARGS": json.dumps(extra_args)' in run_src
          and run_src.index("sandbox_net.down(net_tag)", i_hermes)
          > run_src.index("toolset_arms.stop_sidecar(sidecar)", i_hermes),
          "run.py: the same args in env; the network goes last, after the sidecar")
    g = open(os.path.join(HERE, "grade.py"), encoding="utf-8").read()
    i_up = g.index("net = sandbox_net.up(name)")
    i_run = g.index('"run", "-d", "--name", name')
    check(i_up < g.index('if not net["ok"]:') < i_run
          and "*sandbox_net.network_args(name, \"octo-app\"), *sandbox_net.env_args()" in g,
          "grade.py: the app container (npm install, the game) is on the sandbox network or not run")
    check('"--proxy", sandbox_net.PROXY_URL' in g and "sandbox_net.down(app[\"net_tag\"])" in g,
          "grade.py: the browser gets the gate as proxy; the network is removed after the app")
    bc = open(os.path.join(HERE, "browser_check.py"), encoding="utf-8").read()
    check('f"--proxy-server={a.proxy}"' in bc and "proxy=" not in bc.split("p.chromium.launch")[1][:300],
          "browser_check: a Chromium flag (keeps the loopback bypass), not Playwright's proxy option")


def main() -> int:
    for fn in (test_destination_rule, test_proxy_protocol, test_argv_and_env, test_profile_line,
               test_lifecycle, test_runner_wiring):
        print(f"\n--- {fn.__name__} ---")
        n0 = len(_results)
        try:
            fn()
        except Exception:                                        # noqa: BLE001
            check(False, f"{fn.__name__} itself raised",
                  traceback.format_exc().strip().split("\n")[-1])
            traceback.print_exc()
        for ok, name, detail in _results[n0:]:
            print(("  pass  " if ok else "  FAIL  ") + name
                  + (f"   <- {detail}" if not ok and detail else ""))
    passed = sum(1 for ok, _, _ in _results if ok)
    print(f"\n{'=' * 70}\n  {passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
