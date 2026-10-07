#!/usr/bin/env python
"""The network every model-driven container runs on: internet through an
allow-public-only gate, the Windows host not at all (except, for a harness
container, ONE fixed forward to the proxy or a recording relay). Runner
configuration only -- no harness is ever patched, no Docker or Windows
setting changes.

    python bench/sandbox/sandbox_net.py verify      # the live check (Docker, no model)
    python bench/sandbox/sandbox_net.py plan        # what a run would create

Shared since 2026-09-26 (SELF-IMPROVEMENT-LOG #48/#49). WHO USES IT:
  - bench/octopus: run.py (the Hermes terminal container, control and every
    arm; the browser sidecar) and grade.py (the app container and its
    Playwright browser). bench/octopus/sandbox_net.py is an alias of this
    module, so `import sandbox_net` there is this module.
  - bench/swebench: wsl_side.py brings a network up per instance; mini-swe-
    agent's model-command container joins it (dockerfix.py / the overlay).
  - bench/sandbox/harness_box.py: OpenCode, Pi and Codex in a container,
    the gate forwarding exactly one port to the host's proxy or relay.

THE HOLE (SELF-IMPROVEMENT-LOG #48, checked 2026-09-26 with harmless GETs
and TCP connects from a throwaway container on the pinned image). On Docker
Desktop -- WSL2 engine, `NetworkType: gvisor` in
%APPDATA%\\Docker\\settings-store.json -- a container on the default bridge
reaches the host through host.docker.internal = 192.168.65.254, and a
connection there is made on the host's side to the host's LOOPBACK: every
127.0.0.1-bound service answered (llama-swap :11434 /health, llama-server
:10001 /health, SearXNG :8888 /healthz, Caddy's admin API :2019, the proxy
:1234, the tools API :1235). The host's LAN, ZeroTier and WSL vEthernet
addresses reach its 0.0.0.0 services (:1234, :1235, SMB :445, Caddy :80).
That is Docker Desktop's documented design (host.docker.internal is "the
host"), not a setting that is on by mistake: .wslconfig sets no mirrored
networking, and no key in settings-store.json turns the host route off.
`--add-host host.docker.internal:...` only hides the NAME: 192.168.65.254
answered directly.

THE FIX. Per run unit (a prompt, a grade, an instance, a harness run), a
Docker network created `--internal` (no gateway: checked, "Network is
unreachable" to 192.168.65.254, the LAN, 172.17.0.1 and 8.8.8.8; no
external DNS) and one GATE container (egress_gate.py on the pinned node
image, no download) on the default bridge AND that network, alias `egress`.
The model's containers join only the internal network and get
HTTP(S)_PROXY=http://egress:3128: the gate resolves each name itself and
connects only to a global address on 80/443. npm, pip, git and curl honour
the variables; anything that ignores them has no route at all (fails
closed). The host reaches in only through ports the gate publishes on
127.0.0.1 for a fixed target (Octopus: the browser sidecar's CDP, the host
browser's game port), and a container reaches the host only through a
forward the runner fixed (harness_box.py: egress:<port> -> the host's proxy
or relay port).
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# run.IMAGE (nikolaik/python-nodejs, pinned 2026-09-24): python3 for the gate.
GATE_IMAGE = ("nikolaik/python-nodejs@sha256:"
              "140156d7165a3d18b919bc8e9e21584c0b6099d7c2161efa05ee98b50f9f5d73")
GATE_MOUNT = "/sandbox"               # this directory, read-only, in the gate
GATE_ALIAS = "egress"
GATE_PORT = 3128                      # egress_gate.PROXY_PORT
PROXY_URL = f"http://{GATE_ALIAS}:{GATE_PORT}"
NO_PROXY = "localhost,127.0.0.1,::1"
TERMINAL_ALIAS = "octo-term"          # the Hermes terminal container on the network
BROWSER_ALIAS = "octo-browser"        # the browser sidecar on the network
DEFAULT_PREFIX = "octo"               # names/labels: <prefix>-net-<tag>, <prefix>-gate-<tag>


def proxy_env(no_proxy_extra: tuple[str, ...] | list[str] = ()) -> dict:
    """What the model's containers get. Both cases: curl reads only lowercase
    http_proxy, npm and pip read either. `no_proxy_extra`: hosts reached
    directly on the internal network (harness_box: `egress`, whose fixed
    forward is the one way to the host)."""
    env = {}
    for k in ("HTTP_PROXY", "HTTPS_PROXY"):
        env[k] = env[k.lower()] = PROXY_URL
    env["NO_PROXY"] = env["no_proxy"] = ",".join([NO_PROXY, *no_proxy_extra])
    return env


def env_args(no_proxy_extra: tuple[str, ...] | list[str] = ()) -> list[str]:
    out = []
    for k, v in proxy_env(no_proxy_extra).items():
        out += ["-e", f"{k}={v}"]
    return out


def _safe(tag: str) -> str:
    if tag.startswith("<"):                 # the --dry-run placeholder, printed as is
        return tag
    return re.sub(r"[^A-Za-z0-9_.-]", "-", tag)[:80]


def names(tag: str, prefix: str = DEFAULT_PREFIX) -> dict:
    t = _safe(tag)
    return {"network": f"{prefix}-net-{t}", "gate": f"{prefix}-gate-{t}"}


def network_args(tag: str, alias: str | None = None, prefix: str = DEFAULT_PREFIX) -> list[str]:
    """`docker run` args that put a container on the run's internal network."""
    args = ["--network", names(tag, prefix)["network"]]
    if alias:
        args += ["--network-alias", alias]
    return args


def gate_argv(tag: str, forwards: list[tuple[int, str, int, int | None]] = (),
              prefix: str = DEFAULT_PREFIX) -> list[str]:
    """`docker run` for the gate. `forwards`: (listen, target host, target
    port, host loopback port or None). A host port publishes the forward on
    the host's 127.0.0.1 (the host reaching IN); None keeps it inside the
    internal network (a container reaching OUT to a fixed target, e.g.
    harness_box's egress:1234 -> host.docker.internal:1234)."""
    n = names(tag, prefix)
    argv = ["run", "-d", "--rm", "--init", "--name", n["gate"],
            "--label", f"{prefix}-gate=1", "--label", f"{prefix}-tag={_safe(tag)}",
            "--cpus", "1", "--memory", "256m", "--network", "bridge"]
    for listen, _host, _port, host_port in forwards:
        if host_port is not None:
            argv += ["-p", f"127.0.0.1:{host_port}:{listen}"]
    argv += ["-v", f"{host_path(HERE)}:{GATE_MOUNT}:ro", "--entrypoint", "python3", GATE_IMAGE,
             f"{GATE_MOUNT}/egress_gate.py"]
    for listen, host, port, _hp in forwards:
        argv += ["--forward", f"{listen}:{host}:{port}"]
    return argv


def _remote_engine() -> bool:
    """True when the `docker` this process runs talks to an engine OUTSIDE
    Windows (Docker Engine in WSL Ubuntu since 2026-10-06, docs/DOCKER-WSL.md:
    a `tcp://` DOCKER_HOST or docker context). Such an engine reads a bind
    mount's source as one of ITS paths, and Docker Desktop's own translation
    of `C:\\...` is gone. YAMADORI_DOCKER_PATHS=wsl|native decides it
    outright; an offline suite (YAMADORI_OFFLINE_GUARD=1) is never remote, so
    its argv checks do not depend on this machine's docker context."""
    forced = os.environ.get("YAMADORI_DOCKER_PATHS", "").lower()
    if forced in ("wsl", "native"):
        return forced == "wsl"
    if os.environ.get("YAMADORI_OFFLINE_GUARD") == "1":
        return False
    host = os.environ.get("DOCKER_HOST", "")
    if not host:
        try:
            d = os.environ.get("DOCKER_CONFIG") or os.path.join(os.path.expanduser("~"), ".docker")
            with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                ctx = json.load(f).get("currentContext") or ""
            ctx = os.environ.get("DOCKER_CONTEXT") or ctx
            if ctx and ctx != "default":
                meta = os.path.join(d, "contexts", "meta")
                for sub in os.listdir(meta):
                    with open(os.path.join(meta, sub, "meta.json"), encoding="utf-8") as f:
                        m = json.load(f)
                    if m.get("Name") == ctx:
                        host = (m.get("Endpoints") or {}).get("docker", {}).get("Host", "")
                        break
        except (OSError, ValueError, AttributeError):
            host = ""
    return host.startswith(("tcp://", "ssh://"))


def host_path(p: str) -> str:
    """A Windows path as the engine reads it: `C:\\a\\b` -> `/mnt/c/a/b` for a
    WSL engine (WSL mounts the drive there), unchanged for Docker Desktop
    (which translates it itself) and for a path that is not a drive path."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if not m or not _remote_engine():
        return p
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")


HOST_NAME = "host.docker.internal"        # Docker Desktop's name for the Windows host
_HOST_IP: dict = {}


def host_target() -> str:
    """The Windows host as a container on the engine's default bridge (the
    gate) reaches it. Docker Desktop: `host.docker.internal`. A WSL engine has
    no such name (docs/DOCKER-WSL.md): the host is the Windows side of the WSL
    NAT, the Ubuntu VM's default gateway (the `vEthernet (WSL)` adapter,
    172.29.32.1 on 2026-10-07; it can change when WSL restarts, so it is read
    from the VM each run, cached for the process). `YAMADORI_DOCKER_HOST_IP`
    decides it outright. Measured 2026-10-07 from a bridge container: that
    address answers the host's 0.0.0.0:1234 (200) and a listener bound to it,
    and not a 127.0.0.1-bound service (:11434, 000). Binding a relay to it
    needs no firewall or networking-mode change."""
    forced = os.environ.get("YAMADORI_DOCKER_HOST_IP", "").strip()
    if forced:
        return forced
    if not _remote_engine():
        return HOST_NAME
    if "ip" not in _HOST_IP:
        ip = None
        try:
            r = subprocess.run(["wsl", "-d", "Ubuntu", "-u", "root", "--exec", "ip", "-4",
                                "route", "show", "default"], capture_output=True, text=True,
                               timeout=30)
            m = re.search(r"\bvia\s+(\d+\.\d+\.\d+\.\d+)", r.stdout or "")
            ip = m.group(1) if m else None
        except (OSError, subprocess.TimeoutExpired):
            ip = None
        if not ip:
            raise RuntimeError("the Windows host's address as the WSL engine's containers reach it "
                               "could not be read (wsl -d Ubuntu ip route); set "
                               "YAMADORI_DOCKER_HOST_IP")
        _HOST_IP["ip"] = ip
    return _HOST_IP["ip"]


def _docker(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


# tag -> prefix, for every network this process brought up and has not taken down
_LIVE: dict[str, str] = {}


@atexit.register
def _down_leftovers() -> None:
    for tag, prefix in list(_LIVE.items()):
        down(tag, prefix=prefix)


def up(tag: str, forwards: list[tuple[int, str, int, int | None]] = (), wait_s: int = 30,
       prefix: str = DEFAULT_PREFIX) -> dict:
    """Create the internal network and start the gate on it. `ok` False: the
    caller must not start a model container (fail closed)."""
    n = names(tag, prefix)
    rec: dict = {**n, "proxy": PROXY_URL, "forwards": [list(f) for f in forwards],
                 "t_start": time.time(), "ok": False}
    _docker(["rm", "-f", n["gate"]], 60)
    _docker(["network", "rm", n["network"]], 60)
    r = _docker(["network", "create", "--internal", "--label", f"{prefix}-net=1",
                 "--label", f"{prefix}-tag={_safe(tag)}", n["network"]])
    if r.returncode != 0:
        rec["error"] = "network create: " + (r.stderr or r.stdout)[-400:]
        return rec
    _LIVE[tag] = prefix
    rec["gate_argv"] = gate_argv(tag, forwards, prefix)
    r = _docker(rec["gate_argv"])
    if r.returncode != 0:
        rec["error"] = "gate run: " + (r.stderr or r.stdout)[-400:]
        return rec
    rec["gate_id"] = r.stdout.strip()[:12]
    r = _docker(["network", "connect", "--alias", GATE_ALIAS, n["network"], n["gate"]])
    if r.returncode != 0:
        rec["error"] = "gate connect: " + (r.stderr or r.stdout)[-400:]
        return rec
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if "gate ready" in _docker(["logs", n["gate"]], 30).stdout:
            rec.update(ok=True, ready_s=round(time.time() - t0, 1))
            return rec
        time.sleep(0.5)
    rec["error"] = f"gate not ready within {wait_s}s"
    return rec


def down(tag: str, prefix: str = DEFAULT_PREFIX) -> dict:
    """Remove the gate and the network. Any container still attached (a
    harness that left one behind: mini-swe-agent removes its container on a
    daemon thread that dies with the process) is force-removed first -- it
    belongs to this run unit and has no other network. The gate's decisions
    are counted into the record: what the model tried."""
    n = names(tag, prefix)
    _LIVE.pop(tag, None)
    logs = _docker(["logs", n["gate"]], 60)
    text = (logs.stdout or "") + (logs.stderr or "")
    deny = [ln.split(" ", 1)[1] for ln in text.splitlines() if " DENY " in f" {ln}"]
    left = _docker(["ps", "-aq", "--filter", f"network={n['network']}"], 60)
    ids = [i for i in (left.stdout or "").split() if i]
    gate_id = _docker(["ps", "-aq", "--filter", f"name=^{n['gate']}$"], 60).stdout.split()
    stray = [i for i in ids if i not in gate_id]
    if stray:
        _docker(["rm", "-f", *stray], 120)
    rm = _docker(["rm", "-f", n["gate"]], 60)
    net = _docker(["network", "rm", n["network"]], 60)
    for _ in range(3):
        if net.returncode == 0:
            break
        time.sleep(1)
        net = _docker(["network", "rm", n["network"]], 60)
    return {"t_stop": time.time(), "gate_removed": rm.returncode == 0,
            "network_removed": net.returncode == 0,
            "network_rm_error": None if net.returncode == 0 else (net.stderr or "")[-300:],
            "leftovers_removed": len(stray),
            "allowed": len(re.findall(r"^\S+ ALLOW ", text, re.M)),
            "forwarded": len(re.findall(r"^\S+ FORWARD ", text, re.M)),
            # host:port only (never a path): where the run's traffic went
            "allowed_hosts": sorted(set(re.findall(r"^\S+ ALLOW \S+ (\S+) ->", text, re.M))),
            "failed": len(re.findall(r"^\S+ FAIL ", text, re.M)),
            "denied": len(deny), "denied_sample": deny[:20]}


def plan(tag: str = "<run_id>-p<n>", prefix: str = DEFAULT_PREFIX,
         alias: str = TERMINAL_ALIAS,
         forwards: list[tuple[int, str, int, int | None]] = (),
         no_proxy_extra: tuple[str, ...] = ()) -> str:
    n = names(tag, prefix)
    return "\n".join([
        f"sandbox network (bench/sandbox/sandbox_net.py): docker network create --internal {n['network']}",
        "  gate: docker " + " ".join(gate_argv(tag, forwards, prefix)),
        f"  then: docker network connect --alias {GATE_ALIAS} {n['network']} {n['gate']}",
        f"  model containers: {' '.join(network_args(tag, alias, prefix))} "
        f"{' '.join(env_args(no_proxy_extra))}",
        "  egress: HTTP(S) through the gate to global addresses on 80/443 only; "
        "the host (host.docker.internal, 192.168.65.254, LAN/ZeroTier/WSL IPs) is unreachable"
        + ("" if not any(f[3] is None for f in forwards) else
           "; except the fixed forward(s) "
           + ", ".join(f"{GATE_ALIAS}:{f[0]} -> {f[1]}:{f[2]}" for f in forwards if f[3] is None))])


# ------------------------------------------------------------------ verify

# The probe runs INSIDE a model container (any image with python3). It uses
# only the standard library for the network checks, so it needs no curl; the
# tool checks (npm, pip, git) run only when asked and report `absent` when the
# image lacks the tool. Its one argument is a JSON spec.
PROBE = r"""
import json, os, shutil, socket, subprocess, sys, urllib.request
spec = json.loads(sys.argv[1])
out = {"env": {k: os.environ.get(k) for k in ("HTTPS_PROXY", "NO_PROXY")}}
direct = {}
for ip in spec["direct_ips"]:
    for port in spec["direct_ports"]:
        s = socket.socket(); s.settimeout(3)
        try:
            s.connect((ip, port)); direct[f"{ip}:{port}"] = "OPEN"
        except OSError as e:
            direct[f"{ip}:{port}"] = type(e).__name__ + " " + str(e)[:40]
        finally:
            s.close()
out["direct"] = direct
try:
    out["dns_host_docker_internal"] = socket.gethostbyname("host.docker.internal")
except OSError as e:
    out["dns_host_docker_internal"] = "no: " + type(e).__name__
ph, pp = spec["proxy"]
def via_proxy(t, tunnel):
    # straight to the gate, NO_PROXY irrelevant: the gate itself must refuse
    line = (f"CONNECT {t} HTTP/1.1\r\nHost: {t}\r\n\r\n" if tunnel
            else f"GET http://{t}/health HTTP/1.1\r\nHost: {t}\r\n\r\n")
    try:
        s = socket.create_connection((ph, pp), timeout=10); s.settimeout(15)
        s.sendall(line.encode()); buf = b""
        while b"\r\n" not in buf:
            d = s.recv(4096)
            if not d: break
            buf += d
        s.close()
        first = buf.split(b"\r\n", 1)[0].decode("latin-1")
        return first.split(" ", 1)[1][:80] if " " in first else ("no reply " + first[:40])
    except OSError as e:
        return "error " + type(e).__name__ + " " + str(e)[:60]
out["via_proxy_http"] = {t: via_proxy(t, False) for t in spec["deny"]}
out["via_proxy_connect"] = {t: via_proxy(t, True) for t in spec["deny"]}
def get(url, proxy=True, timeout=20):
    op = urllib.request.build_opener() if proxy else urllib.request.build_opener(
        urllib.request.ProxyHandler({}))
    try:
        with op.open(url, timeout=timeout) as r:
            return f"{r.status} {len(r.read(200000))}B"
    except urllib.error.HTTPError as e:
        return f"{e.code} http error"
    except Exception as e:
        return "error " + type(e).__name__ + " " + str(e)[:80]
out["registry_https"] = get("https://registry.npmjs.org/is-number")
out["pypi_https"] = get("https://pypi.org/simple/six/")
out["allow_http"] = {u: get(u, proxy=False, timeout=10) for u in spec.get("allow_http", [])}
refused = {}
for t in spec.get("refuse_direct", []):
    h, p = t.rsplit(":", 1)
    s = socket.socket(); s.settimeout(5)
    try:
        s.connect((h, int(p))); refused[t] = "OPEN"
    except OSError as e:
        refused[t] = type(e).__name__ + " " + str(e)[:40]
    finally:
        s.close()
out["refuse_direct"] = refused
def sh(cmd, t=300):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
    err = p.stderr.strip()
    return {"rc": p.returncode, "out": p.stdout.strip()[-300:]
            + ("\n[stderr] " + err[-200:] if err else "")}
tools = spec.get("tools", [])
if "npm" in tools:
    out["npm_install"] = ({"rc": None, "out": "absent"} if not shutil.which("npm") else
        sh("mkdir -p /tmp/np && cd /tmp/np && npm init -y >/dev/null && npm install --no-audit "
           "--no-fund is-number@7.0.0 2>&1 | tail -3 && node -e \"console.log(require('is-number')(5))\""))
if "pip" in tools:
    py = sys.executable
    out["pip_install"] = sh(f"{py} -m pip install --no-deps --no-cache-dir --disable-pip-version-check "
                            f"--target /tmp/pp six==1.16.0 2>&1 | tail -2 && "
                            f"PYTHONPATH=/tmp/pp {py} -c 'import six; print(\"six\", six.__version__)'")
if "git" in tools:
    out["git_https"] = ({"rc": None, "out": "absent"} if not shutil.which("git") else
        sh("rm -rf /tmp/gg && git clone -q --depth 1 https://github.com/octocat/Hello-World.git /tmp/gg "
           "2>&1 | tail -2 && git -C /tmp/gg log -1 --format=%H"))
print(json.dumps(out))
"""
_PROBE = PROBE     # the old name


DIRECT_PORTS = (1234, 1235, 8888, 11434, 10001, 2019, 445, 80, 443)


def host_addresses() -> list[str]:
    """The Windows host's own IPv4 addresses (LAN, ZeroTier, WSL/Hyper-V),
    plus Docker Desktop's host and bridge-gateway addresses."""
    out = {"192.168.65.254", "172.17.0.1"}
    try:
        out.update(a for a in socket.gethostbyname_ex(socket.gethostname())[2]
                   if not a.startswith(("127.", "169.254.")))
    except OSError:
        pass
    return sorted(out)


def probe_spec(tools=("npm",), allow_http=(), refuse_direct=(), extra_ports=()) -> dict:
    # :80/:443 are ALLOWED ports, so those rows exercise the address rule
    # (Caddy listens on the host's :80/:443); the rest the port rule
    deny = (["host.docker.internal:11434", "host.docker.internal:80",
             "host.docker.internal:443", "host.docker.internal:1234",
             "gateway.docker.internal:80", "localhost:80", "127.0.0.1:11434",
             f"{GATE_ALIAS}:{GATE_PORT}"]
            + [f"{a}:{p}" for a in host_addresses() for p in (1234, 80)])
    return {"direct_ips": host_addresses(),
            "direct_ports": sorted(set(DIRECT_PORTS) | set(extra_ports)),
            "deny": deny, "proxy": [GATE_ALIAS, GATE_PORT], "tools": list(tools),
            "allow_http": list(allow_http), "refuse_direct": list(refuse_direct)}


def docker_runner(tag: str, image: str, prefix: str, no_proxy_extra=(),
                  alias: str = TERMINAL_ALIAS):
    """The default way to run the probe: a throwaway container with exactly
    the args a model container gets."""
    def run(spec: dict) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "probe.py"), "w", encoding="utf-8") as f:
                f.write(PROBE)
            r = _docker(["run", "--rm", *network_args(tag, alias, prefix),
                         *env_args(no_proxy_extra), "-v", f"{host_path(d)}:/p:ro", "--entrypoint", "python3",
                         image, "/p/probe.py", json.dumps(spec)], 600)
        return r.returncode, r.stdout + ("\n" + r.stderr if r.returncode else "")
    return run


def judge(p: dict, tools=("npm",), allow_http=(), refuse_direct=()) -> dict:
    v = {
        "host_unreachable_direct": bool(p.get("direct")) and not any(
            val == "OPEN" for val in p["direct"].values()),
        # the gate's 403 is the status for plain HTTP and for CONNECT
        "host_refused_via_gate": bool(p.get("via_proxy_http")) and all(
            val.startswith("403") for val in list(p["via_proxy_http"].values())
            + list(p.get("via_proxy_connect", {}).values())),
        "registry_reachable": str(p.get("registry_https", "")).startswith("200"),
    }
    if "npm" in tools:
        v["npm_install_works"] = ((p.get("npm_install") or {}).get("rc") == 0
                                  and "true" in (p.get("npm_install") or {}).get("out", ""))
    if "pip" in tools:
        v["pypi_reachable"] = str(p.get("pypi_https", "")).startswith("200")
        v["pip_install_works"] = ((p.get("pip_install") or {}).get("rc") == 0
                                  and "six 1.16.0" in (p.get("pip_install") or {}).get("out", ""))
    if "git" in tools:
        g = p.get("git_https") or {}
        v["git_https_works"] = g.get("rc") == 0 and bool(re.search(r"\b[0-9a-f]{40}\b",
                                                                   g.get("out", "")))
    if allow_http:
        v["allowed_target_answers"] = all(
            str((p.get("allow_http") or {}).get(u, "")).startswith("200") for u in allow_http)
    if refuse_direct:
        v["other_gate_ports_closed"] = bool(p.get("refuse_direct")) and all(
            val != "OPEN" for val in p["refuse_direct"].values())
    v["ok"] = all(v.values())
    return v


def verify(tag: str = "verify", image: str = GATE_IMAGE, tools=("npm", "pip", "git"),
           prefix: str = DEFAULT_PREFIX, forwards=(), allow_http=(), refuse_direct=(),
           no_proxy_extra=(), extra_ports=(), runner=None) -> dict:
    """Bring a network up, run the probe in a container on it (the same args
    a model container gets, or `runner(spec)` -> (rc, stdout) for a caller
    whose containers are started another way, e.g. mini-swe-agent's), take
    it down. Harmless: TCP connects, GET /health and small public installs;
    nothing is loaded."""
    net = up(tag, forwards, prefix=prefix)
    res: dict = {"up": net}
    spec = probe_spec(tools, allow_http, refuse_direct, extra_ports)
    try:
        if not net["ok"]:
            raise RuntimeError(net.get("error"))
        run = runner or docker_runner(tag, image, prefix, no_proxy_extra)
        rc, text = run(spec)
        res["rc"] = rc
        try:
            res["probe"] = json.loads(text.strip().splitlines()[-1])
        except (ValueError, IndexError):
            res["probe_raw"] = text[-1500:]
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        res["error"] = str(e)
    finally:
        res["down"] = down(tag, prefix=prefix)
    res["verdict"] = judge(res.get("probe") or {}, tools, allow_http, refuse_direct)
    return res


def report(res: dict) -> None:
    p = res.get("probe") or {}
    opened = [k for k, v in (p.get("direct") or {}).items() if v == "OPEN"]
    print(f"direct TCP to the host: {len(p.get('direct') or {})} address:port pairs, "
          f"{len(opened)} open {opened}")
    print(f"host.docker.internal resolves inside: {p.get('dns_host_docker_internal')}")
    for mode in ("via_proxy_http", "via_proxy_connect"):
        for t, v in (p.get(mode) or {}).items():
            print(f"{mode:18s} {t:30s} {v}")
    print(f"https://registry.npmjs.org: {p.get('registry_https')}")
    print(f"https://pypi.org/simple:    {p.get('pypi_https')}")
    for k in ("npm_install", "pip_install", "git_https"):
        if k in p:
            print(f"{k}: {p[k]}")
    for u, v in (p.get("allow_http") or {}).items():
        print(f"allowed forward {u}: {v}")
    for t, v in (p.get("refuse_direct") or {}).items():
        print(f"gate port {t}: {v}")
    if "probe_raw" in res:
        print("probe output: " + res["probe_raw"])
    d = res.get("down", {})
    print(f"gate: {d.get('allowed')} allowed, {d.get('denied')} denied, "
          f"{d.get('forwarded')} forwarded; "
          f"{res.get('up', {}).get('error') or res.get('error') or ''}")
    print("verdict: " + json.dumps(res["verdict"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("verify", "plan"))
    ap.add_argument("--image", default=GATE_IMAGE, help="verify: the model container's image")
    ap.add_argument("--tools", default="npm,pip,git", help="verify: which installs to try")
    ap.add_argument("--json", action="store_true", help="verify: the whole record")
    a = ap.parse_args()
    if a.what == "plan":
        print(plan())
        return 0
    res = verify(image=a.image, tools=tuple(t for t in a.tools.split(",") if t))
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        report(res)
    return 0 if res["verdict"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
