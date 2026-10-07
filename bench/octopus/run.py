#!/usr/bin/env python
"""Run one Octopus Invaders variant through Hermes (Windows install, Docker
terminal backend).

    python bench/octopus/run.py --variant V0 --arm xhigh --rep 1 --tag pilot
    python bench/octopus/run.py --preflight-only
    python bench/octopus/run.py --smoke          # a one-file task, sandbox proof
    python bench/octopus/run.py --task pagoda --arm xhigh --tag pagoda-h1
                                                 # bench/voxel's r3f-stack prompt,
                                                 # single-shot, measured (pagoda.py)
    python bench/octopus/run.py --task pagoda --harness pi --arm xhigh --tag pagoda-p1
                                                 # the same, through Pi in the harness box
    python bench/octopus/run.py --task pagoda [--harness pi] --arm max --tag T --print
                                                 # the exact commands + prompt file; touches nothing

Use the stack interpreter. One run at a time, never beside another GPU
consumer (AGENTS.md "Before you claim anything works", 4).

THE HARNESS (operator, 2026-09-24): the WINDOWS Hermes
(%LOCALAPPDATA%\\hermes\\bin\\hermes.exe) with its own profile,
HERMES_HOME=C:\\Users\\jwals\\octo\\hermes-home (bench/octopus/make_profile.py:
the operator's config with compression on and the docker backend; the
hermes-dogfood key). The operator's own profile is never read or written.

SAFETY: Hermes runs commands the model chooses. Its terminal is the DOCKER
backend (Docker Desktop), with ONLY this run's folder bind-mounted at
/workspace, plus one writable screenshots folder inside Hermes' read-only
cache mount (SCREENSHOT_HOST -> SCREENSHOT_CONTAINER, so vision_analyze can
read a screenshot the model saved; docs/VISION.md route 1); its file tools go
through the same container. If the backend
does not come up, the run fails; there is no fallback to the host and never
--yolo. NETWORK (sandbox_net.py, SELF-IMPROVEMENT-LOG #47): the container is
on a per-prompt `--internal` Docker network whose one way out is a gate
container that proxies HTTP(S) to global addresses on 80/443 only; on the
default bridge it reached every 127.0.0.1-bound service of the Windows host
through host.docker.internal (Docker Desktop's design). No gate, no run. The TERMINAL_* variables below win over config.yaml
(hermes_cli/cli_config_load.py: "Env always wins").

PREFLIGHT refuses to start while a live suite or benchmark is running, unless
every llama-server slot is idle on two reads 5 s apart, or if the proxy or
Docker does not answer.

What a run leaves:

  C:\\Users\\jwals\\octo\\runs\\<run_id>\\   the model's working folder (/workspace)
  C:\\Users\\jwals\\octo\\logs\\<run_id>\\   hermes.jsonl (stream-json), hermes.err,
                                     relay.jsonl (one row per request, with
                                     x_yamadori), session.jsonl (Hermes'
                                     transcript), meta.json, prompt.md
  bench/octopus/results/runs.jsonl   one row per run: ids, versions, exit,
                                     wall time, and the stack-side deltas over
                                     the run window: token ledger rows (every
                                     account), GPU energy from the power
                                     ledger (all cards, idle included, 60 s
                                     flush granularity), corpus ids, proxy
                                     log byte range.

A 429 from the proxy is "not run", never a failure (grade.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import variants  # noqa: E402
import toolset_arms  # noqa: E402  (--tools: which Hermes toolsets, config only)
import sandbox_net  # noqa: E402  (the internal network + egress gate, every arm; #47)

RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(RESULTS, "runs.jsonl")
OCTO = r"C:\Users\jwals\octo"
RUNS_DIR = os.path.join(OCTO, "runs")
LOGS_DIR = os.path.join(OCTO, "logs")
HERMES_HOME = os.path.join(OCTO, "hermes-home")
# WHERE THE MODEL SAVES A SCREENSHOT IT WANTS TO LOOK AT (docs/VISION.md,
# Hermes route 1, checked 2026-09-26 with Hermes' own container code): a
# writable volume nested in Hermes' read-only screenshots cache mount.
# vision_analyze reads a path under it on the host; /workspace and /tmp
# paths fail under the docker backend on Windows (#46). The folder name must
# be the same on both sides (Hermes maps it through its parent mount).
SCREENSHOT_HOST = os.path.join(HERMES_HOME, "cache", "screenshots", "octo")
SCREENSHOT_CONTAINER = "/root/.hermes/cache/screenshots/octo"
HERMES = os.path.join(os.environ.get("LOCALAPPDATA", r"C:\Users\jwals\AppData\Local"),
                      "hermes", "bin", "hermes.exe")
UPSTREAM = "http://127.0.0.1:1234"
RELAY_PORT = 18234
PROVIDER = "octo-relay"          # make_profile.py's named provider -> the relay
DISTRO = "Ubuntu"                # WSL: only used to read nothing now; kept for grade.py
# nikolaik/python-nodejs:python3.12-nodejs22, pulled 2026-09-24 (node 22.23.3,
# python 3.12.14, npm 10.9.9). Pinned so a re-tag cannot change a rerun.
IMAGE = ("nikolaik/python-nodejs@sha256:"
         "140156d7165a3d18b919bc8e9e21584c0b6099d7c2161efa05ee98b50f9f5d73")
SLOTS = "http://127.0.0.1:11434/upstream/bonsai/slots"
PROXY = "http://127.0.0.1:1234/v1/models"
TOKEN_DB = os.path.join(ROOT, "index", "token_ledger.sqlite3")
POWER = os.path.join(ROOT, "index", "power_ledger.json")
CORPUS = os.path.join(ROOT, "index", "corpus.sqlite3")
PROXY_LOG = os.path.join(ROOT, "logs", "proxy.out.log")
KEEPALIVE = "octo-keepalive"

# Command-line fragments of the other GPU consumers this repo runs.
BUSY = ("test_live_stack", "test_tools_live", "run_tests.py --live",
        "queue_runner.py", "bench\\domain\\run.py",
        "bench/domain/run.py", "livecodebench.py", "recipe_oracle.py",
        "octopus\\run.py --variant", "octopus/run.py --variant",
        "octopus\\run.py --task", "octopus/run.py --task",
        "voxel\\run.py", "voxel/run.py")


def wsl_path(p: str) -> str:
    p = os.path.abspath(p)
    return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")


def busy_processes() -> list[str]:
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine } | "
         "ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId) $($_.CommandLine)\" }"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    me = str(os.getpid())
    out = []
    for line in ps.stdout.splitlines():
        pid, _, rest = line.partition(" ")
        ppid, _, cmd = rest.partition(" ")
        if me in (pid, ppid):
            continue
        # Only a Python process is a GPU consumer here; a shell whose command
        # line merely CONTAINS the script name (this runner's own parents) is not.
        exe = cmd.split('" ', 1)[0] if cmd.startswith('"') else cmd.split(" ", 1)[0]
        if "python" not in exe.lower():
            continue
        if any(b in cmd for b in BUSY):
            out.append(f"{pid} {cmd[:200]}")
    return out


def bonsai_ready(swap: str | None = None) -> tuple[bool | None, str]:
    """(ready?, why) from llama-swap's GET /running, which never loads
    anything (mcp/gpu_room.py model_loaded's rule, inline: this runner does
    not import mcp/). None: /running could not be read."""
    swap = swap or SLOTS.split("/upstream/")[0]
    try:
        with urllib.request.urlopen(f"{swap}/running", timeout=10) as r:
            rows = (json.load(r) or {}).get("running") or []
    except Exception as e:                          # noqa: BLE001
        return None, f"llama-swap /running: {type(e).__name__}"
    for x in rows:
        if isinstance(x, dict) and x.get("model") == "bonsai":
            st = str(x.get("state") or "ready")
            return st == "ready", f"bonsai is {st} (llama-swap /running)"
    return False, "bonsai is not loaded (llama-swap /running)"


def slots_idle() -> tuple[bool, list]:
    """Every main-model slot idle. A READ NEVER LOADS A MODEL (2026-09-30):
    /upstream/bonsai/slots only while /running lists bonsai ready; not loaded,
    nothing is generating on it ([("not loaded", why)]); /running unreadable
    is not idle."""
    ready, why = bonsai_ready()
    if not ready:
        return ready is False, [("slots not read", why)]
    with urllib.request.urlopen(SLOTS, timeout=15) as r:
        s = json.load(r)
    state = [(x.get("id"), bool(x.get("is_processing"))) for x in s]
    return not any(p for _i, p in state), state


def proxy_alive() -> str:
    try:
        urllib.request.urlopen(PROXY, timeout=15)
        return "200"
    except urllib.error.HTTPError as e:           # 401 without a key: alive
        return str(e.code)
    except Exception as e:                         # noqa: BLE001
        return f"down: {type(e).__name__}"


def docker_awake() -> str:
    """Docker Desktop's Resource Saver (UseResourceSaver, 300 s) pauses the VM
    when no container runs; seen 2026-09-24, WSL's docker CLI then HUNG while
    `docker desktop status` said paused. A tiny container is kept running for
    the length of the experiment (no setting changed), and the engine is
    asked to do a real job (list containers) with a timeout."""
    def ps_ok() -> bool:
        try:
            p = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=30)
            return p.returncode == 0 and KEEPALIVE in p.stdout.split()
        except subprocess.TimeoutExpired:
            return False
    if ps_ok():
        return "ok (keepalive running)"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", KEEPALIVE, "--cpus", "0.1",
                    "--memory", "32m", IMAGE, "sleep", "infinity"],
                   capture_output=True, text=True, timeout=180)
    time.sleep(3)
    return "ok (keepalive started)" if ps_ok() else "FAIL: docker does not answer"


def preflight(check_hermes: bool = True) -> dict:
    """The stack is free (no other GPU consumer, every slot idle on two reads
    5 s apart), the proxy answers, Docker answers; and, for a Hermes run,
    Hermes and its profile are there."""
    out = {"busy": busy_processes(), "proxy": proxy_alive()}
    a, s1 = slots_idle()
    time.sleep(5)
    b, s2 = slots_idle()
    out["slots"] = [s1, s2]
    out["slots_idle"] = a and b
    out["docker"] = docker_awake()
    if check_hermes:
        out["hermes"] = os.path.isfile(HERMES) and os.path.isfile(
            os.path.join(HERMES_HOME, "config.yaml")) and os.path.isfile(
            os.path.join(HERMES_HOME, ".env"))
    out["ok"] = (not out["busy"] and out["slots_idle"] and out["proxy"] in ("200", "401")
                 and out["docker"].startswith("ok") and out.get("hermes", True))
    return out


def snapshot() -> dict:
    snap = {"t": time.time()}
    try:
        con = sqlite3.connect(f"file:{TOKEN_DB}?mode=ro", uri=True, timeout=10)
        con.row_factory = sqlite3.Row
        snap["tokens"] = [dict(r) for r in con.execute("SELECT * FROM daily")]
        con.close()
    except sqlite3.Error as e:
        snap["tokens_error"] = str(e)
    try:
        snap["power_days"] = json.load(open(POWER, encoding="utf-8")).get("days", {})
    except (OSError, ValueError) as e:
        snap["power_error"] = str(e)
    try:
        con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True, timeout=10)
        snap["corpus_max_id"] = con.execute("SELECT MAX(id) FROM events").fetchone()[0]
        con.close()
    except sqlite3.Error as e:
        snap["corpus_error"] = str(e)
    try:
        snap["proxy_log_bytes"] = os.path.getsize(PROXY_LOG)
    except OSError:
        snap["proxy_log_bytes"] = None
    return snap


KINDS = ("generations", "prompt_processed", "prompt_cached", "prompt_unsplit",
         "completion", "reasoning", "reasoning_reported", "no_usage")


def deltas(a: dict, b: dict) -> dict:
    key = lambda r: (r["day"], r["account"], r["role"])      # noqa: E731
    before = {key(r): r for r in a.get("tokens", [])}
    tok = {}
    for r in b.get("tokens", []):
        k = key(r)
        p = before.get(k, {})
        d = {x: (r.get(x) or 0) - (p.get(x) or 0) for x in KINDS}
        if any(d.values()):
            tok.setdefault(r["account"] or "(stack)", {})[r["role"]] = d
    wh = 0.0
    for day, v in b.get("power_days", {}).items():
        wh += float(v.get("gpu_wh") or 0) - float(
            (a.get("power_days", {}).get(day) or {}).get("gpu_wh") or 0)
    return {"tokens": tok, "gpu_wh": round(wh, 2),
            "corpus_ids": [a.get("corpus_max_id"), b.get("corpus_max_id")],
            "proxy_log_bytes": [a.get("proxy_log_bytes"), b.get("proxy_log_bytes")],
            "note_energy": "power ledger delta: all cards, idle draw included, "
                           "flushed every 60 s (mcp/power.py SAVE_EVERY_SECONDS)"}


def append(row: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


NET_LINE = r"(?m)^  docker_extra_args: .*# OCTO-RUN-NET.*$"


def net_line(extra_args: list[str]) -> str:
    return (f"  docker_extra_args: {json.dumps(extra_args)}"
            f"   # OCTO-RUN-NET (run.py rewrites per run; sandbox_net.py)")


def set_profile_run(run_id: str, run_dir: str, tools: str = toolset_arms.DEFAULT_ARM,
                    browser_host: str = toolset_arms.DEFAULT_BROWSER_HOST,
                    sidecar: str = toolset_arms.SIDECAR_PLACEHOLDER,
                    net_tag: str | None = None, path: str | None = None,
                    skills: bool = True) -> list[str]:
    """Point the dogfood profile's marked terminal lines at this run
    (make_profile.py writes them): the /workspace volume, the container label
    key, and the terminal's docker_extra_args -- the run's sandbox network
    (sandbox_net.py, #47) and how this arm joins it. Each must match exactly
    once; a profile made before the network line gets it after the volumes
    line. Then set the tools arm (toolset_arms.apply): every run states it,
    so a control run after an arm run is back to control. Returns the
    docker_extra_args written (run_prompt passes the same in env)."""
    import re
    path = path or os.path.join(HERMES_HOME, "config.yaml")
    text = open(path, encoding="utf-8").read()
    extra = toolset_arms.docker_extra_args(tools, browser_host, sidecar,
                                           net_tag or f"{run_id}-p1")
    if not re.search(NET_LINE, text):
        vol = r"(?m)^(  docker_volumes: .*# OCTO-RUN-VOLUMES.*)$"
        text, n = re.subn(vol, lambda m: m.group(1) + "\n" + net_line(["--network", "none"]), text)
        if n != 1:
            raise SystemExit(f"profile line not found once ({n}): {vol}")
    subs = [(r"(?m)^  docker_shared_container_key: .*# OCTO-RUN-KEY.*$",
             f'  docker_shared_container_key: "octo-{run_id}"   # OCTO-RUN-KEY (run.py rewrites per run)'),
            (r"(?m)^  docker_volumes: .*# OCTO-RUN-VOLUMES.*$",
             f"  docker_volumes: ['{run_dir}:/workspace', "
             f"'{SCREENSHOT_HOST}:{SCREENSHOT_CONTAINER}', "
             f"'{toolset_arms.volume_spec()}']"
             f"   # OCTO-RUN-VOLUMES (run.py rewrites per run)"),
            (NET_LINE, net_line(extra))]
    for pat, rep in subs:
        text, n = re.subn(pat, lambda _m, r=rep: r, text)
        if n != 1:
            raise SystemExit(f"profile line not found once ({n}): {pat}")
    text = toolset_arms.apply(text, tools, browser_host, sidecar)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    if skills:
        import make_profile                    # the loadout's skills (type-check, ...)
        make_profile.install_skills(os.path.dirname(path))
    return extra


def _session_id(hermes_jsonl: str) -> str | None:
    try:
        for ln in open(hermes_jsonl, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            for k in ("session_id", "sessionId"):
                if isinstance(d, dict) and d.get(k):
                    return str(d[k])
    except OSError:
        pass
    return None


def sfx_of(n: int) -> str:
    """File suffix of prompt n: '' for the first, '.p2' ... for follow-ups."""
    return "" if n == 1 else f".p{n}"


def hermes_env(run_id: str, run_dir: str, extra_args: list[str],
               tools: str = toolset_arms.DEFAULT_ARM,
               browser_host: str = toolset_arms.DEFAULT_BROWSER_HOST) -> dict:
    """The environment added to the Hermes process for one prompt (run_prompt
    and --print use this one function)."""
    return {
        "HERMES_HOME": HERMES_HOME,
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_IMAGE": IMAGE,
        "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE": "true",
        # The run folder is ALSO named explicitly as the /workspace volume. The
        # cwd mount alone is not enough: with container_persistent false Hermes
        # isolates per session and mounts only a workspace the SESSION
        # registered (tools/terminal_tool.py _resolve_task_host_cwd); a
        # --oneshot's tool task registers none, so the smoke run of 2026-09-24
        # wrote to a tmpfs /workspace and the file never reached the host.
        # (Hermes' config bridge also lets config.yaml override these env
        # values, so set_profile_run writes the same into the profile.)
        "TERMINAL_DOCKER_VOLUMES": json.dumps([f"{run_dir}:/workspace"]),
        "TERMINAL_CONTAINER_PERSISTENT": "false",
        "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES": "false",
        "TERMINAL_DOCKER_SHARED_CONTAINER_KEY": f"octo-{run_id}",
        # long enough that the idle reaper never recycles the container while
        # the model is generating (minutes between tool calls)
        "TERMINAL_LIFETIME_SECONDS": "21600",
        # the sandbox network (the profile's OCTO-RUN-NET line says the same;
        # an explicit terminal.* key wins over env in Hermes' bridge)
        "TERMINAL_DOCKER_EXTRA_ARGS": json.dumps(extra_args),
        "TERMINAL_DOCKER_NETWORK": "true",
        "PYTHONIOENCODING": "utf-8",
        **toolset_arms.env(tools, browser_host),
    }


def hermes_cmd(prompt_file: str, effort: str, budget: int, max_turns: int | None = None,
               session_id: str | None = None, toolsets: str | None = None) -> list[str]:
    """The Hermes command line for one prompt (run_prompt and --print)."""
    cmd = [HERMES, "chat", "--query-file", prompt_file,
           "--oneshot", "--provider", PROVIDER, "-m", "yamadori", "--reasoning", effort,
           "--run-budget", str(budget),
           "--ignore-rules", "--source", "tool", "--format", "stream-json"]
    # --max-turns only when asked: Hermes' own default applies otherwise
    # (docs/CONSTANTS-AUDIT.md, 2026-09-27: the 1000 was invented).
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]
    if session_id:
        cmd += ["--resume", session_id]
    if toolsets:
        cmd += ["-t", toolsets]
    return cmd


def relay_listen() -> str:
    """The relay's listen address(es): loopback, and -- on the WSL engine
    (docs/DOCKER-WSL.md), whose containers reach the Windows host only through
    the WSL NAT address -- ALSO that one address (never 0.0.0.0: this box is
    on ZeroTier). Docker Desktop: loopback only, as before."""
    host = _hb().sandbox_net.host_target()
    listen = f"127.0.0.1:{RELAY_PORT}"
    return listen if host == _hb().sandbox_net.HOST_NAME else f"{listen},{host}:{RELAY_PORT}"


def relay_cmd(out_path: str) -> list[str]:
    """The recording relay in front of :1234 (both harnesses)."""
    return [sys.executable, os.path.join(HERE, "relay.py"), "--listen",
            relay_listen(), "--upstream", UPSTREAM, "--out", out_path]


def run_prompt(run_id: str, prompt: str, effort: str, max_turns: int, budget: int,
               meta: dict, n: int = 1, session_id: str | None = None,
               toolsets: str | None = None, tools: str = toolset_arms.DEFAULT_ARM,
               browser_host: str = toolset_arms.DEFAULT_BROWSER_HOST) -> dict:
    """One Hermes invocation: prompt 1 starts the session in a FRESH folder;
    prompt n > 1 resumes `session_id` in the same folder (--resume), its logs
    suffixed `.p<n>` (relay.p2.jsonl, hermes.p2.jsonl, meta.p2.json, ...).
    `tools` is the harness tools arm (toolset_arms.ARMS); for `browser`,
    `browser_host` "sandbox" starts the browser sidecar before Hermes and
    removes it after Hermes' containers (they live in its network namespace)."""
    pf = preflight()
    pr = toolset_arms.prereqs(tools, browser_host)
    # the loadout's pinned type checkers (every arm): made from the harness box
    # image when missing or stale; docker only, no download
    tv = toolset_arms.tools_volume(create=pf["ok"] and pr["ok"])
    pr = {**pr, "tools_volume": tv, "ok": pr["ok"] and tv["ok"]}
    arm = {"tools_arm": tools, "browser_host": browser_host if tools == "browser" else None,
           "tools_prereqs": pr}
    if not pf["ok"] or not pr["ok"]:
        row = {"run_id": run_id, "prompt_no": n, "outcome": "not_run",
               "why": "preflight" if not pf["ok"] else "tools_arm_prereqs",
               **meta, "preflight": pf, **arm, "t": time.time()}
        append(row)
        return row
    run_dir = os.path.join(RUNS_DIR, run_id)
    log_dir = os.path.join(LOGS_DIR, run_id)
    sfx = sfx_of(n)
    if n == 1:
        if os.path.exists(run_dir) or os.path.exists(log_dir):
            raise SystemExit(f"refusing: {run_dir} or {log_dir} exists (fresh folder per run)")
        os.makedirs(run_dir)
        os.makedirs(log_dir)
        # skill_manage cannot be turned off alone; what it saved in an earlier
        # run must not reach this one (toolset_arms.RUN_SKILLS_DIR)
        arm["run_skills_cleared"] = toolset_arms.clear_run_skills()
    elif not (os.path.isdir(run_dir) and session_id):
        raise SystemExit(f"prompt {n} needs the existing run folder and a session id")
    shutil.copy2(prompt, os.path.join(log_dir, f"prompt{sfx}.md"))
    if _port_open(RELAY_PORT):
        raise SystemExit(f"port {RELAY_PORT} is taken: another relay is running")
    relay = subprocess.Popen(
        relay_cmd(os.path.join(log_dir, f"relay{sfx}.jsonl")),
        stdout=open(os.path.join(log_dir, f"relay{sfx}.out"), "w"), stderr=subprocess.STDOUT)
    for _ in range(20):
        if _port_open(RELAY_PORT):
            break
        time.sleep(0.25)
    else:
        relay.kill()
        raise SystemExit("relay did not start")

    sidecar = (toolset_arms.sidecar_name(run_id, n)
               if toolset_arms.uses_sidecar(tools, browser_host) else None)
    # THE SANDBOX NETWORK (sandbox_net.py, SELF-IMPROVEMENT-LOG #47), every
    # arm: the terminal container (and the sidecar) join an --internal network
    # whose only way out is the gate -- global addresses on 80/443 through
    # HTTP(S)_PROXY. The Windows host (host.docker.internal = 192.168.65.254,
    # which reaches its 127.0.0.1 services; its LAN/ZeroTier/WSL addresses)
    # is unreachable. No gate, no run: it fails closed.
    net_tag = f"{run_id}-p{n}"
    extra_args = set_profile_run(run_id, run_dir, tools, browser_host,
                                 sidecar or toolset_arms.SIDECAR_PLACEHOLDER, net_tag)
    arm["sandbox_net"] = sandbox_net.up(net_tag, toolset_arms.gate_forwards(tools, browser_host))
    arm["sandbox_net"]["terminal_extra_args"] = extra_args
    if not arm["sandbox_net"]["ok"]:
        arm["sandbox_net"]["stop"] = sandbox_net.down(net_tag)
        relay.terminate()
        row = {"run_id": run_id, "prompt_no": n, "outcome": "not_run",
               "why": "sandbox_net", **meta, "preflight": pf, **arm, "t": time.time()}
        append(row)
        return row
    if tools == "lean":
        # the Playwright MCP server Hermes spawns (the profile's mcp_servers line)
        arm["mcp"] = {"server": toolset_arms.MCP_SERVER, "tools": list(toolset_arms.MCP_TOOLS),
                      "argv": toolset_arms.mcp_argv(sidecar),
                      "image_id": (pr.get("mcp_image") if isinstance(pr, dict) else None)}
    if sidecar:
        arm["browser_sidecar"] = toolset_arms.start_sidecar(sidecar, run_id, net_tag=net_tag,
                                                            arm=tools)
        if not arm["browser_sidecar"]["ok"]:
            arm["browser_sidecar"]["stop"] = toolset_arms.stop_sidecar(sidecar)
            arm["sandbox_net"]["stop"] = sandbox_net.down(net_tag)
            relay.terminate()
            row = {"run_id": run_id, "prompt_no": n, "outcome": "not_run",
                   "why": "browser_sidecar", **meta, "preflight": pf, **arm, "t": time.time()}
            append(row)
            return row
    env = dict(os.environ)
    env.update(hermes_env(run_id, run_dir, extra_args, tools, browser_host))
    cmd = hermes_cmd(os.path.join(log_dir, f"prompt{sfx}.md"), effort, budget, max_turns,
                     session_id, toolsets)
    a = snapshot()
    if n == 1 and a.get("proxy_log_bytes") is not None:
        # where this run's lines start in logs/proxy.out.log (watch.py reads
        # the warm lines from here on; the proxy prints them with no timestamp)
        with open(os.path.join(log_dir, "proxy_offset"), "w") as f:
            f.write(str(a["proxy_log_bytes"]))
    t0 = time.time()
    killed = False
    with open(os.path.join(log_dir, f"hermes{sfx}.jsonl"), "wb") as out, \
            open(os.path.join(log_dir, f"hermes{sfx}.err"), "wb") as err:
        p = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=out, stderr=err,
                             stdin=subprocess.DEVNULL)
        try:
            rc = p.wait(timeout=budget + 900)
        except subprocess.TimeoutExpired:
            killed = True
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)],
                           capture_output=True)
            rc = p.wait()
    t1 = time.time()
    sid = _session_id(os.path.join(log_dir, f"hermes{sfx}.jsonl")) or session_id
    exp = [HERMES, "sessions", "export", "--format", "jsonl",
           os.path.join(log_dir, f"session{sfx}.jsonl")]
    exp += ["--session-id", sid] if sid else ["--source", "tool", "--newer-than", "1d"]
    e = subprocess.run(exp, env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    relay.terminate()
    try:
        relay.wait(timeout=10)
    except subprocess.TimeoutExpired:
        relay.kill()
    # Containers this run left behind (labelled hermes-profile=octo-<run_id>-<digest>).
    ps = subprocess.run(["docker", "ps", "-a", "--filter", "label=hermes-agent=1",
                         "--format", '{{.ID}} {{.Label "hermes-profile"}}'],
                        capture_output=True, text=True)
    left = [ln.split()[0] for ln in ps.stdout.splitlines()
            if len(ln.split()) > 1 and ln.split()[1].startswith(f"octo-{run_id}-")]
    if left:
        subprocess.run(["docker", "rm", "-f", *left], capture_output=True)
    if sidecar and tools == "lean":    # a Playwright MCP container Hermes left (normally --rm'd)
        arm["mcp"]["cleanup"] = toolset_arms.mcp_cleanup(sidecar)
    if sidecar:                        # after Hermes' containers: they use its namespace
        arm["browser_sidecar"]["stop"] = toolset_arms.stop_sidecar(sidecar)
    # last: the network cannot be removed while a container is on it
    arm["sandbox_net"]["stop"] = sandbox_net.down(net_tag)
    meta_run = {"run_id": run_id, "prompt_no": n, "cmd": cmd, "cwd": run_dir, "exit": rc,
                "killed_by_runner": killed, "t_start": t0, "t_end": t1,
                "session_id": sid, "export_rc": e.returncode,
                "export_out": (e.stdout + e.stderr)[-800:],
                "containers_left": left,
                **arm,
                "terminal_env": {k: v for k, v in env.items() if k.startswith("TERMINAL_")}}
    with open(os.path.join(log_dir, f"meta{sfx}.json"), "w", encoding="utf-8") as f:
        json.dump(meta_run, f, indent=1)
    time.sleep(65)                     # one power-ledger flush after the run
    b = snapshot()
    row = {**meta_run, **meta, "effort": effort, "max_turns": max_turns,
           "run_budget_s": budget, "image": IMAGE, "wall_s": round(t1 - t0, 1),
           "preflight": pf, "stack": deltas(a, b)}
    append(row)
    return row


def run_one(run_id: str, prompt: str, effort: str, max_turns: int, budget: int,
            meta: dict, toolsets: str | None = None, tools: str = toolset_arms.DEFAULT_ARM,
            browser_host: str = toolset_arms.DEFAULT_BROWSER_HOST) -> dict:
    """A single-prompt run (the pilot's protocol)."""
    return run_prompt(run_id, prompt, effort, max_turns, budget, meta, 1, None, toolsets,
                      tools, browser_host)


def run_iterative(run_id: str, variant: str, prompt: str, effort: str, max_turns: int,
                  budget: int, meta: dict, max_prompts: int = 6,
                  tools: str = toolset_arms.DEFAULT_ARM,
                  browser_host: str = toolset_arms.DEFAULT_BROWSER_HOST) -> list[dict]:
    """THE ITERATIVE PROTOCOL (coordinator, 2026-09-24): up to `max_prompts`
    prompts in ONE Hermes session. After each prompt the run is graded
    (grade.py, grade id `<run_id>@p<n>`), and the next prompt is built by
    followup.build() from that grade alone -- build errors, console errors,
    blank canvas, failed spec checks -- so the same grade always yields the
    same follow-up. It stops early when a grade has nothing to report, when a
    prompt is not run (preflight / 429), or when a follow-up is identical to
    the previous one AND the grade did not change (no progress to steer).
    `budget` is PER PROMPT (Hermes' --run-budget is per conversation run)."""
    import followup
    import grade as grademod
    rows = []
    row = run_prompt(run_id, prompt, effort, max_turns, budget, {**meta, "protocol":
                     f"iterative<= {max_prompts}"}, 1, tools=tools,
                     browser_host=browser_host)
    rows.append(row)
    run_dir = os.path.join(RUNS_DIR, run_id)
    log_dir = os.path.join(LOGS_DIR, run_id)
    prev_text = None
    for n in range(2, max_prompts + 2):
        if row.get("outcome") == "not_run":
            break
        g = grademod.grade(f"{run_id}@p{n - 1}", variant, run_dir, log_dir, True,
                           sfx=sfx_of(n - 1))
        if n > max_prompts:
            break
        text = followup.build(variant, g)
        if not text or text == prev_text:
            break
        prev_text = text
        path = os.path.join(log_dir, f"followup.p{n}.md")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        row = run_prompt(run_id, path, effort, max_turns, budget,
                         {**meta, "protocol": f"iterative<= {max_prompts}",
                          "assisted": ASSISTED}, n,
                         session_id=row.get("session_id"), tools=tools,
                         browser_host=browser_host)
        rows.append(row)
    return rows


# Operator decision, 2026-09-26: the iterative protocol's follow-ups
# (followup.py) are built from the grader's failed checks, which is the
# answer key; the grader is a measurement instrument only, and prompt 1
# (single-shot, no feedback) is the headline. Some v1/v2 follow-up items were
# also FALSE (SELF-IMPROVEMENT-LOG #57, #62).
ASSISTED = "assisted: graded follow-up"
GRADED_FOLLOWUPS_WHY = ("the follow-up prompt is built from the grader's failed checks: it "
                        "leaks the answer key, and some of its items were false (grader "
                        "defects, SELF-IMPROVEMENT-LOG #57 / #62)")
GRADED_FOLLOWUPS_REFUSAL = (
    "run.py: --iterative REFUSED. " + GRADED_FOLLOWUPS_WHY + ". Operator decision "
    "2026-09-26: the grader is a measurement instrument only and prompt 1 (single-shot) is "
    "the headline. To run it anyway (every prompt >= 2 is recorded as '" + ASSISTED + "'), "
    "add --allow-graded-followups.")


PAGODA_WIRE_REFUSAL = (
    "run.py --task pagoda: REFUSED. The pagoda runs like V4, on Hermes' Responses wire; "
    "this profile's octo-relay provider speaks {wire}. Make the profile with "
    "`python bench/octopus/make_profile.py --wire responses` first.")


def _hb():
    return toolset_arms._hb()


# Pi's key: the pi-dogfood test key (docs/HARNESS-PI.md), beside the host
# test's other keys. --key-file names another.
PI_KEY_FILE = os.path.join(_hb().SCRATCH, "pi-dogfood.key")


def pi_plan(a) -> dict:
    """Everything a Pi pagoda run uses, computed without touching anything:
    ids, folders, the prompt, the relay, the /v1/models read, the harness box
    command and the docker commands it runs. run_pagoda_pi and --print share it."""
    import pagoda
    hb = _hb()
    rid = pagoda.run_id(a.tag, a.arm, a.rep, "pi")
    run_dir = os.path.join(RUNS_DIR, rid)            # the project: /work, Pi's cwd
    log_dir = os.path.join(LOGS_DIR, rid)
    box_dir = os.path.join(log_dir, "box")           # the run home (Pi's config, sessions) + record
    text, sha = pagoda.prompt()
    models_out = os.path.join(log_dir, "pi-models.json")
    args = pagoda.pi_args(a.arm, text)
    box_cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "sandbox", "harness_box.py"),
               "run", "pi", "--key-file", a.key_file or PI_KEY_FILE, "--project", run_dir,
               "--run-dir", box_dir, "--target-port", str(RELAY_PORT), "--config", models_out,
               "--timeout", str(a.run_budget), "--tag", rid, "--", *args]
    return {"run_id": rid, "run_dir": run_dir, "log_dir": log_dir, "box_dir": box_dir,
            "prompt": text, "prompt_sha256": sha, "prompt_path": pagoda.PROMPT_OVERRIDE or pagoda.PROMPT_PATH,
            "models_src": hb.HARNESSES["pi"]["config_src"], "models_out": models_out,
            "card_url": f"http://127.0.0.1:{RELAY_PORT}/v1/models",
            "relay": relay_cmd(os.path.join(log_dir, "relay.jsonl")),
            "box_cmd": box_cmd, "pi_args": args,
            "net": hb.sandbox_net.plan(rid, hb.PREFIX, "harness", hb.forwards(RELAY_PORT),
                                       (hb.sandbox_net.GATE_ALIAS,)),
            "sidecar": ["docker", *hb.sidecar_argv(rid)],
            "harness": ["docker", *hb.run_argv("pi", rid, run_dir, os.path.join(box_dir, "home"), args)],
            "stdout": os.path.join(log_dir, "pi.jsonl"), "stderr": os.path.join(log_dir, "pi.err")}


def pi_preflight(p: dict, key_file: str) -> dict:
    """The Hermes preflight's stack checks (no other GPU consumer, idle
    slots, the proxy, Docker), then Pi's own: the harness box image built,
    the key file readable and non-empty."""
    out = preflight(check_hermes=False)
    hb = _hb()
    out["box_image"] = hb.image_present(hb.IMAGE)
    try:
        out["key_file"] = bool(open(key_file, encoding="utf-8").read().strip())
    except OSError:
        out["key_file"] = False
    out["ok"] = out["ok"] and out["box_image"] and out["key_file"]
    return out


def _pi_session_id(path: str) -> str | None:
    """Pi's JSON mode opens with the session header {"type":"session","id":...}
    (docs/json.md)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(d, dict) and d.get("type") == "session":
                    return d.get("id")
    except OSError:
        pass
    return None


def run_pagoda_pi(a) -> int:
    """THE PAGODA THROUGH PI (single-shot, like Hermes'): the same prompt, a
    fresh run folder as the project (/work = Pi's current directory, so
    pagoda/ lands in it), the harness box's default loadout, the arm's
    thinking level, every request through the recording relay (x_yamadori
    per row, so pagoda.measure works), then the same measure. Pi's model
    entry takes its window, output ceiling and inputs from the proxy's
    /v1/models, read through the relay at run time (pagoda.pi_models)."""
    import pagoda
    p = pi_plan(a)
    key_file = a.key_file or PI_KEY_FILE
    rid, run_dir, log_dir = p["run_id"], p["run_dir"], p["log_dir"]
    meta = {"task": pagoda.TASK, "variant": pagoda.TASK, "harness": "pi", "arm": a.arm,
            "effort": a.arm, "rep": a.rep, "tag": a.tag, "tools": _hb().DEFAULT_LOADOUT,
            "loadout": _hb().LOADOUT_VERSION, "wire": "chat", "protocol": "single-shot",
            "prompt_sha256": p["prompt_sha256"],
            "prompt_source": "bench/voxel/run.py TASKS['r3f-stack']",
            "versions": {"pi": _hb().VERSIONS["pi"]}}
    pf = pi_preflight(p, key_file)
    if not pf["ok"]:
        row = {"run_id": rid, "prompt_no": 1, "outcome": "not_run", "why": "preflight",
               **meta, "preflight": pf, "t": time.time()}
        append(row)
        print(json.dumps({k: row.get(k) for k in ("run_id", "outcome", "why")}, indent=2))
        return 3
    if os.path.exists(run_dir) or os.path.exists(log_dir):
        raise SystemExit(f"refusing: {run_dir} or {log_dir} exists (fresh folder per run)")
    os.makedirs(run_dir)
    os.makedirs(log_dir)
    path, sha = pagoda.write_prompt()
    shutil.copy2(path, os.path.join(log_dir, "prompt.md"))
    if _port_open(RELAY_PORT):
        raise SystemExit(f"port {RELAY_PORT} is taken: another relay is running")
    relay = subprocess.Popen(p["relay"], stdout=open(os.path.join(log_dir, "relay.out"), "w"),
                             stderr=subprocess.STDOUT)
    for _ in range(20):
        if _port_open(RELAY_PORT):
            break
        time.sleep(0.25)
    else:
        relay.kill()
        raise SystemExit("relay did not start")
    rc, killed, t0, t1 = None, False, None, None
    card_rec: dict = {}
    try:
        key = open(key_file, encoding="utf-8").read().strip()
        try:
            card = pagoda.advertised_card(RELAY_PORT, key)
            src = json.load(open(p["models_src"], encoding="utf-8"))
            cfg, card_rec = pagoda.pi_models(src, card)
            if getattr(a, "features", None):
                # the X-Yamadori-Features provider header (Pi 0.87.1: a provider's `headers`), as
                # bench/mcp/lookup_probe.py sends it: e.g. the model plus the MCP tools only
                json.loads(a.features)
                for prov in (cfg.get("providers") or {}).values():
                    prov.setdefault("headers", {})["X-Yamadori-Features"] = a.features
                card_rec["features"] = a.features
        except (OSError, ValueError, SystemExit) as e:
            row = {"run_id": rid, "prompt_no": 1, "outcome": "not_run", "why": "model_card",
                   "error": str(e)[:400], **meta, "preflight": pf, "t": time.time()}
            append(row)
            print(json.dumps({k: row.get(k) for k in ("run_id", "outcome", "why", "error")}, indent=2))
            return 3
        with open(p["models_out"], "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg, f, indent=2)
        a0 = snapshot()
        t0 = time.time()
        with open(p["stdout"], "wb") as out, open(p["stderr"], "wb") as err:
            proc = subprocess.Popen(p["box_cmd"], stdout=out, stderr=err, stdin=subprocess.DEVNULL)
            try:
                # harness_box's own --timeout removes the container at the run
                # budget; this wait only guards the runner
                rc = proc.wait(timeout=a.run_budget + 900)
            except subprocess.TimeoutExpired:
                killed = True
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
                rc = proc.wait()
        t1 = time.time()
    finally:
        relay.terminate()
        try:
            relay.wait(timeout=10)
        except subprocess.TimeoutExpired:
            relay.kill()
    box_rec = {}
    try:
        lines = open(os.path.join(p["box_dir"], "harness_box.jsonl"), encoding="utf-8").read().split("\n")
        box_rec = json.loads([ln for ln in lines if ln.strip()][-1])
    except (OSError, ValueError, IndexError):
        pass
    meta_run = {"run_id": rid, "prompt_no": 1, "harness": "pi", "cwd": run_dir, "exit": rc,
                "killed_by_runner": killed, "t_start": t0, "t_end": t1,
                "session_id": _pi_session_id(p["stdout"]),
                "box_cmd": p["box_cmd"][:p["box_cmd"].index("--") + 1] + p["pi_args"][:-1]
                + [f"<prompt.md sha256 {sha[:12]}>"],
                "model_card": card_rec,
                "harness_box": {k: box_rec.get(k) for k in ("image", "image_id", "tag", "target", "loadout",
                                                             "rc", "timed_out", "not_run")},
                "browser_sidecar": {k: (box_rec.get("browser_sidecar") or {}).get(k)
                                    for k in ("image", "image_id", "browser", "ok")}}
    with open(os.path.join(log_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_run, f, indent=1)
    time.sleep(65)                     # one power-ledger flush after the run
    b = snapshot()
    row = {**meta_run, **meta, "run_budget_s": a.run_budget,
           "wall_s": round(t1 - t0, 1) if t0 and t1 else None,
           "preflight": pf, "stack": deltas(a0, b)}
    append(row)
    print(json.dumps({k: row.get(k) for k in ("run_id", "harness", "outcome", "wall_s", "exit",
                                               "killed_by_runner", "session_id")}, indent=2), flush=True)
    m = pagoda.measure(rid, run_dir, log_dir, row)
    print(json.dumps(pagoda.brief(m), indent=1)[:6000])
    return 0


def print_pagoda(a) -> int:
    """--print: the exact commands and prompt file a pagoda run of this
    harness would use, and where to watch it. Touches nothing: no file is
    written, no process started, no network or docker call."""
    import pagoda
    text, sha = pagoda.prompt()
    try:
        on_disk = hashlib_sha256(pagoda.PROMPT_PATH)
    except OSError:
        on_disk = None
    status = ("matches" if on_disk == sha else
              "MISSING: the run writes it" if on_disk is None else "DIFFERS: the run rewrites it")
    out = [f"harness: {a.harness}   arm (reasoning / thinking level): {a.arm}   tag: {a.tag}   rep: {a.rep}",
           f"prompt file: {pagoda.PROMPT_PATH}  ({len(text.encode('utf-8'))} bytes, sha256 {sha}; "
           f"on disk: {status})",
           "  = bench/voxel/run.py TASKS['r3f-stack'], byte for byte", ""]
    if a.harness == "pi":
        p = pi_plan(a)
        out += [f"run id: {p['run_id']}",
                f"project (/work, Pi's cwd): {p['run_dir']}",
                f"logs: {p['log_dir']}   (pi.jsonl = Pi's JSON events, pi.err, relay.jsonl = one row "
                "per request with x_yamadori, meta.json, measure.json)",
                "", "1. the recording relay:", "   " + " ".join(p["relay"]),
                "", f"2. the model card, read through the relay at run time: GET {p['card_url']} "
                f"(key from {a.key_file or PI_KEY_FILE}) -> context_length, max_completion_tokens, "
                f"modalities.input written into {p['models_out']} (from {p['models_src']})",
                "", "3. the harness box (this command; it runs the three below):",
                "   " + " ".join(p["box_cmd"][:p["box_cmd"].index("--") + 1]
                                 + p["pi_args"][:-1] + ['"$(cat ' + pagoda.PROMPT_PATH + ')"']),
                "   sandbox network and gate:"] + ["     " + ln for ln in p["net"].split("\n")] + [
                "   browser sidecar: " + " ".join(p["sidecar"]),
                "   Pi: " + " ".join(p["harness"][:-1]) + ' "<the prompt>"',
                "", "4. then: pagoda.measure (serve on :3001, page check, stack check, relay summary)",
                "", "to run it yourself:",
                f"   python bench/octopus/run.py --task pagoda --harness pi --arm {a.arm} --tag {a.tag}"
                + (f" --rep {a.rep}" if a.rep != 1 else "")
                + (f" --key-file {a.key_file}" if a.key_file else ""),
                "to watch it: Get-Content -Wait " + p["stdout"] + "   (and relay.jsonl beside it)"]
    else:
        rid = pagoda.run_id(a.tag, a.arm, a.rep)
        run_dir, log_dir = os.path.join(RUNS_DIR, rid), os.path.join(LOGS_DIR, rid)
        sidecar = (toolset_arms.sidecar_name(rid, 1)
                   if toolset_arms.uses_sidecar(a.tools, a.browser_host) else None)
        extra = toolset_arms.docker_extra_args(a.tools, a.browser_host,
                                               sidecar or toolset_arms.SIDECAR_PLACEHOLDER, f"{rid}-p1")
        env = hermes_env(rid, run_dir, extra, a.tools, a.browser_host)
        cmd = hermes_cmd(os.path.join(log_dir, "prompt.md"), a.arm, a.run_budget, a.max_turns)
        wire = pagoda.profile_wire(os.path.join(HERMES_HOME, "config.yaml"))
        out += [f"run id: {rid}",
                f"project (/workspace): {run_dir}",
                f"logs: {log_dir}   (hermes.jsonl = Hermes' stream-json, hermes.err, relay.jsonl = one "
                "row per request with x_yamadori, session.jsonl, meta.json, measure.json)",
                f"profile: {os.path.join(HERMES_HOME, 'config.yaml')} (wire: {wire}"
                + ("" if wire == "responses" else " -- REFUSED at run time: make_profile.py --wire responses")
                + f"); tools arm: {a.tools} ({toolset_arms.LOADOUT_VERSION})",
                "", "1. the recording relay:", "   " + " ".join(relay_cmd(os.path.join(log_dir, "relay.jsonl"))),
                "", "2. the profile lines this run sets (run.py set_profile_run):",
                "   " + net_line(extra).strip()]
        if a.tools == "lean":
            out += ["   " + toolset_arms.mcp_line(sidecar) + "   # OCTO-ARM:mcp",
                    f"   platform_toolsets.cli: [file, terminal, vision, {toolset_arms.MCP_SERVER}]"]
        out += ["", "3. the sandbox network, gate and sidecar:"]
        out += ["   " + ln for ln in sandbox_net.plan(
            f"{rid}-p1", forwards=toolset_arms.gate_forwards(a.tools, a.browser_host)).split("\n")]
        if sidecar:
            out += ["   sidecar: docker " + " ".join(toolset_arms.sidecar_argv(
                sidecar, rid, f"{rid}-p1", mirror=a.tools == "browser"))]
        if a.tools == "lean":
            out += ["   Playwright MCP (Hermes spawns it): docker " + " ".join(toolset_arms.mcp_argv(sidecar))]
        out += ["", "4. Hermes (cwd = the project; the prompt file is copied to logs\\prompt.md):",
                "   env: " + " ".join(f"{k}={v}" for k, v in env.items()),
                "   " + " ".join(cmd),
                "", "5. then: pagoda.measure (serve on :3001, page check, stack check, relay summary)",
                "", "to run it yourself:",
                f"   python bench/octopus/run.py --task pagoda --arm {a.arm} --tag {a.tag}"
                + (f" --rep {a.rep}" if a.rep != 1 else "")
                + (f" --tools {a.tools}" if a.tools != toolset_arms.DEFAULT_ARM else ""),
                "to watch it: Get-Content -Wait " + os.path.join(log_dir, "hermes.jsonl")
                + "   (and relay.jsonl beside it; bench/octopus/watch.py reads both)"]
    print("\n".join(out))
    return 0


def hashlib_sha256(path: str) -> str:
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_pagoda(a) -> int:
    """THE PAGODA TASK (pagoda.py): one prompt, single-shot, no follow-up;
    then pagoda.measure. Hermes: refused on a chat-wire profile. Both
    harnesses: refused with --iterative (a graded follow-up is never part of
    it). --print shows the commands and touches nothing."""
    import pagoda
    if a.iterative:
        print("run.py --task pagoda: --iterative REFUSED: the pagoda is single-shot, "
              "with no graded follow-up. " + GRADED_FOLLOWUPS_WHY + ".", file=sys.stderr)
        return 2
    if not a.arm:
        print("run.py --task pagoda needs --arm", file=sys.stderr)
        return 2
    if a.print:
        return print_pagoda(a)
    if a.harness == "pi":
        return run_pagoda_pi(a)
    wire = pagoda.profile_wire(os.path.join(HERMES_HOME, "config.yaml"))
    if wire != "responses":
        print(PAGODA_WIRE_REFUSAL.format(wire=wire), file=sys.stderr)
        return 2
    path, sha = pagoda.write_prompt()
    rid = pagoda.run_id(a.tag, a.arm, a.rep)
    meta = {"task": pagoda.TASK, "variant": pagoda.TASK, "harness": "hermes", "arm": a.arm,
            "rep": a.rep, "tag": a.tag, "tools": a.tools, "loadout": toolset_arms.LOADOUT_VERSION,
            "browser_host": a.browser_host if a.tools == "browser" else None,
            "wire": wire, "protocol": "single-shot",
            "prompt_sha256": sha,
            "prompt_source": "bench/voxel/run.py TASKS['r3f-stack']", "versions": {}}
    row = run_one(rid, path, a.arm, a.max_turns, a.run_budget, meta, tools=a.tools,
                  browser_host=a.browser_host)
    print(json.dumps({k: row.get(k) for k in ("run_id", "outcome", "why", "wall_s", "exit",
                                               "killed_by_runner", "session_id")}, indent=2),
          flush=True)
    if row.get("outcome") == "not_run":
        return 3
    m = pagoda.measure(rid, os.path.join(RUNS_DIR, rid), os.path.join(LOGS_DIR, rid), row)
    print(json.dumps(pagoda.brief(m), indent=1)[:6000])
    return 0


SMOKE = ("Create a file named hello.txt containing the single word hi, using the "
         "terminal. Then print it with cat and tell me what `uname -a` and `pwd` print.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(variants.VARIANTS))
    # the reasoning effort Hermes sends (--reasoning) / Pi's thinking level
    # (--thinking): xhigh for the Bonsai arm, max for the Flash-Next arm
    ap.add_argument("--arm", choices=("xhigh", "medium", "low", "max"))
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--tag", default="pilot")
    # No default: Hermes' own --max-turns default applies (the 1000 that stood
    # here was invented; docs/CONSTANTS-AUDIT.md, 2026-09-27).
    ap.add_argument("--max-turns", type=int, default=None)
    # 7 h per prompt (coordinator, 2026-09-24): the prompt author's own V0 run
    # took 5 h on an RTX 3060 in one session; the pilot's 4 h was too short.
    ap.add_argument("--run-budget", type=int, default=25200)
    ap.add_argument("--iterative", type=int, default=0, metavar="MAX_PROMPTS",
                    help="the iterative protocol: up to N prompts, follow-ups from the grader. "
                         "REFUSED unless --allow-graded-followups is also given (operator, "
                         "2026-09-26: a follow-up built from the grader's failed checks leaks "
                         "the answer key)")
    ap.add_argument("--allow-graded-followups", action="store_true",
                    help="permit --iterative; every prompt >= 2 is then recorded as ASSISTED "
                         "and is never the headline")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    # Harness tools arm (toolset_arms.py; SELF-IMPROVEMENT-LOG #47). THE DEFAULT
    # LOADOUT has the browser (operator, 2026-09-26): Hermes' own browser tools
    # against the served page; `--tools default` is the explicit opt-out (the
    # loadout without the browser). Every arm gets the loadout lines.
    ap.add_argument("--tools", choices=list(toolset_arms.ARMS), default=toolset_arms.DEFAULT_ARM,
                    help=f"default {toolset_arms.DEFAULT_ARM} (the default loadout, loadout-2: "
                         "files, terminal, vision_analyze and three Playwright MCP tools); "
                         "`browser` = loadout-1 (Hermes' own browser toolset, the opt-out); "
                         "`default` = no browser")
    # Where that browser runs: "sandbox" (default) = a sidecar container sharing
    # the terminal container's network, only CDP on host loopback; "host" =
    # Chrome on Windows, which can reach host loopback services (opt-in).
    ap.add_argument("--browser-host", choices=toolset_arms.BROWSER_HOSTS,
                    default=toolset_arms.DEFAULT_BROWSER_HOST)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tools arm's profile diff, the sandbox network, env and "
                         "prerequisites; run nothing")
    # THE PAGODA (bench/octopus/pagoda.py, 2026-09-27): bench/voxel's r3f-stack
    # prompt through Hermes, run like V4 (Responses wire, default loadout,
    # single-shot), then served on :3001 and measured with bench/voxel's page
    # checker.
    ap.add_argument("--task", choices=("pagoda",),
                    help="a task other than the Octopus spec: `pagoda` = bench/voxel's "
                         "r3f-stack prompt, single-shot, measured by pagoda.measure")
    ap.add_argument("--measure", metavar="RUN_ID",
                    help="measure a finished pagoda run again (serve + page check + stack "
                         "+ proxy summary); runs no model")
    # THE PAGODA THROUGH PI (2026-09-28): the same prompt, single-shot, through
    # bench/sandbox/harness_box.py (Pi 0.87.1 in the box, its default loadout)
    # and the same relay and measure.
    ap.add_argument("--harness", choices=("hermes", "pi"), default="hermes",
                    help="--task pagoda: which harness runs it (default hermes)")
    ap.add_argument("--key-file", help="--harness pi: the key Pi sends (default the pi-dogfood "
                                       "test key); read into the box's env only, never printed")
    ap.add_argument("--features", help="--harness pi: a JSON X-Yamadori-Features header Pi sends on every request")
    ap.add_argument("--prompt-file", help="--task pagoda: this file's bytes are the prompt (pagoda.PROMPT_OVERRIDE)")
    ap.add_argument("--print", action="store_true",
                    help="--task pagoda: print the exact commands and the prompt file the run "
                         "would use, for either harness, and touch nothing")
    a = ap.parse_args()
    if a.harness != "hermes" and a.task != "pagoda":
        ap.error("--harness pi is for --task pagoda")
    if a.prompt_file:
        import pagoda
        pagoda.PROMPT_OVERRIDE = os.path.abspath(a.prompt_file)

    if a.measure:
        import pagoda
        rr = None
        for r in (json.loads(ln) for ln in open(RUNS, encoding="utf-8") if ln.strip()):
            if r.get("run_id") == a.measure and r.get("outcome") != "not_run":
                rr = r
        m = pagoda.measure(a.measure, os.path.join(RUNS_DIR, a.measure),
                           os.path.join(LOGS_DIR, a.measure), rr)
        print(json.dumps(pagoda.brief(m), indent=1)[:6000])
        return 0
    if a.task == "pagoda":
        return run_pagoda(a)

    if a.dry_run:
        print(toolset_arms.plan(a.tools, a.browser_host))
        return 0
    if a.preflight_only:
        print(json.dumps(preflight(), indent=2))
        return 0
    if a.smoke:
        path = os.path.join(ROOT, "index", "octopus", "smoke.md")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(SMOKE)
        rid = f"smoke-{time.strftime('%H%M%S')}"
        row = run_one(rid, path, a.arm or "low", 10, 900,
                      {"variant": "smoke", "arm": a.arm or "low", "rep": 0, "tag": "smoke"})
        print(json.dumps({k: v for k, v in row.items() if k != "preflight"}, indent=2)[:5000])
        return 0
    if not (a.variant and a.arm):
        ap.error("--variant and --arm are required")
    prompt = os.path.join(variants.PROMPTS, f"{a.variant}.md")
    manifest = json.load(open(os.path.join(variants.PROMPTS, "manifest.json")))
    rid = f"{a.tag}-{a.variant}-{a.arm}-{a.rep}"
    meta = {"variant": a.variant, "arm": a.arm, "rep": a.rep, "tag": a.tag,
            "tools": a.tools, "loadout": toolset_arms.LOADOUT_VERSION,
            "browser_host": a.browser_host if a.tools == "browser" else None,
            "prompt_sha256": manifest["variants"][a.variant]["sha256"],
            "versions": manifest["variants"][a.variant]["versions"]}
    if a.iterative and not a.allow_graded_followups:
        print(GRADED_FOLLOWUPS_REFUSAL, file=sys.stderr)
        return 2
    if a.iterative:
        meta["graded_followups"] = {"allowed": True, "flag": "--allow-graded-followups",
                                    "assisted_from_prompt": 2, "label": ASSISTED,
                                    "why": GRADED_FOLLOWUPS_WHY}
        rows = run_iterative(rid, a.variant, prompt, a.arm, a.max_turns, a.run_budget, meta,
                             a.iterative, tools=a.tools, browser_host=a.browser_host)
        print(json.dumps([{k: r.get(k) for k in ("run_id", "prompt_no", "outcome", "wall_s",
                                                 "exit", "session_id")} for r in rows],
                         indent=2))
        return 0
    row = run_one(rid, prompt, a.arm, a.max_turns, a.run_budget, meta, tools=a.tools,
                  browser_host=a.browser_host)
    print(json.dumps({k: row.get(k) for k in ("run_id", "outcome", "why", "wall_s", "exit",
                                               "killed_by_runner", "session_id")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
