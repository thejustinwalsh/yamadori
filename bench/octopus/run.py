#!/usr/bin/env python
"""Run one Octopus Invaders variant through Hermes (Windows install, Docker
terminal backend).

    python bench/octopus/run.py --variant V0 --arm xhigh --rep 1 --tag pilot
    python bench/octopus/run.py --preflight-only
    python bench/octopus/run.py --smoke          # a one-file task, sandbox proof

Use the stack interpreter. One run at a time, never beside another GPU
consumer (AGENTS.md "Before you claim anything works", 4).

THE HARNESS (operator, 2026-09-24): the WINDOWS Hermes
(%LOCALAPPDATA%\\hermes\\bin\\hermes.exe) with its own profile,
HERMES_HOME=C:\\Users\\jwals\\octo\\hermes-home (bench/octopus/make_profile.py:
the operator's config with compression on and the docker backend; the
hermes-dogfood key). The operator's own profile is never read or written.

SAFETY: Hermes runs commands the model chooses. Its terminal is the DOCKER
backend (Docker Desktop), with ONLY this run's folder bind-mounted at
/workspace; its file tools go through the same container. If the backend
does not come up, the run fails; there is no fallback to the host and never
--yolo. The TERMINAL_* variables below win over config.yaml
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

RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(RESULTS, "runs.jsonl")
OCTO = r"C:\Users\jwals\octo"
RUNS_DIR = os.path.join(OCTO, "runs")
LOGS_DIR = os.path.join(OCTO, "logs")
HERMES_HOME = os.path.join(OCTO, "hermes-home")
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
        "test_laya_head.py --serve", "queue_runner.py", "bench\\domain\\run.py",
        "bench/domain/run.py", "livecodebench.py", "recipe_oracle.py",
        "context_economy.py", "octopus\\run.py --variant", "octopus/run.py --variant")


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


def slots_idle() -> tuple[bool, list]:
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


def preflight() -> dict:
    out = {"busy": busy_processes(), "proxy": proxy_alive()}
    a, s1 = slots_idle()
    time.sleep(5)
    b, s2 = slots_idle()
    out["slots"] = [s1, s2]
    out["slots_idle"] = a and b
    out["docker"] = docker_awake()
    out["hermes"] = os.path.isfile(HERMES) and os.path.isfile(
        os.path.join(HERMES_HOME, "config.yaml")) and os.path.isfile(
        os.path.join(HERMES_HOME, ".env"))
    out["ok"] = (not out["busy"] and out["slots_idle"] and out["proxy"] in ("200", "401")
                 and out["docker"].startswith("ok") and out["hermes"])
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


def set_profile_run(run_id: str, run_dir: str) -> None:
    """Point the dogfood profile's two marked terminal lines at this run
    (make_profile.py writes them): the /workspace volume and the container
    label key. Each must match exactly once."""
    import re
    path = os.path.join(HERMES_HOME, "config.yaml")
    text = open(path, encoding="utf-8").read()
    subs = [(r"(?m)^  docker_shared_container_key: .*# OCTO-RUN-KEY.*$",
             f'  docker_shared_container_key: "octo-{run_id}"   # OCTO-RUN-KEY (run.py rewrites per run)'),
            (r"(?m)^  docker_volumes: .*# OCTO-RUN-VOLUMES.*$",
             f"  docker_volumes: ['{run_dir}:/workspace']   # OCTO-RUN-VOLUMES (run.py rewrites per run)")]
    for pat, rep in subs:
        text, n = re.subn(pat, lambda _m, r=rep: r, text)
        if n != 1:
            raise SystemExit(f"profile line not found once ({n}): {pat}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


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


def run_prompt(run_id: str, prompt: str, effort: str, max_turns: int, budget: int,
               meta: dict, n: int = 1, session_id: str | None = None,
               toolsets: str | None = None) -> dict:
    """One Hermes invocation: prompt 1 starts the session in a FRESH folder;
    prompt n > 1 resumes `session_id` in the same folder (--resume), its logs
    suffixed `.p<n>` (relay.p2.jsonl, hermes.p2.jsonl, meta.p2.json, ...)."""
    pf = preflight()
    if not pf["ok"]:
        row = {"run_id": run_id, "prompt_no": n, "outcome": "not_run", "why": "preflight",
               **meta, "preflight": pf, "t": time.time()}
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
    elif not (os.path.isdir(run_dir) and session_id):
        raise SystemExit(f"prompt {n} needs the existing run folder and a session id")
    shutil.copy2(prompt, os.path.join(log_dir, f"prompt{sfx}.md"))
    if _port_open(RELAY_PORT):
        raise SystemExit(f"port {RELAY_PORT} is taken: another relay is running")
    relay = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "relay.py"), "--listen",
         f"127.0.0.1:{RELAY_PORT}", "--upstream", UPSTREAM, "--out",
         os.path.join(log_dir, f"relay{sfx}.jsonl")],
        stdout=open(os.path.join(log_dir, f"relay{sfx}.out"), "w"), stderr=subprocess.STDOUT)
    for _ in range(20):
        if _port_open(RELAY_PORT):
            break
        time.sleep(0.25)
    else:
        relay.kill()
        raise SystemExit("relay did not start")

    set_profile_run(run_id, run_dir)
    env = dict(os.environ)
    env.update({
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
        "PYTHONIOENCODING": "utf-8",
    })
    cmd = [HERMES, "chat", "--query-file", os.path.join(log_dir, f"prompt{sfx}.md"),
           "--oneshot", "--provider", PROVIDER, "-m", "yamadori", "--reasoning", effort,
           "--max-turns", str(max_turns), "--run-budget", str(budget),
           "--ignore-rules", "--source", "tool", "--format", "stream-json"]
    if session_id:
        cmd += ["--resume", session_id]
    if toolsets:
        cmd += ["-t", toolsets]
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
    meta_run = {"run_id": run_id, "prompt_no": n, "cmd": cmd, "cwd": run_dir, "exit": rc,
                "killed_by_runner": killed, "t_start": t0, "t_end": t1,
                "session_id": sid, "export_rc": e.returncode,
                "export_out": (e.stdout + e.stderr)[-800:],
                "containers_left": left,
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
            meta: dict, toolsets: str | None = None) -> dict:
    """A single-prompt run (the pilot's protocol)."""
    return run_prompt(run_id, prompt, effort, max_turns, budget, meta, 1, None, toolsets)


def run_iterative(run_id: str, variant: str, prompt: str, effort: str, max_turns: int,
                  budget: int, meta: dict, max_prompts: int = 6) -> list[dict]:
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
                     f"iterative<= {max_prompts}"}, 1)
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
                         {**meta, "protocol": f"iterative<= {max_prompts}"}, n,
                         session_id=row.get("session_id"))
        rows.append(row)
    return rows


SMOKE = ("Create a file named hello.txt containing the single word hi, using the "
         "terminal. Then print it with cat and tell me what `uname -a` and `pwd` print.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(variants.VARIANTS))
    ap.add_argument("--arm", choices=("xhigh", "low"))
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--max-turns", type=int, default=1000)   # the last V0 used 16 in 68 min
    # 7 h per prompt (coordinator, 2026-09-24): the prompt author's own V0 run
    # took 5 h on an RTX 3060 in one session; the pilot's 4 h was too short.
    ap.add_argument("--run-budget", type=int, default=25200)
    ap.add_argument("--iterative", type=int, default=0, metavar="MAX_PROMPTS",
                    help="the iterative protocol: up to N prompts, follow-ups from the grader")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

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
            "prompt_sha256": manifest["variants"][a.variant]["sha256"],
            "versions": manifest["variants"][a.variant]["versions"]}
    if a.iterative:
        rows = run_iterative(rid, a.variant, prompt, a.arm, a.max_turns, a.run_budget, meta,
                             a.iterative)
        print(json.dumps([{k: r.get(k) for k in ("run_id", "prompt_no", "outcome", "wall_s",
                                                 "exit", "session_id")} for r in rows],
                         indent=2))
        return 0
    row = run_one(rid, prompt, a.arm, a.max_turns, a.run_budget, meta)
    print(json.dumps({k: row.get(k) for k in ("run_id", "outcome", "why", "wall_s", "exit",
                                               "killed_by_runner", "session_id")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
