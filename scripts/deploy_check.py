#!/usr/bin/env python
"""Is this deploy good? Run the live suite through :1234 right after a restart.

    python scripts/deploy_check.py --key-file PATH
    python scripts/deploy_check.py --key-file PATH --maintenance   # + intrusive
    python scripts/deploy_check.py --key-file PATH --wait 900      # longer boot

A deploy is NOT good until this exits 0 (AGENTS.md, "Before you claim
anything works"). The operator's rule: if we don't have tests that exercise
the real models, we don't have tests.

WHAT IT DOES, IN ORDER

  1. Waits for every service to answer /health: llama-swap :11434, the proxy
     :1234, the tools API :1235 -- and for llama-swap to report the chat
     model `ready`. A stack that is still loading is not failing. (Laya on
     :1237 is retired, 2026-09-24, docs/E1.md: E1's heads replace it, and
     the suites that tested its live service run offline only --
     scripts/run_tests.py RETIRED_LIVE.)
  2. Refuses to start while another live run is on the card (a python
     process running a live suite). Two consumers on one GPU degrade each
     other into 429s and 502s, and the loser looks like the one with the bug.
  3. Waits until the chat model's /slots are idle on two reads 5 s apart.
  4. Runs `scripts/run_tests.py --live --live-only`, which streams every
     check's pass/FAIL with its evidence, treats a 429 as NOT RUN, and exits
     1 on any failure, 3 when nothing failed but something did not run.
  5. Appends one line to logs/deploy_check.jsonl: when, the git HEAD, the
     exit code and the verdict. The key is never printed or recorded.

Exit code: run_tests.py's (0 GOOD, 1 NOT GOOD, 3 INCOMPLETE), or 2 when the
stack never came up or the card was never free (the suite did not start).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
PY = r"C:/Users/jwals/textgen/installer_files/env/python.exe"
if not os.path.exists(PY):
    PY = sys.executable

HEALTH = {"llama-swap": "http://127.0.0.1:11434/health",
          "proxy": "http://127.0.0.1:1234/health",
          "tools-api": "http://127.0.0.1:1235/health"}
SWAP = "http://127.0.0.1:11434"
CHAT_MODEL = os.environ.get("YAMADORI_CHAT_MODEL", "bonsai")
LIVE_MARKERS = ("test_live_stack.py", "test_tools_live.py", "run_tests.py --live",
                "deploy_check.py")
LOG = os.path.join(ROOT, "logs", "deploy_check.jsonl")


def _get(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def services_up() -> dict[str, str]:
    """{service: problem} for every service not yet answering."""
    bad = {}
    for name, url in HEALTH.items():
        try:
            code, _ = _get(url)
            if not 200 <= code < 300:
                bad[name] = f"HTTP {code}"
        except Exception as e:                                   # noqa: BLE001
            bad[name] = f"{type(e).__name__}: {e}"[:120]
    if "llama-swap" not in bad:
        try:
            _, body = _get(f"{SWAP}/running")
            run = {m.get("model"): m.get("state")
                   for m in json.loads(body).get("running") or []}
            if run.get(CHAT_MODEL) != "ready":
                bad["chat model"] = f"{CHAT_MODEL} is {run.get(CHAT_MODEL)!r}"
        except Exception as e:                                   # noqa: BLE001
            bad["chat model"] = f"/running: {type(e).__name__}: {e}"[:120]
    return bad


def other_live_runs() -> list[str]:
    """Command lines of python processes running a live suite, other than
    this process and its parents."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = {os.getpid(), os.getppid()}
    hits = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("\t")
        if not pid.strip().isdigit() or int(pid) in me:
            continue
        if any(m in cmd for m in LIVE_MARKERS):
            hits.append(f"pid {pid.strip()}: {cmd.strip()[:160]}")
    return hits


def slots_busy() -> list | None:
    """The ids of processing slots, or None when /slots cannot be read."""
    try:
        _, body = _get(f"{SWAP}/upstream/{CHAT_MODEL}/slots")
        return [s.get("id") for s in json.loads(body) if s.get("is_processing")]
    except Exception:                                            # noqa: BLE001
        return None


def head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def record(rc: int, verdict: str, secs: float, note: str = "") -> None:
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.time(), "head": head(), "rc": rc,
                            "verdict": verdict, "seconds": round(secs),
                            "note": note}) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--key-file", required=True,
                    help="file holding the proxy API key (never printed)")
    ap.add_argument("--wait", type=int, default=600,
                    help="seconds to wait for the stack and for an idle card")
    ap.add_argument("--maintenance", action="store_true",
                    help="also run the intrusive live tests (restart the proxy)")
    args = ap.parse_args(argv)
    t0 = time.time()
    deadline = t0 + args.wait

    print("  1. waiting for every service to answer /health", flush=True)
    while True:
        bad = services_up()
        if not bad:
            break
        if time.time() > deadline:
            print(f"  NOT STARTED: services not up after {args.wait}s: "
                  + json.dumps(bad))
            record(2, "NOT STARTED", time.time() - t0, "services down")
            return 2
        time.sleep(10)
    print("     all up", flush=True)

    print("  2. checking no other live run is on the card", flush=True)
    others = other_live_runs()
    if others:
        print("  NOT STARTED: another live run is using the card:\n    "
              + "\n    ".join(others))
        record(2, "NOT STARTED", time.time() - t0, "another live run")
        return 2

    print("  3. waiting for idle /slots on two reads 5 s apart", flush=True)
    while True:
        a = slots_busy()
        time.sleep(5)
        b = slots_busy()
        if a == [] and b == []:
            break
        if time.time() > deadline:
            print(f"  NOT STARTED: /slots never idle (busy {a} then {b})")
            record(2, "NOT STARTED", time.time() - t0, "slots busy")
            return 2
        time.sleep(10)
    print("     idle", flush=True)

    print("  4. the live suite, through :1234", flush=True)
    cmd = [PY, "-X", "utf8", os.path.join(ROOT, "scripts", "run_tests.py"),
           "--live", "--live-only", "--key-file", args.key_file]
    if args.maintenance:
        cmd.append("--maintenance")
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    verdict = {0: "GOOD", 3: "INCOMPLETE"}.get(rc, "NOT GOOD")
    record(rc, verdict, time.time() - t0)
    print(f"\n  DEPLOY {verdict} (exit {rc}); recorded in "
          f"{os.path.relpath(LOG, ROOT)}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
