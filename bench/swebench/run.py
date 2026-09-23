#!/usr/bin/env python
"""Run SWE-bench (mini-swe-agent, bash-only) against Yamadori. Windows side.

    python bench/swebench/run.py --run-id pilot-20260922 --arms bonsai \
        --instances django__django-15277 --key-file <path to key file>
    python bench/swebench/run.py --run-id pilot-20260922 \
        --arms bonsai,yamadori-auto,yamadori --pilot --key-file K
    python bench/swebench/run.py --run-id v50-bonsai --arms bonsai \
        --subset verified-mini --all --key-file K
    python bench/swebench/run.py --run-id X --evaluate-only --arms bonsai

Use the stack interpreter (C:\\Users\\jwals\\textgen\\installer_files\\env\\
python.exe). Queue-able through bench/queue_runner.py:
    queue_runner.py add swebench/run.py --args "--run-id ... --key-file ..."
        --needs proxy

WHAT HAPPENS

  1. Waits until no other GPU consumer runs (AGENTS.md "one GPU consumer at a
     time"): the gpu lane must not be paused by someone else, and no other
     known benchmark process may be alive. Then pauses the worker's gpu lane
     under our name, refreshes it every 10 minutes, and resumes it at the end
     -- in a `finally`, so a crash or Ctrl-C still gives the lane back. It
     never removes a pause somebody else holds.
  2. For each arm, for each instance (arm-major, so arms never interleave on
     the card): `wsl_side.py agent` runs mini-swe-agent on ONE instance with
     the leaderboard config plus the endpoint overlay. Wall time is measured
     around that call. Resumable: an instance already in the arm's preds.json
     is skipped by mini-swe-agent itself.
  3. `wsl_side.py evaluate` grades the arm's patches with the swebench
     harness in Docker (CPU only -- the lane is already released by then).
  4. parse_results.py writes results/<run-id>/results.jsonl.

THE KEY is only ever a file path here. The WSL side reads the file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))
import arms  # noqa: E402
import parse_results  # noqa: E402

RESULTS = os.path.join(HERE, "results")
WSL_PY = "~/swebench-yamadori/.venv/bin/python"
LANE = "gpu"
BY = "bench/swebench/run.py"
# Command-line fragments of the other GPU consumers this repo runs. A match on
# any live process other than ourselves means: wait.
OTHER_CONSUMERS = ("bench\\domain\\run.py", "bench/domain/run.py",
                   "mtpbench.py", "livecodebench.py", "queue_runner.py run",
                   "context_economy.py", "bench\\swebench\\run.py",
                   "bench/swebench/run.py")


def to_wsl(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def other_consumers() -> list[str]:
    """Live processes that look like another GPU benchmark -- never ourselves
    or any process we descend from (a shell or queue_runner that started us)."""
    ps = ("Get-CimInstance Win32_Process | ForEach-Object "
          "{ \"$($_.ProcessId)`t$($_.ParentProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ["<process scan failed: cannot prove the card is free>"]
    procs: dict[int, tuple[int, str]] = {}
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            procs[int(parts[0])] = (int(parts[1]), parts[2])
    mine, pid = set(), os.getpid()
    while pid in procs and pid not in mine:
        mine.add(pid)
        pid = procs[pid][0]
    mine.add(pid)
    hits = []
    for pid, (_pp, cmd) in procs.items():
        if pid in mine or ("Win32_Process" in cmd and "powershell" in cmd.lower()):
            continue
        if any(f in cmd for f in OTHER_CONSUMERS):
            hits.append(f"{pid}: {cmd[:160]}")
    return hits


class Lane:
    """The worker's gpu lane, held under our name while the model is busy."""

    def __init__(self, why: str):
        import jobs
        self.jobs = jobs
        self.why = why
        self.held = False
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def wait_until_free(self, poll: int = 60) -> None:
        # Free on TWO consecutive looks, `poll` apart: a series of runs (one
        # benchmark launching the next) has gaps of seconds between them, and
        # a single look can land in one.
        announced, free_once = False, False
        while True:
            rec = self.jobs.paused(LANE)
            others = other_consumers()
            if (not rec or rec.get("by") == BY) and not others:
                if free_once:
                    return
                free_once = True
                time.sleep(poll)
                continue
            free_once = False
            if not announced:
                print(f"  waiting for the card: pause={rec} others={others}",
                      flush=True)
                announced = True
            time.sleep(poll)

    def acquire(self) -> None:
        self.wait_until_free()
        self.jobs.pause(LANE, by=BY, why=self.why, ttl_seconds=1800)
        self.held = True
        self._t = threading.Thread(target=self._refresh, daemon=True)
        self._t.start()
        print(f"  gpu lane paused by {BY} (refreshed every 10 min)", flush=True)

    def _refresh(self) -> None:
        while not self._stop.wait(600):
            rec = self.jobs.paused(LANE)
            if rec and rec.get("by") != BY:
                print(f"  WARNING: gpu lane taken over by {rec.get('by')}",
                      flush=True)
                continue
            self.jobs.pause(LANE, by=BY, why=self.why, ttl_seconds=1800)

    def release(self) -> None:
        self._stop.set()
        if not self.held:
            return
        rec = self.jobs.paused(LANE)
        if rec is None or rec.get("by") == BY:
            self.jobs.resume(LANE)
            print("  gpu lane resumed", flush=True)
        self.held = False


def wsl(args: list[str], timeout: float | None = None) -> tuple[int, str]:
    cmd = ["wsl", "-d", "Ubuntu", "--", "bash", "-lc",
           " ".join([WSL_PY, to_wsl(os.path.join(HERE, "wsl_side.py"))]
                    + [_q(a) for a in args])]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def instance_ids(a) -> list[str]:
    if a.instances:
        return [i.strip() for i in a.instances.split(",") if i.strip()]
    if a.pilot:
        return list(arms.PILOT)
    if a.all:
        code, out = wsl(["ids", "--subset", a.subset])
        if code != 0:
            sys.exit(f"could not list {a.subset}: {out[-500:]}")
        return json.loads(out.strip().splitlines()[-1])
    sys.exit("name instances: --instances, --pilot or --all")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arms", default="bonsai",
                    help=f"comma-separated, from {sorted(arms.ARMS)}")
    ap.add_argument("--instances", default="")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--subset", default="verified", choices=sorted(arms.DATASETS))
    ap.add_argument("--key-file")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--no-evaluate", action="store_true")
    ap.add_argument("--eval-workers", type=int, default=4)
    a = ap.parse_args()
    arm_list = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arm_list if x not in arms.ARMS]
    if bad:
        sys.exit(f"unknown arms {bad}; known {sorted(arms.ARMS)}")
    run_dir = os.path.join(RESULTS, a.run_id)
    os.makedirs(run_dir, exist_ok=True)

    if not a.evaluate_only:
        if not a.key_file or not os.path.exists(a.key_file):
            sys.exit("--key-file must name an existing file (the key is read "
                     "from it on the WSL side and never printed)")
        ids = instance_ids(a)
        print(f"  run {a.run_id}: arms {arm_list}, {len(ids)} instances, "
              f"subset {a.subset}", flush=True)
        lane = Lane(why=f"SWE-bench {a.run_id}")
        try:
            lane.acquire()
            for arm in arm_list:
                out = os.path.join(run_dir, arm)
                for iid in ids:
                    # Between instances: if another consumer appeared, stop
                    # and say so rather than share the card.
                    others = other_consumers()
                    if others:
                        print(f"  STOPPING: another GPU consumer appeared: "
                              f"{others}", flush=True)
                        return 3
                    code, text = wsl(["agent", "--arm", arm, "--instance", iid,
                                      "--out", to_wsl(out), "--subset", a.subset,
                                      "--key-file", to_wsl(a.key_file)])
                    last = (text.strip().splitlines() or [""])[-1]
                    print(f"  {arm:<14} {iid:<40} rc={code} {last[:200]}",
                          flush=True)
        finally:
            lane.release()

    if not a.no_evaluate:
        for arm in arm_list:
            out = os.path.join(run_dir, arm)
            if not os.path.exists(os.path.join(out, "preds.json")):
                print(f"  {arm}: no preds.json, nothing to evaluate", flush=True)
                continue
            code, text = wsl(["evaluate", "--out", to_wsl(out), "--subset",
                              a.subset, "--run-id", f"{a.run_id}.{arm}",
                              "--workers", str(a.eval_workers)])
            print(f"  evaluate {arm}: rc={code} {text.strip()[-300:]}", flush=True)

    rows = parse_results.parse_run(run_dir)
    with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(parse_results.summarize(rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
