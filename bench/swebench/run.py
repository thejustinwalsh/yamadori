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
SHARED_BY = "swebench overnight"
# Command-line fragments of the other GPU consumers this repo runs. A match on
# any live process other than ourselves means: wait.
OTHER_CONSUMERS = ("bench\\domain\\run.py", "bench/domain/run.py",
                   "mtpbench.py", "livecodebench.py", "queue_runner.py run",
                   "context_economy.py", "bench\\swebench\\run.py",
                   "bench/swebench/run.py")


def to_wsl(path: str) -> str:
    if os.environ.get("SWEBENCH_SIDE", "native") != "wsl":
        return os.path.abspath(path)
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


class SharedLane:
    """--shared: the card is shared on purpose with other benchmarks, each
    holding ONE request in flight (the operator's overnight plan). Another
    runner owns the worker's gpu lane pause; this only makes sure SOME pause
    is in force -- setting one as SHARED_BY if none is, refreshed every 20
    minutes while it is ours -- and removes only its own at the end."""

    def __init__(self, why: str):
        import jobs
        self.jobs = jobs
        self.why = why
        self._stop = threading.Event()

    def _ensure(self) -> None:
        rec = self.jobs.paused(LANE)
        if rec is None or rec.get("by") == SHARED_BY:
            self.jobs.pause(LANE, by=SHARED_BY, why=self.why, ttl_seconds=3600)
            if rec is None:
                print(f"  gpu lane was not paused; paused as {SHARED_BY!r}",
                      flush=True)
        else:
            print(f"  gpu lane already paused by {rec.get('by')!r}", flush=True)

    def acquire(self) -> None:
        self._ensure()
        threading.Thread(target=self._refresh, daemon=True).start()

    def _refresh(self) -> None:
        while not self._stop.wait(1200):
            try:
                self._ensure()
            except Exception as e:                               # noqa: BLE001
                print(f"  lane refresh failed: {e}", flush=True)

    def release(self) -> None:
        self._stop.set()
        rec = self.jobs.paused(LANE)
        if rec and rec.get("by") == SHARED_BY:
            self.jobs.resume(LANE)
            print("  gpu lane resumed (our pause)", flush=True)


class Evaluator:
    """Grades each instance as soon as its agent run finishes, in a thread,
    so partial results exist at any moment. CPU and Docker only. One harness
    run per (arm, instance), each with its own run id -- reports never
    overwrite each other -- and results.jsonl is rewritten after each."""

    def __init__(self, run_dir: str, run_id: str, subset: str):
        import queue
        self.q: "queue.Queue" = queue.Queue()
        self.run_dir, self.run_id, self.subset = run_dir, run_id, subset
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def submit(self, arm: str, iid: str) -> None:
        self.q.put((arm, iid))

    def _work(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            arm, iid = item
            out = os.path.join(self.run_dir, arm)
            try:
                code, text = wsl(["evaluate", "--out", to_wsl(out), "--subset",
                                  self.subset, "--instances", iid, "--run-id",
                                  f"{self.run_id}.{arm}.{iid}", "--workers", "1"],
                                 timeout=3600)
                print(f"  graded {arm:<14} {iid:<40} rc={code}", flush=True)
            except Exception as e:                               # noqa: BLE001
                print(f"  grading {arm} {iid} failed: {e}", flush=True)
            self.write()

    def write(self) -> list[dict]:
        with self.lock:
            rows = parse_results.parse_run(self.run_dir)
            tmp = os.path.join(self.run_dir, "results.jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tmp, os.path.join(self.run_dir, "results.jsonl"))
            return rows

    def finish(self) -> None:
        self.q.put(None)
        self.t.join()


def graded(out: str, iid: str) -> bool:
    import glob as _g
    return bool(_g.glob(os.path.join(out, f"*.{iid}.json")))


def done(out: str, iid: str) -> bool:
    p = os.path.join(out, "preds.json")
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as f:
        return iid in json.load(f)


# Where the agent and the harness run. Native Windows since 2026-09-23
# (dockerfix.py explains why); SWEBENCH_SIDE=wsl restores the WSL path.
NATIVE = os.environ.get("SWEBENCH_SIDE", "native") != "wsl"
WIN_PY = os.environ.get(
    "SWEBENCH_WIN_PY", r"C:\Users\jwals\swebench-yamadori\venv-win\Scripts\python.exe")


def wsl(args: list[str], timeout: float | None = None) -> tuple[int, str]:
    if NATIVE:
        cmd = [WIN_PY, os.path.join(HERE, "wsl_side.py"), *args]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONUTF8="1",
                                    MSWEA_SILENT_STARTUP="1"))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
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
        # stderr (HF warnings) is appended after stdout: take the JSON line.
        ids = json.loads([ln for ln in out.splitlines() if ln.startswith("[")][-1])
        if a.seed is not None:
            import random
            ids = sorted(ids)
            random.Random(a.seed).shuffle(ids)
        return ids
    sys.exit("name instances: --instances, --pilot or --all")


# An agent run that ended on one of these never reached an answer: the
# connection or the server failed (a proxy restart, a 502). It is rerun once,
# never scored as unresolved. A full context or the step limit is a real
# outcome and is kept.
TRANSIENT = ("APIConnectionError", "InternalServerError", "ServiceUnavailableError",
             "BadGatewayError", "Timeout", "APIError", "RateLimitError",
             "ConnectionError", "RemoteDisconnected")


def transient_failure(out: str, iid: str) -> str | None:
    p = os.path.join(out, iid, f"{iid}.traj.json")
    try:
        with open(p, encoding="utf-8") as f:
            st = (json.load(f).get("info") or {}).get("exit_status") or ""
    except (OSError, ValueError):
        return None
    return st if any(t in st for t in TRANSIENT) else None


def schedule(arm_list: list[str], ids: list[str], block: int) -> list[tuple[str, str]]:
    """The order of (arm, instance) jobs: every arm on the first `block`
    instances, arm-major (so a comparable block of each arm exists early),
    then the rest interleaved per instance (arm 1 on #21, arm 2 on #21, ...),
    so wherever the run is stopped the arms are paired."""
    jobs = [(arm, i) for arm in arm_list for i in ids[:block]]
    jobs += [(arm, i) for i in ids[block:] for arm in arm_list]
    return jobs


def run_shared(a, arm_list: list[str], ids: list[str], run_dir: str) -> int:
    lane = SharedLane(why=f"SWE-bench {a.run_id}")
    ev = Evaluator(run_dir, a.run_id, a.subset)
    jobs = schedule(arm_list, ids, a.block)
    skip = {tuple(x.split(":", 1)) for x in a.skip.split(",") if ":" in x}
    if skip:
        # Jobs still running in an orphaned agent process from a previous
        # runner: they finish and merge on their own; --evaluate-only grades
        # them afterwards.
        jobs = [j for j in jobs if j not in skip]
        print(f"  skipping in-flight jobs: {sorted(skip)}", flush=True)
    with open(os.path.join(run_dir, "schedule.json"), "w", encoding="utf-8") as f:
        json.dump({"workers": a.workers, "block": a.block, "jobs": jobs}, f, indent=0)
    qlock = threading.Lock()
    retried: set[tuple[str, str]] = set()

    def next_job():
        with qlock:
            return jobs.pop(0) if jobs else None

    def worker(n: int) -> None:
        while True:
            job = next_job()
            if job is None:
                return
            arm, iid = job
            out = os.path.join(run_dir, arm)
            redo = False
            if done(out, iid):
                why = transient_failure(out, iid)
                if why and job not in retried:
                    print(f"  {arm} {iid}: ended on {why}; rerunning once",
                          flush=True)
                    retried.add(job)
                    redo = True
                    import glob as _g
                    for old in _g.glob(os.path.join(out, f"*.{iid}.json")):
                        os.replace(old, old + f".superseded-{int(time.time())}")
                else:
                    if not graded(out, iid) and not a.no_evaluate:
                        ev.submit(arm, iid)
                    continue
            code, text = wsl(["agent", "--arm", arm, "--instance", iid,
                              "--out", to_wsl(out), "--subset", a.subset,
                              "--key-file", to_wsl(a.key_file)]
                             + (["--redo"] if redo else []))
            last = (text.strip().splitlines() or [""])[-1]
            print(f"  {time.strftime('%H:%M:%S')} w{n} {arm:<14} {iid:<36} "
                  f"rc={code} {last[:220]}", flush=True)
            if transient_failure(out, iid) and job not in retried:
                with qlock:
                    jobs.append(job)            # once more, at the end
            elif not a.no_evaluate:
                ev.submit(arm, iid)
            ev.write()

    try:
        lane.acquire()
        threads = []
        for n in range(a.workers):
            t = threading.Thread(target=worker, args=(n,), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(5)       # stagger container starts
        for t in threads:
            t.join()
    finally:
        lane.release()
        ev.finish()
    print(json.dumps(parse_results.summarize(ev.write()), indent=1))
    return 0


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
    ap.add_argument("--seed", type=int, default=None,
                    help="with --all: run the instances in this seeded order, so "
                         "a partial run is a random sample, not one repo")
    ap.add_argument("--workers", type=int, default=1,
                    help="--shared: agent runs in flight at once")
    ap.add_argument("--block", type=int, default=10**6,
                    help="--shared: run every arm on the first BLOCK instances "
                         "arm-major, then interleave the arms per instance")
    ap.add_argument("--skip", default="",
                    help="--shared: arm:instance,... jobs to leave alone")
    ap.add_argument("--shared", action="store_true",
                    help="the card is shared on purpose (one request in flight "
                         "from us); do not wait for other consumers; grade "
                         "each instance as it finishes")
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
        with open(os.path.join(run_dir, "instances.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"subset": a.subset, "dataset": arms.DATASETS[a.subset],
                       "seed": a.seed, "order": ids, "arms": arm_list}, f,
                      indent=1)
        if a.shared:
            return run_shared(a, arm_list, ids, run_dir)
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
