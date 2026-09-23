#!/usr/bin/env python
"""Run GPU experiments one at a time, in order, unattended.

WHY A QUEUE AND NOT JUST PARALLELISM

Building experiments in parallel is free. Running them is not: there is one
llama-swap in front of two GPUs, and several benchmark processes generating at
once do not go faster -- they interleave, contend, and push each other into
model swaps. Two concurrent runs earlier produced 502s and connection refusals
that looked like server instability and were really self-inflicted load, plus a
duplicate process quietly doubling the work.

So the split is: agents build and dry-run their harnesses concurrently, and
everything that needs the GPU goes through here, serially.

WHY IT MATTERS FOR THE SCIENCE, NOT JUST THE THROUGHPUT

A run competing with another run has different latency, different timeout
behaviour, and a different chance of hitting a model swap mid-generation. Those
differences land unevenly across conditions and become a confound. Serialising
is what makes two arms measured an hour apart comparable at all.

FAILURE HANDLING

Each job is checked before it starts: the endpoints it needs must answer. A job
that fails does not stop the queue, because the next experiment is usually
independent and a night of queued work should not be lost to one bad argument.
Failures are recorded and reported at the end rather than swallowed.

Jobs are resumable by construction: every experiment here appends to its own
jsonl and skips completed work, so re-queuing a job continues it.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, "queue.jsonl")
LOG = os.path.join(HERE, "queue_log.jsonl")
PY = sys.executable

ENDPOINTS = {
    "model": "http://127.0.0.1:11434/v1/models",
    "proxy": "http://127.0.0.1:1234/v1/models",
}


def endpoint_alive(url: str, timeout: int = 20) -> bool:
    try:
        req = urllib.request.Request(url)
        key = os.environ.get("YAMADORI_API_KEY", "")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:                                            # noqa: BLE001
        return False


def add(script: str, args: list[str], needs: list[str], note: str = "") -> None:
    job = {"script": script, "args": args, "needs": needs, "note": note,
           "queued": time.time()}
    with open(QUEUE, "a", encoding="utf-8") as f:
        f.write(json.dumps(job) + "\n")
    print(f"  queued {script} {' '.join(args)}")


def pending() -> list[dict]:
    if not os.path.exists(QUEUE):
        return []
    jobs = [json.loads(l) for l in open(QUEUE, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(LOG):
        for line in open(LOG, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add(r["key"])
            except Exception:                                    # noqa: BLE001
                continue
    return [j for j in jobs if _key(j) not in done]


def _key(job: dict) -> str:
    return job["script"] + "\x00" + "\x00".join(job["args"])


def run_one(job: dict) -> dict:
    missing = [n for n in job.get("needs", []) if not endpoint_alive(ENDPOINTS[n])]
    if missing:
        # Not recorded as done: the endpoint may come back, and a skipped job
        # should be retried on the next pass rather than lost.
        return {"key": _key(job), "status": "skipped",
                "why": f"endpoints down: {missing}", "record": False}

    script = os.path.join(HERE, job["script"])
    cmd = [PY, script] + job["args"]
    print(f"\n{'=' * 70}\n  RUNNING  {job['script']} {' '.join(job['args'])}")
    if job.get("note"):
        print(f"  {job['note']}")
    print(f"{'=' * 70}", flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=os.path.dirname(HERE), text=True)
        rc = p.returncode
    except Exception as e:                                       # noqa: BLE001
        return {"key": _key(job), "status": "error",
                "why": f"{type(e).__name__}: {e}",
                "secs": round(time.time() - t0), "record": True}
    return {"key": _key(job), "status": "ok" if rc == 0 else "failed",
            "rc": rc, "secs": round(time.time() - t0), "record": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    # A job's own flags must not be parsed by THIS parser. `--args` taking a
    # list ran straight into two argparse behaviours: `--n 18` was rejected as
    # an ambiguous abbreviation of `--needs`/`--note`, and `--report-only` was
    # read as an unknown option of the queue rather than a value. One quoted
    # string, split here, has neither problem.
    a = sub.add_parser("add")
    a.add_argument("script")
    a.add_argument("--args", default="", help='the job flags, quoted: "--n 18"')
    a.add_argument("--needs", default="model",
                   help="comma-separated: model,proxy")
    a.add_argument("--note", default="")
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--passes", type=int, default=2,
                   help="re-attempt skipped jobs this many times")
    r.add_argument("--wait", type=int, default=60,
                   help="seconds between passes when jobs were skipped")
    args = ap.parse_args()

    if args.cmd == "add":
        add(args.script, shlex.split(args.args),
            [n for n in args.needs.split(",") if n], args.note)
        return

    if args.cmd == "list":
        jobs = pending()
        print(f"  {len(jobs)} pending")
        for j in jobs:
            print(f"    {j['script']} {' '.join(j['args'])}"
                  + (f"   -- {j['note']}" if j.get("note") else ""))
        return

    results = []
    for p in range(args.passes):
        jobs = pending()
        if not jobs:
            break
        print(f"\n  pass {p + 1}: {len(jobs)} jobs pending")
        skipped = 0
        for job in jobs:
            res = run_one(job)
            results.append(res)
            if res.pop("record", True):
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({**res, "at": time.time()}) + "\n")
            else:
                skipped += 1
                print(f"  SKIPPED {job['script']}: {res['why']}", flush=True)
        if not skipped:
            break
        print(f"  {skipped} skipped, waiting {args.wait}s before retrying")
        time.sleep(args.wait)

    print(f"\n{'=' * 70}\n  QUEUE FINISHED\n{'=' * 70}")
    for r_ in results:
        print(f"  {r_['status']:<9} {r_.get('secs', 0):>6}s  {r_['key'].split(chr(0))[0]}"
              + (f"   {r_.get('why', '')}" if r_.get("why") else ""))
    left = pending()
    if left:
        print(f"\n  {len(left)} still pending (endpoints were down):")
        for j in left:
            print(f"    {j['script']} {' '.join(j['args'])}")


if __name__ == "__main__":
    main()
