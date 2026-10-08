#!/usr/bin/env python
"""Pause / resume the worker's gpu job lanes around a training window (operator, 2026-10-07: "pause the job lanes
(jobs.pause('gpu') / jobs.pause('gpu_a4000')) for the training window and resume after").

    python bench/decider/unsloth/lanes.py pause [--ttl 10800] [--why TEXT]
    python bench/decider/unsloth/lanes.py resume
    python bench/decider/unsloth/lanes.py status

A pause is a file next to the jobs DB with an expiry (mcp/jobs.py), so a crash cannot stall the queue past the ttl. A job
already running is not killed; `status` shows whether one still runs (wait for it before a GPU run).
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "mcp"))
import jobs  # noqa: E402

LANES = ("gpu", "gpu_a4000")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pause", "resume", "status"))
    ap.add_argument("--ttl", type=float, default=10800)
    ap.add_argument("--why", default="decision-model training (bench/decider/unsloth)")
    a = ap.parse_args()
    for lane in LANES:
        if a.cmd == "pause":
            jobs.pause(lane, "unsloth-train", a.why, a.ttl)
        elif a.cmd == "resume":
            print(lane, "resumed" if jobs.resume(lane) else "was not paused")
    for lane in LANES:
        p = jobs.paused(lane)
        print(lane, "running", jobs.running(lane), "claimable", jobs.claimable(lane),
              "paused until %s" % time.strftime("%H:%M:%S", time.localtime(p["until"])) if p else "not paused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
