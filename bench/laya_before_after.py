#!/usr/bin/env python
"""Show zero-shot vs trained on the SAME questions, over the real HTTP API.

    .venv-laya/Scripts/python.exe bench/laya_before_after.py

Starts a service on a free port (never 1237 -- the live one stays up), asks
every question with engine=both, and prints the two engines side by side with
their margins against the gate. This is the end-to-end check that the trained
artefact is actually reachable through the API, not just good in a notebook.

Pass --port N to use an already-running service instead of starting one.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = os.path.join(ROOT, ".venv-laya", "Scripts", "python.exe")

# FRESH questions, written for this check and deliberately present in NEITHER
# label file. The shipped artefact is fitted on every label, so anything drawn
# from bench/laya_*_labels*.jsonl would be training data and 12/12 on it would
# mean nothing. bench/test_laya_head.py asserts these stay unseen.
QUESTIONS = [
    ("investigate", "Which module in this repo decides the abstain gate for a "
                    "trained head, and what value is it set to?", ""),
    ("investigate", "Where do we cache extracted Laya features, and what key "
                    "are the records stored under?", ""),
    ("investigate", "What does our watchdog restart when the decision service "
                    "stops answering on its port?", ""),
    ("investigate", "Which crates in this workspace still build against the "
                    "old wgpu surface API?", "crates/ holds the Rust side."),
    ("answer_directly", "What does the repr(transparent) attribute guarantee "
                        "about a Rust struct's layout?", ""),
    ("answer_directly", "In WebGPU, how does a storage texture differ from a "
                        "sampled texture?", ""),
    ("answer_directly", "How does TypeScript's keyof behave when applied to a "
                        "union of object types?", ""),
    ("answer_directly", "In three.js, what does calling dispose() on a "
                        "material actually free, and what does it not?", ""),
    ("clarify", "Same as before, but for the other file.", ""),
    ("clarify", "Can you tidy that bit up?", ""),
    ("clarify", "Does that seem okay to you?", ""),
    ("clarify", "Make it match the other one.", ""),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port: int, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def wait(port: int, proc, timeout: int = 300) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                   timeout=20).read()
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    proc = None
    port = args.port
    if not port:
        port = free_port()
        env = dict(os.environ)
        env["LAYA_HTTP_PORT"] = str(port)
        env["LAYA_HOST"] = "127.0.0.1"
        proc = subprocess.Popen(
            [PY, os.path.join(ROOT, "mcp", "laya_service.py")], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(f"starting test service on {port} (live 1237 untouched)...")
    if port == 1237:
        raise SystemExit("refusing to run against the live service port")
    if not wait(port, proc):
        raise SystemExit("service did not come up")

    print(f"\n{'expected':16s} {'zero-shot':>16s} {'m':>6s}  "
          f"{'trained':>16s} {'m':>6s}   question")
    print("-" * 104)
    zs_ok = tr_ok = zs_abstain = tr_abstain = 0
    try:
        for want, q, ctx in QUESTIONS:
            r = post(port, "/route", {"task": "route_in", "question": q,
                                      "context": ctx, "engine": "both"})
            z, t = r["zero_shot"], r["trained"]
            zs_ok += z["choice"] == want
            tr_ok += t["choice"] == want
            zs_abstain += bool(z["abstain"])
            tr_abstain += bool(t["abstain"])
            zs = z["choice"] + ("*" if z["abstain"] else "")
            tt = t["choice"] + ("*" if t["abstain"] else "")
            print(f"{want:16s} {zs:>16s} {z['margin']:6.3f}  "
                  f"{tt:>16s} {t['margin']:6.3f}   {q[:44]}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    n = len(QUESTIONS)
    print("-" * 104)
    print(f"  zero-shot: {zs_ok}/{n} correct, but abstained on {zs_abstain}/{n}"
          f" -- {n - zs_abstain} usable decision(s)")
    print(f"  trained:   {tr_ok}/{n} correct, abstained on {tr_abstain}/{n}"
          f" -- {n - tr_abstain} usable decision(s)")
    print("\n* = abstained (margin below the engine's gate). An abstained "
          "answer is not a decision:\n  the caller gets nothing and has to "
          "fall back to System 2, which is the cost this head removes.")
    print("These 12 are unseen by the artefact. The held-out estimate over the "
          "full label set\nis the number to quote: see index/laya/route_in.json"
          " -> metrics.held_out_after.")


if __name__ == "__main__":
    main()
