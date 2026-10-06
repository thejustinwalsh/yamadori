#!/usr/bin/env python
"""Run bench/fn_probe/h2d_overlap on the 5060 Ti (or its selftest on a simulated device).

    python bench/fn_probe/run_probe.py --selftest            # no GPU, no CUDA call: the harness's own test (seconds)
    python bench/fn_probe/run_probe.py --go                  # the card. REFUSES unless it is free (see below)

What it answers (docs/FLASH-NEXT.md 13.4): does a host-to-device copy on a second stream overlap a compute kernel on this
card (WDDM), from pinned memory, from pageable memory (an mmapped file), through the staging ring of patch 0030, from
cudaHostRegister, and device-to-device? And how fast can the ring feed the link (threads x chunk x ring sweep)?

THE GPU IS SHARED: run this only on the operator's "GPU go". It needs ~2.5 GiB of VRAM and the link to itself, so it refuses
(exit 2) unless the 5060 Ti has at least --min-free-mib free (default 4096) -- i.e. the stack's flash-next is unloaded
(through the operator's window, never by this script) -- and it never touches :1234 / :11434, config.yaml or a process.

Three runs, one directory bench/results/fn_probe/<stamp>/ (probe-<tag>.json + .txt, nvidia-smi before / after):
  engine   the engine's shape: 3 x 256 MiB tensors a layer, 16 graphs of 16.2 ms, the host synchronising per graph, the ring
           sweep. Expected ~2.5 min.
  nosync   the same without the per-graph host sync (does the engine's sync pattern matter?). Expected ~2 min.
  b2048    4 graphs a layer (-b 2048: 4 ubatches), where the upload dominates. Expected ~1 min.
Expected total ~6 min plus the temp mapped file (768 MiB, created and deleted by the probe in --tmp).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXE_DIR = os.environ.get("OUT", r"C:\Users\jwals\engines\fn_probe")


def smi(fields: str) -> list[list[str]]:
    p = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    return [[c.strip() for c in ln.split(",")] for ln in p.stdout.splitlines() if ln.strip()]


def find_card() -> tuple[str, int, int]:
    """(uuid, used MiB, total MiB) of the 5060 Ti."""
    for idx, name, uuid, used, total in smi("index,name,uuid,memory.used,memory.total"):
        if "5060" in name:
            return uuid, int(float(used)), int(float(total))
    raise SystemExit("no RTX 5060 Ti found by nvidia-smi")


def run(exe: str, args: list[str], env: dict | None = None, log: str | None = None) -> int:
    cmd = [exe] + args
    print("  $", " ".join(cmd), flush=True)
    out = open(log, "w", encoding="utf-8") if log else None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for ln in p.stdout:
        print("   ", ln.rstrip())
        if out:
            out.write(ln)
    p.wait()
    if out:
        out.close()
    return p.returncode


def selftest(tmp: str) -> int:
    """The harness on a simulated device: a pinned or ring source must hide the upload, a pageable one must not."""
    exe = os.path.join(EXE_DIR, "h2d_overlap_sim.exe")
    if not os.path.exists(exe):
        print(f"build it first: bench\\fn_probe\\build.bat sim   ({exe})")
        return 2
    js = os.path.join(tmp, "probe-sim.json")
    # sim sizes keep the real ratio: a graph is ~1/4 of the blocking copy, a layer of compute is ~2x the upload
    rc = run(exe, ["--sim", "--mib", "32", "--tensors", "3", "--graphs", "8", "--kernel-ms", "2", "--layers", "6", "--reps", "3",
                   "--threads", "4", "--chunk-mib", "4", "--ring", "16", "--no-sweep", "--tmp", tmp, "--json", js])
    if rc:
        return rc
    d = json.load(open(js))
    E = {m["mode"]: m.get("E") for m in d["modes"] if "E" in m}
    expect_hi = ["pin", "stage-heap", "stage-mmap"]
    expect_lo = ["pageable", "mmap"]
    bad = [f"{m} E={E.get(m)} should be >= 0.8" for m in expect_hi if E.get(m, 0) < 0.8]
    bad += [f"{m} E={E.get(m)} should be < 0.5" for m in expect_lo if E.get(m, 1) >= 0.5]
    for b in bad:
        print("  FAIL", b)
    print(f"  selftest: {'PASS' if not bad else 'FAIL'} ({len(expect_hi) + len(expect_lo)} checks: "
          + ", ".join(f"{m} E={E[m]:.2f}" for m in expect_hi + expect_lo if m in E) + ")")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--go", action="store_true", help="really run on the card (the operator's 'GPU go')")
    ap.add_argument("--min-free-mib", type=int, default=4096)
    ap.add_argument("--tmp", default=os.environ.get("TEMP", "."))
    ap.add_argument("--out", help="result directory (default bench/results/fn_probe/<stamp>)")
    ap.add_argument("--only", help="engine,nosync,b2048 (default all)")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.tmp)
    if not a.go:
        print("refusing: this uses the GPU; pass --go on the operator's 'GPU go' (or --selftest for the simulated device)")
        return 2
    exe = os.path.join(EXE_DIR, "h2d_overlap.exe")
    if not os.path.exists(exe):
        print(f"build it first: bench\\fn_probe\\build.bat cuda   ({exe})")
        return 2
    uuid, used, total = find_card()
    if total - used < a.min_free_mib:
        print(f"refusing: the 5060 Ti has {total - used} MiB free of {total} ({used} used); the probe needs {a.min_free_mib}. "
              "The stack's model must be unloaded first, through the operator's window.")
        return 2
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = a.out or os.path.join(ROOT, "bench", "results", "fn_probe", stamp)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
    json.dump({"stamp": stamp, "uuid": uuid, "used_mib": used, "total_mib": total, "exe": exe,
               "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout}, open(os.path.join(out, "before.json"), "w"), indent=1)
    base = ["--tensors", "3", "--mib", "256", "--layers", "10", "--reps", "5", "--threads", "4", "--chunk-mib", "16", "--ring", "16",
            "--tmp", a.tmp]
    runs = {
        "engine": base + ["--graphs", "16", "--kernel-ms", "16.2"],
        "nosync": base + ["--graphs", "16", "--kernel-ms", "16.2", "--no-sync-graph", "--no-sweep"],
        "b2048": base + ["--graphs", "4", "--kernel-ms", "16.2", "--no-sweep"],
    }
    rc_all = 0
    for tag, args in runs.items():
        if a.only and tag not in a.only.split(","):
            continue
        print(f"== {tag}", flush=True)
        rc = run(exe, args + ["--json", os.path.join(out, f"probe-{tag}.json")], env, os.path.join(out, f"probe-{tag}.txt"))
        rc_all |= rc
    json.dump({"nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout},
              open(os.path.join(out, "after.json"), "w"), indent=1)
    print("results in", out)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
