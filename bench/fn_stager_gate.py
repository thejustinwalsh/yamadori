#!/usr/bin/env python
"""The engine arms of patch 0030 (the staging ring, docs/FLASH-NEXT.md 13.7 step 3) through bench/flashnext_gate.py, one arm
at a time, on the main card only, WITHOUT the gate's production handling:

  - the gate's quiet check and restore both touch `/upstream/bonsai/...`, which would LOAD bonsai on a card that is
    deliberately empty (the coordinator, 2026-10-06: leave whatever model unloaded rather than loading one), so both are
    replaced here by a check that the card is empty and a restore that only records the card's state;
  - `<out>/YIELD` (a file) is looked at between arms: when it exists the run stops after the arm in progress.

    python bench/fn_stager_gate.py --go [--out DIR] [--arms ...]

The arms (bench/flashnext_gate.py, `pf-st*`): st0 = LLAMA_STAGER off (the control), st1 = on with 3 threads / 4 MiB chunks / a ring
of 16 (the probe's sweep start, bench/results/fn_probe/20261006-024716), lm3 / lm6 = 3 / 6 slots (50 / 38 cache rows),
st1p = the experts pinned. `-b8k` = prefill n=3 (8K and 32K), `-full` = decode, needles, corrupt, KL, checkpoint, cancel,
`-cold` = the first prompt after a load with the file cache evicted. The A4000 is never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLASHNEXT_FLASH_BIN", "C:/Users/jwals/engines/llama-upstream-flash-cand0030/src/build/bin")
import engine_corruption as ec                                        # noqa: E402
import flashnext_gate as fg                                           # noqa: E402

OLD = "C:/Users/jwals/octo/flashnext-gate-20260929"
DEFAULT_OUT = "C:/Users/jwals/octo/flashnext-gate-20261006"
PLAN = [
    ("pf-st0-lm3-b8k", ["prefill"]),
    ("pf-st1-lm3-b8k", ["prefill"]),
    ("pf-st1-lm6-b8k", ["prefill"]),
    ("pf-st0-lm6-b8k", ["prefill"]),
    ("pf-st1p-lm6-b8k", ["prefill"]),
    ("pf-st1-lm6-b8k-full", ["speed", "needles", "corrupt", "kl", "ckpt", "cancel"]),
    ("pf-st0-lm3-b8k-cold", ["cold"]),
    ("pf-st1-lm3-b8k-cold", ["cold"]),
]


def card_used_mib() -> int:
    for ln in subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.splitlines():
        u, used = [c.strip() for c in ln.split(",")]
        if u == ec.CARD_UUID:
            return int(float(used))
    raise SystemExit("the main card was not found")


def no_quiet_check(prod_argv, seconds):                                # noqa: ARG001
    return None


def restore_nothing(prod_argv):                                        # noqa: ARG001
    rec = {"restored": "nothing: the card is left empty on purpose", "gpu": ec.gpu(), "running": list(ec.running())}
    ec.log(f"restore: {rec}")
    return rec


def seed(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for f in ("kl-base.logits", "kl-text.txt"):
        if not os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(OLD, f), os.path.join(out, f))
    gj = os.path.join(out, "gate.json")
    if not os.path.exists(gj):
        y = json.load(open(os.path.join(OLD, "gate.json"), encoding="utf-8")).get("kl_yardstick")
        json.dump({"started": "seeded 2026-10-06", "kl_yardstick": y, "arms": {}}, open(gj, "w"), indent=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--go", action="store_true", help="the coordinator's GPU go for step 3 (the main card only)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    if not a.go:
        print("refusing: this uses the GPU; pass --go")
        return 2
    ec.wait_quiet = no_quiet_check
    ec.restore = restore_nothing
    seed(a.out)
    ran = []
    for name, steps in PLAN:
        if a.arms and name not in a.arms:
            continue
        if os.path.exists(os.path.join(a.out, "YIELD")):
            ec.log("YIELD: stopping before " + name)
            break
        used = card_used_mib()
        if used > 1500 or "flash-next" in ec.running():
            ec.log(f"stopping before {name}: the card is not empty ({used} MiB used, running {list(ec.running())})")
            break
        ec.log(f"=== arm {name} steps {steps}")
        rc = fg.main(["--window", "--out", a.out, "--arms", name, "--steps"] + steps)
        ran.append((name, rc))
    print(json.dumps({"ran": ran, "yielded": os.path.exists(os.path.join(a.out, "YIELD")), "card_used_mib": card_used_mib(),
                      "running": list(ec.running())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
