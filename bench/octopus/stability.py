#!/usr/bin/env python
"""Is each grader check a MEASUREMENT? Grade one unchanged state N times and
count, per check, how often it passed. A check that is not N/N or 0/N on an
unchanged state flips on nothing and goes in grade.FLAKY (its result is then
kept in the row but left out of the headline).

    python bench/octopus/stability.py run [--n 5] [IDS...]   # grade each target N times (Docker)
    python bench/octopus/stability.py report [--version 3] [IDS...]

Targets are fixtures nothing writes to (octo/fixtures): the V0 reference and
v0f-V0-xhigh-1's state after prompt 2 (copied from its run dir, SOURCE.json).
Every row appended carries `stability` {i, n}; rows are ordinary grades
(grade.py), so `report` also reads rows written by regrade.py and earlier
graders (--version 2 is the "before").

Why (SELF-IMPROVEMENT-LOG #62): grader v2's runtime checks flipped on the
unchanged reference fixture (mousemove 1/5, hud 1/5, esc 0/5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grade as G       # noqa: E402
import run as runmod    # noqa: E402

FIX = os.path.join(runmod.OCTO, "fixtures")
TARGETS = {
    "fx-v0-ref": ("V0", os.path.join(FIX, "v0-reference")),
    "fx-v0f-p2": ("V0", os.path.join(FIX, "v0f-p2-state")),
}
# the same states under their other grade ids (the "before" rows)
ALIASES = {"fx-v0f-p2": ["v0f-V0-xhigh-1@p2"]}


def version_of(r: dict) -> int:
    g = r.get("grader")
    return g.get("version") if isinstance(g, dict) else 1


def table(ids: list[str], version: int, batch: str | None = None) -> dict:
    rows = G.jsonl(G.GRADES)
    out = {}
    for gid in ids:
        names = [gid] + ALIASES.get(gid, [])
        sel = [r for r in rows if r.get("grade_id") in names and version_of(r) == version
               and r.get("spec")]
        if version >= 3:
            # only this script's repeats, graded by the CURRENT browser script
            # (rows from while v3 was being written are not the measurement)
            # rows of ONE batch (default: the latest) graded by the current
            # browser script
            # (an explicit --batch skips the hash test: a comment-only edit
            # after a measurement changes the hash; the rows keep theirs)
            bc = G.grader_info()["sha256_16"]["browser_check.py"]
            sel = [r for r in sel if (r.get("stability") or {}).get("batch") and
                   (batch or (r["grader"].get("sha256_16") or {}).get("browser_check.py") == bc)]
            if batch:
                sel = [r for r in sel if r["stability"]["batch"] == batch]
            elif sel:
                last = max(r["stability"]["batch"] for r in sel)
                sel = [r for r in sel if r["stability"]["batch"] == last]
        counts: dict[str, dict] = {}
        for r in sel:
            for c in r["spec"]:
                k = counts.setdefault(c["check"], {"pass": 0, "fail": 0, "none": 0,
                                                   "method": c.get("method")})
                k["pass" if c.get("pass") is True else "fail" if c.get("pass") is False
                  else "none"] += 1
        n = len(sel)
        for k in counts.values():
            k["stable"] = (k["pass"] in (0, n) and k["fail"] in (0, n)
                           and k["none"] in (0, n))
        out[gid] = {"version": version, "n": n,
                    "browser_check_sha256_16": sorted({((r.get("grader") or {}).get(
                        "sha256_16") or {}).get("browser_check.py") for r in sel
                        if isinstance(r.get("grader"), dict)} - {None}),
                    "batch": sorted({(r.get("stability") or {}).get("batch") for r in sel}
                                    - {None}), "grade_ids": sorted({r["grade_id"] for r in sel}),
                    "headline": [r.get("spec_passed") for r in sel],
                    "total": [r.get("spec_total") for r in sel], "checks": counts}
    return out


def print_table(t: dict) -> None:
    for gid, d in t.items():
        print(f"\n{gid}  grader v{d['version']}  n={d['n']}  headline {d['headline']} / "
              f"{d['total']}")
        for name, k in d["checks"].items():
            flag = "" if k["stable"] else "   <-- UNSTABLE"
            print(f"  {name:28s} {k['method']:18s} pass {k['pass']}  fail {k['fail']}  "
                  f"none {k['none']}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--version", type=int, default=G.GRADER_VERSION)
    ap.add_argument("--batch", default=None, help="report this batch (default: the latest)")
    a = ap.parse_args()
    ids = a.ids or list(TARGETS)
    if a.cmd == "run":
        batch = time.strftime("%Y%m%d-%H%M%S")
        for i in range(a.n):
            for gid in ids:
                variant, d = TARGETS[gid]
                row = G.grade(gid, variant, d, None, True,
                              extra={"stability": {"batch": batch, "i": i + 1, "n": a.n,
                                                   "why": "unchanged-state repeat (#62)"}})
                print(gid, i + 1, row.get("spec_passed"), "/", row.get("spec_total"),
                      "unknown", row.get("spec_unknown"), flush=True)
    t = table(ids, a.version, batch if a.cmd == "run" else a.batch)
    print_table(t)
    print(json.dumps({g: {k: v for k, v in d.items() if k != "checks"} for g, d in t.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
