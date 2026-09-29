#!/usr/bin/env python
"""Re-grade EVERY graded Octopus run, snapshot and fixture with the current
grader (grade.GRADER_VERSION), appending new rows to results/grades.jsonl.
Old rows are never touched.

    python bench/octopus/regrade.py --static        # no Docker: the static checks only,
                                                    # old row vs this grader, same state
    python bench/octopus/regrade.py                 # full re-grade (Docker, no GPU), appends
    python bench/octopus/regrade.py v0f-V0-xhigh-1@p1 ...   # only these grade ids

Why it exists (PROTOCOL): grader v2 changed what seven checks accept, after
results had been seen (v0f@p1's `Particles.draw(ctx)` failed a case-sensitive
rule and became prompt 2's item 2). A change made after seeing a result is
applied to every graded state, and the before/after is reported for all of
them, not just the run that exposed it. Grader v3 (2026-09-26, #62) made the
browser run deterministic and changed three runtime checks; the same applies.

THE STATE GRADED. A grade reads files that may have changed since: a run dir
after prompt 2 is not the state after prompt 1. Each grade id names where its
state comes from:
  fixture   a directory under octo/fixtures that nothing writes to
  run       the run's directory; valid only when nothing wrote to the project
            after that grade (checked: v0e prompt 2 made no project edit; the
            pilot's files predate its grade)
  mirror    v0b's root_mirror (the final state; the run wrote in /root)
  recon     rebuilt from Hermes' own write_file/patch/rm calls
            (overthinking/recon.py), written to octo/fixtures/recon/<id>/.
            The rebuild of each run's FINAL state is checked byte for byte
            against the real files before an intermediate state is trusted.
Each new row carries `regrade` {of, state, verified, old_rows}.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "overthinking"))
import grade as G       # noqa: E402
import run as runmod    # noqa: E402

FIX = os.path.join(runmod.OCTO, "fixtures")
RECON = os.path.join(FIX, "recon")

# (grade_id, variant, state kind, source, log run id or None, log sfx)
PLAN = [
    ("fx-v0-ref", "V0", "fixture", "v0-reference", None, ""),
    # v0f's state after prompt 2, copied from its run dir (SOURCE.json) for
    # the v3 stability measurement (bench/octopus/stability.py)
    ("fx-v0f-p2", "V0", "fixture", "v0f-p2-state", None, ""),
    ("fx-v1-skel", "V1", "fixture", "v1-skeleton", None, ""),
    ("fx-v1-skel-net-before", "V1", "fixture", "v1-skeleton", None, ""),
    ("fx-v1-skel-net-after", "V1", "fixture", "v1-skeleton", None, ""),
    ("fx-v2-skel", "V2", "fixture", "v2-skeleton", None, ""),
    ("fx-v2-skel-gl", "V2", "fixture", "v2-skel-gl", None, ""),
    ("fx-v3-skel", "V3", "fixture", "v3-skeleton", None, ""),
    ("pilot-V0-xhigh-1", "V0", "run", "pilot-V0-xhigh-1", "pilot-V0-xhigh-1", ""),
    ("v0b-snap-0048", "V0", "fixture", "v0b-snap-0048", None, ""),
    ("v0b-final", "V0", "mirror", "v0b-V0-xhigh-1", None, ""),
    # graded while prompt 1 was still writing: rebuilt as of the grade's start
    # (the moment its build copied the tree)
    ("v0e-snap1", "V0", "recon", ("v0e1", "v0e-snap1"), None, ""),
    ("v0e-V0-xhigh-1@p1", "V0", "run", "v0e-V0-xhigh-1", "v0e-V0-xhigh-1", ""),
    ("v0e-V0-xhigh-1@p2", "V0", "run", "v0e-V0-xhigh-1", "v0e-V0-xhigh-1", ".p2"),
    ("v0f-V0-xhigh-1@p1", "V0", "recon", ("v0f1", None), "v0f-V0-xhigh-1", ""),
    ("v0f-V0-xhigh-1@p2", "V0", "run", "v0f-V0-xhigh-1", "v0f-V0-xhigh-1", ".p2"),
]


def old_rows(gid: str) -> list[tuple[int, dict]]:
    return [(i, r) for i, r in enumerate(G.jsonl(G.GRADES)) if r.get("grade_id") == gid]


def _same(state: dict, actual: str) -> list[str]:
    bad = []
    for dp, _dn, fns in os.walk(actual):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, actual).replace(os.sep, "/")
            with open(p, encoding="utf-8", newline="") as f:
                if state.get(rel) != f.read():
                    bad.append(rel)
    return bad + [k for k in state if not os.path.exists(os.path.join(actual, k))]


def materialize(gid: str, kind: str, src) -> tuple[str, dict]:
    """Returns (run_dir to grade, provenance)."""
    if kind == "fixture":
        return os.path.join(FIX, src), {"state": "fixture", "dir": os.path.join(FIX, src)}
    if kind == "run":
        return (os.path.join(runmod.RUNS_DIR, src),
                {"state": "run dir (no project write after this grade)"})
    if kind == "mirror":
        d = os.path.join(runmod.LOGS_DIR, src, "root_mirror", "space-shooter")
        return d, {"state": "root_mirror (final state)"}
    from recon import replay, write_state
    key, at_of = src
    run_id = {"v0e1": "v0e-V0-xhigh-1", "v0f1": "v0f-V0-xhigh-1"}[key]
    actual = os.path.join(runmod.RUNS_DIR, run_id, "space-shooter")
    # 1. prove the rebuild: the whole run (every prompt) must equal the real tree
    full, _ = replay(key)
    k2 = key[:-1] + "2"
    if os.path.exists(os.path.join(runmod.LOGS_DIR, run_id, "hermes.p2.jsonl")):
        full, _ = replay(k2, state=full)
    bad = _same(full, actual)
    if bad:
        raise SystemExit(f"{gid}: rebuild of {run_id}'s final state differs from the real "
                         f"files: {bad[:5]} -- not trusting an intermediate state")
    # 2. the state asked for
    stop = None
    if at_of:
        # the ORIGINAL grade's moment (its first row without `regrade`): a
        # re-grade appends rows, and the last row's time is the re-grade's,
        # which rebuilt v0e-snap1 as the run's final state (found by the v3
        # re-grade: readme_run_3001 flipped on a "fixed" state)
        orig = [r for _i, r in old_rows(at_of) if not r.get("regrade")]
        stop = (orig or [r for _i, r in old_rows(at_of)])[0]["graded_at"]
    state, ev = replay(key, stop_at=stop)
    d = os.path.join(RECON, gid.replace("@", "_"))
    write_state(state, os.path.join(d, "space-shooter"))
    return d, {"state": "reconstructed from Hermes' write_file/patch/rm calls "
                        "(bench/octopus/overthinking/recon.py)",
               "replayed": key, "stop_at": stop,
               "stop_at_local": (datetime.datetime.fromtimestamp(stop).isoformat()
                                 if stop else "end of prompt 1"),
               "events": len(ev), "dir": d,
               "verified": f"the same replay through the run's last prompt equals "
                           f"{actual} byte for byte"}


STATIC_METHODS = ("static-parse", "static-text", "static-parse+text")


def static_compare(ids: list[str]) -> list[dict]:
    out = []
    for gid, variant, kind, src, _log, _sfx in PLAN:
        if ids and gid not in ids:
            continue
        rd, prov = materialize(gid, kind, src)
        proj = G.project_root(rd) or (os.path.join(rd, "space-shooter")
                                      if os.path.isdir(os.path.join(rd, "space-shooter"))
                                      else None)
        b = G.files_and_refs(variant, proj) if proj else {}
        new = {c["check"]: c for c in G.spec(variant, proj, b, {"ran": False, "why": "static"})
               if c["method"] in STATIC_METHODS} if proj else {}
        prev = old_rows(gid)
        rows = []
        for i, r in prev:
            if not r.get("spec"):
                continue
            old = {c["check"]: c.get("pass") for c in r["spec"]
                   if c.get("method") in STATIC_METHODS}
            diff = {k: (old.get(k), new[k]["pass"]) for k in new if old.get(k) != new[k]["pass"]}
            rows.append({"row": i, "graded_at": r.get("graded_at"),
                         "old_passed": r.get("spec_passed"), "old_total": r.get("spec_total"),
                         "static_changed": diff})
        out.append({"grade_id": gid, "state": prov, "old": rows,
                    "new_static": {k: v["pass"] for k, v in new.items()},
                    "particles": new.get("particles_draw_ctx", {}).get("evidence")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--no-runtime", action="store_true")
    a = ap.parse_args()
    if a.static:
        print(json.dumps(static_compare(a.ids), indent=1, default=str))
        return 0
    for gid, variant, kind, src, log, sfx in PLAN:
        if a.ids and gid not in a.ids:
            continue
        rd, prov = materialize(gid, kind, src)
        prov.update({"of": gid, "old_rows": [i for i, _ in old_rows(gid)],
                     "why": f"grader v{G.GRADER_VERSION}: every graded state re-graded (PROTOCOL)"})
        row = G.grade(gid, variant, rd, os.path.join(runmod.LOGS_DIR, log) if log else None,
                      not a.no_runtime, sfx, extra={"regrade": prov})
        print(gid, row.get("spec_passed"), "/", row.get("spec_total"),
              "unknown", row.get("spec_unknown"),
              "failed", [c["check"] for c in row.get("spec") or [] if c.get("pass") is False],
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
