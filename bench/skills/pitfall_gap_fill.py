#!/usr/bin/env python
"""FILL THE PITFALL HARNESS'S GAPS FOR EVERY AREA PAST REACT, through the
ONE skills pipeline (GPU: its model stages -- distil, review, the
faithfulness check, prove -- run on the loaded model, in a granted window).

    python bench/skills/pitfall_gap_fill.py --list [--area typegpu]   # offline
    python bench/skills/pitfall_gap_fill.py --check                   # offline
    python bench/skills/pitfall_gap_fill.py --run --model bonsai [--area X]

React's gaps keep their own script (bench/skills/react_gap_fill.py); this is
the same procedure for the areas the operator approved on 2026-09-30 ("Yes
to everything pending, queue it up lets go"): r3f v10 + three, TSL,
TypeGPU, koota, pmndrs math, TypeScript, Rust/WASM.

THE GAP LIST, per area: bench/skills/pitfalls/gaps/<area>.json
  {"area", "sources": [{"name": the skill to build (the case files name it),
                        "url": the PINNED raw page (a commit in the URL, the
                               same bytes as the harness's pinned copy),
                        "file": that copy under pitfalls/sources/,
                        "cases": [case ids it serves],
                        "goal": what the skill should steer to -- UNMEASURED
                                WORDING, given to skills.create's `goal`}]}
Each source goes to skills.create(url=..., goal=...) and
skill_pipeline.run_inline(model=True): fetch (robots.txt obeyed), screen,
model screen, licence, distil, review, classify, tests, validate (the
faithfulness check), prove, arm. The item text is the model's, from the
page, checked by the pipeline: never a hand edit (operator).

--check (offline) verifies each entry: the URL names a commit, the pinned
copy exists and its sha256 is the manifest's, and every case it serves
exists. REFUSES --run unless --model is loaded (GET /running). Writes the
live skill store (that is the pipeline's job).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)
GAPS = os.path.join(HERE, "pitfalls", "gaps")
SOURCES = os.path.join(HERE, "pitfalls", "sources")
PINNED_URL = re.compile(r"/[0-9a-f]{40}/")


def load(area: str | None = None) -> list[dict]:
    out = []
    if not os.path.isdir(GAPS):
        return out
    for fn in sorted(os.listdir(GAPS)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(GAPS, fn), encoding="utf-8") as f:
            d = json.load(f)
        if area and d.get("area") != area:
            continue
        for s in d.get("sources") or []:
            out.append(dict(s, area=d.get("area")))
    return out


def _manifest_sha(file: str) -> str | None:
    d = os.path.dirname(file)
    try:
        with open(os.path.join(SOURCES, d, "MANIFEST.json"),
                  encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    for x in m.get("files") or []:
        if x.get("file") == os.path.basename(file):
            return x.get("sha256")
    return None


def check(entries: list[dict]) -> list[dict]:
    import pitfall_harness as H
    ids = {c["id"] for c in H.load_cases()}
    out = []
    for s in entries:
        why = []
        if not PINNED_URL.search(s.get("url") or ""):
            why.append("the URL names no commit")
        path = os.path.join(SOURCES, s.get("file") or "")
        if not s.get("file") or not os.path.isfile(path):
            why.append("no pinned copy")
        else:
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            if sha != _manifest_sha(s["file"]):
                why.append("the pinned copy's sha256 is not the manifest's")
        bad = [c for c in s.get("cases") or [] if c not in ids]
        if bad or not s.get("cases"):
            why.append(f"cases not found: {bad or 'none named'}")
        if not s.get("goal") or not s.get("name"):
            why.append("no name or goal")
        out.append({"name": s.get("name"), "area": s.get("area"),
                    "ok": not why, "why": why})
    return out


def running(upstream: str) -> list[str]:
    with urllib.request.urlopen(f"{upstream}/running", timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = d.get("running") if isinstance(d, dict) else d
    return [str((x or {}).get("model")) for x in rows or []
            if isinstance(x, dict)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--area")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="bonsai")
    ap.add_argument("--retry", metavar="NAME[,NAME]",
                    help="re-run the EXISTING skills of these entries "
                    "(quarantined or failed) through the worker's pipeline "
                    "from --from-stage, with the pinned-source cleaning "
                    "step on (mcp/skill_clean.py); creates nothing")
    ap.add_argument("--from-stage", default="fetch",
                    help="with --retry: the stage to run again from "
                    "(fetch for a screen quarantine; tests for an "
                    "activation-test one)")
    a = ap.parse_args(argv)
    entries = load(a.area)
    if a.retry:
        import skills
        want = {x.strip() for x in a.retry.split(",") if x.strip()}
        bad = [r for r in check(entries) if not r["ok"]
               and r["name"] in want]
        if bad:
            print("REFUSED: gap entries fail --check:", json.dumps(bad))
            return 2
        for s in entries:
            if s["name"] not in want:
                continue
            row = skills.find(s["name"])
            if row is None or row.get("status") == "armed":
                print(json.dumps({"name": s["name"],
                                  "skipped": "no such skill, or armed"}))
                continue
            skills.update_meta(row["id"], fetch_clean={
                "pinned_sha256": _manifest_sha(s["file"])})
            skills.rerun(row["id"], a.from_stage, author="pitfall_gap_fill")
            print(json.dumps({"name": s["name"], "id": row["id"],
                              "rerun_from": a.from_stage}), flush=True)
        return 0
    if a.check:
        res = check(entries)
        print(json.dumps(res, indent=1))
        return 0 if all(r["ok"] for r in res) else 1
    if a.list or not a.run:
        for s in entries:
            print(json.dumps({k: s.get(k) for k in ("area", "name", "url",
                                                    "cases")}))
        return 0
    bad = [r for r in check(entries) if not r["ok"]]
    if bad:
        print("REFUSED: gap entries fail --check:", json.dumps(bad))
        return 2
    import model as M
    have = running(M.UPSTREAM)
    if a.model not in have:
        print(f"REFUSED: {a.model} is not loaded (GET /running: {have})")
        return 2
    import skill_pipeline
    import skills
    for s in entries:
        rec = skills.create(url=s["url"], name=s["name"], goal=s["goal"],
                            author="pitfall_gap_fill",
                            meta={"gap_fill": {"area": s["area"],
                                               "cases": s["cases"],
                                               "pinned": s["file"]},
                                  # the cleaning step, bound to the pinned
                                  # file's sha256 (mcp/skill_clean.py)
                                  "fetch_clean": {"pinned_sha256":
                                                  _manifest_sha(s["file"])}},
                            enqueue_first=False)
        stages = skill_pipeline.run_inline(rec["id"], 1, model=True,
                                           max_steps=30)
        got = skills.get(rec["id"]) or {}
        print(json.dumps({"area": s["area"], "name": s["name"],
                          "id": rec["id"], "status": got.get("status"),
                          "reason": got.get("reason"),
                          "stages": sorted(stages)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
