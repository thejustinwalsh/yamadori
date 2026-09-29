#!/usr/bin/env python
"""TIER 0: the RELEASED CLM, as-is, as the skill selector's decider on the
daily eval -- against the deterministic selector on the same cases.
(docs/CLM.md "Tier 0".)

    python bench/clm/tier0.py [--heads PATH] [--tag zero_shot]

What runs:
  * the selector exactly as bench/skills/replay_selection.py --daily runs it
    (its rounds build the questions -- skill_deciders.Question: a state and
    each surviving skill's trigger condition plus NONE; OFFLINE: no
    embedding stage, no fallback model), once with the deterministic stub
    as the first decider (the baseline; its checks were tuned on this very
    eval, so the comparison is against an IN-SAMPLE baseline), once with
    `clm` first (skill_deciders.ClmDecider -> mcp/clm.py);
  * CLM's encoder is a llama-server with config.yaml's `clm-encoder` flags
    and the Q8_0 GGUF, started here on the A4000 under gpu_room's room lock
    (bench/clm/standalone.serving): the running llama-swap has no
    `clm-encoder` until it restarts. YAMADORI_CLM_URL points the client at
    it; option vectors go to a scratch cache, not index/clm_actions.npz.

Written: bench/clm/results/tier0_<tag>.json -- per case the checks of both
deciders, CLM's questions (area, option skill names, probabilities, pick,
which decider answered), and every CLM call's timing. Nothing of a `source`
row's text (test material) is stored or printed; the user rows' text is the
eval file's own.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = "C:/Users/jwals/textgen/user_data/models"
PORT = 10992


def _kind(what: str) -> str:
    return ("MUST-NOT" if "MUST NOT" in what else "MUST" if "MUST inject"
            in what else "NOTHING" if "NOTHING" in what else "form/recall")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="")
    ap.add_argument("--tag", default="zero_shot")
    ap.add_argument("--gguf", default=os.path.join(MODELS, "Qwen3-8B-Q8_0.gguf"))
    a = ap.parse_args(argv)
    os.environ["YAMADORI_CLM_URL"] = f"http://127.0.0.1:{PORT}"
    os.environ["YAMADORI_CLM_ACTIONS"] = os.path.join(
        tempfile.mkdtemp(prefix="clm_tier0_"), "actions.npz")
    if a.heads:
        os.environ["YAMADORI_CLM_HEADS"] = a.heads
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    sys.path.insert(0, HERE)
    import replay_selection as rs        # sets its own offline env first
    import skills
    import clm
    import standalone

    calls: list[dict] = []
    real = clm.decide_detail_many

    def timed(state, option_lists, **kw):
        d = real(state, option_lists, **kw)
        calls.append({"questions": len(option_lists),
                      "options": sum(len(o) for o in option_lists),
                      "state_tokens": d["state"]["tokens"],
                      "truncated": d["state"]["truncated"],
                      **d["timing"], "encoded_options":
                          d["options"]["encoded"]})
        return d
    clm.decide_detail_many = timed

    pool = skills.armed()
    out: dict = {"pool": len(pool), "heads": None, "runs": {}}
    os.environ["YAMADORI_SKILL_DECIDER"] = "stub"
    out["runs"]["stub"] = rs.daily(pool)
    with standalone.serving(a.gguf, PORT, tag=f"tier0_{a.tag}") as (base, rec):
        out["server"] = {k: v for k, v in rec.items() if k != "argv"}
        os.environ["YAMADORI_SKILL_DECIDER"] = "clm"
        t0 = time.time()
        out["runs"]["clm"] = rs.daily(pool)
        out["clm_seconds"] = round(time.time() - t0, 1)
    import clm_heads
    out["heads"] = clm_heads.load().namespace
    out["calls"] = calls

    # Per-case comparison and totals.
    table = {}
    for dec, res in out["runs"].items():
        k, ok = collections.Counter(), collections.Counter()
        for good, what in res["checks"]:
            k[_kind(what)] += 1
            ok[_kind(what)] += bool(good)
        table[dec] = {kk: f"{ok[kk]}/{k[kk]}" for kk in k}
        table[dec]["all"] = f"{sum(ok.values())}/{sum(k.values())}"
    out["totals"] = table
    fails = {dec: [w for g, w in res["checks"] if not g]
             for dec, res in out["runs"].items()}
    out["fails"] = fails
    by_id = {r["id"]: r for r in out["runs"]["clm"]["requests"]}
    answered = collections.Counter(q["decider"] for r in by_id.values()
                                   for q in r["questions"])
    out["clm_questions_answered_by"] = dict(answered)
    for dec in out["runs"]:
        out["runs"][dec]["checks"] = [[bool(g), w] for g, w in
                                      out["runs"][dec]["checks"]]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"tier0_{a.tag}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"totals": table, "answered_by": dict(answered),
                      "clm_seconds": out["clm_seconds"],
                      "server": out["server"], "heads": out["heads"],
                      "calls": len(calls)}, indent=1))
    ms = sorted(c["state_encode_ms"] for c in calls)
    if ms:
        print(f"CLM calls: {len(ms)}; state encode ms p50 "
              f"{ms[len(ms) // 2]:.1f}, max {ms[-1]:.1f}")
    for dec, fl in fails.items():
        print(f"\n{dec}: {len(fl)} failed checks")
        for w in fl:
            print(f"  FAIL {w}")
    print(f"\nwritten {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
