#!/usr/bin/env python
"""THE DAILY-WORK SELECTION GATE (operator, 2026-09-27: "evaluate whether
this model is useful in daily work (not only the pmndrs stack)").

    python bench/skills/test_daily_eval.py      -> "N/M checks passed"

bench/skills/daily_eval.jsonl -- everyday requests across the operator's
work, labelled from each request's intent -- replayed through the per-turn
engine (skill_select.decide) against a COPY of the live skill store,
deterministic stages only (bench/skills/replay_selection.py --daily; the
embedder and the fallback model are never run here). Every MUST, MUST-NOT
and NOTHING label is a check; any miss fails the suite. Per-area precision
and recall are printed beside the count.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]
# The gate runs the EVIDENCE decider (skill_deciders.EvidenceStub): the
# questions are asked and answered from the deterministic ranking, so the
# gate does not move with the machine's YAMADORI_SKILL_DECIDER.
os.environ["YAMADORI_SKILL_DECIDER"] = "stub"

import replay_selection as R  # noqa: E402  (copies the store to a temp dir)


def main() -> int:
    pool = R.skills.armed()
    if not pool:
        print("no armed skills in the store: the daily eval cannot run")
        print("0/1 checks passed")
        return 1
    print(f"pool: {len(pool)} armed skills (a copy of the live store)")
    return R.print_daily(R.daily(pool), verbose=False)


if __name__ == "__main__":
    sys.exit(main())
