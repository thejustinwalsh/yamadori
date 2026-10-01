"""The three SWE-bench arms, and the constants every other file here shares.

Standard library only: this module is imported on BOTH sides -- by run.py on
Windows (the stack interpreter) and by wsl_side.py in the WSL venv.

THE ARMS (names are the operator's, 2026-09-22)

  bonsai          everything forced off: the bare model, as it ships.
  yamadori-auto   nothing forced: the tier decides what fires (the header
                  carries only the pinned effort).
  yamadori        everything forced on: skills, the concept seed and the MCP
                  host's tools. The headline row next to the other models.
                  Until 2026-09-29 it also forced retrieval over the HELD
                  package indexes, deep thinking and fan-out N=3; those were
                  removed from the proxy (docs/REMOVED.md), and results
                  recorded before then keep what they recorded.

The header values are bench/domain/run.py's A0, A5 and A6, byte for byte in
meaning. As there, every arm also sends `reasoning_effort: "max"` in the BODY
(through litellm's `extra_body`) so the tier ALLOWS everything and the header
alone decides; `effort` in the header pins the thinking style to `medium` on
every arm, so effort is held fixed across arms (bench/domain/run.py, "THE
ARMS").

"yamadori" does NOT stage, index or bind the SWE-bench repository on the
host. The caller's source reaches the model only through the caller's own
harness tools -- here, the bash commands mini-swe-agent runs in the container
(proxy.resolve_repo returns None by contract). What the server adds is what
it holds: skills, the concept seed, the MCP host's package lookups.
"""
from __future__ import annotations

import json

ARMS: dict[str, dict] = {
    "bonsai": {"skills": False, "seed": False, "mcp_tools": False,
               "effort": "medium"},
    "yamadori-auto": {"effort": "medium"},
    "yamadori": {"skills": True, "seed": True, "mcp_tools": True,
                 "effort": "medium"},
}
ARM_NOTES = {
    "bonsai": "everything off: the bare model (bench/domain A0)",
    "yamadori-auto": "nothing forced, the tier decides (A5)",
    "yamadori": "everything on: skills, concept seed, MCP tools (A6)",
}
# Sent in the body on every arm; see the module docstring.
BODY_EFFORT = "max"

MODEL_NAME = "openai/yamadori"      # litellm: OpenAI-compatible provider
PUBLIC_MODEL = "yamadori"           # what the proxy advertises

# The benchmark definition, pinned. docs/SWE-BENCH.md says where each comes from.
MINI_VERSION = "2.1.0"              # what the leaderboard's v2 trajectories ran
SWEBENCH_VERSION = "4.1.0"          # harness in force for those evaluations
DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",       # 500, test split
    "verified-mini": "MariusHobbhahn/swe-bench-verified-mini",  # 50, test split
    "multilingual": "SWE-bench/SWE-bench_Multilingual",   # 300, test split
}

# What the harness grades against. Verified Mini's rows are byte-identical to
# Verified's on base_commit, patch, FAIL_TO_PASS and PASS_TO_PASS (checked
# 2026-09-22, all 50), so it is graded against the canonical dataset.
EVAL_DATASETS = {"verified": DATASETS["verified"],
                 "verified-mini": DATASETS["verified"],
                 "multilingual": DATASETS["multilingual"]}

# Pilot: random.Random(20260922).sample(sorted(Verified ids), 5).
PILOT = ["sympy__sympy-17655", "scikit-learn__scikit-learn-14629",
         "pytest-dev__pytest-7490", "scikit-learn__scikit-learn-15100",
         "django__django-15277"]


def header(arm: str) -> str:
    """The X-Yamadori-Features value for an arm (KeyError on an unknown arm)."""
    return json.dumps(ARMS[arm], sort_keys=True)
