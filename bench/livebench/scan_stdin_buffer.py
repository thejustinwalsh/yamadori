"""Sensitivity check for a LiveBench grader quirk: `sys.stdin.buffer` / `sys.stdout.buffer`.

(Both streams: the grader's stdin AND its captured stdout are io.StringIO
objects without `.buffer`. The shim below gives each one a `.buffer`.)

LiveBench's LCB grader (lcb_runner, inherited from LiveCodeBench) replaces
sys.stdin with an io.StringIO, which has no `.buffer`. A stdin program that
reads `sys.stdin.buffer` -- valid Python, a common competitive-programming
idiom -- dies with AttributeError on the first test and scores 0 (error_code
-4, "Runtime Error"). Found 2026-09-23 on e1e226d2 (atcoder Loong Tracking):
the official judge scores 0, the same program passes 13/13 tests on a real
stdin.

The OFFICIAL score is what we report: every leaderboard model was graded by
the same code, so changing it would break comparability. This script only
measures how much the quirk moves OUR numbers: for every scored coding answer
that reads sys.stdin.buffer, it re-grades the unchanged program with a
3-line shim prepended that gives the grader's StringIO a `.buffer` holding
the same input as bytes, and reports both scores.

    python scan_stdin_buffer.py results/<run-id>
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import ARMS, answer_path, read_rows  # noqa: E402

from livebench.common import LIVE_BENCH_RELEASES, get_categories_tasks, load_questions  # noqa: E402
from livebench.process_results.coding.utils import LCB_generation_process_results  # noqa: E402

# any spelling that reaches .buffer on stdin: sys.stdin.buffer, `from sys import stdin` + stdin.buffer,
# and aliases such as `import sys as s` -> s.stdin.buffer. (The first version matched only sys.stdin.buffer
# and missed xhigh df00a468; found 2026-09-23 13:40 by re-grading every 0 on a real stdin.)
# stdout too: the grader captures stdout in a StringIO without `.buffer`, so
# sys.stdout.buffer.write(...) fails the same way (xhigh df00a468).
PAT = re.compile(r"\b(?:stdin|stdout)\s*\.\s*buffer\b")


SHIM = ("import sys as _lb_sys, io as _lb_io\n"
        "if not hasattr(_lb_sys.stdin, 'buffer'):\n"
        "    _lb_sys.stdin.buffer = _lb_io.BytesIO(_lb_sys.stdin.read().encode())\n"
        "class _LbOut:\n"
        "    def __init__(self, t): self.t = t\n"
        "    def write(self, b):\n"
        "        self.t.write(bytes(b).decode() if isinstance(b, (bytes, bytearray, memoryview)) else b); return len(b)\n"
        "    def flush(self): self.t.flush()\n"
        "if not hasattr(_lb_sys.stdout, 'buffer'):\n"
        "    _lb_sys.stdout.buffer = _LbOut(_lb_sys.stdout)\n")


def rewrite(answer: str) -> str:
    """Give the grader's StringIO stdin a real `.buffer` (the bytes of the same
    input), leaving the program itself untouched. A text-mode rewrite is NOT
    equivalent: programs compare tokens to bytes literals (b'1')."""
    m = re.search(r"```(?:python|py)?\n", answer)
    if not m:
        return answer
    return answer[:m.end()] + SHIM + answer[m.end():]


def main() -> int:
    run_dir = sys.argv[1]
    rows = {}
    for p in [f for f in os.listdir(run_dir) if f.startswith("rows_") and f.endswith("_coding.jsonl")]:
        for line in open(os.path.join(run_dir, p)):
            r = json.loads(line)
            rows[(r["arm"], r["id"])] = r
    cats, _ = get_categories_tasks("live_bench")
    rel = {r for r in LIVE_BENCH_RELEASES if r <= "2024-11-25"}
    qs = {}
    for task in ("LCB_generation", "coding_completion"):
        for q in load_questions(cats["coding"], rel, "2024-11-25", task):
            qs[q["question_id"]] = q
    out = []
    for (arm, qid), r in sorted(rows.items()):
        if r["status"] not in ("ok", "budget_event") or arm not in ARMS:
            continue
        ans = next((a for a in read_rows(answer_path("coding", r["task"], ARMS[arm]["display"]))
                    if a["question_id"] == qid), None)
        if ans is None:
            continue
        text = ans["choices"][0]["turns"][-1]
        if not PAT.search(text):
            continue
        alt = LCB_generation_process_results(qs[qid], rewrite(text))
        out.append({"arm": arm, "id": qid, "task": r["task"], "official": r["score"], "with_stdin_buffer": alt})
        print(f"{arm:14s} {qid[:8]} {r['task']:18s} official={r['score']} with stdin.buffer provided={alt}")
    path = os.path.join(run_dir, "stdin_buffer_sensitivity.json")
    json.dump({"note": __doc__.split("\n\n")[1], "rows": out}, open(path, "w"), indent=1)
    flips = sum(1 for x in out if x["official"] != x["with_stdin_buffer"])
    print(f"{len(out)} scored coding answers read sys.stdin.buffer; {flips} change score when the grader "
          f"provides stdin.buffer -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
