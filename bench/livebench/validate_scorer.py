"""Prove the official judge scores each task's answer FORMAT before trusting it.

For every task in the given categories it writes answers under a throwaway
model name ("scorer-selftest"), runs LiveBench's unmodified
gen_ground_truth_judgment.py on them, checks the scores, and deletes the
throwaway answers and judgments again. No model is called.

  ground-truth task (reasoning, math, language, data_analysis):
      right  = the ground truth in the prompt's requested format -> expect score 1
      wrong  = a different answer in the same format             -> expect score < 1
  instruction_following (no ground truth; IFEval checkers):
      a plain answer must produce a score in [0, 1] with no eval_error
  coding: skipped here (the smoke answers were judged by the real test runner)

Prints "N/M checks passed" and exits 1 on any failure.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import LB_ROOT  # noqa: E402

from livebench.common import LIVE_BENCH_RELEASES, get_categories_tasks, load_questions  # noqa: E402

NAME = "scorer-selftest"
RELEASE = "2024-11-25"


def formatted(q: dict, text: str) -> str:
    prompt = q["turns"][0]
    if "<solution>" in prompt:
        return f"Reasoning omitted.\n\n<solution>{text}</solution>"
    if "**" in prompt:
        return f"Reasoning omitted.\n\n**{text}**"
    return text


def wrong_of(gt: str) -> str:
    parts = [p.strip() for p in re.split(r",", gt)]
    flip = {"yes": "no", "no": "yes", "true": "false", "false": "true"}
    if all(p.lower() in flip for p in parts):
        return ", ".join(flip[p.lower()] for p in parts)
    if re.fullmatch(r"-?\d+", gt.strip()):
        return str(int(gt.strip()) + 7)
    return "zzzz"


def main() -> int:
    cats_wanted = sys.argv[1:] or ["reasoning", "instruction_following"]
    cats, tasks = get_categories_tasks("live_bench")
    rel = {r for r in LIVE_BENCH_RELEASES if r <= RELEASE}
    plan = []  # (category, task, qid, kind)
    files = {}
    for c in cats_wanted:
        for t in sorted(tasks[c]):
            qs = load_questions(cats[c], rel, RELEASE, t)[:2]
            path = os.path.join(LB_ROOT, "data", "live_bench", c, t, "model_answer", f"{NAME}.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            rows = []
            for i, q in enumerate(qs):
                if c == "instruction_following":
                    ans, kind = "This is a short plain answer written for a scorer self-test.", "if"
                else:
                    gt = str(q["ground_truth"])
                    kind = "right" if i == 0 else "wrong"
                    ans = formatted(q, gt if kind == "right" else wrong_of(gt))
                rows.append({"question_id": q["question_id"], "answer_id": f"selftest{i}", "model_id": NAME,
                             "choices": [{"index": 0, "turns": [ans]}], "tstamp": 0})
                plan.append((c, t, q["question_id"], kind))
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            files[(c, t)] = path

    ids = [p[2] for p in plan]
    cmd = [sys.executable, "gen_ground_truth_judgment.py", "--bench-name", "live_bench", "--model", NAME,
           "--livebench-release-option", RELEASE, "--question-id", *ids, "--ignore-missing-answers"]
    r = subprocess.run(cmd, cwd=LB_ROOT, capture_output=True, text=True)
    checks = [("judge exits 0", r.returncode == 0)]
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])

    got = {}
    for (c, t), path in files.items():
        jp = os.path.join(LB_ROOT, "data", "live_bench", c, t, "model_judgment", "ground_truth_judgment.jsonl")
        keep = []
        if os.path.exists(jp):
            for line in open(jp):
                j = json.loads(line)
                if j.get("model") == NAME:
                    got[j["question_id"]] = j
                else:
                    keep.append(line)
            with open(jp, "w") as f:
                f.writelines(keep)  # remove the throwaway judgments
        os.remove(path)  # and the throwaway answers

    for c, t, qid, kind in plan:
        j = got.get(qid)
        name = f"{c}/{t} {kind} ({qid[:8]})"
        if j is None:
            checks.append((f"{name}: judged", False))
            continue
        s = j.get("score")
        if j.get("eval_status") == "eval_error":
            checks.append((f"{name}: no eval_error ({j.get('error_msg', '')[:80]})", False))
        elif kind == "right":
            checks.append((f"{name}: scores 1 (got {s})", s == 1))
        elif kind == "wrong":
            checks.append((f"{name}: scores < 1 (got {s})", s is not None and s < 1))
        else:
            checks.append((f"{name}: score in [0,1] (got {s})", s is not None and 0 <= s <= 1))
    for name, ok in checks:
        print(("ok   " if ok else "FAIL ") + name)
    passed = sum(ok for _, ok in checks)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
