"""Score answered questions with LiveBench's OFFICIAL judge, then export rows.

    python score.py --run-dir results/<run-id> --arm bonsai --category coding

1. Runs livebench/gen_ground_truth_judgment.py (unmodified) on exactly the
   question ids this arm has answered in this category (re-judging replaces
   the previous judgment row; LiveBench de-duplicates on (question, model)).
2. Joins answer + judgment rows into results/<run-id>/rows_<arm>_<category>.jsonl,
   one line per question: id, category, task, arm, score, finish_reason,
   tokens, seconds, status. `status` separates what LiveBench lumps together:
     ok            -- answered, scored by the judge
     budget_event  -- finish_reason=length (LiveBench scores it 0; so do we)
     not_run       -- transport failure after all retries (429/5xx/timeout);
                      LiveBench would score it 0, we report it separately
     eval_error    -- the judge itself raised (never a model failure)
     stack_error   -- a mechanism broke (mechanisms.py); not scored, re-asked
Each row also carries `mechanisms`, the per-answer record from x_yamadori.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive import ARMS, LB_ROOT, answer_path, is_budget_event, is_error, is_incomplete, read_rows  # noqa: E402
import mechanisms  # noqa: E402


def judgment_path(category: str, task: str) -> str:
    return os.path.join(LB_ROOT, "data", "live_bench", category, task, "model_judgment", "ground_truth_judgment.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--skip-judge", action="store_true")
    a = ap.parse_args()

    # A finished run is frozen: its answers have been archived out of the
    # LiveBench data dir, and the files now there belong to a later run that
    # uses the same display names. Re-scoring would write that run's answers
    # into this one's rows.
    if os.path.exists(os.path.join(a.run_dir, "FROZEN")):
        print(f"{a.run_dir} is FROZEN ({open(os.path.join(a.run_dir, 'FROZEN')).read().strip()}); not re-scoring")
        return 0
    spec = json.load(open(os.path.join(a.run_dir, f"order_{a.category}.json")))
    order = spec["order"]
    tasks = sorted({r["task"] for r in order})
    display = ARMS[a.arm]["display"]
    answers = {}
    for t in tasks:
        for r in read_rows(answer_path(a.category, t, display)):
            answers[r["question_id"]] = (t, r)
    if not answers:
        print("no answers yet")
        return 1

    if not a.skip_judge:
        ids = sorted(answers)
        cmd = [sys.executable, "gen_ground_truth_judgment.py", "--bench-name", "live_bench",
               "--model", display, "--livebench-release-option", spec["release"],
               "--question-id", *ids, "--ignore-missing-answers", "--parallel", "8"]
        log = os.path.join(a.run_dir, "logs", f"judge_{a.arm}_{a.category}.log")
        with open(log, "a") as lf:
            rc = subprocess.run(cmd, cwd=LB_ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"judge exited {rc}; see {log}")
            return rc

    judg = {}
    for t in tasks:
        for j in read_rows(judgment_path(a.category, t)):
            if j.get("model") == display:
                prev = judg.get(j["question_id"])
                if prev is None or j.get("tstamp", 0) >= prev.get("tstamp", 0):
                    judg[j["question_id"]] = j

    # Timing label per answer: which concurrency condition its request STARTED
    # under (condition.json timing_epochs; the latest boundary <= start wins).
    # Correctness is unaffected by concurrency; `seconds` is not comparable
    # across labels.
    epochs = []
    cond_p = os.path.join(a.run_dir, "condition.json")
    if os.path.exists(cond_p):
        epochs = sorted(json.load(open(cond_p)).get("timing_epochs") or [], key=lambda e: e["since_epoch"])

    cond_epochs = []
    if os.path.exists(cond_p):
        cond_epochs = sorted(json.load(open(cond_p)).get("condition_epochs") or [], key=lambda e: e["since_epoch"])

    def label_at(r: dict, eps: list) -> str | None:
        start = (r.get("tstamp") or 0) - (r.get("total_time_s") or 0)
        lab = None
        for e in eps:
            if start >= e["since_epoch"]:
                lab = e["label"]
        return lab

    def timing_of(r: dict) -> str | None:
        return label_at(r, epochs)

    out = os.path.join(a.run_dir, f"rows_{a.arm}_{a.category}.jsonl")
    n = 0
    with open(out, "w") as f:
        for pos, o in enumerate(order):
            qid = o["question_id"]
            if qid not in answers:
                continue
            task, r = answers[qid]
            ai = r.get("api_info") or {}
            j = judg.get(qid)
            mech = mechanisms.record(ai.get("x_yamadori")) if not is_error(r) else None
            if is_error(r):
                status = "budget_event" if is_budget_event(r) else "not_run"
            elif (mech and mech.get("stack_error")) or is_incomplete(r):
                status = "stack_error"  # a mechanism broke or the upstream dropped: not scored, re-asked by drive.py
            elif j is None:
                status = "unjudged"
            elif j.get("eval_status") == "eval_error":
                status = "eval_error"
            else:
                status = "ok"
            fr = ai.get("finish_reason")
            if status == "budget_event":
                fr = "length"
            xy = ai.get("x_yamadori") or {}
            row = {
                "id": qid, "order": pos, "category": a.category, "task": task, "arm": a.arm,
                "score": (j or {}).get("score"), "status": status, "finish_reason": fr,
                "tokens": {"output": r.get("total_output_tokens"), "input": r.get("total_input_tokens"),
                           "reasoning_chars": len(((r["choices"][0].get("reasoning") or [""])[-1]) or ""),
                           "answer_chars": len(r["choices"][0]["turns"][-1] or "")},
                "seconds": r.get("total_time_s"),
                "timing": timing_of(r),
                # the stack condition the request STARTED under (condition.json
                # condition_epochs); an arm spanning two is reported per condition
                "condition": label_at(r, cond_epochs),
                "budget": xy.get("budget"),
                "selection": {k: (xy.get("selection") or {}).get(k) for k in ("hints", "investigate", "fanout_n")},
                "investigate_ran": (xy.get("investigate") or {}).get("ran"),
                "fanout_winner": (xy.get("fanout") or {}).get("winner"),
                "error_msg": (r.get("error_msg") or (j or {}).get("error_msg") or "")[:200],
                "mechanisms": mech,
                "retries_429": ai.get("retries_429"),
            }
            f.write(json.dumps(row) + "\n")
            n += 1
    print(f"wrote {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
