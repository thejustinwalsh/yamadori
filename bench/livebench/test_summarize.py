"""Offline checks for summarize.py. Prints "N/M checks passed"; exits 1 on any failure.

    python bench/livebench/test_summarize.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize as S  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))
    if not ok:
        print(f"FAIL: {name}")


def row(arm, cat, task, i, score, status="ok", fr="stop"):
    return {"id": f"{cat}-{task}-{i}", "order": i, "category": cat, "task": task, "arm": arm,
            "score": score, "status": status, "finish_reason": fr,
            "tokens": {"output": 100 + i}, "seconds": 10.0 + i}


def write(run_dir, arm, cat, rows):
    with open(os.path.join(run_dir, f"rows_{arm}_{cat}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


with tempfile.TemporaryDirectory() as d:
    # coding: task A 3/4 correct, task B 1/2 -> category = mean(75, 50) = 62.5 (NOT pooled 4/6=66.7)
    bon = [row("bonsai", "coding", "A", i, s) for i, s in enumerate([1, 1, 1, 0])]
    bon += [row("bonsai", "coding", "B", i, s) for i, s in enumerate([1, 0])]
    # reasoning: one task, fractional scores, plus a length finish (scores 0) and a transport failure (excluded)
    bon += [row("bonsai", "reasoning", "R", 0, 0.5), row("bonsai", "reasoning", "R", 1, 1.0),
            row("bonsai", "reasoning", "R", 2, 0, status="budget_event", fr="length"),
            row("bonsai", "reasoning", "R", 3, None, status="not_run", fr=None),
            row("bonsai", "reasoning", "R", 4, 0, status="eval_error")]
    write(d, "bonsai", "coding", [r for r in bon if r["category"] == "coding"])
    write(d, "bonsai", "reasoning", [r for r in bon if r["category"] == "reasoning"])
    # yamadori arm: coding only, all right on A, B unchanged; one extra unpaired question
    yam = [row("yamadori", "coding", "A", i, 1) for i in range(4)]
    yam += [row("yamadori", "coding", "B", i, s) for i, s in enumerate([1, 0])]
    yam += [row("yamadori", "coding", "B", 9, 1)]
    write(d, "yamadori", "coding", yam)
    with open(os.path.join(d, "order_coding.json"), "w") as f:
        json.dump({"release": "2024-11-25", "seed": 1, "category": "coding", "order": [{}] * 128}, f)

    s = S.build(d, B=400)
    b = s["arms"]["bonsai"]
    check("task mean is x100 of question mean", abs(b["categories"]["coding"]["tasks"]["A"]["score"] - 75.0) < 1e-9)
    check("category = unweighted mean of task means (62.5, not pooled 66.7)",
          abs(b["categories"]["coding"]["score"] - 62.5) < 1e-9)
    # reasoning scored rows: 0.5, 1.0, 0 (length) -> 50.0; not_run and eval_error excluded
    check("length finish scores 0 and counts", abs(b["categories"]["reasoning"]["score"] - 50.0) < 1e-9
          and b["categories"]["reasoning"]["n"] == 3)
    check("not_run / eval_error excluded but counted",
          b["status"].get("not_run") == 1 and b["status"].get("eval_error") == 1)
    check("overall = mean of category scores", abs(b["overall"]["score"] - (62.5 + 50.0) / 2) < 1e-9)
    check("categories_included lists what was run", b["overall"]["categories_included"] == ["coding", "reasoning"])
    lo, hi = b["categories"]["coding"]["ci95"]
    check("CI brackets the point estimate", lo <= 62.5 <= hi and lo < hi)
    y = s["arms"]["yamadori"]
    lo, hi = y["categories"]["coding"]["tasks"]["A"]["ci95"]
    check("zero-variance task has a zero-width CI", lo == hi == 100.0)
    check("finish_reason distribution counts every row",
          sum(b["finish_reason"].values()) == len(bon) and b["finish_reason"].get("length") == 1)
    check("population recorded from the order file", s["population"] == {"coding": 128})
    p = s["paired_yamadori_minus_bonsai"]
    # paired: only the 6 common coding ids; A: 75 -> 100, B: 50 -> 50 => +12.5
    check("paired uses only common questions", p["categories"]["coding"]["n_pairs"] == 6)
    check("paired diff = category(b) - category(a) on the common set", abs(p["categories"]["coding"]["diff"] - 12.5) < 1e-9)
    check("paired has no reasoning (yamadori did not run it)", p["categories_included"] == ["coding"])
    check("discordant counts", p["categories"]["coding"]["b_only_correct"] == 1
          and p["categories"]["coding"]["a_only_correct"] == 0)
    # a third arm is paired against the baseline too, on its own common questions
    write(d, "minimal", "coding", [row("minimal", "coding", "A", i, 0) for i in range(2)])
    s3 = S.build(d, B=200)
    pm = s3.get("paired_minimal_minus_bonsai")
    check("every non-baseline arm gets its own paired block", pm is not None and pm["n_pairs"] == 2
          and "paired_yamadori_minus_bonsai" in s3)
    check("third-arm diff on common set (A: 100 -> 0 on the 2 shared)", pm is not None
          and abs(pm["categories"]["coding"]["diff"] - (-100.0)) < 1e-9)
    check("arm labels present for known arms", "bare @ medium" in s3["arm_labels"]["bonsai"]
          and "tier max" in s3["arm_labels"]["yamadori"])
    os.remove(os.path.join(d, "rows_minimal_coding.jsonl"))
    # an arm spanning two conditions is reported combined AND per condition
    yam_c = [dict(r, condition=("pre" if i < 3 else "cap10")) for i, r in enumerate(yam)]
    write(d, "yamadori", "coding", yam_c)
    s4 = S.build(d, B=200)
    check("condition split: per-condition arms present", "yamadori@pre" in s4["arms"] and "yamadori@cap10" in s4["arms"])
    check("condition split: combined arm unchanged", s4["arms"]["yamadori"]["categories"]["coding"]["n"] == 7)
    check("condition split: each paired against bonsai", "paired_yamadori@pre_minus_bonsai" in s4
          and s4["paired_yamadori@pre_minus_bonsai"]["n_pairs"] == 3)
    write(d, "yamadori", "coding", yam)
    s2 = S.build(d, B=400)
    check("deterministic under the fixed seed",
          s2["arms"]["bonsai"]["categories"]["coding"]["ci95"] == b["categories"]["coding"]["ci95"])

# exact McNemar against hand values: b=1,c=0 -> p=1.0; b=6,c=0 -> 2*(1/64)=0.03125; b=5,c=5 -> 1.0
check("mcnemar b=1 c=0", S.mcnemar_exact(1, 0) == 1.0)
check("mcnemar b=6 c=0", abs(S.mcnemar_exact(6, 0) - 0.03125) < 1e-12)
check("mcnemar symmetric/balanced", S.mcnemar_exact(5, 5) == 1.0 and S.mcnemar_exact(0, 0) is None)
check("percentile interpolates", S.percentile([0, 10], 0.5) == 5.0)

passed = sum(ok for _, ok in checks)
print(f"{passed}/{len(checks)} checks passed")
sys.exit(0 if passed == len(checks) else 1)
