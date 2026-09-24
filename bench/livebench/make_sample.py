"""Fix the question ORDER for a LiveBench run before any answer exists.

Per category: shuffle each task's questions with a fixed seed, then interleave
the tasks round-robin. Every prefix of the order is therefore a stratified
random sample (task shares equal up to one question), so a run cut short by
the clock still reports an unbiased per-task sample, and both arms run on the
same prefix. Written once; the driver only reads it.

    python make_sample.py --release 2024-11-25 --out results/<run-id> coding reasoning ...
"""
from __future__ import annotations

import argparse
import json
import os
import random

from livebench.common import LIVE_BENCH_RELEASES, get_categories_tasks, load_questions


def order_for(category: str, release: str, seed: int) -> list[dict]:
    # "live_bench", never "live_bench/<category>": LiveBench's get_categories_tasks
    # takes split('_')[0] of the category, so instruction_following and
    # data_analysis resolve to nonexistent datasets "instruction" and "data".
    cats, tasks = get_categories_tasks("live_bench")
    releases = {r for r in LIVE_BENCH_RELEASES if r <= release}
    by_task: dict[str, list[str]] = {}
    for task in sorted(tasks[category]):
        qs = load_questions(cats[category], releases, release, task)
        ids = sorted(q["question_id"] for q in qs)
        if ids:
            random.Random(f"{seed}:{category}:{task}").shuffle(ids)
            by_task[task] = ids
    out, i = [], 0
    while any(i < len(v) for v in by_task.values()):
        for task, ids in by_task.items():
            if i < len(ids):
                out.append({"question_id": ids[i], "task": task, "category": category})
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    ap.add_argument("categories", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in a.categories:
        path = os.path.join(a.out, f"order_{c}.json")
        if os.path.exists(path):
            print(f"{path} exists; order is fixed, not regenerated")
            continue
        order = order_for(c, a.release, a.seed)
        with open(path, "w") as f:
            json.dump({"release": a.release, "seed": a.seed, "category": c, "order": order}, f, indent=0)
        counts: dict[str, int] = {}
        for r in order:
            counts[r["task"]] = counts.get(r["task"], 0) + 1
        print(f"{c}: {len(order)} questions {counts}")


if __name__ == "__main__":
    main()
