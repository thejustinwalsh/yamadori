"""Run inside the LiveBench WSL venv: coding questions per release and task,
so the held-out pool can be counted. Reads the cached HF dataset; no model."""
import json, os
from livebench.common import LIVE_BENCH_RELEASES, get_categories_tasks, load_questions
cats, _ = get_categories_tasks("live_bench")
out = {}
for rel in sorted(LIVE_BENCH_RELEASES):
    rels = {r for r in LIVE_BENCH_RELEASES if r <= rel}
    row = {}
    for task in ("LCB_generation", "coding_completion"):
        try:
            qs = load_questions(cats["coding"], rels, rel, task)
        except Exception as e:  # noqa: BLE001
            row[task] = f"error {type(e).__name__}"
            continue
        row[task] = {"n": len(qs), "ids": [q["question_id"] for q in qs],
                     "titles": [q.get("question_title") for q in qs],
                     "release_dates": sorted({str(q.get("livebench_release_date")) for q in qs})}
    out[rel] = row
    print(rel, {t: (v["n"] if isinstance(v, dict) else v) for t, v in row.items()})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lb_releases_coding.json"), "w"))
