"""Run inside the LiveBench WSL venv. Dumps (a) every coding question of
release 2024-11-25 (all tasks, all releases <= it, as LiveBench selects them),
and (b) the bonsai and tier-xhigh answers, to data/. No model call."""
import json, os
from livebench.common import LIVE_BENCH_RELEASES, get_categories_tasks, load_questions

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LB = os.path.expanduser("~/livebench-run/LiveBench/livebench/data/live_bench/coding")
cats, tasks = get_categories_tasks("live_bench")
rel = {r for r in LIVE_BENCH_RELEASES if r <= "2024-11-25"}
qs = []
for task in ("LCB_generation", "coding_completion"):
    for q in load_questions(cats["coding"], rel, "2024-11-25", task):
        q = dict(q)
        qs.append({k: (v if isinstance(v, (str, int, float, list, dict, type(None), bool)) else str(v)) for k, v in q.items()})
json.dump(qs, open(os.path.join(OUT, "lb_coding_2024-11-25_questions.json"), "w"))
print("questions", len(qs), {t: sum(1 for q in qs if q["task"] == t) for t in ("LCB_generation", "coding_completion")})
for disp in ("yamadori-bonsai-arm", "yamadori-tier-xhigh"):
    rows = []
    for task in ("LCB_generation", "coding_completion"):
        p = os.path.join(LB, task, "model_answer", disp + ".jsonl")
        if os.path.exists(p):
            for line in open(p):
                if line.strip():
                    r = json.loads(line); r["_task"] = task; rows.append(r)
    json.dump(rows, open(os.path.join(OUT, f"answers_{disp}.json"), "w"))
    print(disp, len(rows))
