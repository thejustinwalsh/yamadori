#!/usr/bin/env python
"""OUR SKILL-INJECTION CASES, as flat rows for the trained decision model, with jjava's recorded belief beside them.

    python bench/decider/unsloth/skills_prepare.py --cases C:/Users/jwals/octo/codex-label-kit/cases.jsonl \
        --out index/decider_train/skills_items.jsonl          (stack python; offline; no GPU)

One row per labelled (case, item): the state text, the item's fact (and its craft name), the RUBRIC truth level
(0 OFF, 1 AREA, 2 DONE, 3 NEEDED; bench/skills/inject_labels.truth("rubric"), the diagnosis' truth) and jjava's recorded
level distribution from bench/skills/inject/results/run_<model>.jsonl (variant a = as served, 2026-10-06 run,
reader bonsai-a4000). The two questions asked of the trained model about each item are in eval_skills.py:
RELEVANCE (is this fact about what the state is doing) and NEED (does the assistant need this fact for its very next
action). jjava's matching reads: relevance = 1 - P(level 0), need = P(level 3).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
sys.path.insert(0, os.path.join(ROOT, "mcp"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", default="bonsai-a4000")
    ap.add_argument("--variant", default="a")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import inject_labels as IL
    truth = IL.truth("rubric")
    cases = {}
    for ln in open(a.cases, encoding="utf-8"):
        if ln.strip():
            c = json.loads(ln)
            cases[c["case"]] = c
    runs_path = os.path.join(ROOT, "bench", "skills", "inject", "results", f"run_{a.model}.jsonl")
    runs = [json.loads(ln) for ln in open(runs_path, encoding="utf-8") if ln.strip()]
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in runs:
            if r["variant"] != a.variant:
                continue
            c = cases.get(r["case"])
            if c is None:
                continue
            by = {it["key"]: it for it in c["items"]}
            for it in r["items"]:
                lv = truth.get((r["case"], it["key"]))
                if lv is None:
                    continue
                ci = by.get(it["key"]) or {}
                f.write(json.dumps({"case": r["case"], "key": it["key"], "kind": r["kind"], "group": r["group"],
                                    "state_sha": c["state_sha"], "state": c["state"], "task": c.get("task"),
                                    "skill": ci.get("skill"), "fact": ci.get("fact"), "level": lv,
                                    "jjava_probs": it.get("probs") or {}}, ensure_ascii=False) + "\n")
                n += 1
    print("rows", n, "labelled (case,item) pairs in truth", len(truth), "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
