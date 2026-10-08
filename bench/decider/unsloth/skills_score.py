#!/usr/bin/env python
"""Score eval_skills.py output against the rubric truth, beside jjava's recorded reads (stack python, offline).

    python bench/decider/unsloth/skills_score.py --items index/decider_train/skills_items.jsonl --pred PRED.jsonl [--out result.json]

AUROC (Mann-Whitney, ties half) of: relevance (level >= 1 vs 0) and need (level 3 vs rest; vs AREA level 1; vs OFF level 0),
overall, on steps and on user turns, with a by-case cluster-bootstrap 95% interval; jjava's belief on the same rows
(relevance = 1 - P(level 0), need = P(level 3), variant a as served). The label set is thin (docs/JJAVA.md 9.1 D8:
user turns are 11 distinct states) and the reads are n=1 per row; read the intervals.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skills"))
from inject_diagnose import auroc  # noqa: E402


def boot(rows, score, pos, neg, n=400, seed=7):
    by = {}
    for r in rows:
        by.setdefault(r["case"], []).append(r)
    ids = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [x for c in (rng.choice(ids) for _ in ids) for x in by[c]]
        a = auroc([score(x) for x in s if pos(x)], [score(x) for x in s if neg(x)])
        if a is not None:
            vals.append(a)
    vals.sort()
    if len(vals) < 20:
        return None
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals))], 3)]


def block(rows, score, pos, neg):
    p = [score(x) for x in rows if pos(x)]
    q = [score(x) for x in rows if neg(x)]
    a = auroc(p, q)
    return {"auroc": None if a is None else round(a, 3), "n_pos": len(p), "n_neg": len(q),
            "ci95_by_case": boot(rows, score, pos, neg)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    items = {(r["case"], r["key"]): r for r in map(json.loads, open(a.items, encoding="utf-8"))}
    pred = {(r["case"], r["key"]): r for r in map(json.loads, open(a.pred, encoding="utf-8"))}
    rows = []
    for k, p in pred.items():
        it = items.get(k)
        if it is None:
            continue
        jp = it["jjava_probs"]
        rows.append({**it, "m_rel": p["relevance"], "m_need": p["need"],
                     "j_rel": 1 - float(jp.get("0", 0)), "j_need": float(jp.get("3", 0))})
    res = {"rows": len(rows), "cases": len({r["case"] for r in rows})}
    for kind, sel in (("all", lambda r: True), ("steps", lambda r: r["kind"] != "user"),
                      ("user", lambda r: r["kind"] == "user")):
        sub = [r for r in rows if sel(r)]
        d = {}
        for who, rel, need in (("model", "m_rel", "m_need"), ("jjava", "j_rel", "j_need")):
            d[who] = {
                "relevance (lv>=1 vs OFF)": block(sub, lambda x, k=rel: x[k], lambda x: x["level"] >= 1, lambda x: x["level"] == 0),
                "need (NEEDED vs rest)": block(sub, lambda x, k=need: x[k], lambda x: x["level"] == 3, lambda x: x["level"] != 3),
                "need vs AREA (lv 3 vs 1)": block(sub, lambda x, k=need: x[k], lambda x: x["level"] == 3, lambda x: x["level"] == 1),
                "need vs OFF (lv 3 vs 0)": block(sub, lambda x, k=need: x[k], lambda x: x["level"] == 3, lambda x: x["level"] == 0),
            }
        d["n"] = len(sub)
        d["needed"] = sum(r["level"] == 3 for r in sub)
        res[kind] = d
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
