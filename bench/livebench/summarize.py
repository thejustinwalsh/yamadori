"""Turn LiveBench per-question rows into a dashboard-ready JSON summary.

    python summarize.py results/<run-id>            -> results/<run-id>/summary.json

Scoring follows LiveBench exactly: a task score is the mean over its questions
(x100), a category score is the unweighted mean of its task scores, and the
overall score is the unweighted mean of the category scores. It is computed
only over the categories that were run, and `categories_included` says which.

Row status (see score.py):
  ok, budget_event -> scored (a length finish scores 0, as in LiveBench)
  not_run          -> excluded and counted: a transport failure is not an answer
  stack_error      -> excluded and counted: a mechanism broke; drive.py re-asks it
  eval_error, unjudged -> excluded and counted: the judge failed, not the model

95% intervals: percentile bootstrap over questions, resampled WITHIN each task
(the sample was stratified by task, see make_sample.py), B resamples, fixed
seed. Paired arm difference: questions both arms scored, resampled jointly
by question id within task; plus an exact McNemar test on the discordant
pairs where both scores are binary (0 or 1).

No numpy: the stack interpreter and the WSL venv both run this unchanged.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

SCORED = {"ok", "budget_event"}
B_DEFAULT = 2000
SEED = 20260923


def load_rows(run_dir: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(run_dir, "rows_*_*.jsonl"))):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def percentile(sorted_xs, q):
    if not sorted_xs:
        return None
    k = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def aggregate(by_cat_task: dict[str, dict[str, list[float]]]) -> dict:
    """LiveBench aggregation: task mean -> category mean of tasks -> overall mean of categories."""
    cats = {}
    for c, tasks in by_cat_task.items():
        tmeans = {t: 100.0 * mean(v) for t, v in tasks.items() if v}
        if tmeans:
            cats[c] = {"score": mean(list(tmeans.values())), "tasks": tmeans}
    overall = mean([v["score"] for v in cats.values()]) if cats else None
    return {"overall": overall, "categories": cats}


def resample(by_cat_task, rng):
    return {c: {t: [v[rng.randrange(len(v))] for _ in v] for t, v in tasks.items() if v}
            for c, tasks in by_cat_task.items()}


def bootstrap(by_cat_task, B, seed):
    rng = random.Random(seed)
    overall, cats, tasks = [], defaultdict(list), defaultdict(list)
    for _ in range(B):
        agg = aggregate(resample(by_cat_task, rng))
        overall.append(agg["overall"])
        for c, v in agg["categories"].items():
            cats[c].append(v["score"])
            for t, s in v["tasks"].items():
                tasks[(c, t)].append(s)

    def ci(xs):
        xs = sorted(xs)
        return [percentile(xs, 0.025), percentile(xs, 0.975)]

    return ci(overall), {c: ci(v) for c, v in cats.items()}, {k: ci(v) for k, v in tasks.items()}


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize_arm(rows: list[dict], B: int, seed: int) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    status, finish = Counter(), defaultdict(Counter)
    secs, toks = defaultdict(list), defaultdict(list)
    for r in rows:
        status[r["status"]] += 1
        finish[r["category"]][str(r.get("finish_reason"))] += 1
        if r["status"] in SCORED:
            by[r["category"]][r["task"]].append(float(r["score"] or 0.0))
        if r.get("seconds") is not None:
            secs[r["category"]].append(r["seconds"])
        out = (r.get("tokens") or {}).get("output")
        if out is not None and out >= 0:
            toks[r["category"]].append(out)
    agg = aggregate(by)
    o_ci, c_ci, t_ci = bootstrap(by, B, seed)

    def dist(xs):
        xs = sorted(xs)
        return {"n": len(xs), "median": percentile(xs, 0.5), "p90": percentile(xs, 0.9),
                "max": xs[-1] if xs else None, "total": sum(xs)}

    cats = {}
    for c, v in agg["categories"].items():
        cats[c] = {
            "score": v["score"], "ci95": c_ci[c],
            "n": sum(len(x) for x in by[c].values()),
            "tasks": {t: {"score": s, "ci95": t_ci[(c, t)], "n": len(by[c][t])} for t, s in v["tasks"].items()},
            "finish_reason": dict(finish[c]),
            "seconds": dist(secs[c]), "output_tokens": dist(toks[c]),
        }
    import mechanisms  # local: keeps this module importable where mechanisms.py is absent
    recs = [r["mechanisms"] for r in rows if r.get("mechanisms")]
    return {
        "overall": {"score": agg["overall"], "ci95": o_ci, "categories_included": sorted(cats)},
        "categories": cats,
        "status": dict(status),
        "finish_reason": dict(sum(finish.values(), Counter())),
        # allowed vs decided vs ran vs produced data, over every answer that
        # carried an x_yamadori record (stack_error answers included: they are
        # the evidence that a mechanism broke)
        "mechanism_health": mechanisms.health(recs) if recs else None,
    }


def paired(rows_a: list[dict], rows_b: list[dict], B: int, seed: int) -> dict:
    """b minus a on questions both arms scored."""
    ka = {(r["category"], r["task"], r["id"]): r for r in rows_a if r["status"] in SCORED}
    kb = {(r["category"], r["task"], r["id"]): r for r in rows_b if r["status"] in SCORED}
    common = sorted(set(ka) & set(kb))
    pairs = defaultdict(lambda: defaultdict(list))
    for k in common:
        pairs[k[0]][k[1]].append((float(ka[k]["score"] or 0), float(kb[k]["score"] or 0)))

    def diff(p):
        a = aggregate({c: {t: [x[0] for x in v] for t, v in ts.items()} for c, ts in p.items()})
        b = aggregate({c: {t: [x[1] for x in v] for t, v in ts.items()} for c, ts in p.items()})
        out = {c: b["categories"][c]["score"] - a["categories"][c]["score"] for c in a["categories"]}
        return (b["overall"] - a["overall"]) if a["overall"] is not None else None, out, a, b

    d_all, d_cat, a0, b0 = diff(pairs)
    rng = random.Random(seed + 1)
    boot_all, boot_cat = [], defaultdict(list)
    for _ in range(B):
        rs = {c: {t: [v[rng.randrange(len(v))] for _ in v] for t, v in ts.items()} for c, ts in pairs.items()}
        x, xc, _, _ = diff(rs)
        boot_all.append(x)
        for c, v in xc.items():
            boot_cat[c].append(v)

    def ci(xs):
        xs = sorted(x for x in xs if x is not None)
        return [percentile(xs, 0.025), percentile(xs, 0.975)]

    cats = {}
    for c, ts in pairs.items():
        flat = [x for v in ts.values() for x in v]
        binary = all(x in (0.0, 1.0) and y in (0.0, 1.0) for x, y in flat)
        b_only = sum(1 for x, y in flat if y > x) if binary else None
        a_only = sum(1 for x, y in flat if x > y) if binary else None
        cats[c] = {
            "n_pairs": len(flat), "a_score": a0["categories"][c]["score"], "b_score": b0["categories"][c]["score"],
            "diff": d_cat[c], "ci95": ci(boot_cat[c]),
            "binary": binary, "b_only_correct": b_only, "a_only_correct": a_only,
            "mcnemar_p": mcnemar_exact(b_only, a_only) if binary else None,
        }
    return {"n_pairs": len(common), "diff_overall": d_all, "ci95_overall": ci(boot_all),
            "categories_included": sorted(cats), "categories": cats}


def build(run_dir: str, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    rows = load_rows(run_dir)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    # An arm whose rows span more than one stack condition (condition.json
    # condition_epochs, e.g. before/after the tool-turn cap) is ALSO reported
    # per condition, as "<arm>@<condition>", each paired against bonsai. The
    # baseline itself is never split.
    for arm in [a for a in list(by_arm) if a != "bonsai"]:
        conds = {r.get("condition") for r in by_arm[arm]}
        if len(conds) > 1:
            for c in sorted(conds, key=str):
                by_arm[f"{arm}@{c}"] = [r for r in by_arm[arm] if r.get("condition") == c]
    meta = {}
    for p in glob.glob(os.path.join(run_dir, "order_*.json")):
        spec = json.load(open(p))
        meta[spec["category"]] = {"release": spec["release"], "seed": spec["seed"], "population": len(spec["order"])}
    out = {
        "run_id": os.path.basename(os.path.normpath(run_dir)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "release": sorted({m["release"] for m in meta.values()}),
        "population": {c: m["population"] for c, m in meta.items()},
        "method": {"bootstrap_B": B, "seed": seed, "ci": "percentile, stratified by task",
                   "aggregation": "LiveBench: mean(task) -> mean(tasks) per category -> mean(categories)"},
        "arms": {arm: summarize_arm(rs, B, seed) for arm, rs in sorted(by_arm.items())},
    }
    out["arm_labels"] = {a: (ARM_LABELS.get(a, a) if "@" not in a else
                             f"{ARM_LABELS.get(a.split('@')[0], a.split('@')[0])}; condition {a.split('@')[1]} only")
                         for a in by_arm}
    # every other arm against the effort-matched baseline, on the questions both scored
    if "bonsai" in by_arm:
        for arm in sorted(by_arm):
            if arm != "bonsai":
                out[f"paired_{arm}_minus_bonsai"] = paired(by_arm["bonsai"], by_arm[arm], B, seed)
    return out


# What each arm IS, for anything a person reads. The tier also raises the
# thinking effort sent upstream (minimal=low, medium=medium, max=xhigh), so a
# tier arm against "bare @ medium" mixes effort and tools.
ARM_LABELS = {
    "bonsai": "bare @ medium (all augmentation forced off, effort medium)",
    "yamadori": "tier max (reasoning_effort max -> xhigh upstream; selection decides fan-out/deep thinking)",
    "minimal": ("tier minimal as redefined 2026-09-23 ~11:30: thinking OFF (enable_thinking=false), "
                "vendor instruct sampling 0.7/0.80/20/presence 1.5, no augmentation"),
    "yamadori-xhigh": ("tier xhigh (reasoning_effort xhigh -> MEDIUM upstream; every augmentation allowed: "
                       "retrieval, hints, fan-out 3, deep thinking, check_code, repair, 10-turn tool cap) -- "
                       "the all-on arm, effort-matched to bonsai"),
}


def main() -> int:
    run_dir = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else B_DEFAULT
    s = build(run_dir, B)
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(s, f, indent=1)
    for arm, v in s["arms"].items():
        o = v["overall"]
        print(f"{arm}: overall {o['score']:.1f} [{o['ci95'][0]:.1f}, {o['ci95'][1]:.1f}] over {o['categories_included']}"
              f"  status={v['status']}  finish={v['finish_reason']}")
        for c, cv in v["categories"].items():
            ts = ", ".join(f"{t} {x['score']:.1f} (n={x['n']})" for t, x in cv["tasks"].items())
            print(f"   {c:24s} {cv['score']:5.1f} [{cv['ci95'][0]:.1f}, {cv['ci95'][1]:.1f}] n={cv['n']}  {ts}")
    for key in sorted(k for k in s if k.startswith("paired_")):
        p = s[key]
        if p["diff_overall"] is None:
            continue
        print(f"{key}: {p['diff_overall']:+.1f} [{p['ci95_overall'][0]:+.1f}, {p['ci95_overall'][1]:+.1f}] n={p['n_pairs']}")
        for c, v in p["categories"].items():
            print(f"   {c:24s} {v['diff']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] n={v['n_pairs']} "
                  f"mcnemar_p={v['mcnemar_p']}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
