#!/usr/bin/env python
"""TUNE THE INJECTOR'S QUESTION SETS ON ONE MODEL -- a PROPOSAL, never
applied: separation per question variant, then the THRESHOLDS rows at the
targets, through bench/decider/tune.py's own code (propose, bins, ECE,
Wilson).

    python bench/skills/inject_tune.py --model bonsai [--variant a] [--json]

Reads bench/skills/inject/results/run_<model>.jsonl (bench/skills/
inject_decide.py) and the truth labels (bench/skills/inject/labels.jsonl,
labeller A; bench/skills/inject_labels.py).

THE TRUTH (--truth): `rubric` by default -- the hand labels (pass A on
every item; the blind pass B on its sample), because the hindsight rule H1
did NOT agree with the rubric: on NEEDED vs not, H1@write vs pass A kappa
0.08 (n=470 items), vs the blind pass B 0.008 (n=181), while pass A vs
pass B is 0.75 (n=199, 48 cases) -- 2026-09-29, `inject_labels.py agree`.
`hindsight` (H1 where it settles an item) is reported beside it.

THE QUESTION SETS (mcp/skill_inject.py THRESHOLDS rows):
  skill_item    stage 2, one row per (case, item): the belief is P(top
                level) (variant a/b) or the noul (c); the truth is the
                label's level 3 (NEEDED) -> "true".
  skill_inject  stage 3, one row per case: the noul; the truth is "true"
                when any item it listed is labelled 3.

SEPARATION FIRST (docs/JJAVA.md 4, the operator's "no easy bails"): each
variant's AUROC of its belief against the truth (the Mann-Whitney form; 0.5
is no separation) with the n of each side, and its ECE. The variant that
separates best is the one tuned; a variant whose AUROC's 95% interval (the
Hanley-McNeil standard error at the repo's ALPHA) includes 0.5 does NOT
separate, and is reported as such, never tuned to a default.

THE TARGETS: tune.py takes the owner's target accuracies. Here they are
DERIVED as break-even precisions -- the precision p at which an injected
item's expected gain equals its expected cost, p* = C / (G + C):
  medium  Skills in the Wild 2604.04323 (Qwen3.5-397B, 84 tasks x 3 runs):
          curated force-loaded skills +20.7 pp over none (41.2 vs 20.5);
          adding distractors cost 7.5 pp (41.2 -> 33.7): p* = 7.5 / 28.2 =
          0.266 -- positive expected value on OUR model family's evidence.
  high    SkillsBench 2602.12670v4 (87 tasks, 18 configurations): curated
          +16.6 pp; the worst "confidently wrong" self-generated pack -11.5
          pp: p* = 11.5 / 28.1 = 0.409 -- positive even at the worst cited
          cost.
  The bound is tune.py's Wilson lower bound at package_eval.ALPHA, so a
  tier is proposed only where the LOWER end of its precision clears p*.
  --high-accuracy / --medium-accuracy override them (the operator's call).

Nothing is written. The owner pastes an accepted row into skill_inject.
THRESHOLDS with its n and date.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bench", "decider"))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import inject_labels as IL  # noqa: E402

RESULTS = os.path.join(HERE, "inject", "results")
TRUTH = ["rubric"]               # --truth: rubric (default) | hindsight
TARGET_MEDIUM = round(7.5 / (20.7 + 7.5), 3)     # 0.266, Skills in the Wild
TARGET_HIGH = round(11.5 / (16.6 + 11.5), 3)     # 0.409, SkillsBench


def auroc(pos: list[float], neg: list[float]) -> dict:
    """Mann-Whitney AUROC with the Hanley-McNeil standard error."""
    n1, n2 = len(pos), len(neg)
    if not n1 or not n2:
        return {"auroc": None, "n_true": n1, "n_false": n2}
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    a = wins / (n1 * n2)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt(max((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                        + (n2 - 1) * (q2 - a * a)) / (n1 * n2), 0.0))
    import package_eval
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - float(package_eval.ALPHA) / 2)
    return {"auroc": round(a, 4), "se": round(se, 4),
            "ci": [round(a - z * se, 4), round(a + z * se, 4)],
            "separates": a - z * se > 0.5, "n_true": n1, "n_false": n2}


def load(model: str) -> list[dict]:
    path = os.path.join(RESULTS, f"run_{model}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def rows_for(runs: list[dict], variant: str, truth: dict) -> tuple[list, list]:
    """(stage-2 rows, stage-3 rows) in tune.py's labelled-row shape."""
    s2, s3 = [], []
    for r in runs:
        if r["variant"] != variant:
            continue
        for it in r["items"]:
            lv = truth.get((r["case"], it["key"]))
            if lv is None:
                continue
            t = "true" if lv == 3 else "false"
            s2.append({"id": it["decision_id"], "model": r["model"],
                       "question": "skill_item", "family": "skill_item",
                       "type": "noul", "p": None, "confidence": None,
                       "noul": it["belief"],
                       "argmax": "true" if it["pass"] else "false",
                       "truth": t, "correct": ("true" if it["pass"] else
                                               "false") == t,
                       "how": "label", "kind": r["kind"],
                       "group": r["group"]})
        lv = [truth.get((r["case"], k)) for k in r["stage3"]["listed"]]
        if any(x is None for x in lv):
            continue
        t = "true" if any(x == 3 for x in lv) else "false"
        yes = r["stage3"]["noul"] >= 0.5 and not r["stage3"]["tie"]
        s3.append({"id": r["stage3"]["decision_id"], "model": r["model"],
                   "question": "skill_inject", "family": "skill_inject",
                   "type": "noul", "p": None, "confidence": None,
                   "noul": r["stage3"]["noul"],
                   "argmax": "true" if yes else "false", "truth": t,
                   "correct": ("true" if yes else "false") == t,
                   "how": "label", "kind": r["kind"],
                   "listed_how": r["stage3"]["how"]})
    return s2, s3


def confusion(rows: list[dict], act_yes: float | None) -> dict:
    """Inject decisions at a yes threshold (None: the untuned argmax)."""
    c = collections.Counter()
    for r in rows:
        said = (r["argmax"] == "true") if act_yes is None else \
            (r["noul"] >= act_yes)
        c[("tp" if r["truth"] == "true" else "fp") if said else
          ("fn" if r["truth"] == "true" else "tn")] += 1
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {**{k: c[k] for k in ("tp", "fp", "tn", "fn")},
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "injected": tp + fp}


def report(model: str, variant: str | None, high: float, medium: float
           ) -> dict:
    import tune
    runs = load(model)
    truth = IL.truth(TRUTH[0])
    out = {"model": model, "labels": len(truth),
           "targets": {"high": high, "medium": medium,
                       "derivation": "break-even precision p* = C/(G+C): "
                       "medium Skills in the Wild (7.5 / 28.2), high "
                       "SkillsBench (11.5 / 28.1)"},
           "variants": {}}
    from statistics import NormalDist
    import package_eval
    z = NormalDist().inv_cdf(1 - float(package_eval.ALPHA) / 2)
    for v in sorted({r["variant"] for r in runs}):
        s2, s3 = rows_for(runs, v, truth)
        sep2 = auroc([r["noul"] for r in s2 if r["truth"] == "true"],
                     [r["noul"] for r in s2 if r["truth"] == "false"])
        sep3 = auroc([r["noul"] for r in s3 if r["truth"] == "true"],
                     [r["noul"] for r in s3 if r["truth"] == "false"])
        by_kind = {k: auroc([r["noul"] for r in s2 if r["truth"] == "true"
                             and r["kind"] == k],
                            [r["noul"] for r in s2 if r["truth"] == "false"
                             and r["kind"] == k]) for k in ("user", "step")}
        vr = {"skill_item": {"n": len(s2), "separation": sep2,
                             "by_kind": by_kind,
                             "untuned": confusion(s2, None)},
              "skill_inject": {"n": len(s3), "separation": sep3,
                               "untuned": confusion(s3, None)}}
        for qs, rs in (("skill_item", s2), ("skill_inject", s3)):
            if not rs:
                continue
            p = tune.propose(rs, high, medium, z)
            vr[qs]["proposal"] = p
            vr[qs]["ece_top_label"] = tune.ece(
                [(max(r["noul"], 1 - r["noul"]), r["correct"]) for r in rs])
            vr[qs]["bins"] = tune.bins_of([(r["noul"], r["correct"],
                                            r["truth"] == "true")
                                           for r in rs])
            for side in ("act_yes", "caution_yes"):
                if p["row"].get(side) is not None:
                    vr[qs][f"at_{side}"] = confusion(rs, p["row"][side])
        out["variants"][v] = vr
    best = max(out["variants"].items(), key=lambda kv: (
        (kv[1]["skill_item"]["separation"].get("auroc") or 0.0)))[0] \
        if out["variants"] else None
    out["best_variant"] = variant or best
    return out


# ------------------------------------------------------- held out by run ---
# HOLD OUT BY RUN (coordinator, 2026-09-29): "Every threshold, wording and
# readout choice is selected on some runs and scored on runs it never saw
# (leave-one-run-out ...). Report n and a 95% interval. A setting that
# doesn't hold up on held-out runs doesn't ship." For each run R (a
# transcript's source; the corpus's test turns are one group): the VARIANT
# is the one with the best stage-2 AUROC on the other runs, its act_yes and
# caution_yes rows are tune.propose on the other runs at the targets, and
# R's rows are scored with them. The held-out decisions pooled over every R
# give the precision and recall that count, each with its Wilson 95%
# interval; a tier SHIPS only if its held-out precision's LOWER bound clears
# its target.
def _wilson(k: int, n: int, z: float) -> list:
    import tune
    if not n:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    mid = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(tune.wilson_lower(k, n, z), 4),
            round(min(1.0, (mid + half) / den), 4)]


def loro(model: str, high: float, medium: float,
         variants: list[str] | None = None) -> dict:
    import package_eval
    import tune
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - float(package_eval.ALPHA) / 2)
    runs = load(model)
    truth = IL.truth(TRUTH[0])
    group = {r["case"]: r["source"] if r["group"] != "corpus" else "corpus"
             for r in runs}
    vs = variants or sorted({r["variant"] for r in runs})
    rows = {v: rows_for(runs, v, truth) for v in vs}
    case_of = {}
    for r in runs:
        for it in r["items"]:
            case_of[it["decision_id"]] = r["case"]
        case_of[r["stage3"]["decision_id"]] = r["case"]
    groups = sorted(set(group.values()))
    pooled = {qs: {t: collections.Counter() for t in ("act_yes",
                                                      "caution_yes",
                                                      "untuned")}
              for qs in ("skill_item", "skill_inject")}
    picks = collections.Counter()
    for g in groups:
        def split(rs):
            tr = [r for r in rs if group[case_of[r["id"]]] != g]
            te = [r for r in rs if group[case_of[r["id"]]] == g]
            return tr, te
        best, best_a = None, -1.0
        for v in vs:
            tr, _te = split(rows[v][0])
            a = auroc([r["noul"] for r in tr if r["truth"] == "true"],
                      [r["noul"] for r in tr if r["truth"] == "false"]
                      ).get("auroc") or 0.0
            if a > best_a:
                best, best_a = v, a
        picks[best] += 1
        for qs, rs in (("skill_item", rows[best][0]),
                       ("skill_inject", rows[best][1])):
            tr, te = split(rs)
            row = tune.propose(tr, high, medium, z)["row"] if tr else {}
            for tier_ in ("act_yes", "caution_yes", "untuned"):
                t = row.get(tier_) if tier_ != "untuned" else None
                if tier_ != "untuned" and t is None:
                    pooled[qs][tier_]["unreached_folds"] += 1
                    continue
                c = confusion(te, t)
                for k in ("tp", "fp", "tn", "fn"):
                    pooled[qs][tier_][k] += c[k]
    out = {"model": model, "folds": len(groups), "groups": groups,
           "variant_picks": dict(picks), "targets": {"high": high,
                                                     "medium": medium}}
    for qs, tiers_ in pooled.items():
        out[qs] = {}
        for t, c in tiers_.items():
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            prec_ci = _wilson(tp, tp + fp, z)
            target = high if t == "act_yes" else medium \
                if t == "caution_yes" else None
            out[qs][t] = {
                "tp": tp, "fp": fp, "tn": c["tn"], "fn": fn,
                "unreached_folds": c["unreached_folds"],
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "precision_ci95": prec_ci,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                "recall_ci95": _wilson(tp, tp + fn, z),
                "ships": (target is not None and prec_ci[0] is not None
                          and prec_ci[0] >= target)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant")
    ap.add_argument("--high-accuracy", type=float, default=TARGET_HIGH)
    ap.add_argument("--medium-accuracy", type=float, default=TARGET_MEDIUM)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--truth", choices=("hindsight", "rubric"),
                    default="rubric",
                    help="the labels tuned against (inject_labels.truth)")
    ap.add_argument("--loro", action="store_true",
                    help="leave-one-run-out: select on the other runs, "
                    "score on the held-out one, pooled with 95% intervals")
    a = ap.parse_args(argv)
    TRUTH[0] = a.truth
    if a.loro:
        print(json.dumps(loro(a.model, a.high_accuracy, a.medium_accuracy,
                              [a.variant] if a.variant else None), indent=1))
        return 0
    rep = report(a.model, a.variant, a.high_accuracy, a.medium_accuracy)
    if a.json:
        print(json.dumps(rep, indent=1))
        return 0
    print(f"{a.model}: {rep['labels']} labelled items; targets "
          f"{rep['targets']['high']} / {rep['targets']['medium']}")
    for v, vr in rep["variants"].items():
        for qs in ("skill_item", "skill_inject"):
            x = vr[qs]
            s = x["separation"]
            print(f"  [{v}] {qs:<12} n={x['n']:<4} AUROC {s.get('auroc')} "
                  f"CI {s.get('ci')} separates={s.get('separates')}  "
                  f"untuned {x['untuned']}")
            p = x.get("proposal")
            if p:
                print(f"       proposal {json.dumps(p['row'])} "
                      f"unreached {p['unreached']}")
                for side in ("act_yes", "caution_yes"):
                    if x.get(f"at_{side}"):
                        print(f"       at {side}: {x[f'at_{side}']}")
    print(f"  best variant by stage-2 AUROC: {rep['best_variant']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
