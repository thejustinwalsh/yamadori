#!/usr/bin/env python
"""WHY THE INJECTOR'S QUESTIONS DO NOT SEPARATE -- the diagnosis, offline, on
a run that is already on disk (no GPU, nothing sent anywhere).

    python bench/skills/inject_diagnose.py --cases CASES --model bonsai-a4000

Reads bench/skills/inject/results/run_<model>.jsonl and decisions_<model>.jsonl
(bench/skills/inject_decide.py), the cases (the label kit's cases.jsonl: the
state text, the task, the items' craft names) and the rubric truth
(bench/skills/inject_labels.py: pass A plus the blind pass B). Prints one JSON
object; every claim in mcp/skill_inject.py QUESTION-SET VARIANTS (D1-D8) and
docs/JJAVA.md 9 is a field of it. Nothing is tuned here and nothing is
written: it MEASURES what the reads did, per hypothesis.

THE HYPOTHESES, and the field that answers each
  separation        variants[v].auroc.{all,step,user}: stage-2 AUROC of the
                    belief against NEEDED (the Mann-Whitney form), n of each
                    side, and by-case cluster-bootstrap 95% interval
  what it separates variants[v].vs_level: NEEDED (3) against OFF (0), AREA (1)
                    and DONE (2) one at a time; mean_belief_by_level
  precision         variants[v].at_threshold: injected, precision, recall at
                    belief >= t, overall and on steps
  readout           variants[v].readout: how often each level wins the argmax
                    in each printed order, the two orders' mean total
                    variation, how often their argmaxes agree, the labels'
                    share of the whole distribution (label_mass), the share
                    of reads whose argmax is a MIDDLE level
  state length      by_state_tokens, by_kind, by_kind_and_tokens: AUROC and
                    the needed rate per bucket (state length is confounded
                    with the kind: a user turn is short)
  how many items    by_items_in_case, by_kind_and_items
  the skill         by_skill_step: AUROC and the mean belief of the AREA items
                    per skill on steps (a generic process skill)
  within a case     within_case: the share of cases with a needed item whose
                    top-belief item is needed, against a random item; the
                    pairwise within-(case, skill) AUROC
  the item prior    item_prior: the belief minus the same item's mean belief
                    in OTHER states, against NEEDED
  the goal          stage3: each variant's stage-3 AUROC, by kind, and the
                    paired by-case bootstrap of d minus a
  label set         label_set: levels, the needed rate, distinct states per
                    kind (a repeated prompt is ONE state), the 15 copies of
                    the pagoda request among the user turns
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))

RESULTS = os.path.join(HERE, "inject", "results")
THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9)
BOOT = 400


def auroc(pos: list, neg: list) -> float | None:
    """Mann-Whitney AUROC by ranks (ties count half)."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2 + 1
        i = j + 1
    n1, n2 = len(pos), len(neg)
    sp = sum(r[k] for k, (_v, lab) in enumerate(allv) if lab == 1)
    return (sp - n1 * (n1 + 1) / 2) / (n1 * n2)


def _a(rows: list, key: str = "belief") -> float | None:
    a = auroc([x[key] for x in rows if x["need"]],
              [x[key] for x in rows if not x["need"]])
    return None if a is None else round(a, 3)


def _side(rows: list) -> dict:
    p = sum(x["need"] for x in rows)
    return {"n": len(rows), "needed": p,
            "rate": round(p / len(rows), 3) if rows else None}


def _bucket(rows: list, f) -> dict:
    d = collections.defaultdict(list)
    for x in rows:
        d[f(x)].append(x)
    return {str(k): {**_side(v), "auroc": _a(v)} for k, v in sorted(
        d.items(), key=lambda kv: str(kv[0]))}


def boot_ci(groups: dict, stat, n: int = BOOT, seed: int = 7) -> list | None:
    """The by-CASE cluster-bootstrap 95% interval of stat(rows)."""
    rnd = random.Random(seed)
    ids = sorted(groups)
    out = []
    for _ in range(n):
        rows = [x for c in (rnd.choice(ids) for _ in ids) for x in groups[c]]
        v = stat(rows)
        if v is not None:
            out.append(v)
    if len(out) < n // 2:
        return None
    out.sort()
    return [round(out[int(0.025 * len(out))], 3),
            round(out[int(0.975 * len(out))], 3)]


def stage2_rows(cases: dict, runs: list, truth: dict) -> list:
    rows = []
    for r in runs:
        c = cases.get(r["case"])
        if c is None:
            continue
        by = {it["key"]: it for it in c["items"]}
        for idx, it in enumerate(r["items"]):
            lv = truth.get((r["case"], it["key"]))
            if lv is None:
                continue
            ci = by.get(it["key"]) or {}
            rows.append({
                "v": r["variant"], "case": r["case"], "key": it["key"],
                "sha": it.get("sha"), "lv": lv, "need": int(lv == 3),
                "belief": it["belief"], "kind": r["kind"],
                "group": r["group"], "state_sha": c["state_sha"],
                "tokens": (c.get("state_info") or {}).get("tokens") or 0,
                "n_items": len(c["items"]), "idx": idx,
                "skill": ci.get("skill") or "", "fact": ci.get("fact") or "",
                "probs": it.get("probs") or {},
                "decision_id": it.get("decision_id")})
    return rows


def readout(rows: list, dec: dict) -> dict | None:
    """Per-order argmax, order disagreement and label mass of a variant's
    stage-2 reads (needs the decisions log)."""
    got = [(x, dec.get(x["decision_id"])) for x in rows]
    got = [(x, d) for x, d in got if d and len(d.get("orders") or []) >= 2]
    if not got:
        return None
    first = [max(d["orders"][0]["p"], key=d["orders"][0]["p"].get)
             for _x, d in got]
    second = [max(d["orders"][1]["p"], key=d["orders"][1]["p"].get)
              for _x, d in got]
    keys = sorted(got[0][1]["orders"][0]["p"])
    mid = set(keys[1:-1]) if len(keys) > 2 else set()
    n = len(got)
    return {
        "reads": n, "printed": [got[0][1]["orders"][0]["printed"],
                                got[0][1]["orders"][1]["printed"]],
        "argmax_share_order1": {k: round(first.count(k) / n, 3)
                                for k in keys},
        "argmax_share_order2": {k: round(second.count(k) / n, 3)
                                for k in keys},
        "mean_total_variation": round(statistics.mean(
            d.get("disagreement") or 0.0 for _x, d in got), 3),
        "argmax_agree": round(sum(a == b for a, b in zip(first, second))
                              / n, 3),
        "label_mass_min_mean": round(statistics.mean(
            d.get("label_mass_min") or 0.0 for _x, d in got), 3),
        "middle_level_argmax_share": (round(sum(
            (a in mid) + (b in mid) for a, b in zip(first, second))
            / (2 * n), 3) if mid else None)}


def variant_report(v: str, rows: list, dec: dict) -> dict:
    by_case = collections.defaultdict(list)
    for x in rows:
        by_case[x["case"]].append(x)
    step = [x for x in rows if x["kind"] == "step"]
    user = [x for x in rows if x["kind"] == "user"]
    out = {"n": len(rows), "needed": sum(x["need"] for x in rows),
           "auroc": {"all": _a(rows), "step": _a(step), "user": _a(user),
                     "all_ci95_by_case": boot_ci(by_case, _a)}}
    out["vs_level"] = {
        f"3_vs_{lvl}": (None if not any(x["lv"] == lvl for x in rows) else
                        round(auroc([x["belief"] for x in rows
                                     if x["lv"] == 3],
                                    [x["belief"] for x in rows
                                     if x["lv"] == lvl]) or 0.0, 3))
        for lvl in (0, 1, 2)}
    out["mean_belief_by_level"] = {
        str(lvl): round(statistics.mean(x["belief"] for x in rows
                                        if x["lv"] == lvl), 3)
        for lvl in range(4) if any(x["lv"] == lvl for x in rows)}
    out["at_threshold"] = {}
    for scope, rs in (("all", rows), ("step", step), ("user", user)):
        pos = sum(x["need"] for x in rs)
        tab = {}
        for t in THRESHOLDS:
            sel = [x for x in rs if x["belief"] >= t]
            tp = sum(x["need"] for x in sel)
            tab[str(t)] = {
                "injected": len(sel),
                "precision": round(tp / len(sel), 3) if sel else None,
                "recall": round(tp / pos, 3) if pos else None}
        out["at_threshold"][scope] = {"base_rate": round(
            pos / len(rs), 3) if rs else None, **tab}
    out["readout"] = readout(rows, dec)
    return out


def within_case(rows: list) -> dict:
    by = collections.defaultdict(list)
    for x in rows:
        by[x["case"]].append(x)
    pos = [xs for xs in by.values() if any(y["need"] for y in xs)]
    top1 = [max(xs, key=lambda y: (y["belief"], y["key"]))["need"]
            for xs in pos]
    base = [statistics.mean(y["need"] for y in xs) for xs in pos]
    grp = collections.defaultdict(list)
    for x in rows:
        grp[(x["case"], x["skill"])].append(x)
    tot = w = 0.0
    for xs in grp.values():
        p = [y["belief"] for y in xs if y["need"]]
        n = [y["belief"] for y in xs if not y["need"]]
        a = auroc(p, n)
        if a is not None:
            tot += a * len(p) * len(n)
            w += len(p) * len(n)
    mixed = sum(1 for xs in grp.values()
                if 0 < sum(y["need"] for y in xs) < len(xs))
    return {"cases_with_a_needed_item": len(pos), "cases": len(by),
            "top1_is_needed": round(statistics.mean(top1), 3) if top1
            else None, "random_item_is_needed": round(
                statistics.mean(base), 3) if base else None,
            "within_case_skill_auroc": round(tot / w, 3) if w else None,
            "case_skill_groups": len(grp), "groups_with_mixed_need": mixed}


def item_prior(rows: list) -> dict:
    by = collections.defaultdict(list)
    for x in rows:
        by[x["sha"]].append(x)
    ok = []
    for x in rows:
        others = {y["state_sha"]: y["belief"] for y in by[x["sha"]]
                  if y["state_sha"] != x["state_sha"]}
        if others:
            ok.append(dict(x, prior=statistics.mean(others.values()),
                           resid=x["belief"] - statistics.mean(
                               others.values())))
    out = {"rows": len(ok)}
    for scope, rs in (("all", ok), ("step", [x for x in ok
                                             if x["kind"] == "step"]),
                      ("user", [x for x in ok if x["kind"] == "user"])):
        out[scope] = {"belief": _a(rs), "item_prior_alone": _a(rs, "prior"),
                      "belief_minus_item_prior": _a(rs, "resid")}
    return out


def stage3(cases: dict, runs: list, truth: dict) -> dict:
    per: dict = collections.defaultdict(dict)
    for r in runs:
        lv = [truth.get((r["case"], k)) for k in r["stage3"]["listed"]]
        if any(x is None for x in lv):
            continue
        per[r["variant"]][r["case"]] = {
            "noul": r["stage3"]["noul"], "need": int(any(x == 3 for x in lv)),
            "kind": r["kind"], "how": r["stage3"]["how"]}
    out = {"variants": {}}
    for v, cs in sorted(per.items()):
        rows = list(cs.values())
        out["variants"][v] = {
            "cases": len(rows), "needed": sum(x["need"] for x in rows),
            "auroc": _a(rows, "noul"),
            "step": _a([x for x in rows if x["kind"] == "step"], "noul"),
            "user": _a([x for x in rows if x["kind"] == "user"], "noul"),
            "passed_share": round(sum(x["how"] == "passed" for x in rows)
                                  / len(rows), 3)}
    if "a" in per and "d" in per:
        common = sorted(set(per["a"]) & set(per["d"]))
        rnd = random.Random(7)
        diffs = []
        for _ in range(BOOT * 2):
            s = [rnd.choice(common) for _ in common]
            x = auroc([per["d"][c]["noul"] for c in s if per["d"][c]["need"]],
                      [per["d"][c]["noul"] for c in s
                       if not per["d"][c]["need"]])
            y = auroc([per["a"][c]["noul"] for c in s if per["a"][c]["need"]],
                      [per["a"][c]["noul"] for c in s
                       if not per["a"][c]["need"]])
            if x is not None and y is not None:
                diffs.append(x - y)
        diffs.sort()
        out["d_minus_a"] = None if not diffs else {
            "median": round(statistics.median(diffs), 3),
            "ci95": [round(diffs[int(0.025 * len(diffs))], 3),
                     round(diffs[int(0.975 * len(diffs))], 3)],
            "share_above_zero": round(sum(d > 0 for d in diffs)
                                      / len(diffs), 3)}
    return out


def report(cases: dict, runs: list, dec: dict, truth: dict) -> dict:
    rows = stage2_rows(cases, runs, truth)
    served = sorted({x["v"] for x in rows})
    out: dict = {"variants": {}}
    for v in served:
        out["variants"][v] = variant_report(
            v, [x for x in rows if x["v"] == v], dec)
    ref = "a" if "a" in served else (served[0] if served else None)
    base = [x for x in rows if x["v"] == ref]
    if base:
        step = [x for x in base if x["kind"] == "step"]
        tk = lambda x: ("<=60" if x["tokens"] <= 60 else "<=250"          # noqa: E731
                        if x["tokens"] <= 250 else "<=800"
                        if x["tokens"] <= 800 else ">800")
        out["reference_variant"] = ref
        out["by_kind"] = _bucket(base, lambda x: x["kind"])
        out["by_state_tokens"] = _bucket(base, tk)
        out["by_kind_and_tokens"] = _bucket(
            base, lambda x: (x["kind"], tk(x)))
        out["by_items_in_case"] = _bucket(
            base, lambda x: "1-6" if x["n_items"] <= 6 else "7+")
        out["by_kind_and_items"] = _bucket(
            base, lambda x: (x["kind"], "1-6" if x["n_items"] <= 6
                             else "7+"))
        sk = collections.defaultdict(list)
        for x in step:
            sk[x["skill"]].append(x)
        out["by_skill_step"] = {
            k: {**_side(v), "auroc": _a(v), "mean_belief_of_area_items":
                round(statistics.mean([x["belief"] for x in v
                                       if x["lv"] == 1]), 3)
                if any(x["lv"] == 1 for x in v) else None}
            for k, v in sorted(sk.items(), key=lambda kv: -len(kv[1]))
            if len(v) >= 30}
        nn = [x for x in step if not x["need"] and x["belief"] > 0.7]
        top = collections.Counter(x["skill"] for x in nn).most_common(1)
        out["step_false_yes_above_0.7"] = {
            "non_needed_step_items_above_0.7": len(nn),
            "from_the_most_common_skill": top[0] if top else None,
            "step_rows": len(step)}
        out["within_case"] = within_case(base)
        out["item_prior"] = item_prior(base)
        states = collections.defaultdict(set)
        copies = collections.Counter()
        for x in base:
            states[x["kind"]].add(x["state_sha"])
        for c in {x["case"]: x for x in base}.values():
            if c["kind"] == "user":
                copies[c["state_sha"]] += 1
        lv = collections.Counter(x["lv"] for x in base)
        out["label_set"] = {
            "items": len(base),
            "levels": {str(k): v for k, v in sorted(lv.items())},
            "needed_rate": round(lv[3] / len(base), 3),
            "distinct_states": {k: len(v) for k, v in states.items()},
            "user_cases": sum(copies.values()),
            "user_cases_in_the_most_repeated_state": max(
                copies.values()) if copies else None,
            "user_items": sum(1 for x in base if x["kind"] == "user"),
            "step_items": len(step)}
    out["stage3"] = stage3(cases, runs, truth)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)
    import inject_labels as IL
    with open(a.cases, encoding="utf-8") as f:
        cases = {c["case"]: c for c in (json.loads(ln) for ln in f
                                        if ln.strip())}
    with open(os.path.join(a.results, f"run_{a.model}.jsonl"),
              encoding="utf-8") as f:
        runs = [json.loads(ln) for ln in f if ln.strip()]
    dec = {}
    path = os.path.join(a.results, f"decisions_{a.model}.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("row") == "decision":
                    dec[r["id"]] = r
    print(json.dumps(report(cases, runs, dec, IL.truth("rubric")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
