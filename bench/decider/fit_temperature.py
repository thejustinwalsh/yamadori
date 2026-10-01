#!/usr/bin/env python
"""FIT JJAVA'S TEMPERATURE PER MODEL AND QUESTION SET, HELD OUT BY RUN
(no GPU: it re-scores decisions already logged).

    python bench/decider/fit_temperature.py --model bonsai --inject
    python bench/decider/fit_temperature.py --model bonsai \\
        --decisions logs/decider_decisions.jsonl --labels index/decider/labels.jsonl
    ... --write          # the sets that hold out go into the model's record

WHY (docs/JJAVA.md 8; SGLang's decision models, "How answers are computed",
docs.sglang.io/docs/supported-models/decision_models, read 2026-09-30): a
temperature over the label logits is the one calibration knob -- p_i =
exp(lp_i / T) / sum_j exp(lp_j / T); the vocabulary's normaliser cancels and
the label mass does not depend on T -- and the values are "not a calibrated
probability that the decision is correct. Validate any threshold on labeled
data". decider_bonsai.read() applies a FITTED T (the model profile's field
`temperature`) to each order's label distribution before the orders are
averaged; T = 1 until this script has written one.

THE DATA: logged decisions whose orders carry `raw` (each order's label
distribution as read, before any temperature: decide_turn's DECISIONS rows
since 2026-09-30) joined to a truth:
  --inject     bench/skills/inject/results/run_<model>.jsonl and
               decisions_<model>.jsonl (bench/skills/inject_decide.py) with
               the injector's labels (bench/skills/inject_labels.truth,
               --truth rubric by default, the tuning truth): skill_item's
               levels 0-3 per variant (b folds DONE and NEEDED into its top
               level; c and stage 3 are nouls: NEEDED / any listed item
               NEEDED); the run is the case's source session.
  --decisions + --labels   decide_turn's log and bench/decider/label.py's
               labels (the truth key per decision id); the run is the
               decision's conversation.

THE FIT, per question set (the THRESHOLDS keys: a decision's name up to its
first ':'): T minimises the mean negative log-likelihood of the truth under
decider_bonsai.readout_of(orders_raw, T) -- the exact readout read() makes
-- by golden-section search over log T in SEARCH (a search range, not a
prior: a minimum at its edge is reported `edge` and never written).

HELD OUT BY RUN (the coordinator's rule: "a setting that doesn't hold
held-out doesn't ship"): for each run, T is fitted on the OTHER runs and
scored on this one. `heldout_ok` needs at least two runs and the upper end
of the 95% interval (package_eval.ALPHA, the repo's convention) of the
per-decision NLL difference (fitted T minus T = 1) below zero. Only such a
set is written; the value written is the fit on all runs, with the held-out
evidence beside it. THRESHOLDS are then tuned on tempered values
(bench/skills/inject_tune.py re-scores with the fitted T).

Nothing is sent to any server; the only file written is the model's
profile record, and only with --write (measure_model.write_field).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, HERE)

import decider_bonsai as D  # noqa: E402

FIT_VERSION = "fit-temperature/1"
SEARCH = (math.log(0.01), math.log(100.0))   # log T; see "THE FIT"
FLOOR = 1e-12                                 # log(0) guard for the NLL


# ------------------------------------------------------------------ rows ---
def _jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try:
                    yield json.loads(ln)
                except ValueError:
                    continue


def decision_rows(path: str) -> dict:
    """{decision id: row} of decision rows carrying every order's raw."""
    out = {}
    for r in _jsonl(path):
        if r.get("row") != "decision":
            continue
        orders = r.get("orders") or []
        if orders and all(isinstance(o.get("raw"), dict) for o in orders):
            out[r["id"]] = r
    return out


def qset_of(name: str) -> str:
    return str(name or "").split(":", 1)[0]


def _row(d: dict, truth: str, run: str) -> dict:
    return {"id": d["id"], "set": qset_of(d["question"]["name"]),
            "run": run, "orders_raw": [o["raw"] for o in d["orders"]],
            "excluded": d.get("excluded") or [], "truth": str(truth),
            "type": d["question"].get("type")}


def inject_rows(model: str, truth_kind: str = "rubric") -> list[dict]:
    """The injector's labelled decisions (see the module doc)."""
    sys.path.insert(0, os.path.join(ROOT, "bench", "skills"))
    import inject_labels as IL
    res = os.path.join(ROOT, "bench", "skills", "inject", "results")
    runs = list(_jsonl(os.path.join(res, f"run_{model}.jsonl")))
    dec = decision_rows(os.path.join(res, f"decisions_{model}.jsonl"))
    truth = IL.truth(truth_kind)
    out = []
    for r in runs:
        run = r["source"] if r.get("group") != "corpus" else "corpus"
        v = r["variant"]
        for it in r["items"]:
            lv = truth.get((r["case"], it["key"]))
            d = dec.get(it.get("decision_id"))
            if lv is None or d is None:
                continue
            if v == "c":
                t = "true" if lv == 3 else "false"
            elif v == "b":
                t = str(min(int(lv), 2))
            else:
                t = str(int(lv))
            out.append(_row(d, t, run))
        s3 = r.get("stage3") or {}
        d = dec.get(s3.get("decision_id"))
        lvs = [truth.get((r["case"], k)) for k in s3.get("listed") or []]
        if d is not None and lvs and all(x is not None for x in lvs):
            out.append(_row(d, "true" if any(x == 3 for x in lvs)
                            else "false", run))
    return out


def label_rows(decisions: str, labels: str, model: str) -> list[dict]:
    """decide_turn's log joined to label.py's truth labels by decision id."""
    truth = {}
    for r in _jsonl(labels):
        if r.get("row") == "label" and not r.get("unsure") \
                and r.get("truth") is not None:
            truth[r.get("decision_id")] = r["truth"]
    out = []
    for did, d in decision_rows(decisions).items():
        if did in truth and D.canonical_model(d.get("model")) \
                == D.canonical_model(model):
            t = truth[did]
            t = ("true" if t else "false") if isinstance(t, bool) else t
            out.append(_row(d, t, d.get("conversation") or "unknown"))
    return out


# ------------------------------------------------------------------- fit ---
def dist(row: dict, t: float) -> dict:
    keys = list(row["orders_raw"][0])
    allowed = [k for k in keys if k not in row["excluded"]]
    return D.readout_of(row["orders_raw"], t, allowed)


def nll(rows: list[dict], t: float) -> list[float]:
    return [-math.log(max(dist(r, t).get(r["truth"], 0.0), FLOOR))
            for r in rows]


def fit(rows: list[dict]) -> dict:
    """{t, nll, edge}: golden-section search of the mean NLL over log T."""
    f = lambda lt: sum(nll(rows, math.exp(lt))) / len(rows)  # noqa: E731
    a, b = SEARCH
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    lt = (a + b) / 2
    span = SEARCH[1] - SEARCH[0]
    edge = min(lt - SEARCH[0], SEARCH[1] - lt) < 0.01 * span
    return {"t": round(math.exp(lt), 6), "nll": round(f(lt), 6),
            "edge": edge}


def _metrics(rows: list[dict], t: float) -> dict:
    import tune
    pairs, acc = [], 0
    for r in rows:
        p = dist(r, t)
        top = max(p, key=p.get)
        ok = top == r["truth"]
        acc += ok
        pairs.append((p[top], ok))
    ll = nll(rows, t)
    return {"nll": round(sum(ll) / len(ll), 6),
            "accuracy": round(acc / len(rows), 4),
            "ece": tune.ece(pairs)}


def heldout(rows: list[dict]) -> dict:
    """Leave one run out: T fitted on the others, scored on the run."""
    import package_eval
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - float(package_eval.ALPHA) / 2)
    runs = sorted({r["run"] for r in rows})
    per, diffs, moved = [], [], 0
    for g in runs:
        tr = [r for r in rows if r["run"] != g]
        te = [r for r in rows if r["run"] == g]
        if not tr or not te:
            continue
        ft = fit(tr)
        a, b = nll(te, ft["t"]), nll(te, 1.0)
        diffs += [x - y for x, y in zip(a, b)]
        for r in te:
            p1, pt = dist(r, 1.0), dist(r, ft["t"])
            moved += max(p1, key=p1.get) != max(pt, key=pt.get)
        per.append({"run": g, "n": len(te), "t": ft["t"], "edge": ft["edge"],
                    "nll_fitted": round(sum(a) / len(a), 6),
                    "nll_t1": round(sum(b) / len(b), 6)})
    n = len(diffs)
    if n < 2 or len(per) < 2:
        return {"runs": per, "n": n, "heldout_ok": False,
                "why": "fewer than two runs to hold out"}
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    hi = m + z * sd / math.sqrt(n)
    ok = hi < 0 and not any(p["edge"] for p in per)
    return {"runs": per, "n": n, "mean_nll_diff": round(m, 6),
            "ci95": [round(m - z * sd / math.sqrt(n), 6), round(hi, 6)],
            "argmax_moved": moved, "heldout_ok": ok,
            "why": "" if ok else (
                "a held-out fold's minimum sat at the search edge"
                if hi < 0 else "the held-out NLL did not improve at 95%")}


def report(rows: list[dict]) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        if r["truth"] in r["orders_raw"][0]:
            by[r["set"]].append(r)
    out = {}
    for s, rs in sorted(by.items()):
        f = fit(rs)
        out[s] = {"n": len(rs), "runs": len({r["run"] for r in rs}),
                  "fit_all": f, "at_t1": _metrics(rs, 1.0),
                  "at_fit": _metrics(rs, f["t"]), "heldout": heldout(rs)}
    return out


def write(model: str, rep: dict, source: str) -> dict:
    """The sets that hold out, into the model's profile record."""
    import measure_model as MM
    sets = {s: {"value": x["fit_all"]["t"], "n": x["n"], "runs": x["runs"],
                "heldout_ok": True, "heldout": {
                    k: x["heldout"][k] for k in ("n", "mean_nll_diff",
                                                 "ci95", "argmax_moved")},
                "at_t1": x["at_t1"], "at_fit": x["at_fit"]}
            for s, x in rep.items()
            if x["heldout"].get("heldout_ok") and not x["fit_all"]["edge"]}
    if not sets:
        return {"written": False, "why": "no question set held out by run"}
    path = MM.write_field(model, "temperature", {
        "script": "bench/decider/fit_temperature.py",
        "version": FIT_VERSION, "date": time.strftime("%Y-%m-%d"),
        "source": source, "sets": sets})
    return {"written": True, "path": path, "sets": sorted(sets)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--truth", choices=("rubric", "hindsight"),
                    default="rubric")
    ap.add_argument("--decisions")
    ap.add_argument("--labels")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.inject:
        rows, source = inject_rows(a.model, a.truth), f"inject ({a.truth})"
    elif a.decisions and a.labels:
        rows, source = label_rows(a.decisions, a.labels, a.model), "labels"
    else:
        ap.error("--inject, or --decisions with --labels")
    rep = report(rows)
    out = {"model": a.model, "version": FIT_VERSION, "rows": len(rows),
           "sets": rep}
    if a.write:
        out["write"] = write(a.model, rep, source)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
