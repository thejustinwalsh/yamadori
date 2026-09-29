#!/usr/bin/env python
"""TUNE A QUESTION SET'S TIERS from labelled decisions -- a PROPOSAL the
owner accepts, never applied.

    python bench/decider/tune.py --stats
    python bench/decider/tune.py --question phase \\
        --high-accuracy 0.95 --medium-accuracy 0.85
    python bench/decider/tune.py --question build_intent --model bonsai \\
        --high-accuracy 0.95 --medium-accuracy 0.85 --json

Operator, 2026-09-29: callers own thresholds; "a question set's code owns
tiered thresholds (high: act; medium: proceed with caution/confirm; low:
don't act/fall back), tuned per question set per model on labelled data;
none ship untuned." Jev's own guidance is the same shape (docs.typesafe.ai
/confidence: "Three paths for using confidence in your code"; "Start with
conservative thresholds, test with your own data, and adjust"). The
decider (mcp/decider_bonsai.py) returns belief and certainty only; the
thresholds live in the question set's code, mcp/decide_turn.py THRESHOLDS,
keyed by model then question set. This tool reads the decisions
(logs/decider_decisions.jsonl, via bench/decider/label.py) and the
operator's truth labels (index/decider/labels.jsonl), and for ONE question
set and ONE model prints:

  1. ACCURACY VS CERTAINTY: the labelled decisions in 10 equal-width bins
     (JevBench's ECE convention, jevbench/metrics.py ece_top_label; Guo et
     al. 1706.04599) of Jev's `confidence` (Choice / Score) or of the `noul`
     (a noul has no confidence: Jev /primitives/noul), each bin's n, mean
     and accuracy -- for a noul also the share whose truth is "true" (its
     calibration) -- plus the top-label ECE of the probabilities.
  2. A PROPOSED ROW for decide_turn.THRESHOLDS at the OWNER'S target
     accuracies (--high-accuracy, --medium-accuracy: required, no default --
     "A number or rule exists only if it is the operator's decision ...",
     AGENTS.md):
       Choice / Score  high = the LOWEST confidence t such that the
                       decisions at confidence >= t are correct at a rate
                       whose Wilson lower bound is >= the high target;
                       medium the same at the medium target.
       Noul            act_yes = the LOWEST p >= 0.5 such that the
                       decisions with noul >= p are TRUE at a rate whose
                       lower bound is >= the high target; act_no = the
                       HIGHEST p < 0.5 such that those with noul <= p are
                       FALSE at that rate; caution_
                       yes / caution_no the same at the medium target (each
                       side on its own: Jev /primitives/noul -- raise the
                       yes bar where a false yes is expensive).
     The bound is Wilson's at the repo's significance level (mcp/
     package_eval.py ALPHA = 0.05, two-sided; operator 2026-09-27, "keep
     ours"). A target no threshold reaches is reported as such: that tier
     stays untuned. The row is printed with its n and the coverage of each
     tier on the labelled set; the owner pastes it into THRESHOLDS (with
     the date and the n) or does not.

Nothing is written. Labels marked unsure, or with no truth, are left out.
A decision labelled twice uses its latest label. --reuse-by-state also
counts a label made on ANOTHER model's decision of the same question and
state (the truth of a state does not depend on the model; label.py records
the state hash for this).
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
sys.path.insert(0, os.path.join(ROOT, "mcp"))

import label as L  # noqa: E402

TUNE_VERSION = "decider-tune/1"
BINS = 10          # JevBench metrics.ece_top_label (10 equal-width bins)


def _alpha() -> tuple[float, str]:
    import package_eval
    return float(package_eval.ALPHA), "mcp/package_eval.py ALPHA"


def wilson_lower(k: int, n: int, z: float) -> float:
    """The lower end of the Wilson score interval for k successes in n."""
    if n <= 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    mid = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (mid - half) / den)


# ------------------------------------------------------------- the data ----
def labels_by(path: str = L.LABELS) -> tuple[dict, dict]:
    """({decision id: truth}, {(state sha, question): truth}): the latest
    sure label of each."""
    by_id, by_state = {}, {}
    for _i, r in L._jsonl(path):
        if r.get("row") != "label" or r.get("unsure") \
                or r.get("truth") is None:
            continue
        by_id[r.get("decision_id")] = r["truth"]
        if r.get("state_sha"):
            by_state[(r["state_sha"], r.get("question"))] = r["truth"]
    return by_id, by_state


def labelled(decisions: list[dict], by_id: dict, by_state: dict,
             reuse_by_state: bool = False) -> list[dict]:
    """Decisions with a truth: each {model, question, family, type, p,
    argmax, confidence, noul, truth, correct, how}."""
    out = []
    for d in decisions:
        truth, how = by_id.get(d["id"]), "label"
        if truth is None and reuse_by_state and d.get("state_sha"):
            truth, how = by_state.get((d["state_sha"], d["question"])), \
                "state"
        if truth is None:
            continue
        argmax = d.get("argmax")
        if argmax is None and d.get("p"):
            argmax = max(d["p"], key=d["p"].get)
        out.append({**{k: d.get(k) for k in (
            "id", "model", "question", "family", "type", "p", "confidence",
            "noul")}, "argmax": L._norm_value(argmax), "truth": truth,
            "correct": L._norm_value(argmax) == truth, "how": how})
    return out


def select(rows: list[dict], question: str, model: str | None
           ) -> list[dict]:
    """The rows of one question set (its full name, or its family) and one
    model (None: the rows' only model; several -> refuse, per model)."""
    rs = [r for r in rows if r["question"] == question
          or r["family"] == question]
    if model is not None:
        rs = [r for r in rs if (r["model"] or "") == model]
    return rs


# --------------------------------------------------------------- report ----
def bins_of(xs: list[tuple[float, bool, bool | None]]) -> list[dict]:
    """10 equal-width bins of (certainty, correct, truth_is_true)."""
    out = [{"lo": i / BINS, "hi": (i + 1) / BINS, "n": 0, "sum": 0.0,
            "correct": 0, "true": 0} for i in range(BINS)]
    for c, ok, t in xs:
        b = out[min(int(min(max(c, 0.0), 1.0) * BINS), BINS - 1)]
        b["n"] += 1
        b["sum"] += c
        b["correct"] += int(ok)
        b["true"] += int(bool(t))
    return [{"range": f"[{b['lo']:.1f}, {b['hi']:.1f}"
             + ("]" if b["hi"] >= 1 else ")"), "n": b["n"],
             "mean": round(b["sum"] / b["n"], 4) if b["n"] else None,
             "accuracy": round(b["correct"] / b["n"], 4) if b["n"] else None,
             "true_rate": round(b["true"] / b["n"], 4) if b["n"] else None}
            for b in out]


def ece(pairs: list[tuple[float, bool]]) -> float | None:
    """Top-label ECE, 10 equal-width bins (jevbench/metrics.ece_top_label)."""
    if not pairs:
        return None
    acc = collections.defaultdict(lambda: [0, 0.0, 0])
    for c, ok in pairs:
        i = min(int(min(max(c, 0.0), 1.0) * BINS), BINS - 1)
        acc[i][0] += 1
        acc[i][1] += c
        acc[i][2] += int(ok)
    n = len(pairs)
    return round(sum(v[0] / n * abs(v[2] / v[0] - v[1] / v[0])
                     for v in acc.values()), 6)


def _lowest(cands, keep, z: float, target: float):
    """The lowest candidate t whose kept rows meet `target` at the Wilson
    lower bound: (t, n, k, lower) or None."""
    for t in sorted(set(cands)):
        rs = keep(t)
        k = sum(1 for ok in rs if ok)
        lb = wilson_lower(k, len(rs), z)
        if rs and lb >= target:
            return {"t": round(t, 6), "n": len(rs), "correct": k,
                    "accuracy": round(k / len(rs), 4), "lower": round(lb, 4)}
    return None


def propose(rs: list[dict], high: float, medium: float, z: float) -> dict:
    """The proposed THRESHOLDS row (see the module docstring), each value
    with its evidence; a tier no threshold reaches is None."""
    qtype = rs[0]["type"]
    if qtype == "noul":
        ps = [(float(r["noul"]), r["truth"] == "true") for r in rs
              if r["noul"] is not None]

        # Each side's thresholds stay on its own side of 0.5: act_yes /
        # caution_yes among the decisions that said yes (noul >= 0.5, the
        # argmax "true"), act_no / caution_no among those that said no.
        def yes(t):
            return [tt for p, tt in ps if p >= t]

        def no(t):
            return [not tt for p, tt in ps if p <= t]
        ys = [p for p, _ in ps if p >= 0.5]
        ns = [-p for p, _ in ps if p < 0.5]
        ay = _lowest(ys, yes, z, high)
        cy = _lowest(ys, yes, z, medium)
        an = _lowest(ns, lambda t: no(-t), z, high)
        cn = _lowest(ns, lambda t: no(-t), z, medium)
        for x in (an, cn):
            if x:
                x["t"] = -x["t"]
        ev = {"act_yes": ay, "act_no": an, "caution_yes": cy,
              "caution_no": cn}
    else:
        cs = [(float(r["confidence"]), r["correct"]) for r in rs
              if r["confidence"] is not None]
        ev = {"high": _lowest([c for c, _ in cs],
                              lambda t: [ok for c, ok in cs if c >= t], z,
                              high),
              "medium": _lowest([c for c, _ in cs],
                                lambda t: [ok for c, ok in cs if c >= t], z,
                                medium)}
    row = {k: v["t"] for k, v in ev.items() if v is not None}
    return {"row": row, "evidence": ev,
            "unreached": sorted(k for k, v in ev.items() if v is None)}


def coverage(rs: list[dict], row: dict) -> dict:
    """How the labelled rows fall into the proposed tiers, with each tier's
    accuracy -- through decide_turn.tier itself (the code that will read
    the row)."""
    import decide_turn
    out = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        a = {"type": r["type"], "name": r["question"],
             "noul": r["noul"], "confidence": r["confidence"]}
        save = decide_turn.THRESHOLDS
        decide_turn.THRESHOLDS = {"m": {r["question"]: row}}
        try:
            t = decide_turn.tier(a, "m")["tier"]
        finally:
            decide_turn.THRESHOLDS = save
        out[t][0] += 1
        out[t][1] += int(r["correct"])
    return {t: {"n": n, "accuracy": round(k / n, 4) if n else None}
            for t, (n, k) in sorted(out.items())}


def report(rs: list[dict], question: str, model: str, high: float | None,
           medium: float | None) -> dict:
    alpha, src = _alpha()
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    qtype = rs[0]["type"] if rs else None
    out = {"tool": TUNE_VERSION, "question_set": question, "model": model,
           "type": qtype, "n": len(rs),
           "labels": dict(collections.Counter(r["how"] for r in rs)),
           "accuracy": (round(sum(r["correct"] for r in rs) / len(rs), 4)
                        if rs else None),
           "certainty": "noul" if qtype == "noul" else "confidence",
           "significance": {"alpha": alpha, "source": src, "z": round(z, 4),
                            "bound": "Wilson, lower end"}}
    if not rs:
        out["why"] = ("no labelled decisions for this question set and "
                      "model: label some first (bench/decider/label.py)")
        return out
    if len({r["type"] for r in rs}) > 1:
        out["why"] = "the rows mix question types; name one question"
        return out
    if qtype == "noul":
        xs = [(float(r["noul"]), r["correct"], r["truth"] == "true")
              for r in rs if r["noul"] is not None]
        top = [(max(float(r["noul"]), 1 - float(r["noul"])), r["correct"])
               for r in rs if r["noul"] is not None]
    else:
        xs = [(float(r["confidence"]), r["correct"], None) for r in rs
              if r["confidence"] is not None]
        top = [(max((r["p"] or {}).values() or [0.0]), r["correct"])
               for r in rs]
    out["bins"] = bins_of(xs)
    out["ece_top_label"] = ece(top)
    if high is None or medium is None:
        out["proposal"] = ("none: pass --high-accuracy and --medium-accuracy "
                           "(the owner's targets) for a proposal")
        return out
    p = propose(rs, high, medium, z)
    p["targets"] = {"high": high, "medium": medium}
    p["coverage"] = coverage(rs, p["row"]) if p["row"] else None
    p["paste"] = (f"# decide_turn.THRESHOLDS[{model!r}][{question!r}], "
                  f"proposed by {TUNE_VERSION} on n={len(rs)} labels at "
                  f"targets high {high} / medium {medium}; ACCEPTED BY: "
                  f"<owner, date>\n{json.dumps(p['row'])}") \
        if p["row"] else None
    out["proposal"] = p
    return out


def stats(rows: list[dict]) -> list[dict]:
    c = collections.Counter((r["model"] or "", r["family"], r["type"])
                            for r in rows)
    return [{"model": m, "question_set": f, "type": t, "labelled": n}
            for (m, f, t), n in sorted(c.items())]


def _print(rep: dict) -> None:
    print(f"{rep['question_set']} on {rep['model'] or '(model not recorded)'}"
          f" -- {rep['type']}, n={rep['n']} labelled, accuracy "
          f"{rep['accuracy']}")
    if rep.get("why"):
        print("  " + rep["why"])
        return
    print(f"  accuracy by {rep['certainty']} (10 equal-width bins); "
          f"top-label ECE {rep['ece_top_label']}")
    for b in rep["bins"]:
        if b["n"]:
            extra = (f"  truth=true {b['true_rate']}" if rep["type"] == "noul"
                     else "")
            print(f"    {b['range']:<11} n={b['n']:<4} mean {b['mean']}  "
                  f"accuracy {b['accuracy']}{extra}")
    p = rep["proposal"]
    if isinstance(p, str):
        print("  " + p)
        return
    print(f"  proposal at targets {p['targets']} (Wilson lower bound, "
          f"alpha {rep['significance']['alpha']}):")
    for k, v in p["evidence"].items():
        print(f"    {k:<12} " + (f"{v['t']}  (n={v['n']}, accuracy "
                                  f"{v['accuracy']}, lower {v['lower']})"
                                  if v else "not reached: stays untuned"))
    if p["coverage"]:
        print(f"  coverage on the labelled set: {json.dumps(p['coverage'])}")
    print("  NOT APPLIED. To accept, the owner pastes into "
          "mcp/decide_turn.py THRESHOLDS:" if p["paste"] else
          "  nothing to accept: no tier reached its target")
    if p["paste"]:
        print("    " + p["paste"].replace("\n", "\n    "))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--decisions", default=L.DECISIONS)
    ap.add_argument("--legacy", default=L.LEGACY)
    ap.add_argument("--labels", default=L.LABELS)
    ap.add_argument("--question", help="a question set: a family (phase, "
                    "build_intent, choose, pick, package) or a full name "
                    "(choose:stop)")
    ap.add_argument("--model", help="the model whose decisions to tune on "
                    "(default: the only one labelled)")
    ap.add_argument("--high-accuracy", type=float,
                    help="the owner's target for the high tier (act)")
    ap.add_argument("--medium-accuracy", type=float,
                    help="the owner's target for the medium tier (caution)")
    ap.add_argument("--reuse-by-state", action="store_true",
                    help="count a label made on another model's decision "
                    "of the same question and state")
    ap.add_argument("--stats", action="store_true",
                    help="labelled decisions per model and question set")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    ds, _joins = L.load_decisions(a.decisions, a.legacy)
    by_id, by_state = labels_by(a.labels)
    rows = labelled(ds, by_id, by_state, a.reuse_by_state)
    if a.stats or not a.question:
        s = stats(rows)
        print(json.dumps(s, indent=1) if a.json else "\n".join(
            f"{x['model'] or '-':<24} {x['question_set']:<16} {x['type']:<7}"
            f" {x['labelled']}" for x in s) or "no labelled decisions yet")
        return 0
    rs = select(rows, a.question, a.model)
    models = sorted({r["model"] or "" for r in rs})
    if a.model is None and len(models) > 1:
        print(f"labelled decisions of {a.question} come from {models}: pass "
              "--model (thresholds are per model)")
        return 2
    rep = report(rs, a.question, a.model or (models[0] if models else ""),
                 a.high_accuracy, a.medium_accuracy)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        _print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
