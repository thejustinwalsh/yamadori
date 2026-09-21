#!/usr/bin/env python
"""Measure, then fix, the decision model's calibration on our own data.

WHY CALIBRATION BEFORE FINE-TUNING

The model card reports raw ECE of 0.466 before temperature refitting, and
independent evaluation found the model "often confidently wrong" with 54% of
answers at confidence >= 0.90 being wrong. Those are calibration failures, not
capability failures, and temperature scaling is the standard fix: it changes no
weights, needs no gradients, and cannot make the ranking worse because a single
scalar divisor is monotonic. Whatever the model already knows, it keeps.

Published results put this in perspective: temperature scaling took a network
from 16.53% ECE to 1.26% in-domain. The catch is that it does NOT transfer --
the same technique leaves out-of-domain ECE at 3.61-12.83% -- so it has to be
fitted on data from the distribution we actually serve, which is what the
generated pairs are.

RUN IT ON PAIRS, NOT ON A BAG OF EXAMPLES

Both members of a contrastive pair must land in the same split. Training on one
member and testing on the other leaks the answer: the two differ by a single
token, so a model that memorised one has effectively seen the other.

    .venv-laya/Scripts/python.exe scripts/calibrate_laya.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "..", "index", "pairs.jsonl")
OUT = os.path.join(HERE, "..", "index", "calibration.json")

QUESTION = {
    "correct": {
        "type": "choice",
        "instructions": "Is this code correct, or does it contain a defect?",
        "criteria": {"yes": "the code is correct",
                     "no": "the code contains a defect"},
    }
}


def load_pairs() -> list[list[dict]]:
    rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)
    return [v for v in by_pair.values() if len(v) == 2]


def ece(probs: list[float], labels: list[int], bins: int = 10) -> float:
    """Expected calibration error: how far confidence is from accuracy."""
    total = 0.0
    n = len(probs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo < p <= hi or (b == 0 and p == 0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(conf - acc)
    return total


def apply_temp(p: float, t: float) -> float:
    """Temperature on a binary probability, via its logit."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p)) / t
    return 1 / (1 + math.exp(-logit))


def nll(probs: list[float], labels: list[int], t: float) -> float:
    s = 0.0
    for p, y in zip(probs, labels):
        q = apply_temp(p, t)
        q = min(max(q, 1e-9), 1 - 1e-9)
        s -= math.log(q) if y else math.log(1 - q)
    return s / max(len(probs), 1)


def main() -> None:
    sys.path.insert(0, os.path.join(HERE, "..", "mcp"))
    import laya

    pairs = load_pairs()
    if not pairs:
        print(f"  no pairs at {PAIRS} -- run scripts/make_pairs.py first")
        return
    rnd = random.Random(0)
    rnd.shuffle(pairs)
    # Split BY PAIR, so the two near-identical members never straddle it.
    cut = int(len(pairs) * 0.5)
    fit_pairs, test_pairs = pairs[:cut], pairs[cut:]
    print(f"  {len(pairs)} pairs   fit={len(fit_pairs)} test={len(test_pairs)}")

    agent = laya.load()

    def score(ps: list[list[dict]]) -> tuple[list[float], list[int]]:
        probs, labels = [], []
        for pair in ps:
            for member in pair:
                out = agent.predict(member["state"], QUESTION)["answers"]["correct"]
                p_yes = float((out.get("probabilities") or {}).get("yes", 0.5))
                probs.append(p_yes)
                labels.append(1 if member["label"] == "yes" else 0)
        return probs, labels

    print("  scoring fit split...")
    fp, fl = score(fit_pairs)
    print("  scoring test split...")
    tp, tl = score(test_pairs)

    def acc(probs, labels, t=1.0):
        return sum(1 for p, y in zip(probs, labels)
                   if (apply_temp(p, t) >= 0.5) == bool(y)) / max(len(probs), 1)

    base_acc, base_ece = acc(tp, tl), ece(tp, tl)
    # Majority class is the honest floor: half the examples are correct code by
    # construction, so anything at 0.50 has learned nothing.
    print(f"\n  BEFORE   accuracy {base_acc:.3f}   ECE {base_ece:.3f}"
          f"   (majority baseline 0.500)")

    best_t, best = 1.0, nll(fp, fl, 1.0)
    t = 0.05
    while t <= 10.0:
        v = nll(fp, fl, t)
        if v < best:
            best, best_t = v, t
        t += 0.05
    cal_probs = [apply_temp(p, best_t) for p in tp]
    print(f"  AFTER    accuracy {acc(tp, tl, best_t):.3f}   "
          f"ECE {ece(cal_probs, tl):.3f}   temperature {best_t:.2f}")

    # Accuracy cannot move: temperature is monotonic, so it rescales confidence
    # without reordering anything. If it did move, the fit is broken.
    if abs(acc(tp, tl, best_t) - base_acc) > 1e-9:
        print("  WARNING: accuracy changed under a monotonic transform -- bug")

    json.dump({"temperature": best_t, "fit_pairs": len(fit_pairs),
               "test_pairs": len(test_pairs),
               "ece_before": round(base_ece, 4),
               "ece_after": round(ece(cal_probs, tl), 4),
               "accuracy": round(base_acc, 4)},
              open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\n  written {OUT}")
    if base_acc <= 0.55:
        print("  NOTE: accuracy is at the majority baseline. Calibration makes "
              "the confidence honest, it cannot create capability -- this "
              "checkpoint needs fine-tuning on these pairs, not just scaling.")


if __name__ == "__main__":
    main()
