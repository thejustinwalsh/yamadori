#!/usr/bin/env python
"""Why the 'grounded' head fails, split by what the negative actually is.

    .venv-laya/Scripts/python.exe bench/laya_grounded_diagnostic.py

The grounded label set has two KINDS of negative, and lumping them together
hides which one is unlearnable:

  advice      generic knowledge dressed up as a file finding. Detectable from
              surface form alone -- "you should", "is important", "in general"
              -- and nothing about the cited file is needed to spot it.

  fabricated  confident, file-shaped specifics that are simply false:
              "MAX_TRACES is 1024" when it is 256. Telling this apart from a
              true specific requires the FILE, which the state does not carry.
              No head on these features can do it, because the information is
              not in the input. That is a data design fault, not a model one.

This script trains the same candidates on each sub-task separately. If
grounded-vs-advice is well above chance and grounded-vs-fabricated sits at
chance, the diagnosis is proven and the fix is to put file excerpts in the
state rather than to train harder.

Cross-validated over the whole subset rather than a single held-out split:
n is 36 per sub-task, and one 11-example test split would be noise.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from laya_head import PROMPT_REV, build_vector, render_state  # noqa: E402
from train_laya import CANDIDATES, fit_candidate, kfold, predict_probs  # noqa: E402

LABELS = os.path.join(ROOT, "bench", "laya_grounded_labels.jsonl")
CACHE = os.path.join(ROOT, "index", "laya", "features_grounded.json")


def subtype(rec: dict) -> str:
    if rec["label"] == "grounded":
        return "grounded"
    return "fabricated" if rec["why"].startswith("Fabricated") else "advice"


def main() -> None:
    rows = [json.loads(ln) for ln in open(LABELS, encoding="utf-8") if ln.strip()]
    if not os.path.exists(CACHE):
        raise SystemExit("no feature cache -- run scripts/train_laya.py first")
    blob = json.load(open(CACHE, encoding="utf-8"))
    if blob.get("prompt_rev") != PROMPT_REV:
        raise SystemExit("stale feature cache -- rerun scripts/train_laya.py")
    by_state = blob["by_state"]

    for rec in rows:
        rec["_state"] = render_state("grounded", rec)
        rec["_sub"] = subtype(rec)

    pos = [r for r in rows if r["_sub"] == "grounded"]
    for neg_kind in ("advice", "fabricated"):
        neg = [r for r in rows if r["_sub"] == neg_kind]
        sub = pos + neg
        y = [0] * len(pos) + [1] * len(neg)
        feats = [by_state[r["_state"]] for r in sub]
        idx = list(range(len(sub)))
        cv = kfold(y, idx, 5, 0)
        print(f"\n=== grounded ({len(pos)}) vs {neg_kind} ({len(neg)}) ===")
        print(f"    majority baseline {max(len(pos), len(neg)) / len(sub):.3f}")
        for kind, spec, wds in CANDIDATES:
            X = [build_vector(spec, f) for f in feats]
            for wd in wds:
                accs = []
                for tr, va in cv:
                    W, b = fit_candidate(kind, [X[i] for i in tr],
                                         [y[i] for i in tr], 2, wd, 0)
                    p = predict_probs(W, b, [X[i] for i in va])
                    accs.append(
                        sum(1 for pp, i in zip(p, va)
                            if (0 if pp[0] >= pp[1] else 1) == y[i]) / len(va))
                print(f"    {kind:22s} wd={str(wd):6s} cv_acc "
                      f"{sum(accs) / len(accs):.3f}")


if __name__ == "__main__":
    main()
