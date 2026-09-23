#!/usr/bin/env python
"""Does contrasting against the excerpt rescue the grounding head?

    .venv-laya/Scripts/python.exe bench/laya_grounded_contrast.py

THE PROBLEM THIS TESTS
----------------------
A grounding example is (finding, excerpt). The label depends on the ~20-word
finding; the ~1100-character excerpt is shared by both members of a pair and
carries the opposite label half the time. Measured on standardised pooled CLS
vectors, the two members of a pair sit at cosine 0.73 while unrelated examples
sit at 0.00 -- the representation is mostly the excerpt, and the claim that
decides the label barely moves it. A linear head cannot separate two points
that are nearly the same point.

THE CANDIDATE FIX
-----------------
Run the excerpt ALONE through the same stack and subtract:

    delta = cls(finding + excerpt) - cls(excerpt alone)

The shared excerpt component cancels and what remains is what the finding
contributed. If grounding is linearly decodable at all from these features,
delta is where it lives. This costs one extra forward pass per unique excerpt,
which is cheap and cacheable.

A NEGATIVE RESULT HERE IS STILL A RESULT: it would mean the grounding decision
is not linearly decodable from a frozen Laya and needs encoder training or a
System-2 check, and that is worth knowing before anyone spends another week on
it. Whatever this prints, write it into docs/LAYA.md.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from laya_head import PROMPT_REV, FeatureExtractor, render_state  # noqa: E402
from train_laya import (  # noqa: E402
    evaluate, fit_candidate, predict_probs,
    stratified_split,
)

LABELS = os.path.join(ROOT, "bench", "laya_grounded_excerpt_labels.jsonl")
CACHE = os.path.join(ROOT, "index", "laya", "features_grounded_excerpt.json")
CONTRAST_CACHE = os.path.join(ROOT, "index", "laya",
                              "features_grounded_excerpt_contrast.json")
TASK = "grounded_excerpt"


def excerpt_only_state(rec: dict) -> str:
    """The same rendering with the finding removed -- the shared component."""
    return render_state(TASK, {**rec, "finding": ""})


def main() -> None:
    rows = [json.loads(ln) for ln in open(LABELS, encoding="utf-8") if ln.strip()]
    full_cache = json.load(open(CACHE, encoding="utf-8"))["by_state"]

    base_states = sorted({excerpt_only_state(r) for r in rows})
    if os.path.exists(CONTRAST_CACHE):
        blob = json.load(open(CONTRAST_CACHE, encoding="utf-8"))
        base = blob["by_state"] if blob.get("prompt_rev") == PROMPT_REV else {}
    else:
        base = {}
    missing = [s for s in base_states if s not in base]
    if missing:
        print(f"  extracting {len(missing)} excerpt-only state(s)...")
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES",
                              os.environ.get("LAYA_GPU", "1"))
        import laya

        fx = FeatureExtractor(laya.load(
            os.environ.get("LAYA_MODEL", "convaiinnovations/laya")))
        for s, f in zip(missing, fx.features(TASK, missing)):
            base[s] = f
        json.dump({"prompt_rev": PROMPT_REV, "by_state": base},
                  open(CONTRAST_CACHE, "w", encoding="utf-8"))
        print(f"  contrast cache -> {CONTRAST_CACHE}")

    y = [0 if r["label"] == "grounded" else 1 for r in rows]
    groups = [r["pair"] for r in rows]

    def vec(r: dict, spec: str):
        f = full_cache[render_state(TASK, r)]
        g = base[excerpt_only_state(r)]
        if spec == "delta_cls":
            return [a - b for a, b in zip(f["cls"], g["cls"])]
        if spec == "delta_logits":
            return [a - b for a, b in zip(f["logits"], g["logits"])]
        if spec == "delta_cls_logits":
            return ([a - b for a, b in zip(f["cls"], g["cls"])]
                    + [a - b for a, b in zip(f["logits"], g["logits"])])
        if spec == "cls_plus_delta":
            return list(f["cls"]) + [a - b for a, b in
                                     zip(f["cls"], g["cls"])]
        raise ValueError(spec)

    specs = ["delta_cls", "delta_logits", "delta_cls_logits", "cls_plus_delta"]
    kinds = [("logistic", [1e-3, 1e-2, 1e-1, 1.0]),
             ("centroid_cls", [None])]

    print(f"\n  {len(rows)} rows, {len(set(groups))} pairs, "
          f"majority baseline 0.500")
    print("\n  5-seed group-aware repeated holdout (test_frac 0.3):")
    best = None
    for spec in specs:
        X = [vec(r, spec) for r in rows]
        for kind, wds in kinds:
            for wd in wds:
                accs = []
                for seed in range(5):
                    tr, te = stratified_split(y, 0.3, seed, groups)
                    kd = "logistic_cls" if kind == "logistic" else kind
                    W, b = fit_candidate(kd, [X[i] for i in tr],
                                         [y[i] for i in tr], 2, wd, seed)
                    p = predict_probs(W, b, [X[i] for i in te])
                    accs.append(evaluate(p, [y[i] for i in te], 0.0)["accuracy"])
                mean = sum(accs) / len(accs)
                var = sum((a - mean) ** 2 for a in accs) / len(accs)
                print(f"    {spec:18s} {kind:13s} wd={str(wd):6s} "
                      f"acc {mean:.3f} +/-{var ** 0.5:.3f}")
                if best is None or mean > best[0]:
                    best = (mean, spec, kind, wd)

    print(f"\n  best: {best[1]} / {best[2]} (wd={best[3]}) "
          f"held-out acc {best[0]:.3f} vs 0.500 baseline")
    if best[0] < 0.60:
        print("  VERDICT: grounding is NOT linearly decodable from frozen "
              "Laya features, even after contrasting away the excerpt.\n"
              "  Do not ship a grounding gate on this checkpoint.")
    else:
        print("  VERDICT: contrast features recover real signal -- fold "
              "'delta_cls' into laya_head.build_vector and retrain.")


if __name__ == "__main__":
    main()
