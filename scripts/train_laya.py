#!/usr/bin/env python
"""Train a decision head for Laya on labelled routing data, and prove it helps.

    .venv-laya/Scripts/python.exe scripts/train_laya.py --task all

WHY A TRAINED HEAD AND NOT MORE PROMPTING
-----------------------------------------
Laya is an option-scoring cross-encoder (see mcp/laya_head.py for the evidence).
Its label space is open, which is what makes zero-shot prompting possible and
also what makes it mediocre: nothing in the checkpoint has ever seen OUR
routing boundary. Measured zero-shot on this repo's questions, it abstains on
most realistic cases because the top-two margin sits under the gate.

Prior work here reached for temperature scaling. Temperature is monotonic, so
it rescales confidence and CANNOT move accuracy -- index/calibration.json
records exactly that outcome: ECE 0.279 -> 0.032, accuracy 0.507, i.e. chance.
A head with more than one parameter is required to move the decision itself.

WHAT IS FITTED
--------------
Candidates, all of which reduce to `softmax(W x + b)` so they share one
artefact format and one serving path:

    temperature     W = I/T                  1 param    calibration only
    vector_scale    affine on the k logits   k*k+k      fixes option bias
    logistic_cls    linear on pooled CLS     k*1024     learns a new boundary
    logistic_cls_logits   linear on both
    logistic_markers      linear on the per-option marker states
    centroid_cls    nearest centroid, which is exactly affine after the
                    constant -0.5|x|^2 cancels inside the softmax

HONEST MODEL SELECTION
----------------------
Choosing the candidate on the held-out split and then reporting that split
would be selection bias dressed as a result. So: stratified train/test split,
K-fold cross-validation INSIDE train picks the candidate, the L2 strength and
the abstain gate, then the winner is refitted on all of train and scored ONCE
on test. Test is touched exactly once per run.

THE GATE
--------
An abstain gate trades coverage for accuracy, so it needs an objective rather
than a taste. Utility here is +1 for an answered-and-correct decision, -1 for
answered-and-wrong, 0 for abstained: a gate is only worth raising if the
decisions it removes were worse than coin-flips at the margin. Coverage is
reported alongside so the trade is visible.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

from laya_head import (  # noqa: E402
    ARTEFACT_VERSION,
    CONTRAST_SPECS,
    PROMPT_REV,
    TASKS,
    FeatureExtractor,
    TrainedHead,
    build_vector,
    render_base_state,
    render_state,
)

INDEX_DIR = os.path.join(ROOT, "index", "laya")
BENCH = os.path.join(ROOT, "bench")

LABEL_GLOBS = {
    "route_in": ["laya_routing_labels*.jsonl"],
    "grounded": ["laya_grounded_labels*.jsonl"],
    "grounded_excerpt": ["laya_grounded_excerpt_labels*.jsonl"],
}

# Tasks trained by `--task all`. "grounded" is excluded on purpose: it is a
# measured negative result kept for the diagnostic, not something to ship.
# See docs/LAYA.md, "Grounding without the file is unlearnable".
DEFAULT_TASKS = ["route_in", "grounded_excerpt"]


# ------------------------------------------------------------ data ---------


def load_labels(task: str) -> List[Dict[str, Any]]:
    """Every matching jsonl under bench/, so adding a file adds labels.

    Deliberately a glob: another workstream owns
    bench/laya_routing_labels.jsonl and is still appending to it. Reading a
    pattern rather than one path means neither agent has to edit the other's
    file to contribute labels.
    """
    rows: List[Dict[str, Any]] = []
    seen = set()
    for pat in LABEL_GLOBS[task]:
        for path in sorted(glob.glob(os.path.join(BENCH, pat))):
            with open(path, encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise SystemExit(f"{path}:{ln}: bad JSON: {e}")
                    if rec.get("label") not in TASKS[task]["labels"]:
                        raise SystemExit(
                            f"{path}:{ln}: label {rec.get('label')!r} not in "
                            f"{TASKS[task]['labels']}"
                        )
                    state = render_state(task, rec)
                    if state in seen:          # duplicates skew the split
                        continue
                    seen.add(state)
                    rec["_state"] = state
                    rec["_source"] = os.path.basename(path)
                    # Rows sharing a 'pair' share an excerpt and must stay on
                    # the same side of every split.
                    rec["_group"] = rec.get("pair", f"__solo_{len(rows)}")
                    rec["_base_state"] = render_base_state(task, rec)
                    rows.append(rec)
    return rows


def cache_path(task: str) -> str:
    return os.path.join(INDEX_DIR, f"features_{task}.json")


def get_features(task: str, rows: List[Dict[str, Any]], use_cache: bool
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract (or reuse) frozen features. Cache is keyed on the state text."""
    cp = cache_path(task)
    cache: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    if use_cache and os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            blob = json.load(fh)
        if blob.get("prompt_rev") == PROMPT_REV:
            cache = blob.get("by_state", {})
            meta = blob.get("meta", {})
        else:
            print("  prompt_rev changed -- ignoring stale feature cache")

    # Contrast states are cached in the same store, keyed by their own text, so
    # the excerpt-only pass is computed once no matter how many findings cite
    # the same excerpt.
    wanted: List[str] = []
    for r in rows:
        wanted.append(r["_state"])
        if r.get("_base_state"):
            wanted.append(r["_base_state"])
    missing_states = [s for s in dict.fromkeys(wanted) if s not in cache]

    if missing_states:
        print(f"  extracting features for {len(missing_states)} state(s)...")
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES",
                              os.environ.get("LAYA_GPU", "1"))
        import laya

        agent = laya.load(os.environ.get("LAYA_MODEL", "convaiinnovations/laya"))
        meta = {
            "device": str(agent.device),
            "dtype": str(agent.dtype),
            "encoder": agent.cfg.get("encoder"),
            "max_len": agent.cfg.get("max_len", 512),
        }
        # The service's own zero-shot path divides logits by this. Recorded so
        # the BEFORE numbers are the service's numbers, not an approximation.
        from laya.common import QTYPES, temp_bucket

        qt = QTYPES[TASKS[task]["question"]["type"]]
        k = len(TASKS[task]["labels"])
        meta["t_scale"] = float(
            agent.cfg.get("temperature_by_options", {}).get(
                temp_bucket(qt, k),
                agent.cfg.get("temperature", [1.0, 1.0, 1.0])[qt],
            )
        )
        fx = FeatureExtractor(agent)
        for s, f in zip(missing_states, fx.features(task, missing_states)):
            cache[s] = f
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump({"prompt_rev": PROMPT_REV, "meta": meta,
                       "by_state": cache}, fh)
        print(f"  feature cache -> {cp}")
    else:
        print(f"  feature cache hit for all {len(rows)} example(s)")

    out = []
    for r in rows:
        f = dict(cache[r["_state"]])
        if r.get("_base_state"):
            g = cache[r["_base_state"]]
            f["cls_base"] = g["cls"]
            f["logits_base"] = g["logits"]
        out.append(f)
    return out, meta


# --------------------------------------------------------- metrics ---------


def ece(conf: Sequence[float], correct: Sequence[int], bins: int = 10) -> float:
    n = len(conf)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf)
               if (lo < c <= hi) or (b == 0 and c <= 0)]
        if not idx:
            continue
        cf = sum(conf[i] for i in idx) / len(idx)
        ac = sum(correct[i] for i in idx) / len(idx)
        total += (len(idx) / n) * abs(cf - ac)
    return total


def quantiles(xs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)

    def q(p: float) -> float:
        i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return s[i]

    return {"min": round(s[0], 4), "p25": round(q(0.25), 4),
            "median": round(q(0.5), 4), "p75": round(q(0.75), 4),
            "max": round(s[-1], 4), "mean": round(sum(s) / len(s), 4)}


def evaluate(probs: List[List[float]], y: List[int], gate: float
             ) -> Dict[str, Any]:
    n = len(y)
    pred, conf, margin, correct = [], [], [], []
    for p, yy in zip(probs, y):
        order = sorted(range(len(p)), key=lambda i: -p[i])
        pred.append(order[0])
        conf.append(p[order[0]])
        margin.append(p[order[0]] - (p[order[1]] if len(p) > 1 else 0.0))
        correct.append(1 if order[0] == yy else 0)
    answered = [i for i in range(n) if margin[i] >= gate]
    acc = sum(correct) / n if n else float("nan")
    acc_ans = (sum(correct[i] for i in answered) / len(answered)
               if answered else float("nan"))
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "abstention_rate": round(1 - len(answered) / n, 4) if n else None,
        "coverage": round(len(answered) / n, 4) if n else None,
        "accuracy_on_answered": (round(acc_ans, 4) if answered else None),
        "ece": round(ece(conf, correct), 4),
        "margin": quantiles(margin),
        "gate": gate,
        "utility": sum(1 if correct[i] else -1 for i in answered),
    }


# ----------------------------------------------------------- heads ---------


def _fit_linear(X: List[List[float]], y: List[int], k: int, wd: float,
                steps: int = 600, seed: int = 0) -> Tuple[List[List[float]], List[float]]:
    """Multinomial logistic regression, standardised then folded back.

    Standardisation is folded into the returned weights so the served head is
    a plain affine map and needs no stored feature statistics -- one less thing
    that can drift between training and serving.
    """
    import torch

    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    mu = Xt.mean(0)
    sd = Xt.std(0).clamp_min(1e-6)
    Xs = (Xt - mu) / sd

    lin = torch.nn.Linear(Xs.shape[1], k)
    torch.nn.init.zeros_(lin.weight)
    torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(lin(Xs), yt)
        loss.backward()
        opt.step()

    W = lin.weight.detach()
    b = lin.bias.detach()
    W_eff = W / sd
    b_eff = b - (W * mu / sd).sum(1)
    return W_eff.tolist(), b_eff.tolist()


def _fit_centroid(X: List[List[float]], y: List[int], k: int, scale: float
                  ) -> Tuple[List[List[float]], List[float]]:
    """Nearest Euclidean centroid, written as the affine map it already is."""
    import torch

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    cents = []
    for c in range(k):
        sel = Xt[yt == c]
        cents.append(sel.mean(0) if len(sel) else torch.zeros(Xt.shape[1]))
    C = torch.stack(cents)
    W = C / scale
    b = -0.5 * (C * C).sum(1) / scale
    return W.tolist(), b.tolist()


def _fit_temperature(Z: List[List[float]], y: List[int], k: int
                     ) -> Tuple[List[List[float]], List[float], float]:
    """Scalar temperature on the raw marker logits: W = I/T, b = 0."""
    best_t, best_nll = 1.0, float("inf")
    t = 0.05
    while t <= 10.0:
        s = 0.0
        for z, yy in zip(Z, y):
            zz = [v / t for v in z]
            m = max(zz)
            ex = [math.exp(v - m) for v in zz]
            s -= math.log(max(ex[yy] / sum(ex), 1e-12))
        if s < best_nll:
            best_nll, best_t = s, t
        t += 0.05
    W = [[(1.0 / best_t) if i == j else 0.0 for j in range(k)] for i in range(k)]
    return W, [0.0] * k, best_t


def predict_probs(W: List[List[float]], b: List[float],
                  X: List[List[float]]) -> List[List[float]]:
    out = []
    for x in X:
        z = [sum(w * v for w, v in zip(row, x)) + bb for row, bb in zip(W, b)]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        out.append([v / s for v in e])
    return out


CANDIDATES = [
    ("temperature", "logits", [None]),
    ("vector_scale", "logits", [1e-4, 1e-3, 1e-2, 1e-1]),
    ("logistic_cls", "cls", [1e-3, 1e-2, 1e-1, 1.0]),
    ("logistic_cls_logits", "cls_logits", [1e-3, 1e-2, 1e-1, 1.0]),
    ("logistic_markers", "markers", [1e-2, 1e-1, 1.0]),
    ("centroid_cls", "cls", [None]),
    # Contrast heads: only offered for tasks that produce a base pass. On
    # grounding these are the only specs that beat chance, because the raw
    # pooled state is dominated by the shared excerpt.
    ("logistic_delta_cls", "delta_cls", [1e-3, 1e-2, 1e-1]),
    ("logistic_delta_cls_logits", "delta_cls_logits", [1e-3, 1e-2, 1e-1]),
]


def rule_baseline_preds(task: str, rows: List[Dict[str, Any]],
                        labels: List[str]) -> List[int] | None:
    """The six-line regex (mcp/selection.py, ONE copy), on our rows.

    A trained 421M head that cannot beat a regex is not worth serving, so the
    regex is scored on the SAME held-out splits rather than quoted from
    another document against another label set. Returns None when the harness
    is unavailable, so training never depends on another workstream's file.
    """
    if task != "route_in":
        return None
    try:
        # The same function the proxy's selection engine routes with, not a
        # copy of it: bench/laya_calibration.py imports it from here too.
        sys.path.insert(0, os.path.join(ROOT, "mcp"))
        from selection import rule_baseline  # type: ignore

        return [labels.index(rule_baseline(r)) for r in rows]
    except Exception as e:                                       # noqa: BLE001
        print(f"  (rule baseline unavailable: {type(e).__name__}: {e})")
        return None


def usable_candidates(feats: List[Dict[str, Any]]):
    """Drop contrast candidates when the features carry no contrast pass."""
    has_contrast = bool(feats) and "cls_base" in feats[0]
    return [c for c in CANDIDATES
            if has_contrast or c[1] not in CONTRAST_SPECS]


def fit_candidate(kind: str, X: List[List[float]], y: List[int], k: int,
                  wd: Any, seed: int):
    if kind == "temperature":
        W, b, _ = _fit_temperature(X, y, k)
        return W, b
    if kind == "centroid_cls":
        return _fit_centroid(X, y, k, scale=50.0)
    # every remaining kind is multinomial logistic; they differ only in which
    # feature vector they are handed
    return _fit_linear(X, y, k, wd=float(wd), seed=seed)


# ------------------------------------------------------------ split --------


def _groups_of(n: int, groups: List[Any] | None) -> List[Any]:
    """No grouping means every row is its own group, which is the plain case."""
    return list(range(n)) if groups is None else groups


def _by_group(y: List[int], groups: List[Any], idx: List[int]
              ) -> Dict[Any, List[int]]:
    out: Dict[Any, List[int]] = {}
    for i in idx:
        out.setdefault(groups[i], []).append(i)
    return out


def stratified_split(y: List[int], test_frac: float, seed: int,
                     groups: List[Any] | None = None
                     ) -> Tuple[List[int], List[int]]:
    """Stratified split that never splits a group across the boundary.

    Grouping is load-bearing for the excerpt-based grounded task: both members
    of a pair share one file excerpt, so putting one in train and the other in
    test would let a head that memorised the excerpt look like it generalised.
    A group is stratified by the label of its first member, which is exact when
    groups are single-label and a reasonable proxy otherwise.
    """
    rnd = random.Random(seed)
    g = _groups_of(len(y), groups)
    gmap = _by_group(y, g, list(range(len(y))))
    by_class: Dict[int, List[Any]] = {}
    for gid, members in gmap.items():
        by_class.setdefault(y[members[0]], []).append(gid)
    train, test = [], []
    for c, gids in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        gids = sorted(gids, key=str)
        rnd.shuffle(gids)
        cut = max(1, int(round(len(gids) * test_frac)))
        for gid in gids[:cut]:
            test.extend(gmap[gid])
        for gid in gids[cut:]:
            train.extend(gmap[gid])
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


def kfold(y: List[int], idx: List[int], k: int, seed: int,
          groups: List[Any] | None = None
          ) -> List[Tuple[List[int], List[int]]]:
    """K-fold that keeps a group whole, for the same reason as the split."""
    rnd = random.Random(seed)
    g = _groups_of(max(idx) + 1 if idx else 0, groups)
    gmap = _by_group(y, g, idx)
    by_class: Dict[int, List[Any]] = {}
    for gid, members in gmap.items():
        by_class.setdefault(y[members[0]], []).append(gid)
    folds: List[List[int]] = [[] for _ in range(k)]
    for c, gids in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        gids = sorted(gids, key=str)
        rnd.shuffle(gids)
        for j, gid in enumerate(gids):
            folds[j % k].extend(gmap[gid])
    out = []
    for j in range(k):
        val = folds[j]
        tr = [i for jj in range(k) if jj != j for i in folds[jj]]
        if val and tr:
            out.append((tr, val))
    return out


def best_gate(probs: List[List[float]], y: List[int]) -> float:
    """Gate maximising utility (+1 right, -1 wrong, 0 abstained)."""
    best_g, best_u = 0.0, None
    g = 0.0
    while g <= 0.75:
        u = 0
        for p, yy in zip(probs, y):
            o = sorted(range(len(p)), key=lambda i: -p[i])
            margin = p[o[0]] - (p[o[1]] if len(p) > 1 else 0.0)
            if margin >= g:
                u += 1 if o[0] == yy else -1
        if best_u is None or u > best_u:
            best_u, best_g = u, g
        g += 0.05
    return round(best_g, 2)


# ------------------------------------------------------------- run ---------


def _select(feats, y, tr_idx, groups, folds, seed, k):
    """CV inside `tr_idx` only: pick candidate, L2 strength and gate."""
    cv = kfold(y, tr_idx, folds, seed, groups)
    results = []
    for kind, spec, wds in usable_candidates(feats):
        X = [build_vector(spec, f) for f in feats]
        for wd in wds:
            accs, oof_probs, oof_y = [], [], []
            for tr, va in cv:
                W, b = fit_candidate(kind, [X[i] for i in tr],
                                     [y[i] for i in tr], k, wd, seed)
                p = predict_probs(W, b, [X[i] for i in va])
                oof_probs.extend(p)
                oof_y.extend(y[i] for i in va)
                accs.append(sum(1 for pp, i in zip(p, va)
                                if max(range(k), key=lambda c: pp[c]) == y[i])
                            / len(va))
            results.append({"kind": kind, "spec": spec, "wd": wd,
                            "cv_accuracy": round(sum(accs) / len(accs), 4),
                            "cv_ece": evaluate(oof_probs, oof_y, 0.0)["ece"],
                            "oof_probs": oof_probs, "oof_y": oof_y})
    results.sort(key=lambda r: (-r["cv_accuracy"], r["cv_ece"]))
    return results


def train_task(task: str, test_frac: float, seed: int, folds: int,
               use_cache: bool, out_dir: str, repeats: int = 8
               ) -> Dict[str, Any]:
    labels = TASKS[task]["labels"]
    k = len(labels)
    print(f"\n=== {task} ===")
    rows = load_labels(task)
    if len(rows) < 20:
        raise SystemExit(
            f"only {len(rows)} labels for {task!r}; need at least 20. "
            f"Add rows to bench/{LABEL_GLOBS[task][0]}"
        )
    y = [labels.index(r["label"]) for r in rows]
    dist = {lab: y.count(i) for i, lab in enumerate(labels)}
    srcs = sorted({r["_source"] for r in rows})
    print(f"  {len(rows)} labels from {', '.join(srcs)}   {dist}")

    feats, meta = get_features(task, rows, use_cache)
    t_scale = float(meta.get("t_scale", 1.0))

    groups = [r["_group"] for r in rows]
    n_groups = len(set(groups))

    # The zero-shot service decision, exactly as it is served: the same
    # temperature bucket the agent applies, so BEFORE is the real baseline and
    # not a reimplementation of one.
    zs_probs_all = []
    for f in feats:
        z = [v / max(1e-3, t_scale) for v in f["logits"]]
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        zs_probs_all.append([v / s for v in e])

    # ---- repeated holdout ------------------------------------------------
    #
    # One split of ~20 held-out examples moves by 0.10 accuracy when two rows
    # land differently, which is how a result gets declared and then fails to
    # reproduce. Every seed re-splits, re-selects the candidate by CV INSIDE
    # its own train side, refits and scores its own test side, so the spread
    # across seeds is an honest error bar rather than decoration.
    rule_preds = rule_baseline_preds(task, rows, labels)

    reps = []
    for r in range(repeats):
        s_r = seed + r
        tr_idx, te_idx = stratified_split(y, test_frac, s_r, groups)
        res = _select(feats, y, tr_idx, groups, folds, s_r, k)
        w = res[0]
        g = best_gate(w["oof_probs"], w["oof_y"])
        X = [build_vector(w["spec"], f) for f in feats]
        W_r, b_r = fit_candidate(w["kind"], [X[i] for i in tr_idx],
                                 [y[i] for i in tr_idx], k, w["wd"], s_r)
        te_probs = predict_probs(W_r, b_r, [X[i] for i in te_idx])
        bef = evaluate([zs_probs_all[i] for i in te_idx],
                       [y[i] for i in te_idx], gate=0.3)
        aft = evaluate(te_probs, [y[i] for i in te_idx], gate=g)
        rule_acc = None
        if rule_preds is not None:
            rule_acc = round(sum(1 for i in te_idx if rule_preds[i] == y[i])
                             / len(te_idx), 4)

        # The adversarial slice, where the regex is documented to collapse.
        # Average accuracy hides it: a rule that wins overall and scores zero
        # on the items written to defeat surface cues is not the same product
        # as one that degrades gracefully.
        hard_te = [i for i in te_idx if rows[i].get("hard")]
        if hard_te:
            pred_h = predict_probs(W_r, b_r, [X[i] for i in hard_te])
            aft["hard_n"] = len(hard_te)
            aft["hard_accuracy"] = round(
                sum(1 for p, i in zip(pred_h, hard_te)
                    if max(range(k), key=lambda c: p[c]) == y[i])
                / len(hard_te), 4)
            bef["hard_accuracy"] = round(
                sum(1 for i in hard_te
                    if max(range(k), key=lambda c: zs_probs_all[i][c]) == y[i])
                / len(hard_te), 4)
            if rule_preds is not None:
                aft["hard_rule_accuracy"] = round(
                    sum(1 for i in hard_te if rule_preds[i] == y[i])
                    / len(hard_te), 4)
        reps.append({"seed": s_r, "kind": w["kind"], "wd": w["wd"], "gate": g,
                     "n_train": len(tr_idx), "n_test": len(te_idx),
                     "cv_accuracy": w["cv_accuracy"], "rule_accuracy": rule_acc,
                     "before": bef, "after": aft})
        print(f"  seed {s_r}: train={len(tr_idx)} test={len(te_idx)}  "
              f"pick={w['kind']}(wd={w['wd']}) gate={g:.2f}  "
              f"acc {bef['accuracy']:.3f} -> {aft['accuracy']:.3f}  "
              f"abstain {bef['abstention_rate']:.3f} -> "
              f"{aft['abstention_rate']:.3f}")

    def agg(which: str, field: str) -> Dict[str, float]:
        vals = [rp[which][field] for rp in reps
                if rp[which][field] is not None]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4),
                "min": round(min(vals), 4), "max": round(max(vals), 4)}

    before_agg = {f: agg("before", f) for f in
                  ("accuracy", "abstention_rate", "ece",
                   "accuracy_on_answered", "coverage")}
    after_agg = {f: agg("after", f) for f in
                 ("accuracy", "abstention_rate", "ece",
                  "accuracy_on_answered", "coverage")}
    before_agg["margin_median"] = agg_margin_before = {
        "mean": round(sum(rp["before"]["margin"]["median"] for rp in reps)
                      / len(reps), 4)}
    after_agg["margin_median"] = {
        "mean": round(sum(rp["after"]["margin"]["median"] for rp in reps)
                      / len(reps), 4)}
    del agg_margin_before

    print(f"  BEFORE (zero-shot, gate 0.30): acc "
          f"{before_agg['accuracy']['mean']:.3f} "
          f"+/-{before_agg['accuracy']['std']:.3f}   abstain "
          f"{before_agg['abstention_rate']['mean']:.3f}   ECE "
          f"{before_agg['ece']['mean']:.3f}")
    print(f"  AFTER  (trained):              acc "
          f"{after_agg['accuracy']['mean']:.3f} "
          f"+/-{after_agg['accuracy']['std']:.3f}   abstain "
          f"{after_agg['abstention_rate']['mean']:.3f}   ECE "
          f"{after_agg['ece']['mean']:.3f}")
    rule_agg = None
    if rule_preds is not None:
        vals = [rp["rule_accuracy"] for rp in reps]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        rule_agg = {"mean": round(mean, 4), "std": round(var ** 0.5, 4)}
        verdict = ("trained head WINS" if after_agg["accuracy"]["mean"] > mean
                   else "REGEX WINS -- do not ship the head")
        print(f"  RULE   (regex, no model):      acc {mean:.3f} "
              f"+/-{var ** 0.5:.3f}   [{verdict}]")

    hard_rows = [rp for rp in reps if rp["after"].get("hard_n")]
    if hard_rows:
        def hmean(where: str, key: str):
            v = [rp[where][key] for rp in hard_rows if key in rp[where]]
            return sum(v) / len(v) if v else None

        nh = sum(rp["after"]["hard_n"] for rp in hard_rows) / len(hard_rows)
        h_zs, h_tr = hmean("before", "hard_accuracy"), hmean("after", "hard_accuracy")
        h_rule = hmean("after", "hard_rule_accuracy")
        after_agg["hard_accuracy"] = {"mean": round(h_tr, 4)}
        before_agg["hard_accuracy"] = {"mean": round(h_zs, 4)}
        if h_rule is not None:
            after_agg["hard_rule_accuracy"] = {"mean": round(h_rule, 4)}
        print(f"  ADVERSARIAL slice (~{nh:.0f}/split): zero-shot "
              f"{h_zs:.3f}   trained {h_tr:.3f}" +
              (f"   regex {h_rule:.3f}" if h_rule is not None else ""))

    # ---- the shipped artefact ------------------------------------------
    #
    # Fitted on EVERY label, after the repeated holdout above has already
    # established what that recipe generalises to. Holding a permanent test
    # split back would throw away a third of a very small label set to
    # re-measure a number we just measured five times.
    full_idx = list(range(len(rows)))
    res_full = _select(feats, y, full_idx, groups, folds, seed, k)
    win = res_full[0]
    gate = best_gate(win["oof_probs"], win["oof_y"])
    for rr in res_full[:6]:
        print(f"    {rr['kind']:22s} wd={str(rr['wd']):6s} "
              f"cv_acc {rr['cv_accuracy']:.3f}  cv_ece {rr['cv_ece']:.3f}")
    print(f"  shipped: {win['kind']} (wd={win['wd']})  "
          f"cv_acc {win['cv_accuracy']:.3f}  gate {gate:.2f}")
    X = [build_vector(win["spec"], f) for f in feats]
    W, b = fit_candidate(win["kind"], X, y, k, win["wd"], seed)

    majority = max(dist.values()) / len(rows)
    blob = {
        "task": task,
        "labels": labels,
        "kind": win["kind"],
        "feature_spec": win["spec"],
        "weight_decay": win["wd"],
        "artefact_version": ARTEFACT_VERSION,
        "prompt_rev": PROMPT_REV,
        "W": W,
        "b": b,
        "gate": gate,
        "trained_on": {"n_total": len(rows), "n_groups": n_groups,
                       "sources": srcs,
                       "class_distribution": dist, "seed": seed,
                       "test_frac": test_frac, "folds": folds,
                       "repeats": repeats,
                       "fitted_on": "all labels; generalisation estimated by "
                                    "repeated group-aware holdout"},
        "metrics": {
            "majority_baseline": round(majority, 4),
            "rule_baseline": rule_agg,
            "cv_accuracy": win["cv_accuracy"],
            "held_out_before": before_agg,
            "held_out_after": after_agg,
            "per_seed": [{kk: rp[kk] for kk in
                          ("seed", "kind", "wd", "gate", "n_test")}
                         | {"before_accuracy": rp["before"]["accuracy"],
                            "after_accuracy": rp["after"]["accuracy"]}
                         for rp in reps],
        },
        "feature_meta": meta,
    }
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{task}.json")
    TrainedHead(blob).save(out)
    print(f"  artefact -> {out}")
    return blob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="all",
                    choices=["all", "every"] + list(TASKS.keys()),
                    help="'all' trains the shippable tasks; 'every' also "
                         "trains the known-negative 'grounded' task")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=8,
                    help="held-out splits to average over (error bars)")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-extract features even if cached")
    ap.add_argument("--out-dir", default=INDEX_DIR)
    args = ap.parse_args()

    if args.task == "all":
        tasks = list(DEFAULT_TASKS)
    elif args.task == "every":
        tasks = list(TASKS.keys())
    else:
        tasks = [args.task]
    summary = {}
    for t in tasks:
        blob = train_task(t, args.test_frac, args.seed, args.folds,
                          not args.no_cache, args.out_dir, args.repeats)
        summary[t] = blob["metrics"]

    print("\n========= SUMMARY (repeated group-aware holdout) =========")
    for t, m in summary.items():
        bf, af = m["held_out_before"], m["held_out_after"]

        def line(name: str, key: str) -> str:
            b, a = bf[key], af[key]
            bs = "n/a" if b["mean"] is None else f"{b['mean']:.3f}+/-{b['std']:.3f}"
            as_ = "n/a" if a["mean"] is None else f"{a['mean']:.3f}+/-{a['std']:.3f}"
            return f"  {name:16s} {bs:16s} -> {as_}"

        print(f"{t}:  (majority baseline {m['majority_baseline']:.3f}, "
              f"cv_acc {m['cv_accuracy']:.3f})")
        print(line("accuracy", "accuracy"))
        print(line("abstention", "abstention_rate"))
        print(line("ECE", "ece"))
        print(line("acc|answered", "accuracy_on_answered"))
        print(f"  {'margin median':16s} "
              f"{bf['margin_median']['mean']:.3f}            -> "
              f"{af['margin_median']['mean']:.3f}")


if __name__ == "__main__":
    main()
