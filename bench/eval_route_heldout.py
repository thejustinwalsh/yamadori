#!/usr/bin/env python
"""Score route_in heads on the HELD-OUT package-domain labels. Evaluation only.

    .venv-laya/Scripts/python.exe bench/eval_route_heldout.py \
        --old index/laya/route_in.json --new <staging>/route_in.json

bench/laya_routing_heldout_packages.jsonl was written independently of every
training file and matches no LABEL_GLOB in scripts/train_laya.py, so nothing
here trains on it. This script never prints a held-out question: it reports
aggregates only, and the per-row file it writes carries row index, label,
domain and predictions -- no text.

Conditions, all on the SAME 120 rows (PROTOCOL rule 6, paired):
  old        the trained head currently in index/laya (argmax, forced)
  new        the staged retrained head (argmax, forced)
  rule       selection.rule_baseline, the one copy of the regex
  selection  selection.decide with the real symbol lookup, Laya absent --
             exactly what mcp/test_selection.py reports (binary only)

Heads are scored by argmax. Their abstain gate is reported separately as
coverage, because an abstention is not a routing decision and forcing both
heads keeps the comparison paired on all 120 rows.

Statistics: exact Clopper-Pearson 95% intervals; exact two-sided McNemar
(binomial on the discordant pairs) for paired comparisons.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "mcp"))

HELD_OUT = os.path.join(HERE, "laya_routing_heldout_packages.jsonl")
TASK = "route_in"


# ------------------------------------------------------------ stats --------


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact binomial CI by bisection on the binomial CDF (no scipy needed)."""
    if n == 0:
        return float("nan"), float("nan")

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper


def mcnemar_exact(a_correct: List[bool], b_correct: List[bool]) -> Dict[str, float]:
    """b = A right & B wrong, c = A wrong & B right; exact two-sided p."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p": p}


def fmt(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n)
    return f"{k:3d}/{n:<3d} {k / n:6.3f}  [{lo:.3f}, {hi:.3f}]" if n else "  n/a"


# ------------------------------------------------------------ data ---------


def load_rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    for r in rows:
        if r.get("label") not in ("investigate", "answer_directly", "clarify"):
            raise SystemExit(f"bad label in {path}: {r.get('label')!r}")
    return rows


def features(rows: List[dict], cache_path: str) -> List[dict]:
    """Frozen Laya features for each row, cached by state text OUTSIDE index/laya."""
    from laya_head import PROMPT_REV, FeatureExtractor, render_state

    states = [render_state(TASK, r) for r in rows]
    cache: Dict[str, dict] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            blob = json.load(fh)
        if blob.get("prompt_rev") == PROMPT_REV:
            cache = blob["by_state"]
    missing = [s for s in dict.fromkeys(states) if s not in cache]
    if missing:
        print(f"  extracting features for {len(missing)} held-out state(s) (batch size 1)")
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("LAYA_GPU", "1"))
        import laya

        agent = laya.load(os.environ.get("LAYA_MODEL", "convaiinnovations/laya"))
        fx = FeatureExtractor(agent)
        for s, f in zip(missing, fx.features(TASK, missing)):
            cache[s] = f
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({"prompt_rev": PROMPT_REV, "by_state": cache}, fh)
    else:
        print(f"  feature cache hit for all {len(rows)} held-out rows")
    out = [cache[s] for s in states]
    # PROTOCOL rule 1: a dead extractor must not look like a live one.
    zero = sum(1 for f in out if not any(abs(v) > 0 for v in f["cls"]))
    if zero:
        raise SystemExit(f"{zero} held-out feature vectors are all zero -- extractor is dead")
    return out


def head_preds(path: str, feats: List[dict]) -> Tuple[List[str], List[bool], dict]:
    from laya_head import TrainedHead

    h = TrainedHead.load(path)
    choices, abstain = [], []
    for f in feats:
        d = h.decide(f)
        choices.append(d["choice"])
        abstain.append(d["abstain"])
    return choices, abstain, {"kind": h.kind, "spec": h.feature_spec,
                              "gate": h.gate, "n_train": h.blob.get("trained_on", {}).get("n_total")}


def rule_and_selection(rows: List[dict], want_selection: bool
                       ) -> Tuple[List[str], Optional[List[bool]], str]:
    # Same isolation as mcp/test_selection.py: no corpus writes, no upstream.
    tmp = tempfile.mkdtemp(prefix="eval_route_heldout_")
    os.environ.setdefault("YAMADORI_CORPUS_DB", os.path.join(tmp, "corpus.sqlite3"))
    os.environ.setdefault("LLAMA_STACK_URL", "http://127.0.0.1:1")
    import selection

    rule = [selection.rule_baseline(r) for r in rows]
    if not want_selection:
        return rule, None, "skipped (--no-selection)"
    import deps
    import domains
    import tiers

    if not domains.held_sources(deps.STORE):
        return rule, None, "skipped: no live package store"
    t = tiers.resolve({"reasoning_effort": "max"}, tiers.from_header(None))
    dbs = selection.symbol_dbs()
    sel = []
    for r in rows:
        msgs = ([{"role": "user", "content": r["context"]}] if r.get("context") else []) \
            + [{"role": "user", "content": r["question"]}]
        d = selection.decide(msgs, t, domains.tool_admission(msgs, None), dbs=dbs)
        sel.append(bool(d["investigate"]))
    return rule, sel, "ok"


# ------------------------------------------------------------ main ---------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default=os.path.join(ROOT, "index", "laya", "route_in.json"))
    ap.add_argument("--new", required=True, help="staged route_in.json")
    ap.add_argument("--heldout", default=HELD_OUT)
    ap.add_argument("--cache", default=None,
                    help="feature cache (default: features_heldout.json beside --new)")
    ap.add_argument("--rows-out", default=None,
                    help="per-row predictions, no text (default: beside --new)")
    ap.add_argument("--no-selection", action="store_true")
    args = ap.parse_args()

    stage = os.path.dirname(os.path.abspath(args.new))
    cache = args.cache or os.path.join(stage, "features_heldout.json")
    rows = load_rows(args.heldout)
    n = len(rows)
    y = [r["label"] for r in rows]
    yb = [lab == "investigate" for lab in y]
    hard = [i for i, r in enumerate(rows) if r.get("hard")]
    print(f"held-out: {n} rows, {sum(yb)} investigate, {len(hard)} hard")

    feats = features(rows, cache)
    old_c, old_ab, old_meta = head_preds(args.old, feats)
    new_c, new_ab, new_meta = head_preds(args.new, feats)
    rule_c, sel_b, sel_status = rule_and_selection(rows, not args.no_selection)
    print(f"old head: {old_meta}\nnew head: {new_meta}\nselection: {sel_status}")

    conds3 = {"old": old_c, "new": new_c, "rule": rule_c}
    condsb = {k: [c == "investigate" for c in v] for k, v in conds3.items()}
    if sel_b is not None:
        condsb["selection"] = sel_b
    corr3 = {k: [p == t for p, t in zip(v, y)] for k, v in conds3.items()}
    corrb = {k: [p == t for p, t in zip(v, yb)] for k, v in condsb.items()}

    def block(title, corr, idx):
        print(f"\n{title}")
        for k, c in corr.items():
            print(f"  {k:10s} {fmt(sum(c[i] for i in idx), len(idx))}")

    allidx = list(range(n))
    block("3-way accuracy (95% exact CI)", corr3, allidx)
    block("investigate-vs-not accuracy (95% exact CI)", corrb, allidx)
    block(f"hard slice, 3-way (n={len(hard)})", corr3, hard)
    block(f"hard slice, investigate-vs-not (n={len(hard)})", corrb, hard)

    print("\nper-domain, investigate-vs-not  (old / new / rule"
          + (" / selection" if sel_b is not None else "") + ")")
    doms = sorted({r.get("domain", "?") for r in rows})
    for d in doms:
        idx = [i for i in allidx if rows[i].get("domain") == d]
        cells = [f"{sum(corrb[k][i] for i in idx):3d}" for k in corrb]
        print(f"  {d:12s} n={len(idx):3d}   " + " / ".join(cells))

    print("\nprediction distribution (3-way):")
    for k, v in conds3.items():
        print(f"  {k:10s} " + ", ".join(f"{lab}={v.count(lab)}" for lab in
                                        ("investigate", "answer_directly", "clarify")))
    print("  truth      " + ", ".join(f"{lab}={y.count(lab)}" for lab in
                                       ("investigate", "answer_directly", "clarify")))
    print(f"\nhead coverage at own gate: old {1 - sum(old_ab) / n:.3f} "
          f"(gate {old_meta['gate']}), new {1 - sum(new_ab) / n:.3f} (gate {new_meta['gate']})")
    for name, ab, c in (("old", old_ab, corrb["old"]), ("new", new_ab, corrb["new"])):
        ans = [i for i in allidx if not ab[i]]
        if ans:
            print(f"  {name} investigate-vs-not on answered: "
                  f"{fmt(sum(c[i] for i in ans), len(ans))}")

    print("\npaired exact McNemar (a_only = first right & second wrong):")
    pairs = [("old", "new", corrb, "investigate-vs-not"),
             ("old", "new", corr3, "3-way"),
             ("rule", "new", corrb, "investigate-vs-not")]
    if sel_b is not None:
        pairs.append(("selection", "new", corrb, "investigate-vs-not"))
    results = {}
    for a, b, corr, what in pairs:
        m = mcnemar_exact(corr[a], corr[b])
        results[f"{a}_vs_{b}_{what}"] = m
        print(f"  {a:9s} vs {b:4s} [{what:18s}] {a}-only {m['a_only']:3d}  "
              f"{b}-only {m['b_only']:3d}  p = {m['p']:.4g}")

    rows_out = args.rows_out or os.path.join(stage, "heldout_rows.jsonl")
    with open(rows_out, "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({"i": i, "label": y[i], "domain": r.get("domain"),
                                 "hard": bool(r.get("hard")), "old": old_c[i],
                                 "new": new_c[i], "rule": rule_c[i],
                                 "selection_investigate": None if sel_b is None else sel_b[i]})
                     + "\n")
    summary = {
        "n": n,
        "binary": {k: sum(v) for k, v in corrb.items()},
        "three_way": {k: sum(v) for k, v in corr3.items()},
        "hard_binary": {k: sum(v[i] for i in hard) for k, v in corrb.items()},
        "mcnemar": results,
    }
    with open(os.path.join(stage, "heldout_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nper-row (no text) -> {rows_out}")


if __name__ == "__main__":
    main()
