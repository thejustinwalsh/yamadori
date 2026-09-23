#!/usr/bin/env python
"""Score generated code on three axes, all against a human reference.

WHY THIS EXISTS

LiveCodeBench scores pass or fail by execution and never looks at the code. It
is the right instrument for correctness and the wrong one for everything else:
a solution that passes in 4 seconds and one that passes in 0.04 are the same
number to it, and a program nobody could maintain is the same number again.

Correctness is the veto, not the whole measure. Among solutions that all pass,
what separates them is speed, memory and how the code reads.

WHY THE REFERENCE SOLUTION IS THE BASELINE

"Style" is where a benchmark turns into its author's taste, and this repo has
already been burned by authored thresholds. So nothing here is scored against
an opinion about good code. Every metric is computed for the MODEL's solution
and for the DATASET's reference solution, and reported as a ratio.

`newfacade/LeetCodeDataset` ships a `completion` for every problem: a working,
human-written answer. That makes the question objective -- not "is this good
code", which I would be deciding, but "is this more or less complex than the
solution a person wrote for this exact problem", which the data decides.

A ratio near 1.0 means the model wrote something of the same shape as the
human. Far above means it is more convoluted. Far below usually means it
solved less of the problem.

WHAT IS MEASURED, AND WHY EACH ONE

  cyclomatic    branch points plus one. The standard measure of how many paths
                a reader has to hold in their head.
  max_depth     deepest nesting. Four levels of indentation is where a function
                stops being readable, and it is independent of length.
  length        statements, not lines: blank lines and formatting do not change
                what the code does.
  names         share of identifiers that are one character outside a loop
                index. Cheap, mechanical, and the one style signal that does
                not need a judgement call.
  branch_ratio  branches per statement. Separates "long but flat" from "short
                but tangled", which cyclomatic alone conflates.

All of it comes from the parse tree. Not a regular expression, for the reason
`docs/PROTOCOL.md` rule 8 gives: a regex cannot tell code from a string that
contains code, and this module is handed model output, which is exactly where
that distinction breaks.

WHAT THIS DOES NOT MEASURE

Whether the code is idiomatic, well named beyond the trivial check, or well
designed. Those need a judge, a judge needs validating against human ratings,
and no such ratings exist here. Reporting them would be inventing a number.
"""
from __future__ import annotations

import ast
import json
import math
import os
import statistics

# Nodes that add a path through the function.
_BRANCH = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
           ast.With, ast.AsyncWith, ast.Assert, ast.IfExp, ast.comprehension)
_BOOLOP = (ast.BoolOp,)
# Nesting that a reader actually has to track.
_NESTS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
          ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _depth(node: ast.AST, d: int = 0) -> int:
    best = d
    for child in ast.iter_child_nodes(node):
        nd = d + 1 if isinstance(child, _NESTS) else d
        best = max(best, _depth(child, nd))
    return best


def metrics(code: str) -> dict | None:
    """Structural metrics, or None if the code does not parse.

    None rather than zeros: a program that does not parse has no complexity,
    and scoring it as zero would make the worst possible output look like the
    simplest. Unparseable is its own outcome and the caller must see it.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None

    branches = 0
    for n in ast.walk(tree):
        if isinstance(n, _BRANCH):
            branches += 1
        elif isinstance(n, _BOOLOP):
            # `a and b and c` is two extra paths, not one.
            branches += len(n.values) - 1

    stmts = sum(1 for n in ast.walk(tree) if isinstance(n, ast.stmt))

    # Loop targets are exempt: `for i in range(n)` is not bad naming, and
    # counting it would punish idiomatic code.
    loop_names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.For, ast.AsyncFor)):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    loop_names.add(t.id)
    named = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    considered = [x for x in named if x not in loop_names]
    terse = sum(1 for x in considered if len(x) == 1)

    return {
        "cyclomatic": branches + 1,
        "max_depth": _depth(tree),
        "statements": stmts,
        "terse_names": terse / len(considered) if considered else 0.0,
        "branch_ratio": branches / stmts if stmts else 0.0,
    }


def _ratio(a: float, b: float) -> float | None:
    """a relative to b, with a floor so near-zero denominators do not explode."""
    if b is None or a is None:
        return None
    if b < 1e-9:
        return None
    return a / b


def compare(generated: str, reference: str) -> dict:
    """Every metric as a ratio against the human solution to the same problem."""
    g, r = metrics(generated), metrics(reference)
    if g is None:
        return {"parsed": False, "why": "generated code does not parse"}
    if r is None:
        return {"parsed": True, "why": "reference does not parse", "generated": g}
    out = {"parsed": True, "generated": g, "reference": r, "ratio": {}}
    for k in ("cyclomatic", "max_depth", "statements", "branch_ratio"):
        out["ratio"][k] = _ratio(g[k], r[k])
    # A rate, not a count: a difference rather than a ratio.
    out["ratio"]["terse_names_delta"] = g["terse_names"] - r["terse_names"]
    return out


def summarise(rows: list[dict], label: str = "") -> dict:
    """Median ratios across a run. Median because these are heavy-tailed.

    One pathological solution with 40x the reference complexity would drag a
    mean far enough to invert the comparison, and the question is what a
    TYPICAL solution looks like.
    """
    out: dict = {"n": len(rows), "label": label}
    for k in ("cyclomatic", "max_depth", "statements", "branch_ratio"):
        vals = [r["ratio"][k] for r in rows
                if r.get("parsed") and r.get("ratio", {}).get(k) is not None]
        out[k] = round(statistics.median(vals), 3) if vals else None
        out[k + "_n"] = len(vals)
    deltas = [r["ratio"]["terse_names_delta"] for r in rows
              if r.get("parsed") and r.get("ratio", {}).get("terse_names_delta") is not None]
    out["terse_names_delta"] = round(statistics.median(deltas), 3) if deltas else None
    out["unparseable"] = sum(1 for r in rows if not r.get("parsed"))
    return out


def sign_test(pairs: list[tuple[float, float]], tol: float = 0.05) -> dict:
    """How often does arm B beat arm A on the same problem?

    Paired and non-parametric. Runtimes and complexity ratios are skewed
    enough that a mean difference is not the thing anyone cares about; what
    matters is how often one arm is better on the same input. `tol` ignores
    differences too small to matter, so noise does not count as a win.
    """
    better = sum(1 for a, b in pairs if b < a * (1 - tol))
    worse = sum(1 for a, b in pairs if b > a * (1 + tol))
    n = better + worse
    if n == 0:
        return {"better": 0, "worse": 0, "p": 1.0, "decided": 0}
    tail = sum(math.comb(n, i) for i in range(0, min(better, worse) + 1))
    return {"better": better, "worse": worse,
            "p": min(1.0, 2.0 * tail / (2 ** n)), "decided": n}


def reference_for(row: dict) -> str:
    """The human solution shipped with the problem."""
    return (row.get("completion") or "").strip()


if __name__ == "__main__":
    # Demonstrate on the dataset's own reference solutions: a solution scored
    # against itself must come out at exactly 1.0, which is the cheapest
    # available check that the metrics are not nonsense.
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "LeetCodeDataset-train.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:200]
    same, unparsed = [], 0
    for r in rows:
        ref = reference_for(r)
        if not ref:
            continue
        c = compare(ref, ref)
        if not c.get("parsed"):
            unparsed += 1
            continue
        same.append(c)
    s = summarise(same, "reference vs itself")
    print(f"  self-comparison on {s['n']} reference solutions "
          f"({unparsed} unparseable)")
    for k in ("cyclomatic", "max_depth", "statements", "branch_ratio"):
        flag = "" if s[k] == 1.0 else "   <-- SHOULD BE 1.0"
        print(f"    {k:<14} {s[k]}{flag}")
    print(f"    terse_names_delta {s['terse_names_delta']} (should be 0.0)")
    print("\n  spread across those references, to show the metrics discriminate:")
    ms = [metrics(reference_for(r)) for r in rows if reference_for(r)]
    ms = [m for m in ms if m]
    for k in ("cyclomatic", "max_depth", "statements"):
        vals = sorted(m[k] for m in ms)
        print(f"    {k:<12} min {vals[0]}  median {vals[len(vals)//2]}  "
              f"p90 {vals[int(len(vals)*0.9)]}  max {vals[-1]}")
