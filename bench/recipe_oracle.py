#!/usr/bin/env python
"""Does a minimal hint change the code a model writes? Measure the CEILING first.

THE QUESTION

The proposal is a corpus of compressed performance recipes, selected per
problem by Laya or by vector relevance, injected as a short hint. Building that
means scraping sources, compressing them, building a selector and tuning it.

None of that is worth starting until one thing is known: does injecting the
BEST POSSIBLE hint change the generated code at all?

That is the oracle condition. Every candidate recipe is tried on every problem,
and the best outcome per problem is kept. No selector can beat it, by
construction. So:

  oracle gains nothing   -> no selector helps. The branch dies for the cost of
                            one experiment instead of a corpus project.
  oracle gains a lot     -> the margin is the budget for selection quality, and
                            the gap between oracle and a cheap selector says
                            whether Laya is worth it over plain cosine.

Measuring the ceiling before building the machine is the whole point.

WHY PYTHON, WHEN THE REAL DOMAIN IS RUST AND WGSL

Because it is the cheapest way to be WRONG. The recipes that matter to this
user are about memory layout and GPU dispatch, in languages with no execution
harness here yet. Python competitive programming has perf deltas that are large,
mechanical and famous -- `sys.stdin.readline` against `input()`, `deque` against
`list.pop(0)`, `join` against `+=` in a loop -- and they show up directly in
wall-clock on the big private test cases.

If a minimal hint cannot move the needle where the effect is enormous and
unambiguous, it will not move it in a subtler domain. This is a falsification
test, not a demonstration, and it is deliberately rigged in favour of the
hypothesis. A null result here is decisive; a positive result only earns the
right to try the real domain.

WHAT IS MEASURED

  pass@1      a hint must not break correctness -- that is the veto
  runtime     slowest test case, which is where an O(n^2) answer shows itself
  peak memory sampled resident set

Correctness first. Slower is survivable, wrong is not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import livecodebench as lcb  # noqa: E402

# Python perf recipes, in the same <=40 word minimal-hint form the corpus will
# use. Deliberately terse: the model knows all of this, the hint only has to
# surface it at the moment of writing.
RECIPES = [
    {"id": "io",
     "text": "Large input: read with sys.stdin.buffer.read().split() or "
             "sys.stdin.readline, never input() in a loop. Write one joined "
             "string to sys.stdout, not print per line."},
    {"id": "structures",
     "text": "Use collections.deque for queue pops, not list.pop(0). Use a "
             "set or dict for membership, never a list. Preallocate with "
             "[0]*n rather than repeated append where the size is known."},
    {"id": "strings",
     "text": "Build strings with ''.join(parts), never += in a loop. Slice "
             "rather than concatenate. Compare lengths before comparing "
             "contents."},
    {"id": "loops",
     "text": "Hoist attribute and global lookups out of hot loops into locals. "
             "Prefer comprehensions and built-ins (sum, max, any) over manual "
             "Python-level loops. Avoid recomputing inside the loop condition."},
    {"id": "complexity",
     "text": "Check the input bound first. Above 10^5, an O(n^2) scan will not "
             "finish. Reach for sorting, prefix sums, a heap, or a hash map "
             "before writing nested loops."},
    # The control. A hint of the same SHAPE and LENGTH that carries no
    # performance information. Without it, any measured gain could be the
    # effect of adding text rather than the effect of adding a RECIPE, and
    # that confound would invalidate the whole result.
    {"id": "placebo",
     "text": "Write the solution in Python. Use clear variable names and give "
             "the program a sensible structure. Handle the input format "
             "described in the problem statement."},
]

HINT_BLOCK = "\n\nKeep this in mind while you write:\n{hint}\n"


def prompt_for(row: dict, recipe: dict | None) -> str:
    functional = bool(row.get("starter_code", "").strip())
    base = (lcb.PROMPT_FUNCTIONAL.format(question=row["question_content"],
                                         starter=row["starter_code"])
            if functional else
            lcb.PROMPT_STDIN.format(question=row["question_content"]))
    if recipe is None:
        return base
    return base + HINT_BLOCK.format(hint=recipe["text"])


def wilcoxon_sign(pairs: list[tuple[float, float]]) -> tuple[int, int, float]:
    """Sign test on paired measurements: (better, worse, two-sided p).

    A sign test rather than a t-test because runtimes are heavily skewed --
    one adversarial case dominates -- and the mean of a skewed paired
    difference is not the thing anyone cares about. What matters is how often
    one condition is faster than the other on the same problem.
    """
    better = sum(1 for a, b in pairs if b < a * 0.95)
    worse = sum(1 for a, b in pairs if b > a * 1.05)
    n = better + worse
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(better, worse) + 1))
    return better, worse, min(1.0, 2.0 * tail / (2 ** n))


def report(path: str) -> None:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by: dict = {}
    for r in rows:
        by.setdefault(r["question_id"], {})[r["arm"]] = r
    arms = ["none"] + [r["id"] for r in RECIPES]
    complete = {q: v for q, v in by.items()
                if all(a in v for a in arms) and not any(x.get("error") for x in v.values())}
    errs = sum(1 for r in rows if r.get("error"))

    print(f"\n{'=' * 76}")
    print(f"  RECIPE ORACLE  --  {len(complete)} problems complete across "
          f"{len(arms)} arms")
    if errs:
        print(f"  {errs} generation errors excluded")
    print(f"{'=' * 76}")
    if not complete:
        print("  nothing complete yet")
        return

    n = len(complete)
    print(f"  {'arm':<14}{'pass@1':<18}{'median slowest case':<24}"
          f"{'median peak KB':<16}")
    print("  " + "-" * 70)
    for a in arms:
        k = sum(1 for v in complete.values() if v[a]["passed"])
        passing = [v[a] for v in complete.values() if v[a]["passed"]]
        ms = statistics.median([p["max_s"] for p in passing]) if passing else 0
        mem = statistics.median([p["peak_kb"] for p in passing]) if passing else 0
        lo, hi = lcb.wilson(k, n)
        print(f"  {a:<14}{k}/{n} = {k / n:5.1%} [{lo:.0%}-{hi:.0%}]".ljust(34)
              + f"{ms:<24.3f}{mem:<16.0f}")

    # The ceiling: the best arm per problem, which no selector can beat.
    oracle_pass = sum(1 for v in complete.values()
                      if any(v[a]["passed"] for a in arms))
    base_pass = sum(1 for v in complete.values() if v["none"]["passed"])
    print("\n  ORACLE (best arm per problem)")
    print(f"    pass@1  {oracle_pass}/{n} = {oracle_pass / n:.1%}   "
          f"against no hint at {base_pass / n:.1%}")

    both = [(q, v) for q, v in complete.items()
            if v["none"]["passed"] and any(v[a]["passed"] for a in arms if a != "none")]
    if both:
        pairs = []
        for _q, v in both:
            best = min(v[a]["max_s"] for a in arms
                       if a != "none" and v[a]["passed"])
            pairs.append((v["none"]["max_s"], best))
        b, w, p = wilcoxon_sign(pairs)
        print(f"\n  Among {len(both)} problems solved BOTH with and without a hint:")
        print(f"    best-hinted run faster on {b}, slower on {w}, p = {p:.4f}")
        if b + w < 10:
            print(f"    only {b + w} decided pairs: underpowered, provisional")

    print("\n  per recipe: problems it FIXED / BROKE against no hint")
    for r in RECIPES:
        a = r["id"]
        fixed = sum(1 for v in complete.values()
                    if v[a]["passed"] and not v["none"]["passed"])
        broke = sum(1 for v in complete.values()
                    if v["none"]["passed"] and not v[a]["passed"])
        print(f"    {a:<14} fixed {fixed:<4} broke {broke}")
    print("\n  'placebo' is the control: same shape and length, no perf content.")
    print("  A real effect must beat it, or the gain was from adding text.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "test6.jsonl"))
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--endpoint", default="direct", choices=list(lcb.CONDITIONS))
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--gen-timeout", type=int, default=900)
    ap.add_argument("--test-timeout", type=float, default=12.0)
    ap.add_argument("--max-cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=os.path.join(HERE, "recipe_results.jsonl"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        report(args.out)
        return

    lcb._KEY = lcb.api_key()
    cond = lcb.CONDITIONS[args.endpoint]
    try:
        _t, meta = lcb.generate(cond, "Reply with the word ready.", 64, 120)
        print(f"  {args.endpoint} alive, {meta['ms']}ms")
    except Exception as e:                                       # noqa: BLE001
        raise SystemExit(f"  {args.endpoint} not answering: {type(e).__name__}: {e}")

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rnd = random.Random(args.seed)
    picked = []
    for diff in ("easy", "medium", "hard"):
        g = [r for r in rows if r.get("difficulty") == diff]
        rnd.shuffle(g)
        picked += g[:max(1, args.n // 3)]
    rnd.shuffle(picked)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["question_id"], r["arm"]))
            except Exception:                                    # noqa: BLE001
                continue

    arms = [("none", None)] + [(r["id"], r) for r in RECIPES]
    print(f"  {len(picked)} problems x {len(arms)} arms = "
          f"{len(picked) * len(arms)} generations, {len(done)} already done")

    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(picked):
            qid = row["question_id"]
            cases = lcb.all_cases(row, args.max_cases)
            for arm, recipe in arms:
                if (qid, arm) in done:
                    continue
                try:
                    text, meta = lcb.generate(cond, prompt_for(row, recipe),
                                              args.max_tokens, args.gen_timeout)
                except Exception as e:                           # noqa: BLE001
                    out.write(json.dumps({
                        "question_id": qid, "arm": arm,
                        "error": f"{type(e).__name__}: {e}"}) + "\n")
                    out.flush()
                    continue
                code = lcb.extract_code(text)
                passed, why, perf = lcb.run_tests(code, row, cases,
                                                  args.test_timeout)
                rec = {"question_id": qid, "arm": arm,
                       "difficulty": row.get("difficulty"),
                       "passed": passed, "why": why, **perf, **meta}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"  [{i + 1}/{len(picked)}] {qid:<12} {arm:<12} "
                      f"{'PASS' if passed else 'fail'} "
                      f"slowest={perf['max_s']:.2f}s peak={perf['peak_kb']}KB "
                      f"{why[:30]}", flush=True)

    report(args.out)


if __name__ == "__main__":
    main()
