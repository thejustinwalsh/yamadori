#!/usr/bin/env python
r"""The benchmark results page, which has to stay honest while n is tiny.

WHY THIS IS A SEPARATE PAGE FROM `dashboard.py`

`dashboard.py` edits a corpus; nothing on it is a measurement. This one shows
numbers that a decision could be made on, and the failure mode is different:
not a bad edit, but a comparison presented as a result when the experiment
could not have produced one. So the rules it follows are `docs/PROTOCOL.md`
rules 4, 6, 10 and 11 rather than anything about editing.

NOTHING HERE COMPUTES A STATISTIC

Every number is produced by the harness that produced the data --
`livecodebench.wilson`, `livecodebench.mcnemar`, `quality.sign_test`,
`retrieval.wilson` / `retrieval.mcnemar` -- imported and called, never
reimplemented. A second implementation of McNemar in a web page is a second
thing that can disagree with the benchmark, and when it did the page would win
the argument, because it is the one with the nice typography.

THE PART THAT TOOK THE THINKING: SMALL n

At three complete problems there are two different questions and the page must
answer both without letting either contaminate the other.

  "did it pass anything at all"   a raw pass/fail count. Valid at n=1. Shown.
  "is one arm better"             needs discordant pairs. At n=3 the exact
                                  test cannot reach p < 0.05 no matter how the
                                  three fall, because 2/2**d < 0.05 first
                                  holds at d = 6. So the page computes that
                                  floor from `mcnemar` itself and says the
                                  comparison is impossible, rather than
                                  printing "no detectable difference", which
                                  reads as a finding of no difference.

Suppressing the raw counts for being underpowered would be the opposite error
and is just as wrong: rule 11 says report what happened.

ERRORS ARE NOT FAILURES

A generation that 502'd is not a model that got the answer wrong. The harness
already keeps them apart (`load_done` refuses to record an error as done); the
page keeps them apart too, counts them per arm, and says out loud that the
paired set is whatever survived -- which is not a random subset of what was
drawn if errors correlate with anything.

AND A FAILURE MAY NOT BE THE MODEL EITHER

`why` is bucketed, because "wrong output" and "crashed before the model's code
ran" are different findings and one of them is a bug in this repo. The
documented case: functional problems whose starter code uses `List[int]` die
at the method signature unless the runner prepends a typing preamble, and 63
of 175 problems in the v6 set are functional. Every one was recorded as a
model failure until someone read a traceback.
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "bench"))
sys.path.insert(0, HERE)

import dash_shell  # noqa: E402  (after the path is set)

LCB_PATH = os.path.join(BENCH, "lcb_tiers.jsonl")
RETRIEVAL_PATH = os.path.join(BENCH, "retrieval_results.jsonl")
RECIPE_PATH = os.path.join(BENCH, "recipe_results.jsonl")

# How to produce each file, shown verbatim in the empty state. A panel that
# says "no data" and stops has told the operator nothing they did not know.
HOW = {
    "lcb": "run  python bench/livecodebench.py --tiers --out bench/lcb_tiers.jsonl",
    "retrieval": "run  python bench/retrieval.py",
    "recipe": "run  python bench/recipe_oracle.py",
}

ALPHA = 0.05

# Names the runner's FUNCTIONAL_PREAMBLE exists to define. A NameError on one
# of these fired before any model logic ran, so it is scaffolding, not a wrong
# answer. The list is the preamble's own import list, not a guess.
_PREAMBLE_NAMES = {
    "List", "Dict", "Optional", "Tuple", "Set", "Union", "Any",
    "Counter", "OrderedDict", "defaultdict", "deque",
    "cache", "lru_cache", "reduce",
    "accumulate", "combinations", "permutations", "product",
    "comb", "gcd", "inf", "isqrt", "lcm",
    "ListNode", "TreeNode",
    "bisect", "collections", "functools", "heapq", "itertools",
}
_NAMEERROR = re.compile(r"NameError: name '([A-Za-z_][A-Za-z0-9_]*)' is not defined")


def _bench(name: str):
    """Import a module out of bench/ so the harness's own functions are used."""
    if BENCH not in sys.path:
        sys.path.insert(0, BENCH)
    return importlib.import_module(name)


def _read(path: str) -> tuple[list[dict], dict]:
    """Rows plus what is knowable about the file itself.

    Malformed lines are counted rather than skipped silently. A jsonl written
    by an appending run can end mid-line if the run was killed, and that is
    worth one number on screen.
    """
    meta = {"file": os.path.relpath(path, os.path.dirname(BENCH)).replace("\\", "/"),
            "exists": os.path.exists(path), "file_age_s": None, "bad_lines": 0}
    if not meta["exists"]:
        return [], meta
    meta["file_age_s"] = round(time.time() - os.path.getmtime(path))
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                meta["bad_lines"] += 1
    return rows, meta


def _empty(kind: str, meta: dict, extra: str = "") -> dict:
    return {"state": "empty", "how": HOW[kind], "note": extra, **meta}


def _mean(vals: list) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def min_discordant_for_significance(mcnemar, alpha: float = ALPHA,
                                    cap: int = 400) -> int | None:
    """The fewest discordant pairs that could ever reach `alpha`.

    Computed by asking the benchmark's own `mcnemar` with every pair falling
    the same way -- the most favourable arrangement there is. Below this many
    discordant pairs no result is possible, which is rule 4 stated as a number
    instead of a warning.
    """
    for d in range(1, cap + 1):
        if mcnemar(d, 0) < alpha:
            return d
    return None


def classify_why(why: str) -> tuple[str, bool]:
    """Bucket a failure reason, and say whether it is scaffolding.

    Returns (bucket, scaffolding). Scaffolding means the model's code did not
    get to run, so the row says nothing about the model.
    """
    w = (why or "").strip()
    if not w:
        return "unstated", False
    low = w.lower()
    if low.startswith("no code"):
        return "no code in the reply", False
    if low.startswith("timeout"):
        return "timeout", False
    if "wrong output" in low:
        return "wrong output", False
    if "exited" in low:
        m = _NAMEERROR.search(w)
        if m and m.group(1) in _PREAMBLE_NAMES:
            return f"crashed: {m.group(1)} undefined (runner preamble skipped)", True
        if m:
            return f"crashed: NameError on {m.group(1)}", False
        for kind in ("SyntaxError", "IndentationError", "ImportError",
                     "ModuleNotFoundError", "RecursionError", "MemoryError"):
            if kind in w:
                return f"crashed: {kind}", kind in ("ImportError", "ModuleNotFoundError")
        return "crashed: non-zero exit", False
    # run_tests returns "<ExcType> on case i: ..." when the RUNNER threw.
    m = re.match(r"([A-Za-z_]+Error|[A-Za-z_]+Exception) on case", w)
    if m:
        return f"harness raised {m.group(1)}", True
    return "other", False


# --------------------------------------------------------------------------
# LiveCodeBench tiers
# --------------------------------------------------------------------------

def lcb_summary() -> dict:
    lcb = _bench("livecodebench")
    quality = _bench("quality")
    arms = list(lcb.TIER_CONDITIONS)

    rows, meta = _read(LCB_PATH)
    if not meta["exists"]:
        return _empty("lcb", meta)
    if not rows:
        return _empty("lcb", meta, "the file exists and holds no rows")

    # Last row wins per (problem, arm): the harness appends retries after a
    # transient failure, and the retry is the outcome. This is exactly what
    # `livecodebench.report` does, so the page and the CLI cannot disagree.
    by: dict[str, dict] = {}
    for r in rows:
        if "question_id" not in r or "condition" not in r:
            continue
        by.setdefault(r["question_id"], {})[r["condition"]] = r

    errors = [r for r in rows if r.get("error")]
    err_by_arm: dict[str, dict[str, int]] = {a: {} for a in arms}
    for r in errors:
        a = r.get("condition")
        if a not in err_by_arm:
            err_by_arm.setdefault(a, {})
        kind = str(r["error"]).split(":")[0]
        err_by_arm[a][kind] = err_by_arm[a].get(kind, 0) + 1

    scored = [r for r in rows if not r.get("error")]

    # The RAW question -- "did it pass anything at all" -- over every scored
    # attempt, pairing or no pairing. Valid at n=1 and reported at n=1.
    raw = []
    for a in arms:
        att = [r for r in scored if r.get("condition") == a]
        k = sum(1 for r in att if r.get("passed"))
        lo, hi = lcb.wilson(k, len(att))
        raw.append({"arm": a, "attempts": len(att), "passed": k,
                    "rate": (k / len(att)) if att else None,
                    "lo": lo, "hi": hi})

    paired = {q: v for q, v in by.items()
              if len(v) == len(arms) and all(a in v for a in arms)
              and not any(x.get("error") for x in v.values())}
    n = len(paired)

    floor = min_discordant_for_significance(lcb.mcnemar)
    power = {"n_paired": n, "min_discordant_for_sig": floor,
             "max_possible_discordant": n, "alpha": ALPHA,
             "comparison_possible": bool(floor is not None and n >= floor)}

    arm_table = []
    for a in arms:
        vals = list(paired.values())
        k = sum(1 for v in vals if v[a].get("passed"))
        lo, hi = lcb.wilson(k, n)
        ms = [v[a].get("ms") for v in vals if isinstance(v[a].get("ms"), (int, float))]
        pin = [v[a].get("prompt_tokens") for v in vals if v[a].get("prompt_tokens")]
        pout = [v[a].get("completion_tokens") for v in vals
                if v[a].get("completion_tokens")]
        mean_ms = _mean(ms)
        arm_table.append({
            "arm": a,
            "k": k, "n": n,
            "rate": (k / n) if n else None, "lo": lo, "hi": hi,
            "mean_s": round(mean_ms / 1000, 1) if mean_ms is not None else None,
            # Cost is not a footnote. An arm that wins by spending three times
            # the tokens has not obviously won, so these sit in the same row.
            "tok_in": round(_mean(pin)) if pin else None,
            "tok_in_reported": len(pin),
            "tok_out": round(_mean(pout)) if pout else None,
            "tok_out_reported": len(pout),
            "empty": sum(1 for v in vals if v[a].get("empty_content")),
            "len_cut": sum(1 for v in vals if v[a].get("finish") == "length"),
            "errors": sum(err_by_arm.get(a, {}).values()),
        })

    # Pairwise, McNemar, plus the sign tests among problems BOTH arms solved.
    pairs = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b_ = arms[i], arms[j]
            b = sum(1 for v in paired.values()
                    if v[b_].get("passed") and not v[a].get("passed"))
            c = sum(1 for v in paired.values()
                    if v[a].get("passed") and not v[b_].get("passed"))
            p = lcb.mcnemar(b, c)
            discordant = b + c
            # A verdict is only issued when the test could have produced one.
            verdict = None
            if discordant >= (floor or 10 ** 9) and p < ALPHA:
                verdict = f"{b_} better" if b > c else f"{a} better" if c > b else None

            both = [v for v in paired.values()
                    if v[a].get("passed") and v[b_].get("passed")]
            signs = []
            for key, label in (("max_s", "slowest case s"),
                               ("peak_kb", "peak KB (lower bound)"),
                               ("q_cyclomatic", "cyclomatic"),
                               ("q_statements", "statements"),
                               ("q_max_depth", "nesting depth")):
                pp = [(v[a].get(key), v[b_].get(key)) for v in both
                      if v[a].get(key) is not None and v[b_].get(key) is not None]
                if len(pp) < 3:
                    signs.append({"metric": label, "pairs": len(pp),
                                  "insufficient": True})
                    continue
                st = quality.sign_test(pp)
                sv = None
                if st["p"] < ALPHA and st["decided"] >= (floor or 10 ** 9):
                    sv = b_ if st["better"] > st["worse"] else a
                signs.append({
                    "metric": label, "pairs": len(pp), "insufficient": False,
                    "median_a": round(statistics.median(x[0] for x in pp), 3),
                    "median_b": round(statistics.median(x[1] for x in pp), 3),
                    "better": st["better"], "worse": st["worse"],
                    "decided": st["decided"], "p": st["p"], "verdict": sv})
            pairs.append({"a": a, "b": b_, "b_only": b, "a_only": c,
                          "discordant": discordant, "p": p, "verdict": verdict,
                          "underpowered": discordant < 10,
                          "both_solved": len(both), "signs": signs})

    difficulty = []
    for d in ("easy", "medium", "hard"):
        g = {q: v for q, v in paired.items()
             if v[arms[0]].get("difficulty") == d}
        if not g:
            continue
        cells = {}
        for a in arms:
            k = sum(1 for v in g.values() if v[a].get("passed"))
            cells[a] = {"k": k, "n": len(g), "rate": k / len(g)}
        difficulty.append({"difficulty": d, "n": len(g), "by_arm": cells})

    failures: dict[str, dict] = {}
    scaffolding_total = 0
    for a in arms:
        att = [r for r in scored if r.get("condition") == a]
        buckets: dict[str, dict] = {}
        for r in att:
            if r.get("passed"):
                continue
            bucket, scaf = classify_why(r.get("why"))
            e = buckets.setdefault(bucket, {"bucket": bucket, "n": 0,
                                            "scaffolding": scaf})
            e["n"] += 1
            if scaf:
                scaffolding_total += 1
        failures[a] = {
            "passed": sum(1 for r in att if r.get("passed")),
            "failed": sum(1 for r in att if not r.get("passed")),
            "buckets": sorted(buckets.values(), key=lambda x: -x["n"]),
        }

    # Rule 3, as a check rather than a habit: a rate identical across every
    # condition, or exactly 0 or 100 percent, is a harness bug until proven
    # otherwise. The page says so instead of leaving the reader to notice.
    ks = {a["k"] for a in arm_table}
    suspect = {
        "identical_across_arms": bool(n and len(ks) == 1 and len(arm_table) > 1),
        "extreme": [a["arm"] for a in arm_table
                    if n and a["rate"] in (0.0, 1.0)],
    }

    return {
        "state": "ready", **meta,
        "arms": arms,
        "suspect": suspect,
        "progress": {
            "rows": len(rows),
            "attempted": len(by),
            "scored_rows": len(scored),
            "errors": len(errors),
            "complete_all_arms": n,
            "errors_by_arm": err_by_arm,
        },
        "power": power,
        "arm_table": arm_table,
        "raw": raw,
        "pairs": pairs,
        "difficulty": difficulty,
        "failures": failures,
        "scaffolding_total": scaffolding_total,
    }


# --------------------------------------------------------------------------
# Retrieval: symbol table vs embedding vs embed+rerank
# --------------------------------------------------------------------------

def retrieval_summary() -> dict:
    rows, meta = _read(RETRIEVAL_PATH)
    if not meta["exists"]:
        return _empty("retrieval", meta)
    rows = [r for r in rows if not r.get("error") and "symbol_rank" in r]
    if not rows:
        return _empty("retrieval", meta, "the file holds no scored tasks")

    r_mod = _bench("retrieval")
    conds = list(r_mod.CONDITIONS)
    n = len(rows)

    arm_table = []
    for label, rank_key, ms_key in conds:
        ranks = [r.get(rank_key) for r in rows]
        h1 = sum(1 for x in ranks if x == 1)
        h5 = sum(1 for x in ranks if isinstance(x, int) and 1 <= x <= 5)
        lo1, hi1 = r_mod.wilson(h1, n)
        lo5, hi5 = r_mod.wilson(h5, n)
        # `rank_of` returns 0 when the truth is absent, so 0 contributes
        # nothing to MRR rather than dividing by zero.
        mrr = sum(1 / x for x in ranks if x) / n
        arm_table.append({
            "arm": label, "n": n,
            "hit1": h1, "hit1_rate": h1 / n, "hit1_lo": lo1, "hit1_hi": hi1,
            "hit5": h5, "hit5_rate": h5 / n, "hit5_lo": lo5, "hit5_hi": hi5,
            "mrr": round(mrr, 3),
            "mean_ms": round(_mean([r.get(ms_key) or 0.0 for r in rows]), 2),
            "absent": sum(1 for x in ranks if not x),
            # Rule 3: exactly 0 or 100 percent is a harness artefact until
            # something proves otherwise, and here something does -- see the
            # provenance note below, which explains the one arm that hits it.
            "extreme": (h1 / n) in (0.0, 1.0),
        })
    base = arm_table[0]["mean_ms"] or 1e-9
    for a in arm_table:
        a["cost_x"] = round(a["mean_ms"] / base, 1)

    floor = min_discordant_for_significance(r_mod.mcnemar)
    pairs = []
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            (la, ka, _), (lb, kb, _) = conds[i], conds[j]
            b = sum(1 for r in rows if r.get(kb) == 1 and r.get(ka) != 1)
            c = sum(1 for r in rows if r.get(ka) == 1 and r.get(kb) != 1)
            p = r_mod.mcnemar(b, c)
            verdict = None
            if b + c >= (floor or 10 ** 9) and p < ALPHA:
                verdict = f"{lb} better" if b > c else f"{la} better" if c > b else None
            pairs.append({"a": la, "b": lb, "b_only": b, "a_only": c,
                          "discordant": b + c, "p": p, "verdict": verdict,
                          "underpowered": (b + c) < 10})

    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r.get("source") or "?", []).append(r)
    sources = []
    for s, g in sorted(by_source.items()):
        sources.append({"source": s, "n": len(g), "by_arm": {
            label: sum(1 for r in g if r.get(rank_key) == 1) / len(g)
            for label, rank_key, _ in conds}})

    return {"state": "ready", **meta, "n": n,
            "arms": [c[0] for c in conds],
            "power": {"n_paired": n, "min_discordant_for_sig": floor,
                      "max_possible_discordant": n, "alpha": ALPHA,
                      "comparison_possible": bool(floor is not None and n >= floor)},
            "arm_table": arm_table, "pairs": pairs, "sources": sources,
            # Rule 11, and neither of these can be fixed from here.
            "caveats": [
                # The symbol arm queries `defs`; bench/tasks.py built the
                # ground truth from `defs`. Its rate is a property of the
                # construction, not a retrieval result, and printing it beside
                # two arms that were genuinely tested would read as a win.
                "The locate ground truth in bench/tasks.py is taken from the "
                "same `defs` table the symbol-table arm queries, so that arm "
                "is being scored against its own source. Its rate is a "
                "consistency check on the index, not a retrieval result, and "
                "the two comparisons involving it inherit that.",
                "Sources whose index failed the aliveness check are skipped by "
                "bench/retrieval.py and never written to this file, so this "
                "page cannot show which were excluded or why. The run log has "
                "them.",
            ]}


# --------------------------------------------------------------------------
# Recipe oracle
# --------------------------------------------------------------------------

def recipe_summary() -> dict:
    rows, meta = _read(RECIPE_PATH)
    if not meta["exists"]:
        return _empty("recipe", meta)
    if not rows:
        return _empty("recipe", meta, "the file exists and holds no rows")

    lcb = _bench("livecodebench")
    by: dict[str, dict] = {}
    for r in rows:
        if "question_id" not in r or "arm" not in r:
            continue
        by.setdefault(r["question_id"], {})[r["arm"]] = r
    seen = {r.get("arm") for r in rows if r.get("arm")}
    arms = (["none"] if "none" in seen else []) + sorted(seen - {"none"})
    errors = [r for r in rows if r.get("error")]
    complete = {q: v for q, v in by.items()
                if all(a in v for a in arms)
                and not any(x.get("error") for x in v.values())}
    n = len(complete)
    floor = min_discordant_for_significance(lcb.mcnemar)

    table = []
    for a in arms:
        k = sum(1 for v in complete.values() if v[a].get("passed"))
        lo, hi = lcb.wilson(k, n)
        passing = [v[a] for v in complete.values() if v[a].get("passed")]
        table.append({
            "arm": a, "k": k, "n": n, "rate": (k / n) if n else None,
            "lo": lo, "hi": hi,
            "median_max_s": (round(statistics.median(
                [p.get("max_s") or 0 for p in passing]), 3) if passing else None),
            "median_peak_kb": (round(statistics.median(
                [p.get("peak_kb") or 0 for p in passing])) if passing else None),
        })
    oracle = sum(1 for v in complete.values()
                 if any(v[a].get("passed") for a in arms)) if n else 0
    return {"state": "ready", **meta, "arms": arms,
            "progress": {"attempted": len(by), "errors": len(errors),
                         "complete_all_arms": n},
            "power": {"n_paired": n, "min_discordant_for_sig": floor,
                      "max_possible_discordant": n, "alpha": ALPHA,
                      "comparison_possible": bool(floor is not None and n >= floor)},
            "arm_table": table,
            "oracle": {"k": oracle, "n": n,
                       "rate": (oracle / n) if n else None}}


def _bench_sub(sub: str, name: str):
    """Import `name` out of bench/<sub>/ (analyse.py exists in two of them, so
    the module is loaded under its own path, never a cached namesake)."""
    import importlib.util
    path = os.path.join(BENCH, sub, name + ".py")
    key = f"_dash_{sub}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    d = os.path.join(BENCH, sub)
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(key, None)
        raise
    return mod


def _wilson(k: int, n: int) -> tuple[float | None, float | None]:
    """The harness's own Wilson interval; None bounds when nothing was scored
    (livecodebench.wilson returns (0, 0) at n=0, which would draw as a
    measured zero)."""
    if not n:
        return None, None
    return _bench("livecodebench").wilson(k, n)


def _load_json(path: str):
    """A json file, or None when it is absent or half-written. A benchmark
    rewriting its summary mid-poll must cost one poll, not the section."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _age(path: str) -> int | None:
    return round(time.time() - os.path.getmtime(path)) if os.path.exists(path) else None


def _rel(path: str) -> str:
    return os.path.relpath(path, os.path.dirname(BENCH)).replace("\\", "/")


def _median(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


# --------------------------------------------------------------------------
# The arms legend: one row per arm, in the columns of README "Effort tiers"
# (thinking sent, retrieval, hints, fan-out, deep thinking) plus check_code.
# A value is "on", "off" or "auto" (the tier allows it and mcp/selection.py
# decides per request). Built from each harness's own arm table, never typed
# here, so a renamed or re-specified arm cannot drift from its legend.
# --------------------------------------------------------------------------

def _tiers_table() -> dict:
    return importlib.import_module("tiers").TIERS


def legend_from_features(arm: str, feats: dict, note: str = "",
                         client_tool: str | None = None) -> dict:
    """An arm forced by X-Yamadori-Features. A key the header leaves out is
    left to the selection engine ("auto")."""
    def onoff(key):
        if key not in feats:
            return "auto"
        return "on" if feats[key] else "off"
    fan = feats.get("fanout")
    return {"arm": arm, "kind": "features", "note": note,
            "thinking_sent": feats.get("effort"),
            "retrieval": onoff("retrieval"), "hints": onoff("hints"),
            "fanout": ("auto" if fan is None else str(fan)),
            "deep_thinking": onoff("investigate"),
            "check": ("on" if feats.get("check_code") or client_tool else "off"),
            "repair": "on" if feats.get("repair") else "off",
            "client_tool": client_tool}


def legend_from_tier(arm: str, tier: str, effort_sent: str | None,
                     note: str = "") -> dict:
    """A product arm: the tier says what is ALLOWED; fan-out and deep
    thinking are then chosen per request, so an allowed one reads "auto"."""
    t = _tiers_table().get(tier) or {}
    fan = int(t.get("fanout") or 1)
    return {"arm": arm, "kind": "tier", "tier": tier, "note": note,
            "thinking_sent": effort_sent,
            "retrieval": "on" if t.get("retrieval") else "off",
            "hints": "on" if t.get("hints") else "off",
            "fanout": f"auto, up to {fan}" if fan > 1 else "1",
            "deep_thinking": "auto" if t.get("investigate") else "off",
            "check": "on" if t.get("check_code") else "off",
            "repair": "on" if t.get("repair") else "off",
            "client_tool": None}


# --------------------------------------------------------------------------
# LiveBench
# --------------------------------------------------------------------------

LIVEBENCH_RESULTS = os.path.join(BENCH, "livebench", "results")
_PROGRESS = re.compile(r"^\[(\d\d:\d\d:\d\d)\].*?answered (\d+)/(\d+)\s+err_rows (\d+)")


def _livebench_legend(arms: list[str]) -> list[dict]:
    try:
        drive = _bench_sub("livebench", "drive")
        spec = drive.ARMS
    except Exception:                                            # noqa: BLE001
        spec = {}
    try:
        labels = _bench_sub("livebench", "summarize").ARM_LABELS
    except Exception:                                            # noqa: BLE001
        labels = {}
    out = []
    for a in arms:
        s = spec.get(a) or {}
        note = labels.get(a, "")
        if "tier" in s:
            out.append(legend_from_tier(a, s["tier"], s.get("effort_sent"), note))
        elif "features" in s:
            out.append(legend_from_features(a, s["features"], note))
        else:
            out.append({"arm": a, "kind": "unknown", "note": note or
                        "not in bench/livebench/drive.py ARMS"})
    return out


def _livebench_progress(run_dir: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "logs", "progress_*.log"))):
        base = os.path.basename(p)[len("progress_"):-len(".log")]
        last = None
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = _PROGRESS.search(line)
                    if m:
                        last = m
        except OSError:
            continue
        arm, _, cat = base.partition("_")
        # arm names contain no underscore except the retired prefix arm
        out.append({"arm": arm, "category": cat, "log": _rel(p), "log_age_s": _age(p),
                    "at": last.group(1) if last else None,
                    "answered": int(last.group(2)) if last else 0,
                    "of": int(last.group(3)) if last else None,
                    "err_rows": int(last.group(4)) if last else 0})
    return out


def _livebench_rows(run_dir: str) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "rows_*_*.jsonl"))):
        rows, _ = _read(p)
        for r in rows:
            if isinstance(r, dict) and r.get("arm"):
                by.setdefault(r["arm"], []).append(r)
    return by


def _livebench_health(rows: list[dict]) -> dict:
    """Mechanism health over one arm's answers, by bench/livebench/mechanisms.py
    `health` -- the harness's own counter, not a second one. Rows scored
    before score.py recorded `mechanisms` are counted as unrecorded."""
    recs = [r["mechanisms"] for r in rows if isinstance(r.get("mechanisms"), dict)]
    out = {"answers": len(rows), "recorded": len(recs)}
    if recs:
        out.update(_bench_sub("livebench", "mechanisms").health(recs))
        out["answers"] = len(rows)
    return out


def _livebench_electricity(cats: dict, rows: list[dict]) -> dict | None:
    """Electricity per question for one arm: wall seconds per answered
    question x the measured generating draw (mcp/power.py
    benchmark_estimate). An ESTIMATE -- nothing was metered during the run.
    Seconds come from summary.json (each category's seconds.total and n);
    an arm with no summary seconds falls back to its rows' `seconds`."""
    try:
        import power
    except Exception:                                            # noqa: BLE001
        return None
    total, n = 0.0, 0
    for v in cats.values():
        sec = v.get("seconds") if isinstance(v, dict) else None
        if isinstance(sec, dict) and isinstance(sec.get("total"), (int, float))                 and sec.get("n"):
            total += float(sec["total"])
            n += int(sec["n"])
    source = "summary"
    if not n:
        secs = [float(r["seconds"]) for r in rows
                if r.get("status") == "ok"
                and isinstance(r.get("seconds"), (int, float))]
        total, n, source = sum(secs), len(secs), "rows"
    est = power.benchmark_estimate(total, n)
    if est is not None:
        est["seconds_from"] = source
    return est


def _livebench_run(run_dir: str) -> dict:
    run_id = os.path.basename(os.path.normpath(run_dir))
    summ = _load_json(os.path.join(run_dir, "summary.json")) or {}
    comp = _load_json(os.path.join(run_dir, "comparison.json")) or {}
    cond = _load_json(os.path.join(run_dir, "condition.json")) or {}
    frozen_p = os.path.join(run_dir, "FROZEN")
    frozen = None
    if os.path.exists(frozen_p):
        try:
            frozen = open(frozen_p, encoding="utf-8").read().strip()[:400]
        except OSError:
            frozen = "FROZEN"
    rows = _livebench_rows(run_dir)
    progress = _livebench_progress(run_dir)
    arms_s = summ.get("arms") if isinstance(summ.get("arms"), dict) else {}
    # A frozen run is a record: nothing in it is running, so a progress log
    # (e.g. the smoke answers moved out of its summary) adds no arm to it.
    names = list(arms_s) + [a for a in rows if a not in arms_s] + \
        ([] if frozen else [p["arm"] for p in progress
                            if p["arm"] not in arms_s and p["arm"] not in rows])
    names = list(dict.fromkeys(names))
    # The retired pre-fix arm is excluded from every summary (its README says
    # so); a progress log for it must not re-introduce it here.
    names = [a for a in names if not a.startswith("yamadori_features")]
    arms = []
    for a in names:
        s = arms_s.get(a) or {}
        cats = {}
        for c, v in (s.get("categories") or {}).items():
            if not isinstance(v, dict):
                continue
            cats[c] = {"score": v.get("score"), "ci95": v.get("ci95"), "n": v.get("n"),
                       "tasks": {t: {"score": x.get("score"), "ci95": x.get("ci95"),
                                     "n": x.get("n")}
                                 for t, x in (v.get("tasks") or {}).items()
                                 if isinstance(x, dict)},
                       "finish_reason": v.get("finish_reason") or {},
                       "seconds": v.get("seconds"), "output_tokens": v.get("output_tokens")}
        ov = s.get("overall") or {}
        health = s.get("mechanism_health")
        if isinstance(health, dict):
            # summarize.py's health() counts the answers that carry a record;
            # the page states that n beside the answers scored.
            health = dict(health)
            health.setdefault("recorded", health.get("answers") or 0)
            health["answers"] = max(len(rows.get(a, [])), health["recorded"])
        else:
            health = _livebench_health(rows.get(a, []))
        arms.append({"arm": a, "label": (summ.get("arm_labels") or {}).get(a),
                     "overall": {"score": ov.get("score"), "ci95": ov.get("ci95"),
                                 "categories_included": ov.get("categories_included") or []},
                     "categories": cats, "status": s.get("status") or {},
                     "finish_reason": s.get("finish_reason") or {},
                     "rows": len(rows.get(a, [])),
                     "mechanism_health": health,
                     "electricity": _livebench_electricity(cats, rows.get(a, []))})
    paired = []
    for k, v in summ.items():
        if not (k.startswith("paired_") and isinstance(v, dict)):
            continue
        arm = k[len("paired_"):].rsplit("_minus_", 1)
        paired.append({"b": arm[0], "a": arm[1] if len(arm) > 1 else "bonsai",
                       "n_pairs": v.get("n_pairs"), "diff": v.get("diff_overall"),
                       "ci95": v.get("ci95_overall"),
                       "categories": {c: {kk: x.get(kk) for kk in (
                           "n_pairs", "a_score", "b_score", "diff", "ci95", "binary",
                           "b_only_correct", "a_only_correct", "mcnemar_p")}
                           for c, x in (v.get("categories") or {}).items()
                           if isinstance(x, dict)}})
    same = comp.get("same_questions") if isinstance(comp.get("same_questions"), dict) else {}
    refs = {}
    for arm, models in same.items():
        if isinstance(models, dict):
            refs[arm] = sorted(
                ({"model": m, "overall": x.get("overall"), "n": x.get("n"),
                  "missing": x.get("missing"), "categories": x.get("categories") or {}}
                 for m, x in models.items() if isinstance(x, dict)),
                key=lambda r: -(r["overall"] if isinstance(r["overall"], (int, float)) else -1))
    cur = comp.get("current_leaderboard") or {}
    base = {m: v for m, v in (cur.get("models") or {}).items()
            if isinstance(v, dict) and m.lower().startswith("qwen3.8 27b")}
    state = "ready" if any(a["overall"]["score"] is not None for a in arms) else (
        "running" if (progress or rows) else "empty")
    return {"run_id": run_id, "dir": _rel(run_dir), "state": state,
            "frozen": frozen, "condition": cond,
            "generated_at": summ.get("generated_at"),
            "summary_age_s": _age(os.path.join(run_dir, "summary.json")),
            "release": summ.get("release") or cond.get("question_release"),
            "population": summ.get("population") or {},
            "method": summ.get("method") or {},
            "arms": arms, "legend": _livebench_legend([a["arm"] for a in arms]),
            "paired": paired, "progress": progress,
            "same_question_refs": refs,
            "electricity_basis": _electricity_basis(),
            "published_2024_11_25": comp.get("published_2024_11_25") or {},
            "current_leaderboard_base": {
                "date": cur.get("leaderboard_date"), "models": base,
                "comparable": False} if base else None}


def _electricity_basis() -> dict | None:
    try:
        import power
        return power.benchmark_basis()
    except Exception:                                            # noqa: BLE001
        return None


def livebench_summary(root: str = LIVEBENCH_RESULTS) -> dict:
    """Every run under bench/livebench/results. The current run is the newest
    one without a FROZEN marker; frozen runs are records, shown as such."""
    dirs = [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(d)]
    if not dirs:
        return {"state": "empty", "file": _rel(root), "exists": os.path.exists(root),
                "file_age_s": None, "bad_lines": 0,
                "how": "run  bench/livebench/run_arm.sh, then score.py and summarize.py",
                "note": "no run directory under bench/livebench/results"}
    runs = [_livebench_run(d) for d in dirs]
    live = [r for r in runs if not r["frozen"]]
    cur = max(live, key=lambda r: os.path.getmtime(os.path.join(root, r["run_id"])),
              default=None)
    return {"state": "ready", "file": _rel(root), "exists": True,
            "file_age_s": None, "bad_lines": 0,
            "current": cur["run_id"] if cur else None,
            "runs": runs}


# --------------------------------------------------------------------------
# SWE-bench Verified (Mini)
# --------------------------------------------------------------------------

SWEBENCH_RESULTS = os.path.join(BENCH, "swebench", "results")
SWEBENCH_DOC = os.path.join(os.path.dirname(BENCH), "docs", "SWE-BENCH.md")
_SCORED_SWE = ("resolved", "unresolved", "empty_patch")
_XY_SUM = ("turns", "hint_turns", "hints_injected", "investigate_ran",
           "investigate_injected", "fanout_turns", "internal_tool_turns",
           "tools_offered_turns")


def md_tables(text: str) -> list[dict]:
    """Every pipe table in a markdown document, with the heading and the
    bold caption line it sits under. Cells keep their text; `num()` reads
    a number out of one."""
    out, heading, caption, cur = [], "", "", None
    para: list[str] = []            # a bold-led paragraph, which may wrap
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            heading, caption, para = s.lstrip("#").strip(), "", []
        elif s.startswith("|") and s.endswith("|"):
            if para:
                caption, para = " ".join(para), []
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cur is None:
                cur = {"heading": heading, "caption": caption, "header": cells, "rows": []}
            elif all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            else:
                cur["rows"].append(cells)
            continue
        elif s.startswith("**"):
            para = [s]
        elif s and para:
            para.append(s)
        elif not s and para:
            caption, para = " ".join(para), []
        if cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def num(cell) -> float | None:
    m = re.search(r"-?\d[\d,]*\.?\d*", str(cell or "").replace("**", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def swebench_leaderboard(doc: str = SWEBENCH_DOC) -> dict:
    """docs/SWE-BENCH.md section 7, tables A (same scaffold) and B (same agent
    family, older version). Published by the SWE-bench leaderboard on the full
    500 -- never measured here."""
    try:
        text = open(doc, encoding="utf-8").read()
    except OSError:
        return {"source": _rel(doc), "tables": []}
    tabs = []
    for t in md_tables(text):
        if not t["heading"].startswith("7."):
            continue
        cap = t["caption"].replace("**", "")
        key = cap[:2].rstrip(".") if cap[:2] in ("A.", "B.") else None
        if key not in ("A", "B") or "Verified" not in t["header"]:
            continue
        h = [c.lower() for c in t["header"]]
        vi = h.index("verified")
        rows = [{"model": r[0], "verified": num(r[vi]),
                 "detail": {t["header"][i]: r[i] for i in range(1, len(r))
                            if i != vi and i < len(t["header"])}}
                for r in t["rows"] if len(r) > vi]
        tabs.append({"key": key, "caption": cap.split(".", 1)[-1].strip(),
                     "same_scaffold": key == "A", "rows": rows})
    return {"source": _rel(doc), "n_instances": 500, "tables": tabs}


def _swe_arm_legend(arms: list[str]) -> list[dict]:
    try:
        m = _bench_sub("swebench", "arms")
        out = [legend_from_features(a, m.ARMS[a], m.ARM_NOTES.get(a, ""))
               for a in arms if a in m.ARMS]
        for r in out:
            r["body_effort"] = getattr(m, "BODY_EFFORT", None)
        return out + [{"arm": a, "kind": "unknown", "note": "not in bench/swebench/arms.py"}
                      for a in arms if a not in m.ARMS]
    except Exception:                                            # noqa: BLE001
        return [{"arm": a, "kind": "unknown", "note": ""} for a in arms]


def swebench_summary(root: str = SWEBENCH_RESULTS, doc: str = SWEBENCH_DOC) -> dict:
    runs = [d for d in glob.glob(os.path.join(root, "*"))
            if os.path.exists(os.path.join(d, "results.jsonl"))]
    board = swebench_leaderboard(doc)
    if not runs:
        return {"state": "empty", "file": _rel(root), "exists": os.path.exists(root),
                "file_age_s": None, "bad_lines": 0, "leaderboard": board,
                "how": "run  python bench/swebench/run.py --run-id ID --arms bonsai "
                       "--subset verified-mini --all --key-file K",
                "note": "no results.jsonl under bench/swebench/results"}
    run_dir = max(runs, key=lambda d: os.path.getmtime(os.path.join(d, "results.jsonl")))
    rows, meta = _read(os.path.join(run_dir, "results.jsonl"))
    inst = _load_json(os.path.join(run_dir, "instances.json")) or {}
    planned = len(inst.get("order") or []) or None
    last: dict[tuple, dict] = {}
    for r in rows:
        if isinstance(r, dict) and r.get("instance") and r.get("arm"):
            last[(r["arm"], r["instance"])] = r
    arm_names = list(dict.fromkeys(a for a, _ in last))
    arms = []
    for a in arm_names:
        rs = [r for (x, _), r in last.items() if x == a]
        outcome: dict[str, int] = {}
        for r in rs:
            k = r.get("eval_status") or "unknown"
            outcome[k] = outcome.get(k, 0) + 1
        sc = [r for r in rs if r.get("eval_status") in _SCORED_SWE]
        k = sum(1 for r in sc if r.get("eval_status") == "resolved")
        lo, hi = _wilson(k, len(sc))
        fin: dict[str, int] = {}
        xy: dict[str, int] = {x: 0 for x in _XY_SUM}
        tiers_seen: dict[str, int] = {}
        for r in rs:
            for f, c in (r.get("finish_reasons") or {}).items():
                fin[f] = fin.get(f, 0) + int(c or 0)
            x = r.get("x_yamadori") or {}
            for key in _XY_SUM:
                if isinstance(x.get(key), (int, float)):
                    xy[key] += x[key]
            for t, c in (x.get("tiers") or {}).items():
                tiers_seen[t] = tiers_seen.get(t, 0) + int(c or 0)
        arms.append({
            "arm": a, "attempted": len(rs), "scored": len(sc), "resolved": k,
            "rate": (k / len(sc)) if sc else None, "lo": lo, "hi": hi,
            "outcomes": outcome,
            "exit_status": {s: sum(1 for r in rs if r.get("exit_status") == s)
                            for s in sorted({str(r.get("exit_status")) for r in rs})},
            "median_steps": _median([r.get("steps") for r in rs]),
            "median_seconds": _median([r.get("seconds") for r in rs]),
            "median_prompt_tokens": _median([r.get("prompt_tokens") for r in rs]),
            "median_completion_tokens": _median([r.get("completion_tokens") for r in rs]),
            "format_errors": sum(int(r.get("format_errors") or 0) for r in rs),
            "length_events": sum(int(r.get("length_events") or 0) for r in rs),
            "proxy_error_lines": sum(int(r.get("proxy_error_lines") or 0) for r in rs),
            "finish_reasons": fin, "mechanisms": xy, "tiers_reported": tiers_seen,
            "instances": [{"instance": r.get("instance"), "eval_status": r.get("eval_status"),
                           "exit_status": r.get("exit_status"), "steps": r.get("steps"),
                           "seconds": r.get("seconds"),
                           "prompt_tokens": r.get("prompt_tokens"),
                           "completion_tokens": r.get("completion_tokens")} for r in rs],
        })
    stopped = None
    log = os.path.join(run_dir, "run.log")
    if os.path.exists(log):
        try:
            for line in open(log, encoding="utf-8", errors="replace"):
                if "STOPPED" in line:
                    stopped = line.strip()[:400]
        except OSError:
            pass
    return {"state": "ready" if arms else "running", **meta,
            "run_id": os.path.basename(run_dir), "subset": inst.get("subset"),
            "dataset": inst.get("dataset"), "planned": planned,
            "stopped": stopped, "arms": arms,
            "legend": _swe_arm_legend(arm_names or ["bonsai", "yamadori"]),
            "leaderboard": board}


# --------------------------------------------------------------------------
# Speed: the MTP staging measurements, and bench/longctx if it has run
# --------------------------------------------------------------------------

MTP_DOC = os.path.join(os.path.dirname(BENCH), "docs", "MTP-STAGING.md")
LONGCTX_RESULTS = os.path.join(BENCH, "longctx", "results")
_FRAC = re.compile(r"\((\d[\d,]*)/(\d[\d,]*)\)")


def mtp_tables(doc: str = MTP_DOC) -> dict:
    """Sections 6 (decode tok/s per build) and 7 (draft acceptance by content)
    of docs/MTP-STAGING.md, cell for cell. The heading goes with each table,
    because it names the card the numbers were taken on."""
    try:
        text = open(doc, encoding="utf-8").read()
    except OSError:
        return {"source": _rel(doc), "builds": None, "acceptance": []}
    builds, acc = None, []
    for t in md_tables(text):
        h = [c.lower() for c in t["header"]]
        if t["heading"].startswith("6.") and "mean" in h and builds is None:
            i = {k: h.index(k) for k in ("run", "head", "bi", "mean") if k in h}
            content = [c for c in ("ts", "bash", "prose") if c in h]
            rows = []
            for r in t["rows"]:
                if len(r) != len(h):
                    continue
                rows.append({"run": r[i["run"]].replace("**", ""),
                             "build": r[1].replace("**", ""),
                             "head": r[i.get("head", 2)], "bi": r[i.get("bi", 3)],
                             "mean_tps": num(r[i["mean"]]),
                             "by_content": {c: num(r[h.index(c)]) for c in content},
                             "acceptance": r[-1] if "acceptance" in h[-1] else None})
            builds = {"heading": t["heading"], "rows": rows,
                      "method": "median of 3 reps per cell; server timings.predicted_per_second"}
        elif t["heading"].startswith(("Lean file", "Abliterated graft")) and "acceptance" in h:
            rows = []
            for r in t["rows"]:
                if len(r) != len(h):
                    continue
                a = r[h.index("acceptance")]
                m = _FRAC.search(a)
                rows.append({"content": r[0], "n_prompts": num(r[h.index("n prompts")])
                             if "n prompts" in h else None,
                             "acceptance": num(a.split("(")[0]),
                             "accepted": num(m.group(1)) if m else None,
                             "drafted": num(m.group(2)) if m else None,
                             "tps_on": num(r[h.index("tok/s head on")]) if "tok/s head on" in h else None,
                             "tps_off": num(r[h.index("tok/s head off")]) if "tok/s head off" in h else None})
            acc.append({"file": t["heading"], "rows": rows})
    card = None
    m = re.search(r"^## 6\. (.+)$", text, re.M)
    if m:
        card = m.group(1).strip()
    return {"source": _rel(doc), "section6": card, "builds": builds,
            "acceptance": acc, "file_age_s": _age(doc)}


def longctx_summary(root: str = LONGCTX_RESULTS) -> dict:
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.exists(os.path.join(d, "rows.jsonl")))
    if not dirs:
        return {"state": "empty", "file": _rel(root), "exists": os.path.exists(root),
                "file_age_s": None, "bad_lines": 0,
                "how": "run  python bench/longctx/run.py --url URL --run-id ID",
                "note": "no run directory under bench/longctx/results holds rows"}
    lc = _bench_sub("longctx", "analyse")
    runs = [lc.load_run(d) for d in dirs]
    sums = [lc.summarise(r) for r in runs]
    return lc.dashboard_block(sums, [], os.path.join(root, "_compare", "summary.json"))


def speed_summary() -> dict:
    return {"state": "ready", "mtp": mtp_tables(), "longctx": longctx_summary()}


# --------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------

IMAGEGEN_RESULTS = os.path.join(BENCH, "imagegen", "results.jsonl")


def _img_status(r: dict) -> str:
    if r.get("skipped"):
        return "skipped"
    if r.get("killed"):
        return "killed"
    if r.get("rc") not in (0, None) or not r.get("image"):
        return "failed"
    return "errors_logged" if r.get("errors") else "ok"


def imagegen_summary(path: str = IMAGEGEN_RESULTS) -> dict:
    rows, meta = _read(path)
    if not meta["exists"] or not rows:
        return {"state": "empty", **meta,
                "how": "run  python bench/imagegen/measure.py --config CONFIG",
                "note": "" if meta["exists"] else "results.jsonl not found"}
    groups: dict[tuple, list[dict]] = {}
    probes, servers = [], []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("probe"):
            probes.append(r)
        elif isinstance(r.get("images"), list):
            servers.append(r)
        elif r.get("config"):
            groups.setdefault((r["config"], r.get("size") or "?"), []).append(r)
    configs = []
    for (cfg, size), rs in groups.items():
        st: dict[str, int] = {}
        for r in rs:
            s = _img_status(r)
            st[s] = st.get(s, 0) + 1
        ok = [r for r in rs if _img_status(r) in ("ok", "errors_logged")]
        ran = [r for r in rs if not r.get("skipped")]
        configs.append({
            "config": cfg, "size": size, "rows": len(rs), "status": st,
            "images": len(ok), "steps": _median([r.get("steps") for r in ok]),
            "median_wall_s": _median([r.get("wall_s") for r in ok]),
            "min_wall_s": min((r["wall_s"] for r in ok if isinstance(r.get("wall_s"), (int, float))), default=None),
            "max_wall_s": max((r["wall_s"] for r in ok if isinstance(r.get("wall_s"), (int, float))), default=None),
            "median_sample_s": _median([r.get("sample_s") for r in ok]),
            "median_sec_per_step": _median([r.get("sec_per_step") for r in ok]),
            "median_decode_s": _median([r.get("decode_s") for r in ok]),
            "median_encode_s": _median([r.get("encode_s") for r in ok]),
            # What an image costs is read off the images that were made; a
            # killed run's peak can include a second consumer on the card
            # (docs/IMAGEGEN.md "Incidents") and is listed with its kill.
            "max_delta_peak_mib": max((r["delta_peak_mib"] for r in ok
                                       if isinstance(r.get("delta_peak_mib"), (int, float))), default=None),
            "min_free_mib": min((r["min_free_mib"] for r in ok
                                 if isinstance(r.get("min_free_mib"), (int, float))), default=None),
            "max_temp_c": max((r["peak_temp_c"] for r in ok
                               if isinstance(r.get("peak_temp_c"), (int, float))), default=None),
            "attempts_run": len(ran),
            "kills": [{"why": r.get("kill_why") or "", "min_free_mib": r.get("min_free_mib"),
                       "delta_peak_mib": r.get("delta_peak_mib")} for r in rs if r.get("killed")],
            "skips": [r.get("why") or "skipped" for r in rs if r.get("skipped")],
            "errors": sorted({e for r in rs for e in (r.get("errors") or [])})[:6],
        })
    srv = []
    for r in servers:
        ims = [i for i in r["images"] if isinstance(i, dict)]
        srv.append({"config": r.get("config"), "images": len(ims),
                    "ok": sum(1 for i in ims if i.get("ok")),
                    "median_seconds": _median([i.get("seconds_client") for i in ims]),
                    "delta_peak_mib": r.get("delta_peak_mib"),
                    "min_free_mib": r.get("min_free_mib"), "killed": r.get("killed")})
    parti = sorted(glob.glob(os.path.join(os.path.dirname(path), "**", "*parti*"),
                             recursive=True))
    parti_meta = None
    for p in parti:
        d = _load_json(p) if p.endswith(".json") else None
        if d is not None:
            items = d if isinstance(d, list) else (d.get("prompts") or d.get("items") or [])
            cats: dict[str, int] = {}
            for it in items if isinstance(items, list) else []:
                c = it.get("category") if isinstance(it, dict) else None
                if c:
                    cats[c] = cats.get(c, 0) + 1
            parti_meta = {"file": _rel(p), "prompts": len(items) if isinstance(items, list) else None,
                          "categories": cats}
            break
    return {"state": "ready", **meta, "configs": configs, "server": srv,
            "idle_probes": [{"base_used_mib": p.get("base_used_mib"),
                             "idle_after_mib": _median(p.get("idle_after_mib") or []),
                             "after_kill_mib": p.get("after_kill_mib"),
                             "gen_512_s": p.get("gen_512_s"), "at": p.get("at")}
                            for p in probes],
            "partiprompts": parti_meta}


# --------------------------------------------------------------------------
# Published by PrismML. Not measured here, and labelled so wherever shown.
# --------------------------------------------------------------------------

MODEL_CARD = {
    "state": "ready",
    "publisher": "PrismML",
    "source": "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf",
    "measured_here": False,
    "columns": ["Ternary Bonsai 2 27B", "Qwen3.8 27B FP16"],
    "rows": [
        {"bench": "LiveCodeBench", "values": [90.05, 90.07]},
        {"bench": "HumanEval+", "values": [93.29, 95.12]},
        {"bench": "MBPP+", "values": [83.86, 83.07]},
        {"bench": "IFEval", "values": [91.50, 91.31]},
        {"bench": "BFCL v3", "values": [76.74, 74.92]},
        {"bench": "AIME25", "values": [96.67, 95.00]},
        {"bench": "20-benchmark mean", "values": [85.4, 83.9]},
    ],
}


def model_card_summary() -> dict:
    return MODEL_CARD


# --------------------------------------------------------------------------
# Domain suite
# --------------------------------------------------------------------------

# Unset: the newest group (the one running now). A group name pins one.
DOMAIN_GROUP = os.environ.get("YAMADORI_DOMAIN_GROUP") or None
DOMAIN_RESULTS = os.path.join(BENCH, "domain", "results")


def _domain_groups(root: str, an) -> dict[str, dict]:
    """Every bench/domain run, merged by its manifest `group` (a dir without
    one is its own group): what it planned, what it holds, what was excluded
    as stale and why. Uses analyse.py's own load/split_stale/scored."""
    groups: dict[str, dict] = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        mp, rp = os.path.join(d, "manifest.json"), os.path.join(d, "rows.jsonl")
        if not os.path.isdir(d) or not (os.path.exists(mp) or os.path.exists(rp)):
            continue
        man = _load_json(mp) or {}
        key = man.get("group") or os.path.basename(d)
        g = groups.setdefault(key, {"group": key, "dirs": [], "mtime": 0.0, "rows": 0,
                                    "stale_rows": 0, "stale": {}, "planned": set(),
                                    "done": set(), "arms": set(), "conditions": {}})
        g["dirs"].append(os.path.basename(d))
        g["mtime"] = max([g["mtime"]] + [os.path.getmtime(x) for x in (mp, rp) if os.path.exists(x)])
        ss = man.get("server_sampling") or {}
        for k, v in (("temperature", man.get("temperature")), ("effort", man.get("effort")),
                     ("min_p", ss.get("min_p")), ("suite", man.get("suite"))):
            if v is not None:
                g["conditions"][k] = v
        for run in man.get("runs") or []:
            for a in run.get("arms") or []:
                g["arms"].add(a)
                for t in run.get("tasks") or []:
                    g["planned"].add((t, a))
        rows, _ = an.load(d)
        cur, stale = an.split_stale(rows)
        g["rows"] += len(cur)
        g["stale_rows"] += len(stale)
        for r in stale:
            sc = r.get("stale_code")
            why = (sc.get("reason") if isinstance(sc, dict) else str(sc)) or "?"
            e = g["stale"].setdefault(why, {"reason": why, "rows": 0, "arms": {}})
            e["rows"] += 1
            e["arms"][r.get("arm") or "?"] = e["arms"].get(r.get("arm") or "?", 0) + 1
        for (t, a), r in an.last_by_pair(getattr(an, "reclassify", list)(cur)).items():
            g["arms"].add(a)
            if an.scored(r):
                g["done"].add((t, a))
    return groups


def _group_view(g: dict) -> dict:
    return {"group": g["group"], "dirs": g["dirs"], "age_s": round(time.time() - g["mtime"]),
            "rows": g["rows"], "stale_rows": g["stale_rows"],
            "stale": sorted(g["stale"].values(), key=lambda e: -e["rows"]),
            "planned_pairs": len(g["planned"]) or None, "scored_pairs": len(g["done"]),
            "arms": sorted(g["arms"]), "conditions": g["conditions"]}


def _domain_legend(arms: list[str], conditions: dict) -> list[dict]:
    try:
        run = _bench_sub("domain", "run")
        spec, notes, twin = run.ARMS, run.ARM_NOTES, run.SELF_CHECK
    except Exception:                                            # noqa: BLE001
        return [{"arm": a, "kind": "unknown", "note": ""} for a in arms]
    out = []
    for a in arms:
        if a not in spec:
            out.append({"arm": a, "kind": "unknown", "note": "not in bench/domain/run.py ARMS"})
            continue
        f = dict(spec[a])
        f.setdefault("effort", conditions.get("effort"))
        row = legend_from_features(a, f, notes.get(a, ""),
                                   client_tool="check_solution" if a in twin else None)
        if f.get("reasoning_cap"):
            row["reasoning_cap"] = f["reasoning_cap"]
        out.append(row)
    return out


def domain_summary(root: str = DOMAIN_RESULTS, group: str | None = DOMAIN_GROUP) -> dict:
    """The newest bench/domain group (or the one named), analysed by
    bench/domain/analyse.py -- which owns every statistic, so the page and
    the markdown report cannot disagree. Its lcb-shaped `dashboard` block is
    kept and the rest of the analysis rides beside it: every domain
    (contaminated ones flagged, not dropped), the per-arm headline, mechanism
    triggers and caveats. `groups` lists every group with what it planned,
    what it holds and what was excluded as stale, with the reason."""
    d = os.path.join(BENCH, "domain")
    if d not in sys.path:
        sys.path.insert(0, d)
    an = _bench_sub("domain", "analyse")
    groups = _domain_groups(root, an)
    if not groups:
        return an.dashboard_section(root)
    key = group if group in groups else max(groups, key=lambda k: groups[k]["mtime"])
    g = groups[key]
    views = sorted((_group_view(x) for x in groups.values()), key=lambda v: v["age_s"])
    first = os.path.join(root, g["dirs"][0])
    legend_conditions = {"effort": g["conditions"].get("effort")}
    if not g["rows"]:
        # Planned or running, nothing scorable yet (or every row stale).
        return {"state": "running", "file": _rel(first), "exists": True,
                "file_age_s": round(time.time() - g["mtime"]), "bad_lines": 0,
                "how": "run  python bench/domain/run.py --key-file KEY --run-id ID",
                "run_id": key, "group": key, "merged_run_dirs": g["dirs"],
                "current": _group_view(g), "groups": views,
                "legend": _domain_legend(sorted(g["arms"]), legend_conditions)}
    s = an.analyse(first)
    out = dict(s["dashboard"])

    def strip(block):
        return {**block, "comparisons": [{k: v for k, v in c.items() if k != "verdict"}
                                         for c in block.get("comparisons") or []]}
    out.update({
        "run_id": s.get("run_id"), "group": key,
        "merged_run_dirs": s.get("merged_run_dirs"),
        "conditions": s.get("conditions"),
        "rows_total": s.get("rows"), "stale_rows": s.get("stale_rows"),
        "stale": _group_view(g)["stale"],
        "current": _group_view(g), "groups": views,
        "reclassified_server_reasoning_cap": s.get("reclassified_server_reasoning_cap"),
        "task_counts": s.get("tasks"),
        "caveats": s.get("caveats") or [],
        "headline": strip(s.get("headline_uncontaminated") or {}),
        "contaminated_block": strip(s.get("contaminated_three_tsl") or {}),
        "per_arm_headline": s.get("per_arm_uncontaminated") or {},
        "by_domain": s.get("by_domain") or {},
        "triggers": s.get("triggers") or {},
        "legend": _domain_legend(s.get("arms") or [], s.get("conditions") or {}),
    })
    out["pairs"] = [{k: v for k, v in p.items() if k != "verdict"} for p in out.get("pairs") or []]
    return out


SECTIONS = (("domain", domain_summary), ("livebench", livebench_summary),
            ("swebench", swebench_summary), ("speed", speed_summary),
            ("imagegen", imagegen_summary), ("model_card", model_card_summary),
            ("lcb", lcb_summary), ("retrieval", retrieval_summary),
            ("recipe", recipe_summary))


def results() -> dict:
    out: dict = {"served_at": time.time(), "sections": {}}
    for name, fn in SECTIONS:
        try:
            out["sections"][name] = fn()
        except Exception as e:                                   # noqa: BLE001
            # Named component, and what to check. Never "something went wrong".
            out["sections"][name] = {
                "state": "error",
                "component": f"mcp/dash_results.py {fn.__name__}",
                "error": f"{type(e).__name__}: {e}",
                "check": "the result files under bench/ for this section and "
                         "that its bench/ imports resolve"}
    return out


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Yamadori results</title>
<link rel="stylesheet" media="print" onload="this.media='all'"
 href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>
/* Tokens straight from DESIGN.md frontmatter. Borders carry their value in
   the token and are drawn at FULL opacity -- #516854 measures 3.06:1 on the
   ground, and the same colour at 35% measures 1.38:1, which is the bug the
   design system was written to stop. */
:root{
  --bg:#0f131c; --lowest:#0a0e17; --low:#181c25; --raised:#1c2029;
  --high:#262a34; --highest:#31353f;
  --fg:#dfe2ef; --fg-var:#b9cbb9; --outline:#849585; --outline-variant:#516854;
  --primary:#00ff88;      /* alive, healthy, primary action */
  --tertiary:#5cf2ff;     /* throughput and measurement in flight */
  --cut:#d4004b;          /* destructive, refused, cut */
  --error:#ffb4ab;        /* a fault the operator must resolve */
  --caution:#ffb2ba;      /* a number that cannot answer the question */
  --r:0.125rem; --rlg:0.25rem;
  --xs:0.25rem; --sm:0.5rem; --md:0.75rem; --lg:1.25rem; --xl:2rem;
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --disp:'Syne',var(--mono);
}
*{box-sizing:border-box}
body{margin:0;padding:var(--md);background:var(--bg);color:var(--fg);
  font:400 14px/1.6 var(--mono);letter-spacing:-0.01em;
  font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1,"zero" 1}
h1{font:800 28px/1.2 var(--disp);letter-spacing:-0.03em;margin:0 0 var(--xs)}
h2{font:700 18px/1.3 var(--disp);letter-spacing:0;margin:0}
h3{font:600 15px/1.4 var(--mono);letter-spacing:-0.01em;margin:0 0 var(--sm)}
.label{font:500 11px/1.4 var(--mono);letter-spacing:0.06em;
  text-transform:uppercase;color:var(--outline)}
.small{font-size:12px;letter-spacing:0;color:var(--fg-var)}
.dim{color:var(--fg-var)}
a{color:var(--fg-var)}

header{display:flex;flex-wrap:wrap;gap:var(--md);align-items:flex-end;
  justify-content:space-between;margin-bottom:var(--lg)}
.rule{max-width:62ch;color:var(--fg-var);font-size:12px;letter-spacing:0}

/* Elevation is substrate plus a 1px line. No drop shadows anywhere. */
section{background:var(--low);border:1px solid var(--outline-variant);
  border-radius:var(--rlg);padding:var(--lg);margin-bottom:var(--lg)}
.group{background:var(--raised);border:1px solid var(--outline-variant);
  border-radius:var(--r);padding:var(--md);margin-top:var(--md)}
.sechead{display:flex;flex-wrap:wrap;gap:var(--md);align-items:baseline;
  justify-content:space-between;margin-bottom:var(--md)}

/* Age. Grey past 30s -- a number from four minutes ago shown as current is
   the same failure as an invented one. */
.age{font:500 11px/1.4 var(--mono);letter-spacing:0.06em;color:var(--fg-var);
  border:1px solid var(--outline-variant);border-radius:var(--r);
  padding:2px 6px;white-space:nowrap}
.age.live{color:var(--primary)}
.age.stale{color:var(--outline);border-color:var(--outline-variant)}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;
  background:var(--primary);margin-right:6px;vertical-align:middle}
.age.stale .dot{background:var(--outline)}
@media (prefers-reduced-motion: no-preference){
  .age.live .dot{animation:breathe 2.4s ease-in-out infinite}
}
@keyframes breathe{0%,100%{opacity:1}50%{opacity:.35}}

table{width:100%;border-collapse:collapse;font-size:13px}
th{font:500 11px/1.4 var(--mono);letter-spacing:0.06em;text-transform:uppercase;
  color:var(--outline);text-align:right;padding:var(--xs) var(--sm);
  border-bottom:1px solid var(--outline-variant);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:var(--xs) var(--sm);
  border-bottom:1px solid var(--outline-variant);white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
.scroll{overflow-x:auto}
.ci{color:var(--fg-var);font-size:12px}
.cost{color:var(--tertiary)}
.faulty{color:var(--error)}
.nil{color:var(--outline)}

.note{border:1px solid var(--outline-variant);border-left-width:3px;
  border-radius:var(--r);padding:var(--md);margin-top:var(--md);
  font-size:13px;background:var(--raised)}
.note .label{display:block;margin-bottom:var(--xs)}
.note.caution{border-left-color:var(--caution)}
.note.caution .label{color:var(--caution)}
.note.fault{border-left-color:var(--error)}
.note.fault .label{color:var(--error)}
.note.cut{border-left-color:var(--cut)}
.note.cut .label{color:var(--cut)}

.strip{display:flex;flex-wrap:wrap;gap:var(--lg)}
.stat{min-width:9ch}
.stat b{display:block;font:600 15px/1.4 var(--mono);font-weight:600}
.stat .label{display:block}
.stat.fault b{color:var(--error)}

.cards{display:grid;gap:var(--md);
  grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr));
  margin-top:var(--md)}
.card{background:var(--raised);border:1px solid var(--outline-variant);
  border-radius:var(--r);padding:var(--md)}
.card.caution{border-left:3px solid var(--caution)}
.kv{display:flex;justify-content:space-between;gap:var(--md);
  padding:2px 0;font-size:13px}
.kv span:last-child{color:var(--fg)}

code,.code{font-family:var(--mono);background:var(--lowest);
  border:1px solid var(--outline-variant);border-radius:var(--r);
  padding:1px 5px;font-size:12px;color:var(--fg-var);white-space:nowrap}

button,input{font:inherit;background:var(--high);color:var(--fg);
  border:1px solid var(--outline-variant);border-radius:var(--r);
  padding:var(--sm) var(--md);min-height:44px}
button{cursor:pointer;min-width:44px}
button:hover{border-color:var(--primary);color:var(--primary)}
:focus-visible{outline:2px solid var(--primary);outline-offset:2px}

.skel{background:var(--high);border-radius:var(--r);height:12px;
  margin:var(--sm) 0;max-width:46ch}
@media (prefers-reduced-motion: no-preference){
  .skel{animation:breathe 1.6s ease-in-out infinite}
}
@media (prefers-reduced-motion: reduce){
  *{animation:none !important;transition:none !important}
}
</style>

<header>
  <div>
    <h1>Results</h1>
    <div class="rule">Every number here is measured. Cost sits beside
      correctness; errors are counted apart from failures; a comparison is
      only called when the test could have produced a call.</div>
  </div>
  <div id="age" class="age">connecting</div>
</header>

<div id="login" class="group" style="display:none">
  <h3>API key</h3>
  <div class="small">The same key your editor sends. It is kept in this
    browser only.</div>
  <div style="display:flex;gap:var(--sm);margin-top:var(--sm);flex-wrap:wrap">
    <input id="key" type="password" size="34" placeholder="sk-...">
    <button onclick="saveKey()">sign in</button>
  </div>
  <div class="small" id="loginmsg"></div>
</div>

<main id="main"></main>

<script>
const KEY=()=>localStorage.getItem('yamadori_key')||'';
const H=()=>({'Content-Type':'application/json','Authorization':'Bearer '+KEY()});
const $=id=>document.getElementById(id);
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,c=>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

let LAST_OK=0, DATA=null, FAILED=null, SIG=null, RENDER_AT=0;

function saveKey(){
  localStorage.setItem('yamadori_key',$('key').value.trim());
  $('login').style.display='none'; poll();
}
function showLogin(msg){
  $('login').style.display='block'; $('loginmsg').textContent=msg||'';
}

/* ---- formatting. Nothing here invents a value: null renders as a dash and
   the reason it is null is printed next to it. ---- */
const pct=x=>x==null?'--':(x*100).toFixed(1)+'%';
const pc0=x=>x==null?'--':Math.round(x*100)+'%';
const num=(x,d)=>x==null?'--':Number(x).toFixed(d==null?0:d);
const nil='<span class="nil">not reported</span>';
function dur(s){
  if(s==null)return '--';
  if(s<90)return Math.round(s)+'s';
  if(s<5400)return Math.round(s/60)+'m';
  if(s<172800)return Math.round(s/3600)+'h';
  return Math.round(s/86400)+'d';
}
function pval(p){ return p==null?'--':(p<0.0001?'<0.0001':Number(p).toFixed(4)); }

function tickAge(){
  const el=$('age');
  if(FAILED){el.className='age stale';
    el.innerHTML='<span class="dot"></span>'+esc(FAILED);return;}
  if(!LAST_OK){el.className='age';el.textContent='loading';return;}
  const s=(Date.now()-LAST_OK)/1000;
  el.className='age '+(s>30?'stale':'live');
  el.innerHTML='<span class="dot"></span>polled '+dur(s)+' ago'+
    (s>30?' -- STALE':'');
  // File age counts from the poll that drew it, which is not always the most
  // recent poll -- the DOM is only rewritten when a number changed.
  const since=RENDER_AT?(Date.now()-RENDER_AT)/1000:0;
  document.querySelectorAll('.fileage').forEach(n=>{
    const base=Number(n.dataset.base||0);
    n.textContent='file written '+dur(base+since)+' ago';
    n.className='age fileage '+(s>30?'stale':'');
  });
}

function fileBadge(sec){
  const a=sec.file_age_s;
  return a==null?'':'<span class="age fileage" data-base="'+a+'">file written '+
    dur(a)+' ago</span>';
}

/* ---- the five states ---- */
function shell(title,sec,body){
  return '<section><div class="sechead"><div><h2>'+esc(title)+'</h2>'+
    '<div class="label">'+esc(sec.file||'')+
    (sec.bad_lines?' -- '+sec.bad_lines+' unparseable lines':'')+'</div></div>'+
    fileBadge(sec)+'</div>'+body+'</section>';
}
function stateEmpty(title,sec){
  return shell(title,sec,
    '<div class="note"><span class="label">empty</span>'+
    'No results in <span class="code">'+esc(sec.file)+'</span>'+
    (sec.note?' -- '+esc(sec.note):'')+'.<br>To produce them, '+
    '<span class="code">'+esc(sec.how)+'</span></div>');
}
function stateError(title,sec){
  return shell(title,sec||{},
    '<div class="note fault"><span class="label">error</span>'+
    esc(sec.component)+' raised <span class="code">'+esc(sec.error)+'</span>'+
    '<br>Check: '+esc(sec.check)+'</div>');
}
function stateLoading(title){
  return '<section><div class="sechead"><h2>'+esc(title)+'</h2>'+
    '<span class="age">loading</span></div>'+
    '<div class="skel" style="width:60%"></div>'+
    '<div class="skel" style="width:40%"></div>'+
    '<div class="skel" style="width:52%"></div></section>';
}

/* ---- power, stated before any comparison is shown (protocol rule 4) ---- */
function powerNote(p,what){
  const floor=p.min_discordant_for_sig;
  if(!p.comparison_possible){
    return '<div class="note caution"><span class="label">'+
      'underpowered -- no comparison is possible</span>'+
      p.n_paired+' '+what+' complete under every arm. McNemar\'s exact test '+
      'cannot reach p &lt; '+p.alpha+' below '+floor+' discordant pairs even '+
      'if every one falls the same way, and at most '+p.max_possible_discordant+
      ' are possible here. The rates below are raw pass/fail, which is a '+
      'different and valid question -- did it pass anything at all -- and '+
      'they are not a comparison between arms.</div>';
  }
  return '<div class="note"><span class="label">power</span>'+
    p.n_paired+' '+what+' complete under every arm. '+floor+
    ' discordant pairs are the fewest that could ever reach p &lt; '+p.alpha+
    '; each comparison below prints its own discordant count.</div>';
}

function armRow(a){
  const ci='<span class="ci"> ['+pc0(a.lo)+'-'+pc0(a.hi)+']</span>';
  const tin=a.tok_in==null?nil:num(a.tok_in);
  const tout=a.tok_out==null?nil:num(a.tok_out);
  const rep=(a.tok_in_reported||a.tok_out_reported)&&
    (a.tok_in_reported<a.n||a.tok_out_reported<a.n)
      ? '<span class="ci"> ('+a.tok_in_reported+'/'+a.n+')</span>':'';
  return '<tr><td>'+esc(a.arm)+'</td>'+
    '<td>'+a.k+'/'+a.n+' = '+pct(a.rate)+ci+'</td>'+
    '<td class="cost">'+num(a.mean_s,1)+'</td>'+
    '<td class="cost">'+tin+rep+'</td>'+
    '<td class="cost">'+tout+'</td>'+
    '<td'+(a.empty?' class="faulty"':'')+'>'+a.empty+'</td>'+
    '<td'+(a.len_cut?' class="faulty"':'')+'>'+a.len_cut+'</td>'+
    '<td'+(a.errors?' class="faulty"':'')+'>'+a.errors+'</td></tr>';
}

function mcnemarCard(p,unit){
  const warn=p.discordant<10
    ? '<div class="note caution" style="margin-top:var(--sm)">'+
      '<span class="label">cannot detect anything but a large effect</span>'+
      'only '+p.discordant+' discordant '+unit+'. Problems both arms get '+
      'right carry no information about the difference and are excluded, so '+
      'this is the n the test actually runs on.</div>'
    : '';
  const v=p.verdict
    ? '<div class="kv"><span>verdict</span><span>'+esc(p.verdict)+'</span></div>'
    : '<div class="kv"><span>verdict</span><span class="nil">none issued</span></div>';
  return '<div class="card'+(p.discordant<10?' caution':'')+'">'+
    '<h3>'+esc(p.a)+' vs '+esc(p.b)+'</h3>'+
    '<div class="kv"><span>'+esc(p.b)+' solved, '+esc(p.a)+' missed</span><span>'+
      p.b_only+'</span></div>'+
    '<div class="kv"><span>'+esc(p.a)+' solved, '+esc(p.b)+' missed</span><span>'+
      p.a_only+'</span></div>'+
    '<div class="kv"><span>discordant pairs</span><span>'+p.discordant+
      '</span></div>'+
    '<div class="kv"><span>McNemar exact p</span><span>'+pval(p.p)+'</span></div>'+
    v+warn+'</div>';
}

function signTable(p){
  const usable=(p.signs||[]).filter(s=>!s.insufficient);
  if(!usable.length){
    const most=Math.max(0,...(p.signs||[]).map(s=>s.pairs||0));
    return '<div class="small dim" style="margin-top:var(--sm)">'+
      'no sign test: '+p.both_solved+' problems solved by both arms, and a '+
      'sign test needs at least 3 paired values (most complete metric has '+
      most+').</div>';
  }
  return '<div class="scroll"><table style="margin-top:var(--sm)"><thead><tr>'+
    '<th>metric (both solved, n='+p.both_solved+')</th><th>'+esc(p.a)+
    ' median</th><th>'+esc(p.b)+' median</th><th>better-worse</th><th>p</th>'+
    '<th>verdict</th></tr></thead><tbody>'+
    usable.map(s=>'<tr><td>'+esc(s.metric)+'</td><td>'+num(s.median_a,2)+
      '</td><td>'+num(s.median_b,2)+'</td><td>'+s.better+'-'+s.worse+
      '</td><td>'+pval(s.p)+'</td><td>'+
      (s.verdict?esc(s.verdict):'<span class="nil">none issued</span>')+
      '</td></tr>').join('')+'</tbody></table></div>'+
    '<div class="small dim">sign test on the same problem; ties within 5% '+
    'ignored. Peak memory is a sampled lower bound, not an exact accounting.'+
    '</div>';
}

function renderLcb(sec){
  if(sec.state==='empty')return stateEmpty('LiveCodeBench v6 -- tiers',sec);
  if(sec.state==='error')return stateError('LiveCodeBench v6 -- tiers',sec);
  const pr=sec.progress, arms=sec.arms;
  let h='';

  /* progress: errors are their own outcome and are never folded into a miss
     rate, because a 502 is not a wrong answer. */
  h+='<div class="group"><div class="strip">'+
    '<div class="stat"><span class="label">problems attempted</span><b>'+
      pr.attempted+'</b></div>'+
    '<div class="stat"><span class="label">complete under all '+arms.length+
      ' arms</span><b>'+pr.complete_all_arms+'</b></div>'+
    '<div class="stat"><span class="label">scored attempts</span><b>'+
      pr.scored_rows+'</b></div>'+
    '<div class="stat'+(pr.errors?' fault':'')+'"><span class="label">'+
      'generation errors</span><b>'+pr.errors+'</b></div>'+
    '</div>';
  if(pr.errors){
    const rows=[];
    for(const a of Object.keys(pr.errors_by_arm)){
      const k=pr.errors_by_arm[a];
      const tot=Object.values(k).reduce((x,y)=>x+y,0);
      if(!tot)continue;
      rows.push('<div class="kv"><span>'+esc(a)+'</span><span>'+tot+' -- '+
        Object.entries(k).sort((x,y)=>y[1]-x[1])
          .map(e=>esc(e[0])+' '+e[1]).join(', ')+'</span></div>');
    }
    h+='<div class="note fault"><span class="label">errors, excluded from '+
      'scoring</span>A generation that never happened is not a failed '+
      'solution. These are excluded from every rate on this page, which also '+
      'means the paired set is whatever survived rather than the sample that '+
      'was drawn.'+rows.join('')+'</div>';
  }
  h+='</div>';

  h+=powerNote(sec.power,'problems');

  /* protocol rule 3, checked rather than assumed */
  const sus=sec.suspect||{};
  if(sus.identical_across_arms||(sus.extreme||[]).length){
    const bits=[];
    if(sus.identical_across_arms)bits.push('every arm scored the same count');
    if((sus.extreme||[]).length)bits.push('exactly 0% or 100% on '+
      sus.extreme.map(esc).join(', '));
    h+='<div class="note caution"><span class="label">check the harness '+
      'first</span>'+bits.join('; and ')+'. A rate identical across every '+
      'condition, or exactly 0 or 100 percent, is a harness bug until '+
      'something proves otherwise -- at this n it is also what a working '+
      'harness looks like, and the two are not distinguishable from here.'+
      '</div>';
  }

  h+='<div class="group"><h3>Per arm, on the '+pr.complete_all_arms+
    ' problems complete under every arm</h3><div class="scroll"><table>'+
    '<thead><tr><th>arm</th><th>pass@1 (Wilson 95%)</th><th>mean s</th>'+
    '<th>prompt tok</th><th>completion tok</th><th>empty reply</th>'+
    '<th>length-cut</th><th>errors</th></tr></thead><tbody>'+
    sec.arm_table.map(armRow).join('')+'</tbody></table></div>'+
    '<div class="small dim">Latency and tokens are the cost half of the '+
    'comparison and sit in the same row as correctness: an arm that wins by '+
    'spending three times the tokens has not obviously won. Token counts are '+
    'the server\'s own usage block; where it sent none, nothing is shown.'+
    '</div></div>';

  h+='<div class="group"><h3>Raw pass/fail, every scored attempt</h3>'+
    '<div class="small dim">Not a comparison. Different denominators per arm, '+
    'unpaired, and it answers a different question: did this arm pass '+
    'anything at all.</div><div class="scroll"><table><thead><tr><th>arm</th>'+
    '<th>scored attempts</th><th>passed</th><th>rate (Wilson 95%)</th>'+
    '</tr></thead><tbody>'+
    sec.raw.map(r=>'<tr><td>'+esc(r.arm)+'</td><td>'+r.attempts+'</td><td>'+
      r.passed+'</td><td>'+(r.attempts?pct(r.rate)+
      '<span class="ci"> ['+pc0(r.lo)+'-'+pc0(r.hi)+']</span>':
      '<span class="nil">nothing scored</span>')+'</td></tr>').join('')+
    '</tbody></table></div></div>';

  h+='<div class="group"><h3>Paired comparison -- McNemar exact</h3>'+
    '<div class="cards">'+
    sec.pairs.map(p=>mcnemarCard(p,'pairs')).join('')+'</div>'+
    sec.pairs.map(p=>'<div style="margin-top:var(--md)"><span class="label">'+
      esc(p.a)+' vs '+esc(p.b)+' -- among problems both solved</span>'+
      signTable(p)+'</div>').join('')+'</div>';

  if(sec.difficulty.length){
    h+='<div class="group"><h3>Pass rate by difficulty</h3><div class="scroll">'+
      '<table><thead><tr><th>difficulty</th><th>n</th>'+
      arms.map(a=>'<th>'+esc(a)+'</th>').join('')+'</tr></thead><tbody>'+
      sec.difficulty.map(d=>'<tr><td>'+esc(d.difficulty)+'</td><td>'+d.n+
        '</td>'+arms.map(a=>'<td>'+d.by_arm[a].k+'/'+d.by_arm[a].n+' = '+
        pct(d.by_arm[a].rate)+'</td>').join('')+'</tr>').join('')+
      '</tbody></table></div><div class="small dim">Stratified sampling puts '+
      'roughly equal counts in each tier, so these n are small by '+
      'construction and no interval is shown for them.</div></div>';
  }else{
    h+='<div class="group"><h3>Pass rate by difficulty</h3>'+
      '<div class="note"><span class="label">empty</span>No problem is yet '+
      'complete under every arm, so there is nothing to break down by '+
      'difficulty. '+esc(sec.how||'')+'</div></div>';
  }

  h+='<div class="group"><h3>Why the failures failed</h3>'+
    '<div class="small dim">"Wrong output" is a model result. "Crashed before '+
    'the model\'s code ran" is a bug in the scaffolding. Folding them '+
    'together is how a missing typing import once turned every functional '+
    'problem into a model failure.</div>'+
    '<div class="cards">'+arms.map(a=>{
      const f=sec.failures[a]||{passed:0,failed:0,buckets:[]};
      return '<div class="card"><h3>'+esc(a)+'</h3>'+
        '<div class="kv"><span>passed</span><span>'+f.passed+'</span></div>'+
        '<div class="kv"><span>failed</span><span>'+f.failed+'</span></div>'+
        (f.buckets.length?f.buckets.map(b=>'<div class="kv"><span'+
          (b.scaffolding?' class="faulty"':'')+'>'+esc(b.bucket)+'</span>'+
          '<span>'+b.n+'</span></div>').join('')
          :'<div class="small dim">no scored failures</div>')+'</div>';
    }).join('')+'</div>';
  if(sec.scaffolding_total){
    h+='<div class="note fault"><span class="label">scaffolding, not model '+
      'error</span>'+sec.scaffolding_total+' scored failures crashed before '+
      'the model\'s code ran -- an undefined name the runner\'s preamble '+
      'exists to provide, or an exception raised by the runner itself. They '+
      'are counted in the pass@1 denominator above because the harness scored '+
      'them, and they should not be read as model results.</div>';
  }
  h+='</div>';

  return shell('LiveCodeBench v6 -- tiers',sec,h);
}

function renderRetrieval(sec){
  const t='Retrieval -- symbol table vs embedding vs embed+rerank';
  if(sec.state==='empty')return stateEmpty(t,sec);
  if(sec.state==='error')return stateError(t,sec);
  let h=powerNote(sec.power,'tasks');
  h+='<div class="group"><div class="scroll"><table><thead><tr><th>arm</th>'+
    '<th>hit@1 (Wilson 95%)</th><th>hit@5</th><th>MRR</th><th>mean ms</th>'+
    '<th>cost vs first</th><th>truth absent</th></tr></thead><tbody>'+
    sec.arm_table.map(a=>'<tr><td>'+esc(a.arm)+
      (a.extreme?' <span class="ci">(see note)</span>':'')+'</td>'+
      '<td>'+a.hit1+'/'+a.n+' = '+pct(a.hit1_rate)+'<span class="ci"> ['+
        pc0(a.hit1_lo)+'-'+pc0(a.hit1_hi)+']</span></td>'+
      '<td>'+pct(a.hit5_rate)+'<span class="ci"> ['+pc0(a.hit5_lo)+'-'+
        pc0(a.hit5_hi)+']</span></td>'+
      '<td>'+num(a.mrr,3)+'</td>'+
      '<td class="cost">'+num(a.mean_ms,2)+'</td>'+
      '<td class="cost">'+num(a.cost_x,1)+'x</td>'+
      '<td>'+a.absent+'</td></tr>').join('')+'</tbody></table></div>'+
    '<div class="small dim">Latency sits beside accuracy for the same reason '+
    'as above. "Truth absent" is the ground-truth path missing from the top-k '+
    'entirely, which is rank 0 and contributes nothing to MRR.</div></div>';
  h+='<div class="cards">'+sec.pairs.map(p=>mcnemarCard(p,'tasks')).join('')+
    '</div>';
  h+='<div class="group"><h3>hit@1 by source</h3><div class="scroll"><table>'+
    '<thead><tr><th>source</th><th>n</th>'+
    sec.arms.map(a=>'<th>'+esc(a)+'</th>').join('')+'</tr></thead><tbody>'+
    sec.sources.map(s=>'<tr><td>'+esc(s.source)+'</td><td>'+s.n+'</td>'+
      sec.arms.map(a=>'<td>'+pct(s.by_arm[a])+'</td>').join('')+
      '</tr>').join('')+'</tbody></table></div></div>';
  h+=(sec.caveats||[]).map(c=>'<div class="note cut">'+
    '<span class="label">what these numbers are not</span>'+esc(c)+
    '</div>').join('');
  return shell(t,sec,h);
}

function renderRecipe(sec){
  const t='Recipe oracle';
  if(sec.state==='empty')return stateEmpty(t,sec);
  if(sec.state==='error')return stateError(t,sec);
  let h='<div class="group"><div class="strip">'+
    '<div class="stat"><span class="label">problems attempted</span><b>'+
      sec.progress.attempted+'</b></div>'+
    '<div class="stat"><span class="label">complete under all '+
      sec.arms.length+' arms</span><b>'+sec.progress.complete_all_arms+
      '</b></div>'+
    '<div class="stat'+(sec.progress.errors?' fault':'')+'">'+
      '<span class="label">generation errors</span><b>'+sec.progress.errors+
      '</b></div></div></div>';
  h+=powerNote(sec.power,'problems');
  h+='<div class="group"><div class="scroll"><table><thead><tr><th>arm</th>'+
    '<th>pass@1 (Wilson 95%)</th><th>median slowest case s</th>'+
    '<th>median peak KB</th></tr></thead><tbody>'+
    sec.arm_table.map(a=>'<tr><td>'+esc(a.arm)+'</td><td>'+a.k+'/'+a.n+' = '+
      pct(a.rate)+'<span class="ci"> ['+pc0(a.lo)+'-'+pc0(a.hi)+']</span></td>'+
      '<td class="cost">'+num(a.median_max_s,3)+'</td>'+
      '<td class="cost">'+num(a.median_peak_kb)+'</td></tr>').join('')+
    '</tbody></table></div>'+
    '<div class="kv"><span>oracle (best arm per problem, the ceiling no '+
    'selector can beat)</span><span>'+sec.oracle.k+'/'+sec.oracle.n+' = '+
    pct(sec.oracle.rate)+'</span></div></div>';
  return shell(t,sec,h);
}

function render(){
  if(!DATA){
    $('main').innerHTML=stateLoading('LiveCodeBench v6 -- tiers')+
      stateLoading('Retrieval')+stateLoading('Recipe oracle');
    return;
  }
  const s=DATA.sections;
  /* Rewritten only when the measurements changed. A ten-second poll that
     replaces the DOM regardless throws away the operator's scroll position
     and selection to redraw identical numbers. */
  const sig=JSON.stringify(s,(k,v)=>k==='file_age_s'?undefined:v);
  if(sig!==SIG){
    SIG=sig; RENDER_AT=LAST_OK||Date.now();
    $('main').innerHTML=renderLcb(s.lcb)+renderRetrieval(s.retrieval)+
      renderRecipe(s.recipe);
  }
  tickAge();
}

async function poll(){
  if(!KEY()){showLogin('');FAILED='no API key';tickAge();return;}
  try{
    const res=await fetch('/dash/api/results',{headers:H()});
    if(res.status===401){showLogin('that key was not accepted');
      FAILED='401 unauthorised';tickAge();return;}
    if(!res.ok){FAILED='HTTP '+res.status+' from /dash/api/results';
      tickAge();return;}
    DATA=await res.json(); LAST_OK=Date.now(); FAILED=null;
    $('login').style.display='none';
    render();
  }catch(e){
    FAILED='mcp/server.py unreachable: '+e.message;
    tickAge();
  }
}

render();
poll();
setInterval(poll,10000);
setInterval(tickAge,1000);
</script>
"""


# The shared header, injected after this page's own stylesheet rather than
# baked into the template: the page was measured against the stylesheet above,
# and `nav_block` brings only the rules its own markup needs, scoped under
# `.shell-head`. Without this the three pages exist but nothing links them.
PAGE = PAGE.replace("</style>",
                    "</style>" + dash_shell.nav_block("results"), 1)



def handle_get(path: str):
    """(status, content_type, body_bytes), or None if this path is not ours."""
    p = path.rstrip("/")
    if p == "/dash/results":
        return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")
    if p == "/dash/api/results":
        return 200, "application/json", json.dumps(results()).encode()
    return None


if __name__ == "__main__":
    d = results()
    for name, sec in d["sections"].items():
        print(f"\n== {name}: {sec.get('state')}  {sec.get('file', '')}")
        if sec.get("state") == "empty":
            print(f"   {sec['how']}")
            continue
        if sec.get("state") == "error":
            print(f"   {sec['error']}")
            continue
        print("   " + json.dumps(sec.get("progress") or sec.get("power"))[:200])
        for a in sec.get("arm_table", []):
            print("   " + json.dumps(a))
        for p in sec.get("pairs", []):
            print(f"   {p['a']} vs {p['b']}: discordant={p['discordant']} "
                  f"p={p['p']:.4f} verdict={p['verdict']}")
