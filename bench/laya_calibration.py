#!/usr/bin/env python
"""Does Laya's routing decision work, and does structured state fix it?

WHAT THIS ANSWERS, AND WHY IT EXISTS

`mcp/shomen.py:route_in` asks Laya to sort an incoming question into
`investigate` / `answer_directly` / `clarify`, averaged over option orderings,
and ABSTAINS when the top-two margin falls under a gate. Spot-checked on five
questions it abstained on four. Five questions is an anecdote, and the gate it
abstains against (0.3) was copied from a different experiment in
`bench/mechanisms/selectors.py` and marked UNCALIBRATED in the source.

RESULTS ARE IN docs/LAYA.md. Read that first; this file regenerates it.
Short version: the three-way question cannot be made usable at any gate, the
structured-state hypothesis is refuted, and exactly one configuration works --
a BINARY "does this need our source?" question on a PROSE state at gate 0.30.

This file replaces the anecdote with n=59 labelled questions and answers three
separate questions that were previously answered as one:

  1. HOW OFTEN does route_in abstain, and when it does decide, IS IT RIGHT?
  2. Does handing Laya a STRUCTURED state -- cheap signals computed by
     `mcp/domains.py`, `mcp/discover.py` and the index -- instead of raw prose
     change either number? This is run PAIRED: the same 49 items, the same
     primitive, the same options, only the state string differs.
  3. Where should the gate actually sit, given the labels?

and the same for `distil`'s two questions over 29 labelled findings, half of
which are deliberately fabricated -- confident prose citing files it does not
describe, which is the exact failure the callosum exists to catch.

WHAT IS DELIBERATELY NOT HERE

No fallback. If Laya does not answer, the item is recorded as unanswered and
counted as such. A harness that quietly substitutes a keyword rule for a dead
router measures the keyword rule.

No tuning on the test set pretending to be a result. The gate sweep in
`gate_sweep()` is fit on the SAME labels it reports accuracy for. That is an
upper bound, it is labelled as one, and `holdout()` runs the honest split.

RUNNING IT

    python -X utf8 bench/laya_calibration.py              # everything
    python -X utf8 bench/laya_calibration.py --route      # the three-way choice
    python -X utf8 bench/laya_calibration.py --binary     # the two-binary cascade
    python -X utf8 bench/laya_calibration.py --distil     # grounding + verdict
    python -X utf8 bench/laya_calibration.py --replay     # no GPU, cached runs
    python -X utf8 bench/laya_calibration.py --route --orders 6 --tag ord6

Raw per-item probabilities are written to bench/data/laya_routing_runs.json
and bench/data/laya_distil_runs.json so the tables, the gate sweep and
bench/test_laya_calibration.py can all be recomputed without touching the GPU.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "mcp"))

# Nothing here may write to the shared corpus. The index is read through a
# read-only sqlite URI below, never through code_search._db(), which does a
# CREATE TABLE IF NOT EXISTS on open and would therefore touch the file.
os.environ.setdefault("YAMADORI_CORPUS_DB",
                      os.path.join(HERE, "data", "_calibration_corpus.sqlite3"))

LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")
LAYA_TIMEOUT = float(os.environ.get("YAMADORI_LAYA_TIMEOUT", "30"))
LAYA_STATE_CHARS = 6000

# The gate mcp/shomen.py ships with. Reported against, not assumed.
SHIPPED_GATE = 0.3

# THE MEASURED RECOMMENDATION. See docs/LAYA.md finding 4. It applies ONLY to
# the binary "does answering this need our code?" question on a PROSE state --
# it is not a gate for the three-way question, which has no usable gate at any
# value, and not for the distil questions, which are constants.
RECOMMENDED_GATE = 0.30

ROUTE_LABELS = os.path.join(HERE, "laya_routing_labels.jsonl")
DISTIL_LABELS = os.path.join(HERE, "laya_distil_labels.jsonl")
ROUTE_RUNS = os.path.join(HERE, "data", "laya_routing_runs.json")
DISTIL_RUNS = os.path.join(HERE, "data", "laya_distil_runs.json")

ROUTE_OPTIONS = {
    "investigate": ("requires reading source code that is not already in the "
                    "conversation"),
    "answer_directly": ("answerable from general knowledge, or from what is "
                        "already in the conversation"),
    "clarify": "too underspecified to act on without asking first",
}
ROUTE_INSTRUCTIONS = "Decide how this question should be handled before answering it."

VERDICT_OPTIONS = {
    "answers_it": "the finding answers the question that was asked",
    "partial": "it answers part of the question and leaves the rest open",
    "off_target": "it does not address what was asked",
}
VERDICT_INSTRUCTIONS = "Judge this finding against the question it was meant to answer."

GROUNDING_OPTIONS = {
    "from_the_files": "it describes what is in the files that were read",
    "general_knowledge": "it is general knowledge, not specific to those files",
}
GROUNDING_INSTRUCTIONS = "Where does the content of this finding come from?"

# ------------------------------------------------- THE BINARY CASCADE -------
#
# The three-way argmax confusion says the same thing in every variant: the
# `clarify` option is almost never chosen for a `clarify` item, and its
# probability mass leaks into `answer_directly`. A three-way choice where one
# option is dead is a two-way choice with a handicap.
#
# So: ask two BINARY questions instead. Laya's measured-good regime is a fixed
# state with typed options, and a binary question is the cleanest form of it.
# Each sub-question also gets its own accuracy, which the three-way number was
# hiding -- "55% on three classes" could be a good specificity detector and a
# broken code detector, or the reverse, and the aggregate cannot tell you.

SPEC_OPTIONS = {
    "actionable": ("the request names a specific thing to look at or a "
                   "specific question to answer"),
    "underspecified": ("the request does not say what it refers to, so it "
                       "cannot be acted on without asking first"),
}
SPEC_INSTRUCTIONS = "Is this request specific enough to act on?"

CODE_OPTIONS = {
    "needs_the_codebase": ("answering it requires reading this project's own "
                           "source files"),
    "general_knowledge": ("answering it requires only knowledge of the "
                          "language or library, not this project's code"),
}
CODE_INSTRUCTIONS = "What does answering this request depend on?"


# ====================================================== THE PRIMITIVE =======
#
# A byte-for-byte reimplementation of mcp/shomen.py:_choice_averaged,
# including the degenerate-order guard. It is COPIED rather than imported for
# one reason: shomen.py is being edited by another workstream and importing
# it would make these numbers describe whatever it says at the moment the
# script ran. test_laya_calibration.py asserts the two stay in agreement.

def _laya(state: str, questions: dict, timeout: float = LAYA_TIMEOUT) -> dict:
    req = urllib.request.Request(
        LAYA_URL + "/decide",
        data=json.dumps({"state": state[:LAYA_STATE_CHARS],
                         "questions": questions}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _orders(keys: list[str], n: int) -> list[list[str]]:
    """Option orderings to average over.

    n=2 is forward and reversed, which is what shomen.py uses. n>=6 on a
    three-option question is every permutation, which is the honest version of
    "position bias removed" -- with only two orders, an option that sits in the
    middle both times never gets to be first.
    """
    if n <= 1:
        return [keys]
    if n == 2:
        return [keys, list(reversed(keys))]
    perms = list(itertools.permutations(keys))
    return [list(p) for p in perms[:n]]


def choice_averaged(state: str, instructions: str, criteria: dict,
                    n: int = 2) -> dict | None:
    keys = list(criteria)
    if len(keys) < 2:
        return None
    acc = {k: 0.0 for k in keys}
    used = 0
    per_order = []
    for order in _orders(keys, n):
        try:
            d = _laya(state, {"pick": {
                "type": "choice", "instructions": instructions,
                "criteria": {k: criteria[k] for k in order}}})
            probs = (d.get("answers", {}).get("pick", {}).get("probabilities")
                     or {})
        except Exception as e:                                   # noqa: BLE001
            per_order.append({"order": order, "error": f"{type(e).__name__}: {e}"})
            continue
        # An order whose mass is ~0 is a degenerate answer. Averaging it in
        # halves every margin and makes undecided look decided.
        if sum(float(v) for v in probs.values()) < 1e-6:
            per_order.append({"order": order, "degenerate": True})
            continue
        for k in keys:
            acc[k] += float(probs.get(k, 0.0))
        used += 1
        per_order.append({"order": order,
                          "probabilities": {k: float(probs.get(k, 0.0))
                                            for k in keys}})
    if not used:
        return None
    avg = {k: v / used for k, v in acc.items()}
    ranked = sorted(avg.items(), key=lambda kv: -kv[1])
    return {"choice": ranked[0][0], "probabilities": avg,
            "margin": ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0),
            "confidence": ranked[0][1], "orders_used": used,
            "per_order": per_order}


# ================================================== THE TWO STATE FORMS =====

def prose_state(item: dict) -> str:
    """Exactly what mcp/shomen.py:route_in builds today."""
    return (f"A user asked: {item['question']}\n\n"
            f"Conversation context: {item.get('context', '')[:2000] or '(none)'}")


# `named_symbols`, `index_facts` and `signals` MOVED to mcp/selection.py on
# 2026-09-22, with `rule_baseline` below, so the benchmark, the trainer and the
# proxy's selection engine run ONE copy (docs/SELECTION-BUILD.md step 3). They
# are re-exported under the old names so every caller of this module, and
# bench/data/*_runs.json regenerated from it, is unchanged.
from selection import (  # noqa: E402,F401
    _CLARIFY_HINT,
    _PROSE_CAMEL_OK,
    _SYMBOL_PATTERNS,
    _index_facts,
    index_facts,
    named_symbols,
    rule_baseline,
    signals,
)


def _signal_block(s: dict) -> str:
    idx = (f"available, {s['index_chunks']} chunks over {s['index_files']} files"
           if s["index_exists"] else "none")
    return "\n".join([
        f"domains: {', '.join(s['domains']) or 'none detected'}",
        f"libraries named: {', '.join(s['libraries']) or 'none'}",
        f"concrete symbols named: {', '.join(s['symbols']) or 'none'}",
        f"code block pasted: {'yes' if s['code_block'] else 'no'}",
        f"request length: {s['words']} words",
        f"conversation context: {s['context_words']} words",
        f"codebase index: {idx}",
    ])


def structured_state(item: dict) -> str:
    """Signals ONLY. The strict form of the hypothesis: no prose at all."""
    return _signal_block(signals(item))


def hybrid_state(item: dict) -> str:
    """Signals FIRST, then the request. The form you would actually ship."""
    return (_signal_block(signals(item)) + "\n\n"
            f"request: {item['question']}\n"
            f"context: {(item.get('context', '') or '(none)')[:1500]}")


VARIANTS = {"prose": prose_state,
            "structured": structured_state,
            "hybrid": hybrid_state}


# ==================================================== DETERMINISTIC FLOOR ===
#
# Without this, "Laya gets 62%" is unreadable: the majority class alone gets
# 39%, and six lines of regex may get more than the model does. A router is
# only worth a GPU if it beats the thing you would have written anyway.

# `rule_baseline` and `_CLARIFY_HINT` live in mcp/selection.py now (imported
# above). The floor is the same six lines; it simply has one home.


# ============================================================== RUNNING =====

def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_routing(items: list[dict], variants: list[str], n_orders: int,
                verbose: bool = True) -> dict:
    runs: dict = {"n_orders": n_orders, "items": []}
    t0 = time.time()
    for i, item in enumerate(items):
        rec = {"question": item["question"], "label": item["label"],
               "hard": bool(item.get("hard")),
               "rule": rule_baseline(item), "signals": signals(item),
               "variants": {}}
        for v in variants:
            state = VARIANTS[v](item)
            r = choice_averaged(state, ROUTE_INSTRUCTIONS, ROUTE_OPTIONS,
                                n_orders)
            rec["variants"][v] = (None if r is None else
                                  {"choice": r["choice"],
                                   "margin": r["margin"],
                                   "probabilities": r["probabilities"],
                                   "orders_used": r["orders_used"]})
        runs["items"].append(rec)
        if verbose:
            print(f"  [{i+1:>2}/{len(items)}] {item['question'][:58]:<58} "
                  + "  ".join(
                      f"{v}={(rec['variants'][v] or {}).get('choice','ERR')[:4]}"
                      f"/{(rec['variants'][v] or {}).get('margin',0):.3f}"
                      for v in variants), flush=True)
    runs["elapsed_s"] = round(time.time() - t0, 1)
    return runs


def run_binary(items: list[dict], variants: list[str], n_orders: int,
               verbose: bool = True) -> dict:
    """The cascade: 'specific enough?' then 'does it need our code?'.

    Both sub-questions are asked unconditionally so each can be scored on its
    OWN labels. The cascade only combines them afterwards; running them
    conditionally would leave the second question with a sample that depends
    on the first question's errors, which is unscoreable.
    """
    runs: dict = {"n_orders": n_orders, "items": []}
    t0 = time.time()
    for i, item in enumerate(items):
        rec = {"question": item["question"], "label": item["label"],
               "hard": bool(item.get("hard")),
               "rule": rule_baseline(item), "variants": {}}
        for v in variants:
            state = VARIANTS[v](item)
            spec = choice_averaged(state, SPEC_INSTRUCTIONS, SPEC_OPTIONS,
                                   min(n_orders, 2))
            code = choice_averaged(state, CODE_INSTRUCTIONS, CODE_OPTIONS,
                                   min(n_orders, 2))
            if spec is None or code is None:
                rec["variants"][v] = None
                continue
            if spec["choice"] == "underspecified":
                choice, margin = "clarify", spec["margin"]
            else:
                choice = ("investigate"
                          if code["choice"] == "needs_the_codebase"
                          else "answer_directly")
                margin = min(spec["margin"], code["margin"])
            rec["variants"][v] = {
                "choice": choice, "margin": margin,
                "probabilities": {**spec["probabilities"],
                                  **code["probabilities"]},
                "spec": {"choice": spec["choice"], "margin": spec["margin"],
                         "probabilities": spec["probabilities"]},
                "code": {"choice": code["choice"], "margin": code["margin"],
                         "probabilities": code["probabilities"]},
                "orders_used": spec["orders_used"] + code["orders_used"]}
        runs["items"].append(rec)
        if verbose:
            print(f"  [{i+1:>2}/{len(items)}] {item['label']:<15} "
                  + "  ".join(
                      f"{v}={(rec['variants'][v] or {}).get('choice', 'ERR')[:4]}"
                      f"/{(rec['variants'][v] or {}).get('margin', 0):.3f}"
                      for v in variants)
                  + f"   {item['question'][:44]}", flush=True)
    runs["elapsed_s"] = round(time.time() - t0, 1)
    return runs


def table_binary(runs: dict, variants: list[str]) -> None:
    """Each sub-question on its OWN labels. This is the diagnostic."""
    print(f"\nBINARY CASCADE SUB-QUESTIONS, n={len(runs['items'])} "
          f"(argmax, no gate)")
    print("-" * 100)
    print(f"{'variant':<12} {'Q1 specific?':>26} {'Q2 needs codebase?':>30} "
          f"{'cascade 3-class':>18}")
    print(f"{'':<12} {'(all 49; label=clarify -> underspec)':>26} "
          f"{'(37 non-clarify only)':>30}")
    print("-" * 100)
    for v in variants:
        s_n = s_ok = c_n = c_ok = casc_ok = casc_n = 0
        for rec in runs["items"]:
            r = rec["variants"].get(v)
            if not r:
                continue
            want_under = rec["label"] == "clarify"
            s_n += 1
            s_ok += (r["spec"]["choice"] == "underspecified") == want_under
            if not want_under:
                c_n += 1
                c_ok += ((r["code"]["choice"] == "needs_the_codebase")
                         == (rec["label"] == "investigate"))
            casc_n += 1
            casc_ok += r["choice"] == rec["label"]
        print(f"{v:<12} {s_ok:>3}/{s_n:<3} = {s_ok/s_n*100:>5.1f}%{'':>9} "
              f"{c_ok:>3}/{c_n:<3} = {c_ok/c_n*100:>5.1f}%{'':>11} "
              f"{casc_ok:>3}/{casc_n:<3} = {casc_ok/casc_n*100:>5.1f}%")
    # THE SLICE THAT MATTERS. The easy items are separable by surface cues --
    # "our", a path, a CamelCase name -- so a regex scores near ceiling on
    # them and so would anything. The `hard` items were written to break
    # exactly those cues: an investigate question with no repo marker, an
    # answer_directly question wearing a file path. A model that beats the
    # regex has to do it here or it is not doing anything the regex was not.
    hard = [r for r in runs["items"] if r.get("hard")]
    easy = [r for r in runs["items"] if not r.get("hard")]
    if hard:
        print("\nQ2 ('needs the codebase?') SPLIT BY SLICE  "
              f"(easy n={sum(1 for r in easy if r['label'] != 'clarify')}, "
              f"hard n={sum(1 for r in hard if r['label'] != 'clarify')})")
        print(f"{'variant':<14}{'easy':>12}{'hard':>12}")
        for v in variants + ["rule"]:
            cells = []
            for group in (easy, hard):
                n = ok = 0
                for rec in group:
                    if rec["label"] == "clarify":
                        continue
                    want = rec["label"] == "investigate"
                    if v == "rule":
                        got = rec["rule"] == "investigate"
                    else:
                        r = rec["variants"].get(v)
                        if not r:
                            continue
                        got = r["code"]["choice"] == "needs_the_codebase"
                    n += 1
                    ok += got == want
                cells.append(f"{ok}/{n} = {ok/n*100:.0f}%" if n else "n/a")
            tag = v + (" (regex)" if v == "rule" else "")
            print(f"{tag:<14}{cells[0]:>12}{cells[1]:>12}")

    print("-" * 100)
    print("Q1 chance = 75.5% (always 'actionable'); "
          "Q2 chance = 51.4% (always 'general_knowledge')")
    print("A sub-question at or below its chance line is not a classifier, "
          "whatever its margins say.")

    # DEGENERACY. A constant answer with a large margin looks exactly like a
    # confident classifier in every aggregate. It is the failure this whole
    # file exists to catch, so it is printed as its own line, not inferred.
    print("\nDEGENERACY CHECK (what fraction of items got the SAME answer)")
    for v in variants:
        for sub, opts in (("spec", SPEC_OPTIONS), ("code", CODE_OPTIONS)):
            picks = [rec["variants"][v][sub]["choice"] for rec in runs["items"]
                     if rec["variants"].get(v)]
            mgs = [rec["variants"][v][sub]["margin"] for rec in runs["items"]
                   if rec["variants"].get(v)]
            top = max(set(picks), key=picks.count)
            frac = picks.count(top) / len(picks)
            flag = "  <-- CONSTANT" if frac >= 0.95 else ""
            print(f"  {v:<11} {sub:<5} {top:<20} on {picks.count(top):>2}/"
                  f"{len(picks)} = {frac*100:5.1f}%  "
                  f"mean margin {statistics.fmean(mgs):.3f}{flag}")


def table_subquestion_gate(runs: dict, variant: str, sub: str,
                           positive: str, want) -> None:
    """Gate sweep for ONE binary sub-question, scored on its own labels.

    The three-way gate sweep cannot see this: a cascade margin is the MINIMUM
    of two sub-margins, so a gate on it throws away good answers to the
    working question because the broken question was unsure.
    """
    rows = []
    for rec in runs["items"]:
        r = rec["variants"].get(variant)
        if not r:
            continue
        w = want(rec["label"])
        if w is None:
            continue
        rows.append((r[sub]["margin"], (r[sub]["choice"] == positive) == w))
    print(f"\nGATE SWEEP for sub-question '{sub}' alone, variant={variant}, "
          f"n={len(rows)}")
    print(f"{'gate':>6} {'decided':>10} {'abstain%':>10} {'acc|decided':>13} "
          f"{'cov-acc':>10}")
    for g in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        d = [ok for m, ok in rows if m >= g]
        acc = f"{sum(d)/len(d)*100:.1f}%" if d else "n/a"
        print(f"{g:>6.2f} {len(d):>4}/{len(rows):<5} "
              f"{(len(rows)-len(d))/len(rows)*100:>9.1f}% {acc:>13} "
              f"{sum(d)/len(rows)*100:>9.1f}%")


def run_distil(items: list[dict], n_orders: int,
               verbose: bool = True) -> dict:
    runs: dict = {"n_orders": n_orders, "items": []}
    t0 = time.time()
    for i, item in enumerate(items):
        state = (f"Question: {item['question']}\n\n"
                 f"Finding from the investigation: {item['finding'][:4000]}\n\n"
                 f"Files actually read: {', '.join(item['paths'][:20]) or '(none)'}")
        rec = {"question": item["question"],
               "grounded_label": item["grounded"],
               "verdict_label": item["verdict"]}
        g = choice_averaged(state, GROUNDING_INSTRUCTIONS, GROUNDING_OPTIONS,
                            n_orders)
        v = choice_averaged(state, VERDICT_INSTRUCTIONS, VERDICT_OPTIONS,
                            n_orders)
        rec["grounding"] = None if g is None else {
            "choice": g["choice"], "margin": g["margin"],
            "probabilities": g["probabilities"]}
        rec["verdict"] = None if v is None else {
            "choice": v["choice"], "margin": v["margin"],
            "probabilities": v["probabilities"]}
        runs["items"].append(rec)
        if verbose:
            print(f"  [{i+1:>2}/{len(items)}] grounded={item['grounded']!s:<5} "
                  f"-> {(g or {}).get('choice','ERR')[:9]:<9} "
                  f"m={(g or {}).get('margin',0):.3f}   "
                  f"verdict={item['verdict']:<11} -> "
                  f"{(v or {}).get('choice','ERR'):<11} "
                  f"m={(v or {}).get('margin',0):.3f}", flush=True)
    runs["elapsed_s"] = round(time.time() - t0, 1)
    return runs


# ============================================================== SCORING =====

def score_variant(runs: dict, variant: str, gate: float) -> dict:
    n = decided = correct = unanswered = 0
    margins_ok: list[float] = []
    margins_bad: list[float] = []
    confusion: dict = {}
    for rec in runs["items"]:
        n += 1
        r = rec["variants"][variant]
        if r is None:
            unanswered += 1
            continue
        if r["margin"] < gate:
            continue
        decided += 1
        hit = r["choice"] == rec["label"]
        correct += hit
        (margins_ok if hit else margins_bad).append(r["margin"])
        confusion.setdefault(rec["label"], {}).setdefault(r["choice"], 0)
        confusion[rec["label"]][r["choice"]] += 1
    return {"n": n, "decided": decided, "unanswered": unanswered,
            "abstained": n - decided - unanswered,
            "abstention_rate": (n - decided - unanswered) / n if n else 0.0,
            "correct": correct,
            "accuracy": correct / decided if decided else float("nan"),
            "coverage_accuracy": correct / n if n else 0.0,
            "margins_correct": margins_ok, "margins_incorrect": margins_bad,
            "confusion": confusion}


def argmax_accuracy(runs: dict, variant: str) -> dict:
    """Accuracy with NO gate: what the model says when forced to answer.

    This separates the two failure modes. If argmax accuracy is at chance, no
    threshold rescues it and the gate discussion is moot; if argmax accuracy is
    high and the gate is throwing it away, the gate is the bug.
    """
    n = hit = 0
    for rec in runs["items"]:
        r = rec["variants"][variant]
        if r is None:
            continue
        n += 1
        hit += r["choice"] == rec["label"]
    return {"n": n, "correct": hit, "accuracy": hit / n if n else float("nan")}


def _fmt_dist(xs: list[float]) -> str:
    if not xs:
        return "      --          "
    return (f"{min(xs):.3f}/{statistics.median(xs):.3f}/{max(xs):.3f} "
            f"(mean {statistics.fmean(xs):.3f})")


def table_routing(runs: dict, variants: list[str], gate: float) -> None:
    labels = [r["label"] for r in runs["items"]]
    print(f"\nROUTING, n={len(labels)}  "
          f"({labels.count('investigate')} investigate / "
          f"{labels.count('answer_directly')} answer_directly / "
          f"{labels.count('clarify')} clarify)   "
          f"gate={gate}  orders={runs['n_orders']}")
    print("-" * 100)
    print(f"{'variant':<12} {'decided':>9} {'abstain%':>9} {'acc|decided':>12} "
          f"{'acc|all(argmax)':>16} {'correct-margin min/med/max':>30}")
    print("-" * 100)
    for v in variants:
        s = score_variant(runs, v, gate)
        a = argmax_accuracy(runs, v)
        acc = "n/a" if s["decided"] == 0 else f"{s['accuracy']*100:5.1f}%"
        print(f"{v:<12} {s['decided']:>4}/{s['n']:<4} "
              f"{s['abstention_rate']*100:>8.1f}% {acc:>12} "
              f"{a['accuracy']*100:>15.1f}% {_fmt_dist(s['margins_correct']):>30}")
        print(f"{'':<12} {'':>9} {'':>9} {'':>12} {'':>16} "
              f"incorrect: {_fmt_dist(s['margins_incorrect'])}")
    if "rule" not in runs["items"][0]:
        return
    rules = [(r["rule"] == r["label"]) for r in runs["items"]]
    print("-" * 100)
    print(f"{'rule (regex)':<12} {len(rules):>4}/{len(rules):<4} "
          f"{0.0:>8.1f}% {sum(rules)/len(rules)*100:>11.1f}% "
          f"{sum(rules)/len(rules)*100:>15.1f}%     (deterministic floor, no GPU)")
    maj = max(set(labels), key=labels.count)
    print(f"{'majority':<12} {len(labels):>4}/{len(labels):<4} "
          f"{0.0:>8.1f}% {labels.count(maj)/len(labels)*100:>11.1f}% "
          f"{labels.count(maj)/len(labels)*100:>15.1f}%     (always '{maj}')")


def table_confusion(runs: dict, variant: str, gate: float) -> None:
    s = score_variant(runs, variant, gate)
    order = ["investigate", "answer_directly", "clarify"]
    print(f"\nCONFUSION, variant={variant}, gate={gate} "
          f"(rows = label, cols = Laya's choice; abstentions excluded)")
    print(f"{'':<18}" + "".join(f"{c:>17}" for c in order))
    for lab in order:
        row = s["confusion"].get(lab, {})
        print(f"{lab:<18}" + "".join(f"{row.get(c, 0):>17}" for c in order))
    print("(a nearly empty table means the gate discarded almost everything)")


def table_argmax_confusion(runs: dict, variant: str) -> None:
    order = ["investigate", "answer_directly", "clarify"]
    conf: dict = {}
    for rec in runs["items"]:
        r = rec["variants"][variant]
        if r is None:
            continue
        conf.setdefault(rec["label"], {}).setdefault(r["choice"], 0)
        conf[rec["label"]][r["choice"]] += 1
    print(f"\nARGMAX CONFUSION (no gate), variant={variant}")
    print(f"{'':<18}" + "".join(f"{c:>17}" for c in order))
    for lab in order:
        row = conf.get(lab, {})
        print(f"{lab:<18}" + "".join(f"{row.get(c, 0):>17}" for c in order))


def gate_sweep(runs: dict, variants: list[str],
               gates=(0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
                      0.40, 0.50)) -> None:
    print("\nGATE SWEEP  -- fit and reported on the SAME 49 labels, so these "
          "are an\nUPPER BOUND on what a tuned gate would do on unseen "
          "questions. Use --holdout\nfor the honest number.")
    print("-" * 100)
    print(f"{'gate':>6}" + "".join(
        f"{v + ' dec/acc':>26}" for v in variants))
    print("-" * 100)
    for g in gates:
        cells = []
        for v in variants:
            s = score_variant(runs, v, g)
            acc = "  n/a" if s["decided"] == 0 else f"{s['accuracy']*100:5.1f}%"
            cells.append(f"{s['decided']:>3}/{s['n']:<3} {acc}"
                         f"  (cov-acc {s['coverage_accuracy']*100:4.1f}%)")
        print(f"{g:>6.2f}" + "".join(f"{c:>26}" for c in cells))
    print("cov-acc = correct / ALL items, i.e. an abstention counts as a miss.")
    print("It is the number that matters if abstaining has a real cost.")


def holdout(runs: dict, variant: str, folds: int = 5) -> None:
    """Pick the gate on 4/5 of the labels, report on the held-out 1/5.

    The sweep above is fit on the data it scores. This is the same procedure
    done honestly. If the two disagree, the sweep was fitting noise.
    """
    items = runs["items"]
    grid = [i / 100 for i in range(0, 51, 2)]
    tot = hit = 0
    picked = []
    for f in range(folds):
        test = [x for i, x in enumerate(items) if i % folds == f]
        train = [x for i, x in enumerate(items) if i % folds != f]
        best, best_g = -1.0, 0.0
        for g in grid:
            s = score_variant({"items": train}, variant, g)
            if s["coverage_accuracy"] > best:
                best, best_g = s["coverage_accuracy"], g
        picked.append(best_g)
        s = score_variant({"items": test}, variant, best_g)
        tot += s["n"]
        hit += s["correct"]
    print(f"\nHOLD-OUT ({folds}-fold, gate chosen on train, scored on test), "
          f"variant={variant}")
    print(f"  gates picked per fold : {picked}")
    print(f"  coverage-accuracy     : {hit}/{tot} = {hit/tot*100:.1f}%  "
          f"(abstention counts as a miss)")


def paired(runs: dict, a: str, b: str, gate: float) -> None:
    """McNemar on the SAME items. This is the paired comparison, as asked.

    Exact binomial two-sided p over the discordant pairs, computed here rather
    than imported, because scipy is not in this venv and the arithmetic is
    four lines.
    """
    a_only = b_only = both = neither = 0
    a_dec = b_dec = 0
    for rec in runs["items"]:
        ra, rb = rec["variants"].get(a), rec["variants"].get(b)
        ok_a = bool(ra and ra["margin"] >= gate and ra["choice"] == rec["label"])
        ok_b = bool(rb and rb["margin"] >= gate and rb["choice"] == rec["label"])
        a_dec += bool(ra and ra["margin"] >= gate)
        b_dec += bool(rb and rb["margin"] >= gate)
        if ok_a and ok_b:
            both += 1
        elif ok_a:
            a_only += 1
        elif ok_b:
            b_only += 1
        else:
            neither += 1
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        from math import comb
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    print(f"\nPAIRED ({a} vs {b}) at gate={gate}, correct-and-decided per item")
    print(f"  both correct        : {both}")
    print(f"  {a} only            : {a_only}")
    print(f"  {b} only            : {b_only}")
    print(f"  neither             : {neither}")
    print(f"  decided count       : {a}={a_dec}  {b}={b_dec}")
    print(f"  McNemar exact p     : {p:.4f}"
          + ("   -- NOT significant" if p > 0.05 else "   -- significant"))


# ----------------------------------------------------------------- distil ---

def table_distil(runs: dict, gate: float) -> None:
    items = runs["items"]
    print(f"\nDISTIL, n={len(items)}  gate={gate}  orders={runs['n_orders']}")

    # --- grounding: a binary question, so also report it as a separator ---
    dec = cor = 0
    tp = [r["grounding"]["probabilities"]["from_the_files"]
          for r in items if r["grounding"] and r["grounded_label"]]
    fp = [r["grounding"]["probabilities"]["from_the_files"]
          for r in items if r["grounding"] and not r["grounded_label"]]
    mg_ok, mg_bad = [], []
    for r in items:
        g = r["grounding"]
        if not g or g["margin"] < gate:
            continue
        dec += 1
        hit = (g["choice"] == "from_the_files") == r["grounded_label"]
        cor += hit
        (mg_ok if hit else mg_bad).append(g["margin"])
    print("-" * 100)
    print("GROUNDING  ('does this describe the cited files, or is it general "
          "knowledge?')")
    print(f"  decided            : {dec}/{len(items)}  "
          f"(abstained {len(items)-dec}, {(len(items)-dec)/len(items)*100:.1f}%)")
    print("  accuracy | decided : "
          + (f"{cor}/{dec} = {cor/dec*100:.1f}%" if dec else "n/a"))
    print(f"  margins  correct   : {_fmt_dist(mg_ok)}")
    print(f"  margins  incorrect : {_fmt_dist(mg_bad)}")
    if tp and fp:
        print(f"  p(from_the_files)  grounded findings   : "
              f"mean {statistics.fmean(tp):.3f}  min {min(tp):.3f}")
        print(f"  p(from_the_files)  fabricated findings : "
              f"mean {statistics.fmean(fp):.3f}  max {max(fp):.3f}")
        print(f"  SEPARATION (mean gap): {statistics.fmean(tp)-statistics.fmean(fp):+.3f}"
              f"   AUC: {_auc(tp, fp):.3f}")
        sep, thr = _best_threshold(tp, fp)
        print(f"  best single threshold on p(from_the_files): {thr:.3f} -> "
              f"{sep*100:.1f}% accuracy over all {len(tp)+len(fp)} findings "
              f"(NO abstention)")

    # --- verdict: three-way ---
    dec = cor = 0
    mv_ok, mv_bad = [], []
    for r in items:
        v = r["verdict"]
        if not v or v["margin"] < gate:
            continue
        dec += 1
        hit = v["choice"] == r["verdict_label"]
        cor += hit
        (mv_ok if hit else mv_bad).append(v["margin"])
    argmax = [r for r in items if r["verdict"]]
    amx = sum(r["verdict"]["choice"] == r["verdict_label"] for r in argmax)
    print("-" * 100)
    print("VERDICT  ('does this finding answer the question?')")
    print(f"  decided            : {dec}/{len(items)}  "
          f"(abstained {len(items)-dec}, {(len(items)-dec)/len(items)*100:.1f}%)")
    print("  accuracy | decided : "
          + (f"{cor}/{dec} = {cor/dec*100:.1f}%" if dec else "n/a"))
    print(f"  accuracy | argmax  : {amx}/{len(argmax)} = "
          f"{amx/len(argmax)*100:.1f}%  (no gate)")
    print(f"  margins  correct   : {_fmt_dist(mv_ok)}")
    print(f"  margins  incorrect : {_fmt_dist(mv_bad)}")

    # The collapsed binary: 'is this finding usable at all?'. A three-way
    # judgement may fail only because 'partial' and 'answers_it' are not
    # separable, which is a different and much less serious defect.
    ok = sum(1 for r in argmax
             if (r["verdict"]["choice"] != "off_target")
             == (r["verdict_label"] != "off_target"))
    print(f"  collapsed to usable/off_target (argmax): {ok}/{len(argmax)} = "
          f"{ok/len(argmax)*100:.1f}%")


def _auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random grounded finding outranks a random fabricated one."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    best, bt = 0.0, 0.0
    for t in sorted(set(pos + neg)):
        acc = (sum(p >= t for p in pos) + sum(n < t for n in neg)) / (len(pos) + len(neg))
        if acc > best:
            best, bt = acc, t
    return best, bt


# ================================================================= MAIN =====

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", action="store_true")
    ap.add_argument("--distil", action="store_true")
    ap.add_argument("--binary", action="store_true",
                    help="the two-binary-question cascade instead of the "
                         "three-way choice")
    ap.add_argument("--replay", action="store_true",
                    help="score the cached runs, do not call Laya")
    ap.add_argument("--orders", type=int, default=2,
                    help="option orderings to average (2 = what ships, 6 = all)")
    ap.add_argument("--gate", type=float, default=SHIPPED_GATE)
    ap.add_argument("--variants", default="prose,structured,hybrid")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for the cached run files, so a control run "
                         "(e.g. --orders 6) does not overwrite the main one")
    args = ap.parse_args()
    global ROUTE_RUNS, DISTIL_RUNS
    if args.tag:
        ROUTE_RUNS = ROUTE_RUNS.replace(".json", f"_{args.tag}.json")
        DISTIL_RUNS = DISTIL_RUNS.replace(".json", f"_{args.tag}.json")
    if not args.route and not args.distil and not args.binary:
        args.route = args.distil = args.binary = True
    variants = [v for v in args.variants.split(",") if v in VARIANTS]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

    if not args.replay:
        try:
            with urllib.request.urlopen(LAYA_URL + "/health", timeout=5) as r:
                h = json.load(r)
            if not h.get("loaded"):
                print(f"laya at {LAYA_URL} is up but not loaded: {h}")
                return 2
        except Exception as e:                                   # noqa: BLE001
            print(f"laya at {LAYA_URL} unreachable: {type(e).__name__}: {e}\n"
                  f"nothing is measured and nothing is guessed. "
                  f"use --replay to score cached runs.")
            return 2

    if args.route:
        items = load_jsonl(ROUTE_LABELS)
        if args.replay:
            runs = json.load(open(ROUTE_RUNS, encoding="utf-8"))
        else:
            print(f"\n=== ROUTING: {len(items)} labelled questions x "
                  f"{len(variants)} state forms x {args.orders} orders "
                  f"= {len(items)*len(variants)*args.orders} laya calls ===")
            runs = run_routing(items, variants, args.orders, not args.quiet)
            json.dump(runs, open(ROUTE_RUNS, "w", encoding="utf-8"), indent=1)
            print(f"  raw -> {ROUTE_RUNS}  ({runs['elapsed_s']}s)")
        have = [v for v in variants if v in runs["items"][0]["variants"]]
        table_routing(runs, have, args.gate)
        for v in have:
            table_confusion(runs, v, args.gate)
            table_argmax_confusion(runs, v)
        gate_sweep(runs, have)
        for v in have:
            holdout(runs, v)
        if "prose" in have:
            for v in have:
                if v != "prose":
                    paired(runs, "prose", v, args.gate)

    if args.binary:
        items = load_jsonl(ROUTE_LABELS)
        path = ROUTE_RUNS.replace(".json", "_binary.json")
        if args.replay:
            runs = json.load(open(path, encoding="utf-8"))
        else:
            print(f"\n=== BINARY CASCADE: {len(items)} questions x "
                  f"{len(variants)} state forms x 2 sub-questions x 2 orders "
                  f"= {len(items)*len(variants)*4} laya calls ===")
            runs = run_binary(items, variants, args.orders, not args.quiet)
            json.dump(runs, open(path, "w", encoding="utf-8"), indent=1)
            print(f"  raw -> {path}  ({runs['elapsed_s']}s)")
        have = [v for v in variants if runs["items"][0]["variants"].get(v)]
        table_binary(runs, have)
        for v in have:
            table_subquestion_gate(
                runs, v, "code", "needs_the_codebase",
                lambda lab: None if lab == "clarify" else lab == "investigate")
        table_routing(runs, have, args.gate)

    if args.distil:
        items = load_jsonl(DISTIL_LABELS)
        if args.replay:
            runs = json.load(open(DISTIL_RUNS, encoding="utf-8"))
        else:
            print(f"\n=== DISTIL: {len(items)} labelled findings x 2 questions "
                  f"x {args.orders} orders = "
                  f"{len(items)*2*args.orders} laya calls ===")
            runs = run_distil(items, args.orders, not args.quiet)
            json.dump(runs, open(DISTIL_RUNS, "w", encoding="utf-8"), indent=1)
            print(f"  raw -> {DISTIL_RUNS}  ({runs['elapsed_s']}s)")
        table_distil(runs, args.gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
