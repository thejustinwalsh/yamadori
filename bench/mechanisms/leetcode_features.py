#!/usr/bin/env python
"""Build the decision tree from 2,641 human-labelled problems, on the local GPU.

WHY THIS DATASET CHANGES THE PLAN

Tree induction needs labels: for each problem, which technique is right. The
plan until now was to get them from an oracle run -- generate under every arm,
see which won -- which costs one generation per problem per arm and yields
maybe 18 labelled problems for an evening of GPU time.

LeetCode already labelled 2,641 problems with 63 topic tags, and
`newfacade/LeetCodeDataset` publishes them under Apache-2.0. Prefix Sum 164,
Two Pointers 195, Binary Search 246, Monotonic Stack, Sliding Window, Heap.
That is 147x the labels, free, and labelled by the people who set the problems
rather than by whoever is building the tree -- which removes the author's taste
from the one place it does the most damage.

WHICH TAG IS THE LABEL

Problems carry several tags: two-sum is ["Array", "Hash Table"]. "Array" is
true of 1,619 of 2,641 problems and therefore says almost nothing; "Hash Table"
is the technique that actually solves it. So the label is the RAREST tag on the
problem, which is a crude proxy for the most specific one, and it is recorded
alongside the full tag set so a better rule can be tried without re-running.

Tags under a floor are dropped rather than kept as singleton classes: a class
with three examples cannot support a measured split, and including it inflates
entropy while teaching nothing.

WHAT RUNS WHERE

Nothing here spends a frontier model. Candidate questions come from bonsai on a
free slot of the model already loaded; the answers come from Laya at ~49 ms
each. 2,641 problems by ten questions is about 22 minutes of local GPU, and the
cost of asking a different ten questions tomorrow is another 22 minutes.

ONE AXIS ONLY: PROBLEM AND SOLUTION SPACE

What this tree selects is an ALGORITHMIC APPROACH -- prefix sums, two pointers,
a monotonic stack. It must never select a coding STYLE.

They are independent axes and conflating them corrupts both. A problem that
wants a sliding window wants a sliding window whether it is written in idiomatic
Rust or in TypeScript, and "name things with verbs first" is true regardless of
whether the answer is O(n) or O(n log n). A leaf that carried both would inject
style opinions every time it fired on an algorithmic question, and the measured
effect of the algorithmic hint could no longer be separated from the style hint
riding along with it.

So the leaves of THIS tree map only into the complexity corpus. Style, naming,
modern-idiom and language-practice recipes are a different corpus, selected by
a different question, and `axis_guard()` below refuses to mix them.

WHAT THIS STILL DOES NOT ESTABLISH

That the technique a problem is tagged with is the hint that would most improve
THIS model's output. Those are different claims, and only the oracle run can
connect them. This builds a tree that predicts the human label; whether acting
on that label helps is the experiment that follows.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BENCH)

LAYA = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")
UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("BENCH_MODEL", "bonsai")
DATA = os.path.join(BENCH, "data", "LeetCodeDataset-train.jsonl")

# Tags that describe the DATA rather than the technique. They are nearly
# universal and carry no decision content: 1,619 of 2,641 problems are tagged
# Array. Keeping them as labels would make the majority class "Array" and the
# tree would learn to predict a tautology.
UNINFORMATIVE = {"Array", "String", "Math", "Matrix", "Simulation",
                 "Database", "Interactive", "Concurrency", "Shell"}

PROPOSE = """You are designing yes/no questions that decide which algorithmic
technique a programming problem needs.

Here are the techniques: {techniques}

Write {k} yes/no questions about a problem statement. Rules:
- Answerable from the statement alone, before any code is written.
- Each asks about ONE property. Never use "and" or "or".
- Concrete and checkable, not a matter of judgement.
- 15 words or fewer.
- Ask about the SHAPE of the problem, not about the technique by name.

Good: Does the problem ask for a value over every contiguous subarray?
Good: Are the input values already sorted?
Bad: Should you use a hash table and a two pointer scan?

Return one question per line. No numbering, no explanation."""



# The axis this tree is allowed to select on. LeetCode tags describe problem
# and solution space; nothing here may reach into style, naming or language
# idiom, which are a separate corpus selected by a separate question.
AXIS = "algorithm"
FOREIGN_AXES = {"naming_api_style", "canonical_style", "modern_practice"}


def axis_guard(leaf_recipes: list[dict]) -> None:
    """Refuse a leaf that carries a recipe from another axis."""
    bad = [r.get("recipe_id", "?") for r in leaf_recipes
           if (r.get("category") in FOREIGN_AXES
               or r.get("area") in FOREIGN_AXES)]
    if bad:
        raise ValueError(
            f"leaf mixes axes: {bad}. This tree selects an algorithmic "
            f"approach only. Style and idiom are a separate corpus, selected "
            f"by a separate question, so that the effect of each can be "
            f"measured without the other riding along.")


def load(limit: int = 0) -> list[dict]:
    if not os.path.exists(DATA):
        raise SystemExit(
            f"missing {DATA}\nFetch it first:\n"
            f"  huggingface_hub.hf_hub_download('newfacade/LeetCodeDataset',\n"
            f"      'LeetCodeDataset-train.jsonl', repo_type='dataset',\n"
            f"      local_dir='bench/data')")
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


def label_of(tags: list[str], freq: dict[str, int]) -> str | None:
    """The rarest informative tag: a proxy for the most specific technique."""
    useful = [t for t in (tags or []) if t not in UNINFORMATIVE]
    if not useful:
        return None
    return min(useful, key=lambda t: freq.get(t, 0))


def statement(row: dict, cap: int = 1500) -> str:
    """Problem text, trimmed at the END.

    Laya silently drops the tail of a long state, so the part that must survive
    goes first. Constraints usually sit at the bottom of a LeetCode statement,
    which is unfortunate -- the input bound is decision-relevant -- so the
    constraints block is lifted to the front when it can be found.
    """
    text = (row.get("problem_description") or "").strip()
    low = text.lower()
    i = low.rfind("constraints:")
    if i > 0:
        head, cons = text[:i].strip(), text[i:].strip()
        text = f"{cons}\n\n{head}"
    return text[:cap]


def propose(techniques: list[str], k: int, timeout: int = 600) -> list[str]:
    body = {"model": MODEL,
            "messages": [{"role": "system",
                          "content": PROPOSE.format(k=k,
                                                    techniques=", ".join(techniques))},
                         {"role": "user", "content": "Write the questions."}],
            "max_tokens": 3000, "temperature": 0.7}
    req = urllib.request.Request(f"{UPSTREAM}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "")
    out = []
    for line in text.splitlines():
        q = line.strip().lstrip("-*0123456789. ").strip()
        low = q.lower()
        if (q.endswith("?") and 4 <= len(q.split()) <= 18
                and " and " not in low and " or " not in low and q not in out):
            out.append(q)
    return out


def ask_laya(state: str, question: str, timeout: int = 60) -> tuple[str, float]:
    """One closed yes/no choice, permutation-averaged over option order.

    Averaged because Laya is not permutation invariant: option order changes
    the answer. Two options means forward plus reversed is a complete de-bias,
    not a sample of one.
    """
    probs: dict[str, float] = {}
    for order in (["yes", "no"], ["no", "yes"]):
        body = {"state": state,
                "questions": {"v": {"type": "choice",
                                    "instructions": question,
                                    "criteria": {o: "" for o in order}}}}
        req = urllib.request.Request(f"{LAYA}/decide",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            a = json.load(r)["answers"]["v"]
        for k, v in (a.get("probabilities") or {}).items():
            probs[k] = probs.get(k, 0.0) + v / 2.0
    if not probs:
        return "unknown", 0.0
    best = max(probs, key=probs.get)
    rest = sorted(probs.values(), reverse=True)
    margin = rest[0] - (rest[1] if len(rest) > 1 else 0.0)
    return best, margin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400,
                    help="problems to featurise; 0 means all 2641")
    ap.add_argument("--questions", type=int, default=12)
    ap.add_argument("--min-class", type=int, default=25,
                    help="drop techniques with fewer examples than this")
    ap.add_argument("--gate", type=float, default=0.15,
                    help="below this margin the answer is recorded unknown")
    ap.add_argument("--out", default=os.path.join(HERE, "leetcode_features.json"))
    ap.add_argument("--questions-file",
                    default=os.path.join(HERE, "leetcode_questions.json"))
    ap.add_argument("--propose-only", action="store_true")
    args = ap.parse_args()

    rows = load()
    freq = collections.Counter(t for r in rows for t in (r.get("tags") or []))
    labelled = []
    for r in rows:
        lab = label_of(r.get("tags") or [], freq)
        if lab:
            labelled.append((r, lab))
    counts = collections.Counter(l for _r, l in labelled)
    keep = {t for t, n in counts.items() if n >= args.min_class}
    labelled = [(r, l) for r, l in labelled if l in keep]
    print(f"  {len(rows)} problems -> {len(labelled)} with an informative label")
    print(f"  {len(keep)} techniques kept (>= {args.min_class} examples each)")
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        if t in keep:
            print(f"    {n:>5}  {t}")

    if os.path.exists(args.questions_file):
        questions = json.load(open(args.questions_file, encoding="utf-8"))
        print(f"\n  {len(questions)} questions loaded from disk")
    else:
        print(f"\n  asking bonsai for {args.questions} candidate questions "
              f"(free slot, no frontier tokens)...")
        questions = propose(sorted(keep), args.questions)
        json.dump(questions, open(args.questions_file, "w", encoding="utf-8"),
                  indent=2)
        print(f"  {len(questions)} well-formed questions:")
        for q in questions:
            print(f"    {q}")
    if args.propose_only:
        return
    if not questions:
        raise SystemExit("no well-formed questions; re-run --propose-only")

    work = labelled[:args.limit] if args.limit else labelled
    total = len(work) * len(questions)
    print(f"\n  featurising {len(work)} problems x {len(questions)} questions "
          f"= {total} Laya calls")

    features: dict[str, dict[str, str]] = {q: {} for q in questions}
    labels: dict[str, str] = {}
    margins: list[float] = []
    unknown = 0
    t0 = time.time()
    for i, (row, lab) in enumerate(work):
        qid = str(row.get("question_id") or row.get("task_id") or i)
        labels[qid] = lab
        st = statement(row)
        for q in questions:
            try:
                ans, margin = ask_laya(st, q)
            except Exception:                                    # noqa: BLE001
                ans, margin = "unknown", 0.0
            if margin < args.gate:
                ans = "unknown"
                unknown += 1
            margins.append(margin)
            features[q][qid] = ans
        if (i + 1) % 25 == 0:
            done = (i + 1) * len(questions)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {i + 1}/{len(work)} problems  {rate:.0f} calls/s  "
                  f"eta {(total - done) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)

    json.dump({"features": features, "labels": labels,
               "gate": args.gate, "questions": questions},
              open(args.out, "w", encoding="utf-8"), indent=2)
    secs = time.time() - t0
    mid = sorted(margins)[len(margins) // 2] if margins else 0.0
    print(f"\n  {total} calls in {secs / 60:.1f} min "
          f"({total / max(secs, 1e-9):.0f}/s), median margin {mid:.3f}")
    print(f"  {unknown} answers ({unknown * 100 // max(total, 1)}%) fell below "
          f"the {args.gate} gate and were recorded unknown")
    print(f"  written {args.out}")
    print("\n  Next: induce the tree over these features with tree_build.py, "
          "which will\n  report information gain per question and prune any "
          "node whose branches agree.")


if __name__ == "__main__":
    main()
