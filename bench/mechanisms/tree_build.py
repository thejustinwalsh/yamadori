#!/usr/bin/env python
"""Induce the decision tree from data. System 2 proposes, information gain disposes.

THE ARCHITECTURE THIS SERVES

Enrichment selection is a two-phase thing. Offline and slow, the big model
decomposes a problem space into simple questions with direct answers and
encodes the criteria ahead of time. Online and fast, a small classifier walks
that tree and the leaf names the hint to inject.

That split matters for three reasons, and none of them is elegance:

  it plays to the classifier's ONE strength   a closed-set choice over a fixed
                                              state, which is the only shape it
                                              measurably does well
  it never compares across passages           each node is a self-contained
                                              question, so the failure that
                                              killed score-ranking cannot occur
  selection leaves the context window         200 candidate recipes are never
                                              put in front of the model; only
                                              the chosen leaf's hint is

WHY THE TREE CANNOT BE HAND-WRITTEN

Because then its author's taste decides which questions matter, and the whole
apparatus becomes an elaborate way of encoding a guess. A question earns a node
only if it REDUCES UNCERTAINTY about which hint actually wins, and that is a
measurable quantity: information gain against oracle labels.

So this is ordinary decision-tree induction with two substitutions. The
candidate features are proposed by a language model instead of being columns in
a table, and the feature extractor is a classifier instead of a lookup. The
selection criterion is the same one Quinlan used in 1986, for the same reason:
it is the thing that makes the tree about the data rather than about the
author.

WHAT COMES OUT, AND WHAT IT IS WORTH

A tree, plus the numbers that say whether to trust it: per-node information
gain, per-node discrimination, the label distribution at each leaf, and the
expected end-to-end accuracy given a per-node error rate. A depth-5 tree at 90%
per node is 59% end to end, and that arithmetic has to be visible BEFORE anyone
ships a deep tree, not discovered afterwards.

A node whose branches lead to the same answer is not neutral. It is pure added
error, and it gets pruned.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.join(BENCH, "..", "mcp"))

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("BENCH_MODEL", "bonsai")

# Asking for questions in the abstract produces essays. Every constraint here
# exists to force the output into the shape a classifier can actually answer:
# self-contained, closed-answer, and about the PROBLEM rather than about a
# solution that does not exist yet at selection time.
PROPOSE = """You are designing a decision tree that picks which performance
hint to give a programmer before they solve a problem.

Write {k} yes/no questions about a programming problem statement. Rules:
- Answerable from the problem statement alone, before any code exists.
- Each asks about ONE property. No question may contain "and" or "or".
- Concrete and checkable, not a matter of judgement.
- 15 words or fewer.

Good: Does the input size exceed 100000?
Good: Does the problem require removing items from the front of a sequence?
Bad: Is the problem hard and does it need optimisation?

Return one question per line. No numbering, no explanation."""


def entropy(labels: list[str]) -> float:
    """Bits of uncertainty in a label distribution."""
    if not labels:
        return 0.0
    n = len(labels)
    return -sum((c / n) * math.log2(c / n)
                for c in collections.Counter(labels).values() if c)


def information_gain(labels: list[str], answers: list[str]) -> float:
    """How much answering this question reduces uncertainty about the label.

    This is the whole selection criterion. A question that splits the examples
    into groups whose best hint is the same within each group has high gain; a
    question that splits them at random has none, however sensible it sounds.
    """
    if not labels:
        return 0.0
    before = entropy(labels)
    after = 0.0
    by: dict[str, list[str]] = {}
    for lab, ans in zip(labels, answers):
        by.setdefault(ans, []).append(lab)
    for group in by.values():
        after += (len(group) / len(labels)) * entropy(group)
    return before - after


def discrimination(labels: list[str], answers: list[str]) -> dict:
    """Does this question's answer actually change which hint wins?

    A node whose branches agree on the majority label is not a harmless extra
    step. Every node multiplies in another chance to be wrong, so a node that
    cannot change the outcome strictly lowers end-to-end accuracy and must be
    pruned.
    """
    by: dict[str, list[str]] = {}
    for lab, ans in zip(labels, answers):
        by.setdefault(ans, []).append(lab)
    majorities = {a: collections.Counter(g).most_common(1)[0][0]
                  for a, g in by.items() if g}
    return {"branches": {a: len(g) for a, g in by.items()},
            "majority_per_branch": majorities,
            "discriminates": len(set(majorities.values())) > 1}


def expected_accuracy(depth: int, per_node: float) -> float:
    """Independent per-node errors compound. Make the cost visible up front."""
    return per_node ** max(depth, 0)


def propose_questions(k: int, examples: list[str], timeout: int = 600) -> list[str]:
    """Ask the big model for candidate questions. It is System 2 here, once."""
    sample = "\n\n".join(f"Example problem:\n{e[:600]}" for e in examples[:4])
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": PROPOSE.format(k=k)},
                         {"role": "user", "content": sample}],
            "max_tokens": 3000, "temperature": 0.7}
    req = urllib.request.Request(f"{UPSTREAM}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text and msg.get("reasoning_content"):
        text = msg["reasoning_content"]
    out = []
    for line in text.splitlines():
        q = line.strip().lstrip("-*0123456789. ").strip()
        # Enforce the shape that was asked for rather than trusting it. A
        # compound question cannot be answered by one closed-set choice.
        if (q.endswith("?") and 4 <= len(q.split()) <= 18
                and " and " not in q.lower() and " or " not in q.lower()):
            out.append(q)
    return out


def build(labels: list[str], features: dict[str, list[str]], depth: int,
          min_gain: float = 0.02, min_samples: int = 4,
          idx: list[int] | None = None, used: set | None = None) -> dict:
    """Greedy induction. Each node is the question with the highest gain here.

    Greedy rather than optimal because optimal tree construction is NP-hard and
    because the questions are noisy anyway -- a classifier answers them, not an
    oracle. Buying a globally optimal split on labels this uncertain would be
    precision the inputs do not support.
    """
    idx = list(range(len(labels))) if idx is None else idx
    used = set() if used is None else used
    here = [labels[i] for i in idx]
    counts = collections.Counter(here)
    leaf = {"leaf": counts.most_common(1)[0][0] if counts else "none",
            "n": len(idx), "distribution": dict(counts)}
    if depth <= 0 or len(idx) < min_samples or len(counts) <= 1:
        return leaf

    best, best_gain, best_disc = None, min_gain, None
    for q, answers in features.items():
        if q in used:
            continue
        a_here = [answers[i] for i in idx]
        if len(set(a_here)) < 2:
            continue                       # constant here: splits nothing
        g = information_gain(here, a_here)
        if g > best_gain:
            disc = discrimination(here, a_here)
            # Gain alone is not enough. A question can raise purity while every
            # branch still ends at the same answer, which adds a chance to be
            # wrong and changes nothing.
            if disc["discriminates"]:
                best, best_gain, best_disc = q, g, disc

    if best is None:
        return leaf

    children = {}
    for ans in sorted(set(features[best][i] for i in idx)):
        sub = [i for i in idx if features[best][i] == ans]
        children[ans] = build(labels, features, depth - 1, min_gain,
                              min_samples, sub, used | {best})
    return {"question": best, "gain": round(best_gain, 4),
            "n": len(idx), "discrimination": best_disc, "children": children}


def walk_stats(node: dict, d: int = 0) -> tuple[int, int]:
    """(max depth, node count) -- what the end-to-end error will be built on."""
    if "leaf" in node:
        return d, 1
    depths, total = [], 1
    for c in node["children"].values():
        dd, n = walk_stats(c, d + 1)
        depths.append(dd)
        total += n
    return (max(depths) if depths else d), total


def render(node: dict, indent: str = "  ") -> str:
    if "leaf" in node:
        return f"{indent}-> {node['leaf']}  (n={node['n']}, {node['distribution']})"
    out = [f"{indent}? {node['question']}   [gain {node['gain']}, n={node['n']}]"]
    for ans, child in node["children"].items():
        out.append(f"{indent}  {ans}:")
        out.append(render(child, indent + "    "))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(BENCH, "recipe_results.jsonl"),
                    help="oracle run: per problem, which arm won")
    ap.add_argument("--features", default=os.path.join(HERE, "features.json"),
                    help="{question: {question_id: answer}} from the classifier")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--per-node-accuracy", type=float, default=0.9)
    ap.add_argument("--out", default=os.path.join(HERE, "tree.json"))
    ap.add_argument("--propose", type=int, default=0,
                    help="ask the model for N candidate questions and exit")
    ap.add_argument("--data", default=os.path.join(BENCH, "data", "test6.jsonl"))
    args = ap.parse_args()

    if args.propose:
        rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
        qs = propose_questions(args.propose,
                               [r["question_content"] for r in rows[:4]])
        print(f"  {len(qs)} well-formed questions proposed:")
        for q in qs:
            print(f"    {q}")
        path = os.path.join(HERE, "candidate_questions.json")
        json.dump(qs, open(path, "w", encoding="utf-8"), indent=2)
        print(f"\n  written {path}")
        print("  These are CANDIDATES. None earns a node until its information "
              "gain is measured against oracle labels.")
        return

    if not os.path.exists(args.labels):
        raise SystemExit(
            f"no oracle labels at {args.labels}.\n"
            f"Run bench/recipe_oracle.py first -- the tree is induced FROM "
            f"which hint actually won, and without that there is nothing to "
            f"induce it from.")

    # Label per problem = the arm that passed with the lowest runtime.
    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    by: dict = {}
    for r in rows:
        if r.get("error"):
            continue
        by.setdefault(r["question_id"], {})[r["arm"]] = r
    best: dict[str, str] = {}
    for qid, arms in by.items():
        passing = {a: v for a, v in arms.items() if v.get("passed")}
        if not passing:
            continue
        best[qid] = min(passing, key=lambda a: passing[a].get("max_s", 1e9))
    print(f"  {len(best)} problems have a winning arm")
    print(f"  label distribution: {dict(collections.Counter(best.values()))}")
    print(f"  baseline entropy: {entropy(list(best.values())):.3f} bits")

    if not os.path.exists(args.features):
        raise SystemExit(
            f"\nno feature matrix at {args.features}.\n"
            f"Produce {{question: {{question_id: answer}}}} by running the "
            f"candidate questions through the classifier over these problems.")

    raw = json.load(open(args.features, encoding="utf-8"))
    qids = sorted(best)
    labels = [best[q] for q in qids]
    features = {q: [raw[q].get(i, "unknown") for i in qids] for q in raw}

    print(f"\n  per-question information gain ({len(features)} candidates):")
    scored = sorted(((information_gain(labels, a), q)
                     for q, a in features.items()), reverse=True)
    for g, q in scored[:12]:
        d = discrimination(labels, features[q])
        flag = "" if d["discriminates"] else "   BRANCHES AGREE -- prune"
        print(f"    {g:6.4f}  {q[:64]}{flag}")

    tree = build(labels, features, args.depth)
    depth, nodes = walk_stats(tree)
    json.dump(tree, open(args.out, "w", encoding="utf-8"), indent=2)

    print(f"\n{render(tree)}")
    print(f"\n  depth {depth}, {nodes} nodes")
    print(f"  expected end-to-end accuracy at {args.per_node_accuracy:.0%} "
          f"per node: {expected_accuracy(depth, args.per_node_accuracy):.1%}")
    print(f"  written {args.out}")
    print("\n  This tree is a HYPOTHESIS induced from the oracle labels. It is "
          "not validated until it is run on problems it was not built from.")


if __name__ == "__main__":
    main()
