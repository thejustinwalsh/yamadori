#!/usr/bin/env python
"""Generate contrastive training pairs from real code, mechanically.

WHY THIS UNBLOCKS EVERYTHING

The decision model is documented as "a fast base to specialise, not a
zero-shot decision engine", and its own card reports zero-shot typed decisions
at 0.362 against a 0.461 majority baseline. So every zero-shot measurement of
it -- including all of the ones in this repo -- tested a configuration its
authors say does not work. It cannot be honestly evaluated until it is tuned,
and it cannot be tuned without labelled data in the target shape.

Waiting for organic corpus is slow. Generating labels is not, because a
compiler-checkable defect can be introduced on purpose: take real code, change
one thing, and the label is known by construction.

WHY MINIMAL PAIRS AND NOT JUST "GOOD CODE / BAD CODE"

This is the failure mode every groundedness classifier in the literature falls
into: keying on surface features rather than the proposition. A classifier
reading only the hypothesis and never the premise scores ~67% on SNLI against
a 33% baseline; a claim-only model that never reads the evidence gets 61.7% on
fact verification. Trained on unmatched good/bad examples, a model learns
"long and tidy means correct" and nothing about correctness.

The fix that works is contrastive minimal pairs, where surface form is held
nearly constant and only the proposition flips. VitaminC built ~400k such
pairs from Wikipedia revisions that change a fact and reported +10% on
adversarial fact verification. Each mutation below changes one token or one
line, so the two members of a pair are almost identical text with opposite
labels -- there is no surface shortcut to learn.
"""
from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))

# Each mutation states the axis it flips and how, so a generated example can be
# traced back to why it is labelled the way it is.
MUTATIONS = [
    # (name, pattern, replacement, axis, correct_label, mutated_label)
    ("boundary", re.compile(r"(\bfor\s*\([^;]*;\s*\w+\s*)<(\s*\w+(?:\.\w+)*\s*;)"),
     r"\1<=\2", "correct", "yes", "no"),
    ("off_by_one", re.compile(r"(\[\s*\w+\s*)\+\s*1(\s*\])"),
     r"\1\2", "correct", "yes", "no"),
    ("inverted_guard", re.compile(r"\bif\s*\(\s*!\s*(\w+)\s*\)"),
     r"if ( \1 )", "correct", "yes", "no"),
    ("dropped_await", re.compile(r"\bawait\s+(\w+\()"),
     r"\1", "correct", "yes", "no"),
    ("flipped_operator", re.compile(r"(\breturn\s+\w+\s*)\+(\s*\w+\s*;)"),
     r"\1-\2", "correct", "yes", "no"),
    ("weakened_equality", re.compile(r"([\w\)\]]\s*)===(\s*[\w\'\"])"),
     r"\1==\2", "correct", "yes", "no"),
    ("dropped_null_check", re.compile(r"(\w+)\s*\?\.\s*(\w+)"),
     r"\1.\2", "correct", "yes", "no"),
]


def mutate(text: str, rnd: random.Random) -> tuple[str, str, str, str] | None:
    """Return (mutated_text, mutation_name, axis, mutated_label) or None."""
    opts = [m for m in MUTATIONS if m[1].search(text)]
    if not opts:
        return None
    name, pat, repl, axis, _ok, bad = rnd.choice(opts)
    # Exactly one occurrence changes, so the pair differs by one token.
    new = pat.sub(repl, text, count=1)
    if new == text:
        return None
    return new, name, axis, bad


def chunks_from(db: str, limit: int, rnd: random.Random) -> list[tuple[str, str]]:
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT path, text FROM chunks WHERE length(text) BETWEEN 120 AND 1400"
    ).fetchall()
    con.close()
    rnd.shuffle(rows)
    return rows[:limit]


# `===` to `==` is the easiest mutation to find and the weakest label: loose
# equality is a lint preference, not an unambiguous defect, and at 279 of 500
# it dominated the first generated set. Training on a majority of arguable
# labels teaches the arguable thing. Unambiguous mutations are uncapped.
CAPS = {"weakened_equality": 0.15}


def build(dbs: list[str], want: int = 400, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    out: list[dict] = []
    used: dict[str, int] = {}
    per = max(want // max(len(dbs), 1), 1) * 6
    for db in dbs:
        if not os.path.exists(db):
            continue
        for path, text in chunks_from(db, per, rnd):
            if len(out) >= want * 2:
                break
            m = mutate(text, rnd)
            if not m:
                continue
            bad_text, name, axis, bad_label = m
            cap = CAPS.get(name)
            if cap is not None and used.get(name, 0) >= int(want * cap):
                continue
            used[name] = used.get(name, 0) + 1
            # The pair is emitted together and must stay together in any
            # split, or the model sees one member in train and the other in
            # test and the evaluation is meaningless.
            pair_id = f"{os.path.basename(db)}:{len(out)}"
            out.append({"pair": pair_id, "member": "original", "axis": axis,
                        "label": "yes", "mutation": name, "path": path,
                        "state": text})
            out.append({"pair": pair_id, "member": "mutated", "axis": axis,
                        "label": bad_label, "mutation": name, "path": path,
                        "state": bad_text})
    return out


def main() -> None:
    import repos
    roots = [e["root"] for e in repos.known()]
    dbs = [repos.db_path(r) for r in roots]
    pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "index", "packages")
    if os.path.isdir(pkg_dir):
        dbs += [os.path.join(pkg_dir, f) for f in os.listdir(pkg_dir)
                if f.endswith(".sqlite3")]
    want = int(os.environ.get("PAIRS", "400"))
    rows = build(dbs, want=want)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "index", "pairs.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_mut: dict[str, int] = {}
    for r in rows:
        if r["member"] == "mutated":
            by_mut[r["mutation"]] = by_mut.get(r["mutation"], 0) + 1
    print(f"  sources : {len([d for d in dbs if os.path.exists(d)])} indexes")
    print(f"  pairs   : {len(rows) // 2}   ({len(rows)} examples)")
    for k, v in sorted(by_mut.items(), key=lambda x: -x[1]):
        print(f"    {k:<20} {v}")
    print(f"  written : {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
