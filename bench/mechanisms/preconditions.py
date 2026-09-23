#!/usr/bin/env python
"""Read each recipe's own applicability condition, and build the flowchart from it.

TWO WAYS TO GET THE QUESTIONS, AND WHY THIS IS THE BETTER ONE

`tree_build.py` asks the big model to propose candidate questions and then
keeps whichever ones carry information gain against oracle labels. That works,
but it is guessing forward: most proposed questions turn out to discriminate
nothing, and every recipe is only reachable if some proposed question happens
to select for it.

Working backwards is stronger, because a good recipe ALREADY STATES the
condition under which it applies. From the collected corpus, verbatim:

    "If the loop reads a[i] and b[i] together, one array of structs beats two
     arrays. SoA only wins when the loop touches a subset of fields."

The discriminating question is inside the recipe -- does the hot loop read
several fields together? Nothing has to be invented. Extract it and you get,
for free, both a question that is certainly relevant and the mapping from its
answer to this recipe. Coverage is guaranteed: every recipe is reachable
because its own condition is a node.

That also matters for the recipe corpus itself. A recipe whose condition cannot
be extracted is a recipe with no stated applicability -- advice that claims to
be true always. "Always use struct-of-arrays" is exactly the overgeneralisation
that the Agner Fog source contradicts, so failure to extract a condition is a
signal about the RECIPE, not about the extractor, and is reported as such.

HOW THE TWO DIRECTIONS COMPOSE

Backward extraction builds the tree; information gain prunes it. Extraction
says which questions are meaningful; only measurement says which ones change
the outcome, and a node whose branches lead to the same answer is pure added
error however sensible its question reads.

WHAT THIS IS NOT

It is not validated. A condition the model extracted is a hypothesis about when
a recipe applies, written by the same kind of system that will later answer it.
Nothing here is trustworthy until the questions are answered over real problems
and checked against which hint actually won.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
RECIPES = os.path.join(BENCH, "recipes")

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("BENCH_MODEL", "bonsai")

# The instruction is tight because the output has to be answerable by a
# classifier in one forward pass. Every constraint removes a shape that cannot
# be answered: compound questions have no single answer, questions about the
# solution cannot be asked before the solution exists, and a question needing
# judgement will not be answered consistently.
EXTRACT = """A performance recipe applies only in certain situations.
Read the recipe and write the question that decides whether it applies.

Rules:
- One yes/no question about the PROBLEM or the EXISTING CODE.
- Answerable before the solution is written.
- Exactly one property. Never use "and" or "or".
- Concrete and checkable, not a matter of opinion.
- 15 words or fewer.
- If the recipe applies always, with no condition, answer exactly: ALWAYS

Recipe: Use collections.deque for queue pops, not list.pop(0).
Question: Does the algorithm remove items from the front of a sequence?

Recipe: If the loop reads a[i] and b[i] together, array-of-structs beats two arrays.
Question: Does the hot loop read several fields of the same item together?

Recipe: Write clear variable names.
Question: ALWAYS

Return only the question."""

_WORD = re.compile(r"[a-z]{4,}")
_STOP = {"does", "the", "this", "that", "with", "from", "have", "when",
         "which", "what", "code", "problem", "program", "will", "need"}


def ask(recipe: str, timeout: int = 300) -> str | None:
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": EXTRACT},
                         {"role": "user", "content": f"Recipe: {recipe}\nQuestion:"}],
            "max_tokens": 1200, "temperature": 0.2}
    try:
        req = urllib.request.Request(f"{UPSTREAM}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        out = (msg.get("content") or "").strip()
        if not out and msg.get("reasoning_content"):
            tail = msg["reasoning_content"].strip().splitlines()
            out = tail[-1].strip() if tail else ""
    except Exception:                                            # noqa: BLE001
        return None
    return out.strip().strip('"').split("\n")[0].strip() or None


def well_formed(q: str) -> tuple[bool, str]:
    """Enforce the shape that was asked for rather than trusting it."""
    if not q:
        return False, "empty"
    if q.upper().startswith("ALWAYS"):
        return False, "recipe states no condition"
    if not q.endswith("?"):
        return False, "not a question"
    n = len(q.split())
    if n > 18:
        return False, f"too long ({n} words)"
    if n < 4:
        return False, "too short"
    low = q.lower()
    if " and " in low or " or " in low:
        return False, "compound: two questions in one"
    return True, ""


def signature(q: str) -> frozenset:
    """Content words, for spotting questions that are the same question.

    Many recipes share a condition -- input size, allocation in a loop -- and
    the tree needs each asked ONCE. Every duplicate node is another chance to
    be wrong about the same fact, and the answers would not even be guaranteed
    to agree.
    """
    return frozenset(w for w in _WORD.findall(q.lower()) if w not in _STOP)


def cluster(items: list[dict]) -> list[dict]:
    """Group questions by overlapping content words, keeping the shortest.

    Deliberately crude. A better clustering would embed the questions, and
    that is worth doing once there is evidence the grouping is what limits
    tree quality rather than the questions themselves.
    """
    groups: list[dict] = []
    for it in items:
        sig = signature(it["question"])
        for g in groups:
            overlap = len(sig & g["sig"]) / max(len(sig | g["sig"]), 1)
            if overlap >= 0.6:
                g["recipes"].append(it["recipe_id"])
                g["sig"] = g["sig"] | sig
                if len(it["question"]) < len(g["question"]):
                    g["question"] = it["question"]
                break
        else:
            groups.append({"question": it["question"], "sig": sig,
                           "recipes": [it["recipe_id"]]})
    for g in groups:
        g.pop("sig", None)
    return sorted(groups, key=lambda g: -len(g["recipes"]))


def load_recipes() -> list[dict]:
    out = []
    if not os.path.isdir(RECIPES):
        return out
    for fn in sorted(os.listdir(RECIPES)):
        if not fn.endswith(".jsonl"):
            continue
        for i, line in enumerate(open(os.path.join(RECIPES, fn), encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["recipe_id"] = f"{fn[:-6]}:{i}"
            r["domain"] = r.get("language") or r.get("area") or fn[:-6]
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--domain", default="", help="only this language/area")
    ap.add_argument("--out", default=os.path.join(HERE, "conditions.jsonl"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        rows = [json.loads(l) for l in open(args.out, encoding="utf-8")]
        groups = cluster([r for r in rows if r.get("question")])
        print(f"  {len(rows)} conditions -> {len(groups)} distinct questions")
        for g in groups[:25]:
            print(f"    {len(g['recipes']):>3} recipes  {g['question']}")
        return

    recipes = load_recipes()
    if args.domain:
        recipes = [r for r in recipes if r["domain"] == args.domain]
    if args.limit:
        recipes = recipes[:args.limit]
    if not recipes:
        raise SystemExit(f"no recipes found under {RECIPES}")

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["recipe_id"])
            except Exception:                                    # noqa: BLE001
                continue

    print(f"  {len(recipes)} recipes across "
          f"{len(set(r['domain'] for r in recipes))} domains, "
          f"{len(done)} already extracted\n")
    rejected: dict[str, int] = {}
    kept = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for i, r in enumerate(recipes):
            if r["recipe_id"] in done:
                continue
            q = ask(r["recipe"])
            ok, why = well_formed(q or "")
            rec = {"recipe_id": r["recipe_id"], "domain": r["domain"],
                   "recipe": r["recipe"], "raw": q,
                   "question": q if ok else None, "rejected": None if ok else why}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if ok:
                kept += 1
            else:
                rejected[why] = rejected.get(why, 0) + 1
            print(f"  [{i + 1}/{len(recipes)}] {r['domain']:<16} "
                  f"{'ok ' if ok else 'no '} {(q or '')[:66]}", flush=True)

    total = kept + sum(rejected.values())
    print(f"\n  {kept} of {total} recipes yielded a well-formed condition")
    for why, n in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f"    rejected, {why}: {n}")
    print("\n  A recipe with no condition claims to apply ALWAYS. That is the "
          "overgeneralisation\n  the corpus itself contradicts, so treat those "
          "as a finding about the recipe.")

    rows = [json.loads(l) for l in open(args.out, encoding="utf-8")]
    groups = cluster([r for r in rows if r.get("question")])
    print(f"\n  {len(groups)} distinct questions after clustering "
          f"(shared conditions asked once):")
    for g in groups[:20]:
        print(f"    {len(g['recipes']):>3} recipes  {g['question']}")
    path = os.path.join(HERE, "questions.json")
    json.dump(groups, open(path, "w", encoding="utf-8"), indent=2)
    print(f"\n  written {path}")
    print("  These are HYPOTHESES. None earns a node until its answers are "
          "measured\n  against which hint actually won.")


if __name__ == "__main__":
    main()
