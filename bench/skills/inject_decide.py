#!/usr/bin/env python
"""READ THE INJECTOR'S QUESTIONS ON ONE MODEL (GPU): every labelled case's
stage-2 and stage-3 questions through the real decider (decide_turn.Turn ->
decider_bonsai.read, the typed readout), in each question VARIANT, the
decisions logged for bench/skills/inject_tune.py.

    python bench/skills/inject_decide.py --cases DIR/cases.jsonl \\
        --model bonsai [--variants a,b,c] [--limit N] [--dry-run]

ONE GPU CONSUMER AT A TIME (AGENTS.md): run only inside a window the
coordinator granted, with `--model` loaded. Before anything is sent it reads
llama-swap's GET /running (which never loads a model) and REFUSES unless
`--model` is loaded and nothing else is using the card's main slot.

THE VARIANTS (docs/JJAVA.md 4: "if a question set won't separate, rewrite
the question and state and measure again" -- the operator: "no easy
bails"). All are read in one window so a rewrite needs no second one:

  a  skill_inject.ITEM_Q / ITEM_LEVELS as served (a Score, 4 levels; the
     belief is P(top level))
  b  a Score of 3 levels: the "already done" level folded out (is the fact
     off the material / in its area but not the thing at hand / the thing
     at hand)
  c  a Noul per item: the top level as a statement to judge
  d  variant a's question over a REWRITTEN STATE (goal_state): the
     session's goal named first ("GOAL ... NOW: ..."), the turn after it
     -- the state rewrite of docs/JJAVA.md 4.2, read on a second Turn per
     case (one more state placement)
  stage 3 (all variants): skill_inject.INJECT_Q over the shortlist the
     variant's own untuned gate passes; with none passed, over the variant's
     two items of highest belief (labelled `forced`, so stage 3 has
     decisions on every case to tune on).

WHAT IS WRITTEN (never conversation text): the decisions log (decide_turn
DECISIONS, pointed at bench/skills/inject/results/decisions_<model>.jsonl
before import) and one row per case and variant in
bench/skills/inject/results/run_<model>.jsonl: the item keys, each item's
belief, expected level, confidence, argmax and decision id, the stage-3
noul and what it listed, and the milliseconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "inject", "results")
sys.path.insert(0, os.path.join(ROOT, "mcp"))

RUN_VERSION = "inject-decide/1"
VARIANTS = ("a", "b", "c", "d")
LEVELS_B = [
    "It is about a library, API or task that the material does not "
    "involve.",
    "It is about a library the material involves, but not about the code, "
    "command or error the assistant is working on now.",
    "It is about the code, command or error the assistant is working on "
    "now.",
]
NOUL_C = ("This fact is about the code, command or error the assistant is "
          "working on now, and the material does not yet show it done the "
          "way the fact says.\n\nFACT: {fact}")


def running(upstream: str) -> list[str]:
    with urllib.request.urlopen(f"{upstream}/running", timeout=10) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = d.get("running") if isinstance(d, dict) else d
    return [str((x or {}).get("model")) for x in rows or []
            if isinstance(x, dict)]


def questions(variant: str, items: list[dict]) -> list[dict]:
    import decider_bonsai as D
    import skill_inject as I
    out = []
    for it in items:
        name = f"{I.QSET_ITEM}{'' if variant == 'a' else '_' + variant}:" \
               f"{it['key']}"
        if variant in ("a", "d"):
            out.append(D.q_score(name, I.ITEM_Q.format(fact=it["fact"]),
                                 I.ITEM_LEVELS))
        elif variant == "b":
            out.append(D.q_score(name, I.ITEM_Q.format(fact=it["fact"]),
                                 LEVELS_B))
        else:
            out.append(D.q_noul(name, NOUL_C.format(fact=it["fact"])))
    return out


def belief(variant: str, a: dict) -> float:
    if variant == "c":
        return float(a["noul"])
    p = a.get("probabilities") or {}
    top = str(len(p) - 1)
    return float(p.get(top, 0.0))


def passes(variant: str, a: dict) -> bool:
    if (a.get("diagnostics") or {}).get("tie"):
        return False
    if variant == "c":
        return float(a["noul"]) >= 0.5
    p = a.get("probabilities") or {}
    return bool(p) and max(p, key=p.get) == str(len(p) - 1)


def run_case(c: dict, variants, model: str) -> list[dict]:
    rows = []
    groups = [("base", [v for v in variants if v != "d"]),
              ("goal", [v for v in variants if v == "d"])]
    for which, vs in groups:
        if vs:
            rows += _run_state(c, vs, model, goal_state(c) if which ==
                               "goal" else c["state"])
    return rows


def goal_state(c: dict) -> str:
    """Variant d's STATE: the session's goal named first, then the turn
    (docs/JJAVA.md 4.2 "Name the parts"); when the turn was cut to the
    lane's STATE_TOKENS, the goal takes its room from the turn's head so
    the state stays that size."""
    goal = "GOAL (the session's opening request): " + " ".join(
        str(c.get("task") or "").split())
    st = c["state"]
    if (c.get("state_info") or {}).get("cut"):
        st = st[len(goal) + 2:]
    return goal + "\n\nNOW:\n" + st


def _run_state(c: dict, variants, model: str, state: str) -> list[dict]:
    import decide_turn as T
    import decider_bonsai as D
    import skill_inject as I
    rows = []
    with T.Turn([], state=state, state_info=dict(
            c.get("state_info") or {}, kind=c["kind"]), on=True,
            key=f"inject:{c['case']}", request=c["case"]) as t:
        for v in variants:
            t0 = time.time()
            ans = t.decide(questions(v, c["items"]))
            items = []
            for it, a in zip(c["items"], ans):
                items.append({
                    "key": it["key"], "sha": it["sha"],
                    "belief": round(belief(v, a), 6),
                    "score": a.get("score"), "confidence": a.get("confidence"),
                    "argmax": (a.get("diagnostics") or {}).get("argmax"),
                    "pass": passes(v, a), "tie": bool((a.get("diagnostics")
                                                       or {}).get("tie")),
                    "disagreement": (a.get("diagnostics") or {})
                    .get("disagreement"), "decision_id": a.get("decision_id"),
                    "probs": (a.get("probabilities") if v != "c" else
                              {"true": a["noul"]})})
            ranked = sorted(items, key=lambda r: (-r["belief"], r["key"]))
            short = [r for r in ranked if r["pass"]]
            how = "passed"
            if not short:
                short, how = ranked[:2], "forced"
            by = {it["key"]: it for it in c["items"]}
            listed_items = [{"key": r["key"], "form": "DO", "situation": "",
                             "text": by[r["key"]]["fact"]} for r in short]
            q, listed = I.inject_question(listed_items)
            q = D.q_noul(f"{I.QSET_INJECT}{'' if v == 'a' else '_' + v}",
                         q["text"])
            a3 = t.decide([q])[0]
            rows.append({"v": RUN_VERSION, "model": model, "variant": v,
                         "case": c["case"], "source": c["source"],
                         "group": c["group"], "kind": c["kind"],
                         "state_sha": c["state_sha"], "items": items,
                         "stage3": {"noul": a3["noul"], "how": how,
                                    "listed": [x["key"] for x in listed],
                                    "tie": bool((a3.get("diagnostics") or {})
                                                .get("tie")),
                                    "decision_id": a3.get("decision_id")},
                         "ms": round((time.time() - t0) * 1000, 1)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-labelled", action="store_true",
                    help="only cases bench/skills/inject/labels.jsonl labels")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every question, send nothing")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    os.environ["YAMADORI_DECIDER_DECISIONS"] = os.path.join(
        RESULTS, f"decisions_{a.model}.jsonl")
    os.environ["YAMADORI_DECIDER_LOG"] = os.path.join(
        RESULTS, f"disagreements_{a.model}.jsonl")
    variants = [v for v in a.variants.split(",") if v in VARIANTS]
    with open(a.cases, encoding="utf-8") as f:
        cases = [json.loads(ln) for ln in f if ln.strip()]
    if a.only_labelled:
        sys.path.insert(0, HERE)
        import inject_labels
        lab = {k[0] for k in inject_labels.labels("A")}
        cases = [c for c in cases if c["case"] in lab]
    cases = cases[:a.limit] if a.limit else cases
    if a.dry_run:
        n = sum(len(questions(v, c["items"])) + 1 for c in cases
                for v in variants)
        print(json.dumps({"cases": len(cases), "variants": variants,
                          "questions": n, "reads": 2 * n}))
        return 0
    import model as M
    up = M.UPSTREAM
    have = running(up)
    if a.model not in have:
        print(f"REFUSED: {a.model} is not loaded (GET /running: {have}); "
              "this bench never loads a model -- ask the coordinator for "
              "the window with it loaded", flush=True)
        return 2
    import cancel
    import max_mode
    out = os.path.join(RESULTS, f"run_{a.model}.jsonl")
    t0 = time.time()
    with cancel.bound(cancel.Token()):
        max_mode.set_current(a.model)
        with open(out, "a", encoding="utf-8") as f:
            for k, c in enumerate(cases):
                try:
                    for r in run_case(c, variants, a.model):
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                except Exception as e:                           # noqa: BLE001
                    print(f"  case {c['case']}: {type(e).__name__}: {e}"
                          [:300], flush=True)
                if k % 10 == 9:
                    print(f"  {k + 1}/{len(cases)} cases, "
                          f"{round(time.time() - t0)} s", flush=True)
    print(json.dumps({"cases": len(cases), "variants": variants, "out": out,
                      "seconds": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
