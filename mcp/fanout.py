#!/usr/bin/env python
"""Answer a question several ways at once and keep what they agree on.

WHY THIS IS AFFORDABLE

Measured on this box, concurrent completions against one model instance:

    N=1   5.5s   36.6 tok/s
    N=2   8.8s   45.3 tok/s
    N=4  14.9s   53.8 tok/s    <- best throughput
    N=8  32.5s   49.2 tok/s    <- batch saturates

Four answers cost 2.7x the wall clock of one, not 4x, because batching
recovers memory bandwidth a single stream leaves idle. A metered model cannot
do this at any price; a local one pays only in latency.

WHY AGREEMENT AND NOT A SCORER

The obvious design is to train something to pick the best answer. Published
results say do not bother yet: on MATH500 at 16 generations, plain
self-consistency scored 86.00 against 85.00 for best-of-N with an external
reward model, and 82.80 for beam search. Majority voting beat the trained
scorer. So consensus ships first and a learned selector is an upgrade, not a
prerequisite.

WHY DIVERSE VARIANTS AND NOT JUST TEMPERATURE

Sampling the same prompt four times explores one region of the space. Asking
four genuinely different ways -- with retrieval and without, terse and
thorough -- produces disagreement that means something. When four different
approaches land on the same file, that is evidence. When four samples of one
prompt agree, that is mostly temperature being low.

MEASURED -- WHEN NOT TO USE THIS

On eight file-location questions against koota, four-way consensus scored 7/8
against 7/8 for a single answer, with ZERO discordant pairs, at 3.2x the wall
clock. That is a null result on a task class where it could not have won:
"where is X defined" has one right answer and retrieval either finds it or
does not, so there is no quality variance for diversity to exploit and both
arms sit at the ceiling.

The cost is also higher than raw sampling suggested -- 3.2x rather than the
2.7x measured for generation alone -- because each variant runs its own tool
loop, so tool calls multiply too.

So this is gated on the task having room for a better answer:

  lookup, definition, reference, "where is"   -> N=1, always
  design, approach, refactor, "how should I"  -> N=4

Fanning out on a lookup is pure latency, which is now measured rather than
assumed.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import urllib.request
from collections import Counter

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:1234")

# Each variant is a different way of approaching the same question, not a
# different random seed. `nudge` is appended to the system message.
VARIANTS = [
    {"name": "direct", "temperature": 0.3, "nudge": ""},
    {"name": "evidence", "temperature": 0.3,
     "nudge": "\n\nGround every claim in a file you have actually read. "
              "Name the path and line."},
    {"name": "skeptical", "temperature": 0.7,
     "nudge": "\n\nConsider the obvious answer, then check whether a second "
              "place in the codebase is a better fit before committing."},
    {"name": "terse", "temperature": 0.2,
     "nudge": "\n\nAnswer in as few words as the question allows. No preamble."},
]

# What a consensus is taken OVER. Free prose cannot be voted on, but the
# artefacts that matter in these answers can.
_PATH = re.compile(r"[\w./\\-]+\.(?:ts|tsx|js|jsx|mjs|cjs|rs|c|h|cpp|hpp|py|go|zig|wgsl|glsl|lua|json|toml)")
_SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{2,})`")


def claims(text: str) -> tuple[set, set]:
    """The checkable assertions in an answer: which files, which symbols."""
    paths = {p.replace("\\", "/").lstrip("./") for p in _PATH.findall(text or "")}
    syms = set(_SYMBOL.findall(text or ""))
    return paths, syms


# Seed concepts come from concept_seed, which draws a random direction in
# embedding space and takes the nearest real word. A curated list was tried
# first and is the author's taste in shuffled order -- the same narrow, lumpy
# slice of concept space every time. Random directions in high dimensions are
# near-orthogonal, so consecutive draws genuinely differ.
import concept_seed  # noqa: E402


def _one(payload: dict, variant: dict, timeout: int) -> dict:
    body = json.loads(json.dumps(payload))     # deep copy
    msgs = body.get("messages") or []
    if variant["nudge"]:
        for m in msgs:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] += variant["nudge"]
                break
        else:
            msgs.insert(0, {"role": "system", "content": variant["nudge"].strip()})
    body["temperature"] = variant["temperature"]
    body.pop("stream", None)
    if variant.get("seed"):
        # USER message, not system. The originating project tested both and
        # found the model would ignore a system-message seed and fall back to
        # its default approach; in the user turn it cannot.
        for m in reversed(body["messages"]):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] += concept_seed.phrase(variant["seed"])
                break

    # Each variant must run the WHOLE tool loop. Doing a single completion
    # returns an empty answer and a pending tool call, which then scores zero
    # on every path-based grader -- measured, 0/8, and it was this bug rather
    # than anything about consensus.
    d = _tool_loop(body, timeout)
    msg = d["choices"][0]["message"]
    return {"variant": variant["name"],
            "content": msg.get("content") or "",
            "raw": d}


def _tool_loop(body: dict, timeout: int) -> dict:
    """Same loop the proxy runs, so a variant sees the same tool results."""
    import proxy as _p
    injected = {t["function"]["name"] for t in (body.get("tools") or [])
                if t.get("function", {}).get("name") in _p.OUR_NAMES}
    convo = body["messages"]
    for _ in range(_p.MAX_TOOL_HOPS):
        req = urllib.request.Request(f"{UPSTREAM}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        calls = [c for c in (msg.get("tool_calls") or [])
                 if c.get("function", {}).get("name") in injected]
        if not calls:
            return d
        convo.append({"role": "assistant", "content": msg.get("content") or "",
                      "tool_calls": msg.get("tool_calls") or []})
        for c in calls:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            convo.append({"role": "tool", "tool_call_id": c["id"],
                          "content": _p.run_our_tool(c["function"]["name"],
                                                     args, None)[:6000]})
        body["messages"] = convo
    return d


def run(payload: dict, n: int = 4, timeout: int = 900,
        variants: list | None = None) -> dict:
    """Run n variants concurrently and report what they agreed on."""
    pool = variants if variants is not None else VARIANTS
    use = pool[:max(1, min(n, len(pool)))]
    with cf.ThreadPoolExecutor(max_workers=len(use)) as ex:
        futs = [ex.submit(_one, payload, v, timeout) for v in use]
        results = []
        for f in futs:
            try:
                results.append(f.result())
            except Exception as e:                               # noqa: BLE001
                results.append({"variant": "?", "content": "",
                                "error": f"{type(e).__name__}: {e}"})

    good = [r for r in results if r.get("content")]
    if not good:
        return {"results": results, "agreement": None, "winner": None}

    path_votes: Counter = Counter()
    sym_votes: Counter = Counter()
    for r in good:
        p, s = claims(r["content"])
        # One vote per answer per claim, so a variant repeating a path five
        # times does not outvote three variants naming it once.
        for x in p:
            path_votes[x] += 1
        for x in s:
            sym_votes[x] += 1

    top_path, top_n = (path_votes.most_common(1) or [(None, 0)])[0]
    agreement = top_n / len(good) if good else 0.0

    # The winning ANSWER is the one that best represents the consensus, not the
    # first or the longest: the answer that names the most agreed-upon claims.
    def score(r):
        p, s = claims(r["content"])
        return (sum(path_votes[x] for x in p) + sum(sym_votes[x] for x in s),
                -len(r["content"]))
    winner = max(good, key=score)

    return {
        "n": len(good),
        "agreement": round(agreement, 2),
        "consensus_path": top_path,
        "path_votes": dict(path_votes.most_common(5)),
        "winner": winner,
        "results": results,
    }


def _choice_averaged(state: str, instructions: str, criteria: dict,
                     timeout: int = 30) -> dict | None:
    """A choice, averaged over option orderings.

    The model is NOT permutation invariant. Reported upstream at a 27-33%
    semantic flip rate on two options, and reproduced here: asking which file
    declares a type was stable, but "which loop visits every element once"
    flipped its answer purely on option order at a margin of 0.441, well above
    any gate worth setting.

    Averaging the distributions over forward and reversed order cancels the
    positional component. It also makes the MARGIN honest, which matters more:
    measured, cases that flip average down to margins of 0.012-0.184 while a
    genuinely decided case stays at 0.800. So a reordering-sensitive answer
    stops being a confident answer and starts being an undecided one, which is
    the correct outcome and what the margin gate is then able to catch.

    Costs two calls instead of one, about 100ms at this model's latency.
    """
    import code_search as cs
    keys = list(criteria)
    if len(keys) < 2:
        return None
    orders = [tuple(keys), tuple(reversed(keys))]
    acc = {k: 0.0 for k in keys}
    for order in orders:
        try:
            d = cs._post_json(LAYA_URL_ + "/decide", {
                "state": state[:2000],
                "questions": {"pick": {
                    "type": "choice", "instructions": instructions,
                    "criteria": {k: criteria[k] for k in order}}}},
                timeout=timeout)
            probs = (d["answers"]["pick"].get("probabilities") or {})
        except Exception:                                        # noqa: BLE001
            return None
        for k in keys:
            acc[k] += float(probs.get(k, 0.0)) / len(orders)
    ranked = sorted(acc.items(), key=lambda x: -x[1])
    return {"choice": ranked[0][0],
            "margin": round(ranked[0][1] - ranked[1][1], 3),
            "probabilities": {k: round(v, 3) for k, v in acc.items()}}


def break_tie(question: str, results: list[dict], timeout: int = 30) -> dict | None:
    """Ask the decision model to pick, as ONE choice whose options are the
    candidates.

    This is the primitive it is actually good at. Scoring candidates
    independently does not work -- its scores are not comparable across
    different inputs -- and handing it a LIST of states does not batch, it
    concatenates them and returns a single verdict for the blob.

    Used only when consensus fails. When the variants agree, the agreement is
    free and better evidenced than any classifier.
    """
    good = [r for r in results if r.get("content")]
    if len(good) < 2:
        return None
    criteria = {}
    for i, r in enumerate(good):
        paths, _ = claims(r["content"])
        label = chr(ord("a") + i)
        criteria[label] = (", ".join(sorted(paths)[:3]) or
                           r["content"][:120].replace("\n", " "))
    v = _choice_averaged(question, "Which answer best answers the question?",
                         criteria, timeout)
    # Below the floor the options did not separate once position bias is
    # removed. Reporting a pick it did not really make is inventing a decision.
    if not v or v["margin"] < 0.15:
        return None
    idx = ord(v["choice"]) - ord("a")
    if not 0 <= idx < len(good):
        return None
    return {"winner": good[idx], "margin": v["margin"],
            "ranking": v["probabilities"]}


LAYA_URL_ = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")


def dissent_note(v: dict) -> str:
    """What to tell the user when the variants did NOT agree.

    Disagreement is information and hiding it is the failure mode that makes a
    confident wrong answer indistinguishable from a confident right one.
    """
    if not v or v.get("agreement") is None:
        return ""
    if v["agreement"] >= 0.75:
        return ""
    alts = [p for p in (v.get("path_votes") or {}) if p != v.get("consensus_path")]
    note = (f"\n\n> Answered {v['n']} ways; only {int(v['agreement'] * 100)}% "
            f"agreed on the same file. Treat this as unsettled.")
    if alts:
        note += " Other candidates: " + ", ".join(alts[:3]) + "."
    return note
