#!/usr/bin/env python
"""Shomen: the deep-thinking context. Study the thing, then answer.

NAMED FROM BONSAI, like `yamadori` and `nebari` and for the same reason. The
shomen is the front of a tree -- the angle it is meant to be seen from. Finding
it is not a step you rush: you turn the tree, study it from every side, and only
then make the first cut. Nobody watching the finished tree sees the turning.

That is exactly this module. It searches, reads and reasons in a context of its
own, and only the conclusion is shown. To a user it is never "a second context"
or "a hemisphere" -- it is the system thinking deeply, and every user-facing
string says so.

THE PROBLEM THIS SOLVES, WHICH IS NOT PARALLELISM

Answering "why does this render black in r185 but not r180" takes six searches,
four file reads and a page of reasoning about what came back. Done in the
user's own context, all of it stays there: re-prefilled every turn, occupying
the window that long-context recall degrades over, and ninety percent of it is
scaffolding nobody wanted to read.

Done in a SECOND context, only the conclusion crosses back.

MEASURED, n=26 paired tasks, 78 generations, 0 errors (docs/CONTEXT-ECONOMY.md):

    main-context tokens, median   8,326 -> 2,410   (0.29x)
    main-context peak window      3,670 -> 1,498   (0.41x)
    correct on facts alone        26/26 -> 26/26   (zero discordant pairs)
    total tokens across both      1.76x to 2.06x
    wall clock                      737s -> 1,423-1,752s

Fewer main-context tokens on 26 of 26 rows, sign test p=2.98e-08, and the
main-context figures replicate to within 3%. The claim holds.

Two honest corrections to the sentence that used to be here. It said "six
thousand tokens of searching becomes a two hundred token finding" and both
magnitudes were wrong: measured, 10,764-15,798 tokens of searching become 531
crossing back -- better compression than advertised (21-30x) from a larger
finding than advertised (2.6x). The finding size is the stable number, so
there is no sampling excuse for it.

AND THE SAVING IS NOT FREE. Total tokens roughly double, because the main
model REWRITES the question at the delegation boundary and invents context --
captured verbatim: "the pymrem Python library", "TSL (TypeScript Style
Linter)", "@deprecated Javadoc". Where the rewrite is faithful, delegating is
CHEAPER (0.31x, 0.75x, 0.79x), so 2x is an upper bound on a fixable defect
rather than a property of the design. The fix is to stop the model guessing
what is indexed: put the index roots in the tool description, or force
describe_index on hop 0.

That is the whole argument. It is the same insight as retroactive conversation
compaction, except it never pays for the noise in the first place.

The slots to do it on already exist: llama-server defaults to four with a
unified KV buffer, measured concurrent here -- a helper query answered on slot
0 while a benchmark held slot 1, at zero additional VRAM, because the weights
are shared. A second copy of the model would have cost 5.95 GB to buy
something already running.

THE RISK, AND WHY CITATIONS ARE MANDATORY

A distilled finding hides the evidence that would show it is wrong. Raw search
results let the reading model notice that the top hit is a minified bundle or a
demo rather than the library; a confident summary of those same results does
not. Summarisation trades context economy for undetectable error, and that
trade is only acceptable if the error stays detectable.

So a finding is refused unless it cites paths that the investigation actually
retrieved. Cited paths are checked against the tool results, not taken on
trust, and the full trace is kept under a handle the caller can ask for. The
finding is a claim WITH receipts, never a claim instead of them.

WHAT IS DELIBERATELY NOT HERE

Autonomous delegation. The main model asks for an investigation by calling a
tool; the proxy does not decide on its own that a question needs one. Routing
that automatically is a closed-set decision and therefore a plausible job for
the classifier later, but it is a separate mechanism that would need its own
measurement, and bolting it on now would make the two indistinguishable.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

UPSTREAM = os.environ.get("LLAMA_STACK_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("YAMADORI_HELPER_MODEL", "bonsai")
# No hop count. MAX_HOPS (16) was removed on 2026-09-22 at the operator's
# instruction: a count hides a lying tool behind a slow failure. This context
# ends when it stops asking for tools, or when its window fills its 3/8 share
# of the KV pool (see investigate()).

# ---------------------------------------------------------------------------
# LAYA IS THE ROUTER, IN AND OUT. IT IS NOT OPTIONAL.
#
# The hemisphere previously decided nothing on the way in -- the model called
# a tool when it felt like it -- and summarised itself on the way out, with
# 2,500 max_tokens of the big model compressing text the big model had just
# written. Laya was built, measured and then routed around, which is why the
# service has been running with no callers.
#
# WHY THESE QUESTIONS AND NOT OTHERS. Laya is measured strong on FIXED STATE
# with VARYING TYPED OPTIONS, and measured weak comparing scores across
# different passages -- varying both collapses the real-vs-nonsense gap from
# 0.497 to roughly zero. Every question below holds the state constant (one
# request, or one finding) and varies only the options, which is the regime
# that works. None of them asks Laya to rank passages against each other.
#
# CONFIDENCE IS NOT CALIBRATED. Measured: a choice returning p=0.672 for its
# top option reported confidence 0.22. Calibration on ~50 labels halved ECE
# and moved accuracy 64.6% -> 75.1%. Until that is done for THESE questions,
# the thresholds below are placeholders, marked as such, and the raw
# probabilities are carried through so a caller can apply its own cutoff.
# ---------------------------------------------------------------------------
# The finding must be small or the whole point is lost -- but a cap that cuts
# silently hands the caller half a conclusion as if it were whole. 1400 cut 2
# of 26 measured findings with no marker (docs/CONSTRAINTS.md 9). The cap is
# now a breaker well above a normal finding, and a cut says so.
MAX_FINDING_CHARS = int(os.environ.get("YAMADORI_FINDING_CHARS", "6000"))

# Traces are kept so a finding can be audited, and evicted so a long session
# does not accumulate every investigation it ever ran.
_TRACES: dict[str, dict] = {}
MAX_TRACES = 32

# NOTHING HERE STATES A BUDGET, AND NOTHING COUNTS TURNS AT IT.
#
# There used to be a "BUDGET: at most N searching turns" line, justified by a
# live run that spent 51,206 tokens over eight hops and returned 48
# characters. That measurement was taken on a stack whose tools returned empty
# strings on failure: the model could not tell "nothing matched" from "this is
# broken", so it re-searched until the loop ended. It was not exploring out of
# appetite, it was retrying a broken call.
#
# Tool results now state the situation, whether it is retryable as a fact, and
# a remedy with an owner. That is what ends a search -- an answer, or an error
# it can believe. The number was treating the symptom.
#
# So: the loop ends when this context stops asking for tools. Telling it to
# hurry is the kind of instruction that buys a shallower answer and nothing
# else, and it is a second brain -- the whole reason it exists is to spend
# effort somewhere the main context does not have to.
SYSTEM = """You are investigating a question about a codebase for another
engineer. You have search tools. They have your conclusion and nothing else.

Work in this order:
1. Search for what you need. Read the code before describing it.
2. Stop as soon as you can answer. Do not explore further out of interest.
3. Write the finding.

The finding must be under 200 words and must cite the file paths and line
numbers you actually read. Write what the code DOES, not what you did to find
it. If the code does not answer the question, say exactly that -- an honest
"the index does not contain this" is useful and a guess is not.

Never state anything you did not read. Your reader cannot see your searches
and cannot check you.

WRITE IN PLAIN ENGINEERING ENGLISH. This finding is what crosses back. It is
read by another model, and may be read by a classifier, so structure it rather
than narrate it:

- One idea per sentence. Split compound sentences.
- Name the thing. Do not write "none", "it" or "this" where a noun fits.
- Active voice with a concrete subject: "the loader caches the plan", not
  "the plan is cached".
- No embedded clauses. Instead of "X, which suppresses Y", write "X. This
  suppresses Y."
- State a condition as a condition: "If N is zero, the branch returns early."
- Keep identifiers, paths, line numbers and measured values verbatim.

Instead of: "A MouseEvent has none and resolves to a different pointer with no
recorded initial click, which suppresses the synthetic event."
Write: "A MouseEvent has no recorded initial click. It resolves to a different
pointer. This condition suppresses the synthetic event\""""

# TWO BUGS LIVED IN THE OLD VERSION OF THIS PATTERN.
#
# 1. `yaml|yml` were absent, and config.yaml is where a large share of the
#    answers in this repo live -- so a finding citing it counted as citing
#    nothing.
# 2. Alternation is LEFTMOST-FIRST, and `js` came before `json`. So
#    `package.json` matched `.js` and left `on` unconsumed: every JSON
#    citation was recorded as a `.js` path that could never match anything
#    actually retrieved. Confirmed: `.npmrc.json` -> `.npmrc.js`.
#
# Fixed by ordering longest-first AND refusing a match that has more word
# characters after it, so a prefix extension can never win against the real
# one. Order alone is fragile -- the next person appending an extension would
# reintroduce it.
_PATH = re.compile(
    r"[\w./\\-]+\.(?:tsx|ts|jsx|js|mjs|cjs|json|jsonc|toml|yaml|yml"
    r"|hpp|cpp|rs|py|go|zig|wgsl|glsl|h|c)(?![A-Za-z0-9])")


def _post(path: str, payload: dict, timeout: int = 3600) -> dict:
    """One upstream call to the helper context, through the one door.

    Shaped by mcp/model.py -- the proxy's own effort mapping and budget rule
    at `xhigh`, the effort deep thinking runs at -- so this context can never
    again carry its own budget. It used to send max_tokens=2500 total, below
    what this model thinks for on an ordinary prompt (docs/CONSTRAINTS.md 8).
    `max_tokens` in the payload is the answer allowance: a tool call, or a
    finding.
    """
    import model
    # role="helper": deep thinking's thinking comes out of ITS 3/8 of the
    # pool (mcp/budget.py), never the conversation's 5/8.
    return model.post(model.shape(payload, effort="xhigh", role="helper"),
                      timeout=timeout)


def _cited(text: str) -> set[str]:
    """Paths a finding claims to have read.

    `lstrip("./")` was a CHARACTER-SET strip, not a prefix strip: it removes
    every leading `.` and `/` it finds, so `.eslintrc.js` became `eslintrc.js`
    and never matched what was actually retrieved. Confirmed live. Any finding
    citing a dotfile was silently treated as citing nothing -- and since the
    Laya grounding check was removed on measurement, this deterministic
    comparison is now the ONLY thing standing between a distilled finding and
    an undetectable fabrication.
    """
    out = set()
    for raw in _PATH.findall(text or ""):
        p = raw.replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        p = p.lstrip("/")          # leading separators only -- never dots
        if p:
            out.add(p.lower())
    return out


LAYA_URL = os.environ.get("LAYA_URL", "http://127.0.0.1:1237")
LAYA_TIMEOUT = float(os.environ.get("YAMADORI_LAYA_TIMEOUT", "20"))

# UNCALIBRATED. Replace from measured data; see the note above.
T_INVESTIGATE = float(os.environ.get("YAMADORI_T_INVESTIGATE", "0.5"))
T_ANSWERED = float(os.environ.get("YAMADORI_T_ANSWERED", "0.5"))



def _helper_budget() -> int:
    """The helper's slice of the unified KV pool, from budget.py.

    Read rather than hardcoded: the pool follows `-c` in config.yaml, and a
    second copy of the number here would be wrong the first time that changes.
    """
    try:
        import budget
        return int(budget.budgets().get("helper") or 61440)
    except Exception:                                            # noqa: BLE001
        return 61440


def _laya(state: str, questions: dict) -> dict:
    """One forward pass. Raises rather than guessing if the router is down.

    A router that silently falls back to "yes, investigate" is not a router;
    it is an always-on feature with a decorative dependency. The caller
    decides what to do when the decision cannot be made, and says so.
    """
    req = urllib.request.Request(
        LAYA_URL + "/decide",
        data=json.dumps({"state": state[:6000], "questions": questions}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=LAYA_TIMEOUT) as r:
        return json.load(r)


# Measured in bench/mechanisms/selectors.py, and reused rather than reinvented:
# Laya is ORDER-SENSITIVE, so a single call is not a reading, and a low margin
# is an ABSTENTION rather than a "no". The recorded numbers are that
# reordering-sensitive cases land at 0.012-0.184 while a decided case sits at
# 0.800, so anything under the gate is undecided.
LAYA_PERMUTATIONS = int(os.environ.get("YAMADORI_LAYA_PERMS", "2"))
MARGIN_GATE = float(os.environ.get("YAMADORI_LAYA_MARGIN", "0.3"))


def _choice_averaged(state: str, instructions: str, criteria: dict,
                     n: int = LAYA_PERMUTATIONS) -> dict | None:
    """One closed-set choice, averaged over option orderings.

    The first version of this asked a bare `noul` once and read the raw
    probability as a decision. It returned 0.162 for a question that plainly
    needed investigating, and I read that as "no". It was not a no -- it was
    the undecided band, measured, in a file that already said so.
    """
    keys = list(criteria)
    if len(keys) < 2:
        return None
    orders = [keys, list(reversed(keys))][:max(1, n)]
    acc = {k: 0.0 for k in keys}
    used = 0
    for order in orders:
        try:
            d = _laya(state, {"pick": {
                "type": "choice", "instructions": instructions,
                "criteria": {k: criteria[k] for k in order}}})
            probs = (d.get("answers", {}).get("pick", {}).get("probabilities")
                     or {})
        except Exception:                                        # noqa: BLE001
            continue
        # An order whose mass is ~0 is degenerate. Averaging it in would halve
        # every margin and make an undecided case look decided.
        if sum(float(v) for v in probs.values()) < 1e-6:
            continue
        for k in keys:
            acc[k] += float(probs.get(k, 0.0))
        used += 1
    if not used:
        return None
    avg = {k: v / used for k, v in acc.items()}
    ranked = sorted(avg.items(), key=lambda kv: -kv[1])
    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return {"choice": ranked[0][0], "probabilities": avg, "margin": margin,
            "orders_used": used, "decided": margin >= MARGIN_GATE}


def route_in(question: str, context: str = "") -> dict:
    """Should the second hemisphere run, and on what kind of question?

    A CLOSED CHOICE over a fixed state, which is the regime Laya is measured
    good at -- not a bare boolean, which is what failed. Under the margin gate
    this ABSTAINS: `run` is None rather than False, because "undecided" and
    "no" are different answers and a caller that cannot tell them apart will
    silently disable the feature it thinks it is gating.
    """
    state = (f"A user asked: {question}\n\n"
             f"Conversation context: {context[:2000] or '(none)'}")
    v = _choice_averaged(
        state,
        ("Decide how this question should be handled before answering it."),
        {"investigate": ("requires reading source code that is not already in "
                         "the conversation"),
         "answer_directly": ("answerable from general knowledge, or from what "
                             "is already in the conversation"),
         "clarify": "too underspecified to act on without asking first"})
    if v is None:
        return {"ok": False, "run": None,
                "reason": ("the decision router did not answer; no "
                           "investigation was started and none was ruled out")}
    if not v["decided"]:
        return {"ok": True, "run": None, "abstained": True,
                "margin": round(v["margin"], 4), "gate": MARGIN_GATE,
                "probabilities": v["probabilities"],
                "reason": ("below the margin gate -- undecided, which is not "
                           "the same as no")}
    return {"ok": True, "run": v["choice"] == "investigate",
            "choice": v["choice"], "abstained": False,
            "margin": round(v["margin"], 4), "gate": MARGIN_GATE,
            "probabilities": v["probabilities"], "calibrated": False}


def distil(question: str, finding: str, paths: list[str]) -> dict:
    """What crosses back over the callosum, as structure rather than prose.

    NOT A GATE, AND THE DOCSTRING USED TO CLAIM OTHERWISE. This said
    "MANDATORY, unlike route_in" while the code ~250 lines below said it "does
    not gate anything" -- both in the same file, and the code was right.

    The intent was sound: the deep-thinking context's own text should not
    enter the primary context unexamined, because otherwise the second brain
    is a nested chat whose output we paste. What failed was this mechanism.
    Measured at n=29 it chose `from_the_files` 29 times out of 29, AUC 0.667 --
    a constant, not a discriminator. The earlier 0.925-vs-0.070 separation that
    justified it was n=2. See docs/FINDINGS.md #17.

    The authority is the DETERMINISTIC citation test, which compares claimed
    paths against what was actually retrieved: exact, free, and unable to
    hallucinate agreement. This is kept and tested as the reference
    implementation of the call pattern, and it decides nothing.

    Built on the same averaged closed-choice primitive as route_in, for the
    same reason: a single `noul` on this returned 0.313 for a finding that
    answered the question and cited the two files it came from. That is the
    undecided band, not a no.

    Abstention is reported, never resolved by guessing. A finding we could not
    judge is handed back LABELLED as unjudged, because the alternative is
    asserting a confidence nobody measured.
    """
    state = (f"Question: {question}\n\n"
             f"Finding from the investigation: {finding[:4000]}\n\n"
             f"Files actually read: {', '.join(paths[:20]) or '(none)'}")

    verdict = _choice_averaged(
        state,
        "Judge this finding against the question it was meant to answer.",
        {"answers_it": "the finding answers the question that was asked",
         "partial": "it answers part of the question and leaves the rest open",
         "off_target": "it does not address what was asked"})
    grounding = _choice_averaged(
        state,
        "Where does the content of this finding come from?",
        {"from_the_files": ("it describes what is in the files that were "
                            "read"),
         "general_knowledge": ("it is general knowledge, not specific to "
                               "those files")})

    if verdict is None and grounding is None:
        return {"ok": False,
                "reason": ("the router did not answer, so the finding is "
                           "returned unjudged"),
                "judged": False}
    out = {"ok": True, "judged": True, "calibrated": False,
           "gate": MARGIN_GATE}
    if verdict is None or not verdict["decided"]:
        out["verdict"] = None
        out["verdict_abstained"] = True
        out["verdict_margin"] = round((verdict or {}).get("margin", 0.0), 4)
    else:
        out["verdict"] = verdict["choice"]
        out["verdict_abstained"] = False
        out["verdict_margin"] = round(verdict["margin"], 4)
    if grounding is None or not grounding["decided"]:
        out["grounded"] = None
        out["grounded_abstained"] = True
        out["grounded_margin"] = round((grounding or {}).get("margin", 0.0), 4)
    else:
        out["grounded"] = grounding["choice"] == "from_the_files"
        out["grounded_abstained"] = False
        out["grounded_margin"] = round(grounding["margin"], 4)
    # The citation check is DETERMINISTIC and stays the authority. Laya judges
    # the prose; only the paths actually retrieved prove it was read.
    out["cited_paths"] = list(paths[:20])
    return out


def investigate(question: str, tools: list[dict], run_tool,
                context: str = "", hops: int | None = None,
                on_think=None) -> dict:
    """Run a tool loop in a private context and return only the conclusion.

    `run_tool(name, args) -> str` is injected rather than imported so this can
    be exercised with a fake in tests, and so the caller decides which index
    and which repository the investigation is scoped to.
    """
    started = time.time()
    convo = [{"role": "system", "content": SYSTEM}]
    if context:
        convo.append({"role": "user",
                      "content": f"Context from the conversation:\n{context[:1500]}"})
    convo.append({"role": "user", "content": question})

    trace: list[dict] = []
    seen_paths: set[str] = set()
    spent = 0

    # NO HOP COUNT (removed 2026-09-22, operator's call). This context ends
    # when it stops asking for tools, or when its window reaches its QUARTER
    # of the KV pool (the helper-budget check below), which writes up what it
    # has. `hops` exists only so a test can bound a fake loop.
    hop = -1
    while True:
        hop += 1
        last = hops is not None and hop >= hops - 1
        if last:
            # GRACEFUL DEGRADATION OF A TRIPPED BREAKER, and nothing else.
            # This is no longer part of a normal run -- a healthy
            # investigation stops asking for tools and never gets here.
            #
            # It is kept because a breaker that simply expires returns
            # nothing for everything it spent, while one that asks for the
            # finding returns whatever was actually found. Withdrawing the
            # tools makes the request unambiguous.
            convo.append({"role": "user", "content":
                          "Stop searching. Write the finding now from what you "
                          "have, and cite the paths you read. If it is "
                          "incomplete, say what is missing."})
        payload = {"model": MODEL, "messages": convo,
                   "tools": [] if last else tools,
                   "max_tokens": 1500, "temperature": 0.2}
        try:
            d = _post("/v1/chat/completions", payload)
        except Exception as e:                                   # noqa: BLE001
            return _fail(f"{type(e).__name__}: {e}", trace, started, spent)
        msg = d["choices"][0]["message"]
        usage = d.get("usage") or {}
        spent += int(usage.get("total_tokens") or 0)
        # What this context OCCUPIES right now: this call's prompt plus what it
        # generated. `spent` sums every hop's whole prompt, so it grows as the
        # square of the conversation -- measured 3.1x the real peak at the
        # median (docs/CONSTRAINTS.md 7). It stays as the compute cost that is
        # reported to the caller; the KV check below uses the window.
        window = (int(usage.get("prompt_tokens") or 0)
                  + int(usage.get("completion_tokens") or 0))

        # THE SECOND BRAIN'S OWN REASONING, HANDED TO THE CALLER.
        #
        # This model thinks before it acts, and that reasoning was being
        # discarded -- only `content` was ever read. But the whole of this
        # context IS thinking: its reasoning and its searches are one
        # continuous thought, and a caller streaming it back to a person wants
        # the thought, not a summary of it afterwards.
        #
        # A callback rather than a return value, because the thinking happens
        # DURING a call that blocks for minutes. Returning it at the end would
        # be the summary this is meant to replace.
        if on_think and msg.get("reasoning_content"):
            try:
                on_think(msg["reasoning_content"])
            except Exception:                                    # noqa: BLE001
                pass    # a watcher must never be able to fail the work

        # THE HELPER'S KV ALLOCATION, ENFORCED.
        #
        # budget.py splits the one unified KV pool 5/8 to main and 3/8 to ONE
        # helper -- at 147,456 that is 92,160 + 55,296, at the 163,840
        # config.yaml launches with 102,400 + 61,440 (the operator's split of
        # 2026-09-22; earlier that day it was 1/2 + 2 x 1/4, main 73,728 and
        # 36,864 per helper, and before that main 88,473, one helper 36,864,
        # reserve 22,119).
        # The helper's share was a number nothing checked. This function's
        # first live run was recorded as "51,206 tokens, 39% over", but that
        # figure summed every hop's whole prompt -- docs/CONSTRAINTS.md item 7
        # -- so it is not evidence of a KV overrun.
        #
        # Not the same stop as the hop cap. This is the point where continuing
        # would take context the other side is using, so it writes up what it
        # has rather than being cut mid-thought.
        if window >= _helper_budget() and not last:
            trace.append({"tool": "(budget)", "args": {},
                          "result": f"helper KV budget reached: {window} tokens in context"})
            convo.append({"role": "user", "content":
                          "You have used the whole context budget for this "
                          "investigation. Write the finding now from what you "
                          "have, and say what is still unknown."})
            try:
                d = _post("/v1/chat/completions",
                          dict(payload, tools=[], messages=convo))
                spent += int((d.get("usage") or {}).get("total_tokens") or 0)
                msg = d["choices"][0]["message"]
            except Exception as e:                               # noqa: BLE001
                return _fail(f"{type(e).__name__}: {e}", trace, started, spent)
            return _finish((msg.get("content") or "").strip(), trace,
                           seen_paths, started, spent, question)

        calls = msg.get("tool_calls") or []

        if not calls:
            finding = (msg.get("content") or "").strip()
            if not finding and msg.get("reasoning_content"):
                # Budget exhausted before writing. Reported as such rather
                # than returned as an empty conclusion, which would read to
                # the caller as "investigated, found nothing".
                return _fail("ran out of budget while reasoning, no conclusion",
                             trace, started, spent)
            return _finish(finding, trace, seen_paths, started, spent,
                           question)

        convo.append({"role": "assistant", "content": msg.get("content") or "",
                      "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {}).get("name", "")
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            t0 = time.time()
            try:
                out = run_tool(fn, args)
            except Exception as e:                               # noqa: BLE001
                out = f"{fn} failed: {type(e).__name__}: {e}"
            seen_paths |= _cited(out)
            trace.append({"hop": hop, "tool": fn, "args": args,
                          "chars": len(out),
                          "ms": round((time.time() - t0) * 1000)})
            # Same breaker as the proxy's loops, with a marker and the next
            # range to request -- a silent cut handed this context half a file
            # under a header promising all of it (docs/CONSTRAINTS.md 12).
            import repeats
            convo.append({"role": "tool", "tool_call_id": c.get("id"),
                          "content": repeats.cap_tool_result(out, fn, args)})

    # Reaching here is pathological, not routine. The breaker tripped, which
    # means this context asked for tools MAX_HOPS times without ever deciding
    # it had an answer. With truthful tool results that should not happen, so
    # the message names it as a defect rather than reporting a spent budget.
    return _fail(
        f"runaway breaker tripped after {hops} tool calls without a "
        f"conclusion -- this indicates a tool returning results the model "
        f"cannot act on, not a budget being exhausted; check the trace",
        trace, started, spent)


def _finish(finding: str, trace: list, seen: set[str], started: float,
            spent: int, question: str = "") -> dict:
    """Attach the receipts, and refuse a finding that has none.

    A conclusion citing nothing is indistinguishable from a conclusion the
    model invented, and the caller cannot tell the difference because it never
    saw the searches. Checking the citations against what was actually
    retrieved is the only thing standing between context economy and confident
    fabrication.
    """
    claimed = _cited(finding)
    supported = {p for p in claimed
                 if any(p == s or s.endswith("/" + p) or p.endswith("/" + s)
                        for s in seen)}
    unsupported = sorted(claimed - supported)

    handle = uuid.uuid4().hex[:8]
    if len(_TRACES) >= MAX_TRACES:
        _TRACES.pop(next(iter(_TRACES)))
    _TRACES[handle] = {"trace": trace, "finding": finding,
                       "retrieved": sorted(seen)}

    note = ""
    if not claimed and trace:
        note = ("\n\n[This finding cites no file. It was not checked against "
                "anything retrieved -- treat it as unverified.]")
    elif unsupported:
        note = ("\n\n[Cites paths that were never retrieved: "
                + ", ".join(unsupported[:4])
                + ". Those are unsupported by this investigation.]")

    if len(finding) > MAX_FINDING_CHARS:
        finding = (finding[:MAX_FINDING_CHARS]
                   + f"\n\n[finding cut at {MAX_FINDING_CHARS} characters "
                   f"of {len(finding)}]")
    out = {"ok": True, "finding": finding + note,
           "handle": handle, "hops": len(trace),
           "tools_used": sorted({t["tool"] for t in trace}),
           "cited": sorted(supported), "unsupported": unsupported,
           "helper_tokens": spent,
           "seconds": round(time.time() - started, 1)}

    # LAYA DOES NOT DISTIL HERE, AND THAT IS A MEASURED DECISION.
    #
    # It was wired in on the strength of a two-point result -- grounded finding
    # 0.925 against a vague one 0.070 -- which did NOT replicate. At n=29 with
    # 12 deliberately fabricated findings it chose `from_the_files` 29/29:
    # AUC 0.667, probability gap +0.029, mean margin 0.867. Confident, constant,
    # and carrying no information.
    #
    # The reason is structural rather than a weakness. Laya is an
    # option-scoring cross-encoder and can only judge what is IN its input.
    # Asking "does this finding describe those files" without the file contents
    # is undecidable: "MAX_TRACES is 256" and "MAX_TRACES is 1024" are the same
    # sentence to a model that cannot see the file. Supplying excerpts does not
    # fix it either -- pair members sit at standardised cosine 0.727, so the
    # excerpt swamps a twenty-word claim.
    #
    # So the fabrication check is the DETERMINISTIC citation test above, which
    # compares claimed paths against what was actually retrieved. That is exact,
    # free, and already the authority. A second opinion that answers 29/29 the
    # same way is not a second opinion.
    #
    # `distil()` is kept and tested -- it is the reference implementation of the
    # call pattern -- but it does not gate anything. See docs/LAYA.md Part II.
    return out


def _fail(why: str, trace: list, started: float, spent: int) -> dict:
    # The trace is kept on failure TOO. A failed investigation is exactly the
    # one worth inspecting, and the first live run discarded its own evidence
    # because only the success path wrote to the store.
    handle = uuid.uuid4().hex[:8]
    if len(_TRACES) >= MAX_TRACES:
        _TRACES.pop(next(iter(_TRACES)))
    _TRACES[handle] = {"trace": trace, "finding": why, "retrieved": [],
                       "failed": True}
    return {"ok": False, "finding": f"investigation failed: {why}",
            "handle": handle, "hops": len(trace),
            "tools_used": sorted({t["tool"] for t in trace}),
            "cited": [], "unsupported": [], "helper_tokens": spent,
            "seconds": round(time.time() - started, 1)}


def get_trace(handle: str) -> dict | None:
    """The full trace behind a finding, for when the caller wants the receipts."""
    return _TRACES.get(handle)


TOOL = {
    "type": "function",
    "function": {
        "name": "delegate_investigation",
        # The model is told what this DOES, not how it is built. It does not
        # need to know a second context exists, any more than it needs the
        # shape of the KV pool -- and describing our plumbing in a tool
        # description invites it to reason about the plumbing instead of the
        # question. Same reason the user-facing strings say "thinking deeply".
        "description": (
            "Think deeply about one self-contained question before answering "
            "it. This runs several searches and reads their results without "
            "filling this conversation, then returns one short finding with "
            "file citations. Use it when an answer needs searching you do not "
            "need to watch. Do not use it for a single lookup -- call the "
            "search tool yourself, it is faster."),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "One self-contained question. The investigator cannot "
                        "see this conversation, so name the symbols, files or "
                        "versions it needs."),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional background it would otherwise lack, such as "
                        "the library version in use."),
                },
            },
            "required": ["question"],
        },
    },
}


def as_thinking(result: dict) -> str:
    """The investigation rendered for a thinking block, NOT for the context.

    Clients render `reasoning_content` and do not echo it back in the next
    request, which is the property that matters here. Emitting this as
    `content` instead would put it in the transcript, the harness would send
    it back on the following turn, and the context saving would be undone by
    the client after the server had carefully avoided it.

    So this is a window into the other hemisphere for the person watching,
    while the model still receives only the finding.
    """
    tr = get_trace(result.get("handle") or "") or {}
    lines = [f"thinking deeply ({result.get('seconds', 0)}s):"]
    for step in tr.get("trace", [])[:12]:
        arg = ""
        for k in ("symbol", "pattern", "query", "path"):
            if step.get("args", {}).get(k):
                arg = json.dumps(step["args"][k])[:48]
                break
        lines.append(f"  {step['tool']} {arg} -> {step['chars']} chars "
                     f"({step['ms']} ms)")
    if result.get("cited"):
        lines.append(f"  cited: {', '.join(result['cited'][:6])}")
    if result.get("unsupported"):
        lines.append(f"  UNSUPPORTED citations: {', '.join(result['unsupported'])}")
    return "\n".join(lines)
